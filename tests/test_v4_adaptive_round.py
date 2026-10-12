"""Actual FIFO/socket cancellation and fixed-shape tests; no GPU speed claims."""
from copy import deepcopy
import time
import pytest

from v4_adaptive_round import RoundPolicy, CALIBRATION_SCHEMA, calibration_binding


OPTIONS = {"binding": {"cohort": "a"*64, "runtime": "b"*64, "source": "c"*64},
           "loaded_block_size": 3, "depth": 8, "floor": 1}


def config(*, mode="shadow", preferred=1, now=None, dma=None):
    now = time.time() if now is None else now
    return {"enabled": True, "mode": mode, "min_hold_windows": 1,
        "calibration": {"schema": CALIBRATION_SCHEMA, "binding_sha256": calibration_binding(OPTIONS),
            "created_at": now-1, "expires_at": now+60,
            "profiles": [{"draft_prefix_cap": cap, "committed_tokens": 100,
                "elapsed_s": .5 if cap == preferred else 10., "expert_dma_bytes": dma}
                for cap in (0, 1, 2, 3)]}}


class Clock:
    def __init__(self): self.now = 100.
    def __call__(self): return self.now
    def advance(self, value): self.now += value


def make_policy(value=None, clock=None):
    options = dict(OPTIONS)
    if clock is not None:
        options.update(clock=clock, wall_clock=clock)
    return RoundPolicy(value or config(), **options)


@pytest.mark.parametrize("retry,explicit_greedy", [(False, False), (True, False), (False, True)])
def test_gateway_wires_fresh_policy_only_to_unreplayed_pipeline_attempts(retry, explicit_greedy):
    from test_v4_gateway import Pipe, backend, job
    class RoundPipe(Pipe):
        V4_SPEC_DEPTH, V4_REFILL_FLOOR, V4_LAZY_DRAFT = 8, 1, False
        def __init__(self):
            super().__init__(fail_first=retry)
            self.policies = []
        def coordinate(self, *args, cancel_check, expected_by_signer, strict_job_binding=False, **kwargs):
            policy = kwargs.get("round_policy")
            self.policies.append(policy)
            if policy is not None:
                assert kwargs["draft_block_limit"] == 3
                policy.refill(committed_tokens=0, available_drafts=3)
            result = super().coordinate(*args, cancel_check=cancel_check,
                expected_by_signer=expected_by_signer, strict_job_binding=strict_job_binding, **kwargs)
            if policy is not None:
                result["round_speculation"] = policy.finish(len(result["tokens"]))
            return result
        coordinate_dspark = coordinate
        coordinate_dspark_pipelined = coordinate
    producer = RoundPipe()
    service, request = backend(producer), job(maximum=6)
    # Admission is tested using complete signed reports in round_calibration;
    # this seam isolates per-attempt delivery and actual signed completion.
    service._round_options = deepcopy(OPTIONS)
    service._round_options["binding"]["lazy"] = False
    service._round_config = config(mode="enforce")
    service._round_config["calibration"]["binding_sha256"] = calibration_binding(service._round_options)
    if explicit_greedy:
        request.payload["mode"] = "greedy"
    try:
        result = service.execute(request, request.emitter(), request.check_stop)
        assert result["proof"]["verified"] and request.tokens == list(range(65, 71))
        if retry:
            assert producer.policies[0] is not None and producer.policies[1] is None
            assert "round_speculation" not in result.get("optimizations", {})
        elif explicit_greedy:
            assert producer.policies == [None]
        else:
            assert result["optimizations"]["round_speculation"]["mode"] == "enforce"
    finally:
        service.close()


@pytest.mark.parametrize("change", [
    lambda c: c.update(unknown=True),
    lambda c: c.update(mode="automatic"),
    lambda c: c.update(enabled=1),
    lambda c: c["calibration"].update(binding_sha256="f"*64),
    lambda c: c["calibration"].update(created_at=time.time()+100),
    lambda c: c["calibration"]["profiles"][0].update(elapsed_s=0),
    lambda c: c["calibration"]["profiles"][0].update(expert_dma_bytes=True),
    lambda c: c["calibration"]["profiles"][0].update(draft_prefix_cap=3),
    lambda c: c["calibration"].update(profiles=[]),
])
def test_incomplete_or_mismatched_calibrations_are_rejected(change):
    value = config(); change(value)
    with pytest.raises(ValueError):
        make_policy(value)


def test_shadow_does_not_measure_counterfactual_candidate_rates():
    clock = Clock(); policy = make_policy(config(now=clock(), preferred=1), clock)
    assert policy.refill(committed_tokens=2, available_drafts=3) == 3
    clock.advance(.2)
    policy.observe_reply_cost(None)
    policy.refill(committed_tokens=5, available_drafts=3)
    record = policy.report()
    assert 1 not in policy._runtime and 3 in policy._runtime
    assert any(row.get("selected_cap") == 1 and row["applied_cap"] == 3 for row in record["records"])
    measured = [row for row in record["records"] if row["kind"] == "window"][-1]
    assert measured["expert_dma_bytes"] is None
    assert measured["cost_coverage"] == "unknown"


def test_actual_rate_can_replace_the_preapproved_choice_without_changing_shapes():
    clock = Clock(); policy = make_policy(config(mode="enforce", preferred=1, now=clock()), clock)
    assert policy.refill(committed_tokens=2, available_drafts=3) == 1
    clock.advance(1.)
    # The measured 1-token/s execution loses to the other measured 10-token/s
    # calibration. This does not infer speed from acceptance percentage.
    assert policy.refill(committed_tokens=3, available_drafts=3) == 3
    assert policy.block_size == 3 and policy.depth == 8
    assert policy.report()["switches"] == 2


def test_expired_calibration_returns_to_baseline_without_aborting_service():
    clock = Clock(); value = config(mode="enforce", preferred=0, now=clock())
    policy = make_policy(value, clock)
    assert policy.refill(committed_tokens=2, available_drafts=3) == 0
    clock.advance(61.)
    assert policy.refill(committed_tokens=3, available_drafts=3) == 3
    assert policy.report()["records"][-1]["reason"] == "calibration_expired"
    restarted = make_policy(value, clock)
    assert restarted.refill(committed_tokens=2, available_drafts=3) == 3


def test_missing_dma_never_becomes_a_zero_cost_budget_pass():
    value = config(mode="enforce", preferred=0)
    value["max_dma_bytes_per_committed_token"] = 1000
    policy = make_policy(value)
    assert policy.refill(committed_tokens=2, available_drafts=3) == 3
    assert policy.report()["records"][-1]["reason"] == "missing_or_over_budget_dma_measurement"
    policy.observe_reply_cost({"coverage": "tail", "expert_dma_bytes": 0})
    policy.finish(3)
    assert policy.report()["records"][-1]["expert_dma_bytes"] is None


def test_real_zero_dma_is_distinct_and_complete_costs_can_be_added():
    clock = Clock(); value = config(mode="enforce", now=clock(), dma=0)
    value["max_dma_bytes_per_committed_token"] = 1000
    policy = make_policy(value, clock)
    assert policy.refill(committed_tokens=2, available_drafts=3) == 1
    policy.observe_reply_cost({"coverage": "all_stages", "expert_dma_bytes": 30})
    policy.observe_reply_cost({"coverage": "all_stages", "expert_dma_bytes": 50})
    clock.advance(1.)
    policy.refill(committed_tokens=4, available_drafts=3)
    window = [row for row in policy.report()["records"] if row["kind"] == "window"][-1]
    assert window["expert_dma_bytes"] == 80 and window["learned"]


def test_observation_budget_and_terminal_window_are_bounded():
    clock = Clock(); value = config(now=clock()); value["max_records"] = 8
    policy = make_policy(value, clock)
    for index in range(30):
        clock.advance(.01)
        policy.refill(committed_tokens=index+2, available_drafts=3)
    policy.finish(32)
    report = policy.report()
    assert len(report["records"]) == 8
    assert report["records"][-1]["truncated"] and not report["records"][-1]["learned"]
    report["records"].clear()
    assert len(policy.report()["records"]) == 8
    with pytest.raises(ValueError, match="reused"):
        policy.validate_runtime(depth=8, floor=1, draft_block_limit=3)


def run_ring(policy=None, *, lazy=False, wrong=False, maximum=18, eos_ids=()):
    from test_v4_pipe import _ScriptedPipeRing, TRUTH, PIPE_PROMPT, VP, _perfect_block
    ring = _ScriptedPipeRing(TRUTH, (lambda q: [901, 902, 903]) if wrong else _perfect_block)
    ring.lazy = lazy
    try:
        result = VP.coordinate_dspark_pipelined(ring.pipe_a, ring.ret_b, PIPE_PROMPT, maximum,
            timeout=3, depth=8, floor=1, lazy=lazy, draft_block_limit=3,
            round_policy=policy, eos_ids=eos_ids)
        return result, list(ring.frames), dict(ring.dprev)
    finally:
        ring.close()


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("wrong", [False, True])
@pytest.mark.parametrize("cap", [0, 1, 2, 3])
def test_enforced_caps_keep_actual_socket_greedy_output_and_s1_frames(cap, wrong, lazy):
    from test_v4_pipe import TRUTH, PIPE_PROMPT
    policy = make_policy(config(mode="enforce", preferred=cap))
    result, frames, hints = run_ring(policy, lazy=lazy, wrong=wrong)
    assert result["tokens"] == [TRUTH[p] for p in range(len(PIPE_PROMPT), len(PIPE_PROMPT)+18)]
    assert all(len(ids) == 1 for ids, pos, _ in frames if pos)
    assert result["depth"] == 8 and result["floor"] == 1
    assert not any(hints.values()), "variable caps must not promise the NEXT burst's width"
    assert result["round_speculation"]["loaded_block_size"] == 3


@pytest.mark.parametrize("lazy", [False, True])
def test_shadow_keeps_same_frame_decisions_as_unmodified_pipeline(lazy):
    original, _, _ = run_ring(lazy=lazy)
    shadow, _, _ = run_ring(make_policy(), lazy=lazy)
    for key in ("tokens", "frames", "drafted", "cancels", "accept_by_depth", "topup_frames"):
        assert shadow[key] == original[key]
    assert shadow["round_speculation"]["mode"] == "shadow"


def test_eos_and_single_token_requests_remain_bounded():
    first, _, _ = run_ring(make_policy(), maximum=1)
    assert len(first["tokens"]) == 1 and first["round_speculation"]["windows"] == 0
    result, _, _ = run_ring(make_policy(config(mode="enforce", preferred=1)), eos_ids=(107,))
    assert result["tokens"] == [103, 104, 105, 106, 107]


def test_controller_rejects_runtime_depth_change_before_sending_reset():
    from test_v4_pipe import _ScriptedPipeRing, TRUTH, PIPE_PROMPT, VP, _perfect_block
    ring = _ScriptedPipeRing(TRUTH, _perfect_block)
    try:
        with pytest.raises(ValueError, match="fixed runtime"):
            VP.coordinate_dspark_pipelined(ring.pipe_a, ring.ret_b, PIPE_PROMPT, 8,
                timeout=3, depth=9, floor=1, draft_block_limit=3, round_policy=make_policy())
        assert ring.frames == []
    finally:
        ring.close()

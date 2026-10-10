"""Policy accounting and request-boundary behavior; no GPU throughput claims."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from shard.benchmark_metrics import COUNTERS, make_coordinator_diagnostics
from shard.speculation_policy import (PolicyDecision, RequestFeedback, RequestPolicy, SpeculationContext,
    SpeculationPolicyError, SpeculationRecipe, feedback_from_result)


class Clock:
    def __init__(self):
        self.now = 100.0
    def __call__(self):
        return self.now
    def advance(self, seconds):
        self.now += seconds


def recipes():
    return [SpeculationRecipe("greedy", "greedy"), SpeculationRecipe("pipe8", "pipelined", 8, 1, True)]


def context(**changes):
    value = SpeculationContext("tenant-private-alpha", "a"*64, "ring", "generation-1", "b"*64, 512, 512, "code")
    return replace(value, **changes)


def policy(clock=None, **options):
    return RequestPolicy(recipes(), "greedy", enabled=True, clock=clock or Clock(), **options)


def feedback(mode, *, rate=20, tokens=101, prefill_s=2, drain_s=0, accepted=None, proposed=None,
             judged=None, success=True, replayed=False, cancelled=False):
    speculative = mode == "pipelined"
    decode = tokens-1
    accepted = min(60, decode) if speculative and accepted is None else accepted or 0
    proposed = decode+10 if speculative and proposed is None else proposed or 0
    sent = max(decode+10, proposed-5) if speculative else decode
    wasted = sent-decode
    counts = {"accepted_predictions": accepted, "proposed_predictions": proposed, "cancel_events": int(speculative),
        "speculation_cycles": 2 if speculative else None, "frames_enqueued": sent+5 if speculative else sent,
        "frames_sent": sent, "replies_received": sent, "frames_judged": decode,
        "stale_replies": wasted//2, "drained_replies": wasted-wasted//2,
        "unsent_frames": 5 if speculative else 0}
    value = make_coordinator_diagnostics(mode, committed_tokens=tokens, counters=counts,
        timing={"first_token_s": prefill_s, "last_token_s": prefill_s+decode/rate,
                "request_elapsed_s": prefill_s+decode/rate+drain_s, "receipt_sweep_s": 20},
        inflight_intervals=[{"duration_s": decode/rate, "level": 4}] if speculative else None)
    return RequestFeedback.from_diagnostics(value, judged_predictions=judged, success=success, replayed=replayed, cancelled=cancelled,
        observed_recipe=recipes()[int(speculative)])


def run(p, ctx, *, rate=20, **kwargs):
    decision = p.choose(ctx)
    observation = p.observe(decision, feedback(decision.recipe.mode, rate=rate, **kwargs))
    return decision, observation


def train(p, ctx, *, baseline_rate=20, candidate_rate=30, **candidate_options):
    for rate in (baseline_rate, baseline_rate, candidate_rate, candidate_rate):
        chosen = p.choose(ctx)
        p.observe(chosen, feedback(chosen.recipe.mode, rate=rate,
            **(candidate_options if chosen.recipe.mode == "pipelined" else {})))


def test_default_off_preserves_baseline_without_allocating_learning_state_or_leaking_tenant():
    p = RequestPolicy(recipes(), "greedy", clock=Clock())
    for _ in range(8):
        chosen, result = run(p, context())
        assert chosen.recipe == recipes()[0] and chosen.reason == "disabled_baseline"
        assert not result["learned"]
        assert "tenant-private-alpha" not in json.dumps(result)
        assert "workload" not in chosen.to_dict()
    assert not p._states and not p._pending


@pytest.mark.parametrize("kwargs", [
    {"mode": "dspark"}, {"mode": "pipelined", "depth": 1}, {"mode": "pipelined", "depth": 8, "floor": 9},
    {"mode": "pipelined", "depth": True}, {"mode": "greedy", "lazy": True}, {"mode": "greedy", "depth": 8},
])
def test_policy_cannot_invent_invalid_recipe_shapes(kwargs):
    with pytest.raises(SpeculationPolicyError):
        SpeculationRecipe("bad", **kwargs)


def test_approved_recipes_are_finite_immutable_and_require_known_baseline():
    with pytest.raises(SpeculationPolicyError, match="duplicate"):
        RequestPolicy([SpeculationRecipe("a", "greedy"), SpeculationRecipe("b", "greedy")], "a")
    with pytest.raises(SpeculationPolicyError, match="baseline"):
        RequestPolicy(recipes(), "unknown")
    with pytest.raises(SpeculationPolicyError, match="budget"):
        policy(probe_attempts=1)
    p = policy()
    with pytest.raises(TypeError):
        p.recipes["unapproved"] = SpeculationRecipe("unapproved", "pipelined", 16)
    assert p.recipes["pipe8"].coordinator_kwargs() == {"depth": 8, "floor": 1, "lazy": True}
    assert "block" not in json.dumps(p.recipes["pipe8"].to_dict())


def test_actual_decode_gain_selects_candidate_after_bounded_probes_and_keeps_prefill_separate():
    p = policy(); ctx = context()
    reasons = []
    for rate in (20, 20, 35, 35):
        chosen = p.choose(ctx); reasons.append(chosen.reason)
        result = p.observe(chosen, feedback(chosen.recipe.mode, rate=rate,
            prefill_s=1000 if chosen.recipe.mode == "pipelined" else 1))
        assert result["learned"]
    chosen = p.choose(ctx)
    assert reasons == ["baseline_warmup", "baseline_warmup", "bounded_probe", "bounded_probe"]
    assert chosen.recipe.recipe_id == "pipe8" and chosen.reason == "measured_gain"
    assert "tenant" not in json.dumps(chosen.to_dict())


def test_high_acceptance_and_fast_kernel_do_not_hide_expensive_drain():
    p = policy(); ctx = context()
    train(p, ctx, candidate_rate=200, accepted=99, proposed=100, judged=100, drain_s=10)
    selected = p.choose(ctx)
    assert selected.recipe.mode == "greedy"
    assert selected.reason == "incumbent_hysteresis"


def test_unjudged_cancelled_frames_are_cost_not_false_acceptance_trials():
    value = feedback("pipelined", tokens=8, accepted=3, proposed=40, judged=3)
    raw = value.to_dict()
    assert raw["conditional_acceptance"] == 1
    assert raw["proposal_yield"] == 3/40
    assert raw["counters"]["stale_replies"] == 14
    assert raw["counters"]["unsent_frames"] == 5
    unknown = feedback("pipelined", tokens=8, accepted=3, proposed=40)
    assert unknown.to_dict()["conditional_acceptance"] is None


@pytest.mark.parametrize("status", ["cancelled", "replayed", "success"])
def test_failed_cancelled_replayed_jobs_never_train_or_trigger_unbounded_probe(status):
    p = policy(); ctx = context()
    options = {status: False if status == "success" else True}
    for _ in range(10):
        chosen, result = run(p, ctx, rate=1000000, **options)
        assert chosen.recipe.mode == "greedy" and not result["learned"]
    assert chosen.reason == "insufficient_baseline_evidence"


def test_tiny_warmup_is_not_a_decode_performance_measurement():
    p = policy()
    for _ in range(5):
        chosen, result = run(p, context(max_new_tokens=2), tokens=2, rate=10000)
        assert not result["learned"] and result["reason"] == "insufficient_decode_measurement"
        assert chosen.recipe.mode == "greedy"


@pytest.mark.parametrize("changes", [
    {"tenant_id": "another-tenant"}, {"cohort_id": "c"*64}, {"ring_id": "another-ring"},
    {"ring_generation": "new-generation"}, {"runtime_config_sha256": "d"*64},
    {"context_tokens": 4096}, {"max_new_tokens": 64}, {"workload": "prose"},
])
def test_learning_is_isolated_by_tenant_model_generation_runtime_context_and_workload(changes):
    p = policy(); train(p, context())
    selected = p.choose(context(**changes))
    assert selected.recipe.mode == "greedy" and selected.reason == "baseline_warmup"


def test_hysteresis_rejects_small_or_noisy_gain():
    p = policy(min_gain=.10)
    train(p, context(), baseline_rate=20, candidate_rate=21)
    assert p.choose(context()).recipe.mode == "greedy"
    p = policy()
    for rate in (20, 20, 10, 40):
        chosen = p.choose(context()); p.observe(chosen, feedback(chosen.recipe.mode, rate=rate))
    assert p.choose(context()).recipe.mode == "greedy"  # lower-quartile candidate has no robust 5% gain


def test_feedback_expiry_forces_new_baseline_and_capped_probe_budget_is_renewed():
    clock = Clock(); p = policy(clock, feedback_ttl_s=60, max_probe_requests=2)
    train(p, context())
    chosen = p.choose(context()); p.observe(chosen, feedback(chosen.recipe.mode, rate=30))
    clock.advance(61)
    assert p.choose(context()).reason == "baseline_warmup"


def test_expired_pending_decision_never_teaches_new_generation_and_keeps_actual_feedback_record():
    clock = Clock(); p = policy(clock, decision_ttl_s=10)
    old = p.choose(context()); clock.advance(11)
    new = p.choose(context())
    result = p.observe(old, feedback("greedy", rate=1000))
    assert not result["learned"] and result["reason"] == "expired_feedback"
    assert result["feedback"]["committed_decode_tokens"] == 100
    p.discard(new, reason="request_cancelled")


def test_feedback_idempotency_and_wrong_mode_fail_closed_without_consuming_valid_pending():
    p = policy(); chosen = p.choose(context()); value = feedback("greedy")
    with pytest.raises(SpeculationPolicyError, match="mode differs"):
        p.observe(chosen, feedback("pipelined"))
    first = p.observe(chosen, value)
    second = p.observe(chosen, value)
    assert first == second
    second["feedback"]["committed_tokens"] = 0
    assert p.observe(chosen, value)["feedback"]["committed_tokens"] == 101
    with pytest.raises(SpeculationPolicyError, match="different feedback"):
        p.observe(chosen, feedback("greedy", rate=30))
    other = policy()
    with pytest.raises(SpeculationPolicyError, match="issued"):
        other.observe(chosen, value)
    with pytest.raises(SpeculationPolicyError, match="issued"):
        p.observe(chosen.to_dict(), value)


def test_state_pending_and_completion_storage_remain_bounded_under_many_tenants():
    p = policy(max_states=2, max_pending=2, completion_capacity=3)
    a = p.choose(context(tenant_id="a")); b = p.choose(context(tenant_id="b"))
    c = p.choose(context(tenant_id="c"))
    assert c.reason == "pending_capacity_baseline"
    p.observe(c, feedback("greedy"))
    p.discard(a); p.discard(b)
    for i in range(20):
        chosen = p.choose(context(tenant_id=str(i))); p.observe(chosen, feedback("greedy"))
    assert len(p._states) <= 2 and len(p._pending) == 0 and len(p._completed) == 3


def test_concurrent_pending_requests_do_not_duplicate_exploration_or_learning():
    p = policy()
    with ThreadPoolExecutor(max_workers=8) as pool:
        selected = list(pool.map(lambda _: p.choose(context()), range(8)))
    assert sum(row.reason == "baseline_warmup" for row in selected) == 1
    for row in selected:
        result = p.observe(row, feedback("greedy"))
        assert result["learned"] == (row.reason == "baseline_warmup")


def test_unknown_and_inconsistent_counters_are_rejected_without_made_up_zeros():
    original = feedback("pipelined")
    counts = original.counters; counts["stale_replies"] = None
    with pytest.raises(SpeculationPolicyError, match="incomplete"):
        replace(original, counter_items=tuple(counts.items()))
    with pytest.raises(ValueError):
        replace(original, judged_predictions=1)
    with pytest.raises(SpeculationPolicyError, match="clock"):
        replace(original, request_elapsed_s=1)


def test_import_stays_standard_library_only():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run([sys.executable, "-c", "import sys; import shard.speculation_policy; assert 'torch' not in sys.modules"],
        cwd=root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_actual_recipe_binding_and_request_token_budget_are_required_for_learning():
    p = policy(); chosen = p.choose(context())
    with pytest.raises(SpeculationPolicyError, match="actual feedback bound"):
        p.observe(chosen, replace(feedback("greedy"), observed_recipe=None))
    with pytest.raises(SpeculationPolicyError, match="request authorized"):
        p.observe(chosen, feedback("greedy", tokens=513))
    clone = replace(chosen, _learning=False)
    with pytest.raises(SpeculationPolicyError, match="object differs"):
        p.observe(clone, feedback("greedy"))
    assert p.observe(chosen, feedback("greedy"))["learned"]


def test_feedback_result_rejects_unknown_or_different_actual_coordinator_recipe():
    chosen = recipes()[1]
    value = feedback("pipelined")
    result = {"ok": True, "tokens": list(range(value.committed_tokens)), "mode": "pipelined",
        "coordinator_counters": value.counters, "receipt_sweep_s": 0, "depth": 8, "floor": 1, "lazy": True}
    args = {"first_token_s": value.first_token_s, "last_token_s": value.last_token_s,
            "request_elapsed_s": value.request_elapsed_s, "expected_recipe": chosen}
    assert feedback_from_result(result, **args).observed_recipe == chosen
    for key, actual in (("depth", 16), ("floor", 2), ("lazy", False), ("depth", None)):
        changed = dict(result, **{key: actual})
        with pytest.raises(SpeculationPolicyError, match="depth/floor/lazy"):
            feedback_from_result(changed, **args)
    with pytest.raises(SpeculationPolicyError, match="successful committed"):
        feedback_from_result(dict(result, ok=False), **args)


def test_public_configuration_is_detached_and_binds_implementation_without_history():
    p = policy(); before = p.configuration(); train(p, context())
    after = p.configuration()
    assert before == after and len(before["implementation_sha256"]) == 64
    assert "tenant-private-alpha" not in json.dumps(after)
    after["recipes"][0]["mode"] = "pipelined"
    assert p.configuration()["recipes"][0]["mode"] == "greedy"


def test_probe_attempts_stay_capped_when_one_candidate_has_no_usable_samples():
    p = policy(max_probe_requests=2)
    for _ in range(2):
        run(p, context())
    for _ in range(2):
        selected = p.choose(context()); assert selected.is_probe
        p.discard(selected, reason="request_failed")
    for _ in range(10):
        selected, _ = run(p, context())
        assert selected.recipe.mode == "greedy" and not selected.is_probe


@pytest.mark.parametrize("blocks", ["accepted", "rejected"])
@pytest.mark.parametrize("recipe", [SpeculationRecipe("p8f1", "pipelined", 8, 1, True), SpeculationRecipe("p9f2", "pipelined", 9, 2, False)])
def test_policy_feedback_from_actual_socket_coordinator_preserves_token_stream_and_accounting(blocks, recipe):
    # Existing harness runs the REAL coordinator/frame codec against a scripted
    # socket tail that derives its frontier verdict. No physical model/GPU is used.
    from test_v4_pipe import _ScriptedPipeRing, TRUTH, PIPE_PROMPT, _perfect_block, VP
    import time
    p = RequestPolicy([recipe], recipe.recipe_id, enabled=True)
    decision = p.choose(context())
    proposer = _perfect_block if blocks == "accepted" else lambda q: [901, 902, 903]
    ring = _ScriptedPipeRing(TRUTH, proposer); ring.lazy = recipe.lazy
    start = time.perf_counter(); offsets = []
    try:
        result = VP.coordinate_dspark_pipelined(ring.pipe_a, ring.ret_b, PIPE_PROMPT, 12,
            timeout=10, on_token=lambda token: offsets.append(time.perf_counter()-start), **decision.recipe.coordinator_kwargs())
        elapsed = time.perf_counter()-start
    finally:
        ring.close()
    assert result["tokens"] == [TRUTH[i] for i in range(3, 15)]
    value = feedback_from_result(result, first_token_s=offsets[0], last_token_s=offsets[-1], request_elapsed_s=elapsed,
        expected_recipe=decision.recipe)
    observation = p.observe(decision, value)
    assert observation["learned"]
    assert value.counters["frames_sent"] == len(ring.frames)-1
    assert value.judged_predictions == sum(row[1] for table in (result["accept_by_depth"], result["topup_accept_by_depth"]) for row in table.values())
    assert observation["decision"]["recipe"] == recipe.to_dict()
    assert result["coordinator_counters"]["replies_received"] == result["coordinator_counters"]["frames_sent"]

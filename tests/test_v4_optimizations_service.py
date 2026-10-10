"""Service integration of approved choices with real callbacks and signatures.

The model producer is explicit CPU-only test code; none of these rates are GPU
performance evidence.
"""
from concurrent.futures import ThreadPoolExecutor
import threading
import pytest

from test_v4_gateway import Pipe, backend, job
from engines.deepseek_v4.v4_request_features import build_policy
from shard.speculation_policy import SpeculationRecipe
from shard.service_queue import JobCancelled


class MeasuredPipe(Pipe):
    def coordinate(self, *args, cancel_check, expected_by_signer, strict_job_binding=False, **kwargs):
        result = super().coordinate(*args, cancel_check=cancel_check,
            expected_by_signer=expected_by_signer, strict_job_binding=strict_job_binding, **kwargs)
        count = len(result["tokens"]) - 1
        result.update(mode="pipelined", depth=kwargs.get("depth", 16), floor=kwargs.get("floor", 1),
            lazy=kwargs.get("lazy", False), receipt_sweep_s=0.0,
            accept_by_depth={}, topup_accept_by_depth={}, inflight_intervals=[],
            coordinator_counters={"accepted_predictions": 0, "proposed_predictions": 0,
                "cancel_events": 0, "speculation_cycles": 1,
                "frames_enqueued": count, "frames_sent": count, "replies_received": count,
                "frames_judged": count, "stale_replies": 0, "drained_replies": 0, "unsent_frames": 0})
        return result

    coordinate_dspark_pipelined = coordinate


def policy():
    configs = {}
    for index in range(2):
        configs[f"node-{index}"] = {"lo": index, "hi": index+1, "head": index == 0, "tail": index == 1,
            "dspark": index == 1, "dspark_loaded": index == 1, "dspark_block_size": 8 if index else None,
            "spec_depth": 16, "args": {"n_layers": 2, "max_seq_len": 8192, "dspark_block_size": 8, "n_mtp_layers": 3}}
    return build_policy({"enabled": True, "baseline_id": "baseline", "recipes": [
        {"recipe_id": "baseline", "mode": "pipelined", "depth": 16, "floor": 1, "lazy": False},
        {"recipe_id": "smaller", "mode": "pipelined", "depth": 9, "floor": 1, "lazy": False}]},
        SpeculationRecipe("baseline", "pipelined", 16, 1, False), configs, "a"*64,
        ring_id="test-ring", lease_fences={node: {"lease_id": node+"-lease", "fencing_token": 1} for node in configs})


def test_approved_policy_learns_only_after_full_signed_completion():
    producer = MeasuredPipe()
    service = backend(producer)
    service._policy = policy()  # Explicit service seam; admission itself is tested separately.
    request = job(maximum=8)
    request.payload["policy_allowed"] = True
    try:
        result = service.execute(request, request.commit, request.check_stop)
        record = result["optimizations"]["speculation_policy"]
        assert result["proof"]["verified"] is True
        assert request.tokens == list(range(65, 73))
        assert record["schema"] == "shard-speculation-feedback/1"
        assert "a" not in record.get("tenant_id", {})
        assert record["decision"]["recipe"]["depth"] == 16
    finally:
        service.close()


def test_explicit_mode_and_bad_receipts_do_not_train_policy():
    producer = MeasuredPipe(bad_receipt=True)
    service = backend(producer)
    service._policy = policy()
    request = job(maximum=8)
    request.payload["policy_allowed"] = True
    try:
        with pytest.raises(Exception, match="signed assignment"):
            service.execute(request, request.commit, request.check_stop)
        assert service._last_policy_record is None
        assert service._policy.choose(service._policy.context(tenant_id="a", context_tokens=2, max_new_tokens=8)).reason == "baseline_warmup"
    finally:
        service.close()
    producer = MeasuredPipe()
    service = backend(producer); service._policy = policy()
    request = job(maximum=8); request.payload["policy_allowed"] = False
    try:
        result = service.execute(request, request.commit, request.check_stop)
        assert "optimizations" not in result
    finally:
        service.close()


def test_request_audit_notes_are_isolated_between_parallel_rings():
    import v4_levers
    v4_levers.note("test-scope", "outside")
    ready = threading.Barrier(2)
    def run(depth):
        recipe = SpeculationRecipe("r"+str(depth), "pipelined", depth, 1, False)
        with v4_levers.request_recipe_audit(recipe.to_dict()):
            v4_levers.note("V4_SPEC_DEPTH", depth)
            ready.wait(timeout=2)
            assert v4_levers.notes()["V4_SPEC_DEPTH"] == str(depth)
            assert v4_levers._recipe_request("depth", "unknown") == str(depth)
        return depth
    with ThreadPoolExecutor(max_workers=2) as threads:
        assert sorted(threads.map(run, [8, 16])) == [8, 16]
    assert v4_levers.notes()["test-scope"] == "outside"


def test_cancel_before_first_attempt_discards_the_selected_policy_decision():
    producer = MeasuredPipe()
    service = backend(producer); service._policy = policy()
    request = job(maximum=8); request.payload["policy_allowed"] = True
    chosen = []
    original = service._policy.choose
    def choose(context):
        decision = original(context)
        chosen.append(decision)
        request.cancelled.set()
        return decision
    service._policy.choose = choose
    try:
        with pytest.raises(JobCancelled):
            service.execute(request, request.commit, request.check_stop)
        assert producer.calls == producer.connects == 0
        record = service._policy.discard(chosen[0], reason="request_cancelled")
        assert record["learned"] is False and record["reason"] == "request_cancelled"
    finally:
        service.close()


def test_replayed_attempts_do_not_teach_an_uninterrupted_decode_rate():
    producer = MeasuredPipe(fail_first=True)
    service = backend(producer); service._policy = policy()
    request = job(maximum=8); request.payload["policy_allowed"] = True
    try:
        result = service.execute(request, request.commit, request.check_stop)
        record = result["optimizations"]["speculation_policy"]
        assert producer.calls == 2 and result["proof"]["verified"]
        assert record["learned"] is False and record["reason"] == "request_replayed"
    finally:
        service.close()


def test_approved_request_override_has_truthful_audit_with_environment_on(monkeypatch):
    import v4_levers
    for name in ("V4_LAZY_DRAFT", "V4_PIPELINED_SPEC"):
        monkeypatch.setenv(name, "1")
    recipe = SpeculationRecipe("safe", "greedy")
    with v4_levers.request_recipe_audit(recipe.to_dict()):
        for name, value in (("V4_LAZY_DRAFT", False), ("V4_PIPELINED_SPEC", False),
                            ("V4_SPEC_DEPTH", 1), ("V4_REFILL_FLOOR", 1)):
            v4_levers.note(name, value)
        findings = {f.env: f for f in v4_levers.audit(v4_levers.COORDINATOR)}
        for name in ("V4_LAZY_DRAFT", "V4_PIPELINED_SPEC", "V4_SPEC_DEPTH", "V4_REFILL_FLOOR"):
            assert findings[name].verdict == "OK", findings[name]


def test_missing_sweep_measurement_keeps_valid_signed_output_without_learning():
    class UnmeasuredPipe(MeasuredPipe):
        def coordinate(self, *args, cancel_check, expected_by_signer, strict_job_binding=False, **kwargs):
            result = super().coordinate(*args, cancel_check=cancel_check,
                expected_by_signer=expected_by_signer, strict_job_binding=strict_job_binding, **kwargs)
            result.pop("receipt_sweep_s")
            return result
        coordinate_dspark_pipelined = coordinate
    producer = UnmeasuredPipe()
    service = backend(producer); service._policy = policy()
    request = job(maximum=8); request.payload["policy_allowed"] = True
    try:
        result = service.execute(request, request.commit, request.check_stop)
        assert result["proof"]["verified"] and len(result["tokens"]) == 8
        record = result["optimizations"]["speculation_policy"]
        assert record["reason"] == "invalid_measurement" and not record["learned"]
    finally:
        service.close()


def test_explicit_greedy_request_retains_its_full_context_budget():
    service = backend(MeasuredPipe()); service._policy = policy(); service.max_context = 10
    try:
        payload, count, maximum, _ = service.prepare({"messages": [{"role": "user", "content": "x"}],
            "max_tokens": 8, "shard_mode": "greedy"})
        assert count + maximum == 10 and payload["policy_allowed"] is False
    finally:
        service.close()


def test_actual_socket_tail_cannot_extend_an_approved_lookahead_contract():
    from test_v4_pipe import _ScriptedPipeRing, TRUTH, PIPE_PROMPT, VP
    ring = _ScriptedPipeRing(TRUTH, lambda q: [901] * 65)
    try:
        with pytest.raises(RuntimeError, match="approved loaded block limit"):
            VP.coordinate_dspark_pipelined(ring.pipe_a, ring.ret_b, PIPE_PROMPT, 8,
                timeout=3, depth=256, floor=256, draft_block_limit=8)
        # Only the true first decode input may have been submitted. No oversized
        # speculative burst reaches a worker's bounded context/rollback state.
        assert all(pos <= len(PIPE_PROMPT) for _, pos, _ in ring.frames)
    finally:
        ring.close()

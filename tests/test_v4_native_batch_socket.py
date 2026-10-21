"""Production-connected native request batches on actual strict CPU TCP rings."""
import time

import pytest

from shard.service_queue import JobCancelled
from test_v4_native_lane_socket import ring, job, reference
from v4_native_batch import BATCH_CONTRACT
from v4_prefill_budget import TimeBudget, SCHEMA


def request(prompt, count=3, tenant="a"):
    value = job(prompt, count, tenant)
    value.payload["numerical_contract"] = BATCH_CONTRACT
    return value


def advance(backend, jobs, checks=None):
    return backend.advance_batch(jobs, emit={j.id: j.commit for j in jobs},
        cancel_check=checks or {j.id: j.check_stop for j in jobs}, max_step_tokens=1)


@pytest.mark.parametrize("ring", ["batch"], indirect=True)
def test_real_socket_batch_has_independent_tokens_hashes_nonces_receipts_and_true_batch_counts(ring):
    backend, control = ring
    jobs = [request(range(1, 15)), request(range(31, 43), tenant="b")]
    wanted = {j.id: reference(control, j.payload["prompt_ids"], j.max_new) for j in jobs}
    pending, results = list(jobs), {}
    while pending:
        outputs = advance(backend, pending)
        for value in tuple(pending):
            if outputs[value.id] is not None:
                result = outputs[value.id]
                assert not isinstance(result, Exception)
                backend.verify_result_proof(value, result)
                assert result["tokens"] == wanted[value.id]
                assert result["numerical_contract"] == BATCH_CONTRACT
                assert all(row["nonce"] == result["proof"]["nonce"] for row in result["proof"]["receipts"])
                results[value.id] = result
                backend.release_batch_job(value.id)
                pending.remove(value)
    assert results[jobs[0].id]["proof"]["nonce"] != results[jobs[1].id]["proof"]["nonce"]
    status = backend._native_runner.status()
    assert status["compute"] == "native_equal_frontier_batch"
    assert status["pipeline"]["batch_forward_cohorts"] >= 12
    assert status["pipeline"]["scalar_forward_cohorts"] >= 1
    assert status["pipeline"]["sent_cohorts"] == status["pipeline"]["received_cohorts"]
    assert status["cohort_timing_samples"] > 0
    exported = backend.export_native_prefill_samples(quota_ms=100)
    assert not exported["ready"]  # Each exact context has fewer than five actual samples.


@pytest.mark.parametrize("ring", ["batch"], indirect=True)
def test_mixed_frontiers_pipeline_real_frames_and_cancel_after_submit_drains_then_reuses_slot(ring, monkeypatch):
    backend, _ = ring
    first, cancelled = request(range(1, 16)), request(range(31, 46), tenant="b")
    for _ in range(3):
        assert advance(backend, [first])[first.id] is None
    # Different live frontiers create two disjoint real in-flight cohorts.
    stopped, receives = False, 0
    original = backend.vp.recv_msg
    def receive(channel):
        nonlocal stopped, receives
        value = original(channel)
        if isinstance(value, dict) and value.get("op") == "native_lane_ack" and value.get("acks"):
            # The first ack while opening the second job is admission; setting
            # the flag only after both jobs exist targets an already sent step.
            if len(backend._native_runner.jobs) == 2:
                receives += 1
                stopped = True
        return value
    monkeypatch.setattr(backend.vp, "recv_msg", receive)
    def stop():
        if stopped:
            raise JobCancelled("cancelled while ring work is in flight")
    results = advance(backend, [first, cancelled], {first.id: first.check_stop, cancelled.id: stop})
    assert results[first.id] is None and isinstance(results[cancelled.id], JobCancelled)
    assert receives == 2  # Both already submitted cohorts drained before return.
    native = backend._native_runner
    assert native.pipeline_stats["peak_inflight_cohorts"] == 2
    assert native.jobs[first.id]["pos"] == 4 and native.jobs[cancelled.id]["pos"] == 1
    backend.release_batch_job(cancelled.id)
    assert cancelled.id not in native.jobs and first.id in native.jobs
    monkeypatch.setattr(backend.vp, "recv_msg", original)
    successor = request([51, 52, 53], 1, "c")
    while True:
        result = advance(backend, [successor])[successor.id]
        if result is not None:
            backend.verify_result_proof(successor, result)
            break
    assert native.jobs[first.id]["pos"] == 4
    backend.release_batch_job(successor.id)
    backend.release_batch_job(first.id)
    assert not native.jobs


@pytest.mark.parametrize("ring", ["batch"], indirect=True)
def test_measured_quantum_runs_real_fair_prefill_rounds_without_extra_decode_output(ring):
    backend, _ = ring
    native = backend._native_runner
    native.prefill_quantum = 4
    binding = {key: native.contract[key] for key in ("numerical_contract", "cohort_id", "source_sha256",
               "runtime_config_sha256", "max_context_tokens", "max_jobs")}
    native.prefill_budget = TimeBudget({"schema": SCHEMA, "binding": binding,
        "measured_at": time.time(), "ttl_s": 300, "quota_ms": 100,
        "samples": [{"context_tokens": context, "batch_jobs": 2, "quantum_tokens": 4,
                     "elapsed_ms": [1, 1.1, 1.2, 1.3, 1.4]} for context in (0, 1, 4)]}, binding)
    long, short = request(range(1, 14)), request(range(31, 37), tenant="b")
    assert all(value is None for value in advance(backend, [long, short]).values())
    assert native.jobs[long.id]["pos"] == 4
    assert native.jobs[short.id]["pos"] == 4
    assert not long.tokens and not short.tokens
    result = advance(backend, [short, long])
    assert result[short.id] is None
    assert not short.tokens and not long.tokens
    assert native.jobs[short.id]["pos"] == native.jobs[long.id]["pos"] == 5
    # The short prompt has no complete pure-prefill quantum left, so mixed
    # first-output/decode work is one round even with a matching cost row.
    advance(backend, [short, long])
    assert len(short.tokens) == 1 and not long.tokens
    assert native.jobs[long.id]["pos"] == 6
    backend.release_batch_job(short.id); backend.release_batch_job(long.id)
    # Losing one lane after the first verified drain ends the quantum. The
    # survivor cannot reuse the previous B=2 cost to run further B=1 rounds.
    survivor, stopped = request(range(1, 14), tenant="c"), request(range(21, 34), tenant="d")
    checks = 0
    def cancel_after_submit():
        nonlocal checks
        checks += 1
        if checks >= 3:
            raise JobCancelled("stop after this real cohort drains")
    output = advance(backend, [survivor, stopped],
        {survivor.id: survivor.check_stop, stopped.id: cancel_after_submit})
    assert output[survivor.id] is None and isinstance(output[stopped.id], JobCancelled)
    assert native.jobs[survivor.id]["pos"] == native.jobs[stopped.id]["pos"] == 1
    backend.release_batch_job(survivor.id); backend.release_batch_job(stopped.id)
    # Even with a measured row for max(frontier)=1 and nominal B=2, two ragged
    # B=1 cohorts are a different shape and use one round.
    ahead, behind = request(range(1, 14), tenant="e"), request(range(21, 34), tenant="f")
    advance(backend, [ahead])
    advance(backend, [ahead, behind])
    assert native.jobs[ahead.id]["pos"] == 2 and native.jobs[behind.id]["pos"] == 1
    assert not ahead.tokens and not behind.tokens
    backend.release_batch_job(ahead.id); backend.release_batch_job(behind.id)


@pytest.mark.parametrize("ring", ["batch"], indirect=True)
def test_prefill_calibration_export_requires_five_actual_bound_cohort_observations(ring):
    backend, _ = ring
    for iteration in range(5):
        jobs = [request(range(1, 9), tenant=f"{iteration}-a"),
                request(range(21, 29), tenant=f"{iteration}-b")]
        for _ in range(4):
            assert all(value is None for value in advance(backend, jobs).values())
        for value in jobs:
            backend.release_batch_job(value.id)
    exported = backend.export_native_prefill_samples(quota_ms=100)
    assert exported["ready"] and exported["gpu_qualified"] is False
    config = exported["calibration"]
    assert config["binding"]["numerical_contract"] == BATCH_CONTRACT
    assert config["measurement_scope"] == "observed_consecutive_token_round_span_including_cpu_gaps"
    shape = next(value for value in config["samples"] if value["context_tokens"] == 0 and value["quantum_tokens"] == 1)
    assert (shape["context_tokens"], shape["batch_jobs"], shape["quantum_tokens"]) == (0, 2, 1)
    assert len(shape["elapsed_ms"]) == 5 and all(value > 0 for value in shape["elapsed_ms"])
    spans = next(value for value in config["samples"] if value["context_tokens"] == 0 and value["quantum_tokens"] == 4)
    assert len(spans["elapsed_ms"]) == 5
    assert all(span >= token for span, token in zip(spans["elapsed_ms"], shape["elapsed_ms"]))
    assert exported["single_advance_quantum_observed"] is False

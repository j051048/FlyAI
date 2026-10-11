"""Real queue/leases/Ed25519 evidence with a CPU isolated-lane producer."""
import threading
import time

import pytest

from shard.continuous_batch import contract_digest
from shard.receipt import ReceiptSigner, gen_key, pub_b64, verify_coverage
from shard.service_queue import JobCancelled, JobExpired
from test_ring_pool import Backend, add_ready, COHORT
from test_ring_router import service, body, wait
from shard.ring_pool import RingPool, RingState


class BatchedBackend(Backend):
    def __init__(self, *, hold=False, overrun=False):
        super().__init__()
        self.keys = [gen_key(), gen_key()]
        self.spans = {pub_b64(key): (i, i + 1) for i, key in enumerate(self.keys)}
        self.steps, self.live = [], {}
        self.hold, self.overrun = hold, overrun
        self.entered, self.proceed = threading.Event(), threading.Event()
        if not hold:
            self.proceed.set()

    def batch_contract(self):
        return {"schema": "shard-token-batch/1", "model_id": self.model_id, "cohort_id": COHORT,
            "source_sha256": "a" * 64, "runtime_config_sha256": "b" * 64,
            "numerical_contract": "cpu-test-identity/1", "max_jobs": 4, "max_step_tokens": 2,
            "max_prompt_tokens": 16, "max_new_tokens": 16, "max_context_tokens": 32,
            "per_job_state_bytes": 10, "reserved_state_bytes": 40,
            "prefill_chunk_tokens": 4, "cancellation": "per_job"}

    def warmup(self, **kwargs):
        return {**super().warmup(**kwargs), "batch_contract_sha256": contract_digest(self.batch_contract())}

    def advance_batch(self, jobs, *, emit, cancel_check, max_step_tokens):
        self.entered.set()
        assert self.proceed.wait(2)
        self.steps.append([job.id for job in jobs])
        out = {}
        for job in jobs:
            self.live[job.id] = job.payload["label"]
            try:
                cancel_check[job.id]()
                count = max_step_tokens + 1 if self.overrun else min(max_step_tokens, job.max_new - len(job.tokens))
                for _ in range(count):
                    emit[job.id](65 + len(job.tokens))
                out[job.id] = None
                if len(job.tokens) == job.max_new:
                    receipts = []
                    for index, key in enumerate(self.keys):
                        signer = ReceiptSigner(key, "cpu-test-ring", job.id, index, index + 1, nonce=job.id)
                        for token in job.tokens:
                            signer.observe(f"{job.id}:{index}:{token}".encode(), f"{job.id}:{index + 1}:{token}".encode())
                        receipts.append(signer.finalize())
                    out[job.id] = {"tokens": job.checkpoint(), "text": self.decode(job.tokens),
                        "finish_reason": "length", "recovery": {"strategy": "cpu-isolated-lane"},
                        "proof": {"verified": True, "scope": "complete_final_attempt", "receipts": receipts}}
            except (JobCancelled, JobExpired) as error:
                out[job.id] = error
        time.sleep(.005)
        return out

    def verify_result_proof(self, job, value):
        verify_coverage(value["proof"]["receipts"], self.layers, self.spans,
                        expected_nonce=job.id, check_chain=True)
        assert all(row["job_id"] == job.id for row in value["proof"]["receipts"])
        return True

    def release_batch_job(self, job_id):
        self.live.pop(job_id, None)


def test_new_short_job_joins_an_active_token_batch():
    pool = RingPool(); producer = BatchedBackend(hold=True); add_ready(pool, backend=producer)
    queue = service(pool, batching={"enabled": True, "max_jobs": 2, "max_step_tokens": 1})
    queue.start()
    try:
        first = queue.submit_request("a", body("long", max_tokens=12))[0]
        assert producer.entered.wait(1)
        second = queue.submit_request("b", body("short", max_tokens=2))[0]
        producer.proceed.set()
        wait(first); wait(second)
        assert first.state == second.state == "completed"
        assert any(len(step) == 2 for step in producer.steps)
        assert second.finished_mono < first.finished_mono
        assert first.tokens == list(range(65, 77)) and second.tokens == [65, 66]
        assert not producer.live
        assert queue.metrics()["mode"] == "qualified_token_batches"
    finally:
        producer.proceed.set(); queue.shutdown()


def test_one_lane_cancel_does_not_abort_other_job_or_ring():
    pool = RingPool(); producer = BatchedBackend(hold=True); ring = add_ready(pool, backend=producer)
    queue = service(pool, batching={"enabled": True})
    first = queue.submit_request("a", body("cancel", max_tokens=5))[0]
    other = queue.submit_request("b", body("continue", max_tokens=4))[0]
    queue.start()
    try:
        assert producer.entered.wait(1)
        queue.cancel("a", first.id); producer.proceed.set()
        wait(first); wait(other)
        assert first.state == "cancelled" and other.state == "completed"
        assert ring.state == RingState.READY and not producer.live
    finally:
        producer.proceed.set(); queue.shutdown()


def test_serial_backend_cannot_enable_batching_by_configuration_only():
    pool = RingPool(); add_ready(pool)
    try:
        with pytest.raises(ValueError, match="qualified"):
            service(pool, batching={"enabled": True})
    finally:
        pool.shutdown()


def test_contract_change_after_warmup_cannot_execute_jobs():
    pool = RingPool(); producer = BatchedBackend(); ring = add_ready(pool, backend=producer)
    original = producer.batch_contract
    producer.batch_contract = lambda: {**original(), "runtime_config_sha256": "c" * 64}
    try:
        with pytest.raises(ValueError, match="warmup"):
            service(pool, batching={"enabled": True})
        assert ring.bound_jobs == 0
    finally:
        pool.shutdown()


def test_backend_cannot_exceed_its_token_step_budget():
    pool = RingPool(); producer = BatchedBackend(overrun=True); ring = add_ready(pool, backend=producer)
    queue = service(pool, batching={"enabled": True}); queue.start()
    try:
        request = queue.submit_request("a", body(max_tokens=6))[0]
        wait(request)
        assert request.state == "failed" and len(request.tokens) == 1
        assert ring.state == RingState.FAILED
        assert not producer.live and not ring.guards[0].works
    finally:
        queue.shutdown()


def test_shortest_with_aging_preserves_tenant_fairness_and_starvation_bound():
    pool = RingPool(); ring = add_ready(pool)
    queue = service(pool, scheduling={"policy": "shortest_with_aging", "aging_s": 100})
    long = queue.submit_request("a", body("long", max_tokens=6))[0]
    short = queue.submit_request("a", body("short", max_tokens=2))[0]
    other = queue.submit_request("b", body("other", max_tokens=3))[0]
    queue.start()
    try:
        for job in (long, short, other):
            wait(job)
        assert ring.backend.order == ["short", "other", "long"]
    finally:
        queue.shutdown()

"""Cross-ring concurrency, global admission, tenant fairness and fixed bindings."""
import threading
import time

import pytest

from shard.ring_pool import RingPool, RingState
from shard.ring_router import RingRouter, MultiRingQueue
from shard.service_queue import TenantLimits, AdmissionError
from test_ring_pool import Backend, Guard, add_ready, COHORT, OTHER


def body(label="request", **options):
    return {"model": "default", "messages": [{"role": "user", "content": label}],
            "max_tokens": 2, **options}


def service(pool, **options):
    pool.set_alias("default", "test-model", COHORT)
    return MultiRingQueue(RingRouter(pool, default_model="default"),
                         {"a": TenantLimits(100, 1000, 10000), "b": TenantLimits(100, 1000, 10000)}, **options)


def wait(job):
    end = time.monotonic() + 3
    with job.changed:
        while not job.terminal and time.monotonic() < end:
            job.changed.wait(0.01)
    assert job.terminal, job.snapshot()


def test_two_rings_execute_concurrently_but_each_ring_is_serial():
    pool = RingPool()
    a = add_ready(pool, "a", backend=Backend(block=True))
    b = add_ready(pool, "b", backend=Backend(block=True))
    queue = service(pool); queue.start()
    first = queue.submit_request("a", body("first"))[0]
    second = queue.submit_request("b", body("second"))[0]
    assert a.backend.started.wait(1) and b.backend.started.wait(1)
    assert queue.metrics()["running"] == 2
    third = queue.submit_request("a", body("third"))[0]
    assert third.state == "queued" and first.ring_binding.backend.calls == 1
    a.backend.release.set(); b.backend.release.set()
    for job in (first, second, third):
        wait(job); assert job.state == "completed"
    assert first.ring_binding.ring is a and second.ring_binding.ring is b
    assert queue.metrics()["mode"] == "parallel_serial_rings"
    assert queue.shutdown()


def test_global_tenant_active_quota_cannot_be_bypassed_by_multiple_rings():
    pool = RingPool(); add_ready(pool, "a"); add_ready(pool, "b")
    pool.set_alias("default", "test-model", COHORT)
    queue = MultiRingQueue(RingRouter(pool, default_model="default"), {"a": TenantLimits(1, 100, 1000)})
    job = queue.submit_request("a", body())[0]
    with pytest.raises(AdmissionError):
        queue.submit_request("a", body("overflow"))
    assert sum(r.bound_jobs for r in pool.rings()) == 1
    queue.cancel("a", job.id); assert not any(r.bound_jobs for r in pool.rings())
    queue.shutdown()


def test_region_preference_then_available_cross_region_and_measured_backlog():
    pool = RingPool(); local = add_ready(pool, "local", region="east"); remote = add_ready(pool, "remote", region="west")
    queue = service(pool)
    first = queue.submit_request("a", body(shard_region="east"))[0]
    assert first.ring_binding.ring is local
    local.guards[0].live = False
    assert local.backend.release.wait(1)  # asynchronous guard revocation was observed
    second = queue.submit_request("b", body("cross", shard_region="east"))[0]
    assert second.ring_binding.ring is remote
    assert second.ring_binding.estimated_seconds == 4
    queue.shutdown()


def test_per_ring_tenant_round_robin_preserves_fifo():
    pool = RingPool(); ring = add_ready(pool)
    queue = service(pool)
    jobs = [queue.submit_request("a", body("a1"))[0], queue.submit_request("a", body("a2"))[0],
            queue.submit_request("a", body("a3"))[0], queue.submit_request("b", body("b1"))[0],
            queue.submit_request("b", body("b2"))[0]]
    queue.start()
    for job in jobs:
        wait(job)
    assert ring.backend.order == ["a1", "b1", "a2", "b2", "a3"]
    queue.shutdown()


def test_alias_cutover_and_idempotency_keep_old_request_backend_and_decoder():
    pool = RingPool(); old = add_ready(pool, "old", backend=Backend(offset=0, block=True))
    queue = service(pool); queue.start()
    first = queue.submit_request("a", body(), idempotency_key="stable")[0]
    assert old.backend.started.wait(1)
    new = add_ready(pool, "new", cohort=OTHER, backend=Backend(offset=10))
    pool.set_alias("default", "test-model", OTHER)
    same, fresh = queue.submit_request("a", body(stream=True), idempotency_key="stable")
    assert same is first and fresh is False and same.bound_backend is old.backend
    with pytest.raises(AdmissionError) as exc:
        queue.submit_request("a", body("changed"), idempotency_key="stable")
    assert exc.value.status == 409
    second = queue.submit_request("b", body("new"))[0]
    wait(second)
    assert second.result["text"] == "KL" and second.ring_binding.ring is new
    old.backend.release.set(); wait(first)
    assert first.result["text"] == "AB" and first.ring_binding.ring is old
    assert first.served_cohort == COHORT and second.served_cohort == OTHER
    queue.shutdown()


def test_drain_honors_all_already_bound_jobs_and_new_requests_use_other_ring():
    pool = RingPool(); old = add_ready(pool, "old", backend=Backend(block=True))
    queue = service(pool); queue.start()
    first = queue.submit_request("a", body())[0]
    assert old.backend.started.wait(1)
    queued = queue.submit_request("a", body("already admitted"))[0]
    pool.drain("old")
    assert old.state == RingState.DRAINING and not old.backend.closed
    new = add_ready(pool, "new")
    following = queue.submit_request("b", body("after drain"))[0]
    assert following.ring_binding.ring is new
    old.backend.release.set()
    for job in (first, queued, following):
        wait(job); assert job.state == "completed"
    assert old.state == RingState.STOPPED and old.backend.closed
    queue.shutdown()


def test_failed_ring_job_keeps_prefix_and_never_silently_moves_to_healthy_ring():
    class Failure(Backend):
        def execute(self, job, emit, check):
            self.started.set(); emit(65); self.healthy = False
            raise OSError("explicit ring failure")
    pool = RingPool(); failed = add_ready(pool, "a", backend=Failure()); healthy = add_ready(pool, "b")
    queue = service(pool); queue.start()
    job = queue.submit_request("a", body())[0]; wait(job)
    assert job.state == "failed" and job.tokens == [65]
    assert job.ring_binding.ring is failed and healthy.backend.calls == 0
    assert failed.state == RingState.FAILED
    following = queue.submit_request("b", body("healthy"))[0]; wait(following)
    assert following.state == "completed" and following.ring_binding.ring is healthy
    queue.shutdown()


def test_unknown_or_incompatible_cohort_and_shutdown_fail_closed():
    pool = RingPool(); add_ready(pool); queue = service(pool)
    with pytest.raises(AdmissionError) as exc:
        queue.submit_request("a", body(shard_cohort=OTHER))
    assert exc.value.status == 503
    queue.shutdown()
    with pytest.raises(AdmissionError) as exc:
        queue.submit_request("a", body())
    assert exc.value.status == 503


def test_expired_queued_job_releases_binding_before_drain_stops_ring():
    pool = RingPool(); ring = add_ready(pool)
    now = [time.monotonic()]
    queue = service(pool, clock=lambda: now[0])
    job = queue.submit_request("a", body(timeout_s=0.01))[0]
    pool.drain(ring.ring_id)
    now[0] += 1
    queue.start(); wait(job)
    assert job.state == "expired" and ring.state == RingState.STOPPED
    assert ring.backend.calls == 0
    queue.shutdown()


@pytest.mark.parametrize("key", ["", [], True, "x" * 129])
def test_bad_idempotency_keys_never_reserve_a_ring(key):
    pool = RingPool(); add_ready(pool); queue = service(pool)
    with pytest.raises(AdmissionError) as exc:
        queue.submit_request("a", body(), idempotency_key=key)
    assert exc.value.status == 400 and pool.rings()[0].bound_jobs == 0
    queue.shutdown()


def test_stopped_workers_are_retired_and_old_job_decoder_survives_pool_history_eviction():
    pool = RingPool(max_rings=1, max_history=1)
    add_ready(pool, "first")
    queue = service(pool); queue.start()
    original = queue.submit_request("a", body())[0]; wait(original)
    pool.drain("first")
    second_ring = add_ready(pool, "second", backend=Backend(offset=10))
    second = queue.submit_request("b", body("second"))[0]; wait(second)
    pool.drain("second")
    assert original.ring_binding.ring not in pool.rings()
    assert original.bound_backend.decode(original.tokens) == "AB"
    assert second.bound_backend.decode(second.tokens) == "KL"
    end = time.monotonic() + 2
    with queue._condition:
        while queue._ring_workers and time.monotonic() < end:
            queue._condition.wait(0.1)
    assert queue.metrics()["ring_workers"] == 0 and not queue._running_by_ring
    assert queue.shutdown()

"""Tenant admission/fairness and job lifecycle; no HTTP, model or GPU required."""
import threading
import time

import pytest

from shard.service_queue import ServiceQueue, TenantLimits, AdmissionError, JobCancelled


class Backend:
    def __init__(self):
        self.order = []

    def execute(self, job, emit, check):
        check()
        self.order.append(job.payload["label"])
        emit(7)
        return {"tokens": job.checkpoint()}


def queue(backend=None, **options):
    limits = TenantLimits(max_active=100, requests_per_minute=10000, tokens_per_minute=100000)
    return ServiceQueue(backend or Backend(), {"a": limits, "b": limits, "c": limits}, **options)


def submit(service, tenant="a", label="job", **options):
    return service.submit(tenant, {"label": label}, prompt_tokens=1, max_new=1, **options)[0]


def wait_terminal(job, timeout=3):
    end = time.monotonic() + timeout
    with job.changed:
        while not job.terminal and time.monotonic() < end:
            job.changed.wait(0.02)
    assert job.terminal, job.snapshot()


def test_round_robin_tenants_keep_fifo_without_claiming_continuous_batch():
    backend = Backend()
    service = queue(backend)
    jobs = [submit(service, "a", "a1"), submit(service, "a", "a2"), submit(service, "a", "a3"),
            submit(service, "b", "b1"), submit(service, "b", "b2"), submit(service, "c", "c1")]
    service.start()
    try:
        for job in jobs:
            wait_terminal(job)
        assert backend.order == ["a1", "b1", "c1", "a2", "b2", "a3"]
        assert service.metrics()["mode"] == "serial_one_ring"
        assert all(job.payload == {} for job in jobs)
    finally:
        assert service.shutdown()


def test_idempotent_retries_do_not_consume_extra_quota_or_rerun_and_conflicts_fail():
    service = queue(max_queue=1)
    original, fresh = service.submit("a", {"label": "x"}, prompt_tokens=1, max_new=1, idempotency_key="request-1")
    same, fresh_again = service.submit("a", {"label": "x"}, prompt_tokens=1, max_new=1, idempotency_key="request-1")
    assert original is same and fresh is True and fresh_again is False
    assert service.metrics()["counts"]["submitted"] == 1
    with pytest.raises(AdmissionError) as exc:
        service.submit("a", {"label": "changed"}, prompt_tokens=1, max_new=1, idempotency_key="request-1")
    assert exc.value.status == 409
    assert service.get("b", original.id) is None and service.cancel("b", original.id) is False
    service.shutdown()


def test_queue_and_reserved_token_bounds_are_admission_limits():
    service = queue(max_queue=2, max_queued_tokens=3)
    submit(service)
    with pytest.raises(AdmissionError):
        submit(service, "b")
    assert service.metrics()["queued_reserved_tokens"] == 2
    service.shutdown()


def test_tenant_inflight_and_rolling_minute_request_and_token_quotas():
    now = [100.0]
    service = ServiceQueue(Backend(), {"a": TenantLimits(2, 2, 4)}, clock=lambda: now[0])
    first = submit(service)
    second = submit(service, label="other")
    with pytest.raises(AdmissionError):
        submit(service, label="quota")
    service.cancel("a", first.id); service.cancel("a", second.id)
    with pytest.raises(AdmissionError):
        submit(service, label="minute-still-reserved")
    now[0] += 61
    submit(service, label="new-minute")
    service.shutdown()


def test_cancel_flood_does_not_leave_unbounded_queue_tombstones():
    service = queue(max_queue=1, max_history=4)
    for _ in range(200):
        job = submit(service)
        service.cancel("a", job.id)
    assert len(service._queues["a"]) == 0 and len(service._round) == 0
    assert service.metrics()["retained_jobs"] <= 5
    assert service.metrics()["queued"] == 0
    service.shutdown()


def test_queued_deadline_expires_while_another_job_is_running():
    class Blocker(Backend):
        def __init__(self):
            super().__init__()
            self.started, self.release = threading.Event(), threading.Event()

        def execute(self, job, emit, check):
            self.started.set()
            while not self.release.wait(0.01):
                check()
            return super().execute(job, emit, check)

    backend = Blocker()
    service = queue(backend)
    service.start()
    running = submit(service)
    assert backend.started.wait(1)
    queued = submit(service, "b", timeout_s=0.05)
    try:
        wait_terminal(queued)
        assert queued.state == "expired" and queued.tokens == []
        assert running.state == "running"
        backend.release.set(); wait_terminal(running)
    finally:
        service.shutdown()


def test_running_cancel_retains_exact_observed_frontier_and_releases_capacity():
    class Waiter(Backend):
        def __init__(self):
            self.committed = threading.Event()

        def execute(self, job, emit, check):
            emit(13); self.committed.set()
            while True:
                check(); time.sleep(0.01)

    backend = Waiter()
    service = queue(backend)
    service.start(); job = submit(service)
    assert backend.committed.wait(1)
    service.cancel("a", job.id)
    wait_terminal(job)
    assert job.state == "cancelled" and job.tokens == [13]
    assert service.metrics("a")["tenant_active"] == 0
    service.shutdown()


def test_callback_frontier_must_match_returned_result():
    class Wrong(Backend):
        def execute(self, job, emit, check):
            emit(7)
            return {"tokens": [8]}
    service = queue(Wrong()); service.start(); job = submit(service)
    wait_terminal(job)
    assert job.state == "failed" and "frontier" in job.error
    service.shutdown()


def test_shutdown_cancels_running_and_queued_and_refuses_new_admission():
    class Waiter(Backend):
        def execute(self, job, emit, check):
            while True:
                check(); time.sleep(0.01)
    service = queue(Waiter()); service.start()
    jobs = [submit(service), submit(service, "b")]
    assert service.shutdown(timeout=1)
    assert all(job.state == "cancelled" for job in jobs)
    with pytest.raises(AdmissionError) as exc:
        submit(service)
    assert exc.value.status == 503


def test_cancellation_winning_terminal_lock_cannot_be_overwritten_by_completed():
    class Ready(Backend):
        def __init__(self):
            self.started, self.release = threading.Event(), threading.Event()
        def execute(self, job, emit, check):
            emit(7); self.started.set(); self.release.wait(1)
            return {"tokens": job.checkpoint()}
    backend = Ready()
    service = queue(backend); service.start(); job = submit(service)
    assert backend.started.wait(1)
    checked = threading.Event()
    original = job.check_stop
    def after_result(now=None):
        original(now); checked.set()
    job.check_stop = after_result
    with service._condition:
        backend.release.set()
        assert checked.wait(1)
        service.cancel("a", job.id)
    wait_terminal(job)
    assert job.state == "cancelled"
    service.shutdown()


def test_disconnect_cancel_checks_current_subscriber_count_under_queue_lock():
    service = queue()
    job = submit(service)
    with job.changed:
        job.clients = 1  # an idempotent reconnect acquired the job before disconnect cleanup
    assert service.cancel_if_unobserved("a", job.id) is False
    assert not job.cancelled.is_set()
    with job.changed:
        job.clients = 0
    assert service.cancel_if_unobserved("a", job.id) is True
    assert job.state == "cancelled"
    service.shutdown()


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_invalid_deadline_never_enters_queue(value):
    service = queue()
    with pytest.raises(AdmissionError):
        submit(service, timeout_s=value)
    service.shutdown()

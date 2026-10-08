"""Lease-bound lifecycle tests; explicit CPU backend/guard, no model or networking."""
from types import SimpleNamespace
import threading
import time

import pytest

from shard.ring_pool import RingPool, RingState, RingUnavailable
from shard.service_queue import Job
from shard.leases import LeaseResources, LeaseLedger, LeaseRequest, LeaseConflict

COHORT = "a" * 64
OTHER = "b" * 64


class Guard:
    def __init__(self, ring_id, gpu_uuid, cohort=COHORT):
        self.ring_id, self.gpu_uuid, self.model_cohort_sha256 = ring_id, gpu_uuid, cohort
        self.node_id, self.memory_domain_id = "node-" + gpu_uuid, "host-" + gpu_uuid
        self.resources = LeaseResources(100, 100, 50)
        self.expires_at = time.time() + 3600
        self.live = True
        self.works = set()
        self.begin_failure = False
        self.checks = 0
    def assert_live(self):
        self.checks += 1
        if not self.live or self.expires_at <= time.time():
            raise RingUnavailable("node fence is invalid")
        return SimpleNamespace(ring_id=self.ring_id, model_cohort_sha256=self.model_cohort_sha256,
                               expires_at=self.expires_at)
    def begin_work(self, work_id):
        self.assert_live()
        if self.begin_failure:
            raise RingUnavailable("begin work rejected")
        self.works.add(work_id)
        return work_id
    def end_work(self, work):
        self.works.discard(work)


class Backend:
    model_id, layers = "test-model", 2
    def __init__(self, *, offset=0, block=False, forged=False):
        self.offset, self.block, self.forged = offset, block, forged
        self.healthy, self.closed = False, False
        self.started, self.release = threading.Event(), threading.Event()
        self.calls, self.order = 0, []
    def warmup(self, timeout_s=300):
        self.healthy = True
        return {"proof_verified": not self.forged, "committed_tokens": 2}
    def ready(self):
        return self.healthy and not self.closed, "fake_established_ring"
    def abort(self):
        self.healthy = False; self.closed = True; self.release.set()
    close = abort
    def prepare(self, body):
        assert body["model"] == self.model_id
        maximum = body.get("max_tokens", 2)
        return {"label": body["messages"][0]["content"]}, 2, maximum, body.get("timeout_s", 3)
    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)
    def execute(self, job, emit, check):
        self.calls += 1; self.order.append(job.payload["label"])
        self.started.set()
        while self.block and not self.release.wait(0.005):
            check()
        check()
        for index in range(job.max_new):
            check(); emit(65 + self.offset + index)
        return {"tokens": job.checkpoint(), "text": self.decode(job.checkpoint()),
                "finish_reason": "length", "proof": {"verified": True, "scope": "complete_final_attempt"},
                "recovery": {"strategy": "test"}}


def add_ready(pool, ring_id="ring-a", *, cohort=COHORT, gpu=None, region=None, backend=None, guards=None):
    backend = backend or Backend()
    gpu = gpu or "GPU-" + ring_id
    ring = pool.add(ring_id, backend, model_id=backend.model_id, cohort_id=cohort,
                    gpu_uuids=[gpu], region=region)
    guards = guards or [Guard(ring_id, gpu, cohort)]
    pool.reserve(ring_id, guards); pool.mark_loading(ring_id); pool.warmup(ring_id)
    ring.seconds_per_token = 1.0  # deterministic measured-cost stand-in, never production evidence
    return ring


def make_job():
    now = time.monotonic()
    return Job("tenant", {"label": "test"}, 2, 2, now + 3, "fingerprint", None, now, state="running")


def test_ready_requires_full_lifecycle_leases_and_signed_warmup():
    pool = RingPool(); backend = Backend()
    ring = pool.add("a", backend, model_id=backend.model_id, cohort_id=COHORT, gpu_uuids=["GPU-a"])
    assert ring.state == RingState.PLANNED and not pool.ready()[0]
    with pytest.raises(RingUnavailable):
        pool.mark_loading("a")
    with pytest.raises(RingUnavailable):
        pool.reserve("a", [])
    pool.reserve("a", [Guard("a", "GPU-a")]); assert ring.state == RingState.RESERVED
    pool.mark_loading("a"); assert ring.state == RingState.LOADING
    assert not pool.ready()[0]
    pool.warmup("a"); assert ring.state == RingState.READY and pool.ready()[0]
    assert not ring.guards[0].works
    pool.shutdown(); assert ring.state == RingState.STOPPED


@pytest.mark.parametrize("wrong", ["ring", "cohort", "coverage"])
def test_node_guard_binding_is_not_a_caller_ready_boolean(wrong):
    pool = RingPool(); backend = Backend()
    pool.add("a", backend, model_id=backend.model_id, cohort_id=COHORT, gpu_uuids=["GPU-a"])
    guard = Guard("else" if wrong == "ring" else "a", "else" if wrong == "coverage" else "GPU-a",
                  OTHER if wrong == "cohort" else COHORT)
    with pytest.raises(RingUnavailable):
        pool.reserve("a", [guard])
    pool.shutdown()


def test_invalid_cohort_model_or_duplicate_physical_gpu_rejected():
    pool = RingPool()
    with pytest.raises(ValueError, match="SHA256"):
        pool.add("a", Backend(), model_id="test-model", cohort_id="version-label", gpu_uuids=["GPU"])
    with pytest.raises(ValueError, match="backend model"):
        pool.add("a", Backend(), model_id="different", cohort_id=COHORT, gpu_uuids=["GPU"])
    with pytest.raises(ValueError, match="distinct"):
        pool.add("a", Backend(), model_id="test-model", cohort_id=COHORT, gpu_uuids=["GPU", "GPU"])


def test_join_never_rewrites_live_ring_and_duplicate_gpu_remains_blocked_while_draining():
    pool = RingPool(); old = add_ready(pool, "old", gpu="GPU")
    binding = pool.acquire("test-model", COHORT, estimated_tokens=2)
    new = pool.add("new", Backend(), model_id="test-model", cohort_id=COHORT, gpu_uuids=["GPU"])
    assert old.state == RingState.READY and binding.backend is old.backend
    pool.drain("old"); assert old.state == RingState.DRAINING and not old.backend.closed
    with pytest.raises(RingUnavailable, match="already"):
        pool.reserve("new", [Guard("new", "GPU")])
    pool.release(binding); assert old.state == RingState.STOPPED and old.backend.closed
    pool.reserve("new", [Guard("new", "GPU")]); assert new.state == RingState.RESERVED
    pool.shutdown()


def test_forged_warmup_and_dead_established_connection_clear_readiness():
    pool = RingPool()
    with pytest.raises(RingUnavailable, match="signed"):
        add_ready(pool, "forged", backend=Backend(forged=True))
    assert pool.rings()[0].state == RingState.FAILED
    good = add_ready(pool, "good")
    good.backend.healthy = False
    assert not pool.ready()[0]
    assert good.state == RingState.WARMING and not good.signed_warmup
    pool.shutdown()


def test_only_ready_version_can_receive_alias_and_old_binding_keeps_backend():
    pool = RingPool(); old = add_ready(pool, "old")
    pool.set_alias("default", "test-model", COHORT)
    bound = pool.acquire(*pool.resolve("default"), estimated_tokens=2)
    with pytest.raises(RingUnavailable):
        pool.set_alias("default", "test-model", OTHER)
    new = add_ready(pool, "new", cohort=OTHER)
    pool.set_alias("default", "test-model", OTHER)
    assert pool.resolve("default") == ("test-model", OTHER)
    assert bound.backend is old.backend and bound.backend is not new.backend
    pool.release(bound); pool.shutdown()


def test_lease_revocation_interrupts_blocked_backend_without_releasing_active_work_early():
    pool = RingPool(); ring = add_ready(pool, backend=Backend(block=True))
    binding = pool.acquire("test-model", COHORT, estimated_tokens=2)
    job, errors = make_job(), []
    def run():
        try:
            pool.execute(binding, job, job.commit, job.check_stop)
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=run); thread.start()
    assert ring.backend.started.wait(1) and ring.guards[0].works
    ring.guards[0].live = False
    thread.join(2)
    assert not thread.is_alive() and errors and ring.backend.closed
    assert not ring.guards[0].works and job.tokens == []
    pool.release(binding); pool.shutdown()


def test_partial_begin_failure_unwinds_previously_acquired_node_work():
    pool = RingPool(); backend = Backend()
    ring = pool.add("a", backend, model_id="test-model", cohort_id=COHORT, gpu_uuids=["GPU-a", "GPU-b"])
    guards = [Guard("a", "GPU-a"), Guard("a", "GPU-b")]
    pool.reserve("a", guards); pool.mark_loading("a")
    guards[1].begin_failure = True
    with pytest.raises(RingUnavailable, match="begin"):
        pool.warmup("a")
    assert not guards[0].works and ring.local_work == 0
    assert ring.state == RingState.FAILED
    pool.shutdown()


def test_live_connection_with_invalid_final_proof_is_removed_from_admission():
    class BadFinal(Backend):
        def execute(self, job, emit, check):
            result = super().execute(job, emit, check)
            result["proof"]["verified"] = False
            return result
    pool = RingPool(); ring = add_ready(pool, backend=BadFinal())
    bound = pool.acquire("test-model", COHORT, estimated_tokens=2); job = make_job()
    with pytest.raises(RingUnavailable, match="proof"):
        pool.execute(bound, job, job.commit, job.check_stop)
    assert ring.state == RingState.FAILED and not pool.ready()[0]
    pool.release(bound); pool.shutdown()


def test_backend_connection_owner_cannot_be_shared_between_distinct_rings():
    pool = RingPool(); ring = add_ready(pool)
    with pytest.raises(ValueError, match="connection owner"):
        pool.add("another", ring.backend, model_id="test-model", cohort_id=COHORT, gpu_uuids=["OTHER-GPU"])
    pool.shutdown()


def test_recovery_requires_idle_unverified_ring_and_a_new_signed_warmup():
    pool = RingPool(); ring = add_ready(pool)
    bound = pool.acquire("test-model", COHORT, estimated_tokens=2)
    ring.backend.healthy = False
    assert not pool.ready()[0]
    with pytest.raises(RingUnavailable, match="idle"):
        pool.recover(ring.ring_id)
    pool.release(bound)
    assert pool.recover(ring.ring_id)["proof_verified"] and pool.ready()[0]
    pool.drain(ring.ring_id)
    with pytest.raises(RingUnavailable):
        pool.recover(ring.ring_id)


def test_drain_during_actual_warmup_waits_for_work_cleanup_before_stop():
    class SlowWarmup(Backend):
        def warmup(self, timeout_s=300):
            self.started.set(); self.release.wait(2)
            return super().warmup(timeout_s)
    pool = RingPool(); backend = SlowWarmup()
    ring = pool.add("a", backend, model_id=backend.model_id, cohort_id=COHORT, gpu_uuids=["GPU"])
    guard = Guard("a", "GPU")
    pool.reserve("a", [guard]); pool.mark_loading("a")
    errors = []
    def warming():
        try:
            pool.warmup("a")
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=warming); thread.start()
    assert backend.started.wait(1) and guard.works
    pool.drain("a")
    assert ring.state == RingState.DRAINING and not backend.closed
    backend.release.set(); thread.join(2)
    assert errors and ring.state == RingState.STOPPED and not guard.works


def test_cleanup_failure_blocks_recovery_and_gpu_reuse():
    class FailedEnd(Guard):
        def end_work(self, work):
            raise OSError("node cleanup acknowledgement lost")
    pool = RingPool()
    guard = FailedEnd("a", "GPU")
    ring = pool.add("a", Backend(), model_id="test-model", cohort_id=COHORT, gpu_uuids=["GPU"])
    pool.reserve("a", [guard]); pool.mark_loading("a")
    with pytest.raises(OSError, match="cleanup"):
        pool.warmup("a")
    assert ring.state == RingState.FAILED and ring.local_work == 1
    with pytest.raises(RingUnavailable):
        pool.recover("a")
    pool.drain("a"); assert ring.state == RingState.DRAINING
    pool.add("b", Backend(), model_id="test-model", cohort_id=COHORT, gpu_uuids=["GPU"])
    with pytest.raises(RingUnavailable, match="already"):
        pool.reserve("b", [Guard("b", "GPU")])


def test_pool_registration_has_an_explicit_bound():
    pool = RingPool(max_rings=1); add_ready(pool)
    with pytest.raises(ValueError):
        pool.add("extra", Backend(), model_id="test-model", cohort_id=COHORT, gpu_uuids=["GPU-extra"])
    pool.shutdown()


def test_live_ring_capacity_survives_many_normal_reformations_with_bounded_history():
    pool = RingPool(max_rings=1, max_history=2)
    saved = None
    for index in range(20):
        ring = add_ready(pool, str(index), gpu="GPU-reused")
        bound = pool.acquire("test-model", COHORT, estimated_tokens=2)
        if index == 0:
            saved = bound
        pool.release(bound); pool.drain(ring.ring_id)
        assert ring.state == RingState.STOPPED
        assert len(pool.rings()) <= 2
    assert saved.ring.ring_id == "0" and saved.backend.closed
    assert saved.ring not in pool.rings()  # old clients still own the original object
    assert len(pool.rings()) == 2
    pool.shutdown()


def test_slow_remote_guard_refresh_does_not_hold_pool_lock_or_block_other_ring():
    class SlowGuard(Guard):
        def __init__(self, *args):
            super().__init__(*args)
            self.slow = False
            self.entered, self.unblock = threading.Event(), threading.Event()
        def assert_live(self):
            if self.slow:
                self.entered.set(); self.unblock.wait(2)
            return super().assert_live()
    pool = RingPool()
    slow = SlowGuard("slow", "GPU-slow")
    add_ready(pool, "slow", guards=[slow])
    healthy = add_ready(pool, "healthy", region="east")
    slow.slow = True
    assert slow.entered.wait(1)
    done, selected = threading.Event(), []
    def admission():
        selected.append(pool.acquire("test-model", COHORT, estimated_tokens=2, region="east"))
        done.set()
    thread = threading.Thread(target=admission); thread.start()
    try:
        assert done.wait(0.3), "guard RPC held the global pool admission lock"
        assert selected[0].ring is healthy
    finally:
        slow.unblock.set(); thread.join(1)
        for binding in selected:
            pool.release(binding)
        pool.shutdown()


def test_idle_guard_refresh_tracks_external_renewal_without_needing_a_request():
    pool = RingPool(); ring = add_ready(pool)
    initial = ring.lease_expires_at
    ring.guards[0].expires_at = initial + 600
    updated = threading.Event()
    original = ring.guards[0].assert_live
    def observe():
        lease = original(); updated.set(); return lease
    ring.guards[0].assert_live = observe
    assert updated.wait(1)
    # Event is set just before the monitor stores the lease snapshot. Acquire the
    # next guard cycle deterministically if that write has not happened yet.
    end = time.monotonic() + 1
    while ring.lease_expires_at == initial and time.monotonic() < end:
        updated.clear(); updated.wait(0.1)
    assert ring.lease_expires_at == initial + 600 and pool.ready()[0]
    pool.shutdown()


def test_real_durable_ledger_holds_gpu_until_pool_work_is_cleaned(tmp_path):
    now = [time.time()]
    ledger = LeaseLedger(tmp_path / "leases.db", node_id="node", clock=lambda: now[0],
                         authorize=lambda principal, action, binding: principal)
    ledger.register_capacity("host", available_ram_bytes=200, pinnable_ram_bytes=100,
                             gpu_capacity_bytes={"GPU": 100})
    request = LeaseRequest("a", COHORT, "node", "GPU", "host", LeaseResources(50, 50, 20), 100)
    lease = ledger.prepare(request, principal="controller", idempotency_key="one")
    ledger.commit(lease.lease_id, lease.fencing_token, principal="controller")
    guard = ledger.guard(lease.lease_id, lease.fencing_token, principal="controller", ring_id="a", model_cohort_sha256=COHORT)
    work = guard.begin_work("real-inflight")
    now[0] += 101
    with pytest.raises(LeaseConflict):
        ledger.prepare(LeaseRequest("b", COHORT, "node", "GPU", "host", request.resources, 100),
                       principal="controller", idempotency_key="two")
    guard.end_work(work)
    assert ledger.prepare(LeaseRequest("b", COHORT, "node", "GPU", "host", request.resources, 100),
                          principal="controller", idempotency_key="two").fencing_token > lease.fencing_token

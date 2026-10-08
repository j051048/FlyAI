"""Durable node-local lease races, shared budgets, and execution cleanup fencing."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import threading

import pytest

from shard.leases import (LeaseLedger, LeaseRequest, LeaseResources, Lease, WorkLease,
                          LeaseError, LeaseConflict, LeaseExpired, LeaseFenceError, LeaseAuthorizationError)

COHORT = "ab" * 32


@dataclass(frozen=True)
class VerifiedPeer:
    # Test transport seam; a request body string/dict is deliberately not accepted.
    subject: str


OWNER = VerifiedPeer("peer-controller")
OTHER = VerifiedPeer("other-controller")


def authorize(principal, action, binding):
    return principal.subject if isinstance(principal, VerifiedPeer) else None


@pytest.fixture
def setup(tmp_path):
    now = [100.0]
    path = tmp_path / "host-leases.sqlite"
    ledger = LeaseLedger(path, node_id="node-a", authorize=authorize, clock=lambda: now[0])
    ledger.register_capacity("host", available_ram_bytes=100, pinnable_ram_bytes=60,
                             gpu_capacity_bytes={"GPU-a": 60, "GPU-b": 60, "GPU-c": 60})
    return ledger, now, path


def request(**changes):
    values = dict(ring_id="ring", model_cohort_sha256=COHORT, node_id="node-a", gpu_uuid="GPU-a",
                  memory_domain_id="host", resources=LeaseResources(40, 50, 30), ttl_s=10)
    values.update(changes)
    return LeaseRequest(**values)


def prepare(ledger, req=None, key="prepare", principal=OWNER):
    return ledger.prepare(req or request(), principal=principal, idempotency_key=key)


def commit(ledger, lease):
    return ledger.commit(lease.lease_id, lease.fencing_token, principal=OWNER)


def guard(ledger, lease):
    return ledger.guard(lease.lease_id, lease.fencing_token, principal=OWNER,
                        ring_id=lease.ring_id, model_cohort_sha256=lease.model_cohort_sha256)


def test_prepare_commit_guard_and_serialized_bindings(setup):
    ledger, now, _ = setup
    req = request()
    assert LeaseRequest.from_dict(req.to_dict()) == req
    lease = prepare(ledger, req)
    assert lease.controller_id == OWNER.subject and lease.state == "prepared"
    assert Lease.from_dict(lease.to_dict()) == lease
    g = guard(ledger, lease)
    with pytest.raises(LeaseConflict, match="committed"):
        g.assert_live()
    with pytest.raises(LeaseConflict):
        g.begin_work("job")
    committed = commit(ledger, lease)
    assert commit(ledger, lease) == committed
    assert g.assert_live().state == "committed"
    assert (g.gpu_uuid, g.memory_domain_id, g.ring_id, g.model_cohort_sha256) == ("GPU-a", "host", "ring", COHORT)
    work = g.begin_work("job")
    assert g.begin_work("job") == work
    assert ledger.get(lease.lease_id, principal=OWNER).active_work == 1
    assert g.end_work(work).active_work == 0
    assert g.end_work(work).active_work == 0
    with pytest.raises(LeaseConflict, match="already ended"):
        g.begin_work("job")


def test_prepare_idempotency_conflict_and_gpu_exclusivity(setup):
    ledger, _, _ = setup
    first = prepare(ledger)
    assert prepare(ledger) == first
    with pytest.raises(LeaseConflict, match="idempotency"):
        prepare(ledger, request(ring_id="different"))
    with pytest.raises(LeaseConflict, match="GPU remains"):
        prepare(ledger, key="another")
    with pytest.raises(LeaseConflict):
        prepare(ledger, key="other-owner", principal=OTHER)


def test_expired_prepared_fence_is_never_revived_after_restart(setup):
    ledger, now, path = setup
    old = prepare(ledger)
    now[0] = 110
    with pytest.raises(LeaseExpired):
        commit(ledger, old)
    reopened = LeaseLedger(path, node_id="node-a", authorize=authorize, clock=lambda: now[0])
    current = prepare(reopened, key="next")
    assert current.fencing_token == old.fencing_token + 1
    assert prepare(reopened).state == "expired", "idempotency must not reprepare an expired lease"
    with pytest.raises(LeaseFenceError):
        reopened.commit(current.lease_id, old.fencing_token, principal=OWNER)


def test_expiry_blocks_reuse_until_all_actual_work_ends(setup):
    ledger, now, _ = setup
    lease = commit(ledger, prepare(ledger, request(resources=LeaseResources(40, 80, 40))))
    g = guard(ledger, lease)
    a, b = g.begin_work("a"), g.begin_work("b")
    now[0] = 111
    with pytest.raises(LeaseExpired):
        g.assert_live()
    expired = ledger.get(lease.lease_id, principal=OWNER)
    assert expired.state == "draining" and expired.active_work == 2
    with pytest.raises(LeaseExpired):
        g.begin_work("new")
    with pytest.raises(LeaseConflict, match="GPU remains"):
        prepare(ledger, key="reassign")
    with pytest.raises(LeaseConflict, match="shared RAM"):
        prepare(ledger, request(gpu_uuid="GPU-b", resources=LeaseResources(20, 30, 0)), key="second-gpu")
    assert g.end_work(a).state == "draining"
    assert g.end_work(a).active_work == 1
    assert g.end_work(b).state == "expired"
    assert prepare(ledger, key="reassign").fencing_token == lease.fencing_token + 1


@pytest.mark.parametrize("operation,terminal", [("release", "released"), ("revoke", "revoked")])
def test_revocation_and_release_do_not_reassign_live_gpu(setup, operation, terminal):
    ledger, now, _ = setup
    lease = commit(ledger, prepare(ledger))
    g = guard(ledger, lease)
    work = g.begin_work("running")
    retire = getattr(ledger, operation)
    assert retire(lease.lease_id, lease.fencing_token, principal=OWNER).state == "draining"
    assert retire(lease.lease_id, lease.fencing_token, principal=OWNER).active_work == 1
    now[0] = 200
    with pytest.raises(LeaseExpired):
        g.assert_live()
    with pytest.raises(LeaseConflict):
        prepare(ledger, key="new")
    assert g.end_work(work).state == terminal
    assert retire(lease.lease_id, lease.fencing_token, principal=OWNER).state == terminal
    successor = prepare(ledger, key="new")
    g.end_work(work)  # delayed duplicate cleanup cannot touch the successor
    assert successor == ledger.get(successor.lease_id, principal=OWNER)


def test_shared_ram_and_pinned_subset_are_separate_limits(setup):
    ledger, _, _ = setup
    prepare(ledger, request(resources=LeaseResources(40, 80, 40)))
    prepare(ledger, request(gpu_uuid="GPU-b", resources=LeaseResources(20, 20, 20)), key="b")
    # RAM=100, pinned=60 fit: pinned must NOT be charged again as extra RAM.
    with pytest.raises(LeaseConflict, match="shared RAM"):
        prepare(ledger, request(gpu_uuid="GPU-c", resources=LeaseResources(20, 1, 0)), key="c")


def test_pinned_limit_not_bypassed_by_free_host_ram(setup):
    ledger, _, _ = setup
    prepare(ledger, request(resources=LeaseResources(20, 40, 40)))
    with pytest.raises(LeaseConflict, match="shared pinned"):
        prepare(ledger, request(gpu_uuid="GPU-b", resources=LeaseResources(20, 30, 30)), key="b")


def test_unknown_host_capacity_only_allows_known_zero_host_requirements(setup):
    ledger, _, _ = setup
    ledger.register_capacity("unknown", available_ram_bytes=None, pinnable_ram_bytes=None,
                             gpu_capacity_bytes={"GPU-new": 60})
    zero = prepare(ledger, request(gpu_uuid="GPU-new", memory_domain_id="unknown", resources=LeaseResources(20, 0, 0)), key="zero")
    ledger.release(zero.lease_id, zero.fencing_token, principal=OWNER)
    with pytest.raises(LeaseConflict, match="unknown"):
        prepare(ledger, request(gpu_uuid="GPU-new", memory_domain_id="unknown", resources=LeaseResources(20, 1, 0)), key="ram")


def test_renewal_idempotent_does_not_extend_twice_or_resurrect(setup):
    ledger, now, _ = setup
    lease = commit(ledger, prepare(ledger))
    now[0] = 105
    renewed = ledger.renew(lease.lease_id, lease.fencing_token, principal=OWNER, ttl_s=10, idempotency_key="renew")
    assert renewed.expires_at == 115
    now[0] = 107
    assert ledger.renew(lease.lease_id, lease.fencing_token, principal=OWNER, ttl_s=10, idempotency_key="renew").expires_at == 115
    with pytest.raises(LeaseConflict, match="idempotency"):
        ledger.renew(lease.lease_id, lease.fencing_token, principal=OWNER, ttl_s=20, idempotency_key="renew")
    now[0] = 116
    assert ledger.renew(lease.lease_id, lease.fencing_token, principal=OWNER, ttl_s=10, idempotency_key="renew").state == "expired"
    with pytest.raises(LeaseExpired):
        ledger.renew(lease.lease_id, lease.fencing_token, principal=OWNER, ttl_s=10, idempotency_key="new-renew")


def test_caller_authentication_cannot_be_claimed_from_request(setup):
    ledger, _, path = setup
    for claimed in (OWNER.subject, {"subject": OWNER.subject}, {"authenticated": True}, None):
        with pytest.raises(LeaseAuthorizationError):
            prepare(ledger, principal=claimed)
    with pytest.raises(LeaseError):
        LeaseRequest.from_dict({**request().to_dict(), "controller_id": OWNER.subject})
    lease = prepare(ledger)
    for operation in (ledger.commit, ledger.release, ledger.revoke, ledger.assert_fence):
        with pytest.raises(LeaseAuthorizationError):
            operation(lease.lease_id, lease.fencing_token, principal=OTHER)
    with pytest.raises(LeaseAuthorizationError):
        LeaseLedger(path, node_id="node-a", authorize=None)
    dishonest = LeaseLedger(path, node_id="node-a", authorize=lambda *_: True)
    with pytest.raises(LeaseAuthorizationError):
        prepare(dishonest)


def test_ring_cohort_and_stale_work_guards_fail_closed(setup):
    ledger, _, _ = setup
    lease = commit(ledger, prepare(ledger))
    for ring, cohort in (("other", COHORT), ("ring", "cd" * 32)):
        with pytest.raises(LeaseFenceError):
            ledger.guard(lease.lease_id, lease.fencing_token, principal=OWNER, ring_id=ring, model_cohort_sha256=cohort)
    work = guard(ledger, lease).begin_work()
    with pytest.raises(LeaseFenceError):
        ledger.end_work(replace(work, fencing_token=work.fencing_token + 1), principal=OWNER)
    assert ledger.get(lease.lease_id, principal=OWNER).active_work == 1


@pytest.mark.parametrize("same_gpu", [False, True])
def test_multiconnection_atomic_resource_conflict(setup, same_gpu):
    _, now, path = setup
    connections = [LeaseLedger(path, node_id="node-a", authorize=authorize, clock=lambda: now[0]) for _ in range(2)]
    barrier = threading.Barrier(2)
    def attempt(index):
        barrier.wait()
        try:
            return prepare(connections[index], request(gpu_uuid="GPU-a" if same_gpu or index == 0 else "GPU-b",
                                                      resources=LeaseResources(40, 70, 30)), key=str(index))
        except LeaseConflict:
            return None
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(attempt, range(2)))
    assert sum(result is not None for result in results) == 1


def test_concurrent_identical_prepare_is_one_reservation(setup):
    _, now, path = setup
    connections = [LeaseLedger(path, node_id="node-a", authorize=authorize, clock=lambda: now[0]) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as workers:
        leases = list(workers.map(lambda ledger: prepare(ledger), connections))
    assert len({lease.lease_id for lease in leases}) == 1
    assert {lease.fencing_token for lease in leases} == {1}


def test_same_host_nodes_share_budgets_but_not_operation_or_work_keys(setup):
    first, now, path = setup
    second = LeaseLedger(path, node_id="node-b", authorize=authorize, clock=lambda: now[0])
    a = commit(first, prepare(first, request(resources=LeaseResources(20, 50, 30))))
    req = request(node_id="node-b", gpu_uuid="GPU-b", resources=LeaseResources(20, 50, 30))
    b = second.prepare(req, principal=OWNER, idempotency_key="prepare")
    b = second.commit(b.lease_id, b.fencing_token, principal=OWNER)
    assert a.lease_id != b.lease_id
    ga, gb = guard(first, a), guard(second, b)
    wa, wb = ga.begin_work("same-job"), gb.begin_work("same-job")
    assert ga.end_work(wa).active_work == 0 and gb.end_work(wb).active_work == 0
    with pytest.raises(LeaseAuthorizationError):
        first.get(b.lease_id, principal=OWNER)


def test_restart_retains_execution_until_local_cleanup_confirms_stopped(setup):
    first, now, path = setup
    lease = commit(first, prepare(first))
    guard(first, lease).begin_work("orphaned-runner")
    now[0] = 111
    reopened = LeaseLedger(path, node_id="node-a", authorize=authorize, clock=lambda: now[0])
    assert reopened.get(lease.lease_id, principal=OWNER).active_work == 1
    assert reopened.recover_stopped_work(lambda work: False) == 0
    with pytest.raises(LeaseConflict):
        prepare(reopened, key="replacement")
    cleaned = []
    assert reopened.recover_stopped_work(lambda work: cleaned.append(work) or True) == 1
    assert cleaned[0].work_id == "orphaned-runner"
    assert reopened.recover_stopped_work(lambda work: True) == 0
    assert prepare(reopened, key="replacement").fencing_token == lease.fencing_token + 1


def test_failed_expiry_assert_persists_clock_floor_preventing_rollback_revival(setup):
    ledger, now, _ = setup
    lease = commit(ledger, prepare(ledger))
    now[0] = 111
    with pytest.raises(LeaseExpired):
        guard(ledger, lease).assert_live()
    now[0] = 101
    with pytest.raises(LeaseExpired):
        ledger.begin_work(lease.lease_id, lease.fencing_token, principal=OWNER)
    assert ledger.get(lease.lease_id, principal=OWNER).state == "expired"


def test_capacity_updates_cannot_undercut_reservations_or_move_busy_gpu(setup):
    ledger, _, _ = setup
    lease = prepare(ledger)
    for ram, pin, vram in ((49, 49, 60), (100, 29, 60), (100, 60, 39)):
        with pytest.raises(LeaseConflict):
            ledger.register_capacity("host", available_ram_bytes=ram, pinnable_ram_bytes=pin,
                                     gpu_capacity_bytes={"GPU-a": vram})
    with pytest.raises(LeaseConflict):
        ledger.register_capacity("different-domain", available_ram_bytes=100, pinnable_ram_bytes=60,
                                 gpu_capacity_bytes={"GPU-a": 60})
    assert ledger.get(lease.lease_id, principal=OWNER).state == "prepared"


@pytest.mark.parametrize("resources", [(0, 0, 0), (1, 1, 2), (True, 0, 0), (1, -1, 0), (1, 1.0, 0)])
def test_invalid_byte_contracts_rejected(resources):
    with pytest.raises(LeaseError):
        LeaseResources(*resources)

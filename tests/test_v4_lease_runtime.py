"""Real CPU V4 stage sockets preserve tokens and receipts with local fencing."""
import pytest

from test_v4_privacy_ring import _PrivacyRing, tiny, VP
from shard.leases import LeaseLedger, LeaseRequest, LeaseResources


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("mode", ["greedy", "pipelined"])
def test_fenced_runtime_preserves_real_reference_ring_output(tiny, tmp_path, monkeypatch, fp8, mode):
    checkpoint, args, _ = tiny
    monkeypatch.setattr(VP, "V4_FP8_WIRE", fp8)
    monkeypatch.setattr(VP, "V4_SEALED_IDS", False)
    baseline = _PrivacyRing(checkpoint, args, dspark=mode == "pipelined")
    try:
        expected = baseline.coordinate(mode)
    finally:
        baseline.close()
    guards, resident = [], []
    for index in range(4):
        ledger = LeaseLedger(tmp_path / "cpu-fixture.sqlite", node_id=f"cpu-stage-{index}", authorize=lambda p,a,b:p)
        ledger.register_capacity("cpu-fixture", available_ram_bytes=10000, pinnable_ram_bytes=10000,
                                 gpu_capacity_bytes={f"virtual-device-{i}": 1000 for i in range(4)})
        lease = ledger.prepare(LeaseRequest("cpu-ring", "d" * 64, f"cpu-stage-{index}", f"virtual-device-{index}",
            "cpu-fixture", LeaseResources(1, 1, 0), 120), principal="cpu-test", idempotency_key="resident")
        ledger.commit(lease.lease_id, lease.fencing_token, principal="cpu-test")
        guard = ledger.guard(lease.lease_id, lease.fencing_token, principal="cpu-test", ring_id="cpu-ring", model_cohort_sha256="d" * 64)
        guards.append(guard)
        resident.append(guard.begin_work("resident-stage-test"))
    ring = None
    try:
        ring = _PrivacyRing(checkpoint, args, dspark=mode == "pipelined", guards=guards)
        actual = ring.coordinate(mode)
        assert actual["tokens"] == expected["tokens"]
        assert actual["receipts_ok"] is True and not ring.errors
        assert all(g.assert_live().active_work == 1 for g in guards)
    finally:
        if ring is not None:
            ring.close()
        for guard, handle in zip(guards, resident):
            guard.end_work(handle)
            guard.ledger.release(guard.lease_id, guard.fencing_token, principal="cpu-test")

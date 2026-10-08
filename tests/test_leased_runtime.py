from datetime import datetime, timezone
import sys
import time

import pytest

from shard.leases import LeaseConflict, LeaseLedger, LeaseRequest, LeaseResources, LeaseExpired
from shard.leased_runtime import LeasedProcessRunner, install_stage_lease_checks, process_lease_watchdog


def guarded(tmp_path, clock=time.time, ttl=120):
    path = tmp_path / "host.sqlite"
    ledger = LeaseLedger(path, node_id="node", authorize=lambda p, a, b: p, clock=clock)
    ledger.register_capacity("host", available_ram_bytes=1000, pinnable_ram_bytes=500,
                             gpu_capacity_bytes={"GPU-0": 1000})
    req = LeaseRequest("ring", "1" * 64, "node", "GPU-0", "host", LeaseResources(100, 100, 50), ttl)
    lease = ledger.prepare(req, principal="controller", idempotency_key="start")
    ledger.commit(lease.lease_id, lease.fencing_token, principal="controller")
    guard = ledger.guard(lease.lease_id, lease.fencing_token, principal="controller", ring_id="ring", model_cohort_sha256="1" * 64)
    return ledger, guard, req, path


def assignment():
    return {"ring_id": "ring", "cohort_id": "1" * 64, "node_id": "node", "gpu_uuid": "GPU-0"}


def test_resident_process_retains_resources_until_actual_process_exit(tmp_path):
    ledger, guard, req, path = guarded(tmp_path)
    runner = LeasedProcessRunner(guard, ledger_path=path,
                                 command_factory=lambda _: [sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        runner.start(assignment())
        assert guard.assert_live().active_work == 1
        retired = ledger.release(guard.lease_id, guard.fencing_token, principal="controller")
        assert retired.state == "draining"
        with pytest.raises(LeaseConflict):
            ledger.prepare(req, principal="next-controller", idempotency_key="next")
        runner.stop()
        assert not runner.status()["resident_work_held"]
        new = ledger.prepare(req, principal="next-controller", idempotency_key="next")
        assert new.fencing_token > guard.fencing_token
    finally:
        runner.stop()


def test_expiry_stops_only_the_owned_stage_and_acknowledges_cleanup(tmp_path):
    clock = [1000]
    ledger, guard, req, path = guarded(tmp_path, clock=lambda: clock[0], ttl=1)
    runner = LeasedProcessRunner(guard, ledger_path=path,
                                 command_factory=lambda _: [sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        runner.start(assignment())
        clock[0] = 1002
        deadline = time.monotonic() + 3
        while runner.status()["resident_work_held"] and time.monotonic() < deadline:
            time.sleep(.02)
        assert not runner.status()["running"]
        assert not runner.status()["resident_work_held"]
        assert runner.status()["error"] == "LeaseExpired"
    finally:
        runner.stop()


def test_failed_process_start_releases_work_but_not_the_controller_lease(tmp_path):
    ledger, guard, req, path = guarded(tmp_path)
    runner = LeasedProcessRunner(guard, ledger_path=path, command_factory=lambda _: ["this-program-does-not-exist-86753"])
    with pytest.raises(OSError):
        runner.start(assignment())
    assert guard.assert_live().active_work == 0
    assert guard.assert_live().state == "committed"


def test_node_local_fence_rejects_a_frame_before_state_mutation(tmp_path):
    clock = [1000]
    _, guard, _, _ = guarded(tmp_path, clock=lambda: clock[0], ttl=1)
    class Stage:
        calls = 0
        def forward(self, value):
            self.calls += 1
            return value + 1
    stage = install_stage_lease_checks(Stage(), guard)
    assert stage.forward(3) == 4
    clock[0] = 1002
    with pytest.raises(LeaseExpired):
        stage.forward(3)
    assert stage.calls == 1


def test_cli_watchdog_exits_an_idle_expired_stage():
    called = []
    class Guard:
        def assert_live(self):
            raise LeaseExpired("expired")
    stop = process_lease_watchdog(Guard(), interval_s=.01, exit_process=called.append)
    try:
        deadline = time.monotonic() + 1
        while not called and time.monotonic() < deadline:
            time.sleep(.01)
        assert called == [75]
    finally:
        stop.set()


@pytest.mark.parametrize("mismatch", ["environment", "placeholder"])
def test_calibrated_template_cannot_launch_another_environment_or_dynamic_command(tmp_path, monkeypatch, mismatch):
    import os
    from shard.leased_runtime import configured_stage_factory
    from shard.deployment import config_digest
    from shard.resources import CalibrationProvenance, GpuRequirements, HostRequirements, PlacementRequirements
    for key in list(os.environ):
        if key.startswith("V4_"):
            monkeypatch.delenv(key)
    ledger, guard, _, path = guarded(tmp_path)
    cfg = {"lo": 0, "hi": 1, "head": True, "tail": False, "environment": {"V4_EXPERT_PLACEMENT": "ram"}}
    req = PlacementRequirements("cpu-fixture", 0, 1, GpuRequirements(resident_weights_bytes=1), HostRequirements(),
        CalibrationProvenance("cpu-weights", config_digest(cfg), datetime.now(timezone.utc).isoformat(), "node", "cpu-fixture", "test"))
    row = {"cohort_id": "1" * 64, "lo": 0, "hi": 1, "head": True, "tail": False,
           "requirements": req.to_dict(), "runtime_config": cfg,
           "argv": [sys.executable, "local-stage.py", "--lo={lo}"], "environment": dict(cfg["environment"])}
    if mismatch == "environment":
        row["environment"]["V4_EXPERT_PLACEMENT"] = "gpu"
    else:
        row["argv"].append("{unvalidated_path}")
    assigned = {**assignment(), "lo": 0, "hi": 1, "head": True, "tail": False, "stage": 0, "nstages": 2}
    with pytest.raises(ValueError, match="environment|coordinates"):
        configured_stage_factory(path, [row])(assigned, guard)
    assert guard.assert_live().active_work == 0

"""Real pool/queue request boundaries and configured formation fallbacks."""
from copy import deepcopy
import threading
import time
from types import SimpleNamespace

import pytest

from shard.leases import LeaseConflict
from shard.network_service import OpenNetworkService, PreparedManagedRingBackend
from shard.ring_pool import RingPool, RingState, RingUnavailable
from shard.ring_router import RingRouter, MultiRingQueue
from shard.service_queue import TenantLimits
from shard.offers import ModelCohort
from test_ring_pool import Backend, Guard, add_ready, COHORT, OTHER


def body(name):
    return {"model": "default", "messages": [{"role": "user", "content": name}], "max_tokens": 2}


def wait(job):
    deadline = time.monotonic() + 3
    with job.changed:
        while not job.terminal and time.monotonic() < deadline: job.changed.wait(.01)
    assert job.terminal


def test_replacement_publishes_after_warmup_and_keeps_running_and_queued_bindings():
    pool = RingPool(); old = add_ready(pool, "old", backend=Backend(block=True))
    pool.set_alias("default", "test-model", COHORT)
    queue = MultiRingQueue(RingRouter(pool, default_model="default"), {"a": TenantLimits(10, 100, 1000)})
    queue.start()
    try:
        running = queue.submit_request("a", body("running"), idempotency_key="held")[0]
        assert old.backend.started.wait(1)
        queued = queue.submit_request("a", body("queued"))[0]
        new = add_ready(pool, "new", backend=Backend(offset=10))
        publication = pool.publish_replacement("new", ["old"], aliases=["default"])
        assert publication["published"] and old.state == RingState.DRAINING
        assert not old.backend.closed and old.bound_jobs == 2
        same, fresh = queue.submit_request("a", body("running"), idempotency_key="held")
        assert same is running and not fresh and same.bound_backend is old.backend
        following = queue.submit_request("a", body("following"))[0]
        assert following.ring_binding.ring is new
        wait(following); assert following.state == "completed" and following.result["text"] == "KL"
        old.backend.release.set()
        for job in (running, queued):
            wait(job); assert job.state == "completed" and job.ring_binding.ring is old
        deadline = time.monotonic() + 2
        while old.state != RingState.STOPPED and time.monotonic() < deadline: time.sleep(.001)
        assert old.state == RingState.STOPPED and old.backend.closed
        assert old.replacement_ring_id == "new"
    finally:
        old.backend.release.set(); queue.shutdown()


def test_unwarmed_or_wrong_cohort_cannot_retire_existing_route():
    pool = RingPool(); old = add_ready(pool, "old")
    new = pool.add("new", Backend(), model_id="test-model", cohort_id=COHORT, gpu_uuids=["new-gpu"])
    with pytest.raises(RingUnavailable, match="warmup"): pool.publish_replacement("new", ["old"])
    assert old.state == RingState.READY
    other = add_ready(pool, "other", cohort=OTHER)
    with pytest.raises(RingUnavailable, match="cohort"): pool.publish_replacement("other", ["old"])
    assert old.state == RingState.READY
    pool.shutdown()


def test_old_cleanup_failure_does_not_destroy_new_ready_route_or_release_old_identity():
    class BrokenClose(Backend):
        def close(self): raise RuntimeError("node cleanup not acknowledged")
    pool = RingPool(); old = add_ready(pool, "old", backend=BrokenClose()); new = add_ready(pool, "new")
    assert pool.publish_replacement("new", ["old"])["published"]
    deadline = time.monotonic() + 2
    while old.reason != "replacement_cleanup_unconfirmed" and time.monotonic() < deadline: time.sleep(.001)
    assert old.state == RingState.DRAINING and old.reason == "replacement_cleanup_unconfirmed"
    assert new.state == RingState.READY and pool.acquire("test-model", COHORT, estimated_tokens=1).ring is new
    # The failed close is not reported as STOPPED or eligible for reuse.
    assert not old.guard_stop.is_set() or old.state != RingState.STOPPED
    new.guard_stop.set(); old.guard_stop.set()


def test_slow_old_stop_rpc_never_blocks_admission_to_published_new_ring():
    entered, proceed, stopped = threading.Event(), threading.Event(), threading.Event()
    class SlowClose(Backend):
        def close(self): entered.set(); assert proceed.wait(3); super().close()
    pool = RingPool()
    old = add_ready(pool, "old", backend=SlowClose())
    old.on_stopped = lambda _: stopped.set()
    new = add_ready(pool, "new")
    pool.publish_replacement("new", ["old"])
    assert entered.wait(1) and old.state == RingState.DRAINING
    start = time.monotonic()
    binding = pool.acquire("test-model", COHORT, estimated_tokens=1)
    assert binding.ring is new and time.monotonic() - start < .1
    assert not stopped.is_set() and not old.backend.closed
    proceed.set(); assert stopped.wait(1)
    assert old.state == RingState.STOPPED and old.backend.closed
    pool.release(binding); pool.shutdown()


def test_all_nodes_prepare_before_any_start_and_bad_pins_never_start():
    events = []
    expected = {"artifact_id": "id", "checkpoint_id": "check", "manifest_sha256": "hash"}
    class Client:
        def __init__(self, name, bad=False): self.node_id, self.bad = name, bad
        def operation(self, action, lease, **kwargs):
            events.append((self.node_id, action))
            if action == "prepare_stage": return {"job_id": self.node_id, "state": "fetching"}
            if action == "prepare_status": return {"job_id": self.node_id, "state": "ready",
                "payload_integrity_verified": True, "source_artifact_id": "bad" if self.bad else "id",
                "checkpoint_id": "check", "manifest_sha256": "hash"}
            return {"running": action == "start_stage", "resident_work_held": False}
    backend = Backend()
    records = [(Client(str(i)), {}, {"weight_artifacts": expected}) for i in range(2)]
    managed = PreparedManagedRingBackend(backend, records, poll_s=.001)
    managed.load()
    first_start = next(index for index, (_, action) in enumerate(events) if action == "start_stage")
    assert sum(action == "prepare_status" for _, action in events[:first_start]) == 2
    events.clear()
    managed = PreparedManagedRingBackend(Backend(), [(Client("bad", True), {}, {"weight_artifacts": expected})], poll_s=.001)
    with pytest.raises(ValueError, match="pinned"): managed.load()
    assert not any(action == "start_stage" for _, action in events)


def cohort():
    return ModelCohort("fixture/model", "a" * 64, "checkpoint", "b" * 64,
        "bf16", "cpu-fixture/1", "wire/1", "numeric/1", 3).to_dict()


def test_only_explicit_same_cohort_calibrated_alternatives_are_tried_on_capacity_failure():
    service = OpenNetworkService.__new__(OpenNetworkService)
    calls = []
    def form(row):
        calls.append(row["ring_id"])
        if row["ring_id"] == "large": raise LeaseConflict("shared RAM capacity")
        return {"ring_id": row["ring_id"], "ready": True}
    service.form_one = form
    row = {"ring_id": "large", "cohort": cohort(), "calibrated_alternatives": [
        {"ring_id": "small", "cohort": cohort(), "profile": {"n_layers": 3, "measured": "existing"}}]}
    result = service.form_with_alternatives(row)
    assert calls == ["large", "small"] and result["capacity_fallback"][0]["ring_id"] == "large"
    altered = deepcopy(row); altered["calibrated_alternatives"][0]["cohort"]["checkpoint_id"] = "different"
    calls.clear()
    with pytest.raises(ValueError, match="cohort"): service.form_with_alternatives(altered)
    assert calls == []
    service.form_one = lambda _: (_ for _ in ()).throw(ValueError("route or payload integrity error"))
    with pytest.raises(ValueError, match="integrity"): service.form_with_alternatives(row)


def reconciliation_service(pool, definition, *, cooldown=.02):
    service = OpenNetworkService.__new__(OpenNetworkService)
    service.pool = pool
    service._formation_lock = threading.RLock()
    service._reconcile_stop = threading.Event()
    service._reconcile_thread = None
    service._reconcile_interval = .01
    service._reconcile_cooldown = cooldown
    service._generation_sequence = 0
    service._replacement_policies = {"old": {"definition": definition, "active_ring_id": "old",
        "attempted": {"old"}, "running": False, "next_attempt": 0.0}}
    service.controller = SimpleNamespace(stop=lambda ring_id: pool.drain(ring_id))
    return service


def test_actual_background_trigger_publishes_same_cohort_without_moving_held_binding():
    pool = RingPool(); old = add_ready(pool, "old")
    pool.set_alias("default", "test-model", COHORT)
    held = pool.acquire("test-model", COHORT, estimated_tokens=2)
    calls = []
    service = reconciliation_service(pool, {"calibrated_alternatives": [{"ring_id": "new", "cohort": cohort()}]})
    def form(candidate):
        calls.append(candidate["ring_id"])
        new = add_ready(pool, candidate["ring_id"])
        return {"ring_id": new.ring_id, "ready": True}
    service.form_with_alternatives = form
    service.start_reconciliation()
    try:
        time.sleep(.04)
        assert calls == []  # A healthy resident ring is never replaced from free-VRAM readings.
        pool.fail("old", "guard_failed")
        deadline = time.monotonic() + 2
        while not pool.snapshot().get("reconciliation", {}).get("old", {}).get("state") == "published" and time.monotonic() < deadline:
            time.sleep(.01)
        assert calls == ["new"] and old.state == RingState.DRAINING
        assert held.ring is old and held.backend is old.backend and not old.backend.closed
        following = pool.acquire("test-model", COHORT, estimated_tokens=1)
        assert following.ring.ring_id == "new"
        assert pool.resolve("default") == ("test-model", COHORT)
        pool.release(following); pool.release(held)
    finally:
        service._reconcile_stop.set(); service._reconcile_thread.join(1)
        pool.release(held); pool.shutdown()


def test_automatic_capacity_failure_is_cooled_and_does_not_reuse_candidate_ids_or_stop_old():
    pool = RingPool(); old = add_ready(pool, "old")
    held = pool.acquire("test-model", COHORT, estimated_tokens=2)
    service = reconciliation_service(pool, {"calibrated_alternatives": [
        {"ring_id": "first"}, {"ring_id": "second"}], "auto_generations": False}, cooldown=60)
    calls = []
    def form(candidate): calls.append(candidate["ring_id"]); raise LeaseConflict("no spare calibrated GPUs")
    service.form_with_alternatives = form
    pool.fail("old")
    service.reconcile_once()
    deadline = time.monotonic() + 1
    while service._replacement_policies["old"]["running"] and time.monotonic() < deadline: time.sleep(.001)
    assert calls == ["first"]
    for _ in range(5): service.reconcile_once()
    assert calls == ["first"] and not old.backend.closed and held.ring is old
    service._replacement_policies["old"]["next_attempt"] = 0
    service.reconcile_once()
    while service._replacement_policies["old"]["running"] and time.monotonic() < deadline: time.sleep(.001)
    assert calls == ["first", "second"]
    service._replacement_policies["old"]["next_attempt"] = 0
    for _ in range(5): service.reconcile_once()
    assert calls == ["first", "second"]
    assert pool.snapshot()["reconciliation"]["old"]["reason"] == "configured_calibrated_candidates_exhausted"
    assert not old.backend.closed
    pool.release(held); pool.shutdown()


def test_temporary_capacity_failure_retries_exact_template_in_a_fresh_epoch_then_recovers():
    pool = RingPool(); old = add_ready(pool, "old")
    held = pool.acquire("test-model", COHORT, estimated_tokens=1)
    template = {"ring_id": "approved", "cohort": cohort(), "workload": {"frame_tokens": 1},
                "profile": {"require_exact_calibrations": True}, "measurements": {"pinned": "same"}}
    service = reconciliation_service(pool, {"calibrated_alternatives": [template]}, cooldown=.02)
    calls = []
    def form(candidate):
        calls.append(candidate)
        if len(calls) == 1: raise LeaseConflict("currently occupied")
        assert {key: value for key, value in candidate.items() if key not in ("ring_id", "calibrated_alternatives")} == {
            key: value for key, value in template.items() if key != "ring_id"}
        ready = add_ready(pool, candidate["ring_id"])
        return {"ring_id": ready.ring_id, "ready": True}
    service.form_with_alternatives = form
    pool.fail("old")
    service.reconcile_once()
    deadline = time.monotonic() + 2
    while service._replacement_policies["old"]["running"] and time.monotonic() < deadline: time.sleep(.001)
    assert calls[0]["ring_id"] == "approved" and not old.backend.closed
    time.sleep(.03); service.reconcile_once()
    while service._replacement_policies["old"]["running"] and time.monotonic() < deadline: time.sleep(.001)
    assert len(calls) == 2 and calls[1]["ring_id"].startswith("approved.r")
    assert calls[1]["ring_id"] != calls[0]["ring_id"]
    assert pool.snapshot()["reconciliation"]["old"]["state"] == "published"
    assert held.ring is old and old.state == RingState.DRAINING
    assert len(service._replacement_policies["old"]["attempted"]) == 2
    pool.release(held); pool.shutdown()


def test_preplanning_known_capacity_refusal_uses_an_already_calibrated_routed_alternative(tmp_path):
    from datetime import datetime, timezone
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from shard.control_plane import FormationController, LocalLeaseClient, CapacityUnavailable
    from shard.leases import LeaseLedger
    from shard.offers import OfferRegistry, sign_offer
    from shard.resources import GpuRequirements, HostRequirements, PlacementRequirements, CalibrationProvenance
    from shard.deployment import config_digest
    from test_node_offers import offer, cohort as model_descriptor
    descriptor = {**model_descriptor(), "n_layers": 2}
    model = ModelCohort.from_dict(descriptor)
    registry, pool, agents, identities = OfferRegistry(), RingPool(), {}, []
    for index in range(4):
        key = Ed25519PrivateKey.generate()
        body = offer(key, gpu=f"GPU-{index}", now=time.time(), models=[])
        stage = index % 2
        cfg = {"lo": stage, "hi": stage + 1, "head": stage == 0, "tail": stage == 1,
               "stage": stage, "nstages": 2, "environment": {}}
        req = PlacementRequirements(model.model_id, stage, stage + 1,
            GpuRequirements(resident_weights_bytes=100), HostRequirements(reserve_bytes=200),
            CalibrationProvenance(model.checkpoint_id, config_digest(cfg), datetime.now(timezone.utc).isoformat(),
                body["node_id"], "CPU control fixture", "not a GPU calibration"))
        domain, ram = f"host-{index}", 100 if index < 2 else 1000
        body["memory_domain_id"] = body["host_id"] = domain
        body["resources"].update(available_ram_bytes=ram, pinnable_ram_bytes=0)
        body["models"] = [{"cohort": descriptor, "profile": {"layer_ms": 1, "cap_layers": 1},
            "measured_at": time.time(), "calibrations": [{"requirements": req.to_dict(), "runtime_config": cfg}]}]
        body = sign_offer(body, key); registry.announce(body); identities.append(body["node_id"])
        ledger = LeaseLedger(tmp_path / f"host{index}.sqlite", node_id=body["node_id"], authorize=lambda p,a,b:p)
        ledger.register_capacity(domain, available_ram_bytes=ram, pinnable_ram_bytes=0,
                                 gpu_capacity_bytes={f"GPU-{index}": 1 << 30})
        agents[body["node_id"]] = LocalLeaseClient(ledger, "controller")
    def backend(plan, cohort, contracts):
        value = Backend(); value.model_id = cohort.model_id; return value
    controller = FormationController(registry, pool, agents, requirements=OpenNetworkService.requirements,
                                     backend_factory=backend)
    profile = {"n_layers": 2, "layer_vram_mb": 1, "kv_mb_per_layer": 0, "reserve_mb": 0,
        "head_reserve_mb": 0, "tail_reserve_mb": 0, "cap_layers": 1, "layer_ms_base": 1,
        "head_layer_ms_mult": 1, "require_exact_calibrations": True}
    def mesh(ids):
        return {"schema": "shard-link-measurements/1", "edges": [
            {"src": ids[0], "dst": ids[1], "rtt_ms": 1, "measured_at": time.time(), "ttl_s": 120},
            {"src": ids[1], "dst": ids[0], "rtt_ms": 1, "measured_at": time.time(), "ttl_s": 120}]}
    service = OpenNetworkService.__new__(OpenNetworkService)
    service.form_one = lambda row: controller.form(row["ring_id"], model, profile, measurements=row["measurements"])
    row = {"ring_id": "small-host", "cohort": descriptor, "measurements": mesh(identities[:2]),
        "calibrated_alternatives": [{"ring_id": "available-host", "cohort": descriptor,
                                    "measurements": mesh(identities[2:])}]}
    try:
        result = service.form_with_alternatives(row)
        assert result["ready"] and result["ring_id"] == "available-host"
        assert result["capacity_fallback"][0]["ring_id"] == "small-host"
        assert set(result["plan"]["order"]) == set(identities[2:])
    finally:
        controller.close()
        deadline = time.monotonic() + 2
        while controller._formations and time.monotonic() < deadline: time.sleep(.01)
        registry.close()

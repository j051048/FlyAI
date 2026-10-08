"""Control-plane integration with real SQLite leases and explicit CPU backends."""
from datetime import datetime, timezone
import threading
import time

import pytest

from shard.control_plane import ControlError, FormationController, LocalLeaseClient
from shard.leases import LeaseLedger
from shard.offers import OfferRegistry, ModelCohort
from shard.resources import CalibrationProvenance, GpuRequirements, HostRequirements, PlacementRequirements
from shard.ring_pool import RingPool, RingState
from test_node_offers import offer, cohort
from test_ring_pool import Backend


PROFILE = {"n_layers": 2, "layer_vram_mb": 1, "kv_mb_per_layer": 0, "reserve_mb": 0,
           "head_reserve_mb": 0, "tail_reserve_mb": 0, "cap_layers": 1,
           "layer_ms_base": 1, "head_layer_ms_mult": 1}


def network(tmp_path, *, count=4, backend_factory=None, requirements=None):
    now = time.time()
    descriptor = {**cohort(), "n_layers": 2}
    model = ModelCohort.from_dict(descriptor)
    registry, pool, agents = OfferRegistry(clock=time.time), RingPool(), {}
    for i in range(count):
        body = offer(gpu=f"GPU-{i}", now=now, models=[{"cohort": descriptor,
            "profile": {"layer_ms": 1, "cap_layers": 1}, "measured_at": now}])
        registry.announce(body)
        ledger = LeaseLedger(tmp_path / "host.sqlite", node_id=body["node_id"], authorize=lambda p,a,b:p)
        ledger.register_capacity("host-0", available_ram_bytes=1000, pinnable_ram_bytes=500,
                                 gpu_capacity_bytes={f"GPU-{j}": 2**30 for j in range(count)})
        agents[body["node_id"]] = LocalLeaseClient(ledger, "controller")
    def measured(stage, node):
        return PlacementRequirements(model.model_id, stage["lo"], stage["hi"],
            GpuRequirements(resident_weights_bytes=100), HostRequirements(routed_experts_bytes=100, pinned_bytes=100),
            CalibrationProvenance(model.checkpoint_id, "1" * 64, datetime.now(timezone.utc).isoformat(),
                                  node["node_id"], "cpu-test-contract", "fixture"))
    def factory(plan, cohort, contracts):
        backend = Backend()
        backend.model_id = cohort.model_id
        return backend
    controller = FormationController(registry, pool, agents, requirements=requirements or measured,
                                     backend_factory=backend_factory or factory)
    nodes = registry.snapshot(model.cohort_id)
    mesh = {"schema": "shard-link-measurements/1", "edges": [
        {"src": a["id"], "dst": b["id"], "rtt_ms": 1, "measured_at": now, "ttl_s": 300}
        for a in nodes for b in nodes if a is not b]}
    return controller, pool, registry, agents, model, mesh


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.01)
    assert predicate()


def test_offers_to_local_plan_committed_leases_warmup_and_second_disjoint_ring(tmp_path):
    controller, pool, registry, agents, model, mesh = network(tmp_path)
    try:
        first = controller.form("first", model, PROFILE, measurements=mesh)
        second = controller.form("second", model, PROFILE, measurements=mesh)
        assert first["ready"] and second["ready"]
        assert set(first["plan"]["order"]).isdisjoint(second["plan"]["order"])
        assert all(info["state"] == "READY" for info in pool.snapshot()["rings"])
        # Registration does not rebuild either ready ring.
        before = [info["plan"]["order"][:] for info in controller._formations.values()]
        registry.announce(offer(gpu="GPU-new", now=time.time(), models=[]))
        assert before == [info["plan"]["order"] for info in controller._formations.values()]
    finally:
        controller.close()
        wait_for(lambda: not controller._formations)


def test_failed_warmup_rolls_back_every_reservation(tmp_path):
    def factory(plan, cohort, contracts):
        backend = Backend(forged=True)
        backend.model_id = cohort.model_id
        return backend
    controller, pool, _, agents, model, mesh = network(tmp_path, count=2, backend_factory=factory)
    try:
        with pytest.raises(RuntimeError):
            controller.form("bad", model, PROFILE, measurements=mesh)
        wait_for(lambda: not controller._formations)
        # Same GPUs can subsequently form a healthy ring.
        def good(plan, cohort, contracts):
            backend = Backend(); backend.model_id = cohort.model_id
            return backend
        controller.backend_factory = good
        assert controller.form("good", model, PROFILE, measurements=mesh)["ready"]
    finally:
        controller.close()
        wait_for(lambda: not controller._formations)


def test_loading_longer_than_initial_ttl_keeps_existing_and_new_ring_leases_live(tmp_path):
    loading, finish = threading.Event(), threading.Event()
    calls = [0]
    def factory(plan, cohort, contracts):
        backend = Backend(); backend.model_id = cohort.model_id
        calls[0] += 1
        if calls[0] == 2:
            def load():
                loading.set()
                assert finish.wait(3)
            backend.load = load
        return backend
    controller, pool, _, _, model, mesh = network(tmp_path, backend_factory=factory)
    errors = []
    try:
        controller.form("old", model, PROFILE, measurements=mesh, ttl_s=.6)
        def form_new():
            try:
                controller.form("new", model, PROFILE, measurements=mesh, ttl_s=.6)
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=form_new)
        thread.start()
        assert loading.wait(2)
        time.sleep(1.0)
        assert pool.ready()[0]
        for info in controller._formations.values():
            for client, lease in info["leases"]:
                assert client.guard(lease, ring_id=lease["ring_id"], cohort_id=model.cohort_id).assert_live()
        finish.set(); thread.join(3)
        assert not thread.is_alive() and not errors
    finally:
        finish.set(); controller.close()
        wait_for(lambda: not controller._formations)


def test_graceful_drain_preserves_lease_until_previously_bound_request_finishes(tmp_path):
    controller, pool, _, _, model, mesh = network(tmp_path, count=2)
    try:
        controller.form("ring", model, PROFILE, measurements=mesh, ttl_s=.6)
        binding = pool.acquire(model.model_id, model.cohort_id, estimated_tokens=2)
        controller.stop("ring")
        assert binding.ring.state == RingState.DRAINING
        time.sleep(.8)
        for client, lease in controller._formations["ring"]["leases"]:
            assert client.guard(lease, ring_id="ring", cohort_id=model.cohort_id).assert_live()
        pool.release(binding)
        wait_for(lambda: not controller._formations)
        assert binding.ring.state == RingState.STOPPED
    finally:
        controller.close()


def test_unmeasured_contract_is_rejected_before_any_gpu_reservation(tmp_path):
    controller, _, _, _, model, mesh = network(tmp_path, count=2, requirements=lambda *_: {})
    try:
        with pytest.raises(ValueError):
            controller.form("bad", model, PROFILE, measurements=mesh)
        assert not controller._formations
    finally:
        controller.close()

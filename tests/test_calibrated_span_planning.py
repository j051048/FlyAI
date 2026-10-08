"""Exact measured resource templates constrain executable spans, not estimates."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from shard.offers import OfferError, OfferRegistry, canonical, model_cohort_id, sign_offer
from shard.plan import plan_ring
from shard.resources import CalibrationProvenance, GpuRequirements, HostRequirements, PlacementRequirements
from test_node_offers import cohort, offer
from test_planning_locality import pool, mesh, profile


def span(lo, hi, *, depth=6, cfg="a", gpu=100, host=0, pinned=0, index=None, nstages=None):
    row = dict(lo=lo, hi=hi, head=lo == 0, tail=hi == depth, gpu_bytes=gpu,
               host_bytes=host, pinned_bytes=pinned, runtime_config_sha256=cfg * 64)
    if index is not None:row["stage_index"] = index
    if nstages is not None:row["nstages"] = nstages
    return row


def nodes_with_spans(rows, *, speed=None, host_capacity=1000, pin_capacity=1000):
    nodes = pool(["A"] * len(rows), caps=[6] * len(rows), speed=speed)
    for node, choices in zip(nodes, rows):
        node["allowed_spans"] = choices
        node["resource_capacity"] = dict(available_vram_bytes=1024,
            available_ram_bytes=host_capacity, pinnable_ram_bytes=pin_capacity)
        node["memory_domain_id"] = "shared-host"
    return nodes


def test_exact_templates_choose_deployable_three_three_instead_of_greedy_five_one():
    nodes = nodes_with_spans([[span(0, 3)], [span(3, 6)]], speed=[1, 10])
    result = plan_ring(nodes, mesh(2), profile())
    assert [(s["lo"], s["hi"]) for s in result["stages"]] == [(0, 3), (3, 6)]
    assert all(s["runtime_config_sha256"] == "a" * 64 for s in result["stages"])
    assert result["planning"]["search"]["allocation"] == "exact measured span templates"


def test_joint_template_order_search_finds_an_alternative_to_the_cheapest_invalid_order():
    nodes = nodes_with_spans([[span(0, 2)], [span(4, 6)], [span(2, 4)]])
    result = plan_ring(nodes, [[0, 1, 8], [1, 0, 1], [1, 1, 0]], profile())
    assert result["order"] == ["n0", "n2", "n1"]
    assert [(s["lo"], s["hi"]) for s in result["stages"]] == [(0, 2), (2, 4), (4, 6)]


def test_actual_template_peaks_are_not_docked_again_by_scalar_reserves():
    nodes = nodes_with_spans([[span(0, 3)], [span(3, 6)]])
    p = {**profile(), "reserve_mb": 10000, "head_reserve_mb": 10000, "tail_reserve_mb": 10000,
         "load_peak_extra_mb": 10000, "require_exact_calibrations": True}
    result = plan_ring(nodes, mesh(2), p)
    assert result is not None
    assert all(stage["calibrated_resources"]["gpu_bytes"] == 100 for stage in result["stages"])


def test_native_weight_storage_floor_rejects_an_implausibly_small_signed_template():
    nodes = nodes_with_spans([[span(0, 3)], [span(3, 6)]])
    p = {**profile(), "require_exact_calibrations": True,
         "gpu_weight_storage_floor": dict(version=1, layer_bytes=[30] * 6, head_bytes=10, tail_bytes=10)}
    assert plan_ring(nodes, mesh(2), p) is not None
    nodes[1]["allowed_spans"][0]["gpu_bytes"] = 99
    assert plan_ring(nodes, mesh(2), p) is None


@pytest.mark.parametrize("host,pinned,ram,pin", [(70, 0, 100, 100), (50, 30, 100, 50)])
def test_exact_shared_host_and_pinned_reservations_are_aggregated(host, pinned, ram, pin):
    nodes = nodes_with_spans([[span(0, 3, host=host, pinned=pinned)], [span(3, 6, host=host, pinned=pinned)]],
        host_capacity=ram, pin_capacity=pin)
    assert plan_ring(nodes, mesh(2), profile()) is None


def test_unknown_required_memory_and_missing_templates_cannot_become_free_capacity():
    nodes = nodes_with_spans([[span(0, 3, host=1)], [span(3, 6)]])
    nodes[0]["resource_capacity"]["available_ram_bytes"] = None
    assert plan_ring(nodes, mesh(2), profile()) is None
    unknown = pool(["A", "A"])
    diagnostics = {}
    assert plan_ring(unknown, mesh(2), {**profile(), "require_exact_calibrations": True}, diagnostics=diagnostics) is None
    assert "exact stage calibrations" in diagnostics["reason"]


@pytest.mark.parametrize("index,nstages", [(2, 2), (2, 3)])
def test_stage_index_and_formation_width_are_part_of_executable_template(index, nstages):
    nodes = nodes_with_spans([[span(0, 3, index=0, nstages=2)], [span(3, 6, index=index, nstages=nstages)]])
    if index == nstages:
        with pytest.raises(ValueError):plan_ring(nodes, mesh(2), profile())
    else:
        assert plan_ring(nodes, mesh(2), profile()) is None


def test_valid_stage_geometry_and_gpu_uuid_guards_survive_template_search():
    nodes = nodes_with_spans([[span(0, 3, index=0, nstages=2)], [span(3, 6, index=1, nstages=2)]])
    assert plan_ring(nodes, mesh(2), profile()) is not None
    nodes[1]["gpu_uuid"] = nodes[0]["gpu_uuid"]
    assert plan_ring(nodes, mesh(2), profile()) is None


def add_calibration(body, lo, hi, *, stage=None, nstages=None, measured_at=1000, gpu=100):
    cfg = dict(lo=lo, hi=hi, head=lo == 0, tail=hi == 12, n_layers=12)
    if stage is not None:cfg["stage"] = stage
    if nstages is not None:cfg["nstages"] = nstages
    digest = hashlib.sha256(canonical(cfg)).hexdigest()
    req = PlacementRequirements("model/v1", lo, hi, GpuRequirements(resident_weights_bytes=gpu), HostRequirements(),
        CalibrationProvenance("weights-v1", digest, datetime.fromtimestamp(measured_at, timezone.utc).isoformat(),
            body["node_id"], "CPU schema fixture", "no hardware acceptance"))
    capability = body["models"][0]
    capability["profile"]["cap_layers"] = 12
    capability.setdefault("calibrations", []).append(dict(requirements=req.to_dict(), runtime_config=cfg))
    return digest


def test_registry_snapshot_exports_only_fresh_fitting_calibrated_templates():
    key = Ed25519PrivateKey.generate()
    body = offer(key)
    digest = add_calibration(body, 0, 6, stage=0, nstages=2)
    add_calibration(body, 0, 5, measured_at=600)
    add_calibration(body, 0, 4, gpu=2**31)
    registry = OfferRegistry(clock=lambda: 1000)
    registry.announce(sign_offer(body, key))
    node = registry.snapshot(model_cohort_id(cohort()))[0]
    assert len(node["allowed_spans"]) == 1
    row = node["allowed_spans"][0]
    assert (row["lo"], row["hi"], row["stage_index"], row["nstages"]) == (0, 6, 0, 2)
    assert row["runtime_config_sha256"] == digest
    assert node["resource_capacity"]["available_vram_bytes"] == 2**30


def test_same_span_different_runtime_widths_can_register_and_are_selected_by_exact_identity():
    key = Ed25519PrivateKey.generate()
    body = offer(key)
    first = add_calibration(body, 0, 6, stage=0, nstages=2)
    second = add_calibration(body, 0, 6, stage=0, nstages=3)
    registry = OfferRegistry(clock=lambda: 1000)
    registry.announce(sign_offer(body, key))
    assert {row["runtime_config_sha256"] for row in registry.snapshot(model_cohort_id(cohort()))[0]["allowed_spans"]} == {first, second}


def test_registry_to_real_planner_selects_the_available_templates_without_an_engine_import():
    registry = OfferRegistry(clock=lambda: 1000)
    for i, (lo, hi) in enumerate(((0, 5), (5, 12))):
        key = Ed25519PrivateKey.generate()
        body = offer(key, gpu=f"GPU-{i}")
        add_calibration(body, lo, hi, stage=i, nstages=2)
        registry.announce(sign_offer(body, key))
    nodes = registry.snapshot(model_cohort_id(cohort()))
    result = plan_ring(nodes, mesh(2), {**profile(12), "require_exact_calibrations": True})
    assert result is not None
    assert [(s["lo"], s["hi"]) for s in result["stages"]] == [(0, 5), (5, 12)]


def test_real_chunk_template_must_bind_the_measured_config_and_layer_range():
    from test_route_chunk_planning import chunk_nodes, chunk_work
    nodes = chunk_nodes()
    for i, node in enumerate(nodes):
        node["allowed_spans"] = [span(2*i, 2*i+2, index=i, nstages=3)]
        node["resource_capacity"] = dict(available_vram_bytes=1024, available_ram_bytes=0, pinnable_ram_bytes=0)
    p = {**profile(), "require_exact_calibrations": True}
    L = [[0, 1, 9000], [9000, 0, 1], [1, 9000, 0]]
    result = plan_ring(nodes, L, p, objective="pipeline", workload=chunk_work(), now=100)
    assert result is not None
    assert all(row["source"] == "fresh_stage_trace" for row in result["planning"]["stages"])
    nodes[1]["allowed_spans"][0]["runtime_config_sha256"] = "b" * 64
    assert plan_ring(nodes, L, p, objective="pipeline", workload=chunk_work(), now=100) is None


def test_template_frontier_preserves_a_rare_exact_middle_outside_legacy_latency_trim():
    rows = [[span(0, 2)] for _ in range(13)] + [[span(2, 4)], [span(4, 6)]]
    nodes = nodes_with_spans(rows)
    L = mesh(len(nodes), 1)
    for i in range(len(nodes)):
        L[i][13] = L[13][i] = 200
    result = plan_ring(nodes, L, {**profile(), "require_exact_calibrations": True}, locality="global")
    assert result is not None and "n13" in result["order"] and "n14" in result["order"]

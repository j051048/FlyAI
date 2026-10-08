"""Pure planner contracts: locality is a search tier, not an IP guess or admission ban."""
import json
import subprocess
import sys

import pytest

from shard.plan import plan_ring, V4_DUAL_RESOURCE_PROFILE, V4_ALL_RESIDENT_PROFILE, PROFILES
from shard.locality import link_snapshot
from shard.scheduler import Scheduler, JoinedNode


def profile(layers=6):
    return dict(model_id="fixture/model", n_layers=layers, layer_vram_mb=100,
        kv_mb_per_layer=0, layer_ms_base=1, reserve_mb=0, head_reserve_mb=0,
        tail_reserve_mb=0, cap_layers=layers, head_layer_ms_mult=1,
        decode_bytes=32, prefill_bytes=320, decode_steps=10)


def pool(regions, caps=None, speed=None):
    caps = caps or [3] * len(regions)
    return [dict(id=f"n{i}", region=region, free_vram_mb=caps[i] * 100,
                 layer_ms=(speed or [1] * len(regions))[i], gpu_uuid=f"GPU-{i}", host_id=f"host-{i}")
            for i, region in enumerate(regions)]


def mesh(n, latency=2):
    return [[0 if i == j else latency for j in range(n)] for i in range(n)]


def measurements(nodes, pairs, now=100):
    return {"schema": "shard-link-measurements/1", "edges": [
        dict(src=nodes[a]["id"], dst=nodes[b]["id"], rtt_ms=value,
             bandwidth_mbps=1000, measured_at=now, ttl_s=10)
        for a, b, value in pairs]}


def test_local_fit_wins_even_when_remote_cards_are_faster():
    nodes = pool(["A", "A", "B", "B"], speed=[2, 2, .1, .1])
    result = plan_ring(nodes, mesh(4), profile(), locality={"region": "A"})
    assert set(result["order"]) == {"n0", "n1"}
    assert result["planning"]["locality"]["selected_tier"] == "region"
    assert len(result["planning"]["locality"]["attempts"]) == 1


def test_expand_only_after_local_capacity_failed():
    nodes = pool(["A", "B", "B"], caps=[2, 2, 2])
    result = plan_ring(nodes, mesh(3), profile(), locality={"region": "A"})
    local = result["planning"]["locality"]
    assert local["selected_tier"] == "expanded"
    assert local["attempts"][0]["feasible"] is False
    assert local["expansion_reason"] == "no_local_feasible_plan"
    assert sum(stage["layers"] for stage in result["stages"]) == 6


def test_unknown_region_can_join_local_pool_by_measured_closeness_without_fake_label():
    nodes = pool(["A", None, "B"], caps=[3, 3, 3])
    edges = measurements(nodes, [(0, 1, 3), (1, 0, 3), (0, 2, 80), (2, 0, 80), (1, 2, 80), (2, 1, 80)])
    result = plan_ring(nodes, None, profile(), measurements=edges, now=100, locality={"region": "A", "max_rtt_ms": 10})
    assert set(result["order"]) == {"n0", "n1"}
    assert nodes[1]["region"] is None
    assert result["planning"]["locality"]["regions"] == ["A"]


def test_fresh_sparse_unknown_regions_search_latency_neighborhood_before_global():
    nodes = pool([None, None, None], caps=[3, 3, 3], speed=[1, 1, .1])
    edges = measurements(nodes, [(0, 1, 3), (1, 0, 3), (0, 2, 80), (2, 0, 80), (1, 2, 80), (2, 1, 80)])
    result = plan_ring(nodes, None, profile(), measurements=edges, now=100)
    assert set(result["order"]) == {"n0", "n1"}
    assert result["planning"]["locality"]["selected_tier"] == "measured_neighborhood"


def test_full_pool_central_remote_node_cannot_force_a_cross_region_head():
    nodes = pool(["A", "A", "B", "B", "B", "B"], caps=[3] * 6)
    rtt = mesh(6, 20)
    rtt[0][1] = rtt[1][0] = 5
    for i in range(6):
        if i != 2:rtt[i][2] = rtt[2][i] = 1
    result = plan_ring(nodes, rtt, profile(), locality={"region": "A"})
    assert result["head"] in {"n0", "n1"}
    assert result["coordinator_placement"]["preferred_host"] == result["head"]


def test_expired_sparse_measurements_cannot_fall_back_to_a_cheap_dense_zero():
    nodes = pool(["A", "A"])
    edges = measurements(nodes, [(0, 1, 1), (1, 0, 1)])
    assert plan_ring(nodes, mesh(2, 0), profile(), measurements=edges, now=111) is None


def test_missing_directed_return_edge_is_not_a_route():
    nodes = pool(["A", "A"])
    edges = measurements(nodes, [(0, 1, 1)])
    assert plan_ring(nodes, None, profile(), measurements=edges, now=100) is None


def test_local_head_is_chosen_with_role_capacity_not_full_network_centrality():
    nodes = pool(["A", "A", "A"], caps=[1, 8, 8])
    p = profile(); p["head_reserve_mb"] = 200
    result = plan_ring(nodes, [[0, 1, 1], [1, 0, 3], [1, 3, 0]], p)
    assert result is not None and result["head"] != "n0"
    assert result["planning"]["search"]["head_roles"] == "joint"


def test_tail_role_is_budgeted_per_candidate_not_permanently_docked_from_middle():
    nodes = pool(["A", "A", "A"], caps=[4, 4, 2])
    p = profile(7); p["tail_reserve_mb"] = 300
    result = plan_ring(nodes, mesh(3), p)
    assert result is not None
    assert result["order"][-1] != "n2"
    for stage in result["stages"]:
        available = nodes[int(stage["id"][1:])]["free_vram_mb"]
        assert stage["layers"] * 100 + (300 if stage["tail"] else 0) <= available


def test_three_32g_cards_not_rejected_by_clipping_soft_cap_before_boundary_reserves():
    nodes = [dict(id=f"n{i}", region="A", free_vram_mb=32768, total_vram_mb=32768,
        free_ram_mb=192 * 1024, pinnable_ram_mb=192 * 1024, host_id=f"h{i}", gpu_uuid=f"GPU-{i}") for i in range(3)]
    result = plan_ring(nodes, mesh(3), V4_DUAL_RESOURCE_PROFILE)
    assert result is not None and result["k"] == 3
    assert sum(stage["layers"] for stage in result["stages"]) == 43
    assert max(stage["layers"] for stage in result["stages"]) <= 15


def test_shared_ram_and_duplicate_gpu_constraints_survive_regional_search():
    nodes = pool(["A", "A", "A"], caps=[4, 4, 4])
    for node in nodes:
        node.update(host_id="shared", free_ram_mb=400, pinnable_ram_mb=400)
    p = profile(); p.update(placement="ram", layer_host_ram_mb=100)
    assert plan_ring(nodes, mesh(3), p) is None
    for node in nodes:node.update(free_ram_mb=800, pinnable_ram_mb=800)
    nodes[1]["gpu_uuid"] = nodes[0]["gpu_uuid"]
    result = plan_ring(nodes, mesh(3), p)
    assert not {"n0", "n1"} <= set(result["order"])


def test_explicit_other_model_never_becomes_v4(monkeypatch):
    monkeypatch.setitem(PROFILES, "fixture/future", profile(6))
    scheduler = Scheduler("v4", 6)
    scheduler.register(JoinedNode("a", 1, {"b": 2}, region="A"))
    scheduler.register(JoinedNode("b", 1, {"a": 2}, region="A"))
    result = scheduler.plan(model_id="fixture/future")
    assert sum(stage["layers"] for stage in result["stages"]) == 6
    assert result["model_id"] == "fixture/model"
    with pytest.raises(ValueError, match="no engine profile"):
        scheduler.plan(model_id="unknown/future")


def test_v4_transport_is_four_hc_streams_for_both_placements():
    for p in (V4_DUAL_RESOURCE_PROFILE, V4_ALL_RESIDENT_PROFILE):
        assert p["decode_bytes"] == 32768
        assert p["prefill_bytes"] == 4096 * 32768
        assert p["prefill_chunks"] == 1


def test_explicit_future_profile_wire_geometry_does_not_inherit_m25():
    p = profile(); p.pop("prefill_bytes"); p.pop("decode_bytes")
    nodes = pool(["A", "A"])
    for node in nodes:node["up_mbps"] = 1
    result = plan_ring(nodes, mesh(2), p)
    assert result["prefill_ms"] < 100  # Missing model geometry is unknown/zero estimate, never M25's 25MiB.


def test_explicit_scheduler_profile_identity_must_agree_with_model_id():
    scheduler = Scheduler("v4", 6)
    scheduler.register(JoinedNode("a", 1, {}, region="A"))
    with pytest.raises(ValueError, match="identity differs"):
        scheduler.plan(profile=profile(), model_id="another/model")


@pytest.mark.parametrize("bad", [float("nan"), -1, True])
def test_invalid_link_metrics_reject_before_search(bad):
    nodes = pool(["A", "A"])
    edges = measurements(nodes, [(0, 1, 1)])
    edges["edges"][0]["rtt_ms"] = bad
    with pytest.raises(ValueError):link_snapshot(nodes, None, edges, now=100)


def test_region_and_sparse_policy_json_cli_matches_library():
    nodes = pool(["A", "A"])
    edges = measurements(nodes, [(0, 1, 1), (1, 0, 1)])
    request = dict(nodes=nodes, model=profile(), locality={"region": "A"}, measurements=edges, now=100)
    out = subprocess.run([sys.executable, "-m", "shard.plan"], input=json.dumps(request), text=True, capture_output=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert json.loads(out.stdout)["planning"]["locality"]["selected_tier"] == "region"


def test_ten_thousand_open_offers_are_funnelled_before_any_dense_mesh(monkeypatch):
    import shard.locality as locality
    nodes = [dict(id=f"n{i}", region="A", free_vram_mb=300, gpu_uuid=f"GPU-{i}") for i in range(10_000)]
    nodes[-1]["free_vram_mb"] = 10_000
    sizes = []
    original = locality.link_snapshot
    def guard(selected, *args, **kwargs):
        sizes.append(len(selected))
        assert len(selected) <= 32, "open inventory must not become a 10k squared mesh"
        return original(selected, *args, **kwargs)
    monkeypatch.setattr(locality, "link_snapshot", guard)
    result = plan_ring(nodes, None, profile(), measurements={"schema": "shard-link-measurements/1", "edges": []}, now=100)
    assert result is not None and sizes == [32]
    assert result["order"] == ["n9999"]
    assert result["planning"]["search"]["truncated"] is True
    assert len(result["dropped"]) == 9999


def test_bounded_no_plan_returns_retry_diagnostics_not_global_infeasibility_claim():
    nodes = [dict(id=f"n{i}", region="A", free_vram_mb=100) for i in range(100)]
    diagnostics = {}
    result = plan_ring(nodes, None, profile(43), locality={"max_candidates": 16},
        measurements={"schema": "shard-link-measurements/1", "edges": []}, now=100, diagnostics=diagnostics)
    assert result is None
    assert diagnostics["status"] == "bounded_search_exhausted"
    assert diagnostics["search"]["examined_nodes"] == 16
    assert "widen" in diagnostics["reason"]


def test_new_policy_enforces_stage_limit_and_explicit_widening():
    nodes = pool(["A"] * 7, caps=[1] * 7)
    assert plan_ring(nodes, mesh(7), profile(7)) is None
    result = plan_ring(nodes, mesh(7), profile(7), locality={"max_stages": 7})
    assert result is not None and result["k"] == 7


def test_head_shortlist_filters_tiny_central_nodes_before_its_bound():
    nodes = pool(["A"] * 14, caps=[1] * 13 + [8])
    p = profile(); p["head_reserve_mb"] = 200
    result = plan_ring(nodes, mesh(14), p, locality={"max_head_candidates": 2})
    assert result is not None and result["head"] == "n13"


def test_raw_planner_cannot_combine_different_known_model_cohorts():
    nodes = pool(["A", "A"])
    nodes[0]["cohort_id"] = "model-one"
    nodes[1]["cohort_id"] = "model-two"
    with pytest.raises(ValueError, match="one exact model cohort"):
        plan_ring(nodes, mesh(2), profile())

"""Effective route channels and bound verify chunks, without GPU/remote work."""
from copy import deepcopy
import pytest

from shard.locality import link_snapshot, shortlist_candidates
from shard.plan import plan_ring
from shard.planning_cost import estimate, stage_observation
from test_planning_locality import pool, mesh, profile


def route(a, b, delay=2, *, route_id=None, kind="one_way", policy="measured_one_way", reachable=True, dialer=None):
    return dict(src=a, dst=b, route_id=route_id or f"{a}-{b}", channel="ssh_local",
        src_endpoint="127.0.0.1:39601", dst_endpoint="198.51.100.1:29501",
        dialer_id=dialer or a, reachable=reachable, latency_kind=kind, hop_policy=policy,
        latency_ms=delay, bandwidth_mbps=1000, measured_at=100, ttl_s=10)


def observations(*rows):
    return {"schema": "shard-link-measurements/2", "edges": list(rows)}


def chunk_work(**kw):
    return dict(frame_tokens=9, draft_tokens=8, context_tokens=2048, warmness="warm",
                depth=4, acceptance_gain=3, generated_tokens=64, cancel_rate=.5,
                stale_frames_per_valid=1.5, **kw)


def chunk_nodes():
    result = []
    for i in range(3):
        node = dict(id=f"n{i}", gpu_uuid=f"GPU-{i}", cohort_id="cohort", runtime_config_sha256="a" * 64,
                    free_vram_mb=200, cap_layers=2, up_mbps=1000, region="A")
        node["stage_trace"] = dict(schema="shard-stage-trace/2", node_id=node["id"], gpu_uuid=node["gpu_uuid"],
            cohort_id=node["cohort_id"], runtime_config_sha256=node["runtime_config_sha256"],
            layer_start=2*i, layer_end=2*i+2, frame_tokens=9, context_tokens=2048, warmness="warm",
            frame_ms=6+i, prefill_ms=20, measured_at=100, ttl_s=10)
        result.append(node)
    return result


def test_valid_directed_cycle_does_not_need_a_direct_bidirectional_head_star():
    nodes = pool(["A"] * 3, caps=[2] * 3)
    result = plan_ring(nodes, [[0, 2, 9000], [9000, 0, 2], [2, 9000, 0]], profile())
    assert result is not None
    matrix = [[0, 2, 9000], [9000, 0, 2], [2, 9000, 0]]
    order = [int(n[1:]) for n in result["order"]]
    assert all(matrix[a][b] < 9000 for a, b in zip(order, order[1:] + order[:1]))


def test_directional_chain_without_actual_return_still_cannot_form_local_coord_ring():
    nodes = pool(["A"] * 3, caps=[2] * 3)
    assert plan_ring(nodes, [[0, 2, 9000], [9000, 0, 2], [9000, 9000, 0]], profile()) is None


def test_external_coordinator_makes_a_valid_open_directed_stage_chain():
    nodes = pool(["A"] * 3, caps=[2] * 3)
    links = observations(route("coord", "n0", 5), route("n0", "n1", 2), route("n1", "n2", 3),
                         route("n2", "coord", 7, dialer="coord"))
    result = plan_ring(nodes, None, profile(), measurements=links, coordinator_id="coord", now=100)
    assert result is not None and result["order"] == ["n0", "n1", "n2"]
    assert result["coordinator_placement"]["preferred_host"] == "coord"
    assert len(result["planning"]["routes"]) == 4
    assert result["step_ms"] == 5 + 2 + 3 + 7 + 6


def test_missing_external_entry_or_return_does_not_become_a_one_ms_local_hop():
    nodes = pool(["A", "A"], caps=[3, 3])
    links = observations(route("coord", "n0"), route("n0", "n1"))
    assert plan_ring(nodes, None, profile(), measurements=links, coordinator_id="coord", now=100) is None


def test_rtt_policies_are_explicit_and_legacy_numbers_do_not_silently_halve():
    nodes = pool(["A", "A"])
    for policy, expected in (("conservative_rtt", 262), ("half_rtt_assumption", 131)):
        result = link_snapshot(nodes, None, observations(route("n0", "n1", 262, kind="rtt", policy=policy)), now=100)
        assert result["rtt"][0][1] == expected
    legacy = {"schema": "shard-link-measurements/1", "edges": [dict(src="n0", dst="n1", rtt_ms=262, measured_at=100, ttl_s=10)]}
    assert link_snapshot(nodes, None, legacy, now=100)["rtt"][0][1] == 262


def test_same_public_ip_can_have_a_blocked_direct_route_and_a_working_tunnel():
    nodes = pool(["A", "A"])
    for node in nodes:node["public_ip"] = "198.51.100.1"
    links = observations(route("n0", "n1", route_id="blocked", reachable=False),
                         route("n0", "n1", 3, route_id="tunnel"))
    snapshot = link_snapshot(nodes, None, links, now=100)
    assert snapshot["edges"]["n0", "n1"]["route_id"] == "tunnel"
    forced = link_snapshot(nodes, None, links, now=100, route_ids=[dict(src="n0", dst="n1", route_id="blocked")])
    assert forced["rtt"][0][1] == 9000


def test_k_plus_one_uses_measured_chunk_service_not_single_token_linear_speedup():
    nodes = chunk_nodes()
    result = plan_ring(nodes, [[0, 1, 9000], [9000, 0, 1], [1, 9000, 0]], profile(),
        objective="pipeline", workload=chunk_work(), now=100, locality={"max_head_candidates": 1})
    assert result is not None
    assert result["planning"]["frame_tokens"] == 9
    assert sum(s["service_ms"] for s in result["planning"]["stages"]) == 21
    assert all(s["source"] == "fresh_stage_trace" for s in result["planning"]["stages"])
    assert result["planning"]["prediction_only"] is True


@pytest.mark.parametrize("field,value", [("frame_tokens", 1), ("context_tokens", 1024), ("warmness", "cold"),
                                       ("gpu_uuid", "GPU-other"), ("runtime_config_sha256", "b" * 64)])
def test_wrong_chunk_binding_is_rejected(field, value):
    node = chunk_nodes()[0]
    node["stage_trace"][field] = value
    with pytest.raises(ValueError):stage_observation(node, 2, .01, now=100, start=0, workload=chunk_work())


def test_missing_or_expired_chunk_calibration_cannot_use_a_fake_fast_scalar():
    node = chunk_nodes()[0]; node["layer_ms"] = .0001
    for bad in ({}, {"stage_trace": deepcopy(node["stage_trace"])}):
        candidate = {**node, **bad}
        if not bad:candidate.pop("stage_trace")
        with pytest.raises(ValueError):stage_observation(candidate, 2, .0001, now=111, start=0, workload=chunk_work())


def test_chunk_depth_one_is_serial_and_stale_replay_refill_work_is_not_free():
    nodes = dict(enumerate(chunk_nodes()))
    model = dict(wire_bytes_per_token=0, prefill_bytes=0)
    def cost(objective, *, replay=0, cancel=0, refill=0):
        workload = dict(frame_tokens=9, draft_tokens=8, context_tokens=2048, warmness="warm",
            depth=1, acceptance_gain=3, generated_tokens=10, cancel_rate=cancel,
            stale_frames_per_valid=0, replay_frames_per_cancel=replay, refill_ms=refill)
        return estimate([0, 1, 2], {0: 2, 1: 2, 2: 2}, {0: 3, 1: 3, 2: 3},
            [[0, 2, 9000], [9000, 0, 3], [2, 9000, 0]], [5, 9000, 9000], [9000, 9000, 7],
            nodes, model, workload, now=100, objective=objective)
    serial, pipeline = cost("serial"), cost("pipeline")
    assert serial["predicted_request_ms"] == pipeline["predicted_request_ms"]
    assert cost("serial", replay=2, cancel=1, refill=4)["predicted_request_ms"] > serial["predicted_request_ms"]
    assert cost("pipeline", replay=2, cancel=1, refill=4)["predicted_request_ms"] > pipeline["predicted_request_ms"]


def test_route_selection_cannot_silently_overwrite_another_pinned_channel():
    nodes = pool(["A", "A"])
    links = observations(route("n0", "n1", route_id="direct"), route("n0", "n1", route_id="tunnel"))
    with pytest.raises(ValueError, match="conflicting"):
        link_snapshot(nodes, measurements=links, now=100, route_ids=[
            dict(src="n0", dst="n1", route_id="direct"), dict(src="n0", dst="n1", route_id="tunnel")])
    with pytest.raises(ValueError, match="selection"):
        link_snapshot(nodes, measurements=links, now=100, route_ids=[dict(src="n0", dst="n1")])


def test_v2_routes_can_preserve_an_anchor_neighbor_in_a_bounded_open_pool():
    nodes = pool(["A"] * 40, caps=[2] * 40)
    links = observations(route("external", "n39", 2), route("external", "n38", 1, reachable=False))
    selected, report = shortlist_candidates(nodes, profile(), dict(max_candidates=8, anchor_id="external"), links, now=100)
    assert 39 in selected and report["truncated"] is True and len(selected) == 8


def test_default_head_coordinator_return_is_in_the_selected_route_manifest():
    links = observations(route("n0", "n1"), route("n1", "n2"), route("n2", "n0"))
    result = plan_ring(pool(["A"] * 3, caps=[2] * 3), measurements=links, model=profile(), now=100)
    assert result["planning"]["coordinator_id"] == result["head"]
    pairs = {(row["src"], row["dst"]) for row in result["planning"]["routes"]}
    assert (result["order"][-1], result["head"]) in pairs
    assert all(row["measured_at"] == 100 and row["ttl_s"] == 10 for row in result["planning"]["routes"])

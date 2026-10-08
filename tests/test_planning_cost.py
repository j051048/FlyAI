"""Deterministic timing examples are estimator tests, not hardware acceptance."""
import pytest
from shard.planning_cost import simulate_frames, estimate, stage_observation, workload_spec
from shard.plan import plan_ring
from test_planning_locality import pool, mesh, profile


def stages(times, *, domain=None, busy=0, queue=0):
    return [dict(service_ms=t, queue_ms=queue, shared_id=domain, shared_ms=busy) for t in times]


def test_depth_one_really_is_serial_including_feedback():
    result = simulate_frames(stages([2, 3]), [4], [1], 2, 6, 1, 3)
    assert result == [18, 36, 54]


def test_finite_depth_fill_drain_and_feedback_are_not_just_max_stage():
    shallow = simulate_frames(stages([2, 3]), [4], [1], 2, 6, 2, 6)
    deep = simulate_frames(stages([2, 3]), [4], [1], 2, 6, 6, 6)
    assert shallow[-1] > deep[-1]
    assert deep[0] == shallow[0] == 18
    assert deep[-1] > 6 * 3  # Startup, links and return remain present.


def test_shared_dma_calendar_restricts_other_gpus_without_double_pricing_dma():
    independent = simulate_frames(stages([3, 3]), [0], [0], 0, 0, 8, 8)
    shared = simulate_frames(stages([3, 3], domain="host-pcie", busy=3), [0], [0], 0, 0, 8, 8)
    assert independent == [6 + i * 3 for i in range(8)]
    assert shared == [6 + i * 6 for i in range(8)]


def test_current_queue_cost_is_not_added_to_every_frame():
    reply = simulate_frames(stages([2], queue=10), [], [], 0, 0, 4, 4)
    assert reply == [12, 14, 16, 18]


def trace(lo, hi, frame=4, *, node="a", **kw):
    return dict(schema="shard-stage-trace/1", layer_start=lo, layer_end=hi,
                frame_ms=frame, measured_at=100, ttl_s=10, node_id=node,
                gpu_uuid=f"GPU-{node}", cohort_id="fixture-cohort", frame_tokens=1,
                runtime_config_sha256="a" * 64, **kw)


def test_matching_trace_and_scaled_trace_have_distinct_provenance():
    node = dict(id="a", gpu_uuid="GPU-a", cohort_id="fixture-cohort", runtime_config_sha256="a" * 64,
                stage_trace=trace(0, 4, frame=8))
    assert stage_observation(node, 4, 50, now=100, start=0)["source"] == "fresh_stage_trace"
    scaled = stage_observation(node, 2, 50, now=100, start=0)
    assert scaled["service_ms"] == 4 and scaled["source"] == "scaled_stage_trace"
    other_range = stage_observation(node, 4, 50, now=100, start=4)
    assert other_range["uncertainty"]
    expired = stage_observation(node, 4, 50, now=111)
    assert expired["service_ms"] == 200 and "expired" in expired["uncertainty"][0]


def prediction(**workload):
    nodes = {0: dict(id="a", gpu_uuid="GPU-a", cohort_id="fixture-cohort", runtime_config_sha256="a" * 64,
                    up_mbps=1000, stage_trace=trace(0, 2, prefill_ms=4)),
             1: dict(id="b", gpu_uuid="GPU-b", cohort_id="fixture-cohort", runtime_config_sha256="a" * 64,
                     up_mbps=1000, stage_trace=trace(2, 4, node="b", prefill_ms=4))}
    return estimate([0, 1], {0: 2, 1: 2}, {0: 2, 1: 2}, [[0, 1], [1, 0]], [1, 1], [1, 1],
        nodes, dict(decode_bytes=32, prefill_bytes=320), workload, now=100)


def test_acceptance_replay_and_refill_costs_are_accounted():
    full = prediction(depth=4, block_tokens=4, acceptance_gain=4, generated_tokens=100, cancel_rate=0)
    poor = prediction(depth=4, block_tokens=4, acceptance_gain=1, generated_tokens=100,
                      cancel_rate=1, replay_frames_per_cancel=2, refill_ms=3)
    assert poor["predicted_request_ms"] > full["predicted_request_ms"]
    assert full["prediction_only"] and "hardware acceptance" in full["scope"]
    assert full["bottleneck"]["kind"] in {"stage", "feedback_window"}


def test_mean_gain_does_not_invent_an_observed_rejection_distribution():
    result = prediction(depth=4, acceptance_gain=2, generated_tokens=20)
    assert any("distribution absent" in note for note in result["uncertainty"])


def test_heterogeneous_pipeline_balances_service_and_serial_keeps_sum():
    nodes = pool(["A", "A"], caps=[8, 8], speed=[1, 1.5])
    p = profile(8)
    # One GPU can fit all 8, so add a hard per-card cap to force the same two-stage topology.
    for node in nodes:node["cap_layers"] = 6
    serial = plan_ring(nodes, mesh(2, 0), p, locality="global")
    pipeline = plan_ring(nodes, mesh(2, 0), p, locality="global", objective="pipeline",
        workload=dict(depth=8, block_tokens=8, acceptance_gain=8, generated_tokens=100, cancel_rate=0), now=100)
    serial_counts = {s["id"]: s["layers"] for s in serial["stages"]}
    pipeline_counts = {s["id"]: s["layers"] for s in pipeline["stages"]}
    assert serial_counts == {"n0": 6, "n1": 2}
    assert pipeline_counts == {"n0": 5, "n1": 3}
    assert pipeline["planning"]["finite_depth"] == 8
    assert pipeline["step_ms"] >= serial["step_ms"]  # Optimize throughput without relabeling sum as max.


@pytest.mark.parametrize("workload", [{"depth": 0}, {"depth": True}, {"depth": 2, "acceptance_gain": 3},
                                      {"cancel_rate": 2}, {"replay_frames_per_cancel": -1}, {"frame_tokens": 8}])
def test_invalid_pipeline_assumptions_are_rejected(workload):
    with pytest.raises(ValueError):workload_spec(workload)


def test_pipeline_requires_explicit_workload():
    with pytest.raises(ValueError, match="explicit workload"):
        plan_ring(pool(["A", "A"]), mesh(2), profile(), objective="pipeline")


@pytest.mark.parametrize("key,value", [("node_id", "another-peer"), ("gpu_uuid", "GPU-other"),
                                      ("cohort_id", "another-model"), ("frame_tokens", 8),
                                      ("runtime_config_sha256", "b" * 64)])
def test_foreign_or_wrong_geometry_trace_cannot_supply_measured_service(key, value):
    node = dict(id="a", gpu_uuid="GPU-a", cohort_id="fixture-cohort", runtime_config_sha256="a" * 64,
                stage_trace=trace(0, 4))
    node["stage_trace"][key] = value
    with pytest.raises(ValueError, match="differs"):
        stage_observation(node, 4, 50, now=100, start=0)


def test_unbound_trace_is_explicitly_ignored_rather_than_a_free_fast_gpu():
    node = dict(id="a", stage_trace=trace(0, 4, frame=.001))
    result = stage_observation(node, 4, 50, now=100, start=0)
    assert result["source"] == "scalar_estimate" and result["service_ms"] == 200
    assert any("binding unavailable" in note for note in result["uncertainty"])


def test_context_config_hash_is_required_even_when_gpu_and_model_match():
    node = dict(id="a", gpu_uuid="GPU-a", cohort_id="fixture-cohort", stage_trace=trace(0, 4, frame=.001))
    result = stage_observation(node, 4, 50, now=100, start=0)
    assert result["source"] == "scalar_estimate" and result["service_ms"] == 200


def test_physical_shared_io_can_be_the_reported_bottleneck():
    nodes = {0: dict(id="a", gpu_uuid="GPU-a", cohort_id="fixture-cohort", runtime_config_sha256="a" * 64,
        stage_trace=trace(0, 2, prefill_ms=4, shared_resource_id="host", shared_busy_ms=4)),
        1: dict(id="b", gpu_uuid="GPU-b", cohort_id="fixture-cohort", runtime_config_sha256="a" * 64,
        stage_trace=trace(2, 4, node="b", prefill_ms=4, shared_resource_id="host", shared_busy_ms=4))}
    result = estimate([0, 1], {0: 2, 1: 2}, {0: 1, 1: 1}, [[0, 0], [0, 0]], [0, 0], [0, 0],
        nodes, dict(decode_bytes=0, prefill_bytes=0),
        dict(depth=8, block_tokens=8, acceptance_gain=8, generated_tokens=50, cancel_rate=0), now=100)
    assert result["bottleneck"] == {"kind": "shared_resource", "resource_id": "host", "service_ms": 8}


def test_actual_trace_speeds_seed_heterogeneous_partition_instead_of_equal_scalar_guesses():
    nodes = pool(["A", "A"], caps=[6, 6])
    for i, node in enumerate(nodes):
        node.pop("layer_ms")
        node.update(cohort_id="fixture-cohort", runtime_config_sha256="a" * 64)
        node["stage_trace"] = trace(0 if i == 0 else 4, 4 if i == 0 else 8, node=f"n{i}", frame=4 if i == 0 else 8)
        node["stage_trace"]["gpu_uuid"] = node["gpu_uuid"]
    result = plan_ring(nodes, mesh(2, 0), profile(8), locality="global", workload=dict(generated_tokens=20), now=100)
    counts = {s["id"]: s["layers"] for s in result["stages"]}
    assert counts == {"n0": 6, "n1": 2}
    assert any("reused for another block/range" in note for note in result["planning"]["uncertainty"])

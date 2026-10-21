from copy import deepcopy
import pytest
from shard.planning_cost import stage_observation
from shard.stage_cost import shape_trace, validate_surface


def node():
    sample = dict(context_tokens=4096, frame_tokens=1, warmness="warm",
                  frame_samples_ms=[40.] * 5, prefill_samples_ms=[100.] * 5,
                  layer_weights=[1., 2., 3., 4.])
    return dict(id="n", gpu_uuid="GPU-x", cohort_id="c", runtime_config_sha256="b" * 64,
        source_sha256="a" * 64, stage_trace=dict(schema="shard-stage-trace/3", node_id="n",
        gpu_uuid="GPU-x", cohort_id="c", runtime_config_sha256="b" * 64,
        layer_start=0, layer_end=4, measured_at=100, ttl_s=300, frame_ms=10,
        context_tokens=128, warmness="warm", frame_tokens=1,
        cost_surface=dict(schema="shard-stage-cost-surface/1", source_sha256="a" * 64, samples=[sample])))


def test_measured_layer_weights_change_heterogeneous_placement_cost_and_label_prediction():
    n = node(); w = dict(frame_tokens=1, context_tokens=2048, warmness="warm")
    early = stage_observation(n, 2, 1, now=150, start=0, workload=w, collect_quality=True)
    late = stage_observation(n, 2, 1, now=150, start=2, workload=w, collect_quality=True)
    assert early["service_ms"] == 12 and late["service_ms"] == 28
    assert early["source"] == "context_layer_cost_prediction"
    assert not early["exact_bound"] and early["quality"]["frame_p95_ms"] is None
    assert "context bucket differs" in early["uncertainty"][-1]


def test_full_exact_shape_can_be_used_but_stale_or_new_frame_never_gets_surface_price():
    n = node(); w = dict(frame_tokens=1, context_tokens=4096, warmness="warm")
    r = stage_observation(n, 4, 1, now=150, start=0, workload=w, collect_quality=True)
    assert r["service_ms"] == 40 and r["prefill_ms"] == 100 and r["exact_bound"]
    assert stage_observation(n, 4, 1, now=401, start=0, workload=w)["source"] == "scalar_estimate"
    with pytest.raises(ValueError, match="frame configuration"):
        stage_observation(n, 4, 1, now=150, start=0, workload=dict(w, frame_tokens=2))
    assert shape_trace(n["stage_trace"], dict(w, context_tokens=4097), start=0, layers=4,
                       source_sha256=n["source_sha256"]) is None


def test_shape_source_and_corrupt_measurements_fail_closed():
    n = node(); n["source_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="source differs"):
        stage_observation(n, 4, 1, now=150, start=0, workload=dict(context_tokens=4096, warmness="warm"))
    for change in (lambda s: s.update(layer_weights=[1]),
                   lambda s: s.update(frame_samples_ms=[float("nan")] * 5),
                   lambda s: s.update(context_tokens=True)):
        trace = deepcopy(node()["stage_trace"]); change(trace["cost_surface"]["samples"][0])
        with pytest.raises(ValueError): validate_surface(trace)

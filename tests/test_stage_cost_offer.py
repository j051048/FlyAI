"""Source-bound cost surfaces traverse real signed offer admission and planning."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from shard.offers import OfferError, OfferRegistry, canonical, model_cohort_id, sign_offer
from shard.planning_cost import stage_observation
from shard.resources import CalibrationProvenance, GpuRequirements, HostRequirements, PlacementRequirements
from test_node_offers import cohort, offer


def hashed(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def surface_offer():
    key = Ed25519PrivateKey.generate()
    body = offer(key)
    inventory = {"v4_stage.py": "a" * 64, "v4_native_batch.py": "b" * 64}
    source = hashed(inventory)
    config = {"lo": 0, "hi": 6, "head": True, "tail": False, "n_layers": 12,
              "environment": {}, "engine_source_sha256": inventory}
    runtime = hashed(config)
    provenance = CalibrationProvenance("weights-v1", runtime,
        datetime.fromtimestamp(1000, timezone.utc).isoformat(), body["node_id"], "CPU schema test", "explicit fixture")
    requirements = PlacementRequirements("model/v1", 0, 6,
        GpuRequirements(resident_weights_bytes=1024), HostRequirements(), provenance)
    capability = body["models"][0]
    capability["calibrations"] = [{"requirements": requirements.to_dict(), "runtime_config": config}]
    capability["profile"].update(source_sha256=source, runtime_config_sha256=runtime,
        stage_trace=dict(schema="shard-stage-trace/3", node_id=body["node_id"], gpu_uuid=body["gpu_uuid"],
            cohort_id=model_cohort_id(cohort()), runtime_config_sha256=runtime, layer_start=0, layer_end=6,
            measured_at=1000, ttl_s=120, frame_ms=10, frame_tokens=1, context_tokens=128, warmness="warm",
            cost_surface=dict(schema="shard-stage-cost-surface/1", source_sha256=source,
                samples=[dict(context_tokens=4096, frame_tokens=1, warmness="warm",
                    frame_samples_ms=[42.] * 5, prefill_samples_ms=[100.] * 5,
                    layer_weights=[1., 2., 3., 4., 5., 6.])])) )
    return body, key


def test_signed_cost_surface_survives_registry_and_prices_the_selected_layers():
    body, key = surface_offer()
    registry = OfferRegistry(clock=lambda: 1000)
    try:
        registry.announce(sign_offer(body, key))
        node, = registry.snapshot(model_cohort_id(cohort()))
        assert node["source_sha256"] == body["models"][0]["profile"]["source_sha256"]
        workload = dict(frame_tokens=1, context_tokens=4096, warmness="warm")
        full = stage_observation(node, 6, 1, now=1001, start=0, workload=workload, collect_quality=True)
        early = stage_observation(node, 2, 1, now=1001, start=0, workload=workload)
        late = stage_observation(node, 2, 1, now=1001, start=4, workload=workload)
        assert full["service_ms"] == 42 and full["source"] == "fresh_stage_trace" and full["exact_bound"]
        assert early["service_ms"] == 6 and late["service_ms"] == 22
        assert early["source"] == late["source"] == "context_layer_cost_prediction"
    finally:
        registry.close()


@pytest.mark.parametrize("change", ["missing_profile_source", "bad_profile_source", "foreign_surface_source",
    "foreign_trace_source", "missing_calibration", "different_runtime", "missing_inventory", "foreign_inventory"])
def test_re_signed_surface_cannot_bypass_runtime_inventory_binding(change):
    body, key = surface_offer()
    capability = body["models"][0]
    profile, record = capability["profile"], capability["calibrations"][0]
    if change == "missing_profile_source": profile.pop("source_sha256")
    elif change == "bad_profile_source": profile["source_sha256"] = "not-a-digest"
    elif change == "foreign_surface_source": profile["stage_trace"]["cost_surface"]["source_sha256"] = "f" * 64
    elif change == "foreign_trace_source": profile["stage_trace"]["source_sha256"] = "f" * 64
    elif change == "missing_calibration": capability.pop("calibrations")
    elif change == "different_runtime": record["requirements"]["provenance"]["runtime_config_sha256"] = "f" * 64
    else:
        if change == "missing_inventory": record["runtime_config"].pop("engine_source_sha256")
        else: record["runtime_config"]["engine_source_sha256"]["v4_stage.py"] = "f" * 64
        # Rehash the runtime and all its labels: the independent advertised
        # surface source still must equal the resulting calibrated inventory.
        runtime = hashed(record["runtime_config"])
        record["requirements"]["provenance"]["runtime_config_sha256"] = runtime
        profile["runtime_config_sha256"] = profile["stage_trace"]["runtime_config_sha256"] = runtime
    registry = OfferRegistry(clock=lambda: 1000)
    try:
        with pytest.raises(OfferError): registry.announce(sign_offer(body, key))
        assert registry.active() == []
    finally:
        registry.close()


@pytest.mark.parametrize("corruption", ["missing_profile_source", "foreign_inventory", "malformed_provenance"])
def test_old_persisted_surface_cannot_skip_source_binding_or_poison_honest_pool(corruption):
    body, key = surface_offer()
    registry = OfferRegistry(clock=lambda: 1000)
    try:
        registry.announce(sign_offer(body, key))
        honest = offer(gpu="GPU-honest")
        registry.announce(honest)
        saved = deepcopy(body)
        capability = saved["models"][0]
        if corruption == "missing_profile_source": capability["profile"].pop("source_sha256")
        elif corruption == "malformed_provenance": capability["calibrations"][0]["requirements"]["provenance"] = "bad"
        else: capability["calibrations"][0]["runtime_config"]["engine_source_sha256"]["v4_stage.py"] = "f" * 64
        registry._db.execute("UPDATE offers SET body=? WHERE node_id=?", (json.dumps(saved), body["node_id"]))
        assert [node["id"] for node in registry.snapshot(model_cohort_id(cohort()))] == [honest["node_id"]]
    finally:
        registry.close()


def test_existing_offer_without_cost_surface_keeps_legacy_contract():
    from test_chunk_offer_admission import traced_offer
    body, key = traced_offer()
    registry = OfferRegistry(clock=lambda: 1000)
    try:
        registry.announce(sign_offer(body, key))
        node, = registry.snapshot(model_cohort_id(cohort()))
        assert "source_sha256" not in node and node["stage_trace"]["schema"] == "shard-stage-trace/2"
    finally:
        registry.close()

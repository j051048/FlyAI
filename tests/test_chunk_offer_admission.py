"""Untrusted observations cannot poison a cohort's healthy planning candidates."""
from copy import deepcopy
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from shard.offers import OfferError, OfferRegistry, model_cohort_id, sign_offer
from test_node_offers import cohort, offer


def traced_offer(*, scalar=True, **changes):
    key = Ed25519PrivateKey.generate()
    body = offer(key)
    profile = body["models"][0]["profile"]
    if not scalar:
        profile.pop("layer_ms")
    profile["runtime_config_sha256"] = "a" * 64
    profile["stage_trace"] = dict(schema="shard-stage-trace/2", node_id=body["node_id"],
        gpu_uuid=body["gpu_uuid"], cohort_id=model_cohort_id(cohort()), runtime_config_sha256="a" * 64,
        frame_tokens=9, context_tokens=2048, warmness="warm", layer_start=0, layer_end=6,
        frame_ms=12, prefill_ms=20, measured_at=1000, ttl_s=10)
    profile["stage_trace"].update(changes)
    return body, key


@pytest.mark.parametrize("scalar", [True, False])
def test_bound_chunk_observation_can_supply_an_eligible_open_offer(scalar):
    body, key = traced_offer(scalar=scalar)
    registry = OfferRegistry(clock=lambda: 1000)
    registry.announce(sign_offer(body, key))
    nodes = registry.snapshot(model_cohort_id(cohort()))
    assert len(nodes) == 1
    assert nodes[0]["stage_trace"]["frame_tokens"] == 9
    assert nodes[0]["hardware_attested"] is False


@pytest.mark.parametrize("changes", [dict(node_id="other/GPU"), dict(gpu_uuid="GPU-other"),
    dict(cohort_id="b"*64), dict(runtime_config_sha256="c"*64), dict(frame_tokens=True),
    dict(context_tokens=True), dict(warmness="very-warm"), dict(layer_end=13),
    dict(frame_ms=float("inf")), dict(measured_at=980), dict(shared_busy_ms=13)])
def test_malformed_mismatched_or_stale_trace_is_rejected_before_pool_planning(changes):
    body, key = traced_offer(**changes)
    registry = OfferRegistry(clock=lambda: 1000)
    with pytest.raises((OfferError, ValueError)):
        registry.announce(sign_offer(body, key))
    assert registry.active() == []


@pytest.mark.parametrize("scalar", [True, False])
def test_unbound_capability_can_register_but_is_not_eligible_even_with_scalar_speed(scalar):
    body, key = traced_offer(scalar=scalar)
    body["models"][0]["profile"].pop("runtime_config_sha256")
    registry = OfferRegistry(clock=lambda: 1000)
    registry.announce(sign_offer(body, key))
    assert len(registry.active()) == 1
    assert registry.snapshot(model_cohort_id(cohort())) == []


def test_trace_expiry_after_admission_does_not_poison_healthy_nodes():
    clock = [1000]
    registry = OfferRegistry(clock=lambda: clock[0])
    body, key = traced_offer()
    registry.announce(sign_offer(body, key))
    healthy = offer(gpu="GPU-healthy")
    registry.announce(healthy)
    clock[0] = 1011
    assert [n["id"] for n in registry.snapshot(model_cohort_id(cohort()))] == [healthy["node_id"]]


def test_legacy_persisted_bad_trace_is_quarantined_at_snapshot():
    registry = OfferRegistry(clock=lambda: 1000)
    body, key = traced_offer()
    registry.announce(sign_offer(body, key))
    healthy = offer(gpu="GPU-healthy")
    registry.announce(healthy)
    corrupt = deepcopy(body)
    corrupt["models"][0]["profile"]["stage_trace"]["frame_ms"] = "bad"
    registry._db.execute("UPDATE offers SET body=? WHERE node_id=?", (json.dumps(corrupt), body["node_id"]))
    assert [n["id"] for n in registry.snapshot(model_cohort_id(cohort()))] == [healthy["node_id"]]

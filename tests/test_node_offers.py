import copy
from concurrent.futures import ThreadPoolExecutor

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from shard.offers import (ModelCohort, OfferError, OfferRegistry, load_sidecar_key,
                          model_cohort_id, peer_id_from_public_key, sign_offer, validate_offer)


def cohort(model="model/v1"):
    return ModelCohort(model, "1" * 64, "weights-v1", "2" * 64, "fp4", "runtime/1",
                       "activation/1", "exact-token/1", 12).to_dict()


def offer(key=None, *, gpu="GPU-0", seq=1, now=1000, models=None):
    key = key or Ed25519PrivateKey.generate()
    body = {"gpu_uuid": gpu, "memory_domain_id": "host-0", "host_id": "host-0",
            "endpoints": ["/ip4/198.51.100.1/udp/1234/quic-v1"], "region": "asia",
            "resources": {"available_vram_bytes": 2**30, "available_ram_bytes": 2**32,
                          "pinnable_ram_bytes": 2**31, "available_disk_bytes": None,
                          "measured_at": now}, "issued_at": now, "ttl_s": 120, "sequence": seq,
            "models": models if models is not None else [
                {"cohort": cohort(), "profile": {"layer_ms": 1, "cap_layers": 6}, "measured_at": now}]}
    return sign_offer(body, key)


def test_any_identity_can_register_and_unmeasured_nodes_remain_in_pool():
    registry = OfferRegistry(clock=lambda: 1000)
    measured, waiting = offer(), offer(models=[])
    registry.announce(measured)
    registry.announce(waiting)
    assert len(registry.active()) == 2
    nodes = registry.snapshot(model_cohort_id(cohort()))
    assert [n["id"] for n in nodes] == [measured["node_id"]]
    assert nodes[0]["hardware_attested"] is False
    assert nodes[0]["free_vram_mb"] == 1024


@pytest.mark.parametrize("mutate", [
    lambda b: b.update(region="europe"),
    lambda b: b.update(peer_id="other"),
    lambda b: b.update(node_id="other/GPU-0"),
    lambda b: b["models"][0]["cohort"].update(config_sha256="3" * 64),
    lambda b: b["resources"].update(available_vram_bytes=2**40),
])
def test_tampered_offer_cannot_relabel_identity_hardware_or_model(mutate):
    body = offer()
    mutate(body)
    with pytest.raises(OfferError):
        validate_offer(body, now=1000)


def test_expiry_does_not_erase_sequence_tombstone_on_restart(tmp_path):
    clock = [1000]
    key = Ed25519PrivateKey.generate()
    path = tmp_path / "registry.sqlite"
    registry = OfferRegistry(path, clock=lambda: clock[0])
    registry.announce(offer(key, seq=4))
    registry.close()
    clock[0] = 2000
    registry = OfferRegistry(path, clock=lambda: clock[0])
    assert registry.active() == []
    with pytest.raises(OfferError, match="sequence"):
        registry.announce(offer(key, seq=3, now=2000))
    registry.announce(offer(key, seq=5, now=2000))
    assert len(registry.active()) == 1


def test_concurrent_announcements_do_not_roll_back_a_newer_offer(tmp_path):
    key = Ed25519PrivateKey.generate()
    registries = [OfferRegistry(tmp_path / "r.sqlite", clock=lambda: 1000) for _ in range(2)]
    def put(i):
        try:
            registries[i % 2].announce(offer(key, seq=i + 1))
        except OfferError:
            pass
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(put, range(20)))
    assert registries[0].active()[0]["sequence"] == 20


def test_registration_is_idempotent_but_sequence_equivocation_is_rejected():
    key = Ed25519PrivateKey.generate()
    registry = OfferRegistry(clock=lambda: 1000)
    body = offer(key)
    registry.announce(body)
    assert registry.announce(body)["unchanged"]
    changed = copy.deepcopy(body)
    changed["region"] = "europe"
    with pytest.raises(OfferError, match="sequence"):
        registry.announce(sign_offer(changed, key))


def test_departed_identities_release_registration_slots_without_erasing_recent_sequences():
    clock = [1000]
    registry = OfferRegistry(clock=lambda: clock[0], max_nodes=1)
    key = Ed25519PrivateKey.generate()
    old = offer(key, seq=4)
    registry.announce(old)
    with pytest.raises(OfferError, match="capacity"):
        registry.announce(offer())
    clock[0] = 1200
    registry.announce(offer(now=1200))
    assert len(registry.active()) == 1
    with pytest.raises(OfferError, match="sequence"):
        registry.announce(offer(key, seq=3, now=1200))


def test_model_version_quantization_and_wire_are_separate_cohorts():
    base = cohort()
    for name, value in (("model_id", "model/v2"), ("quantization", "bf16"),
                        ("wire_version", "activation/2"), ("checkpoint_id", "weights-v2")):
        other = {**base, name: value}
        assert model_cohort_id(other) != model_cohort_id(base)
    registry = OfferRegistry(clock=lambda: 1000)
    registry.announce(offer())
    assert registry.snapshot(model_cohort_id(cohort("model/v2"))) == []


def test_unknown_ram_is_preserved_and_stale_compute_measurements_are_not_eligible():
    key = Ed25519PrivateKey.generate()
    body = offer(key)
    body["resources"].update(available_ram_bytes=None, pinnable_ram_bytes=None)
    body["models"][0]["measured_at"] = 600
    registry = OfferRegistry(clock=lambda: 1000)
    registry.announce(sign_offer(body, key))
    assert registry.snapshot(model_cohort_id(cohort())) == []


def test_sidecar_identity_is_reused_and_corrupt_key_is_rejected(tmp_path):
    key = Ed25519PrivateKey.generate()
    raw = key.private_bytes_raw() + key.public_key().public_bytes_raw()
    path = tmp_path / "node.key"
    path.write_bytes(b"\x08\x01\x12\x40" + raw)
    loaded = load_sidecar_key(path)
    assert peer_id_from_public_key(loaded.public_key().public_bytes_raw()) == offer(key)["peer_id"]
    path.write_bytes(b"\x08\x01\x12\x40" + raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(OfferError, match="public half"):
        load_sidecar_key(path)


@pytest.mark.parametrize("trace", [
    {"schema": "bad"},
    {"schema": "shard-stage-trace/1", "measured_at": 1000, "ttl_s": 100,
     "layer_start": 0, "layer_end": 2, "frame_ms": 1, "node_id": "another-node"},
    {"schema": "shard-stage-trace/1", "measured_at": 1000, "ttl_s": 100,
     "layer_start": 0, "layer_end": 200, "frame_ms": 1},
])
def test_malformed_or_cross_node_trace_is_rejected_before_it_can_poison_the_pool(trace):
    key = Ed25519PrivateKey.generate()
    malicious = offer(key)
    malicious["models"][0]["profile"]["stage_trace"] = trace
    registry = OfferRegistry(clock=lambda: 1000)
    honest = offer()
    registry.announce(honest)
    with pytest.raises(OfferError, match="stage trace"):
        registry.announce(sign_offer(malicious, key))
    assert len(registry.snapshot(model_cohort_id(cohort()))) == 1


@pytest.mark.parametrize("change", [
    {"ttl_s": 0}, {"ttl_s": 301}, {"issued_at": 1031}, {"sequence": True},
    {"sequence": 0}, {"issued_at": float("nan")},
])
def test_invalid_time_or_sequence_never_enters_registry(change):
    key = Ed25519PrivateKey.generate()
    body = offer(key)
    body.update(change)
    with pytest.raises(OfferError):
        validate_offer(sign_offer(body, key), now=1000)

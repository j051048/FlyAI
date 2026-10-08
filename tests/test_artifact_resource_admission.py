"""Real file geometry and shared SQLite budgets; no native GPU acceptance claim."""
from dataclasses import replace
from datetime import datetime, timezone
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from shard import weight_artifacts as W
from shard.leases import LeaseLedger, LeaseRequest, LeaseResources, PrepareRequest, PrepareResources, LeaseConflict, LeaseExpired
from shard.resources import (StorageRequirements, ResourceError, NodeResources, CalibrationProvenance,
                             PlacementRequirements, GpuRequirements, HostRequirements, evaluate_fit)
from shard.offers import ModelCohort, OfferRegistry, sign_offer, canonical, peer_id_from_public_key
from shard.plan import plan_ring
from engines.deepseek_v4.v4_resources import storage_requirements_from_artifacts
import hashlib


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "source"; directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"n_layers": 3, "n_mtp_layers": 3, "dspark_target_layer_ids": [2]}))
    tensors = {f"layers.{index}.weight": torch.arange(250000, dtype=torch.int32) for index in range(3)}
    tensors.update({name: torch.ones(4) for name in ("embed.weight", "head.weight", "norm.weight",
        "hc_head_fn", "hc_head_base", "hc_head_scale")})
    tensors.update({f"mtp.{index}.weight": torch.ones(4) for index in range(3)})
    save_file(tensors, str(directory / "model0-mp1.safetensors"))
    catalog, pack = W.catalogue_directory(directory, "fixture/resource-geometry")
    W.write_metadata(directory, catalog, pack)
    return directory, catalog, pack


def storage(source, lo=0, hi=1, filesystem="volume", **roles):
    _, catalog, pack = source
    return storage_requirements_from_artifacts(catalog, pack, lo, hi, filesystem_id=filesystem, **roles)


def req(storage, node="node", now=1000):
    return PlacementRequirements(storage.model_id, storage.layer_start, storage.layer_end,
        GpuRequirements(resident_weights_bytes=10), HostRequirements(reserve_bytes=10),
        CalibrationProvenance(storage.checkpoint_id, "b" * 64,
            datetime.fromtimestamp(now, timezone.utc).isoformat(), node, "resource-vector fixture",
            "local unit test; no GPU measurement or speed claim"))


def test_full_source_file_does_not_become_selected_tensor_bytes(source):
    directory, catalog, pack = source
    requirement = storage(source, head=True)
    whole = (directory / "model0-mp1.safetensors").stat().st_size
    assert sum(row.size for row in requirement.file_records) >= whole
    assert requirement.storage_bytes >= whole
    assert whole > sum(catalog["tensors"][name]["size"] for name in ("layers.0.weight", "embed.weight"))
    assert all(row.sha256 != "0" * 64 for row in requirement.file_records)
    assert StorageRequirements.from_dict(json.loads(json.dumps(requirement.to_dict()))) == requirement


def test_147_over_167_capacity_admits_actual_bounded_range_geometry(source):
    requirement = storage(source, head=True)
    quota = requirement.storage_bytes * 147 // 167
    capacity = NodeResources(100, 1 << 28, 1 << 28, quota)
    assert not evaluate_fit(req(requirement), capacity, requirement, preparation_mode="fetch")["fits"]
    result = evaluate_fit(req(requirement), capacity, requirement)
    assert result["fits"] and result["preparation_mode"] == "range_repack"
    assert result["required_disk_bytes"] < quota < requirement.storage_bytes


def test_repacked_real_verified_cache_has_zero_new_materialization_cost(source, tmp_path):
    directory, catalog, pack = source
    requirement = storage(source, head=True)
    selected = W.select_stage_artifacts(catalog, pack, 0, 1, head=True)
    def read(row, offset, size):
        with (directory / row["path"]).open("rb") as stream:
            stream.seek(offset); return stream.read(size)
    W.repack_stage(catalog, pack, selected, tmp_path / "cached", read)
    witness = W.verify_stage_artifacts(tmp_path / "cached")
    assert witness["artifact_id"] != requirement.artifact_id
    budget = requirement.preparation_budget(witness)
    assert budget["disk_peak_bytes"] == budget["ram_bytes"] == budget["pinned_bytes"] == 0
    assert budget["verified_existing_bytes"] < requirement.storage_bytes
    assert evaluate_fit(req(requirement), NodeResources(10, 10, 0, 0), requirement, verified_existing=witness)["fits"]
    with pytest.raises(ValueError): requirement.preparation_budget(dict(witness))
    payload = next((tmp_path / "cached").glob("*.safetensors"))
    with payload.open("r+b") as handle:
        handle.seek(-1, 2); handle.write(b"changed")
    with pytest.raises(ValueError): requirement.preparation_budget(witness)


@pytest.mark.parametrize("change", [lambda body: body["file_records"][0].update(path="../escape"),
    lambda body: body["file_records"][0].update(sha256="not-a-hash"),
    lambda body: body.update(storage_bytes=0), lambda body: body.pop("prepare_ram_bytes")])
def test_malformed_serialized_disk_contract_fails_closed(source, change):
    body = json.loads(json.dumps(storage(source, head=True).to_dict())); change(body)
    with pytest.raises(ResourceError): StorageRequirements.from_dict(body)


def ledger(tmp_path, now=None, node="node"):
    now = now or [1000.0]
    value = LeaseLedger(tmp_path / "ledger.sqlite", node_id=node,
        authorize=lambda principal, action, binding: principal if principal == "owner" else None,
        clock=lambda: now[0])
    value.register_capacity("host", available_ram_bytes=100, pinnable_ram_bytes=80,
                            gpu_capacity_bytes={"GPU-1": 100, "GPU-2": 100})
    value.register_filesystem("volume", tmp_path, available_disk_bytes=100)
    return value, now


def prepare(value, key="x", disk=60, ram=40, pinned=20):
    request = PrepareRequest("source-stage", "a" * 64, value.node_id, "host",
                             PrepareResources("volume", disk, ram, pinned), 10)
    return value.prepare_artifact(request, principal="owner", idempotency_key=key)


def test_prepare_and_runtime_share_host_quotas_both_directions(tmp_path):
    value, _ = ledger(tmp_path)
    runtime = LeaseRequest("old-ring", "a" * 64, value.node_id, "GPU-1", "host", LeaseResources(10, 50, 30), 100)
    old = value.prepare(runtime, principal="owner", idempotency_key="old")
    value.commit(old.lease_id, old.fencing_token, principal="owner")
    with pytest.raises(LeaseConflict, match="RAM"): prepare(value, ram=51)
    artifact = prepare(value)
    with pytest.raises(LeaseConflict, match="RAM"):
        value.prepare(replace(runtime, gpu_uuid="GPU-2", resources=LeaseResources(10, 20, 0)), principal="owner", idempotency_key="next")
    artifact.release()
    assert value.get(old.lease_id, principal="owner").state == "committed"


def test_expiry_restart_and_release_do_not_free_active_artifact_work(tmp_path):
    value, now = ledger(tmp_path)
    guard = prepare(value); work = guard.begin_work("exact-local-job")
    now[0] += 11
    with pytest.raises(LeaseExpired): guard.assert_live()
    restarted = LeaseLedger(value.path, node_id=value.node_id, authorize=value.authorize, clock=value.clock)
    with pytest.raises(LeaseConflict, match="disk"): prepare(restarted, "new")
    assert restarted.recover_stopped_artifact_work(lambda handle: False) == 0
    with pytest.raises(LeaseConflict): prepare(restarted, "new")
    assert restarted.recover_stopped_artifact_work(lambda handle: handle == work) == 1
    assert prepare(restarted, "new").fencing_token > guard.fencing_token


def test_multi_connection_disk_race_and_physical_alias_cannot_double_reserve(tmp_path):
    first, _ = ledger(tmp_path)
    second = LeaseLedger(first.path, node_id=first.node_id, authorize=first.authorize, clock=first.clock)
    with pytest.raises(LeaseConflict, match="one physical filesystem"):
        second.register_filesystem("another-label", tmp_path)
    def reserve(pair):
        try: return prepare(pair[0], pair[1])
        except LeaseConflict: return None
    with ThreadPoolExecutor(2) as workers:
        guards = list(workers.map(reserve, [(first, "a"), (second, "b")]))
    assert sum(guard is not None for guard in guards) == 1


def signed_pool(source, *, shared_filesystem=False, quota_override=None):
    _, catalog, _ = source
    cohort = ModelCohort(catalog["model_id"], catalog["manifest_sha256"], catalog["checkpoint_id"],
        catalog["config_sha256"], "fixture-bytes", "fixture-resource/1", "fixture-wire/1", "fixture-no-gpu/1", 3)
    registry = OfferRegistry(clock=lambda: 1000)
    for index in range(3):
        key = Ed25519PrivateKey.generate()
        node = peer_id_from_public_key(key.public_key().public_bytes_raw()) + f"/GPU-{index}"
        fs_id = "shared-volume" if shared_filesystem else f"volume-{index}"
        st = storage(source, index, index + 1, filesystem=fs_id, head=index == 0, tail=index == 2)
        cfg = dict(lo=index, hi=index+1, head=index == 0, tail=index == 2, dspark=False, n_layers=3, stage=index, nstages=3)
        measured = replace(req(st, node), provenance=replace(req(st, node).provenance,
            runtime_config_sha256=hashlib.sha256(canonical(cfg)).hexdigest()))
        quota = st.storage_bytes * 147 // 167
        if quota_override is not None:
            quota = quota_override
        body = dict(gpu_uuid=f"GPU-{index}", memory_domain_id=f"host-{index}", endpoints=["127.0.0.1:9000"],
            resources=dict(available_vram_bytes=100, available_ram_bytes=1 << 28, pinnable_ram_bytes=1 << 28,
                available_disk_bytes=None, measured_at=1000, filesystems={fs_id:dict(available_disk_bytes=quota, measured_at=1000)}),
            models=[dict(cohort=cohort.to_dict(), profile=dict(layer_ms=1, cap_layers=1), measured_at=1000,
                calibrations=[dict(requirements=measured.to_dict(), runtime_config=cfg, storage=st.to_dict())])],
            sequence=1, issued_at=1000, ttl_s=120)
        registry.announce(sign_offer(body, key))
    profile = dict(model_id=cohort.model_id, n_layers=3, layer_vram_mb=100, kv_mb_per_layer=0, layer_ms_base=1,
        reserve_mb=0, head_reserve_mb=0, tail_reserve_mb=0, cap_layers=3, head_layer_ms_mult=1, decode_bytes=32)
    return registry, cohort, profile


def test_signed_snapshot_repack_candidates_keep_model_identity_and_fs_budget(source, tmp_path):
    registry, cohort, profile = signed_pool(source)
    nodes = registry.snapshot(cohort.cohort_id)
    assert all(len(node["allowed_spans"]) == 1 for node in nodes)
    assert all(node["allowed_spans"][0]["preparation_mode"] == "range_repack" for node in nodes)
    result = plan_ring(nodes, [[0 if i == j else 2 for j in range(3)] for i in range(3)], profile)
    assert result is not None and len(result["stages"]) == 3
    assert all(stage["preparation_mode"] == "range_repack" for stage in result["stages"])
    assert all(stage["weight_artifacts"]["checkpoint_id"] == source[1]["checkpoint_id"] for stage in result["stages"])


def test_shared_filesystem_peak_is_summed_across_distinct_memory_domains(source):
    registry, cohort, profile = signed_pool(source, shared_filesystem=True)
    nodes = registry.snapshot(cohort.cohort_id)
    assert all(node["allowed_spans"] for node in nodes)
    assert plan_ring(nodes, [[0 if i == j else 2 for j in range(3)] for i in range(3)], profile) is None


def test_capacity_typed_refusal_requires_fresh_geometry_and_reachable_counterfactual(source):
    from shard.control_plane import FormationController, CapacityUnavailable, ControlError
    registry, cohort, profile = signed_pool(source, quota_override=0)
    nodes = registry.snapshot(cohort.cohort_id)
    assert all(node["capacity_rejected_spans"] and not node["allowed_spans"] for node in nodes)
    controller = FormationController(registry, object(),
        {node["id"]:SimpleNamespace(node_id=node["id"]) for node in nodes},
        requirements=lambda *_:pytest.fail("capacity refused before any lease"),
        backend_factory=lambda *_:pytest.fail("capacity refused before engine start"),
        maintain=False, clock=lambda:1000)
    matrix = [[0 if i == j else 2 for j in range(3)] for i in range(3)]
    with pytest.raises(CapacityUnavailable): controller.form("no-disk", cohort, profile, rtt=matrix)
    with pytest.raises(ControlError) as refused:
        controller.form("no-route", cohort, profile, rtt=[[0 if i == j else 9999 for j in range(3)] for i in range(3)])
    assert not isinstance(refused.value, CapacityUnavailable)

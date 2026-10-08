"""Real safetensors, exact byte identities and range reads; no GPU/model download."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file, load_file

from shard import weight_artifacts as W


@pytest.fixture
def native(tmp_path):
    root = tmp_path / "source"; root.mkdir()
    config = {"n_layers": 3, "n_mtp_layers": 3, "dspark_target_layer_ids": [2]}
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text('{"test": true}')
    values = {f"layers.{i}.weight": torch.arange(40, dtype=torch.float32) + i for i in range(3)}
    values.update({f"mtp.{i}.weight": torch.arange(4, dtype=torch.float32) for i in range(3)})
    values.update({name: torch.ones(4) for name in (
        "embed.weight", "head.weight", "norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale")})
    save_file(values, str(root / "model0-mp1.safetensors"))
    catalog, pack = W.catalogue_directory(root, "test/native")
    W.write_metadata(root, catalog, pack)
    return root, catalog, pack, values


def reader(root, calls=None):
    def read(row, offset, count):
        if calls is not None: calls.append((offset, count))
        with (root / row["path"]).open("rb") as stream:
            stream.seek(offset); return stream.read(count)
    return read


def test_native_catalogue_is_full_payload_evidence_and_preview_is_not(native):
    root, catalog, pack, values = native
    verified = W.verify_weight_pack(root, expected_checkpoint_id=catalog["checkpoint_id"])
    assert verified["payload_integrity_verified"] and verified["verification_scope"].startswith("complete")
    W.verified_stage_artifact_descriptor(verified)
    with pytest.raises(W.ArtifactError, match="verification"):
        W.verified_stage_artifact_descriptor(W.verify_weight_pack(root, verify_files=False))
    verified["checkpoint_id"] = "changed"
    with pytest.raises(W.ArtifactError, match="unchanged"):
        W.verified_stage_artifact_descriptor(verified)


def test_range_repacking_keeps_model_identity_and_only_assigned_payload(native, tmp_path):
    root, catalog, pack, values = native
    stage = W.select_stage_artifacts(catalog, pack, 0, 1, head=True)
    calls = []
    destination = tmp_path / "head"
    actual = W.repack_stage(catalog, pack, stage, destination, reader(root, calls),
                            max_file_bytes=64, chunk_bytes=17)
    verified = W.verify_stage_artifacts(destination, lo=0, hi=1, head=True, tail=False, dspark=False,
        expected_checkpoint_id=catalog["checkpoint_id"], expected_manifest_sha256=catalog["manifest_sha256"])
    assert actual["checkpoint_id"] == catalog["checkpoint_id"]
    assert actual["artifact_id"] != stage["artifact_id"]
    assert actual["tensor_names"] == ["embed.weight", "layers.0.weight"]
    assert max(n for _, n in calls) <= 17
    assert sum(row["size"] for row in actual["files"]) < sum(row["size"] for row in pack["files"])
    got = {}
    for file in destination.glob("*.safetensors"): got.update(load_file(str(file)))
    assert set(got) == set(actual["tensor_names"])
    for name, tensor in got.items(): assert torch.equal(tensor, values[name])
    assert W.verified_stage_artifact_descriptor(verified) == actual
    with pytest.raises(W.ArtifactError): W.verify_weight_pack(destination)


def test_tail_manifest_contains_complete_draft_and_alias_dependencies(native, tmp_path):
    root, catalog, pack, values = native
    stage = W.select_stage_artifacts(catalog, pack, 2, 3, tail=True, dspark=True)
    assert set(stage["tensor_names"]) == {"layers.2.weight", "mtp.0.weight", "mtp.1.weight", "mtp.2.weight",
        "embed.weight", "head.weight", "norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"}
    W.repack_stage(catalog, pack, stage, tmp_path / "tail", reader(root), max_file_bytes=32)
    W.verify_stage_artifacts(tmp_path / "tail", lo=2, hi=3, tail=True, dspark=True)
    with pytest.raises(W.ArtifactError, match="tap"):
        W.select_stage_artifacts(catalog, pack, 0, 2, dspark=True)


@pytest.mark.parametrize("change", ["catalog", "stage", "payload", "asset", "extra", "assignment"])
def test_changes_cannot_pass_stage_verification(native, tmp_path, change):
    root, catalog, pack, _ = native
    stage = W.select_stage_artifacts(catalog, pack, 0, 1, head=True)
    dest = tmp_path / change
    W.repack_stage(catalog, pack, stage, dest, reader(root))
    kwargs = {}
    if change == "catalog":
        body = W.read_json(dest / W.GLOBAL_FILE); body["tensors"]["layers.1.weight"]["sha256"] = "f" * 64
        (dest / W.GLOBAL_FILE).write_text(json.dumps(body))
    elif change == "stage":
        body = W.read_json(dest / W.STAGE_FILE); body["tail"] = True
        (dest / W.STAGE_FILE).write_text(json.dumps(body))
    elif change == "payload":
        path = next(dest.glob("*.safetensors")); data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    elif change == "asset": (dest / "config.json").write_text('{"n_layers": 99}')
    elif change == "extra": save_file({"rogue": torch.ones(4)}, str(dest / "model999999-mp1.safetensors"))
    elif change == "assignment": kwargs = {"lo": 1, "hi": 2}
    with pytest.raises((W.ArtifactError, ValueError)):
        W.verify_stage_artifacts(dest, **kwargs)


def test_corrupted_or_short_range_never_writes_completion_marker(native, tmp_path):
    root, catalog, pack, _ = native
    stage = W.select_stage_artifacts(catalog, pack, 0, 1, head=True)
    for label, source in (("short", lambda row, offset, length: b""),
                          ("wrong", lambda row, offset, length: b"\xff" * length)):
        dest = tmp_path / label
        with pytest.raises(W.ArtifactError): W.repack_stage(catalog, pack, stage, dest, source)
        assert not (dest / W.STAGE_FILE).exists()


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a\\b", "a//b", "a/./b", "C:/other", "bad\nname"])
def test_artifact_paths_are_safe(path):
    with pytest.raises(W.ArtifactError): W.relative_path(path)


def test_catalogue_and_pack_mutations_fail_closed(native):
    _, catalog, pack, _ = native
    changed = deepcopy(pack); changed["offsets"]["layers.0.weight"][1] += 1
    with pytest.raises(W.ArtifactError): W.validate_pack(catalog, changed)
    changed = deepcopy(pack); changed["files"].append(changed["files"][0])
    with pytest.raises(W.ArtifactError): W.validate_pack(catalog, changed)


def test_model_content_identity_does_not_depend_on_conversion_location(native):
    _, catalog, _, _ = native
    another = W.make_catalog(catalog["model_id"], catalog["config"], catalog["tensors"], catalog["assets"],
                            source={"kind": "another verified conversion provenance"})
    assert another["checkpoint_id"] == catalog["checkpoint_id"]
    assert another["manifest_sha256"] != catalog["manifest_sha256"]


def test_witness_seals_the_original_entire_verification_batch(native, monkeypatch):
    root, _, _, _ = native
    original = W.hash_file
    changed = False
    def mutate_an_already_verified_file(path, **kwargs):
        nonlocal changed
        result = original(path, **kwargs)
        if Path(path).name == "config.json" and not changed:
            file = root / "model0-mp1.safetensors"
            raw = bytearray(file.read_bytes()); raw[-1] ^= 1; file.write_bytes(raw)
            changed = True
        return result
    monkeypatch.setattr(W, "hash_file", mutate_an_already_verified_file)
    with pytest.raises(W.ArtifactError, match="changed.*witness"):
        W.verify_weight_pack(root)


def test_geometric_repack_budget_covers_actual_output_and_rename_preserves_proof(native, tmp_path):
    root, catalog, pack, _ = native
    stage = W.select_stage_artifacts(catalog, pack, 0, 1, head=True)
    bound = W.repack_storage_bound(catalog, pack, stage, 64, 17)
    staging = tmp_path / "staging"
    W.repack_stage(catalog, pack, stage, staging, reader(root), max_file_bytes=64, chunk_bytes=17)
    actual = sum(path.stat().st_size for path in staging.iterdir())
    assert bound["disk_peak_bytes"] >= actual and bound["host_ram_bytes"] > 0
    value = W.verify_stage_artifacts(staging)
    target = tmp_path / "published"; staging.rename(target)
    moved = W.relocate_verified_artifacts(value, target)
    assert W.verified_stage_artifact_descriptor(moved)["lo"] == 0
    file = next(target.glob("*.safetensors")); raw = file.read_bytes(); file.write_bytes(raw)
    with pytest.raises(W.ArtifactError, match="changed|replaced"):
        W.verified_stage_artifact_descriptor(moved)

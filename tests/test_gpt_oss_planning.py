"""Metadata fixtures and explicit budgets; no pretrained model or GPU involved."""
from copy import deepcopy
import hashlib
import json
import struct

import pytest
from shard.gpt_oss_planning import inspect_checkpoint, planning_profile
from shard.download_inventory import build_inventory


def checkpoint(path, *, layers=3, hidden=32):
    path.mkdir(parents=True)
    config = dict(model_type="gpt_oss", num_hidden_layers=layers, hidden_size=hidden, torch_dtype="bfloat16",
                  quantization_config={"quant_method": "mxfp4"})
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    names = [f"model.layers.{i}.mlp.experts.gate_up_proj_blocks" for i in range(layers)]
    names += ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    header, offset = {}, 0
    for name in names:
        header[name] = dict(dtype="U8", shape=[8], data_offsets=[offset, offset + 8]); offset += 8
    blob = json.dumps(header, separators=(",", ":")).encode()
    filename = "model-00001-of-00001.safetensors"
    (path / filename).write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(offset))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(names, filename)}), encoding="utf-8")
    return path


def calibration(inventory):
    return dict(schema="gpt-oss-planning-calibration/1", kind="measured",
        checkpoint_id=inventory["checkpoint_id"], config_sha256=inventory["config_sha256"],
        node_id="fixture-node", gpu_uuid="GPU-fixture", runtime_config_sha256="a" * 64,
        method="synthetic budget validation fixture", evidence="not hardware acceptance", measured_at=100, ttl_s=10,
        layer_vram_bytes=16, kv_bytes_per_layer=4, reserve_bytes=8, head_reserve_bytes=8,
        tail_reserve_bytes=16, load_peak_extra_bytes=4, cap_layers=3, layer_ms=2.5)


def test_structure_and_geometry_come_from_actual_config_and_header(tmp_path):
    inventory = inspect_checkpoint(checkpoint(tmp_path / "model", layers=5, hidden=48), model_id="openai/gpt-oss-fixture")
    assert inventory["n_layers"] == 5 and inventory["hidden_size"] == 48
    assert inventory["layer_storage_bytes"] == [8] * 5
    assert inventory["embedding_bytes"] == 8 and inventory["tail_bytes"] == 16
    assert inventory["payload_integrity_verified"] is False  # Header scan cannot vouch for weights.
    profile = planning_profile(inventory, calibration(inventory), now=100, prompt_tokens=128)
    assert profile["n_layers"] == 5 and profile["layer_ms_base"] == 2.5
    assert profile["decode_bytes"] == 48 * 2
    assert profile["prefill_bytes"] == 128 * 48 * 2
    assert profile["layer_vram_mb"] == 16 / 1024**2


@pytest.mark.parametrize("field,value", [("checkpoint_id", "another-model"), ("config_sha256", "b" * 64),
    ("layer_vram_bytes", 7), ("head_reserve_bytes", 0), ("tail_reserve_bytes", 0),
    ("kind", "estimated"), ("layer_ms", 0), ("runtime_config_sha256", "unbound")])
def test_false_or_underbudget_calibration_cannot_make_a_runtime_profile(tmp_path, field, value):
    inventory = inspect_checkpoint(checkpoint(tmp_path / "model"))
    evidence = calibration(inventory); evidence[field] = value
    with pytest.raises(ValueError):planning_profile(inventory, evidence, now=100)


def test_no_explicit_measurement_no_profile_and_stale_is_rejected(tmp_path):
    inventory = inspect_checkpoint(checkpoint(tmp_path / "model"))
    with pytest.raises(ValueError):planning_profile(inventory, None, now=100)
    with pytest.raises(ValueError, match="expired"):
        planning_profile(inventory, calibration(inventory), now=111)


def test_user_checkpoint_id_cannot_override_real_download_identity(tmp_path):
    path = checkpoint(tmp_path / "model")
    with pytest.raises((ValueError, OSError)):
        inspect_checkpoint(path, checkpoint_id="sha256:" + "a" * 64)


def test_mxfp4_is_not_inferred_from_directory_name(tmp_path):
    path = checkpoint(tmp_path / "gpt-oss-mxfp4")
    config = json.loads((path / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "bf16"}
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="MXFP4"):
        inspect_checkpoint(path)


def test_index_path_escape_is_rejected(tmp_path):
    path = checkpoint(tmp_path / "model")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    for key in index["weight_map"]:index["weight_map"][key] = "../outside.safetensors"
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="outside"):
        inspect_checkpoint(path)


def bind_download(path):
    rows = []
    for file in sorted(path.iterdir()):
        payload = file.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        rows.append(dict(path=file.name, size=len(payload), algorithm="sha256", digest=digest, payload_sha256=digest))
    inventory = build_inventory("openai/gpt-oss-fixture", "1" * 40, rows)
    (path / ".shard-download.json").write_text(json.dumps(inventory), encoding="utf-8")
    return inventory


def test_planning_metadata_and_complete_payload_verification_have_different_scope(tmp_path):
    path = checkpoint(tmp_path / "model")
    download = bind_download(path)
    metadata = inspect_checkpoint(path, checkpoint_id=download["checkpoint_id"], verify_payload=False)
    verified = inspect_checkpoint(path, checkpoint_id=download["checkpoint_id"], verify_payload=True)
    assert metadata["checkpoint_id"] == verified["checkpoint_id"] == download["checkpoint_id"]
    assert metadata["payload_integrity_verified"] is False
    assert verified["payload_integrity_verified"] is True
    assert verified["model_id"] == "openai/gpt-oss-fixture"
    weights = next(path.glob("*.safetensors"))
    payload = bytearray(weights.read_bytes()); payload[-1] ^= 1; weights.write_bytes(payload)
    # A cheap header inspection must not claim that a modified payload is good.
    assert inspect_checkpoint(path, verify_payload=False)["payload_integrity_verified"] is False
    with pytest.raises(ValueError, match="payload"):
        inspect_checkpoint(path)


def test_download_repository_and_local_config_identity_cannot_be_relabelled(tmp_path):
    path = checkpoint(tmp_path / "model")
    bind_download(path)
    with pytest.raises(ValueError, match="repository"):
        inspect_checkpoint(path, model_id="other/model", verify_payload=False)
    config = json.loads((path / "config.json").read_text())
    config["hidden_size"] = 64  # Same byte length: cheap identity checks must still bind config digest.
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="config"):
        inspect_checkpoint(path, verify_payload=False)


def test_wire_geometry_follows_auto_dtype_or_explicit_measured_bytes(tmp_path):
    path = checkpoint(tmp_path / "model")
    config = json.loads((path / "config.json").read_text())
    config["torch_dtype"] = "float32"
    (path / "config.json").write_text(json.dumps(config))
    inventory = inspect_checkpoint(path)
    p = planning_profile(inventory, calibration(inventory), now=100)
    assert p["decode_bytes"] == 32 * 4
    cal = calibration(inventory); cal["wire_element_bytes"] = 2
    with pytest.raises(ValueError, match="wire bytes"):
        planning_profile(inventory, cal, now=100)
    config.pop("torch_dtype")
    (path / "config.json").write_text(json.dumps(config))
    inventory = inspect_checkpoint(path)
    with pytest.raises(ValueError, match="wire_element_bytes"):
        planning_profile(inventory, calibration(inventory), now=100)
    cal = calibration(inventory); cal["wire_element_bytes"] = 2
    assert planning_profile(inventory, cal, now=100)["decode_bytes"] == 32 * 2

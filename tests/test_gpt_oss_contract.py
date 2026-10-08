"""Actual temporary-file cohort evidence and fail-closed compatibility checks."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from shard.download_inventory import (FILENAME, InventoryError, VerifiedInventory,
                                      build_inventory, verified_config, verify_inventory)
from shard.gpt_oss_contract import (GPTOSSContractError, NUMERIC_CONTRACT, QUANTIZATION,
                                    RUNTIME_ABI, WIRE_VERSION, build_cohort_from_inventory,
                                    validate_supported_cohort)


CONFIG = {"model_type": "gpt_oss", "num_hidden_layers": 24,
          "quantization_config": {"quant_method": "mxfp4", "dequantize": False}}


def checkpoint(directory, *, config=None, sharded=False, target="model-00001.safetensors"):
    files = {"config.json": json.dumps(config or CONFIG, indent=2).encode(),
             target if sharded else "model.safetensors": b"packed-checkpoint-fixture"}
    if sharded:
        files["model.safetensors.index.json"] = json.dumps({"weight_map": {"weight": target}}).encode()
    rows = []
    for path, data in files.items():
        output = directory / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        rows.append({"path": path, "size": len(data), "algorithm": "sha256",
                     "digest": digest, "payload_sha256": digest})
    marker = build_inventory("openai/gpt-oss-20b", "a" * 40, rows)
    (directory / FILENAME).write_text(json.dumps(marker), encoding="utf-8")
    return marker


def descriptor():
    return {"model_id": "openai/gpt-oss-20b", "manifest_sha256": "a" * 64,
            "checkpoint_id": "sha256:" + "a" * 64, "config_sha256": "b" * 64,
            "runtime_abi": RUNTIME_ABI, "wire_version": WIRE_VERSION,
            "numeric_contract": NUMERIC_CONTRACT, "quantization": QUANTIZATION, "n_layers": 24}


def test_public_recipe_binds_fullcontent_manifest_and_actual_raw_config(tmp_path):
    marker = checkpoint(tmp_path)
    inventory = verify_inventory(tmp_path)
    assert isinstance(inventory, VerifiedInventory)
    assert json.loads(json.dumps(inventory))["manifest_sha256"] == marker["manifest_sha256"]
    cohort = build_cohort_from_inventory(inventory)
    assert cohort.manifest_sha256 == marker["manifest_sha256"]
    assert cohort.checkpoint_id == "sha256:" + marker["manifest_sha256"]
    assert cohort.config_sha256 == hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    assert cohort.model_id == marker["repo"]
    assert validate_supported_cohort(cohort, CONFIG) == cohort.to_dict()
    config = verified_config(inventory)
    config["num_hidden_layers"] = 1
    assert verified_config(inventory)["num_hidden_layers"] == 24


def test_validator_import_requires_only_standard_library():
    # A clean interpreter actively rejects importing the optional model/GPU/
    # identity dependencies, rather than inheriting already-loaded test modules.
    source = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'cryptography', 'safetensors'}:
            raise RuntimeError('unexpected optional dependency ' + fullname)
sys.meta_path.insert(0, Block())
import shard.gpt_oss_contract as contract
assert contract.RUNTIME_ABI == 'gpt-oss-hf/1'
"""
    run = subprocess.run([sys.executable, "-c", source], cwd=Path(__file__).resolve().parents[1],
                         text=True, capture_output=True, timeout=20)
    assert run.returncode == 0, run.stderr


def test_same_config_with_different_payload_has_a_different_cohort(tmp_path):
    marker = checkpoint(tmp_path)
    first = build_cohort_from_inventory(verify_inventory(tmp_path))
    payload = b"a-second-packed-checkpoint"
    (tmp_path / "model.safetensors").write_bytes(payload)
    for row in marker["files"]:
        if row["path"] == "model.safetensors":
            digest = hashlib.sha256(payload).hexdigest()
            row.update(size=len(payload), digest=digest, payload_sha256=digest)
    (tmp_path / FILENAME).write_text(json.dumps(build_inventory(marker["repo"], marker["revision"], marker["files"])))
    second = build_cohort_from_inventory(verify_inventory(tmp_path))
    assert first.config_sha256 == second.config_sha256
    assert first.checkpoint_id != second.checkpoint_id
    assert first.cohort_id != second.cohort_id


@pytest.mark.parametrize("field", ["runtime_abi", "wire_version", "numeric_contract", "quantization"])
@pytest.mark.parametrize("value", ["unsupported/2", "", None, 1])
def test_every_unsupported_implementation_field_is_rejected(field, value):
    cohort = descriptor()
    cohort[field] = value
    with pytest.raises(GPTOSSContractError):
        validate_supported_cohort(cohort, CONFIG)


@pytest.mark.parametrize("change", [
    {"model_type": "llama"}, {"model_type": None}, {"quantization_config": None},
    {"quantization_config": {"quant_method": "bf16"}},
    {"quantization_config": {"quant_method": "mxfp4", "dequantize": True}},
    {"quantization_config": {"quant_method": "mxfp4", "dequantize": "false"}},
    {"num_hidden_layers": True}, {"num_hidden_layers": 0},
    {"num_hidden_layers": 24.0}, {"num_hidden_layers": 32},
])
def test_supported_labels_do_not_override_actual_config(change):
    config = {**copy.deepcopy(CONFIG), **change}
    with pytest.raises(GPTOSSContractError):
        validate_supported_cohort(descriptor(), config)


def test_cohort_exact_fields_types_and_no_fake_config_hash_validation():
    cohort = descriptor()
    assert validate_supported_cohort(cohort, CONFIG) == cohort
    # Caller hashes its raw config bytes separately: validation cannot derive
    # that hash from a parsed dict with altered whitespace/key order.
    cohort["config_sha256"] = "c" * 64
    assert validate_supported_cohort(cohort, CONFIG)["config_sha256"] == "c" * 64
    for malformed in ({**cohort, "extra": True}, {k: v for k, v in cohort.items() if k != "wire_version"},
                      {**cohort, "n_layers": True}, {**cohort, "manifest_sha256": "invalid"}):
        with pytest.raises(GPTOSSContractError):
            validate_supported_cohort(malformed, CONFIG)


def test_declared_json_and_header_inventory_cannot_become_verified_cohort(tmp_path):
    marker = checkpoint(tmp_path)
    checked = verify_inventory(tmp_path)
    for inventory in (marker, dict(checked), json.loads(json.dumps(checked)),
                      verify_inventory(tmp_path, verify_files=False),
                      {**dict(checked), "schema": "gpt-oss-storage-inventory/1", "payload_integrity_verified": True}):
        with pytest.raises(GPTOSSContractError, match="complete local file verification"):
            build_cohort_from_inventory(inventory)
    with pytest.raises(InventoryError, match="actual file verification"):
        VerifiedInventory(dict(checked))


@pytest.mark.parametrize("mutation", ["identity", "nested_file", "verification_flag"])
def test_mutating_verified_evidence_cannot_change_the_cohort(tmp_path, mutation):
    checkpoint(tmp_path)
    checked = verify_inventory(tmp_path)
    if mutation == "identity":
        checked["checkpoint_id"] = "sha256:" + "c" * 64
    elif mutation == "nested_file":
        checked["files"][0]["payload_sha256"] = "c" * 64
    else:
        checked["payload_integrity_verified"] = False
    with pytest.raises(GPTOSSContractError, match="complete local file verification"):
        build_cohort_from_inventory(checked)


@pytest.mark.parametrize("sharded", [False, True])
def test_unreferenced_nested_original_and_metal_backups_are_allowed(tmp_path, sharded):
    marker = checkpoint(tmp_path, sharded=sharded)
    for folder in ("original", "metal"):
        nested = tmp_path / folder
        nested.mkdir()
        (nested / "model.safetensors").write_bytes(b"unlisted-backup")
        (nested / "config.json").write_bytes(b'{"model_type":"other"}')
    assert verify_inventory(tmp_path)["checkpoint_id"] == marker["checkpoint_id"]


def test_unlisted_root_override_remains_rejected_with_nested_backups(tmp_path):
    checkpoint(tmp_path, sharded=True)
    (tmp_path / "original").mkdir()
    (tmp_path / "original" / "model.safetensors").write_bytes(b"irrelevant-backup")
    (tmp_path / "model.safetensors").write_bytes(b"loader-override")
    with pytest.raises(InventoryError, match="unlisted safetensors"):
        verify_inventory(tmp_path)


def test_index_cannot_reference_an_unlisted_nested_backup(tmp_path):
    marker = checkpoint(tmp_path, sharded=True)
    (tmp_path / "metal").mkdir()
    (tmp_path / "metal" / "model.safetensors").write_bytes(b"unverified-backup")
    index = json.dumps({"weight_map": {"weight": "metal/model.safetensors"}}).encode()
    (tmp_path / "model.safetensors.index.json").write_bytes(index)
    rows = marker["files"]
    for row in rows:
        if row["path"] == "model.safetensors.index.json":
            digest = hashlib.sha256(index).hexdigest()
            row.update(size=len(index), digest=digest, payload_sha256=digest)
    (tmp_path / FILENAME).write_text(json.dumps(build_inventory(marker["repo"], marker["revision"], rows)))
    with pytest.raises(InventoryError, match="unverified weight"):
        verify_inventory(tmp_path)


def test_listed_nested_index_target_is_content_verified(tmp_path):
    checkpoint(tmp_path, sharded=True, target="weights/model-00001.safetensors")
    assert build_cohort_from_inventory(verify_inventory(tmp_path)).n_layers == 24
    (tmp_path / "weights" / "model-00001.safetensors").write_bytes(b"x" * len(b"packed-checkpoint-fixture"))
    with pytest.raises(InventoryError, match="payload"):
        verify_inventory(tmp_path)

"""Validation for heterogeneous RAM/SSD placement and capacity-driven tiering.

Covers:
  1. 32GB pinned RAM node accepted with smaller block instead of 64GB rejection.
  2. Mixed 64GB + 32GB nodes co-existing in one ring with asymmetric blocks,
     covering [0, 43), with tail stage owning layers 40, 41, 42 + 3 MTP blocks.
  3. pinnable_ram_bytes = None fails closed (no fallback to infinite or MemAvailable).
  4. Selective pull derives exact shard files and detects missing/corrupted files.
  5. Machine-readable impairment report with binding constraint and step penalty.
  6. Dynamic formula: pinned_experts_bytes scales strictly with assigned layers.
"""
import json
from pathlib import Path
import shutil
import struct
import pytest

from engines.deepseek_v4 import v4_resources as vr
from shard import weight_artifacts as artifacts
from shard.plan import V4_DUAL_RESOURCE_PROFILE, plan_ring
from shard.resources import (
    NodeResources,
    PlacementRequirements,
    ResourceError,
    STORAGE_PREPARE_SCHEMA,
    StorageRequirements,
    evaluate_fit,
)


def _make_nodes_4_with_slim():
    """1 fat VRAM node (48GB/64GB RAM) + 2 standard nodes (32GB/64GB RAM) + 1 slim node (32GB/32GB RAM)."""
    nodes = [
        {
            "id": "node_0_fat_vram",
            "free_vram_mb": 48.0 * 1024.0,
            "total_vram_mb": 48.0 * 1024.0,
            "free_ram_mb": 64.0 * 1024.0,
            "pinnable_ram_mb": 60.0 * 1024.0,
            "h2d_gbps": 24.0,
            "subnet": "10.0.0.0/24",
            "cpu_factor": 1.0,
        },
        {
            "id": "node_1_standard",
            "free_vram_mb": 32.0 * 1024.0,
            "free_ram_mb": 64.0 * 1024.0,
            "pinnable_ram_mb": 60.0 * 1024.0,
            "h2d_gbps": 24.0,
            "subnet": "10.0.1.0/24",
            "cpu_factor": 1.0,
        },
        {
            "id": "node_2_standard",
            "free_vram_mb": 32.0 * 1024.0,
            "free_ram_mb": 64.0 * 1024.0,
            "pinnable_ram_mb": 60.0 * 1024.0,
            "h2d_gbps": 24.0,
            "subnet": "10.0.2.0/24",
            "cpu_factor": 1.0,
        },
        {
            "id": "node_3_slim_ram",
            "free_vram_mb": 32.0 * 1024.0,
            "free_ram_mb": 32.0 * 1024.0,
            "pinnable_ram_mb": 28.0 * 1024.0,  # ~28GB pinnable, fits 8 layers
            "h2d_gbps": 24.0,
            "subnet": "10.0.3.0/24",
            "cpu_factor": 1.0,
        },
    ]
    rtt = [[0.0 if i == j else 15.0 for j in range(4)] for i in range(4)]
    return nodes, rtt


def _make_nodes_5_mixed():
    """3 fat nodes (64GB RAM) + 2 slim nodes (32GB RAM)."""
    nodes = [
        {
            "id": f"node_{i}_fat" if i < 3 else f"node_{i}_slim",
            "free_vram_mb": 32.0 * 1024.0,
            "free_ram_mb": 64.0 * 1024.0 if i < 3 else 32.0 * 1024.0,
            "pinnable_ram_mb": 60.0 * 1024.0 if i < 3 else 28.0 * 1024.0,
            "h2d_gbps": 24.0,
            "subnet": f"10.0.{i}.0/24",
            "cpu_factor": 1.0,
        }
        for i in range(5)
    ]
    rtt = [[0.0 if i == j else 15.0 for j in range(5)] for i in range(5)]
    return nodes, rtt


def test_32gb_pinned_node_accepted_with_smaller_block():
    """A 32GB RAM node (pinnable 28GB) was previously rejected under fixed 64GB rule,
    now admitted with an appropriately sized layer block."""
    nodes, rtt = _make_nodes_4_with_slim()
    plan = plan_ring(nodes, rtt, model=V4_DUAL_RESOURCE_PROFILE)
    assert plan is not None, "A 32GB RAM node should be admitted in a 4-node ring"
    assert len(plan["stages"]) == 4

    # Verify node_3_slim_ram receives a block that safely fits in its 28GB pinnable RAM
    stage_slim = next(s for s in plan["stages"] if s["id"] == "node_3_slim_ram")
    assert stage_slim["layers"] <= 8
    assert stage_slim["host_pinned_mb"] <= 28.0 * 1024.0


def test_mixed_64gb_and_32gb_nodes_placed_asymmetrically():
    """Heterogeneous memory ring: fat nodes take more layers, slim nodes take fewer,
    covering [0, 43), with tail stage owning layers 40..42."""
    nodes, rtt = _make_nodes_5_mixed()
    plan = plan_ring(nodes, rtt, model=V4_DUAL_RESOURCE_PROFILE)
    assert plan is not None
    assert len(plan["stages"]) == 5

    stages = sorted(plan["stages"], key=lambda s: s["index"])
    # 1. Total layers must strictly cover 43 without gaps
    assert sum(s["layers"] for s in stages) == 43
    lo = 0
    for s in stages:
        assert s["lo"] == lo
        assert s["hi"] > s["lo"]
        lo = s["hi"]
    assert lo == 43

    # 2. Tail stage must own 40, 41, 42
    tail_stage = stages[-1]
    assert tail_stage["tail"] is True
    assert tail_stage["lo"] <= 40
    assert tail_stage["hi"] == 43
    assert {40, 41, 42} <= set(range(tail_stage["lo"], tail_stage["hi"]))

    # 3. Asymmetric allocation: fat nodes absorb more layers than slim nodes
    fat_layers = [s["layers"] for s in stages if "fat" in s["id"]]
    slim_layers = [s["layers"] for s in stages if "slim" in s["id"]]
    assert max(fat_layers) >= max(slim_layers)


def test_pinnable_ram_none_fails_closed():
    """If pinnable_ram_mb is None in ram placement, planner fails closed (no infinite assumption)."""
    nodes, rtt = _make_nodes_4_with_slim()
    for n in nodes:
        n["pinnable_ram_mb"] = None  # Unmeasured pinnable RAM

    plan = plan_ring(nodes, rtt, model=V4_DUAL_RESOURCE_PROFILE)
    assert plan is None, "When pinnable RAM is unmeasured (None), planner must fail-closed"

    # Single node unknown admissibility check in v4_resources
    eval_res = vr.evaluate_node_host_admissibility(0, 10, None)
    assert eval_res["admissible"] is False
    assert eval_res["status"] == "unknown"
    assert "fail-closed" in eval_res["reason"]


def test_dynamic_host_budget_scales_with_layers():
    """Dynamic host budget scales strictly with assigned layer slice."""
    # 4 layers
    b4 = vr.compute_node_host_budget(0, 4)
    # 11 layers
    b11 = vr.compute_node_host_budget(0, 11)

    assert b4["layers_assigned"] == 4
    assert b11["layers_assigned"] == 11
    assert b4["pinned_experts_bytes"] < b11["pinned_experts_bytes"]
    # Single layer expert size ~3.42 GB
    single_layer_expert = b4["pinned_experts_bytes"] / 4
    assert 3.4e9 < single_layer_expert < 3.5e9

    # Admissibility check: 32GB pinnable RAM easily accommodates 4 layers
    pinnable_32gb = 32 * 1024 * 1024 * 1024
    admit_4 = vr.evaluate_node_host_admissibility(0, 4, pinnable_32gb)
    assert admit_4["admissible"] is True
    assert admit_4["status"] == "fits"


def test_every_stage_reports_impairment_record():
    """Every admission decision emits machine-readable impairment report."""
    nodes, rtt = _make_nodes_4_with_slim()
    plan = plan_ring(nodes, rtt, model=V4_DUAL_RESOURCE_PROFILE)
    assert plan is not None
    assert "impairment_report" in plan

    report = plan["impairment_report"]
    assert len(report) == len(plan["stages"])
    for item in report:
        assert "node_id" in item
        assert "binding_constraint" in item
        assert "vram_cap" in item
        assert "ram_cap" in item
        assert "step_penalty_ms" in item
        assert "reason" in item

    # Check slim node reported pinned_ram as binding constraint
    slim_entry = next(r for r in report if r["node_id"] == "node_3_slim_ram")
    assert slim_entry["binding_constraint"] == "pinned_ram"
    assert slim_entry["ram_cap"] == 8
    assert "pinnable RAM" in slim_entry["reason"]
    # Step penalty is non-negative and measurable
    for r in report:
        assert r["step_penalty_ms"] >= 0.0


def _write_safetensors(path, tensors):
    header, cursor = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = 1
        for dim in shape:
            size *= dim
        size = size * vr._BITS[dtype] // 8
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + size]}
        cursor += size
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(cursor))


def _selective_stage_pull(source, destination, catalog, pack, requirements):
    """Fetch complete required files and publish their exact stage descriptor."""
    destination.mkdir()
    selected = artifacts.select_stage_artifacts(
        catalog, pack, requirements.layer_start, requirements.layer_end,
        head=requirements.head, tail=requirements.tail, dspark=requirements.dspark,
    )
    paths = set(requirements.files)
    local_pack = {
        **pack,
        "files": [row for row in pack["files"] if row["path"] in paths],
        "weight_map": {name: path for name, path in pack["weight_map"].items() if path in paths},
        "offsets": {name: span for name, span in pack["offsets"].items()
                    if pack["weight_map"][name] in paths},
    }
    for path in paths:
        shutil.copyfile(source / path, destination / path)
    artifacts.write_metadata(destination, catalog, local_pack, selected)
    return selected


def test_selective_pull_requirements_and_verification(tmp_path):
    """Published selective pulls bind complete files, assets and role dependencies."""
    ckpt_dir = tmp_path / "model_ckpt"
    ckpt_dir.mkdir()
    config = {
        "n_layers": 43,
        "dim": 4096,
        "hc_mult": 4,
        "dspark_target_layer_ids": [40, 41, 42],
        "n_mtp_layers": 3,
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (ckpt_dir / "tokenizer.json").write_text('{"fixture": true}', encoding="utf-8")

    # Shard 0: layers 0..10 and embedding
    t0 = {f"layers.{i}.attn.weight": ("BF16", [2, 2]) for i in range(11)}
    t0["embed.weight"] = ("BF16", [2, 2])
    _write_safetensors(ckpt_dir / "model0-mp1.safetensors", t0)

    # Shard 1: layers 11..42, tail norm, head, MTP blocks
    t1 = {f"layers.{i}.attn.weight": ("BF16", [2, 2]) for i in range(11, 43)}
    t1.update({
        "norm.weight": ("BF16", [2]),
        "head.weight": ("BF16", [2, 2]),
        "hc_head_fn": ("F32", [4, 4]),
        "hc_head_base": ("F32", [4]),
        "hc_head_scale": ("F32", [1]),
    })
    for i in range(3):
        t1[f"mtp.{i}.ffn.experts.0.w1.weight"] = ("F4", [2, 4])
        t1[f"mtp.{i}.markov_head.markov_w2.weight"] = ("BF16", [2, 2])
        t1[f"mtp.{i}.confidence_head.proj.weight"] = ("BF16", [1, 2])
    _write_safetensors(ckpt_dir / "model1-mp1.safetensors", t1)

    # Header-only inventories are estimates, not evidence of the downloaded bytes.
    legacy_req = vr.derive_stage_storage_requirements(
        vr.inspect_checkpoint(ckpt_dir), 0, 11, head=True,
    )
    with pytest.raises(ResourceError, match="legacy storage estimates lack payload hashes"):
        vr.verify_selective_pull(ckpt_dir, legacy_req)

    # Publication hashes actual tensor payloads, complete containers and encoder assets.
    catalog, pack = artifacts.catalogue_directory(ckpt_dir, "fixture/v4-native-storage")
    artifacts.write_metadata(ckpt_dir, catalog, pack)
    complete = artifacts.verify_weight_pack(ckpt_dir)
    assert complete["payload_integrity_verified"] is True
    inventory = vr.inspect_checkpoint(ckpt_dir)
    assert inventory["payload_integrity_verified"] is False  # Inventory remains a preview.

    cases = (
        ("head", 0, 11, True, False, False, {"model0-mp1.safetensors"}),
        ("middle", 15, 25, False, False, False, {"model1-mp1.safetensors"}),
        ("tail", 40, 43, False, True, False, {"model1-mp1.safetensors"}),
        ("dspark_tail", 40, 43, False, True, True,
         {"model0-mp1.safetensors", "model1-mp1.safetensors"}),
    )
    pulls = {}
    for label, lo, hi, head, tail, dspark, weights in cases:
        requirements = vr.derive_stage_storage_requirements(
            inventory, lo, hi, head=head, tail=tail, dspark=dspark,
            filesystem_id="fixture-filesystem",
        )
        assert requirements.schema == STORAGE_PREPARE_SCHEMA
        assert requirements.model_id == catalog["model_id"]
        assert requirements.checkpoint_id == catalog["checkpoint_id"]
        assert requirements.manifest_sha256 == catalog["manifest_sha256"]
        assert set(requirements.files) == weights | {"config.json", "tokenizer.json"}
        # A middle stage must budget all of its source container, including tensors
        # outside the assigned span; selected tensor byte sizes cannot replace it.
        assert {row.path: row.size for row in requirements.file_records} == {
            path: (ckpt_dir / path).stat().st_size for path in requirements.files
        }
        assert requirements.storage_bytes > sum(row.size for row in requirements.file_records)
        assert all(row.sha256 == artifacts.hash_file(ckpt_dir / row.path)[0]
                   for row in requirements.file_records)

        destination = tmp_path / label
        selected = _selective_stage_pull(ckpt_dir, destination, catalog, pack, requirements)
        verified = vr.verify_selective_pull(destination, requirements)
        assert verified["payload_integrity_verified"] is True
        assert verified["verification_scope"] == "assigned stage payload hashes"
        assert artifacts.verified_stage_artifact_descriptor(verified) == selected
        assert set(path.name for path in destination.glob("*.safetensors")) == weights
        names = set(selected["tensor_names"])
        expected_names = {f"layers.{i}.attn.weight" for i in range(lo, hi)}
        if head or dspark:
            expected_names.add("embed.weight")
        if tail:
            expected_names.update({"norm.weight", "head.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"})
        if dspark:
            expected_names.update(name for name in t1 if name.startswith("mtp."))
        assert names == expected_names
        pulls[label] = destination, requirements

    middle, req_mid = pulls["middle"]
    path = middle / "model1-mp1.safetensors"
    original = path.read_bytes()
    header_before = vr.read_safetensors_header(path)
    changed = bytearray(original)
    changed[-1] ^= 1  # A DSpark tensor outside the middle span, inside its fetched file.
    path.write_bytes(changed)
    assert vr.read_safetensors_header(path) == header_before
    with pytest.raises(artifacts.ArtifactError, match="artifact payload hash mismatch"):
        vr.verify_selective_pull(middle, req_mid)

    path.write_bytes(original)
    assert vr.verify_selective_pull(middle, req_mid)["payload_integrity_verified"] is True
    path.unlink()
    with pytest.raises((artifacts.ArtifactError, FileNotFoundError)):
        vr.verify_selective_pull(middle, req_mid)

    head, req_head = pulls["head"]
    asset = head / "tokenizer.json"
    asset.write_bytes(asset.read_bytes().replace(b"true", b"null"))
    with pytest.raises(artifacts.ArtifactError, match="artifact payload hash mismatch"):
        vr.verify_selective_pull(head, req_head)

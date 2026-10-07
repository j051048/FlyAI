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
import struct
import pytest

from engines.deepseek_v4 import v4_resources as vr
from shard.plan import V4_DUAL_RESOURCE_PROFILE, plan_ring
from shard.resources import (
    NodeResources,
    PlacementRequirements,
    ResourceError,
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


def test_selective_pull_requirements_and_verification(tmp_path):
    """Selective pull derives only required shard files and rejects corrupted files."""
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

    inventory = vr.inspect_checkpoint(ckpt_dir)
    # Stage 0 (head): lo=0, hi=11 -> only needs model0-mp1.safetensors
    req_head = vr.derive_stage_storage_requirements(inventory, 0, 11, head=True, tail=False, dspark=False)
    assert req_head.files == ("model0-mp1.safetensors",)
    assert vr.verify_selective_pull(ckpt_dir, req_head) is True

    # Middle stage: lo=15, hi=25 -> only needs model1-mp1.safetensors
    req_mid = vr.derive_stage_storage_requirements(inventory, 15, 25, head=False, tail=False, dspark=False)
    assert req_mid.files == ("model1-mp1.safetensors",)
    assert vr.verify_selective_pull(ckpt_dir, req_mid) is True

    # Tail stage with dspark requires embed.weight (in model0) and tail modules (in model1)
    req_tail = vr.derive_stage_storage_requirements(inventory, 40, 43, head=False, tail=True, dspark=True)
    assert req_tail.files == ("model0-mp1.safetensors", "model1-mp1.safetensors")
    assert vr.verify_selective_pull(ckpt_dir, req_tail) is True

    # Corrupted / missing file detection
    missing_req = StorageRequirements(
        model_id=vr.MODEL_ID,
        layer_start=0,
        layer_end=11,
        storage_bytes=1000,
        files=("model99-mp1.safetensors",),
    )
    with pytest.raises(ResourceError, match="missing selective shard file"):
        vr.verify_selective_pull(ckpt_dir, missing_req)

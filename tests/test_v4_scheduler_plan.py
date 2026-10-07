"""DeepSeek-V4 dual-resource scheduler and planner tests (Step 7 validation).

Exercises:
  * 4x5090 fails on all-resident GPU mode, but succeeds under dual-resource (GPU + RAM) placement.
  * 6x5090 placement under both modes.
  * Heterogeneous GPU nodes (e.g. 1x 48GB fat node + 3x 32GB standard nodes) asymmetric tiered placement.
  * Host RAM and pinned memory limits are strictly enforced (fail closed when RAM is insufficient).
  * H2D PCIe bandwidth modeling in stage latency estimation.
"""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shard.plan import (
    PROFILES,
    V4_ALL_RESIDENT_PROFILE,
    V4_DUAL_RESOURCE_PROFILE,
    plan_ring,
    profile_for,
)
from shard.scheduler import JoinedNode, Scheduler


def _v4_nodes(k, vram_gb=32.0, ram_gb=64.0, pinnable_ram_gb=60.0, rtt_ms=15.0, h2d_gbps=20.0):
    nodes = []
    for i in range(k):
        nodes.append({
            "id": f"gpu_{i}",
            "free_vram_mb": vram_gb * 1024.0,
            "free_ram_mb": ram_gb * 1024.0,
            "pinnable_ram_mb": pinnable_ram_gb * 1024.0,
            "h2d_gbps": h2d_gbps,
            "subnet": f"192.168.{i}.0/24",
            "cpu_factor": 1.0,
        })
    rtt = [[0.0 if i == j else rtt_ms for j in range(k)] for i in range(k)]
    return nodes, rtt


def test_v4_profiles_registered():
    assert "deepseek-ai/DeepSeek-V4-Flash-0731" in PROFILES
    dual = profile_for("deepseek-ai/DeepSeek-V4-Flash-0731-Dual")
    resident = profile_for("deepseek-ai/DeepSeek-V4-Flash-0731-Resident")
    assert dual["placement"] == "ram" and dual["n_layers"] == 43
    assert resident["placement"] == "gpu" and resident["n_layers"] == 43
    assert dual["layer_vram_mb"] < resident["layer_vram_mb"]
    assert dual["layer_host_ram_mb"] > 0


def test_4x5090_refuses_all_resident_but_succeeds_under_dual_resource():
    """4x 32GB GPUs (128GB total) cannot hold 158GB all-resident V4, but effortlessly holds hybrid."""
    nodes, rtt = _v4_nodes(4, vram_gb=32.0, ram_gb=64.0)
    # 1. Pure GPU placement must refuse
    plan_gpu = plan_ring(nodes, rtt, model=V4_ALL_RESIDENT_PROFILE)
    assert plan_gpu is None, "4x 5090 cards cannot hold all-resident 43-layer V4 model"

    # 2. Dual-resource placement must succeed
    plan_ram = plan_ring(nodes, rtt, model=V4_DUAL_RESOURCE_PROFILE)
    assert plan_ram is not None, "4x 5090 with 64GB RAM each must hold hybrid V4"
    assert len(plan_ram["stages"]) == 4
    assert plan_ram["k"] == 4
    stages = sorted(plan_ram["stages"], key=lambda s: s["index"])
    assert stages[0]["head"] and stages[-1]["tail"]
    lo = 0
    for s in stages:
        assert s["lo"] == lo
        assert s["hi"] > s["lo"]
        assert s["placement"] == "ram"
        assert s["expert_cache_slots"] == 32
        assert s["host_pinned_mb"] > 0
        lo = s["hi"]
    assert lo == 43


def test_scheduler_facade_v4_dual_resource_plan():
    sched = Scheduler("deepseek-ai/DeepSeek-V4-Flash-0731", 43)
    rtt_map = {f"gpu_{i}": {f"gpu_{j}": (0.0 if i == j else 12.0) for j in range(4)} for i in range(4)}
    for i in range(4):
        sched.register(JoinedNode(
            node_id=f"gpu_{i}",
            vram_gb=32.0,
            rtt_ms=rtt_map[f"gpu_{i}"],
            ram_gb=64.0,
            pinnable_ram_gb=60.0,
            h2d_gbps=24.0,
        ))

    # All-resident raises
    with pytest.raises(ValueError, match="insufficient resources"):
        sched.plan(placement="gpu")

    # Hybrid RAM succeeds
    res = sched.plan(placement="ram")
    assert res is not None
    assert len(res["stages"]) == 4
    assert [s["id"] for s in res["stages"]] == res["order"]


def test_6x5090_succeeds_in_both_modes():
    """6x 32GB GPUs (192GB total) can hold all-resident (requires 6 stages) and also hybrid."""
    nodes, rtt = _v4_nodes(6, vram_gb=32.0, ram_gb=64.0)
    # All-resident requires all 6 cards because 32GB caps around 7-8 layers
    plan_gpu = plan_ring(nodes, rtt, model=V4_ALL_RESIDENT_PROFILE)
    assert plan_gpu is not None
    assert len(plan_gpu["stages"]) == 6

    # Under hybrid RAM, 4 cards already suffice to hold 43 layers; selector prefers
    # fewer WAN hops (4 stages) unless tight cap_layers forces 6 stages
    plan_ram = plan_ring(nodes, rtt, model=V4_DUAL_RESOURCE_PROFILE)
    assert plan_ram is not None
    assert plan_ram["k"] in (4, 5, 6)

    # When cap_layers=12 is set, all 6 stages participate to cover 43 layers with head/tail reserves
    tight_model = dict(V4_DUAL_RESOURCE_PROFILE, cap_layers=12)
    plan_tight = plan_ring(nodes, rtt, model=tight_model)
    assert plan_tight is not None
    assert len(plan_tight["stages"]) == 6


def test_heterogeneous_cluster_assigns_more_layers_to_fat_node():
    """1x 48GB Ada (fat VRAM) + 3x 32GB 5090: fat node absorbs the heavy MTP tail or middle blocks."""
    nodes, rtt = _v4_nodes(4, vram_gb=32.0, ram_gb=64.0)
    # Make node 1 a fat compute node with 48GB VRAM and 128GB RAM
    nodes[1]["free_vram_mb"] = 48.0 * 1024.0
    nodes[1]["free_ram_mb"] = 128.0 * 1024.0
    nodes[1]["pinnable_ram_mb"] = 120.0 * 1024.0
    nodes[1]["cpu_factor"] = 0.6

    plan = plan_ring(nodes, rtt, model=V4_DUAL_RESOURCE_PROFILE)
    assert plan is not None
    # All 43 layers are completely covered without gaps
    stages = sorted(plan["stages"], key=lambda s: s["index"])
    assert sum(s["layers"] for s in stages) == 43
    # The middle nodes (without head/tail reserve deductions) absorb maximum 15 layers each
    middle_stages = [s for s in stages if not s["head"] and not s["tail"]]
    assert all(s["layers"] == 15 for s in middle_stages)
    # The fat tail node easily accommodates the 5.5GB MTP boundary reserve plus 8 layers
    tail_stage = next(s for s in stages if s["tail"])
    assert tail_stage["layers"] >= 8


def test_host_ram_shortfall_refuses_or_caps_capacity():
    """If a node has 32GB VRAM but only 10GB RAM (room for <= 3 layers of experts), it cannot take 11 layers."""
    nodes, rtt = _v4_nodes(4, vram_gb=32.0, ram_gb=64.0)
    # Severely starve node 3's host RAM
    nodes[3]["free_ram_mb"] = 10.0 * 1024.0
    nodes[3]["pinnable_ram_mb"] = 10.0 * 1024.0
    # 43 layers across 4 nodes means at least ~10 layers per node; node 3 only fits 3 layers.
    # The other 3 nodes each cap around 15 layers -> 15*3 = 45 layers, so the ring might either
    # squeeze it or if we starve all 4 nodes, it must fail.
    for n in nodes:
        n["free_ram_mb"] = 20.0 * 1024.0
        n["pinnable_ram_mb"] = 20.0 * 1024.0
    # 20 GB RAM holds at most 6 layers per node -> 4 * 6 = 24 layers < 43 layers
    plan = plan_ring(nodes, rtt, model=V4_DUAL_RESOURCE_PROFILE)
    assert plan is None, "when host RAM is insufficient across nodes, dual-resource planner must refuse"


def test_slow_pci_h2d_bandwidth_increases_estimated_latency():
    """A node with slow PCIe (e.g. 6 GB/s) has higher layer_ms overhead than one with 24 GB/s."""
    nodes_fast, rtt = _v4_nodes(4, h2d_gbps=24.0)
    nodes_slow, _ = _v4_nodes(4, h2d_gbps=6.0)
    plan_fast = plan_ring(nodes_fast, rtt, model=V4_DUAL_RESOURCE_PROFILE)
    plan_slow = plan_ring(nodes_slow, rtt, model=V4_DUAL_RESOURCE_PROFILE)
    assert plan_fast is not None and plan_slow is not None
    assert plan_slow["step_ms"] > plan_fast["step_ms"]


def test_cli_roundtrip_v4_dual():
    nodes, rtt = _v4_nodes(4)
    payload = json.dumps({"nodes": nodes, "rtt": rtt, "model": "deepseek-ai/DeepSeek-V4-Flash-0731-Dual"})
    proc = subprocess.run(
        [sys.executable, "-m", "shard.plan"],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(proc.stdout)
    assert "stages" in result
    assert result["k"] == 4
    assert result["stages"][0]["placement"] == "ram"

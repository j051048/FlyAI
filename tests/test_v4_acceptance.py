"""Unit tests for Phase 1 Hardware Acceptance Suite (Step 8 validation)."""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0.v4_acceptance import (
    ACCEPTANCE_TARGETS,
    NodeTradeoffAnalysis,
    ScenarioResult,
    V4AcceptanceHarness,
)


def test_acceptance_targets_contract():
    assert ACCEPTANCE_TARGETS[4] == 40.0
    assert ACCEPTANCE_TARGETS[6] == 30.0


def test_cluster_tradeoff_modeling_and_wan_advantage():
    harness = V4AcceptanceHarness(rtt_ms=10.0, h2d_gbps=24.0)
    tradeoffs = harness.analyze_tradeoffs()

    # All three configurations modeled
    assert "6_node_all_resident" in tradeoffs
    assert "6_node_dual_resource" in tradeoffs
    assert "4_node_dual_resource" in tradeoffs

    t6_res = tradeoffs["6_node_all_resident"]
    t6_dual = tradeoffs["6_node_dual_resource"]
    t4_dual = tradeoffs["4_node_dual_resource"]

    # 1. 6-node all-resident: 0 DMA overhead, high VRAM
    assert t6_res.stages == 6
    assert t6_res.dma_overhead_ms_per_step == 0.0
    assert t6_res.vram_per_node_gb > 25.0
    assert t6_res.host_ram_per_node_gb == 0.0
    assert t6_res.meets_target is True

    # 2. 6-node dual-resource: small DMA overhead, drastically reduced VRAM
    assert t6_dual.stages == 6
    assert t6_dual.dma_overhead_ms_per_step > 0.0
    assert t6_dual.vram_per_node_gb < 15.0
    assert t6_dual.host_ram_per_node_gb > 20.0
    assert t6_dual.meets_target is True

    # 3. 4-node dual-resource: 4 hops saves 2 hops = 20ms WAN!
    assert t4_dual.stages == 4
    wan_saved = t6_dual.total_wan_roundtrip_ms - t4_dual.total_wan_roundtrip_ms
    assert wan_saved == 20.0, "Saving 2 WAN hops @10ms/hop must save 20.0ms"
    # Even with slightly higher DMA overhead, net step latency is significantly lower
    assert t4_dual.total_step_ms < t6_dual.total_step_ms
    # 4-node throughput easily achieves the >= 40.0 tok/s target
    assert t4_dual.estimated_tok_per_sec >= 40.0
    assert t4_dual.meets_target is True


def test_five_stress_scenarios_pass_without_lease_leaks():
    harness = V4AcceptanceHarness()
    results = harness.run_stress_scenarios(stages=4, slot_count=32)

    assert len(results) == 5
    names = [r.name for r in results]
    assert "cold_cache_convergence" in names
    assert "warm_cache_cruise" in names
    assert "high_eviction_pressure" in names
    assert "multi_turn_dialogue" in names
    assert "speculative_rollback" in names

    for r in results:
        assert r.passed is True, f"Scenario {r.name} failed"
        assert r.lease_leak == 0, f"Scenario {r.name} leaked leases: {r.lease_leak}"

    # Specific assertions per scenario
    cold = next(r for r in results if r.name == "cold_cache_convergence")
    assert cold.dma_miss_count > 0

    warm = next(r for r in results if r.name == "warm_cache_cruise")
    assert warm.cache_hit_rate >= 0.70

    tight = next(r for r in results if r.name == "high_eviction_pressure")
    assert tight.eviction_count > 50

    rollback = next(r for r in results if r.name == "speculative_rollback")
    assert rollback.lease_leak == 0


def test_generate_report_and_markdown_summary():
    harness = V4AcceptanceHarness(rtt_ms=10.0, h2d_gbps=20.0)
    report = harness.generate_report()

    assert report["overall_status"] == "SIMULATION_TARGETS_MET"
    assert report["hardware_verified"] is False and report["speed_pass"] is False
    assert "wan_hop_advantage" in report
    adv = report["wan_hop_advantage"]
    assert adv["4_node_vs_6_node_wan_ms_saved"] > 0
    assert adv["net_step_latency_advantage_ms"] > 0

    summary_md = harness.render_markdown_summary(report)
    assert "# DeepSeek-V4 Placement and Cache Simulation" in summary_md
    assert "Hardware acceptance is pending" in summary_md
    assert "4-Node vs 6-Node WAN Hop Advantage" in summary_md
    assert "speculative_rollback" in summary_md


def test_cli_acceptance_tool_roundtrip():
    cmd = [sys.executable, "-m", "phase0.v4_acceptance", "--rtt-ms", "10.0", "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["overall_status"] == "SIMULATION_TARGETS_MET"
    assert payload["hardware_verified"] is False and payload["speed_pass"] is False
    assert len(payload["stress_scenarios"]) == 5

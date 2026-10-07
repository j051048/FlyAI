"""Unit tests for Post-Speedline Capability Expansion (Step 11)."""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.deepseek_v4.v4_expansion import (
    CPUExpertFallbackExecutor,
    CrossNodeReplicaCoordinator,
    V4ClusterExpansionPolicy,
)


def test_cpu_expert_fallback_numerical_parity():
    executor = CPUExpertFallbackExecutor(dtype=torch.float32)

    dim = 64
    inter = 128
    tokens = 4

    # Synthetic weights
    torch.manual_seed(42)
    x = torch.randn((tokens, dim))
    w1 = torch.randn((inter, dim))
    w2 = torch.randn((dim, inter))
    w3 = torch.randn((inter, dim))

    # Reference standard SwiGLU computation
    ref_gate = F.linear(x, w1)
    ref_up = F.linear(x, w3)
    ref_out = F.linear(F.silu(ref_gate) * ref_up, w2)

    # Executed through fallback executor
    fallback_out = executor.compute_expert_forward(x, w1, w2, w3)

    assert torch.allclose(ref_out, fallback_out, atol=1e-5, rtol=1e-5)
    assert executor.fallback_invocations == 1


def test_cluster_expansion_policy_cpu_fallback_thresholds():
    policy = V4ClusterExpansionPolicy(
        enable_cpu_fallback=True,
        cpu_fallback_dma_wait_ms_threshold=5.0,
    )

    # Low DMA wait: should NOT fallback to CPU (use standard fast GPU DMA path)
    assert policy.should_fallback_to_cpu(dma_queue_wait_ms=1.2) is False

    # High DMA congestion (>= 5.0ms): should safely fallback to CPU
    assert policy.should_fallback_to_cpu(dma_queue_wait_ms=5.5) is True

    # When fallback explicitly disabled: never fallback
    policy.enable_cpu_fallback = False
    assert policy.should_fallback_to_cpu(dma_queue_wait_ms=10.0) is False


def test_cross_node_replica_coordinator_heterogeneous_fat_node():
    coord = CrossNodeReplicaCoordinator(replica_slots_fat_node=3)

    # Register 3x 32GB 5090 standard nodes + 1x 48GB Ada fat node
    coord.register_node("gpu_0", vram_gb=32.0, ram_gb=64.0)
    coord.register_node("gpu_1", vram_gb=32.0, ram_gb=64.0)
    coord.register_node("gpu_2", vram_gb=32.0, ram_gb=64.0)
    coord.register_node("gpu_fat", vram_gb=48.0, ram_gb=128.0)

    assert coord.nodes["gpu_0"].is_fat_node is False
    assert coord.nodes["gpu_fat"].is_fat_node is True

    # Simulate multi-step traffic: experts 5, 8, 12 receive extreme heat
    for _ in range(50):
        coord.record_accesses([5, 8, 12])
    for _ in range(5):
        coord.record_accesses([1, 2, 3])

    # Rebalance replicas
    coord.rebalance_replicas()

    # Fat node must persistently hold top hot experts {5, 8, 12}
    fat_replicas = coord.nodes["gpu_fat"].replicated_experts
    assert fat_replicas == {5, 8, 12}

    # Standard nodes hold none
    assert len(coord.nodes["gpu_0"].replicated_experts) == 0

    # Query replica holders
    assert coord.get_replica_holder(5) == "gpu_fat"
    assert coord.get_replica_holder(8) == "gpu_fat"
    assert coord.get_replica_holder(99) is None

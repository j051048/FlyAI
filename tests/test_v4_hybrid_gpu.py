"""Actual CUDA gates for host expert pools, H2D slots and hybrid graph execution.

No performance assertion and no model download. CPU runs skip these tests explicitly.
"""
from pathlib import Path
import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu, pytest.mark.hardware,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="hybrid H2D/graph acceptance requires a CUDA GPU")]


@pytest.fixture
def core():
    return pytest.importorskip("v4_expert_cache")


@pytest.fixture(autouse=True)
def fresh_rotary_tables():
    # The original cache key omits device. GPU twins must not inherit the CPU checkpoint
    # writer's rotary table, nor leave a cached CUDA table for later CPU tests.
    REF = pytest.importorskip("v4_ref_cpu")
    try:
        M = REF.load_ref()
    except ImportError as exc:
        pytest.skip(f"V4 kernel dependency unavailable on this GPU process: {exc}")
    M.precompute_freqs_cis.cache_clear()
    yield
    M.precompute_freqs_cis.cache_clear()


def cpu_moe(*, packed=False):
    REF = pytest.importorskip("v4_ref_cpu")
    V4 = pytest.importorskip("v4_stage")
    args = REF.cpu_args(n_routed_experts=6, n_activated_experts=2,
                        expert_dtype="fp4" if packed else None, moe_inter_dim=128)
    M = REF.load_ref()
    V4._set_globals(M, args)
    with torch.device("cpu"), M.set_dtype(torch.bfloat16):
        moe = M.MoE(0, args)
    with torch.no_grad():
        for index, expert in enumerate(moe.experts):
            for kind in ("w1", "w2", "w3"):
                linear = getattr(expert, kind)
                if packed:
                    raw = linear.weight.view(torch.uint8)
                    raw.copy_((torch.arange(raw.numel()) + 31 * index).remainder(256).to(torch.uint8).reshape(raw.shape))
                    linear.scale.view(torch.uint8).fill_(127 + index)
                else:
                    linear.weight.fill_(index + 1)
    return moe


@pytest.mark.parametrize("packed", [False, True])
def test_private_pinned_pool_fixed_slot_h2d_preserves_weight_and_scale_bits(core, packed):
    if packed and not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("packed-FP4 acceptance requires PyTorch float4_e2m1fn_x2")
    moe = cpu_moe(packed=packed)
    old_ptrs = {p.untyped_storage().data_ptr() for expert in moe.experts for p in expert.parameters()}
    pool = core.HostExpertPool.from_moe(moe, pin=True, preserve=True)
    actual_banks = {name: bank for name, bank in pool.banks.items() if bank is not None}
    assert all(bank.device.type == "cpu" and bank.is_pinned() for bank in actual_banks.values())
    assert all(bank.untyped_storage().data_ptr() not in old_ptrs for bank in actual_banks.values())
    cache = core.FixedSlotCache(pool, 1, device="cuda:0")
    addresses = {name: bank.data_ptr() for name, bank in cache.banks.items() if bank is not None}
    for logical_id in (0, 1, 0, 5):
        lease = cache.acquire([logical_id])
        lease.wait_on()
        slot = lease.mapping[logical_id]
        for name in actual_banks:
            src = pool.banks[name][logical_id].view(torch.uint8)
            dst = cache.banks[name][slot].view(torch.uint8).cpu()
            assert torch.equal(src, dst), f"{name}: packed checkpoint bytes changed during H2D"
        for kind in ("w1", "w2", "w3"):
            linear = getattr(cache.experts[slot], kind)
            if getattr(linear, "scale", None) is not None:
                assert linear.weight.scale is linear.scale
        assert {name: bank.data_ptr() for name, bank in cache.banks.items() if bank is not None} == addresses
        lease.release()
    assert cache.is_resident(5)
    assert not cache.is_resident(0)


def test_cross_stream_consumer_lease_prevents_premature_slot_overwrite(core):
    pool = core.HostExpertPool.from_moe(cpu_moe(), pin=True, preserve=True)
    cache = core.FixedSlotCache(pool, 1, device="cuda:0")
    first = cache.acquire([0])
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        first.wait_on(consumer)
        # Queue independent work before the slot read, so transfer-stream eviction cannot rely
        # on DMA completion as evidence that the consumer has finished reading the previous slot.
        matrix = torch.ones(512, 512, device="cuda")
        for _ in range(12):
            matrix = matrix @ matrix
        if hasattr(torch.cuda, "_sleep"):
            torch.cuda._sleep(20_000_000)  # force a real cross-stream overlap; no wall-clock speed claim
        original = cache.banks["w13"][first.mapping[0]].clone()
    with pytest.raises((RuntimeError, ValueError), match="lease|pinned|busy|slot|capacity"):
        cache.acquire([1])
    first.release(consumer)
    second = cache.acquire([1])
    second.wait_on()
    torch.cuda.synchronize()
    assert torch.equal(original.cpu(), pool.banks["w13"][0])
    assert torch.equal(cache.banks["w13"][second.mapping[1]].cpu(), pool.banks["w13"][1])
    second.release()


def test_reload_cannot_change_canonical_weights_while_a_consumer_lease_is_live(core):
    moe = cpu_moe()
    pool = core.HostExpertPool.from_moe(moe, pin=True, preserve=True)
    cache = core.FixedSlotCache(pool, 1, device="cuda:0")
    lease = cache.acquire([0])
    lease.wait_on()
    before = pool.banks["w13"].view(torch.uint8).clone()
    replacement = {name: tensor.clone() for name, tensor in moe.state_dict().items()}
    for name, value in replacement.items():
        if ".experts." in "." + name and value.dtype == torch.bfloat16:
            value.fill_(7)
    with pytest.raises((RuntimeError, ValueError), match="lease|busy|load|reload"):
        moe.load_state_dict(replacement, strict=True)
    assert torch.equal(before, pool.banks["w13"].view(torch.uint8))
    lease.release()
    moe.load_state_dict(replacement, strict=True)
    assert not cache.is_resident(0)


def test_bf16_hybrid_whole_graph_split_matches_resident_eager_after_rewind(tmp_path, monkeypatch):
    # BF16 GPU reference math is the model's own Torch operations; packed-FP4/TileLang is checked
    # independently below in a fresh interpreter to avoid the shared CPU-backend module cache.
    KERNELS = pytest.importorskip("v4_kernels_cpu")
    if KERNELS.backend() != "cpu":
        pytest.skip("BF16 graph twin uses the Torch reference backend; use the TileLang FP4 gate below")
    REF, V4 = pytest.importorskip("v4_ref_cpu"), pytest.importorskip("v4_stage")
    from test_v4_hybrid import checkpoint, assert_equal_stages, ids
    args = REF.cpu_args(n_layers=4, n_routed_experts=6, n_activated_experts=2,
                        compress_ratios=(0, 4, 8, 0, 0, 0), dspark_target_layer_ids=(1, 2, 3))
    directory, _ = checkpoint(tmp_path / "weights", args)
    V4.ref().precompute_freqs_cis.cache_clear()
    monkeypatch.setattr(V4, "V4_CUDA_GRAPH", "0")
    resident = V4.Stage(0, 4, args, head=True, tail=True, device="cuda:0").load(str(directory))
    monkeypatch.setattr(V4, "V4_CUDA_GRAPH", "whole")
    cached = V4.Stage(0, 4, args, head=True, tail=True, device="cuda:0", expert_placement="ram",
                      expert_cache_slots=2, runtime_metrics=True).load(str(directory))
    assert cached._block_graphs is not None
    cached._spec = True
    sequence = ids(args, 17).cuda()
    assert_equal_stages(resident, cached, sequence[:, :13], 0)
    for pos in (13, 14, 15):
        assert_equal_stages(resident, cached, sequence[:, pos:pos + 1], pos)
    # Reject a speculative future and return to a previously captured graph position.
    cached.forward(cached.embed(sequence[:, 16:17]), sequence[:, 16:17], 16)
    cached._seek(16)
    token = assert_equal_stages(resident, cached, sequence[:, 16:17], 16)
    for pos in (17, 18):
        token = assert_equal_stages(resident, cached, token, pos)
    metrics = cached.runtime_metrics()
    assert metrics["mode"] == "gpu_expert_cache"
    assert metrics["totals"]["dma_misses"] > 0 and metrics["totals"]["dma_bytes"] > 0
    assert metrics["totals"]["cpu_misses"] == 0 and metrics["totals"]["reference_routes"] == 0
    assert metrics["totals"]["routed_entries"] == (metrics["totals"]["resident_hits"] +
                                                   metrics["totals"]["dma_misses"])
    resident.reset(); cached.reset()
    assert_equal_stages(resident, cached, sequence[:, :13], 0)


@pytest.mark.parametrize("hash_routing", [False, True])
def test_original_tilelang_grouped_fp4_matches_full_resident_reference(hash_routing):
    pytest.importorskip("tilelang", reason="original grouped FP4 acceptance needs TileLang")
    if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("original V4 packed formats require recent PyTorch")
    if torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("this V4 grouped-FP4 hardware acceptance is for SM120 / RTX 5090")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, V4_KERNELS="tilelang", V4_MOE_GROUPED="0", V4_MOE_MULTI="0",
               V4_MOE_DECODE="0", V4_FP8_GEMV="0", V4_FP8_SHARED="0", V4_CUDA_GRAPH="0",
               V4_EXPERT_PLACEMENT="gpu")
    script = textwrap.dedent(f"""
        import torch
        import v4_ref_cpu, v4_stage, v4_moe_grouped, v4_hybrid, v4_expert_cache
        M = v4_ref_cpu.load_ref()
        args = v4_ref_cpu.cpu_args(n_layers=4,n_routed_experts=8,n_activated_experts=3,
            compress_ratios=(0,0,0,0,0,0),dspark_target_layer_ids=(1,2,3),
            dim=256,moe_inter_dim=128,dtype='fp8',expert_dtype='fp4',scale_dtype='fp8')
        v4_stage._set_globals(M,args)
        layer_id = {0 if hash_routing else 3}
        reference = v4_moe_grouped.build_real_dims_moe(M,args,seed=3,layer_id=layer_id)
        manager = v4_expert_cache.StageBudgetManager(slots_per_pool=3,device='cuda:0')
        Hybrid = v4_hybrid.hybrid_block_cls(M.Block,M,manager,device='cuda:0')
        with torch.device('cuda:0'), M.set_dtype(torch.bfloat16):
            layer = Hybrid(layer_id,args)
        layer.ffn.load_state_dict(reference.state_dict(),strict=True)
        caches = manager.allocate()
        v4_hybrid.bind_caches([layer],caches)
        if reference.gate.hash:
            with torch.no_grad():
                for moe in (reference,layer.ffn):
                    moe.gate.tid2eid.copy_(torch.tensor([1,1,3],dtype=torch.int32,device='cuda').expand_as(moe.gate.tid2eid))
        for seed in (3,7,11):
            torch.manual_seed(seed)
            x = torch.randn(1,1,args.dim,dtype=torch.bfloat16,device='cuda:0')
            token = torch.tensor([[seed]],device='cuda:0')
            with torch.no_grad():
                expected = reference(x,token)
                actual = layer.ffn(x,token)
            assert torch.equal(expected,actual), (seed,(expected.float()-actual.float()).abs().max().item())
        assert layer.ffn._hybrid_runtime.pool.banks['w13'].is_pinned()
        assert layer.ffn._hybrid_runtime.grouped_steps == 3, 'FP4 comparison must execute the grouped kernel'
        print('ORIGINAL_TILELANG_HYBRID_BITEXACT')
    """)
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            text=True, capture_output=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ORIGINAL_TILELANG_HYBRID_BITEXACT" in result.stdout

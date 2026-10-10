"""Actual raw-byte/lifecycle and tiny native-model gates for expert copy lookahead.

CPU cases emulate cache transfers explicitly; CUDA cases below require real DMA
and compute and make no throughput claim. No model files are downloaded.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")
import v4_expert_cache as EC
import v4_prefill_expert_pipeline as PP

from test_v4_expert_cache import pool, PendingEvent
from test_v4_hybrid import checkpoint, ids, assert_equal_stages, run
import v4_ref_cpu as REF
import v4_stage as V4


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def slots(capacity=4, *, packed=False):
    host = pool(count=6, dim=32 if packed else 4, inter=32 if packed else 4,
                dtype=torch.float4_e2m1fn_x2 if packed else torch.bfloat16, scales=packed)
    return EC.FixedSlotCache(host, capacity, device="cpu", emulation=True)


@pytest.mark.parametrize("options", [dict(enabled=1), dict(depth=True), dict(depth=0),
    dict(depth=5), dict(batch_size=-1), dict(batch_size=1.5)])
def test_invalid_controls_fail_before_any_copy(options):
    cache = slots()
    with pytest.raises(ValueError):
        PP.plan(cache, **options)
    assert cache.stats()["new_copies"] == 0


@pytest.mark.parametrize("capacity,depth,batch_size,reason", [
    (1, 2, 0, "insufficient_cache_slots"), (4, 1, 0, "depth_has_no_lookahead"),
    (4, 2, 3, "insufficient_cache_slots")])
def test_insufficient_budget_is_explicit_demand_fallback(capacity, depth, batch_size, reason):
    cache = slots(capacity)
    config = PP.plan(cache, enabled=True, depth=depth, batch_size=batch_size)
    assert not config["active"] and config["reason"] == reason
    assert config["byte_budget"] == config["additional_gpu_bytes"] == config["additional_pinned_bytes"] == 0
    with pytest.raises(EC.ExpertCacheError):
        PP.ExpertBatchPipeline(cache, range(6), depth=depth, batch_size=batch_size)


@pytest.mark.parametrize("packed", [False, True])
def test_actual_bank_bytes_and_scales_with_bounded_early_production(packed):
    cache = slots(packed=packed)
    pointers = {key: bank.data_ptr() for key, bank in cache.banks.items() if bank is not None}
    stream = PP.ExpertBatchPipeline(cache, range(6), depth=2, batch_size=2)
    seen = []
    with stream:
        assert cache.stats()["new_copies"] == 4, "lookahead is really queued before the first consumer"
        assert cache.stats()["live_leases"] == 4
        for batch, lease in stream:
            lease.wait_on()
            for eid in batch:
                seen.append(eid)
                for name, source in cache.pool.banks.items():
                    if source is not None:
                        assert torch.equal(source[eid].view(torch.uint8),
                                           cache.banks[name][lease.mapping[eid]].view(torch.uint8))
            lease.release()
    assert seen == list(range(6))
    assert stream.stats()["peak_batches"] == 2 and stream.stats()["peak_slots"] == 4
    assert stream.stats()["consumed_batches"] == 3 and stream.stats()["dma_bytes"] == 0
    assert cache.stats()["live_leases"] == 0
    assert pointers == {key: bank.data_ptr() for key, bank in cache.banks.items() if bank is not None}
    json.dumps({"config": stream.config, "observations": stream.stats()}, allow_nan=False)


def test_consumer_event_completes_before_slot_reuse_and_producer_cannot_advance_live_lease():
    cache = slots()
    with PP.ExpertBatchPipeline(cache, range(6), depth=2, batch_size=2) as stream:
        batch, first = next(stream)
        first.wait_on()
        with pytest.raises(EC.ExpertCacheError, match="release the current"):
            next(stream)
        with pytest.raises(EC.ExpertCacheError, match="active leases"):
            cache.acquire([4])
        pending = PendingEvent()
        original = cache.banks["w13"][first.mapping[0]].clone()
        original_sync = pending.synchronize
        def synchronize():
            if not pending.done:
                assert torch.equal(cache.banks["w13"][first.mapping[0]], original)
            original_sync()
        pending.synchronize = synchronize
        first.release(done_event=pending)
        batch, second = next(stream)
        assert pending.done and pending.synchronizations >= 1
        second.wait_on(); second.release()
    assert cache.stats()["live_leases"] == 0


def test_pinned_source_reload_cannot_race_live_lookahead_and_waits_pending_copy_before_write():
    cache = slots()
    original = cache.pool.banks["w13"].view(torch.uint8).clone()
    stream = PP.ExpertBatchPipeline(cache, range(6), depth=2, batch_size=2)
    with stream:
        with pytest.raises(EC.ExpertCacheError, match="lease is active"):
            cache.pool.begin_reload()
        assert torch.equal(cache.pool.banks["w13"].view(torch.uint8), original)
    pending = PendingEvent()
    cache._slots[0].ready = pending
    original_sync = pending.synchronize
    def synchronize():
        assert torch.equal(cache.pool.banks["w13"].view(torch.uint8), original)
        original_sync()
    pending.synchronize = synchronize
    with cache.pool.reloading():
        assert pending.done, "source overwrite must wait for the recorded DMA completion"
        cache.pool.banks["w13"].view(torch.uint8).fill_(19)
    assert cache.pool.loaded and not cache.stats()["owners"]
    with PP.ExpertBatchPipeline(cache, range(6), depth=2) as again:
        for batch, lease in again:
            lease.wait_on()
            for eid in batch:
                assert torch.equal(cache.pool.banks["w13"][eid].view(torch.uint8),
                                   cache.banks["w13"][lease.mapping[eid]].view(torch.uint8))
            lease.release()


def test_cancel_and_partial_producer_failure_release_every_reserved_slot(monkeypatch):
    cache = slots()
    stream = PP.ExpertBatchPipeline(cache, range(6), depth=2)
    with pytest.raises(RuntimeError, match="consumer cancelled"):
        with stream:
            _, lease = next(stream)
            lease.wait_on()
            try:
                raise RuntimeError("consumer cancelled")
            finally:
                lease.release()
    assert stream.closed and cache.stats()["live_leases"] == 0
    assert stream.stats()["cancelled_batches"] == 1
    stream.close()  # deterministic, idempotent cleanup
    original_acquire = cache.acquire
    calls = 0
    def acquire(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("copy failed")
        return original_acquire(*args, **kwargs)
    monkeypatch.setattr(cache, "acquire", acquire)
    failed = PP.ExpertBatchPipeline(cache, range(6), depth=2)
    with pytest.raises(RuntimeError, match="copy failed"):
        with failed:
            pytest.fail("failed producer must not license a compute read")
    assert failed.closed and cache.stats()["live_leases"] == 0


@pytest.mark.parametrize("invalid", [[1, 0], [0, 0], [-1], [6]])
def test_bad_logical_producer_order_never_copies(invalid):
    cache = slots()
    with pytest.raises(EC.ExpertCacheError, match="ascending, valid"):
        PP.ExpertBatchPipeline(cache, invalid)
    assert cache.stats()["new_copies"] == 0


def tiny_args():
    return REF.cpu_args(n_layers=4, n_routed_experts=6, n_activated_experts=2,
                       compress_ratios=(0, 4, 8, 0, 0, 0), dspark_target_layer_ids=(1, 2, 3))


@pytest.mark.parametrize("batch,prompt", [(1, 13), (2, 7), (1, 33)])
def test_native_prefill_shapes_logits_kv_and_rewind_stay_bit_exact(tmp_path, batch, prompt):
    args = tiny_args()
    directory, _ = checkpoint(tmp_path / "weights", args)
    resident = V4.Stage(0, 4, args, head=True, tail=True, device="cpu").load(str(directory))
    cached = V4.Stage(0, 4, args, head=True, tail=True, device="cpu", expert_placement="ram",
        expert_cache_reference=True, expert_cache_slots=4, prefill_expert_pipeline=True,
        prefill_expert_depth=2, prefill_expert_batch=2, runtime_metrics=True).load(str(directory))
    stable = cached.prefill_pipeline_config()
    sequence = ids(args, prompt + 7, batch=batch)
    assert_equal_stages(resident, cached, sequence[:, :prompt], 0)
    prefill_status = cached.prefill_pipeline_status()
    assert any(pool["observations"]["calls"] > 0 for pool in prefill_status["pools"].values())
    assert all(pool["observations"]["peak_slots"] <= pool["config"]["slot_budget"]
               for pool in prefill_status["pools"].values())
    cached._spec = True
    run(cached, sequence[:, prompt:prompt + 4], prompt)
    for position in range(prompt, prompt + 2):
        assert_equal_stages(resident, cached, sequence[:, position:position + 1], position)
    # Rewind the speculative future; correction starts at the committed prefix.
    cached._seek(prompt + 2)
    assert_equal_stages(resident, cached, sequence[:, prompt + 2:prompt + 3], prompt + 2)
    assert cached.prefill_pipeline_status() == prefill_status, "decode/replay must not silently use prefill lookahead"
    assert cached.prefill_pipeline_config() == stable
    metrics = cached.runtime_metrics()
    assert metrics["totals"]["cpu_misses"] == 0 and metrics["totals"]["dma_bytes"] == 0
    resident.reset(); cached.reset()
    assert all(pool["observations"]["calls"] == 0 for pool in cached.prefill_pipeline_status()["pools"].values())
    assert_equal_stages(resident, cached, sequence[:, :prompt], 0)


def test_pipeline_disabled_resident_and_one_slot_fallback_are_truthful(tmp_path):
    args = tiny_args()
    directory, _ = checkpoint(tmp_path / "weights", args)
    with pytest.raises(ValueError, match="RAM expert placement"):
        V4.Stage(0, 4, args, device="cpu", prefill_expert_pipeline=True)
    resident = V4.Stage(0, 4, args, head=True, tail=True, device="cpu").load(str(directory))
    assert not resident.prefill_pipeline_config()["enabled"] and resident.prefill_pipeline_status()["pools"] == {}
    cached = V4.Stage(0, 4, args, head=True, tail=True, device="cpu", expert_placement="ram",
        expert_cache_reference=True, expert_cache_slots=1, prefill_expert_pipeline=True).load(str(directory))
    assert_equal_stages(resident, cached, ids(args, 13), 0)
    for pool_status in cached.prefill_pipeline_status()["pools"].values():
        assert not pool_status["config"]["active"]
        assert pool_status["config"]["reason"] == "insufficient_cache_slots"
        assert pool_status["observations"]["calls"] == 0 and pool_status["observations"]["fallback_calls"] == 1


def test_hash_duplicate_scatter_and_dspark_prefill_keep_original_row_shapes(tmp_path):
    import safetensors.torch as ST
    args = tiny_args()
    directory, model = checkpoint(tmp_path / "weights", args)
    with torch.no_grad():
        for layer in model.layers:
            if layer.ffn.gate.hash:
                route = torch.arange(args.vocab_size).remainder(args.n_routed_experts).to(torch.int32)
                layer.ffn.gate.tid2eid.copy_(route[:, None].expand_as(layer.ffn.gate.tid2eid))
    weights = {name: tensor.detach().clone().contiguous() for name, tensor in model.state_dict().items()
               if not (name.startswith("mtp.") and (".embed." in name or ".head." in name))}
    ST.save_file(weights, str(directory / "model0-mp1.safetensors"))
    resident = V4.Stage(0, 4, args, head=True, tail=True, dspark=True, device="cpu").load(str(directory))
    cached = V4.Stage(0, 4, args, head=True, tail=True, dspark=True, device="cpu", expert_placement="ram",
        expert_cache_reference=True, expert_cache_slots=4, prefill_expert_pipeline=True).load(str(directory))
    actual, original = [], []
    for tag, stage in ((original, resident), (actual, cached)):
        for eid, expert in enumerate(stage.layers[0].ffn.experts if stage is resident else
                                     stage.layers[0].ffn._hybrid_runtime.cache.experts):
            # Slot identity differs; only shape/weight-shape/order is mathematical input.
            expert.register_forward_pre_hook(lambda module, inputs, record=tag:
                record.append((tuple(inputs[0].shape), tuple(inputs[1].shape))))
    prompt = torch.arange(13).unsqueeze(0)
    assert_equal_stages(resident, cached, prompt, 0)
    assert actual == original and len(actual) == 6
    import v4_dspark_draft as DS
    r_draft, c_draft = DS.DSparkTail(resident).load(str(directory)), cached._hybrid_ring_drafter.tail
    assert c_draft is not None
    # MTP uses its normal complete small-block shape, never tokenwise chunks.
    residual = resident.tail_main_hidden()
    r_draft.prefill(torch.tensor([3]), residual)
    c_draft.prefill(torch.tensor([3]), cached.tail_main_hidden())
    token = torch.tensor([[3]])
    assert_equal_stages(resident, cached, token, 13)
    r_result = r_draft.advance_and_draft(token, resident.tail_main_hidden(), start_pos=13)
    c_result = c_draft.advance_and_draft(token, cached.tail_main_hidden(), start_pos=13)
    assert all(torch.equal(x, y) for x, y in zip(r_result, c_result))


def test_zero_position_replay_and_capture_warmup_never_arm_prefill_producer(tmp_path):
    args = tiny_args()
    directory, _ = checkpoint(tmp_path / "weights", args)
    cached = V4.Stage(0, 4, args, head=True, tail=True, device="cpu", expert_placement="ram",
        expert_cache_reference=True, expert_cache_slots=4, prefill_expert_pipeline=True,
        runtime_metrics=True).load(str(directory))
    layer = cached.layers[0]
    runtime = layer.ffn._hybrid_runtime
    token = torch.arange(14).view(2, 7)
    h = cached.embed(token)
    with runtime.phase_context("replay"):
        layer(h, 0, token)
    assert runtime.last_event["phase"] == "replay"
    assert runtime.prefill_pipeline_status()["observations"]["calls"] == 0
    before = dict(runtime.prefill_pipeline_status()["observations"])
    with runtime.phase_context("prefill"), runtime.capture_warmup():
        layer(h, 0, token)
    assert runtime.prefill_pipeline_status()["observations"] == before


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA DMA/compute gate unavailable")


@pytest.mark.gpu
@pytest.mark.hardware
@requires_cuda
def test_actual_cuda_fifo_consumer_stress_preserves_source_and_evicted_slot_bits():
    from test_v4_hybrid_gpu import cpu_moe
    host = EC.HostExpertPool.from_moe(cpu_moe(), pin=True, preserve=True)
    cache = EC.FixedSlotCache(host, 4, device="cuda:0")
    snapshots, consumer = [], torch.cuda.Stream()
    try:
        for _ in range(8):
            with PP.ExpertBatchPipeline(cache, range(6), depth=2, batch_size=2) as pipeline:
                for batch, lease in pipeline:
                    with torch.cuda.stream(consumer):
                        lease.wait_on(consumer)
                        if hasattr(torch.cuda, "_sleep"):
                            torch.cuda._sleep(2_000_000)
                        for eid in batch:
                            snapshots.append((eid, cache.banks["w13"][lease.mapping[eid]].clone()))
                    lease.release(consumer)
        torch.cuda.synchronize()
        assert all(torch.equal(tensor.cpu(), host.banks["w13"][eid]) for eid, tensor in snapshots)
        assert cache.stats()["evictions"] > 0 and cache.stats()["live_leases"] == 0
        assert all(bank.is_pinned() for bank in host.banks.values() if bank is not None)
    finally:
        cache.close()


@pytest.mark.gpu
@pytest.mark.hardware
@requires_cuda
def test_actual_cuda_full_prefill_logits_kv_and_rewind_match_resident(tmp_path, monkeypatch):
    import v4_kernels_cpu
    if v4_kernels_cpu.backend() != "cpu":
        pytest.skip("BF16 tiny Stage twin requires Torch reference kernels; original FP4 gate is separate")
    args = tiny_args()
    directory, _ = checkpoint(tmp_path / "weights", args)
    V4.ref().precompute_freqs_cis.cache_clear()
    monkeypatch.setattr(V4, "V4_CUDA_GRAPH", "0")
    try:
        resident = V4.Stage(0, 4, args, head=True, tail=True, device="cuda:0").load(str(directory))
        cached = V4.Stage(0, 4, args, head=True, tail=True, device="cuda:0", expert_placement="ram",
            expert_cache_slots=4, prefill_expert_pipeline=True, runtime_metrics=True).load(str(directory))
        sequence = ids(args, 37).cuda()
        assert_equal_stages(resident, cached, sequence[:, :33], 0)
        cached._spec = True
        run(cached, sequence[:, 33:37], 33)
        cached._seek(33)
        for position in (33, 34, 35):
            assert_equal_stages(resident, cached, sequence[:, position:position + 1], position)
        status = cached.prefill_pipeline_status()
        assert any(pool["observations"]["dma_bytes"] > 0 for pool in status["pools"].values())
        assert all(pool["observations"]["peak_slots"] <= 4 for pool in status["pools"].values())
        assert cached.runtime_metrics()["totals"]["cpu_misses"] == 0
    finally:
        V4.ref().precompute_freqs_cis.cache_clear()


@pytest.mark.gpu
@pytest.mark.hardware
@requires_cuda
def test_original_tilelang_fp4_prefill_pipeline_preserves_exact_expert_math():
    pytest.importorskip("tilelang", reason="original FP4 CUDA parity needs TileLang")
    if torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("original FP4 acceptance targets SM120 / RTX 5090")
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "V4_KERNELS": "tilelang", "V4_HADAMARD": "torch", "V4_MOE_GROUPED": "0",
           "V4_MOE_MULTI": "0", "V4_MOE_DECODE": "0", "V4_FP8_GEMV": "0", "V4_FP8_SHARED": "0",
           "V4_CUDA_GRAPH": "0", "V4_EXPERT_PLACEMENT": "gpu", "V4_PREFILL_EXPERT_PIPELINE": "0"}
    script = textwrap.dedent("""
        import torch,v4_ref_cpu,v4_stage,v4_moe_grouped,v4_hybrid,v4_expert_cache
        M=v4_ref_cpu.load_ref()
        a=v4_ref_cpu.cpu_args(n_layers=4,n_routed_experts=6,n_activated_experts=3,
            compress_ratios=(0,0,0,0,0,0),dspark_target_layer_ids=(1,2,3),dim=256,
            moe_inter_dim=128,dtype='fp8',expert_dtype='fp4',scale_dtype='fp8')
        v4_stage._set_globals(M,a)
        reference=v4_moe_grouped.build_real_dims_moe(M,a,seed=7,layer_id=0)
        with torch.no_grad():
            route=torch.arange(a.vocab_size,device='cuda').remainder(6).to(torch.int32)
            reference.gate.tid2eid.copy_(torch.stack([route,route,(route+1)%6],dim=-1))
        manager=v4_expert_cache.StageBudgetManager(slots_per_pool=4,device='cuda:0')
        Hybrid=v4_hybrid.hybrid_block_cls(M.Block,M,manager,device='cuda:0')
        with torch.device('cuda:0'),M.set_dtype(torch.bfloat16): layer=Hybrid(0,a)
        layer.ffn.load_state_dict(reference.state_dict(),strict=True)
        v4_hybrid.bind_caches([layer],manager.allocate())
        runtime=layer.ffn._hybrid_runtime
        runtime.configure_prefill_pipeline(enabled=True,depth=2,batch_size=2)
        for rows in (7,13,33):
            torch.manual_seed(rows)
            x=torch.randn(1,rows,a.dim,device='cuda',dtype=torch.bfloat16)
            token=torch.arange(rows,device='cuda').view(1,rows)
            with torch.no_grad(),runtime.phase_context('prefill'):
                wanted=reference(x,token);got=layer.ffn(x,token)
            assert torch.equal(wanted,got), (rows,(wanted.float()-got.float()).abs().max().item())
        assert runtime.prefill_pipeline_status()['observations']['calls']==3
        assert runtime.prefill_pipeline_status()['observations']['peak_slots']<=4
        print('ORIGINAL_FP4_PREFILL_PIPELINE_BITEXACT')
    """)
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ORIGINAL_FP4_PREFILL_PIPELINE_BITEXACT" in result.stdout

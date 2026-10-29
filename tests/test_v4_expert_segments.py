"""Real local expert dispatch and explicit CPU graph-reference lifecycle tests."""
from types import SimpleNamespace

import pytest
import torch

import v4_stage as V
import v4_whole_layer_graph as W
from v4_expert_graph import ExpertSegmentGraphs
from test_v4_hybrid import args, checkpoint, ids, assert_equal_stages


@pytest.fixture(autouse=True)
def deterministic_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def hybrid(directory, args, *, exclusive=False, slots=3):
    options = dict(device="cpu", head=True, tail=True, expert_placement="ram",
        expert_cache_reference=True, expert_cache_slots=slots)
    if exclusive:
        options.update(expert_storage="exclusive", expert_storage_host_bytes=8 << 20)
    return V.Stage(0, args.n_layers, args, **options).load(directory)


@pytest.mark.parametrize("exclusive", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_ready_first_and_fixed_slot_segments_match_native_state_across_swaps(tmp_path, args, exclusive, graph):
    directory, _ = checkpoint(tmp_path / "weights", args)
    baseline = V.Stage(0, args.n_layers, args, head=True, tail=True, device="cpu").load(directory)
    stage = hybrid(directory, args, exclusive=exclusive)
    for layer in stage.layers:
        layer.ffn._hybrid_runtime.configure_execution(ready_first=True,
            graph_bytes=(2 << 20) if graph else 0, reference=True, max_graphs=16)
    tokens = ids(args, 27)
    with torch.no_grad():
        assert_equal_stages(baseline, stage, tokens[:, :13], 0)
        for pos in range(13, 27):
            assert_equal_stages(baseline, stage, tokens[:, pos:pos + 1], pos)
    for layer in stage.layers:
        runtime = layer.ffn._hybrid_runtime
        assert runtime.execution_status()["ready_observations"]["batches"] > 0
        assert runtime.cache.stats()["live_leases"] == 0
        if graph:
            status = runtime.segment_graphs.status()
            assert status["reference_calls"] > status["entries"] > 0
            assert status["captures"] == status["replays"] == 0
            assert status["accounted_bytes"] <= status["max_bytes"]


def test_completed_hit_computes_before_wait_for_unrelated_miss(tmp_path, args, monkeypatch):
    directory, _ = checkpoint(tmp_path / "weights", args)
    stage = hybrid(directory, args)
    runtime = stage.layers[0].ffn._hybrid_runtime
    runtime.configure_execution(ready_first=True, reference=True)
    cache = runtime.cache
    lease = cache.acquire([5]); lease.wait_on(); lease.release()
    events = []
    class Pending:
        done = False
        def query(self):
            return self.done
        def synchronize(self):
            events.append("wait:0")
            self.done = True
    acquire = cache.acquire
    def delayed(ids, **kw):
        events.append("acquire:" + ",".join(str(eid) for eid in ids))
        lease = acquire(ids, **kw)
        if 0 in lease.mapping:
            cache._slots[lease.mapping[0]].ready = Pending()
        return lease
    monkeypatch.setattr(cache, "acquire", delayed)
    monkeypatch.setattr(runtime.moe.gate, "forward", lambda x, _: (
        torch.tensor([[0.4, 0.6]], dtype=x.dtype), torch.tensor([[0, 5]])))
    for slot, expert in enumerate(cache.experts):
        expert.register_forward_pre_hook(lambda module, inputs, index=slot:
            events.append("compute:" + str(cache._slots[index].owner)))
    with torch.no_grad():
        runtime.forward(torch.ones(1, args.dim, dtype=torch.bfloat16), torch.tensor([1]))
    assert events.index("compute:5") < events.index("wait:0") < events.index("compute:0")
    assert events.index("compute:5") < events.index("acquire:0")
    assert runtime.execution_status()["ready_observations"]["completed_before_pending"] == 1
    assert cache.stats()["live_leases"] == 0


def test_partial_lease_wait_does_not_wait_another_expert_or_lose_later_dependency():
    from test_v4_expert_cache import cache, PendingEvent
    cached = cache()
    lease = cached.acquire([0, 1])
    pending = PendingEvent()
    cached._slots[lease.mapping[1]].ready = pending
    assert lease.ready_ids() == (0,)
    lease.wait_for([0])
    assert pending.synchronizations == 0
    lease.wait_on()
    assert pending.synchronizations == 1
    with pytest.raises(Exception, match="outside"):
        lease.wait_for([5])
    lease.release()
    with pytest.raises(Exception, match="released"):
        lease.wait_for([0])


def test_staging_quota_uses_original_streaming_fold_when_too_small(tmp_path, args):
    directory, _ = checkpoint(tmp_path / "weights", args)
    baseline = hybrid(directory, args)
    stage = hybrid(directory, args)
    for layer in stage.layers:
        layer.ffn._hybrid_runtime.configure_execution(ready_first=True, ready_bytes=1, reference=True)
    with torch.no_grad():
        assert_equal_stages(baseline, stage, ids(args, 13), 0)
    assert all(layer.ffn._hybrid_runtime.execution_status()["ready_observations"]["batches"] == 0
               for layer in stage.layers)


def test_whole_split_warmup_does_not_bind_or_count_segment_programs(tmp_path, args):
    directory, _ = checkpoint(tmp_path / "weights", args)
    stage = hybrid(directory, args, exclusive=True)
    runtime = stage.layers[0].ffn._hybrid_runtime
    runtime.configure_execution(ready_first=True, graph_bytes=2 << 20, reference=True)
    x = torch.ones(1, args.dim, dtype=torch.bfloat16)
    token = torch.tensor([1])
    with torch.no_grad(), runtime.capture_warmup():
        expected = runtime.forward(x, token)
    status = runtime.segment_graphs.status()
    assert status["entries"] == status["accounted_bytes"] == status["reference_calls"] == 0
    assert status["captures"] == status["replays"] == 0
    assert runtime.segment_graphs._stamp is None
    assert runtime.last_event is None
    assert runtime.execution_status()["ready_observations"]["batches"] == 0
    with torch.no_grad():
        actual = runtime.forward(x, token)
    assert torch.equal(expected, actual)
    assert runtime.segment_graphs.status()["entries"] > 0
    assert runtime.segment_graphs._stamp is not None


def test_segment_weight_identity_cannot_silently_survive_reload(tmp_path, args):
    directory, _ = checkpoint(tmp_path / "weights", args)
    stage = hybrid(directory, args)
    runtime = stage.layers[0].ffn._hybrid_runtime
    runtime.configure_execution(graph_bytes=1 << 20, reference=True)
    values = torch.ones(1, args.dim, dtype=torch.bfloat16)
    runtime._shared(values)
    with torch.no_grad():
        next(runtime.moe.shared_experts.parameters()).add_(1)
    with pytest.raises(RuntimeError, match="storage changed"):
        runtime._shared(values)
    assert runtime.segment_graphs.status()["faulted"]


def test_same_physical_slot_program_reads_new_expert_after_eviction(tmp_path, args):
    directory, _ = checkpoint(tmp_path / "weights", args)
    stage = hybrid(directory, args, exclusive=True, slots=1)
    runtime = stage.layers[0].ffn._hybrid_runtime
    graphs = ExpertSegmentGraphs(runtime, max_bytes=1 << 20, reference=True)
    xv = torch.ones(1, args.dim, dtype=torch.bfloat16)
    weight = torch.ones(1, 1, dtype=torch.float32)
    with torch.no_grad():
        lease = runtime.cache.acquire([0]); lease.wait_on()
        before = graphs.run(("slot", 0), runtime.cache.experts[0], xv, weight)
        lease.release()
        lease = runtime.cache.acquire([5]); lease.wait_on()
        after = graphs.run(("slot", 0), runtime.cache.experts[0], xv, weight)
        expected = runtime.pool.experts[5](xv, weight)
        lease.release()
    assert not torch.equal(before, after)
    assert torch.equal(after, expected)
    assert graphs.status()["entries"] == 1


def test_bounded_segment_pool_uses_eager_math_without_growing(tmp_path, args):
    directory, _ = checkpoint(tmp_path / "weights", args)
    runtime = hybrid(directory, args).layers[0].ffn._hybrid_runtime
    graphs = ExpertSegmentGraphs(runtime, max_bytes=1 << 20, max_graphs=1, reference=True)
    x = torch.ones(1, args.dim, dtype=torch.bfloat16)
    first = graphs.run("first", runtime.moe.shared_experts, x)
    second = graphs.run("other", runtime.moe.shared_experts, x + 1)
    assert torch.equal(first, runtime.moe.shared_experts(x))
    assert torch.equal(second, runtime.moe.shared_experts(x + 1))
    assert graphs.status()["entries"] == 1
    assert graphs.status()["eager_fallbacks"] == 1
    graphs.close()
    with pytest.raises(RuntimeError, match="closed"):
        graphs.run("first", runtime.moe.shared_experts, x)


def test_failed_split_capture_restores_all_live_kv_before_eager_fallback(monkeypatch):
    buffers = [torch.arange(7, dtype=torch.float32), torch.arange(3, dtype=torch.int64)]
    saved = [tensor.clone() for tensor in buffers]
    graphs = W.WholeBlockGraphs.__new__(W.WholeBlockGraphs)
    graphs.L, graphs.dev = object(), "cpu"
    monkeypatch.setattr(W, "_layer_state", lambda layer: buffers)
    def fail(key, ids):
        for value in buffers:
            value.add_(17)
        raise RuntimeError("injected expert warmup failure")
    graphs._capture_unchecked = fail
    with pytest.raises(RuntimeError, match="expert warmup"):
        graphs._capture((16, False))
    assert all(torch.equal(a, b) for a, b in zip(buffers, saved))


def test_exclusive_whole_eager_composes_with_real_attention_and_fresh_slot_weights(tmp_path, args, monkeypatch):
    from test_v4_hybrid import state
    directory, _ = checkpoint(tmp_path / "weights", args)
    monkeypatch.setattr(V, "V4_CUDA_GRAPH", "whole")
    monkeypatch.setattr(V, "V4_EXPERT_SEGMENT_GRAPH_MIB", 8)
    monkeypatch.setattr(V, "V4_EXPERT_READY_FIRST", True)
    monkeypatch.setenv("V4_MOE_IN_GRAPH", "0")
    stage = hybrid(directory, args, exclusive=True)
    monkeypatch.setattr(V, "V4_CUDA_GRAPH", False)
    monkeypatch.setattr(V, "V4_EXPERT_SEGMENT_GRAPH_MIB", 0)
    monkeypatch.setattr(V, "V4_EXPERT_READY_FIRST", False)
    baseline = V.Stage(0, args.n_layers, args, head=True, tail=True, device="cpu").load(directory)
    tokens = ids(args, 30)
    graphs = [W.WholeBlockGraphs(layer, stage, moe_mode="eager") for layer in stage.layers]
    with torch.no_grad():
        assert_equal_stages(baseline, stage, tokens[:, :13], 0)
        for pos in range(13, 30):
            token = tokens[:, pos:pos + 1]
            expected = baseline.forward(baseline.embed(token), token, pos)
            actual = stage.embed(token)
            for graph in graphs:
                actual = graph._eager(actual, token, pos)
            stage._pos = pos + 1
            assert torch.equal(expected, actual)
            assert all(torch.equal(a, b) for a, b in zip(state(baseline), state(stage)))
    assert sum(cache.stats()["committed_swaps"] for cache in stage._expert_cache.caches.values()) > 0


@pytest.mark.gpu
def test_cuda_fixed_slot_segment_replay_reads_new_input_and_migrated_weights(tmp_path, args):
    if not torch.cuda.is_available():
        pytest.skip("owned CUDA accelerator required")
    from v4_expert_cache import HostExpertPool, FixedSlotCache
    from test_v4_expert_cache import MoE
    host = HostExpertPool.from_moe(MoE(), pin=True, emulation=False, preserve=True)
    cache = FixedSlotCache(host, 1, device="cuda:0")
    # Dense stand-in tests actual CUDA graph lifecycle, not native FP4 quality.
    runtime = SimpleNamespace(pool=host, cache=cache, moe=SimpleNamespace(
        gate=torch.nn.Linear(4, 4).to("cuda:0"), shared_experts=torch.nn.Linear(4, 4).to("cuda:0")),
        device="cuda:0", emulation=False, _require_cache=lambda: None)
    graphs = ExpertSegmentGraphs(runtime, max_bytes=4 << 20)
    with torch.no_grad():
        x = torch.ones(1, 4, device="cuda:0", dtype=torch.bfloat16)
        for eid in (0, 5, 2):
            lease = cache.acquire([eid]); lease.wait_on()
            expected = cache.experts[0](x).clone()
            actual = graphs.run("slot0", cache.experts[0], x)
            assert torch.equal(actual, expected)
            lease.release()
            x += 1
    assert graphs.status()["captures"] == 1
    assert graphs.status()["replays"] == 3
    graphs.close(); cache.close()

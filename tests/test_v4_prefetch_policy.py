"""Bounded optional prediction and router-copy orchestration; CPU is not a DMA benchmark."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import v4_expert_cache as EC
import v4_hybrid as H
from v4_chunked_prefill import ControlledPrefetcher
from test_v4_expert_cache import cache, release


def test_optional_prefetch_preserves_active_leases_and_remains_bounded():
    slots = cache(capacity=3)
    active = slots.acquire([0])
    active.wait_on()
    pointer = slots.experts[active.mapping[0]].w1.weight.data_ptr()
    ticket = slots.try_prefetch([1, 2, 3], max_copies=2, max_evictions=2, reserve_slots=1)
    assert ticket is not None and ticket.released
    assert len(ticket.new_copy_ids) == 1
    assert slots.lookup(0) == active.mapping[0]
    assert slots.experts[active.mapping[0]].w1.weight.data_ptr() == pointer
    assert slots.stats()["live_leases"] == 1
    assert slots.stats()["dma_bytes"] == 0
    active.release()
    assert slots.stats()["live_leases"] == 0


def test_no_room_or_bad_prediction_skips_without_changing_owners():
    slots = cache(capacity=2)
    active = slots.acquire([0, 1])
    before = slots.stats()
    assert slots.try_prefetch([2], reserve_slots=0) is None
    assert slots.stats() == before
    for prediction in ([999], [-1], [True], [None], None):
        assert slots.try_prefetch(prediction, reserve_slots=0) is None
        assert slots.stats() == before
    active.release()
    with slots.acquire([2]) as demanded:
        assert torch.equal(slots.experts[demanded.mapping[2]].w1.weight, slots.pool.experts[2].w1.weight)


def test_optional_prefetch_protects_last_working_set_and_eviction_budget():
    slots = cache(capacity=3)
    release(slots, [0, 1, 2])
    before = slots.stats()
    assert slots.try_prefetch([3, 4], max_copies=2, max_evictions=0, reserve_slots=0) is None
    assert slots.stats() == before
    assert slots.try_prefetch([3, 4], reserve_slots=0, protect_ids=[0, 1, 2]) is None
    assert slots.stats() == before
    ticket = slots.try_prefetch([3, 4], max_copies=2, max_evictions=1,
                                reserve_slots=0, protect_ids=[0, 1])
    assert ticket is not None and ticket.new_copy_ids == (3,)
    assert slots.lookup(0) is not None and slots.lookup(1) is not None
    assert slots.lookup(2) is None
    assert slots.cached_ids(include_pending=True) == {0, 1, 3}


def test_slot_schedule_reuses_unchanged_mapping_and_refreshes_on_epoch():
    class Slots:
        def __init__(self):
            self.copies = 0
        def copy_(self, host, *, non_blocking):
            assert non_blocking
            self.copies += 1
    runtime = H.HybridMoE(SimpleNamespace(), None, SimpleNamespace(epoch=1),
                         ("main", 0), device="cuda:0")
    scratch = {"slots_host": [0, 0], "slots_gpu": Slots()}
    first = runtime._physical_slots([1, 2], {1: 3, 2: 4}, None, scratch)
    second = runtime._physical_slots([1, 2], {1: 3, 2: 4}, None, scratch)
    assert first is second and first.copies == 1
    runtime.pool.epoch += 1
    runtime._physical_slots([1, 2], {1: 3, 2: 4}, None, scratch)
    assert first.copies == 2  # reload generation cannot reuse a stale schedule
    runtime._physical_slots([2, 1], {1: 3, 2: 4}, None, scratch)
    assert first.copies == 3 and scratch["slots_host"] == [4, 3]


def test_real_copy_fault_is_not_disguised_as_optional_prediction_skip(monkeypatch):
    slots = cache()
    def failed_copy(eid, index):
        raise RuntimeError("injected transfer failure")
    monkeypatch.setattr(slots, "_copy_expert", failed_copy)
    with pytest.raises(RuntimeError, match="transfer failure"):
        slots.try_prefetch([1], reserve_slots=0)
    with pytest.raises(EC.ExpertCacheError, match="copy failed"):
        slots.acquire([1])


def test_production_policy_requires_real_history_and_bounds_duplicate_heat():
    policy = ControlledPrefetcher(max_prefetch_slots=2, expert_count=6,
                                 warmup_steps=2, heat_threshold=.1)
    assert policy.predict_prefetch_set([1, 5, 999]) == []
    policy.update_access_history([1])
    assert policy.predict_prefetch_set([1]) == []
    policy.update_access_history([1] * 100 + [2])
    prediction = policy.predict_prefetch_set([4, 999, True, -1, 1])
    assert set(prediction) == {1, 2}
    assert all(0 <= heat <= 1 for heat in policy.expert_heat.values())
    heat = dict(policy.expert_heat)
    policy.reset_job()
    assert dict(policy.expert_heat) == heat and policy.observations == 0
    assert policy.predict_prefetch_set() == []


def test_calibration_metadata_is_detached_and_scratch_inventory_has_no_controls():
    runtime = H.HybridMoE(SimpleNamespace(), None, SimpleNamespace(expert_count=6),
                         ("main", 0), device="cpu", emulation=True)
    runtime.configure_prefetch(True, max_slots=2, warmup_steps=3)
    settings = runtime.prefetch_config()
    assert settings["enabled"] is True and settings["max_slots"] == 2 and settings["warmup_steps"] == 3
    assert "heat" not in settings and "requested" not in settings
    settings["enabled"] = False
    assert runtime.prefetch_config()["enabled"] is True
    tensors = {name: torch.zeros(2) for name in ("ids", "slots_host", "slots_gpu")}
    runtime._router_buffers[1] = dict(tensors, gate_ready=object(), consumer=object())
    inventory = dict(runtime.router_scratch_tensors())
    assert set(inventory) == {"stream_1.ids", "stream_1.slots_host", "stream_1.slots_gpu"}
    assert all(inventory[f"stream_1.{name}"] is tensor for name, tensor in tensors.items())


def test_actual_misprediction_quality_and_reset_are_separate_from_dma_counts():
    slots = cache(capacity=3)
    release(slots, [0, 1])
    runtime = H.HybridMoE(SimpleNamespace(layer_id=0), None, slots.pool,
                         ("main", 0), device="cpu", emulation=True)
    runtime.cache = slots
    runtime.configure_prefetch(True, max_slots=1, warmup_steps=2,
                               heat_threshold=.01, min_cache_spare=0)
    runtime.recent_ids = [0]
    runtime._observe_actual_routes([3])
    runtime._observe_actual_routes([3])
    ticket = runtime.prefetch_for_attention()
    assert ticket is not None and ticket.released and ticket.new_copy_ids == (3,)
    runtime._observe_actual_routes([4])
    assert runtime.prefetch_stats() == dict(requested=1, used=0, wasted=1, skipped=0, candidate_count=1)
    assert slots.stats()["dma_bytes"] == 0 and slots.stats()["live_leases"] == 0
    with slots.acquire([4]) as demanded:
        assert torch.equal(slots.experts[demanded.mapping[4]].w1.weight, slots.pool.experts[4].w1.weight)
    runtime.reset_job()
    assert not any(runtime.prefetch_stats().values())
    assert runtime.prefetch_for_attention() is None


def test_replay_neither_prefetches_nor_counts_as_prediction_warmup():
    slots = cache(capacity=3)
    runtime = H.HybridMoE(SimpleNamespace(layer_id=0), None, slots.pool,
                         ("main", 0), device="cpu", emulation=True)
    runtime.cache = slots
    runtime.configure_prefetch(True, warmup_steps=1)
    runtime.set_phase("replay")
    runtime._observe_actual_routes([1])
    assert runtime.prefetch_for_attention() is None
    assert runtime._prefetch_policy.observations == 0
    assert not any(runtime.prefetch_stats().values())


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_chunk_utility_rejects_invalid_sizes_without_enabling_model_prefill(value):
    from v4_chunked_prefill import ChunkedPrefillExecutor
    with pytest.raises(ValueError):
        ChunkedPrefillExecutor(default_chunk_size=value)


def test_chunk_utility_rejects_overrun_and_handles_empty_prompt():
    from v4_chunked_prefill import ChunkedPrefillExecutor, PrefillChunkState
    state = PrefillChunkState("one", total_prompt_length=3, chunk_size=2)
    with pytest.raises(ValueError):
        state.advance(3)
    assert state.processed_tokens == 0
    executor = ChunkedPrefillExecutor()
    assert executor.execute_chunked(torch.empty((1, 0), dtype=torch.long),
                                    lambda *args: pytest.fail("empty prompt was processed")) == []


def test_router_readback_queues_shared_work_before_the_host_wait(monkeypatch):
    """Control-flow proof with events; no claim about actual PCIe overlap."""
    log = []
    class Event:
        def __init__(self, name):
            self.name = name
        def record(self, stream):
            log.append(("record", self.name, stream.cuda_stream))
        def synchronize(self):
            log.append(("sync", self.name))
    class Stream:
        def __init__(self, number):
            self.cuda_stream = number
        def wait_event(self, event):
            log.append(("wait", event.name))
    class HostIds:
        def __getitem__(self, key):
            return self
        def copy_(self, source, *, non_blocking):
            log.append(("copy", non_blocking))
        def tolist(self):
            log.append(("tolist",))
            return [[1, 2]]
    @contextmanager
    def stream_scope(stream):
        log.append(("enter_stream", stream.cuda_stream))
        yield
        log.append(("exit_stream", stream.cuda_stream))
    consumer, copy = Stream(1), Stream(2)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: consumer)
    monkeypatch.setattr(torch.cuda, "stream", stream_scope)
    moe = SimpleNamespace(shared_experts=lambda x: log.append(("shared",)) or "shared-output")
    runtime = H.HybridMoE(moe, None, SimpleNamespace(expert_count=6), ("main", 0), device="cuda:0")
    runtime.configure_router_overlap(True)  # experimental ordering is explicit, never the production default
    runtime._router_stream = copy
    scratch = {"ids": HostIds(), "gate_ready": Event("gate"), "ids_ready": Event("ids")}
    monkeypatch.setattr(runtime, "_router_buffers_for", lambda indices, stream: scratch)
    indices = SimpleNamespace(shape=(1, 2), detach=lambda: "gpu-ids")
    rows, shared, returned = runtime._read_routes_and_shared(None, indices)
    assert rows == [[1, 2]] and shared == "shared-output" and returned is scratch
    assert log.index(("wait", "gate")) < log.index(("copy", True))
    assert log.index(("copy", True)) < log.index(("shared",)) < log.index(("sync", "ids"))
    assert log.index(("sync", "ids")) < log.index(("tolist",))


def test_prefetch_mispredictions_preserve_tiny_stage_math_and_reset(tmp_path):
    import v4_ref_cpu as REF
    from test_v4_hybrid import checkpoint, stage, assert_equal_stages, ids
    args = REF.cpu_args(n_layers=4, n_routed_experts=6, n_activated_experts=2,
                        compress_ratios=(0, 4, 8, 0, 0, 0), dspark_target_layer_ids=(1, 2, 3))
    directory, _ = checkpoint(tmp_path / "weights", args)
    resident = stage(directory, args, hybrid=False)
    import v4_stage as V4
    cached = V4.Stage(0, args.n_layers, args, head=True, tail=True, device="cpu",
                      expert_placement="ram", expert_cache_reference=True,
                      expert_cache_slots=3, expert_prefetch=True)
    cached.load(str(directory))
    for layer in cached.layers:
        runtime = layer.ffn._hybrid_runtime
        runtime.configure_prefetch(True, max_slots=1, warmup_steps=2,
                                   heat_threshold=.01, min_cache_spare=0)
        assert runtime.prefetch_for_attention() is None
        assert runtime.prefetch_stats()["requested"] == 0
    sequence = ids(args, 16)
    assert_equal_stages(resident, cached, sequence[:, :6], 0)
    for position in range(6, 16):
        assert_equal_stages(resident, cached, sequence[:, position:position + 1], position)
        assert all(layer.ffn._hybrid_runtime.cache.stats()["live_leases"] == 0 for layer in cached.layers)
    assert any(layer.ffn._hybrid_runtime.prefetch_stats()["candidate_count"] > 0 for layer in cached.layers)
    for layer in cached.layers:
        runtime = layer.ffn._hybrid_runtime
        stats = runtime.prefetch_stats()
        assert stats["used"] + stats["wasted"] == stats["requested"]
        assert runtime.cache.stats()["dma_bytes"] == 0
        runtime.reset_job()
        assert not any(runtime.prefetch_stats().values())


def test_default_miss_schedule_queues_weight_copy_before_shared(tmp_path, monkeypatch):
    import v4_ref_cpu as REF
    from test_v4_hybrid import checkpoint, stage, run
    args = REF.cpu_args(n_layers=4, n_routed_experts=6, n_activated_experts=2,
                        compress_ratios=(0, 4, 8, 0, 0, 0), dspark_target_layer_ids=(1, 2, 3))
    directory, _ = checkpoint(tmp_path / "weights", args)
    cached = stage(directory, args, hybrid=True, slots=3)
    runtime = cached.layers[0].ffn._hybrid_runtime
    assert runtime.router_overlap_enabled is False
    log = []
    acquire = runtime.cache.acquire
    def observed(ids, **kwargs):
        result = acquire(ids, **kwargs)
        log.append("acquire")
        return result
    monkeypatch.setattr(runtime.cache, "acquire", observed)
    hook = runtime.moe.shared_experts.register_forward_pre_hook(lambda m, args: log.append("shared"))
    try:
        run(cached, torch.tensor([[1, 2, 3, 4]]), 0)
    finally:
        hook.remove()
    assert log.index("acquire") < log.index("shared")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pinned-router acceptance requires GPU")
def test_cuda_router_and_slot_scratch_reuse_and_stream_separation():
    device = "cuda:0"
    moe = SimpleNamespace(shared_experts=torch.nn.Linear(4, 4, bias=False, device=device))
    runtime = H.HybridMoE(moe, None, SimpleNamespace(expert_count=6), ("main", 0), device=device)
    runtime.configure_router_overlap(True)
    x = torch.ones(1, 4, device=device)
    indices = torch.tensor([[1, 2]], device=device)
    rows, shared, first = runtime._read_routes_and_shared(x, indices)
    assert rows == [[1, 2]] and first["ids"].is_pinned()
    assert torch.equal(shared, moe.shared_experts(x))
    first_slots = runtime._physical_slots(rows[0], {1: 3, 2: 4}, x, first)
    rows, _, again = runtime._read_routes_and_shared(x, indices)
    second_slots = runtime._physical_slots(rows[0], {1: 0, 2: 1}, x, again)
    assert first is again and first_slots.data_ptr() == second_slots.data_ptr()
    assert second_slots.cpu().tolist() == [0, 1]
    another = torch.cuda.Stream(device=device)
    with torch.cuda.stream(another):
        _, _, separate = runtime._read_routes_and_shared(x, indices)
    assert separate is not first and separate["slots_gpu"].data_ptr() != first["slots_gpu"].data_ptr()

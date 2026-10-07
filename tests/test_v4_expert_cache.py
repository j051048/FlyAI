"""Local cache correctness on CPU; CUDA DMA acceptance lives in separate GPU tests."""
import pytest
import torch
from torch import nn
import torch.nn.functional as F

import v4_expert_cache as EC


class Matrix(nn.Module):
    def __init__(self, rows, columns, dtype, *, device="cpu", scales=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(rows, columns, dtype=dtype, device=device), requires_grad=False)
        if scales:
            self.scale = nn.Parameter(torch.empty(rows, 1, dtype=torch.float8_e8m0fnu, device=device), requires_grad=False)
        else:
            self.register_parameter("scale", None)
        self.weight.scale = self.scale

    def forward(self, x):
        return F.linear(x, self.weight)


class Expert(nn.Module):
    def __init__(self, dim, inter, dtype, *, device="cpu", scales=False):
        super().__init__()
        packed = dtype == torch.float4_e2m1fn_x2
        self.w1 = Matrix(inter, dim // 2 if packed else dim, dtype, device=device, scales=scales)
        self.w3 = Matrix(inter, dim // 2 if packed else dim, dtype, device=device, scales=scales)
        self.w2 = Matrix(dim, inter // 2 if packed else inter, dtype, device=device, scales=scales)
        self.swiglu_limit = 0.0

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, count=6, dim=4, inter=4, dtype=torch.bfloat16, *, device="cpu", scales=False):
        super().__init__()
        self.experts = nn.ModuleList([Expert(dim, inter, dtype, device=device, scales=scales) for _ in range(count)])
        self.shared_experts = nn.Linear(dim, dim, bias=False, device=device, dtype=torch.bfloat16)
        if device != "meta":
            for eid, expert in enumerate(self.experts):
                for projection in ("w1", "w2", "w3"):
                    linear = getattr(expert, projection)
                    if dtype == torch.float4_e2m1fn_x2:
                        linear.weight.view(torch.uint8).fill_(0x12 + eid)
                    else:
                        linear.weight.fill_(0.01 * (eid + 1))
                    if linear.scale is not None:
                        linear.scale.view(torch.uint8).fill_(127 + eid)


def pool(count=6, **kwargs):
    moe = MoE(count, **kwargs)
    return EC.HostExpertPool.from_moe(moe, pin=False, emulation=True, preserve=True)


def cache(capacity=2, **kwargs):
    host = pool(**kwargs)
    return EC.FixedSlotCache(host, capacity, device="cpu", emulation=True)


def release(cache, ids):
    lease = cache.acquire(ids)
    lease.wait_on()
    lease.release()
    return lease


def snapshot(module):
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


class PendingEvent:
    def __init__(self):
        self.done = False
        self.synchronizations = 0

    def query(self):
        return self.done

    def synchronize(self):
        self.synchronizations += 1
        self.done = True


def test_meta_experts_become_canonical_cpu_banks_and_strict_load_marks_loaded():
    source = MoE()
    state = snapshot(source)
    target = MoE(device="meta")
    shared = target.shared_experts.weight
    original_keys = set(target.state_dict())
    host = EC.HostExpertPool.from_moe(target, pin=False, emulation=True)
    assert not host.loaded
    assert target.shared_experts.weight is shared and shared.is_meta
    assert set(target.state_dict()) == original_keys
    assert all(parameter.device.type == "cpu" for expert in target.experts for parameter in expert.parameters())
    # Resident materialization belongs to Stage; emulate that one submodule here.
    target.shared_experts.to_empty(device="cpu")
    target.load_state_dict(state, strict=True)
    assert host.loaded and host.epoch == 1
    slots = EC.FixedSlotCache(host, 2, device="cpu", emulation=True)
    with slots.acquire([1, 4]) as lease:
        for eid, index in lease.mapping.items():
            x = torch.arange(4, dtype=torch.bfloat16).reshape(1, 4)
            assert torch.equal(source.experts[eid](x), slots.experts[index](x))


def test_fp4_packed_bytes_scales_and_weight_scale_aliases_are_preserved():
    host = pool(dim=32, inter=32, dtype=torch.float4_e2m1fn_x2, scales=True)
    slots = EC.FixedSlotCache(host, 2, device="cpu", emulation=True)
    assert host.bytes_per_expert == 3 * 32 * 16 + 3 * 32
    assert host.banks["w13"].shape == (6, 64, 16)
    assert host.banks["w13_s"].shape == (6, 64, 1)
    assert host.banks["w2"].shape == (6, 32, 16)
    with slots.acquire([5, 1]) as lease:
        for eid, index in lease.mapping.items():
            for projection in ("w1", "w2", "w3"):
                source, destination = getattr(host.experts[eid], projection), getattr(slots.experts[index], projection)
                assert destination.weight.scale is destination.scale
                assert source.weight.scale is source.scale
                assert torch.equal(source.weight.view(torch.uint8), destination.weight.view(torch.uint8))
                assert torch.equal(source.scale.view(torch.uint8), destination.scale.view(torch.uint8))
    assert slots.stats()["dma_bytes"] == 0
    assert lease.dma_timings() == []


def test_meta_fp4_strict_loader_fills_cpu_canonical_banks_without_conversion():
    source = MoE(2, dim=32, inter=32, dtype=torch.float4_e2m1fn_x2, scales=True)
    target = MoE(2, dim=32, inter=32, dtype=torch.float4_e2m1fn_x2, device="meta", scales=True)
    host = EC.HostExpertPool.from_moe(target, pin=False, emulation=True)
    target.shared_experts.to_empty(device="cpu")
    target.load_state_dict(snapshot(source), strict=True)
    assert host.loaded
    slots = EC.FixedSlotCache(host, 1, device="cpu", emulation=True)
    with slots.acquire([1]) as lease:
        for projection in ("w1", "w2", "w3"):
            original = getattr(source.experts[1], projection)
            cached = getattr(slots.experts[lease.mapping[1]], projection)
            assert torch.equal(original.weight.view(torch.uint8), cached.weight.view(torch.uint8))
            assert torch.equal(original.scale.view(torch.uint8), cached.scale.view(torch.uint8))


def test_cache_slots_do_not_rebind_source_parameters_or_change_captured_addresses():
    slots = cache()
    parameters = [expert.w1.weight for expert in slots.pool.experts]
    pointers = {key: bank.data_ptr() for key, bank in slots.banks.items() if bank is not None}
    for eid in [0, 1, 2, 3, 0, 4, 5, 1]:
        with slots.acquire([eid]) as lease:
            slot = lease.mapping[eid]
            assert torch.equal(slots.experts[slot].w1.weight, slots.pool.experts[eid].w1.weight)
    assert [expert.w1.weight for expert in slots.pool.experts] == parameters
    assert {key: bank.data_ptr() for key, bank in slots.banks.items() if bank is not None} == pointers
    assert not any("cache" in name for name in slots.pool.moe.state_dict())


def test_hits_deduplicate_ids_and_lfu_eviction_is_deterministic():
    slots = cache()
    first = release(slots, [0, 0, 1])
    assert first.new_copy_ids == (0, 1) and first.copied_bytes == 2 * slots.pool.bytes_per_expert
    second = release(slots, [0, 0])
    assert second.hit_ids == (0,) and second.copied_bytes == 0
    release(slots, [2])
    assert slots.is_resident(0) and slots.is_resident(2) and not slots.is_resident(1)
    assert slots.stats()["hits"] == 1 and slots.stats()["new_copies"] == 3
    assert slots.stats()["dma_bytes"] == 0 and slots.stats()["mode"] == "cpu_reference"


def test_lfu_decay_moves_with_the_conversation_instead_of_preserving_old_hot_experts():
    host = pool()
    slots = EC.FixedSlotCache(host, 2, device="cpu", emulation=True, decay_interval=2)
    release(slots, [0, 1])
    for _ in range(5):
        release(slots, [0])
    for _ in range(10):
        release(slots, [1])
    release(slots, [2])
    assert not slots.is_resident(0) and slots.is_resident(1) and slots.is_resident(2)


def test_live_leases_cannot_be_evicted_and_capacity_failure_is_atomic():
    slots = cache(capacity=1)
    lease = slots.acquire([0])
    before = slots.stats()
    with pytest.raises(EC.ExpertCacheError, match="active leases"):
        slots.acquire([1])
    assert slots.stats() == before
    lease.wait_on()
    lease.release()
    release(slots, [1])
    assert slots.is_resident(1)
    with pytest.raises(EC.ExpertCacheError, match="released"):
        lease.wait_on()
    with pytest.raises(EC.ExpertCacheError, match="released"):
        lease.release()


def test_pending_copy_dedup_does_not_publish_a_resident_hit_before_ready():
    slots = cache(capacity=1)
    first = release(slots, [0])
    index = first.mapping[0]
    event = PendingEvent()
    slots._slots[index].ready = event  # Model an unfinished DMA event without claiming hardware.
    assert slots.lookup(0) is None
    assert slots.lookup(0, include_pending=True) == index
    copies = slots.stats()["new_copies"]
    lease = slots.acquire([0])
    assert lease.pending_ids == (0,) and lease.miss_ids == (0,) and lease.hit_ids == ()
    assert lease.new_copy_ids == () and lease.copied_bytes == 0
    assert slots.stats()["new_copies"] == copies
    lease.wait_on()
    assert event.synchronizations == 1 and slots.is_resident(0)
    lease.release()


def test_eviction_waits_all_prior_released_consumer_events():
    slots = cache(capacity=1)
    first, second = slots.acquire([0]), slots.acquire([0])
    event_a, event_b = PendingEvent(), PendingEvent()
    first.release(done_event=event_a)
    with pytest.raises(EC.ExpertCacheError, match="active leases"):
        slots.acquire([1])
    second.release(done_event=event_b)
    release(slots, [1])
    assert event_a.synchronizations == event_b.synchronizations == 1


def test_prefetch_releases_ticket_but_keeps_its_content_for_demand():
    slots = cache()
    ticket = slots.prefetch([1, 2])
    assert ticket.released and slots.stats()["live_leases"] == 0
    assert slots.stats()["acquires"] == 0
    demanded = release(slots, [2, 1])
    assert demanded.hit_ids == (2, 1) and demanded.copied_bytes == 0
    assert slots.stats()["acquires"] == 1


def test_full_and_partial_weight_reloads_invalidate_cache_before_copying():
    slots = cache(capacity=1)
    pointers = {key: bank.data_ptr() for key, bank in slots.banks.items() if bank is not None}
    release(slots, [0])
    previous_epoch = slots.pool.epoch
    state = snapshot(slots.pool.moe)
    state["experts.0.w1.weight"].fill_(0.8)
    slots.pool.moe.load_state_dict(state, strict=True)
    assert slots.pool.epoch == previous_epoch + 1 and slots.pool.loaded
    assert slots.lookup(0) is None
    with slots.acquire([0]) as lease:
        assert torch.equal(slots.experts[lease.mapping[0]].w1.weight, state["experts.0.w1.weight"])
    linear = slots.pool.experts[0].w2
    state = snapshot(linear)
    state["weight"].fill_(0.9)
    linear.load_state_dict(state, strict=True)
    assert slots.lookup(0) is None and slots.pool.loaded
    assert {key: bank.data_ptr() for key, bank in slots.banks.items() if bank is not None} == pointers


def test_reload_with_a_live_lease_fails_before_source_mutation():
    slots = cache(capacity=1)
    before = slots.pool.experts[0].w1.weight.clone()
    state = snapshot(slots.pool.moe)
    state["experts.0.w1.weight"].fill_(0.8)
    lease = slots.acquire([0])
    epoch = slots.pool.epoch
    with pytest.raises(EC.ExpertCacheError, match="active"):
        slots.pool.moe.load_state_dict(state, strict=True)
    assert slots.pool.loaded and slots.pool.epoch == epoch
    assert torch.equal(before, slots.pool.experts[0].w1.weight)
    lease.release()


def test_failed_strict_reload_leaves_pool_unloaded_and_cannot_serve_stale_cache():
    slots = cache(capacity=1)
    release(slots, [0])
    state = snapshot(slots.pool.moe)
    state["experts.0.w1.weight"] = torch.zeros(1, 1, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="size mismatch"):
        slots.pool.moe.load_state_dict(state, strict=True)
    assert not slots.pool.loaded and slots.lookup(0) is None
    with pytest.raises(EC.ExpertCacheError, match="loading"):
        slots.acquire([0])


def test_manual_reload_preserves_pending_consumer_dependency_for_next_overwrite():
    slots = cache(capacity=1)
    lease = slots.acquire([0])
    consumed = PendingEvent()
    lease.release(done_event=consumed)
    with slots.pool.reloading():
        slots.pool.experts[0].w1.weight.fill_(0.5)
    assert consumed.synchronizations == 0  # Host reload did not overwrite GPU/slot bytes.
    release(slots, [0])
    assert consumed.synchronizations == 1


def test_manager_budget_includes_main_and_mtp_and_registration_freezes():
    a, b = pool(), pool()
    manager = EC.SharedCacheManager(7 * a.bytes_per_expert, device="cpu", emulation=True)
    manager.register(a, ("main", 40), min_slots=2)
    manager.register(b, ("draft", 0), min_slots=2)
    caches = manager.allocate()
    assert [caches[key].capacity for key in manager.pools] == [4, 3]
    assert manager.allocated_bytes == manager.budget_bytes
    assert manager.host_bytes == a.host_bytes + b.host_bytes
    assert manager.allocate() is caches
    with pytest.raises(EC.ExpertCacheError, match="frozen"):
        manager.register(pool(), ("draft", 1))


def test_too_small_stage_budget_never_allocates_partial_caches():
    a, b = pool(), pool()
    manager = EC.SharedCacheManager(3 * a.bytes_per_expert, device="cpu", emulation=True)
    manager.register(a, "main", min_slots=2)
    manager.register(b, "draft", min_slots=2)
    with pytest.raises(EC.ExpertCacheError, match="below required"):
        manager.allocate()
    assert manager.caches == {} and manager.allocated_bytes == 0
    assert not list(a._caches) and not list(b._caches)


def test_manager_allocation_exception_closes_already_created_caches(monkeypatch):
    a, b = pool(), pool()
    manager = EC.SharedCacheManager(slots_per_pool=2, device="cpu", emulation=True)
    manager.register(a, "main")
    manager.register(b, "draft")
    real = EC.FixedSlotCache
    def construct(pool, *args, **kwargs):
        if pool is b:
            raise MemoryError("allocation fixture")
        return real(pool, *args, **kwargs)
    monkeypatch.setattr(EC, "FixedSlotCache", construct)
    with pytest.raises(MemoryError):
        manager.allocate()
    assert manager.caches == {} and manager.allocated_bytes == 0
    assert not list(a._caches)


@pytest.mark.parametrize("ids", [[True], [-1], [6], [0, 1, 2], [torch.tensor(0)]])
def test_invalid_or_oversized_requests_never_mutate_cache(ids):
    slots = cache()
    before = slots.stats()
    with pytest.raises(EC.ExpertCacheError):
        slots.acquire(ids)
    assert slots.stats() == before


def test_copy_failure_does_not_publish_partially_written_slots_and_faults_cache(monkeypatch):
    slots = cache()
    real = slots._copy_expert
    def fail_second(eid, index):
        if eid == 1:
            raise RuntimeError("copy fixture")
        return real(eid, index)
    monkeypatch.setattr(slots, "_copy_expert", fail_second)
    with pytest.raises(RuntimeError, match="fixture"):
        slots.acquire([0, 1])
    assert slots.lookup(0) is None and slots.lookup(1) is None
    with pytest.raises(EC.ExpertCacheError, match="failed"):
        slots.acquire([0])
    slots.close()


def test_production_and_reference_modes_cannot_be_silently_mixed():
    with pytest.raises(EC.ExpertCacheError, match="pin=False"):
        EC.HostExpertPool.from_moe(MoE(), emulation=True)
    with pytest.raises(EC.ExpertCacheError, match="require pinned"):
        EC.HostExpertPool.from_moe(MoE(), pin=False)
    with pytest.raises(EC.ExpertCacheError, match="modes"):
        EC.FixedSlotCache(pool(), 2, device="cpu", emulation=False)
    with pytest.raises(EC.ExpertCacheError, match="CPU slots"):
        EC.FixedSlotCache(pool(), 2, device="cuda", emulation=True)

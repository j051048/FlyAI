"""Actual exclusive hot/cold storage and byte-preserving migration transactions."""
import pytest
import torch
from contextlib import contextmanager

import v4_expert_cache as EC
from v4_expert_tiers import ExclusiveExpertPool, ExclusiveTierCache
from test_v4_expert_cache import MoE, snapshot, PendingEvent


def build(*, count=6, hot=3, packed=False):
    kwargs = dict(dim=32, inter=32, dtype=torch.float4_e2m1fn_x2, scales=True) if packed else {}
    model = MoE(count, **kwargs)
    expected = snapshot(model)
    pool = ExclusiveExpertPool(model, hot_slots=hot, device="cpu", identity=("main", 7),
                              pin=False, emulation=True, preserve=True)
    cache = ExclusiveTierCache(pool, hot, device="cpu", emulation=True)
    return pool, cache, expected


def assert_bytes(module, expected):
    actual = module.state_dict()
    assert set(actual) == set(expected)
    assert all(torch.equal(value.view(torch.uint8), expected[name].view(torch.uint8)) for name, value in actual.items())


@pytest.mark.parametrize("packed", [False, True])
def test_constructs_only_cold_complement_and_one_exchange_without_a_hidden_full_bank(packed):
    pool, cache, expected = build(packed=packed)
    assert all(value is None for value in pool.banks.values())
    cold = [(name, value) for name, value in pool.storage_tensors() if not name.startswith("hot.")]
    assert sum(value.numel() * value.element_size() for _, value in cold) == pool.host_bytes
    assert pool.host_bytes == 4 * pool.bytes_per_expert  # six total, three hot, one exchange
    assert pool.host_bytes < pool.expert_count * pool.bytes_per_expert
    assert cache.banks is pool.hot_banks
    assert cache.cached_ids() == {0, 1, 2}
    assert_bytes(pool.moe, expected)
    assert cache.stats()["emulated_hot_cpu_bytes"] == pool.hot_bytes
    assert cache.stats()["qualification"] == "cpu_reference_only"


@pytest.mark.parametrize("packed", [False, True])
def test_many_swaps_keep_every_original_expert_and_reuse_fixed_allocations(packed):
    pool, cache, expected = build(packed=packed)
    # Real distinct backing storages rotate between cold owner and exchange;
    # no whole-host copy or new exchange is allocated per request.
    addresses = {value.untyped_storage().data_ptr() for _, value in pool.storage_tensors()}
    for ids in ([4, 5], [0, 1, 2], [3, 4, 5], [1], [2, 5], [0, 3]) * 5:
        with cache.acquire(ids) as lease:
            for eid, index in lease.mapping.items():
                for name, value in cache.experts[index].state_dict().items():
                    original = expected[f"experts.{eid}.{name}"]
                    assert torch.equal(value.view(torch.uint8), original.view(torch.uint8))
        assert_bytes(pool.moe, expected)
        assert len(pool._cold) == 3 and len(pool._hot_owners) == 3
        assert {value.untyped_storage().data_ptr() for _, value in pool.storage_tensors()} == addresses
    assert cache.stats()["committed_swaps"] > 30
    assert cache.stats()["dma_bytes"] == 0


def test_active_lease_and_old_consumers_protect_the_unique_gpu_owner():
    pool, cache, expected = build(count=4, hot=2)
    first = cache.acquire([0, 1])
    with pytest.raises(EC.ExpertCacheError, match="active leases"):
        cache.acquire([2])
    a, b = PendingEvent(), PendingEvent()
    first.release(done_event=a)
    # Both event fences are honored before an original GPU-only owner is reused.
    for slot in cache._slots:
        slot.use_events.append(b)
    with cache.acquire([2, 3]):
        pass
    assert a.synchronizations >= 1 and b.synchronizations >= 1
    assert_bytes(pool.moe, expected)


def test_failure_after_preserving_the_old_owner_never_discards_its_only_weights(monkeypatch):
    pool, cache, expected = build(count=4, hot=2)
    target = cache.banks["w13"].untyped_storage().data_ptr()
    copy = torch.Tensor.copy_

    def fail(tensor, source, *args, **kwargs):
        if tensor.untyped_storage().data_ptr() == target:
            raise RuntimeError("injected incoming transfer failure")
        return copy(tensor, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail)
    with pytest.raises(RuntimeError, match="incoming transfer"):
        cache.acquire([3])
    assert cache._faulted and pool._poisoned
    assert_bytes(pool.moe, expected)
    with pytest.raises(EC.ExpertCacheError, match="copy failed"):
        cache.acquire([0])
    with pytest.raises(EC.ExpertCacheError, match="restart"):
        pool.begin_reload()
    cache.close()
    with pytest.raises(EC.ExpertCacheError, match="restart"):
        ExclusiveTierCache(pool, 2, device="cpu", emulation=True)


def test_reload_updates_canonical_hot_and_cold_bytes_and_invalidates_old_leases():
    pool, cache, _ = build()
    with cache.acquire([3, 5]):
        pass
    changed = MoE()
    with torch.no_grad():
        for parameter in changed.parameters():
            parameter.add_(.125)
    expected = snapshot(changed)
    epoch = pool.epoch
    pool.moe.load_state_dict(expected, strict=True)
    assert pool.loaded and pool.epoch > epoch
    for eid in range(6):
        with cache.acquire([eid]) as lease:
            assert lease.epoch == pool.epoch
            assert_bytes(cache.experts[lease.mapping[eid]],
                         {name.removeprefix(f"experts.{eid}."): value for name, value in expected.items()
                          if name.startswith(f"experts.{eid}.")})
    assert_bytes(pool.moe, expected)


def test_cache_close_keeps_canonical_weights_readable_and_can_rewrap_same_hot_banks():
    pool, cache, expected = build()
    with cache.acquire([4, 5]):
        pass
    pointer = cache.banks["w13"].data_ptr()
    cache.close()
    assert_bytes(pool.moe, expected)
    again = ExclusiveTierCache(pool, 3, device="cpu", emulation=True)
    assert again.banks["w13"].data_ptr() == pointer
    for eid in range(6):
        with again.acquire([eid]):
            pass
    assert_bytes(pool.moe, expected)


def test_lease_binds_pool_identity_weight_epoch_and_slot_generation():
    pool, cache, _ = build()
    lease = cache.acquire([0])
    index = lease.mapping[0]
    cache._generations[index] += 1
    with pytest.raises(EC.ExpertCacheError, match="generation"):
        lease.wait_on()
    cache._generations[index] -= 1
    identity = pool.identity
    pool.identity = ("another-pool", "main", 7, "cpu")
    with pytest.raises(EC.ExpertCacheError, match="owner"):
        lease.wait_on()
    pool.identity = identity
    lease.release()


def test_manager_checks_actual_cold_exchange_and_reference_hot_budget_before_allocation(monkeypatch):
    model = MoE()
    size = sum(spec.bytes_per_expert for spec in EC._bank_specs(list(model.experts)))
    manager = EC.StageBudgetManager(slots_per_pool=3, storage_mode="exclusive",
        host_budget_bytes=7 * size - 1, device="cpu", emulation=True)
    import v4_expert_tiers as tiers
    monkeypatch.setattr(tiers, "_allocate_banks", lambda *a, **kw: pytest.fail("allocation preceded the host quota"))
    with pytest.raises(EC.ExpertCacheError, match="host tensor budget"):
        manager.new_pool(model, ("main", 0), preserve=True)


def test_exclusive_requires_fixed_capacity_and_bounded_host_admission():
    with pytest.raises(EC.ExpertCacheError, match="fixed slots"):
        EC.StageBudgetManager(budget_bytes=1024, storage_mode="exclusive")
    with pytest.raises(EC.ExpertCacheError, match="host tensor budget"):
        EC.StageBudgetManager(slots_per_pool=2, storage_mode="exclusive")


def test_shared_copy_backpressure_preserves_owners_and_optional_prefetch_can_skip():
    pool, cache, expected = build()
    class Busy:
        @contextmanager
        def copy(self, **kwargs):
            raise TimeoutError("another GPU owns the copy budget")
            yield
    pool._host_compute = Busy()
    before = cache.stats()
    with pytest.raises(EC.ExpertCopyAdmissionError, match="before any expert transfer"):
        cache.acquire([4])
    assert cache.stats() == before
    assert not cache._faulted and not pool._poisoned
    assert cache.try_prefetch([4], max_copies=1, reserve_slots=0) is None
    assert_bytes(pool.moe, expected)
    pool._host_compute = None
    with cache.acquire([4]):
        pass
    assert_bytes(pool.moe, expected)


def test_copy_budget_covers_the_real_transfer_completion_before_release():
    pool, cache, expected = build()
    events = []
    class Budget:
        @contextmanager
        def copy(self, *, completion):
            events.append("acquire")
            try:
                yield
            finally:
                completion()
                assert_bytes(pool.moe, expected)
                events.append("released_after_fence")
    pool._host_compute = Budget()
    with cache.acquire([4, 5]):
        pass
    assert events == ["acquire", "released_after_fence"] * 2
    assert_bytes(pool.moe, expected)


@pytest.mark.parametrize("cancel", [False, True])
def test_existing_copy_lookahead_preserves_exclusive_weights_and_releases_leases(cancel):
    from v4_prefill_expert_pipeline import ExpertBatchPipeline
    pool, cache, expected = build(count=6, hot=4, packed=True)
    pipeline = ExpertBatchPipeline(cache, range(6), depth=2, batch_size=2)
    seen = []
    with pipeline:
        assert pipeline.config["source"] == "exclusive_local_tiers"
        for batch, lease in pipeline:
            with lease:
                for eid in batch:
                    assert_bytes(cache.experts[lease.mapping[eid]], {
                        name.removeprefix(f"experts.{eid}."): value
                        for name, value in expected.items() if name.startswith(f"experts.{eid}.")})
                    seen.append(eid)
            if cancel:
                break
    assert seen == ([0, 1] if cancel else list(range(6)))
    assert cache.stats()["live_leases"] == 0
    assert pipeline.stats()["pending_slots"] == 0
    assert_bytes(pool.moe, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="exclusive tier DMA qualification requires actual CUDA")
@pytest.mark.parametrize("packed", [False, True])
def test_actual_cuda_exclusive_migrations_preserve_all_checkpoint_bytes(packed):
    kwargs = dict(dim=32, inter=32, dtype=torch.float4_e2m1fn_x2, scales=True) if packed else {}
    original = MoE(6, **kwargs)
    expected = snapshot(original)
    pool = ExclusiveExpertPool(original, hot_slots=3, device="cuda:0", identity=("main", 4),
                              pin=True, emulation=False, preserve=True)
    cache = ExclusiveTierCache(pool, 3, device="cuda:0", emulation=False)
    for batch in ([3, 4], [0, 1, 2], [4, 5], [0, 3, 5]) * 8:
        with cache.acquire(batch) as lease:
            for eid, index in lease.mapping.items():
                for name, value in cache.experts[index].state_dict().items():
                    assert torch.equal(value.cpu().view(torch.uint8), expected[f"experts.{eid}.{name}"].view(torch.uint8))
        pool.drain(wait=True)
        for name, value in original.state_dict().items():
            assert torch.equal(value.cpu().view(torch.uint8), expected[name].view(torch.uint8))
    assert cache.stats()["dma_bytes"] > 0
    assert cache.stats()["committed_swaps"] > 30
    assert all(t.is_pinned() for name, t in pool.storage_tensors() if not name.startswith("hot."))
    cache.close()

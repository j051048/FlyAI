"""Production Hybrid/Stage uses exclusive tiers before its first weight load."""
import pytest
import torch

import v4_stage as V
from test_v4_hybrid import args, checkpoint, ids, assert_equal_stages


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def exclusive(directory, args, *, dspark=False):
    return V.Stage(0, args.n_layers, args, head=True, tail=True, dspark=dspark,
        device="cpu", expert_placement="ram", expert_cache_reference=True,
        expert_cache_slots=3, expert_storage="exclusive", expert_storage_host_bytes=8 << 20).load(directory)


def test_real_stage_constructs_cold_only_then_matches_prefill_decode_and_reset(tmp_path, args):
    directory, _ = checkpoint(tmp_path / "weights", args)
    control = V.Stage(0, args.n_layers, args, head=True, tail=True, device="cpu").load(directory)
    run = exclusive(directory, args)
    manager = run._expert_cache
    assert manager.storage_mode == "exclusive"
    assert manager.preallocated_device_bytes == manager.allocated_bytes
    assert all(pool.host_bytes < pool.bytes_per_expert * pool.expert_count for pool in manager.pools.values())
    assert all(all(value is None for value in pool.banks.values()) for pool in manager.pools.values())
    assert all(pool.model_identity == "unverified-cpu-reference" for pool in manager.pools.values())
    tokens = ids(args, 23)
    with torch.no_grad():
        assert_equal_stages(run, control, tokens[:, :7], 0)
        for pos in range(7, 23):
            assert_equal_stages(run, control, tokens[:, pos:pos + 1], pos)
    assert sum(cache.stats()["committed_swaps"] for cache in manager.caches.values()) > 10
    # Storage ownership survives resets; no full host reload is used to recover.
    pointers = {name: value.data_ptr() for name, value in manager.storage_tensors() if ".hot." in name}
    run.reset(); control.reset()
    with torch.no_grad():
        assert_equal_stages(run, control, tokens[:, :5], 0)
        for pos in range(5, 13):
            assert_equal_stages(run, control, tokens[:, pos:pos + 1], pos)
    assert pointers == {name: value.data_ptr() for name, value in manager.storage_tensors() if ".hot." in name}


def test_mtp_pools_share_the_fixed_stage_host_budget_and_use_the_exclusive_backend(tmp_path, args):
    directory, _ = checkpoint(tmp_path / "weights", args)
    run = exclusive(directory, args, dspark=True)
    manager = run._expert_cache
    assert {role for role, _ in manager.pools} == {"main", "draft"}
    assert len(manager.pools) == args.n_layers + args.n_mtp_layers
    assert all(pool.storage_mode == "exclusive" for pool in manager.pools.values())
    assert manager.host_bytes + manager.preallocated_device_bytes <= manager.host_budget_bytes
    assert manager.allocated_bytes == manager.preallocated_device_bytes


def test_verified_native_checkpoint_identity_keeps_its_catalogue_namespace(tmp_path, args):
    from shard import weight_artifacts as artifacts
    directory, _ = checkpoint(tmp_path / "weights", args)
    catalog, pack = artifacts.catalogue_directory(directory, "test/native-v4-exclusive")
    artifacts.write_metadata(directory, catalog, pack)
    run = exclusive(directory, args, dspark=True)
    assert run._artifact_verification["payload_integrity_verified"] is True
    assert catalog["checkpoint_id"].startswith("tensor-sha256:")
    assert run._expert_cache.model_identity == catalog["checkpoint_id"]
    assert all(pool.model_identity == catalog["checkpoint_id"] for pool in run._expert_cache.pools.values())


def test_host_compute_owner_reaches_main_and_subsequently_constructed_mtp_pools(tmp_path, args):
    directory, _ = checkpoint(tmp_path / "weights", args)
    run = V.Stage(0, args.n_layers, args, head=True, tail=True, dspark=True,
        device="cpu", expert_placement="ram", expert_cache_reference=True,
        expert_cache_slots=3, expert_storage="exclusive", expert_storage_host_bytes=8 << 20)
    owner = object()  # No compute is licensed or executed by this metadata-only probe.
    manager = run._expert_cache
    manager.set_host_compute(owner)
    assert {role for role, _ in manager.pools} == {"main"}
    run.load(directory)
    assert {role for role, _ in manager.pools} == {"main", "draft"}
    assert all(pool._host_compute is owner for pool in manager.pools.values())


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_exclusive_swaps_preserve_real_speculative_rollback_and_correction(tmp_path, args, accepted):
    directory, _ = checkpoint(tmp_path / "weights", args)
    reference = V.Stage(0, args.n_layers, args, head=True, tail=True, device="cpu").load(directory)
    run = exclusive(directory, args)
    run._spec = True
    sequence = ids(args, 17)
    with torch.no_grad():
        assert_equal_stages(reference, run, sequence[:, :13], 0)
        run.forward(run.embed(sequence[:, 13:]), sequence[:, 13:], 13)
        for offset in range(accepted):
            token = sequence[:, 13 + offset:14 + offset]
            reference.forward(reference.embed(token), token, 13 + offset)
        run._seek(13 + accepted)
        correction = ids(args, 1, seed=5)
        for pos in range(13 + accepted, 17 + accepted):
            correction = assert_equal_stages(reference, run, correction, pos)
    assert all(cache.stats()["live_leases"] == 0 for cache in run._expert_cache.caches.values())


def test_exclusive_mtp_drafts_remain_equal_to_original_native_tail(tmp_path, args):
    import v4_dspark_draft as DS
    directory, _ = checkpoint(tmp_path / "weights", args)
    full = V.Stage(0, args.n_layers, args, head=True, tail=True, dspark=True, device="cpu").load(directory)
    run = exclusive(directory, args, dspark=True)
    original = DS.DSparkTail(full).load(str(directory))
    draft = run._hybrid_ring_drafter.tail
    prompt = ids(args, 13)
    with torch.no_grad():
        current = assert_equal_stages(full, run, prompt, 0)[:, 0]
        original.prefill(current, full.tail_main_hidden()); draft.prefill(current, run.tail_main_hidden())
        for pos in range(13, 16):
            next_token = assert_equal_stages(full, run, current.unsqueeze(1), pos)
            expected = original.advance_and_draft(next_token, full.tail_main_hidden(), start_pos=pos)
            actual = draft.advance_and_draft(next_token, run.tail_main_hidden(), start_pos=pos)
            assert all(torch.equal(a, b) for a, b in zip(expected, actual))
            current = next_token[:, 0]


@pytest.mark.parametrize("change", [{"expert_cache_slots": 0}, {"expert_storage_host_bytes": 0},
    {"expert_storage_host_bytes": 1}, {"expert_placement": "gpu"}])
def test_stage_refuses_unknown_or_unbudgeted_exclusive_admission_before_load(args, change):
    opts = dict(device="cpu", expert_placement="ram", expert_cache_reference=True,
        expert_cache_slots=3, expert_storage="exclusive", expert_storage_host_bytes=8 << 20)
    opts.update(change)
    with pytest.raises((ValueError, RuntimeError), match="exclusive|RAM placement"):
        V.Stage(0, args.n_layers, args, head=True, tail=True, **opts)


def test_exclusive_refuses_scalar_graph_capture_before_constructing_pool(args, monkeypatch):
    monkeypatch.setattr(V, "V4_CUDA_GRAPH", "whole")
    with pytest.raises(ValueError, match="graph-off"):
        V.Stage(0, args.n_layers, args, head=True, tail=True, device="cpu",
            expert_placement="ram", expert_cache_reference=True, expert_cache_slots=3,
            expert_storage="exclusive", expert_storage_host_bytes=8 << 20)


def test_preallocated_hot_slots_still_require_the_declared_free_runtime_reserve(monkeypatch):
    from types import SimpleNamespace
    stage = V.Stage.__new__(V.Stage)
    stage._expert_cache = SimpleNamespace(pools={0: SimpleNamespace(loaded=True)})
    stage._hybrid_cache_ready = False
    stage._dspark = False
    stage._expert_cache_reference = False
    stage._expert_storage = "exclusive"
    stage._expert_cache_reserve_bytes = 1024
    stage.device = "cuda:0"
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1023, 4096))
    with pytest.raises(RuntimeError, match="runtime VRAM reserve"):
        stage.ensure_expert_cache()

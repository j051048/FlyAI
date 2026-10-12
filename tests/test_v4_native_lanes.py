"""Original V4 mathematics interleaved through independent bounded state slots."""
import pytest
import torch
from test_v4_conversation_cache import stage, forward
import v4_ref_cpu as R
from v4_native_lanes import NativeLanes, CONTRACT
from v4_conversation_cache import _buffers, _same


@pytest.fixture
def model():
    args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
                     dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=64)
    return args, R.build_oracle(args, 7)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def step(runner, job, token, pos):
    runner.activate(job, pos)
    result, logits = forward(runner.stage, torch.tensor([[token]], device=runner.stage.device), pos)
    runner.finish_step(job)
    return result, logits


@pytest.mark.parametrize("count", [2, 4])
def test_original_v4_interleaved_lanes_match_independent_tokenwise_runs_through_window_and_compression(model, count):
    shared = stage(model)
    control = [stage(model) for _ in range(count)]
    runner = NativeLanes(shared, max_jobs=count, host_bytes=64 << 20)
    pointers = {key: value.data_ptr() for key, value in _buffers(shared, None).items()}
    for index in range(count):
        runner.open(str(index), identity=str(index) * 64, horizon=64)
    for position in range(21):
        for index in reversed(range(count)):
            token = 1 + index * 13 + position
            want, logits = forward(control[index], torch.tensor([[token]]), position)
            got, actual = step(runner, str(index), token, position)
            assert torch.equal(got, want) and torch.equal(actual, logits)
            assert _same(_buffers(control[index], None), _buffers(shared, None))
    assert pointers == {key: value.data_ptr() for key, value in _buffers(shared, None).items()}
    assert runner.config()["numerical_contract"] == CONTRACT
    assert runner.config()["reserved_state_bytes"] >= count * runner.tensor_bytes


def test_real_incremental_prefix_restores_after_other_lane_and_preserves_state_bytes(model):
    shared, control = stage(model), stage(model)
    runner = NativeLanes(shared, max_jobs=2, host_bytes=32 << 20)
    runner.open("first", identity="a" * 64, horizon=64)
    for pos in range(13):
        step(runner, "first", pos + 1, pos)
        forward(control, torch.tensor([[pos + 1]]), pos)
    saved = runner.close("first", retain=True, prefix={"token_count": 13, "digest": "d" * 64})
    runner.open("other", identity="b" * 64, horizon=64)
    for pos in range(9):
        step(runner, "other", pos + 50, pos)
    restored = runner.open("continued", identity="a" * 64, horizon=64, restore_entry=saved["entry_id"])
    assert restored["prefix"]["token_count"] == 13
    for pos in range(13, 24):
        got, actual = step(runner, "continued", pos + 1, pos)
        want, logits = forward(control, torch.tensor([[pos + 1]]), pos)
        assert torch.equal(got, want) and torch.equal(actual, logits)
    runner.close("continued")
    # Cancelling one lane does not change the other's consumed frontier.
    assert runner.active["other"]["pos"] == 9
    step(runner, "other", 59, 9)


def test_budget_owner_tenant_and_horizon_fail_closed(model):
    shared = stage(model)
    with pytest.raises(ValueError, match="reservation"):
        NativeLanes(shared, max_jobs=2, host_bytes=1)
    runner = NativeLanes(shared, max_jobs=2, host_bytes=32 << 20)
    runner.bind_epoch("owner:1")
    runner.open("a", identity="a" * 64, horizon=64)
    step(runner, "a", 1, 0)
    entry = runner.close("a", retain=True, prefix={"token_count": 1, "digest": "d" * 64})
    with pytest.raises(ValueError, match="identity"):
        runner.open("b", identity="b" * 64, horizon=64, restore_entry=entry["entry_id"])
    with pytest.raises(ValueError, match="horizon"):
        runner.open("b", identity="a" * 64, horizon=63, restore_entry=entry["entry_id"])
    runner.bind_epoch("owner:2")
    with pytest.raises(ValueError, match="expired"):
        runner.open("b", identity="a" * 64, horizon=64, restore_entry=entry["entry_id"])
    runner.open("b", identity="a" * 64, horizon=64)
    with pytest.raises(ValueError, match="frontier"):
        step(runner, "b", 1, 1)


def test_corrupted_cached_native_state_is_not_reused(model):
    runner = NativeLanes(stage(model), max_jobs=2, host_bytes=32 << 20)
    runner.open("a", identity="a" * 64, horizon=64)
    step(runner, "a", 1, 0)
    entry = runner.close("a", retain=True, prefix={"token_count": 1, "digest": "d" * 64})
    next(iter(runner.slots[0]["buffers"].values())).add_(1)
    with pytest.raises(ValueError, match="content changed"):
        runner.open("b", identity="a" * 64, horizon=64, restore_entry=entry["entry_id"])


@pytest.mark.parametrize("field,value", [("max_jobs", 10 ** 9), ("max_jobs", True),
    ("ttl_s", float("nan")), ("reserved_state_bytes", 1), ("per_job_state_bytes", -1),
    ("restore_workspace_bytes", float("inf"))])
def test_untrusted_signed_native_capacity_is_bounded_before_coordinator_warmup(model, field, value):
    from v4_native_lane_backend import _validate_capacity
    cfg = NativeLanes(stage(model), max_jobs=2, host_bytes=32 << 20).config()
    _validate_capacity(cfg, 64)
    cfg[field] = value
    with pytest.raises(ValueError):
        _validate_capacity(cfg, 64)


def test_durable_native_prefix_restores_all_math_state_only_under_same_runtime_and_epoch(model, tmp_path):
    def build(epoch="boot:1", runtime="a"):
        runner = NativeLanes(stage(model), max_jobs=2, host_bytes=32 << 20,
            state_directory=tmp_path / "private", disk_bytes=16 << 20)
        runner.bind_epoch(epoch)
        runner.bind_storage_identity({"source": runtime, "cohort": "test", "lease_fence": 1})
        return runner
    original, control = build(), stage(model)
    original.open("first", identity="a" * 64, horizon=64)
    for pos in range(13):
        step(original, "first", pos + 1, pos)
        forward(control, torch.tensor([[pos + 1]]), pos)
    entry = original.close("first", retain=True, prefix={"token_count": 13, "digest": "d" * 64})
    assert original.store.status()["written"] == 1
    # A fresh runner and original Stage buffers must use the actual persisted
    # tensor payload; no in-memory slot survives this reconstruction.
    fresh = build()
    restored = fresh.open("continued", identity="a" * 64, horizon=64, restore_entry=entry["entry_id"])
    assert restored["state_digest"] == entry["state_digest"]
    assert fresh.store.status()["loaded"] == 1
    for pos in range(13, 22):
        got, actual = step(fresh, "continued", pos + 1, pos)
        wanted, logits = forward(control, torch.tensor([[pos + 1]]), pos)
        assert torch.equal(got, wanted) and torch.equal(actual, logits)
    for changed in (build(epoch="boot:2"), build(runtime="new-source")):
        with pytest.raises(ValueError, match="expired"):
            changed.open("refused", identity="a" * 64, horizon=64, restore_entry=entry["entry_id"])
    assert fresh.config()["restore_workspace_bytes"] >= fresh.tensor_bytes + (2 << 20)
    from v4_resources import measure_stage_resources
    fresh.stage._native_lanes = fresh
    measured = measure_stage_resources(fresh.stage, checkpoint_id="test-native")
    assert measured["native_lane_reservation"]["host_reserved_bytes"] == fresh.reserved_bytes
    assert measured["native_lane_reservation"]["remaining_host_reserved_bytes"] == fresh.metadata_bytes * 2 + fresh.restore_workspace_bytes
    assert measured["module_storage"]["by_kind"]["host_native_lane_state_bytes"] == fresh.tensor_bytes * 2


@pytest.mark.parametrize("kind", ["integrity", "permission", "unknown"])
def test_durable_native_optional_cache_does_not_hide_integrity_or_unknown_io_failure(model, tmp_path, monkeypatch, kind):
    import errno
    from v4_durable_state import DurableStateError
    runner = NativeLanes(stage(model), max_jobs=2, host_bytes=32 << 20,
        state_directory=tmp_path / "private", disk_bytes=16 << 20)
    runner.bind_epoch("boot:1")
    runner.bind_storage_identity({"source": "test"})
    runner.open("first", identity="a" * 64, horizon=64)
    step(runner, "first", 1, 0)
    error = {"integrity": DurableStateError("checkpoint integrity failed"),
             "permission": OSError(errno.EACCES, "permission denied"),
             "unknown": OSError(errno.EIO, "unknown storage failure")}[kind]
    def fail(*a, **kw): raise error
    monkeypatch.setattr(runner.store, "put", fail)
    with pytest.raises(type(error)):
        runner.close("first", retain=True, prefix={"token_count": 1, "digest": "d" * 64})
    assert runner.last_cache_write_error is None
    assert runner.active["first"]["cache"] is None
    runner.close("first")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA lane gate requires an idle accelerator")
def test_actual_cuda_native_lane_interleaving_preserves_original_v4_math(model):
    import v4_stage as V
    import v4_kernels_cpu as K
    if K.backend() != "cpu":
        pytest.skip("this CUDA state-isolation gate uses Torch reference math; native FP4 gate is separate")
    args, oracle = model
    V.ref().precompute_freqs_cis.cache_clear()
    def build():
        st = V.Stage(0, args.n_layers, args, head=True, tail=True, device="cuda")
        for target, source in zip(st.layers, oracle.layers): target.load_state_dict(source.state_dict())
        st.embed_tokens.load_state_dict(oracle.embed.state_dict())
        st.norm.load_state_dict(oracle.norm.state_dict()); st.lm_head.load_state_dict(oracle.head.state_dict())
        with torch.no_grad():
            for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                getattr(st, name).copy_(getattr(oracle, name))
        return st
    try:
        control = [build(), build()]
        runner = NativeLanes(build(), max_jobs=2, host_bytes=32 << 20)
        for index in range(2): runner.open(str(index), identity=str(index) * 64, horizon=64)
        for pos in range(17):
            for index in range(2):
                token = pos + 10 * index + 1
                want, logits = forward(control[index], torch.tensor([[token]], device="cuda"), pos)
                got, actual = step(runner, str(index), token, pos)
                assert torch.equal(want, got) and torch.equal(logits, actual)
    finally:
        V.ref().precompute_freqs_cis.cache_clear()

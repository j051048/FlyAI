"""Actual native top-K, compressed history, replay and resets; CPU is not CUDA evidence."""
import pytest
import torch
import v4_ref_cpu as R
import v4_stage as V
from test_v4_kv_runtime import stage, forward, assert_state_equal, args


def test_explicit_indexed_constructor_changes_calibrated_identity_and_reserves_scratch():
    import v4_resources as resources
    a = args(); oracle = R.build_oracle(a, 23)
    original = stage(oracle, a, layer=True, kv_indexed_gather=False)
    indexed = stage(oracle, a, layer=True, kv_indexed_gather=True)
    assert resources.runtime_config_payload(indexed)["kv_indexed_gather"] is True
    assert resources.runtime_config_payload(original)["kv_indexed_gather"] is False
    assert resources.runtime_config_identity(indexed) != resources.runtime_config_identity(original)
    report = resources.measure_stage_resources(indexed, checkpoint_id="cpu-fixture")
    reserve = report["kv_scratch_reservation"]
    assert reserve["remaining_host_reserved_bytes"] == indexed._kv_runtime.gather_scratch_bytes > 0
    assert reserve["remaining_device_reserved_bytes"] == reserve["pinned_reserved_bytes"] == 0
    assert resources.measure_stage_resources(original, checkpoint_id="cpu-fixture")["kv_scratch_reservation"]["remaining_host_reserved_bytes"] == 0


@pytest.mark.parametrize("resource", ["host", "device", "pinned", "missing"])
def test_central_placement_refuses_unreserved_gather_scratch(tmp_path, monkeypatch, resource):
    from types import SimpleNamespace
    import v4_resources as resources
    from test_resource_contract import write_checkpoint
    from shard.resources import (PlacementRequirements, GpuRequirements, HostRequirements,
                                 CalibrationProvenance, ResourceError)
    inventory = resources.inspect_checkpoint(write_checkpoint(tmp_path))
    actual = SimpleNamespace(args=SimpleNamespace(n_layers=43), lo=40, hi=43, head=False,
        tail=True, _dspark=True, dtype="bf16", device="cuda:0", _resource_checkpoint_dir=str(tmp_path),
        _kv_indexed_gather=True)
    requirements = PlacementRequirements(resources.MODEL_ID, 40, 43,
        GpuRequirements(resident_weights_bytes=100), HostRequirements(prefill_bytes=100, pinned_bytes=80),
        CalibrationProvenance(inventory["checkpoint_id"], resources.runtime_config_identity(actual),
            "2026-10-29T00:00:00Z", "test-node", "CPU admission fixture", "fixture"))
    report = {"module_storage": {"gpu_bytes": 50, "host_bytes": 100, "host_pinned_bytes": 80},
        "allocator": {"allocated_bytes": 80, "reserved_bytes": 100}, "draft_measurement_status": "measured",
        "kv_scratch_reservation": {"remaining_host_reserved_bytes": 0,
            "remaining_device_reserved_bytes": 0, "pinned_reserved_bytes": 0}}
    if resource == "missing":
        report.pop("kv_scratch_reservation")
    else:
        key = {"host": "remaining_host_reserved_bytes", "device": "remaining_device_reserved_bytes",
               "pinned": "pinned_reserved_bytes"}[resource]
        report["kv_scratch_reservation"][key] = 1
    monkeypatch.setattr(resources, "measure_stage_resources", lambda *a, **kw: report)
    with pytest.raises(ResourceError):
        resources.placement_requirements_for_stage(actual, requirements.to_dict())


def test_stage_reset_records_completion_after_all_compressor_writes(monkeypatch):
    a = args(); actual = stage(R.build_oracle(a, 24), a, layer=True, kv_indexed_gather=True)
    for compressor, _ in actual._compressors():
        compressor.kv_state.fill_(2); compressor.score_state.fill_(0)
    finish = actual._kv_runtime.finish_stage_reset
    calls = []
    def checked_finish():
        for compressor, _ in actual._compressors():
            assert not compressor.kv_state.count_nonzero()
            assert torch.isneginf(compressor.score_state).all()
        assert all(not entry["index_device"].count_nonzero()
                   for entry in actual._kv_runtime.entries.values() if "index_device" in entry)
        calls.append("all_writes_completed_on_owner_stream")
        finish()
    monkeypatch.setattr(actual._kv_runtime, "finish_stage_reset", checked_finish)
    actual.reset()
    assert calls == ["all_writes_completed_on_owner_stream"]


@pytest.mark.parametrize("prefix", [7, 16, 36])
def test_indexed_gather_matches_original_full_history_and_replay(prefix):
    a = args()
    oracle = R.build_oracle(a, 17)
    expected = stage(oracle, a, layer=True)
    actual = stage(oracle, a, layer=True, kv_indexed_gather=True)
    prompt = torch.arange(prefix).reshape(1, -1) % a.vocab_size
    assert torch.equal(forward(expected, prompt, 0), forward(actual, prompt, 0))
    for pos in range(prefix, prefix + 10):
        token = torch.tensor([[pos % a.vocab_size]])
        assert torch.equal(forward(expected, token, pos), forward(actual, token, pos))
        assert_state_equal(expected, actual)
    assert actual._kv_runtime.gather_calls > 0
    expected.reset(); actual.reset()
    assert torch.equal(forward(expected, prompt, 0), forward(actual, prompt, 0))
    assert_state_equal(expected, actual)


def test_indexed_gather_restores_index_residency_after_seek():
    a = args()
    oracle = R.build_oracle(a, 18)
    expected = stage(oracle, a, layer=True, spec_depth=8)
    actual = stage(oracle, a, layer=True, spec_depth=8, kv_indexed_gather=True)
    expected._spec = actual._spec = True
    prompt = torch.arange(13).reshape(1, -1)
    forward(expected, prompt, 0); forward(actual, prompt, 0)
    for pos in list(range(13, 19)) + list(range(15, 20)):
        ids = torch.tensor([[pos + 2]])
        assert torch.equal(forward(expected, ids, pos), forward(actual, ids, pos))
        assert_state_equal(expected, actual)


def test_indexed_gather_requires_its_explicit_reservations():
    a = args()
    oracle = R.build_oracle(a, 19)
    with pytest.raises(ValueError, match="indexed gather"):
        stage(oracle, a, kv_indexed_gather=True)
    with pytest.raises(ValueError, match="quota"):
        stage(oracle, a, layer=True, kv_indexed_gather=True, kv_gpu_budget_bytes=100)


@pytest.mark.gpu
def test_actual_cuda_indexed_storage_parity_transfer_and_reset(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("owned CUDA device required")
    a = args()
    with torch.device("cpu"):
        oracle = R.build_oracle(a, 20)
    previous = torch.get_default_device()
    helpers = ("precompute_freqs_cis", "get_window_topk_idxs", "get_compress_topk_idxs")
    for name in helpers:
        getattr(V.ref(), name).cache_clear()
    try:
        torch.set_default_device("cuda")
        monkeypatch.setattr(V, "_graph_mode", lambda: "off")
        runs = []
        for selected in (False, True):
            s = V.Stage(0, a.n_layers, a, head=True, tail=True, device="cuda",
                kv_placement="layer", kv_gpu_budget_bytes=4 << 20,
                kv_host_budget_bytes=16 << 20, kv_indexed_gather=selected)
            for target, source in zip(s.layers, oracle.layers):
                target.load_state_dict(source.state_dict())
            s.embed_tokens.load_state_dict(oracle.embed.state_dict())
            s.norm.load_state_dict(oracle.norm.state_dict())
            s.lm_head.load_state_dict(oracle.head.state_dict())
            with torch.no_grad():
                for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                    getattr(s, name).copy_(getattr(oracle, name))
            runs.append(s)
        for _ in range(2):
            for s in runs:
                s.reset()
            prompt = torch.arange(36, device="cuda").reshape(1, -1) % a.vocab_size
            assert torch.equal(forward(runs[0], prompt, 0), forward(runs[1], prompt, 0))
            for pos in range(36, 48):
                ids = torch.tensor([[pos]], device="cuda")
                assert torch.equal(forward(runs[0], ids, pos), forward(runs[1], ids, pos))
                assert_state_equal(*runs)
        assert runs[1]._kv_runtime.gather_calls > 0
        assert runs[1]._kv_runtime.gather_index_d2h_bytes > 0
    finally:
        torch.set_default_device(previous)
        for name in helpers:
            getattr(V.ref(), name).cache_clear()


@pytest.mark.gpu
def test_actual_cuda_indexed_reset_stream_a_then_forward_stream_b(monkeypatch):
    """Real asynchronous reset ordering; this test is pending owned-GPU execution."""
    if not torch.cuda.is_available():
        pytest.skip("owned CUDA device required")
    a = args()
    with torch.device("cpu"):
        oracle = R.build_oracle(a, 25)
    previous = torch.get_default_device()
    helpers = ("precompute_freqs_cis", "get_window_topk_idxs", "get_compress_topk_idxs")
    for name in helpers:
        getattr(V.ref(), name).cache_clear()
    try:
        torch.set_default_device("cuda")
        monkeypatch.setattr(V, "_graph_mode", lambda: "off")
        actual = V.Stage(0, a.n_layers, a, head=True, tail=True, device="cuda", runtime_profile=False,
            runtime_metrics=False, kv_placement="layer", kv_gpu_budget_bytes=4 << 20,
            kv_host_budget_bytes=16 << 20, kv_indexed_gather=True)
        for target, source in zip(actual.layers, oracle.layers):
            target.load_state_dict(source.state_dict())
        actual.embed_tokens.load_state_dict(oracle.embed.state_dict())
        actual.norm.load_state_dict(oracle.norm.state_dict())
        actual.lm_head.load_state_dict(oracle.head.state_dict())
        with torch.no_grad():
            for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                getattr(actual, name).copy_(getattr(oracle, name))
            prompt = torch.arange(17, device="cuda").reshape(1, -1) % a.vocab_size
            actual.reset()
            expected = forward(actual, prompt, 0).clone()
            forward(actual, torch.tensor([[19]], device="cuda"), 17)
            torch.cuda.synchronize()
            reset_stream, compute_stream = torch.cuda.Stream(), torch.cuda.Stream()
            with torch.cuda.stream(reset_stream):
                torch.cuda._sleep(100_000_000)
                actual.reset()
            assert actual._kv_runtime._stream_id == reset_stream.cuda_stream
            with torch.cuda.stream(compute_stream):
                observed = forward(actual, prompt, 0).clone()
            torch.cuda.synchronize()
            assert torch.equal(expected, observed)
            assert actual._kv_runtime._stream_id == compute_stream.cuda_stream
    finally:
        torch.cuda.synchronize()
        torch.set_default_device(previous)
        for name in helpers:
            getattr(V.ref(), name).cache_clear()

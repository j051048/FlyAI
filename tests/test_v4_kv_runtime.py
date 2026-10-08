"""Real V4 Attention/Indexer/Compressor gates, not a conventional K/V mock."""
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")
R = pytest.importorskip("v4_ref_cpu")
V = pytest.importorskip("v4_stage")
import v4_resources as resources
import v4_chunked_prefill as chunks


def args(**kw):
    return R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
        dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=96,
        **kw)


def stage(oracle, a, *, layer=False, chunk=0, **kw):
    s = V.Stage(0, a.n_layers, a, head=True, tail=True, device="cpu",
        kv_placement="layer" if layer else "gpu", kv_reference=layer,
        kv_gpu_budget_bytes=kw.pop("kv_gpu_budget_bytes", 2_000_000 if layer else 0),
        kv_host_budget_bytes=kw.pop("kv_host_budget_bytes", 10_000_000 if layer else 0),
        prefill_query_chunk=chunk, **kw)
    for target, source in zip(s.layers, oracle.layers):
        target.load_state_dict(source.state_dict(), strict=True)
    s.embed_tokens.load_state_dict(oracle.embed.state_dict())
    s.norm.load_state_dict(oracle.norm.state_dict())
    s.lm_head.load_state_dict(oracle.head.state_dict())
    with torch.no_grad():
        for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            getattr(s, name).copy_(getattr(oracle, name))
    return s


def logical_state(s):
    runtime = s._kv_runtime
    if runtime is not None:
        runtime.synchronize_host()
    result = {}
    for i, layer in enumerate(s.layers):
        attn = layer.attn
        entry = runtime.entries.get(id(attn)) if runtime is not None else None
        result[f"{i}.attention"] = (torch.cat([entry["ring"].cpu(), entry["history"]], dim=1)
                                   if entry is not None else attn.kv_cache.cpu().clone())
        indexer = getattr(attn, "indexer", None)
        if indexer is not None:
            result[f"{i}.indexer"] = (entry["index_history"].clone() if entry is not None
                                       else indexer.kv_cache.cpu().clone())
        for name, owner in (("attention", attn), ("indexer", indexer)):
            compressor = getattr(owner, "compressor", None)
            if compressor is not None:
                for field in ("kv_state", "score_state"):
                    result[f"{i}.{name}.{field}"] = getattr(compressor, field).cpu().clone()
    return result


def assert_state_equal(a, b):
    expected, actual = logical_state(a), logical_state(b)
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name


def forward(s, ids, pos):
    return s.forward(s.embed(ids), ids, pos)


@pytest.mark.parametrize("length,chunk,batch", [(1, 3, 1), (3, 2, 1), (4, 3, 1),
    (7, 3, 1), (8, 3, 1), (9, 4, 1), (13, 3, 1), (17, 4, 2)])
@pytest.mark.parametrize("layer", [False, True])
def test_query_prefill_preserves_reference_full_projection_and_all_state(length, chunk, batch, layer):
    a = args(max_batch_size=2)
    oracle = R.build_oracle(a, 7)
    resident = stage(oracle, a)
    actual = stage(oracle, a, layer=layer, chunk=chunk)
    ids = torch.arange(batch * length).reshape(batch, length).remainder(a.vocab_size)
    expected, observed = forward(resident, ids, 0), forward(actual, ids, 0)
    assert torch.equal(expected, observed)
    assert torch.equal(resident.logits_all(expected), actual.logits_all(observed))
    assert_state_equal(resident, actual)
    for pos in range(length, length + 10):
        ids = torch.full((batch, 1), pos % a.vocab_size)
        assert torch.equal(forward(resident, ids, pos), forward(actual, ids, pos))
    assert_state_equal(resident, actual)
    if length > chunk:
        assert actual.prefill_runtime_status()["gate_checks"] == a.n_layers
        assert actual.prefill_runtime_status()["executed_query_chunks"] == a.n_layers * ((length + chunk - 1) // chunk)
    assert actual._kv_runtime is None or actual._kv_runtime.h2d_bytes == actual._kv_runtime.d2h_bytes == 0


def test_query_gate_reuses_verified_shape_and_invalidates_parameter_reload():
    a = args()
    s = stage(R.build_oracle(a, 7), a, chunk=3)
    ids = torch.arange(13).reshape(1, -1)
    first = forward(s, ids, 0)
    s.reset()
    assert torch.equal(first, forward(s, ids, 0))
    assert s.prefill_runtime_status()["gate_checks"] == 0
    s.reset()
    with torch.no_grad():
        s.layers[0].attn.wq_a.weight.add_(0)  # Version changes even when bytes do not.
    assert torch.equal(first, forward(s, ids, 0))
    assert s.prefill_runtime_status()["gate_checks"] == 1
    s.reset()
    s.layers[1].attn.load_state_dict(s.layers[1].attn.state_dict())
    forward(s, ids, 0)
    assert s.prefill_runtime_status()["gate_checks"] == 1


def test_gate_rejects_a_wrong_chunk_result_instead_of_serving_approximate_logits(monkeypatch):
    a = args()
    s = stage(R.build_oracle(a, 7), a, chunk=3)
    original = chunks.query_chunk_attention
    monkeypatch.setattr(chunks, "query_chunk_attention", lambda *a, **kw: original(*a, **kw) + 1)
    ids = torch.arange(13).reshape(1, -1)
    with pytest.raises(RuntimeError, match="output/state bit parity"):
        forward(s, ids, 0)
    assert s._pos == 0
    assert not getattr(s.layers[0].attn, "_prefill_gate_verified", set())


def test_tiered_rollback_replay_matches_resident_across_both_compression_boundaries():
    a = args()
    oracle = R.build_oracle(a, 7)
    resident, tiered = stage(oracle, a, spec_depth=8), stage(oracle, a, layer=True, chunk=3, spec_depth=8)
    for s in (resident, tiered):
        s._spec = True
    ids = torch.arange(7).reshape(1, -1)
    assert torch.equal(forward(resident, ids, 0), forward(tiered, ids, 0))
    # One covering checkpoint, then two more futures. Rewind into its prefix.
    ids = torch.tensor([[8, 9, 10, 11, 12]])
    assert torch.equal(forward(resident, ids, 7), forward(tiered, ids, 7))
    for pos in (12, 13):
        ids = torch.tensor([[pos]])
        assert torch.equal(forward(resident, ids, pos), forward(tiered, ids, pos))
    for s in (resident, tiered):
        s._seek(9)
        assert s._pos == 9
        s.commit(9)
    assert_state_equal(resident, tiered)
    for pos in range(9, 21):
        ids = torch.tensor([[pos + 1]])
        assert torch.equal(forward(resident, ids, pos), forward(tiered, ids, pos))
    assert_state_equal(resident, tiered)
    assert all(value.device.type == "cpu" for checkpoint in tiered._spec_ckpts
               for state in checkpoint["state"] for value in state.values())


def test_tiered_stale_compressed_slots_are_rewritten_before_first_read():
    a = args()
    oracle = R.build_oracle(a, 7)
    resident, tiered = stage(oracle, a), stage(oracle, a, layer=True)
    for s in (resident, tiered):
        s._spec = True
    ids = torch.arange(7).reshape(1, -1)
    forward(resident, ids, 0); forward(tiered, ids, 0)
    for pos in range(7, 17):
        ids = torch.tensor([[pos]])
        forward(resident, ids, pos); forward(tiered, ids, pos)
    for s in (resident, tiered):
        s._seek(7)
    for entry in tiered._kv_runtime.entries.values():
        first = 7 // entry["attention"].compress_ratio
        entry["history"][:, first:].fill_(float("nan"))
        if entry["index_history"] is not None:
            entry["index_history"][:, first:].fill_(float("nan"))
    for pos in range(7, 18):
        ids = torch.tensor([[pos + 1]])
        assert torch.equal(forward(resident, ids, pos), forward(tiered, ids, pos))


def test_workspace_capacity_refuses_a_frame_before_seek_or_any_state_mutation():
    a = args()
    oracle = R.build_oracle(a, 7)
    probe = stage(oracle, a, layer=True)
    fixed = probe._kv_runtime.fixed_bytes
    # Enough for ring + three ratio-4 compressed rows in both caches, not four.
    workspace = a.max_batch_size * ((a.window_size + 3) * a.head_dim + 3 * a.index_head_dim) * 2
    s = stage(oracle, a, layer=True, kv_gpu_budget_bytes=fixed + workspace)
    ids = torch.arange(13).reshape(1, -1)
    forward(s, ids, 0)
    s._spec = True
    before = logical_state(s)
    with pytest.raises(ValueError, match="active prefix"):
        forward(s, torch.tensor([[2, 3, 4]]), 13)
    assert s._pos == 13 and not s._spec_ckpts
    after = logical_state(s)
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert s.kv_runtime_status()["max_supported_tokens"] == 15


def test_tiered_resource_inventory_counts_host_history_and_workspace_once():
    a = args()
    s = stage(R.build_oracle(a, 7), a, layer=True, chunk=3, runtime_metrics=True)
    ids = torch.arange(13).reshape(1, -1)
    forward(s, ids, 0)
    report = resources.measure_stage_resources(s, checkpoint_id="fixture")
    names = [entry["name"] for entry in report["module_storage"]["storages"]]
    assert "kv_runtime.workspace" in names
    assert len([name for name in names if name.endswith(".history")]) == 2
    assert not any(name.endswith("indexer.kv_cache") for name in names)  # empty workspace alias
    payload = resources.runtime_config_payload(s)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == resources.runtime_config_identity(s) == report["runtime_config_sha256"]
    assert payload == report["runtime_config_payload"]
    assert payload["kv_placement"] == "layer" and payload["prefill_query_chunk_tokens"] == 3
    assert report["kv_policy"]["host_history_bytes"] > 0
    assert s.runtime_metrics()["kv"]["gpu_bytes"] == 0
    from shard.runtime_metrics import validate_runtime_metrics
    signed_metrics = validate_runtime_metrics(s.runtime_metrics())
    assert signed_metrics["kv_policy"]["mode"] == "reference_cpu"
    assert signed_metrics["prefill_policy"]["mode"] == "query_chunks"
    s.reset()
    assert all(not tensor.count_nonzero() for entry in s._kv_runtime.entries.values()
               for tensor in (entry["history"], entry["index_history"]) if tensor is not None)


@pytest.mark.parametrize("kwargs,message", [
    ({"kv_placement": "layer"}, "positive"),
    ({"kv_placement": "invalid"}, "kv_placement"),
    ({"kv_reference": "false"}, "kv_reference"),
    ({"prefill_query_chunk": True}, "prefill_query_chunk"),
    ({"kv_gpu_budget_bytes": -1}, "kv_gpu_budget_bytes"),
])
def test_invalid_configuration_precedes_reference_allocation(monkeypatch, kwargs, message):
    monkeypatch.setattr(V, "ref", lambda: pytest.fail("allocated before configuration validation"))
    with pytest.raises(ValueError, match=message):
        V.Stage(0, 1, device="cpu", **kwargs)


def test_whole_graph_and_fast_verify_are_explicitly_incompatible(monkeypatch):
    opts = dict(kv_placement="layer", kv_gpu_budget_bytes=2_000_000, kv_host_budget_bytes=10_000_000,
                kv_reference=True, device="cpu")
    monkeypatch.setattr(V, "V4_CUDA_GRAPH", "whole")
    with pytest.raises(ValueError, match="not whole"):
        V.Stage(0, 1, **opts)
    monkeypatch.setattr(V, "V4_CUDA_GRAPH", "off")
    with pytest.raises(ValueError, match="reference verify"):
        V.Stage(0, 1, fast_verify=True, **opts)


def test_insufficient_host_quota_accounts_for_rollback_and_gate():
    a = args()
    oracle = R.build_oracle(a, 7)
    with pytest.raises(ValueError, match="host quota"):
        stage(oracle, a, layer=True, chunk=3, kv_host_budget_bytes=1)


def test_layer_kv_query_chunks_compose_with_real_host_expert_cache_and_taps(tmp_path):
    from test_v4_hybrid import checkpoint
    a = args(n_routed_experts=6, n_activated_experts=2)
    directory, oracle = checkpoint(tmp_path / "weights", a)
    resident = stage(oracle, a, dspark=True)
    hybrid = stage(oracle, a, layer=True, chunk=3, dspark=True,
        expert_placement="ram", expert_cache_reference=True, expert_cache_slots=2,
        expert_prefetch=True, runtime_metrics=True)
    hybrid.load(str(directory))  # Load/register actual MTP pools before shared cache allocation.
    for s in (resident, hybrid):
        s._spec = True
    for pos, width in ((0, 17), (17, 3), (18, 1), (19, 1)):
        ids = torch.arange(pos, pos + width).reshape(1, -1).remainder(a.vocab_size)
        assert torch.equal(forward(resident, ids, pos), forward(hybrid, ids, pos))
        assert torch.equal(resident.tail_main_hidden(), hybrid.tail_main_hidden())
    assert_state_equal(resident, hybrid)
    assert hybrid.kv_runtime_status()["draft_reserve_bytes"] == a.n_mtp_layers * a.max_batch_size * a.window_size * a.head_dim * 2
    metrics = hybrid.runtime_metrics()
    assert metrics["totals"]["dma_misses"] == 0
    assert metrics["totals"]["reference_routes"] > 0
    assert not any(slot.references for cache in hybrid._expert_cache.caches.values() for slot in cache._slots)


def test_real_kv_and_prefill_observations_are_signed_and_tamper_detected():
    from copy import deepcopy
    from shard.receipt import ReceiptError, ReceiptSigner, gen_key, verify_receipt
    a = args()
    s = stage(R.build_oracle(a, 7), a, layer=True, chunk=3, runtime_metrics=True)
    forward(s, torch.arange(13).reshape(1, -1), 0)
    receipt = ReceiptSigner(gen_key(), "s", "j", 0, a.n_layers).finalize(runtime_metrics=s.runtime_metrics())
    verify_receipt(receipt)
    tampered = deepcopy(receipt)
    tampered["runtime_metrics"]["prefill_policy"]["executed_query_chunks"] += 1
    with pytest.raises(ReceiptError, match="signature"):
        verify_receipt(tampered)


def test_tiered_constructor_never_builds_full_device_history_or_mutates_model_args():
    a = args()
    s = stage(R.build_oracle(a, 7), a, layer=True)
    assert a.max_seq_len == 96
    for layer in s.layers:
        assert layer.attn.kv_cache.shape[1] == a.window_size
        indexer = getattr(layer.attn, "indexer", None)
        assert indexer is None or indexer.kv_cache.numel() == 0
    pointers = [(name, tensor.data_ptr()) for name, tensor in s._kv_runtime.tensors() if tensor is not None]
    forward(s, torch.arange(17).reshape(1, -1), 0)
    s.reset()
    assert pointers == [(name, tensor.data_ptr()) for name, tensor in s._kv_runtime.tensors() if tensor is not None]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA V4 bit/state gate needs CUDA")
@pytest.mark.parametrize("batch", [1, 2])
def test_cuda_tiered_and_query_chunk_prefill_bit_gate(batch):
    # Run with the real Tilelang backend, not the CPU kernel shim on a CUDA tensor.
    if R.v4_kernels_cpu.backend() != "tilelang":
        pytest.skip("requires V4_KERNELS=tilelang before reference import")
    a = args(max_batch_size=2)
    def gpu_stage(**kw):
        return V.Stage(0, a.n_layers, a, head=True, tail=True, device="cuda:0", **kw)
    resident = gpu_stage(runtime_metrics=True)
    R.init_random(resident.layers, 7)
    tiered = gpu_stage(kv_placement="layer", kv_gpu_budget_bytes=2_000_000,
                       kv_host_budget_bytes=10_000_000, prefill_query_chunk=3, runtime_metrics=True)
    tiered.layers.load_state_dict(resident.layers.state_dict())
    # Compare real layer output/state; boundary parameters are irrelevant here.
    torch.cuda.reset_peak_memory_stats()
    resident._spec = tiered._spec = True
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for i, (pos, width) in enumerate(((0, 13), (13, 1), (14, 1), (15, 1), (16, 1))):
        with torch.cuda.stream(streams[i % 2]):
            ids = torch.arange(pos, pos + width, device="cuda:0").reshape(1, -1).expand(batch, -1)
            h = torch.randn(batch, width, a.hc_mult, a.dim, device="cuda:0", dtype=torch.bfloat16)
            assert torch.equal(resident.forward(h, ids, pos), tiered.forward(h, ids, pos))
    assert_state_equal(resident, tiered)
    status = tiered.kv_runtime_status()
    assert status["h2d_bytes"] > 0 and status["d2h_bytes"] > 0
    assert all(entry["history"].is_pinned() for entry in tiered._kv_runtime.entries.values())
    assert all(value.is_pinned() for checkpoint in tiered._spec_ckpts
               for state in checkpoint["state"] for value in state.values())
    assert tiered.runtime_metrics()["kv"]["gpu_bytes"] <= status["gpu_budget_bytes"]
    with torch.cuda.stream(streams[1]):
        resident._seek(14); tiered._seek(14)
        ids = torch.full((batch, 1), 42, device="cuda:0")
        h = torch.randn(batch, 1, a.hc_mult, a.dim, device="cuda:0", dtype=torch.bfloat16)
        assert torch.equal(resident.forward(h, ids, 14), tiered.forward(h, ids, 14))
    # Compare reachable state after rewind; future compressed slots may legally
    # retain the old speculation until their write-before-first-read boundary.
    assert_state_equal(resident, tiered)

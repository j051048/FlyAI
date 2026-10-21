"""Native request batches: exact-width graph math, ownership and real CUDA gates."""
from types import SimpleNamespace

import pytest
import torch

import v4_ref_cpu as R
import v4_stage as V
from test_v4_conversation_cache import stage
from v4_batch_invariant import request_row_math
from v4_conversation_cache import _buffers
from v4_native_batch import DeviceNativeLanes
from v4_native_batch_graph import NativeBatchGraphPool, graph_key


def _bytes_equal(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


@pytest.fixture(autouse=True)
def one_thread():
    count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(count)
    V.ref().precompute_freqs_cis.cache_clear()


@pytest.fixture
def cuda_device_defaults():
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    try:
        yield
    finally:
        torch.set_default_device(previous)
        for name in ("precompute_freqs_cis", "get_window_topk_idxs", "get_compress_topk_idxs"):
            getattr(V.ref(), name).cache_clear()


def _model(seed=7, *, plain_ratio=8):
    args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, plain_ratio, 0, 0, 0),
        dspark_target_layer_ids=(1, 2, 3), window_size=8,
        max_seq_len=256 if plain_ratio == 128 else 64,
        index_topk=2)
    return args, R.build_oracle(args, seed)


def _runner(model, device="cpu", *, max_graphs=512):
    if device == "cpu":
        st = stage(model)
    else:
        args, oracle = model
        st = V.Stage(0, args.n_layers, args, head=True, tail=True, device=device)
        for target, source in zip(st.layers, oracle.layers):
            target.load_state_dict(source.state_dict())
        st.embed_tokens.load_state_dict(oracle.embed.state_dict())
        st.norm.load_state_dict(oracle.norm.state_dict())
        st.lm_head.load_state_dict(oracle.head.state_dict())
        with torch.no_grad():
            for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                getattr(st, name).copy_(getattr(oracle, name))
    run = DeviceNativeLanes(st, max_jobs=4, host_bytes=64 << 20,
        device_bytes=64 << 20, reference=device == "cpu")
    pool = NativeBatchGraphPool(st, run.workspace, max_jobs=4,
        max_graphs=max_graphs, graph_bytes=(64 if device == "cpu" else 512) << 20,
        reference=device == "cpu")
    return run, pool


def _advance(run, pool, jobs, tokens, pos, *, eager=False):
    ids = torch.tensor(tokens, dtype=torch.long, device=run.stage.device).reshape(-1, 1)
    rows = [run.active[job] for job in jobs]
    with torch.no_grad(), run._bind_batch(rows, pos), request_row_math():
        out = (pool.eager_twin if eager else pool.forward)(run.stage.embed(ids), ids, pos)
        logits = run.stage.logits_all(out, full_logits=False)
        for index, slot in enumerate(rows):
            for name, value in run.workspace.items():
                slot["buffers"][name].copy_(value[index:index + 1])
            slot["pos"] = pos + 1
    return out, logits


@pytest.mark.parametrize("seed", [7, 11, 23])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_exact_width_batch_graph_eager_twin_preserves_scalar_topk_and_all_native_state(seed, batch_size):
    model = _model(seed)
    run, pool = _runner(model)
    controls = [stage(model) for _ in range(batch_size)]
    for index in range(batch_size):
        run.open(str(index), identity=str(index) * 64, horizon=64)
    workspace_addresses = {name: value.data_ptr() for name, value in run.workspace.items()}
    for pos in range(37):
        order = list(range(batch_size))[::-1] if pos % 2 else list(range(batch_size))
        tokens = [(1 + index * 13 + pos * 3) % run.stage.args.vocab_size for index in order]
        out, logits = _advance(run, pool, [str(index) for index in order], tokens, pos)
        for row, index in enumerate(order):
            ids = torch.tensor([[tokens[row]]], dtype=torch.long)
            with torch.no_grad():
                want = controls[index].forward(controls[index].embed(ids), ids, pos)
                want_logits = controls[index].logits_all(want, full_logits=False)
            assert _bytes_equal(out[row:row + 1], want), (seed, batch_size, pos, index, "hidden")
            assert _bytes_equal(logits[row:row + 1], want_logits), (seed, batch_size, pos, index, "logits")
            for name, value in _buffers(controls[index], None).items():
                assert _bytes_equal(run.active[str(index)]["buffers"][name], value[0:1]), (pos, index, name)
    status = pool.status()
    assert status["captured_graphs"] == status["live_graphs"] == status["replays"] == 0
    assert status["reference_eager"] == 36 * len(run.stage.layers)
    assert status["qualification"] == "cpu_reference_only"
    assert status["selection"] == "native_exact_width_topk"
    assert workspace_addresses == {name: value.data_ptr() for name, value in run.workspace.items()}


def test_context_key_keeps_batch_branch_bounds_with_live_native_width_and_position():
    args, oracle = _model()
    layer = next(layer for layer in oracle.layers if layer.attn.compress_ratio == 8)
    make = lambda count, pos: graph_key(layer, requests=count, position=pos, max_jobs=4, horizon=64)
    assert make(4, 9) == make(4, 10)  # device-side position differs within the same exact read width.
    assert make(4, 9) == make(4, 17)  # original narrow scoring is eager, native widths may differ.
    assert make(4, 14) != make(4, 15)  # compress branch and read width change.
    assert make(4, 9) != make(3, 9)
    assert make(4, 15).read_width == 2 and make(4, 15).compress
    for count, pos in ((0, 1), (5, 1), (4, 0), (4, 64), (True, 1)):
        with pytest.raises(ValueError):
            make(count, pos)


def test_coarse_bucket_math_preserves_original_b4_state_across_native_ratio128_and_read_rungs():
    model = _model(7, plain_ratio=128)
    run, pool = _runner(model)
    controls = [stage(model) for _ in range(4)]
    for index in range(4):
        run.open(str(index), identity=str(index) * 64, horizon=256)
    observed_rungs = set()
    for pos in range(135):
        order = [3, 1, 0, 2] if pos % 2 else [0, 2, 3, 1]
        tokens = [(1 + index * 13 + pos * 3) % run.stage.args.vocab_size for index in order]
        got, logits = _advance(run, pool, [str(index) for index in order], tokens, pos)
        if pos:
            observed_rungs.add(graph_key(run.stage.layers[1], requests=4,
                position=pos, max_jobs=4, horizon=256).context_bucket)
        for row, index in enumerate(order):
            ids = torch.tensor([[tokens[row]]], dtype=torch.long)
            with torch.no_grad():
                want = controls[index].forward(controls[index].embed(ids), ids, pos)
                want_logits = controls[index].logits_all(want, full_logits=False)
            assert _bytes_equal(got[row:row + 1], want), (pos, index, "hidden")
            assert _bytes_equal(logits[row:row + 1], want_logits), (pos, index, "logits")
            for name, value in _buffers(controls[index], None).items():
                assert _bytes_equal(run.active[str(index)]["buffers"][name], value[0:1]), (pos, index, name)
    assert observed_rungs == {16, 64}


def test_signed_prefill_suppression_uses_the_same_eager_math_without_capture(monkeypatch):
    run, pool = _runner(_model())
    run.open("a", identity="a" * 64, horizon=64)
    monkeypatch.setattr(pool, "_capture", lambda *args: pytest.fail("suppressed prefill attempted capture"))
    for pos in range(3):
        with torch.no_grad(), run._bind_batch([run.active["a"]], pos):
            ids = torch.tensor([[pos + 1]], dtype=torch.long)
            pool.forward(run.stage.embed(ids), ids, pos, capture_allow=False)
            for name, value in run.workspace.items():
                run.active["a"]["buffers"][name].copy_(value[0:1])
            run.active["a"]["pos"] = pos + 1
    assert pool.status()["capture_suppressed"] == 3
    assert pool.status()["captured_graphs"] == pool.status()["replays"] == 0
    with pytest.raises(ValueError, match="boolean"):
        pool.forward(torch.empty(0), torch.empty(0), 3, capture_allow="false")


def test_bound_workspace_signature_and_quota_are_checked_before_state_mutation():
    run, pool = _runner(_model())
    for value in (1, pool.backup_bytes):
        with pytest.raises(ValueError, match="backup"):
            NativeBatchGraphPool(run.stage, run.workspace, max_jobs=4,
                max_graphs=2, graph_bytes=value, reference=True)
    with pytest.raises(ValueError, match="bounded"):
        NativeBatchGraphPool(run.stage, run.workspace, max_jobs=4,
            max_graphs=1, graph_bytes=64 << 20, reference=True)
    run.open("a", identity="a" * 64, horizon=64)
    ids = torch.tensor([[3]], dtype=torch.long)
    with pytest.raises(ValueError, match="addresses"):
        pool.forward(run.stage.embed(ids), ids, 0)  # workspace is not bound yet.
    with run._bind_batch([run.active["a"]], 0):
        original = run.stage.layers[0].attn.kv_cache
        run.stage.layers[0].attn.kv_cache = original.clone()
        try:
            with pytest.raises(ValueError, match="addresses"):
                pool.forward(run.stage.embed(ids), ids, 0)
        finally:
            run.stage.layers[0].attn.kv_cache = original
    assert run.active["a"]["pos"] == 0


@pytest.mark.parametrize("mutation", ["weight_replace", "weight_inplace", "rope_replace", "rope_inplace", "compressor_alias"])
def test_active_graph_owner_refuses_changed_immutable_model_and_stale_alias(mutation):
    run, pool = _runner(_model())
    run.open("a", identity="a" * 64, horizon=64)
    _advance(run, pool, ["a"], [3], 0)
    with run._bind_batch([run.active["a"]], 1), torch.no_grad():
        if mutation == "weight_replace":
            run.stage.layers[0].attn.wq_a.weight = torch.nn.Parameter(run.stage.layers[0].attn.wq_a.weight.clone())
        elif mutation == "weight_inplace":
            run.stage.layers[0].attn.wq_a.weight.add_(1)
        elif mutation == "rope_replace":
            run.stage.layers[0].attn.freqs_cis = run.stage.layers[0].attn.freqs_cis.clone()
        elif mutation == "rope_inplace":
            run.stage.layers[0].attn.freqs_cis.add_(1)
        else:
            run.stage.layers[1].attn.compressor.kv_cache = run.stage.layers[1].attn.compressor.kv_cache.clone()
        ids = torch.tensor([[4]], dtype=torch.long)
        with pytest.raises((ValueError, RuntimeError), match="immutable|alias"):
            pool.forward(run.stage.embed(ids), ids, 1)
    assert run.active["a"]["pos"] == 1


def test_failed_capture_restores_before_eager_fallback_and_budget_is_bounded(monkeypatch):
    run, pool = _runner(_model())
    control, twin = _runner(_model())
    for job in ("a", "b"):
        for owner in (run, control):
            owner.open(job, identity=job * 64, horizon=64)
    _advance(run, pool, ["a", "b"], [3, 7], 0)
    _advance(control, twin, ["a", "b"], [3, 7], 0)
    # Inject at the transaction seam, rather than pretending CPU is CUDA.
    # This verifies parent fallback ordering; the CUDA test checks real capture.
    pool.reference = False
    calls = []
    def rejected(layer, h, ids, key):
        pool._snapshot()
        try:
            for value in run.workspace.values():
                value.zero_()
            raise RuntimeError("capture refused after native state writes")
        finally:
            pool._restore()
    native_eager = pool._eager_block
    def checked(layer, h, ids, key):
        calls.append(key)
        return native_eager(layer, h, ids, key)
    monkeypatch.setattr(pool, "_capture", rejected)
    monkeypatch.setattr(pool, "_eager_block", checked)
    out, _ = _advance(run, pool, ["a", "b"], [4, 8], 1)
    expected, _ = _advance(control, twin, ["a", "b"], [4, 8], 1)
    assert torch.isfinite(out).all()
    assert torch.equal(out, expected)
    for job in ("a", "b"):
        for name, value in run.active[job]["buffers"].items():
            assert torch.equal(value, control.active[job]["buffers"][name])
    assert len(calls) == len(run.stage.layers)
    assert pool.status()["capture_failures"] == len(run.stage.layers)
    assert pool.status()["captured_graphs"] == pool.status()["replays"] == 0
    assert pool.status()["used_bytes"] == pool.backup_bytes
    assert all(slot["pos"] == 2 for slot in run.active.values())


@pytest.mark.parametrize("near_tie", [False, True])
def test_native_indexer_topk_tie_order_matches_original_narrow_width_at_every_k_boundary(monkeypatch, near_tie):
    import v4_whole_layer_graph as whole
    monkeypatch.setattr(whole, "_compressor_decode_cs", lambda *args, **kwargs: None)
    ref = SimpleNamespace(apply_rotary_emb=lambda *args: None,
        rotate_activation=lambda value: value, fp4_act_quant=lambda *args: None,
        fp4_block_size=32)
    for width in range(13):
        scores = torch.zeros(4, width, 1)
        if near_tie and width:
            scores[:, -1] = 1e-7
        indexer = SimpleNamespace(n_local_heads=1, head_dim=1, rope_head_dim=0,
            n_heads=1, index_topk=2, softmax_scale=1.,
            wq_b=lambda value: torch.ones(4, 1, 1), compressor=object(),
            weights_proj=lambda value: torch.ones(4, 1, 1), kv_cache=scores)
        with request_row_math():
            got = whole._indexer_decode_cs(ref, indexer, torch.ones(4, 1, 1),
                torch.ones(4, 1, 1), torch.tensor([3]), torch.tensor([width]),
                None, 8, None, False, width, batch_size=4, native_topk=True)
        for row in range(4):
            expected = (scores[row:row + 1, :, 0].unsqueeze(1).topk(min(2, width), -1).indices + 8).int()
            assert torch.equal(got[row:row + 1], expected), (width, row, near_tie)


@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA capture and replay required")
def test_cuda_batch_graph_matches_eager_twin_freshness_and_state(batch_size, cuda_device_defaults):
    """A real CUDA gate; CPU mode never substitutes or reports a pass here.

    BF16 tiny native equations test graph state/selection ownership. Real V4
    packed FP4/FP8 checkpoint qualification is an additional hardware gate.
    """
    with torch.device("cpu"):
        model = _model()
    for name in ("precompute_freqs_cis", "get_window_topk_idxs", "get_compress_topk_idxs"):
        getattr(V.ref(), name).cache_clear()
    run, pool = _runner(model, "cuda")
    control, twin = _runner(model, "cuda")
    scalar_controls = [_runner(model, "cuda")[0].stage for _ in range(batch_size)]
    for index in range(batch_size):
        for owner in (run, control):
            owner.open(str(index), identity=str(index) * 64, horizon=64)
    addresses = {name: value.data_ptr() for name, value in run.workspace.items()}
    outputs = []
    for pos in range(19):
        order = list(range(batch_size))[::-1] if pos % 2 else list(range(batch_size))
        jobs = [str(index) for index in order]
        tokens = [(1 + index * 13 + pos * 3) % run.stage.args.vocab_size for index in order]
        got, logits = _advance(run, pool, jobs, tokens, pos)
        want, want_logits = _advance(control, twin, jobs, tokens, pos, eager=True)
        assert torch.equal(got, want) and torch.equal(logits, want_logits), (batch_size, pos)
        for row, index in enumerate(order):
            scalar = scalar_controls[index]
            scalar_ids = torch.tensor([[tokens[row]]], dtype=torch.long, device="cuda")
            # This independent oracle deliberately does NOT enter request_row_math.
            with torch.no_grad():
                scalar_h = scalar.forward(scalar.embed(scalar_ids), scalar_ids, pos)
                scalar_logits = scalar.logits_all(scalar_h, full_logits=False)
            assert _bytes_equal(got[row:row + 1], scalar_h), (batch_size, pos, index, "native hidden")
            assert _bytes_equal(logits[row:row + 1], scalar_logits), (batch_size, pos, index, "native logits")
            for name, value in _buffers(scalar, None).items():
                assert _bytes_equal(run.active[str(index)]["buffers"][name], value[0:1]), (pos, index, name)
        for job in jobs:
            for name, value in run.active[job]["buffers"].items():
                assert torch.equal(value, control.active[job]["buffers"][name]), (pos, job, name)
        outputs.append(got.clone())
    # Revisit an existing exact-width/branch key with changed hidden and token
    # IDs from a reset slot owner. No graph identity is tied to a job string.
    for owner in (run, control):
        for index in range(batch_size):
            owner.close(str(index))
            owner.open("new" + str(index), identity=hex(index)[-1] * 64, horizon=64)
    jobs = ["new" + str(index) for index in range(batch_size)]
    for scalar in scalar_controls:
        scalar.reset()
    for pos in range(3):
        tokens = [42 + index + pos for index in range(batch_size)]
        got, _ = _advance(run, pool, jobs, tokens, pos)
        want, _ = _advance(control, twin, jobs, tokens, pos, eager=True)
        assert torch.equal(got, want)
        for index, scalar in enumerate(scalar_controls):
            scalar_ids = torch.tensor([[tokens[index]]], dtype=torch.long, device="cuda")
            with torch.no_grad():
                scalar_h = scalar.forward(scalar.embed(scalar_ids), scalar_ids, pos)
            assert _bytes_equal(got[index:index + 1], scalar_h), (pos, index, "reused native owner")
    torch.cuda.synchronize()
    status = pool.status()
    assert status["captured_graphs"] > 0 and status["replays"] > status["captured_graphs"]
    assert status["capture_failures"] == status["eager_fallbacks"] == 0
    assert status["used_bytes"] <= status["graph_bytes"]
    assert not torch.equal(outputs[1], outputs[2]), "position or token input appears frozen"
    assert addresses == {name: value.data_ptr() for name, value in run.workspace.items()}


@pytest.mark.gpu
@pytest.mark.parametrize("refusal", ["warmup_failure", "graph_count"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA stream recovery required")
def test_cuda_batch_graph_capture_refusal_restores_before_eager_and_commits_exact_state(refusal, cuda_device_defaults, monkeypatch):
    with torch.device("cpu"):
        model = _model()
    for name in ("precompute_freqs_cis", "get_window_topk_idxs", "get_compress_topk_idxs"):
        getattr(V.ref(), name).cache_clear()
    run, pool = _runner(model, "cuda", max_graphs=2 if refusal == "graph_count" else 512)
    control, twin = _runner(model, "cuda")
    jobs = [str(index) for index in range(4)]
    for owner in (run, control):
        for job in jobs:
            owner.open(job, identity=job * 64, horizon=64)
    if refusal == "warmup_failure":
        native_pre = pool._pre
        entered = 0
        def fail_after_side_write(layer, buffers, key):
            nonlocal entered
            out = native_pre(layer, buffers, key)
            if layer is run.stage.layers[0] and key.position == 1:
                entered += 1
                if entered == 2:  # shape probe passed; first SIDE warmup wrote KV.
                    raise RuntimeError("injected after enqueued side-stream native KV write")
            return out
        monkeypatch.setattr(pool, "_pre", fail_after_side_write)
    for pos in range(13):
        tokens = [5 + index * 13 + pos for index in range(4)]
        got, logits = _advance(run, pool, jobs, tokens, pos)
        want, want_logits = _advance(control, twin, jobs, tokens, pos, eager=True)
        assert torch.equal(got, want) and torch.equal(logits, want_logits), (refusal, pos)
        for job in jobs:
            for name, value in run.active[job]["buffers"].items():
                assert torch.equal(value, control.active[job]["buffers"][name]), (pos, job, name)
    torch.cuda.synchronize()
    status = pool.status()
    assert status["captured_graphs"] > 0 and status["replays"] > 0
    assert status["eager_fallbacks"] > 0
    assert status["accounted_device_bytes"] <= status["graph_bytes"]
    if refusal == "graph_count":
        assert status["live_graphs"] == 2 and status["budget_refusals"] > 0
        assert status["capture_failures"] == 0
    else:
        assert status["capture_failures"] == 1

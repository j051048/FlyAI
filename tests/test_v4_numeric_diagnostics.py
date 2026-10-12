import json

import pytest

torch = pytest.importorskip("torch")
R = pytest.importorskip("v4_ref_cpu")
V = pytest.importorskip("v4_stage")
WL = pytest.importorskip("v4_whole_layer_graph")
D = pytest.importorskip("v4_numeric_diagnostics")


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(V, "V4_CUDA_GRAPH", "0")
    yield
    torch.set_num_threads(old)


@pytest.fixture
def stages():
    if __import__("v4_kernels_cpu").backend() != "cpu":
        pytest.skip("CPU numerical fixtures require CPU reference kernels")
    from test_v4_stage import stage_from_oracle
    args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 4, 0, 0),
                      dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=96)
    oracle = R.build_oracle(args, 7)
    return [stage_from_oracle(oracle, args, 0, 4) for _ in range(2)]


def tokens(n=46):
    return torch.randint(0, 128, (1, n), generator=torch.Generator().manual_seed(19))


@pytest.mark.parametrize("position,width,accepted", [(11, 6, 3), (29, 7, 4), (33, 8, 5)])
def test_whole_twin_rejection_rebuilds_same_state_and_following_logits(stages, position, width, accepted):
    # Ratio 4/8 compression, window wrap, indexer selection and bucket steps are real arithmetic.
    left, right = stages
    want = D.trace_stage(left, tokens(), prompt_tokens=9, execution="whole_eager")
    got = D.trace_stage(right, tokens(), prompt_tokens=9, execution="whole_eager",
                        rejection={"position": position, "width": width, "accepted": accepted})
    result = D.compare_traces(want, got)
    assert result["exact_match"], result["first_difference"]
    assert got["evidence"] == "cpu_reference" and not got["speed_evidence"]
    assert left._pos == right._pos == 0
    assert all(row["captured_variants"] == 0 for row in got["graphs"])


def test_replay_does_not_call_reference_indexer_or_touch_graph_buffers(stages, monkeypatch):
    left, right = stages
    with D.execution_path(right, "whole_eager"):
        graphs = right._block_graphs
        for bg in graphs:
            bg.h_buf.fill_(123)
            bg.ids_buf.fill_(17)
        original = right._replay
        def replay(h, ids, pos):
            M = V.ref()
            def forbidden(*args, **kwargs):
                raise AssertionError("whole rollback switched to reference Indexer.forward")
            with monkeypatch.context() as patch:
                patch.setattr(M.Indexer, "forward", forbidden)
                return original(h, ids, pos)
        monkeypatch.setattr(right, "_replay", replay)
        want = D.trace_stage(left, tokens(24), prompt_tokens=9, execution="whole_eager")
        got = D.trace_stage(right, tokens(24), prompt_tokens=9, execution="whole_eager",
                            rejection={"position": 11, "width": 6, "accepted": 3})
        assert D.compare_traces(want, got)["exact_match"]
        assert all(bool((bg.h_buf == 123).all()) and bool((bg.ids_buf == 17).all()) for bg in graphs)


def test_diagnostic_catches_old_reference_replay_mutation(stages, monkeypatch):
    left, right = stages
    slim = pytest.importorskip("v4_ref_slim")
    monkeypatch.setattr(slim, "_REF_INDEXER_FORWARD", V.ref().Indexer.forward)
    monkeypatch.setattr(slim, "_GET_COMPRESS_TOPK_IDXS", V.ref().get_compress_topk_idxs)
    monkeypatch.setattr(slim, "_ACTIVE", True)
    monkeypatch.setattr(slim, "_JOB_MAX_POS", 30)
    monkeypatch.setattr(V.ref().Indexer, "forward", slim.slim_indexer_forward)
    want = D.trace_stage(left, tokens(30), prompt_tokens=9, execution="whole_eager")
    good = D.trace_stage(right, tokens(30), prompt_tokens=9, execution="whole_eager",
                         rejection={"position": 11, "width": 6, "accepted": 3})
    assert D.compare_traces(want, good)["exact_match"]
    def wrong(self, h, ids, pos):
        return self.L(h, pos, ids)
    monkeypatch.setattr(WL.WholeBlockGraphs, "replay_eager", wrong)
    # Only replace the replay, not normal whole math, to reproduce the previous state bug.
    with D.execution_path(right, "whole_eager"):
        for bg in right._block_graphs:
            def run(h, ids, pos, bg=bg):
                return bg._eager(h, ids, pos)
            bg.run = run
        right._numeric_eager_twin = False
        got = D.trace_stage(right, tokens(30), prompt_tokens=9,
                            rejection={"position": 11, "width": 6, "accepted": 3})
    assert not D.compare_traces(want, got)["exact_match"]


def test_same_teacher_input_exposes_slim_vs_whole_difference(stages):
    slim = pytest.importorskip("v4_ref_slim")
    mod = V.ref()
    was = getattr(mod.Indexer.forward, "_v4_ref_slim", False)
    slim.install(mod, item1=True, item2=False)
    try:
        a = D.trace_stage(stages[0], tokens(), prompt_tokens=9, execution="reference")
        b = D.trace_stage(stages[1], tokens(), prompt_tokens=9, execution="whole_eager")
        result = D.compare_traces(a, b)
        assert not result["exact_match"] and result["first_difference"]["left"]["position"] >= 9
    finally:
        if not was:
            slim.uninstall(mod)


def test_trace_integrity_input_and_resource_bounds(stages):
    st = stages[0]
    for kw in ({"prompt_tokens": 0}, {"prompt_tokens": 9, "max_tokens": 10},
               {"prompt_tokens": 9, "max_tensor_bytes": 0},
               {"prompt_tokens": 9, "rejection": {"position": 10, "width": 2, "accepted": 2}}):
        with pytest.raises(ValueError):
            D.trace_stage(st, tokens(20), **kw)
    a = D.trace_stage(st, tokens(15), prompt_tokens=9)
    changed = json.loads(json.dumps(a)); changed["records"][0]["position"] = 1
    with pytest.raises(ValueError, match="modified"):
        D.compare_traces(a, changed)
    b = D.trace_stage(stages[1], tokens(16), prompt_tokens=9)
    with pytest.raises(ValueError, match="identities"):
        D.compare_traces(a, b)


def test_whole_coverage_distinguishes_armed_and_retained_variants(stages):
    st = stages[0]
    with D.execution_path(st, "whole_eager"):
        summary = WL.coverage_summary(st)
        assert summary["captured_graphs"] == 0 and not summary["all_contexts_warmed"]
        assert len(summary["layers"]) == 4
        assert summary["process_budget"] == WL.V4_GRAPH_MAX


def test_cli_real_cpu_trace_and_compare(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    from safetensors.torch import save_file
    root = Path(__file__).resolve().parents[1]
    inputs, trace, result = (tmp_path / name for name in ("input.safetensors", "trace.json", "comparison.json"))
    save_file({"token_ids": tokens(6)}, str(inputs))
    env = dict(os.environ, V4_KERNELS="cpu", V4_CUDA_GRAPH="0")
    cmd = [sys.executable, str(root / "phase0/v4_numeric_diagnose.py")]
    child = subprocess.run(cmd + ["trace", "--fixture", "--input", str(inputs), "--prompt-tokens", "2",
        "--hi", "8", "--execution", "whole_eager", "--out", str(trace)], env=env,
        capture_output=True, text=True, timeout=45)
    assert child.returncode == 0, child.stderr
    row = json.loads(trace.read_text())
    assert row["evidence"] == "cpu_reference" and not row["identity"]["model"]["verified_artifacts"]
    child = subprocess.run(cmd + ["compare", str(trace), str(trace), "--out", str(result)], env=env,
        capture_output=True, text=True, timeout=15)
    assert child.returncode == 0, child.stderr
    assert json.loads(result.read_text())["exact_match"]


def test_equal_nan_outputs_never_pass_a_numeric_gate(stages, monkeypatch):
    st = stages[0]
    original = st.logits_all
    def broken(*args, **kwargs):
        value = original(*args, **kwargs)
        value.fill_(float("nan"))
        return value
    monkeypatch.setattr(st, "logits_all", broken)
    trace = D.trace_stage(st, tokens(12), prompt_tokens=9)
    result = D.compare_traces(trace, trace)
    assert not result["exact_match"] and result["first_difference"]["invalid_numeric"]


def test_forged_artifact_witness_cannot_claim_production_identity(stages, monkeypatch):
    from shard.weight_artifacts import ArtifactError
    monkeypatch.setattr(stages[0], "_artifact_verification", {
        "checkpoint_id": "a" * 64, "manifest_sha256": "b" * 64}, raising=False)
    with pytest.raises(ArtifactError, match="verification"):
        D.trace_stage(stages[0], tokens(12), prompt_tokens=9)


def test_diagnostic_owns_and_restores_horizon_and_records_effective_twin(stages, monkeypatch):
    slim = pytest.importorskip("v4_ref_slim")
    monkeypatch.setattr(slim, "_JOB_MAX_POS", 1)
    report = D.trace_stage(stages[0], tokens(12), prompt_tokens=9, execution="whole_eager")
    assert report["recipe"]["ref_slim_max_position"] == 12
    assert report["recipe"]["graph_mode"] == "whole"
    assert report["recipe"]["kernel_policy_scope"] == "numerical_twin_of_capture"
    assert not report["recipe"]["performance_evidence"]
    assert slim._JOB_MAX_POS == 1
    assert stages[0]._block_graphs is None


def test_host_paged_kv_requires_its_own_state_diagnostic(stages, monkeypatch):
    monkeypatch.setattr(stages[0], "_kv_runtime", object())
    with pytest.raises(ValueError, match="resident KV"):
        D.trace_stage(stages[0], tokens(12), prompt_tokens=9)


@pytest.mark.gpu
def test_cuda_actual_whole_graph_matches_its_eager_twin_and_rejection(tmp_path, monkeypatch):
    # Real CUDA capture over the CPU-reference math backend; this is not native FP4 qualification.
    if not torch.cuda.is_available():
        pytest.skip("requires an actual CUDA device")
    if __import__("v4_kernels_cpu").backend() != "cpu":
        pytest.skip("this CUDA state gate uses reference math; native FP4 has a separate gate")
    from test_v4_whole_layer import _stage
    args = R.cpu_args(max_seq_len=96, window_size=8)
    monkeypatch.setattr(V, "V4_CUDA_GRAPH", "whole")
    a = _stage(R.build_oracle(args, 7), args, "cuda", graph="whole")
    b = _stage(R.build_oracle(args, 7), args, "cuda", graph="whole")
    # Exercise CUDA directly; don't label these generated toy weights as production artifacts.
    ids = tokens(46).to("cuda")
    for st in (a, b):
        st.forward(st.embed(ids[:, :9]), ids[:, :9], 0)
        st._spec = True
    for p in range(9, 33):
        x = a.forward(a.embed(ids[:, p:p+1]), ids[:, p:p+1], p)
        with D.execution_path(b, "whole_eager"):
            y = b.forward(b.embed(ids[:, p:p+1]), ids[:, p:p+1], p)
        assert torch.equal(x, y)
    wrong = ids[:, 33:41].clone(); wrong[:, 5:] = (wrong[:, 5:]+1) % args.vocab_size
    a.forward(a.embed(wrong), wrong, 33)
    for p in range(33, 38):
        with D.execution_path(b, "whole_eager"):
            b.forward(b.embed(ids[:, p:p+1]), ids[:, p:p+1], p)
    for p in range(38, 46):
        x = a.forward(a.embed(ids[:, p:p+1]), ids[:, p:p+1], p)
        with D.execution_path(b, "whole_eager"):
            y = b.forward(b.embed(ids[:, p:p+1]), ids[:, p:p+1], p)
        assert torch.equal(x, y)
        assert torch.equal(a.logits_all(x, False), b.logits_all(y, False))
        for la, lb in zip(a.layers, b.layers):
            assert D.layer_state(la, p+1) == D.layer_state(lb, p+1)
    assert WL.coverage_summary(a)["captured_layers"] == args.n_layers

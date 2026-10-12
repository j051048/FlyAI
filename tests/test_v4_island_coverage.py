"""CPU coverage/fallback metadata tests; actual CUDA capture is separately marked."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
V = pytest.importorskip("v4_stage")


def stage():
    class Layer:
        def __init__(self, layer_id): self.layer_id = layer_id
        def __call__(self, hidden, position, ids): return hidden + position
    layers = [Layer(i) for i in (5, 6)]
    value = SimpleNamespace(lo=5, layers=layers, _graph_mode="island", _graph_refusal_reason=None)
    value._block_graphs = [V._BlockGraphs(layer, value) for layer in layers]
    return value


def test_armed_and_refused_stage_never_claim_captured_islands(monkeypatch):
    st = stage()
    row = V.island_coverage_summary(st)
    assert row["schema"] == "flyai-v4-island-coverage/1"
    assert row["captured_layers"] == row["captured_graphs"] == 0
    assert row["undecided_layers"] == 2 and row["refused_layers"] == 0
    assert all(r["replayed_steps"] == 0 and r["required_graphs"] == 3 for r in row["layers"])
    st._block_graphs = None
    st._graph_refusal_reason = "CPU cannot capture CUDA graphs"
    row = V.island_coverage_summary(st)
    assert row["refused_layers"] == 2 and row["undecided_layers"] == 0
    assert all(r["fallback_reason"] == st._graph_refusal_reason for r in row["layers"])
    assert not row["all_contexts_warmed"]


def test_real_budget_fallback_does_not_invent_capture_or_replay(monkeypatch):
    st = stage()
    monkeypatch.setattr(V, "V4_GRAPH_MAX", 2)
    monkeypatch.setattr(V, "_GRAPH_COUNT", 0)
    monkeypatch.setattr(V, "_GRAPH_SKIPPED", 0)
    hidden = torch.tensor([3.0])
    for runner in st._block_graphs:
        assert torch.equal(runner.run(hidden, None, 4), hidden + 4)
    row = V.island_coverage_summary(st)
    assert row["refused_layers"] == 2 and row["captured_layers"] == 0
    assert row["process_skipped_total"] == 6 and row["process_captured_total"] == 0
    assert all(r["fallback_reason"] == "graph_budget_exhausted" and r["replayed_steps"] == 0
               for r in row["layers"])


def test_partial_capture_exception_drops_runner_coverage(monkeypatch):
    st = stage()
    monkeypatch.setattr(V, "_GRAPH_COUNT", 0)
    monkeypatch.setattr(V, "_GRAPH_SKIPPED", 0)
    monkeypatch.setattr(V, "V4_GRAPH_MAX", 100)
    # Simulated failure exercises host error accounting only, never GPU qualification.
    def failed(): raise RuntimeError("capture fixture failure")
    monkeypatch.setattr(st._block_graphs[0], "_build", failed)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    assert st._block_graphs[0].run(torch.tensor([1]), None, 2).item() == 3
    row = V.island_coverage_summary(st)
    assert row["captured_layers"] == 0 and row["refused_layers"] == 1
    assert row["layers"][0]["fallback_reason"] == "capture_failed:RuntimeError"
    assert row["process_skipped_total"] == 3


def test_retained_metadata_requires_three_graphlets_and_matching_layer():
    st = stage()
    # Explicit metadata fixture, not CUDA evidence: no kernel/capture is executed.
    def fixture():
        graph = object.__new__(V._Graphlet)
        graph.graph = object()
        return graph
    st._block_graphs[0].g = (fixture(), fixture())
    st._block_graphs[1].g = (fixture(), fixture(), fixture())
    row = V.island_coverage_summary(st)
    assert row["captured_graphs"] == 5 and row["captured_layers"] == 1
    st._block_graphs[1].eager = True
    assert V.island_coverage_summary(st)["captured_layers"] == 0
    st._block_graphs.reverse()
    assert V.island_coverage_summary(st)["captured_graphs"] == 0


@pytest.mark.gpu
def test_actual_island_capture_reports_three_per_layer_after_decode(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("actual CUDA capture required")
    import v4_ref_cpu as R
    import v4_kernels_cpu
    if v4_kernels_cpu.backend() != "cpu":
        pytest.skip("reference-math CUDA fixture, not native checkpoint qualification")
    from test_v4_whole_layer import _stage
    monkeypatch.setattr(V, "V4_CUDA_GRAPH", "island")
    monkeypatch.setattr(V, "_GRAPH_COUNT", 0)
    monkeypatch.setattr(V, "_GRAPH_SKIPPED", 0)
    monkeypatch.setattr(V, "V4_GRAPH_MAX", 192)
    args = R.cpu_args()
    st = _stage(R.build_oracle(args, 7), args, "cuda", graph="island")
    ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    st.forward(st.embed(ids[:, :3]), ids[:, :3], 0)
    assert V.island_coverage_summary(st)["captured_layers"] == 0
    st.forward(st.embed(ids[:, 3:]), ids[:, 3:], 3)
    result = V.island_coverage_summary(st)
    assert result["captured_layers"] == args.n_layers
    assert result["captured_graphs"] == 3 * args.n_layers
    assert all(row["replayed_steps"] == 1 for row in result["layers"])

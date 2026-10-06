"""Tiny V4 monitoring stays outside the math, records real execution and resets per job."""
import pytest

torch = pytest.importorskip("torch")
REF = pytest.importorskip("v4_ref_cpu")
V4 = pytest.importorskip("v4_stage")

from test_v4_stage import stage_from_oracle


@pytest.fixture
def pair(monkeypatch):
    args = REF.cpu_args()
    oracle = REF.build_oracle(args, 7)
    monkeypatch.setattr(V4, "V4_RUNTIME_METRICS", False)
    plain = stage_from_oracle(oracle, args, 0, 2, tail=False)
    monkeypatch.setattr(V4, "V4_RUNTIME_METRICS", True)
    monitored = stage_from_oracle(oracle, args, 0, 2, tail=False)
    return args, plain, monitored


def test_stage_route_counts_and_reset_leave_outputs_unchanged(pair):
    args, plain, stage = pair
    ids = torch.tensor([[1, 2, 3, 4]])
    h = plain.embed(ids)
    expected = plain.forward(h, ids, 0)
    actual = stage.forward(h, ids, 0)
    assert torch.equal(actual, expected)
    value = stage.runtime_metrics()
    assert value["work"]["main"]["prefill"]["routed_entries"] == 4 * 2 * args.n_activated_experts
    assert value["totals"]["resident_hits"] == 0
    assert value["totals"]["reference_routes"] == value["totals"]["routed_entries"]
    assert value["kv"]["host_bytes"] > 0 and value["kv"]["gpu_bytes"] == 0
    assert plain.runtime_metrics() is None
    host = value["kv"]["host_bytes"]
    stage.reset()
    reset = stage.runtime_metrics()
    assert reset["totals"]["routed_entries"] == 0
    assert reset["kv"]["host_bytes"] == reset["kv"]["host_peak_bytes"] == host
    assert torch.equal(stage.forward(h, ids, 0), expected)


def test_rollback_prefix_replay_is_separate_work_and_views_are_not_double_counted(pair):
    args, plain, stage = pair
    prompt = torch.tensor([[1, 2, 3, 4]])
    stage.forward(stage.embed(prompt), prompt, 0)
    before = stage.runtime_metrics()["kv"]["host_bytes"]
    stage._spec = True
    ids = torch.tensor([[5, 6, 7]])
    stage.forward(stage.embed(ids), ids, 4)
    stage.forward(stage.embed([[8]]), torch.tensor([[8]]), 5)  # restore + replay position 4
    value = stage.runtime_metrics()
    unit = 2 * args.n_activated_experts
    assert value["work"]["main"]["decode"]["routed_entries"] == 4 * unit
    assert value["work"]["main"]["replay"]["routed_entries"] == unit
    assert value["kv"]["host_peak_bytes"] > before  # live rollback clones, not estimates
    # Every live base/view and retained KV snapshot is deduplicated by backing storage.
    tensors = list(stage._runtime_kv_tensors())
    storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                for t in tensors if t is not None}
    assert value["kv"]["host_bytes"] == sum(storages.values())
    stage.commit(8)
    assert stage.runtime_metrics()["kv"]["host_bytes"] == before
    assert stage.runtime_metrics()["work"]["main"]["replay"]["routed_entries"] == unit


def test_stage_counts_graph_execution_once_when_python_gate_is_bypassed(pair):
    args, plain, stage = pair
    ids = torch.tensor([[1]])
    h = stage.embed(ids)
    stage.forward(h, ids, 0)
    class Replay:
        def __init__(self):
            self.calls = 0
        def run(self, h, ids, pos):
            self.calls += 1
            return h  # models an already-captured layer whose Python Gate is not called
    stage._block_graphs = [Replay() for _ in stage.layers]
    stage.forward(h, ids, 1)
    assert stage.runtime_metrics()["work"]["main"]["decode"]["routed_entries"] == 2 * args.n_activated_experts
    assert all(replay.calls == 1 for replay in stage._block_graphs)


def test_runtime_receipt_call_uses_old_finalize_shape_when_disabled():
    import v4_pipe
    from types import SimpleNamespace
    class OldSigner:
        def finalize(self):
            return {"old": True}
    assert v4_pipe._finalize_stage_receipt(OldSigner(), SimpleNamespace()) == {"old": True}


def test_draft_observer_counts_actual_moe_calls_and_resets_with_main_job(monkeypatch):
    from types import SimpleNamespace
    from test_v4_dspark import tail_from_oracle
    args = REF.cpu_args()
    oracle = REF.build_oracle(args, 7)
    monkeypatch.setattr(V4, "V4_RUNTIME_METRICS", False)
    plain_stage, plain = tail_from_oracle(oracle, args)
    monkeypatch.setattr(V4, "V4_RUNTIME_METRICS", True)
    stage, draft = tail_from_oracle(oracle, args)
    before = stage.runtime_metrics()["kv"]["host_bytes"]
    wrapper = SimpleNamespace(tail=draft)
    stage.observe_runtime_drafter(wrapper)
    stage.observe_runtime_drafter(wrapper)  # cached RingDrafter is armed again each job
    assert stage.runtime_metrics()["kv"]["host_bytes"] > before
    prompt = torch.randn(1, 4, draft.hidden_dim, generator=torch.Generator().manual_seed(19))
    plain.prefill([1], prompt)
    draft.prefill([1], prompt)
    assert stage.runtime_metrics()["work"]["draft"]["prefill"]["routed_entries"] == 0
    # DSpark prefill is attention-only; only real decode MoEs add routes.
    hidden = torch.randn(1, 2, draft.hidden_dim, generator=torch.Generator().manual_seed(20))
    expected = plain.advance_and_draft([[2, 3]], hidden, 4)
    actual = draft.advance_and_draft([[2, 3]], hidden, 4)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    entries = 2 * draft.block_size * args.n_mtp_layers * args.n_activated_experts
    assert stage.runtime_metrics()["work"]["draft"]["decode"]["routed_entries"] == entries
    stage.reset()
    draft.reset()
    assert stage.runtime_metrics()["totals"]["routed_entries"] == 0


def test_flag_is_propagated_to_stage_launch_and_registered(monkeypatch):
    import v4_pipe
    import v4_levers
    monkeypatch.setenv("V4_RUNTIME_METRICS", "1")
    assert "V4_RUNTIME_METRICS=1 " in v4_pipe.stage_launch_cmd(0, 2, 0, 2, cuda_graph=False)
    assert "V4_RUNTIME_METRICS" in v4_levers.NON_LEVER_ENV


def test_two_stage_receipts_sign_metrics_and_reset_across_real_socket_jobs(tmp_path, monkeypatch):
    import v4_pipe
    from test_v4_pipe import _Ring
    from shard.receipt import verify_receipt, wire_receipt
    args = REF.cpu_args()
    oracle = REF.build_oracle(args, 7)
    directory = str(tmp_path)
    v4_pipe._write_tiny_checkpoint(directory, args, oracle)
    monkeypatch.setattr(V4, "V4_RUNTIME_METRICS", True)
    ring = _Ring(directory, args, [(0, 4), (4, 8)], receipts=True)
    try:
        first = ring.coordinate([1, 2, 3, 4], 3, nonce="metrics-job-one")
        second = ring.coordinate([1, 2], 1, nonce="metrics-job-two")
    finally:
        ring.close()
    assert first["receipts_ok"] is True and second["receipts_ok"] is True
    for receipt in second["receipts"]:
        verify_receipt(wire_receipt(receipt))
        value = receipt["runtime_metrics"]
        assert value["work"]["main"]["prefill"]["routed_entries"] == 2 * 4 * args.n_activated_experts
        assert value["work"]["main"]["decode"]["routed_entries"] == 0
        assert value["work"]["main"]["replay"]["routed_entries"] == 0
        assert receipt["nonce"] == "metrics-job-two"

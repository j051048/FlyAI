"""Real tiny native V4 batch math and independent recurrent state boundaries."""
import pytest
import torch

import v4_ref_cpu as R
from test_v4_conversation_cache import stage, forward
from v4_conversation_cache import _buffers
from v4_native_batch import DeviceNativeLanes, BATCH_CONTRACT
from v4_native_lane_backend import _validate_capacity
from v4_native_pipeline import drain_pipeline, observed_token_spans


@pytest.fixture
def model():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
        dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=64)
    # A preceding hardware fixture must not move the independent scalar
    # reference weights to the accelerator through a process default device.
    with torch.device("cpu"):
        oracle = R.build_oracle(args, 7)
    yield args, oracle
    torch.set_num_threads(old)


def runner(model, **options):
    return DeviceNativeLanes(stage(model), max_jobs=2, host_bytes=32 << 20,
        device_bytes=32 << 20, reference=True, **options)


def batch(run, jobs, tokens, pos):
    ids = torch.tensor(tokens, dtype=torch.long, device=run.stage.device).reshape(-1, 1)
    with torch.no_grad():
        h = run.forward_batch(jobs, run.stage.embed(ids), ids, pos)
        return h, run.logits_all(h, full_logits=False)


def test_one_actual_native_call_sees_request_batch_and_preserves_independent_state(model, monkeypatch):
    run = runner(model)
    controls = [stage(model), stage(model)]
    for index in range(2):
        run.open(str(index), identity=str(index) * 64, horizon=64)
    allocation_addresses = {name: value.data_ptr() for name, value in run.tensors()}
    original_addresses = {name: value.data_ptr() for name, value in _buffers(run.stage, None).items()}
    actual = run.stage.forward
    calls = []
    def observe(h, ids, pos):
        calls.append(tuple(ids.shape))
        return actual(h, ids, pos)
    monkeypatch.setattr(run.stage, "forward", observe)
    # Window wrapping plus both compression boundaries, with different tokens.
    for pos in range(21):
        order = [1, 0] if pos % 2 else [0, 1]
        tokens = [1 + 13 * index + pos for index in order]
        h, logits = batch(run, [str(index) for index in order], tokens, pos)
        for row, index in enumerate(order):
            want, want_logits = forward(controls[index], torch.tensor([[tokens[row]]]), pos)
            # The native batch uses the same equations, with different library
            # kernel shapes. Its separate contract permits measured rounding.
            assert torch.allclose(h[row:row + 1].float(), want.float(), atol=.0625, rtol=.05)
            assert int(logits[row].argmax()) == int(want_logits.argmax())
            for name, value in _buffers(controls[index], None).items():
                got = run.active[str(index)]["buffers"][name]
                assert torch.allclose(got.float(), value[0:1].float(), atol=.0625, rtol=.05)
        assert all(slot["pos"] == pos + 1 for slot in run.active.values())
    assert calls == [(2, 1)] * 21  # A per-request Python forward loop fails this.
    assert run.counters == {"batch_forward_calls": 21, "batched_request_rows": 42,
                            "scalar_forward_calls": 0, "state_host_copy_bytes": 0}
    assert allocation_addresses == {name: value.data_ptr() for name, value in run.tensors()}
    assert original_addresses == {name: value.data_ptr() for name, value in _buffers(run.stage, None).items()}
    assert run.config()["numerical_contract"] == BATCH_CONTRACT
    _validate_capacity(run.config(), 64)


def test_ragged_frontier_validation_cancellation_slot_reuse_and_failure_are_isolated(model, monkeypatch):
    run = runner(model)
    run.open("a", identity="a" * 64, horizon=64)
    run.open("b", identity="b" * 64, horizon=64)
    batch(run, ["a"], [1], 0)
    before = {k: v.clone() for k, v in run.active["a"]["buffers"].items()}
    with pytest.raises(ValueError, match="frontiers"):
        batch(run, ["a", "b"], [2, 3], 1)
    with pytest.raises(ValueError, match="unique"):
        batch(run, ["a", "a"], [2, 2], 1)
    assert run.active["a"]["pos"] == 1 and run.active["b"]["pos"] == 0
    for name, value in before.items():
        assert torch.equal(value, run.active["a"]["buffers"][name])
    run.close("b")  # A cancelled lane cannot disturb the survivor.
    run.open("c", identity="c" * 64, horizon=64)
    batch(run, ["c"], [7], 0)
    batch(run, ["c", "a"], [8, 2], 1)
    assert run.active["a"]["pos"] == run.active["c"]["pos"] == 2
    saved = {job: {k: v.clone() for k, v in slot["buffers"].items()} for job, slot in run.active.items()}
    native = run.stage.forward
    def fail(h, ids, pos):
        native(h, ids, pos)
        raise RuntimeError("injected after local forward")
    monkeypatch.setattr(run.stage, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        batch(run, ["a", "c"], [3, 9], 2)
    assert not run._batch_active and run.stage._pos == 0
    for job, buffers in saved.items():
        assert run.active[job]["pos"] == 2
        assert all(torch.equal(value, run.active[job]["buffers"][name]) for name, value in buffers.items())


def test_resident_batch_refuses_missing_device_quota_cpu_masquerade_or_graphs(model):
    st = stage(model)
    with pytest.raises(ValueError, match="CUDA"):
        DeviceNativeLanes(st, max_jobs=2, host_bytes=32 << 20, device_bytes=32 << 20)
    with pytest.raises(ValueError, match="reservation"):
        DeviceNativeLanes(st, max_jobs=2, host_bytes=32 << 20, device_bytes=1, reference=True)
    st._graph_mode = "whole"
    with pytest.raises(ValueError, match="graphs"):
        DeviceNativeLanes(st, max_jobs=2, host_bytes=32 << 20, device_bytes=32 << 20, reference=True)


@pytest.mark.parametrize("field,value", [("device_reserved_bytes", 1), ("batch_workspace_bytes", -1),
    ("host_reserved_bytes", 0), ("async_depth", True), ("graph_qualified", True),
    ("reference", False), ("prefill_chunk_tokens", 9)])
def test_signed_device_batch_capacity_cannot_understate_allocations_or_claim_capture(model, field, value):
    config = runner(model).config()
    config[field] = value
    with pytest.raises(ValueError):
        _validate_capacity(config, 64)


def test_pipeline_has_real_bounded_inflight_work_and_drains_before_owner_cancellation():
    events, sent, received = [], [], []
    stats = {"sent_cohorts": 0, "received_cohorts": 0, "peak_inflight_cohorts": 0}
    def send(item):
        events.append(("send", item)); sent.append(item)
    def receive(item):
        assert item in sent
        events.append(("receive", item)); received.append(item)
        return item * 10
    assert drain_pipeline([1, 2, 3, 4], depth=2, send=send, receive=receive, stats=stats) == [10, 20, 30, 40]
    assert events[:3] == [("send", 1), ("send", 2), ("receive", 1)]
    assert received == sent == [1, 2, 3, 4]
    assert stats == {"sent_cohorts": 4, "received_cohorts": 4, "peak_inflight_cohorts": 2}


@pytest.mark.parametrize("changed,value", [("cohort_sequence", 8), ("job_set_sha256", "other"),
    ("owner_epoch", "other:2"), ("context_tokens", 8), ("batch_jobs", 3), ("prefill", False)])
def test_observed_span_cannot_cross_missing_round_changed_job_owner_batch_or_frontier(changed, value):
    samples = [{"context_tokens": index, "batch_jobs": 2, "prefill": True, "owner_epoch": "owner:1",
                "job_set_sha256": "a" * 64, "cohort_sequence": index + 1,
                "submitted_at": index * .1, "drained_at": index * .1 + .01, "measured_at": 100 + index}
               for index in range(4)]
    spans = observed_token_spans(samples)
    full = next(value for value in spans if value["context_tokens"] == 0 and value["quantum_tokens"] == 4)
    assert full["elapsed_ms"] == pytest.approx(310)  # Includes the real 90ms inter-round gaps.
    samples[1][changed] = value
    spans = observed_token_spans(samples)
    assert not any(row["context_tokens"] == 0 and row["quantum_tokens"] > 1 for row in spans)


@pytest.fixture
def native_batch_accelerator_defaults(model):
    """The reference index-list caches depend on the ambient default device."""
    import v4_stage as V
    previous = torch.get_default_device()
    names = ("precompute_freqs_cis", "get_window_topk_idxs", "get_compress_topk_idxs")
    for name in names:
        getattr(V.ref(), name).cache_clear()
    try:
        # The model fixture dependency has already constructed its CPU oracle.
        # Native helper arange/empty calls below must now target the real GPU.
        torch.set_default_device("cuda")
        yield
    finally:
        for name in names:
            getattr(V.ref(), name).cache_clear()
        torch.set_default_device(previous)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="resident native batch qualification requires actual CUDA")
@pytest.mark.parametrize("count", [2, 3, 4])
def test_actual_cuda_resident_batch_preserves_independent_native_trajectories(model, count, native_batch_accelerator_defaults):
    # This gate deliberately refuses a CPU substitute; execution is opt-in on an
    # idle CUDA worker. Full FP4 checkpoint token agreement is a separate gate.
    import v4_stage as V
    args, oracle = model
    V.ref().precompute_freqs_cis.cache_clear()
    def build():
        value = V.Stage(0, args.n_layers, args, head=True, tail=True, device="cuda")
        for target, source in zip(value.layers, oracle.layers):
            target.load_state_dict(source.state_dict())
        value.embed_tokens.load_state_dict(oracle.embed.state_dict())
        value.norm.load_state_dict(oracle.norm.state_dict()); value.lm_head.load_state_dict(oracle.head.state_dict())
        with torch.no_grad():
            for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                getattr(value, name).copy_(getattr(oracle, name))
        return value
    try:
        controls = [build() for _ in range(count)]
        run = DeviceNativeLanes(build(), max_jobs=count, host_bytes=4 << 20, device_bytes=64 << 20)
        assert all(t.device.type == "cuda" for _, t in run.tensors())
        for index in range(count):
            run.open(str(index), identity=str(index) * 64, horizon=64)
        addresses = {name: value.data_ptr() for name, value in run.tensors()}
        trajectories = [[1 + 17 * index + pos for pos in range(13)] for index in range(count)]
        generated = [[] for _ in range(count)]
        for pos in range(21):
            order = list(reversed(range(count))) if pos % 2 else list(range(count))
            tokens = [trajectories[index][pos] if pos < 13 else generated[index][-1] for index in order]
            h, logits = batch(run, [str(index) for index in order], tokens, pos)
            assert torch.isfinite(h).all() and torch.isfinite(logits).all()
            for row, index in enumerate(order):
                want_h, want_logits = forward(controls[index], torch.tensor([[tokens[row]]], device="cuda"), pos)
                observed, expected = logits[row:row + 1].float(), want_logits.float()
                assert int(observed.argmax()) == int(expected.argmax())
                assert torch.equal(observed.topk(5).indices, expected.topk(5).indices)
                assert torch.allclose(observed, expected, atol=.0625, rtol=.05)
                assert torch.allclose(observed.log_softmax(-1), expected.log_softmax(-1), atol=.05, rtol=.01)
                assert torch.allclose(h[row:row + 1].float(), want_h.float(), atol=.0625, rtol=.05)
                for name, value in _buffers(controls[index], None).items():
                    assert torch.allclose(run.active[str(index)]["buffers"][name].float(),
                                          value[0:1].float(), atol=.0625, rtol=.05)
                if pos >= 12:
                    generated[index].append(int(observed.argmax()))
        assert all(len(tokens) == 9 for tokens in generated)
        assert addresses == {name: value.data_ptr() for name, value in run.tensors()}
        assert run.counters["batch_forward_calls"] == 21 and run.counters["state_host_copy_bytes"] == 0
        assert run.config()["state_tier"] == "device" and run.config()["reference"] is False
    finally:
        V.ref().precompute_freqs_cis.cache_clear()

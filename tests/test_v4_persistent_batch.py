"""Actual native ragged request batches; no complete per-step state copies."""
import pytest
import torch

from test_v4_native_batch import model, batch, native_batch_accelerator_defaults
from test_v4_conversation_cache import stage
from v4_native_batch import DeviceNativeLanes, PERSISTENT_BATCH_CONTRACT
from v4_native_lane_backend import _validate_capacity


def runner(model, jobs=2):
    return DeviceNativeLanes(stage(model), max_jobs=jobs, host_bytes=32 << 20,
        device_bytes=32 << 20, reference=True, storage_mode="persistent")


def step(run, jobs, tokens):
    return batch(run, jobs, tokens, [run.active[job]["pos"] for job in jobs])


def test_persistent_ragged_one_stage_call_matches_independent_native_row_trajectories(model, monkeypatch):
    run = runner(model, 4)
    controls = [DeviceNativeLanes(stage(model), max_jobs=2, host_bytes=32 << 20,
        device_bytes=32 << 20, reference=True) for _ in range(4)]
    for i in range(4):
        run.open(str(i), identity=str(i) * 64, horizon=64)
        controls[i].open(str(i), identity=str(i) * 64, horizon=64)
    addresses = {name: value.data_ptr() for name, value in run.tensors()}
    calls = []
    native = run.stage.forward
    def observed(hidden, ids, position):
        calls.append(tuple(ids.shape))
        return native(hidden, ids, position)
    monkeypatch.setattr(run.stage, "forward", observed)
    for tick in range(27):
        order = list(reversed(range(4))) if tick % 2 else list(range(4))
        order = [i for i in order if tick >= 2 * i and (tick % 5 == 0 or (tick + i) % 3)]
        if not order:
            continue
        jobs = [str(i) for i in order]
        tokens = [3 + 13 * i + run.active[str(i)]["pos"] for i in order]
        hidden, logits = step(run, jobs, tokens)
        for row, i in enumerate(order):
            expected_h, expected_logits = batch(controls[i], [str(i)], [tokens[row]], controls[i].active[str(i)]["pos"])
            assert torch.equal(hidden[row:row + 1], expected_h)
            assert torch.equal(logits[row:row + 1], expected_logits)
            for name, value in run.active[str(i)]["buffers"].items():
                assert torch.equal(value, controls[i].active[str(i)]["buffers"][name]), name
    assert run.counters["ragged_forward_calls"] > 0
    assert run.counters["state_full_copy_bytes"] == 0
    assert len(calls) == run.counters["batch_forward_calls"] + run.counters["scalar_forward_calls"]
    assert any(shape == (4, 1) for shape in calls)
    assert addresses == {name: value.data_ptr() for name, value in run.tensors()}
    config = run.config()
    assert config["numerical_contract"] == PERSISTENT_BATCH_CONTRACT
    assert config["batch_workspace_bytes"] < config["tensor_bytes_per_job"] * 4
    _validate_capacity(config, 64)


def test_write_journal_restores_actual_ring_wrap_and_compression_mutations(model, monkeypatch):
    run = runner(model)
    for i in range(2):
        run.open(str(i), identity=str(i) * 64, horizon=64)
    for pos in range(15):
        step(run, ["0", "1"], [pos + 1, pos + 21])
    before = {job: {name: value.clone() for name, value in slot["buffers"].items()}
              for job, slot in run.active.items()}
    native = run.stage.forward
    def failed(hidden, ids, pos):
        native(hidden, ids, pos)
        raise RuntimeError("after all native writes")
    monkeypatch.setattr(run.stage, "forward", failed)
    with pytest.raises(RuntimeError, match="native writes"):
        step(run, ["1", "0"], [49, 29])
    assert not run._batch_fault and not run._batch_active
    assert all(slot["pos"] == 15 for slot in run.active.values())
    assert run.counters["state_rollback_copy_bytes"] > 0
    for job, values in before.items():
        for name, value in values.items():
            assert torch.equal(value, run.active[job]["buffers"][name]), name
    monkeypatch.setattr(run.stage, "forward", native)
    step(run, ["0", "1"], [29, 49])
    assert all(slot["pos"] == 16 for slot in run.active.values())


def test_journal_allocation_independent_of_context_history_and_rejects_graph_claim(model):
    run = runner(model)
    assert sum(t.numel() * t.element_size() for t in run.journal.values()) == run.batch_workspace_bytes
    config = run.config()
    config["persistent_policy"]["full_state_copy_per_step"] = True
    with pytest.raises(ValueError, match="persistent"):
        _validate_capacity(config, 64)
    with pytest.raises(ValueError, match="graph-off"):
        DeviceNativeLanes(stage(model), max_jobs=2, host_bytes=32 << 20, device_bytes=32 << 20,
            reference=True, storage_mode="persistent", graph_mode="partial", graph_bytes=16 << 20)


def test_cancellation_and_slot_reuse_do_not_change_survivor_or_slot_storage(model):
    run = runner(model)
    run.open("a", identity="a" * 64, horizon=64)
    run.open("b", identity="b" * 64, horizon=64)
    for tick in range(9):
        step(run, ["a", "b"], [tick + 1, tick + 20])
    survivor = {name: value.clone() for name, value in run.active["a"]["buffers"].items()}
    ptr = {name: value.data_ptr() for name, value in run.active["b"]["buffers"].items()}
    run.close("b")
    run.open("c", identity="c" * 64, horizon=64)
    assert ptr == {name: value.data_ptr() for name, value in run.active["c"]["buffers"].items()}
    for name, value in survivor.items():
        assert torch.equal(value, run.active["a"]["buffers"][name])
    step(run, ["c", "a"], [50, 10])
    assert run.active["c"]["pos"] == 1 and run.active["a"]["pos"] == 10


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="persistent ragged batch qualification requires actual CUDA")
@pytest.mark.parametrize("count", [2, 4])
def test_actual_cuda_persistent_ragged_preserves_native_row_tokens_and_state(model, count, native_batch_accelerator_defaults):
    import v4_stage as V
    args, oracle = model
    def build():
        value = V.Stage(0, args.n_layers, args, head=True, tail=True, device="cuda")
        for target, source in zip(value.layers, oracle.layers):
            target.load_state_dict(source.state_dict())
        value.embed_tokens.load_state_dict(oracle.embed.state_dict())
        value.norm.load_state_dict(oracle.norm.state_dict())
        value.lm_head.load_state_dict(oracle.head.state_dict())
        with torch.no_grad():
            for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                getattr(value, name).copy_(getattr(oracle, name))
        return value
    run = DeviceNativeLanes(build(), max_jobs=count, host_bytes=8 << 20, device_bytes=64 << 20,
                            storage_mode="persistent")
    controls = [DeviceNativeLanes(build(), max_jobs=2, host_bytes=8 << 20, device_bytes=64 << 20)
                for _ in range(count)]
    for i in range(count):
        run.open(str(i), identity=str(i) * 64, horizon=64)
        controls[i].open(str(i), identity=str(i) * 64, horizon=64)
    for tick in range(22):
        order = [i for i in reversed(range(count)) if tick >= i and (tick % 5 == 0 or (tick + i) % 3)]
        if not order:
            continue
        tokens = [3 + 13 * i + run.active[str(i)]["pos"] for i in order]
        hidden, logits = step(run, [str(i) for i in order], tokens)
        for row, i in enumerate(order):
            expected_h, expected_logits = batch(controls[i], [str(i)], [tokens[row]], controls[i].active[str(i)]["pos"])
            # This gate deliberately requires exact /3 vs isolated native-row
            # execution. A mismatch must fix the new implementation, never be
            # hidden by broadening the scalar or existing /2 tolerances.
            assert torch.equal(hidden[row:row + 1], expected_h)
            assert torch.equal(logits[row:row + 1], expected_logits)
            for name, value in run.active[str(i)]["buffers"].items():
                assert torch.equal(value, controls[i].active[str(i)]["buffers"][name]), name
    assert run.counters["ragged_forward_calls"] > 0 and run.counters["state_full_copy_bytes"] == 0

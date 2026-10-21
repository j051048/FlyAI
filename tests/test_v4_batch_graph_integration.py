"""Production-connected partial graph policy with honest CPU reference evidence."""
from copy import deepcopy

import pytest
import torch

from test_v4_native_lane_socket import ring, job, reference
from test_v4_native_batch import model
from test_v4_conversation_cache import stage
from v4_native_batch import DeviceNativeLanes, BATCH_CONTRACT
from v4_native_lane_backend import _validate_capacity
from v4_batch_diagnostics import compare_native_batch, comparison_passed
from v4_batch_invariant import CONTRACT as MATH_CONTRACT
from shard.runtime_observation import validate_runtime_observation


def test_runner_integrates_real_request_axis_partial_eager_twin_and_static_quotas(model):
    st = stage(model)
    run = DeviceNativeLanes(st, max_jobs=4, host_bytes=64 << 20, device_bytes=32 << 20,
        reference=True, graph_mode="partial", graph_bytes=16 << 20, max_graphs=32)
    config = deepcopy(run.config())
    _validate_capacity(config, 64)
    for i in range(4):
        run.open(str(i), identity=str(i) * 64, horizon=64)
    for pos in range(17):
        ids = torch.tensor([[i * 17 + pos + 1] for i in range(4)])
        out = run.forward_batch([str(i) for i in range(4)], st.embed(ids), ids, pos)
        assert tuple(run.logits_all(out, full_logits=False).shape) == (4, st.args.vocab_size)
    assert run.config() == config  # captures/counters cannot mutate signed capacity
    status = run.graph_status()
    assert status["reference_eager"] == 16 * len(st.layers)
    assert status["captured_graphs"] == status["replays"] == 0
    assert status["coverage"] == "attention_dense_partial" and not status["graph_qualified"]
    assert all(slot["pos"] == 17 for slot in run.active.values())
    assert config["math_contract"] == MATH_CONTRACT


@pytest.mark.parametrize("ring", ["batch_graph"], indirect=True)
def test_real_signed_ring_receipts_bind_partial_graph_and_cannot_claim_cuda_in_cpu_reference(ring):
    backend, control = ring
    jobs = [job([1, 2, 3, 4, 5], 2), job([11, 12, 13, 14, 15], 2, "b")]
    expected = {value.id: reference(control, value.payload["prompt_ids"], value.max_new) for value in jobs}
    for value in jobs:
        value.payload["numerical_contract"] = BATCH_CONTRACT
    pending = list(jobs)
    while pending:
        results = backend.advance_batch(pending, emit={j.id: j.commit for j in pending},
            cancel_check={j.id: j.check_stop for j in pending}, max_step_tokens=1)
        for value in tuple(pending):
            result = results[value.id]
            if result is None:
                continue
            assert not isinstance(result, Exception)
            backend.verify_result_proof(value, result)
            assert result["tokens"] == expected[value.id]
            for receipt in result["proof"]["receipts"]:
                observation = receipt["runtime_observation"]
                native = observation["optimizations"]["native_batch"]
                assert native["scope"] == "stage_lifetime_not_per_job"
                assert native["config"]["graph_policy"]["mode"] == "partial"
                assert native["graphs"]["reference_eager"] > 0
                assert native["graphs"]["captured_graphs"] == native["graphs"]["replays"] == 0
                changed = deepcopy(observation)
                changed["optimizations"]["native_batch"]["graphs"]["replays"] = 1
                with pytest.raises(ValueError, match="CPU reference|reference batch"):
                    validate_runtime_observation(changed)
            backend.release_batch_job(value.id)
            pending.remove(value)


def test_graph_diagnostic_binds_actual_runner_and_cpu_scope_without_relaxing_scalar(model):
    st = stage(model)
    ids = torch.arange(4 * 17).reshape(4, 17) + 1
    report = compare_native_batch(st, ids, host_bytes=128 << 20, device_bytes=64 << 20,
        execution_workspace_bytes=8 << 20, reference=True, max_tokens=64,
        graph_mode="partial", graph_bytes=16 << 20, max_graphs=32)
    assert comparison_passed(report)
    assert report["qualification"]["old_scalar_abi_bitwise_invariant"]
    assert report["binding"]["batch_runtime"]["graph_policy"]["mode"] == "partial"
    assert not report["qualification"]["graph_capture_observed"]
    assert not report["qualification"]["release_qualified"]
    assert report["batch_graphs"]["reference_eager"] > 0
    forged = deepcopy(report)
    forged["binding"]["reference"] = False
    assert not comparison_passed(forged)


@pytest.mark.parametrize("ring", ["batch_graph"], indirect=True)
def test_signed_phase_admission_splits_equal_frontier_prefill_from_decode(ring, monkeypatch):
    backend, control = ring
    actual = DeviceNativeLanes.forward_batch
    observed = []

    def forward(run, jobs, h, ids, pos, *, capture_graph=True):
        observed.append((pos, len(jobs), capture_graph))
        return actual(run, jobs, h, ids, pos, capture_graph=capture_graph)

    monkeypatch.setattr(DeviceNativeLanes, "forward_batch", forward)
    jobs = [job([1, 2, 3], 2), job([11, 12, 13, 14, 15], 2, "b")]
    wanted = {value.id: reference(control, value.payload["prompt_ids"], value.max_new) for value in jobs}
    pending = list(jobs)
    for value in jobs:
        value.payload["numerical_contract"] = BATCH_CONTRACT
    while pending:
        outputs = backend.advance_batch(pending, emit={j.id: j.commit for j in pending},
            cancel_check={j.id: j.check_stop for j in pending}, max_step_tokens=1)
        for value in tuple(pending):
            result = outputs[value.id]
            if result is None:
                continue
            assert not isinstance(result, Exception)
            backend.verify_result_proof(value, result)
            assert result["tokens"] == wanted[value.id]
            assert all(row["runtime_observation"]["optimizations"]["native_batch"]
                       ["graphs"]["capture_suppressed"] >= 2 for row in result["proof"]["receipts"])
            backend.release_batch_job(value.id)
            pending.remove(value)
    assert (0, 2, False) in observed and (1, 2, False) in observed
    assert (2, 1, True) in observed and (2, 1, False) in observed
    assert not any(pos == 2 and count == 2 for pos, count, _ in observed)


def test_partial_graph_failure_does_not_commit_any_request_state(model, monkeypatch):
    st = stage(model)
    run = DeviceNativeLanes(st, max_jobs=4, host_bytes=64 << 20, device_bytes=32 << 20,
        reference=True, graph_mode="partial", graph_bytes=16 << 20, max_graphs=32)
    names = [str(index) for index in range(4)]
    for name in names:
        run.open(name, identity=name * 64, horizon=64)
    ids = torch.tensor([[1], [11], [21], [31]])
    for pos in range(2):
        run.forward_batch(names, st.embed(ids), ids, pos)
    saved = {name: {key: value.clone() for key, value in run.active[name]["buffers"].items()} for name in names}
    original = run.graph_pool.forward

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("failure after partial graph execution")

    monkeypatch.setattr(run.graph_pool, "forward", fail)
    with pytest.raises(RuntimeError, match="after partial"):
        run.forward_batch(names, st.embed(ids), ids, 2)
    assert st._pos == 0 and not run._batch_active
    for name in names:
        assert run.active[name]["pos"] == 2
        assert all(torch.equal(value, run.active[name]["buffers"][key]) for key, value in saved[name].items())
    monkeypatch.setattr(run.graph_pool, "forward", original)
    run.forward_batch(names, st.embed(ids), ids, 2)
    assert all(run.active[name]["pos"] == 3 for name in names)


def test_resource_inventory_preserves_unallocated_graph_and_host_metadata_reservation(model):
    from v4_resources import measure_stage_resources
    st = stage(model)
    run = DeviceNativeLanes(st, max_jobs=4, host_bytes=64 << 20, device_bytes=32 << 20,
        reference=True, graph_mode="partial", graph_bytes=16 << 20, max_graphs=32)
    st._native_lanes = run
    measured = measure_stage_resources(st, checkpoint_id="a" * 64)
    host_storages = {}
    for _, tensor in run.tensors():
        storage = tensor.untyped_storage()
        host_storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    reservation = measured["native_lane_reservation"]
    assert reservation["remaining_host_reserved_bytes"] == run.reserved_bytes - sum(host_storages.values())
    assert reservation["remaining_device_reserved_bytes"] == run.graph_bytes - run.graph_pool.accounted_device_bytes
    assert reservation["remaining_host_reserved_bytes"] > run.graph_host_bytes + run.hash_workspace_bytes


def test_alias_restore_failure_cannot_commit_kv_or_reopen_a_poisoned_owner(model, monkeypatch):
    import v4_runtime_init as init
    st = stage(model)
    run = DeviceNativeLanes(st, max_jobs=2, host_bytes=32 << 20, device_bytes=32 << 20, reference=True)
    names = ["a", "b"]
    for name in names:
        run.open(name, identity=name * 64, horizon=64)
    ids = torch.tensor([[1], [11]])
    run.forward_batch(names, st.embed(ids), ids, 0)
    saved = {name: {key: value.clone() for key, value in run.active[name]["buffers"].items()} for name in names}
    original = init.bind_attention_aliases

    def restore(attn, *, rebind=False):
        if rebind and attn.kv_cache is st.layers[0].attn.kv_cache and attn.kv_cache is run.buffers["main.0.kv"]:
            raise RuntimeError("injected alias restoration failure")
        return original(attn, rebind=rebind)

    monkeypatch.setattr(init, "bind_attention_aliases", restore)
    with pytest.raises(RuntimeError, match="alias restoration"):
        run.forward_batch(names, st.embed(ids), ids, 1)
    assert st._pos == 0 and not run._batch_active and run._batch_fault
    for name in names:
        assert run.active[name]["pos"] == 1
        assert all(torch.equal(value, run.active[name]["buffers"][key]) for key, value in saved[name].items())
    monkeypatch.setattr(init, "bind_attention_aliases", original)
    run.clear()
    with pytest.raises(RuntimeError, match="restart the owned stage"):
        run.open("c", identity="c" * 64, horizon=64)


def test_unsafe_capture_stream_cannot_be_reused_after_a_new_epoch(model, monkeypatch):
    from v4_native_batch_graph import UnsafeCaptureState
    st = stage(model)
    run = DeviceNativeLanes(st, max_jobs=2, host_bytes=32 << 20, device_bytes=32 << 20,
        reference=True, graph_mode="partial", graph_bytes=8 << 20, max_graphs=8)
    run.bind_epoch("owner:1")
    run.open("a", identity="a" * 64, horizon=64)
    ids = torch.tensor([[1]])

    def unsafe(*args, **kwargs):
        raise UnsafeCaptureState("injected unjoined CUDA capture stream")

    monkeypatch.setattr(run.graph_pool, "forward", unsafe)
    with pytest.raises(UnsafeCaptureState):
        run.forward_batch(["a"], st.embed(ids), ids, 0)
    assert run.active["a"]["pos"] == 0 and not run._batch_active
    run.bind_epoch("owner:2")
    with pytest.raises(RuntimeError, match="restart the owned stage"):
        run.open("b", identity="b" * 64, horizon=64)

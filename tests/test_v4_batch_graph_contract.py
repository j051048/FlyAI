"""Independent batch graph acceptance accounting; fixtures are not CUDA proof."""
from copy import deepcopy
import hashlib
from xml.etree.ElementTree import Element, SubElement, tostring

import pytest

import v4_release as release
from v4_batch_invariant import CONTRACT as MATH_CONTRACT
from v4_native_batch import BATCH_CONTRACT
from v4_native_lane_backend import _validate_capacity
from v4_native_lane_protocol import runtime_binding


def _capacity_fixture(mode="partial"):
    """Untrusted declared capacity fixture, without any claimed execution."""
    jobs, horizon, tensor, graph_count = 4, 64, 1024, 8
    metadata, hash_bytes = 65536 + horizon * 64, 2 << 20
    graph_bytes = 8 << 20 if mode == "partial" else 0
    graph_host = 65536 * graph_count if mode == "partial" else 0
    workspace = tensor * jobs
    device = 2 * workspace + graph_bytes
    per_job = metadata + tensor
    return {"enabled": True, "numerical_contract": BATCH_CONTRACT, "max_jobs": jobs,
            "horizon": horizon, "per_job_state_bytes": per_job,
            "reserved_state_bytes": per_job * jobs + workspace + hash_bytes + graph_bytes + graph_host,
            "tensor_bytes_per_job": tensor, "metadata_bytes_per_job": metadata,
            "prefill_chunk_tokens": 1, "restore_workspace_bytes": 0,
            "durable_prefix": {"enabled": False, "max_bytes": 0}, "ttl_s": 300.,
            "state_tier": "cpu_reference", "compute": "native_equal_frontier_batch",
            "host_reserved_bytes": metadata * jobs + hash_bytes + graph_host + device,
            "device_reserved_bytes": device, "batch_workspace_bytes": workspace,
            "async_depth": 2, "reference": True, "graph_qualified": False,
            "hash_workspace_bytes": hash_bytes, "math_contract": MATH_CONTRACT,
            "graph_policy": {"mode": mode, "graph_bytes": graph_bytes,
                             "host_metadata_bytes": graph_host,
                             "max_graphs": graph_count if mode == "partial" else 0,
                             "coverage": "attention_dense_partial" if mode == "partial" else "off",
                             "selection": "native_exact_width_topk"}}


@pytest.mark.parametrize("mode", ["off", "partial"])
def test_declared_batch_graph_capacity_is_separate_from_hardware_qualification(mode):
    value = _capacity_fixture(mode)
    _validate_capacity(value, 64)
    assert value["graph_qualified"] is False
    assert value["math_contract"] == MATH_CONTRACT
    assert BATCH_CONTRACT.endswith("/2")


@pytest.mark.parametrize("field,value", [
    ("mode", "whole"), ("mode", "island"), ("mode", False),
    ("graph_bytes", 0), ("graph_bytes", True), ("graph_bytes", -1),
    ("host_metadata_bytes", 0), ("host_metadata_bytes", True),
    ("max_graphs", 1), ("max_graphs", True), ("max_graphs", 4097),
    ("coverage", "whole"), ("coverage", "off"),
    ("selection", "stable_width_invariant_topk"),
])
def test_untrusted_graph_policy_cannot_overstate_scope_or_understate_reservation(field, value):
    config = _capacity_fixture()
    config["graph_policy"][field] = value
    with pytest.raises(ValueError):
        _validate_capacity(config, 64)


@pytest.mark.parametrize("field,value", [
    ("math_contract", "same_as_scalar_without_evidence"),
    ("graph_qualified", True), ("graph_qualified", 0),
    ("device_reserved_bytes", 8192), ("host_reserved_bytes", 1),
    ("reserved_state_bytes", 1),
])
def test_batch_math_and_graph_storage_are_frozen_capacity_obligations(field, value):
    config = _capacity_fixture()
    config[field] = value
    with pytest.raises(ValueError):
        _validate_capacity(config, 64)


def test_scalar_whole_graph_metadata_cannot_be_inserted_as_batch_capability():
    config = _capacity_fixture()
    config["graph_policy"]["whole_layers"] = 4
    with pytest.raises(ValueError):
        _validate_capacity(config, 64)


@pytest.mark.parametrize("policy", [None, [], {}, {"graph_bytes": 1},
    {"mode": "partial", "graph_bytes": "invalid", "host_metadata_bytes": 0,
     "max_graphs": 8, "coverage": "attention_dense_partial", "selection": "native_exact_width_topk"}])
def test_malformed_untrusted_graph_shape_is_rejected_before_reservation_arithmetic(policy):
    config = _capacity_fixture()
    config["graph_policy"] = policy
    with pytest.raises(ValueError):
        _validate_capacity(config, 64)


def test_runtime_signature_binds_graph_policy_math_and_reservations():
    config = _capacity_fixture()
    identity = runtime_binding({"native_lanes": config})
    for field, value in (("math_contract", "different/1"), ("device_reserved_bytes", 1)):
        changed = deepcopy(config)
        changed[field] = value
        assert runtime_binding({"native_lanes": changed}) != identity
    changed = deepcopy(config)
    changed["graph_policy"]["selection"] = "stable_topk"
    assert runtime_binding({"native_lanes": changed}) != identity


def _reference_graph_runner():
    import v4_ref_cpu as R
    from test_v4_conversation_cache import stage
    from v4_native_batch import DeviceNativeLanes
    from v4_runtime_init import bind_attention_aliases
    args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
        dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=64)
    owned = stage((args, R.build_oracle(args, 7)))
    # Rotary tables are normally shared through the reference's LRU. Mutation
    # tests own their tables so they cannot corrupt another fixture's oracle.
    for layer in owned.layers:
        layer.attn.freqs_cis = layer.attn.freqs_cis.clone()
        bind_attention_aliases(layer.attn, rebind=True)
    run = DeviceNativeLanes(owned, max_jobs=2,
        host_bytes=32 << 20, device_bytes=32 << 20, reference=True,
        graph_mode="partial", graph_bytes=8 << 20, max_graphs=8)
    for job in ("a", "b"):
        run.open(job, identity=job * 64, horizon=64)
    return run


def _reference_forward(run, pos):
    import torch
    ids = torch.tensor([[pos + 1], [pos + 21]])
    with torch.no_grad():
        return run.forward_batch(["a", "b"], run.stage.embed(ids), ids, pos)


def test_observed_reference_graph_execution_does_not_mutate_admitted_runtime_signature():
    import torch
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run = _reference_graph_runner()
        config = deepcopy(run.config())
        signature = runtime_binding({"native_lanes": config})
        for pos in range(3):
            _reference_forward(run, pos)
        assert run.config() == config
        assert runtime_binding({"native_lanes": run.config()}) == signature
        status = run.graph_status()
        assert status["reference_eager"] > 0
        assert status["captured_graphs"] == status["replays"] == 0
        assert status["qualification"] == "cpu_reference_only"
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize("mutation", ["rotary_storage", "rotary_version", "weight_version"])
def test_active_reference_graph_refuses_changed_immutable_model_without_committing_state(mutation):
    import torch
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run = _reference_graph_runner()
        _reference_forward(run, 0)
        saved = {job: {name: tensor.clone() for name, tensor in slot["buffers"].items()}
                 for job, slot in run.active.items()}
        with torch.no_grad():
            if mutation == "rotary_storage":
                run.stage.layers[0].attn.freqs_cis = run.stage.layers[0].attn.freqs_cis.clone()
            elif mutation == "rotary_version":
                run.stage.layers[0].attn.freqs_cis.add_(1)
            else:
                next(run.stage.layers.parameters()).add_(1)
        with pytest.raises(ValueError):
            _reference_forward(run, 1)
        assert all(slot["pos"] == 1 for slot in run.active.values())
        assert all(torch.equal(saved[job][name], tensor)
                   for job, slot in run.active.items() for name, tensor in slot["buffers"].items())
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize("unsafe", ["none", "join", "restore"])
def test_failed_side_warmup_is_joined_before_workspace_restore(monkeypatch, unsafe):
    """CPU transaction seam, not execution or capture evidence for an accelerator."""
    from contextlib import contextmanager
    import torch
    from v4_native_batch_graph import graph_key, UnsafeCaptureState
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run = _reference_graph_runner()
        pool = run.graph_pool
        _reference_forward(run, 0)
        events = []

        class Stream:
            def __init__(self, name):
                self.name = name

            def wait_stream(self, other):
                events.append(("wait", self.name, other.name))
                if unsafe == "join" and self.name == "caller" and other.name == "side":
                    raise RuntimeError("injected side join failure")

        caller, side = Stream("caller"), Stream("side")
        current = [caller]

        @contextmanager
        def use(stream):
            before = current[0]
            current[0] = stream
            try:
                yield
            finally:
                current[0] = before

        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args: 0)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *args: 0)
        monkeypatch.setattr(torch.cuda, "Stream", lambda *args, **kwargs: side)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: current[0])
        monkeypatch.setattr(torch.cuda, "stream", use)
        original_restore = pool._restore

        def restore():
            events.append(("restore", current[0].name))
            if unsafe == "restore" and current[0] is caller and len(calls) == 2:
                raise RuntimeError("injected workspace restore failure")
            original_restore()

        calls = []

        def pre(layer, buffers, key):
            calls.append(current[0].name)
            if current[0] is side:
                # Represent already enqueued side-stream work followed by a
                # host exception. A caller-stream restore must depend on it.
                for tensor in run.workspace.values():
                    tensor.zero_()
                raise RuntimeError("injected during side warmup")
            return (buffers["h"],) * 4

        monkeypatch.setattr(pool, "_restore", restore)
        monkeypatch.setattr(pool, "_pre", pre)
        if unsafe != "none":
            pool.reference = False  # Dispatch seam only, with real CPU tensors.
            monkeypatch.setattr(pool, "_eager_block", lambda *args: pytest.fail("unsafe graph failure reached eager fallback"))
        ids = torch.tensor([[2], [22]])
        rows = [run.active["a"], run.active["b"]]
        with torch.no_grad(), run._bind_batch(rows, 1):
            saved = {name: tensor.clone() for name, tensor in run.workspace.items()}
            key = graph_key(run.stage.layers[0], requests=2, position=1, max_jobs=2, horizon=64)
            if unsafe == "none":
                with pytest.raises(RuntimeError, match="injected during side warmup"):
                    pool._capture(run.stage.layers[0], run.stage.embed(ids), ids, key)
                assert all(torch.equal(value, run.workspace[name]) for name, value in saved.items())
            else:
                with pytest.raises(UnsafeCaptureState):
                    pool._block(run.stage.layers[0], run.stage.embed(ids), ids, key, False)
            assert all(slot["pos"] == 1 for slot in run.active.values())
            for row, slot in enumerate(rows):
                assert all(torch.equal(value, saved[name][row:row + 1]) for name, value in slot["buffers"].items())
        assert calls == ["caller", "side"]
        if unsafe == "join":
            assert events[-1] == ("wait", "caller", "side")
        else:
            assert events[-2:] == [("wait", "caller", "side"), ("restore", "caller")]
    finally:
        torch.set_num_threads(old)


def test_partial_graph_diagnostic_binds_actual_policy_and_cannot_relabel_reference_as_capture():
    import torch
    import v4_ref_cpu as R
    from test_v4_conversation_cache import stage
    from engines.deepseek_v4.v4_batch_diagnostics import compare_native_batch, comparison_passed
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
            dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=64)
        owned = stage((args, R.build_oracle(args, 7)))
        ids = torch.arange(20).reshape(4, 5) + 1
        report = compare_native_batch(owned, ids, host_bytes=64 << 20, device_bytes=64 << 20,
            execution_workspace_bytes=4 << 20, max_tokens=64, reference=True,
            graph_mode="partial", graph_bytes=8 << 20, max_graphs=8)
        policy = report["binding"]["batch_runtime"]["graph_policy"]
        assert policy == {"mode": "partial", "graph_bytes": 8 << 20,
            "host_metadata_bytes": 65536 * 8, "max_graphs": 8,
            "coverage": "attention_dense_partial", "selection": "native_exact_width_topk"}
        assert report["batch_graphs"]["coverage"] == "attention_dense_partial"
        assert report["batch_graphs"]["reference_eager"] > 0
        assert report["batch_graphs"]["captured_graphs"] == report["batch_graphs"]["replays"] == 0
        assert report["qualification"]["graph_capture_observed"] is False
        assert report["qualification"]["hardware_claim"] is False
        assert report["qualification"]["release_qualified"] is False
        assert comparison_passed(report)  # Arithmetic result, explicitly CPU.
        # Verifier boundary only: claiming an actual-device execution with this
        # all-eager observation must fail, even when all numeric fields agree.
        relabelled = deepcopy(report)
        relabelled["binding"]["reference"] = False
        assert not comparison_passed(relabelled)
    finally:
        torch.set_num_threads(old)


def _gate_accounting_fixture(case_names):
    """Synthetic evidence exercises the verifier, never probes this hardware."""
    suite = Element("testsuite")
    for name in case_names:
        SubElement(suite, "testcase", name=name)
    xml = tostring(suite, encoding="unicode")
    return {"schema": "flyai-v4-gpu-gate/1",
            "hardware": {"available": True, "idle": True, "capability": [12, 0]},
            "process_cleanup": {"confirmed": True}, "source_unchanged": True,
            "test_sources": {name: "a" * 64 for name in release.GPU_FILES},
            "source": {"files": [{"path": "synthetic_fixture", "sha256": "b" * 64}]},
            "exit_code": 0, "junit": xml,
            "junit_sha256": hashlib.sha256(xml.encode()).hexdigest()}


def test_release_freezes_real_batch_graph_test_source_and_all_four_actual_widths():
    assert "tests/test_v4_native_batch_graph.py" in release.GPU_FILES
    for width in (1, 2, 3, 4):
        assert f"test_cuda_batch_graph_matches_eager_twin_freshness_and_state[{width}]" in release.REQUIRED_CASES
    for failure in ("warmup_failure", "graph_count"):
        assert f"test_cuda_batch_graph_capture_refusal_restores_before_eager_and_commits_exact_state[{failure}]" in release.REQUIRED_CASES


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_missing_one_actual_bucket_cannot_pass_other_bucket_results(width):
    absent = f"test_cuda_batch_graph_matches_eager_twin_freshness_and_state[{width}]"
    report = _gate_accounting_fixture(name for name in release.REQUIRED_CASES if name != absent)
    verdict = release.evaluate_gpu_gate(report)
    assert verdict["status"] == "unverified"
    assert any("mandatory" in value for value in verdict["missing_evidence"])


def test_a_required_graph_case_prefix_does_not_cover_the_required_bucket():
    required = "test_cuda_batch_graph_matches_eager_twin_freshness_and_state[4]"
    names = [name if name != required else required + "_renamed" for name in release.REQUIRED_CASES]
    assert release.evaluate_gpu_gate(_gate_accounting_fixture(names))["status"] == "unverified"


@pytest.mark.parametrize("case_name", [
    pytest.param("test_cuda_batch_graph_matches_eager_twin_freshness_and_state[4]", id="graph4"),
    pytest.param("test_actual_cuda_resident_batch_preserves_independent_native_trajectories[4]", id="resident4"),
])
def test_skipped_or_failed_b4_remains_visible_in_raw_junit(case_name):
    report = _gate_accounting_fixture(release.REQUIRED_CASES)
    for element, expected in (("skipped", "unverified"), ("failure", "failed")):
        changed = deepcopy(report)
        before = f'<testcase name="{case_name}" />'
        after = f'<testcase name="{case_name}"><{element} /></testcase>'
        assert before in changed["junit"]
        changed["junit"] = changed["junit"].replace(before, after)
        changed["junit_sha256"] = hashlib.sha256(changed["junit"].encode()).hexdigest()
        if element == "failure":
            changed["exit_code"] = 1
        assert release.evaluate_gpu_gate(changed)["status"] == expected


def test_cpu_or_wrong_capability_cannot_qualify_even_with_all_case_names():
    report = _gate_accounting_fixture(release.REQUIRED_CASES)
    for hardware in ({"available": False}, {"available": True, "idle": True, "capability": [9, 0]}):
        changed = deepcopy(report)
        changed["hardware"] = hardware
        assert release.evaluate_gpu_gate(changed)["status"] == "unverified"

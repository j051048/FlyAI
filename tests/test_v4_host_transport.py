"""Actual local shared-memory ownership and strict frame integration.

Reference tests execute CPU/shared memory, never pretend CUDA IPC succeeded.
The separate GPU gate uses independent spawned CUDA workers.
"""
import base64
from copy import deepcopy
import json
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys

import pytest
import torch

import v4_pipe as VP
import v4_host_transport as H


def _config(directory, *, keys=None, backend="shared_memory_reference", uuids=("reference-0", "reference-1")):
    directory.mkdir(exist_ok=True)
    if os.name == "posix":
        os.chmod(directory, 0o700)
    keys = keys or [base64.b64encode(bytes([i]) * 32).decode() for i in (1, 2)]
    return {"schema": H.CONFIG_SCHEMA, "host_id": "rig-test", "directory": str(directory.resolve()),
            "secret": secrets.token_hex(32), "backend": backend, "max_bytes": 8192, "slots": 2,
            "ttl_s": 30, "channels": [{"id": "0-1", "cohort_id": "a" * 64,
                "src_pubkey": keys[0], "dst_pubkey": keys[1], "src_uuid": uuids[0], "dst_uuid": uuids[1]}]}


def _write(path, config):
    path.write_text(json.dumps(config), encoding="utf-8")
    if os.name == "posix":
        os.chmod(path, 0o600)


def _side(config, index, *, device="cpu", clock=None):
    row = config["channels"][0]
    kwargs = {} if clock is None else {"clock": clock}
    return H.HostTransport(config, local_pubkey=row["src_pubkey" if index == 0 else "dst_pubkey"],
        local_uuid=row["src_uuid" if index == 0 else "dst_uuid"], cohort_id=row["cohort_id"], device=device, **kwargs)


def _binding(pos=0, *, owner="boot:1", nonce="b" * 64):
    return H.request_binding(owner, {"job_id": "job", "nonce": nonce, "swarm_id": "ring"}, start_pos=pos)


@pytest.fixture
def pair(tmp_path):
    config = _config(tmp_path / "private")
    source, target = _side(config, 0), _side(config, 1)
    yield config, source, target
    target.close(); source.close()


def test_actual_shared_memory_reuses_only_acknowledged_slots_and_returns_owned_tensors(pair):
    config, source, target = pair
    value = torch.arange(128, dtype=torch.float32).reshape(1, 1, 4, 32)
    first = source.publish(value, _binding())
    second = source.publish(value + 1, _binding(1))
    assert source.publish(value + 2, _binding(2)) is None
    one = target.receive(first, _binding())
    assert torch.equal(one, value)
    two = target.receive(second, _binding(1))
    third = source.publish(value + 2, _binding(2))
    assert third is not None and third != first
    assert torch.equal(one, value)  # A reused source slot cannot overwrite its consumer.
    assert torch.equal(two, value + 1)
    assert torch.equal(target.receive(third, _binding(2)), value + 2)
    assert source.status()["socket_fallbacks"] == 1
    assert target.status()["ipc_copies"] == 0
    assert target.status()["shared_memory_copies"] == 3
    assert target.status()["performance_qualified"] is False
    assert source.reservation()["host_reserved_bytes"] == config["max_bytes"] + 4 * source.slot_bytes + config["slots"] * 65536


def test_reference_frame_hashes_actual_bytes_and_keeps_fp8_scales_on_the_wire(pair, monkeypatch):
    _, source, target = pair
    for packed in (False, True):
        monkeypatch.setattr(VP, "V4_FP8_WIRE", packed)
        value = torch.linspace(-2, 3, 128).reshape(1, 1, 4, 32).bfloat16()
        ids = torch.tensor([[17]])
        expected, expected_out = VP._make_step_frame(value, ids, 7, object())
        frame, out = VP._make_step_frame(value, ids, 7, object(), host_transport=source, host_binding=_binding(7))
        assert "host_ref" in frame and "h" not in frame
        if packed:
            assert "h8" in frame
        got, got_ids, incoming = VP._recv_hids(frame, object(), host_transport=target, host_binding=_binding(7))
        control = VP._recv_hids(expected, object())
        assert torch.equal(got, control[0]) and torch.equal(got_ids, ids)
        assert out == incoming == expected_out


@pytest.mark.parametrize("alter", ["owner", "nonce", "generation", "path", "tamper", "replay"])
def test_authenticated_capability_rejects_wrong_owner_tamper_path_and_replay(pair, alter):
    _, source, target = pair
    value = torch.ones(1, 1, 4, 16)
    binding, original = _binding(), None
    original = source.publish(value, binding)
    reference, changed = deepcopy(original), deepcopy(binding)
    if alter == "owner": changed["owner_epoch"] = "other:2"
    if alter == "nonce": changed["nonce"] = "c" * 64
    if alter == "generation": reference["generation"] += 1
    if alter == "path": reference["capability"] = "../../credentials"
    if alter == "tamper":
        path = target._manifest(reference["capability"])
        raw = json.loads(path.read_text()); raw["shape"][-1] += 1
        path.write_text(json.dumps(raw))
        with pytest.raises(H.HostTransportError, match="authentication"):
            target.receive(reference, changed)
        # Restore the authenticated original for orderly reference-test cleanup.
        raw["shape"][-1] -= 1; path.write_text(json.dumps(raw))
    elif alter == "replay":
        target.receive(reference, binding)
        with pytest.raises(H.HostTransportError, match="replay"):
            target.receive(reference, binding)
        return
    else:
        with pytest.raises(H.HostTransportError):
            target.receive(reference, changed)
    assert torch.equal(target.receive(original, binding), value)


def test_expiry_never_recycles_unacknowledged_data_and_fenced_discard_reclaims(pair):
    _, source, target = pair
    value = torch.ones(1, 1, 4, 16)
    ref = source.publish(value, _binding())
    body_path = target._manifest(ref["capability"])
    body = H._authenticated(body_path, target.key)
    body["expires_at"] = target.clock() - 1
    H._write(target.directory, body_path.name, body, target.key)
    with pytest.raises(H.HostTransportError, match="expiry"):
        target.receive(ref, _binding())
    assert source._retire(source._slots[0]) is False
    target.discard(ref, _binding())  # A fenced frame performs no read or forward.
    assert source._retire(source._slots[0]) is True
    assert target.status()["received"] == 0 and target.status()["discarded"] == 1


def test_close_retains_unacknowledged_slot_then_reclaims_only_after_actual_peer_ack(pair):
    _, source, target = pair
    value = torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 16)
    ref = source.publish(value, _binding())
    source.close()
    assert source.status()["allocated_host_bytes"] == source.slot_bytes
    assert source.status()["quarantined_slots"] == 1
    source.close()
    assert source.status()["quarantined"] == 1
    with pytest.raises(H.HostTransportError, match="closed"):
        source.publish(value, _binding(1))
    assert torch.equal(target.receive(ref, _binding()), value)
    source.close()
    assert source.status()["allocated_host_bytes"] == 0
    assert source.status()["quarantined_slots"] == 0


def test_failed_manifest_publication_leaves_no_live_capability_and_can_close(pair, monkeypatch):
    _, source, _ = pair
    original = H._write
    def fail_slot(directory, name, body, key):
        if name.startswith("slot-"):
            raise OSError("injected local manifest failure")
        return original(directory, name, body, key)
    monkeypatch.setattr(H, "_write", fail_slot)
    with pytest.raises(OSError, match="manifest failure"):
        source.publish(torch.ones(1, 1, 4, 16), _binding())
    assert source.status()["live_slots"] == 0 and source.status()["published"] == 0
    source.close()
    assert source.status()["allocated_host_bytes"] == 0


def test_independent_reader_exit_cannot_unlink_the_producer_owned_mapping(pair):
    _, source, target = pair
    value = torch.ones(1, 1, 4, 16)
    ref = source.publish(value, _binding())
    body = H._authenticated(source._manifest(ref["capability"]), source.key)
    code = ("import sys; sys.path.insert(0,sys.argv[1]); "
            "import v4_host_transport as H; "
            "print(H._sha(H._shared_bytes(sys.argv[2],int(sys.argv[3]))))")
    process = subprocess.run([sys.executable, "-c", code, str(Path(H.__file__).parent),
        body["memory_name"], str(body["nbytes"])], capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == body["sha256"]
    assert "resource_tracker" not in process.stderr
    assert torch.equal(target.receive(ref, _binding()), value)


def test_request_binding_refuses_coercion_or_incomplete_request():
    with pytest.raises(H.HostTransportError, match="request object"):
        H.request_binding("owner", None, start_pos=0)
    for position in (True, 1.5, "1"):
        with pytest.raises(H.HostTransportError, match="binding"):
            H.request_binding("owner", {"job_id": "job", "nonce": "b" * 64, "swarm_id": "ring"}, start_pos=position)


def test_real_host_copy_budget_refuses_before_slot_allocation_and_fences_release(pair, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from v4_host_compute import HostComputeSession
    from shard.host_compute_budget import HostBudgetBusy, observed_cpu_ids
    if len(observed_cpu_ids()) < 2:
        pytest.skip("one compute CPU and one control CPU are required")
    _, source, target = pair
    policy = dict(schema="v4-host-compute-config/1", host_id="rig-test", budget_path=str(tmp_path / "host.db"),
        cpu_capacity=2, control_reserve=1, copy_capacity=1, workers=1, copies=1, cpu_ids=[],
        strict_affinity=False, ttl_s=300, wait_s=0)
    owners = [HostComputeSession(SimpleNamespace(device="cpu"), policy) for _ in range(2)]
    source.host_compute, target.host_compute = owners
    fences = []
    def complete():
        fences.append(owners[0].budget.status()["active_copies"])
    for owner in owners:
        monkeypatch.setattr(owner, "_completion", complete)
    try:
        with owners[1].copy():
            with pytest.raises(HostBudgetBusy):
                source.publish(torch.ones(1, 1, 4, 16), _binding())
            assert source.status()["allocated_host_bytes"] == 0
        value = torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 16)
        reference = source.publish(value, _binding())
        assert torch.equal(target.receive(reference, _binding()), value)
        assert fences == [1, 1, 1]
        assert owners[0].budget.status()["active_copies"] == 0
        assert source.status()["host_copy_budget"] == "completion_fenced"
    finally:
        for owner in owners:
            owner.close()


def test_service_startup_bind_failure_closes_real_local_owners(pair, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from v4_host_compute import HostComputeSession
    from test_v4_host_compute import policy
    _, _, target = pair
    owner = HostComputeSession(SimpleNamespace(device="cpu"), policy(tmp_path / "startup.db"))
    stage = SimpleNamespace(_host_compute=owner, device="cpu")
    monkeypatch.setattr(VP, "_v4", lambda: SimpleNamespace(Stage=lambda *a, **kw: stage))
    monkeypatch.setattr(H, "configured", lambda *a, **kw: target)
    monkeypatch.setattr(VP.v4_levers, "report", lambda **kw: None)
    import v4_observability
    monkeypatch.setattr(v4_observability, "runtime_observation", lambda *a, **kw: {})
    blocker = socket.socket(); blocker.bind(("127.0.0.1", 0)); blocker.listen(1)
    try:
        with pytest.raises(OSError):
            VP.serve_stage(0, 1, 0, 1, blocker.getsockname()[1], bind="127.0.0.1", device="cpu",
                args=SimpleNamespace(n_hash_layers=3), receipts=False)
        assert owner.status()["closed"] is True
        assert target.status()["closed"] is True
        assert not (target.directory / "ready-0-1.json").exists()
        assert owner.budget.status()["active_permits"] == 0
    finally:
        blocker.close()


def test_service_local_cleanup_never_forces_an_uncertain_copy_permit(tmp_path):
    from types import SimpleNamespace
    from v4_host_compute import HostComputeSession
    from test_v4_host_compute import policy
    owner = HostComputeSession(SimpleNamespace(device="cpu"), policy(tmp_path / "uncertain.db"))
    def failed():
        raise RuntimeError("completion cannot be established")
    try:
        with pytest.raises(RuntimeError, match="cannot be established"):
            with owner.copy(completion=failed):
                pass
        VP._close_stage_local_resources(SimpleNamespace(_host_compute=owner), 0)
        assert owner.budget.status()["active_copies"] == 1
        assert not owner.status()["closed"] and owner.status()["faulted"]
    finally:
        owner.close(confirmed_stopped=True)


def test_missing_peer_or_oversized_frame_keeps_original_socket_payload(tmp_path):
    config = _config(tmp_path / "private")
    source = _side(config, 0)
    try:
        assert source.publish(torch.ones(1, 1, 4, 16), _binding()) is None
        target = _side(config, 1)
        try:
            assert source.publish(torch.ones(1, 1, 4, 4096), _binding()) is None
            assert source.status()["allocated_host_bytes"] == 0
        finally:
            target.close()
    finally:
        source.close()


def test_config_identity_omits_private_path_and_key_but_binds_content(tmp_path):
    config = _config(tmp_path / "private")
    path = tmp_path / "config.json"; _write(path, config)
    public = H.public_config(path)
    assert config["secret"] not in json.dumps(public) and "directory" not in public
    config["slots"] = 3; _write(path, config)
    assert H.public_config(path)["config_sha256"] != public["config_sha256"]
    config["slots"] = True; _write(path, config)
    with pytest.raises(H.HostTransportError, match="bounded"):
        H.load_config(path)


def _receiver_worker(path, port_queue, backend, device):
    transport = None
    try:
        config = H.load_config(path)
        transport = _side(config, 1, device=device)
        server = socket.socket(); server.bind(("127.0.0.1", 0)); server.listen(1)
        server.settimeout(20)
        port_queue.put({"port": server.getsockname()[1]})
        conn, _ = server.accept(); conn.settimeout(20)
        try:
            frame = VP.recv_msg(conn)
            got, _, incoming = VP._recv_hids(frame, object(), device=device,
                host_transport=transport, host_binding=_binding(3))
            VP.send_msg(conn, {"values": got.cpu().tolist(), "sha256": H._sha(incoming), "status": transport.status()})
        finally:
            conn.close(); server.close()
    except BaseException as error:
        port_queue.put({"error": type(error).__name__ + ": " + str(error)})
        raise
    finally:
        if transport is not None:
            transport.close()


def _cross_process(config, path, *, source_device="cpu", target_device="cpu"):
    _write(path, config)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_receiver_worker, args=(str(path), queue, config["backend"], target_device))
    process.start()
    source = None
    try:
        hello = queue.get(timeout=30)
        assert "error" not in hello, hello
        source = _side(config, 0, device=source_device)
        value = torch.arange(128, dtype=torch.float32, device=source_device).reshape(1, 1, 4, 32)
        frame, outgoing = VP._make_step_frame(value, torch.tensor([[3]]), 3, object(),
                                             host_transport=source, host_binding=_binding(3))
        assert "host_ref" in frame and "h" not in frame
        with socket.create_connection(("127.0.0.1", hello["port"]), timeout=20) as conn:
            VP.send_msg(conn, frame)
            response = VP.recv_msg(conn)
        assert response["values"] == value.cpu().tolist()
        assert response["sha256"] == H._sha(outgoing)
        process.join(15)
        assert process.exitcode == 0
        return response["status"]
    finally:
        if source is not None:
            source.close()
        if process.is_alive():
            process.terminate(); process.join(10)  # Only this test's owned child.
        queue.close(); queue.join_thread()


def test_actual_spawned_process_receives_authenticated_shared_memory_and_equal_receipt_bytes(tmp_path):
    config = _config(tmp_path / "private")
    result = _cross_process(config, tmp_path / "config.json")
    assert result["shared_memory_copies"] == 1 and result["ipc_copies"] == 0
    assert result["reference"] is True and result["performance_qualified"] is False


def test_owned_probe_reports_real_process_cleanup_and_never_qualifies_reference():
    from phase0.v4_host_transport_probe import measure
    result = measure(repeats=2, tokens=2, dim=16, max_mib=1)
    assert result["gpu_copy_observed"] is False and result["performance_qualified"] is False
    assert result["process_cleanup"] == {"confirmed": True, "quarantined_slots": 0}
    assert result["source"]["allocated_host_bytes"] == 0
    assert result["target"]["shared_memory_copies"] == 2
    assert all(sample["owner_ready_to_verified_reply_ms"] > 0 for sample in result["samples"])


def test_device_digest_probe_cannot_qualify_cpu_reference():
    from phase0.v4_host_transport_probe import measure
    with pytest.raises(ValueError, match="actual CUDA"):
        measure(digest_backend="device_sha256")


@pytest.mark.gpu
def test_actual_device_digest_ipc_probe_reads_target_gpu_and_preserves_receipt_bytes():
    if os.name != "posix" or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two owned Linux CUDA devices required")
    from phase0.v4_host_transport_probe import measure
    result = measure(source_device="cuda:0", target_device="cuda:1", execute_gpu=True,
                     digest_backend="device_sha256", repeats=3, tokens=1, dim=4096)
    assert result["gpu_copy_observed"] and not result["performance_qualified"]
    assert result["receipt_hash_path"] == "actual_bytes_device_sha256"
    assert result["target"]["shared_memory_copies"] == 0
    assert "no_lease_no_strict_session_not_planner_evidence" in result["scope"]


def test_probe_refuses_accelerator_side_effects_without_explicit_opt_in():
    from phase0.v4_host_transport_probe import measure
    with pytest.raises(ValueError, match="execute-gpu"):
        measure(source_device="cuda:0", target_device="cuda:1")
    with pytest.raises(ValueError, match="bounded"):
        measure(tokens=10000)


@pytest.fixture
def host_ring(tmp_path, monkeypatch, request):
    """Reuse the real tiny native ring, provisioning its exact pinned peers."""
    import test_v4_native_lane_socket as sockets
    keys, installed = [], []
    generate_key, build_plan, configured = sockets.gen_key, sockets.build_plan, H.configured

    def key():
        value = generate_key(); keys.append(value); return value

    def plan(*args, **kwargs):
        value = build_plan(*args, **kwargs)
        config = _config(tmp_path / "host-private", keys=[sockets.pub_b64(k) for k in keys[:2]])
        config["channels"][0]["cohort_id"] = value["cohort_id"]
        for index, row in enumerate(value["stages"]):
            row["gpu_uuid"] = "reference-" + str(index)
        path = tmp_path / "host-config.json"; _write(path, config)
        monkeypatch.setenv("V4_HOST_TRANSPORT_CONFIG", str(path))
        return value

    def install(stage, session):
        result = configured(stage, session)
        if result is not None:
            installed.append(result)
        return result

    monkeypatch.setattr(sockets, "gen_key", key)
    monkeypatch.setattr(sockets, "build_plan", plan)
    monkeypatch.setattr(H, "configured", install)
    factory = sockets.ring.__wrapped__(tmp_path, monkeypatch, request)
    backend, control = next(factory)
    try:
        yield backend, control, installed
    finally:
        with pytest.raises(StopIteration):
            next(factory)


@pytest.mark.parametrize("host_ring", [None, "batch"], indirect=True)
def test_real_strict_ring_local_slots_preserve_original_tokens_and_signed_receipts(host_ring):
    import test_v4_native_lane_socket as sockets
    backend, control, installed = host_ring
    values = [sockets.job([1, 2, 3, 4, 5, 6], 3), sockets.job([11, 12, 13, 14], 3, "b")]
    contract = backend._native_runner.numerical_contract
    wanted = {value.id: sockets.reference(control, value.payload["prompt_ids"], value.max_new) for value in values}
    for value in values:
        value.payload["numerical_contract"] = contract
    pending = list(values)
    while pending:
        result = backend.advance_batch(pending, emit={j.id: j.commit for j in pending},
            cancel_check={j.id: j.check_stop for j in pending}, max_step_tokens=1)
        for value in tuple(pending):
            output = result[value.id]
            if output is None:
                continue
            assert not isinstance(output, Exception)
            assert output["tokens"] == wanted[value.id]
            assert backend.verify_result_proof(value, output)
            backend.release_batch_job(value.id)
            pending.remove(value)
    assert len(installed) == 2
    assert sum(owner.status()["published"] for owner in installed) > 0
    assert sum(owner.status()["received"] for owner in installed) > 0
    assert all(owner.status()["ipc_copies"] == 0 and owner.reference for owner in installed)


@pytest.mark.gpu
def test_actual_two_device_interprocess_activation_copy_preserves_payload_and_receipt(tmp_path):
    if os.name != "posix" or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two idle Linux CUDA devices are required for real IPC qualification")
    from v4_observability import gpu_uuid_text
    uuids = tuple(gpu_uuid_text(getattr(torch.cuda.get_device_properties(i), "uuid", None)) for i in (0, 1))
    assert all(uuids)
    config = _config(tmp_path / "private", backend="cuda_ipc", uuids=uuids)
    result = _cross_process(config, tmp_path / "config.json", source_device="cuda:0", target_device="cuda:1")
    assert result["ipc_copies"] == 1 and result["shared_memory_copies"] == 0
    assert result["reference"] is False and result["performance_qualified"] is False

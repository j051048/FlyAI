"""Actual HTTP/SSE and callback-driven transport recovery using an explicit fake model."""
import http.client
import json
import io
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import v4_gateway as GW
from shard.service_queue import Job, JobCancelled, JobExpired, TenantLimits
from shard.receipt import ReceiptSigner, gen_key, pub_b64

KEY_A, KEY_B = "tenant-a-test-api-key-0123456789", "tenant-b-test-api-key-9876543210"


class Tokenizer:
    eos_token_id = None
    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


class Channel:
    def __init__(self):
        self.closed = threading.Event()
        self.socket, self.peer = socket.socketpair()
    def settimeout(self, timeout):
        self.timeout = timeout
        self.socket.settimeout(timeout)
    def gettimeout(self):
        return self.socket.gettimeout()
    def setblocking(self, enabled):
        self.socket.setblocking(enabled)
    def fileno(self):
        return self.socket.fileno()
    def recv(self, size, flags=0):
        return self.socket.recv(size, flags)
    def shutdown(self, how):
        self.closed.set()
        self.socket.shutdown(how)
    def close(self):
        self.closed.set()
        self.socket.close(); self.peer.close()


class Pipe:
    V4_MODEL_ID, N_LAYERS, SWARM_TOKEN = "test-v4-model", 2, None
    def __init__(self, *, fail_first=False, mismatch=False, bad_receipt=False, block=False):
        self.keys = [gen_key(), gen_key()]
        self.assignments = {pub_b64(key): (i, i + 1) for i, key in enumerate(self.keys)}
        self.calls, self.connects = 0, 0
        self.fail_first, self.mismatch, self.bad_receipt, self.block = fail_first, mismatch, bad_receipt, block
        self.started = threading.Event()
        self.recorded = []

    def connect_ring(self, *args, **kwargs):
        self.connects += 1
        return Channel(), Channel()

    def _encode_prompt(self, tokenizer, job):
        return [1, 2]

    def coordinate(self, pipe, ret, prompt_ids, maximum, *, cancel_check, expected_by_signer,
                   strict_job_binding=False, **kw):
        assert strict_job_binding and expected_by_signer == self.assignments
        self.calls += 1
        self.recorded.append((list(prompt_ids), maximum, kw["job_id"], kw["nonce"]))
        tokens = list(range(65, 65 + maximum))
        if self.mismatch and self.calls > 1:
            tokens[0] = 90
        self.started.set()
        if self.block:
            kw["on_token"](tokens[0])
            assert ret.closed.wait(2), "blocked receive was not actively interrupted"
            raise OSError("receive interrupted")
        for index, token in enumerate(tokens):
            cancel_check(); kw["on_token"](token)
            if self.fail_first and self.calls == 1 and index == 1:
                raise OSError("actual callback frontier before transport failure")
        receipts = []
        for i, key in enumerate(self.keys):
            signer = ReceiptSigner(key, kw["swarm_id"], kw["job_id"], i, i + 1, nonce=kw["nonce"])
            for step in range(len(tokens)):
                signer.observe(f"boundary-{i}-{step}".encode(), f"boundary-{i+1}-{step}".encode())
            receipts.append(signer.finalize())
        if self.bad_receipt:
            receipts[0]["nonce"] = "forged"
        return {"ok": True, "tokens": tokens, "receipts": receipts, "receipts_ok": True}

    coordinate_dspark = coordinate
    coordinate_dspark_pipelined = coordinate


def backend(pipe):
    return GW.V4RingBackend("unused", "head:1", "tail:2", pipe.assignments,
        vp=pipe, tokenizer=Tokenizer(), max_retries=1)


def job(maximum=4, timeout=3):
    return Job("a", {"prompt_ids": [1, 2], "mode": "pipelined"}, 2, maximum,
               time.monotonic() + timeout, "test", None, time.monotonic(), state="running")


def test_real_adapter_replays_original_request_checks_prefix_and_never_duplicates_commit():
    pipe = Pipe(fail_first=True)
    adapter, request = backend(pipe), job()
    result = adapter.execute(request, request.commit, request.check_stop)
    assert request.tokens == [65, 66, 67, 68] and result["tokens"] == request.tokens
    assert pipe.connects == 2 and pipe.calls == 2
    assert [call[:2] for call in pipe.recorded] == [([1, 2], 4), ([1, 2], 4)]
    assert pipe.recorded[0][3] != pipe.recorded[1][3]
    assert result["proof"]["verified"] and result["proof"]["scope"] == "complete_final_attempt"
    assert result["recovery"]["replayed_committed_tokens"] == 2
    adapter.close()


def test_replayed_prefix_mismatch_fails_without_false_continuation():
    pipe = Pipe(fail_first=True, mismatch=True)
    adapter, request = backend(pipe), job()
    with pytest.raises(GW.ReplayMismatch):
        adapter.execute(request, request.commit, request.check_stop)
    assert request.tokens == [65, 66]
    adapter.close()


def test_receipt_failure_is_not_retryable_even_when_receipts_ok_boolean_is_true():
    pipe = Pipe(bad_receipt=True)
    adapter, request = backend(pipe), job()
    with pytest.raises(GW.ReceiptValidationError):
        adapter.execute(request, request.commit, request.check_stop)
    assert pipe.calls == 1 and adapter.stats()["receipt_failures"] == 1
    adapter.close()


@pytest.mark.parametrize("deadline", [False, True])
def test_blocked_receive_is_shutdown_on_cancel_or_deadline(deadline):
    pipe = Pipe(block=True)
    adapter, request = backend(pipe), job(timeout=0.08 if deadline else 3)
    errors = []
    def run():
        try:
            adapter.execute(request, request.commit, request.check_stop)
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=run); thread.start()
    assert pipe.started.wait(1)
    if not deadline:
        request.cancelled.set()
    thread.join(1)
    assert not thread.is_alive()
    assert isinstance(errors[0], JobExpired if deadline else JobCancelled)
    assert request.tokens == [65] and pipe.connects == 1
    adapter.close()


def test_legacy_coordinator_and_missing_assignments_refuse_production_backend():
    pipe = Pipe()
    with pytest.raises(ValueError, match="assignments"):
        GW.V4RingBackend("unused", "h:1", "t:2", {}, vp=pipe, tokenizer=Tokenizer())
    pipe.coordinate = lambda *args, **kw: None
    with pytest.raises(ValueError, match="capability"):
        backend(pipe)


@pytest.fixture
def live():
    pipe = Pipe(fail_first=True)
    adapter = backend(pipe)
    assert adapter.warmup()["proof_verified"]
    pipe.calls = pipe.connects = 0  # startup canary is not a tenant request
    pipe.started.clear()
    auth = GW.AuthRegistry({KEY_A: "a", KEY_B: "b"}, {"a": TenantLimits(), "b": TenantLimits()})
    gateway = GW.Gateway(adapter, auth, max_body=2048)
    server = gateway.server()
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    yield server.server_address, gateway, pipe
    server.shutdown(); gateway.shutdown(); server.server_close(); thread.join(1)


def request(address, method, path, body=None, *, key=KEY_A, headers=None):
    connection = http.client.HTTPConnection(*address, timeout=3)
    supplied = {"Authorization": "Bearer " + key} if key else {}
    supplied.update(headers or {})
    data = json.dumps(body) if body is not None else None
    if data is not None:
        supplied["Content-Type"] = "application/json"
    connection.request(method, path, data, supplied)
    response = connection.getresponse()
    status, content = response.status, response.read()
    connection.close()
    return status, content


def chat(**fields):
    return {"model": Pipe.V4_MODEL_ID, "messages": [{"role": "user", "content": "test text"}],
            "max_tokens": 4, **fields}


def test_http_auth_json_idempotency_and_tenant_job_isolation(live):
    address, gateway, pipe = live
    assert request(address, "GET", "/health", key=None)[0] == 200
    assert request(address, "GET", "/v1/models", key=None)[0] == 401
    status, content = request(address, "POST", "/v1/chat/completions", chat(), headers={"Idempotency-Key": "same-job"})
    assert status == 200
    result = json.loads(content)
    assert result["choices"][0]["message"]["content"] == "ABCD"
    assert result["usage"]["completion_tokens"] == 4 and result["shard"]["proof_verified"] is True
    calls = pipe.calls
    again = json.loads(request(address, "POST", "/v1/chat/completions", chat(), headers={"Idempotency-Key": "same-job"})[1])
    assert again["id"] == result["id"] and pipe.calls == calls
    assert request(address, "GET", "/v1/jobs/" + result["id"], key=KEY_B)[0] == 404
    assert request(address, "POST", "/v1/jobs/" + result["id"] + "/cancel", key=KEY_B)[0] == 404


def test_sse_only_emits_new_suffix_after_backend_transport_replay(live):
    address, _, _ = live
    status, content = request(address, "POST", "/v1/chat/completions", chat(stream=True))
    assert status == 200 and content.endswith(b"data: [DONE]\n\n")
    events = [json.loads(line[6:]) for line in content.decode().splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    text = "".join(event["choices"][0]["delta"].get("content", "") for event in events)
    assert text == "ABCD"


def test_sse_resume_cursor_and_conflicting_idempotency_are_explicit(live):
    address, _, _ = live
    header = {"Idempotency-Key": "resume-job"}
    first = json.loads(request(address, "POST", "/v1/chat/completions", chat(), headers=header)[1])
    status, content = request(address, "POST", "/v1/chat/completions", chat(stream=True),
        headers={**header, "Last-Event-ID": first["id"] + ":2"})
    assert status == 200
    events = [json.loads(line[6:]) for line in content.decode().splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert "".join(event["choices"][0]["delta"].get("content", "") for event in events) == "CD"
    assert request(address, "POST", "/v1/chat/completions", chat(max_tokens=3), headers=header)[0] == 409


def test_request_limits_and_unsupported_sampling_do_not_enter_ring(live):
    address, _, pipe = live
    assert request(address, "POST", "/v1/chat/completions", chat(temperature=0.5))[0] == 400
    assert request(address, "POST", "/v1/chat/completions", chat(tools=[{"x": "y"}]))[0] == 400
    assert request(address, "POST", "/v1/chat/completions", chat(messages=[{"role": "user", "content": "x" * 3000}]))[0] == 413
    assert pipe.calls == 0


def test_endpoints_reachable_do_not_establish_readiness_and_signed_warmup_does(monkeypatch):
    pipe = Pipe()
    adapter = backend(pipe)
    class Reachable:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(GW.socket, "create_connection", lambda *a, **kw: Reachable())
    assert adapter.ready() == (False, "signed_warmup_required")
    auth = GW.AuthRegistry({KEY_A: "a"}, {"a": TenantLimits()})
    with pytest.raises(ValueError, match="warmup"):
        GW.Gateway(adapter, auth)
    assert adapter.warmup(timeout_s=1)["proof_verified"] is True
    assert pipe.recorded[0][:2] == ([1, 2], 2)
    assert adapter.ready() == (True, "verified_established_ring")
    gateway = GW.Gateway(adapter, auth)
    assert gateway.jobs.metrics()["counts"]["submitted"] == 0
    gateway.shutdown()


def test_forged_warmup_receipt_cannot_start_service():
    adapter = backend(Pipe(bad_receipt=True))
    with pytest.raises(GW.ReceiptValidationError):
        adapter.warmup(timeout_s=1)
    auth = GW.AuthRegistry({KEY_A: "a"}, {"a": TenantLimits()})
    with pytest.raises(ValueError, match="warmup"):
        GW.Gateway(adapter, auth)
    adapter.close()


def test_idle_verified_ring_keeps_ready_after_60_seconds_without_new_requests():
    adapter = backend(Pipe())
    adapter.warmup(timeout_s=1)
    adapter._last_ok = time.monotonic() - 3600
    assert adapter.ready() == (True, "verified_established_ring")
    adapter.close()


def test_dead_established_socket_invalidates_old_signed_proof():
    adapter = backend(Pipe())
    adapter.warmup(timeout_s=1)
    adapter._channels[1].peer.close()
    assert adapter.ready() == (False, "established_ring_closed")
    assert adapter._last_ok == 0 and adapter._channels is None
    assert adapter.ready() == (False, "signed_warmup_required")


def test_idle_readiness_peek_is_nonblocking_and_never_consumes_pending_reply():
    adapter = backend(Pipe())
    adapter.warmup(timeout_s=1)
    channel = adapter._channels[1]
    channel.peer.sendall(b"pending-reply")
    timeout = channel.gettimeout()
    assert adapter.ready() == (True, "verified_established_ring")
    assert channel.gettimeout() == timeout
    assert channel.recv(13) == b"pending-reply"
    adapter.close()


def test_active_attempt_readiness_does_not_change_socket_mode_or_read_reply(monkeypatch):
    adapter = backend(Pipe())
    adapter.warmup(timeout_s=1)
    adapter._attempt_owner = object()
    for channel in adapter._channels:
        monkeypatch.setattr(channel, "recv", lambda *a: pytest.fail("readiness raced active reader"))
        monkeypatch.setattr(channel, "setblocking", lambda *a: pytest.fail("readiness changed active mode"))
    assert adapter.ready() == (True, "verified_ring_in_use")
    adapter.close()


def stream_harness(job, decoder, *, last=None):
    handler = GW.Handler.__new__(GW.Handler)
    handler.server = SimpleNamespace(gateway=SimpleNamespace(backend=SimpleNamespace(
        model_id="test", decode=decoder), write_timeout=1))
    handler.headers = {"Last-Event-ID": last} if last else {}
    handler.connection = SimpleNamespace(settimeout=lambda value: None)
    handler.wfile = io.BytesIO()
    handler.send_response = handler.send_header = lambda *a: None
    handler.end_headers = lambda: None
    handler._client_gone = lambda: False
    handler._stream(job)
    return handler.wfile.getvalue().decode()


def final_result(tokens):
    return {"tokens": tokens, "finish_reason": "length", "proof": {"scope": "test"}, "recovery": {}}


def test_atomic_stream_snapshot_cannot_drop_last_token_when_worker_finishes_during_decode():
    request = job(maximum=2)
    request.tokens = [65]
    def decode(tokens):
        if tokens == [65] and request.state == "running":
            with request.changed:
                request.tokens.append(66)
                request.result = final_result(request.tokens)
                request.state = "completed"
        return Tokenizer().decode(tokens)
    output = stream_harness(request, decode)
    events = [json.loads(line[6:]) for line in output.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
    assert "".join(event["choices"][0]["delta"].get("content", "") for event in events) == "AB"
    assert "[DONE]" in output


def test_character_cursor_prevents_repeated_terminal_replacement_character():
    request = job(maximum=1)
    request.tokens = [65533]
    request.result = final_result(request.tokens); request.state = "completed"
    first = stream_harness(request, Tokenizer().decode)
    content_id = [line[4:] for line in first.splitlines() if line.startswith("id: ")][-1]
    resumed = stream_harness(request, Tokenizer().decode, last=content_id)
    events = [json.loads(line[6:]) for line in resumed.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
    assert "".join(event["choices"][0]["delta"].get("content", "") for event in events) == ""


def test_http_expired_job_and_raw_receipt_endpoint_are_tenant_scoped(live):
    address, gateway, pipe = live
    completed = json.loads(request(address, "POST", "/v1/chat/completions", chat())[1])
    status, proof = request(address, "GET", "/v1/jobs/" + completed["id"] + "/receipts")
    assert status == 200 and json.loads(proof)["verified"] is True
    assert request(address, "GET", "/v1/jobs/" + completed["id"] + "/receipts", key=KEY_B)[0] == 404
    pipe.block = True
    assert request(address, "POST", "/v1/chat/completions", chat(timeout_s=0.08))[0] == 408
    assert gateway.jobs.metrics()["counts"]["expired"] == 1


def test_http_cancel_interrupts_an_actual_blocked_backend_and_ends_sse(live):
    address, gateway, pipe = live
    pipe.block = True; pipe.started.clear()
    connection = http.client.HTTPConnection(*address, timeout=3)
    connection.request("POST", "/v1/chat/completions", json.dumps(chat(stream=True)),
        {"Authorization": "Bearer " + KEY_A, "Content-Type": "application/json"})
    response = connection.getresponse()
    assert response.status == 200
    first = response.readline().decode().strip()
    assert first.startswith("id: ")
    job_id = first[4:].split(":")[0]
    assert pipe.started.wait(1)
    assert request(address, "POST", "/v1/jobs/" + job_id + "/cancel")[0] == 200
    content = response.read().decode()
    connection.close()
    assert "job_cancelled" in content and "[DONE]" in content
    assert gateway.jobs.get("a", job_id).state == "cancelled"
    assert gateway.jobs.get("a", job_id).tokens == [65]


def test_ambiguous_legacy_replacement_cursor_is_rejected_instead_of_repeated():
    request_job = job(maximum=1)
    request_job.tokens = [65533]
    request_job.result = final_result(request_job.tokens); request_job.state = "completed"
    handler = GW.Handler.__new__(GW.Handler)
    handler.server = SimpleNamespace(gateway=SimpleNamespace(backend=SimpleNamespace(decode=Tokenizer().decode)))
    handler.headers = {"Last-Event-ID": request_job.id + ":1"}
    errors = []
    handler._error = lambda message, status, code: errors.append((status, code))
    handler._stream(request_job)
    assert errors == [(409, "invalid_resume_cursor")]


def test_cli_requires_deployment_before_loading_model_or_starting_server():
    with pytest.raises(SystemExit) as exc:
        GW.main(["--dir", "unused", "--auth-file", "unused"])
    assert exc.value.code == 2


def test_flat_gateway_help_has_no_repo_or_model_import_dependency(tmp_path):
    shutil.copy2(GW.__file__, tmp_path / "v4_gateway.py")
    shutil.copy2(GW.ROOT / "shard" / "service_queue.py", tmp_path / "service_queue.py")
    env = dict(os.environ); env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(tmp_path / "v4_gateway.py"), "--help"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0 and "--deployment" in result.stdout, result.stderr


def test_deployment_identity_privacy_and_coordinator_environment_cannot_be_downgraded():
    bundle = {"model_id": "test", "trust_mode": "trusted_nodes",
        "coord_env": {"V4_WIRE_FUSED": "0"},
        "stages": [{"signer_pubkey": "a", "lo": 0, "hi": 2, "env": {}}]}
    checker = lambda value: {"ready": True}
    assert GW.validate_deployment_bundle(bundle, env={"V4_WIRE_FUSED": "0"}, checker=checker) == {"a": (0, 2)}
    with pytest.raises(ValueError, match="not ready"):
        GW.validate_deployment_bundle(bundle, checker=lambda value: {"ready": False, "errors": ["capacity"]})
    with pytest.raises(ValueError, match="assignments"):
        GW.validate_deployment_bundle(bundle, assignments={"b": [0, 2]}, env={"V4_WIRE_FUSED": "0"}, checker=checker)
    with pytest.raises(ValueError, match="environment"):
        GW.validate_deployment_bundle(bundle, env={"V4_WIRE_FUSED": "1"}, checker=checker)
    bundle["trust_mode"] = "sealed_ids"
    bundle["stages"][0]["env"] = {"V4_SEALED_IDS": "1", "V4_TOKEN_PRIVACY_KEY_ID": "epoch-a"}
    with pytest.raises(ValueError, match="privacy mode"):
        GW.validate_deployment_bundle(bundle, env={"V4_WIRE_FUSED": "0"}, checker=checker)
    with pytest.raises(ValueError, match="key identity"):
        GW.validate_deployment_bundle(bundle, env={"V4_WIRE_FUSED": "0", "V4_SEALED_IDS": "1", "V4_TOKEN_PRIVACY_KEY_ID": "wrong"}, checker=checker)

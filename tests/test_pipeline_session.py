"""Real sockets/Ed25519 HELLO, remote ownership, fencing and independent probes."""
import copy
import socket
import threading
import time
import struct

import pytest

from shard.pipeline_plan import build_plan
from shard.pipeline_session import (SessionConfig, SessionAcceptor, SessionBusy, ProtocolError,
    hello_client, send_message, recv_message, prepare_message, _handshake_recv)
from shard.receipt import gen_key, pub_b64
from shard.transport import send_msg, recv_msg


@pytest.fixture
def network():
    keys = [gen_key(), gen_key(), gen_key()]
    listeners = [socket.socket(), socket.socket()]
    for listener in listeners:
        listener.bind(("127.0.0.1", 0)); listener.listen(32)
    addresses = [f"127.0.0.1:{listener.getsockname()[1]}" for listener in listeners]
    plan = build_plan({"num_hidden_layers": 4}, ring_id="test-ring", cohort_id="a" * 64,
                      endpoints=addresses)
    for index in range(2):
        plan["stages"][index]["signer_pubkey"] = pub_b64(keys[index])
    plan["coordinator"].update(signer_pubkey=pub_b64(keys[2]), node_id="controller")
    configs = [SessionConfig.from_plan(plan, index, caller_key=keys[index], ttl_s=2) for index in range(2)]
    coordinator = SessionConfig.from_plan(plan, -1, caller_key=keys[2], ttl_s=2)
    acceptors = [SessionAcceptor(listener, config, send_msg, recv_msg, key=keys[index], timeout=2)
                 for index, (listener, config) in enumerate(zip(listeners, configs))]
    clients = []
    def dial(index, *, purpose="drive", config=None, grant=None, session_id="session", tap=None):
        raw = socket.create_connection(listeners[index].getsockname(), timeout=2)
        clients.append(raw)
        def send(channel, payload):
            if tap is not None:
                tap.append(copy.deepcopy(payload))
            return send_msg(channel, payload)
        wrapped = hello_client(raw, config or coordinator, index, purpose, send, recv_msg,
                               grant=grant, session_id=session_id)
        clients.append(wrapped)
        return wrapped
    try:
        yield plan, configs, coordinator, acceptors, dial, keys
    finally:
        for client in clients:
            client.close()
        for acceptor in acceptors:
            acceptor.close()


def test_authenticated_role_handshake_and_busy_are_independent_of_the_model_loop(network):
    plan, configs, coordinator, acceptors, dial, keys = network
    first = dial(0)
    assert first.grant["session_id"] == "session"
    started = time.monotonic()
    with pytest.raises(SessionBusy):
        dial(0, session_id="other")
    assert time.monotonic() - started < 1
    assert acceptors[0].registry.grant["owner"] == first.grant["owner"]


@pytest.mark.parametrize("mutation", [
    lambda plan: plan.update(cohort_id="b" * 64),
    lambda plan: plan.update(ring_id="wrong-ring"),
    lambda plan: plan["stages"][0].update(node_id="wrong-head"),
])
def test_wrong_role_plan_or_cohort_fails_before_session_or_model_work(network, mutation):
    plan, configs, coordinator, acceptors, dial, keys = network
    wrong = copy.deepcopy(plan); mutation(wrong)
    with pytest.raises(ProtocolError):
        dial(0, config=SessionConfig.from_plan(wrong, -1, caller_key=keys[2]))
    assert acceptors[0].registry.grant is None


def test_a_dialer_knowing_the_plan_cannot_impersonate_the_controller(network):
    plan, configs, coordinator, acceptors, dial, keys = network
    spoof = SessionConfig.from_plan(plan, -1, caller_key=gen_key())
    with pytest.raises(ProtocolError):
        dial(0, config=spoof)
    assert acceptors[0].registry.grant is None


def test_forward_role_signature_is_pinned_and_wrong_stage_is_rejected(network):
    plan, configs, coordinator, acceptors, dial, keys = network
    with pytest.raises(ProtocolError):
        dial(1, purpose="forward", config=coordinator)
    forward = dial(1, purpose="forward", config=configs[0])
    accepted = acceptors[1].get("forward", timeout=1)
    assert forward.grant is None and accepted.grant is None


def test_tail_return_and_forward_envelopes_bind_to_the_same_signed_head_owner(network):
    plan, configs, coordinator, acceptors, dial, keys = network
    head = dial(0); head_server = acceptors[0].get("drive", timeout=1)
    ret = dial(1, purpose="return", grant=head.grant); ret_server = acceptors[1].get("return", timeout=1)
    forward = dial(1, purpose="forward", config=configs[0]); pred = acceptors[1].get("forward", timeout=1)
    send_message(send_msg, head, {"op": "reset"})
    assert recv_message(recv_msg, head_server) == {"op": "reset"}
    send_message(send_msg, forward, {"op": "reset"})
    assert recv_message(recv_msg, pred) == {"op": "reset"}
    send_message(send_msg, ret_server, [42])
    assert recv_message(recv_msg, ret) == [42]
    assert ret.grant["owner"] == head.grant["owner"]


def test_old_coordinator_return_is_fenced_after_a_new_owner_even_if_old_grant_unexpired(network):
    plan, configs, coordinator, acceptors, dial, keys = network
    first = dial(0); old = dict(first.grant)
    ret = dial(1, purpose="return", grant=old)
    acceptors[0].get("drive", timeout=1).close()
    second = dial(0, session_id="second")
    assert second.grant["fence"] > old["fence"]
    new_ret = dial(1, purpose="return", grant=second.grant)
    with pytest.raises(ProtocolError):
        dial(1, purpose="return", grant=old)
    assert acceptors[1].last_return_grant["owner"] == new_ret.grant["owner"]


def test_async_frame_envelope_keeps_its_original_owner_after_reset(network):
    plan, configs, coordinator, acceptors, dial, keys = network
    first = dial(0); server = acceptors[0].get("drive", timeout=1)
    forward = dial(1, purpose="forward", config=configs[0])
    send_message(send_msg, first, {"op": "reset"}); recv_message(recv_msg, server)
    prepared = prepare_message(forward, {"op": "verify", "start": 1})
    old_owner = prepared.envelope["grant"]["owner"]
    server.close(); second = dial(0, session_id="second")
    assert prepared.envelope["grant"]["owner"] == old_owner != second.grant["owner"]


def test_idle_ping_renews_ownership_without_consuming_tail_job_replies(network):
    plan, configs, coordinator, acceptors, dial, keys = network
    head = dial(0); server = acceptors[0].get("drive", timeout=1)
    done = threading.Event()
    def model_loop():
        assert recv_message(recv_msg, server) == {"op": "stop"}
        done.set()
    worker = threading.Thread(target=model_loop); worker.start()
    old_expiry = head.grant["expires_at"]
    send_message(send_msg, head, {"op": "session_ping"})
    pong = recv_message(recv_msg, head)
    assert pong["op"] == "session_pong" and head.grant["expires_at"] >= old_expiry
    send_message(send_msg, head, {"op": "stop"})
    assert done.wait(1); worker.join(1)


def test_signed_probe_works_while_the_head_session_is_busy_and_does_not_replace_it(network):
    plan, configs, coordinator, acceptors, dial, keys = network
    head = dial(0)
    probe = dial(0, purpose="probe")
    for seq in range(3):
        ping = {"op": "probe_ping", "seq": seq, "nonce": str(seq) * 64}
        send_message(send_msg, probe, ping)
        assert recv_message(recv_msg, probe) == {**ping, "op": "probe_pong"}
    assert acceptors[0].registry.grant["owner"] == head.grant["owner"]


def test_replayed_signed_hello_proof_cannot_acquire_a_fresh_challenge(network):
    plan, configs, coordinator, acceptors, dial, keys = network
    transcript = []
    first = dial(0, tap=transcript)
    acceptors[0].get("drive", timeout=1).close()
    raw = socket.create_connection(acceptors[0].srv.getsockname(), timeout=2)
    try:
        send_msg(raw, transcript[0])
        challenge = _handshake_recv(recv_msg, raw)
        assert challenge["op"] == "hello_challenge"
        send_msg(raw, transcript[1])  # valid signature, but on the OLD server nonce
        rejected = _handshake_recv(recv_msg, raw)
        assert rejected["status"] == "error" and acceptors[0].registry.grant is None
    finally:
        raw.close()


def test_oversized_initial_hello_is_rejected_before_generic_model_codec_allocation(network):
    plan, configs, coordinator, acceptors, dial, keys = network
    raw = socket.create_connection(acceptors[0].srv.getsockname(), timeout=2)
    try:
        raw.sendall(struct.pack("!Q", 10**9))
        rejected = _handshake_recv(recv_msg, raw)
        assert rejected["status"] == "error" and "64 KiB" in rejected["error"]
        assert acceptors[0].registry.grant is None
    finally:
        raw.close()


def test_many_signed_head_epochs_keep_bounded_history_and_reject_evicted_old_boot(network):
    plan, configs, coordinator, acceptors, dial, keys = network
    channel = dial(0); first = dict(channel.grant)
    for index in range(400):
        grant = acceptors[0].registry._sign({**first, "boot_id": "epoch-" + str(index),
            "boot_started_at": first["boot_started_at"] + (index + 1) * .001,
            "expires_at": time.time() + 60})
        channel.adopt(grant)
        acceptors[1].check_forward_fence(grant)
    assert len(channel.retired_boots) == 256
    assert len(acceptors[1].forward_retired_boots) == 256
    assert first["boot_id"] not in channel.retired_boots
    with pytest.raises(ProtocolError, match="stale"):
        channel.adopt(first)
    with pytest.raises(ProtocolError, match="stale"):
        acceptors[1].check_forward_fence(first)

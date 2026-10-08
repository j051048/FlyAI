"""Real strict tail serve-loop churn on CPU; no model/GPU performance evidence."""
from contextlib import suppress
import importlib
import socket
import threading
import time

import pytest

from shard.pipeline_plan import build_plan
from shard.pipeline_session import SessionAcceptor, SessionConfig, hello_client, send_message, recv_message
from shard.receipt import gen_key, pub_b64
from shard.transport import send_msg, recv_msg


class StopServe(BaseException):
    """Fixture-only stop, outside the production loop's recoverable errors."""


class StoppingAcceptor(SessionAcceptor):
    def get(self, purpose, timeout=None):
        try:
            return super().get(purpose, timeout)
        except ConnectionError:
            if self.stop.is_set():
                raise StopServe
            raise


class Harness:
    def __init__(self, monkeypatch):
        torch = importlib.import_module("torch")
        self.sp = importlib.import_module("specpipe")
        self.channels, self.errors = [], []
        self.head_server = None
        self.owner_number = 0
        self.block_next = False
        self.in_compute, self.release_compute = threading.Event(), threading.Event()
        self.calls = []
        self.keys = [gen_key(), gen_key(), gen_key()]
        self.listeners = [socket.socket(), socket.socket()]
        for listener in self.listeners:
            listener.bind(("127.0.0.1", 0)); listener.listen(32)
        self.plan = build_plan({"n_layers": 4}, ring_id="tail-churn", cohort_id="a"*64,
            endpoints=[f"127.0.0.1:{s.getsockname()[1]}" for s in self.listeners])
        for index in range(2):
            self.plan["stages"][index]["signer_pubkey"] = pub_b64(self.keys[index])
        self.plan["coordinator"].update(signer_pubkey=pub_b64(self.keys[2]), node_id="controller")
        self.configs = [SessionConfig.from_plan(self.plan, i, caller_key=self.keys[i], ttl_s=30) for i in range(2)]
        self.coordinator = SessionConfig.from_plan(self.plan, -1, caller_key=self.keys[2], ttl_s=30)
        self.acceptors = [StoppingAcceptor(s, cfg, send_msg, recv_msg, key=self.keys[i], timeout=2)
            for i, (s, cfg) in enumerate(zip(self.listeners, self.configs))]
        harness = self
        class FakeFastVerify:
            def __init__(self, parts, maxlen, dev):
                self.dev = "cpu"
            def reset(self):
                harness.calls.append(("reset", None))
            def prefill(self, x, start):
                harness.calls.append(("prefill", start))
                if harness.block_next:
                    harness.block_next = False
                    harness.in_compute.set()
                    assert harness.release_compute.wait(5), "fixture compute barrier was not released"
                return x
            def decode(self, x, start):
                harness.calls.append(("decode", start))
                return x
        monkeypatch.setattr(self.sp, "FastVerify", FakeFastVerify)
        monkeypatch.setattr(self.sp, "RECEIPTS", False)
        monkeypatch.setattr(self.sp, "_raw_send_msg", send_msg)
        monkeypatch.setattr(self.sp, "_raw_recv_msg", recv_msg)
        monkeypatch.setattr(self.sp, "_server", lambda *args: (self.listeners[1], self.acceptors[1]))
        self.parts = {"lo": 2, "hi": 4, "norm": torch.nn.Identity(), "lm_head": torch.nn.Identity(),
            "_session_config": self.configs[1], "_session_key": self.keys[1]}
        def serve():
            try:
                self.sp.serve_tail_fast(self.parts, self.listeners[1].getsockname()[1], 2, "cpu", max_ctx=64)
            except StopServe:
                pass
            except BaseException as exc:
                self.errors.append(exc)
        self.worker = threading.Thread(target=serve, daemon=True)
        self.worker.start()
        self.new_owner()

    def dial(self, purpose, *, index=1):
        raw = socket.create_connection(self.listeners[index].getsockname(), timeout=2)
        self.channels.append(raw)
        cfg = self.configs[0] if purpose == "forward" else self.coordinator
        wrapped = hello_client(raw, cfg, index, purpose, send_msg, recv_msg,
            session_id=f"owner-{self.owner_number}", grant=self.head.grant if purpose == "return" else None)
        self.channels.append(wrapped)
        return wrapped

    def new_owner(self):
        if self.head_server is not None:
            self.head_server.close()
            self.head.close()
        self.owner_number += 1
        self.head = self.dial("drive", index=0)
        self.head_server = self.acceptors[0].get("drive", timeout=1)
        # A real authenticated head receive establishes the thread-local grant
        # that the ordinary forwarding API captures in its immutable envelope.
        send_message(send_msg, self.head, {"op": "reset"})
        assert recv_message(recv_msg, self.head_server) == {"op": "reset"}

    def reset(self, predecessor, returned):
        send_message(send_msg, predecessor, {"op": "reset", "job_id": f"job-{self.owner_number}"})
        assert recv_message(recv_msg, returned) == "ok"

    def verify(self, predecessor, returned, *, token=4, start=0):
        torch = importlib.import_module("torch")
        x = torch.zeros(1, 1, 16); x[..., token] = 1
        send_message(send_msg, predecessor, {"op": "verify", "h": x, "start": start, "prefill": True})
        assert recv_message(recv_msg, returned) == [token]

    def close(self):
        self.release_compute.set()
        for acceptor in self.acceptors:
            acceptor.stop.set()
        for channel in self.channels:
            with suppress(OSError):channel.shutdown(socket.SHUT_RDWR)
            with suppress(OSError):channel.close()
        for acceptor in self.acceptors:
            acceptor.close()
        self.worker.join(3)
        assert not self.worker.is_alive(), "CPU serve thread outlived its fixture"
        assert not self.errors, self.errors


@pytest.fixture
def tail(monkeypatch):
    harness = Harness(monkeypatch)
    try:
        yield harness
    finally:
        harness.close()


def test_same_grant_return_reconnect_delivers_reset_ack_and_runs_the_model(tail):
    predecessor, old_return = tail.dial("forward"), tail.dial("return")
    tail.reset(predecessor, old_return)
    tail.verify(predecessor, old_return, token=4)
    returned = tail.dial("return")
    tail.reset(predecessor, returned)
    tail.verify(predecessor, returned, token=5)
    assert tail.calls.count(("prefill", 0)) == 2


def test_old_predecessor_eof_does_not_close_the_new_owner_return_or_publish_old_result(tail):
    predecessor, old_return = tail.dial("forward"), tail.dial("return")
    tail.reset(predecessor, old_return)
    tail.block_next = True
    torch = importlib.import_module("torch")
    x = torch.zeros(1, 1, 16); x[..., 3] = 1
    send_message(send_msg, predecessor, {"op": "verify", "h": x, "start": 0, "prefill": True})
    assert tail.in_compute.wait(2)
    tail.new_owner()
    returned = tail.dial("return")
    predecessor.close()
    next_predecessor = tail.dial("forward")
    tail.release_compute.set()
    # Any old [3] reply here is a cross-owner leak; a closed socket is a churn
    # failure. The new owner's first response must be this reset acknowledgement.
    tail.reset(next_predecessor, returned)
    tail.verify(next_predecessor, returned, token=6)


def test_forward_peer_eof_can_reconnect_while_tail_is_waiting_for_a_return_channel(tail):
    predecessor = tail.dial("forward")
    # Let the real serve loop adopt the old forward, then block waiting for its
    # return. No model receive can consume the forward EOF in this state.
    deadline = time.monotonic() + 1
    while not tail.acceptors[1].queues["forward"].empty() and time.monotonic() < deadline:
        time.sleep(.01)
    predecessor.close()
    next_predecessor = tail.dial("forward")
    returned = tail.dial("return")
    tail.reset(next_predecessor, returned)
    tail.verify(next_predecessor, returned, token=7)


def test_single_stage_head_tail_runs_actual_coordinator_and_complete_signed_receipt(monkeypatch):
    torch, sp = importlib.import_module("torch"), importlib.import_module("specpipe")
    node_key, coordinator_key = gen_key(), gen_key()
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(32)
    address = f"127.0.0.1:{srv.getsockname()[1]}"
    plan = build_plan({"n_layers": 4}, ring_id="one-stage", cohort_id="b"*64, endpoints=[address])
    plan["stages"][0]["signer_pubkey"] = pub_b64(node_key)
    plan["coordinator"].update(signer_pubkey=pub_b64(coordinator_key), node_id="controller")
    cfg = SessionConfig.from_plan(plan, 0, caller_key=node_key, ttl_s=30)
    coordinator = SessionConfig.from_plan(plan, -1, caller_key=coordinator_key, ttl_s=30)
    acceptor = StoppingAcceptor(srv, cfg, send_msg, recv_msg, key=node_key, timeout=2)
    class IdentityVerify:
        def __init__(self, parts, maxlen, dev):
            self.dev = "cpu"
        def reset(self):
            pass
        def prefill(self, x, start):
            return x
        decode = prefill
    parts = {"lo": 0, "hi": 4, "embed": lambda ids: torch.nn.functional.one_hot((ids + 1) % 32, 32).float(),
        "norm": torch.nn.Identity(), "lm_head": torch.nn.Identity(), "_session_config": cfg, "_session_key": node_key}
    monkeypatch.setattr(sp, "FastVerify", IdentityVerify)
    monkeypatch.setattr(sp, "RECEIPTS", True)
    monkeypatch.setattr(sp, "load_or_make_node_key", lambda _path: node_key)
    monkeypatch.setattr(sp, "_raw_send_msg", send_msg)
    monkeypatch.setattr(sp, "_raw_recv_msg", recv_msg)
    monkeypatch.setattr(sp, "_server", lambda *args: (srv, acceptor))
    errors = []
    def serve():
        try:
            sp.serve_tail_fast(parts, srv.getsockname()[1], 2, "cpu", max_ctx=64)
        except StopServe:
            pass
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=serve, daemon=True); worker.start()
    channels = []
    class Tokenizer:
        eos_token_id = 999
        def decode(self, ids, **kwargs):
            return ",".join(map(str, ids))
    try:
        channels = list(sp.connect_ring(address, address, session_config=coordinator, timeout=2))
        result = sp.coordinate_pipe(None, channels[0], Tokenizer(), "", 0, 2, 2, 1,
            ret_sock=channels[1], prompt_ids=[1, 2], max_ctx=64,
            swarm_id="one-stage", job_id="signed-cpu-lifecycle", nonce="c"*64,
            expected_by_signer={pub_b64(node_key): (0, 4)}, strict_job_binding=True)
        assert result["output_ids"] == [3, 4]
        assert result["proof_verified"] is True and result["receipts_ok"] is True
        assert len(result["receipts"]) == 1 and result["receipts"][0]["n_chunks"] == 2
        assert result["receipts"][0]["job_id"] == "signed-cpu-lifecycle"
        assert result["metrics"]["new_decode_tokens"] == 1
    finally:
        acceptor.stop.set()
        for channel in channels:
            with suppress(OSError):channel.shutdown(socket.SHUT_RDWR)
            with suppress(OSError):channel.close()
        acceptor.close(); worker.join(3)
        assert not worker.is_alive() and not errors

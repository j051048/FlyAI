"""Actual legacy TCP reset/verify paths with CPU math, never GPU acceptance.

The old five-second classifier deadline is compressed to 50ms by the fixture;
the 30-second inference edge timeout is left intact. This deterministically
models a stage loading/connecting after the tail has accepted an idle forward.
"""
from contextlib import suppress
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import torch

import specpipe as SP


class StopServe(BaseException):
    pass


class Tokenizer:
    eos_token_id = 127
    def apply_chat_template(self, *args, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]])}
    def decode(self, ids, **kwargs):
        return ",".join(map(str, ids))


class SuccessorEmbedding(torch.nn.Module):
    def forward(self, ids):
        return torch.nn.functional.one_hot((ids + 1) % 128, 128).to(torch.float32)


class Draft:
    def request(self, ids, k):
        self.pending = self.propose(ids, k)
    def fetch(self):
        return self.pending
    def propose(self, ids, k):
        return list(range(ids[-1] + 1, ids[-1] + k + 1))
    def note_accepted(self, count):
        pass


class Accepted:
    def __init__(self, raw, harness):
        self.raw, self.harness = raw, harness
    def __getattr__(self, name):
        return getattr(self.raw, name)
    def settimeout(self, timeout):
        # Compress ONLY the regressed first-frame deadline. Ordinary inference
        # sockets retain the actual, much longer timeout supplied by the stage.
        self.raw.settimeout(.05 if timeout == 5 else timeout)


class Listener:
    def __init__(self, raw, harness):
        self.raw, self.harness = raw, harness
        raw.settimeout(.05)
    def __getattr__(self, name):
        return getattr(self.raw, name)
    def accept(self):
        while not self.harness.stopping.is_set():
            try:
                raw, address = self.raw.accept()
                self.harness.channels.append(raw)
                self.harness.accepted.set()
                return Accepted(raw, self.harness), address
            except socket.timeout:
                continue
            except OSError:
                if self.harness.stopping.is_set():
                    raise StopServe
                raise
        raise StopServe


class Ring:
    def __init__(self, monkeypatch, *, direct, fast=True):
        self.direct, self.fast = direct, fast
        self.stopping, self.accepted = threading.Event(), threading.Event()
        self.channels, self.workers, self.errors, self.calls = [], [], [], []
        self.servers = []
        self.ready = [threading.Event() for _ in range(3)]
        self.listeners = {}
        for index in range(3):
            raw = socket.socket()
            raw.bind(("127.0.0.1", 0)); raw.listen(32)
            self.listeners[index] = Listener(raw, self)
            self.servers.append(raw)
        self.endpoints = [f"127.0.0.1:{self.listeners[i].getsockname()[1]}" for i in range(3)]
        harness = self
        class FastVerify:
            def __init__(self, parts, maxlen, dev):
                self.index = parts["lo"]
            def reset(self):
                harness.calls.append((self.index, "reset"))
            def prefill(self, x, start):
                harness.calls.append((self.index, "prefill")); return x
            def decode(self, x, start):
                harness.calls.append((self.index, "decode")); return x
        monkeypatch.setattr(SP, "FastVerify", FastVerify)
        monkeypatch.setattr(SP, "RECEIPTS", False)
        monkeypatch.setattr(SP, "_run_block", lambda x, *args, **kwargs: x)
        monkeypatch.setattr(SP.wire, "_KEY", SP.wire._KEY)
        SP.wire.use_key("legacy-reset-regression-only")
        # Preserve the real wire codec, session passthrough, forward dialer,
        # async sender, and coordinator functions. Only model math is fake.
        def server(parts, port, timeout):
            index = parts["lo"]
            self.ready[index].set()
            return self.listeners[index], None
        monkeypatch.setattr(SP, "_server", server)
        original_forward = SP._forward_connect
        def forward(*args, **kwargs):
            if self.stopping.is_set():
                raise StopServe
            channel = original_forward(*args, **kwargs)
            self.channels.append(channel)
            return channel
        monkeypatch.setattr(SP, "_forward_connect", forward)

    def start(self, index):
        parts = {"lo": index, "hi": index + 1, "embed": SuccessorEmbedding(),
                 "norm": torch.nn.Identity(), "lm_head": torch.nn.Identity(),
                 "_session_config": None, "_lease_guard": None}
        def run():
            try:
                port = self.listeners[index].getsockname()[1]
                nxt = self.endpoints[index + 1] if index < 2 else None
                if index == 2 and self.direct:
                    fn = SP.serve_tail_fast if self.fast else SP.serve_tail_direct
                    fn(parts, port, 30, "cpu")
                else:
                    fn = SP.serve_spec_fast if self.fast else SP.serve_spec
                    fn(parts, index, 3, port, nxt, 30, "cpu", direct=self.direct)
            except StopServe:
                pass
            except BaseException as error:
                self.errors.append(error)
        worker = threading.Thread(target=run, daemon=True)
        self.workers.append(worker); worker.start()
        assert self.ready[index].wait(2), f"stage {index} listener did not start"

    def connect(self):
        pipe, returned = SP.connect_ring(self.endpoints[0], self.endpoints[2] if self.direct else None, timeout=2)
        self.channels.extend(c for c in (pipe, returned) if c is not None)
        return pipe, returned

    def close(self):
        self.stopping.set()
        for channel in self.channels + self.servers:
            with suppress(OSError): channel.shutdown(socket.SHUT_RDWR)
            with suppress(OSError): channel.close()
        for worker in self.workers:
            worker.join(3)
        assert not any(w.is_alive() for w in self.workers), "legacy fixture left a serving worker"
        assert not self.errors, self.errors


@pytest.mark.parametrize("mode,direct", [("sync", False), ("sync", True), ("local_sync", True), ("pipe", True)])
def test_fast_legacy_three_stage_reset_prefill_and_decode(monkeypatch, mode, direct):
    ring = Ring(monkeypatch, direct=direct)
    draft_worker = None
    try:
        ring.start(2); ring.start(1); ring.start(0)
        pipe, returned = ring.connect()
        if mode == "pipe":
            result = SP.coordinate_pipe(None, pipe, Tokenizer(), "prompt", 2, 8, 2, 3,
                ret_sock=returned, local_draft=Draft())
        elif mode == "local_sync":
            # Non-pipe --ngram-draft has no draft socket. It must use the
            # declared local proposer, not attempt wire I/O through None.
            result = SP.coordinate(None, pipe, Tokenizer(), "prompt", 2, 8, 2,
                ret_sock=returned, local_draft=Draft())
        else:
            client, server = socket.socketpair(); ring.channels.extend((client, server))
            def draft_server():
                try:
                    while not ring.stopping.is_set():
                        request = SP.recv_msg(server)
                        SP.send_msg(server, Draft().propose(request["ids"], request["k"]))
                except SP.EDGE_ERRORS:
                    pass
            draft_worker = threading.Thread(target=draft_server, daemon=True); draft_worker.start()
            result = SP.coordinate(client, pipe, Tokenizer(), "prompt", 0, 8, 2, ret_sock=returned)
        assert result["output_ids"] == list(range(3, 11))
        for index in range(3):
            assert (index, "reset") in ring.calls and (index, "prefill") in ring.calls
    finally:
        ring.close()
        if draft_worker:
            draft_worker.join(2); assert not draft_worker.is_alive()


def test_eager_legacy_direct_tail_does_not_confuse_a_ready_reset_with_return_hello(monkeypatch):
    ring = Ring(monkeypatch, direct=True, fast=False)
    try:
        ring.start(2)
        predecessor = socket.create_connection(SP._address(ring.endpoints[2]), timeout=2)
        ring.channels.append(predecessor)
        SP.send_msg(predecessor, {"op": "reset"})
        # Both channels have bytes queued before tail role classification.
        # A select-ready socket is not necessarily the return channel.
        returned = socket.create_connection(SP._address(ring.endpoints[2]), timeout=2)
        ring.channels.append(returned)
        SP.send_msg(returned, {"op": "hello_return"})
        assert SP.recv_msg(returned) == "ok"
        h = torch.nn.functional.one_hot(torch.tensor([[9]]), 128).float()
        SP.send_msg(predecessor, {"op": "verify", "h": h, "start": 0})
        assert SP.recv_msg(returned) == [9]
    finally:
        ring.close()


@pytest.mark.parametrize("mode,direct", [("sync", False), ("sync", True), ("pipe", False), ("pipe", True)])
def test_reset_failure_names_actual_edge_peer_op_and_receive_phase(monkeypatch, mode, direct):
    monkeypatch.setattr(SP.wire, "_KEY", SP.wire._KEY)
    SP.wire.use_key("legacy-reset-diagnostic-only")
    head = socket.socket(); head.bind(("127.0.0.1", 0)); head.listen(1)
    tail = socket.socket(); tail.bind(("127.0.0.1", 0)); tail.listen(1)
    peers, errors = [], []
    def fail_head():
        try:
            peer, _ = head.accept(); peers.append(peer)
            message = SP.recv_msg(peer)
            assert message["op"] == "reset"
            peer.close()
        except BaseException as error:
            errors.append(error)
    def fail_return():
        try:
            peer, _ = tail.accept(); peers.append(peer)
            assert SP.recv_msg(peer) == {"op": "hello_return"}
            peer.close()
        except BaseException as error:
            errors.append(error)
    workers = [threading.Thread(target=fail_head, daemon=True)]
    if direct:
        workers.append(threading.Thread(target=fail_return, daemon=True))
    for worker in workers:
        worker.start()
    pipe = returned = None
    try:
        endpoint = lambda srv: f"127.0.0.1:{srv.getsockname()[1]}"
        pipe, returned = SP.connect_ring(endpoint(head), endpoint(tail) if direct else None, timeout=2)
        expected_peer = endpoint(tail if direct else head)
        with pytest.raises(SP.TransportError) as failure:
            if mode == "sync":
                SP.coordinate(None, pipe, Tokenizer(), "private prompt", 0, 2, 2, ret_sock=returned)
            else:
                SP.coordinate_pipe(None, pipe, Tokenizer(), "private prompt", 0, 2, 2, 1, ret_sock=returned)
        diagnostic = str(failure.value)
        assert f"edge={'tail-return' if direct else 'head(s0)'}" in diagnostic
        assert f"peer={expected_peer}" in diagnostic
        assert "op=reset" in diagnostic and "phase=recv" in diagnostic and "token 0" in diagnostic
        assert "private prompt" not in diagnostic and "token_ids" not in diagnostic
    finally:
        for peer in peers + [pipe, returned, head, tail]:
            if peer is not None:
                with suppress(OSError): peer.shutdown(socket.SHUT_RDWR)
                with suppress(OSError): peer.close()
        for worker in workers:
            worker.join(2); assert not worker.is_alive()
        assert not errors, errors


def test_legacy_direct_tail_keeps_idle_predecessor_until_stage_edge_deadline(monkeypatch):
    ring = Ring(monkeypatch, direct=True)
    try:
        ring.start(2); ring.start(1)
        assert ring.accepted.wait(2)
        # Loading the head/coordinator takes longer than the old classifier
        # budget, but remains well within the configured inference deadline.
        time.sleep(.15)
        ring.start(0)
        pipe, returned = ring.connect()
        result = SP.coordinate_pipe(None, pipe, Tokenizer(), "prompt", 2, 8, 2, 3,
            ret_sock=returned, local_draft=Draft())
        assert result["output_ids"] == list(range(3, 11))
    finally:
        ring.close()


def test_send_failure_names_head_reset_phase_without_secret_payload(monkeypatch):
    monkeypatch.setattr(SP.wire, "_KEY", SP.wire._KEY)
    SP.wire.use_key("legacy-send-diagnostic-only")
    client, server = socket.socketpair()
    try:
        client.shutdown(socket.SHUT_WR)
        with pytest.raises(SP.TransportError) as failure:
            SP.coordinate_pipe(None, client, Tokenizer(), "private prompt", 0, 2, 2, 1)
        assert "edge=head(s0)" in str(failure.value) and "op=reset" in str(failure.value)
        assert "phase=send" in str(failure.value) and "private prompt" not in str(failure.value)
    finally:
        client.close(); server.close()


def test_abort_does_not_replace_the_request_bound_to_a_pending_return_read(monkeypatch):
    monkeypatch.setattr(SP.wire, "_KEY", SP.wire._KEY)
    SP.wire.use_key("legacy-fifo-diagnostic-only")
    client, server = socket.socketpair()
    try:
        edges = SP._CoordinatorEdges(client)
        edges.send(client, {"op": "verify", "token_ids": [1, 2], "start": 0})
        edges.send(client, {"op": "abort", "discard": 1})
        assert SP.recv_msg(server)["op"] == "verify"
        assert SP.recv_msg(server)["op"] == "abort"
        server.shutdown(socket.SHUT_RDWR); server.close()
        with pytest.raises(SP.EDGE_ERRORS) as failure:
            edges.receive(client)
        diagnostic = edges.failure(failure.value, 1)
        assert "op=verify" in diagnostic and "phase=recv" in diagnostic
        assert list(edges.pending) == ["verify"]
    finally:
        client.close(); server.close()


def test_ngram_compare_cli_branch_passes_one_real_local_draft_to_sync_and_pipe(monkeypatch):
    ring = Ring(monkeypatch, direct=True)
    observed = []
    try:
        ring.start(2); ring.start(1); ring.start(0)
        monkeypatch.setattr(SP.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
        for name in ("coordinate", "coordinate_pipe"):
            original = getattr(SP, name)
            def checked(*args, _original=original, _name=name, **kwargs):
                assert args[0] is None
                assert isinstance(kwargs.get("local_draft"), SP.NgramDrafter)
                result = _original(*args, **kwargs)
                observed.append((_name, kwargs["local_draft"], result["output_ids"]))
                return result
            monkeypatch.setattr(SP, name, checked)
        args = SimpleNamespace(ngram_draft=True, tree="", tree_fast="", compare=True,
            model="cpu-test-only", ngram_n=2, draft_server="", next=ring.endpoints[0],
            tail=ring.endpoints[2], direct_return=True, timeout=2, depths="2", ks="2",
            prompt="prompt", max_new=8, dump="")
        SP._run_coordinator(args, None, None)
        assert [name for name, _, _ in observed] == ["coordinate"] * 2 + ["coordinate_pipe"] * 2
        assert all(draft is observed[0][1] for _, draft, _ in observed)
        assert all(ids == list(range(3, 11)) for _, _, ids in observed)
    finally:
        ring.close()


@pytest.mark.parametrize("tree,tree_fast", [("2,3", ""), ("", "2,3")])
def test_ngram_tree_cli_fails_before_loading_tokenizer_or_connecting(monkeypatch, tree, tree_fast):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported mode touched the model/network before refusal")
    monkeypatch.setattr(SP.AutoTokenizer, "from_pretrained", forbidden)
    monkeypatch.setattr(SP, "connect_ring", forbidden)
    args = SimpleNamespace(ngram_draft=True, tree=tree, tree_fast=tree_fast)
    with pytest.raises(ValueError, match="linear sync/pipe/compare"):
        SP._run_coordinator(args, None, None)

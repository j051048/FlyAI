"""Versioned engine role/session contracts over the existing authenticated codec.

No new network identity or NAT layer. The existing head receipt key signs owner
grants; its public key is pinned by the deployment. Static forward links carry a
signed per-session envelope, so return and forward channels cannot mix owners.
"""
from dataclasses import dataclass
from collections import deque
import base64
import hashlib
import json
import math
import queue
import secrets
import select
import socket
import threading
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

PROTOCOL = "shard-pipeline-session/1"
_CURRENT = threading.local()
_CONTROL = object()


@dataclass(frozen=True)
class PreparedFrame:
    envelope: dict


class ProtocolError(ConnectionError):
    pass


class SessionBusy(ProtocolError):
    pass


class _PrefixSocket:
    def __init__(self, raw, prefix):
        self.raw, self.prefix = raw, bytearray(prefix)
    def __getattr__(self, name):
        return getattr(self.raw, name)
    def recv(self, size, flags=0):
        if self.prefix:
            value = bytes(self.prefix[:size]); del self.prefix[:size]
            return value
        return self.raw.recv(size, flags)
    def recv_into(self, buffer, size=0):
        size = size or len(buffer)
        if self.prefix:
            value = self.recv(size); memoryview(buffer)[:len(value)] = value
            return len(value)
        return self.raw.recv_into(buffer, size)


def _handshake_recv(recv, sock):
    """Bound the declared body BEFORE the generic codec allocates model-sized data."""
    prefix = bytearray()
    while len(prefix) < 8:
        value = sock.recv(8 - len(prefix))
        if not value:
            raise ProtocolError("peer closed during the initial handshake")
        prefix.extend(value)
    if int.from_bytes(prefix, "big") > 65536:
        raise ProtocolError("HELLO/proof/control frame exceeds 64 KiB")
    return recv(_PrefixSocket(sock, prefix))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class SessionConfig:
    plan: dict
    index: int = -1
    ttl_s: float = 120.0
    caller_key: object = None

    def __post_init__(self):
        if type(self.index) is not int or not -1 <= self.index < self.plan["nstages"]:
            raise ValueError("invalid stage index")
        if type(self.ttl_s) not in (float, int) or not math.isfinite(self.ttl_s) or not 1 <= self.ttl_s <= 3600:
            raise ValueError("session TTL must be finite in 1..3600")
        for stage in self.plan["stages"]:
            raw = base64.b64decode(stage["signer_pubkey"], validate=True)
            if len(raw) != 32:
                raise ValueError("strict protocol requires pinned stage signing keys")
        if len(base64.b64decode(self.plan["coordinator"]["signer_pubkey"], validate=True)) != 32:
            raise ValueError("strict protocol requires a pinned coordinator signing key")

    @classmethod
    def from_plan(cls, plan, index=-1, *, ttl_s=120, caller_key=None):
        from .pipeline_plan import validate_plan
        return cls(validate_plan(plan), index, ttl_s, caller_key)

    def stage(self, index):
        return self.plan["stages"][index]

    def descriptor(self, index=None):
        index = self.index if index is None else index
        row = self.stage(index) if index >= 0 else self.plan["coordinator"]
        role = ("head_tail" if self.plan["nstages"] == 1 else "head" if index == 0 else
                "tail" if index == self.plan["nstages"] - 1 else "stage") if index >= 0 else "coordinator"
        return {"ring_id": self.plan["ring_id"], "cohort_id": self.plan["cohort_id"],
                "nstages": self.plan["nstages"], "n_layers": self.plan["n_layers"],
                "node_id": row.get("node_id", "coordinator"), "stage_index": index,
                "role": role, "lo": row.get("lo"), "hi": row.get("hi"),
                "signer_pubkey": row["signer_pubkey"]}

    @property
    def contract_digest(self):
        return hashlib.sha256(canonical([self.descriptor(-1)] +
            [self.descriptor(i) for i in range(self.plan["nstages"])] )).hexdigest()

    def verify_grant(self, grant, *, check_expiry=True):
        try:
            if not isinstance(grant, dict) or set(grant) != {"ring_id", "cohort_id", "session_id", "fence", "boot_id", "boot_started_at", "owner", "expires_at", "signature"}:
                raise ValueError("invalid grant fields")
            if grant["ring_id"] != self.plan["ring_id"] or grant["cohort_id"] != self.plan["cohort_id"]:
                raise ValueError("grant belongs to another ring/cohort")
            if type(grant["fence"]) is not int or grant["fence"] < 1:
                raise ValueError("invalid fence")
            for name in ("session_id", "boot_id", "owner"):
                if not isinstance(grant[name], str) or not 1 <= len(grant[name]) <= 128:
                    raise ValueError("invalid grant identity")
            if type(grant["expires_at"]) not in (int, float) or not math.isfinite(grant["expires_at"]):
                raise ValueError("invalid expiration")
            if type(grant["boot_started_at"]) not in (int, float) or not math.isfinite(grant["boot_started_at"]):
                raise ValueError("invalid head boot time")
            if check_expiry and grant["expires_at"] <= time.time():
                raise ValueError("owner grant expired")
            key = Ed25519PublicKey.from_public_bytes(base64.b64decode(self.stage(0)["signer_pubkey"], validate=True))
            key.verify(base64.b64decode(grant["signature"], validate=True),
                       b"shard-owner-grant/1\0" + canonical({k: v for k, v in grant.items() if k != "signature"}))
            return dict(grant)
        except Exception as error:
            raise ProtocolError("owner grant signature/binding/expiry rejected") from error


class SessionRegistry:
    """Remote head exclusion; a local PID lock is not the authority."""
    def __init__(self, config, key):
        from cryptography.hazmat.primitives import serialization
        raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        if base64.b64encode(raw).decode() != config.stage(0)["signer_pubkey"]:
            raise ProtocolError("head signing key differs from deployment")
        self.config, self.key = config, key
        self.lock, self.boot_id, self.fence = threading.RLock(), secrets.token_hex(16), 0
        self.boot_started_at = time.time()
        self.grant, self.sock = None, None

    def _sign(self, grant):
        body = {k: v for k, v in grant.items() if k != "signature"}
        return {**body, "signature": base64.b64encode(self.key.sign(b"shard-owner-grant/1\0" + canonical(body))).decode()}

    def acquire(self, session_id, sock):
        with self.lock:
            if self.grant is not None:
                alive = self.grant["expires_at"] > time.time() and self.sock is not None and self.sock.fileno() >= 0
                if alive:
                    readable, _, _ = select.select([self.sock], [], [], 0)
                    if readable:
                        try:
                            alive = self.sock.recv(1, socket.MSG_PEEK) != b""
                        except OSError:
                            alive = False
                if alive:
                    raise SessionBusy("head already has a live coordinator session")
                self.release(self.grant)
            self.fence += 1
            self.grant = self._sign({"ring_id": self.config.plan["ring_id"], "cohort_id": self.config.plan["cohort_id"],
                "session_id": session_id, "fence": self.fence, "boot_id": self.boot_id, "boot_started_at": self.boot_started_at,
                "owner": secrets.token_hex(32), "expires_at": time.time() + self.config.ttl_s})
            self.sock = sock
            return self.grant

    def touch(self, grant):
        with self.lock:
            self.config.verify_grant(grant, check_expiry=False)
            if self.grant is None or any(grant[k] != self.grant[k] for k in ("owner", "session_id", "fence", "boot_id")):
                raise ProtocolError("coordinator session has been fenced")
            if self.grant["expires_at"] <= time.time():
                self.release(grant)
                raise ProtocolError("coordinator session TTL expired")
            self.grant = self._sign({**self.grant, "expires_at": time.time() + self.config.ttl_s})
            return self.grant

    def release(self, grant):
        with self.lock:
            if self.grant is None or grant.get("owner") != self.grant["owner"]:
                return
            sock, self.sock, self.grant = self.sock, None, None
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass


class SessionSocket:
    def __init__(self, sock, config, purpose, *, grant=None, registry=None, on_close=None):
        self.raw, self.config, self.purpose = sock, config, purpose
        self.grant, self.registry, self.on_close = grant, registry, on_close
        self._last = None
        self.retired_boots = deque(maxlen=256)
        self.acceptor = None
        self.closed = False

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def adopt(self, grant):
        grant = self.config.verify_grant(grant)
        if self.grant is not None:
            if grant["boot_id"] != self.grant["boot_id"]:
                if grant["boot_id"] in self.retired_boots or grant["boot_started_at"] <= self.grant["boot_started_at"]:
                    raise ProtocolError("stale head process epoch")
                self.retired_boots.append(self.grant["boot_id"])
            if grant["boot_id"] == self.grant["boot_id"] and grant["fence"] < self.grant["fence"]:
                raise ProtocolError("stale coordinator fence")
            if grant["boot_id"] == self.grant["boot_id"] and grant["fence"] == self.grant["fence"] and grant["owner"] != self.grant["owner"]:
                raise ProtocolError("owner changed within one fence")
        self.grant = grant

    def prepare(self, payload):
        if isinstance(payload, PreparedFrame):
            return payload
        if isinstance(payload, dict) and payload.get("op") == "noop":
            return PreparedFrame({"protocol": PROTOCOL, "grant": None, "payload": payload})
        current = getattr(_CURRENT, "grant", None)
        if self.purpose == "forward" and current is not None:
            self.adopt(current)
        if self.purpose == "return" and self.config.index >= 0 and current is not None and self.grant is not None:
            if any(current[k] != self.grant[k] for k in ("owner", "session_id", "fence", "boot_id")):
                return None  # a late old-owner result never reaches the new return channel
            self.adopt(current)  # renew the same owner's signed expiry after idle pings
        grant = self.registry.touch(self.grant) if self.registry is not None else self.grant
        if self.registry is not None:
            self.grant = grant
        if grant is None:
            raise ProtocolError("job frame has no owner session")
        return PreparedFrame({"protocol": PROTOCOL, "grant": dict(grant), "payload": payload})

    def send_message(self, send, payload):
        if self.purpose == "probe":
            return send(self.raw, _sign(self.config.caller_key, b"shard-probe-ping/1\0", payload))
        prepared = self.prepare(payload)
        return send(self.raw, prepared.envelope) if prepared is not None else 0

    def _recv_one(self, recv):
        frame = recv(self.raw)
        if not isinstance(frame, dict) or set(frame) != {"protocol", "grant", "payload"} or frame["protocol"] != PROTOCOL:
            raise ProtocolError("missing versioned session envelope")
        payload, grant = frame["payload"], frame["grant"]
        if grant is None and isinstance(payload, dict) and payload.get("op") == "noop":
            return payload
        if self.registry is not None:
            grant = self.registry.touch(grant)
        else:
            grant = self.config.verify_grant(grant)
        if self.acceptor is not None and self.purpose == "forward":
            self.acceptor.check_forward_fence(grant)
        reset = isinstance(payload, dict) and payload.get("op") == "reset"
        if self.grant is not None and not reset and any(grant[k] != self.grant[k] for k in ("owner", "session_id", "fence", "boot_id")):
            raise ProtocolError("reply/frame belongs to another coordinator session")
        self.adopt(grant)
        self._last = grant
        _CURRENT.grant = grant
        if self.registry is not None and isinstance(payload, dict) and payload.get("op") == "session_ping":
            self.send_message(self.control_send, {"op": "session_pong", "session_id": grant["session_id"]})
            return _CONTROL
        return payload

    def recv_message(self, recv):
        if self.purpose == "probe":
            return _verify_signed(self.config.stage(self.target_index)["signer_pubkey"],
                                  b"shard-probe-pong/1\0", recv(self.raw))
        while True:
            result = self._recv_one(recv)
            if result is not _CONTROL:
                return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.registry is not None and self.grant is not None:
            self.registry.release(self.grant)
        self.raw.close()
        if self.on_close is not None:
            self.on_close(self)


def send_message(send, sock, payload):
    return sock.send_message(send, payload) if isinstance(sock, SessionSocket) else send(sock, payload)


def recv_message(recv, sock):
    return sock.recv_message(recv) if isinstance(sock, SessionSocket) else recv(sock)


def prepare_message(sock, payload):
    return sock.prepare(payload) if isinstance(sock, SessionSocket) else payload


def _sign(key, domain, body):
    return {**body, "signature": base64.b64encode(key.sign(domain + canonical(body))).decode()}


def _verify_signed(key_text, domain, value):
    try:
        body = {k: v for k, v in value.items() if k != "signature"}
        Ed25519PublicKey.from_public_bytes(base64.b64decode(key_text, validate=True)).verify(
            base64.b64decode(value["signature"], validate=True), domain + canonical(body))
        return body
    except Exception as error:
        raise ProtocolError("pinned peer challenge/signature rejected") from error


def hello_client(sock, config, target_index, purpose, send, recv, *, session_id=None, grant=None, key=None):
    key = key or config.caller_key
    if key is None:
        raise ProtocolError("authenticated HELLO requires this caller's existing Ed25519 key")
    target = config.descriptor(target_index)
    client_nonce = secrets.token_hex(32)
    hello = {"op": "hello", "protocol": PROTOCOL, **config.descriptor(), "purpose": purpose,
             "expected_peer": target["node_id"], "session_id": session_id, "grant": grant,
             "client_nonce": client_nonce, "plan_digest": config.contract_digest}
    send(sock, hello)
    challenge = _handshake_recv(recv, sock)
    challenge_body = _verify_signed(target["signer_pubkey"], b"shard-hello-challenge/1\0", challenge)
    if (challenge_body.get("op") != "hello_challenge" or challenge_body.get("peer") != target or
            challenge_body.get("hello_sha256") != hashlib.sha256(canonical(hello)).hexdigest() or
            not isinstance(challenge_body.get("server_nonce"), str) or len(challenge_body["server_nonce"]) != 64):
        raise ProtocolError("server role/nonce/contract challenge rejected")
    proof = {"op": "hello_proof", "hello_sha256": challenge_body["hello_sha256"],
             "server_nonce": challenge_body["server_nonce"], "client_nonce": client_nonce}
    send(sock, _sign(key, b"shard-hello-proof/1\0", proof))
    ack = _handshake_recv(recv, sock)
    ack = _verify_signed(target["signer_pubkey"], b"shard-hello-ack/1\0", ack)
    if ack.get("client_nonce") != client_nonce or ack.get("server_nonce") != challenge_body["server_nonce"]:
        raise ProtocolError("HELLO acknowledgement belongs to another connection")
    if not isinstance(ack, dict) or ack.get("protocol") != PROTOCOL or ack.get("op") != "hello_ack":
        raise ProtocolError("peer does not support the required engine session protocol")
    if ack.get("status") == "busy":
        raise SessionBusy(ack.get("error", "head session busy"))
    if ack.get("status") != "ok" or ack.get("peer") != target or ack.get("purpose") != purpose:
        raise ProtocolError("peer role/index/layer/model contract rejected")
    grant = ack.get("grant")
    if purpose in ("drive", "return"):
        config.verify_grant(grant)
    wrapped = SessionSocket(sock, config, purpose, grant=grant)
    wrapped.target_index = target_index
    wrapped.caller_key = key
    return wrapped


class SessionAcceptor:
    """Accept HELLO concurrently with a live job, so BUSY is an immediate reply."""
    def __init__(self, srv, config, send, recv, *, key=None, timeout=600, max_pending=32):
        self.srv, self.config, self.send, self.recv, self.timeout = srv, config, send, recv, timeout
        self.key = key or config.caller_key
        if self.key is None:
            raise ProtocolError("stage HELLO service requires its existing node key")
        self.registry = SessionRegistry(config, self.key) if config.index == 0 else None
        self.queues = {name: queue.Queue(maxsize=1) for name in ("drive", "forward", "return")}
        self.lock, self.slots = threading.RLock(), {}
        self.last_return_grant = None
        self.last_forward_grant = None
        self.forward_retired_boots = deque(maxlen=256)
        self.retired_boots = deque(maxlen=256)
        self.stop = threading.Event(); self.limit = threading.BoundedSemaphore(max_pending)
        srv.settimeout(0.2)
        self.thread = threading.Thread(target=self._accept, daemon=True, name="engine-session-accept")
        self.thread.start()

    def _accept(self):
        while not self.stop.is_set():
            try:
                conn, _ = self.srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            if not self.limit.acquire(False):
                conn.close(); continue
            threading.Thread(target=self._hello, args=(conn,), daemon=True).start()

    def _hello(self, conn):
        wrapped = None
        client_nonce, server_nonce = None, secrets.token_hex(32)
        try:
            conn.settimeout(min(5.0, self.timeout))
            hello = _handshake_recv(self.recv, conn)
            if not isinstance(hello, dict) or hello.get("protocol") != PROTOCOL or hello.get("op") != "hello":
                raise ProtocolError("versioned role HELLO required")
            purpose = hello.get("purpose")
            idx = self.config.index
            if purpose == "drive" and idx == 0:
                expected = self.config.descriptor(-1)
            elif purpose == "return" and idx == self.config.plan["nstages"] - 1:
                expected = self.config.descriptor(-1)
            elif purpose == "forward" and idx > 0:
                expected = self.config.descriptor(idx - 1)
            elif purpose == "probe" and hello.get("stage_index") in (-1, idx - 1):
                expected = self.config.descriptor(hello["stage_index"])
            else:
                raise ProtocolError("wrong connection purpose for this stage role")
            if {k: hello.get(k) for k in expected} != expected or hello.get("expected_peer") != self.config.descriptor()["node_id"]:
                raise ProtocolError("wrong peer/ring/cohort/stage index/layer range")
            if hello.get("plan_digest") != self.config.contract_digest:
                raise ProtocolError("deployment execution contract digest differs")
            client_nonce = hello.get("client_nonce")
            if not isinstance(client_nonce, str) or len(client_nonce) != 64:
                raise ProtocolError("client challenge nonce required")
            digest = hashlib.sha256(canonical(hello)).hexdigest()
            self.send(conn, _sign(self.key, b"shard-hello-challenge/1\0", {
                "op": "hello_challenge", "peer": self.config.descriptor(),
                "hello_sha256": digest, "server_nonce": server_nonce}))
            proof = _verify_signed(expected["signer_pubkey"], b"shard-hello-proof/1\0", _handshake_recv(self.recv, conn))
            if proof != {"op": "hello_proof", "hello_sha256": digest, "server_nonce": server_nonce,
                         "client_nonce": client_nonce}:
                raise ProtocolError("client proof belongs to another connection/role")
            if purpose == "probe":
                self.send(conn, _sign(self.key, b"shard-hello-ack/1\0", {
                    "op": "hello_ack", "protocol": PROTOCOL, "status": "ok", "purpose": purpose,
                    "peer": self.config.descriptor(), "grant": None,
                    "client_nonce": client_nonce, "server_nonce": server_nonce}))
                for seq in range(3):
                    ping = _verify_signed(expected["signer_pubkey"], b"shard-probe-ping/1\0", _handshake_recv(self.recv, conn))
                    if (set(ping) != {"op", "seq", "nonce"} or ping["op"] != "probe_ping" or ping["seq"] != seq or
                            not isinstance(ping["nonce"], str) or len(ping["nonce"]) != 64):
                        raise ProtocolError("invalid bounded control probe")
                    self.send(conn, _sign(self.key, b"shard-probe-pong/1\0", {**ping, "op": "probe_pong"}))
                conn.close()
                return
            grant = hello.get("grant")
            if purpose == "drive":
                sid = hello.get("session_id")
                if not isinstance(sid, str) or not 1 <= len(sid) <= 128:
                    raise ProtocolError("coordinator session_id required")
                grant = self.registry.acquire(sid, conn)
            elif purpose == "return":
                grant = self.config.verify_grant(grant)
            with self.lock:
                if purpose == "return" and self.last_return_grant is not None:
                    previous_grant = self.last_return_grant
                    if grant["boot_id"] == previous_grant["boot_id"]:
                        if grant["fence"] < previous_grant["fence"] or (grant["fence"] == previous_grant["fence"] and grant["owner"] != previous_grant["owner"]):
                            raise ProtocolError("stale return session fence")
                    else:
                        if grant["boot_id"] in self.retired_boots or grant["boot_started_at"] <= previous_grant["boot_started_at"]:
                            raise ProtocolError("stale return head epoch")
                        self.retired_boots.append(previous_grant["boot_id"])
                previous = self.slots.get(purpose)
                if previous is not None and purpose == "forward" and not previous.closed:
                    try:
                        readable, _, _ = select.select([previous.raw], [], [], 0)
                        if previous.fileno() < 0 or (readable and previous.raw.recv(1, socket.MSG_PEEK) == b""):
                            previous.close()
                    except OSError:
                        previous.close()
                if previous is not None and not previous.closed and previous.fileno() >= 0:
                    if purpose == "drive" and previous.grant["owner"] != grant["owner"]:
                        previous.close()
                    elif purpose != "return":
                        raise SessionBusy("stage channel already has a live owner")
                    else:
                        previous.adopt(grant)  # refuses an older fence before replacement
                        previous.close()
                def released(value):
                    with self.lock:
                        if self.slots.get(purpose) is value:
                            self.slots.pop(purpose, None)
                wrapped = SessionSocket(conn, self.config, purpose, grant=grant,
                    registry=self.registry if purpose == "drive" else None, on_close=released)
                wrapped.control_send = self.send
                wrapped.acceptor = self
                self.slots[purpose] = wrapped
                if purpose == "return":
                    self.last_return_grant = grant
            self.send(conn, _sign(self.key, b"shard-hello-ack/1\0", {
                "op": "hello_ack", "protocol": PROTOCOL, "status": "ok", "purpose": purpose,
                "peer": self.config.descriptor(), "grant": grant,
                "client_nonce": client_nonce, "server_nonce": server_nonce}))
            conn.settimeout(self.timeout)
            try:
                self.queues[purpose].put_nowait(wrapped)
            except queue.Full:
                stale = self.queues[purpose].get_nowait(); stale.close()
                self.queues[purpose].put_nowait(wrapped)
        except Exception as error:
            try:
                self.send(conn, _sign(self.key, b"shard-hello-ack/1\0", {"op": "hello_ack", "protocol": PROTOCOL,
                    "status": "busy" if isinstance(error, SessionBusy) else "error", "error": str(error)[:200],
                    "client_nonce": client_nonce, "server_nonce": server_nonce}))
            except Exception:
                pass
            if wrapped is not None:
                wrapped.close()
            else:
                conn.close()
        finally:
            self.limit.release()

    def check_forward_fence(self, grant):
        """Retain the high watermark across authenticated TCP reconnects."""
        with self.lock:
            old = self.last_forward_grant
            if old is not None:
                if grant["boot_id"] == old["boot_id"]:
                    if grant["fence"] < old["fence"] or (grant["fence"] == old["fence"] and grant["owner"] != old["owner"]):
                        raise ProtocolError("stale forward owner fence after reconnect")
                else:
                    if grant["boot_id"] in self.forward_retired_boots or grant["boot_started_at"] <= old["boot_started_at"]:
                        raise ProtocolError("stale forward head epoch after reconnect")
                    self.forward_retired_boots.append(old["boot_id"])
            self.last_forward_grant = dict(grant)

    def get(self, purpose, timeout=None):
        while not self.stop.is_set():
            try:
                value = self.queues[purpose].get(timeout=timeout or 0.2)
                if not value.closed:
                    return value
            except queue.Empty:
                if timeout is not None:
                    raise TimeoutError("waiting for authenticated stage session")
        raise ConnectionError("session acceptor stopped")

    def close(self):
        self.stop.set()
        self.srv.close()
        with self.lock:
            for wrapped in tuple(self.slots.values()):
                wrapped.close()

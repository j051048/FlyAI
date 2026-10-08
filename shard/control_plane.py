"""Reference open control plane: signed offers, leases and ring formation.

The engine imports none of this service. c0mpute can call the same contracts.
No account, payment or reputation system is embedded here. Production ring
backends are supplied by the existing engine gateway; factories never receive
shell commands from a peer. Lease RPC uses the existing libp2p signing identity
and requires TLS for non-loopback HTTP endpoints.
"""
from __future__ import annotations

import argparse
import base64
import copy
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import socket
import sqlite3
import ssl
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .offers import (ModelCohort, OfferRegistry, canonical, load_sidecar_key,
                     peer_id_from_public_key, model_cohort_id)

RPC_SCHEMA = "shard-control-rpc/1"
MAX_BODY = 256 * 1024


class ControlError(ValueError):
    pass


from .leases import LeaseConflict


class ResourceControlError(ControlError, LeaseConflict):
    """A signed capacity refusal, also compatible with existing RPC callers."""
    pass


class CapacityUnavailable(ResourceControlError):
    """A valid measured geometry can form only after lifting insufficient quotas."""
    pass


def _identity(key):
    raw = key.public_key().public_bytes_raw()
    return peer_id_from_public_key(raw), base64.b64encode(raw).decode()


def _signed(kind, payload, key, *, clock=time.time):
    peer, pub = _identity(key)
    body = {"schema": RPC_SCHEMA, "kind": kind, "peer_id": peer, "public_key": pub,
            "issued_at": clock(), "ttl_s": 30, "nonce": secrets.token_hex(32), "payload": payload}
    body["signature"] = base64.b64encode(key.sign(b"shard-control-rpc/1\0" + canonical(body))).decode()
    return body


def _verify(body, kind, *, expected_peer=None, clock=time.time):
    fields = {"schema", "kind", "peer_id", "public_key", "issued_at", "ttl_s", "nonce", "payload", "signature"}
    if not isinstance(body, dict) or set(body) != fields or body["schema"] != RPC_SCHEMA or body["kind"] != kind:
        raise ControlError("invalid signed RPC envelope")
    stamp, ttl = body["issued_at"], body["ttl_s"]
    if (type(stamp) not in (int, float) or type(ttl) not in (int, float) or
            not math.isfinite(stamp) or not math.isfinite(ttl) or not 0 < ttl <= 30 or
            stamp > clock() + 10 or stamp + ttl <= clock()):
        raise ControlError("expired RPC envelope")
    nonce = body["nonce"]
    if not isinstance(nonce, str) or len(nonce) != 64 or any(c not in "0123456789abcdef" for c in nonce):
        raise ControlError("invalid RPC nonce")
    try:
        pub = base64.b64decode(body["public_key"], validate=True)
        peer = peer_id_from_public_key(pub)
        if peer != body["peer_id"] or (expected_peer is not None and peer != expected_peer):
            raise ControlError("RPC peer identity mismatch")
        unsigned = {k: v for k, v in body.items() if k != "signature"}
        Ed25519PublicKey.from_public_bytes(pub).verify(base64.b64decode(body["signature"], validate=True),
                                                     b"shard-control-rpc/1\0" + canonical(unsigned))
    except (ValueError, TypeError, InvalidSignature) as exc:
        raise ControlError("RPC signature failed verification") from exc
    return peer


class ReplayCache:
    def __init__(self, path=":memory:", *, clock=time.time, max_entries=10000):
        self.clock, self.max_entries = clock, max_entries
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False, timeout=10)
        self._db.execute("CREATE TABLE IF NOT EXISTS rpc_nonce(peer TEXT,nonce TEXT,expires REAL,PRIMARY KEY(peer,nonce))")

    def consume(self, peer, body):
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute("DELETE FROM rpc_nonce WHERE expires<=?", (self.clock(),))
                if self._db.execute("SELECT count(*) FROM rpc_nonce").fetchone()[0] >= self.max_entries:
                    raise ControlError("RPC replay cache capacity reached")
                self._db.execute("INSERT INTO rpc_nonce VALUES(?,?,?)", (peer, body["nonce"], body["issued_at"] + body["ttl_s"]))
                self._db.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                self._db.execute("ROLLBACK")
                raise ControlError("replayed RPC nonce") from exc
            except BaseException:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise

    def close(self):
        self._db.close()


class NodeLeaseAgent:
    """Authenticate requests before the ledger's ownership and resource checks."""
    def __init__(self, ledger, key, *, cohorts, replay=None, clock=time.time, stage_factory=None):
        self.ledger, self.key, self.clock = ledger, key, clock
        self.cohorts = set(cohorts)
        self.replay = replay or ReplayCache(clock=clock)
        self.peer_id = _identity(key)[0]
        self.stage_factory, self._runners = stage_factory, {}
        self._runner_lock = threading.RLock()
        if not ledger.node_id.startswith(self.peer_id + "/"):
            raise ControlError("lease node identity differs from sidecar identity")

    def _assignment_guard(self, assignment, lease, peer):
        allowed = {"ring_id", "cohort_id", "node_id", "gpu_uuid", "lo", "hi", "head", "tail", "dspark",
                   "stage", "nstages", "next", "runtime_config_sha256", "deployment_plan", "weight_artifacts", "preparation_mode"}
        if not isinstance(assignment, dict) or set(assignment) - allowed:
            raise ControlError("remote assignment cannot contain commands, paths or environment")
        if any(assignment.get(key) != expected for key, expected in (
                ("ring_id", lease.ring_id), ("cohort_id", lease.model_cohort_sha256),
                ("node_id", lease.node_id), ("gpu_uuid", lease.gpu_uuid))):
            raise ControlError("stage/preparation assignment differs from the authenticated lease")
        artifact = assignment.get("weight_artifacts")
        if "preparation_mode" in assignment and (assignment["preparation_mode"] not in {"fetch", "range_repack"}
                or artifact is None):
            raise ControlError("preparation mode must bind a pinned artifact and be fetch or range_repack")
        geometry = {"lo", "hi", "stage", "nstages", "head", "tail"}
        if artifact is not None or geometry.intersection(assignment) or assignment.get("deployment_plan") is not None:
            for name in ("lo", "hi", "stage", "nstages"):
                if type(assignment.get(name)) is not int:
                    raise ControlError("stage coordinates must be integers")
            if not (0 <= assignment["lo"] < assignment["hi"] and 0 <= assignment["stage"] < assignment["nstages"]):
                raise ControlError("invalid stage coordinates")
            if (type(assignment.get("head")) is not bool or type(assignment.get("tail")) is not bool
                    or assignment["head"] != (assignment["stage"] == 0)
                    or assignment["tail"] != (assignment["stage"] == assignment["nstages"] - 1)):
                raise ControlError("stage roles differ from execution coordinates")
        if artifact is not None and (not isinstance(artifact, dict) or
                set(artifact) != {"artifact_id", "checkpoint_id", "manifest_sha256"} or
                any(not isinstance(value, str) or not value for value in artifact.values())):
            raise ControlError("only a pinned artifact identity may appear in a remote assignment")
        if assignment.get("deployment_plan") is not None:
            from .pipeline_plan import validate_plan
            plan = validate_plan(assignment["deployment_plan"])
            slot = plan["stages"][assignment["stage"]]
            if (plan["ring_id"] != lease.ring_id or plan["cohort_id"] != lease.model_cohort_sha256
                    or any(slot[key] != assignment[key] for key in ("node_id", "gpu_uuid", "lo", "hi", "head", "tail"))):
                raise ControlError("deployment plan differs from the node assignment")
        guard = self.ledger.guard(lease.lease_id, lease.fencing_token, principal=peer,
                                 ring_id=lease.ring_id, model_cohort_sha256=lease.model_cohort_sha256)
        guard.assert_live()
        return guard

    def dispatch(self, envelope):
        peer = _verify(envelope, "request", clock=self.clock)
        self.replay.consume(peer, envelope)
        payload = envelope["payload"]
        try:
            if not isinstance(payload, dict) or set(payload) != {"action", "body"}:
                raise ControlError("invalid RPC payload")
            action, body = payload["action"], payload["body"]
            if not isinstance(body, dict):
                raise ControlError("invalid RPC body")
            from .leases import LeaseRequest, WorkLease
            if action == "prepare":
                if set(body) != {"request", "idempotency_key"}:
                    raise ControlError("invalid prepare arguments")
                request = LeaseRequest.from_dict(body["request"])
                if request.gpu_uuid != self.ledger.node_id.split("/", 1)[1]:
                    raise ControlError("GPU differs from authenticated node endpoint")
                if request.model_cohort_sha256 not in self.cohorts:
                    raise ControlError("node does not support this model cohort")
                value = self.ledger.prepare(request, principal=peer, idempotency_key=body["idempotency_key"])
            elif action in {"commit", "release", "revoke", "assert_fence", "renew", "begin_work"}:
                common = {"lease_id", "fencing_token"}
                extras = {"renew": {"ttl_s", "idempotency_key"}, "begin_work": {"work_id"}}.get(action, set())
                if set(body) != common | extras:
                    raise ControlError(f"invalid {action} arguments")
                kwargs = {"principal": peer}
                if action == "renew":
                    kwargs.update(ttl_s=body["ttl_s"], idempotency_key=body["idempotency_key"])
                elif action == "begin_work":
                    work_id = body["work_id"]
                    if not isinstance(work_id, str) or not work_id or len(work_id) > 128:
                        raise ControlError("bounded RPC work_id required")
                    # A remote controller cannot create or end a node-local
                    # resident engine handle, even when it owns the lease.
                    kwargs["work_id"] = "rpc:" + peer + ":" + work_id
                value = getattr(self.ledger, action)(body["lease_id"], body["fencing_token"], **kwargs)
            elif action in {"prepare_stage", "prepare_status"}:
                expected = {"lease_id", "fencing_token", "assignment"} | ({"job_id"} if action == "prepare_status" else set())
                if set(body) != expected:
                    raise ControlError("invalid preparation lifecycle arguments")
                lease = self.ledger.get(body["lease_id"], principal=peer)
                if lease.fencing_token != body["fencing_token"]:
                    raise ControlError("preparation stage fencing token differs")
                guard = self._assignment_guard(body["assignment"], lease, peer)
                preparer = getattr(self.stage_factory, "weight_preparer", None)
                if preparer is None:
                    if body["assignment"].get("weight_artifacts") is not None:
                        raise ControlError("node has no locally configured artifact preparation source")
                    value = {"ready": True, "state": "not_required"}
                elif action == "prepare_stage":
                    # Construction validates an exact locally calibrated template;
                    # it does not start a process or load model weights.
                    self.stage_factory(body["assignment"], guard)
                    if body["assignment"].get("preparation_mode") is not None:
                        choices = preparer.options(body["assignment"], guard)
                        if body["assignment"]["preparation_mode"] not in choices["options"]:
                            raise ControlError("requested preparation mode is not locally approved")
                    value = preparer.submit(body["assignment"], guard)
                else:
                    if not isinstance(body["job_id"], str) or not body["job_id"]:
                        raise ControlError("bounded preparation job identity required")
                    value = preparer.status(body["assignment"], guard, body["job_id"])
            elif action in {"start_stage", "stop_stage", "stage_status"}:
                expected = {"lease_id", "fencing_token"} | ({"assignment"} if action == "start_stage" else set())
                if set(body) != expected:
                    raise ControlError("invalid stage lifecycle arguments")
                lease = self.ledger.get(body["lease_id"], principal=peer)
                if lease.fencing_token != body["fencing_token"]:
                    raise ControlError("stage lifecycle fencing token differs")
                with self._runner_lock:
                    runner = self._runners.get(lease.lease_id)
                    if action == "start_stage":
                        if self.stage_factory is None:
                            raise ControlError("node has no locally configured engine runner")
                        assignment = body["assignment"]
                        guard = self._assignment_guard(assignment, lease, peer)
                        if runner is None or not runner.status()["resident_work_held"]:
                            if self.ledger.resident_work(lease.lease_id, principal=peer):
                                raise ControlError("resident stage cleanup is unconfirmed after restart")
                            runner = self.stage_factory(assignment, guard)
                            value = runner.start(assignment)
                            self._runners[lease.lease_id] = runner
                        else:
                            value = runner.status()
                    elif runner is None:
                        orphaned = bool(self.ledger.resident_work(lease.lease_id, principal=peer))
                        value = {"running": None if orphaned else False, "resident_work_held": orphaned,
                                 "cleanup_unconfirmed": orphaned}
                    elif action == "stop_stage":
                        runner.stop()
                        value = runner.status()
                        self._runners.pop(lease.lease_id, None)
                    else:
                        value = runner.status()
            elif action == "end_work":
                if set(body) != {"work"} or set(body["work"]) != {"work_id", "lease_id", "fencing_token"}:
                    raise ControlError("invalid work release")
                if not body["work"]["work_id"].startswith("rpc:" + peer + ":"):
                    raise ControlError("remote controller cannot acknowledge local engine cleanup")
                value = self.ledger.end_work(WorkLease(**body["work"]), principal=peer)
            else:
                raise ControlError("unknown lease action")
            result = value.to_dict() if hasattr(value, "to_dict") else asdict(value) if hasattr(value, "__dataclass_fields__") else value
            answer = {"request_nonce": envelope["nonce"], "ok": True, "result": result}
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            answer = {"request_nonce": envelope["nonce"], "ok": False, "error": str(exc)}
            from .leases import LeaseConflict
            if isinstance(exc, LeaseConflict):
                answer["error_code"] = "resource_conflict"
        return _signed("response", answer, self.key, clock=self.clock)


class _BoundedHTTP(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, handler, *, max_connections=32):
        self.slots = threading.BoundedSemaphore(max_connections)
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def _json_loads(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ControlError("duplicate JSON key")
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ControlError("nonfinite JSON")))


def control_server(address, dispatch, *, certificate=None, private_key=None):
    """Bounded JSON HTTP adapter; public bindings require configured TLS."""
    host, port = address
    if bool(certificate) != bool(private_key):
        raise ControlError("provide both TLS certificate and key")
    if host not in {"127.0.0.1", "::1", "localhost"} and not (certificate and private_key):
        raise ControlError("non-loopback control listener requires TLS")
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def do_POST(self):
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ControlError("chunked control requests are unsupported")
                sizes = self.headers.get_all("Content-Length") or []
                if len(sizes) != 1 or not sizes[0].isdigit() or not 0 < int(sizes[0]) <= MAX_BODY:
                    raise ControlError("bounded Content-Length required")
                size = int(sizes[0])
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ControlError("incomplete control body")
                value = dispatch(self.path, _json_loads(raw))
                status = 200
            except (ValueError, KeyError, TypeError, OSError) as exc:
                value, status = {"error": str(exc)}, 400
            raw = canonical(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)
            self.close_connection = True

        def log_message(self, *args):
            pass
    server = _BoundedHTTP(address, Handler)
    if certificate and private_key:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certificate, private_key)
        # Handshake in the bounded worker after its socket timeout is set;
        # an idle TLS client must not block the accept/renewal loop.
        server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
    return server


class LeaseRPCClient:
    def __init__(self, address, node_id, key, *, timeout_s=10, clock=time.time):
        parsed = urlparse(address)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ControlError("invalid lease RPC address")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
            raise ControlError("remote lease RPC requires HTTPS")
        if not isinstance(node_id, str) or "/" not in node_id:
            raise ControlError("lease RPC node_id must be PeerId/GPU UUID")
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ControlError("positive RPC timeout required")
        self.address, self.node_id, self.key = address.rstrip("/") + "/lease", node_id, key
        self.expected_peer, self.clock, self.timeout_s = node_id.split("/", 1)[0], clock, timeout_s

    def call(self, action, body):
        envelope = _signed("request", {"action": action, "body": body}, self.key, clock=self.clock)
        request = Request(self.address, canonical(envelope), {"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=self.timeout_s) as response:
            raw = response.read(MAX_BODY + 1)
        if len(raw) > MAX_BODY:
            raise ControlError("oversized lease reply")
        answer = _json_loads(raw)
        _verify(answer, "response", expected_peer=self.expected_peer, clock=self.clock)
        payload = answer["payload"]
        if not isinstance(payload, dict) or payload.get("request_nonce") != envelope["nonce"]:
            raise ControlError("lease reply belongs to another request")
        if payload.get("ok") is not True:
            if payload.get("error_code") == "resource_conflict":
                raise ResourceControlError(payload.get("error", "node resource reservation failed"))
            raise ControlError(payload.get("error", "lease request failed"))
        return payload["result"]

    def prepare(self, request, *, idempotency_key):
        return self.call("prepare", {"request": request.to_dict(), "idempotency_key": idempotency_key})

    def operation(self, action, lease, **extra):
        return self.call(action, {"lease_id": lease["lease_id"], "fencing_token": lease["fencing_token"], **extra})


class RemoteLeaseGuard:
    def __init__(self, client, lease_id, fencing_token, *, ring_id, cohort_id):
        self.client, self.lease_id, self.fencing_token = client, lease_id, fencing_token
        self.ring_id, self.cohort_id = ring_id, cohort_id
        self.watch_interval_s = 1.0
        self._lease = self.assert_live()
        self.gpu_uuid, self.memory_domain_id = self._lease.gpu_uuid, self._lease.memory_domain_id
        from .leases import LeaseResources
        self.node_id = self._lease.node_id
        self.resources = LeaseResources.from_dict(self._lease.resources)

    def assert_live(self):
        row = self.client.call("assert_fence", {"lease_id": self.lease_id, "fencing_token": self.fencing_token})
        if (row.get("ring_id") != self.ring_id or row.get("model_cohort_sha256") != self.cohort_id or
                row.get("node_id") != self.client.node_id or row.get("fencing_token") != self.fencing_token):
            raise ControlError("remote lease binding mismatch")
        return SimpleNamespace(**row)

    def begin_work(self, work_id=None):
        return self.client.call("begin_work", {"lease_id": self.lease_id, "fencing_token": self.fencing_token,
                                               "work_id": work_id or secrets.token_hex(16)})

    def end_work(self, work):
        return self.client.call("end_work", {"work": work})


def remote_lease_guard(spec, *, ring_id, cohort_id):
    client = LeaseRPCClient(spec["rpc_address"], spec["node_id"], load_sidecar_key(spec["sidecar_key"]),
                            timeout_s=spec.get("timeout_s", 10))
    return RemoteLeaseGuard(client, spec["lease_id"], spec["fencing_token"], ring_id=ring_id, cohort_id=cohort_id)


class LocalLeaseClient:
    """The same protocol seam for a locally authenticated controller identity."""
    def __init__(self, ledger, principal):
        self.ledger, self.principal, self.node_id = ledger, principal, ledger.node_id

    def prepare(self, request, *, idempotency_key):
        return self.ledger.prepare(request, principal=self.principal, idempotency_key=idempotency_key).to_dict()

    def operation(self, action, lease, **extra):
        result = getattr(self.ledger, action)(lease["lease_id"], lease["fencing_token"], principal=self.principal, **extra)
        return result.to_dict() if hasattr(result, "to_dict") else result

    def guard(self, lease, *, ring_id, cohort_id):
        return self.ledger.guard(lease["lease_id"], lease["fencing_token"], principal=self.principal,
                                 ring_id=ring_id, model_cohort_sha256=cohort_id)


class ManagedRingBackend:
    """Existing inference backend plus node-local process lifecycle acknowledgements."""
    def __init__(self, backend, nodes):
        self.backend, self.nodes = backend, tuple(nodes)
        self._telemetry_lock = threading.Lock()
        self._telemetry = {}
        self._telemetry_at = 0.0
        self._telemetry_loading = False
        self._telemetry_closed = False

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def load(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(16, len(self.nodes))) as workers:
            futures = [workers.submit(client.operation, "start_stage", lease, assignment=assignment)
                       for client, lease, assignment in self.nodes]
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    errors.append(exc)
            if errors:
                # Stop every selected node, including starts whose reply was lost.
                self.close()
                raise errors[0]

    def close(self):
        self._telemetry_closed = True
        self.backend.close()
        errors = []
        for client, lease, _ in self.nodes:
            try:
                state = client.operation("stop_stage", lease)
                if state.get("resident_work_held") or state.get("cleanup_unconfirmed"):
                    raise ControlError("node engine cleanup was not acknowledged")
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def stats(self):
        """Metrics reads never wait for node RPC; stale observations retain timestamps."""
        now = time.monotonic()
        with self._telemetry_lock:
            if not self._telemetry_closed and not self._telemetry_loading and now - self._telemetry_at >= 5:
                self._telemetry_loading = True
                threading.Thread(target=self._refresh_telemetry, daemon=True,
                                 name="ring-stage-telemetry").start()
            snapshot = dict(self._telemetry)
        return {**self.backend.stats(), "node_telemetry": snapshot}

    def _refresh_telemetry(self):
        from concurrent.futures import ThreadPoolExecutor
        def read(record):
            client, lease, _ = record
            try:
                status = client.operation("stage_status", lease)
                return client.node_id, {"available": status.get("telemetry") is not None,
                                        "sample": status.get("telemetry"), "observed_at": time.time()}
            except Exception:
                return client.node_id, {"available": False, "observed_at": time.time()}
        try:
            with ThreadPoolExecutor(max_workers=min(16, len(self.nodes))) as workers:
                observed = dict(workers.map(read, self.nodes))
            with self._telemetry_lock:
                self._telemetry, self._telemetry_at = observed, time.monotonic()
        finally:
            with self._telemetry_lock:
                self._telemetry_loading = False



class FormationController:
    """Plan, reserve and start rings; maintenance continues during slow loading.

    The caller supplies measured block contracts and a configured engine adapter.
    Global locks cover only state snapshots, never network or model operations.
    """
    def __init__(self, registry, ring_pool, agents, *, requirements, backend_factory,
                 clock=time.time, maintain=True, agent_factory=None):
        self.registry, self.pool, self.agents = registry, ring_pool, dict(agents)
        self.requirements, self.backend_factory, self.clock = requirements, backend_factory, clock
        self.agent_factory = agent_factory
        self._lock = threading.RLock()
        self._formations = {}
        self._closed = False
        self._maintenance_stop = threading.Event()
        self._maintenance = None
        if maintain:
            self._maintenance = threading.Thread(target=self._maintain, daemon=True, name="ring-lease-renewal")
            self._maintenance.start()

    def _maintain(self):
        while not self._maintenance_stop.wait(.1):
            self.tick(asynchronous=True)

    def form(self, ring_id, cohort, profile, *, rtt=None, measurements=None, locality=None,
             objective="serial", workload=None, ttl_s=120, warmup_timeout_s=60, coordinator_id=None, route_ids=None):
        from .plan import plan_ring
        from .leases import LeaseRequest, LeaseResources
        from .resources import PlacementRequirements
        from datetime import datetime
        cohort = cohort if isinstance(cohort, ModelCohort) else ModelCohort.from_dict(cohort)
        cid = cohort.cohort_id
        if int(profile.get("n_layers", -1)) != cohort.n_layers:
            raise ControlError("placement profile layer count differs from model cohort")
        info = {"leases": [], "gpus": [], "cohort": cid, "ttl_s": ttl_s, "next_renewal": self.clock() + ttl_s / 3,
                "renewal_sequence": 0, "io_lock": threading.RLock(), "failed": threading.Event(),
                "added": False, "draining": False, "backend": None}
        with self._lock:
            if self._closed or ring_id in self._formations:
                raise ControlError("controller is closed or ring_id already has a formation")
            self._formations[ring_id] = info
            busy = {uuid for other in self._formations.values() if other is not info for uuid in other["gpus"]}
        guards, contracts = [], []
        try:
            nodes = self.registry.snapshot(cid)
            all_ids = [n["id"] for n in nodes]
            if self.agent_factory is not None:
                for node in nodes:
                    if node["id"] not in self.agents:
                        try:
                            client = self.agent_factory(self.registry.get(node["id"]))
                            if client.node_id != node["id"]:
                                raise ControlError("discovered lease agent identity mismatch")
                            with self._lock:
                                self.agents[node["id"]] = client
                        except (ValueError, KeyError, OSError):
                            continue  # Registered nodes without a usable endpoint stay in the pool.
            nodes = [n for n in nodes if n["gpu_uuid"] not in busy and n["id"] in self.agents]
            if rtt is not None:
                # Caller dense matrices align to the complete cohort snapshot.
                positions = [all_ids.index(n["id"]) for n in nodes]
                rtt = [[rtt[a][b] for b in positions] for a in positions]
            diagnostics = {}
            plan = plan_ring(nodes, rtt, profile, locality=locality or {"mode": "prefer_local"},
                             objective=objective, workload=workload, measurements=measurements, now=self.clock(),
                             diagnostics=diagnostics, **({"coordinator_id": coordinator_id} if coordinator_id is not None else {}),
                             **({"route_ids": route_ids} if route_ids is not None else {}))
            if plan is None:
                # Prove this is a capacity refusal, not an invalid cohort,
                # missing calibration, unsupported shape or unreachable route.
                relaxed = copy.deepcopy(nodes)
                evidence = False
                total_ram = total_pin = total_disk = 0
                for node in relaxed:
                    rejected = node.get("capacity_rejected_spans", [])
                    if rejected:
                        evidence = True
                        node["allowed_spans"] = [*node.get("allowed_spans", []), *(item["span"] for item in rejected)]
                    for span in node.get("allowed_spans", []):
                        total_ram += span["host_bytes"] + span.get("prepare_ram_bytes", 0)
                        total_pin += span["pinned_bytes"] + span.get("prepare_pinned_bytes", 0)
                        total_disk += span.get("disk_bytes", 0)
                if evidence:
                    for node in relaxed:
                        rows = node.get("allowed_spans", [])
                        if not rows:
                            continue
                        caps = node["resource_capacity"]
                        caps["available_vram_bytes"] = max(caps["available_vram_bytes"], max(row["gpu_bytes"] for row in rows))
                        node["free_vram_mb"] = caps["available_vram_bytes"] / 1024**2
                        if caps["available_ram_bytes"] is not None:
                            caps["available_ram_bytes"] = max(caps["available_ram_bytes"], total_ram)
                        if caps["pinnable_ram_bytes"] is not None:
                            caps["pinnable_ram_bytes"] = max(caps["pinnable_ram_bytes"], total_pin)
                        for filesystem in caps.get("filesystems", {}).values():
                            if filesystem["available_disk_bytes"] is not None:
                                filesystem["available_disk_bytes"] = max(filesystem["available_disk_bytes"], total_disk)
                    possible = plan_ring(relaxed, rtt, profile, locality=locality or {"mode": "prefer_local"},
                        objective=objective, workload=workload, measurements=measurements, now=self.clock(),
                        **({"coordinator_id": coordinator_id} if coordinator_id is not None else {}),
                        **({"route_ids": route_ids} if route_ids is not None else {}))
                    if possible is not None:
                        raise CapacityUnavailable("fresh calibrated stages/routes exist but measured GPU/host/filesystem capacity is insufficient")
                raise ControlError(diagnostics.get("reason", "no compatible ring found in the examined candidates"))
            plan["ring_id"] = ring_id
            info["plan"] = plan
            selected = {node["id"]: node for node in nodes}
            with self._lock:
                info["gpus"] = [self.registry.get(stage["id"])["gpu_uuid"] for stage in plan["stages"]]
            for stage in plan["stages"]:
                stage["cohort_id"] = cid
                offer = self.registry.get(stage["id"])
                if (offer["sequence"] != selected[stage["id"]]["offer_sequence"] or
                        cid not in {model_cohort_id(item["cohort"]) for item in offer["models"]}):
                    raise ControlError("selected offer changed during planning; form a fresh plan")
                req = self.requirements(copy.deepcopy(stage), copy.deepcopy(offer))
                if not isinstance(req, PlacementRequirements):
                    req = PlacementRequirements.from_dict(req)
                if ((req.layer_start, req.layer_end, req.model_id) != (stage["lo"], stage["hi"], cohort.model_id)
                        or req.provenance.node_id != offer["node_id"] or req.provenance.checkpoint_id != cohort.checkpoint_id):
                    raise ControlError("measured block contract differs from model assignment")
                measured = datetime.fromisoformat(req.provenance.measured_at.replace("Z", "+00:00")).timestamp()
                if not -30 <= self.clock() - measured <= self.registry.max_ttl_s:
                    raise ControlError("block calibration is stale")
                if stage.get("storage") is not None:
                    from .resources import StorageRequirements, NodeResources, evaluate_fit
                    storage = StorageRequirements.from_dict(stage["storage"])
                    offered = [row for capability in offer["models"] if model_cohort_id(capability["cohort"]) == cid
                               for row in capability.get("calibrations", []) if row.get("storage") is not None
                               and row["requirements"]["provenance"]["runtime_config_sha256"] == req.provenance.runtime_config_sha256
                               and canonical(row["storage"]) == canonical(storage.to_dict())]
                    if len(offered) != 1:
                        raise ControlError("selected storage/preparation contract differs from signed calibration")
                    if (storage.checkpoint_id != cohort.checkpoint_id or storage.manifest_sha256 != cohort.manifest_sha256
                            or storage.model_id != cohort.model_id or (storage.layer_start, storage.layer_end) != (stage["lo"], stage["hi"])):
                        raise ControlError("storage artifact identity differs from the formation cohort")
                    values = offer["resources"]
                    filesystem = values.get("filesystems", {}).get(storage.filesystem_id, {})
                    if self.clock() - filesystem.get("measured_at", 0) > self.registry.max_ttl_s:
                        raise LeaseConflict("target filesystem measurement is absent or stale")
                    available = NodeResources(values["available_vram_bytes"], values["available_ram_bytes"],
                        values["pinnable_ram_bytes"], filesystem.get("available_disk_bytes"))
                    fit = evaluate_fit(req, available, storage, preparation_mode=stage.get("preparation_mode"))
                    if not fit["fits"]:
                        raise LeaseConflict("selected preparation/GPU/host capacity became insufficient or unknown")
                contracts.append((stage, offer, req))
            for stage, offer, req in contracts:
                request = LeaseRequest(ring_id, cid, offer["node_id"], offer["gpu_uuid"], offer["memory_domain_id"],
                                       LeaseResources(req.gpu.peak_bytes, req.host.peak_bytes, req.host.pinned_bytes), ttl_s)
                client = self.agents[offer["node_id"]]
                if client.node_id != offer["node_id"]:
                    raise ControlError("lease agent identity differs from selected offer")
                with info["io_lock"]:
                    if info["failed"].is_set():
                        raise ControlError("formation lease maintenance failed")
                    lease = client.prepare(request, idempotency_key=f"{ring_id}:{offer['node_id']}")
                    info["leases"].append((client, lease))
            with info["io_lock"]:
                if any(self.registry.get(stage["id"])["sequence"] != selected[stage["id"]]["offer_sequence"]
                       for stage, _, _ in contracts):
                    raise ControlError("selected offer changed before commit")
                for i, (client, lease) in enumerate(info["leases"]):
                    lease = client.operation("commit", lease)
                    info["leases"][i] = client, lease
                    guard = (client.guard(lease, ring_id=ring_id, cohort_id=cid) if isinstance(client, LocalLeaseClient)
                             else RemoteLeaseGuard(client, lease["lease_id"], lease["fencing_token"], ring_id=ring_id, cohort_id=cid))
                    guards.append(guard)
            backend = self.backend_factory(copy.deepcopy(plan), cohort, [req.to_dict() for _, _, req in contracts])
            info["backend"] = backend
            regions = {offer.get("region") for _, offer, _ in contracts}
            region = next(iter(regions)) if len(regions) == 1 and None not in regions else None
            self.pool.add(ring_id, backend, model_id=cohort.model_id, cohort_id=cid,
                          gpu_uuids=info["gpus"], region=region, on_stopped=self._stopped)
            info["added"] = True
            self.pool.reserve(ring_id, guards)
            self.pool.mark_loading(ring_id)
            load = getattr(backend, "load", None)
            if load is not None:
                handles = []
                try:
                    for guard in guards:
                        handles.append((guard, guard.begin_work()))
                    load()
                    if info["failed"].is_set():
                        raise ControlError("lease maintenance failed during loading")
                finally:
                    errors = []
                    for guard, work in reversed(handles):
                        try:
                            guard.end_work(work)
                        except Exception as error:
                            errors.append(error)
                    if errors:
                        raise errors[0]
            self.pool.warmup(ring_id, timeout_s=warmup_timeout_s)
            if info["failed"].is_set():
                raise ControlError("lease maintenance failed during warmup")
            return {"ring_id": ring_id, "cohort_id": cid, "plan": plan, "ready": True}
        except BaseException:
            info["failed"].set()
            if info["added"]:
                self.pool.fail(ring_id, "formation failed")
                self.pool.drain(ring_id)
            else:
                if info["backend"] is not None:
                    info["backend"].close()
                self._release_formation(ring_id, info)
            raise

    def tick(self, *, asynchronous=False):
        with self._lock:
            snapshot = list(self._formations.items())
        for ring_id, info in snapshot:
            if info["failed"].is_set() or self.clock() < info["next_renewal"]:
                continue
            with self._lock:
                if info.get("renewing"):
                    continue
                info["renewing"] = True
            if asynchronous:
                threading.Thread(target=self._renew_one, args=(ring_id, info), daemon=True,
                                 name="formation-lease-renewal").start()
            else:
                self._renew_one(ring_id, info)

    def _renew_one(self, ring_id, info):
        acquired = info["io_lock"].acquire(False)
        try:
            if not acquired or info["failed"].is_set():
                return
            info["renewal_sequence"] += 1
            for i, (client, lease) in enumerate(info["leases"]):
                renewed = client.operation("renew", lease, ttl_s=info["ttl_s"],
                                           idempotency_key=f"{ring_id}:renew:{info['renewal_sequence']}")
                info["leases"][i] = client, renewed
            info["next_renewal"] = self.clock() + info["ttl_s"] / 3
        except (ValueError, RuntimeError, OSError):
            info["failed"].set()
            backend = info["backend"]
            if backend is not None:
                backend.abort()
            if info["added"]:
                self.pool.fail(ring_id, "lease renewal failed")
                self.pool.drain(ring_id)
        finally:
            if acquired:
                info["io_lock"].release()
            with self._lock:
                info["renewing"] = False

    def _stopped(self, ring_id):
        # RingPool invokes its callback under its own lock; never perform RPC or
        # acquire the controller lock synchronously on that callback stack.
        threading.Thread(target=self._release_formation, args=(ring_id,), daemon=True,
                         name="stopped-ring-cleanup").start()

    def _release_formation(self, ring_id, expected=None):
        with self._lock:
            info = self._formations.get(ring_id)
        if info is None or (expected is not None and info is not expected):
            return
        info["failed"].set()
        with info["io_lock"]:
            for client, lease in reversed(info["leases"]):
                try:
                    client.operation("release", lease)
                except (ValueError, RuntimeError, OSError):
                    pass  # Node-side expiry/real process cleanup still prevents reuse.
        with self._lock:
            if self._formations.get(ring_id) is info:
                self._formations.pop(ring_id)
            if self._closed and not self._formations:
                self._maintenance_stop.set()

    def stop(self, ring_id):
        with self._lock:
            info = self._formations.get(ring_id)
        if info is None:
            return
        info["draining"] = True
        if info["added"]:
            # Keep renewing until queued and running bindings have all finished.
            self.pool.drain(ring_id)
        else:
            info["failed"].set()
            self._release_formation(ring_id, info)

    def close(self):
        with self._lock:
            self._closed = True
            rings = list(self._formations)
            if not rings:
                self._maintenance_stop.set()
        for ring_id in rings:
            self.stop(ring_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    registry = commands.add_parser("registry")
    registry.add_argument("--db", required=True)
    registry.add_argument("--host", default="127.0.0.1")
    registry.add_argument("--port", type=int, default=29100)
    node = commands.add_parser("node-leases")
    node.add_argument("--db", required=True)
    node.add_argument("--sidecar-key", required=True)
    node.add_argument("--capacity", required=True, help="local measured capacity and supported cohort IDs JSON")
    node.add_argument("--host", default="127.0.0.1")
    node.add_argument("--port", type=int, default=29101)
    for child in (registry, node):
        child.add_argument("--tls-cert")
        child.add_argument("--tls-key")
    args = parser.parse_args(argv)
    if args.command == "registry":
        offers = OfferRegistry(args.db)
        def dispatch(path, body):
            if path == "/offers":
                return offers.announce(body)
            if path == "/snapshot" and set(body) == {"cohort_id"}:
                return {"nodes": offers.snapshot(body["cohort_id"])}
            raise ControlError("unknown registry endpoint")
    else:
        from .leases import LeaseLedger
        key = load_sidecar_key(args.sidecar_key)
        capacity = _json_loads(Path(args.capacity).read_bytes())
        # Node ID is derived from the same authenticated libp2p identity.
        node_id = f"{_identity(key)[0]}/{capacity['gpu_uuid']}"
        # NodeLeaseAgent verifies the signature before it supplies this principal.
        ledger = LeaseLedger(args.db, node_id=node_id,
                             authorize=lambda principal, action, binding: principal if isinstance(principal, str) else None)
        ledger.register_capacity(capacity["memory_domain_id"], available_ram_bytes=capacity["available_ram_bytes"],
                                 pinnable_ram_bytes=capacity["pinnable_ram_bytes"],
                                 gpu_capacity_bytes=capacity["gpu_capacity_bytes"])
        for row in capacity.get("filesystems", []):
            if not isinstance(row, dict) or not {"filesystem_id", "path"} <= set(row) or set(row) - {"filesystem_id", "path", "available_disk_bytes"}:
                raise ControlError("filesystem capacity needs a local identity/path and optional measured free bytes")
            ledger.register_filesystem(row["filesystem_id"], row["path"], available_disk_bytes=row.get("available_disk_bytes"))
        from .leased_runtime import configured_stage_factory
        preparer = None
        if capacity.get("weight_sources"):
            from .weight_prepare import configured_prepare_factory
            preparer = configured_prepare_factory(args.db, capacity["weight_sources"])
        factory = configured_stage_factory(args.db, capacity["stages"], weight_preparer=preparer) if capacity.get("stages") else None
        agent = NodeLeaseAgent(ledger, key, cohorts=capacity["cohort_ids"], replay=ReplayCache(str(args.db) + ".rpc"),
                               stage_factory=factory)
        def dispatch(path, body):
            if path != "/lease":
                raise ControlError("unknown node endpoint")
            return agent.dispatch(body)
    server = control_server((args.host, args.port), dispatch, certificate=args.tls_cert, private_key=args.tls_key)
    print(json.dumps({"listening": server.server_address, "registration_policy": "open"}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

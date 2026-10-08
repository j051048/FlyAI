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
    def __init__(self, ledger, key, *, cohorts, replay=None, clock=time.time):
        self.ledger, self.key, self.clock = ledger, key, clock
        self.cohorts = set(cohorts)
        self.replay = replay or ReplayCache(clock=clock)
        self.peer_id = _identity(key)[0]
        if not ledger.node_id.startswith(self.peer_id + "/"):
            raise ControlError("lease node identity differs from sidecar identity")

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
                    kwargs["work_id"] = body["work_id"]
                value = getattr(self.ledger, action)(body["lease_id"], body["fencing_token"], **kwargs)
            elif action == "end_work":
                if set(body) != {"work"} or set(body["work"]) != {"work_id", "lease_id", "fencing_token"}:
                    raise ControlError("invalid work release")
                value = self.ledger.end_work(WorkLease(**body["work"]), principal=peer)
            else:
                raise ControlError("unknown lease action")
            result = value.to_dict() if hasattr(value, "to_dict") else asdict(value) if hasattr(value, "__dataclass_fields__") else value
            answer = {"request_nonce": envelope["nonce"], "ok": True, "result": result}
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            answer = {"request_nonce": envelope["nonce"], "ok": False, "error": str(exc)}
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
        server.socket = context.wrap_socket(server.socket, server_side=True)
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


class FormationController:
    """Joint planning then all-node reservation, loading and signed warmup.

    requirements(stage,offer) must produce the exact measured block resource
    contract; backend_factory receives that immutable assignment. This prevents
    turning a scalar planning estimate into a production resource admission.
    """
    def __init__(self, registry, ring_pool, agents, *, requirements, backend_factory, clock=time.time):
        self.registry, self.pool, self.agents = registry, ring_pool, dict(agents)
        self.requirements, self.backend_factory, self.clock = requirements, backend_factory, clock
        self._lock = threading.RLock()
        self._formations = {}

    def form(self, ring_id, cohort, profile, *, rtt=None, measurements=None, locality=None,
             objective="serial", workload=None, ttl_s=120, warmup_timeout_s=60):
        from .plan import plan_ring
        from .leases import LeaseRequest, LeaseResources
        from .resources import PlacementRequirements
        cohort = cohort if isinstance(cohort, ModelCohort) else ModelCohort.from_dict(cohort)
        cid = cohort.cohort_id
        if int(profile.get("n_layers", -1)) != cohort.n_layers:
            raise ControlError("placement profile layer count differs from model cohort")
        with self._lock:
            if ring_id in self._formations:
                raise ControlError("ring_id already has a formation")
            nodes = self.registry.snapshot(cid)
            busy = {uuid for info in self._formations.values() for uuid in info["gpus"]}
            nodes = [n for n in nodes if n["gpu_uuid"] not in busy and n["id"] in self.agents]
            # Sparse unknown links never become zero-cost network connections.
            if rtt is None:
                rtt = [[0 if a["id"] == b["id"] else 9000 for b in nodes] for a in nodes]
            plan = plan_ring(nodes, rtt, profile, locality=locality or {"mode": "prefer_local"},
                             objective=objective, workload=workload, measurements=measurements, now=self.clock())
            if plan is None:
                raise ControlError("no compatible ring satisfies locality and resource requirements")
            leases, contracts, guards = [], [], []
            added = False
            try:
                # Validate every measured contract before reserving any node.
                for stage in plan["stages"]:
                    offer = self.registry.get(stage["id"])
                    req = self.requirements(copy.deepcopy(stage), copy.deepcopy(offer))
                    if not isinstance(req, PlacementRequirements):
                        req = PlacementRequirements.from_dict(req)
                    if ((req.layer_start, req.layer_end, req.model_id) != (stage["lo"], stage["hi"], cohort.model_id)
                            or req.provenance.node_id != offer["node_id"] or req.provenance.checkpoint_id != cohort.checkpoint_id):
                        raise ControlError("measured block contract differs from model assignment")
                    # Provenance timestamp may be ISO; no stale evidence is used for admission.
                    from datetime import datetime
                    measured = datetime.fromisoformat(req.provenance.measured_at.replace("Z", "+00:00")).timestamp()
                    if not -30 <= self.clock() - measured <= self.registry.max_ttl_s:
                        raise ControlError("block calibration is stale")
                    contracts.append((stage, offer, req))
                for stage, offer, req in contracts:
                    request = LeaseRequest(ring_id, cid, offer["node_id"], offer["gpu_uuid"],
                                           offer["memory_domain_id"],
                                           LeaseResources(req.gpu.peak_bytes, req.host.peak_bytes, req.host.pinned_bytes), ttl_s)
                    client = self.agents[offer["node_id"]]
                    if client.node_id != offer["node_id"]:
                        raise ControlError("lease agent identity differs from selected offer")
                    lease = client.prepare(request, idempotency_key=f"{ring_id}:{offer['node_id']}")
                    leases.append((client, lease))
                for i, (client, lease) in enumerate(leases):
                    lease = client.operation("commit", lease)
                    leases[i] = client, lease
                    guard = (client.guard(lease, ring_id=ring_id, cohort_id=cid) if isinstance(client, LocalLeaseClient)
                             else RemoteLeaseGuard(client, lease["lease_id"], lease["fencing_token"], ring_id=ring_id, cohort_id=cid))
                    guards.append(guard)
                backend = self.backend_factory(copy.deepcopy(plan), cohort, [req.to_dict() for _, _, req in contracts])
                gpus = [offer["gpu_uuid"] for _, offer, _ in contracts]
                region = next((offer.get("region") for _, offer, _ in contracts if offer.get("region")), None)
                self.pool.add(ring_id, backend, model_id=cohort.model_id, cohort_id=cid, gpu_uuids=gpus, region=region)
                added = True
                self.pool.reserve(ring_id, guards)
                self.pool.mark_loading(ring_id)
                # Optional loader is an explicit, configured engine adapter, never peer-supplied code.
                load = getattr(backend, "load", None)
                if load is not None:
                    handles = [guard.begin_work() for guard in guards]
                    try:
                        load()
                    finally:
                        for guard, work in zip(guards, handles):
                            guard.end_work(work)
                self.pool.warmup(ring_id, timeout_s=warmup_timeout_s)
                self._formations[ring_id] = {"plan": plan, "leases": leases, "gpus": gpus, "cohort": cid,
                                             "ttl_s": ttl_s, "next_renewal": self.clock() + ttl_s / 3,
                                             "renewal_sequence": 0}
                return {"ring_id": ring_id, "cohort_id": cid, "plan": plan, "ready": True}
            except BaseException:
                if added:
                    self.pool.fail(ring_id, "formation failed")
                for client, lease in reversed(leases):
                    try:
                        client.operation("release", lease)
                    except (ValueError, RuntimeError, OSError):
                        pass  # failed RPC retains the node-side reservation until expiry/cleanup
                raise

    def tick(self):
        """Renew live formations; revoke readiness on any lost lease."""
        with self._lock:
            for ring_id, info in list(self._formations.items()):
                if self.clock() < info["next_renewal"]:
                    continue
                info["renewal_sequence"] += 1
                try:
                    for i, (client, lease) in enumerate(info["leases"]):
                        renewed = client.operation("renew", lease, ttl_s=info["ttl_s"],
                                                   idempotency_key=f"{ring_id}:renew:{info['renewal_sequence']}")
                        info["leases"][i] = client, renewed
                    info["next_renewal"] = self.clock() + info["ttl_s"] / 3
                except (ValueError, RuntimeError, OSError):
                    self.pool.fail(ring_id, "lease renewal failed")
                    self.stop(ring_id)

    def stop(self, ring_id):
        with self._lock:
            info = self._formations.get(ring_id)
            if info is None:
                return
            self.pool.drain(ring_id)
            for client, lease in info["leases"]:
                try:
                    client.operation("release", lease)
                except (ValueError, RuntimeError, OSError):
                    pass
            # Node ledgers retain active-work resources until the executor finishes.
            self._formations.pop(ring_id, None)


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
        agent = NodeLeaseAgent(ledger, key, cohorts=capacity["cohort_ids"], replay=ReplayCache(str(args.db) + ".rpc"))
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

"""Node-local durable GPU leases, shared host budgets, and execution fencing.

The protocol adapter MUST supply authenticated principals to the mandatory
authorizer. Its returned subject is the controller; no request can name its own
controller. SQLite records survive restart. Expiry/revocation prevents new work
but does not make an executing GPU reusable until its work is actually cleaned.
"""
from contextlib import contextmanager, closing
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time
import uuid

SCHEMA = "shard-node-leases/1"
_LIVE = ("prepared", "committed", "draining")


class LeaseError(ValueError):
    pass


class LeaseConflict(LeaseError):
    pass


class LeaseExpired(LeaseError):
    pass


class LeaseFenceError(LeaseError):
    pass


class LeaseAuthorizationError(LeaseError):
    pass


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 1024:
        raise LeaseError(f"{name} must be a bounded nonempty string")
    return value


def _bytes(value, name):
    if type(value) is not int or not 0 <= value <= (1 << 63) - 1:
        raise LeaseError(f"{name} must be a nonnegative SQLite byte integer")
    return value


def _ttl(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 86400:
        raise LeaseError("ttl_s must be finite in (0,86400]")
    return float(value)


def _cohort(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise LeaseError("model_cohort_sha256 must be a lowercase SHA-256 digest")
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class LeaseResources:
    vram_bytes: int
    ram_bytes: int
    pinned_bytes: int = 0

    def __post_init__(self):
        for name in ("vram_bytes", "ram_bytes", "pinned_bytes"):
            _bytes(getattr(self, name), name)
        if self.vram_bytes == 0:
            raise LeaseError("a GPU lease requires a positive VRAM reservation")
        if self.pinned_bytes > self.ram_bytes:
            raise LeaseError("pinned memory is a subset of RAM, not an extra budget")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"vram_bytes", "ram_bytes", "pinned_bytes"}:
            raise LeaseError("invalid lease resource fields")
        return cls(**value)


@dataclass(frozen=True)
class LeaseRequest:
    ring_id: str
    model_cohort_sha256: str
    node_id: str
    gpu_uuid: str
    memory_domain_id: str
    resources: LeaseResources
    ttl_s: float

    def __post_init__(self):
        for name in ("ring_id", "node_id", "gpu_uuid", "memory_domain_id"):
            _text(getattr(self, name), name)
        _cohort(self.model_cohort_sha256)
        if not isinstance(self.resources, LeaseResources):
            raise LeaseError("resources must be LeaseResources")
        object.__setattr__(self, "ttl_s", _ttl(self.ttl_s))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        names = {"ring_id", "model_cohort_sha256", "node_id", "gpu_uuid", "memory_domain_id", "resources", "ttl_s"}
        if not isinstance(value, dict) or set(value) != names:
            raise LeaseError("invalid lease request fields (controller is supplied by authentication)")
        return cls(**{**value, "resources": LeaseResources.from_dict(value["resources"])})


@dataclass(frozen=True)
class Lease:
    lease_id: str
    fencing_token: int
    controller_id: str
    ring_id: str
    model_cohort_sha256: str
    node_id: str
    gpu_uuid: str
    memory_domain_id: str
    resources: LeaseResources
    state: str
    expires_at: float
    active_work: int
    created_at: float
    drain_reason: str | None = None

    def __post_init__(self):
        for name in ("lease_id", "controller_id", "ring_id", "node_id", "gpu_uuid", "memory_domain_id"):
            _text(getattr(self, name), name)
        _cohort(self.model_cohort_sha256)
        _bytes(self.fencing_token, "fencing_token")
        _bytes(self.active_work, "active_work")
        if self.fencing_token < 1 or not isinstance(self.resources, LeaseResources):
            raise LeaseError("invalid lease token or resources")
        if self.state not in ("prepared", "committed", "draining", "expired", "released", "revoked"):
            raise LeaseError("invalid lease state")
        if self.drain_reason not in (None, "expired", "released", "revoked"):
            raise LeaseError("invalid drain reason")
        if self.state == "draining" and self.drain_reason is None:
            raise LeaseError("draining lease needs a reason")
        if self.state == "prepared" and self.active_work:
            raise LeaseError("prepared lease cannot have active execution")
        for name in ("expires_at", "created_at"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise LeaseError(f"{name} must be finite Unix time")
        if self.expires_at < self.created_at:
            raise LeaseError("lease expiry precedes creation")

    def to_dict(self):
        return {"schema": SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        names = set(cls.__dataclass_fields__) | {"schema"}
        if not isinstance(value, dict) or set(value) != names or value["schema"] != SCHEMA:
            raise LeaseError("invalid serialized lease")
        body = {key: item for key, item in value.items() if key != "schema"}
        body["resources"] = LeaseResources.from_dict(body["resources"])
        return cls(**body)


@dataclass(frozen=True)
class WorkLease:
    work_id: str
    lease_id: str
    fencing_token: int

    def __post_init__(self):
        _text(self.work_id, "work_id")
        _text(self.lease_id, "lease_id")
        _bytes(self.fencing_token, "fencing_token")
        if self.fencing_token < 1:
            raise LeaseFenceError("work handle fencing token must be positive")

    def to_dict(self):
        return asdict(self)


class LeaseLedger:
    """One shared SQLite file per host/resource domain; multiple node IDs may use it.

    authorize(principal, action, binding) must return the canonical authenticated
    subject string or deny by returning None/raising. A bare True is never identity.
    register_capacity and recover_stopped_work are LOCAL maintenance methods, never
    remote requests. Capacity is a reservation budget excluding unrelated usage,
    not an after-load free-memory sample that would double-count existing leases.
    """

    def __init__(self, path, *, node_id, authorize, clock=time.time, busy_timeout_s=5.0):
        self.path, self.node_id = str(path), _text(node_id, "node_id")
        if self.path == ":memory:":
            raise LeaseError("lease ledger requires a persistent SQLite file")
        if not callable(authorize):
            raise LeaseAuthorizationError("an authenticated protocol authorizer is mandatory")
        if not math.isfinite(busy_timeout_s) or busy_timeout_s <= 0:
            raise LeaseError("invalid SQLite busy timeout")
        self.authorize, self.clock, self.busy_timeout_s = authorize, clock, busy_timeout_s
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS domains(id TEXT PRIMARY KEY,ram INTEGER,pinned INTEGER);
                CREATE TABLE IF NOT EXISTS gpus(uuid TEXT PRIMARY KEY,domain_id TEXT NOT NULL REFERENCES domains(id),
                                               vram INTEGER NOT NULL,fence INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS leases(
                    id TEXT PRIMARY KEY,fence INTEGER NOT NULL,controller TEXT NOT NULL,node TEXT NOT NULL,
                    ring TEXT NOT NULL,cohort TEXT NOT NULL,gpu TEXT NOT NULL REFERENCES gpus(uuid),
                    domain_id TEXT NOT NULL REFERENCES domains(id),resources TEXT NOT NULL,
                    state TEXT NOT NULL,expires REAL NOT NULL,created REAL NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0,drain_reason TEXT);
                CREATE TABLE IF NOT EXISTS operations(
                    node TEXT NOT NULL,controller TEXT NOT NULL,action TEXT NOT NULL,key TEXT NOT NULL,fingerprint TEXT NOT NULL,
                    lease_id TEXT NOT NULL REFERENCES leases(id),PRIMARY KEY(node,controller,action,key));
                CREATE TABLE IF NOT EXISTS work(
                    lease_id TEXT NOT NULL REFERENCES leases(id),id TEXT NOT NULL,fence INTEGER NOT NULL,
                    started REAL NOT NULL,ended REAL,PRIMARY KEY(lease_id,id));
                CREATE INDEX IF NOT EXISTS lease_gpu ON leases(gpu,state);
                CREATE INDEX IF NOT EXISTS lease_domain ON leases(domain_id,state);
            """)
            conn.execute("INSERT OR IGNORE INTO meta VALUES('schema',?)", (SCHEMA,))
            if conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0] != SCHEMA:
                raise LeaseError("unsupported lease ledger schema")
            conn.execute("INSERT OR IGNORE INTO meta VALUES('clock_floor','0')")

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=self.busy_timeout_s, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    @contextmanager
    def _transaction(self):
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = self._observed_now(conn)
            conn.execute("UPDATE meta SET value=? WHERE key='clock_floor'", (str(now),))
            self._cleanup(conn, now)
            # Preserve monotonic clock/expiry even if the requested operation is
            # rejected. Otherwise a later wall-clock rollback could revive a
            # lease after an expired assert_fence rolled its cleanup back.
            conn.execute("SAVEPOINT requested_operation")
            try:
                yield conn, now
            except BaseException:
                conn.execute("ROLLBACK TO requested_operation")
                conn.execute("RELEASE requested_operation")
                conn.commit()
                raise
            conn.execute("RELEASE requested_operation")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _observed_now(self, conn):
        now = self.clock()
        if type(now) not in (int, float) or not math.isfinite(now) or now < 0:
            raise LeaseError("clock must provide finite nonnegative Unix time")
        return max(float(now), float(conn.execute("SELECT value FROM meta WHERE key='clock_floor'").fetchone()[0]))

    @staticmethod
    def _cleanup(conn, now):
        conn.execute("UPDATE leases SET state=CASE WHEN active>0 THEN 'draining' ELSE 'expired' END,"
                     "drain_reason='expired' WHERE state IN ('prepared','committed') AND expires<=?", (now,))
        conn.execute("UPDATE leases SET state=drain_reason WHERE state='draining' AND active=0")

    @staticmethod
    def _lease(row):
        if row is None:
            raise LeaseError("lease not found")
        return Lease(row["id"], row["fence"], row["controller"], row["ring"], row["cohort"], row["node"],
                     row["gpu"], row["domain_id"], LeaseResources.from_dict(json.loads(row["resources"])),
                     row["state"], row["expires"], row["active"], row["created"], row["drain_reason"])

    def _read(self, conn, lease_id):
        lease = self._lease(conn.execute("SELECT * FROM leases WHERE id=?", (_text(lease_id, "lease_id"),)).fetchone())
        if lease.node_id != self.node_id:
            raise LeaseAuthorizationError("lease belongs to another node agent")
        return lease

    def _subject(self, principal, action, binding):
        try:
            subject = self.authorize(principal, action, dict(binding))
        except Exception as exc:
            raise LeaseAuthorizationError("authenticated lease operation denied") from exc
        if not isinstance(subject, str) or not subject.strip() or len(subject) > 1024:
            raise LeaseAuthorizationError("authorizer must return an authenticated subject, never a claimed owner")
        return subject

    def _owned(self, conn, lease_id, fence, principal, action, ring_id=None, cohort=None):
        if type(fence) is not int or fence < 1:
            raise LeaseFenceError("fencing token must be a positive integer")
        lease = self._read(conn, lease_id)
        subject = self._subject(principal, action, lease.to_dict())
        if subject != lease.controller_id:
            raise LeaseAuthorizationError("lease controller differs from authenticated caller")
        if fence != lease.fencing_token:
            raise LeaseFenceError("stale or incorrect fencing token")
        if ring_id is not None and ring_id != lease.ring_id:
            raise LeaseFenceError("lease belongs to another ring")
        if cohort is not None and cohort != lease.model_cohort_sha256:
            raise LeaseFenceError("lease belongs to another model cohort")
        return lease

    @staticmethod
    def _live(lease, now, committed=False):
        if lease.expires_at <= now or lease.state in ("expired", "draining", "revoked", "released"):
            raise LeaseExpired("lease expired, revoked, released, or awaiting execution cleanup")
        if lease.state != ("committed" if committed else "prepared"):
            raise LeaseConflict("operation requires a committed lease" if committed else "operation requires a prepared lease")

    def register_capacity(self, memory_domain_id, *, available_ram_bytes, pinnable_ram_bytes, gpu_capacity_bytes):
        _text(memory_domain_id, "memory_domain_id")
        for name, value in (("available_ram_bytes", available_ram_bytes), ("pinnable_ram_bytes", pinnable_ram_bytes)):
            if value is not None:
                _bytes(value, name)
        if (available_ram_bytes is not None and pinnable_ram_bytes is not None
                and pinnable_ram_bytes > available_ram_bytes):
            raise LeaseError("pinnable budget cannot exceed the host RAM budget")
        if not isinstance(gpu_capacity_bytes, dict) or not gpu_capacity_bytes:
            raise LeaseError("at least one locally observed GPU capacity is required")
        for gpu, capacity in gpu_capacity_bytes.items():
            _text(gpu, "gpu_uuid"); _bytes(capacity, "GPU capacity")
        with self._transaction() as (conn, now):
            rows = conn.execute("SELECT resources FROM leases WHERE domain_id=? AND state IN ('prepared','committed','draining')",
                                (memory_domain_id,)).fetchall()
            reserved = [LeaseResources.from_dict(json.loads(row[0])) for row in rows]
            for used, budget in ((sum(r.ram_bytes for r in reserved), available_ram_bytes),
                                 (sum(r.pinned_bytes for r in reserved), pinnable_ram_bytes)):
                if used and (budget is None or used > budget):
                    raise LeaseConflict("capacity update undercuts existing shared reservations")
            conn.execute("INSERT INTO domains VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET ram=excluded.ram,pinned=excluded.pinned",
                         (memory_domain_id, available_ram_bytes, pinnable_ram_bytes))
            for gpu, capacity in gpu_capacity_bytes.items():
                existing = conn.execute("SELECT * FROM gpus WHERE uuid=?", (gpu,)).fetchone()
                occupied = conn.execute("SELECT * FROM leases WHERE gpu=? AND state IN ('prepared','committed','draining')",
                                        (gpu,)).fetchall()
                if occupied and (existing["domain_id"] != memory_domain_id or
                                 any(json.loads(row["resources"])["vram_bytes"] > capacity for row in occupied)):
                    raise LeaseConflict("cannot move or underbudget an occupied GPU")
                conn.execute("INSERT INTO gpus(uuid,domain_id,vram) VALUES(?,?,?) ON CONFLICT(uuid) "
                             "DO UPDATE SET domain_id=excluded.domain_id,vram=excluded.vram", (gpu, memory_domain_id, capacity))

    def _idempotent(self, conn, subject, action, key, body):
        _text(key, "idempotency_key")
        fingerprint = hashlib.sha256(_json(body).encode()).hexdigest()
        prior = conn.execute("SELECT * FROM operations WHERE node=? AND controller=? AND action=? AND key=?",
                             (self.node_id, subject, action, key)).fetchone()
        if prior is not None and prior["fingerprint"] != fingerprint:
            raise LeaseConflict("idempotency key names different lease operation")
        return prior, fingerprint

    def prepare(self, request, *, principal, idempotency_key):
        if not isinstance(request, LeaseRequest) or request.node_id != self.node_id:
            raise LeaseError("prepare requires a request for this node")
        subject = self._subject(principal, "prepare", request.to_dict())
        with self._transaction() as (conn, now):
            prior, fingerprint = self._idempotent(conn, subject, "prepare", idempotency_key, request.to_dict())
            if prior is not None:
                return self._read(conn, prior["lease_id"])
            gpu = conn.execute("SELECT * FROM gpus WHERE uuid=?", (request.gpu_uuid,)).fetchone()
            if gpu is None or gpu["domain_id"] != request.memory_domain_id:
                raise LeaseConflict("GPU is absent or belongs to another memory domain")
            if conn.execute("SELECT 1 FROM leases WHERE gpu=? AND state IN ('prepared','committed','draining')",
                            (request.gpu_uuid,)).fetchone():
                raise LeaseConflict("GPU remains reserved or awaits execution cleanup")
            if request.resources.vram_bytes > gpu["vram"]:
                raise LeaseConflict("GPU VRAM reservation exceeds local capacity")
            domain = conn.execute("SELECT * FROM domains WHERE id=?", (request.memory_domain_id,)).fetchone()
            reserved = conn.execute("SELECT resources FROM leases WHERE domain_id=? AND state IN ('prepared','committed','draining')",
                                    (request.memory_domain_id,)).fetchall()
            resources = [LeaseResources.from_dict(json.loads(row[0])) for row in reserved]
            for label, demand, limit in (("RAM", request.resources.ram_bytes + sum(r.ram_bytes for r in resources), domain["ram"]),
                                        ("pinned", request.resources.pinned_bytes + sum(r.pinned_bytes for r in resources), domain["pinned"])):
                if demand and (limit is None or demand > limit):
                    raise LeaseConflict(f"shared {label} capacity is unknown or insufficient")
            lease_id, fence = "lease-" + uuid.uuid4().hex, gpu["fence"] + 1
            conn.execute("UPDATE gpus SET fence=? WHERE uuid=?", (fence, request.gpu_uuid))
            conn.execute("INSERT INTO leases VALUES(?,?,?,?,?,?,?,?,?,'prepared',?,?,0,NULL)",
                         (lease_id, fence, subject, self.node_id, request.ring_id, request.model_cohort_sha256,
                          request.gpu_uuid, request.memory_domain_id, _json(request.resources.to_dict()), now + request.ttl_s, now))
            conn.execute("INSERT INTO operations VALUES(?,?,?,?,?,?)",
                         (self.node_id, subject, "prepare", idempotency_key, fingerprint, lease_id))
            return self._read(conn, lease_id)

    def commit(self, lease_id, fencing_token, *, principal):
        with self._transaction() as (conn, now):
            lease = self._owned(conn, lease_id, fencing_token, principal, "commit")
            if lease.state != "committed":
                self._live(lease, now)
                conn.execute("UPDATE leases SET state='committed' WHERE id=?", (lease_id,))
            else:
                self._live(lease, now, committed=True)
            return self._read(conn, lease_id)

    def renew(self, lease_id, fencing_token, *, principal, ttl_s, idempotency_key):
        ttl_s = _ttl(ttl_s)
        with self._transaction() as (conn, now):
            lease = self._owned(conn, lease_id, fencing_token, principal, "renew")
            prior, fingerprint = self._idempotent(conn, lease.controller_id, "renew", idempotency_key,
                                                  {"lease_id": lease_id, "fencing_token": fencing_token, "ttl_s": ttl_s})
            if prior is not None:
                return lease  # never revive an expired lease or repeatedly extend its deadline
            self._live(lease, now, committed=lease.state == "committed")
            conn.execute("UPDATE leases SET expires=MAX(expires,?) WHERE id=?", (now + ttl_s, lease_id))
            conn.execute("INSERT INTO operations VALUES(?,?,?,?,?,?)",
                         (self.node_id, lease.controller_id, "renew", idempotency_key, fingerprint, lease_id))
            return self._read(conn, lease_id)

    def _retire(self, lease_id, fencing_token, principal, reason):
        with self._transaction() as (conn, now):
            lease = self._owned(conn, lease_id, fencing_token, principal,
                                "release" if reason == "released" else "revoke")
            if lease.state in ("prepared", "committed"):
                conn.execute("UPDATE leases SET state=?,drain_reason=? WHERE id=?",
                             ("draining" if lease.active_work else reason, reason, lease_id))
            return self._read(conn, lease_id)

    def release(self, lease_id, fencing_token, *, principal):
        return self._retire(lease_id, fencing_token, principal, "released")

    def revoke(self, lease_id, fencing_token, *, principal):
        return self._retire(lease_id, fencing_token, principal, "revoked")

    def assert_fence(self, lease_id, fencing_token, *, principal, ring_id=None, model_cohort_sha256=None):
        # Valid execution checks need no fsync or writer lock. Expiry must still
        # be durably recorded before refusal so clock rollback cannot revive it.
        with closing(self._connect()) as conn:
            conn.execute("BEGIN")
            now = self._observed_now(conn)
            lease = self._owned(conn, lease_id, fencing_token, principal, "assert_fence", ring_id, model_cohort_sha256)
            if lease.expires_at > now:
                self._live(lease, now, committed=True)
                return lease
        with self._transaction() as (conn, now):
            lease = self._owned(conn, lease_id, fencing_token, principal, "assert_fence", ring_id, model_cohort_sha256)
            self._live(lease, now, committed=True)
            return lease

    def begin_work(self, lease_id, fencing_token, *, principal, work_id=None, ring_id=None, model_cohort_sha256=None):
        work_id = _text("work-" + uuid.uuid4().hex if work_id is None else work_id, "work_id")
        with self._transaction() as (conn, now):
            lease = self._owned(conn, lease_id, fencing_token, principal, "begin_work", ring_id, model_cohort_sha256)
            self._live(lease, now, committed=True)
            prior = conn.execute("SELECT * FROM work WHERE lease_id=? AND id=?", (lease_id, work_id)).fetchone()
            if prior is not None:
                if prior["ended"] is not None:
                    raise LeaseConflict("work ID already ended; use a new work ID")
            else:
                conn.execute("INSERT INTO work VALUES(?,?,?,?,NULL)", (lease_id, work_id, fencing_token, now))
                conn.execute("UPDATE leases SET active=active+1 WHERE id=?", (lease_id,))
            return WorkLease(work_id, lease_id, fencing_token)

    def end_work(self, work, *, principal):
        if not isinstance(work, WorkLease):
            raise LeaseError("end_work requires a WorkLease")
        with self._transaction() as (conn, now):
            self._owned(conn, work.lease_id, work.fencing_token, principal, "end_work")
            self._end(conn, work, now)
            self._cleanup(conn, now)
            return self._read(conn, work.lease_id)

    @staticmethod
    def _end(conn, work, now):
        row = conn.execute("SELECT * FROM work WHERE lease_id=? AND id=?", (work.lease_id, work.work_id)).fetchone()
        if row is None or row["fence"] != work.fencing_token:
            raise LeaseFenceError("unknown or stale work handle")
        if row["ended"] is None:
            conn.execute("UPDATE work SET ended=? WHERE lease_id=? AND id=?", (now, work.lease_id, work.work_id))
            conn.execute("UPDATE leases SET active=active-1 WHERE id=? AND active>0", (work.lease_id,))

    def get(self, lease_id, *, principal):
        with self._transaction() as (conn, now):
            lease = self._read(conn, lease_id)
            if self._subject(principal, "get", lease.to_dict()) != lease.controller_id:
                raise LeaseAuthorizationError("lease controller differs from authenticated caller")
            return lease

    def sweep(self):
        """Local expiry cleanup. Active work is NEVER assumed gone on time/restart."""
        with self._transaction() as (conn, now):
            return [self._lease(row) for row in conn.execute("SELECT * FROM leases WHERE node=?", (self.node_id,))]

    def recover_stopped_work(self, confirm_stopped):
        """Local runner callback must verify each abandoned work actually stopped.

        Do not expose this maintenance operation as a remote `cleanup_confirmed`
        boolean. A restart alone is not evidence that another GPU process stopped.
        """
        if not callable(confirm_stopped):
            raise LeaseError("local execution cleanup verifier is required")
        with self._transaction() as (conn, now):
            rows = conn.execute("SELECT w.* FROM work w JOIN leases l ON l.id=w.lease_id "
                                "WHERE l.node=? AND l.state='draining' AND w.ended IS NULL", (self.node_id,)).fetchall()
        count = 0
        for row in rows:
            work = WorkLease(row["id"], row["lease_id"], row["fence"])
            if confirm_stopped(work) is True:
                with self._transaction() as (conn, now):
                    before = conn.execute("SELECT ended FROM work WHERE lease_id=? AND id=?",
                                          (work.lease_id, work.work_id)).fetchone()[0]
                    self._end(conn, work, now)
                    self._cleanup(conn, now)
                    count += before is None
        return count

    def guard(self, lease_id, fencing_token, *, principal, ring_id, model_cohort_sha256):
        lease = self.get(lease_id, principal=principal)
        if (lease.fencing_token != fencing_token or lease.ring_id != ring_id or
                lease.model_cohort_sha256 != model_cohort_sha256):
            raise LeaseFenceError("guard binding differs from ledger")
        return LeaseGuard(self, lease, principal)

    def resident_work(self, lease_id, *, principal):
        """Local runner recovery query, never proof that a vanished process exited."""
        with closing(self._connect()) as conn:
            lease = self._read(conn, lease_id)
            subject = self._subject(principal, "get", lease.to_dict())
            if subject != lease.controller_id or lease.node_id != self.node_id:
                raise LeaseAuthorizationError("resident work belongs to another controller/node")
            rows = conn.execute("SELECT id,fence FROM work WHERE lease_id=? AND ended IS NULL AND id LIKE 'resident-stage-%'",
                                (lease_id,)).fetchall()
            return [WorkLease(row[0], lease_id, row[1]) for row in rows]


class LeaseGuard:
    """Reusable execution seam: begin once, assert during work, end after cleanup."""
    def __init__(self, ledger, lease, principal):
        self.ledger, self.principal = ledger, principal
        for name in ("lease_id", "fencing_token", "ring_id", "model_cohort_sha256", "node_id",
                     "gpu_uuid", "memory_domain_id", "resources"):
            setattr(self, name, getattr(lease, name))

    def assert_live(self):
        return self.ledger.assert_fence(self.lease_id, self.fencing_token, principal=self.principal,
                                       ring_id=self.ring_id, model_cohort_sha256=self.model_cohort_sha256)

    def begin_work(self, work_id=None):
        return self.ledger.begin_work(self.lease_id, self.fencing_token, principal=self.principal,
                                      work_id=work_id, ring_id=self.ring_id, model_cohort_sha256=self.model_cohort_sha256)

    def end_work(self, work):
        if work.lease_id != self.lease_id or work.fencing_token != self.fencing_token:
            raise LeaseFenceError("work does not belong to this guard")
        return self.ledger.end_work(work, principal=self.principal)

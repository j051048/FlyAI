"""Artifact preparation reservations sharing the node GPU lease SQLite ledger.

No GPU lease, old ring or model file is removed here. Expired active preparation
continues to occupy disk/RAM until local worker cleanup is acknowledged.
"""
from dataclasses import asdict, dataclass
from dataclasses import replace
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


def _api():
    # Deferred to permit LeaseLedger to inherit this mixin without import cycles.
    from . import leases
    return leases


@dataclass(frozen=True)
class PrepareResources:
    filesystem_id: str
    disk_peak_bytes: int
    ram_bytes: int
    pinned_bytes: int = 0

    def __post_init__(self):
        api = _api()
        api._text(self.filesystem_id, "filesystem_id")
        for name in ("disk_peak_bytes", "ram_bytes", "pinned_bytes"):
            api._bytes(getattr(self, name), name)
        if self.pinned_bytes > self.ram_bytes:
            raise api.LeaseError("prepare pinned bytes must be a subset of RAM")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"filesystem_id", "disk_peak_bytes", "ram_bytes", "pinned_bytes"}:
            raise _api().LeaseError("exact prepare resource fields required")
        return cls(**value)


@dataclass(frozen=True)
class PrepareRequest:
    artifact_id: str
    model_cohort_sha256: str
    node_id: str
    memory_domain_id: str
    resources: PrepareResources
    ttl_s: float = 120

    def __post_init__(self):
        api = _api()
        for name in ("artifact_id", "node_id", "memory_domain_id"):
            api._text(getattr(self, name), name)
        api._cohort(self.model_cohort_sha256)
        api._ttl(self.ttl_s)
        if not isinstance(self.resources, PrepareResources):
            raise api.LeaseError("validated prepare resources required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        expected = {"artifact_id", "model_cohort_sha256", "node_id", "memory_domain_id", "resources", "ttl_s"}
        if not isinstance(value, dict) or set(value) != expected:
            raise _api().LeaseError("exact artifact prepare request required")
        body = dict(value)
        body["resources"] = PrepareResources.from_dict(body["resources"])
        return cls(**body)


@dataclass(frozen=True)
class PrepareLease:
    lease_id: str
    fencing_token: int
    controller_id: str
    artifact_id: str
    model_cohort_sha256: str
    node_id: str
    memory_domain_id: str
    resources: PrepareResources
    state: str
    expires_at: float
    active_work: int

    def to_dict(self):
        return asdict(self)


class ArtifactPrepareGuard:
    def __init__(self, ledger, lease, principal):
        self.ledger, self.principal = ledger, principal
        for name in ("lease_id", "fencing_token", "artifact_id", "model_cohort_sha256", "node_id", "memory_domain_id", "resources"):
            setattr(self, name, getattr(lease, name))

    def assert_live(self):
        return self.ledger.artifact_assert(self.lease_id, self.fencing_token, principal=self.principal)

    def begin_work(self, work_id=None):
        return self.ledger.artifact_begin(self.lease_id, self.fencing_token, principal=self.principal, work_id=work_id)

    def end_work(self, work):
        if work.lease_id != self.lease_id or work.fencing_token != self.fencing_token:
            raise _api().LeaseFenceError("artifact work differs from reservation")
        return self.ledger.artifact_end(work, principal=self.principal)

    def renew(self, *, ttl_s, idempotency_key):
        return self.ledger.artifact_renew(self.lease_id, self.fencing_token, principal=self.principal,
                                          ttl_s=ttl_s, idempotency_key=idempotency_key)

    def release(self):
        return self.ledger.artifact_release(self.lease_id, self.fencing_token, principal=self.principal)

    def finish(self, verified):
        return self.ledger.artifact_finish(self.lease_id, self.fencing_token, verified, principal=self.principal)


class ArtifactPreparationLedgerMixin:
    @staticmethod
    def _initialize_artifact_tables(conn):
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS filesystems(id TEXT PRIMARY KEY,path TEXT NOT NULL,physical TEXT NOT NULL UNIQUE,
                capacity INTEGER NOT NULL,fence INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS artifact_preparations(id TEXT PRIMARY KEY,fence INTEGER NOT NULL,controller TEXT NOT NULL,
                node TEXT NOT NULL,artifact TEXT NOT NULL,cohort TEXT NOT NULL,domain_id TEXT NOT NULL REFERENCES domains(id),
                filesystem_id TEXT NOT NULL REFERENCES filesystems(id),resources TEXT NOT NULL,state TEXT NOT NULL,expires REAL NOT NULL,
                active INTEGER NOT NULL DEFAULT 0,drain_reason TEXT);
            CREATE TABLE IF NOT EXISTS artifact_operations(node TEXT NOT NULL,controller TEXT NOT NULL,action TEXT NOT NULL,
                key TEXT NOT NULL,fingerprint TEXT NOT NULL,reservation TEXT NOT NULL REFERENCES artifact_preparations(id),
                PRIMARY KEY(node,controller,action,key));
            CREATE TABLE IF NOT EXISTS artifact_work(reservation TEXT NOT NULL REFERENCES artifact_preparations(id),id TEXT NOT NULL,
                fence INTEGER NOT NULL,started REAL NOT NULL,ended REAL,PRIMARY KEY(reservation,id));
            CREATE TABLE IF NOT EXISTS artifact_cache(filesystem_id TEXT NOT NULL REFERENCES filesystems(id),directory TEXT NOT NULL,
                artifact TEXT NOT NULL,source_artifact TEXT NOT NULL,bytes INTEGER NOT NULL,
                PRIMARY KEY(filesystem_id,directory));
        """)

    @staticmethod
    def _cleanup_artifacts(conn, now):
        conn.execute("UPDATE artifact_preparations SET state=CASE WHEN active>0 THEN 'draining' ELSE 'expired' END,"
                     "drain_reason='expired' WHERE state='prepared' AND expires<=?", (now,))
        conn.execute("UPDATE artifact_preparations SET state=drain_reason WHERE state='draining' AND active=0")

    @staticmethod
    def _prepare_host_usage(conn, domain):
        rows = conn.execute("SELECT resources FROM artifact_preparations WHERE domain_id=? AND state IN ('prepared','draining')",
                            (domain,)).fetchall()
        values = [PrepareResources.from_dict(json.loads(row[0])) for row in rows]
        return sum(row.ram_bytes for row in values), sum(row.pinned_bytes for row in values)

    def register_filesystem(self, filesystem_id, path, *, available_disk_bytes=None):
        """Register current free bytes on a locally configured target filesystem.

        Existing charged cache bytes are added back to the accounting baseline:
        disk free already excludes them. Never derive this identity from an IP.
        """
        api = _api()
        api._text(filesystem_id, "filesystem_id")
        target = Path(path).expanduser().resolve(strict=True)
        if not target.is_dir():
            raise api.LeaseError("filesystem target must be an existing local directory")
        observed = shutil.disk_usage(target).free
        if available_disk_bytes is not None:
            api._bytes(available_disk_bytes, "available_disk_bytes")
            observed = min(observed, available_disk_bytes)
        physical = f"{os.name}:{target.stat().st_dev}:{target.anchor.casefold()}"
        with self._transaction() as (conn, now):
            other = conn.execute("SELECT id FROM filesystems WHERE physical=?", (physical,)).fetchone()
            if other is not None and other[0] != filesystem_id:
                raise api.LeaseConflict("one physical filesystem must use one shared filesystem_id")
            existing = conn.execute("SELECT * FROM filesystems WHERE id=?", (filesystem_id,)).fetchone()
            if existing is not None and (existing["physical"] != physical or existing["path"] != str(target)):
                raise api.LeaseConflict("cannot change a registered filesystem identity/path")
            cached = conn.execute("SELECT COALESCE(SUM(bytes),0) FROM artifact_cache WHERE filesystem_id=?", (filesystem_id,)).fetchone()[0]
            reserved = [PrepareResources.from_dict(json.loads(row[0])).disk_peak_bytes for row in conn.execute(
                "SELECT resources FROM artifact_preparations WHERE filesystem_id=? AND state IN ('prepared','draining')", (filesystem_id,))]
            if sum(reserved) > observed:
                raise api.LeaseConflict("filesystem capacity update undercuts active preparation reservations")
            conn.execute("INSERT INTO filesystems(id,path,physical,capacity) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET capacity=excluded.capacity",
                         (filesystem_id, str(target), physical, observed + cached))

    def _artifact_owned(self, conn, reservation, fence, principal, action):
        api = _api()
        row = conn.execute("SELECT * FROM artifact_preparations WHERE id=?", (api._text(reservation, "reservation_id"),)).fetchone()
        if row is None:
            raise api.LeaseError("artifact reservation not found")
        lease = PrepareLease(row["id"], row["fence"], row["controller"], row["artifact"], row["cohort"], row["node"],
            row["domain_id"], PrepareResources.from_dict(json.loads(row["resources"])), row["state"], row["expires"], row["active"])
        subject = self._subject(principal, action, lease.to_dict())
        if lease.node_id != self.node_id or subject != lease.controller_id:
            raise api.LeaseAuthorizationError("artifact reservation belongs to another authenticated node/controller")
        if type(fence) is not int or fence != lease.fencing_token:
            raise api.LeaseFenceError("artifact preparation fencing token differs")
        return lease

    @staticmethod
    def _artifact_live(lease, now):
        if lease.state != "prepared" or lease.expires_at <= now:
            raise _api().LeaseExpired("artifact preparation expired, retired or awaits cleanup")

    @staticmethod
    def _publication(verified):
        from .weight_artifacts import verified_stage_artifact_descriptor, GLOBAL_FILE, PACK_FILE, STAGE_FILE, safe_path
        descriptor = verified_stage_artifact_descriptor(verified)
        directory = Path(verified["directory"]).resolve(strict=True)
        files = {row["path"]: row["size"] for row in descriptor["files"]}
        for name in (GLOBAL_FILE, PACK_FILE, STAGE_FILE):
            path = safe_path(directory, name)
            if path.is_file():
                files.setdefault(name, path.stat().st_size)
        for name, size in files.items():
            if safe_path(directory, name).stat().st_size != size:
                raise _api().LeaseConflict("verified artifact changed before disk accounting")
        return descriptor, directory, sum(files.values())

    def adopt_artifact(self, request, verified, *, principal, idempotency_key):
        """Local-only full-hash cache adoption: existing bytes are not a new write.

        A JSON claim, pathname or RPC cleanup flag can never authorize a credit.
        Publication identity/stat snapshots must be verified before entering here.
        """
        return self.prepare_artifact(request, principal=principal, idempotency_key=idempotency_key,
                                     _verified_adoption=verified)

    def prepare_artifact(self, request, *, principal, idempotency_key, _verified_adoption=None):
        api = _api()
        if not isinstance(request, PrepareRequest) or request.node_id != self.node_id:
            raise api.LeaseError("artifact prepare request must target this node")
        adoption = self._publication(_verified_adoption) if _verified_adoption is not None else None
        original = request.to_dict()
        if adoption is not None:
            request = replace(request, resources=replace(request.resources, disk_peak_bytes=0))
            original.update(adopted_directory=str(adoption[1]), adopted_artifact=adoption[0]["artifact_id"])
        subject = self._subject(principal, "prepare_artifact", original)
        api._text(idempotency_key, "idempotency_key")
        fingerprint = hashlib.sha256(api._json(original).encode()).hexdigest()
        with self._transaction() as (conn, now):
            prior = conn.execute("SELECT * FROM artifact_operations WHERE node=? AND controller=? AND action='prepare' AND key=?",
                                 (self.node_id, subject, idempotency_key)).fetchone()
            if prior is not None:
                if prior["fingerprint"] != fingerprint:
                    raise api.LeaseConflict("idempotency key names a different artifact preparation")
                row = conn.execute("SELECT fence FROM artifact_preparations WHERE id=?", (prior["reservation"],)).fetchone()
                lease = self._artifact_owned(conn, prior["reservation"], row[0], principal, "prepare_artifact")
                self._artifact_live(lease, now)
                return ArtifactPrepareGuard(self, lease, principal)
            fs = conn.execute("SELECT * FROM filesystems WHERE id=?", (request.resources.filesystem_id,)).fetchone()
            domain = conn.execute("SELECT * FROM domains WHERE id=?", (request.memory_domain_id,)).fetchone()
            if fs is None or domain is None:
                raise api.LeaseConflict("locally measured filesystem/memory domain is absent")
            if adoption is not None:
                descriptor, directory, amount = adoption
                if not directory.is_relative_to(Path(fs["path"])) or directory.stat().st_dev != Path(fs["path"]).stat().st_dev:
                    raise api.LeaseConflict("adopted artifact differs from reserved local filesystem")
                held = conn.execute("SELECT * FROM artifact_cache WHERE filesystem_id=? AND directory=?", (fs["id"], str(directory))).fetchone()
                if held is not None and (held["artifact"] != descriptor["artifact_id"] or held["bytes"] != amount):
                    raise api.LeaseConflict("existing cache identity/bytes changed")
                if held is None:
                    # These bytes pre-exist and are already absent from observed
                    # free space. Add back before recording their charge once.
                    conn.execute("UPDATE filesystems SET capacity=capacity+? WHERE id=?", (amount, fs["id"]))
                    conn.execute("INSERT INTO artifact_cache VALUES(?,?,?,?,?)", (fs["id"], str(directory), descriptor["artifact_id"], request.artifact_id, amount))
                    fs = conn.execute("SELECT * FROM filesystems WHERE id=?", (fs["id"],)).fetchone()
            cached = conn.execute("SELECT COALESCE(SUM(bytes),0) FROM artifact_cache WHERE filesystem_id=?", (fs["id"],)).fetchone()[0]
            disks = [PrepareResources.from_dict(json.loads(row[0])).disk_peak_bytes for row in conn.execute(
                "SELECT resources FROM artifact_preparations WHERE filesystem_id=? AND state IN ('prepared','draining')", (fs["id"],))]
            # Existing .partial files, even after failed workers/restart, remain
            # visible in actual free space. Full peaks may double-count written
            # active bytes conservatively; they can never create phantom space.
            current_free = shutil.disk_usage(fs["path"]).free
            available = min(current_free, max(0, fs["capacity"] - cached))
            if request.resources.disk_peak_bytes + sum(disks) > available:
                raise api.LeaseConflict("shared disk preparation capacity is insufficient")
            runtime = [api.LeaseResources.from_dict(json.loads(row[0])) for row in conn.execute(
                "SELECT resources FROM leases WHERE domain_id=? AND state IN ('prepared','committed','draining')", (request.memory_domain_id,))]
            host, pinned = self._prepare_host_usage(conn, request.memory_domain_id)
            for label, demand, limit in (("RAM", request.resources.ram_bytes + host + sum(r.ram_bytes for r in runtime), domain["ram"]),
                                        ("pinned", request.resources.pinned_bytes + pinned + sum(r.pinned_bytes for r in runtime), domain["pinned"])):
                if demand and (limit is None or demand > limit):
                    raise api.LeaseConflict(f"shared prepare {label} capacity is unknown or insufficient")
            reservation, fence = "artifact-prepare-" + uuid.uuid4().hex, fs["fence"] + 1
            conn.execute("UPDATE filesystems SET fence=? WHERE id=?", (fence, fs["id"]))
            conn.execute("INSERT INTO artifact_preparations VALUES(?,?,?,?,?,?,?,?,?,'prepared',?,0,NULL)",
                (reservation, fence, subject, self.node_id, request.artifact_id, request.model_cohort_sha256,
                 request.memory_domain_id, fs["id"], api._json(request.resources.to_dict()), now + request.ttl_s))
            conn.execute("INSERT INTO artifact_operations VALUES(?,?,'prepare',?,?,?)", (self.node_id, subject, idempotency_key, fingerprint, reservation))
            return ArtifactPrepareGuard(self, self._artifact_owned(conn, reservation, fence, principal, "prepare_artifact"), principal)

    def artifact_assert(self, reservation, fence, *, principal):
        with self._transaction() as (conn, now):
            lease = self._artifact_owned(conn, reservation, fence, principal, "artifact_assert")
            self._artifact_live(lease, now)
            return lease

    def artifact_begin(self, reservation, fence, *, principal, work_id=None):
        api = _api()
        work_id = api._text(work_id or "prepare-work-" + uuid.uuid4().hex, "work_id")
        with self._transaction() as (conn, now):
            lease = self._artifact_owned(conn, reservation, fence, principal, "artifact_begin")
            self._artifact_live(lease, now)
            previous = conn.execute("SELECT ended FROM artifact_work WHERE reservation=? AND id=?", (reservation, work_id)).fetchone()
            if previous is not None:
                if previous[0] is not None:
                    raise api.LeaseConflict("artifact work ID already ended")
            else:
                conn.execute("INSERT INTO artifact_work VALUES(?,?,?,?,NULL)", (reservation, work_id, fence, now))
                conn.execute("UPDATE artifact_preparations SET active=active+1 WHERE id=?", (reservation,))
            return api.WorkLease(work_id, reservation, fence)

    def artifact_end(self, work, *, principal):
        api = _api()
        if not isinstance(work, api.WorkLease):
            raise api.LeaseError("actual artifact WorkLease required")
        with self._transaction() as (conn, now):
            self._artifact_owned(conn, work.lease_id, work.fencing_token, principal, "artifact_end")
            row = conn.execute("SELECT fence,ended FROM artifact_work WHERE reservation=? AND id=?", (work.lease_id, work.work_id)).fetchone()
            if row is None or row[0] != work.fencing_token:
                raise api.LeaseFenceError("unknown artifact work handle")
            if row[1] is None:
                conn.execute("UPDATE artifact_work SET ended=? WHERE reservation=? AND id=?", (now, work.lease_id, work.work_id))
                conn.execute("UPDATE artifact_preparations SET active=active-1 WHERE id=? AND active>0", (work.lease_id,))
            self._cleanup_artifacts(conn, now)
            return self._artifact_owned(conn, work.lease_id, work.fencing_token, principal, "artifact_end")

    def artifact_renew(self, reservation, fence, *, principal, ttl_s, idempotency_key):
        api = _api()
        ttl_s = api._ttl(ttl_s)
        api._text(idempotency_key, "idempotency_key")
        with self._transaction() as (conn, now):
            lease = self._artifact_owned(conn, reservation, fence, principal, "artifact_renew")
            body = {"reservation": reservation, "fence": fence, "ttl_s": ttl_s}
            fingerprint = hashlib.sha256(api._json(body).encode()).hexdigest()
            prior = conn.execute("SELECT * FROM artifact_operations WHERE node=? AND controller=? AND action='renew' AND key=?",
                                 (self.node_id, lease.controller_id, idempotency_key)).fetchone()
            if prior is not None:
                if prior["fingerprint"] != fingerprint:
                    raise api.LeaseConflict("artifact renewal idempotency key differs")
                return lease
            self._artifact_live(lease, now)
            conn.execute("UPDATE artifact_preparations SET expires=MAX(expires,?) WHERE id=?", (now + ttl_s, reservation))
            conn.execute("INSERT INTO artifact_operations VALUES(?,?,'renew',?,?,?)", (self.node_id, lease.controller_id, idempotency_key, fingerprint, reservation))
            return self._artifact_owned(conn, reservation, fence, principal, "artifact_renew")

    def artifact_release(self, reservation, fence, *, principal):
        with self._transaction() as (conn, now):
            lease = self._artifact_owned(conn, reservation, fence, principal, "artifact_release")
            if lease.state == "prepared":
                conn.execute("UPDATE artifact_preparations SET state=?,drain_reason='released' WHERE id=?",
                             ("draining" if lease.active_work else "released", reservation))
            return self._artifact_owned(conn, reservation, fence, principal, "artifact_release")

    def artifact_finish(self, reservation, fence, verified, *, principal):
        api = _api()
        descriptor, directory, amount = self._publication(verified)
        with self._transaction() as (conn, now):
            lease = self._artifact_owned(conn, reservation, fence, principal, "artifact_finish")
            if lease.active_work:
                raise api.LeaseConflict("artifact worker cleanup must finish before publication accounting")
            fs = conn.execute("SELECT * FROM filesystems WHERE id=?", (lease.resources.filesystem_id,)).fetchone()
            if not directory.is_relative_to(Path(fs["path"])) or directory.stat().st_dev != Path(fs["path"]).stat().st_dev:
                raise api.LeaseConflict("artifact was published outside the reserved filesystem")
            prior = conn.execute("SELECT * FROM artifact_cache WHERE filesystem_id=? AND directory=?", (fs["id"], str(directory))).fetchone()
            if prior is not None and (prior["artifact"] != descriptor["artifact_id"] or prior["bytes"] != amount):
                raise api.LeaseConflict("immutable cached artifact identity/bytes changed")
            if lease.state == "finished" and prior is not None:
                return lease
            self._artifact_live(lease, now)
            if prior is None and amount > lease.resources.disk_peak_bytes:
                raise api.LeaseConflict("published files exceed the reserved disk peak")
            conn.execute("INSERT OR IGNORE INTO artifact_cache VALUES(?,?,?,?,?)", (fs["id"], str(directory), descriptor["artifact_id"], lease.artifact_id, amount))
            conn.execute("UPDATE artifact_preparations SET state='finished' WHERE id=?", (reservation,))
            return self._artifact_owned(conn, reservation, fence, principal, "artifact_finish")

    def recover_stopped_artifact_work(self, confirm_stopped):
        """Local callback proves each worker stopped; restart/remote flags cannot.

        An in-process preparer can hold its exclusive OS job lock and match an
        exact work-ID namespace. Unknown/detached worker lifetimes must refuse.
        """
        api = _api()
        if not callable(confirm_stopped):
            raise api.LeaseError("local artifact worker cleanup verifier required")
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT w.* FROM artifact_work w JOIN artifact_preparations p ON p.id=w.reservation "
                                "WHERE p.node=? AND w.ended IS NULL", (self.node_id,)).fetchall()
        count = 0
        for row in rows:
            work = api.WorkLease(row["id"], row["reservation"], row["fence"])
            if confirm_stopped(work) is not True:
                continue
            with self._transaction() as (conn, now):
                current = conn.execute("SELECT ended FROM artifact_work WHERE reservation=? AND id=?", (work.lease_id, work.work_id)).fetchone()
                if current[0] is None:
                    conn.execute("UPDATE artifact_work SET ended=? WHERE reservation=? AND id=?", (now, work.lease_id, work.work_id))
                    conn.execute("UPDATE artifact_preparations SET active=active-1 WHERE id=? AND active>0", (work.lease_id,))
                    count += 1
                conn.execute("UPDATE artifact_preparations SET state=CASE WHEN active>0 THEN 'draining' ELSE 'released' END,"
                             "drain_reason='released' WHERE id=? AND state IN ('prepared','draining','expired')", (work.lease_id,))
        return count

"""Private, bounded SQLite job log and coordinator ownership fencing.

Local/shared-file recovery, not a distributed consensus service or KV migration.
Each observed token is committed before publication. Stable job/sequence IDs
support client deduplication; delivery over a broken network is not exactly once.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import threading
import time

SCHEMA = "shard-job-journal/1"
TERMINAL = ("completed", "failed", "cancelled", "expired")


class JournalError(RuntimeError):
    pass


class JournalOwnerLost(JournalError):
    pass


class JournalBusy(JournalError):
    pass


class JournalOwnerBusy(JournalError):
    pass


class JournalCapacity(JournalError):
    pass


class JournalConflict(JournalError):
    pass


class JournalDeadline(JournalConflict):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _text(value, name, limit=512):
    if not isinstance(value, str) or not 1 <= len(value) <= limit:
        raise JournalError(f"bounded nonempty {name} required")
    return value


def _number(value, name, positive=True):
    if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise JournalError(f"invalid {name}")
    return float(value)


def _windows_acl(path, *, create=False):
    """Current-account protected DACL; never chmod an existing public directory."""
    import ctypes
    from ctypes import wintypes as W
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = W.HANDLE
    advapi.OpenProcessToken.argtypes = [W.HANDLE, W.DWORD, ctypes.POINTER(W.HANDLE)]
    advapi.GetTokenInformation.argtypes = [W.HANDLE, W.DWORD, ctypes.c_void_p, W.DWORD, ctypes.POINTER(W.DWORD)]
    advapi.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(W.LPWSTR)]
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [W.LPCWSTR, W.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    advapi.GetSecurityDescriptorDacl.argtypes = [ctypes.c_void_p, ctypes.POINTER(W.BOOL), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(W.BOOL)]
    advapi.SetNamedSecurityInfoW.argtypes = [W.LPWSTR, W.DWORD, W.DWORD, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    advapi.GetNamedSecurityInfoW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [ctypes.c_void_p, W.DWORD, W.DWORD, ctypes.POINTER(W.LPWSTR), ctypes.c_void_p]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [W.HANDLE]
    token, needed = W.HANDLE(), W.DWORD()
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
        raise JournalError("cannot query Windows account for journal access policy")
    try:
        advapi.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
        buffer = ctypes.create_string_buffer(needed.value)
        if not advapi.GetTokenInformation(token, 1, buffer, needed, ctypes.byref(needed)):
            raise JournalError("cannot query Windows journal account SID")
        sid_pointer = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        sid_string = W.LPWSTR()
        if not advapi.ConvertSidToStringSidW(sid_pointer, ctypes.byref(sid_string)):
            raise JournalError("cannot encode Windows journal account SID")
        sid = sid_string.value
        kernel.LocalFree(sid_string)
    finally:
        kernel.CloseHandle(token)
    descriptor = ctypes.c_void_p()
    if create:
        if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                f"D:P(A;OICI;FA;;;{sid})", 1, ctypes.byref(descriptor), None):
            raise JournalError("cannot construct private journal ACL")
        try:
            present, defaulted, dacl = W.BOOL(), W.BOOL(), ctypes.c_void_p()
            if not advapi.GetSecurityDescriptorDacl(descriptor, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)):
                raise JournalError("cannot read private journal ACL")
            if advapi.SetNamedSecurityInfoW(str(path), 1, 4 | 0x80000000, None, None, dacl, None):
                raise JournalError("cannot apply private journal directory ACL")
        finally:
            kernel.LocalFree(descriptor)
    if advapi.GetNamedSecurityInfoW(str(path), 1, 4, None, None, None, None, ctypes.byref(descriptor)):
        raise JournalError("cannot inspect journal ACL")
    output = W.LPWSTR()
    try:
        if not advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW(descriptor, 1, 4, ctypes.byref(output), None):
            raise JournalError("cannot inspect journal ACL entries")
        sddl = output.value
        allowed = re.findall(r"\(A[^)]*;;;([^)]*)\)", sddl)
        if not allowed or any(value != sid for value in allowed):
            raise JournalError("journal directory/files must grant access only to the current Windows account")
    finally:
        if output:
            kernel.LocalFree(output)
        kernel.LocalFree(descriptor)


def _private_path(value):
    path = Path(value).absolute()
    if str(value) == ":memory:":
        raise JournalError("job journal requires a persistent private file")
    for ancestor in (path, *path.parents):
        if ancestor.exists() and (ancestor.is_symlink() or getattr(ancestor.lstat(), "st_file_attributes", 0) & 0x400):
            raise JournalError("journal path must not contain symlinks/reparse points")
    existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "nt":
        _windows_acl(path.parent, create=not existed)
    else:
        info = path.parent.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise JournalError("journal requires a current-user private directory (0700)")
    if not path.exists():
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    elif not path.is_file():
        raise JournalError("journal path must be a regular file")
    if os.name == "nt":
        _windows_acl(path)
    else:
        info = path.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise JournalError("journal file must have current-user permissions 0600")
    return path


@dataclass(frozen=True)
class OwnerLease:
    namespace: str
    owner_id: str
    token: str = field(repr=False)
    fence: int
    expires_at: float


class JobJournal:
    def __init__(self, path, *, namespace="service", max_jobs=256, max_bytes=64 << 20,
                 max_payload_bytes=1 << 20, max_result_bytes=2 << 20, max_tokens=16384,
                 max_recovery_attempts=8, retention_s=600, busy_timeout_s=5, clock=time.time):
        self.namespace = _text(namespace, "namespace", 128)
        for name, value in (("max_jobs", max_jobs), ("max_bytes", max_bytes), ("max_payload_bytes", max_payload_bytes),
                            ("max_result_bytes", max_result_bytes), ("max_tokens", max_tokens), ("max_recovery_attempts", max_recovery_attempts)):
            if type(value) is not int or value < 1:
                raise JournalError(f"{name} must be positive")
            setattr(self, name, value)
        self.retention_s = _number(retention_s, "retention")
        self.busy_timeout_s = _number(busy_timeout_s, "SQLite timeout")
        self.clock, self.path = clock, _private_path(path)
        self._publication = threading.local()
        self._metrics_lock = threading.Lock()
        self._io_metrics = {"transactions": 0, "transaction_seconds": 0.0, "token_commits": 0,
                            "token_commit_seconds": 0.0}
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA max_page_count={max(32, max_bytes // 4096)}")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS journal_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS owner(namespace TEXT PRIMARY KEY,id TEXT,token TEXT,fence INTEGER NOT NULL,
                    expires REAL NOT NULL,clock_floor REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,tenant TEXT NOT NULL,idem TEXT,
                    fingerprint TEXT NOT NULL,payload TEXT NOT NULL,context TEXT NOT NULL,prompt INTEGER NOT NULL,
                    maximum INTEGER NOT NULL,created REAL NOT NULL,deadline REAL NOT NULL,state TEXT NOT NULL,
                    seq INTEGER NOT NULL DEFAULT 0,n INTEGER NOT NULL DEFAULT 0,attempts INTEGER NOT NULL DEFAULT 0,
                    owner_fence INTEGER NOT NULL,cancel_kind TEXT,result TEXT,error TEXT,error_code TEXT,
                    finished REAL,expires REAL,charge INTEGER NOT NULL);
                CREATE UNIQUE INDEX IF NOT EXISTS job_idempotency ON jobs(tenant,idem) WHERE idem IS NOT NULL;
                CREATE TABLE IF NOT EXISTS tokens(job TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    offset INTEGER NOT NULL,value INTEGER NOT NULL,PRIMARY KEY(job,offset));
                CREATE TABLE IF NOT EXISTS admissions(job TEXT PRIMARY KEY,tenant TEXT NOT NULL,created REAL NOT NULL,charge INTEGER NOT NULL);
            """)
            conn.execute("INSERT OR IGNORE INTO journal_meta VALUES('schema',?)", (SCHEMA,))
            conn.execute("INSERT OR IGNORE INTO journal_meta VALUES('namespace',?)", (self.namespace,))
            if dict(conn.execute("SELECT key,value FROM journal_meta")) != {"schema": SCHEMA, "namespace": self.namespace}:
                raise JournalError("journal schema/namespace mismatch")
            conn.execute("INSERT OR IGNORE INTO owner VALUES(?,NULL,NULL,0,0,0)", (self.namespace,))
        finally:
            conn.close()

    def _connect(self):
        conn = None
        try:
            wal = Path(str(self.path) + "-wal")
            if wal.exists() and wal.stat().st_size > self.max_bytes * 2 + (8 << 20):
                raise JournalCapacity("journal WAL exceeded bounded safety capacity")
            # Never silently recreate a deleted journal while retaining an old
            # in-memory owner handle; recovery needs the original file authority.
            conn = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True,
                                   timeout=self.busy_timeout_s, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA secure_delete=ON")
            conn.execute("PRAGMA wal_autocheckpoint=128")
            conn.execute(f"PRAGMA max_page_count={max(32, self.max_bytes // 4096)}")
            return conn
        except sqlite3.Error as error:
            if conn is not None:
                conn.close()
            if getattr(error, "sqlite_errorcode", None) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise JournalBusy("private journal temporarily busy") from error
            raise JournalError("private journal unavailable") from error

    @contextmanager
    def _transaction(self, lease=None):
        started = time.perf_counter()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM owner WHERE namespace=?", (self.namespace,)).fetchone()
            now = max(_number(self.clock(), "wall clock", positive=False), row["clock_floor"])
            conn.execute("UPDATE owner SET clock_floor=? WHERE namespace=?", (now, self.namespace))
            conn.execute("SAVEPOINT operation")
            try:
                if lease is not None:
                    self._check(row, lease, now)
                yield conn, now
                conn.execute("RELEASE operation")
            except Exception:
                conn.execute("ROLLBACK TO operation")
                conn.execute("RELEASE operation")
                conn.execute("COMMIT")  # Preserve clock floor even on rejected stale writes.
                raise
            conn.execute("COMMIT")
        except sqlite3.Error as error:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            if getattr(error, "sqlite_errorcode", None) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise JournalBusy("private journal transaction temporarily busy") from error
            raise JournalError("private journal transaction unavailable") from error
        finally:
            conn.close()
            with self._metrics_lock:
                self._io_metrics["transactions"] += 1
                self._io_metrics["transaction_seconds"] += time.perf_counter() - started

    def metrics(self):
        with self._metrics_lock:
            return {**self._io_metrics, "scope": "this process; wall time includes SQLite wait/commit and publication lock holding; not isolated fsync cost"}

    def _check(self, row, lease, now):
        if (not isinstance(lease, OwnerLease) or lease.namespace != self.namespace or row["id"] != lease.owner_id
                or row["token"] != lease.token or row["fence"] != lease.fence or now >= row["expires"]):
            raise JournalOwnerLost("durable coordinator ownership expired or was fenced")

    def claim_owner(self, *, ttl_s=30, owner_id=None):
        ttl = _number(ttl_s, "owner TTL")
        ident, secret = _text(owner_id or secrets.token_hex(16), "owner identity", 128), secrets.token_hex(32)
        with self._transaction() as (conn, now):
            row = conn.execute("SELECT * FROM owner WHERE namespace=?", (self.namespace,)).fetchone()
            if row["id"] is not None and now < row["expires"]:
                raise JournalOwnerBusy("another coordinator owns this journal")
            fence = row["fence"] + 1
            conn.execute("UPDATE owner SET id=?,token=?,fence=?,expires=? WHERE namespace=?",
                         (ident, secret, fence, now + ttl, self.namespace))
            return OwnerLease(self.namespace, ident, secret, fence, now + ttl)

    def assert_owner(self, lease):
        active = getattr(self._publication, "active", None)
        if active is not None:
            conn, started = active
            row = conn.execute("SELECT * FROM owner WHERE namespace=?", (self.namespace,)).fetchone()
            now = max(_number(self.clock(), "wall clock", positive=False), row["clock_floor"], started)
            self._check(row, lease, now)
            return True
        with self._transaction(lease):
            return True

    def renew(self, lease, *, ttl_s=30):
        ttl = _number(ttl_s, "owner TTL")
        with self._transaction(lease) as (conn, now):
            conn.execute("UPDATE owner SET expires=? WHERE namespace=?", (now + ttl, self.namespace))
            return OwnerLease(lease.namespace, lease.owner_id, lease.token, lease.fence, now + ttl)

    def release_owner(self, lease):
        with self._transaction(lease) as (conn, now):
            conn.execute("UPDATE owner SET id=NULL,token=NULL,expires=? WHERE namespace=?", (now, self.namespace))

    @contextmanager
    def publish_guard(self, lease):
        # A bounded HTTP write holds the ownership transaction. Takeover cannot
        # interleave a newer response while an older coordinator is publishing.
        if getattr(self._publication, "active", None) is not None:
            self.assert_owner(lease)
            yield
            return
        with self._transaction(lease) as (conn, now):
            self._publication.active = (conn, now)
            try:
                yield
            finally:
                self._publication.active = None

    def _prune(self, conn, now, max_terminal=None):
        conn.execute("DELETE FROM admissions WHERE created<=?", (now - 60,))
        conn.execute("DELETE FROM jobs WHERE expires IS NOT NULL AND expires<=?", (now,))
        if max_terminal is not None:
            conn.execute("DELETE FROM jobs WHERE id IN (SELECT id FROM jobs WHERE finished IS NOT NULL ORDER BY finished DESC,id DESC LIMIT -1 OFFSET ?)", (max_terminal,))

    def prune(self, lease, *, max_terminal=None):
        with self._transaction(lease) as (conn, now):
            before = {row[0] for row in conn.execute("SELECT id FROM jobs")}
            self._prune(conn, now, max_terminal)
            return before - {row[0] for row in conn.execute("SELECT id FROM jobs")}

    def _row(self, conn, row):
        if row is None:
            return None
        value = dict(row)
        for key in ("payload", "context", "result"):
            value[key] = json.loads(value[key]) if value[key] is not None else None
        tokens = list(conn.execute("SELECT offset,value FROM tokens WHERE job=? ORDER BY offset", (row["id"],)))
        value["tokens"] = [item[1] for item in tokens]
        if len(tokens) != row["n"] or [item[0] for item in tokens] != list(range(row["n"])):
            raise JournalError("durable prefix sequence is incomplete")
        return value

    def get(self, lease, *, tenant, job_id=None, idempotency_key=None):
        with self._transaction(lease) as (conn, now):
            self._prune(conn, now)
            row = (conn.execute("SELECT * FROM jobs WHERE tenant=? AND id=?", (tenant, job_id)).fetchone()
                   if job_id is not None else conn.execute("SELECT * FROM jobs WHERE tenant=? AND idem=?", (tenant, idempotency_key)).fetchone())
            return self._row(conn, row)

    def admit(self, lease, value):
        payload, context = canonical(value["payload"]), canonical(value["context"])
        size = len(payload.encode()) + len(context.encode())
        if size > self.max_payload_bytes or type(value["max_new"]) is not int or not 1 <= value["max_new"] <= self.max_tokens:
            raise JournalCapacity("durable payload/token budget exceeds configured limit")
        binding = value["context"]
        if (not isinstance(binding, dict) or not isinstance(binding.get("model_id"), str) or not binding["model_id"]
                or not isinstance(binding.get("cohort_id"), str) or not re.fullmatch("[0-9a-f]{64}", binding["cohort_id"])):
            raise JournalError("durable jobs require an exact model/cohort binding")
        with self._transaction(lease) as (conn, now):
            self._prune(conn, now)
            if value["idempotency_key"] is not None:
                prior = conn.execute("SELECT * FROM jobs WHERE tenant=? AND idem=?", (value["tenant"], value["idempotency_key"])).fetchone()
                if prior:
                    if prior["fingerprint"] != value["fingerprint"]:
                        raise JournalConflict("idempotency key names another durable request")
                    return self._row(conn, prior), False
            # Reserve the bounded final response and complete token log before
            # accepting: disk/logical capacity cannot fail only at finalization.
            charge = size + 4096 + value["max_new"] * 64 + self.max_result_bytes
            deadline = value.get("deadline_unix", now + value["timeout_s"])
            if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= now:
                raise JournalDeadline("request deadline expired before durable admission")
            count, used = conn.execute("SELECT COUNT(*),COALESCE(SUM(charge),0) FROM jobs").fetchone()
            if count >= self.max_jobs or used + charge > self.max_bytes or conn.execute("SELECT COUNT(*) FROM admissions").fetchone()[0] >= self.max_jobs * 16:
                raise JournalCapacity("durable journal capacity exceeded")
            conn.execute("INSERT INTO jobs(id,tenant,idem,fingerprint,payload,context,prompt,maximum,created,deadline,state,owner_fence,charge) VALUES(?,?,?,?,?,?,?,?,?,?,'queued',?,?)",
                (value["id"], value["tenant"], value["idempotency_key"], value["fingerprint"], payload, context,
                 value["prompt_tokens"], value["max_new"], now, deadline, lease.fence, charge))
            conn.execute("INSERT INTO admissions VALUES(?,?,?,?)", (value["id"], value["tenant"], now, value["prompt_tokens"] + value["max_new"]))
            return self._row(conn, conn.execute("SELECT * FROM jobs WHERE id=?", (value["id"],)).fetchone()), True

    def claim_jobs(self, lease):
        with self._transaction(lease) as (conn, now):
            self._prune(conn, now)
            rows = conn.execute("SELECT *,rowid AS arrival FROM jobs ORDER BY created,rowid").fetchall()
            for row in rows:
                if row["state"] not in TERMINAL and row["owner_fence"] != lease.fence:
                    conn.execute("UPDATE jobs SET state='recovering',owner_fence=?,seq=seq+1 WHERE id=?", (lease.fence, row["id"]))
            return [self._row(conn, row) for row in conn.execute("SELECT *,rowid AS arrival FROM jobs ORDER BY created,rowid")], now

    def rate_records(self, lease):
        with self._transaction(lease) as (conn, now):
            self._prune(conn, now)
            return [dict(row) for row in conn.execute("SELECT * FROM admissions ORDER BY created")], now

    def running(self, lease, job_id):
        with self._transaction(lease) as (conn, now):
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or row["owner_fence"] != lease.fence or row["state"] not in ("queued", "recovering"):
                raise JournalConflict("durable job cannot enter running")
            if row["cancel_kind"] or now >= row["deadline"]:
                raise JournalConflict("cancelled/expired durable job cannot run")
            if row["attempts"] >= self.max_recovery_attempts:
                raise JournalConflict("durable recovery attempt budget exhausted")
            conn.execute("UPDATE jobs SET state='running',seq=seq+1,attempts=attempts+1 WHERE id=?", (job_id,))

    def append_token(self, lease, job_id, offset, token):
        started = time.perf_counter()
        if type(token) is not int or not 0 <= token < (1 << 31):
            raise JournalError("invalid durable token")
        with self._transaction(lease) as (conn, now):
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if (row is None or row["owner_fence"] != lease.fence or row["state"] != "running"
                    or type(offset) is not int or offset != row["n"] or offset >= row["maximum"]):
                raise JournalConflict("durable token sequence/state mismatch")
            conn.execute("INSERT INTO tokens VALUES(?,?,?)", (job_id, offset, token))
            conn.execute("UPDATE jobs SET n=n+1,seq=seq+1 WHERE id=?", (job_id,))
        with self._metrics_lock:
            self._io_metrics["token_commits"] += 1
            self._io_metrics["token_commit_seconds"] += time.perf_counter() - started

    def request_cancel(self, lease, job_id, kind):
        if kind not in ("cancel", "deadline"):
            raise JournalError("invalid durable cancellation")
        with self._transaction(lease) as (conn, now):
            conn.execute("UPDATE jobs SET cancel_kind=?,seq=seq+1 WHERE id=? AND owner_fence=? AND finished IS NULL", (kind, job_id, lease.fence))

    def finish(self, lease, job_id, state, *, result=None, error=None, code=None):
        if state not in TERMINAL:
            raise JournalError("invalid durable terminal state")
        result_json = canonical(result) if result is not None else None
        extra = len(result_json.encode()) if result_json is not None else 0
        if extra > self.max_result_bytes:
            raise JournalCapacity("durable result exceeds configured bound")
        with self._transaction(lease) as (conn, now):
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or row["owner_fence"] != lease.fence:
                raise JournalOwnerLost("durable job belongs to another coordinator generation")
            if row["state"] in TERMINAL:
                return
            if state == "completed":
                if now >= row["deadline"]:
                    raise JournalDeadline("deadline won the durable terminal transition")
                if row["cancel_kind"]:
                    raise JournalConflict("cancellation/deadline won the durable terminal transition")
                tokens = [r[0] for r in conn.execute("SELECT value FROM tokens WHERE job=? ORDER BY offset", (job_id,))]
                if not isinstance(result, dict) or result.get("tokens") != tokens:
                    raise JournalConflict("durable result differs from committed prefix")
            context = {k: v for k, v in json.loads(row["context"]).items()
                       if k in ("model_id", "cohort_id", "request_fingerprint")}
            context_json = canonical(context)
            actual_charge = 4096 + 2 + len(context_json.encode()) + row["n"] * 64 + extra
            conn.execute("UPDATE jobs SET state=?,result=?,error=?,error_code=?,finished=?,expires=?,seq=seq+1,charge=?,payload='{}',context=? WHERE id=?",
                (state, result_json, str(error)[:1024] if error is not None else None,
                 str(code)[:128] if code is not None else None, now, now + self.retention_s,
                 actual_charge, context_json, job_id))

    def checkpoint(self):
        conn = self._connect()
        try:
            return conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
        finally:
            conn.close()

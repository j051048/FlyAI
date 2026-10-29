"""Opt-in host-wide CPU/copy concurrency and calling-thread affinity.

SQLite permits are shared by local GPU workers. Expiry prevents admission but
never reclaims active work; a crashed owner needs explicit stopped confirmation.
This is a local scheduling resource, not an offer or an execution attestation.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid


SCHEMA = "shard-host-compute-budget/1"


class HostBudgetError(RuntimeError):
    pass


class HostBudgetBusy(HostBudgetError):
    pass


def _integer(value, name, minimum=1, maximum=65536):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value or len(value) > 256 or "\x00" in value:
        raise ValueError(f"invalid host budget {name}")
    return value


def _seconds(value, name, maximum):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError(f"invalid {name}")
    return float(value)


def observed_cpu_ids():
    if hasattr(os, "sched_getaffinity"):
        try:
            return sorted(os.sched_getaffinity(0))
        except OSError:
            pass
    return list(range(os.cpu_count() or 1))


class HostComputeBudget:
    def __init__(self, path, *, host_id, cpu_capacity, control_reserve=1,
                 copy_capacity=1, allowed_cpu_ids=None, clock=time.time):
        _text(host_id, "host_id")
        _integer(cpu_capacity, "cpu_capacity")
        _integer(control_reserve, "control_reserve", 0)
        _integer(copy_capacity, "copy_capacity", 1, 64)
        if control_reserve >= cpu_capacity:
            raise ValueError("host CPU budget must retain control-thread headroom")
        cpus = observed_cpu_ids() if allowed_cpu_ids is None else list(allowed_cpu_ids)
        if (not cpus or any(type(x) is not int or x < 0 for x in cpus) or len(set(cpus)) != len(cpus)
                or len(cpus) < cpu_capacity):
            raise ValueError("CPU capacity requires an explicit available CPU inventory")
        self.path, self.clock = Path(path), clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.host_id, self.cpus = host_id, sorted(cpus)
        self.cpu_capacity, self.control_reserve = cpu_capacity, control_reserve
        self.worker_capacity = cpu_capacity - control_reserve
        self.copy_capacity = copy_capacity
        self.affinity_status = "not_requested"
        self._initialize()

    def _geometry(self):
        return {"schema": SCHEMA, "host_id": self.host_id,
                "cpu_capacity": self.cpu_capacity, "control_reserve": self.control_reserve,
                "worker_capacity": self.worker_capacity, "copy_capacity": self.copy_capacity,
                "allowed_cpu_ids": list(self.cpus), "scope": "shared_local_active_work"}

    def config(self):
        # A path-independent persisted identity prevents two separate databases
        # with identical limits from being described as one shared budget.
        return {**self._geometry(), "budget_id": self.budget_id}

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA busy_timeout=5000")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _initialize(self):
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS config (value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS owners (owner TEXT PRIMARY KEY, epoch TEXT NOT NULL, "
                "workers INTEGER NOT NULL, copies INTEGER NOT NULL, expires REAL NOT NULL, "
                "active_workers INTEGER NOT NULL DEFAULT 0, active_copies INTEGER NOT NULL DEFAULT 0, "
                "cpu_ids TEXT NOT NULL, strict_affinity INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS permits (id TEXT PRIMARY KEY, owner TEXT NOT NULL, "
                       "epoch TEXT NOT NULL, kind TEXT NOT NULL, slots INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS identity (id TEXT NOT NULL)")
            encoded = json.dumps(self._geometry(), sort_keys=True, separators=(",", ":"))
            row = db.execute("SELECT value FROM config").fetchone()
            if row is None:
                db.execute("INSERT INTO config VALUES (?)", (encoded,))
            elif row["value"] != encoded:
                raise HostBudgetError("shared host budget geometry changed; use the frozen host configuration")
            identity = db.execute("SELECT id FROM identity").fetchone()
            if identity is None:
                self.budget_id = uuid.uuid4().hex
                db.execute("INSERT INTO identity VALUES (?)", (self.budget_id,))
            else:
                self.budget_id = identity["id"]

    def register(self, owner, epoch, *, workers=1, copies=1, ttl_s=300,
                 cpu_ids=(), strict_affinity=False):
        _text(owner, "owner"); _text(epoch, "epoch")
        _integer(workers, "workers", 1, self.worker_capacity)
        _integer(copies, "copies", 1, self.copy_capacity)
        ttl_s = _seconds(ttl_s, "ttl_s", 86400)
        if (type(strict_affinity) is not bool or not isinstance(cpu_ids, (list, tuple))
                or any(type(x) is not int or x not in self.cpus for x in cpu_ids)
                or len(set(cpu_ids)) != len(cpu_ids)):
            raise ValueError("affinity needs distinct observed local CPUs")
        if cpu_ids and len(cpu_ids) < workers:
            raise ValueError("worker allowance exceeds its CPU affinity set")
        with self._db() as db:
            old = db.execute("SELECT * FROM owners WHERE owner=?", (owner,)).fetchone()
            if old is not None and (old["active_workers"] or old["active_copies"]):
                raise HostBudgetBusy("active host work must drain before owner re-registration")
            db.execute("INSERT OR REPLACE INTO owners VALUES (?,?,?,?,?,0,0,?,?)",
                (owner, epoch, workers, copies, self.clock() + ttl_s,
                 json.dumps(list(cpu_ids)), int(strict_affinity)))
        return HostBudgetOwner(self, owner, epoch)

    def _acquire(self, owner, epoch, kind, slots):
        field, cap = ("active_workers", self.worker_capacity) if kind == "compute" else ("active_copies", self.copy_capacity)
        allowance = "workers" if kind == "compute" else "copies"
        with self._db() as db:
            row = db.execute("SELECT * FROM owners WHERE owner=?", (owner,)).fetchone()
            if row is None or row["epoch"] != epoch or row["expires"] <= self.clock():
                raise HostBudgetError("host compute owner expired or epoch changed")
            total = db.execute(f"SELECT COALESCE(SUM({field}),0) FROM owners").fetchone()[0]
            if row[field] + slots > row[allowance] or total + slots > cap:
                raise HostBudgetBusy("host compute/copy concurrency budget is busy")
            token = uuid.uuid4().hex
            db.execute(f"UPDATE owners SET {field}={field}+? WHERE owner=? AND epoch=?", (slots, owner, epoch))
            db.execute("INSERT INTO permits VALUES (?,?,?,?,?)", (token, owner, epoch, kind, slots))
            return token, json.loads(row["cpu_ids"]), bool(row["strict_affinity"])

    def renew(self, owner, epoch, *, ttl_s=300):
        ttl_s = _seconds(ttl_s, "ttl_s", 86400)
        with self._db() as db:
            row = db.execute("SELECT * FROM owners WHERE owner=? AND epoch=?", (owner, epoch)).fetchone()
            if row is None or row["expires"] <= self.clock():
                raise HostBudgetError("expired host owner cannot renew an old epoch")
            db.execute("UPDATE owners SET expires=? WHERE owner=? AND epoch=?", (self.clock()+ttl_s,owner,epoch))

    def _release(self, token, owner, epoch):
        with self._db() as db:
            row = db.execute("SELECT * FROM permits WHERE id=? AND owner=? AND epoch=?", (token, owner, epoch)).fetchone()
            if row is None:
                raise HostBudgetError("host permit no longer belongs to this owner")
            field = "active_workers" if row["kind"] == "compute" else "active_copies"
            db.execute(f"UPDATE owners SET {field}={field}-? WHERE owner=? AND epoch=?", (row["slots"], owner, epoch))
            db.execute("DELETE FROM permits WHERE id=?", (token,))

    def unregister(self, owner, epoch, *, confirmed_stopped=False):
        if type(confirmed_stopped) is not bool:
            raise ValueError("stopped confirmation must be explicit")
        with self._db() as db:
            row = db.execute("SELECT * FROM owners WHERE owner=? AND epoch=?", (owner, epoch)).fetchone()
            if row is None:
                return
            if (row["active_workers"] or row["active_copies"]) and not confirmed_stopped:
                raise HostBudgetBusy("expired or cancelled work retains host budget until confirmed stopped")
            db.execute("DELETE FROM permits WHERE owner=? AND epoch=?", (owner, epoch))
            db.execute("DELETE FROM owners WHERE owner=? AND epoch=?", (owner, epoch))

    def status(self):
        with self._db() as db:
            totals = db.execute("SELECT COALESCE(SUM(active_workers),0), COALESCE(SUM(active_copies),0), COUNT(*) FROM owners").fetchone()
            active = db.execute("SELECT COUNT(*) FROM permits").fetchone()[0]
        return {**self.config(), "active_workers": totals[0], "active_copies": totals[1],
                "owners": totals[2], "active_permits": active, "affinity_status": self.affinity_status,
                "gpu_qualified": False}


class HostBudgetOwner:
    def __init__(self, budget, owner, epoch):
        self.budget, self.owner, self.epoch = budget, owner, epoch

    @contextmanager
    def work(self, kind="compute", *, slots=1, timeout_s=0):
        if kind not in ("compute", "copy"):
            raise ValueError("host work kind must be compute or copy")
        _integer(slots, "slots")
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 <= timeout_s <= 30:
            raise ValueError("host budget wait must be bounded to 0..30 seconds")
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                token, cpus, strict = self.budget._acquire(self.owner, self.epoch, kind, slots)
                break
            except HostBudgetBusy:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(min(.005, max(0., deadline - time.monotonic())))
        previous = None
        try:
            if cpus:
                if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
                    self.budget.affinity_status = "unsupported_os"
                    if strict:
                        raise HostBudgetError("calling-thread CPU affinity is unsupported")
                else:
                    try:
                        previous = os.sched_getaffinity(0)
                        os.sched_setaffinity(0, set(cpus))
                        self.budget.affinity_status = "calling_thread_bound"
                    except OSError as error:
                        previous = None
                        self.budget.affinity_status = "permission_or_topology_changed"
                        if strict:
                            raise HostBudgetError("requested CPU affinity could not be applied") from error
            yield
        finally:
            try:
                if previous is not None:
                    try:
                        os.sched_setaffinity(0, previous)
                    except OSError:
                        self.budget.affinity_status = "restore_failed"
                        raise
            finally:
                self.budget._release(token, self.owner, self.epoch)

    def close(self, *, confirmed_stopped=False):
        self.budget.unregister(self.owner, self.epoch, confirmed_stopped=confirmed_stopped)

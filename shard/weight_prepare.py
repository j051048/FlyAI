"""Node-local, resumable preparation of pinned stage artifacts before loading.

Sources and converters are configured locally, never taken from assignments.
The separate jobs database stores progress only; capacity is reserved in the
same LeaseLedger as resident GPU/RAM allocations. Published directories are
immutable and never overwritten, including when a later verification fails.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.request import Request

from .fetch import Provider, LocalDirProvider, MirrorProvider, urlopen

SCHEMA = "shard-weight-prepare/1"
STATES = ("planned", "reserved", "fetching", "converting", "verifying", "published", "ready", "failed")


class PreparationError(ValueError):
    pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 64 * 1024 * 1024:
        raise PreparationError("invalid local artifact metadata")
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise PreparationError("duplicate artifact metadata field")
            result[name] = value
        return result
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(PreparationError("nonfinite metadata")))


def _path(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or ":" in relative or relative.startswith("/")
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise PreparationError("artifact file path must remain inside its directory")
    root = Path(root).resolve()
    target = root.joinpath(*relative.split("/"))
    if any(parent.is_symlink() for parent in (target, *target.parents) if parent != root.parent):
        raise PreparationError("symlink artifact path refused")
    if not target.resolve().is_relative_to(root):
        raise PreparationError("artifact path escapes directory")
    return target


def _hash_matches(path, row):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != row["size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == row["sha256"]


@contextmanager
def _artifact_lock(path):
    """An OS lock survives heartbeat delays and is released on process exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if handle.tell() == 0:
                handle.write(b"0"); handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise PreparationError("artifact preparation is already running") from None
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise PreparationError("artifact preparation is already running") from None
        yield
    finally:
        handle.close()


class RangeReader:
    """Bounded local/HTTP reads for raw tensor repacking; no full-file fallback."""
    def __init__(self, provider, *, chunk_bytes=1024 * 1024):
        if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 64 * 1024 * 1024:
            raise PreparationError("invalid range read chunk")
        self.provider, self.chunk_bytes = provider, chunk_bytes

    def __call__(self, row, offset, length):
        if (type(offset) is not int or type(length) is not int or offset < 0 or length < 0
                or length > self.chunk_bytes or offset + length > row["size"]):
            raise PreparationError("out-of-bounds tensor range")
        if length == 0:
            return b""
        provider = self.provider.for_file(row) if isinstance(self.provider, RoutedProvider) else self.provider
        if isinstance(provider, LocalDirProvider):
            source = _path(provider.root, row["path"])
            with source.open("rb") as stream:
                stream.seek(offset); result = stream.read(length)
        elif isinstance(provider, MirrorProvider):
            # _path validates manifest-controlled URL components as well.
            _path(Path.cwd(), row["path"])
            request = Request(provider._url(row), headers={**provider.headers,
                "Range": f"bytes={offset}-{offset + length - 1}", "Accept-Encoding": "identity"})
            with urlopen(request, timeout=120) as response:
                expected = f"bytes {offset}-{offset + length - 1}/{row['size']}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise PreparationError("provider must return the exact requested byte range")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise PreparationError("encoded tensor ranges refused")
                result = response.read(length + 1)
        else:
            raise PreparationError("provider does not implement bounded range reads")
        if len(result) != length:
            raise PreparationError("incomplete or oversized tensor range")
        return result


class RoutedProvider(Provider):
    """Node-configured per-file mirrors, including layers resident elsewhere.

    These are weight preparation file sources, not inference expert routing.
    A route does not establish an HTTP file server or reinterpret an SSH alias.
    """
    def __init__(self, routes, *, default=None):
        if not isinstance(routes, dict) or len(routes) > 16384:
            raise PreparationError("bounded per-file provider routes required")
        self.routes = {}
        for name, config in routes.items():
            _path(Path.cwd(), name)
            self.routes[name] = provider_from_config(config, allow_routed=False)
        self.default = None if default is None else provider_from_config(default, allow_routed=False)

    def for_file(self, row):
        result = self.routes.get(row["path"], self.default)
        if result is None:
            raise PreparationError("no locally configured source route for " + row["path"])
        return result

    def fetch(self, row, destination):
        return self.for_file(row).fetch(row, destination)


def provider_from_config(source, *, allow_routed=True):
    if not isinstance(source, dict):
        raise PreparationError("local provider configuration required")
    if source.get("kind") == "local":
        return LocalDirProvider(source["root"])
    if source.get("kind") == "http":
        return MirrorProvider(source["base_url"], headers=source.get("headers"), retries=source.get("retries", 3))
    if source.get("kind") == "routed" and allow_routed:
        return RoutedProvider(source["routes"], default=source.get("default"))
    raise PreparationError("unsupported locally configured provider")


class WeightPrepareManager:
    def __init__(self, cache_root, jobs_db, *, catalog, pack, cohort_id, provider,
                 reserve, resources, convert=None, repack_resources=None, repack=None,
                 adopt=None, recover=None, ttl_s=120, clock=time.time):
        from .weight_artifacts import validate_catalog, validate_pack
        self.catalog = validate_catalog(catalog if isinstance(catalog, dict) else _json(catalog))
        self.pack = validate_pack(self.catalog, pack if isinstance(pack, dict) else _json(pack))
        if not isinstance(cohort_id, str) or not re.fullmatch(r"[0-9a-f]{64}", cohort_id):
            raise PreparationError("pinned model cohort required")
        if type(ttl_s) not in (int, float) or not 0 < ttl_s <= 86400:
            raise PreparationError("invalid preparation TTL")
        self.cohort_id, self.provider = cohort_id, provider
        self.reserve, self.resources, self.convert = reserve, dict(resources), convert
        self.adopt, self.recover = adopt, recover
        self.repack_resources = None if repack_resources is None else dict(repack_resources)
        self.repack = dict(repack or {})
        self.ttl_s, self.clock = float(ttl_s), clock
        self.root = Path(cache_root).absolute()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink():
            raise PreparationError("cache root must not be a symlink")
        self.db = Path(jobs_db).absolute()
        self.db.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._threads = {}
        with self._connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, binding TEXT NOT NULL, state TEXT NOT NULL, attempt INTEGER NOT NULL, updated REAL NOT NULL, result TEXT, error TEXT)")
            if "error_code" not in {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}:
                conn.execute("ALTER TABLE jobs ADD COLUMN error_code TEXT")
            conn.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT NOT NULL, state TEXT NOT NULL, updated REAL NOT NULL)")

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=FULL")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def selection(self, assignment):
        from .weight_artifacts import select_stage_artifacts
        if assignment.get("cohort_id") != self.cohort_id:
            raise PreparationError("assignment differs from locally pinned cohort")
        mode = assignment.get("preparation_mode")
        if mode not in (None, "fetch", "range_repack"):
            raise PreparationError("unsupported planned preparation mode")
        if mode == "range_repack" and (self.convert is None or self.repack_resources is None):
            raise PreparationError("planned range repacking is not locally configured")
        selected = select_stage_artifacts(self.catalog, self.pack, assignment["lo"], assignment["hi"],
            head=assignment["head"], tail=assignment["tail"], dspark=assignment.get("dspark", False))
        expected = {key: selected[key] for key in ("artifact_id", "checkpoint_id", "manifest_sha256")}
        if assignment.get("weight_artifacts") != expected:
            raise PreparationError("assignment artifact identity differs from local catalog selection")
        return selected

    def _binding(self, assignment):
        selected = self.selection(assignment)
        binding = {"cohort_id": self.cohort_id, "node_id": assignment["node_id"],
                   "source_artifact_id": selected["artifact_id"], "lo": assignment["lo"], "hi": assignment["hi"],
                   "head": assignment["head"], "tail": assignment["tail"], "dspark": assignment.get("dspark", False),
                   "cache_scope": str(self.root.resolve()), "planned_preparation_mode": assignment.get("preparation_mode")}
        job = hashlib.sha256(_canonical(binding).encode()).hexdigest()
        return job, binding, selected

    def _record(self, job, state, *, binding=None, result=None, error=None, error_code=None):
        if state not in STATES:
            raise PreparationError("invalid preparation state")
        with self._connection() as conn:
            if binding is not None:
                conn.execute("INSERT OR IGNORE INTO jobs(id,binding,state,attempt,updated,result,error) VALUES(?,?,?,0,?,NULL,NULL)",
                             (job, _canonical(binding), "planned", self.clock()))
            conn.execute("UPDATE jobs SET state=?,updated=?,result=?,error=?,attempt=attempt+? WHERE id=?",
                (state, self.clock(), None if result is None else _canonical(result), error,
                 int(state == "planned"), job))
            conn.execute("UPDATE jobs SET error_code=? WHERE id=?", (error_code, job))
            conn.execute("INSERT INTO events(job,state,updated) VALUES(?,?,?)", (job, state, self.clock()))
            conn.execute("DELETE FROM events WHERE job=? AND id NOT IN (SELECT id FROM events WHERE job=? ORDER BY id DESC LIMIT 128)", (job, job))

    def status(self, job):
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
            if row is None:
                raise PreparationError("unknown preparation job")
            value = {"schema": SCHEMA, "job_id": job, "state": row["state"], "attempt": row["attempt"],
                     "updated_at": row["updated"], "ready": row["state"] == "ready", "error": row["error"]}
            if row["error_code"] is not None:
                value["error_code"] = row["error_code"]
            if row["result"] is not None:
                value.update(json.loads(row["result"]))
            value["history"] = [item[0] for item in conn.execute("SELECT state FROM events WHERE job=? ORDER BY id", (job,))]
            return value

    def submit(self, assignment, guard):
        job, binding, _ = self._binding(assignment)
        guard.assert_live()
        if assignment["node_id"] != guard.node_id:
            raise PreparationError("preparation belongs to another node")
        with self._lock:
            thread = self._threads.get(job)
            if thread is None or not thread.is_alive():
                # Do not overwrite another process's active job progress before
                # this worker obtains the OS artifact lock.
                with self._connection() as conn:
                    conn.execute("INSERT OR IGNORE INTO jobs(id,binding,state,attempt,updated,result,error) VALUES(?,?,?,0,?,NULL,NULL)",
                                 (job, _canonical(binding), "planned", self.clock()))
                def work():
                    try:
                        self.prepare(assignment, guard)
                    except Exception:
                        pass  # persisted failure is returned by prepare_status
                    finally:
                        with self._lock:
                            if self._threads.get(job) is threading.current_thread():
                                self._threads.pop(job, None)
                thread = threading.Thread(target=work, daemon=True, name="weight-prepare-" + job[:12])
                self._threads[job] = thread
                thread.start()
        return self.status(job)

    def prepare(self, assignment, guard):
        from .weight_artifacts import GLOBAL_FILE, PACK_FILE, STAGE_FILE, fsync_directory
        job, binding, selected = self._binding(assignment)
        if assignment["node_id"] != guard.node_id:
            raise PreparationError("preparation belongs to another node")
        guard.assert_live()
        with _artifact_lock(_path(self.root, ".locks/" + job + ".lock")):
            if self.recover is not None:
                # Production sources are synchronous in-process I/O. Holding
                # their exact cache/job OS lock proves the former writer ended.
                self.recover(job, guard)
            self._record(job, "planned", binding=binding)
            target, staging = self.root / job, self.root / (".partial-" + job)
            if target.exists():
                self._record(job, "verifying")
                try:
                    if target.is_symlink():
                        raise PreparationError("published artifact directory must not be a symlink")
                    result = self._verify_existing(target, binding, selected, guard, job)
                    self._record(job, "ready", result=result)
                    return self.status(job)
                except Exception as error:
                    self._record(job, "failed", error=str(error)[:500])
                    raise  # never replace a previously published/live directory
            reservation = work = None
            stopping, invalid = threading.Event(), []
            mode = "converting" if assignment.get("preparation_mode") == "range_repack" else "fetching"
            try:
                if assignment.get("preparation_mode") is not None:
                    reservation = self.reserve(selected, guard, self._resource_budget(selected, mode), self.ttl_s)
                else:
                    try:
                        reservation = self.reserve(selected, guard, self._resource_budget(selected, "fetching"), self.ttl_s)
                    except Exception as error:
                        from .leases import LeaseConflict
                        if not isinstance(error, LeaseConflict) or self.convert is None or self.repack_resources is None:
                            raise
                        reservation = self.reserve(selected, guard, self._resource_budget(selected, "converting"), self.ttl_s)
                        mode = "converting"
                self._record(job, "reserved")
                work = reservation.begin_work("artifact-prepare-" + job + "-" + str(time.time_ns()))
                def check():
                    guard.assert_live()
                    reservation.assert_live()
                    if invalid:
                        raise invalid[0]
                def renew():
                    sequence = 0
                    while not stopping.wait(min(30, self.ttl_s / 3)):
                        try:
                            guard.assert_live()
                            sequence += 1
                            reservation.renew(ttl_s=self.ttl_s, idempotency_key=f"{job}:{sequence}:{time.time_ns()}")
                        except Exception as error:
                            invalid.append(error)
                            return
                watcher = threading.Thread(target=renew, daemon=True, name="weight-prepare-renewal")
                watcher.start()
                staging.mkdir(exist_ok=True, mode=0o700)
                if staging.is_symlink():
                    raise PreparationError("staging directory must not be a symlink")
                self._record(job, mode)
                if mode == "converting":
                    # Only this job's private, unpublished staging is reset.
                    # The raw repacker requires an empty destination. Published
                    # model paths are excluded from this cleanup entirely.
                    self._clear_staging(staging)
                    self.convert(staging, self.catalog, self.pack, selected, self.provider, check)
                else:
                    for row in selected["files"]:
                        check()
                        path = _path(staging, row["path"])
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if not _hash_matches(path, row):
                            path.unlink(missing_ok=True)
                            self.provider.fetch(row, str(path))
                            if not _hash_matches(path, row):
                                # Only this task's unpublished file is discarded.
                                path.unlink(missing_ok=True)
                                raise PreparationError("fetched stage file failed full SHA256 verification")
                        partial = _path(staging, row["path"] + ".part")
                        partial.unlink(missing_ok=True)
                    local_pack = self._local_pack(selected)
                    for name, body in ((GLOBAL_FILE, self.catalog), (PACK_FILE, local_pack), (STAGE_FILE, selected)):
                        path = _path(staging, name)
                        with path.open("w", encoding="utf-8") as out:
                            out.write(_canonical(body)); out.flush(); os.fsync(out.fileno())
                    fsync_directory(staging)
                check()
                self._record(job, "verifying")
                verified = self._verify(staging, binding)
                # Flush payloads before rename; no READY row precedes durable files.
                for row in verified["files"]:
                    with _path(staging, row["path"]).open("r+b") as stream:
                        os.fsync(stream.fileno())
                check()
                if target.exists():
                    raise PreparationError("published artifact directory already exists")
                staging.rename(target)
                fsync_directory(target.parent)
                self._record(job, "published")
                # Preserve the real hash witness across a same-filesystem rename;
                # the helper checks inode/file stamps and metadata unchanged.
                from .weight_artifacts import relocate_verified_artifacts
                verified = relocate_verified_artifacts(verified, target)
                result = self._result(target, selected, verified)
                result["preparation_mode"] = "range_repack" if mode == "converting" else "fetch"
                result["planned_preparation_mode"] = assignment.get("preparation_mode")
                stopping.set()
                watcher.join(1)
                reservation.end_work(work)
                work = None
                reservation.finish(verified)
                reservation = None
                self._record(job, "ready", result=result)
                return self.status(job)
            except BaseException as error:
                from .leases import LeaseConflict
                self._record(job, "failed", error=f"{type(error).__name__}: {str(error)[:450]}",
                             error_code="resource_conflict" if isinstance(error, LeaseConflict) else "preparation_failed")
                raise
            finally:
                stopping.set()
                if "watcher" in locals():
                    watcher.join(1)
                if reservation is not None:
                    if work is not None:
                        reservation.end_work(work)
                    reservation.release()

    def _local_pack(self, selected):
        needed = {row["path"] for row in selected["files"]}
        return {**self.pack, "files": [row for row in self.pack["files"] if row["path"] in needed],
            "weight_map": {name: path for name, path in self.pack["weight_map"].items() if path in needed},
            "offsets": {name: span for name, span in self.pack["offsets"].items()
                        if self.pack["weight_map"][name] in needed}}

    def _resource_budget(self, selected, mode):
        from .leases import LeaseConflict
        if mode == "converting":
            from .weight_artifacts import repack_storage_bound
            bound = repack_storage_bound(self.catalog, self.pack, selected,
                self.repack.get("max_file_bytes", 512 << 20), self.repack.get("chunk_bytes", 1 << 20))
            declared = self.repack_resources
            required_disk = bound["disk_peak_bytes"]
            required_ram = max(4 << 20, bound["host_ram_bytes"])
        else:
            metadata = sum(len(_canonical(body).encode()) + 1 for body in
                           (self.catalog, self._local_pack(selected), selected))
            required_disk = sum(row["size"] for row in selected["files"]) + metadata
            required_ram = metadata * 12 + (4 << 20)
            declared = self.resources
        if declared is None or not isinstance(declared.get("filesystem_id"), str):
            raise PreparationError("locally registered filesystem budget required")
        result = {"filesystem_id": declared["filesystem_id"], "disk_peak_bytes": required_disk,
                  "ram_bytes": required_ram, "pinned_bytes": 0}
        for name in ("disk_peak_bytes", "ram_bytes", "pinned_bytes"):
            value = declared.get(name, result[name])
            if type(value) is not int or value < 0:
                raise PreparationError("preparation budgets must be nonnegative byte integers")
            if value < result[name]:
                raise LeaseConflict("configured preparation " + name + " is below the pre-write geometry bound")
            result[name] = value
        return result

    def preparation_options(self, assignment):
        """Public pre-write geometry; no claim of measured GPU/model capacity."""
        selected = self.selection(assignment)
        options = {"fetch": self._resource_budget(selected, "fetching")}
        if self.convert is not None and self.repack_resources is not None:
            options["range_repack"] = self._resource_budget(selected, "converting")
        return {"source_artifact_id": selected["artifact_id"], "options": options,
                "scope": "pinned tensor geometry and bounded CPU I/O preparation only"}

    def _adoption_resources(self, verified):
        directory = Path(verified["directory"])
        actual = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        metadata = sum(len(_canonical(body).encode()) for body in
            (verified["catalog"], verified["pack"], verified["stage"]))
        return {"filesystem_id": self.resources["filesystem_id"], "disk_peak_bytes": actual,
                "ram_bytes": metadata * 12 + (4 << 20), "pinned_bytes": 0}

    def _verify_existing(self, target, binding, selected, guard, job):
        # Reserve verification buffers BEFORE parsing/hashing an existing
        # artifact. Its payload is already on disk, so no second copy is needed.
        metadata = sum(len(_canonical(body).encode()) for body in
                       (self.catalog, self._local_pack(selected), selected))
        required_ram = metadata * 12 + (4 << 20)
        approved = self.repack_resources if binding.get("planned_preparation_mode") == "range_repack" else self.resources
        ram = approved.get("ram_bytes", required_ram)
        if type(ram) is not int or ram < required_ram:
            from .leases import LeaseConflict
            raise LeaseConflict("configured verification RAM is below the pre-read geometry bound")
        verification = self.reserve(selected, guard, {
            "filesystem_id": self.resources["filesystem_id"], "disk_peak_bytes": 0,
            "ram_bytes": ram, "pinned_bytes": 0}, self.ttl_s)
        work = verification.begin_work("artifact-prepare-" + job + "-verify-" + str(time.time_ns()))
        stopping, invalid = threading.Event(), []
        def renew():
            sequence = 0
            while not stopping.wait(min(30, self.ttl_s / 3)):
                try:
                    guard.assert_live(); sequence += 1
                    verification.renew(ttl_s=self.ttl_s, idempotency_key=f"{job}:verify:{sequence}:{time.time_ns()}")
                except Exception as error:
                    invalid.append(error); return
        watcher = threading.Thread(target=renew, daemon=True, name="artifact-verify-renewal")
        watcher.start()
        try:
            verified = self._verify(target, binding)
            guard.assert_live(); verification.assert_live()
            if invalid:
                raise invalid[0]
            if self.adopt is not None:
                self.adopt(selected, guard, verified)
            else:
                # The callback seam can be used by tests/custom local ledgers;
                # production uses the stat-sealed no-second-copy adoption API.
                adopted = self.reserve(selected, guard, self._adoption_resources(verified), self.ttl_s)
                try:
                    adopted.finish(verified)
                except BaseException:
                    adopted.release(); raise
            result = self._result(target, selected, verified)
            if binding.get("planned_preparation_mode") is not None:
                result["preparation_mode"] = binding["planned_preparation_mode"]
                result["planned_preparation_mode"] = binding["planned_preparation_mode"]
            return result
        finally:
            stopping.set(); watcher.join(1)
            verification.end_work(work)
            verification.release()

    def _clear_staging(self, directory):
        if (directory.parent != self.root or re.fullmatch(r"\.partial-[0-9a-f]{64}", directory.name) is None
                or directory.is_symlink() or not directory.resolve().is_relative_to(self.root.resolve())):
            raise PreparationError("refusing cleanup outside job-owned staging")
        for path in sorted(directory.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_symlink():
                raise PreparationError("symlink found in staging cleanup")
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()

    def _verify(self, directory, binding):
        from .weight_artifacts import verify_stage_artifacts
        return verify_stage_artifacts(directory, expected_checkpoint_id=self.catalog["checkpoint_id"],
            expected_manifest_sha256=self.catalog["manifest_sha256"], lo=binding["lo"], hi=binding["hi"],
            head=binding["head"], tail=binding["tail"], dspark=binding["dspark"], verify_files=True)

    @staticmethod
    def _result(directory, selection, verified):
        if verified.get("payload_integrity_verified") is not True:
            raise PreparationError("full stage payload verification required before publication")
        return {"directory": str(directory), "artifact_id": verified["artifact_id"],
                "source_artifact_id": selection["artifact_id"], "checkpoint_id": verified["checkpoint_id"],
                "manifest_sha256": verified["manifest_sha256"], "payload_integrity_verified": True}


def configured_prepare_factory(ledger_path, sources):
    """Return a preparation hook from node-local sources, not network commands."""
    from .leases import LeaseLedger, PrepareRequest, PrepareResources
    configured = list(sources)
    managers = {}
    lock = threading.RLock()
    def manager_for(assignment, guard):
        matches = [row for row in configured if row.get("cohort_id") == assignment.get("cohort_id")]
        if len(matches) != 1:
            raise PreparationError("no unique locally configured weight source for cohort")
        row = matches[0]
        with lock:
            key = (guard.node_id, assignment["cohort_id"])
            manager = managers.get(key)
            if manager is None:
                source = row["provider"]
                provider = provider_from_config(source)
                def reserve(selection, stage_guard, resources, ttl):
                    ledger = stage_guard.ledger
                    request = PrepareRequest(selection["artifact_id"], assignment["cohort_id"], stage_guard.node_id,
                        stage_guard.memory_domain_id, PrepareResources(**resources), ttl)
                    return ledger.prepare_artifact(request, principal=stage_guard.principal,
                        idempotency_key="weight:" + selection["artifact_id"] + ":" + str(time.time_ns()))
                def adopt(selection, stage_guard, verified):
                    ledger = stage_guard.ledger
                    request = PrepareRequest(selection["artifact_id"], assignment["cohort_id"], stage_guard.node_id,
                        stage_guard.memory_domain_id, PrepareResources(row["resources"]["filesystem_id"], 0, 0, 0),
                        row.get("ttl_s", 120))
                    adopted = ledger.adopt_artifact(request, verified, principal=stage_guard.principal,
                        idempotency_key="adopt:" + selection["artifact_id"] + ":" + str(time.time_ns()))
                    try:
                        adopted.finish(verified)
                    except BaseException:
                        adopted.release()
                        raise
                    return adopted
                def recover(job, stage_guard):
                    # Both configured providers and the raw repacker execute
                    # inside this Python process, with no detached child writer.
                    return stage_guard.ledger.recover_stopped_artifact_work(
                        lambda work: work.work_id.startswith("artifact-prepare-" + job + "-"))
                convert = None
                if row.get("repack_resources") is not None:
                    limits = row.get("repack", {})
                    def convert(destination, catalog, pack, stage, actual_provider, check):
                        from .weight_artifacts import repack_stage
                        return repack_stage(catalog, pack, stage, destination,
                            RangeReader(actual_provider, chunk_bytes=limits.get("chunk_bytes", 1 << 20)),
                            max_file_bytes=limits.get("max_file_bytes", 512 << 20),
                            chunk_bytes=limits.get("chunk_bytes", 1 << 20), cancel_check=check)
                manager = WeightPrepareManager(row["cache_root"], row["jobs_db"], catalog=row["catalog"],
                    pack=row["pack"], cohort_id=row["cohort_id"], provider=provider, reserve=reserve,
                    resources=row["resources"], repack_resources=row.get("repack_resources"),
                    convert=convert, repack=row.get("repack"), adopt=adopt, recover=recover,
                    ttl_s=row.get("ttl_s", 120))
                managers[key] = manager
        return manager
    def prepare(assignment, guard):
        return manager_for(assignment, guard).prepare(assignment, guard)
    def submit(assignment, guard):
        return manager_for(assignment, guard).submit(assignment, guard)
    def status(assignment, guard, job_id):
        manager = manager_for(assignment, guard)
        job, _, _ = manager._binding(assignment)
        guard.assert_live()
        if job != job_id or guard.node_id != assignment["node_id"]:
            raise PreparationError("preparation status belongs to another assignment")
        return manager.status(job)
    def options(assignment, guard):
        return manager_for(assignment, guard).preparation_options(assignment)
    prepare.submit, prepare.status, prepare.options = submit, status, options
    return prepare

"""Start/stop only locally recorded deployment processes, without a shell.

The public network uses LeasedProcessRunner. This manual/SSH utility is not a GPU
resource lease. Durable pre-spawn reservations survive failed bookkeeping and
restart; an unresolved launch is never interpreted as an available slot.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import subprocess
import threading
import time

_TOKEN_ENV = "SHARD_MANAGED_LAUNCH_TOKEN"
_STARTING = set()
_STARTING_LOCK = threading.Lock()


class ManagedLauncher:
    def __init__(self, state_dir):
        target = Path(state_dir)
        if target.is_symlink():
            raise ValueError("state directory cannot be a symlink")
        self.directory = target.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = self.directory / "processes.sqlite"
        if self.db.is_symlink():
            raise ValueError("process database cannot be a symlink")
        with closing(self._db()) as conn:
            # Keep the original five columns for old locally recorded processes.
            conn.execute("CREATE TABLE IF NOT EXISTS processes (name TEXT PRIMARY KEY, pid INTEGER, birth REAL, argv_hash TEXT, log TEXT)")
            conn.execute("""CREATE TABLE IF NOT EXISTS launches(
                name TEXT PRIMARY KEY,token TEXT NOT NULL,state TEXT NOT NULL,
                starter_pid INTEGER NOT NULL,starter_birth REAL NOT NULL,
                pid INTEGER,birth REAL)""")
            conn.commit()

    def _db(self):
        conn = sqlite3.connect(self.db, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    @staticmethod
    def _name(name):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name):
            raise ValueError("process name must be a safe deployment identifier")
        return name

    @staticmethod
    def _timeout(timeout):
        import math
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("cleanup timeout must be finite and positive")
        return float(timeout)

    @staticmethod
    def _process(row):
        import psutil
        if row is None or row["pid"] is None or row["birth"] is None:
            return None
        try:
            process = psutil.Process(row["pid"])
            if abs(process.create_time() - row["birth"]) > .01 or process.status() == psutil.STATUS_ZOMBIE:
                return None
            return process
        except psutil.NoSuchProcess:
            return None

    @staticmethod
    def _token_processes(token, since=None):
        """Explicit local recovery only: identify this launch, never a port/GPU pattern."""
        import psutil
        if not token:
            return []
        owner = psutil.Process(os.getpid()).username()
        result = []
        for process in psutil.process_iter():
            try:
                if process.username() != owner:
                    continue
                # A child cannot predate its recorded starter. In particular,
                # unrelated elevated Windows apps may deny environ() even to
                # the same username and must not poison every launch recovery.
                if since is not None and process.create_time() < since:
                    continue
                if process.environ().get(_TOKEN_ENV) == token and process.status() != psutil.STATUS_ZOMBIE:
                    result.append(process)
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                # Do not silently classify an inaccessible same-user process idle.
                try:
                    same_user = process.username() == owner
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    same_user = False
                if same_user:
                    raise RuntimeError("owned-process recovery cannot inspect all same-user candidates")
        return result

    @staticmethod
    def _terminate_targets(targets, timeout):
        """Birth-identity psutil objects protect signals against PID reuse."""
        import psutil
        unique = {p.pid: p for p in targets}
        for process in list(unique.values()):
            try:
                for child in process.children(recursive=True):
                    unique.setdefault(child.pid, child)
            except psutil.NoSuchProcess:
                pass
        values = list(unique.values())
        for process in reversed(values):
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(values, timeout=timeout)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(alive, timeout=timeout)
        if alive:
            raise RuntimeError("owned process cleanup not acknowledged")

    def _cleanup_child(self, process, token, timeout, since=None):
        import psutil
        targets = self._token_processes(token, since)
        if process.poll() is None:
            try:
                root = psutil.Process(process.pid)
                targets.append(root)
            except psutil.NoSuchProcess:
                pass
        self._terminate_targets(targets, timeout)
        # Reap the actual Popen child, including a process that ignored SIGTERM.
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout)
        # Catch an owned descendant forked between enumeration and termination.
        remaining = self._token_processes(token, since)
        if remaining:
            self._terminate_targets(remaining, timeout)
        if self._token_processes(token, since):
            raise RuntimeError("owned descendant cleanup not acknowledged")

    def _clear(self, name, token=None):
        with closing(self._db()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT token FROM launches WHERE name=?", (name,)).fetchone()
            if token is not None and (row is None or row["token"] != token):
                raise RuntimeError("launch reservation changed during cleanup")
            conn.execute("DELETE FROM processes WHERE name=?", (name,))
            conn.execute("DELETE FROM launches WHERE name=?", (name,))
            conn.commit()

    def _orphan(self, name, token, process, birth):
        # The starting reservation already exists if the DB cannot accept this
        # diagnostic update. Failure here must NEVER erase that reservation.
        try:
            with closing(self._db()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("UPDATE launches SET state='orphan',pid=?,birth=? WHERE name=? AND token=?",
                             (None if process is None else process.pid, birth, name, token))
                conn.commit()
        except Exception:
            pass

    def start(self, name, argv, *, cwd=None, environment=None, cleanup_timeout=5):
        import psutil
        self._name(name)
        timeout = self._timeout(cleanup_timeout)
        if not isinstance(argv, (list, tuple)) or not argv or any(not isinstance(a, str) or "\x00" in a for a in argv):
            raise ValueError("argv must be a nonempty list of strings")
        explicit = dict(environment or {})
        if any(not isinstance(k, str) or not isinstance(v, str) or "\x00" in k or "\x00" in v for k, v in explicit.items()):
            raise ValueError("environment must be a string mapping")
        digest = hashlib.sha256(json.dumps({"argv": list(argv), "cwd": str(cwd),
            "environment": explicit}, sort_keys=True).encode()).hexdigest()
        token = secrets.token_hex(32)
        starter_birth = psutil.Process(os.getpid()).create_time()
        log_path = self.directory / (name + ".log")
        if log_path.is_symlink():
            raise ValueError("deployment log cannot be a symlink")
        with closing(self._db()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            launch = conn.execute("SELECT * FROM launches WHERE name=?", (name,)).fetchone()
            row = conn.execute("SELECT * FROM processes WHERE name=?", (name,)).fetchone()
            if launch is not None and launch["state"] != "running":
                raise RuntimeError("unresolved launch reservation; explicit local recover required")
            existing = self._process(row)
            if existing is not None:
                if row["argv_hash"] != digest:
                    raise RuntimeError("deployment process already running with different configuration; stop it explicitly")
                return {"name": name, "pid": existing.pid, "running": True, "reused": True}
            if launch is not None and self._token_processes(launch["token"], launch["starter_birth"]):
                raise RuntimeError("old owned descendants remain; explicit local recover required")
            if row is not None and launch is None:
                raise RuntimeError("legacy dead process record requires explicit stop/recover before restart")
            conn.execute("DELETE FROM processes WHERE name=?", (name,))
            conn.execute("INSERT OR REPLACE INTO launches VALUES(?,?,'starting',?,?,NULL,NULL)",
                         (name, token, os.getpid(), starter_birth))
            # If this commit fails, no child exists yet.
            with _STARTING_LOCK:
                _STARTING.add(token)
            try:
                conn.commit()
            except BaseException:
                with _STARTING_LOCK:
                    _STARTING.discard(token)
                raise
        process, birth = None, None
        try:
            env = {**os.environ, **explicit, _TOKEN_ENV: token}
            with log_path.open("wb") as log:
                options = dict(cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT, shell=False)
                if os.name == "nt":
                    options["creationflags"] = subprocess.CREATE_NO_WINDOW
                else:
                    options["start_new_session"] = True
                process = subprocess.Popen(list(argv), **options)
            birth = psutil.Process(process.pid).create_time()
            with closing(self._db()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                current = conn.execute("SELECT token FROM launches WHERE name=?", (name,)).fetchone()
                if current is None or current["token"] != token:
                    raise RuntimeError("launch reservation changed before process registration")
                conn.execute("INSERT OR REPLACE INTO processes VALUES (?,?,?,?,?)",
                             (name, process.pid, birth, digest, str(log_path)))
                conn.execute("UPDATE launches SET state='running',pid=?,birth=? WHERE name=? AND token=?",
                             (process.pid, birth, name, token))
                # Critical: commit belongs to the spawn cleanup try, not __exit__.
                conn.commit()
            return {"name": name, "pid": process.pid, "running": process.poll() is None,
                    "reused": False, "log": str(log_path)}
        except BaseException:
            self._orphan(name, token, process, birth)
            try:
                if process is not None:
                    self._cleanup_child(process, token, timeout, starter_birth)
                self._clear(name, token)
            except BaseException as cleanup_error:
                raise RuntimeError("launch failed; cleanup/reservation unresolved; explicit local recover required") from cleanup_error
            raise
        finally:
            with _STARTING_LOCK:
                _STARTING.discard(token)

    def _stop(self, name, timeout):
        self._name(name)
        timeout = self._timeout(timeout)
        with closing(self._db()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            launch = conn.execute("SELECT * FROM launches WHERE name=?", (name,)).fetchone()
            row = conn.execute("SELECT * FROM processes WHERE name=?", (name,)).fetchone()
            if launch is not None and launch["state"] == "starting":
                starter = self._process({"pid": launch["starter_pid"], "birth": launch["starter_birth"]})
                with _STARTING_LOCK:
                    active_here = launch["token"] in _STARTING
                if starter is not None and (starter.pid != os.getpid() or active_here):
                    raise RuntimeError("launch starter still active; cleanup must wait")
            token = launch["token"] if launch else None
            since = launch["starter_birth"] if launch else None
            root = self._process(row) or self._process(launch)
            targets = self._token_processes(token, since)
            if root is not None:
                targets.append(root)
            self._terminate_targets(targets, timeout)
            remaining = self._token_processes(token, since)
            if remaining:
                self._terminate_targets(remaining, timeout)
            if self._token_processes(token, since):
                raise RuntimeError("owned process cleanup not acknowledged")
            conn.execute("DELETE FROM processes WHERE name=?", (name,))
            conn.execute("DELETE FROM launches WHERE name=?", (name,))
            conn.commit()
            return {"name": name, "running": False, "stopped": bool(targets)}

    def stop(self, name, *, timeout=5):
        return self._stop(name, timeout)

    def recover(self, name, *, timeout=5):
        """Explicit local orphan recovery; never discard an unacknowledged live child."""
        result = self._stop(name, timeout)
        return {**result, "recovered": True}

    def status(self, name):
        self._name(name)
        with closing(self._db()) as conn:
            row = conn.execute("SELECT * FROM processes WHERE name=?", (name,)).fetchone()
            launch = conn.execute("SELECT * FROM launches WHERE name=?", (name,)).fetchone()
        process = self._process(row) or self._process(launch)
        return {"name": name, "running": process is not None,
                "pid": None if process is None else process.pid,
                "reserved": launch is not None,
                "recovery_required": launch is not None and launch["state"] != "running"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default=".shard-processes")
    parser.add_argument("action", choices=("start", "stop", "status", "recover"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--cwd")
    parser.add_argument("--environment-file")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    launcher = ManagedLauncher(args.state_dir)
    if args.action == "start":
        environment = json.loads(Path(args.environment_file).read_text()) if args.environment_file else None
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        result = launcher.start(args.name, command, cwd=args.cwd, environment=environment)
    elif args.action in ("stop", "recover"):
        result = getattr(launcher, args.action)(args.name)
    else:
        result = launcher.status(args.name)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

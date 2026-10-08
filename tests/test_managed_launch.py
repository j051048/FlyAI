import sqlite3
import sys
import os
import subprocess
import threading
import time
import pytest

from shard.managed_launch import ManagedLauncher


def test_owned_process_idempotency_and_stop(tmp_path):
    pytest.importorskip("psutil")
    launcher = ManagedLauncher(tmp_path)
    argv = [sys.executable, "-c", "import time; time.sleep(60)"]
    try:
        result = launcher.start("ring.stage0", argv)
        assert result["running"]
        assert launcher.start("ring.stage0", argv)["pid"] == result["pid"]
        with pytest.raises(RuntimeError):
            launcher.start("ring.stage0", argv + ["changed"])
        assert launcher.status("ring.stage0")["running"]
    finally:
        launcher.stop("ring.stage0", timeout=2)
    assert not launcher.status("ring.stage0")["running"]


def test_pid_reuse_cannot_signal_another_process(tmp_path):
    import os
    launcher = ManagedLauncher(tmp_path)
    with sqlite3.connect(launcher.db) as conn:
        conn.execute("INSERT INTO processes VALUES (?,?,?,?,?)", ("old", os.getpid(), 0, "hash", "log"))
    assert launcher.stop("old")["stopped"] is False


def commit_failure(launcher, monkeypatch, *, before_failure=None, fail_all=False):
    original = launcher._db
    commits = [0]
    class Connection:
        def __init__(self, inner): self.inner = inner
        def __getattr__(self, name): return getattr(self.inner, name)
        def commit(self):
            commits[0] += 1
            if commits[0] == 2 or fail_all and commits[0] >= 2:
                if before_failure: before_failure()
                raise sqlite3.OperationalError("injected process-registration commit failure")
            return self.inner.commit()
    monkeypatch.setattr(launcher, "_db", lambda: Connection(original()))
    return original, commits


def capture_child(monkeypatch):
    import shard.managed_launch as managed
    original = subprocess.Popen
    children = []
    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(managed.subprocess, "Popen", spawn)
    return children


def wait_file(path):
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert path.exists()


def test_registration_commit_failure_kills_only_spawned_child_and_rolls_back(tmp_path, monkeypatch):
    launcher = ManagedLauncher(tmp_path)
    children = capture_child(monkeypatch)
    original_db, commits = commit_failure(launcher, monkeypatch)
    with pytest.raises(sqlite3.OperationalError, match="commit failure"):
        launcher.start("failing", [sys.executable, "-c", "import time; time.sleep(60)"], cleanup_timeout=.2)
    assert len(children) == 1 and children[0].poll() is not None
    monkeypatch.setattr(launcher, "_db", original_db)
    assert launcher.status("failing")["reserved"] is False
    with sqlite3.connect(launcher.db) as conn:
        assert not conn.execute("SELECT 1 FROM processes WHERE name='failing'").fetchone()


@pytest.mark.skipif(os.name == "nt", reason="Windows terminate already forces process exit; POSIX SIGTERM-ignore path")
def test_commit_failure_escalates_real_sigterm_ignoring_child_to_kill(tmp_path, monkeypatch):
    launcher = ManagedLauncher(tmp_path)
    ready = tmp_path / "ready"
    argv = [sys.executable, "-c", "import signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            f"pathlib.Path({str(ready)!r}).write_text('ready'); time.sleep(60)"]
    children = capture_child(monkeypatch)
    original_db, _ = commit_failure(launcher, monkeypatch, before_failure=lambda: wait_file(ready))
    with pytest.raises(sqlite3.OperationalError):
        launcher.start("ignore-term", argv, cleanup_timeout=.1)
    assert ready.exists() and children[0].poll() is not None
    assert children[0].returncode == -9
    monkeypatch.setattr(launcher, "_db", original_db)
    assert not launcher.status("ignore-term")["reserved"]


def test_cleanup_unacknowledged_blocks_restart_and_explicit_recovery_finishes(tmp_path, monkeypatch):
    launcher = ManagedLauncher(tmp_path)
    children = capture_child(monkeypatch)
    original_db, _ = commit_failure(launcher, monkeypatch)
    real_cleanup = launcher._cleanup_child
    monkeypatch.setattr(launcher, "_cleanup_child", lambda *args: (_ for _ in ()).throw(RuntimeError("unacknowledged")))
    try:
        with pytest.raises(RuntimeError, match="unresolved"):
            launcher.start("orphan", [sys.executable, "-c", "import time; time.sleep(60)"], cleanup_timeout=.1)
        monkeypatch.setattr(launcher, "_db", original_db)
        state = launcher.status("orphan")
        assert state["reserved"] and state["recovery_required"] and children[0].poll() is None
        with pytest.raises(RuntimeError, match="recover"):
            launcher.start("orphan", [sys.executable, "-c", "pass"])
        monkeypatch.setattr(launcher, "_cleanup_child", real_cleanup)
        result = launcher.recover("orphan", timeout=.2)
        children[0].wait(2)
        assert result["recovered"] and children[0].poll() is not None
        assert not launcher.status("orphan")["reserved"]
    finally:
        monkeypatch.setattr(launcher, "_db", original_db)
        monkeypatch.setattr(launcher, "_cleanup_child", real_cleanup)
        launcher.recover("orphan", timeout=.2)


def test_persistent_pending_marker_survives_all_post_spawn_database_failures(tmp_path, monkeypatch):
    launcher = ManagedLauncher(tmp_path)
    children = capture_child(monkeypatch)
    original_db, _ = commit_failure(launcher, monkeypatch, fail_all=True)
    with pytest.raises(RuntimeError, match="reservation unresolved"):
        launcher.start("pending", [sys.executable, "-c", "import time; time.sleep(60)"], cleanup_timeout=.1)
    assert children[0].poll() is not None
    reopened = ManagedLauncher(tmp_path)
    assert reopened.status("pending")["reserved"]
    with pytest.raises(RuntimeError, match="recover"):
        reopened.start("pending", [sys.executable, "-c", "pass"])
    assert reopened.recover("pending", timeout=.1)["recovered"]


def test_pre_spawn_commit_failure_creates_no_process(tmp_path, monkeypatch):
    launcher = ManagedLauncher(tmp_path)
    children = capture_child(monkeypatch)
    original = launcher._db
    class Connection:
        def __init__(self, inner): self.inner = inner
        def __getattr__(self, name): return getattr(self.inner, name)
        def commit(self): raise sqlite3.OperationalError("initial reservation unavailable")
    monkeypatch.setattr(launcher, "_db", lambda: Connection(original()))
    with pytest.raises(sqlite3.OperationalError):
        launcher.start("none", [sys.executable, "-c", "pass"])
    assert children == []
    monkeypatch.setattr(launcher, "_db", original)
    assert not launcher.status("none")["reserved"]


def test_recovery_cannot_overtake_active_start(tmp_path, monkeypatch):
    launcher = ManagedLauncher(tmp_path)
    entered, proceed = threading.Event(), threading.Event()
    original = subprocess.Popen
    def blocked_spawn(*args, **kwargs):
        entered.set()
        assert proceed.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr("shard.managed_launch.subprocess.Popen", blocked_spawn)
    result, failures = [], []
    def launch():
        try: result.append(launcher.start("race", [sys.executable, "-c", "import time; time.sleep(60)"]))
        except Exception as error: failures.append(error)
    thread = threading.Thread(target=launch); thread.start()
    assert entered.wait(2)
    try:
        with pytest.raises(RuntimeError, match="starter still active"):
            ManagedLauncher(tmp_path).recover("race", timeout=.1)
        with pytest.raises(RuntimeError, match="reservation"):
            ManagedLauncher(tmp_path).start("race", [sys.executable, "-c", "pass"])
    finally:
        proceed.set(); thread.join(5)
        launcher.stop("race", timeout=.2)
    assert result and not failures


def test_concurrent_starts_cannot_spawn_two_owned_children(tmp_path, monkeypatch):
    first, second = ManagedLauncher(tmp_path), ManagedLauncher(tmp_path)
    children = capture_child(monkeypatch)
    barrier = threading.Barrier(2)
    results, errors = [], []
    argv = [sys.executable, "-c", "import time; time.sleep(60)"]
    def start(launcher):
        barrier.wait()
        try: results.append(launcher.start("shared", argv))
        except RuntimeError as error: errors.append(error)
    threads = [threading.Thread(target=start, args=(launcher,)) for launcher in (first, second)]
    try:
        for thread in threads: thread.start()
        for thread in threads: thread.join(5)
        assert results and len(children) == 1
        assert all("reservation" in str(error) for error in errors)
        assert first.start("shared", argv)["reused"]
    finally:
        first.stop("shared", timeout=.2)


def test_restart_recovers_pre_registration_orphan_by_local_token_only(tmp_path):
    import psutil
    import secrets
    launcher = ManagedLauncher(tmp_path)
    token = secrets.token_hex(32)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             env={**os.environ, "SHARD_MANAGED_LAUNCH_TOKEN": token},
                             **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
    try:
        birth = psutil.Process(child.pid).create_time()
        with sqlite3.connect(launcher.db) as conn:
            conn.execute("INSERT INTO launches VALUES(?,?,'starting',?,?,NULL,NULL)",
                         ("unregistered", token, 99999999, birth - .001))
        restarted = ManagedLauncher(tmp_path)
        assert restarted.status("unregistered")["recovery_required"]
        with pytest.raises(RuntimeError, match="reservation"):
            restarted.start("unregistered", [sys.executable, "-c", "pass"])
        assert restarted.recover("unregistered", timeout=.2)["stopped"]
        child.wait(2)
        assert not restarted.status("unregistered")["reserved"]
    finally:
        if child.poll() is None: child.kill()
        child.wait(2)

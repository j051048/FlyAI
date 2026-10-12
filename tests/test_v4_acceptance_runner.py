"""Real owned CPU subprocesses; no GPU, model, remote or Git execution."""
import json
import os
from pathlib import Path
import sys

import pytest

from tools import v4_5090_acceptance as campaign


def runner(tmp_path):
    pytest.importorskip("psutil")
    return campaign.Runner(tmp_path, sys.executable)


def assert_clean(owner, name):
    state = owner.manager.status(name)
    assert state["running"] is False and state["reserved"] is False
    assert owner.records[-1]["cleanup_acknowledged"] is True


def test_actual_child_returncode_is_persisted_without_launcher_status_fields(tmp_path):
    owner = runner(tmp_path)
    record = owner.command("baseline", [sys.executable, "-c", "print('actual child output'); raise SystemExit(2)"],
        15, allowed=(0, 2))
    assert record["exit_code"] == 2
    completion = campaign.read(owner.private / "baseline.completion.json")
    assert completion["nonce"] == record["request_nonce"]
    assert completion["request_sha256"] == record["request_sha256"]
    assert completion["exit_code"] == 2
    assert "actual child output" in (tmp_path / record["log"]).read_text(encoding="utf-8")
    assert_clean(owner, "baseline")


def test_disallowed_child_exit_preserves_logs_and_cleanup(tmp_path):
    owner = runner(tmp_path)
    with pytest.raises(RuntimeError, match="failed; inspect"):
        owner.command("bad-exit", [sys.executable, "-c", "print('failed child'); raise SystemExit(3)"], 15)
    assert owner.records[-1]["exit_code"] == 3
    assert campaign.read(tmp_path / "commands.json") == owner.records
    assert_clean(owner, "bad-exit")


def test_existing_completion_and_repeated_command_names_cannot_be_reused(tmp_path):
    owner = runner(tmp_path)
    campaign.write(owner.private / "stale.completion.json", {"exit_code": 0})
    with pytest.raises(ValueError, match="unique command name"):
        owner.command("stale", [sys.executable, "-c", "raise RuntimeError('must not launch')"], 15)
    assert owner.manager.status("stale")["reserved"] is False
    owner.command("once", [sys.executable, "-c", "pass"], 15)
    with pytest.raises(ValueError, match="unique command name"):
        owner.command("once", [sys.executable, "-c", "pass"], 15)
    assert_clean(owner, "once")


@pytest.mark.parametrize("field", ["name", "nonce", "request_sha256"])
def test_completion_must_bind_actual_name_nonce_and_request(tmp_path, monkeypatch, field):
    owner = runner(tmp_path)
    real_read = campaign.read
    def corrupt_completion(path):
        value = real_read(path)
        if str(path).endswith(".completion.json"):
            value[field] = "another-request"
        return value
    monkeypatch.setattr(campaign, "read", corrupt_completion)
    with pytest.raises(RuntimeError, match="completion is absent, stale"):
        owner.command("binding", [sys.executable, "-c", "pass"], 15)
    assert owner.records[-1]["exit_code"] is None
    assert_clean(owner, "binding")


@pytest.mark.parametrize("failure", ["start", "status"])
def test_launch_or_status_failure_always_stops_created_tree(tmp_path, monkeypatch, failure):
    owner = runner(tmp_path)
    real_start, real_status = owner.manager.start, owner.manager.status
    if failure == "start":
        def fail_after_spawn(*args, **kwargs):
            real_start(*args, **kwargs)
            raise OSError("injected start observation failure")
        monkeypatch.setattr(owner.manager, "start", fail_after_spawn)
    else:
        monkeypatch.setattr(owner.manager, "status", lambda *args: (_ for _ in ()).throw(OSError("injected status failure")))
    with pytest.raises(OSError, match="injected"):
        owner.command("broken-observation", [sys.executable, "-c", "import time; time.sleep(60)"], 15)
    monkeypatch.setattr(owner.manager, "status", real_status)
    assert_clean(owner, "broken-observation")


def test_oversized_log_failure_cannot_skip_cleanup_or_export_unbounded_bytes(tmp_path, monkeypatch):
    owner = runner(tmp_path)
    monkeypatch.setattr(campaign, "MAX_LOG_BYTES", 4)
    with pytest.raises(RuntimeError, match="bounded log budget"):
        owner.command("large-log", [sys.executable, "-c", "print('too much output', flush=True)"], 15)
    assert not (tmp_path / "logs/large-log.log").exists()
    assert_clean(owner, "large-log")


def test_public_log_is_copied_only_after_owned_cleanup(tmp_path, monkeypatch):
    owner = runner(tmp_path)
    real_stop, real_write = owner.manager.stop, Path.write_bytes
    stopped = []
    def stop(*args, **kwargs):
        value = real_stop(*args, **kwargs)
        stopped.append(True)
        return value
    def write_bytes(path, data):
        if path == tmp_path / "logs/order.log":
            assert stopped
        return real_write(path, data)
    monkeypatch.setattr(owner.manager, "stop", stop)
    monkeypatch.setattr(Path, "write_bytes", write_bytes)
    owner.command("order", [sys.executable, "-c", "print('done')"], 15)
    assert_clean(owner, "order")


def test_timeout_preserves_launch_token_and_cleans_real_descendants(tmp_path):
    import psutil
    from shard.managed_launch import _TOKEN_ENV
    owner = runner(tmp_path)
    evidence = owner.private / "descendant.json"
    grandchild = "import time; time.sleep(60)"
    child = ("import json,os,pathlib,psutil,subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
        f"pathlib.Path({str(evidence)!r}).write_text(json.dumps({{'pid':p.pid,"
        f"'birth':psutil.Process(p.pid).create_time(),'token':os.environ.get({_TOKEN_ENV!r})}}),encoding='utf-8'); "
        "time.sleep(60)")
    with pytest.raises(TimeoutError, match="owned process timeout"):
        owner.command("descendants", [sys.executable, "-c", child], 4)
    value = json.loads(evidence.read_text(encoding="utf-8"))
    assert isinstance(value["token"], str) and value["token"]
    assert owner.manager._process({"pid": value["pid"], "birth": value["birth"]}) is None
    assert_clean(owner, "descendants")


def test_command_cannot_override_owned_launch_token(tmp_path):
    from shard.managed_launch import _TOKEN_ENV
    owner = runner(tmp_path)
    with pytest.raises(ValueError, match="cannot override managed"):
        owner.command("bad-token", [sys.executable, "-c", "pass"], 15, {_TOKEN_ENV: "foreign-token"})
    assert owner.records == []

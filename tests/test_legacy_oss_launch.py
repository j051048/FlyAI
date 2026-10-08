"""Compatibility launchers must report failed deployments without starting a coordinator."""
from types import SimpleNamespace
import subprocess
import sys

import pytest

import launch_oss
import launch_ngram


def test_managed_remote_spawn_failure_is_not_ignored_or_leaked(monkeypatch):
    monkeypatch.setattr(launch_oss, "rssh", lambda *a: SimpleNamespace(
        returncode=7, stdout="private environment", stderr="private credential"))
    with pytest.raises(RuntimeError) as caught:
        launch_oss.fire({"id": 123}, "opaque remote command")
    assert "node 123" in str(caught.value) and "exit 7" in str(caught.value)
    assert "private" not in str(caught.value)


def test_detached_ssh_timeout_still_defers_to_readiness_barrier(monkeypatch):
    def detached(*args):
        raise subprocess.TimeoutExpired("ssh", 1)
    monkeypatch.setattr(launch_oss, "rssh", detached)
    launch_oss.fire({"id": 123}, "command", timeout=1)


def test_ngram_failed_warmup_exits_nonzero_before_driving_ring(monkeypatch):
    nodes = {i: {"id": i, "public_ipaddr": "127.0.0.1",
        "ports": {"29600/tcp": [{"HostPort": str(30000 + i)}]}} for i in (1, 2, 3)}
    launched = []
    monkeypatch.setattr(sys, "argv", ["launch_ngram.py", "--stages", "1,2,3", "--layers", "12,12,12"])
    monkeypatch.setattr(launch_ngram, "instances", lambda: nodes)
    monkeypatch.setattr(launch_ngram, "remote_config", lambda *a: {"num_hidden_layers": 36})
    monkeypatch.setattr(launch_ngram, "launch_stage_uneven", lambda *a, **k: launched.append(a[1]))
    monkeypatch.setattr(launch_ngram, "warm_stage", lambda *a: ("stage2 owned stage exited", False))
    monkeypatch.setattr(launch_ngram, "rssh", lambda *a, **k: pytest.fail("coordinator ran after failed warmup"))
    with pytest.raises(SystemExit) as caught:
        launch_ngram.main()
    assert caught.value.code and "stage2" in str(caught.value)
    assert launched == [2]


@pytest.mark.parametrize("failure", ["draft", "gateway"])
def test_oss_other_readiness_failures_cannot_report_success(monkeypatch, failure, capsys):
    nodes = {i: {"id": i, "public_ipaddr": "127.0.0.1",
        "ports": {"29600/tcp": [{"HostPort": str(30000 + i)}]}} for i in (1, 2, 3, 4)}
    args = ["launch_oss.py", "--stages", "1,2,3", "--coord", "4", "--skip-draft"]
    if failure == "gateway": args.append("--gateway")
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setitem(sys.modules, "launch_swarm", SimpleNamespace(mesh_rtt=lambda nodes:
        [[9999] * len(nodes) for _ in nodes]))
    monkeypatch.setattr(launch_oss, "instances", lambda: nodes)
    monkeypatch.setattr(launch_oss, "launch_stage", lambda *a, **k: None)
    monkeypatch.setattr(launch_oss, "warm_stage", lambda *a: ("ready", True))
    monkeypatch.setattr(launch_oss, "remote_config", lambda *a: {"num_hidden_layers": 36})
    monkeypatch.setattr(launch_oss, "fire", lambda *a: None)
    monkeypatch.setattr(launch_oss.time, "sleep", lambda *a: None)
    def remote(node, command, *args, **kwargs):
        if hasattr(command, "stdin"):
            pytest.fail("coordinator ran after failed readiness")
        ready = failure != "draft" and "grep -ciE" in command
        return SimpleNamespace(returncode=0, stdout="1\n" if ready else "0\n", stderr="")
    monkeypatch.setattr(launch_oss, "rssh", remote)
    with pytest.raises(SystemExit) as caught:
        launch_oss.main()
    assert caught.value.code and failure in str(caught.value)
    assert "[done]" not in capsys.readouterr().out

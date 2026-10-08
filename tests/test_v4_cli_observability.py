import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "engines" / "deepseek_v4"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))
import v4_pipe as vp


def configure(monkeypatch, stdin, *, strict=False):
    monkeypatch.setattr(vp.sys, "stdin", io.StringIO(stdin))
    monkeypatch.setattr(vp, "ring_args", lambda _: SimpleNamespace(n_layers=2))
    tokenizer = SimpleNamespace(eos_token_id=99, decode=lambda *a, **kw: "text")
    monkeypatch.setitem(sys.modules, "v4_tokenizer", SimpleNamespace(load_v4_tokenizer=lambda _: tokenizer))
    monkeypatch.setattr(vp, "connect_ring", lambda *a, **kw: (None, None))
    monkeypatch.setattr(vp.v4_levers, "report", lambda **kw: None)
    plan = {"stages": [{"signer_pubkey": "key-a", "lo": 0, "hi": 1}, {"signer_pubkey": "key-b", "lo": 1, "hi": 2}]}
    return SimpleNamespace(dir="unused", head="127.0.0.1:1", tail="127.0.0.1:2", timeout=10,
        connect_retry=1, receipts=False, session_config=SimpleNamespace(plan=plan) if strict else None)


def messages(output):
    return [(line.split(" ", 1)[0], json.loads(line.split(" ", 1)[1]))
        for line in output.splitlines() if line.startswith("SHARD_")]


def test_cli_measures_committed_callbacks_and_separates_receipt_cost(monkeypatch, capsys):
    args = configure(monkeypatch, json.dumps({"jobId": "test", "promptIds": [1], "maxNew": 3}) + "\n")
    ticks = iter([100.0, 101.0, 102.0, 103.0, 111.0])
    monkeypatch.setattr(vp.time, "perf_counter", lambda: next(ticks))
    def run(*a, on_token, **kw):
        for token in [2, 3, 4]:
            on_token(token)
        return {"tokens": [2, 3, 4], "receipts": [], "receipts_ok": None, "receipt_sweep_s": 5.0,
            "mode": "greedy", "coordinator_counters": {"frames_enqueued": 2, "frames_sent": 2,
                "replies_received": 2, "frames_judged": 2, "stale_replies": 0, "drained_replies": 0, "unsent_frames": 0,
                "accepted_predictions": 0, "proposed_predictions": 0, "cancel_events": 0}}
    monkeypatch.setattr(vp, "coordinate", run)
    assert vp._coord_cli(args) == 0
    output = messages(capsys.readouterr().out)
    ready = next(row for tag, row in output if tag == "SHARD_COORD_READY")
    done = next(row for tag, row in output if tag == "SHARD_JOB_DONE")
    assert ready["serviceReady"] is False
    assert done["tokenIds"] == [2, 3, 4]
    assert done["decodeTokPerSec"] == 1.0
    assert done["endToEndTokPerSec"] == 0.5
    assert done["coordinatorDiagnostics"]["timing"]["drain_s"] == 3.0
    assert done["coordinatorDiagnostics"]["timing"]["receipt_sweep_s"] == 5.0


def test_strict_cli_binds_job_and_refuses_bad_receipts(monkeypatch, capsys):
    args = configure(monkeypatch, json.dumps({"jobId": "j", "promptIds": [1], "maxNew": 2}) + "\n", strict=True)
    captured = {}
    def run(*a, on_token, **kwargs):
        captured.update(kwargs)
        on_token(2); on_token(3)
        return {"tokens": [2, 3], "receipts": [], "receipts_ok": False}
    monkeypatch.setattr(vp, "coordinate", run)
    vp._coord_cli(args)
    output = capsys.readouterr()
    rows = messages(output.out)
    assert not any(tag == "SHARD_JOB_DONE" for tag, _ in rows)
    assert any(tag == "SHARD_JOB_FATAL" for tag, _ in rows)
    assert "Traceback" in output.err
    assert captured["strict_job_binding"] is True
    assert captured["expected_by_signer"] == {"key-a": (0, 1), "key-b": (1, 2)}
    assert captured["receipts"] is True
    assert len(captured["nonce"]) == 64


def test_invalid_jobs_keep_persistent_coordinator_alive(monkeypatch, capsys):
    args = configure(monkeypatch, '[]\n{"jobId":"bad","maxNew":"oops"}\n{"jobId":"good","promptIds":[1],"maxNew":1}\n')
    def run(*a, on_token, **kwargs):
        on_token(2)
        return {"tokens": [2], "receipts": [], "receipts_ok": None}
    monkeypatch.setattr(vp, "coordinate", run)
    vp._coord_cli(args)
    rows = messages(capsys.readouterr().out)
    assert len([row for tag, row in rows if tag == "SHARD_JOB_FATAL"]) == 2
    assert [row["jobId"] for tag, row in rows if tag == "SHARD_JOB_DONE"] == ["good"]


def test_late_env_and_lazy_unloaded_module_are_distinct(monkeypatch):
    import v4_levers
    monkeypatch.delitem(sys.modules, "v4_dspark_moe", raising=False)
    monkeypatch.setenv("V4_DSPARK_MOE", "1")
    findings = {row.env: row for row in v4_levers.audit(v4_levers.STAGE)}
    assert findings["V4_DSPARK_MOE"].verdict == "UNJUDGED"
    assert "has not been imported" in findings["V4_DSPARK_MOE"].why

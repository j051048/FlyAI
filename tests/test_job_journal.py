"""Real SQLite/owner transactions and queue replay recovery; no model/GPU."""
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from shard.job_journal import (JobJournal, JournalError, JournalOwnerBusy, JournalOwnerLost, JournalBusy,
                              JournalConflict, JournalCapacity)
from shard.service_queue import ServiceQueue, TenantLimits, AdmissionError

COHORT = "a" * 64


def journal(tmp_path, **options):
    return JobJournal(tmp_path / "private" / "jobs.sqlite", max_result_bytes=8192, **options)


def record(*, job_id="job-1", tenant="a", key="stable", maximum=3):
    payload = {"label": "private-prompt", "sequence": [11, 12, 13]}
    return {"id": job_id, "tenant": tenant, "payload": payload,
        "context": {"model_id": "toy", "cohort_id": COHORT, "original_request": payload,
                    "request_fingerprint": "f" * 64},
        "prompt_tokens": 1, "max_new": maximum, "timeout_s": 60,
        "fingerprint": hashlib.sha256(json.dumps(payload).encode()).hexdigest(), "idempotency_key": key}


class ReplayBackend:
    model_id, cohort_id = "toy", COHORT
    supports_prefix_replay = True
    def __init__(self):
        self.calls = 0
        self.prefixes = []
    def execute(self, job, emit, check):
        self.calls += 1
        prefix = job.checkpoint()
        self.prefixes.append(prefix)
        for i, token in enumerate(job.payload["sequence"]):
            check()
            if i < len(prefix):
                if prefix[i] != token:
                    raise RuntimeError("original request replay prefix mismatch")
            else:
                emit(token)
        return {"tokens": job.checkpoint(), "text": "toy-result"}


def queue(log, backend=None, **options):
    return ServiceQueue(backend or ReplayBackend(), {"a": TenantLimits(100, 1000, 10000),
        "b": TenantLimits(100, 1000, 10000)}, journal=log, **options)


def submit(q, **options):
    return q.submit("a", {"label": "private-prompt", "sequence": [11, 12, 13]},
                    prompt_tokens=1, max_new=3, idempotency_key="stable", **options)[0]


def wait(job):
    end = time.monotonic() + 3
    with job.changed:
        while not job.terminal and time.monotonic() < end:
            job.changed.wait(.02)
    assert job.terminal


def test_same_file_other_namespace_rejected_before_second_owner(tmp_path):
    log = journal(tmp_path, namespace="first")
    lease = log.claim_owner()
    with pytest.raises(JournalError, match="namespace"):
        journal(tmp_path, namespace="second")
    assert log.assert_owner(lease)
    log.release_owner(lease)


def test_live_owner_exclusion_expiry_and_fencing_survive_reopen(tmp_path):
    now = [0.0]
    first = journal(tmp_path, clock=lambda: now[0])
    old = first.claim_owner(ttl_s=1)
    second = journal(tmp_path, clock=lambda: now[0])
    with pytest.raises(JournalOwnerBusy):
        second.claim_owner(ttl_s=1)
    now[0] = 2
    fresh = second.claim_owner(ttl_s=1)
    assert fresh.fence == old.fence + 1
    with pytest.raises(JournalOwnerLost):
        first.admit(old, record())
    with pytest.raises(JournalOwnerLost):
        first.release_owner(old)
    assert second.assert_owner(fresh)
    assert old.token not in repr(old)


def test_renewed_owner_old_job_handle_uses_database_expiry(tmp_path):
    now = [0.0]
    log = journal(tmp_path, clock=lambda: now[0])
    initial = log.claim_owner(ttl_s=1)
    now[0] = .5
    renewed = log.renew(initial, ttl_s=2)
    now[0] = 1.5
    assert log.assert_owner(initial)  # Same token/fence; copied expires is not authority.
    assert renewed.expires_at > initial.expires_at


def test_publication_transaction_serializes_takeover(tmp_path):
    now = [0.0]
    log = journal(tmp_path, clock=lambda: now[0])
    old = log.claim_owner(ttl_s=1)
    opened, release, acquired = threading.Event(), threading.Event(), threading.Event()
    events = []
    def publish():
        with log.publish_guard(old):
            events.append("old-start")
            opened.set()
            release.wait(2)
            events.append("old-end")
    def replace():
        lease = journal(tmp_path, clock=lambda: now[0]).claim_owner(ttl_s=2)
        with log.publish_guard(lease):
            events.append("new-start")
        acquired.set()
    a = threading.Thread(target=publish); a.start(); assert opened.wait(1)
    now[0] = 2
    b = threading.Thread(target=replace); b.start()
    assert not acquired.wait(.05)
    release.set(); a.join(2); b.join(2)
    assert acquired.is_set() and events == ["old-start", "old-end", "new-start"]


def test_prefix_sequence_tenant_idempotency_and_terminal_prompt_erasure(tmp_path):
    log = journal(tmp_path)
    lease = log.claim_owner()
    row, fresh = log.admit(lease, record())
    assert fresh and row["payload"]["label"] == "private-prompt"
    assert log.get(lease, tenant="b", job_id=row["id"]) is None
    assert log.admit(lease, record())[1] is False
    changed = record(); changed["fingerprint"] = "0" * 64
    with pytest.raises(JournalConflict):
        log.admit(lease, changed)
    log.running(lease, row["id"])
    log.append_token(lease, row["id"], 0, 11)
    with pytest.raises(JournalConflict):
        log.append_token(lease, row["id"], 0, 12)
    log.finish(lease, row["id"], "completed", result={"tokens": [11], "text": "private-output"})
    saved = log.get(lease, tenant="a", job_id=row["id"])
    assert saved["tokens"] == [11] and saved["payload"] == {}
    assert set(saved["context"]) == {"model_id", "cohort_id", "request_fingerprint"}
    assert saved["result"]["text"] == "private-output"


def test_cancellation_wins_durable_finalization_but_keeps_observed_prefix(tmp_path):
    log = journal(tmp_path)
    lease = log.claim_owner()
    row, _ = log.admit(lease, record())
    log.running(lease, row["id"])
    log.request_cancel(lease, row["id"], "cancel")
    log.append_token(lease, row["id"], 0, 11)  # Racing observed commit is retained.
    with pytest.raises(JournalConflict):
        log.finish(lease, row["id"], "completed", result={"tokens": [11]})
    log.finish(lease, row["id"], "cancelled", error="job cancelled", code="job_cancelled")
    assert log.get(lease, tenant="a", job_id=row["id"])["tokens"] == [11]


def test_capacity_reserves_full_result_and_ttl_removes_terminal_output(tmp_path):
    now = [0.0]
    log = journal(tmp_path, max_jobs=1, retention_s=1, clock=lambda: now[0])
    lease = log.claim_owner(ttl_s=100)
    log.admit(lease, record())
    with pytest.raises(JournalCapacity):
        log.admit(lease, record(job_id="job-2", key="next"))
    log.finish(lease, "job-1", "cancelled")
    now[0] = 2
    assert log.get(lease, tenant="a", job_id="job-1") is None
    assert log.admit(lease, record(job_id="job-2", key="next"))[1]


def test_sqlite_abort_does_not_publish_or_partially_append_token(tmp_path):
    log = journal(tmp_path)
    service = queue(log)
    job = submit(service)
    service._mark_running(job)
    connection = sqlite3.connect(log.path)
    connection.execute("CREATE TRIGGER fail_append BEFORE INSERT ON tokens BEGIN SELECT RAISE(ABORT,'simulated-cut'); END")
    connection.commit(); connection.close()
    with pytest.raises(JournalError):
        job.commit(11)
    assert job.tokens == []
    assert log.get(service._journal_owner, tenant="a", job_id=job.id)["tokens"] == []
    service.shutdown(preserve_pending=True)


def test_queue_completed_idempotent_response_survives_restart_without_execution(tmp_path):
    log = journal(tmp_path)
    first = queue(log); first.start()
    job = submit(first); wait(job)
    assert first.shutdown()
    backend = ReplayBackend()
    second = queue(journal(tmp_path), backend)
    same, fresh = second.submit("a", {"label": "private-prompt", "sequence": [11, 12, 13]},
        prompt_tokens=1, max_new=3, idempotency_key="stable")
    assert same.id == job.id and fresh is False and same.result == job.result
    assert same.tokens == [11, 12, 13] and backend.calls == 0
    assert "original_request" not in second._durable_context(same)
    second.shutdown()


def test_explicit_preserve_pending_recovers_same_job_with_original_deadline(tmp_path):
    first = queue(journal(tmp_path))
    original = submit(first)
    first.shutdown(preserve_pending=True)
    backend = ReplayBackend()
    restored = queue(journal(tmp_path), backend)
    job = restored.get("a", original.id)
    assert job.state == "recovering" and job.recovered and job.deadline <= original.deadline + .1
    restored.start(); wait(job)
    assert job.tokens == [11, 12, 13] and backend.calls == 1
    assert restored.shutdown()


def test_recovery_binding_mismatch_releases_prepared_resource(tmp_path):
    first = queue(journal(tmp_path)); submit(first); first.shutdown(preserve_pending=True)
    cleaned = []
    def bad(row):
        return {"payload": {"changed": True}, "prompt_tokens": row["prompt"], "max_new": row["maximum"],
                "bind": {}, "cleanup": lambda: cleaned.append(True)}
    restored = queue(journal(tmp_path), recovery_resolver=bad)
    with pytest.raises(JournalConflict):
        restored.recover_pending()
    assert cleaned == [True] and restored.metrics()["queued"] == 1
    restored.shutdown(preserve_pending=True)


def test_pending_recovery_waits_for_exact_cohort_and_rolling_quota_survives(tmp_path):
    limits = {"a": TenantLimits(4, 1, 100)}
    first = ServiceQueue(ReplayBackend(), limits, journal=journal(tmp_path))
    original = submit(first); first.shutdown(preserve_pending=True)
    backend = ReplayBackend(); backend.cohort_id = "b" * 64
    second = ServiceQueue(backend, limits, journal=journal(tmp_path))
    assert second.recover_pending() == 0
    assert second.get("a", original.id).state == "recovering"
    with pytest.raises(AdmissionError):
        second.submit("a", {"new": True}, prompt_tokens=1, max_new=1)
    second.shutdown(preserve_pending=True)


def test_stale_owner_worker_cannot_commit_or_publish_after_takeover(tmp_path):
    now = [0.0]
    service = queue(journal(tmp_path, clock=lambda: now[0]), journal_owner_ttl_s=1)
    job = submit(service); service._mark_running(job); job.commit(11)
    service._journal_stop.set(); service._journal_thread.join(1)  # Simulate suspended/crashed renewer.
    now[0] = 2
    fresh = journal(tmp_path, clock=lambda: now[0]).claim_owner(ttl_s=10)
    with pytest.raises(JournalOwnerLost):
        job.commit(12)
    with pytest.raises(JournalOwnerLost):
        with job.publish_guard():
            pytest.fail("stale response was published")
    assert job.tokens == [11]
    service._journal_fenced(JournalOwnerLost("test fence"))
    service.shutdown()
    assert journal(tmp_path, clock=lambda: now[0]).assert_owner(fresh)


def test_real_process_exit_replays_durable_prefix_from_original_payload(tmp_path):
    log = journal(tmp_path)
    code = '''
import os,sys,time
from shard.job_journal import JobJournal
from shard.service_queue import ServiceQueue,TenantLimits
class Backend:
 model_id='toy';cohort_id='a'*64
 def execute(self,job,emit,check):
  emit(11)
  os._exit(17)
q=ServiceQueue(Backend(),{'a':TenantLimits(4,30,1000)},journal=JobJournal(sys.argv[1],max_result_bytes=8192),journal_owner_ttl_s=.2)
q.submit('a',{'label':'private-prompt','sequence':[11,12,13]},prompt_tokens=1,max_new=3,idempotency_key='stable')
q.start()
time.sleep(5)
os._exit(18)
'''
    result = subprocess.run([sys.executable, "-c", code, str(log.path)], capture_output=True, timeout=10)
    assert result.returncode == 17, result.stderr.decode(errors="replace")
    end = time.monotonic() + 3
    while True:
        try:
            backend = ReplayBackend()
            restored = queue(journal(tmp_path), backend)
            break
        except JournalOwnerBusy:
            assert time.monotonic() < end
            time.sleep(.02)
    job = restored.durable_lookup("a", "stable")
    assert job.tokens == [11] and job.recovered
    restored.start(); wait(job)
    assert backend.prefixes == [[11]] and job.tokens == [11, 12, 13]
    assert job.state == "completed" and restored.shutdown()


def test_recovery_refuses_backend_without_prefix_check_contract(tmp_path):
    log = journal(tmp_path)
    old = queue(log); job = submit(old); old._mark_running(job); job.commit(11)
    old.shutdown(preserve_pending=True)
    backend = ReplayBackend(); backend.supports_prefix_replay = False
    new = queue(journal(tmp_path), backend)
    assert new.recover_pending() == 0
    restored = new.get("a", job.id)
    assert restored.state == "failed" and restored.tokens == [11] and backend.calls == 0
    new.shutdown()


def test_nested_publication_guard_and_job_snapshot_do_not_begin_second_writer(tmp_path):
    service = queue(journal(tmp_path))
    job = submit(service)
    with service.publish_guard():
        with job.publish_guard():
            assert job.checkpoint() == [] and job.snapshot()["state"] == "queued"
    service.shutdown()


def test_local_storage_fence_refuses_output_even_while_db_lease_is_live(tmp_path):
    service = queue(journal(tmp_path))
    job = submit(service)
    service._journal_fenced(JournalError("simulated storage failure"))
    assert service.journal.assert_owner(service._journal_owner)
    with pytest.raises(JournalOwnerLost):
        job.checkpoint()
    with pytest.raises(JournalOwnerLost):
        with job.publish_guard():
            pytest.fail("fenced local owner was allowed to respond")
    service.shutdown()


def test_renew_retries_busy_only_until_owner_expiry(tmp_path, monkeypatch):
    log = journal(tmp_path)
    real = log.renew
    calls = []
    def transient(lease, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise JournalBusy("temporary writer")
        return real(lease, **kw)
    monkeypatch.setattr(log, "renew", transient)
    service = queue(log, journal_owner_ttl_s=.6)
    end = time.monotonic() + 2
    while len(calls) < 2 and time.monotonic() < end:
        time.sleep(.01)
    assert len(calls) >= 2 and service._journal_failed is None
    service.assert_owner(); service.shutdown()
    log = journal(tmp_path)
    monkeypatch.setattr(log, "renew", lambda *a, **kw: (_ for _ in ()).throw(JournalBusy("blocked")))
    service = queue(log, journal_owner_ttl_s=.2)
    end = time.monotonic() + 2
    while service._journal_failed is None and time.monotonic() < end:
        time.sleep(.01)
    assert service._journal_failed == "JournalBusy"
    with pytest.raises(JournalOwnerLost):
        service.assert_owner()
    service.shutdown()


def test_io_counters_report_real_commit_wall_time_and_are_not_zero_cost_claim(tmp_path):
    log = journal(tmp_path)
    service = queue(log); service.start(); job = submit(service); wait(job)
    values = service.metrics()["journal_io"]
    assert values["token_commits"] == 3 and values["token_commit_seconds"] > 0
    assert values["transactions"] >= 3 and values["transaction_seconds"] >= values["token_commit_seconds"]
    assert "not isolated fsync" in values["scope"]
    service.shutdown()


def test_database_wait_does_not_extend_original_admission_deadline(tmp_path, monkeypatch):
    now = [0.0]
    log = journal(tmp_path, clock=lambda: now[0])
    service = queue(log, clock=lambda: 0.0)
    original = log.admit
    def delayed(owner, value):
        now[0] = 2
        return original(owner, value)
    monkeypatch.setattr(log, "admit", delayed)
    with pytest.raises(AdmissionError) as rejected:
        submit(service, timeout_s=1)
    assert rejected.value.status == 408
    assert not service._jobs and service.metrics()["queued"] == 0
    service.shutdown()


def test_deadline_winning_terminal_database_transition_does_not_fence_owner(tmp_path):
    now = [0.0]
    log = journal(tmp_path, clock=lambda: now[0])
    service = queue(log, clock=lambda: 0.0)
    job = submit(service, timeout_s=1)
    service._mark_running(job); job.commit(11)
    now[0] = 2
    service._finish(job, "completed", result={"tokens": [11], "text": "early result"})
    assert job.state == "expired" and job.result is None
    assert service._journal_failed is None and log.assert_owner(service._journal_owner)
    service.shutdown()

import sqlite3
from pathlib import Path
import pytest

from shard.job_journal import JobJournal, JournalError, JournalConflict, JournalOwnerLost
from test_job_journal import journal, record, queue, submit


def test_settled_group_is_one_transaction_and_keeps_per_token_sequence(tmp_path):
    log = journal(tmp_path)
    lease = log.claim_owner()
    log.admit(lease, record()); log.running(lease, "job-1")
    before = log.metrics()["transactions"]
    log.append_tokens(lease, "job-1", 0, [11, 12, 13])
    assert log.metrics()["transactions"] == before + 1
    saved = log.get(lease, tenant="a", job_id="job-1")
    assert saved["tokens"] == [11, 12, 13]
    assert log.metrics()["token_commits"] == 3


def test_middle_insert_failure_rolls_back_entire_group_before_publication(tmp_path):
    log = journal(tmp_path)
    service = queue(log); job = submit(service); service._mark_running(job)
    conn = sqlite3.connect(log.path)
    conn.execute("CREATE TRIGGER fail_second BEFORE INSERT ON tokens WHEN NEW.offset=1 BEGIN SELECT RAISE(ABORT,'cut'); END")
    conn.commit(); conn.close()
    with pytest.raises(JournalError):
        job.commit_many([11, 12, 13])
    assert job.tokens == [] and log.get(service._journal_owner, tenant="a", job_id=job.id)["tokens"] == []
    service.shutdown()


def test_reopen_recovers_whole_group_and_rejects_duplicate_or_overbudget(tmp_path):
    log = journal(tmp_path); lease = log.claim_owner()
    log.admit(lease, record()); log.running(lease, "job-1")
    log.append_tokens(lease, "job-1", 0, [11, 12])
    with pytest.raises(JournalConflict):
        log.append_tokens(lease, "job-1", 1, [13])
    with pytest.raises(JournalConflict):
        log.append_tokens(lease, "job-1", 2, [13, 14])
    assert journal(tmp_path).get(lease, tenant="a", job_id="job-1")["tokens"] == [11, 12]


def test_invalid_group_cannot_partially_append(tmp_path):
    log = journal(tmp_path); lease = log.claim_owner()
    log.admit(lease, record()); log.running(lease, "job-1")
    for value in ([], [11, True], [11, -1], list(range(65))):
        with pytest.raises(JournalError):
            log.append_tokens(lease, "job-1", 0, value)
    assert log.get(lease, tenant="a", job_id="job-1")["tokens"] == []


def test_concurrent_sqlite_wal_removal_does_not_kill_owner_heartbeat(tmp_path, monkeypatch):
    log = journal(tmp_path); lease = log.claim_owner()
    original = Path.stat
    def disappears(path, *args, **kwargs):
        if str(path) == str(log.path) + "-wal":
            raise FileNotFoundError("SQLite closed its last connection")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", disappears)
    assert log.assert_owner(lease)
    assert log.renew(lease, ttl_s=30).fence == lease.fence

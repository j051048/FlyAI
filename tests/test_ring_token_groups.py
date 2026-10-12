"""Already-settled groups retain durable frontiers through actual pool callbacks."""
from shard.job_journal import JobJournal
from shard.ring_pool import RingPool, RingState
from test_ring_pool import Backend, add_ready
from test_ring_router import service, body, wait
from test_continuous_ring_batch import BatchedBackend


class GroupedBackend(Backend):
    def __init__(self, *, race_cancel=False):
        super().__init__()
        self.race_cancel = race_cancel
        self.transactions = None

    def execute(self, job, emit, check):
        check()
        tokens = [65 + index for index in range(job.max_new)]
        if self.race_cancel:
            job.cancelled.set()  # arrives after the group has settled
        before = job._journal.metrics()["transactions"]
        emit.commit_many(tokens)
        self.transactions = job._journal.metrics()["transactions"] - before
        return {"tokens": job.checkpoint(), "text": self.decode(job.checkpoint()),
            "finish_reason": "length", "proof": {"verified": True, "scope": "complete_final_attempt"},
            "recovery": {"strategy": "cpu-fixture"}}


def test_serial_pool_preserves_group_and_uses_one_durable_transaction(tmp_path):
    pool, producer = RingPool(), GroupedBackend()
    add_ready(pool, backend=producer)
    journal = JobJournal(tmp_path / "private" / "jobs.sqlite")
    queue = service(pool, journal=journal); queue.start()
    try:
        job = queue.submit_request("a", body(max_tokens=4))[0]
        wait(job)
        assert job.state == "completed" and job.tokens == [65, 66, 67, 68]
        assert producer.transactions == 1
        saved = journal.get(queue._journal_owner, tenant="a", job_id=job.id)
        assert saved["tokens"] == job.tokens
    finally:
        queue.shutdown()


def test_racing_cancellation_keeps_settled_group_without_success_claim(tmp_path):
    pool, producer = RingPool(), GroupedBackend(race_cancel=True)
    add_ready(pool, backend=producer)
    journal = JobJournal(tmp_path / "private" / "jobs.sqlite")
    queue = service(pool, journal=journal); queue.start()
    try:
        job = queue.submit_request("a", body(max_tokens=4))[0]
        wait(job)
        assert job.state == "cancelled" and job.tokens == [65, 66, 67, 68] and job.result is None
        assert producer.transactions == 1
        assert journal.get(queue._journal_owner, tenant="a", job_id=job.id)["tokens"] == job.tokens
    finally:
        queue.shutdown()


def test_batch_group_budget_is_checked_before_any_partial_commit():
    class Overrun(BatchedBackend):
        def advance_batch(self, jobs, *, emit, cancel_check, max_step_tokens):
            emit[jobs[0].id].commit_many(list(range(max_step_tokens + 1)))
            raise AssertionError("invalid group should have been rejected")
    pool = RingPool(); producer = Overrun(); ring = add_ready(pool, backend=producer)
    queue = service(pool, batching={"enabled": True, "max_step_tokens": 2}); queue.start()
    try:
        job = queue.submit_request("a", body(max_tokens=4))[0]
        wait(job)
        assert job.state == "failed" and job.tokens == [] and ring.state == RingState.FAILED
    finally:
        queue.shutdown()

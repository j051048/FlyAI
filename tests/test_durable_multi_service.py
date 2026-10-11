"""Owner-fenced HTTP/SSE restart with actual gateway replay and signed CPU receipts."""
import http.client
import json
import threading
import time

import pytest

import v4_gateway as GW
from shard.job_journal import JobJournal
from shard.ring_pool import RingPool
from shard.service_queue import TenantLimits
from test_ring_pool import add_ready, COHORT, OTHER
from test_v4_gateway import Pipe, backend, KEY_A, request


class InterruptiblePipe(Pipe):
    hold = False
    def coordinate(self, pipe, ret, *args, cancel_check, expected_by_signer, strict_job_binding=False, **kwargs):
        if self.hold:
            kwargs["on_token"](65)
            self.started.set()
            assert ret.closed.wait(3)
            raise OSError("coordinator stopped after an actually committed token")
        return super().coordinate(pipe, ret, *args, cancel_check=cancel_check,
            expected_by_signer=expected_by_signer, strict_job_binding=strict_job_binding, **kwargs)
    coordinate_dspark = coordinate
    coordinate_dspark_pipelined = coordinate


def serving(path, *, cohort=COHORT):
    producer = InterruptiblePipe()
    adapter = backend(producer)
    pool = RingPool(); add_ready(pool, backend=adapter, cohort=cohort)
    pool.set_alias("default", producer.V4_MODEL_ID, cohort)
    auth = GW.AuthRegistry({KEY_A: "a"}, {"a": TenantLimits(10, 100, 10000)})
    gateway = GW.Gateway(auth=auth, ring_pool=pool, default_model="default",
        job_journal=JobJournal(path), journal_owner_ttl_s=1)
    server = gateway.server()
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    return producer, gateway, server, thread


def close(gateway, server, thread, *, preserve=False):
    server.shutdown(); gateway.shutdown(preserve_pending=preserve)
    server.server_close(); thread.join(1)


def body():
    return {"model": "default", "messages": [{"role": "user", "content": "original request"}],
            "max_tokens": 4, "stream": True, "timeout_s": 10}


def interrupted(path):
    producer, gateway, server, thread = serving(path)
    producer.hold = True; producer.started.clear()
    connection = http.client.HTTPConnection(*server.server_address, timeout=3)
    connection.request("POST", "/v1/chat/completions", json.dumps(body()),
        {"Authorization": "Bearer " + KEY_A, "Idempotency-Key": "same-original-job", "Content-Type": "application/json"})
    response = connection.getresponse()
    assert response.status == 200 and producer.started.wait(1)
    job = next(job for job in gateway.jobs._jobs.values() if job.tenant == "a")
    for _ in range(8):
        if '"content": "A"' in response.readline().decode():
            break
    else:
        pytest.fail("committed prefix was not streamed")
    assert job.tokens == [65]
    close(gateway, server, thread, preserve=True)
    connection.close()
    return job.id


def test_restart_replays_original_and_sse_cursor_does_not_duplicate_prefix(tmp_path):
    path = tmp_path / "private-state" / "jobs.sqlite"
    job_id = interrupted(path)
    producer, gateway, server, thread = serving(path)
    try:
        status, output = request(server.server_address, "POST", "/v1/chat/completions", body(),
            headers={"Idempotency-Key": "same-original-job", "Last-Event-ID": f"{job_id}:1:1"})
        assert status == 200
        output = output.decode()
        frames = [json.loads(line[6:]) for line in output.splitlines() if line.startswith("data: {")]
        text = "".join(frame.get("choices", [{}])[0].get("delta", {}).get("content", "") for frame in frames)
        assert text == "BCD"
        restored = gateway.jobs.get("a", job_id)
        assert restored.state == "completed" and restored.tokens == [65, 66, 67, 68]
        assert restored.result["recovery"]["replayed_committed_tokens"] == 1
        assert producer.recorded[-1][:2] == ([1, 2], 4)
        assert gateway.jobs.get("b", job_id) is None
    finally:
        close(gateway, server, thread)


def test_restart_never_rebinds_pending_work_to_another_cohort(tmp_path):
    path = tmp_path / "private-state" / "jobs.sqlite"
    job_id = interrupted(path)
    producer, gateway, server, thread = serving(path, cohort=OTHER)
    try:
        assert gateway.jobs.get("a", job_id).state == "recovering"
        assert gateway.jobs.recover_pending() == 0
        assert all(len(record) >= 2 and record[1] == 2 for record in producer.recorded)  # warmup only
        assert sum(r.bound_jobs for r in gateway.backend.pool.rings()) == 0
    finally:
        close(gateway, server, thread)

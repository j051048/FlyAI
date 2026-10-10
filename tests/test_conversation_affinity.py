"""New admissions prefer tenant-scoped sessions, held bindings never move."""
import pytest

from shard.ring_router import SessionAffinity
from shard.ring_pool import RingPool, RingState
from test_ring_pool import add_ready, COHORT, OTHER
from test_ring_router import service, body
from shard.service_queue import AdmissionError


def test_affinity_is_tenant_cohort_scoped_bounded_and_expires():
    now = [0.0]
    hints = SessionAffinity(max_entries=2, ttl_s=10, clock=lambda: now[0])
    hints.remember("a", COHORT, "session", "first")
    assert hints.preferred("b", COHORT, "session") is None
    assert hints.preferred("a", OTHER, "session") is None
    hints.remember("b", COHORT, "second", "second")
    assert hints.preferred("a", COHORT, "session") == "first"
    hints.remember("c", COHORT, "third", "third")
    assert hints.preferred("b", COHORT, "second") is None
    now[0] = 11
    assert hints.preferred("a", COHORT, "session") is None


def test_pool_preferred_ring_respects_cohort_readiness_and_fixed_binding():
    pool = RingPool()
    a, b = add_ready(pool, "a"), add_ready(pool, "b")
    first = pool.acquire("test-model", COHORT, estimated_tokens=1)
    following = pool.acquire("test-model", COHORT, estimated_tokens=1, preferred_ring_id="a")
    assert first.ring is following.ring is a
    pool.drain("a")
    fallback = pool.acquire("test-model", COHORT, estimated_tokens=1, preferred_ring_id="a")
    assert fallback.ring is b and first.ring is a and a.state == RingState.DRAINING
    for binding in (first, following, fallback):
        pool.release(binding)
    pool.shutdown()


def test_queue_same_session_prefers_old_ring_but_other_tenant_does_not():
    pool = RingPool()
    a, b = add_ready(pool, "a"), add_ready(pool, "b")
    queue = service(pool, session_affinity=True)
    first = queue.submit_request("a", body("first", shard_session_id="s"))[0]
    second = queue.submit_request("a", body("second", shard_session_id="s"))[0]
    assert first.ring_binding.ring is second.ring_binding.ring is a
    other = queue.submit_request("b", body("other", shard_session_id="s"))[0]
    assert other.ring_binding.ring is b
    assert first.conversation_session_id == "s"
    pool.drain("a")
    replacement = queue.submit_request("a", body("following", shard_session_id="s"))[0]
    assert replacement.ring_binding.ring is b and first.ring_binding.ring is a
    queue.shutdown()


def test_queue_affinity_is_default_off_and_session_validated():
    pool = RingPool()
    a, b = add_ready(pool, "a"), add_ready(pool, "b")
    queue = service(pool)
    first = queue.submit_request("a", body("first", shard_session_id="s"))[0]
    second = queue.submit_request("a", body("second", shard_session_id="s"))[0]
    assert first.ring_binding.ring is a and second.ring_binding.ring is b
    with pytest.raises(AdmissionError):
        queue.submit_request("a", body(shard_session_id={"fake": "tenant"}))
    queue.shutdown()

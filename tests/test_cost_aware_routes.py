"""Admission decisions with synthetic timings; never GPU performance evidence."""
import copy
import time

import pytest

from shard.ring_pool import RingPool
from shard.routing_cost import RoutingCosts
from shard.service_queue import AdmissionError
from test_ring_pool import Backend, COHORT, add_ready
from test_ring_router import body, service, wait


class PreparedBackend(Backend):
    def __init__(self, **options):
        super().__init__(**options)
        self.preparations, self.validations = 0, 0
        self.hint = None

    def routing_identity(self):
        return {"cohort_id": COHORT, "runtime_config_sha256": "c" * 64}

    def prepare(self, request):
        self.preparations += 1
        payload, _, maximum, timeout = super().prepare(request)
        return payload, len(payload["label"]), maximum, timeout

    def validate_prepared_route(self, prepared, request):
        self.validations += 1
        return copy.deepcopy(prepared)

    def routing_cache_hint(self, payload, *, tenant, request_fingerprint, now):
        return self.hint(request_fingerprint, now) if callable(self.hint) else self.hint


def sample(ring, count, prefill, decode=.1, now=None):
    if now is not None:
        ring.routing_costs = RoutingCosts(clock=lambda: now[0], ttl_s=30)
    assert ring.routing_costs.observe(payload={"label": "irrelevant"}, prompt_tokens=count, token_count=3,
        first_s=prefill, last_s=prefill + decode * 2, elapsed_s=prefill + decode * 2 + .05,
        identity=ring.backend.routing_identity())


def test_tokenized_prompt_cost_selects_ring_before_reserving_and_encodes_once():
    pool = RingPool()
    a, b = [add_ready(pool, name, backend=PreparedBackend()) for name in ("a", "b")]
    sample(a, 200, 8)
    sample(b, 200, .5)
    queue = service(pool)
    job = queue.submit_request("a", body("x" * 200))[0]
    assert job.ring_binding.ring is b and a.bound_jobs == 0
    assert a.backend.preparations == 1 and b.backend.preparations == 0 and b.backend.validations == 1
    assert job.ring_binding.admission["source"] == "local_verified_service_phases"
    assert job.ring_binding.admission["prefill_s"] == .5
    queue.shutdown()


def test_same_session_does_not_force_overloaded_ring_and_region_stays_local_first():
    pool = RingPool()
    a, b = [add_ready(pool, name, backend=PreparedBackend(), region="asia") for name in ("a", "b")]
    remote = add_ready(pool, "remote", backend=PreparedBackend(), region="europe")
    remote.seconds_per_token = .0001
    queue = service(pool, session_affinity=True)
    first = queue.submit_request("a", body("first", shard_region="asia", shard_session_id="session"))[0]
    following = queue.submit_request("a", body("first", shard_region="asia", shard_session_id="session"))[0]
    assert first.ring_binding.ring is a and following.ring_binding.ring is b
    assert remote.bound_jobs == 0
    queue.shutdown()


def test_exact_cache_net_savings_can_outweigh_queue_but_not_unknown_restore_cost():
    pool = RingPool()
    a, b = [add_ready(pool, name, backend=PreparedBackend()) for name in ("a", "b")]
    for ring in (a, b):
        sample(ring, 200, 5)
    a.reserved_seconds = 1
    a.backend.hint = lambda fingerprint, now: {"schema": "shard-routing-cache-hint/1",
        **a.backend.routing_identity(), "request_fingerprint": fingerprint, "prefix_tokens": 200,
        "restore_seconds": .1, "first_token_seconds": .2, "measured_at": now, "ttl_s": 20,
        "source": "verified_exact_prefix_restore", "skips_prefill": True}
    queue = service(pool)
    job = queue.submit_request("a", body("x" * 200))[0]
    assert job.ring_binding.ring is a and job.ring_binding.admission["cache_saved_s"] == 4.8
    queue.cancel("a", job.id)
    a.backend.hint = lambda fingerprint, now: {"schema": "shard-routing-cache-hint/1",
        **a.backend.routing_identity(), "request_fingerprint": fingerprint, "prefix_tokens": 200,
        "measured_at": now, "ttl_s": 20, "source": "verified_exact_prefix_restore", "skips_prefill": True}
    second = queue.submit_request("a", body("x" * 200))[0]
    assert second.ring_binding.ring is b
    queue.shutdown()


@pytest.mark.parametrize("field,value", [("runtime_config_sha256", "d" * 64), ("cohort_id", "b" * 64),
    ("request_fingerprint", "other"), ("prefix_tokens", 199), ("skips_prefill", False),
    ("measured_at", 0), ("restore_seconds", float("nan")), ("first_token_seconds", .01)])
def test_stale_wrong_or_shadow_cache_hint_never_discounted(field, value):
    backend = PreparedBackend()
    costs = RoutingCosts(clock=lambda: 100.0)
    costs.observe(payload={}, prompt_tokens=200, token_count=3, first_s=5, last_s=6, elapsed_s=6.1,
                  identity=backend.routing_identity())
    backend.hint = {"schema": "shard-routing-cache-hint/1", **backend.routing_identity(),
        "request_fingerprint": "req", "prefix_tokens": 200, "restore_seconds": .1,
        "first_token_seconds": .2, "measured_at": 100, "ttl_s": 20,
        "source": "verified_exact_prefix_restore", "skips_prefill": True, field: value}
    result = costs.estimate(backend, cohort=COHORT, payload={}, prompt_tokens=200, max_new_tokens=3,
                           fallback_s_per_token=1, tenant="a", request_fingerprint="req")
    assert result["cache_saved_s"] == 0
    assert result["uncertainty"] == ["cache_hint_unusable"]


def test_phase_observations_expire_and_never_cross_shape_or_runtime():
    pool = RingPool()
    ring = add_ready(pool, backend=PreparedBackend())
    now = [100.0]
    sample(ring, 10, .1, now=now)
    def forecast(count=10):
        return ring.routing_costs.estimate(ring.backend, cohort=COHORT, payload={}, prompt_tokens=count,
            max_new_tokens=3, fallback_s_per_token=1, tenant="a", request_fingerprint="req")
    assert forecast()["fresh_samples"] == 1
    assert forecast(11)["fresh_samples"] == 0
    now[0] = 131
    assert forecast()["fresh_samples"] == 0
    pool.shutdown()


def test_live_completed_proof_teaches_cost_but_failed_request_does_not():
    class Timed(PreparedBackend):
        def execute(self, job, emit, check):
            time.sleep(.002)
            def delayed(token):
                time.sleep(.002)
                emit(token)
            return super().execute(job, delayed, check)
    pool = RingPool()
    ring = add_ready(pool, backend=Timed())
    queue = service(pool)
    queue.start()
    job = queue.submit_request("a", body("measured"))[0]
    wait(job)
    assert job.state == "completed"
    next_job = queue.submit_request("a", body("measured"))[0]
    assert next_job.ring_binding.admission["fresh_samples"] == 1
    queue.shutdown()


def test_legacy_small_context_rejection_does_not_hide_valid_larger_context_ring():
    class SmallContext(Backend):
        def prepare(self, body):
            raise AdmissionError("small context", status=400, code="context_length_exceeded")
    pool = RingPool()
    large = add_ready(pool, "large")
    small = add_ready(pool, "small", backend=SmallContext())
    small.seconds_per_token = .01
    queue = service(pool)
    try:
        job = queue.submit_request("a", body("long prompt"))[0]
        assert job.ring_binding.ring is large
        assert small.bound_jobs == 0 and small.reserved_seconds == 0
    finally:
        queue.shutdown()

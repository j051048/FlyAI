"""Bounded real admission selection using synthetic local service measurements."""
from copy import deepcopy

import pytest

from shard.ring_pool import RingPool
from shard.routing_cost import RoutingCosts, production_policy
from test_cost_aware_routes import PreparedBackend
from test_ring_pool import COHORT, add_ready


def policy():
    return {"enabled": True, "context_bucket_tokens": 128, "output_bucket_tokens": 32,
        "min_samples": 3, "uncertainty_fraction": .2, "explore_every": 10, "max_probe_extra_s": 1.}


def train(costs, backend, *, prefill=2, count=200, maximum=16, payload=None):
    for _ in range(3):
        assert costs.observe(payload=payload or {"mode": "greedy"}, prompt_tokens=count, token_count=4,
            first_s=prefill, last_s=prefill+.3, elapsed_s=prefill+.35,
            identity=backend.routing_identity(), max_new_tokens=maximum)


def test_unseen_prompt_lengths_use_bounded_runtime_bound_forecast_only_when_opted_in():
    backend = PreparedBackend(); costs = RoutingCosts(production=policy())
    train(costs, backend)
    args = dict(backend=backend, cohort=COHORT, payload={"mode": "greedy"}, prompt_tokens=210,
        max_new_tokens=16, fallback_s_per_token=1, tenant="t", request_fingerprint="new-content")
    estimate = costs.estimate(**args)
    assert estimate["source"] == "bounded_local_shape_forecast" and estimate["fresh_samples"] == 3
    assert estimate["service_s"] > 3.5 and estimate["cache_saved_s"] == 0
    assert "bucket_forecast_not_exact_workload" in estimate["uncertainty"]
    assert costs.estimate(**{**args, "prompt_tokens": 300})["fresh_samples"] == 0
    assert costs.estimate(**{**args, "payload": {"mode": "dspark"}})["fresh_samples"] == 0
    plain = RoutingCosts(); train(plain, backend)
    assert plain.estimate(**args)["fresh_samples"] == 0


def test_stale_runtime_different_contract_and_cache_hit_do_not_train_buckets():
    now = [100.]; backend = PreparedBackend()
    costs = RoutingCosts(clock=lambda: now[0], ttl_s=10, production=policy())
    train(costs, backend)
    args = dict(backend=backend, cohort=COHORT, payload={"mode": "greedy"}, prompt_tokens=210,
        max_new_tokens=16, fallback_s_per_token=1, tenant="t", request_fingerprint="new")
    assert costs.estimate(**{**args, "payload": {"mode": "greedy", "numerical_contract": "other"}})["fresh_samples"] == 0
    now[0] += 11
    assert costs.estimate(**args)["fresh_samples"] == 0
    changed = PreparedBackend(); changed.routing_identity = lambda: {"cohort_id": COHORT, "runtime_config_sha256": "f"*64}
    assert costs.estimate(**{**args, "backend": changed})["fresh_samples"] == 0


def test_ready_ring_choice_uses_shape_forecast_and_exploration_preserves_locality():
    pool = RingPool(routing_policy=policy())
    a, b = [add_ready(pool, name, backend=PreparedBackend(), region="asia") for name in ("a", "b")]
    remote = add_ready(pool, "remote", backend=PreparedBackend(), region="europe")
    train(a.routing_costs, a.backend, prefill=8)
    train(b.routing_costs, b.backend, prefill=1)
    request = ({"mode": "greedy"}, 210, 16, 10)
    prepared = {r.ring_id: (r, request) for r in (a, b, remote)}
    selected = pool.acquire_prepared(a.model_id, COHORT, prepared, tenant="t", request_fingerprint="new", region="asia")
    assert selected.ring is b and selected.admission["source"] == "bounded_local_shape_forecast"
    pool.release(selected)
    # Near-cost fresh runtimes can receive at most one probe in ten new requests.
    probe = add_ready(pool, "probe", backend=PreparedBackend(), region="asia")
    probe.seconds_per_token = .01
    a.seconds_per_token = b.seconds_per_token = .01
    unseen = ({"mode": "greedy"}, 40, 8, 10)
    prepared = {r.ring_id: (r, unseen) for r in (a, b, probe, remote)}
    decisions = []
    for _ in range(20):
        binding = pool.acquire_prepared(a.model_id, COHORT, prepared, tenant="t", request_fingerprint="u", region="asia")
        decisions.append(binding.admission["production_probe"])
        assert binding.ring is not remote
        pool.release(binding)
    assert sum(decisions) == 2
    pool.shutdown()


@pytest.mark.parametrize("key,value", [("explore_every", 1), ("min_samples", 1), ("max_probe_extra_s", 11),
    ("uncertainty_fraction", float("nan")), ("enabled", 1)])
def test_policy_is_complete_local_and_bounded(key, value):
    item = policy(); item[key] = value
    with pytest.raises(ValueError): production_policy(item)

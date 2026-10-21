from copy import deepcopy
import pytest
from engines.deepseek_v4.v4_prefill_budget import TimeBudget, validate_config


def config():
    binding = dict(numerical_contract="batch/1", cohort_id="c", source_sha256="a" * 64,
                   runtime_config_sha256="b" * 64, max_context_tokens=8192, max_jobs=4)
    return dict(schema="flyai-v4-prefill-budget/1", binding=binding,
        measured_at=100, ttl_s=300, quota_ms=20, samples=[
            dict(context_tokens=4096, batch_jobs=4, quantum_tokens=q, elapsed_ms=[ms] * 5)
            for q, ms in ((1, 4), (2, 9), (4, 17), (8, 35))])


def test_measured_budget_bounds_real_quantum_and_unseen_or_stale_shapes_fall_back():
    c = config(); b = TimeBudget(c, c["binding"])
    assert b.choose(candidates=range(1, 9), context=4096, batch_jobs=4, now=150) == 4
    assert b.choose(candidates=range(1, 9), context=128, batch_jobs=2, now=150) == 1
    assert b.choose(candidates=range(1, 9), context=4097, batch_jobs=2, now=150) == 1
    assert b.choose(candidates=range(1, 9), context=128, batch_jobs=2, now=401) == 1
    assert b.choose(candidates=range(1, 9), context=128, batch_jobs=2, now=99) == 1
    assert b.status()["baseline_decisions"] == 4


def test_unknown_budget_never_invents_token_cost_or_exceeds_declared_quantum():
    c = config(); b = TimeBudget(None, c["binding"])
    assert b.choose(candidates=[1, 2], context=0, batch_jobs=4) == 1
    b = TimeBudget(c, c["binding"])
    assert b.choose(candidates=[1, 2], context=4096, batch_jobs=4, now=150) == 2
    bad = deepcopy(c["binding"]); bad["source_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="another runner"): TimeBudget(c, bad)


@pytest.mark.parametrize("change", [
    lambda c: c.update(quota_ms=True), lambda c: c.update(ttl_s=86401),
    lambda c: c["samples"][0].update(elapsed_ms=[float("nan")] * 5),
    lambda c: c["samples"][0].update(elapsed_ms=[1] * 4),
    lambda c: c["samples"].append(deepcopy(c["samples"][0])),
    lambda c: c["samples"][0].update(batch_jobs=5)])
def test_corrupt_cost_measurements_rejected(change):
    c = config(); change(c)
    with pytest.raises(ValueError): validate_config(c)

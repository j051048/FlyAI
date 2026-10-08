import copy

import pytest

import v4_soak as acceptance
import v4_benchmark as bench
from test_v4_benchmark import Clock, FakeAdapter, protocol


def campaign(*, backend="live_ring"):
    p, keys = protocol()
    clock = Clock()
    adapter = FakeAdapter(p, keys, clock)
    # Synthetic signed fixtures exercise verifier branches; these are not measurements.
    adapter.backend = backend
    report = acceptance.run_soak(adapter, p, cycles=1, duration_s=10,
                                 ttft_p95_s=3, token_gap_p95_s=.1, clock=clock)
    return report, p


def test_soak_reverifies_every_committed_stream_and_signed_job():
    report, p = campaign()
    result = acceptance.evaluate_soak(report, p, report["contract"])
    assert result["status"] == "passed", result
    assert result["jobs"] == 8
    assert result["summary"]["median_decode_tok_s"] == pytest.approx(50)
    assert result["summary"]["ttft_p95_s"] > 2


def test_mock_does_not_qualify_hardware_campaign():
    report, p = campaign(backend="mock")
    assert acceptance.evaluate_soak(report, p, report["contract"])["status"] == "unverified"


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(complete=False),
    lambda r: r.update(elapsed_s=1),
    lambda r: r.update(elapsed_s=r["elapsed_s"] + 3600),
    lambda r: r.update(active_observation_s=r["active_observation_s"] + 3600),
    lambda r: r["samples"][1].update(campaign_interval_s=r["samples"][0]["campaign_interval_s"]),
    lambda r: r["samples"].pop(),
    lambda r: r["samples"][0]["tokens"].__setitem__(0, 999),
    lambda r: r["samples"][0]["receipts"][0].update(job_id="forged"),
    lambda r: r["samples"][0]["measurement"].update(decode_committed_tok_s=999),
    lambda r: r["samples"][1].update(nonce=r["samples"][0]["nonce"]),
])
def test_incomplete_or_fabricated_soak_cannot_pass(mutation):
    report, p = campaign()
    contract = copy.deepcopy(report["contract"])
    mutation(report)
    assert acceptance.evaluate_soak(report, p, contract)["status"] == "failed"


def test_relaxing_saved_slo_cannot_override_independent_contract():
    report, p = campaign()
    contract = copy.deepcopy(report["contract"])
    report["contract"]["ttft_p95_s"] = 999
    assert acceptance.evaluate_soak(report, p, contract)["status"] == "failed"


def test_runtime_metrics_can_qualify_only_gpu_mode_with_real_signed_fields():
    p, keys = protocol(metrics=True)
    clock = Clock()
    adapter = FakeAdapter(p, keys, clock)
    adapter.backend = "live_ring"
    report = acceptance.run_soak(adapter, p, cycles=1, duration_s=10,
                                 ttft_p95_s=3, token_gap_p95_s=.1, clock=clock)
    assert acceptance.evaluate_soak(report, p, report["contract"])["status"] == "passed"


def test_cohosted_benchmark_requires_explicit_different_protocol():
    p, _ = protocol()
    p["hardware"][1]["host_id"] = p["hardware"][0]["host_id"]
    assert bench.protocol_errors(bench.seal_protocol(p))
    p["isolation"] = "none"
    assert bench.protocol_errors(bench.seal_protocol(p)) == []
    p["hardware"][1]["gpu_uuid"] = p["hardware"][0]["gpu_uuid"]
    assert bench.protocol_errors(bench.seal_protocol(p))

import base64
from copy import deepcopy
import pytest

from phase0.v4_profile_report import explain
from shard.receipt import _canonical
from shard.runtime_profile import RuntimeProfiler
from shard.runtime_metrics import RuntimeMetrics
from test_v4_benchmark import observed_protocol, observed_report


def profiled_report():
    protocol, keys = observed_protocol()
    report = observed_report(protocol, keys)
    for sample in report["samples"]:
        for index, receipt in enumerate(sample["receipts"]):
            profiler = RuntimeProfiler()
            profiler.record("stage.send.host", 2.)
            profiler.record("main.decode.graph_replay.gpu", 1.)
            metrics = RuntimeMetrics("cuda:0").snapshot()
            metrics["performance"] = profiler.snapshot()
            receipt["runtime_metrics"] = metrics
            receipt["sig"] = base64.b64encode(keys[index].sign(_canonical(receipt))).decode()
    return report, protocol


def test_signed_profile_diagnosis_keeps_sampling_and_overlap_limits():
    report, protocol = profiled_report()
    result = explain(report, protocol)
    assert result["status"] == "diagnostic_only" and len(result["stage_profiles"]) == 48
    assert result["interpretation"]["cpu_fraction"] is None
    assert result["stage_profiles"][0]["gpu_sample_every"] == 16


def test_profile_substitution_is_rejected_even_with_valid_geometry():
    report, protocol = profiled_report()
    sample = next(s for s in report["samples"] if s["phase"] == "warm")
    sample["receipts"][0]["runtime_metrics"]["performance"]["phases"]["stage.send.host"]["total_ms"] = 3.
    with pytest.raises(Exception):
        explain(report, protocol)


def test_profile_report_refuses_missing_nonce_and_mock_evidence():
    report, protocol = profiled_report()
    sample = next(s for s in report["samples"] if s["phase"] == "warm"); sample["nonce"] = None
    with pytest.raises(ValueError, match="nonce"):
        explain(report, protocol)
    report["backend"] = "mock"
    with pytest.raises(ValueError, match="CPU/mock"):
        explain(report, protocol)

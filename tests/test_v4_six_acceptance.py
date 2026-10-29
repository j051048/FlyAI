"""Frozen acceptance cannot promote plans, skips or CPU evidence to GPU success."""
import json
from pathlib import Path

import pytest

from tools import v4_six_acceptance as S


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    source = {"files": [{"name": "test.py", "sha256": "a" * 64}], "sha256": "b" * 64}
    monkeypatch.setattr(S, "source_identity", lambda: source)
    monkeypatch.setattr(S, "_local_files", lambda: ["tests/test_v4_device_digest.py"])
    path = tmp_path / "plan.json"
    plan = S.prepare(path)
    return path, plan, source


def test_cpu_plan_has_no_native_gpu_qualification_and_binds_all_six(frozen):
    path, plan, _ = frozen
    assert S.require_plan(path) == plan
    assert len(plan["obligations"]) == 6
    assert plan["native_gpu_verified"] is False
    with pytest.raises(ValueError, match="fresh"):
        S.prepare(path)


def test_changed_source_or_fabricated_gpu_status_invalidates_plan(frozen):
    path, plan, source = frozen
    source["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="changed"):
        S.require_plan(path)
    source["sha256"] = "b" * 64
    plan["native_gpu_verified"] = True
    plan["sha256"] = S.A.digest({k: v for k, v in plan.items() if k != "sha256"})
    S.A.write(path, plan)
    with pytest.raises(ValueError, match="fabricated"):
        S.require_plan(path)


@pytest.mark.parametrize("execute,devices", [(False, "0"), (True, ""), (True, "0,0"), (True, "0;1")])
def test_gpu_commands_need_explicit_unique_idle_assignment(frozen, tmp_path, execute, devices):
    path, _, _ = frozen
    directory = tmp_path / "not-created"
    with pytest.raises(ValueError, match="idle"):
        S.checks(path, directory, gpu=True, execute=execute, devices=devices)
    assert not directory.exists()


def test_skipped_gpu_case_is_failed_evidence_with_command_exit_retained(frozen, tmp_path, monkeypatch):
    path, _, _ = frozen
    class Runner:
        def __init__(self, directory, python):
            self.records = []
        def command(self, name, argv, timeout, environment):
            self.records.append({"name": name, "exit_code": 0, "argv": argv})
            Path(argv[argv.index("--junitxml") + 1]).write_text(
                '<testsuite><testcase name="required-cuda"><skipped/></testcase></testsuite>')
    monkeypatch.setattr(S.A, "Runner", Runner)
    directory = tmp_path / "gpu-evidence"
    with pytest.raises(ValueError, match="skipped"):
        S.checks(path, directory, gpu=True, execute=True, devices="0")
    result = json.loads((directory / "report.json").read_text())
    assert result["status"] == "failed"
    assert result["commands"][0]["exit_code"] == 0
    assert result["native_gpu_verified"] is False


def test_digest_probe_reference_contains_real_samples_but_never_performance_claim(tmp_path):
    result = S.digest_probe(tmp_path / "probe", device="cpu", repeats=5, sizes=(1, 64, 129))
    assert result["gpu_observed"] is False
    assert result["performance_qualified"] is False
    assert all(len(row["samples"]) == 5 for row in result["observations"])
    assert all("cpu_ms" in pair and "device_ms" in pair
               for row in result["observations"] for pair in row["samples"])


def test_ab_comparison_needs_two_distinct_reports(frozen, tmp_path):
    plan, _, _ = frozen
    report = tmp_path / "same.json"
    report.write_text("{}")
    with pytest.raises(ValueError, match="distinct"):
        S.strict_ab(plan, report, report, tmp_path / "comparison", vary="source")


def test_ab_needs_independent_protocols_before_any_verifier_runs(frozen, tmp_path):
    plan, _, _ = frozen
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    before.write_text("{}"); after.write_text("{}")
    with pytest.raises(ValueError, match="independently frozen"):
        S.strict_ab(plan, before, after, tmp_path / "comparison", vary="source")

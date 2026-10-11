"""Release gate accounting only; synthetic data never reports this host as CUDA."""
from copy import deepcopy
import hashlib
import pytest
import v4_release as release
import v4_benchmark as bench
from test_v4_benchmark import protocol


def strict_protocol(count):
    p, _ = protocol(count)
    p["evidence_contract"] = {"coordinator_diagnostics_required": True, "runtime_observation_required": True}
    return bench.seal_protocol(p)


def contract():
    four, six = strict_protocol(4), strict_protocol(6)
    slo = dict(cycles=1, duration_s=10, ttft_p95_s=3, token_gap_p95_s=.1, max_idle_gap_s=10)
    return release.freeze_contract(four, six, six, slo)


def test_cpu_gate_never_invokes_gpu_tests_or_passes(monkeypatch, tmp_path):
    monkeypatch.setattr(release, "cuda_probe", lambda: {"available": False})
    monkeypatch.setattr(release.B, "source_identity", lambda: {"sha256": "a" * 64, "files": []})
    monkeypatch.setattr(release.subprocess, "run", lambda *a, **k: pytest.fail("executed hardware suite on CPU"))
    result = release.run_gpu_gate(tmp_path / "cuda.xml")
    assert result["status"] == "unverified" and result["hardware_attested"] is False
    assert not (tmp_path / "cuda.xml").exists()


@pytest.mark.parametrize("xml", ["<testsuite/>", "<testsuite><testcase name='other'/></testsuite>",
    "<testsuite><testcase name='gpu'><skipped/></testcase></testsuite>"])
def test_absent_or_skipped_cuda_cases_cannot_qualify(xml):
    result = release.evaluate_gpu_gate({"schema": "flyai-v4-gpu-gate/1", "hardware": {"available": False},
        "source_unchanged": True, "test_sources": {}, "exit_code": 0, "junit": xml,
        "junit_sha256": hashlib.sha256(xml.encode()).hexdigest()})
    assert result["status"] == "unverified"


def test_bad_junit_digest_and_failed_test_fail_closed():
    xml = "<testsuite><testcase name='gpu'><failure/></testcase></testsuite>"
    result = release.evaluate_gpu_gate({"schema": "flyai-v4-gpu-gate/1", "hardware": {"available": False},
        "source_unchanged": True, "test_sources": {}, "exit_code": 1, "junit": xml, "junit_sha256": "0" * 64})
    assert result["status"] == "failed"


def test_contract_keeps_four_six_targets_and_requires_signed_actual_runtime():
    c = contract(); assert c["targets"] == {"4": 40.0, "6": 30.0}
    weak = strict_protocol(4); weak.pop("evidence_contract")
    with pytest.raises(ValueError, match="signed-runtime"):
        release.freeze_contract(bench.seal_protocol(weak), c["six_protocol"], c["soak_protocol"], c["soak_contract"])
    changed = deepcopy(c["six_protocol"]); changed["source"]["files"][0]["sha256"] = "e" * 64
    changed["source"]["sha256"] = bench.digest(changed["source"]["files"])
    for node in changed["hardware"]: node["engine_source_sha256"] = changed["source"]["sha256"]
    with pytest.raises(ValueError, match="differ"):
        release.freeze_contract(c["four_protocol"], bench.seal_protocol(changed), c["soak_protocol"], c["soak_contract"])


def test_release_contract_cannot_relax_speed_or_omit_any_gate():
    c = contract(); altered = deepcopy(c); altered["targets"]["4"] = 1
    result = release.evaluate_release({"schema": "flyai-v4-release/1", "contract_sha256": release.digest(altered)}, altered)
    assert result["status"] == "failed"
    assert release.evaluate_release({"schema": "flyai-v4-release/1", "contract_sha256": release.digest(c)}, c)["status"] == "failed"

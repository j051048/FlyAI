"""The CPU case producer executes original native math and never signs GPU evidence."""
import json

import pytest
import torch
from safetensors.torch import save_file

from test_v4_native_batch import model
from test_v4_conversation_cache import stage
from test_v4_incremental_evidence import report_fixture
from phase0.v4_incremental_case_probe import compare_cases, main


def frozen():
    return {"token_ids": torch.arange(1, 17, dtype=torch.long).view(1, -1),
            "other_token_ids": torch.arange(21, 37, dtype=torch.long).view(1, -1)}


def test_actual_cpu_case_comparisons_include_reset_and_report_real_mismatches(model):
    st = stage(model)
    values = compare_cases(st, frozen(), parent_count=8, host_bytes=32 << 20)
    assert {row["case"] for row in values} == {"append", "compression_boundary", "rollback", "cancel", "reset", "tenant_isolation"}
    reset = next(row for row in values if row["case"] == "reset")
    assert reset["passed"] and reset["candidate"] == reset["reference"]
    for row in values:
        assert row["passed"] is (row["covered"] and row["candidate"] == row["reference"])
        assert row["compression_ratios"] == [0, 4, 8, 0]
        assert len(row["inputs_sha256"]) == 64
    assert st._pos == 0 and not st._spec and not st._spec_ckpts


def test_fixture_cli_never_emits_signed_native_gpu_case(tmp_path, monkeypatch):
    monkeypatch.setenv("V4_CUDA_GRAPH", "0")
    report = report_fixture()
    spec = {key: report[key] for key in ("binding", "source_sha256", "execution", "max_context", "max_suffix", "parent_alignment")}
    spec["schema"] = "v4-incremental-prefix-campaign/1"
    evidence = report["evidence"]
    evidence["stages"][0]["hi"] = 4
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (tmp_path / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    save_file(frozen(), str(tmp_path / "inputs.safetensors"))
    status = main(["--fixture", "--device", "cpu", "--spec", str(tmp_path / "spec.json"),
        "--evidence", str(tmp_path / "evidence.json"), "--input", str(tmp_path / "inputs.safetensors"),
        "--stage-index", "0", "--lo", "0", "--hi", "4", "--parent-count", "8", "--max-seq", "64",
        "--host-mib", "32", "--device-mib", "32", "--out-dir", str(tmp_path / "output")])
    assert status == 0
    observed = json.loads((tmp_path / "output" / "report.json").read_text(encoding="utf-8"))
    assert observed["gpu_verified"] is False and observed["ready"] is False
    assert observed["native_cases"] == [] and observed["reference"] is True
    assert len(observed["comparisons"]) == 6


def test_native_case_requires_distinct_negative_control_and_bounded_snapshot(model):
    with pytest.raises(ValueError, match="distinct"):
        compare_cases(stage(model), {"token_ids": frozen()["token_ids"], "other_token_ids": frozen()["token_ids"]},
                      parent_count=8, host_bytes=32 << 20)
    with pytest.raises(ValueError, match="quota"):
        compare_cases(stage(model), frozen(), parent_count=8, host_bytes=1)

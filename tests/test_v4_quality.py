"""Offline artifact integrity, actual CPU numerics, and no fixture qualification."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
from engines.deepseek_v4 import v4_quality as Q


def artifact(*, tokens=None, probabilities=None, expected=None):
    identity = {"artifact": {"runtime_abi": Q.FIXTURE_ABI, "weights_sha256": "e" * 64},
        "source_sha256": "a" * 64, "tokenizer_sha256": "b" * 64, "config_sha256": "c" * 64,
        "sampling": {"temperature": 0, "seed": 0, "max_new": 2}, "hardware": {"device": "cpu", "fixture": True}}
    dataset = [{"id": "case-1", "prompt_ids": [1, 2], "expected_tokens": expected or [3, 4]}]
    rows = [{"sample_id": "case-1:0", "case_id": "case-1", "prompt_sha256": Q.digest([1, 2]),
             "tokens": tokens or [3, 4], "token_logprobs": probabilities}]
    return Q.generation_artifact(identity, dataset, rows, origin="explicit-test-fixture")


def test_generation_and_scoring_are_separate_hashed_artifacts(tmp_path):
    header, rows = artifact(probabilities=[-0.25, -1.5])
    path = tmp_path / "generation.jsonl"
    Q.write_jsonl(path, header, rows)
    assert Q.read_jsonl(path) == (header, rows)
    score = Q.score_exact(header, rows)
    assert score["exact_token_match_rate"] == 1 and score["production_qualified"] is False
    assert score["pass_at_1"] is None and score["generation_sha256"] == header["sha256"]
    changed = copy.deepcopy(rows); changed[0]["tokens"][0] = 99
    with pytest.raises(ValueError, match="hash"):
        Q.validate_artifact(header, changed)


@pytest.mark.parametrize("mutation", [
    lambda h, r: h["identity"]["artifact"].update(runtime_abi="deepseek-v4.1-native/1"),
    lambda h, r: r[0].update(prompt_sha256="f" * 64),
    lambda h, r: r[0].update(token_logprobs=[float("nan"), -1]),
    lambda h, r: r[0].update(token_logprobs=[-1]),
    lambda h, r: h.update(production_qualified=True),
])
def test_rehashing_cannot_bypass_schema_or_qualification(mutation):
    header, rows = artifact(); mutation(header, rows)
    try:
        for row in rows:
            row["sha256"] = Q.digest({k: v for k, v in row.items() if k != "sha256"})
        header["records_sha256"] = Q.digest(rows)
        header["identity_sha256"] = Q.digest(header["identity"])
        header["sha256"] = Q.digest({k: v for k, v in header.items() if k != "sha256"})
        Q.validate_artifact(header, rows)
    except ValueError:
        pass
    else:
        pytest.fail("semantically invalid rehashed quality artifact passed")


def test_generation_comparison_pins_frozen_input_and_retains_recipe_identity():
    before = artifact(probabilities=[-1, -2])
    same = artifact(probabilities=[-1, -2])
    assert Q.compare_generations(before, same)["exact_match"]
    other = artifact(tokens=[3, 5], probabilities=[-1, -3])
    assert not Q.compare_generations(before, other)["exact_match"]
    different_input = artifact(expected=[4, 5])
    with pytest.raises(ValueError, match="dataset"):
        Q.compare_generations(before, different_input)


def test_jsonl_size_and_duplicate_fields_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b'{"schema":"a","schema":"b"}\n{}\n')
    with pytest.raises(ValueError, match="duplicate"):
        Q.read_jsonl(path)
    monkeypatch.setattr(Q, "MAX_LINE_BYTES", 16)
    path.write_bytes(b"x" * 17 + b"\n")
    with pytest.raises(ValueError, match="budget"):
        Q.read_jsonl(path)


def test_existing_mock_benchmark_exports_actual_outputs_but_never_qualifies():
    from test_v4_benchmark import protocol, FakeAdapter, Clock
    from phase0.v4_benchmark import run_suite
    p, keys = protocol(max_new=3); clock = Clock()
    report = run_suite(FakeAdapter(p, keys, clock), p, fresh_ring=True, clock=clock)
    header, rows = Q.from_benchmark(report)
    assert len(rows) == len(report["samples"])
    assert rows[0]["tokens"] == report["samples"][0]["tokens"]
    assert rows[0]["token_logprobs"] is None, "benchmark did not record probabilities"
    assert header["production_qualified"] is False and header["pass_at_1"] is None
    assert rows[0]["metadata"]["original_backend"] == "mock"
    with pytest.raises(ValueError, match="no expected"):
        Q.score_exact(header, rows)


def test_cli_score_reads_saved_generation_without_model_or_network(tmp_path):
    path, out = tmp_path / "generation.jsonl", tmp_path / "score.json"
    Q.write_jsonl(path, *artifact())
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "phase0" / "v4_quality.py"),
        "score", str(path), "--out", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["exact_token_match_rate"] == 1
    assert json.loads(result.stdout)["pass_at_1"] is None


def test_actual_cpu_teacher_forced_probabilities_and_legacy_trace_comparison():
    torch = pytest.importorskip("torch")
    import v4_numeric_diagnostics as D
    import v4_ref_cpu as R
    from test_v4_stage import stage_from_oracle
    old = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
            dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=32)
        oracle = R.build_oracle(args, 7)
        stage = stage_from_oracle(oracle, args, 0, 4)
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        trace = D.trace_stage(stage, ids, prompt_tokens=3)
        logits = [row for row in trace["records"] if "logits" in row]
        probabilities = [row for row in logits if "teacher_token" in row]
        assert [row["teacher_token"] for row in probabilities] == [4, 5, 6]
        with torch.no_grad():
            hidden = stage.forward(stage.embed(ids[:, :3]), ids[:, :3], 0)
            value = stage.logits_all(hidden, full_logits=False).reshape(-1, args.vocab_size)[-1].float()
            expected = float(torch.log_softmax(value, dim=-1)[4])
        stage.reset()
        assert math.isclose(probabilities[0]["next_token_logprob"], expected, abs_tol=1e-6)
        legacy = copy.deepcopy(trace)
        for row in legacy["records"]:
            row.pop("teacher_token", None); row.pop("next_token_logprob", None)
        legacy["sha256"] = D.digest({k: v for k, v in legacy.items() if k != "sha256"})
        result = Q.compare_teacher_forced(legacy, trace)
        assert result["exact_match"] and result["teacher_forced"]["compared_token_logprobs"] == 0
        assert result["production_qualified"] is False
    finally:
        torch.set_num_threads(old)

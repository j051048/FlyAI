"""CPU contract tests: no model downloads, GPU inventory or CUDA execution."""
import copy
import json
from pathlib import Path
import tarfile

import pytest

from tools import v4_5090_acceptance as A


def config():
    return json.loads((A.ROOT / "examples/v4_5090_acceptance.json").read_text(encoding="utf-8"))


def test_cpu_plan_freezes_exact_matrix_and_closed_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(A.R, "gpu_inventory", lambda: pytest.fail("planning must not inspect CUDA"))
    value = A.plan(config(), tmp_path / "plan.json")
    assert A.require_plan(value)["ring"]["environment"]["V4_CUDA_GRAPH"] == "whole"
    assert value["matrix"] == {"short": {"prompt_tokens": 512, "max_new": 512},
                                "long": {"prompt_tokens": 4096, "max_new": 64}}
    assert value["modes"] == ["greedy", "spec", "dspark", "pipelined"]
    assert value["numeric_input_tokens"] == 4104
    assert not value["github_push_allowed"]
    assert not value["hardware_verified"]


@pytest.mark.parametrize("change", [
    lambda c: c.update(gpu_gate_index=0),
    lambda c: c.update(numeric_gpu_index=True),
    lambda c: c["ring"]["environment"].update(V4_PREFETCH="1"),
    lambda c: c["ring"]["environment"].update(V4_EXECUTION_PROFILE="resident-island"),
    lambda c: c["ring"]["environment"].update(V4_PREFILL_QUERY_CHUNK="0"),
    lambda c: c["soak"].update(duration_s=59),
    lambda c: c["soak"].update(duration_s=float("nan")),
    lambda c: c["ring"]["stages"][2].update(gpu_index=1),
    lambda c: c["ring"]["stages"][-1].update(lo=41),
])
def test_unsafe_or_relaxed_campaign_is_rejected(change):
    cfg = config()
    change(cfg)
    with pytest.raises((ValueError, TypeError)):
        A.validate_config(cfg)


def test_rehashing_plan_cannot_relax_required_matrix(tmp_path):
    p = A.plan(config(), tmp_path / "p.json")
    p["matrix"]["short"]["max_new"] = 16
    p["sha256"] = A.digest({k: v for k, v in p.items() if k != "sha256"})
    with pytest.raises(ValueError, match="contract"):
        A.require_plan(p)


def test_run_without_execute_is_readonly(tmp_path, monkeypatch):
    A.plan(config(), tmp_path / "p.json")
    monkeypatch.setattr(A.R, "prepare", lambda *a, **k: pytest.fail("no execute"))
    assert A.run(tmp_path / "p.json", tmp_path / "run", [], execute=False)["status"] == "planned_only"
    assert not (tmp_path / "run").exists()


def test_wrong_allocation_refused_before_any_hardware_or_directory(tmp_path, monkeypatch):
    A.plan(config(), tmp_path / "p.json")
    monkeypatch.setattr(A.R, "gpu_inventory", lambda: pytest.fail("unconfirmed hardware"))
    with pytest.raises(ValueError, match="exclusively allocated"):
        A.run(tmp_path / "p.json", tmp_path / "run", [0, 1], execute=True)
    assert not (tmp_path / "run").exists()


def test_missing_real_evidence_cannot_pass(tmp_path):
    A.plan(config(), tmp_path / "campaign-plan.json")
    result = A.verify(tmp_path)
    assert result["status"] == "failed"
    assert result["github_push_allowed"] is False


def test_archive_preserves_logs_inputs_but_excludes_private_signers(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "ring/private").mkdir(parents=True)
    (run_dir / "ring/private/key.json").write_text("DO-NOT-EXPORT")
    (run_dir / "logs").mkdir()
    (run_dir / "logs/run.log").write_text("actual raw log")
    (run_dir / "numeric").mkdir()
    (run_dir / "numeric/short.input.safetensors").write_bytes(b"frozen-case-input")
    (run_dir / "summary.json").write_text('{"status":"failed"}')
    metadata = A.archive_evidence(run_dir, tmp_path / "evidence.tgz")
    with tarfile.open(metadata["archive"]) as tar:
        names = tar.getnames()
        assert "evidence/logs/run.log" in names
        assert "evidence/numeric/short.input.safetensors" in names
        assert not any("private" in name for name in names)
        manifest = json.load(tar.extractfile("evidence/evidence-manifest.json"))
    assert len(manifest["files"]) == metadata["files"]
    assert metadata["sha256"] == A.file_hash(metadata["archive"])
    assert A.verify_evidence_files(run_dir, manifest)["files_verified"] == len(manifest["files"])
    (run_dir / "logs/run.log").write_text("tampered raw log")
    with pytest.raises(ValueError, match="hash/size"):
        A.verify_evidence_files(run_dir, manifest)


def test_verifier_preserves_linux_execution_paths_on_windows(tmp_path):
    p = A.plan(config(), tmp_path / "p.json")
    p["config"]["ring"]["model_dir"] = "/data/models/native-on-linux"
    p["sha256"] = A.digest({k: v for k, v in p.items() if k != "sha256"})
    assert A.require_plan(p)["ring"]["model_dir"] == "/data/models/native-on-linux"


@pytest.mark.parametrize("name", ["../outside", "private/key.json", "processes/state.db", "/absolute", "C:/bad", "logs\\bad"])
def test_evidence_manifest_never_reads_unsafe_paths(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe"):
        A.verify_evidence_files(tmp_path, {"schema": "flyai-v4-5090-evidence-manifest/1", "files": [
            {"name": name, "size": 1, "sha256": "0" * 64}]})


def test_whole_flag_without_actual_signed_layer_captures_is_rejected():
    p = {"stages": [{"node_id": "one"}]}
    report = {"samples": [{"receipts": [{"layer_start": 0, "layer_end": 1,
        "runtime_observation": {"node_id": "one", "graph_mode": "whole",
            "optimizations": {"execution_profile": {"name": "resident-whole", "graph_mode": "whole"}}}}]}]}
    assert A.profile_errors(report, p)


def test_plan_source_change_refused_even_with_valid_plan_hash(tmp_path, monkeypatch):
    p = A.plan(config(), tmp_path / "p.json")
    original = copy.deepcopy(p["source"])
    original["sha256"] = "0" * 64
    monkeypatch.setattr(A.R, "_source_identity", lambda: original)
    with pytest.raises(ValueError, match="source"):
        A.require_plan(p)


def test_coordinator_environment_does_not_leak_stage_device_into_protocol():
    state = {"config": {"environment": {"V4_CUDA_GRAPH": "whole"}, "model_dir": "/data/native"},
             "key_paths": ["stage-private.key", "controller-private.key"]}
    env = A.coordinator_environment(state)
    assert "V4_DEV" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["V4_CUDA_GRAPH"] == "whole"


def test_matrix_prepare_freezes_route_and_uses_distinct_service_command_name(tmp_path):
    class Commands:
        def __init__(self): self.calls = []
        def command(self, *args, **kwargs): self.calls.append((args, kwargs))
    runner = Commands()
    cfg = A.validate_config(config())
    ring_dir = tmp_path / "ring"
    A._bench_prepare(runner, cfg, ring_dir, tmp_path / "matrix/short/greedy", "short", "greedy")
    A._bench_prepare(runner, cfg, ring_dir, tmp_path / "service", "short", "greedy")
    assert runner.calls[0][0][0] != runner.calls[1][0][0]
    argv = runner.calls[0][0][1]
    assert argv[argv.index("--network") + 1] == tmp_path / "network.json"
    assert argv[argv.index("--prompt-tokens") + 1] == 512
    assert argv[argv.index("--max-new") + 1] == 512
    assert runner.calls[0][0][3] == {"CUDA_VISIBLE_DEVICES": ""}

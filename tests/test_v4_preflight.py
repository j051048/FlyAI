"""Read-only entry point and actual owned CPU logs; no remote/GPU/model inference."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch
from safetensors.torch import save_file

from engines.deepseek_v4.v4_preflight import process_observation
from shard.managed_launch import ManagedLauncher
from shard import weight_artifacts as W
from shard.offers import ModelCohort

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("structured", [False, True])
def test_real_owned_cpu_log_listening_format_is_a_milestone_not_ready(tmp_path, structured):
    manager = ManagedLauncher(tmp_path)
    line = 'SHARD_STAGE_EVENT ' + json.dumps({"event": "listening", "pid": os.getpid()})
    code = ("import os,json,time; "
        "print('SHARD_STAGE_EVENT '+json.dumps({'event':'listening','pid':os.getpid()}),flush=True); time.sleep(30)")
    if not structured: code = "import time; print('[s2] listening 127.0.0.1:29610',flush=True); time.sleep(30)"
    manager.start("cpu-stage", [sys.executable, "-c", code])
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            result = process_observation("cpu-stage", state_dir=tmp_path)
            if result["listening"]: break
            time.sleep(.01)
        assert result["running"] and result["listening"]
        assert "ready" not in result
        if structured: assert result["events"][0]["pid"] > 0  # Windows venv launcher may own a Python child.
    finally:
        manager.stop("cpu-stage")


def test_large_log_is_bounded_and_first_traceback_is_retained(tmp_path):
    manager = ManagedLauncher(tmp_path)
    log = manager.directory / "sample.log"
    log.write_bytes(b"Traceback (most recent call last):\nModuleNotFoundError: fixture\n" + b"x" * (3 << 20) + b"\nConnectionError: downstream closed\n")
    result = process_observation("sample", state_dir=tmp_path)
    assert not result["running"] and not result["listening"]
    assert "ModuleNotFoundError" in result["first_traceback"]
    assert len(result["log_tail"]) <= 8192 and "downstream closed" in result["log_tail"]


def package(tmp_path):
    directory = tmp_path / "metadata"; directory.mkdir()
    config = json.loads((ROOT / "vendor/deepseek_v4_ref/inference/config.json").read_text())
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "tokenizer.json").write_text('{"fixture":true}')
    names = {f"layers.{i}.fixture": torch.ones(1) for i in range(43)}
    names.update({name: torch.ones(1) for name in ("embed.weight", "head.weight", "norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale")})
    save_file(names, str(directory / "model0-mp1.safetensors"))
    catalog, pack = W.catalogue_directory(directory, "deepseek-ai/DeepSeek-V4-Flash")
    W.write_metadata(directory, catalog, pack)
    cohort = ModelCohort(catalog["model_id"], catalog["manifest_sha256"], catalog["checkpoint_id"], catalog["config_sha256"],
        "fp4-fp8", "deepseek-v4-native/1", "shard-pipeline-session/1", "greedy-native-fp4-fp8/1", 43)
    from shard.pipeline_plan import build_plan
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from shard.manifest import pub_b64
    key = pub_b64(Ed25519PrivateKey.generate())
    plan = build_plan(config, ring_id="cpu-entry", cohort_id=cohort.cohort_id, endpoints=["127.0.0.1:29610"],
                      model_cohort=cohort.to_dict())
    plan["stages"][0]["signer_pubkey"] = key
    plan["coordinator"]["signer_pubkey"] = key
    return directory, plan


@pytest.mark.parametrize("entry", ["script", "module"])
def test_actual_preflight_entry_outside_repo_cwd_has_no_relative_import_failure(tmp_path, entry):
    directory, plan = package(tmp_path)
    request = {"plan": plan, "index": 0, "node": {"model": str(directory), "node_key": str(tmp_path / "missing-key"),
        "listen_port": 29610, "dspark": False}, "environment": {"V4_KERNELS": "cpu", "V4_HADAMARD": "torch"},
        "verify_files": True, "probe_gpu": False, "check_tokenizer": False}
    command = [sys.executable, str(ROOT / "engines/deepseek_v4/v4_preflight.py"), "--request-stdin"] if entry == "script" else [
        sys.executable, "-m", "engines.deepseek_v4.v4_preflight", "--request-stdin"]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(command, input=json.dumps(request), cwd=tmp_path, env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 1  # Missing identity is refused, not fabricated.
    report = json.loads(result.stdout)
    assert report["checks"]["artifacts"]["payload_integrity_verified"]
    assert not report["runtime_ready"] and any(row["check"] == "signer" for row in report["errors"])
    assert "relative import" not in result.stderr and "Stage.load strict tensor coverage" in report["remaining_checks"]


def test_full_checkout_hadamard_import_after_environment_outside_workspace(tmp_path):
    code = ("from engines.deepseek_v4.v4_preflight import apply_environment; "
        "apply_environment({'V4_KERNELS':'cpu','V4_HADAMARD':'torch'}); "
        "import torch,v4_runtime_init; v4_runtime_init.ensure_hadamard(); "
        "r=v4_runtime_init.probe_hadamard('cpu',torch.bfloat16,(1,1,128)); "
        "assert r['passed']; print(v4_runtime_init.hadamard_identity()['backend'])")
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)}, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "torch"

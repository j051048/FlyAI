"""Owned dual-process probe: actual tiny native MTP execution, no GPU claim."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file, load_file

from v4_draft_chain_probe import _validate, _environment, _child_args
import v4_ref_cpu as R

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "phase0" / "v4_draft_chain_probe.py"


def inputs():
    args = R.cpu_args(max_batch_size=1, max_seq_len=64)
    hidden = len(args.dspark_target_layer_ids)*args.dim
    generator = torch.Generator().manual_seed(29)
    return args, {"prefill_ids": torch.tensor([7], dtype=torch.long),
        "prompt_taps": torch.randn(1, 13, hidden, generator=generator).to(torch.bfloat16),
        "committed_ids": torch.randint(0, args.vocab_size, (1, 6), generator=generator),
        "main_taps": torch.randn(1, 6, hidden, generator=generator).to(torch.bfloat16),
        "run_lengths": torch.tensor([1, 3, 2], dtype=torch.long)}


@pytest.mark.parametrize("bad", ["missing", "dtype", "nonfinite", "runs", "range", "horizon", "quota"])
def test_probe_rejects_invalid_frozen_inputs_before_model_construction(bad):
    args, data = inputs()
    horizon, budget = 64, 64 << 20
    if bad == "missing": data.pop("main_taps")
    elif bad == "dtype": data["prompt_taps"] = data["prompt_taps"].float()
    elif bad == "nonfinite": data["main_taps"][0, 0, 0] = float("nan")
    elif bad == "runs": data["run_lengths"][0] = 0
    elif bad == "range": data["committed_ids"][0, 0] = args.vocab_size
    elif bad == "horizon": horizon = 18
    else: budget = 1
    with pytest.raises(ValueError):
        _validate(data, args, torch.bfloat16, max_seq=horizon, host_bytes=budget)


def test_owned_recipe_discards_ambient_service_flags_and_keeps_host_execution_environment(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv("V4_DRAFT_ADAPTER_DIR", "ambient-adapter")
    monkeypatch.setenv("V4_NATIVE_LANES", "4")
    monkeypatch.setenv("PROBE_HOST_MARKER", "preserved")
    env = _environment(SimpleNamespace(fixture=True, device="cpu", graph_mib=16), "chain")
    assert env["PROBE_HOST_MARKER"] == "preserved"
    assert "V4_NATIVE_LANES" not in env and "V4_DRAFT_ADAPTER_DIR" not in env
    assert env["V4_DSPARK_CHAIN_REFERENCE"] == env["V4_DSPARK_CHAIN_GRAPH"] == "1"
    assert env["V4_DSPARK_MOE"] == "0"


def test_child_paths_resolve_in_the_callers_directory_before_repo_cwd_switch(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(fixture=False, checkpoint="native weights", input="frozen inputs.safetensors",
        out_dir="evidence", lo=40, hi=43, device="cuda:0", host_mib=8192, device_mib=30000,
        graph_mib=1024, max_seq=8192, seed=7, execute_gpu=True)
    command = _child_args(args, "chain")
    assert command[command.index("--checkpoint")+1] == str(tmp_path/"native weights")
    assert command[command.index("--input")+1] == str(tmp_path/"frozen inputs.safetensors")
    assert command[command.index("--out-dir")+1] == str(tmp_path/"evidence")
    assert command[-1] == "--execute-gpu"


def test_actual_fixture_cli_uses_two_sequential_independent_native_processes(tmp_path):
    _, data = inputs()
    frozen = tmp_path / "frozen.safetensors"
    save_file(data, str(frozen))
    out = tmp_path / "evidence"
    env = dict(os.environ, V4_NATIVE_LANES="4", V4_DRAFT_ADAPTER_DIR="must-not-load",
               PYTHONUTF8="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, str(SCRIPT), "--fixture", "--input", str(frozen),
        "--out-dir", str(out), "--lo", "5", "--hi", "8", "--device", "cpu",
        "--host-mib", "512", "--device-mib", "64", "--graph-mib", "16", "--max-seq", "64"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + "\n" + "\n".join(
        p.read_text(encoding="utf-8") for p in out.glob("*.log"))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    scalar = json.loads((out / "scalar.json").read_text(encoding="utf-8"))
    chain = json.loads((out / "chain.json").read_text(encoding="utf-8"))
    assert report["byte_parity"] and not report["gpu_qualified"] and not report["release_qualified"]
    assert report["tensors_compared"] == 2 * 3 * (3 + 2)
    assert chain["chain"]["reference_eager"] == 6
    assert chain["chain"]["captures"] == chain["chain"]["replays"] == 0
    assert scalar["chain"] == {"enabled": False}
    assert scalar["sources"] == chain["sources"] == report["sources"]
    assert scalar["model"] == chain["model"]
    original, replayed = load_file(str(out / "scalar.safetensors")), load_file(str(out / "chain.safetensors"))
    assert set(original) == set(replayed)
    assert all(torch.equal(original[name].view(torch.uint8), replayed[name].view(torch.uint8)) for name in original)


def test_native_probe_requires_explicit_owned_execution_before_checkpoint_access(tmp_path):
    _, data = inputs()
    frozen = tmp_path / "frozen.safetensors"
    save_file(data, str(frozen))
    result = subprocess.run([sys.executable, str(SCRIPT), "--checkpoint", "missing-checkpoint",
        "--input", str(frozen), "--out-dir", str(tmp_path / "must-not-create"),
        "--lo", "40", "--hi", "43", "--device", "cpu", "--host-mib", "512",
        "--device-mib", "64", "--graph-mib", "16"], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
    assert result.returncode == 2
    assert "--execute-gpu" in result.stdout
    assert not (tmp_path / "must-not-create").exists()

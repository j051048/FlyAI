"""Local preparation and actual isolated CPU traces; never start a GPU/ring."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import pytest

from tools import v4_acceptance_numeric as N

ROOT = Path(__file__).resolve().parents[1]


def frozen(path, ids, hidden_shape=None, hidden_dtype="BF16"):
    header = {"token_ids": {"dtype": "I64", "shape": [1, len(ids)], "data_offsets": [0, len(ids) * 8]}}
    payload = struct.pack("<" + "q" * len(ids), *ids)
    if hidden_shape:
        amount = __import__("math").prod(hidden_shape) * N.DTYPE_BYTES[hidden_dtype]
        header["hidden"] = {"dtype": hidden_dtype, "shape": hidden_shape, "data_offsets": [len(payload), len(payload) + amount]}
        payload += bytes(amount)
    raw = N.canonical(header)
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return path


def run_cli(*args, timeout=90, environment=None):
    env = {**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env.update(environment or {})
    return subprocess.run([sys.executable, str(ROOT / "tools/v4_acceptance_numeric.py"), *map(str, args)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)


def fixture_plan(tmp_path, n=7, hi=1, max_tokens=64, extra=()):
    inp = frozen(tmp_path / "frozen.safetensors", list(range(n)))
    directory = tmp_path / "numeric"
    args = N.parser().parse_args(["prepare", "--fixture", "--input", str(inp), "--hi", str(hi),
                                 "--device", "cpu", "--max-tokens", str(max_tokens), "--out-dir", str(directory), *extra])
    return N.prepare(args), directory


def metadata_checkpoint(tmp_path):
    """Tiny metadata fixture with production-like geometry, never model evidence."""
    from shard import weight_artifacts as A
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    config = json.loads((ROOT / "vendor/deepseek_v4_ref/inference/config.json").read_text())
    (directory / "config.json").write_text(json.dumps(config))
    header, payload = {}, bytearray()
    for name in ["embed.weight", *[f"layers.{i}.dummy.weight" for i in range(config["n_layers"])],
                 "norm.weight", "head.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"]:
        header[name] = {"dtype": "BF16", "shape": [1], "data_offsets": [len(payload), len(payload) + 2]}
        payload += bytes(2)
    raw = N.canonical(header)
    raw += b" " * (-len(raw) % 8)
    (directory / "model0-mp1.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    catalog, pack = A.catalogue_directory(directory, "metadata-test-only")
    A.write_metadata(directory, catalog, pack)
    return directory, config


def test_prepare_is_stdlib_only_no_gpu_or_tokenizer_import(tmp_path):
    ids = frozen(tmp_path / "input.safetensors", [1, 2, 3, 4, 5, 6, 7])
    # The -S interpreter cannot import site packages (including torch/tokenizers).
    child = subprocess.run([sys.executable, "-S", str(ROOT / "tools/v4_acceptance_numeric.py"),
        "prepare", "--fixture", "--input", str(ids), "--hi", "1", "--device", "cpu", "--max-tokens", "64",
        "--out-dir", str(tmp_path / "std-only")], cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["status"] == "prepared_unverified"
    assert not list(tmp_path.rglob("*comparison.json"))


def test_prepare_native_head_span_and_actual_bucket_are_derived(tmp_path):
    checkpoint, cfg = metadata_checkpoint(tmp_path)
    ids = frozen(tmp_path / "long.safetensors", [11] * 4104)
    args = N.parser().parse_args(["prepare", "--checkpoint", str(checkpoint), "--input", str(ids),
                                 "--device", "cuda:3", "--out-dir", str(tmp_path / "production")])
    plan = N.prepare(args)
    assert [plan["lo"], plan["hi"]] == [0, 4]
    assert plan["roles"] == {"head": True, "tail": False, "dspark": False}
    assert plan["artifact_preview"]["assigned"]["lo"] == 0
    assert plan["artifact_preview"]["assigned"]["hi"] == 4
    assert not plan["artifact_preview"]["payload_integrity_verified"]
    assert all(row["prepared"] and not row["verified"] for row in plan["coverage"])
    topk = next(row for row in plan["coverage"] if row["kind"] == "indexer_topk_floor")
    bucket = next(row for row in plan["coverage"] if row["kind"] == "indexer_bucket_transition")
    assert topk["end_tokens"] == cfg["index_topk"] * 4 == 2048
    assert topk["selection_starts_end_tokens"] == 2052
    assert bucket["end_tokens"] == 4100
    assert (bucket["bucket_before"], bucket["bucket_after"]) == (1024, 2048)
    case = next(row for row in plan["cases"] if "indexer_bucket_transition" in row["covers"])
    assert case["prompt_tokens"] == 4092 and case["tokens"] == 4104
    assert N.expected_graph_variants(plan, case, 2) == 4
    assert N.expected_graph_variants(plan, case, 3) == 2
    assert plan["runtime_model_args"]["max_batch_size"] == 1
    assert plan["runtime_model_args"]["max_seq_len"] == 8192
    assert not plan["hardware_verified"] and not plan["speed_pass"]
    assert N.checked_plan(tmp_path / "production/numeric-plan.json")["sha256"] == plan["sha256"]


def test_head_zero_single_layer_does_not_claim_indexer_coverage(tmp_path):
    plan, directory = fixture_plan(tmp_path)
    assert [plan["lo"], plan["hi"]] == [0, 1]
    assert not any(row["prepared"] for row in plan["coverage"])
    result = run_cli("run", "--plan", directory / "numeric-plan.json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "prepared_unverified"
    assert not (directory / "numeric-result.json").exists()


def test_nonhead_requires_original_hidden_and_valid_shape(tmp_path):
    cfg = {**N.model_defaults(), **N.literal_assignment(N.ENGINE / "v4_ref_cpu.py", "_CPU_ARGS")}
    ids = frozen(tmp_path / "ids.safetensors", [1, 2, 3, 4])
    with pytest.raises(ValueError, match="original frozen hidden"):
        N.input_descriptor(ids, cfg, 2, 64)
    wrong = frozen(tmp_path / "wrong.safetensors", [1, 2, 3, 4], [1, 4, 4, 8])
    with pytest.raises(ValueError, match="model BF16"):
        N.input_descriptor(wrong, cfg, 2, 64)
    valid = frozen(tmp_path / "geometry-only.safetensors", [1, 2, 3, 4], [1, 4, cfg["hc_mult"], cfg["dim"]])
    row = N.input_descriptor(valid, cfg, 2, 64)
    assert row["tensors"]["hidden"]["shape"] == [1, 4, 4, 256]


def test_input_prefix_preserves_frozen_tensor_bytes(tmp_path):
    cfg = {**N.model_defaults(), **N.literal_assignment(N.ENGINE / "v4_ref_cpu.py", "_CPU_ARGS")}
    original = frozen(tmp_path / "original.safetensors", list(range(10)), [1, 10, 4, 256])
    description = N.input_descriptor(original, cfg, 2, 64)
    prefix = tmp_path / "prefix.safetensors"
    N.prefix_input(description, 6, prefix)
    row = N.input_descriptor(prefix, cfg, 2, 64)
    assert row["tokens"] == 6 and row["tensors"]["hidden"]["shape"] == [1, 6, 4, 256]
    assert row["tensors"]["token_ids"]["sha256"] == __import__("hashlib").sha256(struct.pack("<6q", *range(6))).hexdigest()


def test_modified_plan_and_input_refused_before_execution(tmp_path):
    plan, directory = fixture_plan(tmp_path)
    path = directory / "numeric-plan.json"
    changed = json.loads(path.read_text())
    changed["threads"] = 2
    N.write_json(path, changed)
    with pytest.raises(ValueError, match="modified numeric plan"):
        N.checked_plan(path)
    N.write_json(path, plan)
    input_path = Path(plan["cases"][0]["input"]["path"])
    raw = bytearray(input_path.read_bytes())
    raw[-8:] = struct.pack("<q", 127)
    input_path.write_bytes(raw)
    with pytest.raises(ValueError, match="changed since preparation"):
        N.checked_plan(path)


@pytest.mark.parametrize("override", [{"V4_REF_SLIM": "1"}, {"V4_FAST_VERIFY": "1"},
                                     {"V4_CUDA_GRAPH": "island"}, {"V4_DTYPE": "float32"}, {"V4_UNKNOWN": "0"}])
def test_approximations_or_unknown_environment_cannot_prepare(tmp_path, override):
    env_file = tmp_path / "override.json"
    env_file.write_text(json.dumps(override))
    with pytest.raises(ValueError):
        fixture_plan(tmp_path, extra=("--env-file", str(env_file)))


def test_strict_token_record_and_trace_byte_budgets(tmp_path):
    with pytest.raises(ValueError, match="byte budget"):
        fixture_plan(tmp_path, extra=("--trace-max-bytes", "16"))
    source = frozen(tmp_path / "out-of-vocabulary.safetensors", [1, 2, 700, 4])
    cfg = {**N.model_defaults(), **N.literal_assignment(N.ENGINE / "v4_ref_cpu.py", "_CPU_ARGS")}
    with pytest.raises(ValueError, match="outside checkpoint vocabulary"):
        N.input_descriptor(source, cfg, 0, 64)
    with pytest.raises(ValueError, match="token budget"):
        N.input_descriptor(source, cfg, 0, 3)


def test_freeze_explicit_ids_roundtrip_and_make_input_requires_execute(tmp_path):
    ids = tmp_path / "ids.json"
    ids.write_text("[1,2,3,4,5]")
    out = tmp_path / "ids.safetensors"
    child = run_cli("freeze-ids", "--ids-json", ids, "--out", out)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["tokens"] == 5
    child = run_cli("make-input", "--checkpoint", tmp_path / "nonexistent", "--input", out,
                    "--layer", "2", "--device", "cuda:0", "--out", tmp_path / "hidden.safetensors")
    assert child.returncode == 2 and "requires --execute" in child.stderr


def test_actual_cpu_subprocess_gate_and_independent_portable_verification(tmp_path):
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("safetensors") is None:
        pytest.skip("requires CPU torch and safetensors for actual trace subprocesses")
    plan, directory = fixture_plan(tmp_path)
    child = run_cli("run", "--plan", directory / "numeric-plan.json", "--execute",
                    environment={"V4_REF_SLIM": "1", "V4_FAST_VERIFY": "1", "V4_CUDA_GRAPH": "island"})
    assert child.returncode == 0, child.stderr
    result = json.loads((directory / "numeric-result.json").read_text())
    assert result["status"] == "partial_unverified"
    assert not result["hardware_verified"] and not result["speed_pass"]
    assert result["cases"][0]["exact_match"]
    assert result["cases"][0]["rejection_recovery_checked"]
    assert len(result["cases"][0]["comparisons"]) == 4
    assert len(result["commands"]) == 8
    for reference in result["commands"]:
        command = N.read_json(reference["completion"])
        assert command["returncode"] == 0 and command["cleanup"]["confirmed"]
        assert not command["timeout"] and not command["output_limit"]
        for stream in ("stdout", "stderr"):
            raw_log = Path(command["logs"][stream]["path"])
            assert raw_log.is_file() and raw_log.stat().st_size == command["logs"][stream]["bytes"]
        stdout = Path(command["logs"]["stdout"]["path"]).read_text(encoding="utf-8")
        assert "cpu_reference" in stdout if command["step"].startswith("trace.") else "exact_match" in stdout
    trace_path = Path(result["cases"][0]["traces"]["whole_eager_rejected"])
    trace = json.loads(trace_path.read_text())
    assert trace["evidence"] == "cpu_reference" and not trace["identity"]["model"]["verified_artifacts"]
    assert trace["recipe"]["threads"] == 1 and not trace["recipe"]["ref_slim_active"]
    assert not trace["recipe"]["fast_verify"]
    result["status"], result["hardware_verified"] = "passed_owned_stage_numeric", True
    verified = N.verify_result(result, plan, base_dir=directory)
    assert verified["status"] == "partial_unverified" and not verified["hardware_verified"]
    moved = tmp_path / "extracted"
    shutil.copytree(directory, moved)
    shutil.rmtree(directory)
    portable = N.verify_result(result, plan, base_dir=moved)
    assert portable == verified
    log = next(moved.glob("*.stdout.log"))
    original_log = log.read_bytes()
    log.write_bytes(original_log + b"tampered\n")
    with pytest.raises(ValueError, match="raw numeric command log changed"):
        N.verify_result(result, plan, base_dir=moved)
    log.write_bytes(original_log)
    evidence = moved / trace_path.name
    modified = json.loads(evidence.read_text())
    modified["records"][1]["hidden"]["sha256"] = "0" * 64
    N.write_json(evidence, modified)
    with pytest.raises(ValueError, match="modified bounded numeric trace"):
        N.verify_result(result, plan, base_dir=moved)


def test_boundary_inventory_uses_changed_model_and_runtime_ladder():
    cfg = {"window_size": 32, "compress_ratios": [0, 4, 128], "index_topk": 8}
    boundaries = N.boundary_inventory(cfg, 1, 3, 1024)
    floor = next(b for b in boundaries if b["kind"] == "indexer_topk_floor")
    bucket = next(b for b in boundaries if b["kind"] == "indexer_bucket_transition")
    assert floor["end_tokens"] == 32
    assert bucket["end_tokens"] == 68 and bucket["bucket_before"] == 16 and bucket["bucket_after"] == 64


def test_owned_command_persists_both_streams_and_preserves_outer_ownership(tmp_path):
    pytest.importorskip("psutil")
    command = [sys.executable, "-c", "import os,sys; print(os.environ['SHARD_MANAGED_LAUNCH_TOKEN']); print('raw warning',file=sys.stderr)"]
    completion = N.run_owned_command(command,
        environment={**os.environ, "SHARD_MANAGED_LAUNCH_TOKEN": "test-outer-owned-scope"}, cwd=ROOT,
        stdout_path=tmp_path / "success.stdout.log", stderr_path=tmp_path / "success.stderr.log", timeout_s=10)
    assert N.command_succeeded(completion)
    assert Path(completion["logs"]["stdout"]["path"]).read_text().strip() == "test-outer-owned-scope"
    assert Path(completion["logs"]["stderr"]["path"]).read_text().strip() == "raw warning"
    assert all(log["bytes"] == log["bytes_seen"] for log in completion["logs"].values())


def test_owned_command_caps_noisy_output_without_capture_memory(tmp_path):
    pytest.importorskip("psutil")
    command = [sys.executable, "-c", "import os,sys,time; os.write(sys.stdout.fileno(),b'x'*131072); os.write(sys.stderr.fileno(),b'e'*131072); time.sleep(30)"]
    completion = N.run_owned_command(command, environment=os.environ, cwd=ROOT,
        stdout_path=tmp_path / "noisy.stdout.log", stderr_path=tmp_path / "noisy.stderr.log", timeout_s=10, log_max_bytes=4096)
    assert completion["output_limit"] and not completion["timeout"]
    assert completion["cleanup"]["confirmed"] and not N.command_succeeded(completion)
    assert all(log["bytes"] <= 4096 for log in completion["logs"].values())
    assert completion["logs"]["stdout"]["bytes"] == 4096


def test_owned_timeout_stops_descendants_and_preserves_unrelated_process(tmp_path):
    psutil = pytest.importorskip("psutil")
    base_python = getattr(sys, "_base_executable", sys.executable)
    unrelated = subprocess.Popen([base_python, "-c", "import time; time.sleep(60)"],
                                  **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
    child_code = "import subprocess,sys,time; child=subprocess.Popen([sys.argv[1],'-c','import time; time.sleep(60)']); print(child.pid,flush=True); time.sleep(60)"
    try:
        completion = N.run_owned_command([sys.executable, "-c", child_code, base_python], environment=os.environ, cwd=ROOT,
            stdout_path=tmp_path / "timeout.stdout.log", stderr_path=tmp_path / "timeout.stderr.log", timeout_s=2)
        assert completion["timeout"] and not completion["output_limit"]
        assert completion["cleanup"]["confirmed"] and unrelated.poll() is None
        pid = int(Path(completion["logs"]["stdout"]["path"]).read_text().strip())
        try:
            descendant = psutil.Process(pid)
            assert not descendant.is_running() or descendant.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)

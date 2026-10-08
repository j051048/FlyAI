"""CPU-only tests of the measurement/evidence contract, not GPU performance claims."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import v4_benchmark as bench
from shard.receipt import ReceiptSigner, gen_key, pub_b64
from shard.runtime_metrics import RuntimeMetrics


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def protocol(n=4, *, max_new=8, metrics=False):
    keys = [gen_key() for _ in range(n)]
    ranges = [(0, 12), (12, 24), (24, 36), (36, 43)] if n == 4 else [
        (0, 8), (8, 16), (16, 24), (24, 32), (32, 40), (40, 43)]
    files = [{"name": "config.json", "bytes": 10, "sha256": "a" * 64},
             {"name": "model.safetensors", "bytes": 100, "sha256": "b" * 64},
             {"name": "tokenizer.json", "bytes": 20, "sha256": "c" * 64}]
    source_files = [{"name": "engine.py", "sha256": "d" * 64}]
    env = dict(bench.DEFAULT_ENV, V4_RUNTIME_METRICS="1" if metrics else "0")
    # Deliberately synthetic inventory: this fixture never represents physical GPUs.
    nodes = [{"node_id": f"test-node-{i}", "host_id": f"test-host-{i}",
              "gpu_uuid": f"GPU-test-{i}", "gpu_name": "NVIDIA GeForce RTX 5090",
              "vram_bytes": 32 * 1024**3, "driver": "test", "cuda": "test", "torch": "test", "tilelang": "test",
              "signer_pubkey": pub_b64(key), "process_run_id": f"test-process-{i}",
              "checkpoint_sha256": bench.digest(files), "config_sha256": "a" * 64,
              "engine_source_sha256": bench.digest(source_files), "stage_env_sha256": bench.digest(env),
              "layer_start": lo, "layer_end": hi}
             for i, (key, (lo, hi)) in enumerate(zip(keys, ranges))]
    p = bench.seal_protocol({
        "schema": bench.PROTOCOL_SCHEMA, "model_id": bench.MODEL_ID, "layer_count": 43,
        "checkpoint": {"files": files, "sha256": bench.digest(files), "config_sha256": "a" * 64,
                       "tokenizer_sha256": bench.digest([files[-1]]),
                       "quantization": {"dtype": "fp8", "expert_dtype": "fp4", "scale_dtype": "fp8", "scale_fmt": None}},
        "source": {"git_commit": None, "files": source_files, "sha256": bench.digest(source_files)},
        "env": env, "hardware": nodes,
        "prompts": bench.freeze_prompts(lambda text: list(text.encode()), lambda text: text),
        "run": {"prompt_tokens": 512, "max_new": max_new, "warm_reps": 3, "warmup_reps": 1,
                "mode": "pipelined", "temperature": 0.0, "seed": 0,
                "eos_policy": "fixed_length_ignore_eos",
                "speed_metric": "complete_suite_median_warm_committed_decode_tok_s"},
    })
    return p, keys


class FakeAdapter:
    backend = "mock"
    swarm_id = "test-swarm"
    artifact_verification = {"checkpoint_bytes_verified": True, "config_bytes_verified": True,
                             "tokenizer_bytes_verified": True, "engine_source_verified": True}

    def __init__(self, p, keys, clock, *, rate=50.0, prefill_s=2.0, sweep_s=60.0):
        self.protocol, self.keys, self.clock = p, keys, clock
        self.rate, self.prefill_s, self.sweep_s = rate, prefill_s, sweep_s
        self.last = None

    def generate(self, ids, max_new, *, mode, nonce, job_id, on_token):
        self.last = (nonce, job_id)
        tokens = [(sum(ids) + i) % 257 for i in range(max_new)]
        self.clock.advance(self.prefill_s)
        for token in tokens:
            self.clock.advance(1 / self.rate)
            on_token(token)
        self.clock.advance(0.001)  # coordinator return/drain, before receipt sweep
        return {"ok": True, "tokens": tokens, "prompt_tokens": len(ids),
                "frames": 100 * max_new, "generated": 1000 * max_new,
                "accepted": max_new // 2}

    def sweep(self, nonce):
        assert nonce == self.last[0]
        self.clock.advance(self.sweep_s)
        receipts = []
        for i, (node, key) in enumerate(zip(self.protocol["hardware"], self.keys)):
            signer = ReceiptSigner(key, self.swarm_id, self.last[1], node["layer_start"],
                                   node["layer_end"], nonce=nonce)
            for chunk in range(2):
                signer.observe(f"boundary-{i}-{chunk}".encode(), f"boundary-{i+1}-{chunk}".encode())
            if self.protocol["env"]["V4_RUNTIME_METRICS"] == "1":
                # Synthetic geometry observations only; no real GPU is touched by these fixtures.
                metrics = RuntimeMetrics("cuda:0")
                metrics.resident_routes(12)
                receipts.append(signer.finalize(runtime_metrics=metrics.snapshot()))
            else:
                receipts.append(signer.finalize())
        return receipts, True


def report(n=4, rate=50.0, *, backend="live_ring", prefill_s=2.0, metrics=False):
    p, keys = protocol(n, metrics=metrics)
    clock = Clock()
    adapter = FakeAdapter(p, keys, clock, rate=rate, prefill_s=prefill_s)
    adapter.backend = backend  # only fixtures exercise the passing evaluator branch
    return bench.run_suite(adapter, p, fresh_ring=True, clock=clock)


def test_protocol_freezes_prompts_without_losing_chat_prefix_or_task():
    prefix, suffix = "<BOS><user>\n", "<assistant>"
    prompts = bench.freeze_prompts(lambda text: list(text.encode()),
                                   lambda text: prefix + text + suffix, 512)
    assert [p["workload"] for p in prompts] == list(bench.WORKLOADS)
    assert prompts == bench.freeze_prompts(lambda text: list(text.encode()),
                                           lambda text: prefix + text + suffix, 512)
    for prompt in prompts:
        assert len(prompt["token_ids"]) == 512
        text = bytes(prompt["token_ids"]).decode()
        assert text.startswith(prefix) and text.endswith(bench.PROMPTS[prompt["workload"]] + suffix)
        assert prompt["sha256"] == bench.digest(prompt["token_ids"])


def test_frozen_protocol_rejects_changed_hash_missing_workload_or_tail_split():
    p, _ = protocol()
    assert bench.protocol_errors(p) == []
    changed = copy.deepcopy(p)
    changed["run"]["max_new"] += 1
    assert "protocol hash" in ";".join(bench.protocol_errors(changed))
    changed = copy.deepcopy(p)
    changed["prompts"].pop()
    assert "prompt set" in ";".join(bench.protocol_errors(bench.seal_protocol(changed)))
    changed = copy.deepcopy(p)
    changed["hardware"][-2]["layer_end"] = 41
    changed["hardware"][-1]["layer_start"] = 41
    assert "tail" in ";".join(bench.protocol_errors(bench.seal_protocol(changed)))


def test_measurement_counts_commits_excludes_prefill_for_decode_and_excludes_sweep():
    r = report(prefill_s=90.0)
    s = r["samples"][0]
    assert s["committed_tokens"] == 8
    assert s["coordinator_stats"]["generated"] == 8000
    assert s["measurement"]["elapsed_s"] == pytest.approx(90.161)
    assert s["measurement"]["receipt_sweep_s"] == 60.0
    assert s["measurement"]["decode_committed_tok_s"] == pytest.approx(50.0)
    assert s["measurement"]["end_to_end_committed_tok_s"] < 1
    assert bench.evaluate_report(r)["status"] == "passed"


@pytest.mark.parametrize("n,rate,status,target", [(4, 40.0, "passed", 40), (4, 39.0, "valid_baseline", 40),
                                                (6, 30.0, "passed", 30), (6, 29.0, "valid_baseline", 30)])
def test_exact_hardware_targets(n, rate, status, target):
    evaluation = bench.evaluate_report(report(n, rate))
    assert evaluation["status"] == status, evaluation
    assert evaluation["target_tok_s"] == target
    assert evaluation["complete_suite_median_warm_decode_tok_s"] == pytest.approx(rate)


def test_only_first_job_is_process_cold_and_controls_cannot_prewarm_it():
    r = report()
    samples = r["samples"]
    assert len(samples) == 20
    assert [(s["workload"], s["phase"]) for s in samples if s["phase"] == "cold"] == [("code", "cold")]
    assert [s["phase"] for s in samples[:5]] == ["cold", "warm", "warm", "warm", "greedy_control"]
    assert [s["phase"] for s in samples[5:10]] == ["warmup", "warm", "warm", "warm", "greedy_control"]


@pytest.mark.parametrize("change", ["receipts", "hardware", "parity", "signer", "nonce", "chain", "job", "speed"])
def test_missing_or_forged_evidence_never_passes(change):
    r = report()
    if change == "receipts":
        r["samples"][0]["receipts"] = []
    elif change == "hardware":
        del r["protocol"]["hardware"][0]["gpu_uuid"]
        r["protocol"] = bench.seal_protocol(r["protocol"])
        r["protocol_sha256"] = r["protocol"]["sha256"]
    elif change == "parity":
        s = r["samples"][1]
        s["tokens"][0] += 1
        s["output_sha256"] = bench.digest(s["tokens"])
    elif change == "signer":
        r["samples"][0]["receipts"][0]["pubkey"] = pub_b64(gen_key())
    elif change == "nonce":
        r["samples"][0]["nonce"] = "wrong" * 10
    elif change == "chain":
        r["samples"][0]["receipts"][0]["out_root"] = "0" * 64
    elif change == "job":
        r["samples"][0]["job_id"] = "another-job"
    else:
        r["samples"][0]["measurement"]["decode_committed_tok_s"] = 1000
    assert bench.evaluate_report(r)["status"] in ("failed", "unverified")
    assert bench.evaluate_report(r)["speed_pass"] is False


def test_fully_valid_but_unassigned_signer_fails():
    r = report()
    s = r["samples"][0]
    node = r["protocol"]["hardware"][0]
    signer = ReceiptSigner(gen_key(), s["swarm_id"], s["job_id"],
                           node["layer_start"], node["layer_end"], nonce=s["nonce"])
    signer.observe(b"a", b"b")
    s["receipts"][0] = signer.finalize()
    evaluation = bench.evaluate_report(r)
    assert evaluation["status"] == "failed"
    assert "assignment" in ";".join(evaluation["errors"])


def test_unique_nonce_and_complete_suite_are_required():
    r = report()
    r["samples"][1]["nonce"] = r["samples"][0]["nonce"]
    assert bench.evaluate_report(r)["status"] == "failed"
    r = report()
    r["samples"] = r["samples"][:5]  # cherry-pick the fast code workload
    assert bench.evaluate_report(r)["speed_pass"] is False


def test_mock_or_unknown_artifact_or_no_fresh_ring_cannot_qualify():
    r = report(backend="mock")
    assert bench.evaluate_report(r)["status"] == "unverified"
    for field in ("artifact_verification", "cold_start"):
        r = report()
        r.pop(field)
        assert bench.evaluate_report(r)["status"] == "unverified"


def test_optional_signed_metrics_enabled_without_raw_metrics_is_unverified():
    r = report()
    r["protocol"]["env"]["V4_RUNTIME_METRICS"] = "1"
    for node in r["protocol"]["hardware"]:
        node["stage_env_sha256"] = bench.digest(r["protocol"]["env"])
    r["protocol"] = bench.seal_protocol(r["protocol"])
    r["protocol_sha256"] = r["protocol"]["sha256"]
    result = bench.evaluate_report(r)
    assert result["status"] == "unverified"
    assert "runtime metrics absent" in ";".join(result["missing_evidence"])


def test_signed_runtime_metrics_survive_report_and_tampering_fails():
    r = report(metrics=True)
    original = copy.deepcopy(r["samples"][0]["receipts"])
    assert bench.evaluate_report(r)["status"] == "passed"
    assert r["samples"][0]["receipts"] == original
    r["samples"][0]["receipts"][0]["runtime_metrics"]["totals"]["resident_hits"] += 1
    assert bench.evaluate_report(r)["status"] == "failed"


def test_live_adapter_calls_existing_coordinators_and_separate_sweep(monkeypatch, tmp_path):
    p, _ = protocol()
    calls = []
    sockets = [SimpleNamespace(close=lambda: calls.append("close")) for _ in range(2)]

    def coordinate(pipe, ret, ids, max_new, **kw):
        calls.append((ids, max_new, kw))
        kw["on_token"](7)
        return {"ok": True, "tokens": [7]}

    vp = SimpleNamespace(coordinate=coordinate, coordinate_dspark=coordinate,
                         coordinate_dspark_pipelined=coordinate, SWARM_TOKEN=None,
                         connect_ring=lambda *a, **kw: sockets,
                         _sweep_receipts=lambda pipe, ret, layers, nonce, **kw: ([{"nonce": nonce}], True))
    for key in list(os.environ):
        if key.startswith("V4_"):
            monkeypatch.delenv(key)
    monkeypatch.delitem(sys.modules, "v4_pipe", raising=False)
    monkeypatch.setattr(bench, "verify_checkpoint", lambda *a: dict(FakeAdapter.artifact_verification))
    monkeypatch.setattr(bench, "source_identity", lambda: p["source"])
    monkeypatch.setattr(bench.importlib, "import_module", lambda name: vp)
    # Isolate env updates by registering every env key with monkeypatch before adapter writes them.
    for key in p["env"]:
        monkeypatch.setenv(key, p["env"][key])
    monkeypatch.setenv("V4_DIR", str(tmp_path))
    adapter = bench.LiveRingAdapter(p, tmp_path, "head:29610", "tail:29612")
    tokens = []
    result = adapter.generate([1, 2], 8, mode="pipelined", nonce="nonce", job_id="job", on_token=tokens.append)
    assert result["tokens"] == tokens == [7]
    assert calls[-1][2]["receipts"] is False and calls[-1][2]["eos_ids"] == ()
    assert calls[-1][2]["strict_job_binding"] is True
    assert calls[-1][2]["depth"] == 16 and calls[-1][2]["lazy"] is True
    assert adapter.sweep("nonce") == ([{"nonce": "nonce"}], True)
    adapter.close()
    assert calls[-2:] == ["close", "close"]


def test_compare_rejects_changed_context_or_missing_raw_receipts():
    before, after = report(), report(rate=60.0)
    # Keys/hardware differ, but both sets are pinned and independently verified by the evaluator.
    result = bench.compare_reports(before, after)
    assert result["status"] == "verified_target_pass"
    assert result["workload_speed_ratios"]["code"] == pytest.approx(1.2)
    after["samples"][0]["receipts"] = []
    assert bench.compare_reports(before, after)["status"] == "unverified"
    after = report()
    after["protocol"]["env"]["V4_MAX_SEQ"] = "16384"
    for node in after["protocol"]["hardware"]:
        node["stage_env_sha256"] = bench.digest(after["protocol"]["env"])
    after["protocol"] = bench.seal_protocol(after["protocol"])
    after["protocol_sha256"] = after["protocol"]["sha256"]
    assert bench.compare_reports(before, after)["status"] == "unverified"


def test_checkpoint_hash_detects_content_change_and_unsafe_paths(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"n_layers": 43, "expert_dtype": "fp4"}), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"original")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    inventory = bench.checkpoint_inventory(tmp_path)
    assert bench.verify_checkpoint(tmp_path, inventory)["checkpoint_bytes_verified"]
    (tmp_path / "model.safetensors").write_bytes(b"tampered")  # same size: size-only checks would miss it
    with pytest.raises(bench.BenchmarkError, match="differ"):
        bench.verify_checkpoint(tmp_path, inventory)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"expert.weight": "../outside.safetensors"}}), encoding="utf-8")
    with pytest.raises(bench.BenchmarkError, match="escapes"):
        bench.checkpoint_inventory(tmp_path)


def test_import_and_help_never_import_torch_or_model_modules():
    root = Path(__file__).resolve().parents[1]
    code = "import sys; import phase0.v4_benchmark; assert 'torch' not in sys.modules; assert 'v4_pipe' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
    proc = subprocess.run([sys.executable, str(root / "phase0" / "v4_benchmark.py"), "--help"],
                          cwd=root, text=True, capture_output=True)
    assert proc.returncode == 0 and "existing-ring" in proc.stdout


def test_cli_verify_exit_is_nonzero_without_evidence(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"tokPerSec": 99, "receiptsOk": True}), encoding="utf-8")
    assert bench.main(["verify", str(path)]) == 2
    assert bench.HISTORICAL["verification"] == "historical_repository_claim_not_independently_verified"


def test_standalone_verifier_from_outside_repo_without_pythonpath(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report()), encoding="utf-8")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, str(bench.ROOT / "phase0" / "v4_benchmark.py"),
                           "verify", str(path)], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert json.loads(proc.stdout)["status"] == "passed"

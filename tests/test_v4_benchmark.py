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
    p, keys = observed_protocol()
    def measured(rate):
        clock=Clock(); adapter=ObservedAdapter(p,keys,clock,rate=rate); adapter.backend="live_ring"
        return bench.run_suite(adapter,p,fresh_ring=True,clock=clock)
    before, after = measured(50), measured(60)
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


def logical_checkpoint(tmp_path):
    """Small identity-only files, never a loadable 43-layer model or GPU proof."""
    import shutil
    torch = pytest.importorskip("torch")
    tensors = pytest.importorskip("safetensors.torch")
    from shard import weight_artifacts as art
    full = tmp_path / "full"; full.mkdir()
    config = {"n_layers": 43, "dim": 4096, "n_routed_experts": 256, "n_activated_experts": 6,
              "hc_mult": 4, "n_mtp_layers": 3, "dspark_target_layer_ids": [40, 41, 42],
              "dtype": "fp8", "expert_dtype": "fp4", "scale_fmt": "ue8m0"}
    (full / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (full / "tokenizer.json").write_text("{}", encoding="utf-8")
    tensors.save_file({"embed.weight": torch.ones(2, 2), "layers.0.test_weight": torch.arange(4).reshape(2, 2).float()},
                      str(full / "model0-mp1.safetensors"))
    catalog, pack = art.catalogue_directory(full, bench.MODEL_ID, source={"fixture": "identity-only"})
    art.write_metadata(full, catalog, pack)
    assets = tmp_path / "assets"; assets.mkdir()
    shutil.copyfile(full / art.GLOBAL_FILE, assets / art.GLOBAL_FILE)
    for name in catalog["assets"]:
        shutil.copyfile(full / name, assets / name)
    partial = tmp_path / "partial"
    stage = art.select_stage_artifacts(catalog, pack, 0, 1, head=True)
    def read(row, offset, size):
        with (full / row["path"]).open("rb") as stream:
            stream.seek(offset); return stream.read(size)
    art.repack_stage(catalog, pack, stage, partial, read, max_file_bytes=512, chunk_bytes=17)
    return full, assets, partial


def test_logical_identity_is_packing_independent_and_local_hash_scope_stays_truthful(tmp_path):
    full, assets, partial = logical_checkpoint(tmp_path)
    complete_inventory = bench.checkpoint_inventory(full)
    coordinator_inventory = bench.checkpoint_inventory(assets)
    partial_inventory = bench.checkpoint_inventory(partial)
    assert bench._checkpoint_identity(complete_inventory) == bench._checkpoint_identity(coordinator_inventory)
    assert bench._checkpoint_identity(partial_inventory) == bench._checkpoint_identity(coordinator_inventory)
    assert complete_inventory["local_verification"]["verification_scope"] == "complete packed payload hashes"
    coordinator = bench.verify_checkpoint(assets, complete_inventory)
    local = bench.verify_checkpoint(partial, complete_inventory)
    assert coordinator["global_catalog_verified"] and not coordinator["checkpoint_bytes_verified"]
    assert not coordinator["assigned_checkpoint_bytes_verified"]
    assert local["assigned_checkpoint_bytes_verified"] and not local["checkpoint_bytes_verified"]
    assert bench.verify_checkpoint(full, coordinator_inventory)["checkpoint_bytes_verified"]


def test_logical_benchmark_requires_authenticated_cohort_and_refuses_false_full_local_proof(tmp_path):
    _, assets, _ = logical_checkpoint(tmp_path)
    p, keys = protocol()
    p["checkpoint"] = bench._checkpoint_identity(bench.checkpoint_inventory(assets))
    for node in p["hardware"]:
        node["checkpoint_sha256"] = p["checkpoint"]["sha256"]
        node["config_sha256"] = p["checkpoint"]["config_sha256"]
    p = bench.seal_protocol(p)
    assert not bench.protocol_errors(p)
    clock = Clock(); adapter = FakeAdapter(p, keys, clock)
    adapter.backend = "live_ring"  # Synthetic evaluator branch only, not hardware evidence.
    adapter.artifact_verification = {**bench.verify_checkpoint(assets, p["checkpoint"]),
                                    "engine_source_verified": True, "authenticated_cohort_verified": False}
    r = bench.run_suite(adapter, p, fresh_ring=True, clock=clock)
    assert bench.evaluate_report(r)["status"] == "unverified"
    r["artifact_verification"]["authenticated_cohort_verified"] = True
    assert bench.evaluate_report(r)["status"] == "passed"
    r["artifact_verification"]["checkpoint_bytes_verified"] = True
    assert bench.evaluate_report(r)["status"] == "failed"
    r["artifact_verification"]["checkpoint_bytes_verified"] = False
    r["artifact_verification"]["checkpoint_id"] = "tensor-sha256:" + "0" * 64
    assert bench.evaluate_report(r)["status"] == "unverified"


def test_logical_catalogue_and_encoder_tampering_are_rejected(tmp_path):
    _, assets, _ = logical_checkpoint(tmp_path)
    from shard import weight_artifacts as art
    inventory = bench.checkpoint_inventory(assets)
    (assets / "tokenizer.json").write_text("[]", encoding="utf-8")
    with pytest.raises(bench.BenchmarkError, match="asset bytes differ"):
        bench.verify_checkpoint(assets, inventory)
    (assets / "tokenizer.json").write_text("{}", encoding="utf-8")
    (assets / "special_tokens_map.json").write_text("{}")
    with pytest.raises(bench.BenchmarkError, match="unverified local tokenizer"):
        bench.checkpoint_inventory(assets)
    (assets / "special_tokens_map.json").unlink()
    body = json.loads((assets / art.GLOBAL_FILE).read_text())
    body["tensors"]["embed.weight"]["sha256"] = "0" * 64
    (assets / art.GLOBAL_FILE).write_text(json.dumps(body))
    with pytest.raises(art.ArtifactError, match="logical model identity"):
        bench.checkpoint_inventory(assets)


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


def observed_protocol():
    p, keys = protocol()
    p["source"]["files"] = [{"name": name, "sha256": "d" * 64} for name in (
        "engines/deepseek_v4/v4_pipe.py", "engines/deepseek_v4/v4_stage.py",
        "vendor/deepseek_v4_ref/inference/model.py", "shard/pipeline_session.py", "shard/transport.py")]
    p["source"]["working_tree"] = {"known": True, "dirty": True, "status_sha256": "a" * 64}
    p["evidence_contract"] = {"coordinator_diagnostics_required": True, "runtime_observation_required": True}
    p["network_comparison"] = {"schema": bench.NETWORK_SCHEMA, "transport": "local_test_fixture",
        "route_identity_sha256": "a" * 64, "latency_semantics": "no_latency_claim", "measurement_method": "scripted CPU fixture"}
    return reseal_observed_protocol(p), keys


def reseal_observed_protocol(p):
    """Rebind synthetic declarations, never produce physical hardware evidence."""
    p["source"]["sha256"] = bench.digest(p["source"]["files"])
    for node in p["hardware"]:
        node["engine_source_sha256"] = p["source"]["sha256"]
        node["stage_env_sha256"] = bench.digest(p["env"])
        node["runtime_config_sha256"] = bench.digest({"env": p["env"], "source": p["source"]["sha256"]})
        node["hadamard_backend"] = "extension" if p["env"]["V4_HADAMARD"] == "extension" else "torch"
        node["graph_mode"] = "whole"
    return bench.seal_protocol(p)


def signed_observation(p, node):
    from shard.runtime_observation import SCHEMA, digest
    files = {row["name"]: row["sha256"] for row in p["source"]["files"]}
    env = dict(p["env"])
    flags = {"V4_HADAMARD": {"requested": env["V4_HADAMARD"], "parsed": env["V4_HADAMARD"],
        "observed": node["hadamard_backend"], "verdict": "OK", "reason": "CPU-signed fixture only"}}
    return {"schema": SCHEMA, "node_id": node["node_id"], "gpu_uuid": node["gpu_uuid"],
        "process_run_id": node["process_run_id"], "runtime_config_sha256": node["runtime_config_sha256"],
        "source_files": files, "source_sha256": digest(files), "environment": env, "environment_sha256": digest(env),
        "effective_flags": flags, "effective_flags_sha256": digest(flags), "kernel_backend": "tilelang",
        "hadamard_backend": node["hadamard_backend"], "graph_mode": node["graph_mode"], "wire_mode": "bf16",
        "backend_identity": {"hadamard": {"requested": env["V4_HADAMARD"], "backend": node["hadamard_backend"],
            "reason": "CPU-signed fixture only", "source_sha256": "a"*64,
            "module_files": [{"name": "hadamard_cuda.pyd", "sha256": "b"*64}] if node["hadamard_backend"] == "extension" else
                [{"name": "v4_kernels_cpu.py", "sha256": next((row["sha256"] for row in p["source"]["files"] if row["name"].endswith("/v4_kernels_cpu.py")), "d"*64)}],
            "dependency_version": "fixture-version" if node["hadamard_backend"] == "extension" else None}},
        "transport": "engine-message-socket; external route declared by deployment",
        "versions": {"python": "3.11", "torch": node["torch"], "cuda": node["cuda"], "tilelang": node["tilelang"]},
        "phase": "job_complete"}


class ObservedAdapter(FakeAdapter):
    """Actual Ed25519 fixture signatures, wholly synthetic model and device data."""
    backend = "live_ring"  # Exercises the acceptance branch only, never a GPU speed claim.

    def __init__(self, *args, observation_mutator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_mutator = observation_mutator

    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        decode = len(result["tokens"]) - 1
        speculative = kwargs["mode"] != "greedy"
        result.update(coordinator_counters={"accepted_predictions": min(3, decode) if speculative else 0,
            "proposed_predictions": decode+3 if speculative else 0, "cancel_events": int(speculative),
            "speculation_cycles": 2 if speculative else decode, "frames_enqueued": decode+3 if speculative else decode,
            "frames_sent": decode+2 if speculative else decode, "replies_received": decode+2 if speculative else decode,
            "frames_judged": decode, "stale_replies": int(speculative), "drained_replies": int(speculative),
            "unsent_frames": int(speculative)},
            inflight_intervals=[{"duration_s": .01, "level": 1}, {"duration_s": .1, "level": 4}] if speculative else [],
            prefill_committed_tokens=1, g=len(result["tokens"])/2)
        return result

    def sweep(self, nonce):
        assert nonce == self.last[0]
        self.clock.advance(self.sweep_s)
        receipts = []
        for i, (node, key) in enumerate(zip(self.protocol["hardware"], self.keys)):
            signer = ReceiptSigner(key, self.swarm_id, self.last[1], node["layer_start"], node["layer_end"], nonce=nonce)
            for chunk in range(2):
                signer.observe(f"boundary-{i}-{chunk}".encode(), f"boundary-{i+1}-{chunk}".encode())
            value = signed_observation(self.protocol, node)
            if self.observation_mutator:
                self.observation_mutator(value)
            receipts.append(signer.finalize(runtime_observation=value))
        return receipts, True


def observed_report(p, keys, *, rate=50, observation_mutator=None):
    clock = Clock()
    return bench.run_suite(ObservedAdapter(p, keys, clock, rate=rate, observation_mutator=observation_mutator),
                           p, fresh_ring=True, clock=clock)


def test_complete_diagnostics_signed_observations_and_raw_latency_distributions():
    p, keys = observed_protocol()
    r = observed_report(p, keys)
    evaluation = bench.evaluate_report(r)
    assert evaluation["valid_evidence"], evaluation
    assert evaluation["runtime_evidence"]["signed_observation_count"] == 80
    assert evaluation["runtime_evidence"]["not_remote_execution_attestation"] is True
    sample = r["samples"][0]
    timing = sample["measurement"]
    assert timing["drain_s"] == pytest.approx(.001)
    assert timing["full_service_s"] == pytest.approx(timing["elapsed_s"]+60)
    assert sample["coordinator_stats"]["g"] == 4
    assert sample["coordinator_diagnostics"]["derived"]["g_cycle"] == 3.5
    summary = evaluation["workloads"]["code"]
    assert summary["timing_distributions"]["receipt_sweep_s"]["samples"] == [60, 60, 60]
    assert summary["timing_distributions"]["drain_s"]["p95"] == pytest.approx(.001)
    assert summary["coordinator_distributions"]["inflight_time_avg"]["p50"] == pytest.approx(.41/.11)
    assert summary["coordinator_count_distributions"]["accepted_predictions"]["samples"] == [3, 3, 3]
    assert summary["coordinator_count_distributions"]["cancel_events"]["p95"] == 1
    assert len(summary["inter_token_latency_s"]["samples"]) == 3*7
    assert summary["inter_token_latency_s"]["p95"] == pytest.approx(.02)


def test_generation_must_not_sweep_twice():
    p, keys = observed_protocol(); clock = Clock()
    class AlreadySwept(ObservedAdapter):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            result["receipt_sweep_s"] = 3.0
            return result
    with pytest.raises(bench.BenchmarkError, match="disable internal receipt sweep"):
        bench.run_suite(AlreadySwept(p, keys, clock), p, fresh_ring=True, clock=clock)


@pytest.mark.parametrize("field", ["source", "env", "backend", "process", "audit"])
def test_authenticated_but_wrong_runtime_declarations_fail_frozen_contract(field):
    from shard.runtime_observation import digest
    p, keys = observed_protocol()
    def mutate(value):
        if field == "source":
            value["source_files"]["engines/deepseek_v4/v4_stage.py"] = "e" * 64
            value["source_sha256"] = digest(value["source_files"])
        elif field == "env":
            value["environment"]["V4_SPEC_DEPTH"] = "999"
            value["environment_sha256"] = digest(value["environment"])
        elif field == "backend":
            value["kernel_backend"] = "cpu"
        elif field == "process":
            value["process_run_id"] = "unfrozen-process"
        else:
            value["effective_flags"]["V4_HADAMARD"]["verdict"] = "MISMATCH"
            value["effective_flags_sha256"] = digest(value["effective_flags"])
    r = observed_report(p, keys, observation_mutator=mutate)
    assert bench.evaluate_report(r)["status"] == "failed"


def test_unsigned_source_env_backend_edits_are_rejected_as_signature_tampering():
    p, keys = observed_protocol(); original = observed_report(p, keys)
    for field, value in (("kernel_backend", "cpu"), ("environment_sha256", "e"*64), ("source_sha256", "e"*64)):
        r = copy.deepcopy(original)
        r["samples"][0]["receipts"][0]["runtime_observation"][field] = value
        assert bench.evaluate_report(r)["status"] == "failed"


def test_missing_counters_and_modified_raw_intervals_do_not_claim_verified_diagnostics():
    p, keys = observed_protocol(); original = observed_report(p, keys)
    r = copy.deepcopy(original)
    r["samples"][0]["coordinator_stats"]["inflight_intervals"][0]["duration_s"] = 3
    assert bench.evaluate_report(r)["status"] == "failed"
    r = copy.deepcopy(original)
    from shard.benchmark_metrics import make_coordinator_diagnostics
    s = r["samples"][0]
    s["coordinator_stats"]["coordinator_counters"].pop("cancel_events")
    s["coordinator_diagnostics"] = make_coordinator_diagnostics("pipelined", committed_tokens=8,
        counters=s["coordinator_stats"]["coordinator_counters"], timing=s["coordinator_diagnostics"]["timing"],
        inflight_intervals=s["coordinator_stats"]["inflight_intervals"])
    assert bench.evaluate_report(r)["status"] == "failed"


@pytest.mark.parametrize("vary", ["source", "env:V4_HADAMARD"])
def test_declared_single_variable_comparison_allows_only_derived_identity_changes(vary, tmp_path):
    p, keys = observed_protocol(); p["env"]["V4_HADAMARD"] = "torch"; p = reseal_observed_protocol(p)
    q = copy.deepcopy(p)
    if vary == "source":
        q["source"]["files"][0]["sha256"] = "e" * 64
    else:
        q["env"]["V4_HADAMARD"] = "extension"
    for node in q["hardware"]:
        node["process_run_id"] += "-new-process"
    q = reseal_observed_protocol(q)
    before, after = observed_report(p, keys), observed_report(q, keys, rate=60)
    assert bench.compare_reports(before, after)["status"] == "unverified"
    result = bench.compare_reports(before, after, vary=vary)
    assert result["status"] == "verified_target_pass", result
    assert result["declared_variable"] == vary and result["changes"] == [vary]
    assert result["frozen_controls"]["after"]["hardware"][0]["process_run_id"].endswith("new-process")
    if vary.startswith("env:"):
        assert result["before"]["runtime_evidence"]["stage_identities"]["test-node-0"]["hadamard_backend"] == "torch"
        assert result["after"]["runtime_evidence"]["stage_identities"]["test-node-0"]["hadamard_backend"] == "extension"
    a, b = tmp_path / "before.json", tmp_path / "after.json"
    a.write_text(json.dumps(before)); b.write_text(json.dumps(after))
    assert bench.main(["compare", str(a), str(b), "--vary", vary]) == 0


@pytest.mark.parametrize("change", ["driver", "two_flags", "source_and_flag", "network_semantics", "output", "backend"])
def test_comparison_rejects_mixed_build_hardware_network_semantics_or_output(change):
    p, keys = observed_protocol(); q = copy.deepcopy(p)
    vary = "env:V4_HADAMARD"
    q["env"]["V4_HADAMARD"] = "extension"
    if change == "driver":
        q["hardware"][0]["driver"] = "other-driver"
    elif change == "two_flags":
        q["env"]["V4_SPEC_DEPTH"] = "8"
    elif change == "source_and_flag":
        q["source"]["files"][0]["sha256"] = "e" * 64
    elif change == "network_semantics":
        q["env"]["V4_HADAMARD"] = p["env"]["V4_HADAMARD"]
        q["network_comparison"]["latency_semantics"] = "one_way"
        vary = "network"
    elif change == "backend":
        q["hardware"][0]["graph_mode"] = "off"
    q = reseal_observed_protocol(q)
    if change == "backend":
        q["hardware"][0]["graph_mode"] = "off"; q = bench.seal_protocol(q)
    before, after = observed_report(p, keys), observed_report(q, keys)
    if change == "output":
        # Coherent same-ring parity can still differ BETWEEN before/after arms.
        for sample in after["samples"]:
            sample["tokens"][0] += 1
            sample["output_sha256"] = bench.digest(sample["tokens"])
    assert bench.compare_reports(before, after, vary=vary)["status"] == "unverified"


def test_operator_only_legacy_report_is_not_effective_backend_ab_evidence():
    before = report(); after = copy.deepcopy(before)
    for r in (before, after):
        r["protocol"]["network_comparison"] = {"schema": bench.NETWORK_SCHEMA, "transport": "fixture",
            "route_identity_sha256": "a"*64, "latency_semantics": "no_latency_claim", "measurement_method": "fixture"}
        r["protocol"] = bench.seal_protocol(r["protocol"]); r["protocol_sha256"] = r["protocol"]["sha256"]
    assert bench.evaluate_report(before)["status"] == "passed"
    assert bench.compare_reports(before, after)["status"] == "unverified"


def test_declared_network_route_change_preserves_latency_semantics():
    p, keys = observed_protocol(); q = copy.deepcopy(p)
    q["network_comparison"]["route_identity_sha256"] = "b" * 64
    q = bench.seal_protocol(q)
    result = bench.compare_reports(observed_report(p, keys), observed_report(q, keys), vary="network")
    assert result["status"] == "verified_target_pass", result
    assert result["changes"] == ["network"]


def test_required_signed_observation_missing_cannot_pass_and_compare_invalid_protocol_is_safe():
    p, keys = observed_protocol(); r = observed_report(p, keys)
    # Replace one signed receipt with a valid legacy receipt for the SAME signer/job.
    sample = r["samples"][0]; node = p["hardware"][0]
    signer = ReceiptSigner(keys[0], sample["swarm_id"], sample["job_id"], node["layer_start"], node["layer_end"], nonce=sample["nonce"])
    for chunk in range(2):
        signer.observe(f"boundary-0-{chunk}".encode(), f"boundary-1-{chunk}".encode())
    sample["receipts"][0] = signer.finalize()
    assert bench.evaluate_report(r)["status"] == "unverified"
    invalid = copy.deepcopy(r); invalid["protocol"]["network_comparison"] = {"schema": bench.NETWORK_SCHEMA}
    invalid["protocol"] = bench.seal_protocol(invalid["protocol"]); invalid["protocol_sha256"] = invalid["protocol"]["sha256"]
    assert bench.compare_reports(r, invalid)["status"] == "unverified"


@pytest.mark.parametrize("vary", ["source", "env:V4_SPEC_DEPTH", "env:V4_HADAMARD"])
def test_same_backend_binary_change_cannot_hide_inside_derived_config_digest(vary):
    p, keys = observed_protocol(); q = copy.deepcopy(p)
    if vary == "source":
        q["source"]["files"][0]["sha256"] = "e"*64
    elif vary == "env:V4_SPEC_DEPTH":
        q["env"]["V4_SPEC_DEPTH"] = "8"
    else:
        # auto->torch keeps the actual backend; it must not authorize binary drift.
        q["env"]["V4_HADAMARD"] = "torch"
    q = reseal_observed_protocol(q)
    def binary_change(value):
        value["backend_identity"]["hadamard"]["module_files"][0]["sha256"] = "e"*64
    result = bench.compare_reports(observed_report(p, keys), observed_report(q, keys, observation_mutator=binary_change), vary=vary)
    assert result["status"] == "unverified", result
    assert "observed backends/settings" in result["reason"]


def test_source_experiment_allows_bound_repo_hadamard_function_change():
    p, keys = observed_protocol()
    p["source"]["files"].append({"name": "engines/deepseek_v4/v4_kernels_cpu.py", "sha256": "d"*64})
    p = reseal_observed_protocol(p); q = copy.deepcopy(p)
    q["source"]["files"][-1]["sha256"] = "e"*64; q = reseal_observed_protocol(q)
    def function_change(value):
        value["backend_identity"]["hadamard"]["source_sha256"] = "f"*64
    result = bench.compare_reports(observed_report(p, keys), observed_report(q, keys, observation_mutator=function_change), vary="source")
    assert result["status"] == "verified_target_pass", result

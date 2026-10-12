"""CPU-only schema, real signature and owned-PID tests; no hardware qualification.

Inventories, allocator values and capture coverage below are explicitly synthetic.
They test fail-closed plumbing, never prove physical GPUs or measured capacity.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import v4_acceptance_ring as ring
from shard.manifest import gen_key, pub_b64, save_key
from shard.offers import ModelCohort
from shard.receipt import ReceiptSigner, ReceiptError


def config(model):
    return {"schema": ring.SCHEMA, "ring_id": "synthetic-ring", "model_dir": str(model),
            "python": sys.executable, "environment": {"V4_EXECUTION_PROFILE": "resident-whole"},
            "stages": [{"gpu_index": i, "lo": i * 8, "hi": min((i + 1) * 8, 43), "port": 29610 + i}
                       for i in range(6)]}


@pytest.fixture
def state(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "private").mkdir()
    cfg = ring.validate_config(config(tmp_path / "model"))
    keys = [gen_key() for _ in range(7)]
    paths = []
    for i, key in enumerate(keys):
        path = directory / "private" / f"synthetic-{i}.key"
        save_key(key, str(path)); paths.append(str(path))
    cohort = ModelCohort("deepseek-ai/DeepSeek-V4-Flash-0731", "a" * 64, "tensor-sha256:" + "b" * 64,
                        "c" * 64, "fp4-fp8", "deepseek-v4-native/1", "shard-pipeline-session/1",
                        "greedy-native-fp4-fp8/1", 43).to_dict()
    inventory = {"schema": "flyai-v4-same-host-gpu-inventory/1", "host_id": "synthetic-same-host",
                 "observed_at": datetime.now(timezone.utc).isoformat(), "hardware_attested": False,
                 "gpus": [{"gpu_index": i, "gpu_uuid": f"GPU-{i + 1:08x}-0000-0000-0000-000000000000",
                           "gpu_name": "NVIDIA GeForce RTX 5090", "driver": "synthetic-no-hardware",
                           "vram_bytes": 32 * 1024**3, "free_bytes": 31 * 1024**3,
                           "utilization_percent": 0, "compute_process_present": False}
                          for i in range(6)]}
    value = {"schema": "flyai-v4-acceptance-ring-state/1", "scope": ring.SCOPE, "config": cfg,
             "config_sha256": ring._digest(cfg), "cohort": cohort, "source": ring._source_identity(),
             "checkpoint": {"sha256": "b" * 64, "config_sha256": "c" * 64},
             "signer_pubkeys": [pub_b64(key) for key in keys], "key_paths": paths,
             "gpu_inventory": inventory, "owned_names": [], "status": "PREPARED"}
    ring._save_state(directory, value)
    return directory, value, keys


def measurement(state, index):
    cfg = state["config"]
    row = cfg["stages"][index]
    roles = {"lo": row["lo"], "hi": row["hi"], "head": index == 0, "tail": index == 5, "dspark": index == 5}
    payload = {**roles, "args": {"max_seq_len": int(cfg["environment"]["V4_MAX_SEQ"]), "max_batch_size": 1},
               "environment": {k: v for k, v in ring.stage_environment(state, index).items() if k.startswith("V4_")},
               "engine_source_sha256": {Path(row["name"]).name: row["sha256"] for row in state["source"]["files"]
                                        if row["name"].startswith("engines/deepseek_v4/v4_")},
               "dspark_loaded": index == 5}
    return {"schema": "v4-resource-measurement/1", "calibration_status": "measured", **roles,
            "measured_at": datetime.now(timezone.utc).isoformat(), "checkpoint_id": state["cohort"]["checkpoint_id"],
            "runtime_config_payload": payload,
            "runtime_config_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "hardware": {"device": "cuda:0", "gpu_uuid": state["gpu_inventory"]["gpus"][index]["gpu_uuid"],
                         "gpu_name": "NVIDIA GeForce RTX 5090"},
            "draft_measurement_status": "measured" if index == 5 else "not_requested",
            "public_environment": cfg["environment"],
            "workload": {"kind": "synthetic_resource_probe", "batch": 1, "max_seq": 8192,
                         "prefill_tokens": 16, "decode_tokens": 8},
            "allocator": {"scope": "process", "peak_interval_owned": True, "peak_reserved_bytes": 8 * 1024**3,
                          "peak_allocated_bytes": 7 * 1024**3, "device_free_bytes": 24 * 1024**3,
                          "device_total_bytes": 32 * 1024**3}}


def observation(state, plan, index, report, *, phase="job_complete"):
    row = plan["stages"][index]
    source = {row["name"]: row["sha256"] for row in state["source"]["files"]}
    env = state["config"]["environment"]
    coverage = {"schema": "flyai-v4-whole-coverage/1", "captured_layers": row["hi"] - row["lo"],
                "refused_layers": 0, "undecided_layers": 0,
                "layers": [{"layer": layer, "captured_graphs": 1, "eager": False, "moe_in_graph": True}
                           for layer in range(row["lo"], row["hi"])]}
    return {"schema": "shard-runtime-observation/1", "node_id": row["node_id"], "gpu_uuid": row["gpu_uuid"],
            "process_run_id": f"synthetic-not-hardware-{index}", "runtime_config_sha256": report["runtime_config_sha256"],
            "source_files": source, "source_sha256": ring._digest(source), "environment": env,
            "environment_sha256": ring._digest(env), "effective_flags": {}, "effective_flags_sha256": ring._digest({}),
            "kernel_backend": "tilelang", "hadamard_backend": "torch", "graph_mode": "whole", "wire_mode": "bf16",
            "transport": "synthetic-CPU-schema", "versions": {k: "synthetic-no-hardware" for k in ("python", "torch", "cuda", "tilelang")},
            "phase": phase, "backend_identity": {"hadamard": {"requested": "auto", "backend": "torch",
                "reason": "synthetic CPU schema fixture", "source_sha256": "d" * 64, "module_files": [], "dependency_version": None}},
            "optimizations": {"prefill_expert_pipeline": {}, "conversation_cache": {}, "execution_profile": {
                "name": "resident-whole", "graph_mode": "whole", "coverage": coverage}}}


def final_result(state, plan, keys, reports, job, *, missing=False, bad_coverage=False):
    receipts = []
    for i, (row, key) in enumerate(zip(plan["stages"], keys)):
        signer = ReceiptSigner(key, plan["ring_id"], job["jobId"], row["lo"], row["hi"], job["nonce"])
        signer.observe(f"boundary-{i}".encode(), f"boundary-{i + 1}".encode())
        observed = observation(state, plan, i, reports[i])
        if bad_coverage and i == 0:
            observed["optimizations"]["execution_profile"]["coverage"]["captured_layers"] = 0
        receipts.append(signer.finalize(runtime_observation=observed))
    return {"ok": True, "jobId": job["jobId"], "tokensGenerated": 2, "tokenIds": [7, 8],
            "receipts": [] if missing else receipts}


def test_prepare_is_cpu_only_and_public_exports_omit_private_key_paths(tmp_path, monkeypatch):
    from shard import weight_artifacts as W
    from phase0 import v4_benchmark as B
    model = tmp_path / "model"; model.mkdir()
    for name in (W.GLOBAL_FILE, W.PACK_FILE):
        (model / name).write_text("{}")
    cohort = ModelCohort("deepseek-ai/DeepSeek-V4-Flash-0731", "a" * 64, "tensor-sha256:" + "b" * 64,
                        "c" * 64, "fp4-fp8", "deepseek-v4-native/1", "shard-pipeline-session/1",
                        "greedy-native-fp4-fp8/1", 43).to_dict()
    monkeypatch.setattr(W, "verify_weight_pack", lambda *a, **kw: {"catalog": {"synthetic": True}})
    monkeypatch.setattr(W, "catalog_model_cohort", lambda catalog: cohort)
    monkeypatch.setattr(B, "checkpoint_inventory", lambda model: {"sha256": "b" * 64, "config_sha256": "c" * 64})
    def forbidden(*a, **kw):
        raise AssertionError("prepare must not launch a subprocess or probe GPUs")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(ring, "gpu_inventory", forbidden)
    directory = tmp_path / "run"
    public = ring.prepare(config(model), directory)
    state = ring.load_state(directory)
    assert public["runtime_ready"] is False and public["hardware_qualification"] is False
    assert len(set(state["signer_pubkeys"])) == 7
    text = (directory / "prepared.json").read_text()
    assert "key_paths" not in text and "coordinator.key" not in text
    assert all(Path(path).is_file() for path in state["key_paths"])
    assert not (directory / "hardware.json").exists()
    assert state["owned_names"] == []


@pytest.mark.parametrize("change", ["gap", "overlap", "duplicate_gpu", "duplicate_port", "tail", "implicit_split"])
def test_assignment_rejects_implicit_or_invalid_spans(tmp_path, change):
    cfg = config(tmp_path)
    if change == "gap": cfg["stages"][1]["lo"] = 9
    if change == "overlap": cfg["stages"][1]["lo"] = 7
    if change == "duplicate_gpu": cfg["stages"][1]["gpu_index"] = 0
    if change == "duplicate_port": cfg["stages"][1]["port"] = 29610
    if change == "tail": cfg["stages"][-2]["hi"] = 41; cfg["stages"][-1]["lo"] = 41
    if change == "implicit_split": del cfg["stages"][2]["hi"]
    with pytest.raises(ValueError): ring.validate_config(cfg)


def test_plan_direct_tail_and_per_gpu_singleton_environment(state):
    _, value, _ = state
    plan = ring.make_plan(value)
    assert plan["coordinator"]["tail"] == plan["stages"][-1]["endpoint"] == "127.0.0.1:29615"
    assert [(r["lo"], r["hi"]) for r in plan["stages"]] == [(i * 8, min((i + 1) * 8, 43)) for i in range(6)]
    for i in range(6):
        env = ring.stage_environment(value, i)
        assert env["CUDA_VISIBLE_DEVICES"] == str(i)
        assert env["V4_DEV"] == "cuda:0"
        argv = ring.stage_argv(value, "plan.json", i)
        assert "--legacy-protocol" not in argv and "--ret-relay" not in argv
        assert argv[argv.index("--device") + 1] == "cuda:0"
        assert {k: v for k, v in env.items() if k.startswith("V4_") and k not in ("V4_DIR", "V4_DEV")} == value["config"]["environment"]
    assert value["config"]["environment"]["V4_PROFILE_GPU_EVERY"] == "16"
    assert value["config"]["environment"]["V4_EXPERT_PREFETCH"] == "0"


@pytest.mark.parametrize("change", ["uuid", "stale", "source", "environment", "role", "headroom", "peak", "draft"])
def test_resource_gate_rejects_inexact_or_insufficient_evidence(state, change):
    _, value, _ = state
    index = 5 if change == "draft" else 0
    report = measurement(value, index)
    if change == "uuid": report["hardware"]["gpu_uuid"] = value["gpu_inventory"]["gpus"][1]["gpu_uuid"]
    if change == "stale": report["measured_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    if change == "source": report["runtime_config_payload"]["engine_source_sha256"] = {}
    if change == "environment": report["public_environment"] = dict(report["public_environment"], V4_UNKNOWN="1")
    if change == "role": report["head"] = False
    if change == "headroom": value["gpu_inventory"]["gpus"][index]["free_bytes"] = 9 * 1024**3
    if change == "peak": report["allocator"]["peak_interval_owned"] = False
    if change == "draft": report["draft_measurement_status"] = "missing"
    with pytest.raises(ValueError): ring.validate_measurement(report, value, index)


def test_resource_gate_is_per_role_and_checks_each_assigned_gpu(state):
    _, value, _ = state
    for i in range(6):
        assert ring.validate_measurement(measurement(value, i), value, i)["validated"] is True
    value["gpu_inventory"]["gpus"][4]["compute_process_present"] = True
    with pytest.raises(ValueError, match="idle"):
        ring._assigned_idle(value, value["gpu_inventory"])
    with pytest.raises(ValueError, match="confirm"):
        ring._confirmed(value, [0, 1, 2, 3, 4])


def test_real_signatures_bind_warmup_and_require_actual_profile_coverage(state):
    _, value, keys = state
    plan = ring.make_plan(value)
    reports = [measurement(value, i) for i in range(6)]
    job = {"jobId": "synthetic-CPU-warmup", "maxNew": 2, "nonce": "1" * 64}
    final = final_result(value, plan, keys, reports, job)
    proof = ring.verify_warmup(final, value, plan, job, reports)
    hardware = ring.hardware_from_warmup(value, plan, proof)
    assert proof["verified"] and len(hardware) == 6
    assert len({row["host_id"] for row in hardware}) == 1
    assert hardware[0]["stage_env_sha256"] == ring._digest(value["config"]["environment"])
    assert hardware[0]["process_run_id"] == "synthetic-not-hardware-0"
    broken = deepcopy(final); broken["receipts"][0]["runtime_observation"]["process_run_id"] = "tampered"
    with pytest.raises(ReceiptError): ring.verify_warmup(broken, value, plan, job, reports)
    with pytest.raises(ReceiptError):
        ring.verify_warmup(final_result(value, plan, keys, reports, job, missing=True), value, plan, job, reports)
    with pytest.raises(ValueError, match="capture"):
        ring.verify_warmup(final_result(value, plan, keys, reports, job, bad_coverage=True), value, plan, job, reports)


class FakeManager:
    def __init__(self): self.started = []; self.stopped = []
    def status(self, name): return {"reserved": False}
    def start(self, *a, **kw): raise AssertionError("test supplies _launch directly")
    def stop(self, name): self.stopped.append(name); return {"name": name, "running": False}


def mocked_launch(state, monkeypatch, *, missing_receipts=False):
    from phase0 import v4_benchmark as B
    directory, value, keys = state
    manager = FakeManager()
    requests = {}
    monkeypatch.setattr(ring, "_manager", lambda directory: manager)
    monkeypatch.setattr(B, "verify_checkpoint", lambda *args: {"checkpoint_bytes_verified": True})
    reports = [measurement(value, i) for i in range(6)]
    def launch(directory, current, name, request):
        manager.started.append(name); requests[name] = request
        if name not in current["owned_names"]: current["owned_names"].append(name)
        ring._save_state(directory, current)
        return {"running": True, "reused": False}
    def wait(directory, name, *args, **kwargs):
        plan = ring.make_plan(value)
        if name.endswith(".warmup"):
            job = json.loads(requests[name]["stdin"])
            final = final_result(value, plan, keys, reports, job, missing=missing_receipts)
            return {"running": False, "log_tail": "SHARD_JOB_DONE " + json.dumps(final)}
        i = int(name.rsplit(".", 1)[1])
        return {"running": True, "listening": True, "owned_listener": True,
                "events": [{"event": "loaded", "observation": observation(value, plan, i, reports[i], phase="loaded")}]}
    monkeypatch.setattr(ring, "_launch", launch); monkeypatch.setattr(ring, "_wait", wait)
    monkeypatch.setattr(ring, "_observation", lambda *args, **kwargs: {"running": True, "listening": True, "owned_listener": True})
    return manager, reports


def test_cold_open_is_tail_first_zero_requests_and_never_ready_then_signed_warmup(state, monkeypatch):
    directory, value, _ = state
    manager, reports = mocked_launch(state, monkeypatch)
    result = ring.start(directory, list(range(6)), measurements=reports, cold_open=True,
                        inventory_reader=lambda: value["gpu_inventory"])
    assert manager.started == [f"synthetic-ring.stage.{i}" for i in reversed(range(6))]
    assert result["ready"] is False and result["request_count"] == 0
    assert "unsigned" in result["hardware_evidence"]
    assert not (directory / "signed-warmup.json").exists()
    ready = ring.warmup(directory)
    assert ready["ready"] is True and ready["hardware_qualification"] is False
    assert manager.started[-1] == "synthetic-ring.warmup"
    assert (directory / "signed-warmup.json").is_file()
    exported = (directory / "hardware.json").read_text()
    assert "key_paths" not in exported and "synthetic-6.key" not in exported


def test_listening_without_signed_warmup_rolls_back_only_recorded_names(state, monkeypatch):
    directory, value, _ = state
    manager, reports = mocked_launch(state, monkeypatch, missing_receipts=True)
    with pytest.raises(ReceiptError):
        ring.start(directory, list(range(6)), measurements=reports, inventory_reader=lambda: value["gpu_inventory"])
    assert all(name.startswith("synthetic-ring.") for name in manager.stopped)
    assert all(f"synthetic-ring.stage.{i}" in manager.stopped for i in range(6))
    assert ring.load_state(directory)["status"] == "STOPPED"
    assert not (directory / "signed-warmup.json").exists()


def test_real_owned_cleanup_preserves_foreign_cpu_process_keys_and_weights(state, tmp_path):
    pytest.importorskip("psutil")
    from shard.managed_launch import ManagedLauncher
    directory, value, _ = state
    foreign = ManagedLauncher(tmp_path / "foreign")
    own = ManagedLauncher(directory / "processes")
    name = "synthetic-ring.safe-cpu"
    weight = tmp_path / "preserve-weight.bin"; weight.write_bytes(b"synthetic-preserve")
    foreign.start("foreign-node", [sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        own.start(name, [sys.executable, "-c", "import time; time.sleep(120)"])
        value["owned_names"] = [name]; ring._save_state(directory, value)
        result = ring.stop(directory)
        assert result["cleanup_acknowledged"]
        assert own.status(name)["running"] is False
        assert foreign.status("foreign-node")["running"] is True
        assert weight.read_bytes() == b"synthetic-preserve"
        assert all(Path(path).is_file() for path in value["key_paths"])
    finally:
        own.stop(name); foreign.stop("foreign-node")


def test_persistent_owned_logs_are_capped_before_poll_or_log_read(state, monkeypatch):
    directory, _, _ = state
    processes = directory / "processes"; processes.mkdir()
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized log must fail before process observation or bounded log read")
    from engines.deepseek_v4 import v4_preflight
    monkeypatch.setattr(v4_preflight, "process_observation", forbidden)
    for suffix in ("stage.0", "measure.0", "warmup"):
        name = "synthetic-ring." + suffix
        path = processes / (name + ".log")
        try:
            # Extend the logical file without allocating or writing 256 MiB of
            # fixture data. This is a CPU-only size check, not a GPU log claim.
            with path.open("wb") as stream:
                stream.truncate(ring.MAX_OWNED_LOG_BYTES + 1)
            with pytest.raises(RuntimeError, match="exceeds 256 MiB"):
                ring._observation(directory, name)
            with pytest.raises(RuntimeError, match="exceeds 256 MiB"):
                ring._wait(directory, name, 1)
        finally:
            path.unlink(missing_ok=True)


def test_oversized_initial_stage_log_aborts_standalone_warmup_with_owned_cleanup(state, monkeypatch):
    directory, value, _ = state
    manager, reports = mocked_launch(state, monkeypatch)
    ring.start(directory, list(range(6)), measurements=reports, cold_open=True,
               inventory_reader=lambda: value["gpu_inventory"])
    processes = directory / "processes"; processes.mkdir()
    path = processes / "synthetic-ring.stage.0.log"
    try:
        with path.open("wb") as stream:
            stream.truncate(ring.MAX_OWNED_LOG_BYTES + 1)
        with pytest.raises(RuntimeError, match="exceeds 256 MiB"):
            ring.warmup(directory)
        assert ring.load_state(directory)["status"] == "STOPPED"
        assert all(f"synthetic-ring.stage.{i}" in manager.stopped for i in range(6))
        assert "synthetic-ring.warmup" not in manager.started
    finally:
        path.unlink(missing_ok=True)

"""Explicit, owned-process 5090 campaign. Planning/verification never run GPUs.

This is operator evidence for one same-host ring, not the two-topology official
release certificate and not a cryptographic attestation of rented hardware.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import v4_acceptance_ring as R

SCHEMA = "flyai-v4-5090-campaign-config/1"
MODES = ("greedy", "spec", "dspark", "pipelined")
MATRIX = {"short": {"prompt_tokens": 512, "max_new": 512},
          "long": {"prompt_tokens": 4096, "max_new": 64}}
MAX_LOG_BYTES = 256 * 1024**2


def read(path):
    return R._read(path)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    R._write(path, value)


def digest(value):
    return R._digest(value)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def kit_identity():
    paths = sorted(set((ROOT / "tools").glob("v4*acceptance*.py")) |
                   set((ROOT / "tests").glob("test_v4*acceptance*.py")) |
                   {ROOT / "phase0/v4_numeric_diagnose.py"})
    return [{"name": p.relative_to(ROOT).as_posix(), "sha256": file_hash(p)} for p in paths if p.is_file()]


def validate_config(config, *, preserve_paths=False):
    allowed = {"schema", "ring", "gpu_gate_index", "numeric_gpu_index", "soak", "timeouts"}
    if not isinstance(config, dict) or config.get("schema") != SCHEMA or set(config) - allowed:
        raise ValueError("explicit campaign schema and supported fields required")
    cfg = json.loads(json.dumps(config, allow_nan=False))
    model_path = cfg.get("ring", {}).get("model_dir")
    cfg["ring"] = R.validate_config(cfg.get("ring", {}))
    if preserve_paths:
        # Re-verification may run on Windows after execution on Linux. These are
        # frozen execution declarations, never paths opened by the verifier.
        cfg["ring"]["model_dir"] = model_path
    ring = cfg["ring"]
    if len(ring["stages"]) not in (4, 6):
        raise ValueError("8 stages are diagnostic only; campaign speed acceptance needs 4 or 6")
    env = ring["environment"]
    if (env.get("V4_EXECUTION_PROFILE") != "resident-whole" or env["V4_MAX_SEQ"] != "8192"
            or env["V4_PREFILL_QUERY_CHUNK"] != "128"):
        raise ValueError("this campaign requires resident-whole, max_seq8192 and prefill query chunk128")
    assigned = {row["gpu_index"] for row in ring["stages"]}
    for name in ("gpu_gate_index", "numeric_gpu_index"):
        value = cfg.get(name)
        if type(value) is not int or value < 0 or value in assigned:
            raise ValueError("explicit spare GPU for " + name + " required; cannot overlap running stages")
    soak = {"cycles": 25, "duration_s": 3600, "ttft_p95_s": 30,
            "token_gap_p95_s": .25, "max_idle_gap_s": 10, **cfg.get("soak", {})}
    if set(soak) != {"cycles", "duration_s", "ttft_p95_s", "token_gap_p95_s", "max_idle_gap_s"}:
        raise ValueError("unknown soak setting")
    if type(soak["cycles"]) is not int or soak["cycles"] < 1:
        raise ValueError("soak cycles must be positive")
    for name in set(soak) - {"cycles"}:
        if type(soak[name]) not in (int, float) or not math.isfinite(soak[name]) or soak[name] <= 0:
            raise ValueError("finite positive soak setting required")
    if soak["duration_s"] < 3600:
        raise ValueError("acceptance requires at least one hour of active soak observations")
    cfg["soak"] = soak
    cfg["timeouts"] = {"numeric_s": 14400, "gpu_gate_s": 3600,
        "benchmark_s": 14400, "job_s": 600, "soak_s": 21600,
        "lifecycle_s": 14400, **cfg.get("timeouts", {})}
    if set(cfg["timeouts"]) != {"numeric_s", "gpu_gate_s", "benchmark_s", "job_s", "soak_s", "lifecycle_s"}:
        raise ValueError("unknown campaign timeout")
    for value in cfg["timeouts"].values():
        if type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= 86400:
            raise ValueError("timeouts must be finite seconds in 1..86400")
    return cfg


def plan(config, out):
    """Freeze CPU-only intentions; no model reads, key creation or GPU inventory."""
    cfg = validate_config(config)
    value = {"schema": "flyai-v4-5090-campaign-plan/1", "config": cfg,
        "matrix": json.loads(json.dumps(MATRIX)), "modes": list(MODES), "source": R._source_identity(), "kit": kit_identity(),
        "performance_target_tok_s": 40 if len(cfg["ring"]["stages"]) == 4 else 30,
        "numeric_input_tokens": 4104, "numeric_max_tokens": 8192,
        "lifecycle": {"cycles": 2, "ngram_k": 4, "ngram_order": 3, "cancel_after": 2},
        "scope": "same-host ring; real GPU/numerical/signed evidence pending",
        "cold_scope": "fresh model/process; existing disk JIT cache is retained",
        "hardware_verified": False, "github_push_allowed": False}
    value["sha256"] = digest(value)
    write(out, value)
    return value


def require_plan(value, *, source=True):
    if (value.get("schema") != "flyai-v4-5090-campaign-plan/1" or
            value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"})):
        raise ValueError("campaign plan hash/schema differs")
    cfg = validate_config(value["config"], preserve_paths=True)
    if (cfg != value["config"] or value["matrix"] != MATRIX or value["modes"] != list(MODES)
            or value["numeric_input_tokens"] != 4104 or value["numeric_max_tokens"] != 8192
            or value["lifecycle"] != {"cycles": 2, "ngram_k": 4, "ngram_order": 3, "cancel_after": 2}
            or value["performance_target_tok_s"] != (40 if len(cfg["ring"]["stages"]) == 4 else 30)):
        raise ValueError("campaign contract was relaxed or changed")
    if source and (value["source"] != R._source_identity() or value["kit"] != kit_identity()):
        raise ValueError("source or acceptance helpers changed; create a new plan/run")
    return cfg


class Runner:
    """All test commands have persistent logs and process-birth owned cleanup."""
    def __init__(self, directory, python):
        from shard.managed_launch import ManagedLauncher
        self.directory, self.python = Path(directory), python
        self.private = self.directory / "private"
        self.private.mkdir(parents=True, exist_ok=True)
        self.manager = ManagedLauncher(self.private / "commands")
        self.records = []

    def command(self, name, argv, timeout, environment=None, *, allowed=(0,)):
        from shard.managed_launch import ManagedLauncher, _TOKEN_ENV
        ManagedLauncher._name(name)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("finite positive command timeout required")
        environment = dict(environment or {})
        if _TOKEN_ENV in environment:
            raise ValueError("command cannot override managed process ownership")
        completion = self.private / (name + ".completion.json")
        request = self.private / (name + ".request.json")
        if request.exists() or completion.exists() or any(row["name"] == name for row in self.records):
            raise ValueError("use a unique command name and fresh completion evidence")
        nonce = secrets.token_hex(32)
        payload = {"schema": "flyai-v4-acceptance-command-request/1", "name": name, "nonce": nonce,
                   "argv": [str(x) for x in argv], "environment": environment,
                   "completion": str(completion.resolve())}
        write(request, payload)
        record = {"name": name, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "timeout_s": timeout, "request_nonce": nonce, "request_sha256": digest(payload),
                  "exit_code": None, "cleanup_acknowledged": False}
        self.records.append(record)
        output = self.directory / "logs" / (name + ".log")
        output.parent.mkdir(parents=True, exist_ok=True)
        logfile = self.manager.directory / (name + ".log")
        primary_error = None
        try:
            started = self.manager.start(name, [self.python, str(Path(__file__).resolve()), "_worker", "--request", str(request)], cwd=str(ROOT))
            if started.get("reused"):
                raise RuntimeError("command reused a prior managed process")
            deadline = time.monotonic() + timeout
            while True:
                status = self.manager.status(name)
                if logfile.exists() and logfile.stat().st_size > MAX_LOG_BYTES:
                    raise RuntimeError(name + " exceeded bounded log budget")
                if not status["running"]:
                    done = read(completion)
                    if (done.get("schema") != "flyai-v4-acceptance-command-completion/1"
                            or done.get("name") != name or done.get("nonce") != nonce
                            or done.get("request_sha256") != record["request_sha256"]
                            or type(done.get("exit_code")) is not int or done.get("error")):
                        raise RuntimeError("command completion is absent, stale or differs from this request")
                    record["exit_code"] = done["exit_code"]
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(name + " exceeded owned process timeout")
                time.sleep(.1)
        except BaseException as error:
            primary_error = error
            record["failure"] = {"class": type(error).__name__, "message": str(error)}
        finally:
            # Cleanup is attempted even when start, status or log inspection fails.
            # Keep the launch token intact through both subprocess layers so
            # ManagedLauncher can discover compiler and diagnostic descendants.
            try:
                stopped = self.manager.stop(name)
                record["cleanup_acknowledged"] = stopped.get("running") is False
            except BaseException as error:
                record["cleanup_failure"] = {"class": type(error).__name__, "message": str(error)}
                if primary_error is None:
                    primary_error = error
            try:
                if logfile.is_file():
                    if logfile.stat().st_size > MAX_LOG_BYTES:
                        raise RuntimeError(name + " exceeded bounded log budget")
                    output.write_bytes(logfile.read_bytes())
            except BaseException as error:
                record["log_failure"] = {"class": type(error).__name__, "message": str(error)}
                if primary_error is None:
                    primary_error = error
            record["log"] = output.relative_to(self.directory).as_posix()
            write(self.directory / "commands.json", self.records)
        if primary_error is not None:
            raise primary_error
        if record["exit_code"] not in allowed or not record["cleanup_acknowledged"]:
            raise RuntimeError(name + " failed; inspect " + str(output))
        return record


def _worker(path):
    request = read(path)
    from shard.managed_launch import ManagedLauncher, _TOKEN_ENV
    ManagedLauncher._name(request["name"])
    completion = Path(request["completion"])
    if (request.get("schema") != "flyai-v4-acceptance-command-request/1"
            or not isinstance(request.get("nonce"), str) or len(request["nonce"]) != 64
            or completion.resolve() != Path(path).resolve().with_name(request["name"] + ".completion.json")
            or completion.exists() or _TOKEN_ENV in request["environment"]):
        raise ValueError("worker requires a fresh completion path bound to its named request")
    env = {k: v for k, v in os.environ.items() if not k.startswith("V4_")
           and k not in {"SHARD_STAGE_LEASE_CONFIG", "SHARD_NODE_KEY", "SHARD_COORDINATOR_KEY"}}
    env.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    env.update(request["environment"])
    done = {"schema": "flyai-v4-acceptance-command-completion/1", "name": request["name"],
            "nonce": request["nonce"], "request_sha256": digest(request), "exit_code": None}
    try:
        done["exit_code"] = subprocess.run(request["argv"], cwd=str(ROOT), env=env, check=False).returncode
    except BaseException as error:
        done["error"] = {"class": type(error).__name__, "message": str(error)}
        raise
    finally:
        R._write(completion, done, exclusive=True)
    return done["exit_code"]


def freeze_input(model_dir, tokens, out):
    """CPU tokenizer only, exact benchmark rendering and deterministic padding."""
    from phase0 import v4_benchmark as B
    from engines.deepseek_v4.v4_tokenizer import load_v4_tokenizer
    sys.path.insert(0, str(ROOT / "vendor/deepseek_v4_ref/encoding"))
    from encoding_dsv4 import encode_messages
    from safetensors.torch import save_file
    import torch
    tokenizer = load_v4_tokenizer(model_dir)
    render = lambda text: encode_messages([{"role": "user", "content": text}], "chat")
    prompts = B.freeze_prompts(lambda text: tokenizer.encode(text, add_special_tokens=False), render, tokens)
    ids = prompts[0]["token_ids"]
    save_file({"token_ids": torch.tensor([ids], dtype=torch.int64, device="cpu")}, str(out))
    write(str(out) + ".json", {"schema": "flyai-v4-acceptance-frozen-ids/1", "sha256": file_hash(out),
        "tokens": len(ids), "prompt": prompts[0], "scope": "teacher-forced numeric input; not a speed workload"})


def profile_errors(report, plan):
    """Actual signed receipts must show capture at every owned layer, every job."""
    from engines.deepseek_v4.v4_execution_profiles import require_profile_warmup
    expected = {row["node_id"]: "resident-whole" for row in plan["stages"]}
    rows = report.get("samples", report.get("cases", []))
    errors = []
    for index, row in enumerate(rows):
        if row.get("action") in ("cancel", "reconnect"):
            continue
        try:
            require_profile_warmup(row.get("receipts", []), expected)
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f"job {index}: {error}")
    if not rows:
        errors.append("signed capture observations absent")
    return errors


def _bench_prepare(runner, cfg, ring_dir, case_dir, size, mode):
    python = cfg["ring"]["python"]
    params = MATRIX[size]
    output = case_dir / "protocol.json"
    name = ("service" if case_dir.name == "service" else size) + "-" + mode + "-freeze"
    runner.command(name, [python, ROOT / "phase0/v4_benchmark.py", "prepare",
        "--dir", cfg["ring"]["model_dir"], "--hardware", ring_dir / "hardware.json", "--isolation", "none",
        "--env-file", ring_dir / "environment.json", "--network", ring_dir.parent / "network.json",
        "--prompt-tokens", params["prompt_tokens"],
        "--max-new", params["max_new"], "--mode", mode, "--out", output], 1800,
        {"CUDA_VISIBLE_DEVICES": ""})
    return output


def coordinator_environment(state):
    env = R.stage_environment(state)
    # Benchmark applies the frozen public environment before importing v4_pipe.
    # V4_DEV is stage-local and is deliberately absent from that public contract.
    env.pop("V4_DEV", None)
    return env


def run(plan_path, run_dir, confirmed, *, execute=False):
    frozen = read(plan_path)
    cfg = require_plan(frozen)
    if not execute:
        return {"status": "planned_only", "hardware_verified": False, "github_push_allowed": False}
    assigned = [row["gpu_index"] for row in cfg["ring"]["stages"]]
    required = set(assigned + [cfg["numeric_gpu_index"], cfg["gpu_gate_index"]])
    if set(confirmed) != required or len(confirmed) != len(required):
        raise ValueError("--confirm-gpus must name exactly the exclusively allocated campaign GPUs")
    directory = Path(run_dir).resolve()
    if directory == ROOT or ROOT in directory.parents or directory.exists():
        raise ValueError("fresh evidence directory outside checkout required; never overwrite/resume a run")
    directory.mkdir(parents=True, mode=0o700)
    write(directory / "campaign-plan.json", frozen)
    runner = Runner(directory, cfg["ring"]["python"])
    ring_dir = directory / "ring"
    index = {"schema": "flyai-v4-5090-campaign-index/1", "plan_sha256": frozen["sha256"],
        "cases": [], "complete": False, "owned_cleanup_acknowledged": False,
        "hardware_attested": False, "github_push_performed": False}
    write(directory / "index.json", index)
    prepared = False
    error = None
    try:
        # Read-only inventory includes all reserved devices. Compute is sequential.
        inventory = R.gpu_inventory()
        by_index = {row["gpu_index"]: row for row in inventory["gpus"]}
        if not required <= set(by_index):
            raise ValueError("reserved GPU ordinal missing from actual inventory")
        inventory["gpus"] = [by_index[i] for i in sorted(required)]
        write(directory / "allocated-gpus.json", inventory)
        if any("5090" not in row["gpu_name"] or row["compute_process_present"] or row["utilization_percent"] != 0
               for row in inventory["gpus"]):
            raise ValueError("all reserved devices must be observed idle RTX5090s before acceptance")
        # Artifact verification/explicit metadata creation precedes numeric input
        # preparation, and is CPU-only. Both paths must use the same checkpoint.
        R.prepare(cfg["ring"], ring_dir)
        prepared = True
        python = cfg["ring"]["python"]
        gate_env = {"CUDA_VISIBLE_DEVICES": by_index[cfg["gpu_gate_index"]]["gpu_uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        runner.command("cuda-gates", [python, ROOT / "phase0/v4_release.py", "gpu-check",
            "--junit", directory / "gpu.junit.xml", "--out", directory / "gpu-gate.json",
            "--timeout-s", cfg["timeouts"]["gpu_gate_s"]], cfg["timeouts"]["gpu_gate_s"] + 60, gate_env)
        from phase0 import v4_release as release
        if release.evaluate_gpu_gate(read(directory / "gpu-gate.json"))["status"] != "passed":
            raise RuntimeError("real CUDA gates did not pass")
        inputs = directory / "numeric-input.safetensors"
        freeze_input(cfg["ring"]["model_dir"], frozen["numeric_input_tokens"], inputs)
        numeric_dir = directory / "numeric"
        # Dedicated device remapped to cuda:0 in every numeric child.
        numeric_env = {"CUDA_VISIBLE_DEVICES": by_index[cfg["numeric_gpu_index"]]["gpu_uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        runner.command("numeric-prepare", [python, ROOT / "tools/v4_acceptance_numeric.py", "prepare",
            "--checkpoint", cfg["ring"]["model_dir"], "--input", inputs,
            "--device", "cuda:0", "--max-tokens", 8192, "--out-dir", numeric_dir], 1800, numeric_env)
        runner.command("numeric-run", [python, ROOT / "tools/v4_acceptance_numeric.py", "run",
            "--plan", numeric_dir / "numeric-plan.json", "--execute"], cfg["timeouts"]["numeric_s"], numeric_env)
        if read(numeric_dir / "numeric-result.json").get("status") != "passed_owned_stage_numeric":
            raise RuntimeError("real model whole rollback/capture/bucket coverage did not pass")
        from phase0 import v4_benchmark as B
        route = {"host_id": inventory["host_id"], "transport": "tcp_loopback",
                 "gpus": [by_index[i]["gpu_uuid"] for i in assigned],
                 "stages": cfg["ring"]["stages"], "return": "direct strict tail/coordinator socket"}
        write(directory / "route.json", route)
        write(directory / "network.json", {"schema": B.NETWORK_SCHEMA, "transport": "tcp_loopback",
            "route_identity_sha256": digest(route), "latency_semantics": "no_latency_claim",
            "measurement_method": "fixed observed host/GPU UUIDs and loopback endpoints; no WAN/RTT measurement claim"})
        R.measure(ring_dir, assigned)
        measurements = list(R.load_state(ring_dir)["measurement_paths"])
        short_results = {}
        for size in MATRIX:
            for mode in MODES:
                require_plan(frozen)
                reports = [read(path) for path in measurements]
                load_started = datetime.now(timezone.utc).isoformat()
                R.start(ring_dir, assigned, measurements=reports, cold_open=True)
                try:
                    case_dir = directory / "matrix" / size / mode
                    case_dir.mkdir(parents=True)
                    write(case_dir / "load-check.json", {"started_utc": load_started,
                        "completed_utc": datetime.now(timezone.utc).isoformat(),
                        "measurement_files": [Path(path).name for path in measurements]})
                    protocol_path = _bench_prepare(runner, cfg, ring_dir, case_dir, size, mode)
                    state = R.load_state(ring_dir)
                    protocol = read(protocol_path)
                    plan_obj = state["plan"]
                    head, tail = plan_obj["coordinator"]["head"], plan_obj["coordinator"]["tail"]
                    runner.command(size + "-" + mode + "-bench", [python, ROOT / "phase0/v4_benchmark.py", "run",
                        "--protocol", protocol_path, "--dir", cfg["ring"]["model_dir"], "--head", head, "--tail", tail,
                        "--timeout", cfg["timeouts"]["job_s"], "--connect-retry", 30,
                        "--deployment-plan", ring_dir / "plan.json", "--coordinator-key", state["key_paths"][-1],
                        "--fresh-ring", "--out", case_dir / "report.json"], cfg["timeouts"]["benchmark_s"],
                        coordinator_environment(state), allowed=(0, 2))
                    report = read(case_dir / "report.json")
                    verdict = B.evaluate_report(report, expected_protocol=protocol)
                    capture = profile_errors(report, plan_obj)
                    write(case_dir / "verification.json", {"benchmark": verdict, "capture_errors": capture})
                    if verdict["status"] not in ("passed", "valid_baseline") or capture:
                        raise RuntimeError(size + "/" + mode + " correctness/identity/capture failed")
                    item = {"size": size, "mode": mode, "protocol_sha256": protocol["sha256"],
                            "directory": case_dir.relative_to(directory).as_posix()}
                    index["cases"].append(item)
                    write(directory / "index.json", index)
                    if size == "short":
                        short_results[mode] = (verdict, protocol, item)
                    # Authenticated READY is a distinct post-benchmark check, never precedes cold request.
                    R.warmup(ring_dir, measurements=reports)
                    write(case_dir / "signed-warmup.json", read(ring_dir / "signed-warmup.json"))
                    write(case_dir / "plan.json", plan_obj)
                    write(case_dir / "loaded.json", read(ring_dir / "loaded.json"))
                finally:
                    R.stop(ring_dir)
                    copy_stage_logs(ring_dir, directory / "stage-logs" / (size + "-" + mode))
        # Below-target valid results still get reliability coverage; final verification
        # keeps performance acceptance false. No easy prompt or peak selection.
        best = max(MODES, key=lambda mode: short_results[mode][0]["complete_suite_median_warm_decode_tok_s"])
        index["selected_mode"] = best
        write(directory / "index.json", index)
        # New process has a new authenticated process_run_id; freeze a separate soak protocol.
        reports = [read(path) for path in measurements]
        R.start(ring_dir, assigned, measurements=reports)
        state = R.load_state(ring_dir)
        service_dir = directory / "service"
        service_dir.mkdir()
        service_protocol = _bench_prepare(runner, cfg, ring_dir, service_dir, "short", best)
        head, tail = state["plan"]["coordinator"]["head"], state["plan"]["coordinator"]["tail"]
        base = ["--protocol", service_protocol, "--dir", cfg["ring"]["model_dir"], "--head", head, "--tail", tail,
                "--timeout", cfg["timeouts"]["job_s"], "--connect-retry", 30,
                "--deployment-plan", ring_dir / "plan.json", "--coordinator-key", state["key_paths"][-1]]
        write(service_dir / "soak-contract.json", cfg["soak"])
        from phase0 import v4_soak as S
        life = S.lifecycle_contract(**frozen["lifecycle"])
        write(service_dir / "lifecycle-contract.json", life)
        runner.command("lifecycle", [python, ROOT / "phase0/v4_soak.py", "lifecycle", *base,
            "--cycles", 2, "--ngram-k", 4, "--ngram-order", 3, "--cancel-after", 2,
            "--out", service_dir / "lifecycle.json"], cfg["timeouts"]["lifecycle_s"], coordinator_environment(state))
        extra = [arg for key, value in cfg["soak"].items() for arg in ("--" + key.replace("_", "-"), value)]
        runner.command("soak", [python, ROOT / "phase0/v4_soak.py", "run", *base, *extra,
            "--out", service_dir / "soak.json"], cfg["timeouts"]["soak_s"], coordinator_environment(state))
        write(service_dir / "plan.json", state["plan"])
        index["complete"] = True
    except BaseException as exc:
        error = exc
        index["failure"] = {"class": type(exc).__name__, "message": str(exc)}
    finally:
        if prepared:
            try:
                stop = R.stop(ring_dir)
                index["owned_cleanup_acknowledged"] = stop["cleanup_acknowledged"]
                copy_stage_logs(ring_dir, directory / "stage-logs" / "final")
            except Exception as cleanup:
                index["cleanup_failure"] = type(cleanup).__name__ + ": " + str(cleanup)
        else:
            index["owned_cleanup_acknowledged"] = all(row["cleanup_acknowledged"] for row in runner.records)
        write(directory / "index.json", index)
        verdict = verify(directory, expected_plan=frozen)
        write(directory / "summary.json", verdict)
        archive_evidence(directory, directory.with_suffix(".evidence.tgz"))
    if error:
        raise RuntimeError("campaign stopped; partial evidence preserved in " + str(directory)) from error
    return verdict


def verify(directory, *, expected_plan=None):
    """Recompute signatures, tokens, measured timings and independent contracts."""
    directory = Path(directory).resolve()
    checks, errors = {}, []
    try:
        from phase0 import v4_benchmark as B, v4_release as G, v4_soak as S
        frozen = read(directory / "campaign-plan.json")
        cfg = require_plan(frozen)
        if expected_plan is not None and frozen != expected_plan:
            raise ValueError("independent campaign plan differs")
        index = read(directory / "index.json")
        if index.get("schema") != "flyai-v4-5090-campaign-index/1" or index["plan_sha256"] != frozen["sha256"]:
            raise ValueError("campaign index binding differs")
        if not index.get("complete") or index.get("failure"):
            errors.append("campaign incomplete/failed")
        if not index.get("owned_cleanup_acknowledged") or index.get("github_push_performed"):
            errors.append("owned cleanup absent or unauthorized push recorded")
        commands = read(directory / "commands.json")
        if not commands or any(not row.get("cleanup_acknowledged") or row.get("exit_code") not in (0, 2) for row in commands):
            errors.append("command execution/owned cleanup evidence incomplete")
        manifest_path = directory / "evidence-manifest.json"
        if manifest_path.exists():
            verify_evidence_files(directory, read(manifest_path))
        gate = read(directory / "gpu-gate.json")
        checks["cuda"] = G.evaluate_gpu_gate(gate)
        allocated = {row["gpu_index"]: row for row in read(directory / "allocated-gpus.json")["gpus"]}
        if gate.get("hardware", {}).get("uuid", "").lower() != allocated[cfg["gpu_gate_index"]]["gpu_uuid"].lower():
            errors.append("CUDA gate ran on a different reserved physical device")
        from tools import v4_acceptance_numeric as N
        numeric_plan = read(directory / "numeric/numeric-plan.json")
        numeric = read(directory / "numeric/numeric-result.json")
        checks["numeric"] = N.verify_result(numeric, numeric_plan, base_dir=directory / "numeric")
        if checks["numeric"].get("status") != "passed_owned_stage_numeric":
            errors.append("owned whole/eager rollback numeric evidence absent or invalid")
        expected = [(size, mode) for size in MATRIX for mode in MODES]
        if [(row["size"], row["mode"]) for row in index["cases"]] != expected:
            errors.append("eight independently frozen matrix cells missing/reordered")
        reports = {}
        gpu_rows = None
        previous_process_ids = set()
        network = read(directory / "network.json")
        if network["route_identity_sha256"] != digest(read(directory / "route.json")) or network["latency_semantics"] != "no_latency_claim":
            raise ValueError("same-host route contract differs")
        for row in index["cases"]:
            if row["directory"] != f"matrix/{row['size']}/{row['mode']}" or (row["size"], row["mode"]) not in expected:
                raise ValueError("unsafe/misbound matrix case directory")
            case_dir = directory / row["directory"]
            p, r, deployment = read(case_dir / "protocol.json"), read(case_dir / "report.json"), read(case_dir / "plan.json")
            if p["sha256"] != row["protocol_sha256"] or p["run"]["mode"] != row["mode"]:
                raise ValueError("mode/protocol binding differs")
            if any(p["run"][key] != value for key, value in MATRIX[row["size"]].items()):
                raise ValueError("benchmark length contract changed")
            if p["env"] != cfg["ring"]["environment"] or p["source"]["sha256"] != frozen["source"]["sha256"]:
                raise ValueError("benchmark recipe/source differs from campaign")
            if p.get("network_comparison") != network:
                raise ValueError("mode comparison network route contract differs")
            if p["isolation"] != "none" or len({n["host_id"] for n in p["hardware"]}) != 1:
                raise ValueError("same-host topology evidence differs")
            assignments = [(n["node_id"], n["layer_start"], n["layer_end"]) for n in p["hardware"]]
            if assignments != [(n["node_id"], n["lo"], n["hi"]) for n in cfg["ring"]["stages"]]:
                raise ValueError("per-GPU assigned layers changed")
            invariant = [{k: v for k, v in n.items() if k != "process_run_id"} for n in p["hardware"]]
            if gpu_rows is not None and invariant != gpu_rows:
                raise ValueError("hardware/runtime dependency identities changed between matrix cells")
            gpu_rows = invariant
            result = B.evaluate_report(r, expected_protocol=p)
            loaded = read(case_dir / "loaded.json")
            from tools import v4_acceptance_bindings as bindings
            errors.extend(bindings.deployment_binding_errors(deployment, p, cfg["ring"]))
            errors.extend(bindings.cold_binding_errors(loaded, p, previous_process_ids))
            previous_process_ids.update(n["process_run_id"] for n in p["hardware"])
            errors.extend(bindings.warmup_binding_errors(read(case_dir / "signed-warmup.json"), p, deployment))
            errors.extend(bindings.numeric_binding_errors(numeric_plan, frozen, p["checkpoint"], directory / "numeric-input.safetensors"))
            load_check = read(case_dir / "load-check.json")
            start_at = datetime.fromisoformat(load_check["started_utc"])
            if start_at.tzinfo is None or datetime.fromisoformat(load_check["completed_utc"]) < start_at:
                raise ValueError("resource acceptance clock interval invalid")
            inventory = read(directory / "allocated-gpus.json")
            resource_state = {"config": cfg["ring"], "gpu_inventory": inventory, "source": frozen["source"],
                "cohort": deployment["model_cohort"], "key_paths": [""] * (len(cfg["ring"]["stages"]) + 1)}
            for i, node in enumerate(p["hardware"]):
                measurement = read(directory / "ring" / f"resource-stage-{i}.json")
                calibration = R.validate_measurement(measurement, resource_state, i, now=start_at)
                if calibration["runtime_config_sha256"] != node["runtime_config_sha256"]:
                    errors.append("actual stage config differs from accepted resource calibration")
            capture = profile_errors(r, deployment)
            result["capture_errors"] = capture
            label = row["size"] + "/" + row["mode"]
            checks[label] = result
            if result["status"] not in ("passed", "valid_baseline") or capture:
                errors.append(label + " failed identity/greedy parity/actual captures")
            reports[(row["size"], row["mode"])] = r
            if gate.get("source", {}).get("sha256") != p["source"]["sha256"] or any(
                n.get("gpu_name") != gate.get("hardware", {}).get("name") or any(
                    n.get(k) != gate.get("hardware", {}).get(k) for k in ("torch", "cuda", "tilelang", "driver")) for n in p["hardware"]):
                errors.append("CUDA gate source/GPU/dependencies differ from actual ring")
        for size in MATRIX:
            if (size, "greedy") not in reports:
                continue
            for mode in MODES[1:]:
                if (size, mode) not in reports:
                    continue
                compared = B.compare_reports(reports[(size, "greedy")], reports[(size, mode)], vary="mode")
                checks[size + "/greedy-vs-" + mode] = compared
                if compared["status"] not in ("verified_comparison", "verified_target_pass"):
                    errors.append(size + " controlled mode comparison failed")
        selected = index.get("selected_mode")
        if selected not in MODES or checks.get("short/" + str(selected), {}).get("status") != "passed":
            errors.append("no short-workload mode met the declared 4/6 speed target")
        short_modes = {mode: checks.get("short/" + mode, {}) for mode in MODES}
        if all(row.get("valid_evidence") for row in short_modes.values()):
            if selected != max(MODES, key=lambda mode: short_modes[mode]["complete_suite_median_warm_decode_tok_s"]):
                errors.append("service selection differs from predeclared complete-suite median rule")
        service = directory / "service"
        protocol = read(service / "protocol.json")
        if protocol["run"]["mode"] != selected or protocol["env"] != cfg["ring"]["environment"]:
            errors.append("service mode/recipe differs")
        if protocol["source"]["sha256"] != frozen["source"]["sha256"] or protocol["prompts"] != reports[("short", "greedy")]["protocol"]["prompts"]:
            errors.append("service frozen source/workload differs")
        service_rows = [{k: v for k, v in n.items() if k != "process_run_id"} for n in protocol["hardware"]]
        if service_rows != gpu_rows or protocol["checkpoint"] != reports[("short", "greedy")]["protocol"]["checkpoint"]:
            errors.append("service ring/checkpoint identity differs from matrix")
        from tools import v4_acceptance_bindings as bindings
        errors.extend(bindings.deployment_binding_errors(read(service / "plan.json"), protocol, cfg["ring"]))
        if read(service / "soak-contract.json") != cfg["soak"]:
            errors.append("independent soak contract differs")
        life_contract = S.lifecycle_contract(**frozen["lifecycle"])
        if read(service / "lifecycle-contract.json") != life_contract:
            errors.append("independent lifecycle contract differs")
        for label, evaluator, contract in (("soak", S.evaluate_soak, cfg["soak"]),
                                             ("lifecycle", S.evaluate_lifecycle, life_contract)):
            report = read(service / (label + ".json"))
            checks[label] = evaluator(report, protocol, contract)
            capture = profile_errors(report, read(service / "plan.json"))
            checks[label]["capture_errors"] = capture
            if checks[label]["status"] != "passed" or capture:
                errors.append(label + " incomplete or failed")
        if checks["cuda"]["status"] != "passed":
            errors.append("real CUDA gates failed/unverified")
    except Exception as exc:
        errors.append(type(exc).__name__ + ": " + str(exc))
    return {"schema": "flyai-v4-5090-campaign-verdict/1", "status": "failed" if errors else "passed",
            "errors": errors, "checks": checks, "hardware_attested": False,
            "scope": "one same-host 4/6-card ring; not official combined four+six release",
            "github_push_allowed": not errors, "github_push_performed": False}


def archive_evidence(directory, out):
    """Public raw evidence only; private receipt signing keys are never exported."""
    directory, out = Path(directory).resolve(), Path(out).resolve()
    if out == directory or directory in out.parents or out.exists():
        raise ValueError("new archive outside evidence directory required")
    files = []
    total = 0
    for path in sorted(directory.rglob("*")):
        rel = path.relative_to(directory)
        if "private" in rel.parts or "processes" in rel.parts or rel.as_posix() == "evidence-manifest.json":
            continue
        if path.is_symlink():
            raise ValueError("evidence symlinks are not portable")
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_LOG_BYTES:
            raise ValueError("artifact exceeds bounded archive file budget")
        total += size
        if total > 2 * 1024**3:
            raise ValueError("evidence exceeds 2GiB archive budget; preserve directory and inspect")
        files.append({"name": rel.as_posix(), "sha256": file_hash(path), "size": size})
    manifest = {"schema": "flyai-v4-5090-evidence-manifest/1", "files": files,
        "excluded": "private keys/requests/owned-launch databases; weights are never copied"}
    write(directory / "evidence-manifest.json", manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "x:gz") as archive:
        for row in files:
            archive.add(directory / row["name"], arcname="evidence/" + row["name"], recursive=False)
        archive.add(directory / "evidence-manifest.json", arcname="evidence/evidence-manifest.json", recursive=False)
    return {"archive": str(out), "sha256": file_hash(out), "files": len(files), "bytes": total}


def verify_evidence_files(directory, manifest):
    directory = Path(directory).resolve()
    if manifest.get("schema") != "flyai-v4-5090-evidence-manifest/1":
        raise ValueError("invalid evidence file manifest")
    names = set()
    for row in manifest["files"]:
        name = row["name"]
        parts = name.split("/")
        if (not name or name.startswith("/") or "\\" in name or ":" in name or
                any(p in ("", ".", "..", "private", "processes") for p in parts) or name in names):
            raise ValueError("unsafe/duplicate evidence manifest path")
        names.add(name)
        path = directory.joinpath(*parts)
        if any(directory.joinpath(*parts[:i]).is_symlink() for i in range(1, len(parts) + 1)):
            raise ValueError("evidence path traverses a symlink")
        if not path.is_file() or path.stat().st_size != row["size"] or file_hash(path) != row["sha256"]:
            raise ValueError("evidence raw file hash/size differs: " + name)
    return {"files_verified": len(names)}


def copy_stage_logs(ring_dir, destination):
    """Export only bounded logs, never private worker requests or key files."""
    log_dir = R._manager(ring_dir).directory
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(log_dir.glob("*.log")):
        if path.is_symlink() or path.stat().st_size > MAX_LOG_BYTES:
            raise ValueError("stage log is unsafe or exceeds bounded budget")
        (destination / path.name).write_bytes(path.read_bytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan"); p.add_argument("--config", required=True); p.add_argument("--out", required=True)
    r = commands.add_parser("run"); r.add_argument("--plan", required=True); r.add_argument("--run-dir", required=True)
    r.add_argument("--execute", action="store_true"); r.add_argument("--confirm-gpus", default="")
    v = commands.add_parser("verify"); v.add_argument("--run-dir", required=True); v.add_argument("--plan", required=True)
    a = commands.add_parser("archive"); a.add_argument("--run-dir", required=True); a.add_argument("--out", required=True)
    w = commands.add_parser("_worker", help=argparse.SUPPRESS); w.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    if args.command == "_worker":
        return _worker(args.request)
    if args.command == "plan":
        value = plan(read(args.config), args.out)
        result = {"status": "planned_only", "sha256": value["sha256"], "hardware_verified": False}
    elif args.command == "run":
        indices = [int(i) for i in args.confirm_gpus.split(",") if i]
        result = run(args.plan, args.run_dir, indices, execute=args.execute)
    elif args.command == "verify":
        result = verify(args.run_dir, expected_plan=read(args.plan))
    else:
        result = archive_evidence(args.run_dir, args.out)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 2 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

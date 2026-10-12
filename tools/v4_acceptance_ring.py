"""Same-host V4 acceptance ring, with explicit GPU assignments and owned cleanup.

``prepare`` is the default, CPU-only operation: it hashes local files and creates
an isolated run directory and receipt-format keys. It does not launch a process.
Only explicit ``observe``, ``measure`` and ``start`` commands inspect GPUs or run
hardware work. This is a manual strict experiment, without production leases.
No SSH, package installation, downloads, GPU selection or model-size averaging.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "flyai-v4-acceptance-ring-config/1"
SCOPE = "manual_strict_experiment_without_resource_leases"
UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
MAX_OWNED_LOG_BYTES = 256 * 1024**2


def _digest(value):
    from phase0.v4_benchmark import digest
    return digest(value)


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular local JSON file required: " + str(path))
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path, value, *, exclusive=False):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("refusing symlink output")
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target = path if exclusive else path.with_name(path.name + "." + secrets.token_hex(8) + ".tmp")
    try:
        with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if not exclusive:
            os.replace(target, path)
    finally:
        if not exclusive and target.exists():
            target.unlink()


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an explicit integer")
    return value


def canonical_environment(overrides=None):
    """Freeze registered literal defaults without importing any model or Torch.

    Runtime observations are still checked for exact equality. A new dynamic
    default cannot silently extend the benchmark recipe after preparation.
    """
    from engines.deepseek_v4.v4_execution_profiles import (
        baseline_environment, expanded_environment, _known_environment_names,
    )
    overrides = dict(overrides or {})
    names = _known_environment_names() - {"V4_DIR", "V4_DEV"}
    if any(k not in names or not isinstance(v, str) or "\x00" in v
           for k, v in overrides.items()):
        raise ValueError("environment must contain registered public V4 string settings")
    defaults = {}
    for path in sorted((ROOT / "engines" / "deepseek_v4").glob("v4_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute)
                    or node.func.attr != "get" or not isinstance(node.func.value, ast.Attribute)
                    or node.func.value.attr != "environ" or len(node.args) < 2
                    or not isinstance(node.args[0], ast.Constant)):
                continue
            key = node.args[0].value
            if key not in names or not isinstance(node.args[1], ast.Constant):
                continue
            value = node.args[1].value
            if isinstance(value, (str, int, float)):
                defaults.setdefault(key, str(value))
    # Profile expansion must see only explicit overrides, because its protected
    # settings can differ from the generic baseline (e.g. resident-island).
    recipe = expanded_environment(overrides)
    env = {**defaults, **baseline_environment(), **recipe}
    if env.get("V4_KERNELS") != "tilelang" or env.get("V4_MAX_BATCH") != "1":
        raise ValueError("acceptance ring requires tilelang kernels and batch one")
    if int(env["V4_MAX_SEQ"]) < 64:
        raise ValueError("V4_MAX_SEQ must leave room for warmup and speculative margin")
    return dict(sorted(env.items()))


def _source_identity():
    """Same content catalogue as benchmark.source_identity, without git subprocesses."""
    from phase0.v4_benchmark import file_digest
    paths = set()
    for directory in ("engines/deepseek_v4", "vendor/deepseek_v4_ref", "shard"):
        paths.update((ROOT / directory).rglob("*.py"))
    for name in ("wire", "ngram_draft", "v4_benchmark", "v4_soak", "v4_release",
                 "v4_profile_report", "v4_ring_economics", "v4_round_calibration"):
        path = ROOT / "phase0" / (name + ".py")
        if path.is_file():
            paths.add(path)
    files = [{"name": p.relative_to(ROOT).as_posix(), "sha256": file_digest(p)} for p in sorted(paths)]
    return {"sha256": _digest(files), "files": files, "git_commit": None,
            "working_tree": {"known": False, "dirty": None, "status_sha256": None}}


def validate_config(config):
    config = json.loads(json.dumps(config, allow_nan=False))
    allowed = {"schema", "ring_id", "model_dir", "model_id", "python", "environment",
               "stages", "resource_probe", "timeouts", "catalogue_if_missing"}
    if config.get("schema") != SCHEMA or set(config) - allowed:
        raise ValueError(SCHEMA + " and supported config fields required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", config.get("ring_id", "")):
        raise ValueError("ring_id must be a safe identifier of at most 64 characters")
    if not isinstance(config.get("model_dir"), str) or not config["model_dir"]:
        raise ValueError("existing local native model_dir is required")
    config["model_dir"] = str(Path(config["model_dir"]).resolve())
    config.setdefault("python", sys.executable)
    if not isinstance(config["python"], str) or not config["python"]:
        raise ValueError("explicit Python executable required")
    rows = config.get("stages")
    if not isinstance(rows, list) or len(rows) not in (4, 6, 8):
        raise ValueError("explicit 4, 6 or 8 stage assignments required; 8 is diagnostic only")
    cursor, indices, ports, nodes = 0, set(), set(), set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) - {"gpu_index", "lo", "hi", "port", "node_id"}:
            raise ValueError("stages require explicit gpu_index, lo, hi and port")
        for name in ("gpu_index", "lo", "hi", "port"):
            _integer(row.get(name), name, 1 if name in ("hi", "port") else 0)
        row.setdefault("node_id", f"local-gpu-{row['gpu_index']}")
        if (row["lo"] != cursor or not cursor < row["hi"] <= 43 or row["port"] > 65535
                or row["gpu_index"] in indices or row["port"] in ports
                or not isinstance(row["node_id"], str) or not row["node_id"] or row["node_id"] in nodes):
            raise ValueError("distinct GPU indices/ports/node IDs and contiguous explicit [lo,hi) spans required")
        cursor = row["hi"]
        indices.add(row["gpu_index"]); ports.add(row["port"]); nodes.add(row["node_id"])
    if cursor != 43 or rows[-1]["lo"] > 40:
        raise ValueError("assign all 43 layers exactly once; tail must include layers 40..42")
    config["environment"] = canonical_environment(config.get("environment"))
    probe = {"prefill_tokens": 16, "decode_tokens": 8, "max_age_s": 3600,
             "headroom_bytes": 2 * 1024**3, **config.get("resource_probe", {})}
    if set(probe) != {"prefill_tokens", "decode_tokens", "max_age_s", "headroom_bytes"}:
        raise ValueError("unknown resource probe setting")
    for name in ("prefill_tokens", "decode_tokens", "headroom_bytes"):
        _integer(probe[name], name, 1)
    _positive(probe["max_age_s"], "max_age_s")
    if probe["prefill_tokens"] + probe["decode_tokens"] + 16 > int(config["environment"]["V4_MAX_SEQ"]):
        raise ValueError("resource probe exceeds configured sequence capacity")
    config["resource_probe"] = probe
    config["timeouts"] = {"startup_s": 1800, "warmup_s": 600, "measure_s": 3600,
                          **config.get("timeouts", {})}
    if set(config["timeouts"]) != {"startup_s", "warmup_s", "measure_s"}:
        raise ValueError("unknown timeout")
    for name, value in config["timeouts"].items():
        _positive(value, name)
    if type(config.get("catalogue_if_missing", False)) is not bool:
        raise ValueError("catalogue_if_missing must be a boolean")
    return config


def prepare(config, run_dir):
    """Verify full local artifacts and create keys; never launch or probe CUDA."""
    from shard import weight_artifacts as W
    from shard.manifest import gen_key, pub_b64, save_key
    from phase0.v4_benchmark import checkpoint_inventory
    config = validate_config(config)
    directory = Path(run_dir)
    if directory.exists() or directory.is_symlink():
        raise ValueError("prepare requires a new isolated run directory")
    model = Path(config["model_dir"])
    present = [(model / filename).is_file() for filename in (W.GLOBAL_FILE, W.PACK_FILE)]
    if not all(present):
        if any(present) or not config.get("catalogue_if_missing", False):
            raise ValueError("full native catalogue and packing required; catalogue_if_missing permits explicit creation")
        catalog, pack = W.catalogue_directory(model, config.get("model_id", "deepseek-ai/DeepSeek-V4-Flash-0731"))
        W.catalog_model_cohort(catalog)  # Refuse non-native/tiny checkpoint before metadata writes.
        W.write_metadata(model, catalog, pack)
    full = W.verify_weight_pack(model, verify_files=True)
    cohort = W.catalog_model_cohort(full["catalog"])
    checkpoint = checkpoint_inventory(model)
    directory.mkdir(parents=True, mode=0o700)
    directory = directory.resolve()
    private = directory / "private"
    private.mkdir(mode=0o700)
    signers, keys = [], []
    for i in range(len(config["stages"]) + 1):
        key = gen_key()
        path = private / (f"stage-{i}.key" if i < len(config["stages"]) else "coordinator.key")
        save_key(key, str(path))
        keys.append(str(path)); signers.append(pub_b64(key))
    state = {"schema": "flyai-v4-acceptance-ring-state/1", "scope": SCOPE,
             "config": config, "config_sha256": _digest(config), "cohort": cohort,
             "checkpoint": checkpoint, "source": _source_identity(), "signer_pubkeys": signers,
             "key_paths": keys, "owned_names": [], "status": "PREPARED"}
    _write(directory / "private" / "state.json", state, exclusive=True)
    _write(directory / "environment.json", config["environment"], exclusive=True)
    public = {k: v for k, v in state.items() if k not in {"key_paths", "owned_names"}}
    public.update(runtime_ready=False, resource_leases=False, hardware_qualification=False,
                  gpu_mapping="unobserved; explicit assigned indices await later observation",
                  environment_sha256=_digest(config["environment"]))
    _write(directory / "prepared.json", public, exclusive=True)
    return public


def load_state(run_dir):
    directory = Path(run_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("existing isolated run directory required")
    state = _read(directory / "private" / "state.json")
    if state.get("schema") != "flyai-v4-acceptance-ring-state/1" or state.get("scope") != SCOPE:
        raise ValueError("invalid acceptance state")
    if _digest(state["config"]) != state["config_sha256"]:
        raise ValueError("prepared config changed")
    prefix = state["config"]["ring_id"] + "."
    if any(not isinstance(name, str) or not name.startswith(prefix) for name in state["owned_names"]):
        raise ValueError("invalid owned process names")
    return state


def _save_state(run_dir, state):
    _write(Path(run_dir) / "private" / "state.json", state)


def _confirmed(state, indices):
    expected = [row["gpu_index"] for row in state["config"]["stages"]]
    if not isinstance(indices, (list, tuple)) or any(type(i) is not int for i in indices) or list(indices) != expected:
        raise ValueError("confirm assigned GPU indices in exact stage order before hardware work")


def gpu_inventory(*, executor=subprocess.run):
    """Explicit later read-only nvidia-smi observation; no CUDA context creation."""
    def query(fields, kind):
        argv = ["nvidia-smi", "--query-" + kind + "=" + fields, "--format=csv,noheader,nounits"]
        result = executor(argv, capture_output=True, text=True, timeout=30, check=True)
        return list(csv.reader(io.StringIO(result.stdout)))
    rows = query("index,uuid,name,driver_version,memory.total,memory.free,utilization.gpu", "gpu")
    processes = query("gpu_uuid,pid", "compute-apps")
    active = {row[0].strip() for row in processes if len(row) == 2 and row[1].strip().isdigit()}
    values = []
    for row in rows:
        if len(row) != 7:
            raise ValueError("incomplete nvidia-smi GPU observation")
        index, uuid, name, driver, total, free, utilization = (item.strip() for item in row)
        if not UUID.fullmatch(uuid):
            raise ValueError("complete observed NVIDIA GPU UUID required")
        values.append({"gpu_index": int(index), "gpu_uuid": uuid, "gpu_name": name,
                       "driver": driver, "vram_bytes": int(total) * 1024**2,
                       "free_bytes": int(free) * 1024**2, "utilization_percent": int(utilization),
                       "compute_process_present": uuid in active})
    if not values or len({row["gpu_index"] for row in values}) != len(values):
        raise ValueError("complete unique GPU inventory required")
    return {"schema": "flyai-v4-same-host-gpu-inventory/1", "observed_at": datetime.now(timezone.utc).isoformat(),
            "host_id": socket.gethostname(), "gpus": values, "hardware_attested": False,
            "idle_scope": "zero sampled GPU utilization and no nvidia-smi compute process on each assigned GPU"}


def _assigned_idle(state, inventory):
    by_index = {row["gpu_index"]: row for row in inventory["gpus"]}
    selected = []
    for row in state["config"]["stages"]:
        gpu = by_index.get(row["gpu_index"])
        if gpu is None or gpu["compute_process_present"] or gpu["utilization_percent"] != 0:
            raise ValueError("every assigned GPU must be observed idle before hardware work")
        if not UUID.fullmatch(gpu["gpu_uuid"]) or not re.search(r"\bRTX\s+5090\b", gpu["gpu_name"], re.I):
            raise ValueError("assigned hardware must be observed RTX 5090 GPUs")
        if not 30 * 1024**3 <= gpu["vram_bytes"] <= 34 * 1024**3:
            raise ValueError("assigned GPU must have observed 32 GB class VRAM")
        selected.append(gpu)
    if len({row["gpu_uuid"].lower() for row in selected}) != len(selected):
        raise ValueError("assigned GPUs must have distinct observed UUIDs")
    return selected


def observe(run_dir, confirmed_gpu_indices, *, inventory_reader=gpu_inventory):
    state = load_state(run_dir)
    _confirmed(state, confirmed_gpu_indices)
    inventory = inventory_reader()
    _assigned_idle(state, inventory)
    prior = state.get("gpu_inventory")
    if prior:
        old = {row["gpu_index"]: row["gpu_uuid"] for row in prior["gpus"]}
        if any(old.get(row["gpu_index"], "").lower() != row["gpu_uuid"].lower()
               for row in _assigned_idle(state, inventory)):
            raise ValueError("GPU ordinal-to-UUID mapping changed since preparation observation")
    state["gpu_inventory"] = inventory
    _save_state(run_dir, state)
    _write(Path(run_dir) / "gpu-inventory.json", inventory)
    return inventory


def make_plan(state):
    from shard.pipeline_plan import build_plan, validate_plan
    from shard.offers import model_cohort_id
    config = state["config"]
    gpus = _assigned_idle(state, state["gpu_inventory"])
    rows = config["stages"]
    endpoints = [f"127.0.0.1:{row['port']}" for row in rows]
    plan = build_plan({"n_layers": 43}, ring_id=config["ring_id"], cohort_id=model_cohort_id(state["cohort"]),
                      endpoints=endpoints, split=[row["hi"] - row["lo"] for row in rows],
                      node_ids=[row["node_id"] for row in rows], gpu_uuids=[row["gpu_uuid"] for row in gpus],
                      model_cohort=state["cohort"], head=endpoints[0], tail=endpoints[-1])
    for i, row in enumerate(plan["stages"]):
        row.update(signer_pubkey=state["signer_pubkeys"][i], context_limit=int(config["environment"]["V4_MAX_SEQ"]))
    plan["coordinator"].update(node_id=config["ring_id"] + "-coordinator", signer_pubkey=state["signer_pubkeys"][-1])
    plan["execution"] = {"max_context": int(config["environment"]["V4_MAX_SEQ"])}
    if len(set(state["signer_pubkeys"])) != len(rows) + 1:
        raise ValueError("distinct per-GPU and coordinator receipt-format signer keys required")
    return validate_plan(plan)


def stage_environment(state, index=None):
    env = dict(state["config"]["environment"])
    env.update(V4_DIR=state["config"]["model_dir"], V4_DEV="cuda:0", SHARD_RECEIPTS="1",
               CUDA_DEVICE_ORDER="PCI_BUS_ID")
    if index is None:
        env.update(CUDA_VISIBLE_DEVICES="", SHARD_COORDINATOR_KEY=state["key_paths"][-1])
    else:
        env.update(CUDA_VISIBLE_DEVICES=str(state["config"]["stages"][index]["gpu_index"]),
                   SHARD_NODE_KEY=state["key_paths"][index])
    return env


def stage_argv(state, plan_path, index):
    rows, config = state["config"]["stages"], state["config"]
    row = rows[index]
    argv = [config["python"], str(ROOT / "engines/deepseek_v4/v4_pipe.py"), "stage",
            "--deployment-plan", str(plan_path), "--stage", str(index), "--nstages", str(len(rows)),
            "--lo", str(row["lo"]), "--hi", str(row["hi"]), "--port", str(row["port"]),
            "--dir", config["model_dir"], "--device", "cuda:0", "--bind", "127.0.0.1", "--receipts",
            "--timeout", str(math.ceil(config["timeouts"]["startup_s"]))]
    if index < len(rows) - 1:
        argv += ["--next", f"127.0.0.1:{rows[index + 1]['port']}"]
    else:
        argv += ["--dspark"]
    return argv


def _manager(run_dir):
    from shard.managed_launch import ManagedLauncher
    return ManagedLauncher(Path(run_dir) / "processes")


def _launch(run_dir, state, name, request):
    # Persist ownership before spawning, including launches that fail registration.
    if name not in state["owned_names"]:
        state["owned_names"].append(name)
        _save_state(run_dir, state)
    path = Path(run_dir).resolve() / "private" / (name + ".request.json")
    _write(path, request)
    result = _manager(run_dir).start(name, [state["config"]["python"], str(Path(__file__).resolve()),
                                           "_worker", "--request", str(path)], cwd=str(ROOT))
    if result.get("reused"):
        raise RuntimeError("owned process already exists; stop this run explicitly before restarting")
    return result


def _check_owned_log_size(run_dir, name):
    from shard.managed_launch import ManagedLauncher
    ManagedLauncher._name(name)
    directory = Path(run_dir) / "processes"
    if directory.is_symlink():
        raise ValueError("owned process directory cannot be a symlink")
    path = directory / (name + ".log")
    if path.is_symlink():
        raise ValueError("owned process log cannot be a symlink")
    if path.exists() and path.stat().st_size > MAX_OWNED_LOG_BYTES:
        raise RuntimeError("owned process log exceeds 256 MiB; stop this run and inspect " + name + ".log")


def _observation(run_dir, name, port=None):
    _check_owned_log_size(run_dir, name)
    from engines.deepseek_v4.v4_preflight import process_observation
    return process_observation(name, state_dir=Path(run_dir) / "processes",
                               listener_port=port, log_tail_bytes=1 << 20)


def _wait(run_dir, name, timeout, *, listening=False, port=None):
    deadline = time.monotonic() + timeout
    while True:
        _check_owned_log_size(run_dir, name)
        observation = _observation(run_dir, name, port)
        if listening and observation.get("listening") and observation.get("owned_listener") is True:
            return observation
        if not observation.get("running"):
            if listening:
                raise RuntimeError("owned stage exited before listening; inspect " + name + ".log")
            return observation
        if time.monotonic() >= deadline:
            raise TimeoutError("owned process deadline exceeded: " + name)
        time.sleep(.2)


def validate_measurement(report, state, index, *, now=None):
    """Validate exact per-role evidence; CPU/schema fixtures cannot qualify hardware."""
    from shard.runtime_observation import gpu_uuid_text
    row = state["config"]["stages"][index]
    gpu = _assigned_idle(state, state["gpu_inventory"])[index]
    probe = state["config"]["resource_probe"]
    roles = {"lo": row["lo"], "hi": row["hi"], "head": index == 0,
             "tail": index == len(state["config"]["stages"]) - 1}
    roles["dspark"] = roles["tail"]
    if (report.get("schema") != "v4-resource-measurement/1" or report.get("calibration_status") != "measured"
            or report.get("checkpoint_id") != state["cohort"]["checkpoint_id"]
            or any(type(report.get(k)) is not type(v) or report[k] != v for k, v in roles.items())):
        raise ValueError("measurement must bind the exact checkpoint, span and head/tail/DSpark roles")
    measured = datetime.fromisoformat(report["measured_at"].replace("Z", "+00:00"))
    now = now or datetime.now(timezone.utc)
    if measured.tzinfo is None or not 0 <= (now - measured).total_seconds() <= probe["max_age_s"]:
        raise ValueError("resource measurement is stale or has an invalid clock")
    hardware = report["hardware"]
    actual_uuid = gpu_uuid_text(hardware.get("gpu_uuid"))
    if (hardware.get("device") != "cuda:0" or not actual_uuid or actual_uuid.lower() != gpu["gpu_uuid"].lower()
            or hardware.get("gpu_name") != gpu["gpu_name"]):
        raise ValueError("resource measurement GPU UUID/device/name differs from assigned observed hardware")
    payload = report["runtime_config_payload"]
    payload_sha = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    expected_env = {k: v for k, v in stage_environment(state, index).items() if k.startswith("V4_")}
    source = {Path(x["name"]).name: x["sha256"] for x in state["source"]["files"]
              if x["name"].startswith("engines/deepseek_v4/v4_")}
    if (report.get("runtime_config_sha256") != payload_sha or payload.get("environment") != expected_env
            or payload.get("engine_source_sha256") != source
            or any(payload.get(k) != v for k, v in roles.items())
            or payload.get("args", {}).get("max_seq_len") != int(expected_env["V4_MAX_SEQ"])
            or payload.get("args", {}).get("max_batch_size") != 1):
        raise ValueError("resource measurement source/config/public environment differs from prepared launch")
    workload = report.get("workload", {})
    if any(workload.get(k) != v for k, v in {"kind": "synthetic_resource_probe", "batch": 1,
            "max_seq": int(expected_env["V4_MAX_SEQ"]), "prefill_tokens": probe["prefill_tokens"],
            "decode_tokens": probe["decode_tokens"]}.items()):
        raise ValueError("measurement workload differs from the explicit resource probe")
    if roles["dspark"] and (report.get("draft_measurement_status") != "measured" or payload.get("dspark_loaded") is not True):
        raise ValueError("tail calibration must include actually loaded MTP/drafter resources")
    if report.get("public_environment") != state["config"]["environment"]:
        raise ValueError("resource probe observed unfrozen or differing public defaults")
    allocator = report.get("allocator", {})
    if allocator.get("peak_interval_owned") is not True:
        raise ValueError("load/prefill/decode allocator interval must be owned by the actual probe")
    for key in ("peak_reserved_bytes", "peak_allocated_bytes", "device_free_bytes", "device_total_bytes"):
        _integer(allocator.get(key), key, 1)
    peak = max(allocator["peak_reserved_bytes"], allocator["peak_allocated_bytes"])
    # nvidia-smi reports rounded MiB; CUDA provides the exact byte count.
    if (abs(allocator["device_total_bytes"] - gpu["vram_bytes"]) > 2 * 1024**2
            or allocator["device_free_bytes"] < probe["headroom_bytes"]
            or gpu["free_bytes"] < peak + probe["headroom_bytes"]):
        raise ValueError("assigned GPU lacks free headroom for its exact measured role")
    return {"stage": index, "gpu_uuid": gpu["gpu_uuid"], "runtime_config_sha256": payload_sha,
            "peak_bytes": peak, "headroom_bytes": probe["headroom_bytes"], "validated": True,
            "scope": "sampled real per-role allocation evidence; no resource lease or all-route guarantee"}


def measure(run_dir, confirmed_gpu_indices, *, inventory_reader=gpu_inventory):
    observe(run_dir, confirmed_gpu_indices, inventory_reader=inventory_reader)
    state = load_state(run_dir)
    if _source_identity()["sha256"] != state["source"]["sha256"]:
        raise ValueError("source changed since preparation")
    paths = []
    try:
        for i, row in enumerate(state["config"]["stages"]):
            path = Path(run_dir).resolve() / f"resource-stage-{i}.json"
            if path.exists():
                raise ValueError("measurement output already exists; use a fresh run or supply exact fresh measurements")
            request = {"kind": "measure", "environment": stage_environment(state, i), "output": str(path),
                       "model_dir": state["config"]["model_dir"], "lo": row["lo"], "hi": row["hi"],
                       "head": i == 0, "tail": i == len(state["config"]["stages"]) - 1,
                       "probe": state["config"]["resource_probe"]}
            name = state["config"]["ring_id"] + f".measure.{i}"
            _launch(run_dir, state, name, request)
            _wait(run_dir, name, state["config"]["timeouts"]["measure_s"])
            if not path.is_file():
                raise RuntimeError("actual resource probe failed; inspect owned log " + name)
            validate_measurement(_read(path), state, i)
            _manager(run_dir).stop(name)
            paths.append(str(path))
    except BaseException:
        stop(run_dir)
        raise
    state["measurement_paths"] = paths
    _save_state(run_dir, state)
    return paths


def verify_warmup(final, state, plan, job, measurements):
    from shard.receipt import verify_coverage
    from engines.deepseek_v4.v4_execution_profiles import require_profile_warmup
    if (final.get("jobId") != job["jobId"] or final.get("ok") is not True
            or final.get("tokensGenerated") != job["maxNew"]
            or not isinstance(final.get("tokenIds"), list) or len(final["tokenIds"]) != job["maxNew"]
            or any(type(token) is not int or token < 0 for token in final["tokenIds"])):
        raise ValueError("warmup must commit the complete bound request")
    receipts = final.get("receipts", [])
    expected = {row["signer_pubkey"]: (row["lo"], row["hi"]) for row in plan["stages"]}
    verify_coverage(receipts, 43, expected_by_signer=expected, expected_nonce=job["nonce"], check_chain=True)
    if any(r.get("job_id") != job["jobId"] or r.get("swarm_id") != plan["ring_id"] for r in receipts):
        raise ValueError("warmup receipt belongs to another request or ring")
    require_profile_warmup(receipts, {row["node_id"]: state["config"]["environment"].get("V4_EXECUTION_PROFILE")
                                      for row in plan["stages"]})
    by_signer = {r["pubkey"]: r for r in receipts}
    for i, row in enumerate(plan["stages"]):
        observed = by_signer[row["signer_pubkey"]].get("runtime_observation", {})
        loaded = state.get("loaded_observations", [None] * len(plan["stages"]))[i]
        if (observed.get("phase") != "job_complete" or observed.get("node_id") != row["node_id"]
                or observed.get("gpu_uuid", "").lower() != row["gpu_uuid"].lower()
                or observed.get("environment") != state["config"]["environment"]
                or observed.get("runtime_config_sha256") != measurements[i]["runtime_config_sha256"]
                or loaded is not None and observed.get("process_run_id") != loaded.get("process_run_id")):
            raise ValueError("signed warmup differs from the exact measured GPU/source/config/public environment")
    return {"verified": True, "nonce": job["nonce"], "job_id": job["jobId"], "receipts": receipts,
            "scope": "strict authenticated real-model warmup; independent numerical/performance acceptance remains"}


def _hardware_from_observations(state, plan, observations, *, signed):
    """Export actual facts; unsigned cold-open events cannot establish READY."""
    from phase0.v4_benchmark import runtime_observation_errors
    if len(observations) != len(plan["stages"]):
        raise ValueError("one observed runtime identity per assigned stage required")
    gpus = {row["gpu_uuid"].lower(): row for row in state["gpu_inventory"]["gpus"]}
    result = []
    for stage, observation in zip(plan["stages"], observations):
        gpu = gpus[stage["gpu_uuid"].lower()]
        row = {"node_id": observation["node_id"], "host_id": state["gpu_inventory"]["host_id"],
               "gpu_uuid": observation["gpu_uuid"], "gpu_name": gpu["gpu_name"], "driver": gpu["driver"],
               "vram_bytes": gpu["vram_bytes"], "signer_pubkey": stage["signer_pubkey"],
               "process_run_id": observation["process_run_id"], "runtime_config_sha256": observation["runtime_config_sha256"],
               "layer_start": stage["lo"], "layer_end": stage["hi"],
               "checkpoint_sha256": state["checkpoint"]["sha256"], "config_sha256": state["checkpoint"]["config_sha256"],
               "engine_source_sha256": state["source"]["sha256"], "stage_env_sha256": _digest(observation["environment"]),
               **{key: observation["versions"][key] for key in ("torch", "cuda", "tilelang")}}
        errors = runtime_observation_errors(observation, row, {"source": state["source"],
            "env": state["config"]["environment"], "evidence_contract": {"runtime_observation_required": True}})
        if not signed and observation.get("phase") == "loaded":
            # This is an explicitly unsigned initial inventory. The benchmark
            # authenticates its unchanged identities in each completed receipt.
            errors = [error for error in errors if error != "runtime observation must describe the measured completed job"]
        if errors:
            raise ValueError("hardware observation invalid: " + "; ".join(errors))
        result.append(row)
    return result


def hardware_from_warmup(state, plan, proof):
    from shard.receipt import verify_coverage
    if proof.get("verified") is not True:
        raise ValueError("verified signed warmup required")
    verify_coverage(proof["receipts"], 43,
        expected_by_signer={row["signer_pubkey"]: (row["lo"], row["hi"]) for row in plan["stages"]},
        expected_nonce=proof["nonce"], check_chain=True)
    if any(receipt.get("job_id") != proof["job_id"] or receipt.get("swarm_id") != plan["ring_id"]
           for receipt in proof["receipts"]):
        raise ValueError("hardware proof belongs to another job or ring")
    by_signer = {receipt["pubkey"]: receipt for receipt in proof["receipts"]}
    return _hardware_from_observations(state, plan,
        [by_signer[row["signer_pubkey"]]["runtime_observation"] for row in plan["stages"]], signed=True)


def warmup(run_dir, *, measurements=None):
    """Authenticate a loaded ring; only verified real warmup sets READY."""
    state = load_state(run_dir)
    if state["status"] not in ("LOADED", "READY") or not state.get("plan"):
        raise ValueError("warmup requires this run's loaded owned ring")
    plan = state["plan"]
    reports = measurements if measurements is not None else [_read(path) for path in state["measurement_paths"]]
    path = Path(run_dir).resolve() / "plan.json"
    job = {"jobId": "acceptance-warmup-" + secrets.token_hex(8), "swarmId": plan["ring_id"],
           "nonce": secrets.token_hex(32), "maxNew": 2, "ignoreEOS": True,
           "messages": [{"role": "user", "content": "Reply with a short greeting."}],
           "dspark": True, "pipelined": True}
    warm_name = state["config"]["ring_id"] + ".warmup"
    argv = [state["config"]["python"], str(ROOT / "engines/deepseek_v4/v4_pipe.py"), "coord",
            "--deployment-plan", str(path), "--coordinator-key", state["key_paths"][-1],
            "--dir", state["config"]["model_dir"], "--receipts",
            "--timeout", str(math.ceil(state["config"]["timeouts"]["warmup_s"]))]
    try:
        for i in range(len(plan["stages"])):
            name = state["config"]["ring_id"] + f".stage.{i}"
            _check_owned_log_size(run_dir, name)
            status = _observation(run_dir, name, state["config"]["stages"][i]["port"])
            if not status.get("running") or not status.get("listening") or status.get("owned_listener") is not True:
                raise ValueError("warmup requires every assigned stage's live owned listener")
        _launch(run_dir, state, warm_name, {"kind": "command", "argv": argv,
                                          "environment": stage_environment(state), "stdin": json.dumps(job) + "\n"})
        observed = _wait(run_dir, warm_name, state["config"]["timeouts"]["warmup_s"])
        records = [json.loads(line[len("SHARD_JOB_DONE "):]) for line in observed["log_tail"].splitlines()
                   if line.startswith("SHARD_JOB_DONE ")]
        if len(records) != 1:
            raise RuntimeError("strict warmup exited without its complete signed result; inspect owned log")
        proof = verify_warmup(records[0], state, plan, job, reports)
        hardware = hardware_from_warmup(state, plan, proof)
        _manager(run_dir).stop(warm_name)
        _write(Path(run_dir) / "signed-warmup.json", proof)
        _write(Path(run_dir) / "hardware.json", hardware)
        state.update(status="READY", request_count=state.get("request_count", 0) + 1)
        _save_state(run_dir, state)
        ready = {"schema": "flyai-v4-acceptance-ring-ready/1", "ready": True, "ring_id": plan["ring_id"],
                 "scope": SCOPE, "resource_leases": False, "hardware_qualification": False,
                 "release_stage_count_supported": len(plan["stages"]) in (4, 6), "isolation": "none",
                 "signed_warmup": "signed-warmup.json", "hardware": "hardware.json", "environment": "environment.json"}
        _write(Path(run_dir) / "ready.json", ready)
        return ready
    except BaseException:
        stop(run_dir)
        raise


def start(run_dir, confirmed_gpu_indices, *, measurements=None, cold_open=False, inventory_reader=gpu_inventory):
    if type(cold_open) is not bool:
        raise ValueError("cold_open must be an explicit boolean")
    state = load_state(run_dir)
    _confirmed(state, confirmed_gpu_indices)
    if any(_manager(run_dir).status(name).get("reserved") for name in state["owned_names"]):
        raise ValueError("run owns existing process reservations; stop it explicitly before start")
    paths = measurements if measurements is not None else state.get("measurement_paths")
    if paths is None:
        paths = measure(run_dir, confirmed_gpu_indices, inventory_reader=inventory_reader)
    observe(run_dir, confirmed_gpu_indices, inventory_reader=inventory_reader)
    state = load_state(run_dir)
    if _source_identity()["sha256"] != state["source"]["sha256"]:
        raise ValueError("source changed since preparation")
    from phase0.v4_benchmark import verify_checkpoint
    verify_checkpoint(state["config"]["model_dir"], state["checkpoint"])
    if len(paths) != len(state["config"]["stages"]):
        raise ValueError("one exact fresh resource measurement per assigned GPU required")
    reports = [_read(path) if isinstance(path, (str, os.PathLike)) else path for path in paths]
    for i, report in enumerate(reports):
        validate_measurement(report, state, i)
    plan = make_plan(state)
    path = Path(run_dir).resolve() / "plan.json"
    _write(path, plan)
    try:
        loaded_observations = [None] * len(plan["stages"])
        for i in reversed(range(len(plan["stages"]))):
            name = state["config"]["ring_id"] + f".stage.{i}"
            _launch(run_dir, state, name, {"kind": "command", "argv": stage_argv(state, path, i),
                                          "environment": stage_environment(state, i)})
            observed = _wait(run_dir, name, state["config"]["timeouts"]["startup_s"], listening=True,
                             port=state["config"]["stages"][i]["port"])
            loaded = [event["observation"] for event in observed.get("events", [])
                      if event.get("event") == "loaded" and "observation" in event]
            if len(loaded) != 1 or loaded[0].get("phase") != "loaded":
                raise ValueError("owned listening process lacks its actual loaded runtime observation")
            if (loaded[0].get("runtime_config_sha256") != reports[i]["runtime_config_sha256"]
                    or loaded[0].get("node_id") != plan["stages"][i]["node_id"]
                    or loaded[0].get("gpu_uuid", "").lower() != plan["stages"][i]["gpu_uuid"].lower()
                    or loaded[0].get("environment") != state["config"]["environment"]):
                raise ValueError("loaded stage differs from its exact measured identity/config/environment")
            loaded_observations[i] = loaded[0]
        hardware = _hardware_from_observations(state, plan, loaded_observations, signed=False)
        _write(Path(run_dir) / "hardware.json", hardware)
        # Store accepted supplied reports locally so later warmup uses precisely
        # these measurements and restarting can recheck their freshness.
        local_paths = []
        for i, report in enumerate(reports):
            local_path = Path(run_dir).resolve() / f"accepted-resource-stage-{i}.json"
            _write(local_path, report)
            local_paths.append(str(local_path))
        state.update(status="LOADED", plan=plan, measurement_paths=local_paths, request_count=0,
                     loaded_observations=loaded_observations)
        _save_state(run_dir, state)
        loaded = {"schema": "flyai-v4-acceptance-ring-loaded/1", "ready": False, "ring_id": plan["ring_id"],
                  "scope": SCOPE, "resource_leases": False, "hardware_qualification": False,
                  "request_count": 0, "isolation": "none", "cold_open": bool(cold_open),
                  "hardware_evidence": "unsigned runtime loaded events from this launch's owned PIDs; benchmark receipts must authenticate them",
                  "observations": loaded_observations, "hardware": "hardware.json", "environment": "environment.json"}
        _write(Path(run_dir) / "loaded.json", loaded)
        if cold_open:
            _write(Path(run_dir) / "ready.json", {"ready": False, "status": "LOADED", "request_count": 0, "scope": SCOPE})
            return loaded
        return warmup(run_dir, measurements=reports)
    except BaseException as error:
        _write(Path(run_dir) / "failure.json", {"ready": False, "error_class": type(error).__name__,
                                                "message": str(error), "scope": SCOPE})
        try:
            stop(run_dir)
        except Exception as cleanup:
            error.add_note("Owned cleanup unconfirmed: " + type(cleanup).__name__)
        raise


def stop(run_dir):
    """Stop only recorded names in this run's ManagedLauncher database."""
    state = load_state(run_dir)
    results, errors = [], []
    manager = _manager(run_dir)
    for name in reversed(state["owned_names"]):
        try:
            results.append(manager.stop(name))
        except Exception as error:
            errors.append({"name": name, "error_class": type(error).__name__})
    state["status"] = "CLEANUP_UNCONFIRMED" if errors else "STOPPED"
    _save_state(run_dir, state)
    result = {"ready": False, "cleanup_acknowledged": not errors, "scope": SCOPE,
              "processes": results, "errors": errors, "preserved": "keys, weights, artifacts and foreign processes"}
    _write(Path(run_dir) / "stop.json", result)
    if (Path(run_dir) / "ready.json").exists():
        _write(Path(run_dir) / "ready.json", {"ready": False, "status": state["status"], "scope": SCOPE})
    if errors:
        raise RuntimeError("owned cleanup was not fully acknowledged; inspect stop.json")
    return result


def _worker(path):
    request = _read(path)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("V4_")
                   and k not in {"SHARD_STAGE_LEASE_CONFIG", "SHARD_NODE_KEY", "SHARD_COORDINATOR_KEY"}}
    environment.update(request["environment"])
    if request["kind"] == "command":
        result = subprocess.run(request["argv"], cwd=ROOT, env=environment, text=True,
                                input=request.get("stdin"), check=False)
        return result.returncode
    if request["kind"] != "measure":
        raise ValueError("unknown private worker request")
    os.environ.clear(); os.environ.update(environment)
    engine_dir = str(ROOT / "engines/deepseek_v4")
    sys.path.insert(0, engine_dir)
    # Import the same complete pipe module as serving, so all observed defaults
    # and its environment bindings exist in this isolated probe process too.
    import v4_pipe  # noqa: F401
    import v4_resources
    from v4_observability import public_environment
    probe = request["probe"]
    report = v4_resources.measure_checkpoint(request["model_dir"], request["lo"], request["hi"],
        head=request["head"], tail=request["tail"], dspark=request["tail"], device="cuda:0",
        max_seq=int(environment["V4_MAX_SEQ"]), prefill_tokens=probe["prefill_tokens"], decode_tokens=probe["decode_tokens"])
    report["public_environment"] = public_environment()
    _write(request["output"], report, exclusive=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    for name in ("prepare", "observe", "measure", "start", "warmup", "stop"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", required=True)
        if name == "prepare":
            command.add_argument("--config", required=True)
        if name in ("observe", "measure", "start"):
            command.add_argument("--confirm-gpus", required=True, help="assigned idle GPU indices in stage order, e.g. 0,1,2,3,4,5")
        if name == "start":
            command.add_argument("--measurements", help="JSON array of exact fresh per-role measurement file paths")
            command.add_argument("--cold-open", action="store_true", help="load only; zero requests and ready=false for genuine first benchmark request")
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "_worker":
        return _worker(args.request)
    if args.command == "prepare":
        result = prepare(_read(args.config), args.run_dir)
    elif args.command in ("stop", "warmup"):
        result = globals()[args.command](args.run_dir)
    else:
        indices = [int(part) for part in args.confirm_gpus.split(",")]
        kwargs = {"cold_open": args.cold_open} if args.command == "start" else {}
        if args.command == "start" and args.measurements:
            kwargs["measurements"] = _read(args.measurements)
        result = globals()[args.command](args.run_dir, indices, **kwargs)
    # Return concise public metadata, never private coordinator key paths.
    if args.command == "prepare":
        result = {"prepared": True, "runtime_ready": False, "scope": SCOPE,
                  "run_dir": str(Path(args.run_dir).resolve()), "assigned_gpu_indices":
                  [row["gpu_index"] for row in result["config"]["stages"]]}
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

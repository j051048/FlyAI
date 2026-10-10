"""Bounded lifecycle events and observations of the actual loaded V4 process."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import uuid
import time
from datetime import datetime, timezone

import v4_levers
from shard.runtime_observation import SCHEMA, digest, validate_runtime_observation, gpu_uuid_text


def effective_flags(side, stage=None):
    public = set(v4_levers.LEVERS_BY_ENV) | set(v4_levers.NON_LEVER_ENV)
    return {finding.env: {
        "requested": os.environ.get(finding.env) if finding.env in public else None,
        "parsed": str(finding.requested) if finding.env in public else "unregistered",
        "observed": str(finding.observed),
        "verdict": str(finding.verdict), "reason": finding.why or "",
    } for finding in v4_levers.audit(side=side, stage=stage)}


def public_environment():
    names = set(v4_levers.LEVERS_BY_ENV) | set(v4_levers.NON_LEVER_ENV)
    result = {}
    modules = [sys.modules.get(Path(name).stem) for name in v4_levers.ENGINE_MODULES]
    for key in sorted(names):
        if key in ("V4_DIR", "V4_DEV") or any(part in key for part in ("SECRET", "PASSWORD", "KEY_FILE", "TOKEN_FILE")):
            continue
        value = os.environ.get(key)
        if value is None:
            # Only report parsed defaults of modules that really exist in this
            # process. Lazy modules' defaults are left unknown until loaded.
            parsed = next((getattr(module, key) for module in modules
                if module is not None and hasattr(module, key)), None)
            if isinstance(parsed, bool):
                value = "1" if parsed else "0"
            elif isinstance(parsed, (int, float, str)):
                value = str(parsed)
        if value is not None:
            result[key] = value
    return result


def source_inventory():
    engine = Path(__file__).resolve().parent
    root = engine.parent.parent
    files = list(engine.glob("v4_*.py"))
    vendor = root / "vendor" / "deepseek_v4_ref"
    files += list(vendor.rglob("*.py")) if vendor.is_dir() else []
    for name in ("receipt", "runtime_metrics", "runtime_profile", "runtime_observation", "pipeline_plan", "pipeline_session", "transport", "speculation_policy"):
        path = root / "shard" / (name + ".py")
        if path.is_file():
            files.append(path)
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(files))}


def runtime_observation(stage, *, session_config=None, phase="loaded"):
    """Snapshot source/backend at load; later refuse a file change, never bless it.

    GPU UUID and runtime settings are worker declarations. A receipt signature binds
    these declarations but does not remotely attest the physical hardware.
    """
    from v4_resources import runtime_config_identity
    import v4_kernels_cpu
    from v4_runtime_init import hadamard_identity
    source = source_inventory()
    previous = getattr(stage, "_runtime_observed_sources", None)
    if previous is not None and source != previous:
        raise RuntimeError("V4 source files changed since this process loaded its runtime")
    stage._runtime_observed_sources = source
    if not hasattr(stage, "_runtime_process_run_id"):
        stage._runtime_process_run_id = uuid.uuid4().hex
    device = str(stage.device)
    gpu_uuid = None
    if device.startswith("cuda"):
        import torch
        gpu_uuid = gpu_uuid_text(getattr(torch.cuda.get_device_properties(stage.device), "uuid", None))
    row = session_config.stage(session_config.index) if session_config is not None else None
    flags = effective_flags(v4_levers.STAGE, stage)
    env = public_environment()
    torch_module = sys.modules.get("torch")
    tilelang_module = sys.modules.get("tilelang")
    value = {"schema": SCHEMA, "node_id": row["node_id"] if row else f"local-stage-{stage.lo}-{stage.hi}",
        "gpu_uuid": gpu_uuid, "process_run_id": stage._runtime_process_run_id,
        "runtime_config_sha256": runtime_config_identity(stage), "source_files": source,
        "source_sha256": digest(source), "environment": env, "environment_sha256": digest(env),
        "effective_flags": flags, "effective_flags_sha256": digest(flags),
        "kernel_backend": v4_kernels_cpu.backend(), "hadamard_backend": hadamard_identity()["backend"],
        "graph_mode": stage._graph_mode if stage._block_graphs is not None else "off",
        "wire_mode": "fp8" if env.get("V4_FP8_WIRE", "0") not in ("", "0") else "bf16",
        "transport": "engine-message-socket; external route declared by deployment",
        "backend_identity": {"hadamard": hadamard_identity()},
        "optimizations": {
            "prefill_expert_pipeline": stage.prefill_pipeline_status() if hasattr(stage, "prefill_pipeline_status") else {"enabled": False},
            "conversation_cache": stage._conversation_cache.status() if hasattr(stage, "_conversation_cache") else {"enabled": False}},
        "versions": {"python": platform.python_version(), "torch": str(getattr(torch_module, "__version__", "unknown")),
            "cuda": getattr(getattr(torch_module, "version", None), "cuda", None),
            "tilelang": getattr(tilelang_module, "__version__", None)}, "phase": phase}
    return validate_runtime_observation(value)


def emit_stage_event(event, stage_index, *, operation=None, stream=None, **fields):
    """Lifecycle-only data: no prompts, tensors, private keys or arbitrary env."""
    body = {"schema": "shard-stage-event/1", "event": event, "stage": stage_index,
        "pid": os.getpid(), "operation": operation, "monotonic_ns": time.monotonic_ns(),
        "recorded_at": datetime.now(timezone.utc).isoformat(), **fields}
    print("SHARD_STAGE_EVENT " + json.dumps(body, sort_keys=True, allow_nan=False),
        file=stream or sys.stdout, flush=True)
    return body

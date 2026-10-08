"""Frozen V4 benchmark protocol, existing-ring runner and fail-closed evidence checker.

Preparation hashes LOCAL files; run attaches only to an already-running ring. Neither command
downloads a model, launches/rents nodes, changes the ring or introduces a new decode algorithm.
Importing this module requires only the standard library. CUDA/model imports are live-run only.
See docs/V4_BENCHMARK.md for the measurement boundaries and the limits of signed telemetry.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCHEMA = "flyai-v4-benchmark-protocol/1"
REPORT_SCHEMA = "flyai-v4-benchmark-report/1"
NETWORK_SCHEMA = "flyai-v4-network-comparison/1"
MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
MODEL_IDS = frozenset((MODEL_ID, "deepseek-ai/DeepSeek-V4-Flash"))
WORKLOADS = ("code", "math", "prose", "agentic")
TARGETS = {4: 40.0, 6: 30.0}
PROMPTS = {
    "code": "Write a complete Rust bounded worker pool using std only. Explain ownership, shutdown, "
            "backpressure and panic handling. Include runnable code and tests for a full queue.",
    "math": "Derive a stable online algorithm for the mean and variance of a stream. Prove the "
            "update equations, explain population versus sample variance, and work through 2, 4, 8, 10.",
    "prose": "Write a clear essay about maintaining a public library during a prolonged power outage. "
             "Describe practical choices, disagreements, tradeoffs, and a realistic recovery plan.",
    "agentic": "You operate an incident-response assistant. A database migration stalls while writes "
               "continue. Produce a detailed sequence of tool calls, expected evidence, rollback "
               "conditions and a final status message. Do not assume a successful rollback.",
}
CONTEXT = "Reference note: measure the actual result, preserve the inputs, and record the evidence.\n"

# An explicit baseline recipe, not the engine's mutable implicit defaults. Operators must launch
# their stages with these same values; --env-file can freeze a DIFFERENT experiment recipe.
DEFAULT_ENV = {
    "V4_KERNELS": "tilelang", "V4_DTYPE": "bfloat16", "V4_MAX_SEQ": "8192",
    "V4_HADAMARD": "auto",
    "V4_MAX_BATCH": "1", "V4_CUDA_GRAPH": "whole", "V4_GRAPH_MAX": "192",
    "V4_MOE_GROUPED": "1", "V4_MOE_DECODE": "1", "V4_MOE_MULTI": "1",
    "V4_MOE_MULTI_MAX": "32", "V4_MOE_IN_GRAPH": "1", "V4_FP8_GEMV": "1",
    "V4_FP8_SHARED": "1", "V4_DSPARK_FAST": "0", "V4_DSPARK_GRAPH": "0",
    "V4_DSPARK_MOE": "1", "V4_DSPARK_BLOCK": "8", "V4_DRAFT_TOP2": "0",
    "V4_PIPELINED_SPEC": "1", "V4_SPEC_DEPTH": "16", "V4_LAZY_DRAFT": "1",
    "V4_REFILL_FLOOR": "1", "V4_DSPARK_CONF_GATE": "0", "V4_DSPARK_CONF_THRESH": "0",
    "V4_DSPARK_CONF_MIN": "1", "V4_FP8_WIRE": "0", "V4_REF_SLIM": "0",
    "V4_REF_SLIM_NOQAT": "0", "V4_FAST_VERIFY": "0", "V4_FAST_VERIFY_MAX": "16",
    "V4_KEEPWARM": "0", "V4_KEEPWARM_MS": "150", "V4_TIMING": "0",
    "V4_TIMING_EVERY": "0", "V4_RUNTIME_METRICS": "1", "V4_LEVERS_STRICT": "1",
    "V4_EXPERT_PLACEMENT": "gpu", "V4_EXPERT_CACHE_SLOTS": "0",
    "V4_EXPERT_CACHE_MIB": "0", "V4_EXPERT_CACHE_RESERVE_MIB": "2048",
}
HISTORICAL = {
    "source": "docs/receipts/v4-flash-matrix-20260802.json",
    "date": "2026-08-02", "stages": 6, "gpu": "RTX 5090 32 GB",
    "median_tok_s": 30.15, "max_seq": 8192, "max_new": 512,
    "reported_reps_tok_s": [19.311, 30.178, 30.287, 30.124],
    "reported_protocol": "novel Rust concurrency prompt; first repetition excluded; three warm reps",
    "verification": "historical_repository_claim_not_independently_verified",
    "limitations": "No raw signatures, frozen prompt IDs or checkpoint hashes in the historical file; "
                   "new suite uses different fixed prompts and cannot reproduce that claim exactly.",
}
MEASUREMENT_DEFINITIONS = {
    "first_token_s": "first committed callback minus request start, including reset and prefill",
    "decode_s": "last committed callback minus first committed callback; N-1 committed tokens",
    "elapsed_s": "generation return minus request start, including prefill, decode and speculative return drain",
    "drain_s": "generation return minus last committed callback; includes residual compute/queue/wire, not network RTT",
    "receipt_sweep_s": "one independent receipt sweep after generation has returned",
    "full_service_s": "final observer return minus request start, including generation, drain, sweep and observer overhead",
    "percentiles": "linear interpolation of retained raw warm samples; all four fixed workloads are required",
}


class BenchmarkError(ValueError):
    pass


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sha(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _integer(value, minimum=0) -> bool:
    return type(value) is int and value >= minimum


def _positive(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _duration_valid(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_path(directory: Path, name: str) -> Path:
    """A manifest is data; it must never resolve a model file outside the checkpoint."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise BenchmarkError(f"unsafe checkpoint filename: {name!r}")
    result = (directory / name).resolve()
    if not result.is_relative_to(directory.resolve()):
        raise BenchmarkError(f"checkpoint file escapes directory: {name!r}")
    return result


def _artifacts():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from shard import weight_artifacts
    return weight_artifacts


def _checkpoint_identity(value):
    return {key: item for key, item in value.items() if key != "local_verification"}


def _catalogue_inventory(directory: Path) -> dict:
    artifacts = _artifacts()
    catalog = artifacts.validate_catalog(artifacts.read_json(directory / artifacts.GLOBAL_FILE))
    config = catalog["config"]
    from engines.deepseek_v4.v4_artifact_contract import validate_native_config
    validate_native_config(config, model_id=catalog["model_id"], runtime_abi=catalog["runtime_abi"])
    if catalog["model_id"] not in MODEL_IDS or config.get("n_layers") != 43 or config.get("expert_dtype") != "fp4":
        raise BenchmarkError("this protocol requires the shipped 43-layer V4 FP4 checkpoint")
    files = []
    for name, row in sorted(catalog["assets"].items()):
        if artifacts.hash_file(artifacts.safe_path(directory, name)) != (row["sha256"], row["size"]):
            raise BenchmarkError("checkpoint config/tokenizer asset bytes differ from catalogue")
        files.append({"name": name, "bytes": row["size"], "sha256": row["sha256"]})
    tokenizer_files = [row for row in files if row["name"].startswith(("tokenizer", "vocab", "merges"))]
    if not tokenizer_files:
        raise BenchmarkError("checkpoint has no pinned tokenizer files")
    local_encoder = {p.name for pattern in ("tokenizer*", "vocab*", "merges*", "special_tokens_map.json", "generation_config.json")
                     for p in directory.glob(pattern) if p.is_file()}
    if not local_encoder <= set(catalog["assets"]):
        raise BenchmarkError("unverified local tokenizer override")
    local = {"payload_integrity_verified": False,
             "verification_scope": "pinned catalogue and local model assets only"}
    if (directory / artifacts.PACK_FILE).exists():
        proof = (artifacts.verify_stage_artifacts(directory) if (directory / artifacts.STAGE_FILE).exists()
                 else artifacts.verify_weight_pack(directory))
        artifacts.verified_stage_artifact_descriptor(proof)
        local = {"payload_integrity_verified": True, "verification_scope": proof["verification_scope"]}
    elif any(directory.glob("*.safetensors")):
        raise BenchmarkError("local weights lack a verified packing manifest")
    return {"identity_kind": "logical-tensor-catalogue", "checkpoint_id": catalog["checkpoint_id"],
            "manifest_sha256": catalog["manifest_sha256"], "sha256": catalog["checkpoint_id"].removeprefix("tensor-sha256:"),
            "global_catalog": catalog, "files": files, "config_sha256": catalog["config_sha256"],
            "tokenizer_sha256": digest(tokenizer_files), "local_verification": local,
            "quantization": {k: config.get(k) for k in ("dtype", "expert_dtype", "scale_dtype", "scale_fmt")}}


def checkpoint_inventory(directory: str | Path) -> dict:
    """Read every weight byte once. Header/size alone is insufficient to pin a checkpoint."""
    directory = Path(directory).resolve()
    if (directory / ".shard-model-artifacts.json").exists():
        return _catalogue_inventory(directory)
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("n_layers") != 43 or config.get("expert_dtype") != "fp4":
        raise BenchmarkError("this protocol requires the shipped 43-layer V4 FP4 checkpoint")
    weights = sorted(directory.glob("*.safetensors"))
    if not weights:
        raise BenchmarkError("checkpoint has no safetensors weights")
    auxiliary = {config_path}
    for pattern in ("*.index.json", "tokenizer*", "special_tokens_map.json", "vocab*", "merges*"):
        auxiliary.update(p for p in directory.glob(pattern) if p.is_file())
    tokenizer_files = [p for p in auxiliary if p.name.startswith(("tokenizer", "vocab", "merges"))]
    if not tokenizer_files:
        raise BenchmarkError("checkpoint has no tokenizer files")
    for index in directory.glob("*.safetensors.index.json"):
        mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        if not mapping or any(not _safe_path(directory, name).is_file() for name in mapping.values()):
            raise BenchmarkError(f"weight index {index.name} has absent/invalid weight files")
    files = []
    for p in sorted(set(weights) | auxiliary):
        name = p.relative_to(directory).as_posix()
        _safe_path(directory, name)
        files.append({"name": name, "bytes": p.stat().st_size, "sha256": file_digest(p)})
    return {"files": files, "sha256": digest(files), "config_sha256": file_digest(config_path),
            "tokenizer_sha256": digest([x for x in files if x["name"] in
                                        {p.name for p in tokenizer_files}]),
            "quantization": {k: config.get(k) for k in
                             ("dtype", "expert_dtype", "scale_dtype", "scale_fmt")}}


def verify_checkpoint(directory: str | Path, inventory: dict) -> dict:
    actual = checkpoint_inventory(directory)
    if _checkpoint_identity(actual) != _checkpoint_identity(inventory):
        raise BenchmarkError("checkpoint/config/tokenizer bytes differ from the frozen protocol")
    if actual.get("identity_kind") == "logical-tensor-catalogue":
        local = actual["local_verification"]
        return {"checkpoint_bytes_verified": local["payload_integrity_verified"] and
                    local["verification_scope"] == "complete packed payload hashes",
                "assigned_checkpoint_bytes_verified": local["payload_integrity_verified"] and
                    local["verification_scope"] == "assigned stage payload hashes",
                "global_catalog_verified": True, "checkpoint_id": actual["checkpoint_id"],
                "local_weight_scope": local["verification_scope"],
                "config_bytes_verified": True, "tokenizer_bytes_verified": True}
    return {"checkpoint_bytes_verified": True, "config_bytes_verified": True,
            "tokenizer_bytes_verified": True}


def source_identity(root: Path = ROOT) -> dict:
    paths = set()
    for directory in (root / "engines" / "deepseek_v4", root / "vendor" / "deepseek_v4_ref", root / "shard"):
        paths.update(directory.rglob("*.py"))
    for name in ("phase0/wire.py", "phase0/v4_benchmark.py", "phase0/v4_soak.py"):
        if (root / name).is_file():
            paths.add(root / name)
    files = [{"name": p.relative_to(root).as_posix(), "sha256": file_digest(p)} for p in sorted(paths)]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    working_tree = {"known": False, "dirty": None, "status_sha256": None}
    try:
        status = subprocess.check_output(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all", "--",
                                          *(row["name"] for row in files)], cwd=root, stderr=subprocess.DEVNULL)
        working_tree = {"known": True, "dirty": bool(status), "status_sha256": hashlib.sha256(status).hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        pass
    # The content digest, rather than an optimistic clean flag, also identifies uncommitted code.
    return {"git_commit": commit, "sha256": digest(files), "files": files, "working_tree": working_tree}


def freeze_prompts(encode: Callable[[str], list[int]], render: Callable[[str], str],
                   prompt_tokens: int = 512) -> list[dict]:
    if not _integer(prompt_tokens, 1):
        raise BenchmarkError("prompt_tokens must be a positive integer")
    marker = "FLYAI_BENCHMARK_BODY_SENTINEL"
    rendered_marker = render(marker)
    if rendered_marker.count(marker) != 1:
        raise BenchmarkError("prompt renderer must preserve the message body exactly once")
    prefix = list(encode(rendered_marker.split(marker)[0]))
    prompts = []
    for name in WORKLOADS:
        ids = list(encode(render(PROMPTS[name])))
        if len(ids) > prompt_tokens:
            raise BenchmarkError(f"{name} task uses {len(ids)} tokens, above frozen context {prompt_tokens}")
        if ids[:len(prefix)] != prefix:
            raise BenchmarkError("tokenizer chat prefix is not stable across the frozen messages")
        # Keep BOS, role markers, the entire task and assistant suffix intact. Only deterministic
        # context IDs in the message body are truncated to reach the exact requested context.
        pad_count = prompt_tokens - len(ids)
        padding = []
        repetitions = 1
        while len(padding) < pad_count:
            padding = list(encode(CONTEXT * repetitions))
            repetitions *= 2
        ids = ids[:len(prefix)] + padding[:pad_count] + ids[len(prefix):]
        if not all(_integer(t) for t in ids):
            raise BenchmarkError("tokenizer returned non-integer token IDs")
        prompts.append({"workload": name, "task": PROMPTS[name], "rendered_task": render(PROMPTS[name]),
                        "padding_policy": "deterministic_context_ids_after_chat_prefix",
                        "token_ids": ids, "sha256": digest(ids)})
    return prompts


def seal_protocol(protocol: dict) -> dict:
    body = {k: v for k, v in protocol.items() if k != "sha256"}
    return dict(body, sha256=digest(body))


def protocol_errors(p: dict) -> list[str]:
    errors = []
    if not isinstance(p, dict) or p.get("schema") != PROTOCOL_SCHEMA:
        return ["protocol schema absent or unsupported"]
    try:
        if p.get("sha256") != digest({k: v for k, v in p.items() if k != "sha256"}):
            errors.append("protocol hash does not match contents")
        if p.get("model_id") not in MODEL_IDS or p.get("layer_count") != 43:
            errors.append("protocol is not the fixed V4-Flash 43-layer model")
        ck = p.get("checkpoint", {})
        logical = ck.get("identity_kind") == "logical-tensor-catalogue"
        if (not _sha(ck.get("sha256")) or not _sha(ck.get("config_sha256"))
                or not _sha(ck.get("tokenizer_sha256")) or not ck.get("files")
                or not logical and digest(ck["files"]) != ck["sha256"]):
            errors.append("checkpoint/config/tokenizer hash inventory absent or invalid")
        if logical:
            artifacts = _artifacts()
            catalog = artifacts.validate_catalog(ck.get("global_catalog"))
            from engines.deepseek_v4.v4_artifact_contract import validate_native_config
            validate_native_config(catalog["config"], model_id=catalog["model_id"], runtime_abi=catalog["runtime_abi"])
            expected_assets = [{"name": name, "bytes": row["size"], "sha256": row["sha256"]}
                               for name, row in sorted(catalog["assets"].items())]
            if (ck.get("checkpoint_id"), ck.get("manifest_sha256"), ck.get("sha256"), ck.get("files"),
                    catalog["model_id"], catalog["config"].get("n_layers"), catalog["config_sha256"]) != (
                    catalog["checkpoint_id"], catalog["manifest_sha256"], catalog["checkpoint_id"].removeprefix("tensor-sha256:"), expected_assets,
                    p["model_id"], p["layer_count"], ck.get("config_sha256")):
                errors.append("logical checkpoint/catalogue/assets binding invalid")
        files = ck.get("files", [])
        if not logical and not any(x.get("name", "").endswith(".safetensors") for x in files):
            errors.append("weight bytes are not pinned")
        for x in files:
            if not _sha(x.get("sha256")) or not _integer(x.get("bytes"), 1):
                errors.append("invalid checkpoint file digest/size")
                break
            _safe_path(Path("."), x["name"])
        by_name = {x["name"]: x for x in files}
        tokenizer_files = [x for x in files if x["name"].startswith(("tokenizer", "vocab", "merges"))]
        if (len(by_name) != len(files) or by_name.get("config.json", {}).get("sha256") != ck.get("config_sha256")
                or not tokenizer_files or digest(tokenizer_files) != ck.get("tokenizer_sha256")):
            errors.append("checkpoint config/tokenizer digest binding invalid")
        if ck.get("quantization", {}).get("expert_dtype") != "fp4":
            errors.append("quantization must explicitly pin FP4 experts")
        source = p.get("source", {})
        if not source.get("files") or not _sha(source.get("sha256")) or digest(source["files"]) != source["sha256"]:
            errors.append("engine source contents are not pinned")
        source_names = set()
        for row in source.get("files", []):
            if not isinstance(row, dict) or not _sha(row.get("sha256")) or row.get("name") in source_names:
                errors.append("invalid or duplicate source file digest")
                break
            _safe_path(Path("."), row["name"])
            source_names.add(row["name"])
        state = source.get("working_tree")
        if state is not None and (not isinstance(state, dict) or set(state) != {"known", "dirty", "status_sha256"}
                or type(state.get("known")) is not bool or
                (state["known"] and (type(state.get("dirty")) is not bool or not _sha(state.get("status_sha256")))) or
                (not state["known"] and (state.get("dirty") is not None or state.get("status_sha256") is not None))):
            errors.append("source working-tree summary invalid")
        contract = p.get("evidence_contract")
        if contract is not None and (not isinstance(contract, dict) or set(contract) != {"coordinator_diagnostics_required", "runtime_observation_required"}
                or any(type(value) is not bool for value in contract.values())):
            errors.append("explicit benchmark observation contract invalid")
        network = p.get("network_comparison")
        if network is not None and (not isinstance(network, dict) or
                set(network) != {"schema", "transport", "route_identity_sha256", "latency_semantics", "measurement_method"}
                or network.get("schema") != NETWORK_SCHEMA or not _sha(network.get("route_identity_sha256"))
                or network.get("latency_semantics") not in ("round_trip", "one_way", "no_latency_claim")
                or any(not isinstance(network.get(name), str) or not network[name] for name in ("transport", "measurement_method"))):
            errors.append("network route/latency comparison contract invalid")
        run = p.get("run", {})
        if (not _integer(run.get("prompt_tokens"), 1) or not _integer(run.get("max_new"), 2)
                or run.get("warm_reps") != 3 or run.get("warmup_reps") != 1
                or run.get("temperature") != 0.0 or run.get("seed") != 0
                or run.get("eos_policy") != "fixed_length_ignore_eos"
                or run.get("mode") not in ("greedy", "dspark", "pipelined")):
            errors.append("run context/generation/warmup/sampling protocol absent or invalid")
        if run.get("speed_metric") != "complete_suite_median_warm_committed_decode_tok_s":
            errors.append("acceptance speed metric absent or unsupported")
        prompts = p.get("prompts", [])
        if len(prompts) != 4 or [x.get("workload") for x in prompts] != list(WORKLOADS):
            errors.append("complete ordered code/math/prose/agentic prompt set is required")
        for prompt in prompts:
            ids = prompt.get("token_ids", [])
            if (not ids or len(ids) != run.get("prompt_tokens") or not all(_integer(t) for t in ids)
                    or prompt.get("sha256") != digest(ids) or prompt.get("task") != PROMPTS.get(prompt.get("workload"))):
                errors.append("prompt IDs/content do not match frozen workload protocol")
                break
        env = p.get("env", {})
        if (not all(k in env for k in DEFAULT_ENV) or not all(isinstance(k, str) and
                k.startswith("V4_") and isinstance(v, str) for k, v in env.items())):
            errors.append("explicit V4 kernel/graph/quantization environment is incomplete")
        if env.get("V4_KERNELS") != "tilelang" or env.get("V4_MAX_BATCH") != "1":
            errors.append("hardware acceptance requires GPU tilelang kernels and batch one")
        if env.get("V4_EXPERT_PLACEMENT") not in ("gpu", "ram"):
            errors.append("expert placement must explicitly identify GPU or local RAM pools")
        if any(int(env.get(key, "-1")) < 0 for key in
                ("V4_EXPERT_CACHE_SLOTS", "V4_EXPERT_CACHE_MIB", "V4_EXPERT_CACHE_RESERVE_MIB")):
            errors.append("expert cache budgets cannot be negative")
        if env.get("V4_EXPERT_PLACEMENT") == "ram" and (
                env.get("V4_MOE_IN_GRAPH") not in ("", "0")
                or env.get("V4_DSPARK_MOE") not in ("", "0")):
            errors.append("RAM experts require explicit dynamic-MoE graph seam and local MTP dispatch")
        if int(env.get("V4_MAX_SEQ", 0)) < run.get("prompt_tokens", 0) + run.get("max_new", 0) + 64:
            errors.append("V4_MAX_SEQ does not cover context+generation+speculative margin")
        nodes = p.get("hardware", [])
        if len(nodes) not in TARGETS:
            errors.append("hardware must identify exactly four or six single-GPU stages")
        identities, keys, hosts = set(), set(), set()
        isolation = p.get("isolation", "host")
        if isolation not in ("host", "none"):
            errors.append("benchmark isolation must be host or none")
        cursor = 0
        for node in nodes:
            required = ("node_id", "host_id", "gpu_uuid", "gpu_name", "driver", "cuda", "torch", "tilelang",
                        "signer_pubkey", "process_run_id")
            if any(not isinstance(node.get(k), str) or not node[k].strip() for k in required):
                errors.append("node hardware/software/signing/process identity incomplete")
            for field, expected in (("checkpoint_sha256", ck.get("sha256")),
                                    ("config_sha256", ck.get("config_sha256")),
                                    ("engine_source_sha256", source.get("sha256")),
                                    ("stage_env_sha256", digest(env))):
                if node.get(field) != expected:
                    errors.append(f"node {node.get('node_id')}: declared {field} differs from protocol")
            if not re.search(r"\bRTX\s+5090\b", str(node.get("gpu_name", "")), re.I):
                errors.append("target hardware must be RTX 5090")
            if not _integer(node.get("vram_bytes"), 30 * 1024**3) or node["vram_bytes"] > 34 * 1024**3:
                errors.append("node must report actual 32 GB class VRAM in bytes")
            if (node.get("gpu_uuid") in identities or node.get("signer_pubkey") in keys
                    or (isolation == "host" and node.get("host_id") in hosts)):
                errors.append("each baseline stage must have a distinct GPU, signer and host")
            identities.add(node.get("gpu_uuid")); keys.add(node.get("signer_pubkey")); hosts.add(node.get("host_id"))
            lo, hi = node.get("layer_start"), node.get("layer_end")
            if not _integer(lo) or not _integer(hi, 1) or lo != cursor or hi <= lo or hi > 43:
                errors.append("hardware assignment must tile [0,43) in ring order")
            cursor = hi if _integer(hi) else -1
        if cursor != 43 or (nodes and nodes[-1].get("layer_start", 43) > 40):
            errors.append("tail must contain target layers 40, 41, 42")
    except (KeyError, ValueError, TypeError, BenchmarkError, AttributeError) as exc:
        errors.append(f"malformed protocol: {exc}")
    return errors


def require_protocol(protocol: dict) -> None:
    errors = protocol_errors(protocol)
    if errors:
        raise BenchmarkError("; ".join(errors))


def runtime_observation_errors(observation, node, protocol):
    """Check a declaration already authenticated inside its stage receipt."""
    from shard.runtime_observation import validate_runtime_observation
    try:
        value = validate_runtime_observation(observation)
        errors = []
        if any(value[name] != node[name] for name in ("node_id", "gpu_uuid", "process_run_id")):
            errors.append("runtime node/GPU/process identity differs from frozen inventory")
        if node.get("runtime_config_sha256") is not None and value["runtime_config_sha256"] != node["runtime_config_sha256"]:
            errors.append("effective runtime configuration differs from frozen inventory")
        expected = {row["name"]:row["sha256"] for row in protocol["source"]["files"]}
        required_source = {"engines/deepseek_v4/v4_pipe.py", "engines/deepseek_v4/v4_stage.py",
                           "vendor/deepseek_v4_ref/inference/model.py", "shard/pipeline_session.py", "shard/transport.py"}
        required_source |= {path for path in expected if path.startswith(("engines/deepseek_v4/v4_", "vendor/deepseek_v4_ref/"))
                            or path in {"shard/"+name+".py" for name in (
                                "receipt", "runtime_metrics", "runtime_profile", "runtime_observation", "pipeline_plan", "pipeline_session", "transport")}}
        if not required_source <= set(value["source_files"]):
            errors.append("observed source inventory omits core executed stage/protocol modules")
        if any(expected.get(path) != sha for path, sha in value["source_files"].items()):
            errors.append("observed stage source differs from frozen source bytes")
        env = value["environment"]
        def normalized(setting):
            return str(setting).removeprefix("torch.")
        if any(key not in env or normalized(env[key]) != normalized(setting) for key, setting in protocol["env"].items()):
            errors.append("observed public environment differs from frozen recipe")
        if set(env) - set(protocol["env"]):
            errors.append("observed public environment contains unfrozen runtime flags")
        for name in ("torch", "cuda", "tilelang"):
            if value["versions"][name] != node[name]:
                errors.append("observed runtime software version differs from frozen inventory")
        if value["phase"] != "job_complete":
            errors.append("runtime observation must describe the measured completed job")
        if value["kernel_backend"] != protocol["env"]["V4_KERNELS"]:
            errors.append("actual kernel backend differs from frozen recipe")
        expected_wire = "fp8" if protocol["env"]["V4_FP8_WIRE"] not in ("", "0") else "bf16"
        if value["wire_mode"] != expected_wire:
            errors.append("actual wire dtype differs from frozen recipe")
        if protocol["env"].get("V4_HADAMARD") in ("torch", "extension") and value["hadamard_backend"] != protocol["env"]["V4_HADAMARD"]:
            errors.append("actual Hadamard backend differs from explicitly selected recipe")
        if any(row["verdict"] in ("MISMATCH", "UNKNOWN") for row in value["effective_flags"].values()):
            errors.append("effective runtime audit reports a mismatched or unregistered flag")
        identity = value.get("backend_identity")
        if identity is None and protocol.get("evidence_contract", {}).get("runtime_observation_required", False):
            errors.append("frozen runtime contract requires backend source/binary identity")
        if identity is not None and identity["hadamard"]["requested"] != protocol["env"].get("V4_HADAMARD", "auto"):
            errors.append("Hadamard selection identity differs from frozen request")
        return errors
    except (ValueError, TypeError, KeyError) as exc:
        return [f"runtime observation invalid: {exc}"]


def _runtime_identity(observation):
    """Immutable settings, separate from per-job changing coverage observations."""
    fields = ("node_id", "gpu_uuid", "process_run_id", "runtime_config_sha256", "source_sha256",
              "environment", "environment_sha256", "kernel_backend", "hadamard_backend", "graph_mode",
              "wire_mode", "transport", "versions")
    identity = observation.get("backend_identity")
    return {**{name: observation[name] for name in fields},
            "backend_identity": {"hadamard": {key: value for key, value in identity["hadamard"].items() if key != "reason"}} if identity else None,
            "effective_settings": {name: {field: row[field] for field in ("requested", "parsed")}
                                   for name, row in observation["effective_flags"].items()}}


def measure_job(adapter, prompt: dict, protocol: dict, *, run_id: str, phase: str,
                rep: int, mode: str, clock: Callable[[], float] = time.perf_counter) -> dict:
    """Time committed callbacks, never drafted frames. Sweep receipts after generation stops."""
    nonce = secrets.token_hex(32)
    job_id = f"{run_id}/{prompt['workload']}/{phase}/{rep}/{mode}"
    start = clock()
    events = []

    def on_token(token):
        events.append((int(token), clock() - start))

    result = adapter.generate(prompt["token_ids"], protocol["run"]["max_new"], mode=mode,
                              nonce=nonce, job_id=job_id, on_token=on_token)
    returned_clock = clock()
    returned_at = returned_clock - start  # includes reset/prefill/last-token return/drain; excludes sweep
    if result.get("receipt_sweep_s", 0) != 0:
        raise BenchmarkError("benchmark generation must disable internal receipt sweep; external sweep is timed separately")
    tokens = list(result.get("tokens", []))
    if not result.get("ok") or tokens != [token for token, _ in events]:
        raise BenchmarkError("coordinator result is not exactly its committed callback token stream")
    if len(tokens) != protocol["run"]["max_new"]:
        raise BenchmarkError("generation ended before the fixed committed-token length")
    sweep_start = clock()
    receipts, sweep_ok = adapter.sweep(nonce)
    sweep_end = clock()
    sweep_s = sweep_end - sweep_start
    service_end = clock()
    service_s = service_end - start
    offsets = [offset for _, offset in events]
    decode_s = offsets[-1] - offsets[0]
    sample = {
        "workload": prompt["workload"], "phase": phase, "rep": rep, "mode": mode,
        "job_id": job_id, "swarm_id": adapter.swarm_id, "nonce": nonce,
        "prompt_sha256": prompt["sha256"], "prompt_tokens": len(prompt["token_ids"]),
        "tokens": tokens, "output_sha256": digest(tokens), "committed_tokens": len(tokens),
        "measurement": {"clock": "perf_counter", "elapsed_s": returned_at,
                        "first_token_s": offsets[0], "last_token_s": offsets[-1],
                        "commit_offsets_s": offsets, "receipt_sweep_s": sweep_s,
                        "drain_s": returned_at-offsets[-1], "full_service_s": service_s,
                        "observer_overhead_s": (sweep_start-returned_clock)+(service_end-sweep_end),
                        "decode_s": decode_s,
                        "end_to_end_committed_tok_s": len(tokens) / returned_at,
                        "full_service_committed_tok_s": len(tokens)/service_s,
                        "decode_committed_tok_s": (len(tokens) - 1) / decode_s if decode_s > 0 else None},
        "coordinator_stats": {k: v for k, v in result.items()
                              if k not in ("tokens", "receipts", "receipts_ok")},
        "receipt_sweep_ok": sweep_ok, "receipts": receipts,
    }
    if result.get("coordinator_counters") is not None:
        from shard.benchmark_metrics import make_coordinator_diagnostics
        sample["coordinator_diagnostics"] = make_coordinator_diagnostics(mode,
            committed_tokens=len(tokens), prefill_tokens=result.get("prefill_committed_tokens", 1),
            counters=result["coordinator_counters"], inflight_intervals=result.get("inflight_intervals"),
            timing={"request_elapsed_s": returned_at, "first_token_s": offsets[0], "last_token_s": offsets[-1],
                    "drain_s": returned_at-offsets[-1], "receipt_sweep_s": sweep_s},
            backend=result.get("backend_identity"))
    return sample


def run_suite(adapter, protocol: dict, *, fresh_ring: bool,
              clock: Callable[[], float] = time.perf_counter,
              on_sample: Callable[[dict], None] | None = None) -> dict:
    require_protocol(protocol)
    run_id = secrets.token_hex(16)
    report = {
        "schema": REPORT_SCHEMA, "protocol": protocol, "protocol_sha256": protocol["sha256"],
        "measurement_definitions": dict(MEASUREMENT_DEFINITIONS),
        "run_id": run_id, "created_utc": _utc(), "backend": adapter.backend,
        "artifact_verification": getattr(adapter, "artifact_verification", {}),
        "cold_start": {"fresh_ring_operator_assertion": bool(fresh_ring),
                       "scope": "first_request_after_process_and_model_start",
                       "cache_reset": "existing reset clears sequence state; does not evict expert cache"},
        "samples": [], "historical_reference": dict(HISTORICAL),
    }
    for prompt_index, prompt in enumerate(protocol["prompts"]):
        # Control runs AFTER candidate runs: a greedy pass must not warm the first cold request.
        jobs = [("cold" if prompt_index == 0 else "warmup", 0, protocol["run"]["mode"])]
        jobs += [("warm", rep, protocol["run"]["mode"]) for rep in range(1, 4)]
        jobs += [("greedy_control", 0, "greedy")]
        for phase, rep, mode in jobs:
            sample = measure_job(adapter, prompt, protocol, run_id=run_id, phase=phase,
                                 rep=rep, mode=mode, clock=clock)
            report["samples"].append(sample)
            if on_sample:
                on_sample(report)
    return report


class LiveRingAdapter:
    """Weightless coordinator for an already-running deployment; no CUDA model import here."""
    backend = "live_ring"

    def __init__(self, protocol: dict, directory: Path, head: str, tail: str,
                 *, swarm_id="v4-benchmark", timeout=600.0, retry_s=300.0,
                 deployment_plan=None, coordinator_key=None):
        require_protocol(protocol)
        self.artifact_verification = verify_checkpoint(directory, protocol["checkpoint"])
        source = source_identity()
        if source != protocol["source"]:
            raise BenchmarkError("engine source differs from frozen protocol")
        self.artifact_verification["engine_source_verified"] = True
        extra = {k for k in os.environ if k.startswith("V4_")} - set(protocol["env"]) - {"V4_DIR"}
        if extra:
            raise BenchmarkError(f"unfrozen V4 environment variables: {sorted(extra)}")
        if "v4_pipe" in sys.modules:
            raise BenchmarkError("start a fresh benchmark process: v4_pipe flags are import-time")
        for key, value in protocol["env"].items():
            os.environ[key] = value
        os.environ["V4_DIR"] = str(directory.resolve())
        # Imported only once all identities are frozen, before opening a ring socket.
        for p in (ROOT, ROOT / "engines" / "deepseek_v4"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        self.vp = importlib.import_module("v4_pipe")
        self.timeout, self.swarm_id, self.layer_count = timeout, swarm_id, protocol["layer_count"]
        self.protocol = protocol
        session_options = {}
        if deployment_plan is not None:
            from shard.pipeline_plan import load_plan, validate_plan
            from shard.pipeline_session import SessionConfig
            from shard.manifest import load_key
            plan = load_plan(deployment_plan) if isinstance(deployment_plan, (str, Path)) else validate_plan(deployment_plan)
            if (plan["n_layers"] != self.layer_count or plan["coordinator"]["head"] != head or plan["coordinator"]["tail"] != tail):
                raise BenchmarkError("benchmark endpoints/layers differ from deployment plan")
            expected = {(n["node_id"], n["gpu_uuid"], n["signer_pubkey"], n["layer_start"], n["layer_end"])
                        for n in protocol["hardware"]}
            actual = {(n["node_id"], n["gpu_uuid"], n["signer_pubkey"], n["lo"], n["hi"]) for n in plan["stages"]}
            if actual != expected or plan.get("model_cohort", {}).get("config_sha256") != protocol["checkpoint"]["config_sha256"]:
                raise BenchmarkError("benchmark hardware/config differs from deployment plan")
            if protocol["checkpoint"].get("identity_kind") == "logical-tensor-catalogue":
                checkpoint = protocol["checkpoint"]
                cohort = plan.get("model_cohort", {})
                if (cohort.get("checkpoint_id"), cohort.get("manifest_sha256")) != (
                        checkpoint["checkpoint_id"], checkpoint["manifest_sha256"]):
                    raise BenchmarkError("benchmark global checkpoint differs from deployment cohort")
                self.vp.verify_cohort_directory(directory, cohort)
            key_path = coordinator_key or os.environ.get("SHARD_COORDINATOR_KEY")
            if not key_path:
                raise BenchmarkError("strict benchmark requires a coordinator signing key")
            session_options["session_config"] = SessionConfig.from_plan(plan, -1,
                ttl_s=min(3600, max(30, timeout * 2)), caller_key=load_key(key_path))
            self.swarm_id = plan["ring_id"]
        if protocol["checkpoint"].get("identity_kind") == "logical-tensor-catalogue" and not session_options:
            raise BenchmarkError("logical catalogue benchmarks require a strict deployment plan")
        self.pipe, self.ret = self.vp.connect_ring(head, tail, timeout=timeout,
                                                  token=self.vp.SWARM_TOKEN, retry_s=retry_s, **session_options)
        if protocol["checkpoint"].get("identity_kind") == "logical-tensor-catalogue":
            self.artifact_verification["authenticated_cohort_verified"] = bool(session_options)

    def generate(self, prompt_ids, max_new, *, mode, nonce, job_id, on_token):
        self._job_id = job_id
        methods = {"greedy": self.vp.coordinate, "dspark": self.vp.coordinate_dspark,
                   "pipelined": self.vp.coordinate_dspark_pipelined}
        kw = dict(eos_ids=(), nonce=nonce, swarm_id=self.swarm_id, job_id=job_id,
                  layer_count=self.layer_count, receipts=False, timeout=self.timeout,
                  on_token=on_token, strict_job_binding=True)
        if mode == "greedy":
            kw.update(temp=0.0, seed=0)
        elif mode == "pipelined":
            kw.update(depth=int(self.protocol["env"]["V4_SPEC_DEPTH"]),
                      lazy=self.protocol["env"]["V4_LAZY_DRAFT"] not in ("", "0"),
                      floor=int(self.protocol["env"]["V4_REFILL_FLOOR"]))
        return methods[mode](self.pipe, self.ret, prompt_ids, max_new, **kw)

    def sweep(self, nonce):
        assignment = {node["signer_pubkey"]: (node["layer_start"], node["layer_end"])
                      for node in self.protocol["hardware"]}
        return self.vp._sweep_receipts(self.pipe, self.ret, self.layer_count, nonce,
                                       expected_by_signer=assignment, swarm_id=self.swarm_id,
                                       job_id=self._job_id)

    def close(self):
        # Disconnect the coordinator; do not send stop to the user's running stages.
        self.pipe.close()
        self.ret.close()


def _sample_errors(sample: dict, protocol: dict, prompt: dict) -> list[str]:
    errors = []
    run = protocol["run"]
    tokens = sample.get("tokens")
    if (not isinstance(tokens, list) or not all(_integer(t) for t in tokens)
            or len(tokens) != run["max_new"] or sample.get("committed_tokens") != len(tokens)
            or sample.get("output_sha256") != digest(tokens)):
        errors.append("fixed-length committed token IDs/hash absent or inconsistent")
    if sample.get("prompt_sha256") != prompt["sha256"] or sample.get("prompt_tokens") != run["prompt_tokens"]:
        errors.append("sample uses a different prompt/context")
    m = sample.get("measurement", {})
    offsets = m.get("commit_offsets_s", [])
    elapsed = m.get("elapsed_s")
    if (m.get("clock") != "perf_counter" or not _positive(elapsed)
            or len(offsets) != run["max_new"] or not all(_positive(t) for t in offsets)
            or any(b < a for a, b in zip(offsets, offsets[1:]))):
        return errors + ["monotonic committed-token measurement evidence absent/invalid"]
    if (m.get("first_token_s") != offsets[0] or m.get("last_token_s") != offsets[-1]
            or offsets[-1] > elapsed or m.get("decode_s") != offsets[-1] - offsets[0]
            or not _integer(sample.get("committed_tokens"), 2)):
        errors.append("measurement boundaries/counts are inconsistent")
    expected_speed = run["max_new"] / elapsed
    if (not _positive(m.get("end_to_end_committed_tok_s"))
            or not math.isclose(m["end_to_end_committed_tok_s"], expected_speed, rel_tol=1e-9)):
        errors.append("speed was not calculated from final committed tokens and measured elapsed time")
    decode_s = offsets[-1] - offsets[0]
    decode_speed = m.get("decode_committed_tok_s")
    if (decode_s <= 0 or not _positive(decode_speed)
            or not math.isclose(decode_speed, (run["max_new"] - 1) / decode_s, rel_tol=1e-9)):
        errors.append("decode speed must use N-1 committed tokens between first and last callbacks")
    if type(m.get("receipt_sweep_s")) not in (int, float) or not math.isfinite(m["receipt_sweep_s"]) or m["receipt_sweep_s"] < 0:
        errors.append("receipt sweep duration is absent/invalid")
    if "drain_s" in m and (not _duration_valid(m["drain_s"]) or
            not math.isclose(m["drain_s"], elapsed-offsets[-1], rel_tol=1e-8, abs_tol=1e-8)):
        errors.append("drain must measure generation return minus the final commit")
    if "full_service_s" in m and (not _positive(m["full_service_s"]) or
            not _duration_valid(m.get("observer_overhead_s")) or
            not math.isclose(m["full_service_s"], elapsed+m["receipt_sweep_s"]+m["observer_overhead_s"], rel_tol=1e-8, abs_tol=1e-8) or
            not math.isclose(m.get("full_service_committed_tok_s", -1), run["max_new"]/m["full_service_s"], rel_tol=1e-8)):
        errors.append("full service must include generation/drain and separate receipt sweep")
    if sample.get("coordinator_diagnostics") is not None:
        try:
            from shard.benchmark_metrics import COUNTERS, validate_coordinator_diagnostics
            diagnostics = validate_coordinator_diagnostics(sample["coordinator_diagnostics"])
            if diagnostics["counts"]["committed_total_tokens"] != len(tokens) or diagnostics["mode"] != sample["mode"]:
                errors.append("coordinator raw counts/mode differ from actual committed sample")
            if protocol.get("evidence_contract", {}).get("coordinator_diagnostics_required", False) and diagnostics["missing_counters"]:
                errors.append("frozen diagnostic contract requires complete raw coordinator counters")
            for diagnostic_field, measurement_field in (("request_elapsed_s", "elapsed_s"), ("first_token_s", "first_token_s"),
                    ("last_token_s", "last_token_s"), ("drain_s", "drain_s"), ("receipt_sweep_s", "receipt_sweep_s")):
                if diagnostics["timing"][diagnostic_field] != m.get(measurement_field):
                    errors.append("coordinator diagnostic timing differs from measured callback/return/sweep boundaries")
                    break
            raw_stats = sample.get("coordinator_stats", {})
            if diagnostics["inflight_intervals"] != raw_stats.get("inflight_intervals"):
                errors.append("inflight diagnostics differ from raw coordinator intervals")
            if any(diagnostics["counts"][name] != raw_stats.get("coordinator_counters", {}).get(name) for name in COUNTERS):
                errors.append("coordinator diagnostics differ from raw result counters")
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"coordinator diagnostics invalid: {exc}")
    elif protocol.get("evidence_contract", {}).get("coordinator_diagnostics_required", False):
        errors.append("frozen diagnostic contract requires raw coordinator counters")
    return errors


def evaluate_report(report: dict, expected_protocol: dict | None = None) -> dict:
    """Independently reverify raw evidence; never trust receiptsOk/parity/speed booleans."""
    failed, missing = [], []
    workloads = {}
    protocol = report.get("protocol", {}) if isinstance(report, dict) else {}
    errors = protocol_errors(protocol)
    if errors:
        return {"status": "unverified", "errors": errors, "target_tok_s": None,
                "workloads": {}, "hardware_evidence": "operator_inventory", "speed_pass": False}
    if report.get("schema") != REPORT_SCHEMA or report.get("protocol_sha256") != protocol["sha256"]:
        failed.append("report schema/protocol binding invalid")
    if report.get("measurement_definitions") is not None and report["measurement_definitions"] != MEASUREMENT_DEFINITIONS:
        failed.append("measurement definitions differ from the supported timing/counter contract")
    if expected_protocol is not None and protocol != expected_protocol:
        failed.append("report does not use the independently supplied frozen protocol")
    if report.get("backend") != "live_ring":
        missing.append("not a live hardware benchmark; mock/CPU execution cannot pass the speed gate")
    evidence = report.get("artifact_verification", {})
    logical = protocol["checkpoint"].get("identity_kind") == "logical-tensor-catalogue"
    required_artifacts = (("global_catalog_verified", "authenticated_cohort_verified") if logical else
                          ("checkpoint_bytes_verified",)) + ("config_bytes_verified", "tokenizer_bytes_verified", "engine_source_verified")
    if not all(evidence.get(k) is True for k in required_artifacts):
        missing.append("live checkpoint/config/tokenizer/source verification absent")
    if logical:
        scope = evidence.get("local_weight_scope")
        if evidence.get("checkpoint_id") != protocol["checkpoint"]["checkpoint_id"] or scope not in (
                "complete packed payload hashes", "assigned stage payload hashes", "pinned catalogue and local model assets only"):
            missing.append("local/global artifact verification scope absent or invalid")
        if scope != "complete packed payload hashes" and evidence.get("checkpoint_bytes_verified") is True:
            failed.append("partial/coordinator assets falsely claim complete local checkpoint verification")
    if report.get("cold_start", {}).get("fresh_ring_operator_assertion") is not True:
        missing.append("fresh process/model cold-start assertion absent")
    nodes = protocol["hardware"]
    assignments = {n["signer_pubkey"]: (n["layer_start"], n["layer_end"]) for n in nodes}
    hardware_by_signer = {node["signer_pubkey"]:node for node in nodes}
    runtime_observed = 0
    runtime_identities = {}
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from shard.receipt import verify_coverage, wire_receipt
    except ImportError:
        missing.append("cryptography dependency required to verify raw signatures")
        verify_coverage = wire_receipt = None
    samples = report.get("samples", [])
    if not isinstance(samples, list):
        samples = []
        missing.append("sample list absent")
    nonces, job_ids = set(), set()
    try:
        for prompt_index, prompt in enumerate(protocol["prompts"]):
            name = prompt["workload"]
            group = [s for s in samples if s.get("workload") == name]
            expected_slots = {("cold" if prompt_index == 0 else "warmup", 0)}
            expected_slots.update(("warm", rep) for rep in range(1, 4))
            expected_slots.add(("greedy_control", 0))
            slots = [(s.get("phase"), s.get("rep")) for s in group]
            if len(group) != 5 or set(slots) != expected_slots:
                missing.append(f"{name}: cold/warmup, three warm reps and greedy control required")
                continue
            greedy = next(s for s in group if s["phase"] == "greedy_control")
            for s in group:
                label = f"{name}/{s['phase']}/{s['rep']}"
                failed.extend(f"{label}: {e}" for e in _sample_errors(s, protocol, prompt))
                wanted_mode = "greedy" if s["phase"] == "greedy_control" else protocol["run"]["mode"]
                if s.get("mode") != wanted_mode:
                    failed.append(f"{label}: coordinator mode differs from frozen protocol")
                if s.get("tokens") != greedy.get("tokens"):
                    failed.append(f"{label}: committed token IDs differ from same-ring greedy control")
                nonce, job_id = s.get("nonce"), s.get("job_id")
                if (not isinstance(nonce, str) or len(nonce) < 32 or nonce in nonces
                        or not isinstance(job_id, str) or not job_id or job_id in job_ids):
                    failed.append(f"{label}: unique per-job nonce/job identity absent")
                nonces.add(nonce); job_ids.add(job_id)
                raw = s.get("receipts")
                if not raw:
                    missing.append(f"{label}: raw signed receipts absent")
                elif verify_coverage is not None:
                    try:
                        wired = [wire_receipt(r) for r in raw]
                        verify_coverage(wired, 43, expected_by_signer=assignments,
                                        expected_nonce=nonce, check_chain=True)
                        if any(r.get("job_id") != job_id or r.get("swarm_id") != s.get("swarm_id") for r in wired):
                            raise BenchmarkError("signed job/swarm identity differs from measured job")
                        for receipt in wired:
                            observation = receipt.get("runtime_observation")
                            if observation is not None:
                                runtime_observed += 1
                                problems = runtime_observation_errors(observation, hardware_by_signer[receipt["pubkey"]], protocol)
                                if problems:
                                    raise BenchmarkError("; ".join(problems))
                                identity = _runtime_identity(observation)
                                previous = runtime_identities.setdefault(observation["node_id"], identity)
                                if previous != identity:
                                    raise BenchmarkError("effective runtime settings/backends changed during the frozen suite")
                            elif protocol.get("evidence_contract", {}).get("runtime_observation_required", False):
                                missing.append(f"{label}: signed effective runtime observation absent")
                        if protocol["env"].get("V4_RUNTIME_METRICS") not in ("", "0"):
                            if any(not r.get("runtime_metrics") for r in wired):
                                missing.append(f"{label}: enabled signed runtime metrics absent")
                            elif any(r["runtime_metrics"].get("mode") != (
                                    "gpu_expert_cache" if protocol["env"]["V4_EXPERT_PLACEMENT"] == "ram"
                                    else "gpu_resident") for r in wired):
                                raise BenchmarkError("hardware acceptance requires GPU runtime telemetry")
                            else:
                                from shard.runtime_metrics import validate_runtime_metrics
                                for r in wired:
                                    validate_runtime_metrics(r["runtime_metrics"])
                    except Exception as exc:
                        failed.append(f"{label}: signature/signer/nonce/coverage/chain: {exc}")
            warm = [s for s in group if s["phase"] == "warm"]
            speeds = [s["measurement"]["decode_committed_tok_s"] for s in warm]
            if not all(_positive(speed) for speed in speeds):
                failed.append(f"{name}: warm speeds invalid")
                continue
            workloads[name] = {
                "median_decode_committed_tok_s": statistics.median(speeds),
                "warm_decode_reps_tok_s": speeds,
                "warm_decode_range_tok_s": [min(speeds), max(speeds)],
                "median_end_to_end_committed_tok_s": statistics.median(
                    s["measurement"]["end_to_end_committed_tok_s"] for s in warm),
                "median_first_token_s": statistics.median(s["measurement"]["first_token_s"] for s in warm),
                "token_parity": all(s.get("tokens") == greedy.get("tokens") for s in group),
            }
            from shard.benchmark_metrics import percentile
            raw_timings = {field:[s["measurement"][field] for s in warm if s["measurement"].get(field) is not None]
                           for field in ("first_token_s", "decode_s", "elapsed_s", "drain_s", "receipt_sweep_s", "full_service_s")}
            workloads[name]["timing_distributions"] = {field:{"samples":values, "p50":percentile(values,50), "p95":percentile(values,95)}
                                                       for field, values in raw_timings.items()}
            workloads[name]["inter_token_latency_s"] = {"samples":[b-a for s in warm for a,b in zip(
                s["measurement"]["commit_offsets_s"], s["measurement"]["commit_offsets_s"][1:])]}
            intervals = workloads[name]["inter_token_latency_s"]["samples"]
            workloads[name]["inter_token_latency_s"].update(p50=percentile(intervals,50), p95=percentile(intervals,95))
            diagnostic_values = {field:[s["coordinator_diagnostics"]["derived"][field] for s in warm
                if s.get("coordinator_diagnostics") is not None and s["coordinator_diagnostics"]["derived"][field] is not None]
                for field in ("g_cycle", "g_frame", "acceptance_ratio", "frame_waste_ratio", "inflight_time_avg", "max_inflight")}
            workloads[name]["coordinator_distributions"] = {field:{"samples":values, "p50":percentile(values,50), "p95":percentile(values,95)}
                                                            for field, values in diagnostic_values.items()}
            from shard.benchmark_metrics import COUNTERS
            count_values = {field:[s["coordinator_diagnostics"]["counts"][field] for s in warm
                if s.get("coordinator_diagnostics") is not None and s["coordinator_diagnostics"]["counts"][field] is not None] for field in COUNTERS}
            workloads[name]["coordinator_count_distributions"] = {field:{"samples":values, "p50":percentile(values,50), "p95":percentile(values,95)}
                                                                  for field, values in count_values.items()}
        if len(samples) != 20 or any(s.get("workload") not in WORKLOADS for s in samples):
            failed.append("report has extra/missing benchmark jobs")
    except (KeyError, ValueError, TypeError, AttributeError, BenchmarkError) as exc:
        failed.append(f"malformed benchmark evidence: {exc}")
    target = TARGETS[len(nodes)]
    all_warm = [speed for v in workloads.values() for speed in v["warm_decode_reps_tok_s"]]
    speed = statistics.median(all_warm) if all_warm else None
    speed_pass = bool(len(workloads) == 4 and speed is not None and
                      (speed >= target or math.isclose(speed, target, rel_tol=1e-9)))
    evidence_errors = list(failed)
    # Distinguish structurally valid measurement evidence from final speedline target attainment.
    # Below-target baselines remain genuine valid evidence for comparative A/B analysis.
    has_evidence = len(missing) == 0 and len(evidence_errors) == 0
    return {
        "status": "failed" if evidence_errors else "unverified" if missing else "passed" if speed_pass else "valid_baseline",
        "errors": evidence_errors, "missing_evidence": missing, "target_tok_s": target,
        "complete_suite_median_warm_decode_tok_s": speed, "workloads": workloads,
        "speed_pass": speed_pass and has_evidence,
        "valid_evidence": has_evidence,
        "parity_scope": "committed token IDs vs same-ring greedy, not proof of hidden-state bit equality",
        "hardware_evidence": "operator_inventory_pinned_to_receipt_signers; not remote hardware attestation",
        "runtime_evidence": {"signed_observation_count": runtime_observed,
            "scope": "worker-signed effective runtime declarations" if runtime_observed else "operator declarations only; actual runtime not observed",
            "stage_identities": runtime_identities,
            "not_remote_execution_attestation": True},
    }


def compare_reports(before: dict, after: dict, *, vary=None) -> dict:
    """Compare matching workload/model recipes; decouple target passing from valid A/B comparison."""
    old, new = evaluate_report(before), evaluate_report(after)
    p, q = before.get("protocol", {}), after.get("protocol", {})
    if protocol_errors(p) or protocol_errors(q):
        return {"status":"unverified", "reason":"comparison protocol invalid", "before":old, "after":new}
    # Core model, context length, and prompt recipe must match, allowing intentional runtime cache/knob evolution
    same_recipe = (all(p.get(k) == q.get(k) for k in ("model_id", "prompts", "run"))
                   and _checkpoint_identity(p.get("checkpoint", {})) == _checkpoint_identity(q.get("checkpoint", {}))
                   and p.get("env", {}).get("V4_MAX_SEQ") == q.get("env", {}).get("V4_MAX_SEQ"))
    if not same_recipe:
        return {"status": "unverified", "reason": "model/prompt/context/run recipe differs",
                "before": old, "after": new}
    if vary is not None and (not isinstance(vary, str) or vary != "source" and vary != "network" and not re.fullmatch(r"env:V4_[A-Z0-9_]+", vary)):
        return {"status":"unverified", "reason":"vary must be source, network or env:V4_FLAG", "before":old, "after":new}
    source_changed = p.get("source", {}).get("sha256") != q.get("source", {}).get("sha256")
    ignored = {"process_run_id", "stage_env_sha256", "engine_source_sha256"}
    if vary == "source" or isinstance(vary, str) and vary.startswith("env:"):
        ignored |= {"runtime_config_sha256", "effective_flags_sha256"}
    if vary == "source":
        ignored |= {"hadamard_backend", "graph_mode", "kernel_backend"}
    elif vary in {"env:V4_HADAMARD", "env:V4_HADAMARD_BACKEND"}:
        ignored.add("hadamard_backend")
    elif vary == "env:V4_CUDA_GRAPH":
        ignored.add("graph_mode")
    hardware_changed = [{k:v for k,v in node.items() if k not in ignored} for node in p.get("hardware", [])] != [
        {k:v for k,v in node.items() if k not in ignored} for node in q.get("hardware", [])]
    env_changes = sorted(key for key in set(p.get("env", {})) | set(q.get("env", {})) if p.get("env", {}).get(key) != q.get("env", {}).get(key))
    network_changed = p.get("network_comparison") != q.get("network_comparison")
    actual_changes = (["source"] if source_changed else []) + (["network"] if network_changed else []) + ["env:"+key for key in env_changes]
    reason = None
    if hardware_changed or before.get("backend") != after.get("backend"):
        reason = "hardware/software assignment or backend differs; not a controlled single-variable comparison"
    elif not p.get("network_comparison") or not q.get("network_comparison"):
        reason = "network route/latency measurement contract is not frozen"
    elif any(p["network_comparison"][key] != q["network_comparison"][key] for key in ("latency_semantics", "measurement_method")):
        reason = "network latency semantics or measurement method differs"
    elif actual_changes != ([] if vary is None else [vary]):
        reason = "undeclared or multiple source/environment/network changes: " + str(actual_changes)
    observations_old = old.get("runtime_evidence", {}).get("stage_identities", {})
    observations_new = new.get("runtime_evidence", {}).get("stage_identities", {})
    if reason is None and (old.get("runtime_evidence", {}).get("signed_observation_count") != len(before.get("samples", []))*len(p.get("hardware", []))
            or new.get("runtime_evidence", {}).get("signed_observation_count") != len(after.get("samples", []))*len(q.get("hardware", []))
            or set(observations_old) != {node["node_id"] for node in p.get("hardware", [])}
            or set(observations_new) != set(observations_old)
            or any(row.get("backend_identity") is None for row in (*observations_old.values(), *observations_new.values()))):
        reason = "controlled comparison requires signed effective runtime observations for every stage/job"
    def comparable_runtime(identity, protocol, other):
        result = {name: value for name, value in identity.items() if name != "process_run_id"}
        if vary == "source":
            result = {name:value for name,value in result.items() if name not in
                      {"runtime_config_sha256", "source_sha256", "kernel_backend", "hadamard_backend", "graph_mode", "effective_settings"}}
            # Repo-owned Python code may change in a source experiment. Retain
            # external .so/.pyd and dependency versions, even though config SHA changes.
            backend = result.get("backend_identity", {}).get("hadamard")
            if identity["hadamard_backend"] != other["hadamard_backend"]:
                result.pop("backend_identity", None)
            elif backend is not None:
                source_rows = protocol["source"]["files"]
                repo_files = {(Path(row["name"]).name, row["sha256"]) for row in source_rows}
                hadamard = dict(backend)
                hadamard["module_files"] = [row for row in backend["module_files"] if (row["name"], row["sha256"]) not in repo_files]
                if len(hadamard["module_files"]) != len(backend["module_files"]) and backend["backend"] == "torch":
                    hadamard.pop("source_sha256", None)
                result["backend_identity"] = {"hadamard": hadamard}
        elif isinstance(vary, str) and vary.startswith("env:"):
            changed_flag = vary[4:]
            result.pop("runtime_config_sha256", None)
            result.pop("environment_sha256", None)
            result["environment"] = {name:value for name,value in result["environment"].items() if name != changed_flag}
            result["effective_settings"] = {name:value for name,value in result["effective_settings"].items() if name != changed_flag}
            if changed_flag == "V4_HADAMARD":
                result.pop("hadamard_backend", None)
                if identity["hadamard_backend"] != other["hadamard_backend"]:
                    result.pop("backend_identity", None)
                else:
                    backend = result.get("backend_identity")
                    if backend is not None:
                        result["backend_identity"] = {"hadamard": {key:value for key,value in backend["hadamard"].items() if key != "requested"}}
            if changed_flag == "V4_CUDA_GRAPH":
                result.pop("graph_mode", None)
            if changed_flag == "V4_FP8_WIRE":
                result.pop("wire_mode", None)
        return result
    if reason is None and any(comparable_runtime(observations_old[node], p, observations_new[node]) !=
            comparable_runtime(observations_new[node], q, observations_old[node]) for node in observations_old):
        reason = "observed backends/settings differ beyond the declared variable"
    if reason:
        return {"status":"unverified", "reason":reason, "changes":actual_changes, "before":old, "after":new}
    outputs_before = {sample["workload"]:sample["tokens"] for sample in before.get("samples", []) if sample.get("phase") == "greedy_control"}
    outputs_after = {sample["workload"]:sample["tokens"] for sample in after.get("samples", []) if sample.get("phase") == "greedy_control"}
    if outputs_before != outputs_after:
        return {"status":"unverified", "reason":"before/after committed output IDs differ despite matched prompts/recipe", "before":old, "after":new}
    ratios = {name: new["workloads"][name]["median_decode_committed_tok_s"] /
                    old["workloads"][name]["median_decode_committed_tok_s"]
              for name in WORKLOADS if name in old.get("workloads", {}) and name in new.get("workloads", {})}
    valid_pair = old.get("valid_evidence", False) and new.get("valid_evidence", False)
    status = ("verified_target_pass" if old.get("speed_pass") and new.get("speed_pass")
              else "verified_comparison" if valid_pair
              else "unverified")
    return {"status": status,
            "workload_speed_ratios": ratios, "before": old, "after": new,
            "declared_variable": vary, "changes": actual_changes,
            "frozen_controls": {"before": {"source":p.get("source"), "hardware":p.get("hardware"), "env":p.get("env"), "network":p.get("network_comparison")},
                                "after": {"source":q.get("source"), "hardware":q.get("hardware"), "env":q.get("env"), "network":q.get("network_comparison")}},
            "source_changed": p.get("source") != q.get("source"),
            "hardware_changed": p.get("hardware") != q.get("hardware"),
            "cache_knob_changed": p.get("expert_cache") != q.get("expert_cache")}


def _read(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, obj):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(output)


def _prepare(args):
    inventory = _checkpoint_identity(checkpoint_inventory(args.dir))
    source = source_identity()
    env = dict(DEFAULT_ENV)
    if args.env_file:
        env.update(_read(args.env_file))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from engines.deepseek_v4.v4_tokenizer import load_v4_tokenizer
    tokenizer = load_v4_tokenizer(args.dir)
    enc_dir = ROOT / "vendor" / "deepseek_v4_ref" / "encoding"
    sys.path.insert(0, str(enc_dir))
    from encoding_dsv4 import encode_messages
    render = lambda text: encode_messages([{"role": "user", "content": text}], "chat")
    prompts = freeze_prompts(lambda text: tokenizer.encode(text, add_special_tokens=False), render,
                             args.prompt_tokens)
    p = seal_protocol({
        "schema": PROTOCOL_SCHEMA, "model_id": inventory.get("global_catalog", {}).get("model_id", MODEL_ID), "layer_count": 43,
        "checkpoint": inventory, "source": source, "env": env, "hardware": _read(args.hardware),
        "isolation": args.isolation,
        "prompts": prompts,
        "evidence_contract": {"coordinator_diagnostics_required": True, "runtime_observation_required": True},
        **({"network_comparison": _read(args.network)} if getattr(args, "network", None) else {}),
        "run": {"prompt_tokens": args.prompt_tokens, "max_new": args.max_new,
                "warm_reps": 3, "warmup_reps": 1, "mode": args.mode,
                "temperature": 0.0, "seed": 0, "eos_policy": "fixed_length_ignore_eos",
                "speed_metric": "complete_suite_median_warm_committed_decode_tok_s"},
    })
    require_protocol(p)
    _write(args.out, p)
    print(json.dumps({"protocol": str(Path(args.out).resolve()), "sha256": p["sha256"]}))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="hash all local checkpoint bytes; no model load/download")
    inv.add_argument("--dir", required=True)
    inv.add_argument("--env-file", help="JSON object overriding the explicit baseline recipe")
    inv.add_argument("--out", required=True)
    prep = sub.add_parser("prepare", help="freeze local checkpoint, tokenizer, prompts and node inventory")
    prep.add_argument("--dir", required=True)
    prep.add_argument("--hardware", required=True, help="JSON array of pinned node/GPU/signer assignments")
    prep.add_argument("--isolation", choices=("host", "none"), default="host",
                      help="host preserves scattered-WAN baseline; none admits distinct GPUs on shared hosts")
    prep.add_argument("--env-file", help="JSON object overriding the explicit baseline recipe")
    prep.add_argument("--network", help="frozen public route/latency comparison contract JSON; required for controlled A/B")
    prep.add_argument("--prompt-tokens", type=int, default=512)
    prep.add_argument("--max-new", type=int, default=512)
    prep.add_argument("--mode", choices=("greedy", "dspark", "pipelined"), default="pipelined")
    prep.add_argument("--out", required=True)
    live = sub.add_parser("run", help="drive an existing ring; never launches/stops/rents GPU stages")
    live.add_argument("--protocol", required=True)
    live.add_argument("--dir", required=True)
    live.add_argument("--head", default="127.0.0.1:29610")
    live.add_argument("--tail", default="127.0.0.1:29612")
    live.add_argument("--timeout", type=float, default=600.0)
    live.add_argument("--connect-retry", type=float, default=300.0)
    live.add_argument("--deployment-plan", help="authenticated pipeline plan for a strict running ring")
    live.add_argument("--coordinator-key", help="existing receipt-format signing key; never copied into reports")
    live.add_argument("--fresh-ring", action="store_true", help="operator asserts processes/models are freshly started")
    live.add_argument("--out", required=True)
    check = sub.add_parser("verify", help="reverify raw receipts, token parity, complete protocol and speed")
    check.add_argument("report")
    check.add_argument("--protocol", help="independent frozen protocol; reject changed recipe")
    compare = sub.add_parser("compare", help="compare before/after only with matching benchmark recipe")
    compare.add_argument("before")
    compare.add_argument("after")
    compare.add_argument("--vary", help="one predeclared change: source, network, or env:V4_FLAG; omitted means identical controls")
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            env = dict(DEFAULT_ENV)
            if args.env_file:
                env.update(_read(args.env_file))
            _write(args.out, {"checkpoint": checkpoint_inventory(args.dir), "source": source_identity(),
                              "env": env, "stage_env_sha256": digest(env)})
            return 0
        if args.command == "prepare":
            return _prepare(args)
        if args.command == "run":
            protocol = _read(args.protocol)
            adapter = LiveRingAdapter(protocol, Path(args.dir), args.head, args.tail,
                                      timeout=args.timeout, retry_s=args.connect_retry,
                                      deployment_plan=args.deployment_plan, coordinator_key=args.coordinator_key)
            try:
                report = run_suite(adapter, protocol, fresh_ring=args.fresh_ring,
                                   on_sample=lambda partial: _write(args.out, partial))
            finally:
                adapter.close()
            report["evaluation"] = evaluate_report(report, protocol)
            _write(args.out, report)
            result = report["evaluation"]
        elif args.command == "verify":
            result = evaluate_report(_read(args.report), _read(args.protocol) if args.protocol else None)
        else:
            result = compare_reports(_read(args.before), _read(args.after), vary=args.vary)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result["status"] in ("passed", "verified_comparison", "verified_target_pass") else 2
    except (BenchmarkError, OSError, ValueError, ImportError) as exc:
        print(json.dumps({"status": "unverified", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

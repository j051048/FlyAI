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
MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
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


def checkpoint_inventory(directory: str | Path) -> dict:
    """Read every weight byte once. Header/size alone is insufficient to pin a checkpoint."""
    directory = Path(directory).resolve()
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
    if actual != inventory:
        raise BenchmarkError("checkpoint/config/tokenizer bytes differ from the frozen protocol")
    return {"checkpoint_bytes_verified": True, "config_bytes_verified": True,
            "tokenizer_bytes_verified": True}


def source_identity(root: Path = ROOT) -> dict:
    paths = set()
    for directory in (root / "engines" / "deepseek_v4", root / "vendor" / "deepseek_v4_ref", root / "shard"):
        paths.update(directory.rglob("*.py"))
    for name in ("phase0/wire.py", "phase0/v4_benchmark.py"):
        if (root / name).is_file():
            paths.add(root / name)
    files = [{"name": p.relative_to(root).as_posix(), "sha256": file_digest(p)} for p in sorted(paths)]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    # The content digest, rather than an optimistic clean flag, also identifies uncommitted code.
    return {"git_commit": commit, "sha256": digest(files), "files": files}


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
        if p.get("model_id") != MODEL_ID or p.get("layer_count") != 43:
            errors.append("protocol is not the fixed V4-Flash 43-layer model")
        ck = p.get("checkpoint", {})
        if (not _sha(ck.get("sha256")) or not _sha(ck.get("config_sha256"))
                or not _sha(ck.get("tokenizer_sha256")) or not ck.get("files")
                or digest(ck["files"]) != ck["sha256"]):
            errors.append("checkpoint/config/tokenizer hash inventory absent or invalid")
        files = ck.get("files", [])
        if not any(x.get("name", "").endswith(".safetensors") for x in files):
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
                    or node.get("host_id") in hosts):
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
    returned_at = clock() - start  # includes reset/prefill/last-token return/drain; excludes sweep
    tokens = list(result.get("tokens", []))
    if not result.get("ok") or tokens != [token for token, _ in events]:
        raise BenchmarkError("coordinator result is not exactly its committed callback token stream")
    if len(tokens) != protocol["run"]["max_new"]:
        raise BenchmarkError("generation ended before the fixed committed-token length")
    sweep_start = clock()
    receipts, sweep_ok = adapter.sweep(nonce)
    sweep_s = clock() - sweep_start
    offsets = [offset for _, offset in events]
    decode_s = offsets[-1] - offsets[0]
    return {
        "workload": prompt["workload"], "phase": phase, "rep": rep, "mode": mode,
        "job_id": job_id, "swarm_id": adapter.swarm_id, "nonce": nonce,
        "prompt_sha256": prompt["sha256"], "prompt_tokens": len(prompt["token_ids"]),
        "tokens": tokens, "output_sha256": digest(tokens), "committed_tokens": len(tokens),
        "measurement": {"clock": "perf_counter", "elapsed_s": returned_at,
                        "first_token_s": offsets[0], "last_token_s": offsets[-1],
                        "commit_offsets_s": offsets, "receipt_sweep_s": sweep_s,
                        "decode_s": decode_s,
                        "end_to_end_committed_tok_s": len(tokens) / returned_at,
                        "decode_committed_tok_s": (len(tokens) - 1) / decode_s if decode_s > 0 else None},
        "coordinator_stats": {k: v for k, v in result.items()
                              if k not in ("tokens", "receipts", "receipts_ok")},
        "receipt_sweep_ok": sweep_ok, "receipts": receipts,
    }


def run_suite(adapter, protocol: dict, *, fresh_ring: bool,
              clock: Callable[[], float] = time.perf_counter,
              on_sample: Callable[[dict], None] | None = None) -> dict:
    require_protocol(protocol)
    run_id = secrets.token_hex(16)
    report = {
        "schema": REPORT_SCHEMA, "protocol": protocol, "protocol_sha256": protocol["sha256"],
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
                 *, swarm_id="v4-benchmark", timeout=600.0, retry_s=300.0):
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
        self.pipe, self.ret = self.vp.connect_ring(head, tail, timeout=timeout,
                                                  token=self.vp.SWARM_TOKEN, retry_s=retry_s)

    def generate(self, prompt_ids, max_new, *, mode, nonce, job_id, on_token):
        methods = {"greedy": self.vp.coordinate, "dspark": self.vp.coordinate_dspark,
                   "pipelined": self.vp.coordinate_dspark_pipelined}
        kw = dict(eos_ids=(), nonce=nonce, swarm_id=self.swarm_id, job_id=job_id,
                  layer_count=self.layer_count, receipts=False, timeout=self.timeout,
                  on_token=on_token)
        if mode == "greedy":
            kw.update(temp=0.0, seed=0)
        elif mode == "pipelined":
            kw.update(depth=int(self.protocol["env"]["V4_SPEC_DEPTH"]),
                      lazy=self.protocol["env"]["V4_LAZY_DRAFT"] not in ("", "0"),
                      floor=int(self.protocol["env"]["V4_REFILL_FLOOR"]))
        return methods[mode](self.pipe, self.ret, prompt_ids, max_new, **kw)

    def sweep(self, nonce):
        return self.vp._sweep_receipts(self.pipe, self.ret, self.layer_count, nonce)

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
    if expected_protocol is not None and protocol != expected_protocol:
        failed.append("report does not use the independently supplied frozen protocol")
    if report.get("backend") != "live_ring":
        missing.append("not a live hardware benchmark; mock/CPU execution cannot pass the speed gate")
    evidence = report.get("artifact_verification", {})
    if not all(evidence.get(k) is True for k in ("checkpoint_bytes_verified", "config_bytes_verified",
                                               "tokenizer_bytes_verified", "engine_source_verified")):
        missing.append("live checkpoint/config/tokenizer/source verification absent")
    if report.get("cold_start", {}).get("fresh_ring_operator_assertion") is not True:
        missing.append("fresh process/model cold-start assertion absent")
    nodes = protocol["hardware"]
    assignments = {n["signer_pubkey"]: (n["layer_start"], n["layer_end"]) for n in nodes}
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
    }


def compare_reports(before: dict, after: dict) -> dict:
    """Compare matching workload/model recipes; decouple target passing from valid A/B comparison."""
    old, new = evaluate_report(before), evaluate_report(after)
    p, q = before.get("protocol", {}), after.get("protocol", {})
    # Core model, context length, and prompt recipe must match, allowing intentional runtime cache/knob evolution
    same_recipe = (all(p.get(k) == q.get(k) for k in ("model_id", "checkpoint", "prompts", "run"))
                   and p.get("env", {}).get("V4_MAX_SEQ") == q.get("env", {}).get("V4_MAX_SEQ"))
    if not same_recipe:
        return {"status": "unverified", "reason": "model/prompt/context/run recipe differs",
                "before": old, "after": new}
    ratios = {name: new["workloads"][name]["median_decode_committed_tok_s"] /
                    old["workloads"][name]["median_decode_committed_tok_s"]
              for name in WORKLOADS if name in old.get("workloads", {}) and name in new.get("workloads", {})}
    valid_pair = old.get("valid_evidence", False) and new.get("valid_evidence", False)
    status = ("verified_target_pass" if old.get("speed_pass") and new.get("speed_pass")
              else "verified_comparison" if valid_pair
              else "unverified")
    return {"status": status,
            "workload_speed_ratios": ratios, "before": old, "after": new,
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
    inventory = checkpoint_inventory(args.dir)
    source = source_identity()
    env = dict(DEFAULT_ENV)
    if args.env_file:
        env.update(_read(args.env_file))
    from transformers import AutoTokenizer  # local files only, no GPU/runtime model dependency
    tokenizer = AutoTokenizer.from_pretrained(args.dir, local_files_only=True, trust_remote_code=False)
    enc_dir = ROOT / "vendor" / "deepseek_v4_ref" / "encoding"
    sys.path.insert(0, str(enc_dir))
    from encoding_dsv4 import encode_messages
    render = lambda text: encode_messages([{"role": "user", "content": text}], "chat")
    prompts = freeze_prompts(lambda text: tokenizer.encode(text, add_special_tokens=False), render,
                             args.prompt_tokens)
    p = seal_protocol({
        "schema": PROTOCOL_SCHEMA, "model_id": MODEL_ID, "layer_count": 43,
        "checkpoint": inventory, "source": source, "env": env, "hardware": _read(args.hardware),
        "prompts": prompts,
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
    prep.add_argument("--env-file", help="JSON object overriding the explicit baseline recipe")
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
    live.add_argument("--fresh-ring", action="store_true", help="operator asserts processes/models are freshly started")
    live.add_argument("--out", required=True)
    check = sub.add_parser("verify", help="reverify raw receipts, token parity, complete protocol and speed")
    check.add_argument("report")
    check.add_argument("--protocol", help="independent frozen protocol; reject changed recipe")
    compare = sub.add_parser("compare", help="compare before/after only with matching benchmark recipe")
    compare.add_argument("before")
    compare.add_argument("after")
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
                                      timeout=args.timeout, retry_s=args.connect_retry)
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
            result = compare_reports(_read(args.before), _read(args.after))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result["status"] in ("passed", "verified_comparison") else 2
    except (BenchmarkError, OSError, ValueError, ImportError) as exc:
        print(json.dumps({"status": "unverified", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

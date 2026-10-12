"""Prepare bounded whole-layer numerical acceptance; execution is always opt-in.

Preparation uses only Python's standard library and artifact metadata. No model,
tokenizer, GPU, SSH, or ring is started. A head span can use frozen int64 IDs
directly; a non-head span must use its original frozen hidden input. ``make-input``
can obtain that input from a verified, explicitly executed real head prefix.
Numerical traces are diagnostic evidence, never throughput qualification.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import struct
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engines" / "deepseek_v4"
SCHEMA = "flyai-v4-numeric-acceptance-plan/1"
MAX_INPUT_BYTES = 128 << 20
MAX_JSON_BYTES = 32 << 20
MAX_RECORDS = 16384
MAX_TOKENS = 16384
MAX_LOG_BYTES = 32 << 20
DTYPE_BYTES = {"I64": 8, "BF16": 2, "F32": 4}
# Every numerical flag accepted by the existing diagnose CLI is explicit. Child
# environments also discard inherited V4_* settings, including placement/profile.
NUMERIC_ENV = {
    "V4_CUDA_GRAPH": "whole", "V4_MOE_IN_GRAPH": "1", "V4_MOE_GROUPED": "1",
    "V4_MOE_DECODE": "1", "V4_MOE_MULTI": "1", "V4_MOE_MULTI_MAX": "32",
    "V4_REF_SLIM": "0", "V4_REF_SLIM_NOQAT": "0", "V4_FAST_VERIFY": "0",
    "V4_FAST_VERIFY_MAX": "16", "V4_GRAPH_MAX": "192", "V4_FP8_GEMV": "1",
    "V4_FP8_SHARED": "1", "V4_FP8_AUTOTUNE": "0", "V4_KERNELS": "tilelang",
    "V4_DTYPE": "bfloat16", "V4_HADAMARD": "auto", "V4_EXECUTION_PROFILE": "",
    "V4_MAX_SEQ": "4096", "V4_PREFILL_QUERY_CHUNK": "128",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while part := stream.read(1 << 20):
            value.update(part)
    return value.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def read_json(path, limit=MAX_JSON_BYTES):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("JSON byte budget exceeded")
    return json.loads(raw, object_pairs_hook=_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def write_json(path, value, limit=MAX_JSON_BYTES):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    if len(payload) > limit:
        raise ValueError("JSON byte budget exceeded")
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".acceptance-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_inventory():
    paths = list(ENGINE.glob("v4_*.py")) + list((ROOT / "shard").glob("*.py"))
    paths += list((ROOT / "vendor" / "deepseek_v4_ref").rglob("*.py"))
    paths += [ROOT / "phase0" / "v4_numeric_diagnose.py", Path(__file__).resolve()]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(set(paths))}


def diagnostic_source_sha256():
    # Execute only this trusted repository's pure source-inventory function, so
    # preparation imports no model modules and uses exactly the trace's inventory.
    source = ENGINE / "v4_observability.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "source_inventory")
    scope = {"Path": Path, "hashlib": hashlib, "__file__": str(source)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
    return digest(scope["source_inventory"]())


def literal_assignment(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            # CPU defaults use dict(k=v), while bucket ladders are literal tuples.
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "dict":
                return {item.arg: ast.literal_eval(item.value) for item in node.value.keywords}
            return ast.literal_eval(node.value)
    raise ValueError(f"cannot derive {name} from source")


def model_defaults():
    tree = ast.parse((ROOT / "vendor/deepseek_v4_ref/inference/model.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelArgs")
    values = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                values[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "tuple" and not node.value.args:
                    values[node.target.id] = ()
                else:
                    raise ValueError("unsupported ModelArgs default")
    return values


def input_descriptor(path, cfg, lo, max_tokens):
    """Validate the actual safetensors bytes without importing torch or a tokenizer."""
    path = Path(path).resolve()
    size = path.stat().st_size
    if size > MAX_INPUT_BYTES + 65536 or size < 10:
        raise ValueError("frozen input byte budget exceeded")
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= length <= 65536 or 8 + length > size:
            raise ValueError("invalid bounded safetensors header")
        header = json.loads(stream.read(length), object_pairs_hook=_object)
        header.pop("__metadata__", None)
        if set(header) not in ({"token_ids"}, {"token_ids", "hidden"}):
            raise ValueError("only token_ids and original optional hidden are accepted")
        rows, spans = {}, []
        for name, row in header.items():
            if not isinstance(row, dict) or set(row) != {"dtype", "shape", "data_offsets"}:
                raise ValueError("invalid frozen tensor descriptor")
            shape, offsets = row["shape"], row["data_offsets"]
            if (not isinstance(shape, list) or any(type(n) is not int or n < 1 for n in shape)
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)
                    or row["dtype"] not in DTYPE_BYTES):
                raise ValueError("invalid frozen tensor shape/dtype/offsets")
            amount = math.prod(shape) * DTYPE_BYTES[row["dtype"]]
            if not 0 <= offsets[0] < offsets[1] <= size - 8 - length or offsets[1] - offsets[0] != amount or amount > MAX_INPUT_BYTES:
                raise ValueError("frozen tensor bytes differ from bounded geometry")
            spans.append(offsets)
            rows[name] = {**row, "bytes": amount}
            stream.seek(8 + length + offsets[0])
            value, remaining = hashlib.sha256(), amount
            while remaining:
                part = stream.read(min(remaining, 1 << 20))
                if not part:
                    raise ValueError("frozen payload is truncated")
                value.update(part)
                remaining -= len(part)
            rows[name]["sha256"] = value.hexdigest()
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                raise ValueError("frozen tensor payload is overlapping or has holes")
            cursor = end
        if cursor != size - 8 - length:
            raise ValueError("frozen input has unindexed payload")
        ids = rows["token_ids"]
        if ids["dtype"] != "I64" or len(ids["shape"]) != 2 or ids["shape"][0] != 1:
            raise ValueError("frozen int64 token_ids [1,total] required")
        total = ids["shape"][1]
        if not 2 <= total <= max_tokens:
            raise ValueError("frozen input token budget exceeded")
        stream.seek(8 + length + ids["data_offsets"][0])
        values = struct.iter_unpack("<q", stream.read(ids["bytes"]))
        if any(not 0 <= value[0] < cfg["vocab_size"] for value in values):
            raise ValueError("frozen token ID outside checkpoint vocabulary")
        hidden = rows.get("hidden")
        if hidden is not None and (hidden["dtype"] != "BF16" or hidden["shape"] != [1, total, cfg["hc_mult"], cfg["dim"]]):
            raise ValueError("original hidden must have model BF16 [1,total,hc_mult,dim] geometry")
        if lo and hidden is None:
            raise ValueError("non-head range requires original frozen hidden; use make-input --execute with the real verified prefix")
    return {"path": str(path), "sha256": file_hash(path), "bytes": size,
            "tokens": total, "header_bytes": length, "tensors": rows}


def prefix_input(source, total, destination):
    """Freeze an exact byte prefix of supplied inputs, preserving every ID/hidden."""
    header, cursor = {}, 0
    for name, row in source["tensors"].items():
        shape = list(row["shape"])
        amount = row["bytes"] // source["tokens"] * total
        shape[1] = total
        header[name] = {"dtype": row["dtype"], "shape": shape, "data_offsets": [cursor, cursor + amount]}
        cursor += amount
    raw = canonical(header)
    raw += b" " * (-len(raw) % 8)
    with Path(destination).open("xb") as out, Path(source["path"]).open("rb") as incoming:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        for name, row in source["tensors"].items():
            incoming.seek(8 + source["header_bytes"] + row["data_offsets"][0])
            remaining = header[name]["data_offsets"][1] - header[name]["data_offsets"][0]
            while remaining:
                chunk = incoming.read(min(remaining, 1 << 20))
                if not chunk:
                    raise ValueError("input changed while freezing prefix")
                out.write(chunk)
                remaining -= len(chunk)


def artifact_preview(checkpoint, lo, hi, cfg):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from shard import weight_artifacts as A
    checkpoint = Path(checkpoint).resolve()
    roles = {"head": lo == 0, "tail": hi == cfg["n_layers"], "dspark": False}
    if (checkpoint / A.STAGE_FILE).exists():
        proof = A.verify_stage_artifacts(checkpoint, lo=lo, hi=hi, **roles, verify_files=False)
        span = proof["stage"]
    else:
        proof = A.verify_weight_pack(checkpoint, verify_files=False)
        span = A.select_stage_artifacts(proof["catalog"], proof["pack"], lo, hi, **roles)
    A.validate_native_production_config(proof["catalog"]["config"])
    return {"checkpoint_id": proof["checkpoint_id"], "manifest_sha256": proof["manifest_sha256"],
            "config_sha256": proof["catalog"]["config_sha256"], "assigned": span,
            "payload_integrity_verified": False, "verification_scope": "metadata preview only; each trace verifies actual payload"}


def boundary_inventory(cfg, lo, hi, runtime_seq):
    """Positions are zero-based decode starts; end counts are one-based lengths."""
    buckets = literal_assignment(ENGINE / "v4_whole_layer_graph.py", "INDEXER_BUCKETS")
    result = [{"kind": "window_wrap", "end_tokens": cfg["window_size"] + 1, "layers": list(range(lo, hi))}]
    ratios = cfg["compress_ratios"]
    for ratio in (4, 128):
        layers = [i for i in range(lo, hi) if ratios[i] == ratio]
        result.append({"kind": f"compression_ratio_{ratio}", "end_tokens": ratio, "layers": layers})
    indexers = [i for i in range(lo, hi) if ratios[i] == 4]
    result.append({"kind": "indexer_topk_floor", "end_tokens": cfg["index_topk"] * 4,
                   "selection_starts_end_tokens": (cfg["index_topk"] + 1) * 4, "layers": indexers})
    maxw = runtime_seq // 4
    def width(need):
        return next((min(b, maxw) for b in buckets if b >= max(need, cfg["index_topk"])), maxw)
    first = next((n for n in range(1, maxw + 1) if width(n) != width(n - 1)), None)
    result.append({"kind": "indexer_bucket_transition", "end_tokens": first * 4 if first else None,
                   "layers": indexers, "bucket_before": width(first - 1) if first else None,
                   "bucket_after": width(first) if first else None, "ladder": list(buckets)})
    return result


def prepare(args):
    if (not 2 <= args.max_tokens <= MAX_TOKENS or not 1 <= args.threads <= 32 or
            not 1024 <= args.trace_max_bytes <= MAX_JSON_BYTES or not 1024 <= args.log_max_bytes <= MAX_LOG_BYTES):
        raise ValueError("invalid token/thread/trace byte budget")
    fixture = bool(args.fixture)
    raw = literal_assignment(ENGINE / "v4_ref_cpu.py", "_CPU_ARGS") if fixture else read_json(Path(args.checkpoint) / "config.json")
    if not isinstance(raw, dict):
        raise ValueError("model config must be an object")
    cfg = {**model_defaults(), **raw}
    ratios = cfg["compress_ratios"]
    for key in ("n_layers", "dim", "hc_mult", "vocab_size", "window_size", "index_topk"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError("invalid numeric model geometry")
    if len(ratios) < cfg["n_layers"] or any(type(r) is not int or r < 0 for r in ratios):
        raise ValueError("invalid configured compression ratios")
    lo = args.lo
    hi = args.hi
    if hi is None:
        # Shortest actual head prefix containing production ratio-4 and ratio-128.
        wanted = {4, 128}
        hi = next((i + 1 for i in range(cfg["n_layers"]) if wanted <= set(ratios[:i+1])), cfg["n_layers"])
    if not 0 <= lo < hi <= cfg["n_layers"]:
        raise ValueError("invalid exact layer range")
    if lo and args.hi is None:
        raise ValueError("non-head range requires explicit --hi")
    if fixture and args.device != "cpu" or not fixture and re.fullmatch(r"cuda:\d+", args.device) is None:
        raise ValueError("fixture uses cpu; production requires explicit cuda:N device")
    source = input_descriptor(args.input, cfg, lo, args.max_tokens)
    # A serving override is explicit and hashed. The existing CLI otherwise uses
    # ModelArgs default max_seq=4096; that cannot cross native 1024-slot buckets.
    runtime_seq = args.max_tokens
    cfg["max_seq_len"], cfg["max_batch_size"] = runtime_seq, 1
    environment = dict(NUMERIC_ENV, V4_MAX_SEQ=str(runtime_seq))
    if fixture:
        environment.update(V4_KERNELS="cpu", V4_HADAMARD="torch", V4_MOE_IN_GRAPH="0",
                           V4_MOE_GROUPED="0", V4_MOE_DECODE="0", V4_MOE_MULTI="0",
                           V4_FP8_GEMV="0", V4_FP8_SHARED="0", V4_PREFILL_QUERY_CHUNK="0")
    if args.env_file:
        overrides = read_json(args.env_file, 16384)
        if not isinstance(overrides, dict) or set(overrides) - set(environment) or any(not isinstance(v, str) or len(v) > 128 for v in overrides.values()):
            raise ValueError("unsupported numerical environment override")
        environment.update(overrides)
    if any(environment[k] != expected for k, expected in {
            "V4_CUDA_GRAPH": "whole", "V4_REF_SLIM": "0", "V4_REF_SLIM_NOQAT": "0",
            "V4_FAST_VERIFY": "0", "V4_EXECUTION_PROFILE": "", "V4_DTYPE": "bfloat16",
            "V4_MAX_SEQ": str(runtime_seq), "V4_KERNELS": "cpu" if fixture else "tilelang"}.items()):
        raise ValueError("acceptance requires whole, scalar verify, BF16, pinned kernels/limits, and REF_SLIM=0")
    boundaries = boundary_inventory(cfg, lo, hi, runtime_seq)
    proof = None if fixture else artifact_preview(args.checkpoint, lo, hi, cfg)
    directory = Path(args.out_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "numeric-plan.json").exists():
        raise ValueError("output already has a plan; use a fresh output directory")
    max_prefill = MAX_INPUT_BYTES // (cfg["hc_mult"] * cfg["dim"] * 2)
    cases, coverage = [], []
    for boundary in boundaries:
        end = boundary["end_tokens"]
        last = boundary.get("selection_starts_end_tokens", end)
        possible = bool(boundary["layers"] and end is not None and last + 4 <= source["tokens"])
        coverage.append({**boundary, "prepared": possible, "verified": False,
                         "reason": "scheduled exact boundary bracket" if possible else
                         "selected layers/input/runtime budget cannot cover this boundary"})
        if possible:
            # A bucket bracket must decode compression and non-compression in
            # BOTH rungs, rather than starting after the old rung's last write.
            lead = 8 if boundary["kind"] == "indexer_bucket_transition" else 3
            prompt = max(1, min(end - lead, max_prefill))
            total = last + 4
            if args.prompt_tokens is not None:
                prompt = args.prompt_tokens
            if not 1 <= prompt < end or total - prompt < 4:
                raise ValueError("explicit prompt must precede every scheduled boundary with recovery tokens")
            cases.append({"name": boundary["kind"], "prompt_tokens": prompt, "tokens": total,
                          "boundary": boundary, "rejection": {"position": max(prompt, end - 3), "width": 6, "accepted": 2}})
    if not cases:
        prompt = args.prompt_tokens if args.prompt_tokens is not None else min(2, source["tokens"] - 1)
        total = source["tokens"]
        cases.append({"name": "bounded_smoke", "prompt_tokens": prompt, "tokens": total,
                      "boundary": None, "rejection": {"position": prompt, "width": 3, "accepted": 1}
                      if total - prompt >= 4 else None})
    dedup = {}
    for case in cases:
        key = (case["prompt_tokens"], case["tokens"], digest(case["rejection"]))
        if key not in dedup:
            case["covers"] = [case["name"]]
            dedup[key] = case
        else:
            dedup[key]["covers"].append(case["name"])
    cases = list(dedup.values())
    for case in cases:
        total, prompt = case["tokens"], case["prompt_tokens"]
        records = (total - prompt + 1) * (hi - lo + 1)
        if not 1 <= prompt < total or prompt > max_prefill or records > MAX_RECORDS or records * 16384 > args.trace_max_bytes:
            raise ValueError("case exceeds prompt tensor or record budget; provide a closer prompt or shorter input")
        rejection = case["rejection"]
        if rejection and not prompt <= rejection["position"] < rejection["position"] + rejection["width"] <= total:
            raise ValueError("case cannot hold rejection and recovery suffix")
        target = directory / (case["name"] + ".safetensors")
        prefix_input(source, total, target)
        case["input"] = input_descriptor(target, cfg, lo, args.max_tokens)
        case["record_budget"] = records
    if source["sha256"] != file_hash(source["path"]):
        raise ValueError("frozen source input changed during preparation")
    inventory = source_inventory()
    plan = {"schema": SCHEMA, "fixture": fixture, "seed": args.seed,
            "checkpoint": None if fixture else str(Path(args.checkpoint).resolve()),
            "device": args.device, "threads": args.threads, "lo": lo, "hi": hi,
            "roles": {"head": lo == 0, "tail": hi == cfg["n_layers"], "dspark": False},
            "runtime_model_args": cfg, "runtime_model_args_sha256": digest(cfg),
            "config_asset_sha256": None if fixture else file_hash(Path(args.checkpoint) / "config.json"),
            "artifact_preview": proof, "source_files": inventory, "source_sha256": digest(inventory),
            "diagnostic_source_sha256": diagnostic_source_sha256(),
            "environment": environment, "environment_sha256": digest(environment), "input": source,
            "limits": {"tokens": args.max_tokens, "records": MAX_RECORDS, "input_bytes": MAX_INPUT_BYTES,
                       "trace_bytes": args.trace_max_bytes, "process_timeout_s": args.timeout,
                       "log_bytes_per_stream": args.log_max_bytes},
            "cases": cases, "coverage": coverage, "hardware_verified": False, "speed_pass": False,
            "status": "prepared_unverified", "scope": "exact owned stage; same-host ring and decoding strategies require separate acceptance"}
    plan["sha256"] = digest(plan)
    write_json(directory / "numeric-plan.json", plan)
    write_json(directory / "numeric-env.json", environment, 16384)
    return plan


def checked_plan(path):
    plan = read_json(path)
    if plan.get("schema") != SCHEMA or plan.get("sha256") != digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("invalid or modified numeric plan")
    if source_inventory() != plan["source_files"]:
        raise ValueError("source changed since preparation; prepare a fresh plan")
    for item in [plan["input"], *(case["input"] for case in plan["cases"])]:
        current = input_descriptor(item["path"], plan["runtime_model_args"], plan["lo"], plan["limits"]["tokens"])
        if current != item:
            raise ValueError("frozen input changed since preparation")
    if not plan["fixture"]:
        if file_hash(Path(plan["checkpoint"]) / "config.json") != plan["config_asset_sha256"]:
            raise ValueError("checkpoint config changed since preparation")
        if artifact_preview(plan["checkpoint"], plan["lo"], plan["hi"], plan["runtime_model_args"]) != plan["artifact_preview"]:
            raise ValueError("checkpoint metadata changed since preparation")
    return plan


def child_environment(plan):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("V4_")}
    environment.update(plan["environment"])
    environment.update(OMP_NUM_THREADS=str(plan["threads"]), MKL_NUM_THREADS=str(plan["threads"]))
    return environment


def run_owned_command(command, *, environment, cwd, stdout_path, stderr_path,
                      timeout_s, log_max_bytes=MAX_LOG_BYTES, cleanup_timeout_s=5.0):
    """Persist capped raw streams and supervise only this explicitly owned tree.

    Stream readers hold at most 64KiB each and never write beyond the per-stream
    cap. The worker waits on stdin until its Windows job/POSIX ownership is
    enrolled. Inherited campaign ownership tokens are preserved in descendants.
    """
    if (not isinstance(command, list) or not command or any(not isinstance(arg, str) or "\x00" in arg for arg in command)
            or type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 86400
            or type(log_max_bytes) is not int or not 1 <= log_max_bytes <= MAX_LOG_BYTES
            or not 0 < cleanup_timeout_s <= 30):
        raise ValueError("invalid bounded owned command")
    stdout_path, stderr_path = Path(stdout_path).resolve(), Path(stderr_path).resolve()
    if stdout_path == stderr_path:
        raise ValueError("stdout/stderr require separate fresh log files")
    for path in (stdout_path, stderr_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise ValueError("owned raw command log already exists")
    for path in (ROOT, ENGINE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import psutil
    from v4_native_capability import _ProbeProcessOwner, _PROCESS_TOKEN
    environment = dict(environment)
    environment[_PROCESS_TOKEN] = secrets.token_hex(32)
    environment.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    # Keep this enrolled wrapper alive while the actual command runs. Windows
    # venv launchers cannot safely use POSIX-style exec replacement here.
    gate = "import json,os,sys,subprocess; request=json.loads(sys.stdin.buffer.readline()); sys.exit(subprocess.call(request['argv'],env=os.environ))"
    payload = canonical({"argv": command}) + b"\n"
    if len(payload) > 65536:
        raise ValueError("owned command request exceeds stdin budget")
    events = {"stdout": threading.Event(), "stderr": threading.Event()}
    state = {name: {"bytes_seen": 0, "written": 0, "error": None} for name in events}
    readers, proc, owner = [], None, None
    cleanup = {"confirmed": False, "errors": ["process ownership was not established"]}
    timed_out = output_limit = False
    supervision_error = None
    started = time.monotonic()
    def copy_stream(name, source, destination):
        try:
            while part := os.read(source.fileno(), 65536):
                current = state[name]
                current["bytes_seen"] += len(part)
                amount = min(len(part), max(0, log_max_bytes - current["written"]))
                if amount:
                    destination.write(part[:amount])
                    current["written"] += amount
                if amount != len(part):
                    events[name].set()
        except (OSError, ValueError) as error:
            state[name]["error"] = str(error)[:500]
            events[name].set()
    with stdout_path.open("xb", buffering=0) as stdout_log, stderr_path.open("xb", buffering=0) as stderr_log:
        try:
            proc = subprocess.Popen([sys.executable, "-u", "-c", gate], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=environment,
                **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
                   if os.name == "nt" else {"start_new_session": True}))
            owner = _ProbeProcessOwner(proc, environment[_PROCESS_TOKEN])
            owner.enroll()
            for name, stream, output in (("stdout", proc.stdout, stdout_log), ("stderr", proc.stderr, stderr_log)):
                reader = threading.Thread(target=copy_stream, args=(name, stream, output), daemon=True)
                reader.start()
                readers.append(reader)
            proc.stdin.write(payload)
            proc.stdin.flush()
            proc.stdin.close()
            while proc.poll() is None:
                if any(event.is_set() for event in events.values()):
                    output_limit = any(state[name]["bytes_seen"] > log_max_bytes for name in state)
                    if not output_limit:
                        supervision_error = "raw log stream failed"
                    break
                if time.monotonic() - started >= timeout_s:
                    timed_out = True
                    break
                # Logs are also observed on disk; thread writers enforce the cap
                # even if a compiler emits an entire huge burst between polls.
                if any(path.stat().st_size > log_max_bytes for path in (stdout_path, stderr_path)):
                    output_limit = True
                    break
                time.sleep(.01)
        except (OSError, psutil.Error, ValueError) as error:
            supervision_error = f"{type(error).__name__}: {str(error)[:500]}"
        finally:
            if owner is not None:
                cleanup = owner.cleanup(timeout_s=cleanup_timeout_s)
            elif proc is not None:
                try:
                    proc.kill()
                    proc.wait(timeout=cleanup_timeout_s)
                except (OSError, subprocess.TimeoutExpired) as error:
                    supervision_error = f"unenrolled worker cleanup failed: {error}"
            if proc is not None and proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
            for reader in readers:
                reader.join(timeout=2)
            if any(reader.is_alive() for reader in readers):
                cleanup = {**cleanup, "confirmed": False, "errors": [*cleanup.get("errors", []), "raw log reader did not finish"]}
            output_limit |= any(item["bytes_seen"] > log_max_bytes for item in state.values())
            if any(item["error"] for item in state.values()):
                supervision_error = "raw log stream failed: " + str(state)
            if proc is not None and not any(reader.is_alive() for reader in readers):
                if proc.stdout is not None: proc.stdout.close()
                if proc.stderr is not None: proc.stderr.close()
            stdout_log.flush(); stderr_log.flush()
            os.fsync(stdout_log.fileno()); os.fsync(stderr_log.fileno())
    logs = {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path),
                   "bytes_seen": state[name]["bytes_seen"]}
            for name, path in (("stdout", stdout_path), ("stderr", stderr_path))}
    return {"schema": "flyai-v4-numeric-owned-command/1", "argv": command,
            "returncode": proc.returncode if proc is not None else None,
            "timeout": timed_out, "output_limit": bool(output_limit), "supervision_error": supervision_error,
            "cleanup": cleanup, "logs": logs, "log_max_bytes_per_stream": log_max_bytes,
            "elapsed_s": time.monotonic() - started, "timeout_s": timeout_s}


def command_succeeded(completion, *, allowed_codes=(0,)):
    return (completion["returncode"] in allowed_codes and not completion["timeout"] and not completion["output_limit"]
            and completion["supervision_error"] is None and completion["cleanup"].get("confirmed") is True)


def _log_tail(path, limit=4096):
    with Path(path).open("rb") as stream:
        stream.seek(max(0, Path(path).stat().st_size - limit))
        return stream.read(limit).decode("utf-8", errors="replace")


def trace_worker(args):
    """Adapter around the unchanged CLI, with explicit serving capacity overrides."""
    if not args.execute:
        raise ValueError("GPU/model execution requires --execute")
    plan = checked_plan(args.plan)
    case = next((item for item in plan["cases"] if item["name"] == args.case), None)
    if case is None:
        raise ValueError("unknown prepared numeric case")
    for key in list(os.environ):
        if key.startswith("V4_"):
            del os.environ[key]
    os.environ.update(plan["environment"])
    for path in (ROOT, ROOT / "phase0", ENGINE):
        sys.path.insert(0, str(path))
    import v4_stage as V
    import v4_ref_cpu as R
    import v4_numeric_diagnose as diagnose
    import v4_ref_slim as slim
    # These runtime toggles default true even when their wrappers are absent.
    # Pin both the installation flags and observed toggles for an unambiguous receipt.
    slim.set_active(False)
    slim.set_active_noqat(False)
    old_config, old_cpu = V.config, R.cpu_args
    def config(*a, **kw):
        model = old_config(*a, **kw)
        model.max_seq_len = plan["runtime_model_args"]["max_seq_len"]
        model.max_batch_size = 1
        return model
    def cpu(*a, **kw):
        kw["max_batch_size"] = 1
        return old_cpu(*a, **kw)
    V.config, R.cpu_args = config, cpu
    command = ["trace", "--input", case["input"]["path"], "--prompt-tokens", str(case["prompt_tokens"]),
               "--max-tokens", str(plan["limits"]["tokens"]), "--threads", str(plan["threads"]),
               "--lo", str(plan["lo"]), "--hi", str(plan["hi"]), "--device", plan["device"],
               "--execution", args.execution, "--out", args.out]
    command += ["--fixture", "--seed", str(plan["seed"])] if plan["fixture"] else ["--checkpoint", plan["checkpoint"]]
    if args.rejection:
        if case["rejection"] is None:
            raise ValueError("case has no bounded rejection plan")
        command += ["--rejection", json.dumps(case["rejection"])]
    try:
        return diagnose.main(command)
    finally:
        V.config, R.cpu_args = old_config, old_cpu


def _saved_file(value, base_dir):
    original = Path(value)
    basename = re.split(r"[\\/]", str(value))[-1]
    portable = Path(base_dir) / basename if base_dir else original
    # Prefer the package next to the plan when paths refer to another machine.
    return portable if portable.is_file() else original


def _trace_compare(left, right):
    # Re-use the unchanged comparison algorithm without importing Torch. Its
    # function is pure JSON/hash logic, with these three immutable globals.
    source = ENGINE / "v4_numeric_diagnostics.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compare_traces")
    scope = {"digest": digest, "SCHEMA": "flyai-v4-numeric-trace/1", "MAX_RECORDS": MAX_RECORDS}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
    return scope["compare_traces"](left, right)


def expected_graph_variants(plan, case, layer):
    cfg = plan["runtime_model_args"]
    ratio = cfg["compress_ratios"][layer]
    if not ratio:
        return 1
    ladder = next(row["ladder"] for row in plan["coverage"] if row["kind"] == "indexer_bucket_transition")
    maxw, floor = cfg["max_seq_len"] // ratio, cfg["index_topk"] if ratio == 4 else 0
    keys = set()
    for position in range(case["prompt_tokens"], case["tokens"]):
        need = max((position + 1) // ratio, floor)
        bucket = next((min(b, maxw) for b in ladder if b >= need), maxw)
        keys.add((bucket, (position + 1) % ratio == 0))
    return len(keys)


def verify_result(result, plan, *, base_dir=None):
    """Recompute a saved gate from raw evidence; never trust saved status fields.

    ``base_dir`` is the extracted package directory containing case safetensors,
    traces and comparison JSON. All execution paths remain operator declarations;
    this verifies bounded saved evidence and does not remotely attest a GPU.
    """
    if plan.get("schema") != SCHEMA or plan.get("sha256") != digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("invalid or modified numeric plan")
    if result.get("schema") != "flyai-v4-numeric-acceptance-result/1" or result.get("plan_sha256") != plan["sha256"]:
        raise ValueError("numeric result belongs to another plan")
    if source_inventory() != plan["source_files"] or diagnostic_source_sha256() != plan["diagnostic_source_sha256"]:
        raise ValueError("numeric evidence source binding differs from current repository")
    if (plan["environment_sha256"] != digest(plan["environment"]) or
            plan["runtime_model_args_sha256"] != digest(plan["runtime_model_args"])):
        raise ValueError("numeric recipe/configuration hash mismatch")
    prepared = {case["name"]: case for case in plan["cases"]}
    saved = {case["name"]: case for case in result.get("cases", [])}
    if len(saved) != len(result.get("cases", [])) or set(saved) != set(prepared):
        raise ValueError("numeric result has missing or duplicate cases")
    expected_commands = [{"case": name, **reference} for name in prepared for reference in saved[name].get("commands", [])]
    if result.get("commands") != expected_commands:
        raise ValueError("numeric owned command inventory is incomplete or modified")
    coverage = json.loads(json.dumps(plan["coverage"]))
    checked = []
    for name, case in prepared.items():
        original = case["input"]
        input_path = _saved_file(original["path"], base_dir)
        current = input_descriptor(input_path, plan["runtime_model_args"], plan["lo"], plan["limits"]["tokens"])
        if {k: v for k, v in current.items() if k != "path"} != {k: v for k, v in original.items() if k != "path"}:
            raise ValueError("case frozen input differs from prepared bytes")
        expected_keys = {"configured_clean", "whole_eager_clean"}
        if case["rejection"]:
            expected_keys |= {"configured_rejected", "whole_eager_rejected"}
        references = saved[name].get("traces", {})
        if set(references) != expected_keys:
            raise ValueError("missing configured/eager/rejection raw traces")
        traces = {key: read_json(_saved_file(path, base_dir), plan["limits"]["trace_bytes"]) for key, path in references.items()}
        for key, trace in traces.items():
            _trace_compare(trace, trace)  # checks schema and digest; NaN is handled by the real comparator.
            identity = trace.get("identity", {})
            if identity.get("layers") != [plan["lo"], plan["hi"]] or identity.get("args_sha256") != plan["runtime_model_args_sha256"] or identity.get("source_sha256") != plan["diagnostic_source_sha256"]:
                raise ValueError("trace model arguments/source/layer identity differs from plan")
            expected_inputs = {"prompt_tokens": case["prompt_tokens"], "hidden": None}
            for tensor, target in (("token_ids", "ids"), ("hidden", "hidden")):
                row = original["tensors"].get(tensor)
                if row is not None:
                    expected_inputs[target] = {"sha256": row["sha256"], "shape": row["shape"],
                        "dtype": "torch.int64" if tensor == "token_ids" else "torch.bfloat16", "nan": False, "infinity": False}
            if trace.get("inputs") != expected_inputs:
                raise ValueError("trace does not use the frozen case tensor bytes/prompt")
            if trace.get("execution") != ("configured" if key.startswith("configured") else "whole_eager") or trace.get("rejection") != (case["rejection"] if key.endswith("rejected") else None):
                raise ValueError("trace execution/rejection differs from plan")
            expected_positions = {(0, case["prompt_tokens"], layer) for layer in range(plan["lo"], plan["hi"])}
            expected_positions |= {(position, 1, layer) for position in range(case["prompt_tokens"], case["tokens"])
                                   for layer in range(plan["lo"], plan["hi"])}
            if plan["roles"]["tail"]:
                expected_positions |= {(0, case["prompt_tokens"], plan["hi"])}
                expected_positions |= {(position, 1, plan["hi"]) for position in range(case["prompt_tokens"], case["tokens"])}
            records = trace["records"]
            if len(records) != len(expected_positions) or {(r["position"], r["tokens"], r["layer"]) for r in records} != expected_positions:
                raise ValueError("trace has missing/duplicated boundary or recovery records")
            for record in records:
                if record["layer"] == plan["hi"]:
                    continue
                ratio = plan["runtime_model_args"]["compress_ratios"][record["layer"]]
                expected_state = {"window"}
                if ratio:
                    expected_state |= {"compressed", "compressor.kv_state", "compressor.score_state"}
                if ratio == 4:
                    expected_state |= {"indexer.cache", "indexer.kv_state", "indexer.score_state"}
                if set(record.get("state", {})) != expected_state:
                    raise ValueError("trace omitted valid window/compressor/indexer state")
        pairs = [("configured_clean", "whole_eager_clean")]
        if case["rejection"]:
            pairs += [("configured_rejected", "whole_eager_rejected"),
                      ("configured_clean", "configured_rejected"), ("whole_eager_clean", "whole_eager_rejected")]
        recomputed = [_trace_compare(traces[a], traces[b]) for a, b in pairs]
        reports = saved[name].get("comparisons", [])
        if len(reports) != len(pairs):
            raise ValueError("missing saved pairwise comparison evidence")
        for evidence, (left, right), comparison in zip(reports, pairs, recomputed):
            saved_comparison = read_json(_saved_file(evidence["report"], base_dir), plan["limits"]["trace_bytes"])
            if evidence.get("left") != left or evidence.get("right") != right or saved_comparison != comparison:
                raise ValueError("saved numeric comparison differs from recomputed raw traces")
        command_refs = saved[name].get("commands", [])
        command_map = {item["step"]: item for item in command_refs}
        expected_steps = {"trace." + key for key in expected_keys} | {"compare." + left + "-vs-" + right for left, right in pairs}
        if len(command_map) != len(command_refs) or set(command_map) != expected_steps:
            raise ValueError("numeric raw command evidence is missing or duplicated")
        for step, reference in command_map.items():
            completion_path = _saved_file(reference["completion"], base_dir)
            if file_hash(completion_path) != reference["sha256"]:
                raise ValueError("owned command completion bytes changed")
            completion = read_json(completion_path, 65536)
            if (completion.get("schema") != "flyai-v4-numeric-owned-command/1" or completion.get("case") != name or
                    completion.get("step") != step or completion.get("sha256") != digest({k: v for k, v in completion.items() if k != "sha256"}) or
                    completion.get("timeout_s") != plan["limits"]["process_timeout_s"] or
                    completion.get("log_max_bytes_per_stream") != plan["limits"]["log_bytes_per_stream"]):
                raise ValueError("owned command completion differs from prepared supervision")
            if step.startswith("trace."):
                expected_code = 0
                key = step[len("trace."):]
                argv = completion.get("argv", [])
                execution = "configured" if key.startswith("configured_") else "whole_eager"
                required = ["trace-worker", "--case", name, "--execution", execution, "--execute"]
                if any(value not in argv for value in required) or ("--rejection" in argv) != key.endswith("rejected"):
                    raise ValueError("owned trace command differs from required execution")
            else:
                label = step[len("compare."):]
                index = next(i for i, (left, right) in enumerate(pairs) if left + "-vs-" + right == label)
                expected_code = 0 if recomputed[index]["exact_match"] else 1
                if "compare" not in completion.get("argv", []):
                    raise ValueError("owned comparison command is absent")
            if not command_succeeded(completion, allowed_codes=(expected_code,)):
                raise ValueError("numeric command supervision/owned cleanup was not successful")
            if set(completion.get("logs", {})) != {"stdout", "stderr"}:
                raise ValueError("both persistent raw log streams are required")
            for log in completion["logs"].values():
                log_path = _saved_file(log["path"], base_dir)
                if (type(log.get("bytes")) is not int or not 0 <= log["bytes"] <= plan["limits"]["log_bytes_per_stream"] or
                        log.get("bytes_seen") != log["bytes"] or log_path.stat().st_size != log["bytes"] or file_hash(log_path) != log["sha256"]):
                    raise ValueError("raw numeric command log changed or exceeded its byte cap")
        exact = all(item["exact_match"] for item in recomputed)
        models = [trace["identity"]["model"] for trace in traces.values()]
        proof = plan["artifact_preview"]
        verified = not plan["fixture"] and all(trace.get("evidence") == "gpu_execution" and model.get("verified_artifacts") is True and
            model.get("checkpoint_id") == proof["checkpoint_id"] and model.get("manifest_sha256") == proof["manifest_sha256"] and
            model.get("config_sha256") == proof["config_sha256"] for trace, model in zip(traces.values(), models))
        recipes = all(trace.get("recipe", {}).get("graph_mode") == "whole" and trace["recipe"].get("threads") == plan["threads"] and
            trace["recipe"].get("device") == plan["device"] and trace["recipe"].get("dtype") == "torch.bfloat16" and
            trace["recipe"].get("fast_verify") is False and trace["recipe"].get("ref_slim_active") is False and
            trace["recipe"].get("noqat_active") is False and trace["recipe"].get("ref_slim_installed") is False and
            trace["recipe"].get("noqat_installed") is False and trace["recipe"].get("whole_eager_twin") is key.startswith("whole_eager")
            for key, trace in traces.items())
        captured = True
        for key, trace in traces.items():
            if not key.startswith("configured"):
                continue
            graphs = trace.get("graphs", [])
            captured &= len(graphs) == plan["hi"] - plan["lo"] and {g["layer"] for g in graphs} == set(range(plan["lo"], plan["hi"]))
            for graph in graphs:
                minimum = expected_graph_variants(plan, case, graph["layer"])
                captured &= graph["captured_variants"] >= minimum and not graph["fallback"]
        row = {"name": name, "covers": case["covers"], "exact_match": exact,
               "verified_artifacts_gpu_execution": verified, "every_layer_captured": bool(captured),
               "numerical_recipe_observed": recipes, "rejection_recovery_checked": bool(case["rejection"])}
        checked.append(row)
        for item in coverage:
            if item["kind"] in case["covers"]:
                item["verified"] = item["prepared"] and exact and verified and captured and recipes and bool(case["rejection"])
    exact = all(row["exact_match"] for row in checked)
    complete = exact and all(item["verified"] for item in coverage)
    return {"schema": "flyai-v4-numeric-acceptance-verification/1", "plan_sha256": plan["sha256"],
            "status": "passed_owned_stage_numeric" if complete else "partial_unverified" if exact else "failed",
            "hardware_verified": bool(complete), "speed_pass": False, "full_model_numeric_verified": False,
            "source_sha256": plan["source_sha256"], "runtime_model_args_sha256": plan["runtime_model_args_sha256"],
            "device": plan["device"], "artifact_identity": plan["artifact_preview"], "cases": checked, "coverage": coverage,
            "scope": plan["scope"], "physical_gpu_attestation": False}


def verify(args):
    return verify_result(read_json(args.result), read_json(args.plan), base_dir=Path(args.plan).resolve().parent)


def run(args):
    plan = checked_plan(args.plan)
    if not args.execute:
        return {"status": "prepared_unverified", "hardware_verified": False, "speed_pass": False,
                "message": "Add --execute only on the explicitly reserved acceptance device", "plan_sha256": plan["sha256"]}
    directory = Path(args.plan).resolve().parent
    report_path = directory / "numeric-result.json"
    if report_path.exists():
        raise ValueError("result already exists; prepare a fresh directory for a new run")
    result = {"schema": "flyai-v4-numeric-acceptance-result/1", "plan_sha256": plan["sha256"],
              "status": "running", "hardware_verified": False, "speed_pass": False,
              "coverage": plan["coverage"], "cases": [], "commands": [], "scope": plan["scope"]}
    write_json(report_path, result)
    environment = child_environment(plan)
    def execute_step(case, step, command):
        stem = case["name"] + "." + step
        completion = run_owned_command(command, environment=environment, cwd=ROOT,
            stdout_path=directory / (stem + ".stdout.log"), stderr_path=directory / (stem + ".stderr.log"),
            timeout_s=plan["limits"]["process_timeout_s"], log_max_bytes=plan["limits"]["log_bytes_per_stream"])
        completion.update(case=case["name"], step=step)
        completion["sha256"] = digest(completion)
        path = directory / (stem + ".command.json")
        write_json(path, completion)
        reference = {"step": step, "completion": str(path), "sha256": file_hash(path)}
        result["commands"].append({"case": case["name"], **reference})
        write_json(report_path, result)
        return completion, reference
    def failed_command(case, step, completion):
        result.update(status="failed", error={"case": case["name"], "step": step,
            "returncode": completion["returncode"], "timeout": completion["timeout"],
            "output_limit": completion["output_limit"], "cleanup": completion["cleanup"],
            "detail": _log_tail(completion["logs"]["stderr"]["path"]),
            "supervision_error": completion["supervision_error"]})
        write_json(report_path, result)
        return result
    for case in plan["cases"]:
        traces, comparisons, commands = {}, [], []
        rejection_modes = (False, True) if case["rejection"] else (False,)
        for rejected in rejection_modes:
            for execution in ("configured", "whole_eager"):
                key = execution + ("_rejected" if rejected else "_clean")
                out = directory / (case["name"] + "." + key + ".json")
                command = [sys.executable, str(Path(__file__).resolve()), "trace-worker", "--plan", str(Path(args.plan).resolve()),
                           "--case", case["name"], "--execution", execution, "--out", str(out), "--execute"]
                if rejected:
                    command.append("--rejection")
                completion, reference = execute_step(case, "trace." + key, command)
                commands.append(reference)
                if not command_succeeded(completion):
                    return failed_command(case, key, completion)
                traces[key] = read_json(out, plan["limits"]["trace_bytes"])
                traces[key]["_file"] = str(out)
        pairs = [("configured_clean", "whole_eager_clean")]
        if case["rejection"]:
            pairs += [("configured_rejected", "whole_eager_rejected"),
                      ("configured_clean", "configured_rejected"), ("whole_eager_clean", "whole_eager_rejected")]
        for left, right in pairs:
            out = directory / (case["name"] + "." + left + "-vs-" + right + ".comparison.json")
            command = [sys.executable, str(ROOT / "phase0/v4_numeric_diagnose.py"), "compare", traces[left]["_file"], traces[right]["_file"], "--out", str(out)]
            completion, reference = execute_step(case, "compare." + left + "-vs-" + right, command)
            commands.append(reference)
            if not command_succeeded(completion, allowed_codes=(0, 1)):
                return failed_command(case, left + "-vs-" + right, completion)
            comparison = read_json(out, plan["limits"]["trace_bytes"]) if out.exists() else {"exact_match": False, "error": "comparison JSON missing; inspect raw logs"}
            comparisons.append({"left": left, "right": right, "report": str(out), **comparison})
        clean = traces["configured_clean"]
        verified = (not plan["fixture"] and all(trace.get("evidence") == "gpu_execution" and
                    trace.get("identity", {}).get("model", {}).get("verified_artifacts") is True for trace in traces.values()))
        configured = [trace for key, trace in traces.items() if key.startswith("configured_")]
        captured = all(bool(trace.get("graphs")) and len(trace["graphs"]) == plan["hi"] - plan["lo"] and all(
            graph["captured_variants"] > 0 and not graph["fallback"] for graph in trace["graphs"]) for trace in configured)
        recipes_pinned = all(trace.get("recipe", {}).get("graph_mode") == "whole" and
            trace["recipe"].get("threads") == plan["threads"] and trace["recipe"].get("device") == plan["device"] and
            trace["recipe"].get("dtype") == "torch.bfloat16" and trace["recipe"].get("fast_verify") is False and
            trace["recipe"].get("ref_slim_active") is False and trace["recipe"].get("noqat_active") is False
            for trace in traces.values())
        exact = all(item["exact_match"] for item in comparisons)
        row = {"name": case["name"], "covers": case["covers"], "exact_match": exact,
               "verified_artifacts_gpu_execution": verified, "every_layer_captured": captured,
               "numerical_recipe_observed": recipes_pinned,
               "rejection_recovery_checked": case["rejection"] is not None, "comparisons": comparisons,
               "commands": commands,
               "traces": {key: value["_file"] for key, value in traces.items()}}
        result["cases"].append(row)
        for coverage in result["coverage"]:
            if coverage["kind"] in case["covers"]:
                coverage["verified"] = verified and captured and exact and recipes_pinned and row["rejection_recovery_checked"]
        write_json(report_path, result)
    exact = all(case["exact_match"] for case in result["cases"])
    complete = exact and all(item["verified"] for item in result["coverage"])
    result.update(status="passed_owned_stage_numeric" if complete else "partial_unverified" if exact else "failed",
                  hardware_verified=complete, speed_pass=False,
                  full_model_numeric_verified=False,
                  note="Exact local stage checks do not qualify a same-host ring, DSpark, decoding strategies, or speed")
    write_json(report_path, result)
    verification = verify_result(result, plan, base_dir=directory)
    result.update(status=verification["status"], hardware_verified=verification["hardware_verified"],
                  coverage=verification["coverage"])
    result["independent_verification"] = verification
    write_json(report_path, result)
    return result


def freeze_ids(args):
    ids = read_json(args.ids_json, 1 << 20)
    if not isinstance(ids, list) or not 2 <= len(ids) <= MAX_TOKENS or any(type(i) is not int or not 0 <= i < 1 << 63 for i in ids):
        raise ValueError("provide 2..16384 explicit nonnegative int64 IDs, without tokenizer inference")
    path = Path(args.out).resolve()
    header = canonical({"token_ids": {"dtype": "I64", "shape": [1, len(ids)], "data_offsets": [0, len(ids) * 8]}})
    header += b" " * (-len(header) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        stream.write(struct.pack("<" + "q" * len(ids), *ids))
    return {"input": str(path), "sha256": file_hash(path), "tokens": len(ids), "hardware_verified": False}


def make_input(args):
    if not args.execute:
        raise ValueError("real prefix model/GPU execution requires --execute")
    if re.fullmatch(r"cuda:\d+", args.device) is None or not 1 <= args.threads <= 32 or not 2 <= args.max_tokens <= MAX_TOKENS:
        raise ValueError("real prefix requires an explicit cuda:N device and fixed threads")
    raw = read_json(Path(args.checkpoint) / "config.json")
    cfg = {**model_defaults(), **raw}
    if not 0 < args.layer < cfg["n_layers"]:
        raise ValueError("target non-head layer must be inside the actual model")
    description = input_descriptor(args.input, cfg, 0, args.max_tokens)
    if "hidden" in description["tensors"]:
        raise ValueError("make-input expects frozen IDs only")
    if description["tokens"] * cfg["hc_mult"] * cfg["dim"] * 2 > MAX_INPUT_BYTES:
        raise ValueError("real prefix hidden would exceed the existing diagnostic 128MiB tensor budget")
    if Path(args.out).exists():
        raise ValueError("frozen input output already exists")
    prefix_proof = artifact_preview(args.checkpoint, 0, args.layer, cfg)
    inventory = source_inventory()
    for key in list(os.environ):
        if key.startswith("V4_"):
            del os.environ[key]
    prefix_environment = dict(NUMERIC_ENV, V4_MAX_SEQ=str(args.max_tokens))
    os.environ.update(prefix_environment)
    for path in (ROOT, ENGINE):
        sys.path.insert(0, str(path))
    import torch
    from safetensors.torch import load_file, save_file
    import v4_stage as V
    import v4_numeric_diagnostics as D
    import v4_ref_slim as slim
    slim.set_active(False)
    slim.set_active_noqat(False)
    torch.set_num_threads(args.threads)
    model = V.config(args.checkpoint)
    model.max_batch_size, model.max_seq_len = 1, args.max_tokens
    V.verify_checkpoint_artifacts(args.checkpoint, required=True, lo=0, hi=args.layer,
                                  head=True, tail=False, dspark=False)
    stage = V.Stage(0, args.layer, model, head=True, tail=False, device=args.device)
    stage.load(args.checkpoint)
    ids = load_file(args.input, device="cpu")["token_ids"]
    with torch.no_grad(), D.execution_path(stage, "whole_eager"):
        hidden = stage.forward(stage.embed(ids), ids, 0).detach().cpu().contiguous()
    if description["sha256"] != file_hash(args.input) or inventory != source_inventory():
        raise ValueError("source or frozen IDs changed during prefix generation")
    target = Path(args.out).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file({"token_ids": ids, "hidden": hidden}, str(target))
    provenance = {"schema": "flyai-v4-frozen-prefix-input/1", "input": str(target), "sha256": file_hash(target),
                  "original_ids_sha256": description["sha256"], "prefix_span": [0, args.layer],
                  "recipe": prefix_environment, "threads": args.threads, "device": args.device,
                  "runtime_args_sha256": digest(vars(model)), "artifact_preview": prefix_proof,
                  "verified_artifacts": D._model_identity(stage)["model"],
                  "source_sha256": digest(inventory), "scope": "actual verified prefix prefill activations; standalone owned-stage input"}
    write_json(str(target) + ".provenance.json", provenance)
    return provenance


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="CPU-only recipe preparation; no model or tokenizer import")
    source = prep.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint"); source.add_argument("--fixture", action="store_true")
    prep.add_argument("--input", required=True)
    prep.add_argument("--lo", type=int, default=0); prep.add_argument("--hi", type=int)
    prep.add_argument("--device", default="cuda:0")
    prep.add_argument("--prompt-tokens", type=int, help="override derived nearby prefix for every scheduled boundary")
    prep.add_argument("--max-tokens", type=int, default=8192)
    prep.add_argument("--threads", type=int, default=1); prep.add_argument("--seed", type=int, default=7)
    prep.add_argument("--trace-max-bytes", type=int, default=MAX_JSON_BYTES)
    prep.add_argument("--log-max-bytes", type=int, default=MAX_LOG_BYTES,
                      help="raw log cap per stdout/stderr stream (1KiB..32MiB, at most64MiB per command)")
    prep.add_argument("--timeout", type=int, default=1800)
    prep.add_argument("--env-file"); prep.add_argument("--out-dir", required=True)
    runner = sub.add_parser("run", help="execution only with --execute on a reserved device")
    runner.add_argument("--plan", required=True); runner.add_argument("--execute", action="store_true")
    verifier = sub.add_parser("verify", help="recompute the saved gate from bounded raw evidence, without a GPU")
    verifier.add_argument("--plan", required=True); verifier.add_argument("--result", required=True)
    worker = sub.add_parser("trace-worker", help=argparse.SUPPRESS)
    worker.add_argument("--plan", required=True); worker.add_argument("--case", required=True)
    worker.add_argument("--execution", choices=("configured", "whole_eager"), required=True)
    worker.add_argument("--out", required=True); worker.add_argument("--rejection", action="store_true")
    worker.add_argument("--execute", action="store_true")
    freeze = sub.add_parser("freeze-ids", help="freeze an explicit JSON token-ID list; no tokenizer guessing")
    freeze.add_argument("--ids-json", required=True); freeze.add_argument("--out", required=True)
    make = sub.add_parser("make-input", help="explicit real verified prefix execution for original non-head hidden")
    make.add_argument("--checkpoint", required=True); make.add_argument("--input", required=True)
    make.add_argument("--layer", type=int, required=True); make.add_argument("--device", required=True)
    make.add_argument("--max-tokens", type=int, default=4096); make.add_argument("--threads", type=int, default=1)
    make.add_argument("--out", required=True); make.add_argument("--execute", action="store_true")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare" and not 1 <= args.timeout <= 86400:
        raise ValueError("process timeout must be 1..86400 seconds")
    if args.command == "trace-worker":
        return trace_worker(args)
    try:
        value = {"prepare": prepare, "run": run, "verify": verify, "freeze-ids": freeze_ids, "make-input": make_input}[args.command](args)
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired):
        if args.command == "run" and args.execute:
            destination = Path(args.plan).resolve().parent / "numeric-result.json"
            if destination.exists():
                failed = read_json(destination)
                failed.update(status="failed", hardware_verified=False, error="execution or independent verification refused; inspect stderr")
                write_json(destination, failed)
        raise
    if args.command == "prepare":
        value = {"status": value["status"], "plan": str(Path(args.out_dir).resolve() / "numeric-plan.json"),
                 "layer_span": [value["lo"], value["hi"]], "cases": len(value["cases"]),
                 "prepared_coverage": [item["kind"] for item in value["coverage"] if item["prepared"]],
                 "hardware_verified": False, "speed_pass": False}
    print(json.dumps(value, sort_keys=True))
    return 1 if value.get("status") == "failed" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired, StopIteration) as error:
        print(f"numeric acceptance refused: {error}", file=sys.stderr)
        raise SystemExit(2)

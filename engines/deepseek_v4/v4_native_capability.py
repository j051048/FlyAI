"""Runtime implementation of opt-in native-kernel smoke before model weights.

Synthetic native-format operands are bounded; no checkpoint is downloaded or
loaded. A pass is kernel smoke evidence, never whole-layer, stage or ring
qualification. Cached history never substitutes for executing this process.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import time

SCHEMA = "v4-native-kernel-capability/1"
CASES = ("fp4_quant", "fp8_gemm", "fp4_gemm", "grouped_moe", "sparse_attention")
STATUSES = {"pass", "fail", "unsupported", "unverified", "gpu_unavailable"}
SCOPE = "synthetic native-format kernel smoke; not whole-layer/model/stage/ring qualification"
MARKER = "V4_KERNEL_CAPABILITY "
_PROCESS_TOKEN = "FLYAI_NATIVE_PROBE_PROCESS_TOKEN"
_CLEANUP_TIMEOUT_S = 5.0


class _WindowsProbeJob:
    """Kernel-owned child tree, including compilers after the Python worker exits."""
    def __init__(self):
        import ctypes as C
        from ctypes import wintypes as W
        class Basic(C.Structure):
            _fields_ = [("user", C.c_longlong), ("job_user", C.c_longlong), ("flags", W.DWORD),
                ("min_ws", C.c_size_t), ("max_ws", C.c_size_t), ("active_limit", W.DWORD),
                ("affinity", C.c_size_t), ("priority", W.DWORD), ("scheduling", W.DWORD)]
        class IO(C.Structure):
            _fields_ = [(name, C.c_ulonglong) for name in ("read_ops", "write_ops", "other_ops", "read", "write", "other")]
        class Limits(C.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process_memory", C.c_size_t),
                       ("job_memory", C.c_size_t), ("peak_process", C.c_size_t), ("peak_job", C.c_size_t)]
        class Accounting(C.Structure):
            _fields_ = [(name, C.c_longlong) for name in ("user", "kernel", "period_user", "period_kernel")] + [
                (name, W.DWORD) for name in ("faults", "total", "active", "terminated")]
        self.C, self.Accounting = C, Accounting
        self.api = C.WinDLL("kernel32", use_last_error=True)
        signatures = {"CreateJobObjectW": ([C.c_void_p, W.LPCWSTR], W.HANDLE),
            "SetInformationJobObject": ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD], W.BOOL),
            "QueryInformationJobObject": ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p], W.BOOL),
            "AssignProcessToJobObject": ([W.HANDLE, W.HANDLE], W.BOOL),
            "IsProcessInJob": ([W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)], W.BOOL),
            "TerminateJobObject": ([W.HANDLE, W.UINT], W.BOOL),
            "OpenProcess": ([W.DWORD, W.BOOL, W.DWORD], W.HANDLE),
            "CloseHandle": ([W.HANDLE], W.BOOL)}
        for name, (args, result) in signatures.items():
            fn = getattr(self.api, name); fn.argtypes, fn.restype = args, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle: raise C.WinError(C.get_last_error())
        limits = Limits(); limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, C.byref(limits), C.sizeof(limits)):
            error = C.WinError(C.get_last_error()); self.close(); raise error

    def attach_handle(self, handle):
        from ctypes import wintypes as W
        member = W.BOOL()
        if not self.api.IsProcessInJob(handle, self.handle, self.C.byref(member)):
            raise self.C.WinError(self.C.get_last_error())
        if not member.value and not self.api.AssignProcessToJobObject(self.handle, handle):
            raise self.C.WinError(self.C.get_last_error())

    def attach_child(self, process):
        # A psutil Process retains birth identity; recheck it after opening the
        # handle so PID reuse can never enroll an unrelated application.
        if not process.is_running(): return
        handle = self.api.OpenProcess(0x100 | 0x1000 | 1, False, process.pid)
        if not handle:
            if not process.is_running(): return
            raise self.C.WinError(self.C.get_last_error())
        try:
            if process.is_running(): self.attach_handle(handle)
        finally:
            self.api.CloseHandle(handle)

    def active(self):
        values = self.Accounting()
        if not self.api.QueryInformationJobObject(self.handle, 1, self.C.byref(values), self.C.sizeof(values), None):
            raise self.C.WinError(self.C.get_last_error())
        return values.active

    def terminate(self):
        if not self.api.TerminateJobObject(self.handle, 1):
            raise self.C.WinError(self.C.get_last_error())

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle); self.handle = None


class _ProbeProcessOwner:
    def __init__(self, proc, token):
        import psutil
        self.psutil, self.proc, self.token = psutil, proc, token
        self.root = psutil.Process(proc.pid)
        self.birth = self.root.create_time()
        self.job = None
        self.known = {proc.pid: self.root}

    def enroll(self):
        if os.name == "nt":
            self.job = _WindowsProbeJob()
            self.job.attach_handle(int(self.proc._handle))
            # The Windows venv launcher can already have created its Python
            # child. Actual worker computation is still blocked on stdin.
            children = self.root.children(recursive=True)
            self.known.update({child.pid: child for child in children})
            for child in children: self.job.attach_child(child)

    def _alive(self, process):
        try:
            return process.is_running() and process.status() != self.psutil.STATUS_ZOMBIE
        except self.psutil.NoSuchProcess:
            return False

    def cleanup(self, timeout_s=_CLEANUP_TIMEOUT_S):
        deadline = time.monotonic() + timeout_s
        errors = []
        try:
            if self.job is not None:
                self.job.terminate()  # Only this explicitly created job object.
                for process in self.known.values():  # Also cover partial job enrollment failures.
                    try:
                        if self._alive(process): process.kill()
                    except self.psutil.NoSuchProcess: pass
            else:
                # An alive root anchors both ancestry and its independent PGID.
                if self._alive(self.root):
                    for child in self.root.children(recursive=True): self.known[child.pid] = child
                # Find same-launch orphans after nominal worker completion. The
                # random inherited token and birth floor are specific ownership,
                # never a process-name, executable or GPU-wide match.
                for process in self.psutil.process_iter():
                    if time.monotonic() >= deadline:
                        errors.append("owned-child discovery exceeded cleanup budget")
                        break
                    try:
                        if (process.create_time() >= self.birth and
                                process.environ().get(_PROCESS_TOKEN) == self.token):
                            self.known.setdefault(process.pid, process)
                    except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                        continue
                anchors = [p for p in self.known.values() if self._alive(p)]
                if os.name != "nt" and any(os.getpgid(p.pid) == self.proc.pid for p in anchors):
                    # At least one birth-verified owned member still anchors this
                    # group, so its ID cannot have become another process group.
                    try: os.killpg(self.proc.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                for process in anchors:  # Also catch an owned compiler that called setsid().
                    try: process.kill()
                    except self.psutil.NoSuchProcess: pass
            # Popen is the sole reaper of its direct child. No unbounded second
            # communicate(), and no competing psutil waitpid() for this PID.
            try: self.proc.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired: errors.append("worker did not exit within cleanup budget")
            while time.monotonic() < deadline:
                pending = (self.job.active() if self.job is not None else 0) + sum(self._alive(p) for p in self.known.values())
                if not pending: break
                time.sleep(min(.02, max(0, deadline - time.monotonic())))
            pending = (self.job.active() if self.job is not None else 0) + sum(self._alive(p) for p in self.known.values())
            if pending: errors.append("owned child processes remain active")
        except (OSError, self.psutil.Error) as error:
            errors.append(f"{type(error).__name__}: {error}")
        finally:
            if self.job is not None: self.job.close()
        return {"confirmed": not errors, "errors": errors,
                "scope": "owned Windows job object" if os.name == "nt" else "owned POSIX group and birth-identified children",
                "timeout_s": timeout_s}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _options(device, graph, max_probe_bytes, timeout_s):
    if not isinstance(device, str) or not re.fullmatch(r"cuda(?::[0-9]{1,3})?|cpu", device):
        raise ValueError("probe device must be cpu or a CUDA ordinal")
    if type(graph) is not bool or type(max_probe_bytes) is not int or not (512 << 20) <= max_probe_bytes <= (2048 << 20):
        raise ValueError("explicit graph boolean and 512..2048 MiB allocator budget required")
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 1 <= timeout_s <= 1800:
        raise ValueError("probe timeout must be 1..1800 seconds")
    return dict(device=device, graph=graph, max_probe_bytes=max_probe_bytes, timeout_s=timeout_s)


def _report(options, status, reason, *, identity=None, cases=None):
    return {"schema": SCHEMA, "status": status, "reason": reason, "scope": SCOPE,
        "runtime_ready": False, "qualification": "native_kernel_smoke_only" if status == "pass" else "none",
        "executed_gpu": any(row.get("executed_gpu") is True for row in cases or []),
        "gpu_execution_attempted": bool(cases), "identity": identity, "identity_sha256": _digest(identity) if identity else None,
        "options": options, "cases": cases or [], "measured_at": time.time(),
        "budget_scope": "Torch CUDA allocator cap; excludes driver context and compiler memory",
        "cache_authorizes_execution": False}


def validate_report(value):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("status") not in STATUSES:
        raise ValueError("invalid native capability report")
    _options(**value["options"])
    if value.get("runtime_ready") is not False or value.get("cache_authorizes_execution") is not False or value.get("scope") != SCOPE:
        raise ValueError("kernel smoke cannot authorize runtime readiness")
    identity = value.get("identity")
    if identity is not None and value.get("identity_sha256") != _digest(identity):
        raise ValueError("capability identity digest differs")
    if (type(value.get("measured_at")) not in (int, float) or not math.isfinite(value["measured_at"])
            or type(value.get("executed_gpu")) is not bool):
        raise ValueError("invalid probe execution observation")
    cases = value.get("cases")
    if (not isinstance(cases, list) or len(cases) > len(CASES)
            or any(not isinstance(row, dict) or row.get("name") not in CASES for row in cases)
            or [row["name"] for row in cases] != list(CASES[:len(cases)])):
        raise ValueError("invalid native probe coverage")
    if value["status"] == "pass":
        keys = {"gpu_uuid", "capability", "driver", "cuda", "torch", "tilelang", "source_sha256", "gpu_name"}
        if (value["options"]["device"] == "cpu" or not isinstance(identity, dict) or set(identity) != keys
                or any(identity[k] in (None, "", "unknown") for k in keys)
                or value.get("executed_gpu") is not True or value.get("qualification") != "native_kernel_smoke_only"
                or [row["name"] for row in cases] != list(CASES)
                or any(row.get("status") != "pass" or row.get("executed_gpu") is not True
                       or row.get("graph_replay") is not value["options"]["graph"] for row in cases)):
            raise ValueError("passing capability requires complete actual GPU evidence")
        if (any(not isinstance(identity[k], str) for k in keys - {"capability"})
                or not re.fullmatch(r"[0-9a-f]{64}", identity["source_sha256"])
                or not isinstance(identity["capability"], list) or len(identity["capability"]) != 2
                or any(type(v) is not int or not 0 <= v <= 99 for v in identity["capability"])):
            raise ValueError("invalid GPU/toolchain identity")
    elif value.get("qualification") != "none":
        raise ValueError("incomplete native probe cannot qualify kernels")
    return value


def _classify(error):
    text = str(error).lower()
    if isinstance(error, (ImportError, FileNotFoundError)) or "out of memory" in text:
        return "unverified"
    if isinstance(error, AttributeError) and (
            re.search(r"module ['\"](?:torch|tilelang)(?:\.[^'\"]+)?['\"] has no attribute", text)
            or "type object 'passconfigkey' has no attribute" in text):
        return "unverified"  # Installed toolchain API, not GPU ISA evidence.
    # Only concrete compiler/ISA diagnostics imply unsupported. Unknown kernel
    # failures and numerical mismatches must not be disguised as hardware limits.
    if any(x in text for x in ("no kernel image is available", "invalid device function",
        "unsupported gpu architecture", "requires .target sm_", "requires sm_", "not supported on this architecture")):
        return "unsupported"
    return "fail"


def _identity(torch):
    from shard.runtime_observation import gpu_uuid_text
    from v4_observability import source_inventory
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    uuid = gpu_uuid_text(getattr(props, "uuid", None))
    if not uuid:
        raise ValueError("actual GPU UUID unavailable")
    driver = subprocess.run(["nvidia-smi", "-i", uuid, "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    if not driver or "\n" in driver:
        raise ValueError("unambiguous selected GPU driver identity required")
    return {"gpu_uuid": uuid, "gpu_name": props.name, "capability": [props.major, props.minor],
        "driver": driver, "cuda": torch.version.cuda, "torch": str(torch.__version__),
        "tilelang": importlib.metadata.version("tilelang"), "source_sha256": _digest(source_inventory())}


def _check_graph(torch, call, mutate, enabled):
    expected = call()
    leaves = lambda x: list(x) if isinstance(x, tuple) else [x]
    # Packed E2M1 has no nonfinite encodings and torch does not universally
    # implement its direct .float() conversion, even on a supported GPU.
    if not all(t.dtype == torch.float4_e2m1fn_x2 or torch.isfinite(t.float()).all().item() for t in leaves(expected)):
        raise AssertionError("native kernel returned nonfinite output")
    if not enabled:
        return
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call(); call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = call()
    # Changed input plus two replays detects accidentally captured old values.
    for _ in range(2):
        mutate()
        expected = call()
        graph.replay()
        torch.cuda.synchronize()
        for actual, wanted in zip(leaves(result), leaves(expected)):
            if not torch.equal(actual.view(torch.uint8), wanted.view(torch.uint8)):
                raise AssertionError("native graph replay differs bytewise from native eager")


def _native_case(name, torch, K, graph):
    import v4_kernels_cpu as oracle
    import v4_moe_grouped as grouped
    import v4_sparse_attn_sm120 as attention
    sd = torch.float8_e8m0fnu
    rand = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * .02
    close = lambda a, b: torch.testing.assert_close(a.float(), b.float(), rtol=.02, atol=.001)
    if name == "fp4_quant":
        x = rand(32, 4096)
        def quantize(fn):
            packed, scale = fn(x)
            # The indexer/compressor runtime uses quantize-and-dequantize in
            # place. Packed weight conversion alone does not exercise it.
            inplace = fn(x.clone(), inplace=True)
            return packed, scale, inplace
        actual, expected = quantize(K.fp4_act_quant), quantize(oracle.fp4_act_quant)
        if any(not torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(actual, expected)):
            raise AssertionError("FP4 packed/inplace bytes or scales differ from reference quantizer")
        call, mutate = lambda: quantize(K.fp4_act_quant), lambda: x.copy_(rand(*x.shape))
        shape, comparison = [32, 4096], "exact packed, scale and inplace quant-dequant bytes"
    elif name in ("fp8_gemm", "fp4_gemm"):
        n, k = (1024, 4096) if name == "fp8_gemm" else (2048, 4096)
        x = rand(1, k)
        a, scale = K.act_quant(x, 128, "ue8m0", sd)
        w, ws = (grouped._fp8_block_quant(rand(n, k)) if name == "fp8_gemm" else K.fp4_act_quant(rand(n, k)))
        fn = getattr(K, name)
        call = lambda: fn(a, scale, w, ws, sd)
        def mutate():
            aq, sq = K.act_quant(rand(1, k), 128, "ue8m0", sd)
            a.view(torch.uint8).copy_(aq.view(torch.uint8)); scale.view(torch.uint8).copy_(sq.view(torch.uint8))
        expected = getattr(oracle, name)(a, scale, w[:128].contiguous(),
            ws[:1 if name == "fp8_gemm" else 128].contiguous(), sd)
        close(call()[:, :128], expected)
        shape, comparison = [1, n, k], "first 128 output rows vs dequantized torch oracle; rtol=.02 atol=.001"
    elif name == "grouped_moe":
        # Exact deployed active-expert shapes without allocating 256 experts.
        # This probes the two grouped GEMMs and original scalar arithmetic, not
        # router/model coverage. Each slot has distinct native FP4 bytes.
        def bank(n, k):
            values = [K.fp4_act_quant(rand(n, k)) for _ in range(6)]
            return tuple(torch.stack([row[i].view(torch.uint8) for row in values]).view(values[0][i].dtype)
                         for i in (0, 1))
        w13, s13 = bank(4096, 4096); w2, s2 = bank(4096, 2048)
        x, weights = rand(1, 4096), torch.arange(1, 7, device="cuda", dtype=torch.float32)[:, None] / 21
        def chain(group):
            def gemm(a, scales, w, ws):
                if group: return grouped.grouped_fp4_gemm(a, scales, w, ws, sd)
                return torch.cat([K.fp4_gemm(a[i:i+1], scales[i:i+1], w[i], ws[i], sd) for i in range(6)])
            a, scales = K.act_quant(x, 128, "ue8m0", sd)
            both = gemm(a.expand(6, -1).contiguous(), scales.expand(6, -1).contiguous(), w13, s13)
            h = (torch.nn.functional.silu(both[:, :2048].float().clamp(max=10)) *
                 both[:, 2048:].float().clamp(-10, 10) * weights).to(torch.bfloat16)
            a, scales = K.act_quant(h, 128, "ue8m0", sd)
            return gemm(a, scales, w2, s2)
        if not torch.equal(chain(True), chain(False)):
            raise AssertionError("grouped active expert path differs from six original FP4 experts")
        call, mutate = lambda: chain(True), lambda: x.copy_(rand(1, 4096))
        shape, comparison = [6, 4096, 2048], "six active experts; exact per-expert GEMM outputs; no router qualification"
    else:
        q, kv = rand(1, 1, 64, 512), rand(1, 128, 512)
        sink = torch.zeros(64, device="cuda", dtype=torch.float32)
        ids = torch.arange(128, device="cuda", dtype=torch.int32).view(1, 1, 128)
        fn = K.sparse_attn
        call = lambda: fn(q, kv, sink, ids, 512 ** -.5)
        close(call(), attention.sparse_attn_eager(q, kv, sink, ids, 512 ** -.5))
        mutate = lambda: q.copy_(rand(*q.shape))
        shape, comparison = [1, 1, 64, 512, 128], "attention vs torch oracle; rtol=.02 atol=.001"
    _check_graph(torch, call, mutate, graph)
    torch.cuda.synchronize()
    return {"name": name, "status": "pass", "executed_gpu": True, "graph_replay": graph,
            "shape": shape, "comparison": comparison}


def _worker(options):
    try:
        import torch
    except ImportError as error:
        return _report(options, "unverified", str(error))
    if options["device"] == "cpu" or not torch.cuda.is_available():
        return _report(options, "gpu_unavailable", "CPU cannot qualify native CUDA kernels")
    identity, cases = None, []
    try:
        selected = torch.cuda.current_device() if options["device"] == "cuda" else options["device"]
        torch.cuda.set_device(selected)
        identity = _identity(torch)
        if any(not hasattr(torch, name) for name in ("float4_e2m1fn_x2", "float8_e8m0fnu", "float8_e4m3fn")):
            return _report(options, "unverified", "installed Torch lacks the native checkpoint dtypes", identity=identity)
        free, total = torch.cuda.mem_get_info()
        if free < options["max_probe_bytes"] + (256 << 20):
            return _report(options, "unverified", "insufficient free memory for bounded probe and safety margin", identity=identity)
        torch.cuda.set_per_process_memory_fraction(min(1.0, options["max_probe_bytes"] / total))
        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(1729)
        import v4_ref_cpu
        sys.path.insert(0, v4_ref_cpu.INFERENCE_DIR)
        import kernel as K
        if getattr(K, "_v4_cpu_backend", False):
            raise RuntimeError("native probe refuses a CPU kernel shim")
        import v4_sparse_attn_sm120
        v4_sparse_attn_sm120.install_sm120()
        for name in CASES:
            started = time.perf_counter()
            try:
                result = _native_case(name, torch, K, options["graph"])
            except Exception as error:
                status = _classify(error)
                cases.append({"name": name, "status": status, "executed_gpu": None,
                    "graph_replay": False, "reason": str(error)[:1000], "error_class": type(error).__name__})
                return _report(options, status, "native case did not qualify: " + name, identity=identity, cases=cases)
            result["elapsed_seconds"] = time.perf_counter() - started
            cases.append(result)
            torch.cuda.empty_cache()
        if _identity(torch) != identity:
            raise RuntimeError("native probe source/runtime identity changed during execution")
        return validate_report(_report(options, "pass", "all requested native kernel probes executed", identity=identity, cases=cases))
    except Exception as error:
        return _report(options, "unverified" if identity is None else _classify(error),
                       str(error)[:1000], identity=identity, cases=cases)


def _worker_command():
    return [sys.executable, str(Path(__file__).resolve()), "--worker"]


def _run_isolated(command, payload, timeout_s):
    # Tree ownership is established before releasing stdin, which is the real
    # worker's gate before any imports that can launch native compilers.
    try:
        import psutil  # Required for ownership checks, even when CUDA is absent.
    except ImportError:
        return {"timeout": False, "returncode": None, "tail": "", "cleanup": {
            "confirmed": False, "errors": ["psutil is required before launching a native probe"]}}
    # File-backed output avoids unbounded capture_output RAM if a compiler is noisy.
    with tempfile.TemporaryFile() as output:
        token = secrets.token_hex(32)
        proc = subprocess.Popen(command,
            stdin=subprocess.PIPE, stdout=output, stderr=output,
            env={**os.environ, "V4_KERNELS": "tilelang", "PYTHONUTF8": "1", _PROCESS_TOKEN: token},
            **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
               if os.name == "nt" else {"start_new_session": True}))
        owner = None
        try:
            owner = _ProbeProcessOwner(proc, token)
            owner.enroll()
        except (OSError, psutil.Error) as error:
            # No input has been released: model/compiler work has not begun.
            # We cannot certify complete tree enrollment, so never qualify.
            if owner is not None:
                owner.cleanup()
            else:
                try: proc.kill()
                except OSError: pass
                try: proc.wait(timeout=_CLEANUP_TIMEOUT_S)
                except subprocess.TimeoutExpired: pass
            if proc.stdin is not None: proc.stdin.close()
            return {"timeout": False, "returncode": proc.returncode, "tail": "", "cleanup": {
                "confirmed": False, "errors": ["worker tree enrollment failed: " + str(error)[:500]]}}
        timed_out = False
        try:
            proc.communicate(payload, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            cleanup = owner.cleanup()
            if proc.stdin is not None: proc.stdin.close()
        size = output.seek(0, 2); output.seek(max(0, size - (64 << 10)))
        tail = output.read().decode("utf-8", errors="replace")
    return {"timeout": timed_out, "returncode": proc.returncode, "tail": tail, "cleanup": cleanup}


def _run_worker(options):
    execution = _run_isolated(_worker_command(), json.dumps(options).encode(), options["timeout_s"])
    cleanup = execution["cleanup"]
    if cleanup.get("confirmed") is not True or execution["timeout"]:
        reason = ("native probe exceeded its wall-time budget" if execution["timeout"] else
                  "native probe process-tree cleanup was not confirmed")
        return dict(_report(options, "unverified", reason), process_cleanup=cleanup)
    tail = execution["tail"]
    lines = [line[len(MARKER):] for line in tail.splitlines() if line.startswith(MARKER)]
    if execution["returncode"] or len(lines) != 1:
        return dict(_report(options, "fail", "isolated probe exited without a unique complete report"), process_cleanup=cleanup)
    try:
        value = validate_report(json.loads(lines[0]))
        if value["options"] != options:
            raise ValueError("native worker options differ from request")
    except (ValueError, TypeError, KeyError) as error:
        return dict(_report(options, "fail", "invalid isolated probe report: " + str(error)[:500]), process_cleanup=cleanup)
    return dict(value, process_cleanup=cleanup)


def probe_native_capability(device="cuda:0", *, graph=False, cache_dir=None, max_probe_bytes=512 << 20, timeout_s=300):
    options = _options(device, graph, max_probe_bytes, timeout_s)
    result = _run_worker(options)  # Always execute; cached pass is never authorization.
    result["previous_matching_report"] = False
    if cache_dir is not None and result.get("identity") is not None:
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink():
            raise ValueError("capability cache cannot be a symlink")
        key = _digest({"identity": result["identity"], "options": options})
        path = directory / (key + ".json")
        if path.is_symlink():
            raise ValueError("capability cache report cannot be a symlink")
        if path.is_file() and path.stat().st_size < (64 << 10):
            try:
                prior = validate_report(json.loads(path.read_text(encoding="utf-8")))
                result["previous_matching_report"] = (prior["identity"] == result["identity"] and prior["options"] == options
                    and 0 <= time.time() - prior["measured_at"] <= 86400)
            except (ValueError, TypeError, KeyError):
                pass
        # Store bounded diagnostic history. It never skips a fresh qualification.
        fd, temporary = tempfile.mkstemp(prefix=key, suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(result, stream, sort_keys=True, allow_nan=False)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return validate_report(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--max-probe-mib", type=int, default=512)
    parser.add_argument("--timeout-s", type=float, default=300)
    parser.add_argument("--cache-dir")
    args = parser.parse_args(argv)
    if args.worker:
        result = _worker(_options(**json.load(sys.stdin)))
        print(MARKER + json.dumps(result, allow_nan=False), flush=True)
        return 0
    result = probe_native_capability(args.device, graph=args.graph, max_probe_bytes=args.max_probe_mib << 20,
                                    timeout_s=args.timeout_s, cache_dir=args.cache_dir)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    engine = Path(__file__).resolve().parent
    for path in (engine, engine.parent.parent):
        if str(path) not in sys.path: sys.path.insert(0, str(path))
    raise SystemExit(main())

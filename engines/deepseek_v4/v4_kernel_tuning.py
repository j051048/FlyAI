"""Startup-only, exact-gated kernel selection under the serving measurement mode.

Cached choices never authorize numerical correctness: the current process probes
them again. Timing is local operator evidence, not an end-to-end speed claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import uuid

SCHEMA = "v4-kernel-selection/1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def environment_identity(torch_module, *, policy, source_sha256):
    if policy not in ("eager", "graph"):
        raise ValueError("kernel measurement policy must match eager or graph serving")
    import importlib.metadata
    dev = torch_module.cuda.current_device()
    props = torch_module.cuda.get_device_properties(dev)
    query = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
    if query.returncode or not query.stdout.strip():
        raise ValueError("kernel tuning requires a readable driver identity")
    return {"gpu": str(props.name), "gpu_uuid": str(getattr(props, "uuid", "unknown")),
            "capability": [props.major, props.minor], "torch": str(torch_module.__version__),
            "cuda": torch_module.version.cuda, "driver": query.stdout.splitlines()[0].strip(),
            "tilelang": importlib.metadata.version("tilelang"), "policy": policy,
            "source_sha256": source_sha256}


def measure_cuda(call, torch_module, *, policy, warmup=3, repeats=7, iterations=10):
    """Only called during pre-capture warmup, never on a live token dispatch."""
    if policy not in ("eager", "graph") or min(warmup, repeats, iterations) < 1:
        raise ValueError("invalid bounded kernel measurement")
    if repeats > 31 or iterations > 1000:
        raise ValueError("kernel measurement exceeds startup budget")
    if torch_module.cuda.is_current_stream_capturing():
        raise RuntimeError("kernel tuning is forbidden inside CUDA capture")
    stream = torch_module.cuda.Stream()
    stream.wait_stream(torch_module.cuda.current_stream())
    with torch_module.cuda.stream(stream):
        for _ in range(warmup):
            call()
    stream.synchronize()
    graph = None
    if policy == "graph":
        graph = torch_module.cuda.CUDAGraph()
        with torch_module.cuda.graph(graph, stream=stream):
            call()
    samples = []
    for _ in range(repeats):
        if policy == "graph":
            start, end = (torch_module.cuda.Event(enable_timing=True) for _ in range(2))
            with torch_module.cuda.stream(stream):
                start.record(stream)
                for _ in range(iterations):
                    graph.replay()
                end.record(stream)
            end.synchronize()
            elapsed = start.elapsed_time(end)
        else:
            # Eager includes the submitting host's cost; graph uses replay timing.
            stream.synchronize()
            started = time.perf_counter()
            with torch_module.cuda.stream(stream):
                for _ in range(iterations):
                    call()
            stream.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
        samples.append(elapsed / iterations)
    return statistics.median(samples)


class KernelSelector:
    def __init__(self, identity, directory=None, *, clock=time.time, ttl_s=86400, max_entries=256):
        self.identity = json.loads(json.dumps(identity, allow_nan=False))
        if identity.get("policy") not in ("eager", "graph"):
            raise ValueError("explicit serving measurement policy required")
        if type(ttl_s) not in (int, float) or not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ValueError("finite tuning cache TTL required")
        self.clock, self.ttl_s, self.max_entries = clock, ttl_s, max_entries
        if type(max_entries) is not int or not 1 <= max_entries <= 4096:
            raise ValueError("bounded tuning cache required")
        self.directory = Path(directory).absolute() if directory else None
        if self.directory:
            for p in (self.directory, *self.directory.parents):
                if p.exists() and (p.is_symlink() or getattr(p.lstat(), "st_file_attributes", 0) & 0x400):
                    raise ValueError("tuning cache must not traverse symlinks")
            self.directory.mkdir(parents=True, exist_ok=True)
        self.last = {}

    def _path(self, operation):
        return self.directory / (digest({"environment": self.identity, "operation": operation}) + ".json")

    def _read(self, operation, choices):
        if not self.directory:
            return None
        path = self._path(operation)
        try:
            if path.is_symlink() or path.stat().st_size > 65536:
                return None
            value = json.loads(path.read_text())
            age = self.clock() - value["measured_at"]
            if (value["schema"] != SCHEMA or value["identity"] != self.identity or
                    value["operation"] != operation or not 0 <= age <= self.ttl_s or
                    value["choice"] not in choices or value.get("sha256") != digest(
                        {k: v for k, v in value.items() if k != "sha256"})):
                return None
            return value
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def select(self, operation, choices, *, probe, measure):
        if (not isinstance(choices, (tuple, list)) or not 1 <= len(choices) <= 16 or
                len(set(choices)) != len(choices)):
            raise ValueError("unique bounded candidate names required")
        cached = self._read(operation, choices)
        if cached is not None and probe(cached["choice"]):
            self.last = {**cached, "cache_reused": True, "current_process_reprobed": True}
            return cached["choice"]
        timings, refused = {}, []
        for name in choices:
            try:
                if not probe(name):
                    refused.append(name)
                    continue
                values = measure(name)
                if (not isinstance(values, dict) or not values or len(values) > 32 or
                        any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values.values())):
                    raise ValueError("finite per-workload timings required")
                timings[name] = values
            except Exception:
                refused.append(name)
        baseline = timings.get(choices[0])
        if baseline is None:
            raise RuntimeError("reference kernel did not complete startup measurement")
        # Require a win without >1% regression on any declared representative shape.
        safe = {name: max(values[k] / baseline[k] for k in baseline)
                for name, values in timings.items() if set(values) == set(baseline)}
        candidates = [name for name in choices if safe.get(name, math.inf) <= .98]
        choice = min(candidates, key=lambda name: (safe[name], choices.index(name))) if candidates else choices[0]
        value = {"schema": SCHEMA, "identity": self.identity, "operation": operation,
                 "choice": choice, "timings_ms": timings, "refused": refused,
                 "measured_at": self.clock()}
        value["sha256"] = digest(value)
        self.last = {**value, "cache_reused": False, "current_process_reprobed": True}
        if self.directory:
            # Cache availability is not a precondition for reference execution.
            try:
                paths = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
                if len(paths) < self.max_entries or self._path(operation).exists():
                    path = self._path(operation)
                    temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
                    with temporary.open("x") as stream:
                        json.dump(value, stream, sort_keys=True, allow_nan=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, path)
            except OSError:
                pass
        return choice

"""Bounded per-job phase timing, with CUDA events resolved outside the hot path.

Host times measure dispatch/wait on the calling thread. GPU times measure the
selected stream's interval, so overlapping phases must not be added together.
Percentiles describe a bounded rolling sample, while count/total/min/max cover
all measured intervals. This module itself does not import torch.
"""
from collections import deque
from contextlib import contextmanager
import math
import re
import time

SCHEMA = "shard-runtime-profile/1"
_NAME = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_FIELDS = {"count", "total_ms", "min_ms", "max_ms", "p50_ms", "p95_ms", "samples"}


def validate_profile(value):
    if not isinstance(value, dict) or set(value) != {
            "schema", "gpu_sample_every", "sample_capacity", "phases", "dropped_gpu_samples"}:
        raise ValueError("invalid runtime performance fields")
    if value["schema"] != SCHEMA:
        raise ValueError("unknown runtime performance schema")
    for key in ("gpu_sample_every", "sample_capacity", "dropped_gpu_samples"):
        v = value[key]
        if type(v) is not int or v < (0 if key == "dropped_gpu_samples" else 1):
            raise ValueError(f"invalid performance {key}")
    if value["sample_capacity"] > 8192 or value["gpu_sample_every"] > 1000000:
        raise ValueError("performance sampling budget exceeds its limit")
    phases = value["phases"]
    if not isinstance(phases, dict) or len(phases) > 64:
        raise ValueError("invalid performance phases")
    for name, row in phases.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name) or \
                not isinstance(row, dict) or set(row) != _FIELDS:
            raise ValueError("invalid performance phase")
        for key in ("count", "samples"):
            if type(row[key]) is not int or row[key] < 1:
                raise ValueError(f"invalid performance {key}")
        if row["samples"] > min(row["count"], value["sample_capacity"]):
            raise ValueError("performance sample count exceeds measured intervals")
        for key in _FIELDS - {"count", "samples"}:
            v = row[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"invalid performance {key}")
        if not row["min_ms"] <= row["p50_ms"] <= row["p95_ms"] <= row["max_ms"]:
            raise ValueError("inconsistent performance percentiles")
        mean = row["total_ms"] / row["count"]
        if not row["min_ms"] - 1e-8 <= mean <= row["max_ms"] + 1e-8:
            raise ValueError("inconsistent performance total")
    return value


class RuntimeProfiler:
    def __init__(self, *, gpu_sample_every=16, sample_capacity=512, max_pending=2048, clock=None):
        if any(type(v) is not int or v < 1 for v in (gpu_sample_every, sample_capacity, max_pending)):
            raise ValueError("profile budgets must be positive integers")
        if sample_capacity > 8192 or gpu_sample_every > 1000000 or max_pending > 8192:
            raise ValueError("profile budget exceeds its limit")
        self.every, self.capacity, self.max_pending = gpu_sample_every, sample_capacity, max_pending
        self.clock = clock or time.perf_counter
        self.reset()

    def reset(self):
        # A job reset is a barrier. Do not retain events/tensors from an old job.
        self._phases, self._seen = {}, {}
        self._pending = deque()
        self.dropped = 0

    def record(self, name, elapsed_ms):
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("invalid performance phase name")
        if isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, (int, float)) or \
                not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError("invalid performance duration")
        if name not in self._phases:
            if len(self._phases) >= 64:
                raise ValueError("too many performance phases")
            self._phases[name] = {"count": 0, "total_ms": 0.0, "min_ms": math.inf,
                                  "max_ms": 0.0, "values": deque(maxlen=self.capacity)}
        row = self._phases[name]
        row["count"] += 1
        row["total_ms"] += elapsed_ms
        row["min_ms"] = min(row["min_ms"], elapsed_ms)
        row["max_ms"] = max(row["max_ms"], elapsed_ms)
        row["values"].append(float(elapsed_ms))

    @contextmanager
    def host(self, name):
        start = self.clock()
        try:
            yield
        finally:
            self.record(name, max(0.0, (self.clock() - start) * 1000.0))

    def _drain(self, wait=False):
        # Querying an event does not wait for the GPU. Draining at submission
        # boundaries keeps instrumentation memory bounded during long jobs.
        remaining = deque()
        while self._pending:
            name, start, end = self._pending.popleft()
            if wait:
                end.synchronize()
            if wait or end.query():
                self.record(name, float(start.elapsed_time(end)))
            else:
                remaining.append((name, start, end))
        self._pending = remaining

    @contextmanager
    def gpu(self, name, torch_module, device):
        if not str(device).startswith("cuda") or torch_module.cuda.is_current_stream_capturing():
            yield
            return
        count = self._seen.get(name, 0)
        self._seen[name] = count + 1
        if count % self.every:
            yield
            return
        self._drain()
        if len(self._pending) >= self.max_pending:
            self.dropped += 1
            yield
            return
        with torch_module.cuda.device(device):
            stream = torch_module.cuda.current_stream(device)
            start = torch_module.cuda.Event(enable_timing=True)
            end = torch_module.cuda.Event(enable_timing=True)
            start.record(stream)
            try:
                yield
            finally:
                end.record(stream)
                self._pending.append((name, start, end))

    def snapshot(self):
        self._drain(wait=True)
        rows = {}
        for name, row in self._phases.items():
            values = sorted(row["values"])
            percentile = lambda p: values[min(len(values) - 1, math.ceil(p * (len(values) - 1)))]
            rows[name] = {key: row[key] for key in ("count", "total_ms", "min_ms", "max_ms")}
            rows[name].update(p50_ms=percentile(.5), p95_ms=percentile(.95), samples=len(values))
        return validate_profile({"schema": SCHEMA, "gpu_sample_every": self.every,
            "sample_capacity": self.capacity, "phases": rows, "dropped_gpu_samples": self.dropped})

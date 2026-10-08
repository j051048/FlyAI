"""Bounded local stage timings and independent authenticated route RTT probes.

Durations use one machine's monotonic clock. CUDA events are queried after
completion; profiling never synchronizes the inference stream to collect data.
"""
from __future__ import annotations
from collections import defaultdict, deque
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import secrets
import socket
import tempfile
import threading
import time

from .performance import percentile


def probe_peer(endpoint, config, target_index, send, recv, *, samples=3, timeout=2):
    from .pipeline_session import hello_client, send_message, recv_message
    from urllib.parse import urlsplit
    address = urlsplit("//" + endpoint)
    sock = socket.create_connection((address.hostname, address.port), timeout=timeout)
    values = []
    try:
        channel = hello_client(sock, config, target_index, "probe", send, recv)
        # The channel's identity was checked by HELLO. These frames stay on
        # its independent control connection and never enter a model queue.
        for index in range(samples):
            nonce = secrets.token_hex(32)
            started = time.perf_counter()
            send_message(send, channel, {"op": "probe_ping", "nonce": nonce, "seq": index})
            reply = recv_message(recv, channel)
            elapsed = (time.perf_counter() - started) * 1000
            if reply != {"op": "probe_pong", "nonce": nonce, "seq": index}:
                raise ConnectionError("route probe response mismatch")
            values.append(elapsed)
        return {"scope": "independent_authenticated_control_channel_rtt",
                "samples": len(values), "p50_ms": percentile(values, .5),
                "p95_ms": percentile(values, .95), "measured_at": time.time()}
    finally:
        sock.close()


class StageTelemetry:
    def __init__(self, *, node_id, cohort_id, index, path=None, window=256):
        self.node_id, self.cohort_id, self.index = node_id, cohort_id, index
        self.path = Path(path) if path else None
        self.window = window
        self.values = defaultdict(lambda: deque(maxlen=window))
        self.pending = deque(maxlen=window)
        self.probe = None
        self.lock = threading.RLock()
        self.stop = threading.Event()

    def record(self, name, duration_ms):
        with self.lock:
            self.values[name].append(float(duration_ms))

    @contextmanager
    def measure(self, name, device=None):
        started, events = time.perf_counter(), None
        if device is not None and str(device).startswith("cuda"):
            import torch
            if torch.cuda.is_available():
                with torch.cuda.device(device):
                    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    events[0].record()
        try:
            yield
        finally:
            self.record(name + "_host_ms", (time.perf_counter() - started) * 1000)
            if events is not None:
                with torch.cuda.device(device): events[1].record()
                with self.lock: self.pending.append((name, events))

    def wrap(self, stage):
        for name in ("prefill", "decode", "forward", "load"):
            method = getattr(stage, name, None)
            if method is None: continue
            def measured(*args, _method=method, _name=name, **kwargs):
                with self.measure(_name, getattr(stage, "dev", getattr(stage, "device", None))):
                    return _method(*args, **kwargs)
            setattr(stage, name, functools.wraps(method)(measured))
        return stage

    def snapshot(self):
        with self.lock:
            pending = []
            for name, events in self.pending:
                if events[1].query(): self.values[name + "_gpu_ms"].append(events[0].elapsed_time(events[1]))
                else: pending.append((name, events))
            self.pending.clear(); self.pending.extend(pending)
            return {"schema": "shard-stage-telemetry/1", "node_id": self.node_id,
                "cohort_id": self.cohort_id, "stage_index": self.index, "measured_at": time.time(),
                "scope": "rolling_local_durations; control RTT is not one-way latency or inference wait",
                "durations": {name: {"samples": len(values), "p50": percentile(values, .5),
                                      "p95": percentile(values, .95)} for name, values in self.values.items()},
                "network_probe": self.probe, "pending_gpu_samples": len(self.pending)}

    def write(self):
        if self.path is None: return
        if self.path.is_symlink(): raise ValueError("telemetry path cannot be a symlink")
        fd, temporary = tempfile.mkstemp(prefix=".stage-telemetry-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                json.dump(self.snapshot(), out, allow_nan=False)
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def start(self, *, endpoint=None, config=None, target_index=None, send=None, recv=None, interval_s=5):
        def run():
            while not self.stop.is_set():
                if endpoint and config is not None:
                    try:
                        self.probe = probe_peer(endpoint, config, target_index, send, recv)
                    except (OSError, ValueError):
                        self.probe = {"scope": "independent_authenticated_control_channel_rtt",
                                      "available": False, "measured_at": time.time()}
                try: self.write()
                except OSError: pass
                self.stop.wait(interval_s)
        thread = threading.Thread(target=run, daemon=True, name="stage-route-telemetry")
        thread.start()
        return self.stop

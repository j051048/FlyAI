"""Bounded local boundary transport experiment; never rents or downloads hardware.

This uses the actual V4 frame builder/decoder and loopback TCP with synthetic
activations. It is diagnostic data, not an authenticated ring cost observation,
a model benchmark, or a proof that CUDA peer/IPC transport was used.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import queue
import socket
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
ENGINE = ROOT / "engines" / "deepseek_v4"
if str(ENGINE) not in sys.path: sys.path.insert(0, str(ENGINE))


def measure(*, source_device="cpu", target_device="cpu", tokens=1, dim=4096, hc=4,
            repeats=20, warmup=3, buffer_mib=0, pinned=False):
    if (type(tokens) is not int or not 1 <= tokens <= 64 or type(dim) is not int or not 1 <= dim <= 8192
            or type(hc) is not int or not 1 <= hc <= 4 or type(repeats) is not int or not 1 <= repeats <= 100
            or type(warmup) is not int or not 0 <= warmup <= 10 or type(buffer_mib) is not int or not 0 <= buffer_mib <= 64):
        raise ValueError("local probe geometry/repeats/buffers exceed bounded limits")
    import torch
    import v4_pipe as vp
    wire = vp.transport_module()
    if vp.V4_FP8_WIRE or vp.V4_SEALED_IDS:
        raise ValueError("diagnostic currently requires the declared BF16/raw-ID boundary contract")
    devices = [torch.device(source_device), torch.device(target_device)]
    devices = [torch.device("cuda", torch.cuda.current_device()) if d.type == "cuda" and d.index is None
               and torch.cuda.is_available() else d for d in devices]
    metadata = []
    for device in devices:
        if device.type not in ("cpu", "cuda"):
            raise ValueError("explicit CPU or CUDA devices required")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("actual CUDA device is unavailable; no CPU fallback")
            properties = torch.cuda.get_device_properties(device)
            metadata.append({"device": str(device), "uuid": str(getattr(properties, "uuid", "")),
                             "name": properties.name, "compute_capability": [properties.major, properties.minor]})
            free, _ = torch.cuda.mem_get_info(device)
            if tokens * hc * dim * 2 * 4 > free * .05:
                raise RuntimeError("bounded boundary probe exceeds 5% of device free VRAM")
        else:
            metadata.append({"device": "cpu", "uuid": None, "name": "CPU diagnostic", "compute_capability": None})
    wire.configure_receive_pool(buffer_mib << 20, slots=2, pinned=pinned)
    listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1)
    sender = socket.create_connection(listener.getsockname(), timeout=10)
    receiver, _ = listener.accept(); listener.close()
    sender.settimeout(10); receiver.settimeout(10)
    sender.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    receiver.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    outputs = queue.Queue(maxsize=1)
    def receive():
        try:
            for _ in range(repeats + warmup):
                message = wire.recv_msg(receiver)
                hidden, ids, digest_bytes = vp._recv_hids(message, True)
                started = time.perf_counter()
                target = hidden.to(devices[1])
                if devices[1].type == "cuda": torch.cuda.synchronize(devices[1])
                ready = time.perf_counter()
                expected = message["payload_sha256"]
                if hashlib.sha256(digest_bytes).hexdigest() != expected:
                    raise ValueError("boundary bytes changed in local transport")
                outputs.put({"ready_at": ready, "h2d_host_s": ready - started,
                             "payload_sha256": expected, "target_shape": list(target.shape)}, timeout=10)
                del target, hidden, ids, digest_bytes, message
        except BaseException as error:
            # A traceback would retain message tensors and their borrowed slot,
            # allowing cleanup to mask the actual decoder failure.
            outputs.put({"error_class": type(error).__name__, "error": str(error)[:500]}, timeout=10)
    thread = threading.Thread(target=receive, daemon=True); thread.start()
    samples = []
    try:
        # Build the small synthetic payload on CPU: avoid two simultaneous
        # int64 temporaries on the selected GPU beyond the declared VRAM bound.
        hidden = torch.arange(tokens * hc * dim, device="cpu").remainder(113).to(torch.bfloat16).reshape(1, tokens, hc, dim).to(devices[0])
        ids = torch.arange(tokens, device="cpu").reshape(1, tokens)
        if devices[0].type == "cuda": torch.cuda.synchronize(devices[0])
        for index in range(repeats + warmup):
            started = time.perf_counter()
            frame, payload = vp._make_step_frame(hidden, ids, 1, True)
            frame["payload_sha256"] = hashlib.sha256(payload).hexdigest()
            packed = time.perf_counter()
            sent_bytes = wire.send_msg(sender, frame)
            result = outputs.get(timeout=15)
            if "error_class" in result:
                raise RuntimeError(f"local receiver failed ({result['error_class']}): {result['error']}")
            if result["target_shape"] != list(hidden.shape):
                raise ValueError("local transport geometry differs")
            if index >= warmup:
                samples.append({"payload_ready_to_ready_ms": (result["ready_at"] - started) * 1000,
                    "pack_d2h_and_payload_hash_host_ms": (packed - started) * 1000,
                    "target_h2d_host_ms": result["h2d_host_s"] * 1000, "wire_bytes": sent_bytes})
        thread.join(15)
        if thread.is_alive(): raise TimeoutError("local transport receiver did not finish")
        gc.collect()
        return {"schema": "flyai-v4-local-transport-diagnostic/1", "status": "diagnostic_only",
            "transport": "buffered_tcp" if buffer_mib else "loopback_tcp", "devices": metadata,
            "geometry": {"batch": 1, "tokens": tokens, "hc": hc, "dim": dim, "dtype": "bfloat16"},
            "warmup": warmup, "samples": samples, "receive_pool": wire.receive_pool_status(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "synthetic boundary ready-to-ready; unleased, no strict session envelope; never planner engine cost",
            "gpu_evidence": "actual CUDA copies only" if any(d.type == "cuda" for d in devices) else "none; CPU fixture",
            "performance_qualified": False}
    finally:
        sender.close(); receiver.close()
        thread.join(10)
        gc.collect()
        wire.configure_receive_pool(0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-device", default="cpu")
    parser.add_argument("--target-device", default="cpu")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--buffer-mib", type=int, default=0)
    parser.add_argument("--pinned", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = measure(source_device=args.source_device, target_device=args.target_device, tokens=args.tokens,
        repeats=args.repeats, buffer_mib=args.buffer_mib, pinned=args.pinned)
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "samples": len(report["samples"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

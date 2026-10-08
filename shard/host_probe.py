"""Opt-in, bounded simultaneous H2D measurement for one multi-GPU host.

The small transfer buffer proves only its own pin allocation. A separately requested
pin budget is actually allocated and reported as an observed lower bound, not an OS
promise or a reservation for a later stage process. This module does no work on import.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import threading
import time

try:
    from shard.resources import measure_host_resources
except ImportError:
    from resources import measure_host_resources


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_probe(devices, sample_bytes, repeats, pin_budget_bytes, available_ram_bytes):
    if not devices or any(type(x) is not int or x < 0 for x in devices) or len(set(devices)) != len(devices):
        raise ValueError("devices must be distinct nonnegative local CUDA indices")
    _positive_int(sample_bytes, "sample_bytes")
    _positive_int(repeats, "repeats")
    if repeats > 1000 or sample_bytes > 256 * 1024**2:
        raise ValueError("probe exceeds bounded repeat/transfer limits")
    if type(pin_budget_bytes) is not int or pin_budget_bytes < 0:
        raise ValueError("pin_budget_bytes must be a nonnegative integer")
    required = max(sample_bytes * len(devices), pin_budget_bytes)
    if type(available_ram_bytes) is not int or available_ram_bytes <= 0:
        raise ValueError("available host RAM is unknown")
    if required > available_ram_bytes * .8:
        raise ValueError("probe allocation exceeds 80% of currently available host RAM")
    return required


def summarize_transfers(rows, wall_seconds, sample_bytes, repeats):
    _positive_int(sample_bytes, "sample_bytes")
    _positive_int(repeats, "repeats")
    if (not rows or type(wall_seconds) not in (int, float) or
            not math.isfinite(wall_seconds) or wall_seconds <= 0):
        raise ValueError("positive measured wall interval and device observations required")
    result = []
    for row in rows:
        samples = row["samples_ms"]
        if len(samples) != repeats or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError("complete finite positive CUDA event samples required")
        values = sorted(samples)
        def percentile(p):
            at = p * (len(values) - 1)
            lower = int(at)
            return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (at - lower)
        total_s = sum(samples) / 1000
        result.append({**row, "p50_ms": percentile(.5), "p95_ms": percentile(.95),
                       "h2d_gb_s": sample_bytes * repeats / total_s / 1e9})
    uuids = [x.get("gpu_uuid") for x in result]
    if any(not isinstance(x, str) or not x for x in uuids) or len(set(uuids)) != len(uuids):
        raise ValueError("distinct physical GPU UUID observations required")
    return {"devices": result, "concurrent_wall_s": wall_seconds,
            "aggregate_h2d_gb_s": len(rows) * sample_bytes * repeats / wall_seconds / 1e9,
            "bandwidth_unit": "decimal GB/s", "concurrent": True}


def verify_report(report):
    """Recompute raw timing/allocation consistency; not remote attestation."""
    if report.get("schema") != "shard-host-io-probe/1":
        raise ValueError("unsupported host I/O probe schema")
    rows = report["devices"]
    indices = [row["local_index"] for row in rows]
    required = validate_probe(indices, report["sample_bytes"], report["repeats"],
                              report["pin_budget_requested_bytes"], report["host_capacity"]["available_ram_bytes"])
    if type(report.get("observed_pinned_allocation_bytes")) is not int or report.get("observed_pinned_allocation_bytes") != required or report.get("pin_budget_verified") is not (
            report["pin_budget_requested_bytes"] > 0):
        raise ValueError("observed pinned allocation is inconsistent with the requested experiment")
    if report.get("concurrent") is not True or report.get("reservation") is not False:
        raise ValueError("concurrent probe and non-reservation scope required")
    recomputed = summarize_transfers(rows, report["concurrent_wall_s"], report["sample_bytes"], report["repeats"])
    if report.get("bandwidth_unit") != "decimal GB/s" or not math.isclose(
            report["aggregate_h2d_gb_s"], recomputed["aggregate_h2d_gb_s"], rel_tol=1e-9):
        raise ValueError("aggregate bandwidth differs from raw concurrent timing")
    if report["concurrent_wall_s"] < max(sum(row["samples_ms"]) / 1000 for row in rows):
        raise ValueError("CUDA transfer intervals exceed the declared whole-host wall interval")
    for observed, expected in zip(rows, recomputed["devices"]):
        for field in ("p50_ms", "p95_ms", "h2d_gb_s"):
            if not math.isclose(observed[field], expected[field], rel_tol=1e-9):
                raise ValueError("per-device bandwidth/percentile differs from raw samples")
    return recomputed


def measure(devices, *, host_id, directory=".", sample_bytes=64 * 1024**2,
            repeats=20, pin_budget_bytes=0):
    if not isinstance(host_id, str) or not host_id.strip():
        raise ValueError("an explicit host_id is required; an IP is not a host identity")
    cap = measure_host_resources(directory)
    required = validate_probe(devices, sample_bytes, repeats, pin_budget_bytes, cap["available_ram_bytes"])
    import torch
    if not torch.cuda.is_available() or any(i >= torch.cuda.device_count() for i in devices):
        raise RuntimeError("requested CUDA devices are unavailable; CPU cannot qualify this probe")
    # Allocate before the measured interval. No page locking or allocation in the hot loop.
    host = torch.empty(required, dtype=torch.uint8, pin_memory=True)
    host[:sample_bytes * len(devices)].zero_()
    targets, streams, metadata = [], [], []
    for index in devices:
        with torch.cuda.device(index):
            free, total = torch.cuda.mem_get_info(index)
            if sample_bytes > free * .1:
                raise RuntimeError(f"device {index}: transfer buffer exceeds 10% of free VRAM")
            prop = torch.cuda.get_device_properties(index)
            uuid = str(getattr(prop, "uuid", ""))
            if not uuid:
                raise RuntimeError("runtime does not expose physical GPU UUIDs")
            targets.append(torch.empty(sample_bytes, dtype=torch.uint8, device=f"cuda:{index}"))
            streams.append(torch.cuda.Stream(device=index))
            metadata.append({"local_index": index, "gpu_uuid": uuid, "gpu_name": prop.name,
                             "available_vram_bytes": free, "total_vram_bytes": total})
    barrier = threading.Barrier(len(devices) + 1)
    def worker(j):
        index, stream, target = devices[j], streams[j], targets[j]
        source = host[j * sample_bytes:(j + 1) * sample_bytes]
        with torch.cuda.device(index), torch.cuda.stream(stream):
            target.copy_(source, non_blocking=True)
            stream.synchronize()
            pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                     for _ in range(repeats)]
            barrier.wait(timeout=30)
            for start, end in pairs:
                start.record(stream)
                target.copy_(source, non_blocking=True)
                end.record(stream)
            stream.synchronize()
            return {**metadata[j], "samples_ms": [a.elapsed_time(b) for a, b in pairs]}
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(worker, j) for j in range(len(devices))]
        # Include shared-host scheduling/contended transfers in aggregate wall bandwidth.
        started = time.perf_counter()
        barrier.wait(timeout=30)
        rows = [future.result() for future in futures]
        elapsed = time.perf_counter() - started
    return {"schema": "shard-host-io-probe/1", "host_id": host_id,
            "identity_scope": "operator-declared host; locally observed GPU UUIDs",
            "measured_at": datetime.now(timezone.utc).isoformat(), "host_capacity": cap,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "sample_bytes": sample_bytes, "repeats": repeats,
            "observed_pinned_allocation_bytes": required,
            "pin_budget_requested_bytes": pin_budget_bytes,
            "pin_budget_verified": bool(pin_budget_bytes),
            "reservation": False, **summarize_transfers(rows, elapsed, sample_bytes, repeats)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True, help="comma-separated local CUDA indices")
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--dir", default=".")
    parser.add_argument("--sample-mib", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--pin-budget-mib", type=int, default=0,
                        help="actually allocate this host-wide locked-memory budget; never inferred from RAM")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        report = measure([int(x) for x in args.devices.split(",")], host_id=args.host_id,
                         directory=args.dir, sample_bytes=args.sample_mib * 1024**2,
                         repeats=args.repeats, pin_budget_bytes=args.pin_budget_mib * 1024**2)
        from pathlib import Path
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"ok": True, "report": str(output.resolve())}))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

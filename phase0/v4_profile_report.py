"""Explain signed stage timing from a frozen benchmark; never infer CPU% by subtraction."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from phase0 import v4_benchmark as B
from shard.receipt import verify_coverage, wire_receipt
from shard.runtime_profile import validate_profile


def explain(report, protocol):
    B.require_protocol(protocol)
    B.require_protocol(report.get("protocol", {}))
    if report["protocol"]["sha256"] != protocol["sha256"]:
        raise ValueError("independent frozen protocol differs from benchmark report")
    if report.get("backend") != "live_ring":
        raise ValueError("CPU/mock benchmark cannot provide live stage timing evidence")
    assignments = {node["signer_pubkey"]: (node["layer_start"], node["layer_end"]) for node in protocol["hardware"]}
    nodes = {node["signer_pubkey"]: node for node in protocol["hardware"]}
    rows, missing = [], []
    for sample in report.get("samples", []):
        if sample.get("phase") != "warm":
            continue
        if not isinstance(sample.get("nonce"), str) or not sample["nonce"]:
            raise ValueError("fresh measured-job nonce required for profile receipts")
        receipts = [wire_receipt(value) for value in sample.get("receipts", [])]
        verify_coverage(receipts, 43, expected_by_signer=assignments,
                        expected_nonce=sample.get("nonce"), check_chain=True)
        for receipt in receipts:
            if receipt.get("job_id") != sample.get("job_id") or receipt.get("swarm_id") != sample.get("swarm_id"):
                raise ValueError("profile receipt belongs to another measured job")
            observation = receipt.get("runtime_observation")
            if observation is None or B.runtime_observation_errors(observation, nodes[receipt["pubkey"]], protocol):
                raise ValueError("profile runtime/source identity differs from frozen node")
            profile = receipt.get("runtime_metrics", {}).get("performance")
            if profile is None:
                missing.append({"workload": sample.get("workload"), "rep": sample.get("rep"),
                                "node_id": observation["node_id"]})
                continue
            validate_profile(profile)
            rows.append({"node_id": observation["node_id"], "layers": [receipt["layer_start"], receipt["layer_end"]],
                         "job_id": receipt["job_id"], "workload": sample.get("workload"), "rep": sample.get("rep"),
                         "gpu_sample_every": profile["gpu_sample_every"], "dropped_gpu_samples": profile["dropped_gpu_samples"],
                         "phases": profile["phases"], "receipt_sha256": B.digest(receipt),
                         **({"timeline": profile["timeline"]} if "timeline" in profile else {}),
                         **({"role_cost_frames": profile["role_cost_frames"],
                             "runtime_config_sha256": observation["runtime_config_sha256"],
                             "engine_source_inventory": {Path(name).name: sha for name, sha in observation["source_files"].items()
                                 if name.startswith("engines/deepseek_v4/v4_") and name.endswith(".py")}}
                            if "role_cost_frames" in profile else {})})
    return {"schema": "flyai-v4-profile-diagnosis/1", "protocol_sha256": protocol["sha256"],
            "report_sha256": B.digest(report), "benchmark_verdict": B.evaluate_report(report, protocol),
            "status": "diagnostic_only" if rows else "unverified", "stage_profiles": rows, "missing_profiles": missing,
            "interpretation": {"host": "caller-thread dispatch/wait intervals; not isolated CPU execution",
                "gpu": "sampled CUDA stream intervals; includes stream ordering and may overlap nested phases",
                "percentiles": "bounded per-receipt rolling samples, not global stage or end-to-end p95",
                "totals": "host and GPU totals use different sampling counts and must not be subtracted or summed",
                "cpu_fraction": None, "speed_acceptance": "independent benchmark verdict only; this report does not qualify a release"}}


def role_surfaces(diagnosis):
    """Export authenticated measured partitions; publication still needs calibration.

    Group by source/runtime/node/role; never join frames from different runtimes.
    This artifact is diagnostic input, not an executable placement lease.
    """
    from shard.stage_cost import surface_from_frames
    groups = {}
    for row in diagnosis["stage_profiles"]:
        inventory = row.get("engine_source_inventory")
        if "role_cost_frames" not in row or not inventory:
            continue
        for frame in row["role_cost_frames"]["frames"]:
            if [frame["layer_start"], frame["layer_end"]] != row["layers"]:
                raise ValueError("role frame layer geometry differs from its signed receipt")
            key = (row["node_id"], row["runtime_config_sha256"], B.digest(inventory),
                   frame["decode_mode"], B.digest(frame["role"]), tuple(row["layers"]))
            group = groups.setdefault(key, {"frames": [], "receipts": set()})
            if len(group["frames"]) >= 128:
                raise ValueError("role calibration group exceeds 128 raw intervals; select a bounded campaign")
            group["frames"].append(frame); group["receipts"].add(row["receipt_sha256"])
    exported, missing = [], []
    for key, group in groups.items():
        node, runtime, source, mode, _, layers = key
        try:
            surface = surface_from_frames(group["frames"], source_sha256=source)
        except ValueError as error:
            if len(group["frames"]) < 5 or str(error) == "bounded stage cost samples required":
                missing.append({"node_id": node, "reason": "fewer than five measured intervals for one exact shape"})
                continue
            raise
        exported.append({"node_id": node, "runtime_config_sha256": runtime, "layers": list(layers),
            "source_sha256": source, "measured_at": min(x["measured_at"] for x in group["frames"]),
            "cost_surface": surface, "receipt_sha256": sorted(group["receipts"]),
            "qualification": "diagnostic_input_requires_matching_runtime_resource_calibration"})
    return {"schema": "flyai-role-cost-export/1", "protocol_sha256": diagnosis["protocol_sha256"],
            "report_sha256": diagnosis["report_sha256"], "surfaces": exported, "missing": missing,
            "release_qualified": False}


def chrome_trace(diagnosis):
    """Perfetto/Chrome events; every receipt/clock/stream is an independent lane.

    CUDA epochs measure device events. Aligning them to host submission times
    would invent an enqueue/execute correlation which CUDA events do not give.
    """
    events = []
    for row in diagnosis["stage_profiles"]:
        timeline = row.get("timeline")
        if timeline is None:
            continue
        from shard.runtime_profile import validate_timeline
        validate_timeline(timeline)
        for span in timeline["spans"]:
            # Do not combine even two jobs from the same stage: reset restarts
            # the local origin. The receipt hash binds the exact epoch.
            pid = f"{row['node_id']}:{row['receipt_sha256']}:{span['clock']}"
            events.append({"name": span["name"], "cat": span["clock"], "ph": "X", "pid": pid,
                "tid": span["track"], "ts": span["start_ms"] * 1000, "dur": span["duration_ms"] * 1000,
                "args": {"job_id": row["job_id"], "receipt_sha256": row["receipt_sha256"]}})
    return {"traceEvents": events, "displayTimeUnit": "ms", "metadata": {
        "protocol_sha256": diagnosis["protocol_sha256"], "report_sha256": diagnosis["report_sha256"],
        "scope": "Independent local clock domains; not a synchronized distributed critical path or CPU utilization"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trace-out", help="Optional Perfetto/Chrome JSON; clock domains stay separate")
    parser.add_argument("--role-cost-out", help="Optional role-aware measured cost surfaces; not placement calibration")
    args = parser.parse_args(argv)
    result = explain(json.loads(Path(args.report).read_text(encoding="utf-8")),
                     json.loads(Path(args.protocol).read_text(encoding="utf-8")))
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if args.trace_out:
        Path(args.trace_out).write_text(json.dumps(chrome_trace(result), allow_nan=False), encoding="utf-8")
    if args.role_cost_out:
        Path(args.role_cost_out).write_text(json.dumps(role_surfaces(result), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "profiles": len(result["stage_profiles"]),
                      "missing_profiles": len(result["missing_profiles"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

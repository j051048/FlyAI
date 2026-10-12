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
                         "phases": profile["phases"], "receipt_sha256": B.digest(receipt)})
    return {"schema": "flyai-v4-profile-diagnosis/1", "protocol_sha256": protocol["sha256"],
            "report_sha256": B.digest(report), "benchmark_verdict": B.evaluate_report(report, protocol),
            "status": "diagnostic_only" if rows else "unverified", "stage_profiles": rows, "missing_profiles": missing,
            "interpretation": {"host": "caller-thread dispatch/wait intervals; not isolated CPU execution",
                "gpu": "sampled CUDA stream intervals; includes stream ordering and may overlap nested phases",
                "percentiles": "bounded per-receipt rolling samples, not global stage or end-to-end p95",
                "totals": "host and GPU totals use different sampling counts and must not be subtracted or summed",
                "cpu_fraction": None, "speed_acceptance": "independent benchmark verdict only; this report does not qualify a release"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = explain(json.loads(Path(args.report).read_text(encoding="utf-8")),
                     json.loads(Path(args.protocol).read_text(encoding="utf-8")))
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "profiles": len(result["stage_profiles"]),
                      "missing_profiles": len(result["missing_profiles"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

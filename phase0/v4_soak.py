"""Sustained V4-ring verification, alongside the frozen four/six-card speed suite.

Runs only an existing ring. Every candidate job gets a same-prompt greedy control,
fresh nonce, raw signed receipts and committed-token timing. Partial campaigns are
saved and cannot pass the declared duration/cycle/SLO contract.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import secrets
import statistics
import time

try:
    import v4_benchmark as bench
except ImportError:
    from phase0 import v4_benchmark as bench
from shard.receipt import verify_coverage, wire_receipt
from shard.runtime_metrics import validate_runtime_metrics


def _finite_positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def run_soak(adapter, protocol, *, cycles=25, duration_s=3600, ttft_p95_s=30,
             token_gap_p95_s=.25, max_idle_gap_s=10, clock=time.perf_counter, on_update=None):
    bench.require_protocol(protocol)
    if type(cycles) is not int or cycles < 1:
        raise ValueError("cycles must be a positive integer")
    for name, value in (("duration_s", duration_s), ("ttft_p95_s", ttft_p95_s),
                        ("token_gap_p95_s", token_gap_p95_s), ("max_idle_gap_s", max_idle_gap_s)):
        _finite_positive(value, name)
    report = {"schema": "flyai-v4-soak/1", "protocol": protocol,
              "protocol_sha256": protocol["sha256"], "run_id": secrets.token_hex(16),
              "backend": adapter.backend, "artifact_verification": adapter.artifact_verification,
              "contract": {"cycles": cycles, "duration_s": duration_s,
                           "ttft_p95_s": ttft_p95_s, "token_gap_p95_s": token_gap_p95_s,
                           "max_idle_gap_s": max_idle_gap_s},
              "completed_cycles": 0, "elapsed_s": 0, "active_observation_s": 0,
              "samples": [], "complete": False,
              "timing_scope": "trusted coordinator monotonic observations; receipts do not attest elapsed wall time"}
    started = clock()
    try:
        while report["completed_cycles"] < cycles or report["active_observation_s"] < duration_s:
            cycle = report["completed_cycles"]
            for prompt in protocol["prompts"]:
                for phase, mode in (("soak", protocol["run"]["mode"]), ("greedy_control", "greedy")):
                    job_start = clock() - started
                    sample = bench.measure_job(adapter, prompt, protocol, run_id=report["run_id"],
                                               phase=phase, rep=cycle, mode=mode, clock=clock)
                    sample["campaign_interval_s"] = [job_start, clock() - started]
                    report["active_observation_s"] += (sample["measurement"]["elapsed_s"] +
                                                       sample["measurement"]["receipt_sweep_s"])
                    report["samples"].append(sample)
                    report["elapsed_s"] = clock() - started
                    if on_update:
                        on_update(report)
            report["completed_cycles"] += 1
        report["elapsed_s"] = clock() - started
        report["complete"] = True
    except Exception as exc:
        report["elapsed_s"] = clock() - started
        report["failure"] = f"{type(exc).__name__}: {exc}"
    if on_update:
        on_update(report)
    return report


def _p95(values):
    values = sorted(values)
    at = .95 * (len(values) - 1)
    lower = int(at)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (at - lower)


def evaluate_soak(report, expected_protocol=None, expected_contract=None):
    errors, missing = [], []
    try:
        if report.get("schema") != "flyai-v4-soak/1":
            raise ValueError("unsupported soak schema")
        protocol = report["protocol"]
        errors.extend(bench.protocol_errors(protocol))
        if report["protocol_sha256"] != protocol["sha256"]:
            errors.append("protocol digest mismatch")
        if expected_protocol is not None and expected_protocol != protocol:
            errors.append("campaign differs from independently frozen protocol")
        contract = report["contract"]
        if expected_contract is not None and contract != expected_contract:
            errors.append("campaign differs from independently declared duration/SLO contract")
        if type(contract["cycles"]) is not int or contract["cycles"] < 1:
            raise ValueError("invalid declared cycles")
        for name in ("duration_s", "ttft_p95_s", "token_gap_p95_s", "max_idle_gap_s"):
            _finite_positive(contract[name], name)
        if (type(report["completed_cycles"]) is not int or report["completed_cycles"] < contract["cycles"] or
                report.get("complete") is not True or report.get("failure")):
            errors.append("campaign interrupted or declared cycles incomplete")
        if not math.isfinite(report["elapsed_s"]) or report["elapsed_s"] < contract["duration_s"]:
            errors.append("declared continuous duration not attained")
        if report.get("backend") != "live_ring":
            missing.append("CPU/mock campaign cannot qualify hardware acceptance")
        if not all(report.get("artifact_verification", {}).get(k) is True for k in
                   ("checkpoint_bytes_verified", "config_bytes_verified", "tokenizer_bytes_verified", "engine_source_verified")):
            missing.append("checkpoint/config/tokenizer/source verification absent")
        assignments = {node["signer_pubkey"]: (node["layer_start"], node["layer_end"])
                       for node in protocol["hardware"]}
        nonces, jobs, slots = set(), set(), {}
        ttfts, gaps, rates = [], [], []
        observation_s, last_end = 0.0, 0.0
        samples = report["samples"]
        if len(samples) != report["completed_cycles"] * len(protocol["prompts"]) * 2:
            errors.append("missing or extra campaign jobs")
        for sample in samples:
            prompt = next(p for p in protocol["prompts"] if p["workload"] == sample["workload"])
            errors.extend(bench._sample_errors(sample, protocol, prompt))
            left, right = sample["campaign_interval_s"]
            if (type(left) not in (int, float) or type(right) not in (int, float) or
                    not math.isfinite(left) or not math.isfinite(right) or left < last_end or right < left):
                raise ValueError("invalid or overlapping campaign intervals")
            observed = sample["measurement"]["elapsed_s"] + sample["measurement"]["receipt_sweep_s"]
            if right - left < observed - 1e-6 or right - left - observed > contract["max_idle_gap_s"]:
                errors.append("campaign interval differs from timed generation/sweep")
            if left - last_end > contract["max_idle_gap_s"]:
                errors.append("continuous campaign has an excessive idle gap")
            last_end = right
            observation_s += observed
            slot = sample["rep"], sample["workload"], sample["phase"]
            if (type(sample["rep"]) is not int or not 0 <= sample["rep"] < report["completed_cycles"] or
                    sample["phase"] not in ("soak", "greedy_control") or slot in slots):
                errors.append("duplicate or invalid campaign slot")
            slots[slot] = sample
            wanted_mode = "greedy" if sample["phase"] == "greedy_control" else protocol["run"]["mode"]
            if sample["mode"] != wanted_mode:
                errors.append("job mode differs from frozen protocol")
            nonce, job_id = sample["nonce"], sample["job_id"]
            if not isinstance(nonce, str) or len(nonce) < 32 or nonce in nonces or job_id in jobs:
                errors.append("missing or reused nonce/job identity")
            nonces.add(nonce); jobs.add(job_id)
            raw = sample["receipts"]
            if not raw:
                missing.append("raw stage receipts absent")
            else:
                receipts = [wire_receipt(r) for r in raw]
                verify_coverage(receipts, protocol["layer_count"], expected_by_signer=assignments,
                                expected_nonce=nonce, check_chain=True)
                if any(r["job_id"] != job_id or r["swarm_id"] != sample["swarm_id"] for r in receipts):
                    errors.append("receipt job/swarm binding differs")
                if protocol["env"].get("V4_RUNTIME_METRICS") == "1":
                    for r in receipts:
                        metrics = validate_runtime_metrics(r.get("runtime_metrics"))
                        if metrics.get("mode") != ("gpu_expert_cache" if
                                protocol["env"]["V4_EXPERT_PLACEMENT"] == "ram" else "gpu_resident"):
                            missing.append("GPU runtime observation absent")
            if sample["phase"] == "soak":
                measurement = sample["measurement"]
                ttfts.append(measurement["first_token_s"])
                offsets = measurement["commit_offsets_s"]
                gaps.extend(b - a for a, b in zip(offsets, offsets[1:]))
                rates.append(measurement["decode_committed_tok_s"])
        for cycle in range(report["completed_cycles"]):
            for prompt in protocol["prompts"]:
                left = slots.get((cycle, prompt["workload"], "soak"))
                right = slots.get((cycle, prompt["workload"], "greedy_control"))
                if not left or not right or left["tokens"] != right["tokens"]:
                    errors.append("candidate output differs from same-prompt greedy control or pair missing")
        if (not math.isclose(observation_s, report["active_observation_s"], rel_tol=1e-9) or
                observation_s < contract["duration_s"]):
            errors.append("declared active duration is not supported by complete job/sweep observations")
        if report["elapsed_s"] < last_end or report["elapsed_s"] - last_end > contract["max_idle_gap_s"]:
            errors.append("campaign elapsed scalar differs from the complete monotonic timeline")
        if not ttfts or not gaps:
            missing.append("committed-token latency samples absent")
        summary = {"ttft_p95_s": _p95(ttfts) if ttfts else None,
                   "token_gap_p95_s": _p95(gaps) if gaps else None,
                   "median_decode_tok_s": statistics.median(rates) if rates else None}
        if ttfts and summary["ttft_p95_s"] > contract["ttft_p95_s"]:
            errors.append("declared TTFT P95 exceeded")
        if gaps and summary["token_gap_p95_s"] > contract["token_gap_p95_s"]:
            errors.append("declared token-gap P95 exceeded")
        return {"status": "failed" if errors else "unverified" if missing else "passed",
                "errors": errors, "missing_evidence": missing, "summary": summary,
                "jobs": len(samples), "cold_start_scope": "warm sustained service; use v4_benchmark separately for cold starts"}
    except Exception as exc:
        return {"status": "failed", "errors": errors + [f"invalid evidence: {type(exc).__name__}: {exc}"],
                "missing_evidence": missing}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    live = sub.add_parser("run")
    live.add_argument("--protocol", required=True)
    live.add_argument("--dir", required=True)
    live.add_argument("--head", required=True)
    live.add_argument("--tail", required=True)
    live.add_argument("--cycles", type=int, default=25)
    live.add_argument("--duration-s", type=float, default=3600)
    live.add_argument("--ttft-p95-s", type=float, default=30)
    live.add_argument("--token-gap-p95-s", type=float, default=.25)
    live.add_argument("--max-idle-gap-s", type=float, default=10)
    live.add_argument("--timeout", type=float, default=600)
    live.add_argument("--connect-retry", type=float, default=30)
    live.add_argument("--deployment-plan")
    live.add_argument("--coordinator-key")
    live.add_argument("--out", required=True)
    check = sub.add_parser("verify")
    check.add_argument("report")
    check.add_argument("--protocol", required=True)
    check.add_argument("--contract", required=True, help="independent JSON duration/cycles/SLO contract")
    args = parser.parse_args(argv)
    if args.action == "verify":
        result = evaluate_soak(bench._read(args.report), bench._read(args.protocol), bench._read(args.contract))
    else:
        protocol = bench._read(args.protocol)
        adapter = bench.LiveRingAdapter(protocol, Path(args.dir), args.head, args.tail,
                                        timeout=args.timeout, retry_s=args.connect_retry,
                                        **({"deployment_plan": args.deployment_plan, "coordinator_key": args.coordinator_key}
                                           if args.deployment_plan else {}))
        try:
            report = run_soak(adapter, protocol, cycles=args.cycles, duration_s=args.duration_s,
                              ttft_p95_s=args.ttft_p95_s, token_gap_p95_s=args.token_gap_p95_s,
                              max_idle_gap_s=args.max_idle_gap_s,
                              on_update=lambda report: bench._write(args.out, report))
            result = evaluate_soak(report, protocol, report["contract"])
        finally:
            adapter.close()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

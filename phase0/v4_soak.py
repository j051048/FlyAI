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
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import v4_benchmark as bench
except ImportError:
    from phase0 import v4_benchmark as bench
from shard.receipt import verify_coverage, wire_receipt
from shard.runtime_metrics import validate_runtime_metrics


def _finite_positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _artifact_verification_findings(report, protocol):
    """Keep the frozen benchmark's native catalogue/local payload scopes intact.

    A coordinator's authenticated global cohort and local model assets do not
    claim to contain every stage's weight bytes. Signed stage/job/runtime
    verification remains the responsibility of each campaign evaluator.
    """
    errors, missing = [], []
    evidence = report.get("artifact_verification", {})
    logical = protocol.get("checkpoint", {}).get("identity_kind") == "logical-tensor-catalogue"
    required = (("global_catalog_verified", "authenticated_cohort_verified") if logical else
                ("checkpoint_bytes_verified",)) + ("config_bytes_verified", "tokenizer_bytes_verified", "engine_source_verified")
    if not all(evidence.get(name) is True for name in required):
        missing.append("frozen checkpoint/config/tokenizer/source verification absent")
    if logical:
        scope = evidence.get("local_weight_scope")
        if (evidence.get("checkpoint_id") != protocol["checkpoint"]["checkpoint_id"] or scope not in (
                "complete packed payload hashes", "assigned stage payload hashes", "pinned catalogue and local model assets only")):
            missing.append("local/global artifact verification scope absent or invalid")
        if scope != "complete packed payload hashes" and evidence.get("checkpoint_bytes_verified") is True:
            errors.append("partial/coordinator assets falsely claim complete local checkpoint verification")
    return errors, missing


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


LIFECYCLE_MODES = ("greedy", "spec", "dspark", "pipelined")
LIFECYCLE_SCHEMA = "flyai-v4-lifecycle/1"


class _IntentionalCancel(Exception):
    """An injected coordinator cancellation, never a successful decode."""


def lifecycle_contract(cycles=2, *, ngram_k=4, ngram_order=3, cancel_after=2):
    if (type(cycles) is not int or not 1 <= cycles <= 100
            or type(ngram_k) is not int or not 1 <= ngram_k <= 64
            or type(ngram_order) is not int or not 1 <= ngram_order <= 16
            or type(cancel_after) is not int or not 1 <= cancel_after <= 64):
        raise ValueError("lifecycle cycles, draft parameters and cancellation point must be bounded integers")
    return {"cycles": cycles, "modes": list(LIFECYCLE_MODES), "ngram_k": ngram_k,
            "ngram_order": ngram_order, "cancel_after": cancel_after,
            "rollback_injection": "spec_first_proposal_differs_from_greedy_v1"}


def run_lifecycle(adapter, protocol, *, contract=None, on_update=None):
    """Fault matrix on an existing strict ring; deliberately separate from speed.

    Real cancellation interrupts coordinator execution and drops its channels.
    Recovery obtains a new signed owner grant and uses the ordinary reset barrier.
    The spec-only fault injects a known-wrong proposal to require native rollback.
    This cannot attest GPU execution or replace complete benchmark acceptance.
    """
    contract = lifecycle_contract() if contract is None else contract
    if contract != lifecycle_contract(contract["cycles"], ngram_k=contract["ngram_k"],
            ngram_order=contract["ngram_order"], cancel_after=contract["cancel_after"]):
        raise ValueError("lifecycle contract differs from the supported execution matrix")
    if not getattr(adapter, "session_options", {}).get("session_config"):
        raise ValueError("lifecycle matrix requires strict authenticated ring channels")
    maximum = protocol["run"]["max_new"]
    if maximum < max(4, contract["cancel_after"] + 1):
        raise ValueError("lifecycle output budget must leave work after cancellation and rollback")
    prompt = protocol["prompts"][0]
    report = {"schema": LIFECYCLE_SCHEMA, "protocol": protocol,
        "protocol_sha256": protocol["sha256"], "contract": contract,
        "backend": adapter.backend, "artifact_verification": adapter.artifact_verification,
        "deployment_plan": adapter.session_options["session_config"].plan,
        "run_id": secrets.token_hex(16), "cases": [], "complete": False,
        "scope": "coordinator lifecycle fault injection; no throughput or GPU qualification"}

    def record(case):
        report["cases"].append(case)
        if on_update: on_update(report)

    def job(cycle, mode, action, *, cancel=False, drafter=None):
        nonce = secrets.token_hex(32)
        identity = f"{report['run_id']}/{cycle}/{mode}/{action}"
        committed = []
        row = {"cycle": cycle, "mode": mode, "action": action, "nonce": nonce,
               "job_id": identity, "swarm_id": adapter.swarm_id, "tokens": committed,
               "owner_grant": dict(adapter.pipe.grant)}
        def commit(token):
            committed.append(int(token))
            if cancel and len(committed) >= contract["cancel_after"]:
                raise _IntentionalCancel("declared committed-token boundary")
        try:
            result = adapter.generate(prompt["token_ids"], maximum, mode=mode, nonce=nonce,
                job_id=identity, on_token=commit, drafter=drafter,
                **({"ngram_k": contract["ngram_k"], "ngram_order": contract["ngram_order"]}
                   if mode == "spec" else {}))
        except _IntentionalCancel:
            if not cancel: raise
            row.update(cancelled=True, receipt_scope="cancelled partial job is not settled")
            adapter.close()
        else:
            if cancel: raise RuntimeError("declared cancellation was not reached")
            if committed != result.get("tokens") or result.get("ok") is not True:
                raise RuntimeError("coordinator returned tokens different from committed callbacks")
            receipts, okay = adapter.sweep(nonce)
            row.update(cancelled=False, receipts=[wire_receipt(x) for x in receipts], receipts_ok=okay,
                statistics={k:result[k] for k in ("mode", "rounds", "accepted", "accept_hist", "coordinator_counters") if k in result})
        record(row)
        return list(committed)

    try:
        control = job(-1, "greedy", "control")
        for cycle in range(contract["cycles"]):
            for mode in contract["modes"]:
                job(cycle, mode, "reset_first")
                job(cycle, mode, "reset_repeat")
                previous = dict(adapter.pipe.grant)
                job(cycle, mode, "cancel", cancel=True)
                current = adapter.reconnect()
                record({"cycle": cycle, "mode": mode, "action": "reconnect",
                    "previous_owner": previous,
                    "new_owner": current})
                job(cycle, mode, "recovery")
            def reject(ids, width):
                offset = len(ids) - len(prompt["token_ids"])
                expected = control[offset] if offset < len(control) else 0
                return [0 if expected != 0 else 1] * width
            job(cycle, "spec", "forced_rollback", drafter=reject)
            job(cycle, "greedy", "post_rollback_control")
        report["complete"] = True
    except Exception as error:
        report["failure"] = f"{type(error).__name__}: {error}"
    if on_update: on_update(report)
    return report


def evaluate_lifecycle(report, expected_protocol=None, expected_contract=None):
    errors, missing = [], []
    try:
        if report.get("schema") != LIFECYCLE_SCHEMA: raise ValueError("unsupported lifecycle schema")
        protocol, contract = report["protocol"], report["contract"]
        if (contract != lifecycle_contract(contract["cycles"], ngram_k=contract["ngram_k"],
                ngram_order=contract["ngram_order"], cancel_after=contract["cancel_after"])
                or (expected_contract is not None and contract != expected_contract)):
            raise ValueError("lifecycle matrix differs from independently declared contract")
        if (report["protocol_sha256"] != protocol["sha256"] or
                protocol["sha256"] != bench.digest({k:v for k,v in protocol.items() if k != "sha256"})
                or expected_protocol is not None and expected_protocol != protocol):
            raise ValueError("lifecycle protocol binding differs")
        if not report.get("complete") or report.get("failure"): errors.append("lifecycle execution incomplete")
        expected = [(-1, "greedy", "control")]
        for cycle in range(contract["cycles"]):
            expected += [(cycle, mode, action) for mode in LIFECYCLE_MODES for action in
                         ("reset_first", "reset_repeat", "cancel", "reconnect", "recovery")]
            expected += [(cycle, "spec", "forced_rollback"), (cycle, "greedy", "post_rollback_control")]
        cases = report["cases"]
        if [(x["cycle"], x["mode"], x["action"]) for x in cases] != expected:
            errors.append("missing, reordered or repeated lifecycle cases")
        assignments = {n["signer_pubkey"]:(n["layer_start"], n["layer_end"]) for n in protocol["hardware"]}
        by_signer = {n["signer_pubkey"]:n for n in protocol["hardware"]}
        from shard.pipeline_session import SessionConfig
        session = SessionConfig.from_plan(report["deployment_plan"], -1)
        if ({n["signer_pubkey"]:(n["lo"], n["hi"]) for n in session.plan["stages"]} != assignments
                or session.plan["n_layers"] != protocol["layer_count"]):
            raise ValueError("strict session assignments differ from the frozen protocol")
        baseline = cases[0]["tokens"]
        if len(baseline) != protocol["run"]["max_new"]: errors.append("incomplete greedy control")
        nonces, jobs, identities = set(), set(), {}
        for row in cases:
            if row["action"] == "reconnect":
                before, after = row["previous_owner"], row["new_owner"]
                session.verify_grant(before, check_expiry=False)
                session.verify_grant(after, check_expiry=False)
                if (before["boot_id"] != after["boot_id"] or after["fence"] <= before["fence"]
                        or before["session_id"] == after["session_id"]):
                    errors.append("reconnect did not acquire a fresh owner fence on the same warm ring")
                continue
            session.verify_grant(row["owner_grant"], check_expiry=False)
            if row["nonce"] in nonces or row["job_id"] in jobs: errors.append("reused lifecycle job identity")
            nonces.add(row["nonce"]); jobs.add(row["job_id"])
            if row["action"] == "cancel":
                if row.get("cancelled") is not True or row["tokens"] != baseline[:contract["cancel_after"]]:
                    errors.append("cancellation did not occur at the declared correct prefix")
                if row.get("receipts"): errors.append("cancelled partial work cannot be settled as a complete job")
                continue
            if row.get("cancelled") is not False or row["tokens"] != baseline:
                errors.append("reset/recovery/rollback output differs from greedy control")
            if row.get("receipts_ok") is not True: errors.append("stage receipt sweep did not verify")
            raw = [wire_receipt(r) for r in row["receipts"]]
            verify_coverage(raw, protocol["layer_count"], expected_by_signer=assignments,
                            expected_nonce=row["nonce"], check_chain=True)
            for receipt in raw:
                if receipt["job_id"] != row["job_id"] or receipt["swarm_id"] != row["swarm_id"]:
                    errors.append("lifecycle receipt job/swarm differs")
                observation = receipt.get("runtime_observation")
                if protocol.get("evidence_contract", {}).get("runtime_observation_required"):
                    if observation is None: missing.append("signed runtime observation absent")
                    else:
                        errors.extend(bench.runtime_observation_errors(observation, by_signer[receipt["pubkey"]], protocol))
                        identity = bench._runtime_identity(observation)
                        if identities.setdefault(receipt["pubkey"], identity) != identity:
                            errors.append("stage runtime changed during the lifecycle matrix")
            if row["action"] == "forced_rollback":
                stats = row.get("statistics", {})
                hist = stats.get("accept_hist", {})
                if stats.get("rounds", 0) < 2 or sum(v for k,v in hist.items() if int(k) == 0) < 2:
                    errors.append("forced rejection did not exercise repeated speculative rewinds")
        if report.get("backend") != "live_ring" or bench.protocol_errors(protocol):
            missing.append("CPU/test lifecycle evidence cannot qualify production hardware")
        artifact_errors, artifact_missing = _artifact_verification_findings(report, protocol)
        errors.extend(artifact_errors)
        missing.extend(artifact_missing)
        return {"status": "failed" if errors else "unverified" if missing else "passed",
            "errors": errors, "missing_evidence": missing, "cases": len(cases),
            "scope": "functional lifecycle matrix only; no speed or GPU execution attestation"}
    except Exception as error:
        return {"status":"failed", "errors":errors + [f"invalid lifecycle evidence: {type(error).__name__}: {error}"],
                "missing_evidence":missing}


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
        artifact_errors, artifact_missing = _artifact_verification_findings(report, protocol)
        errors.extend(artifact_errors)
        missing.extend(artifact_missing)
        assignments = {node["signer_pubkey"]: (node["layer_start"], node["layer_end"])
                       for node in protocol["hardware"]}
        hardware_by_signer = {node["signer_pubkey"]: node for node in protocol["hardware"]}
        runtime_identities = {}
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
                for receipt in receipts:
                    observation = receipt.get("runtime_observation")
                    if observation is None:
                        if protocol.get("evidence_contract", {}).get("runtime_observation_required", False):
                            missing.append("signed effective runtime observation absent")
                        continue
                    errors.extend(bench.runtime_observation_errors(observation,
                        hardware_by_signer[receipt["pubkey"]], protocol))
                    identity = bench._runtime_identity(observation)
                    previous = runtime_identities.setdefault(observation["node_id"], identity)
                    if previous != identity:
                        errors.append("effective runtime identity changed during the sustained campaign")
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
    life = sub.add_parser("lifecycle", help="explicit reset/cancel/reconnect/forced-rollback matrix; no speed claim")
    life.add_argument("--protocol", required=True)
    life.add_argument("--dir", required=True)
    life.add_argument("--head", required=True)
    life.add_argument("--tail", required=True)
    life.add_argument("--cycles", type=int, default=2)
    life.add_argument("--ngram-k", type=int, default=4)
    life.add_argument("--ngram-order", type=int, default=3)
    life.add_argument("--cancel-after", type=int, default=2)
    life.add_argument("--timeout", type=float, default=600)
    life.add_argument("--connect-retry", type=float, default=30)
    life.add_argument("--deployment-plan", required=True)
    life.add_argument("--coordinator-key")
    life.add_argument("--out", required=True)
    check = sub.add_parser("verify")
    check.add_argument("report")
    check.add_argument("--protocol", required=True)
    check.add_argument("--contract", required=True, help="independent JSON duration/cycles/SLO contract")
    args = parser.parse_args(argv)
    if args.action == "verify":
        report = bench._read(args.report)
        evaluate = evaluate_lifecycle if report.get("schema") == LIFECYCLE_SCHEMA else evaluate_soak
        result = evaluate(report, bench._read(args.protocol), bench._read(args.contract))
    else:
        protocol = bench._read(args.protocol)
        adapter = bench.LiveRingAdapter(protocol, Path(args.dir), args.head, args.tail,
                                        timeout=args.timeout, retry_s=args.connect_retry,
                                        **({"deployment_plan": args.deployment_plan, "coordinator_key": args.coordinator_key}
                                           if args.deployment_plan else {}))
        try:
            if args.action == "lifecycle":
                contract = lifecycle_contract(args.cycles, ngram_k=args.ngram_k,
                    ngram_order=args.ngram_order, cancel_after=args.cancel_after)
                report = run_lifecycle(adapter, protocol, contract=contract,
                    on_update=lambda report: bench._write(args.out, report))
                result = evaluate_lifecycle(report, protocol, contract)
            else:
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

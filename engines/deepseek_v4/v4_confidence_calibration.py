"""Held-out raw-score calibration and frozen-workload prefix budgets.

The V4 confidence head emits logits, including negative values. These are never
multiplied or passed through an assumed sigmoid. Isotonic regressions estimate
conditional acceptance at each reached draft position. Statistics alone never
qualify hardware: promotion separately re-verifies full signed benchmark reports.
Coordinator timings and labels remain observations, not remote attestations.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from copy import deepcopy
import math
import time

from v4_adaptive_round import calibration_binding, digest

SCHEMA = "v4-confidence-calibration/1"
PROBE_SCHEMA = "v4-confidence-probe/1"
FIT_SCHEMA = "v4-confidence-isotonic/1"
MAX_LABELS = 8192


def _number(value, name, *, lower=None):
    if type(value) not in (int, float) or not math.isfinite(value) or (lower is not None and value < lower):
        raise ValueError(f"invalid finite {name}")
    return float(value)


def _integer(value, name, lower=0, upper=1_000_000_000):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"invalid {name}")
    return value


class ProbeRecorder:
    """Bounded, reached-position labels; canceled/unsubmitted futures have no label."""

    def __init__(self, enabled=False, limit=MAX_LABELS):
        self.enabled = bool(enabled)
        self.limit = _integer(limit, "probe label budget", 1, MAX_LABELS)
        self.labels = []
        self.refills = self.dropped = 0

    def refill(self):
        if self.enabled:
            self.refills += 1

    def observe(self, position, raw_conf, accepted):
        if not self.enabled or raw_conf is None:
            return
        position = _integer(position, "reached draft position", 1, 256)
        raw_conf = _number(raw_conf, "raw confidence")
        if type(accepted) is not bool:
            raise ValueError("acceptance label must be boolean")
        if len(self.labels) >= self.limit:
            self.dropped += 1
            return
        self.labels.append({"position": position, "raw_conf": raw_conf, "accepted": accepted})

    def report(self):
        return {"schema": PROBE_SCHEMA, "labels": deepcopy(self.labels),
                "refills": self.refills, "dropped_labels": self.dropped}


def _labels(probe, *, block_size):
    if (not isinstance(probe, dict) or set(probe) != {"schema", "labels", "refills", "dropped_labels"}
            or probe["schema"] != PROBE_SCHEMA):
        raise ValueError("actual bounded coordinator confidence probe required")
    _integer(probe["refills"], "observed refills", 1)
    if probe["dropped_labels"] != 0 or type(probe["dropped_labels"]) is not int:
        raise ValueError("truncated confidence probe cannot qualify calibration")
    rows = probe["labels"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_LABELS:
        raise ValueError("nonempty bounded reached-position labels required")
    result = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"position", "raw_conf", "accepted"}:
            raise ValueError("probe labels permit only reached position, raw score and actual acceptance")
        position = _integer(row["position"], "reached draft position", 1, block_size)
        score = _number(row["raw_conf"], "raw confidence")
        if type(row["accepted"]) is not bool:
            raise ValueError("acceptance label must be boolean")
        result.append((position, score, int(row["accepted"])))
    return result


def isotonic_fit(rows):
    """Pool adjacent violators, merging tied raw scores before fitting.

    No interpolation/extrapolation law is assumed. Prediction uses a right-closed
    empirical step function and refuses scores outside the training support.
    """
    if not rows:
        raise ValueError("isotonic fit needs observations")
    grouped = {}
    for score, accepted in rows:
        score = _number(score, "raw confidence")
        if type(accepted) not in (int, bool) or accepted not in (0, 1):
            raise ValueError("binary acceptance label required")
        total, count = grouped.get(score, (0, 0))
        grouped[score] = total+int(accepted), count+1
    blocks = []
    for score, (total, count) in sorted(grouped.items()):
        blocks.append([score, score, total, count])
        while len(blocks) >= 2 and blocks[-2][2]/blocks[-2][3] > blocks[-1][2]/blocks[-1][3]:
            right = blocks.pop(); left = blocks.pop()
            blocks.append([left[0], right[1], left[2]+right[2], left[3]+right[3]])
    return {"minimum": min(grouped), "maximum": max(grouped),
            "knots": [[right, total/count] for _, right, total, count in blocks],
            "observations": sum(count for _, count in grouped.values())}


def predict(model, raw_conf):
    score = _number(raw_conf, "raw confidence")
    if score < model["minimum"] or score > model["maximum"]:
        return None
    knots = model["knots"]
    return knots[bisect_left([row[0] for row in knots], score)][1]


def _diagnostics(model, rows):
    evaluated = [(predict(model, score), accepted) for score, accepted in rows]
    covered = [(prob, accepted) for prob, accepted in evaluated if prob is not None]
    buckets = []
    for index in range(10):
        values = [(prob, accepted) for prob, accepted in covered if min(9, int(prob*10)) == index]
        if values:
            buckets.append({"bin": index, "count": len(values),
                "mean_probability": sum(p for p, _ in values)/len(values),
                "acceptance_rate": sum(y for _, y in values)/len(values)})
    count = len(covered)
    return {"observations": len(rows), "covered": count,
        "brier": sum((p-y)**2 for p, y in covered)/count if count else None,
        "ece": sum(b["count"]*abs(b["mean_probability"]-b["acceptance_rate"]) for b in buckets)/count if count else None,
        "reliability": buckets}


def fit_statistics(requests, *, block_size, min_train=32, min_heldout=16):
    """Split whole request IDs before fitting; held-out labels never train PAV.

    Input rows are local statistics only. Their job IDs may later be linked to
    reverified reports, but this function alone grants no execution permission.
    """
    _integer(block_size, "loaded block", 1, 256)
    _integer(min_train, "minimum training count", 2)
    _integer(min_heldout, "minimum held-out count", 2)
    if not isinstance(requests, list) or not 4 <= len(requests) <= 4096:
        raise ValueError("at least four independent request probes required")
    groups = {}
    for request in requests:
        if not isinstance(request, dict) or set(request) != {"job_id", "probe"}:
            raise ValueError("one unique actual job ID per probe required")
        job = request["job_id"]
        if not isinstance(job, str) or not job or len(job) > 512 or job in groups:
            raise ValueError("probe job IDs must be distinct")
        groups[job] = _labels(request["probe"], block_size=block_size)
    # Deterministic whole-request 75/25 holdout; ranking is independent of labels.
    ordered = sorted(groups, key=lambda job: digest(job))
    heldout_ids = set(ordered[:max(1, len(ordered)//4)])
    positions = {}
    for position in range(1, block_size+1):
        train = [(score, y) for job, rows in groups.items() if job not in heldout_ids
                 for pos, score, y in rows if pos == position]
        test = [(score, y) for job, rows in groups.items() if job in heldout_ids
                for pos, score, y in rows if pos == position]
        if len(train) < min_train or len(test) < min_heldout:
            continue
        model = isotonic_fit(train)
        positions[str(position)] = {"model": model, "heldout": _diagnostics(model, test)}
    if "1" not in positions:
        raise ValueError("insufficient independent training/held-out position-one observations")
    return {"schema": FIT_SCHEMA, "method": "position_conditional_isotonic_pav",
        "split": "whole_request_sha256_rank_75_25", "training_job_ids": sorted(set(groups)-heldout_ids),
        "heldout_job_ids": sorted(heldout_ids), "minimum_train": min_train, "minimum_heldout": min_heldout,
        "positions": positions, "hardware_qualified": False}


def _time(now, ttl_s):
    now = _number(time.time() if now is None else now, "calibration creation", lower=0.)
    ttl_s = _number(ttl_s, "calibration lifetime", lower=1e-9)
    if ttl_s > 86400:
        raise ValueError("confidence calibration lifetime is at most one day")
    return now, now+ttl_s


def _qualify(evidence, options, limits):
    """Re-use the independent signed-report verifier; never promote row telemetry."""
    from v4_round_evidence import _profiles
    from phase0 import v4_benchmark as bench
    profiles = _profiles(evidence, options)
    if not isinstance(limits, dict) or set(limits) != {"max_p95_ttft_s", "max_p95_tpot_s", "max_heldout_ece"}:
        raise ValueError("explicit P95 TTFT/TPOT and held-out reliability limits required")
    for name in ("max_p95_ttft_s", "max_p95_tpot_s"):
        _number(limits[name], name, lower=1e-9)
    if not 0 <= _number(limits["max_heldout_ece"], "held-out ECE", lower=0.) <= 1:
        raise ValueError("held-out ECE limit must be a probability")
    baseline = min(options["loaded_block_size"], options["depth"]-1)
    probes, costs = [], []
    cohort = options["binding"].get("model_cohort", {})
    # Config bytes alone cannot distinguish two sets of weights. Existing local
    # statistical fixtures may omit an authenticated logical catalogue, but may
    # never acquire hardware enforcement authority by doing so.
    model_bound = (isinstance(cohort.get("checkpoint_id"), str)
                   and isinstance(cohort.get("manifest_sha256"), str))
    gpu_qualified = model_bound
    for entry in evidence:
        report, protocol = entry["report"], entry["protocol"]
        if model_bound and (cohort["checkpoint_id"], cohort["manifest_sha256"]) != (
                protocol["checkpoint"].get("checkpoint_id"), protocol["checkpoint"].get("manifest_sha256")):
            raise ValueError("confidence evidence model payload differs from the selected logical checkpoint")
        cap = protocol["round_experiment"]["draft_prefix_cap"]
        measured = bench.evaluate_report(report, protocol)
        for workload in bench.WORKLOADS:
            samples = [s for s in report["samples"] if s["phase"] == "warm" and s["workload"] == workload]
            refills = wall = 0
            dma = 0; dma_complete = True
            for sample in samples:
                probe = sample["coordinator_stats"].get("confidence_probe")
                # Cap zero has no reached draft labels but still measures refills.
                if (not isinstance(probe, dict) or probe.get("schema") != PROBE_SCHEMA
                        or type(probe.get("refills")) is not int or probe["refills"] <= 0
                        or probe.get("dropped_labels") != 0):
                    raise ValueError("qualification requires actual untruncated coordinator refill probes")
                counters = sample["coordinator_stats"].get("coordinator_counters", {})
                if (probe["refills"] > counters.get("frames_judged", -1)
                        or len(probe.get("labels", [])) > counters.get("frames_judged", -1)):
                    raise ValueError("confidence probe exceeds independently checked actual frame counts")
                if cap == baseline:
                    labels = _labels(probe, block_size=options["loaded_block_size"])
                    by_depth = sample["coordinator_stats"].get("accept_by_depth")
                    if not isinstance(by_depth, dict) or any(
                            list(by_depth.get(str(pos), by_depth.get(pos, []))) != [
                                sum(y for p, _, y in labels if p == pos),
                                sum(1 for p, _, _ in labels if p == pos)]
                            for pos in {p for p, _, _ in labels}):
                        raise ValueError("confidence labels differ from observed reached-position acceptance counts")
                    if sum(row[1] for row in by_depth.values()) != len(labels):
                        raise ValueError("baseline must observe every judged draft confidence without censor fabrication")
                    if (sum(y for _, _, y in labels) != counters.get("accepted_predictions")
                            or any(sum(1 for p, _, _ in labels if p == pos) > probe["refills"]
                                   for pos in {p for p, _, _ in labels})):
                        raise ValueError("confidence labels differ from actual accepted work/refill counts")
                    probes.append({"job_id": sample["job_id"], "probe": deepcopy(probe)})
                refills += probe["refills"]
                # Return/drain minus the first committed callback includes WAN,
                # GPU work, discarded futures and queue gaps. No additive guess.
                wall += sample["measurement"]["elapsed_s"]-sample["measurement"]["first_token_s"]
                for receipt in sample["receipts"]:
                    metrics = receipt.get("runtime_metrics")
                    if not isinstance(metrics, dict) or metrics.get("mode") == "reference_cpu":
                        gpu_qualified = dma_complete = False
                        continue
                    if metrics.get("mode") == "gpu_expert_cache" and "expert_prefetch" not in metrics:
                        dma_complete = False
                    dma += sum(metrics["work"][role][phase]["dma_bytes"]
                               for role in ("main", "draft") for phase in ("decode", "replay"))
                    dma += metrics.get("expert_prefetch", {}).get("dma_bytes", 0)
            prompt = next(p for p in protocol["prompts"] if p["workload"] == workload)
            summary = measured["workloads"][workload]
            costs.append({"workload": workload, "prompt_sha256": prompt["sha256"],
                "max_new": protocol["run"]["max_new"], "draft_prefix_cap": cap,
                "refills": refills, "wall_s": wall, "wall_per_refill_s": wall/refills,
                "expert_dma_bytes_per_refill": dma/refills if dma_complete else None,
                "p95_ttft_s": summary["timing_distributions"]["first_token_s"]["p95"],
                "p95_tpot_s": summary["inter_token_latency_s"]["p95"]})
    # Hardware mode is read from every verified stage receipt; a benchmark
    # inventory string or a coordinator label cannot qualify a CPU simulation.
    return probes, costs, gpu_qualified, profiles


def build_calibration(evidence, options, *, limits, now=None, ttl_s=900., min_train=32, min_heldout=16):
    created, expires = _time(now, ttl_s)
    probes, costs, gpu, _ = _qualify(evidence, options, limits)
    statistics = fit_statistics(probes, block_size=options["loaded_block_size"], min_train=min_train, min_heldout=min_heldout)
    return {"schema": SCHEMA, "binding_sha256": calibration_binding(options),
        "created_at": created, "expires_at": expires, "statistics": statistics,
        "costs": costs, "limits": deepcopy(limits), "gpu_qualified": gpu,
        "evidence": deepcopy(evidence), "evidence_sha256": digest(evidence)}


def verify_calibration(bundle, options):
    keys = {"schema", "binding_sha256", "created_at", "expires_at", "statistics", "costs", "limits",
            "gpu_qualified", "evidence", "evidence_sha256"}
    if not isinstance(bundle, dict) or set(bundle) != keys or bundle["schema"] != SCHEMA:
        raise ValueError("full raw confidence calibration bundle required")
    if bundle["binding_sha256"] != calibration_binding(options) or bundle["evidence_sha256"] != digest(bundle["evidence"]):
        raise ValueError("confidence evidence differs from the exact selected runtime binding")
    stats = bundle["statistics"]
    rebuilt = build_calibration(bundle["evidence"], options, limits=bundle["limits"], now=bundle["created_at"],
        ttl_s=bundle["expires_at"]-bundle["created_at"], min_train=stats["minimum_train"], min_heldout=stats["minimum_heldout"])
    if rebuilt != bundle:
        raise ValueError("confidence fit/costs differ from independently reverified raw evidence")
    return {key: deepcopy(value) for key, value in bundle.items() if key != "evidence"} | {
        "verification": "raw_benchmark_reverified"}


class ConfidenceBudget:
    """Frozen prefix objective; no per-request learning or guessed DMA costs."""

    def __init__(self, calibration, *, binding_sha256, baseline, mode="shadow", wall_clock=time.time, dma_limit=None,
                 min_gain=.08):
        if (not isinstance(calibration, dict) or calibration.get("schema") != SCHEMA
                or calibration.get("binding_sha256") != binding_sha256
                or calibration.get("verification") != "raw_benchmark_reverified"):
            raise ValueError("confidence selection requires independently reverified fixed-source calibration")
        required = {"schema", "binding_sha256", "created_at", "expires_at", "statistics", "costs", "limits",
                    "gpu_qualified", "evidence_sha256", "verification"}
        if set(calibration) != required or type(calibration["gpu_qualified"]) is not bool:
            raise ValueError("complete compact confidence calibration required")
        evidence_hash = calibration["evidence_sha256"]
        if not isinstance(evidence_hash, str) or len(evidence_hash) != 64 or any(c not in "0123456789abcdef" for c in evidence_hash):
            raise ValueError("invalid confidence evidence digest")
        stats = calibration["statistics"]
        if (not isinstance(stats, dict) or stats.get("schema") != FIT_SCHEMA
                or stats.get("method") != "position_conditional_isotonic_pav"
                or stats.get("hardware_qualified") is not False
                or not isinstance(stats.get("positions"), dict) or not stats["positions"]):
            raise ValueError("held-out conditional isotonic statistics required")
        train = stats.get("training_job_ids"); heldout = stats.get("heldout_job_ids")
        if (not isinstance(train, list) or not train or not isinstance(heldout, list) or not heldout
                or any(not isinstance(job, str) or not job for job in train+heldout)
                or len(set(train+heldout)) != len(train)+len(heldout)):
            raise ValueError("training and held-out requests must be independent")
        _integer(stats.get("minimum_train"), "minimum training count", 2)
        _integer(stats.get("minimum_heldout"), "minimum held-out count", 2)
        for position, row in stats["positions"].items():
            if not isinstance(position, str) or not position.isdigit() or not 1 <= int(position) <= baseline:
                raise ValueError("invalid calibrated draft position")
            if not isinstance(row, dict) or set(row) != {"model", "heldout"}:
                raise ValueError("complete per-position model and held-out diagnostics required")
            model, diagnostics = row["model"], row["heldout"]
            if not isinstance(model, dict) or set(model) != {"minimum", "maximum", "knots", "observations"}:
                raise ValueError("complete isotonic support and knots required")
            low = _number(model["minimum"], "confidence support minimum")
            high = _number(model["maximum"], "confidence support maximum")
            knots = model["knots"]
            if high < low or not isinstance(knots, list) or not 1 <= len(knots) <= 100000:
                raise ValueError("invalid isotonic support")
            previous, probability = None, -1.
            for knot in knots:
                if not isinstance(knot, list) or len(knot) != 2:
                    raise ValueError("invalid isotonic knot")
                score = _number(knot[0], "confidence knot")
                prob = _number(knot[1], "calibrated probability", lower=0.)
                if not low <= score <= high or (previous is not None and score <= previous) or not probability <= prob <= 1.:
                    raise ValueError("isotonic scores and acceptance must be monotone probabilities")
                previous, probability = score, prob
            if previous != high:
                raise ValueError("isotonic support must include its maximum")
            _integer(model["observations"], "training observations", stats["minimum_train"])
            if not isinstance(diagnostics, dict) or set(diagnostics) != {"observations", "covered", "brier", "ece", "reliability"}:
                raise ValueError("complete held-out diagnostics required")
            n = _integer(diagnostics["observations"], "held-out observations", stats["minimum_heldout"])
            _integer(diagnostics["covered"], "covered held-out observations", 0, n)
            for field in ("brier", "ece"):
                if diagnostics[field] is not None and not 0 <= _number(diagnostics[field], field) <= 1:
                    raise ValueError("invalid held-out error")
        limits = calibration["limits"]
        if not isinstance(limits, dict) or set(limits) != {"max_p95_ttft_s", "max_p95_tpot_s", "max_heldout_ece"}:
            raise ValueError("complete frozen P95/reliability limits required")
        _number(limits["max_p95_ttft_s"], "P95 TTFT limit", lower=1e-9)
        _number(limits["max_p95_tpot_s"], "P95 TPOT limit", lower=1e-9)
        if not 0 <= _number(limits["max_heldout_ece"], "held-out ECE limit") <= 1:
            raise ValueError("invalid held-out ECE limit")
        costs = calibration["costs"]
        if not isinstance(costs, list) or not 1 <= len(costs) <= 128:
            raise ValueError("bounded actual workload costs required")
        seen = set()
        for row in costs:
            if not isinstance(row, dict) or set(row) != {"workload", "prompt_sha256", "max_new", "draft_prefix_cap",
                    "refills", "wall_s", "wall_per_refill_s", "expert_dma_bytes_per_refill", "p95_ttft_s", "p95_tpot_s"}:
                raise ValueError("complete measured workload cost required")
            if not isinstance(row["workload"], str) or not row["workload"] or not isinstance(row["prompt_sha256"], str):
                raise ValueError("invalid frozen workload")
            _integer(row["max_new"], "frozen generation limit", 2)
            cap = _integer(row["draft_prefix_cap"], "measured prefix cap", 0, baseline)
            key = (row["workload"], row["prompt_sha256"], row["max_new"], cap)
            if key in seen:
                raise ValueError("duplicate confidence cost profile")
            seen.add(key)
            refills = _integer(row["refills"], "actual refill count", 1)
            wall = _number(row["wall_s"], "actual refill wall", lower=1e-9)
            cost = _number(row["wall_per_refill_s"], "actual wall cost", lower=1e-9)
            if not math.isclose(wall/refills, cost, rel_tol=1e-12):
                raise ValueError("workload wall cost differs from actual work/time")
            for field in ("expert_dma_bytes_per_refill", "p95_ttft_s", "p95_tpot_s"):
                if row[field] is not None:
                    _number(row[field], field, lower=0.)
        self.calibration = deepcopy(calibration)
        self.baseline = _integer(baseline, "unchanged prefix cap", 1, 256)
        self.mode, self.wall_clock, self.dma_limit = mode, wall_clock, dma_limit
        self.min_gain = _number(min_gain, "minimum confidence gain", lower=0.)
        created, expires = calibration["created_at"], calibration["expires_at"]
        _time(created, expires-created)
        if created > wall_clock():
            raise ValueError("confidence calibration is future-dated")
        self.workload = None
        self.records = deque(maxlen=64)

    def bind_request(self, prompt_ids, max_new):
        actual = digest(list(prompt_ids))
        matches = {row["workload"] for row in self.calibration["costs"]
                   if row["prompt_sha256"] == actual and row["max_new"] == max_new}
        self.workload = next(iter(matches)) if len(matches) == 1 else None

    def select(self, raw_conf, available):
        fallback = min(self.baseline, available)
        bundle = self.calibration
        reason = "no_matching_frozen_workload"
        candidates = []
        if not bundle["created_at"] <= self.wall_clock() < bundle["expires_at"]:
            reason = "confidence_calibration_expired"
        elif self.workload is not None:
            reason = "confidence_unqualified_or_unreliable"
            models = bundle["statistics"]["positions"]
            limits = bundle["limits"]
            reliable = bundle["gpu_qualified"] is True and all(
                row["heldout"]["ece"] is not None and row["heldout"]["ece"] <= limits["max_heldout_ece"]
                and row["heldout"]["covered"] >= bundle["statistics"]["minimum_heldout"] for row in models.values())
            if reliable and isinstance(raw_conf, (list, tuple)) and len(raw_conf) == available:
                survival, expected = 1., [1.]
                for index, score in enumerate(raw_conf[:self.baseline], 1):
                    if survival == 0.:
                        expected.append(expected[-1])
                        continue
                    model = models.get(str(index))
                    if model is None:
                        break
                    try:
                        probability = predict(model["model"], score)
                    except (TypeError, ValueError):
                        probability = None
                    if probability is None:
                        break
                    # Only calibrated conditional probabilities are multiplied.
                    survival *= probability
                    expected.append(expected[-1]+survival)
                for row in bundle["costs"]:
                    cap = row["draft_prefix_cap"]
                    if (row["workload"] != self.workload or cap > available or cap >= len(expected)
                            or row["p95_ttft_s"] is None or row["p95_tpot_s"] is None
                            or row["p95_ttft_s"] > limits["max_p95_ttft_s"]
                            or row["p95_tpot_s"] > limits["max_p95_tpot_s"]):
                        continue
                    dma = row["expert_dma_bytes_per_refill"]
                    if self.dma_limit is not None and (dma is None or dma/expected[cap] > self.dma_limit):
                        continue
                    candidates.append((expected[cap]/row["wall_per_refill_s"], cap, expected[cap]))
                # Baseline must also have evaluable scores and measured costs;
                # otherwise a shorter unsupported choice cannot win by omission.
                if any(cap == fallback for _, cap, _ in candidates):
                    rate, selected, accepted = max(candidates, key=lambda r: (r[0], r[1] == fallback, r[1]))
                    baseline = next(row for row in candidates if row[1] == fallback)
                    if selected != fallback and rate <= baseline[0]*(1+self.min_gain):
                        rate, selected, accepted = baseline
                    self.records.append({"reason": "calibrated_expected_committed_per_wall_s", "selected_cap": selected,
                        "expected_committed": accepted, "expected_rate": rate,
                        "applied_cap": selected if self.mode == "enforce" else fallback})
                    return selected
                reason = "missing_supported_baseline_or_p95_or_dma_budget"
        self.records.append({"reason": reason, "selected_cap": fallback, "applied_cap": fallback})
        return fallback

    def report(self):
        return {"schema": SCHEMA, "workload": self.workload, "mode": self.mode,
            "evidence_sha256": self.calibration["evidence_sha256"], "records": deepcopy(list(self.records)),
            "unknown_expert_dma_is_zero": False,
            "survival_model": "product of position-conditional held-out calibrated acceptance estimates",
            "cost_scope": "actual warm coordinator refill wall including WAN and drain"}

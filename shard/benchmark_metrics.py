"""Coordinator measurement definitions and raw-evidence checks, standard library only."""
from copy import deepcopy
import math

SCHEMA = "shard-coordinator-diagnostics/1"
COUNTERS = ("accepted_predictions", "proposed_predictions", "cancel_events", "speculation_cycles",
            "frames_enqueued", "frames_sent", "replies_received", "frames_judged",
            "stale_replies", "drained_replies", "unsent_frames")
TIMINGS = ("request_elapsed_s", "first_token_s", "last_token_s", "drain_s", "receipt_sweep_s")
DEFINITIONS = {
    "counter_scope": "decode_generation_after_prefill",
    "proposed_predictions": "prediction candidates enqueued for verification, including subsequently unsent candidates",
    "accepted_predictions": "enqueued predicted candidates actually committed",
    "g_cycle": "committed_decode_tokens / speculation_cycles; excludes prefill's first committed token",
    "g_frame": "committed_decode_tokens / frames_sent; decode frames only",
    "frame_waste_ratio": "(stale_replies + drained_replies) / frames_sent; does not count unsent work as sent",
    "inflight_level": "speculation_horizon_minus_committed_frontier; may include queued work, not physical network occupancy",
    "inflight_time_avg": "sum(duration_s * level) / sum(duration_s), unrounded raw intervals",
    "receipt_scope": "coordinator observations, not signed GPU attestation or model execution proof",
}


class MetricsContractError(ValueError):
    pass


def _count(value, name):
    if type(value) is not int or value < 0:
        raise MetricsContractError(f"{name} must be a nonnegative integer")
    return value


def _duration(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise MetricsContractError(f"{name} must be a finite nonnegative duration")
    return float(value)


def percentile(values, p):
    """Linear interpolation on sorted raw samples, with no made-up observations."""
    if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 100:
        raise MetricsContractError("percentile must be in 0..100")
    if not values:
        return None
    rows = sorted(_duration(value, "percentile sample") for value in values)
    position = (len(rows) - 1) * p / 100
    low = int(position); high = min(low + 1, len(rows) - 1)
    return rows[low] + (rows[high] - rows[low]) * (position - low)


def make_coordinator_diagnostics(mode, *, committed_tokens, prefill_tokens=1, counters=None,
                                 timing=None, inflight_intervals=None, backend=None):
    """Normalize explicit counters, retaining missing evidence as None.

    Legacy `g`, `drafted`, event-weighted means and rounded fill estimates are
    deliberately not inferred into these counters. Producers must measure their
    documented scope. Timing is relative to one coordinator request start.
    """
    if mode not in ("greedy", "spec", "dspark", "pipelined"):
        raise MetricsContractError("unsupported coordinator mode")
    total = _count(committed_tokens, "committed_tokens")
    seed = _count(prefill_tokens, "prefill_tokens")
    if seed > total:
        raise MetricsContractError("prefill token count exceeds committed output")
    counters, timing = counters or {}, timing or {}
    if not isinstance(counters, dict) or not isinstance(timing, dict):
        raise MetricsContractError("raw counters and timing must be objects")
    counts = {name: None if counters.get(name) is None else _count(counters[name], name) for name in COUNTERS}
    counts.update(committed_total_tokens=total, prefill_committed_tokens=seed, committed_decode_tokens=total-seed)
    times = {name: None if timing.get(name) is None else _duration(timing[name], name) for name in TIMINGS}
    first, last, returned, sweep = (times[name] for name in ("first_token_s", "last_token_s", "request_elapsed_s", "receipt_sweep_s"))
    if first is not None and last is not None:
        if last < first:
            raise MetricsContractError("last token precedes first token")
        times["decode_s"] = last-first
    else:
        times["decode_s"] = None
    if last is not None and returned is not None:
        if returned < last:
            raise MetricsContractError("request returned before the final committed token")
        drain = returned-last
        if times["drain_s"] is not None and not math.isclose(times["drain_s"], drain, rel_tol=1e-8, abs_tol=1e-8):
            raise MetricsContractError("drain boundary differs from return-minus-last-token")
        times["drain_s"] = drain
    times["full_service_s"] = returned+sweep if returned is not None and sweep is not None else None
    intervals = None
    if inflight_intervals is not None:
        if not isinstance(inflight_intervals, list) or len(inflight_intervals) > 1_000_000:
            raise MetricsContractError("bounded raw inflight intervals required")
        intervals = []
        for row in inflight_intervals:
            if not isinstance(row, dict) or set(row) != {"duration_s", "level"}:
                raise MetricsContractError("inflight intervals need exact duration_s/level")
            intervals.append({"duration_s": _duration(row["duration_s"], "interval duration"),
                              "level": _count(row["level"], "interval level")})
    try:
        area = sum(row["duration_s"]*row["level"] for row in intervals) if intervals is not None else None
    except OverflowError as exc:
        raise MetricsContractError("inflight duration/area overflow") from exc
    span = sum(row["duration_s"] for row in intervals) if intervals is not None else None
    if any(value is not None and not math.isfinite(value) for value in (area, span)):
        raise MetricsContractError("inflight duration/area overflow")
    accepted, proposed = counts["accepted_predictions"], counts["proposed_predictions"]
    if accepted is not None and (accepted > total-seed or proposed is not None and accepted > proposed):
        raise MetricsContractError("accepted predictions exceed proposed or committed decode tokens")
    enqueued, sent, unsent = (counts[name] for name in ("frames_enqueued", "frames_sent", "unsent_frames"))
    if None not in (enqueued, sent, unsent) and enqueued != sent+unsent:
        raise MetricsContractError("enqueued frames differ from sent plus unsent")
    received, judged, stale, drained = (counts[name] for name in ("replies_received", "frames_judged", "stale_replies", "drained_replies"))
    if None not in (received, judged, stale, drained) and received != judged+stale+drained:
        raise MetricsContractError("reply accounting differs from judged/stale/drained")
    if sent is not None and received is not None and received > sent:
        raise MetricsContractError("received replies exceed sent decode frames")
    if None not in (sent, stale, drained) and stale+drained > sent:
        raise MetricsContractError("wasted replies exceed sent decode frames")
    def ratio(numerator, denominator):
        return numerator/denominator if numerator is not None and denominator is not None and denominator > 0 else None
    derived = {"g_cycle": ratio(total-seed, counts["speculation_cycles"]),
        "g_frame": ratio(total-seed, sent), "acceptance_ratio": ratio(accepted, proposed),
        "frame_waste_ratio": ratio(stale+drained if stale is not None and drained is not None else None, sent),
        "inflight_area_s": area, "inflight_span_s": span,
        "inflight_time_avg": ratio(area, span),
        "max_inflight": max((row["level"] for row in intervals), default=0) if intervals is not None else None}
    definitions = dict(DEFINITIONS, frame_unit="single_token_decode_frame" if mode == "pipelined" else "decode_verification_traversal_may_contain_multiple_tokens")
    return {"schema": SCHEMA, "mode": mode, "definitions": definitions, "counts": counts,
            "timing": times, "inflight_intervals": intervals, "derived": derived,
            "backend": deepcopy(backend), "missing_counters": [name for name in COUNTERS if counts[name] is None]}


def validate_coordinator_diagnostics(value):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise MetricsContractError("versioned coordinator diagnostics required")
    counts = value.get("counts", {})
    expected = make_coordinator_diagnostics(value.get("mode"), committed_tokens=counts.get("committed_total_tokens"),
        prefill_tokens=counts.get("prefill_committed_tokens"), counters=counts, timing=value.get("timing"),
        inflight_intervals=value.get("inflight_intervals"), backend=value.get("backend"))
    if value != expected:
        raise MetricsContractError("diagnostic definitions/counts/derived values differ from their raw evidence")
    return value

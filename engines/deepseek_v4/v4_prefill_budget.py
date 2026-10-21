"""Bounded measured prefill quanta; predictions never authorize a numerical ABI.

DeepInfer's time-quota scheduling is a useful design, not an available WAN
scheduler. This implementation uses this runner's own measured equal-frontier
cohorts. Missing/stale/unseen shapes fall back to one native token round.
"""
from copy import deepcopy
import math
import time

SCHEMA = "flyai-v4-prefill-budget/1"
BINDING = {"numerical_contract", "cohort_id", "source_sha256", "runtime_config_sha256",
           "max_context_tokens", "max_jobs"}


def _int(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"invalid {name}")
    return value


def _real(value, name, low=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < low:
        raise ValueError(f"invalid {name}")
    return float(value)


def validate_binding(binding):
    if not isinstance(binding, dict) or set(binding) != BINDING:
        raise ValueError("exact prefill runner binding required")
    _int(binding["max_jobs"], "max_jobs", 1, 32)
    _int(binding["max_context_tokens"], "max_context_tokens", 1, 1_000_000)
    for key in ("numerical_contract", "cohort_id"):
        if not isinstance(binding[key], str) or not 1 <= len(binding[key]) <= 256:
            raise ValueError(f"invalid {key}")
    for key in ("source_sha256", "runtime_config_sha256"):
        if (not isinstance(binding[key], str) or len(binding[key]) != 64
                or any(c not in "0123456789abcdef" for c in binding[key])):
            raise ValueError(f"invalid {key}")
    return deepcopy(binding)


def validate_config(config):
    fields = {"schema", "binding", "measured_at", "ttl_s", "quota_ms", "samples"}
    if (not isinstance(config, dict) or not fields.issubset(config) or set(config) - fields - {"measurement_scope"}
            or config["schema"] != SCHEMA):
        raise ValueError("invalid measured prefill budget")
    if config.get("measurement_scope", "operator_supplied_cohort_wall_timings") not in (
            "operator_supplied_cohort_wall_timings", "observed_consecutive_token_round_span_including_cpu_gaps",
            "actual_single_advance_quantum_wall"):
        raise ValueError("unknown prefill timing scope")
    binding = validate_binding(config["binding"])
    _real(config["measured_at"], "measured_at")
    ttl = _real(config["ttl_s"], "ttl_s", 1)
    quota = _real(config["quota_ms"], "quota_ms", .01)
    if ttl > 86400 or quota > 10000:
        raise ValueError("prefill TTL/quota exceeds limit")
    samples = config["samples"]
    if not isinstance(samples, list) or not 1 <= len(samples) <= 512:
        raise ValueError("bounded prefill cost samples required")
    seen = set()
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {
                "context_tokens", "batch_jobs", "quantum_tokens", "elapsed_ms"}:
            raise ValueError("invalid prefill cost sample")
        shape = (_int(sample["context_tokens"], "context_tokens", 0, binding["max_context_tokens"]),
                 _int(sample["batch_jobs"], "batch_jobs", 1, binding["max_jobs"]),
                 _int(sample["quantum_tokens"], "quantum_tokens", 1, 8))
        if shape in seen:
            raise ValueError("duplicate prefill cost shape")
        seen.add(shape)
        times = sample["elapsed_ms"]
        if not isinstance(times, list) or not 5 <= len(times) <= 128:
            raise ValueError("each shape needs 5..128 actual cohort timings")
        for value in times:
            _real(value, "elapsed_ms", .000001)
    return deepcopy(config)


class TimeBudget:
    def __init__(self, config, binding, *, clock=None):
        self.binding = validate_binding(binding)
        self.config = validate_config(config) if config is not None else None
        if self.config is not None and self.config["binding"] != self.binding:
            raise ValueError("prefill calibration belongs to another runner")
        self.clock = clock or time.time
        self.decisions = self.fallbacks = 0
        self.reason = "no_measurements"

    def choose(self, *, candidates, context, batch_jobs, now=None):
        choices = list(candidates)
        if (not choices or len(choices) > 8 or len(set(choices)) != len(choices)
                or any(type(v) is not int or not 1 <= v <= 8 for v in choices) or 1 not in choices):
            raise ValueError("prefill candidate quanta must contain baseline 1")
        _int(context, "context", 0, self.binding["max_context_tokens"])
        _int(batch_jobs, "batch_jobs", 1, self.binding["max_jobs"])
        self.decisions += 1
        now = self.clock() if now is None else _real(now, "now")
        config = self.config
        eligible = []
        self.reason = "no_measurements" if config is None else "unmeasured_shape"
        if config is not None:
            age = now - config["measured_at"]
            if not 0 <= age <= config["ttl_s"]:
                self.reason = "expired_or_future_measurements"
            else:
                for quantum in choices:
                    # Compression boundaries and smaller batches need not be
                    # faster. Never treat a dominating shape as a measured
                    # bound for an unseen frontier.
                    upper = [row for row in config["samples"] if row["quantum_tokens"] == quantum
                             and row["context_tokens"] == context and row["batch_jobs"] == batch_jobs]
                    if not upper:
                        continue
                    p95 = max(sorted(r["elapsed_ms"])[math.ceil(.95 * (len(r["elapsed_ms"]) - 1))] for r in upper)
                    if p95 <= config["quota_ms"]:
                        eligible.append(quantum)
                if eligible:
                    self.reason = "measured_p95_envelope"
        selected = max(eligible, default=1)
        if selected == 1:
            self.fallbacks += 1
        return selected

    def status(self):
        return {"schema": SCHEMA, "prediction_only": True, "decisions": self.decisions,
                "baseline_decisions": self.fallbacks, "reason": self.reason,
                "measurement_scope": self.config.get("measurement_scope", "operator_supplied_cohort_wall_timings")
                    if self.config is not None else None,
                "scope": "local measured time quota; no batch-invariance or GPU qualification"}

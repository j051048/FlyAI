"""Bounded, request-shaped admission forecasts from verified local executions.

Service timings include coordinator/transport costs. They are observations, not
GPU kernel measurements or an end-to-end percentile guarantee. Cache metadata is
advisory and may only reduce the forecast when the backend supplies fresh, exact
request/runtime-bound evidence that the prefill actually can be skipped.
"""
from __future__ import annotations

from collections import OrderedDict, deque
import hashlib
import json
import math
import statistics
import threading
import time


def _number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def runtime_identity(backend, cohort):
    method = getattr(backend, "routing_identity", None)
    if not callable(method):
        return None
    try:
        value = method()
        digest = value.get("runtime_config_sha256")
        if (value.get("cohort_id") != cohort or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            return None
        return {"cohort_id": cohort, "runtime_config_sha256": digest}
    except (TypeError, ValueError, AttributeError):
        return None


def production_policy(value):
    """Local operator opt-in; never accepted from contributor offers."""
    if value is None:
        return None
    fields = {"enabled", "context_bucket_tokens", "output_bucket_tokens", "min_samples",
              "uncertainty_fraction", "explore_every", "max_probe_extra_s"}
    if not isinstance(value, dict) or set(value) != fields or value.get("enabled") is not True:
        raise ValueError("complete production routing policy required")
    for key, low, high in (("context_bucket_tokens", 16, 4096), ("output_bucket_tokens", 1, 512),
                          ("min_samples", 3, 64), ("explore_every", 10, 10000)):
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError("bounded production routing policy required")
    if (not _number(value["uncertainty_fraction"], positive=True) or not .05 <= value["uncertainty_fraction"] <= 2
            or not _number(value["max_probe_extra_s"]) or value["max_probe_extra_s"] > 10):
        raise ValueError("bounded routing uncertainty and probe cost required")
    return dict(value)


class RoutingCosts:
    def __init__(self, *, ttl_s=120.0, max_shapes=64, samples_per_shape=8, clock=time.time, production=None):
        if (not _number(ttl_s, positive=True) or type(max_shapes) is not int or not 1 <= max_shapes <= 1024
                or type(samples_per_shape) is not int or not 1 <= samples_per_shape <= 64):
            raise ValueError("finite bounded routing observation policy required")
        self.ttl_s, self.max_shapes, self.samples_per_shape, self.clock = ttl_s, max_shapes, samples_per_shape, clock
        self._samples = OrderedDict()
        self._traffic = deque(maxlen=256)
        self._lock = threading.RLock()
        self.production = production_policy(production)
        if self.production is not None:
            self.samples_per_shape = max(self.samples_per_shape, self.production["min_samples"])
        self._bucket_samples = OrderedDict()

    def _bucket(self, payload, count, maximum, identity):
        if self.production is None or identity is None:
            return None
        if type(maximum) is not int or maximum <= 0:
            return None
        base = self.shape(payload, count, identity)[1:]
        return (count // self.production["context_bucket_tokens"],
                maximum // self.production["output_bucket_tokens"], *base)

    def note_completion(self, payload, prompt_tokens, max_new_tokens):
        ids = payload.get("prompt_ids")
        prompt_sha256 = None
        if (isinstance(ids, list) and len(ids) == prompt_tokens
                and all(type(token) is int and token >= 0 for token in ids)):
            prompt_sha256 = hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with self._lock:
            self._traffic.append({"at": self.clock(), "prompt_tokens": prompt_tokens,
                "max_new": max_new_tokens, "mode": payload.get("mode"), "prompt_sha256": prompt_sha256})

    def matching_traffic(self, workload, *, window_s):
        now = self.clock()
        with self._lock:
            recent = [row for row in self._traffic if 0 <= now - row["at"] <= window_s]
        hashes = workload.get("prompt_sha256s", [])
        matching = [row for row in recent if row["prompt_sha256"] in hashes
                    and all(row.get(key) == value for key, value in workload.items() if key != "prompt_sha256s")]
        counts = [sum(row["prompt_sha256"] == digest for row in matching) for digest in hashes]
        return {"total": len(recent), "matching": len(matching), "window_s": window_s,
                "balanced_frozen_mix": bool(counts) and min(counts) >= 1 and max(counts)-min(counts) <= 1}

    @staticmethod
    def shape(payload, count, identity):
        # Exact context and execution mode: do not extrapolate quadratic prefill
        # or silently mix a new runtime/quantizer with its predecessor's samples.
        return (count, payload.get("mode"), bool(payload.get("policy_allowed", False)),
                payload.get("numerical_contract"), None if identity is None else identity["runtime_config_sha256"])

    def observe(self, *, payload, prompt_tokens, token_count, first_s, last_s,
                elapsed_s, identity, cache_hit=False, max_new_tokens=None):
        if (type(prompt_tokens) is not int or prompt_tokens <= 0 or type(token_count) is not int
                or token_count < 2 or any(not _number(x) for x in (first_s, last_s, elapsed_s))
                or not 0 < first_s < last_s <= elapsed_s or cache_hit):
            return False
        key = self.shape(payload, prompt_tokens, identity)
        row = {"measured_at": self.clock(), "prefill_s": first_s,
               "decode_token_s": (last_s - first_s) / (token_count - 1),
               "completion_s": elapsed_s - last_s}
        with self._lock:
            rows = self._samples.setdefault(key, [])
            rows.append(row)
            del rows[:-self.samples_per_shape]
            self._samples.move_to_end(key)
            while len(self._samples) > self.max_shapes:
                self._samples.popitem(last=False)
            bucket = self._bucket(payload, prompt_tokens, max_new_tokens, identity)
            if bucket is not None:
                values = self._bucket_samples.setdefault(bucket, [])
                values.append({**row, "prompt_tokens": prompt_tokens})
                del values[:-self.samples_per_shape]
                self._bucket_samples.move_to_end(bucket)
                while len(self._bucket_samples) > self.max_shapes:
                    self._bucket_samples.popitem(last=False)
        return True

    def estimate(self, backend, *, cohort, payload, prompt_tokens, max_new_tokens,
                 fallback_s_per_token, tenant, request_fingerprint):
        now = self.clock()
        identity = runtime_identity(backend, cohort)
        with self._lock:
            rows = [r for r in self._samples.get(self.shape(payload, prompt_tokens, identity), [])
                    if 0 <= now - r["measured_at"] <= self.ttl_s]
            bucket = self._bucket(payload, prompt_tokens, max_new_tokens, identity)
            grouped = [r for r in self._bucket_samples.get(bucket, [])
                       if 0 <= now-r["measured_at"] <= self.ttl_s] if bucket is not None else []
        if not rows and self.production is not None and len(grouped) >= self.production["min_samples"]:
            # A bucket is a bounded forecast, not a numerical qualification.
            # Limit extrapolation and price the worst observed phase with an
            # explicit safety allowance; no prompt hash or tenant text is kept.
            minimum = min(r["prompt_tokens"] for r in grouped)
            maximum = max(r["prompt_tokens"] for r in grouped)
            ratio = prompt_tokens / minimum
            if minimum*.8 <= prompt_tokens <= maximum*1.25:
                margin = 1 + self.production["uncertainty_fraction"]
                prefill = max(r["prefill_s"] for r in grouped) * max(1., ratio**2) * margin
                decode = max(r["decode_token_s"] for r in grouped) * max(0, max_new_tokens-1) * margin
                completion = max(r["completion_s"] for r in grouped) * margin
                return {"service_s": prefill+decode+completion, "source": "bounded_local_shape_forecast",
                    "fresh_samples": len(grouped), "prefill_s": prefill, "decode_s": decode,
                    "completion_s": completion, "cache_saved_s": 0., "restore_s": None,
                    "runtime_identity": identity, "measured_at": max(r["measured_at"] for r in grouped),
                    "context_range": [minimum, maximum], "ttl_s": self.ttl_s,
                    "uncertainty": ["bucket_forecast_not_exact_workload", "worst_phase_with_safety_allowance"]}
        if not rows:
            return {"service_s": (prompt_tokens + max_new_tokens) * fallback_s_per_token,
                    "source": "warmup_or_service_scalar_fallback", "fresh_samples": 0,
                    "cache_saved_s": 0.0, "restore_s": None,
                    "uncertainty": ["no_fresh_matching_phase_observation"]}
        prefill = statistics.median(r["prefill_s"] for r in rows)
        decode = statistics.median(r["decode_token_s"] for r in rows) * max(0, max_new_tokens - 1)
        completion = statistics.median(r["completion_s"] for r in rows)
        saved, restore = 0.0, 0.0
        uncertainty = []
        hook = getattr(backend, "routing_cache_hint", None)
        if identity is not None and callable(hook):
            try:
                hint = hook(payload, tenant=tenant, request_fingerprint=request_fingerprint, now=now)
                if hint is not None:
                    if (not isinstance(hint, dict) or hint.get("schema") != "shard-routing-cache-hint/1"
                            or any(hint.get(k) != v for k, v in identity.items())
                            or hint.get("request_fingerprint") != request_fingerprint
                            or hint.get("source") != "verified_exact_prefix_restore"
                            or hint.get("skips_prefill") is not True
                            or type(hint.get("prefix_tokens")) is not int
                            or hint["prefix_tokens"] != prompt_tokens
                            or not _number(hint.get("measured_at"))
                            or not _number(hint.get("ttl_s"), positive=True)
                            or not 0 <= now - hint["measured_at"] <= min(hint["ttl_s"], self.ttl_s)
                            or not _number(hint.get("restore_seconds"))
                            or not _number(hint.get("first_token_seconds"), positive=True)
                            or hint["first_token_seconds"] < hint["restore_seconds"]):
                        raise ValueError("cache hint is not fresh/exact/request/runtime bound")
                    # A hit still performs first-token work: a pure DMA timer
                    # cannot be subtracted from the entire cold TTFT estimate.
                    restore = hint["restore_seconds"]
                    saved = max(0.0, prefill - hint["first_token_seconds"])
            except (AttributeError, TypeError, ValueError, KeyError):
                uncertainty.append("cache_hint_unusable")
        return {"service_s": prefill - saved + decode + completion,
                "source": "local_verified_service_phases", "fresh_samples": len(rows),
                "measured_at": max(r["measured_at"] for r in rows), "ttl_s": self.ttl_s,
                "prefill_s": prefill - saved, "decode_s": decode, "completion_s": completion,
                "cache_saved_s": saved, "restore_s": restore,
                "runtime_identity": identity, "uncertainty": uncertainty}

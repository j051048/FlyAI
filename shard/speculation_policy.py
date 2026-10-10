"""Opt-in request-boundary selection among locally approved V4 recipes.

This controller never changes model weights, trained block width, layer placement,
in-flight epochs or lazy-hint arithmetic. Its measurements are coordinator
observations, not remote execution proofs or GPU performance certification.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from copy import deepcopy
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import secrets
import statistics
import threading
import time
from types import MappingProxyType

try:
    from .benchmark_metrics import COUNTERS, make_coordinator_diagnostics, percentile, validate_coordinator_diagnostics
except ImportError:  # flat engine deployment
    from benchmark_metrics import COUNTERS, make_coordinator_diagnostics, percentile, validate_coordinator_diagnostics

DECISION_SCHEMA = "shard-speculation-decision/1"
FEEDBACK_SCHEMA = "shard-speculation-feedback/1"
_HEX = re.compile(r"[a-f0-9]{64}\Z")
_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class SpeculationPolicyError(ValueError):
    pass


def _text(value, name, maximum=256):
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise SpeculationPolicyError(f"invalid {name}")
    return value


def _integer(value, name, minimum=0, maximum=1_000_000_000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise SpeculationPolicyError(f"invalid {name}")
    return value


def _seconds(value, name, *, positive=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0 and (not positive or value != 0)
    except OverflowError:
        valid = False
    if not valid:
        raise SpeculationPolicyError(f"invalid {name}")
    return float(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SpeculationRecipe:
    recipe_id: str
    mode: str
    depth: int = 1
    floor: int = 1
    lazy: bool = False

    def __post_init__(self):
        _text(self.recipe_id, "recipe ID", 64)
        _integer(self.depth, "depth", 1, 256)
        _integer(self.floor, "refill floor", 1, self.depth)
        if type(self.lazy) is not bool or self.mode not in ("greedy", "pipelined"):
            raise SpeculationPolicyError("only greedy and pipelined approved recipes are supported")
        if self.mode == "greedy" and (self.depth, self.floor, self.lazy) != (1, 1, False):
            raise SpeculationPolicyError("greedy recipe must use depth=1, floor=1 and lazy=False")
        if self.mode == "pipelined" and self.depth < 2:
            raise SpeculationPolicyError("pipelined recipe requires depth >= 2")

    def to_dict(self):
        return {"recipe_id": self.recipe_id, "mode": self.mode, "depth": self.depth,
                "floor": self.floor, "lazy": self.lazy}

    def coordinator_kwargs(self):
        return {"depth": self.depth, "floor": self.floor, "lazy": self.lazy} if self.mode == "pipelined" else {}


@dataclass(frozen=True)
class SpeculationContext:
    tenant_id: str
    cohort_id: str
    ring_id: str
    ring_generation: str
    runtime_config_sha256: str
    context_tokens: int
    max_new_tokens: int
    workload: str = "unknown"

    def __post_init__(self):
        for name in ("tenant_id", "ring_id", "ring_generation", "workload"):
            _text(getattr(self, name), name, 64 if name == "workload" else 256)
        for name in ("cohort_id", "runtime_config_sha256"):
            if not isinstance(getattr(self, name), str) or not _HEX.fullmatch(getattr(self, name)):
                raise SpeculationPolicyError(f"invalid {name}")
        _integer(self.context_tokens, "context token count", 1)
        _integer(self.max_new_tokens, "maximum output token count", 1)


@dataclass(frozen=True)
class RequestFeedback:
    mode: str
    committed_tokens: int
    prefill_tokens: int
    first_token_s: float
    last_token_s: float
    request_elapsed_s: float
    receipt_sweep_s: float
    counter_items: tuple
    judged_predictions: int | None = None
    inflight_time_avg: float | None = None
    success: bool = True
    replayed: bool = False
    cancelled: bool = False
    observed_recipe: SpeculationRecipe | None = None

    def __post_init__(self):
        if self.mode not in ("greedy", "pipelined"):
            raise SpeculationPolicyError("feedback mode is outside the approved policy scope")
        _integer(self.committed_tokens, "committed token count")
        _integer(self.prefill_tokens, "prefill committed token count", 0, self.committed_tokens)
        for name in ("first_token_s", "last_token_s", "request_elapsed_s", "receipt_sweep_s"):
            _seconds(getattr(self, name), name)
        if not self.first_token_s <= self.last_token_s <= self.request_elapsed_s:
            raise SpeculationPolicyError("feedback clock boundaries are reversed")
        if any(type(getattr(self, name)) is not bool for name in ("success", "replayed", "cancelled")):
            raise SpeculationPolicyError("feedback outcome flags must be boolean")
        if not isinstance(self.counter_items, tuple) or any(not isinstance(row, tuple) or len(row) != 2 for row in self.counter_items):
            raise SpeculationPolicyError("immutable raw feedback counters required")
        counters = dict(self.counter_items)
        if len(counters) != len(self.counter_items) or set(counters) != set(COUNTERS):
            raise SpeculationPolicyError("exact coordinator counter names required")
        needed = set(COUNTERS) - {"speculation_cycles"}
        if any(counters[name] is None for name in needed):
            raise SpeculationPolicyError("raw actual-work counters are incomplete")
        make_coordinator_diagnostics(self.mode, committed_tokens=self.committed_tokens, prefill_tokens=self.prefill_tokens,
            counters=counters, timing={name: getattr(self, name) for name in ("first_token_s", "last_token_s", "request_elapsed_s", "receipt_sweep_s")})
        if self.judged_predictions is not None:
            _integer(self.judged_predictions, "judged prediction count", counters["accepted_predictions"],
                     min(counters["proposed_predictions"], counters["frames_judged"]))
        if counters["proposed_predictions"] > counters["frames_enqueued"]:
            raise SpeculationPolicyError("single-token predicted frames exceed actual enqueued frames")
        if self.mode == "greedy" and (counters["accepted_predictions"] or counters["proposed_predictions"] or self.judged_predictions not in (None, 0)):
            raise SpeculationPolicyError("greedy feedback cannot contain speculative predictions")
        if self.inflight_time_avg is not None:
            _seconds(self.inflight_time_avg, "time-weighted inflight")
        if self.observed_recipe is not None and (not isinstance(self.observed_recipe, SpeculationRecipe) or self.observed_recipe.mode != self.mode):
            raise SpeculationPolicyError("observed recipe differs from actual feedback mode")

    @classmethod
    def from_diagnostics(cls, value, *, judged_predictions=None, success=True, replayed=False, cancelled=False, observed_recipe=None):
        value = validate_coordinator_diagnostics(value)
        counts, timing = value["counts"], value["timing"]
        return cls(value["mode"], counts["committed_total_tokens"], counts["prefill_committed_tokens"],
            *(timing[name] for name in ("first_token_s", "last_token_s", "request_elapsed_s", "receipt_sweep_s")),
            tuple((name, counts[name]) for name in COUNTERS), judged_predictions,
            value["derived"]["inflight_time_avg"], success, replayed, cancelled, observed_recipe)

    @property
    def counters(self):
        return dict(self.counter_items)

    @property
    def committed_decode_tokens(self):
        return self.committed_tokens - self.prefill_tokens

    @property
    def decode_and_drain_s(self):
        return self.request_elapsed_s - self.first_token_s

    def to_dict(self):
        counts = self.counters
        result = {"mode": self.mode, "committed_tokens": self.committed_tokens, "prefill_tokens": self.prefill_tokens,
            "committed_decode_tokens": self.committed_decode_tokens, "first_token_s": self.first_token_s,
            "decode_s": self.last_token_s-self.first_token_s, "drain_s": self.request_elapsed_s-self.last_token_s,
            "request_elapsed_s": self.request_elapsed_s, "receipt_sweep_s": self.receipt_sweep_s,
            "decode_and_drain_s": self.decode_and_drain_s, "counters": counts,
            "judged_predictions": self.judged_predictions,
            "conditional_acceptance": counts["accepted_predictions"]/self.judged_predictions if self.judged_predictions else None,
            "proposal_yield": counts["accepted_predictions"]/counts["proposed_predictions"] if counts["proposed_predictions"] else None,
            "inflight_time_avg": self.inflight_time_avg, "success": self.success, "replayed": self.replayed, "cancelled": self.cancelled}
        result["observed_recipe"] = self.observed_recipe.to_dict() if self.observed_recipe is not None else None
        return result


def feedback_from_result(result, *, first_token_s, last_token_s, request_elapsed_s, receipt_sweep_s=None,
                         success=True, replayed=False, cancelled=False, expected_recipe=None):
    """Build feedback from actual coordinator counters and independently observed clocks.

    Only judged depth-table trials estimate conditional acceptance. Frames cancelled
    before judgment still cost wall time, but do not become rejection observations.
    """
    if not isinstance(result, dict) or not isinstance(result.get("tokens"), list):
        raise SpeculationPolicyError("actual coordinator token result required")
    if any(type(token) is not int or token < 0 for token in result["tokens"]) or success and result.get("ok") is not True:
        raise SpeculationPolicyError("successful committed integer-token result required")
    if receipt_sweep_s is None:
        receipt_sweep_s = result.get("receipt_sweep_s")
    if expected_recipe is not None:
        if not isinstance(expected_recipe, SpeculationRecipe) or expected_recipe.mode != result.get("mode"):
            raise SpeculationPolicyError("actual coordinator mode differs from expected recipe")
        if expected_recipe.mode == "pipelined" and any(result.get(name) != getattr(expected_recipe, name) or
                type(result.get(name)) is not type(getattr(expected_recipe, name)) for name in ("depth", "floor", "lazy")):
            raise SpeculationPolicyError("actual coordinator depth/floor/lazy differs from expected recipe")
    value = make_coordinator_diagnostics(result.get("mode"), committed_tokens=len(result["tokens"]),
        prefill_tokens=result.get("prefill_committed_tokens", min(1, len(result["tokens"]))),
        counters=result.get("coordinator_counters"), inflight_intervals=result.get("inflight_intervals"),
        timing={"first_token_s": first_token_s, "last_token_s": last_token_s,
                "request_elapsed_s": request_elapsed_s, "receipt_sweep_s": receipt_sweep_s})
    judged = 0 if result.get("mode") == "greedy" else None
    if result.get("mode") == "pipelined" and all(isinstance(result.get(name), dict) for name in ("accept_by_depth", "topup_accept_by_depth")):
        hits = trials = 0
        for table in (result["accept_by_depth"], result["topup_accept_by_depth"]):
            seen_depths = set()
            for depth, row in table.items():
                if not (type(depth) is int and depth > 0 or isinstance(depth, str) and re.fullmatch(r"[1-9][0-9]*", depth)):
                    raise SpeculationPolicyError("invalid judged prediction depth")
                if not isinstance(row, (tuple, list)) or len(row) != 2:
                    raise SpeculationPolicyError("invalid judged prediction row")
                depth = int(depth)
                if depth in seen_depths:
                    raise SpeculationPolicyError("duplicate judged prediction depth")
                seen_depths.add(depth)
                hit = _integer(row[0], "judged hits"); trial = _integer(row[1], "judged trials", hit)
                hits += hit; trials += trial
        if hits != value["counts"]["accepted_predictions"]:
            raise SpeculationPolicyError("judged prediction hits differ from actual committed acceptance")
        judged = trials
    return RequestFeedback.from_diagnostics(value, judged_predictions=judged, success=success, replayed=replayed, cancelled=cancelled, observed_recipe=expected_recipe)


@dataclass(frozen=True)
class PolicyDecision:
    decision_id: str
    policy_sha256: str
    recipe: SpeculationRecipe
    reason: str
    enabled: bool
    is_probe: bool
    selected_at: float
    sequence: int
    _issuer: object = field(repr=False, compare=False)
    _learning: bool = field(repr=False, compare=False)

    def to_dict(self):
        # No tenant ID, history, context key, prompt or other requests' statistics.
        return {"schema": DECISION_SCHEMA, "decision_id": self.decision_id, "policy_sha256": self.policy_sha256,
                "recipe": self.recipe.to_dict(), "reason": self.reason, "enabled": self.enabled,
                "is_probe": self.is_probe, "selected_at": self.selected_at, "sequence": self.sequence}


@dataclass
class _Arm:
    samples: deque
    attempts: int = 0
    last_attempt: float | None = None


@dataclass
class _State:
    arms: dict
    incumbent: str
    last_activity: float
    sequence: int = 0
    last_switch: int = 0
    probes: int = 0
    probe_budget_at: float = 0


class RequestPolicy:
    """Bounded in-memory, tenant-isolated policy. Approved recipes are a LOCAL contract.

    The caller must validate every recipe against exact node calibration/templates
    before constructing this object. Public request parameters cannot add recipes.
    """

    def __init__(self, approved_recipes, baseline_id, *, enabled=False, clock=time.monotonic,
                 min_observations=2, baseline_attempts=4, probe_attempts=2, max_probe_requests=32,
                 min_gain=.05, min_hold_requests=3, feedback_ttl_s=600, decision_ttl_s=1800,
                 max_states=256, max_pending=256, sample_capacity=16, completion_capacity=256,
                 min_decode_tokens=4):
        rows = tuple(approved_recipes)
        if not 1 <= len(rows) <= 16 or any(not isinstance(row, SpeculationRecipe) for row in rows):
            raise SpeculationPolicyError("one to sixteen locally approved recipe objects required")
        if len({row.recipe_id for row in rows}) != len(rows) or len({(row.mode,row.depth,row.floor,row.lazy) for row in rows}) != len(rows):
            raise SpeculationPolicyError("duplicate recipe IDs or configurations")
        if baseline_id not in {row.recipe_id for row in rows} or type(enabled) is not bool:
            raise SpeculationPolicyError("known baseline and explicit enabled flag required")
        for name, number in (("min_observations", min_observations), ("baseline_attempts", baseline_attempts),
                ("probe_attempts", probe_attempts), ("max_probe_requests", max_probe_requests), ("min_hold_requests", min_hold_requests),
                ("max_states", max_states), ("max_pending", max_pending), ("sample_capacity", sample_capacity),
                ("completion_capacity", completion_capacity), ("min_decode_tokens", min_decode_tokens)):
            _integer(number, name, 1, 8192)
        if min_observations > sample_capacity or baseline_attempts < min_observations or probe_attempts < min_observations:
            raise SpeculationPolicyError("observation budget exceeds attempts or sample capacity")
        _seconds(min_gain, "minimum gain")
        if min_gain > 1:
            raise SpeculationPolicyError("minimum gain exceeds one")
        _seconds(feedback_ttl_s, "feedback TTL", positive=True); _seconds(decision_ttl_s, "decision TTL", positive=True)
        self.recipes = MappingProxyType({row.recipe_id: row for row in rows})
        self.baseline_id, self.enabled, self.clock = baseline_id, enabled, clock
        self.min_observations, self.baseline_attempts, self.probe_attempts = min_observations, baseline_attempts, probe_attempts
        self.max_probe_requests, self.min_gain, self.min_hold_requests = max_probe_requests, min_gain, min_hold_requests
        self.feedback_ttl_s, self.decision_ttl_s = feedback_ttl_s, decision_ttl_s
        self.max_states, self.max_pending, self.sample_capacity = max_states, max_pending, sample_capacity
        self.completion_capacity, self.min_decode_tokens = completion_capacity, min_decode_tokens
        self._states, self._pending, self._completed = OrderedDict(), {}, OrderedDict()
        self._issuer, self._scope_secret, self._lock = object(), secrets.token_bytes(32), threading.RLock()
        self._configuration = {"schema": "shard-speculation-policy/1", "recipes": [row.to_dict() for row in rows],
            "implementation_sha256": _IMPLEMENTATION_SHA256,
            "baseline_id": baseline_id, "enabled": enabled, "min_observations": min_observations,
            "baseline_attempts": baseline_attempts, "probe_attempts": probe_attempts, "max_probe_requests": max_probe_requests,
            "min_gain": min_gain, "min_hold_requests": min_hold_requests, "feedback_ttl_s": feedback_ttl_s,
            "decision_ttl_s": decision_ttl_s, "max_states": max_states, "max_pending": max_pending,
            "sample_capacity": sample_capacity, "completion_capacity": completion_capacity, "min_decode_tokens": min_decode_tokens}
        self.policy_sha256 = _digest(self._configuration)

    def configuration(self):
        """Public reproducibility metadata, excluding scope keys and request history."""
        return deepcopy(self._configuration)

    def _scope(self, context):
        if not isinstance(context, SpeculationContext):
            raise SpeculationPolicyError("trusted request context object required")
        body = [context.tenant_id, context.cohort_id, context.ring_id, context.ring_generation,
                context.runtime_config_sha256, context.context_tokens.bit_length(), context.max_new_tokens.bit_length(), context.workload]
        return hmac.new(self._scope_secret, json.dumps(body, separators=(",", ":")).encode(), hashlib.sha256).digest()

    def _decision(self, recipe, reason, now, sequence=0, *, probe=False, learning=False):
        return PolicyDecision(secrets.token_hex(16), self.policy_sha256, recipe, reason, self.enabled,
                              probe, now, sequence, self._issuer, learning)

    def choose(self, context):
        """Call once at the request's execution boundary, never in a token loop."""
        scope = self._scope(context)
        now = _seconds(self.clock(), "policy clock")
        with self._lock:
            baseline = self.recipes[self.baseline_id]
            if not self.enabled:
                return self._decision(baseline, "disabled_baseline", now)
            for identifier, pending in list(self._pending.items()):
                if now-pending[0].selected_at >= self.decision_ttl_s:
                    self._pending.pop(identifier)
                    self._completed[identifier] = (pending[0], None, None)
            while len(self._completed) > self.completion_capacity:
                self._completed.popitem(last=False)
            if len(self._pending) >= self.max_pending:
                return self._decision(baseline, "pending_capacity_baseline", now)
            state = self._states.get(scope)
            if state is not None and now-state.last_activity >= self.feedback_ttl_s:
                self._states.pop(scope); state = None
            if state is None:
                busy = {entry[1] for entry in self._pending.values()}
                if len(self._states) >= self.max_states:
                    removable = next((key for key in self._states if key not in busy), None)
                    if removable is None:
                        return self._decision(baseline, "state_capacity_baseline", now)
                    self._states.pop(removable)
                state = _State({name: _Arm(deque(maxlen=self.sample_capacity)) for name in self.recipes}, self.baseline_id, now, probe_budget_at=now)
                self._states[scope] = state
            self._states.move_to_end(scope)
            state.last_activity = now; state.sequence += 1
            if now-state.probe_budget_at >= self.feedback_ttl_s:
                state.probes = 0; state.probe_budget_at = now
            for arm in state.arms.values():
                while arm.samples and now-arm.samples[0][0] >= self.feedback_ttl_s:
                    arm.samples.popleft()
                if arm.last_attempt is not None and now-arm.last_attempt >= self.feedback_ttl_s:
                    arm.attempts = 0; arm.last_attempt = None
            busy_scope = any(entry[1] == scope for entry in self._pending.values())
            chosen, reason, probe = self.baseline_id, "baseline_warmup", False
            base = state.arms[self.baseline_id]
            if busy_scope:
                reason = "concurrent_request_baseline"
            elif len(base.samples) < self.min_observations:
                reason = "baseline_warmup" if base.attempts < self.baseline_attempts else "insufficient_baseline_evidence"
            else:
                candidate = next((name for name,arm in state.arms.items() if name != self.baseline_id and
                                  arm.attempts < self.probe_attempts), None)
                if candidate is not None and state.probes < self.max_probe_requests:
                    chosen, reason, probe = candidate, "bounded_probe", True
                    state.probes += 1
                else:
                    current = state.arms[state.incumbent]
                    if len(current.samples) < self.min_observations:
                        state.incumbent = self.baseline_id
                        current = base
                    current_rate = statistics.median(row[1] for row in current.samples)
                    eligible = [(percentile([row[1] for row in arm.samples], 25), name) for name,arm in state.arms.items()
                                if len(arm.samples) >= self.min_observations]
                    score, best = max(eligible, key=lambda row: (row[0], -list(self.recipes).index(row[1])))
                    if best != state.incumbent and score > current_rate*(1+self.min_gain) and state.sequence-state.last_switch >= self.min_hold_requests:
                        state.incumbent, state.last_switch = best, state.sequence
                        reason = "measured_gain"
                    else:
                        reason = "incumbent_hysteresis"
                    chosen = state.incumbent
            arm = state.arms[chosen]; arm.attempts += 1; arm.last_attempt = now
            decision = self._decision(self.recipes[chosen], reason, now, state.sequence, probe=probe, learning=not busy_scope)
            self._pending[decision.decision_id] = (decision, scope, state, context.max_new_tokens)
            return decision

    def _finish(self, decision, feedback, reason=None):
        if not isinstance(decision, PolicyDecision) or decision._issuer is not self._issuer:
            raise SpeculationPolicyError("decision was not issued by this policy")
        if feedback is None and reason is None or feedback is not None and not isinstance(feedback, RequestFeedback):
            raise SpeculationPolicyError("validated feedback object required")
        now = _seconds(self.clock(), "policy clock")
        raw = feedback.to_dict() if feedback is not None else None
        fingerprint = _digest({"feedback": raw, "discard_reason": reason})
        with self._lock:
            completed = self._completed.get(decision.decision_id)
            expired = completed is not None and completed[0] is decision and completed[1] is None
            if completed is not None:
                if completed[0] is not decision or completed[1] is not None and completed[1] != fingerprint:
                    raise SpeculationPolicyError("completed decision received different feedback")
                if completed[1] is not None:
                    return deepcopy(completed[2])
            pending = self._pending.get(decision.decision_id)
            if pending is not None and pending[0] is not decision:
                raise SpeculationPolicyError("decision object differs from the issued active decision")
            if decision._learning and not expired and (pending is None or pending[0] is not decision):
                raise SpeculationPolicyError("decision expired or is no longer active")
            if feedback is not None and feedback.mode != decision.recipe.mode:
                raise SpeculationPolicyError("actual feedback mode differs from chosen recipe")
            if pending is not None and feedback is not None and feedback.committed_tokens > pending[3]:
                raise SpeculationPolicyError("feedback committed more tokens than the request authorized")
            learned = False
            why = "expired_feedback" if expired else reason or decision.reason
            if pending is not None and decision._learning:
                state = pending[2]
                if now-decision.selected_at >= self.decision_ttl_s or self._states.get(pending[1]) is not state:
                    why = "expired_feedback"
                elif reason is not None:
                    why = reason
                elif not feedback.success or feedback.cancelled or feedback.replayed:
                    why = "failed_cancelled_or_replayed_request"
                elif feedback.committed_decode_tokens < self.min_decode_tokens or feedback.decode_and_drain_s <= 0 or not math.isfinite(feedback.committed_decode_tokens/feedback.decode_and_drain_s):
                    why = "insufficient_decode_measurement"
                else:
                    if feedback.observed_recipe != decision.recipe:
                        raise SpeculationPolicyError("learning requires actual feedback bound to the chosen approved recipe")
                    state.arms[decision.recipe.recipe_id].samples.append((now, feedback.committed_decode_tokens/feedback.decode_and_drain_s))
                    state.last_activity = now
                    learned, why = True, "actual_committed_decode_per_decode_and_drain_time"
            self._pending.pop(decision.decision_id, None)
            result = {"schema": FEEDBACK_SCHEMA, "decision": decision.to_dict(), "learned": learned,
                "reason": why, "feedback": raw, "score_definition": "committed_decode_tokens / (request_return_s - first_committed_token_s); prefill and receipt sweep separate",
                "acceptance_definition": "conditional acceptance uses only actually judged prediction trials; cancelled unjudged frames are cost, not rejection samples"}
            self._completed[decision.decision_id] = (decision, fingerprint, deepcopy(result))
            while len(self._completed) > self.completion_capacity:
                self._completed.popitem(last=False)
            return result

    def observe(self, decision, feedback):
        return self._finish(decision, feedback)

    def discard(self, decision, *, reason="request_failed"):
        if reason not in ("request_failed", "request_cancelled", "request_replayed", "request_expired", "receipt_validation_failed", "invalid_measurement"):
            raise SpeculationPolicyError("unknown request discard reason")
        return self._finish(decision, None, reason)

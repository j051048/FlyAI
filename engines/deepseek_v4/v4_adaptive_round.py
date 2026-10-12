"""Bounded, opt-in prefix selection at settled V4 pipeline refill boundaries.

The loaded drafter width, rollback depth, verification shape and accept rule are
fixed. Only candidates not yet submitted can be omitted. Calibrations are local
operator inputs, not remote execution proofs. Shadow observations cannot measure
the counterfactual speed of a prefix they never executed.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
import math
import time


SCHEMA = "v4-round-speculation/1"
CALIBRATION_SCHEMA = "v4-round-calibration/1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def service_options(plan, node_configs, lease_fences, *, depth, floor, lazy):
    """Bind local calibrations to the controller-selected ring and source bytes.

    Also usable by a calibration collector to construct the identity it records.
    This validates capacities, roles and lease assignments; it does not invent
    performance samples or infer an approval from an unsigned remote benchmark.
    """
    from v4_request_features import validate_selected_bindings, build_policy
    try:
        from shard.speculation_policy import SpeculationRecipe
    except ImportError:
        from speculation_policy import SpeculationRecipe
    from v4_observability import source_inventory
    validate_selected_bindings(plan, node_configs, lease_fences)
    recipe = SpeculationRecipe("baseline", "pipelined", depth, floor, lazy)
    checked = build_policy({"enabled": True, "baseline_id": "baseline", "recipes": [recipe.to_dict()]},
        recipe, node_configs, plan["cohort_id"], lease_fences=lease_fences, ring_id=plan["ring_id"])
    binding = {"capabilities": checked.binding(), "source_sha256": digest(source_inventory()), "lazy": lazy,
        "runtime_config_sha256_by_node": {node: digest(cfg) for node, cfg in sorted(node_configs.items())},
        "checkpoint_config_sha256": plan["model_cohort"]["config_sha256"],
        "stages": [{key: row[key] for key in ("node_id", "gpu_uuid", "signer_pubkey", "lo", "hi")}
                   for row in plan["stages"]]}
    return {"binding": binding, "loaded_block_size": checked.binding()["loaded_block_size"],
            "depth": depth, "floor": floor}


def calibration_binding(options):
    return digest({"identity": options["binding"], "loaded_block_size": options["loaded_block_size"],
                   "depth": options["depth"], "floor": options["floor"]})


def _integer(value, name, lo=0, hi=1_000_000_000):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return value


def _number(value, name, lo=0., hi=float("inf")):
    if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"invalid finite {name}")
    return float(value)


class RoundPolicy:
    """Per-attempt controller. No observations survive a failed/replayed request.

    Measured rate is committed tokens / wall time between refill boundaries,
    including carryover work. It is not a GPU phase attribution. Unknown expert
    traffic remains null; a byte constraint cannot pass on missing measurements.
    """

    def __init__(self, config, *, binding, loaded_block_size, depth, floor,
                 wall_clock=time.time, clock=time.perf_counter):
        if not isinstance(config, dict):
            raise ValueError("round speculation requires an object")
        allowed = {"enabled", "mode", "calibration", "ema_alpha", "min_gain",
                   "min_hold_windows", "sample_ttl_s", "max_records",
                   "max_dma_bytes_per_committed_token"}
        if set(config) - allowed or config.get("enabled") is not True:
            raise ValueError("unknown round speculation options or missing enabled=true")
        self.mode = config.get("mode", "shadow")
        if self.mode not in ("shadow", "enforce"):
            raise ValueError("round speculation mode must be shadow or enforce")
        self.block_size = _integer(loaded_block_size, "loaded block size", 1, 256)
        self.depth = _integer(depth, "fixed rollback depth", 2, 256)
        if floor != 1 or type(floor) is not int:
            raise ValueError("round prefix selection requires fixed refill floor=1")
        self.baseline = min(self.block_size, self.depth - 1)
        if not isinstance(binding, dict) or not binding:
            raise ValueError("round selection requires a pinned local binding")
        self.binding_sha256 = digest({"identity": binding, "loaded_block_size": self.block_size,
                                      "depth": self.depth, "floor": floor})
        self.alpha = _number(config.get("ema_alpha", .2), "EMA alpha", .001, 1.)
        self.min_gain = _number(config.get("min_gain", .08), "minimum gain", 0., 10.)
        self.min_hold = _integer(config.get("min_hold_windows", 4), "minimum hold windows", 1, 10000)
        self.sample_ttl = _number(config.get("sample_ttl_s", 60.), "sample TTL", .001, 86400.)
        self.max_records = _integer(config.get("max_records", 64), "record budget", 1, 1024)
        self.dma_limit = config.get("max_dma_bytes_per_committed_token")
        if self.dma_limit is not None:
            self.dma_limit = _number(self.dma_limit, "DMA bytes per committed token", 0.)
        calibration = config.get("calibration")
        calibration_keys = {
                "schema", "binding_sha256", "created_at", "expires_at", "profiles"}
        if isinstance(calibration, dict) and "evidence_sha256" in calibration:
            calibration_keys.add("evidence_sha256")
            value = calibration["evidence_sha256"]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid round evidence digest")
        if not isinstance(calibration, dict) or set(calibration) != calibration_keys:
            raise ValueError("complete local round calibration required")
        if calibration["schema"] != CALIBRATION_SCHEMA or calibration["binding_sha256"] != self.binding_sha256:
            raise ValueError("round calibration identity differs from loaded runtime")
        self.wall_clock, self.clock = wall_clock, clock
        self.created_at = _number(calibration["created_at"], "calibration creation")
        self.expires_at = _number(calibration["expires_at"], "calibration expiry")
        now = self.wall_clock()
        if (not self.created_at <= now or not 0 < self.expires_at-self.created_at <= 86400):
            raise ValueError("round calibration is future-dated or has an invalid one-day lifetime")
        rows = calibration["profiles"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 32:
            raise ValueError("bounded measured round profiles required")
        self.profiles = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"draft_prefix_cap", "committed_tokens", "elapsed_s", "expert_dma_bytes"}:
                raise ValueError("round profile must include actual work, time and nullable DMA")
            cap = _integer(row["draft_prefix_cap"], "approved prefix cap", 0, self.baseline)
            if cap in self.profiles:
                raise ValueError("duplicate round prefix profile")
            tokens = _integer(row["committed_tokens"], "calibration committed tokens", 2)
            elapsed = _number(row["elapsed_s"], "calibration elapsed seconds", 1e-9)
            dma = row["expert_dma_bytes"]
            if dma is not None:
                _integer(dma, "calibration DMA bytes", 0, 1 << 60)
            self.profiles[cap] = {"rate": tokens/elapsed,
                "dma_per_token": dma/tokens if dma is not None else None}
        if self.baseline not in self.profiles:
            raise ValueError("round calibration must include the unchanged baseline")
        self.calibration_sha256 = digest(calibration)
        self.config_sha256 = digest(config)
        self._runtime = {}
        self._records = deque(maxlen=self.max_records)
        self._window = None
        self._windows = self._switches = 0
        self._last_switch = -self.min_hold
        self._selected = self.baseline
        self._applied = self.baseline
        self._cost_bytes = 0
        self._cost_complete = True
        self._cost_frames = 0
        self._finished = False

    @property
    def changes_frames(self):
        return self.mode == "enforce"

    def validate_runtime(self, *, depth, floor, draft_block_limit):
        if (depth, floor, draft_block_limit) != (self.depth, 1, self.block_size):
            raise ValueError("round controller differs from the actual fixed runtime")
        if self._window is not None or self._finished:
            raise ValueError("round controller cannot be reused across attempts")

    def observe_reply_cost(self, cost):
        """Only complete all-stage costs may constrain an end-to-end decision.

        The current wire does not provide this measurement: absence is normal and
        records unknown, never zero. Tail-only telemetry is also incomplete.
        """
        self._cost_frames += 1
        if not isinstance(cost, dict) or cost.get("coverage") != "all_stages":
            self._cost_complete = False
            return
        value = cost.get("expert_dma_bytes")
        if type(value) is not int or value < 0 or value > 1 << 60:
            self._cost_complete = False
            return
        self._cost_bytes += value

    def _close(self, committed, now, *, truncated=False):
        if self._window is None:
            return
        count = committed - self._window["committed"]
        elapsed = now - self._window["at"]
        cap = self._window["applied_cap"]
        complete = self._cost_complete and self._cost_frames > 0
        dma = self._cost_bytes if complete else None
        measured = count > 0 and elapsed > 0 and math.isfinite(elapsed)
        # EOS/max-new has a partial window; terminal drain is separately reported
        # by the coordinator. Do not train a fully executed profile from it.
        learned = measured and not truncated and self._window["profile_complete"]
        if learned:
            previous = self._runtime.get(cap)
            if previous is None or now-previous["at"] > self.sample_ttl:
                rate = count/elapsed
            else:
                rate = self.alpha*(count/elapsed) + (1-self.alpha)*previous["rate"]
            self._runtime[cap] = {"rate": rate, "at": now,
                "dma_per_token": dma/count if dma is not None else None}
        self._windows += 1
        self._records.append({"kind": "window", "applied_cap": cap, "committed_tokens": count,
            "elapsed_s": elapsed, "expert_dma_bytes": dma, "cost_coverage": "all_stages" if complete else "unknown",
            "learned": learned, "truncated": truncated,
            "scope": "coordinator_refill_interval_including_carryover"})
        self._cost_bytes = self._cost_frames = 0
        self._cost_complete = True

    def _score(self, cap, now):
        row = self._runtime.get(cap)
        if row is None or now-row["at"] > self.sample_ttl:
            row = self.profiles[cap]
        dma = row["dma_per_token"]
        if self.dma_limit is not None and (dma is None or dma > self.dma_limit):
            return None
        return row["rate"]

    def refill(self, *, committed_tokens, available_drafts):
        if self._finished:
            raise ValueError("round controller has already completed")
        committed = _integer(committed_tokens, "actual committed count", 0)
        available = _integer(available_drafts, "available drafts", 0, self.block_size)
        now = self.clock()
        self._close(committed, now)
        # The first refill's incoming reply precedes the first measured window.
        # Later closes reset these counters; do the same for that initial case.
        self._cost_bytes = self._cost_frames = 0
        self._cost_complete = True
        valid = self.created_at <= self.wall_clock() < self.expires_at
        reason = "calibration_expired"
        if valid:
            scored = [(self._score(cap, now), cap) for cap in self.profiles]
            scored = [(score, cap) for score, cap in scored if score is not None]
            if scored:
                best_rate, best = max(scored, key=lambda item: (item[0], item[1] == self.baseline, -item[1]))
                current = self._score(self._selected, now)
                if (best != self._selected and (current is None or self._windows-self._last_switch >= self.min_hold)
                        and (current is None or best_rate > current*(1+self.min_gain))):
                    self._selected, self._last_switch = best, self._windows
                    self._switches += 1
                    reason = "measured_profile_gain"
                else:
                    reason = "hold_hysteresis"
            else:
                self._selected = self.baseline
                reason = "missing_or_over_budget_dma_measurement"
        else:
            self._selected = self.baseline
        wanted = self._selected if valid else self.baseline
        self._applied = wanted if self.changes_frames else self.baseline
        applied = min(self._applied, available)
        self._records.append({"kind": "decision", "selected_cap": wanted, "applied_cap": applied,
            "mode": self.mode, "reason": reason, "fixed_drafter_block_size": self.block_size})
        self._window = {"at": now, "committed": committed, "applied_cap": self._applied,
                        "profile_complete": available >= self._applied}
        return applied

    def finish(self, committed_tokens):
        if not self._finished:
            self._close(_integer(committed_tokens, "actual committed count"), self.clock(), truncated=True)
            self._finished = True
        return self.report()

    def report(self):
        return {"schema": SCHEMA, "mode": self.mode, "config_sha256": self.config_sha256,
            "calibration_sha256": self.calibration_sha256, "binding_sha256": self.binding_sha256,
            "baseline_cap": self.baseline, "loaded_block_size": self.block_size,
            "windows": self._windows, "switches": self._switches,
            "measurement_scope": "local_coordinator_observation_not_execution_proof",
            "expert_cost_missing_is_zero": False, "records": deepcopy(list(self._records))}

"""Request policy admission using exact controller-selected V4 calibrations.

The supplied node configurations and lease identities must come from the local
controller's verified selection, not HTTP request fields or a client JSON claim.
This module does not authenticate remote workers or certify their hardware.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
import time

try:
    from shard.speculation_policy import RequestPolicy, SpeculationContext, SpeculationPolicyError, SpeculationRecipe
except ImportError:  # flat deployed engine beside shared helpers
    from speculation_policy import RequestPolicy, SpeculationContext, SpeculationPolicyError, SpeculationRecipe

SCHEMA = "v4-request-capabilities/1"
# Native coordinators bind reset max_pos with this conservative lookahead.
# The real loop refills relative to c, never relative to the previous horizon.
SPECULATIVE_CONTEXT_MARGIN = 64
_HEX = re.compile(r"[a-f0-9]{64}\Z")
_RECIPE_FIELDS = {"recipe_id", "mode", "depth", "floor", "lazy"}
_LEARNING_FIELDS = {"min_observations", "baseline_attempts", "probe_attempts", "max_probe_requests", "min_gain",
    "min_hold_requests", "feedback_ttl_s", "decision_ttl_s", "max_states", "max_pending", "sample_capacity",
    "completion_capacity", "min_decode_tokens"}


class RequestFeatureError(SpeculationPolicyError):
    pass


def _text(value, name, maximum=256):
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise RequestFeatureError(f"invalid {name}")
    return value


def _integer(value, name, minimum=0, maximum=1_000_000_000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise RequestFeatureError(f"invalid {name}")
    return value


def _hash(value, name):
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise RequestFeatureError(f"invalid {name}")
    return value


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError) as error:
        raise RequestFeatureError("calibrated capabilities must contain finite JSON values") from error


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _capabilities(node_configs, cohort_id, lease_fences, ring_id):
    _hash(cohort_id, "cohort ID"); _text(ring_id, "ring ID")
    if not isinstance(node_configs, dict) or not 1 <= len(node_configs) <= 256:
        raise RequestFeatureError("enabled policy requires exact calibrated node configurations")
    if not isinstance(lease_fences, dict) or set(lease_fences) != set(node_configs):
        raise RequestFeatureError("enabled policy requires each selected node's verified local lease identity")
    detached = deepcopy(node_configs)
    leases, stages = {}, []
    for node_id, cfg in detached.items():
        _text(node_id, "node ID")
        lease = lease_fences[node_id]
        if not isinstance(lease, dict) or set(lease) != {"lease_id", "fencing_token"}:
            raise RequestFeatureError("verified lease binding requires exact lease_id/fencing_token")
        leases[node_id] = {"lease_id": _text(lease["lease_id"], "lease ID"),
                           "fencing_token": _integer(lease["fencing_token"], "lease fence", 1)}
        if not isinstance(cfg, dict) or not isinstance(cfg.get("args"), dict):
            raise RequestFeatureError("exact V4 runtime args are required")
        if cfg.get("cohort_id", cohort_id) != cohort_id:
            raise RequestFeatureError("selected configuration belongs to another model cohort")
        lo = _integer(cfg.get("lo"), "layer start")
        hi = _integer(cfg.get("hi"), "layer end", lo+1)
        layers = _integer(cfg["args"].get("n_layers"), "actual model depth", 1)
        context_limit = _integer(cfg["args"].get("max_seq_len"), "actual model context capacity", 1)
        cap = _integer(cfg.get("spec_depth"), "actual calibrated rollback capacity", 0, 256)
        if any(type(cfg.get(name)) is not bool for name in ("head", "tail", "dspark", "dspark_loaded")):
            raise RequestFeatureError("actual boundary and loaded draft roles must be explicit booleans")
        if hi > layers or cfg["head"] != (lo == 0) or cfg["tail"] != (hi == layers):
            raise RequestFeatureError("calibrated boundary roles differ from actual layer span")
        if cfg["dspark_loaded"] and (not cfg["tail"] or not cfg["dspark"]):
            raise RequestFeatureError("loaded MTP capability is only valid on the calibrated tail")
        if cfg["dspark_loaded"]:
            active_block = _integer(cfg.get("dspark_block_size"), "actual loaded draft block", 1, 256)
            trained_block = _integer(cfg["args"].get("dspark_block_size"), "trained draft block", 1, 256)
            _integer(cfg["args"].get("n_mtp_layers"), "actual MTP layer count", 1)
        else:
            if cfg.get("dspark_block_size") is not None:
                raise RequestFeatureError("an unloaded draft cannot claim an actual draft block size")
            active_block = trained_block = None
        stages.append({"node_id": node_id, "lo": lo, "hi": hi, "n_layers": layers,
                       "spec_depth": cap, "context_limit": context_limit, "loaded_block": active_block, "trained_block": trained_block})
    stages.sort(key=lambda row: row["lo"])
    cursor, layers = 0, stages[0]["n_layers"]
    for stage in stages:
        if stage["lo"] != cursor or stage["n_layers"] != layers:
            raise RequestFeatureError("selected calibrated nodes must exactly tile one model")
        cursor = stage["hi"]
    if cursor != layers:
        raise RequestFeatureError("selected calibrated nodes do not cover the whole model")
    body = {"schema": SCHEMA, "cohort_id": cohort_id, "ring_id": ring_id,
            "node_configs": detached, "lease_fences": leases}
    return body, stages


def capabilities_digest(node_configs, cohort_id, *, lease_fences, ring_id):
    """Full selected configuration/lease digest, never inferred from environment."""
    body, _ = _capabilities(node_configs, cohort_id, lease_fences, ring_id)
    return _digest(body)


def validate_selected_bindings(plan, node_configs, lease_fences):
    """Reject calibrations for another ring or another selected layer assignment."""
    if not isinstance(plan, dict) or not isinstance(plan.get("stages"), list):
        raise RequestFeatureError("feature admission requires the selected pipeline plan")
    stages = plan["stages"]
    selected = {row["node_id"] for row in stages}
    if (not isinstance(node_configs, dict) or set(node_configs) != selected
            or not isinstance(lease_fences, dict) or set(lease_fences) != selected):
        raise RequestFeatureError("feature calibration and leases must match every selected node")
    _capabilities(node_configs, plan.get("cohort_id"), lease_fences, plan.get("ring_id"))
    for row in stages:
        cfg = node_configs[row["node_id"]]
        if (any(cfg.get(name) != row.get(name) for name in ("lo", "hi", "head", "tail"))
                or cfg["args"].get("n_layers") != plan.get("n_layers")):
            raise RequestFeatureError("feature calibration differs from the selected layer assignment")
        expected = _hash(row.get("runtime_config_sha256"), "selected calibration hash")
        if _digest(cfg) != expected:
            raise RequestFeatureError("feature configuration differs from its selected calibration hash")


def _recipe(value):
    if isinstance(value, SpeculationRecipe):
        return value
    if not isinstance(value, dict) or set(value) - _RECIPE_FIELDS or not {"recipe_id", "mode"} <= set(value):
        raise RequestFeatureError("recipe permits only ID/mode/depth/floor/lazy; model and trained-block overrides are forbidden")
    try:
        return SpeculationRecipe(**value)
    except (SpeculationPolicyError, TypeError) as error:
        raise RequestFeatureError(str(error)) from error


class CalibratedRequestPolicy(RequestPolicy):
    def __init__(self, recipes, baseline_id, *, binding, clock, learning):
        self._capability_binding = deepcopy(binding)
        super().__init__(recipes, baseline_id, enabled=True, clock=clock, **learning)
        self._configuration["capabilities"] = deepcopy(binding)
        self.policy_sha256 = _digest(self._configuration)

    def binding(self):
        return deepcopy(self._capability_binding)

    def context(self, *, tenant_id, context_tokens, max_new_tokens, workload="unknown"):
        """tenant_id must be taken from authenticated service identity, not request JSON."""
        bound = self._capability_binding
        return SpeculationContext(tenant_id, bound["cohort_id"], bound["ring_id"], bound["ring_generation"],
            bound["capabilities_sha256"], context_tokens, max_new_tokens, workload)

    def choose(self, context):
        if not isinstance(context, SpeculationContext) or any(getattr(context, name) != self._capability_binding[key]
                for name, key in (("cohort_id", "cohort_id"), ("ring_id", "ring_id"), ("ring_generation", "ring_generation"),
                                  ("runtime_config_sha256", "capabilities_sha256"))):
            raise RequestFeatureError("request policy scope differs from its calibrated ring binding")
        total = context.context_tokens + context.max_new_tokens
        if total > self._capability_binding["context_limit"]:
            raise RequestFeatureError("request exceeds calibrated stage context capacity")
        if any(row.mode == "pipelined" for row in self.recipes.values()) and total+SPECULATIVE_CONTEXT_MARGIN > self._capability_binding["context_limit"]:
            # Match the existing gateway's conservative speculative margin. Never
            # teach a normal pipeline bucket with this capacity-limited request.
            greedy = next((row for row in self.recipes.values() if row.mode == "greedy"), None)
            if greedy is None:
                raise RequestFeatureError("request lacks speculative context margin and no greedy recipe is approved")
            now = self.clock()
            if type(now) not in (int, float) or not math.isfinite(now) or now < 0:
                raise RequestFeatureError("invalid policy clock")
            return self._decision(greedy, "context_capacity_greedy", float(now))
        return super().choose(context)


def build_policy(config, baseline_recipe, node_configs, cohort_id, *, lease_fences=None, ring_id=None, clock=time.monotonic):
    """Build from the LOCAL speculation_policy dict and controller-selected inputs.

    Missing configuration or enabled=False returns None before touching capacities,
    preserving the previous backend path. Enabling it fails closed on unknown
    calibration/lease evidence. No field here alters stage environment or weights.
    """
    if config is None:
        return None
    if not isinstance(config, dict) or type(config.get("enabled", False)) is not bool:
        raise RequestFeatureError("speculation_policy must be a local object with boolean enabled")
    if not config.get("enabled", False):
        return None
    if set(config) - {"enabled", "baseline_id", "recipes", "learning"}:
        raise RequestFeatureError("unknown enabled speculation policy fields")
    current = _recipe(baseline_recipe)
    values = config.get("recipes")
    if not isinstance(values, list) or not 1 <= len(values) <= 16:
        raise RequestFeatureError("enabled policy requires one to sixteen finite approved recipes")
    recipes = [_recipe(row) for row in values]
    by_id = {row.recipe_id: row for row in recipes}
    baseline_id = _text(config.get("baseline_id", current.recipe_id), "baseline recipe ID", 64)
    if baseline_id not in by_id or by_id[baseline_id].to_dict() != {**current.to_dict(), "recipe_id": baseline_id}:
        raise RequestFeatureError("policy baseline differs from the existing backend mode/depth/floor/lazy")
    body, stages = _capabilities(node_configs, cohort_id, lease_fences, ring_id)
    cap = min(row["spec_depth"] for row in stages)
    tail = stages[-1]
    for recipe in recipes:
        if recipe.mode == "pipelined" and (recipe.depth > cap or tail["loaded_block"] is None):
            raise RequestFeatureError("pipelined candidate requires loaded tail MTP and every stage's actual rollback capacity")
        if recipe.mode == "pipelined" and min(tail["loaded_block"], recipe.depth - 1) > SPECULATIVE_CONTEXT_MARGIN:
            raise RequestFeatureError("approved lookahead exceeds the native coordinator context margin")
    learning = config.get("learning", {})
    if not isinstance(learning, dict) or set(learning) - _LEARNING_FIELDS:
        raise RequestFeatureError("unknown policy learning settings")
    binding = {"schema": SCHEMA, "cohort_id": cohort_id, "ring_id": ring_id,
        "capabilities_sha256": _digest(body),
        "ring_generation": _digest({"cohort_id": cohort_id, "ring_id": ring_id, "lease_fences": body["lease_fences"]}),
        "rollback_capacity": cap, "context_limit": min(row["context_limit"] for row in stages),
        "trained_block_size": tail["trained_block"], "loaded_block_size": tail["loaded_block"]}
    try:
        return CalibratedRequestPolicy(recipes, baseline_id, binding=binding, clock=clock, learning=deepcopy(learning))
    except SpeculationPolicyError as error:
        raise RequestFeatureError(str(error)) from error

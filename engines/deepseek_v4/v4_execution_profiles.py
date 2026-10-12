"""Explicit deployment recipes, not GPU or performance qualification certificates.

The frozen benchmark and named launch profiles share these defaults. Expansion
must happen before importing the model modules which read environment constants.
An operator may tune budgets, but a named numerical recipe cannot silently turn
on approximation or change its expert/graph placement contract.
"""
from __future__ import annotations

from copy import deepcopy
import ast
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys

BASELINE_ENV = {
    "V4_FP8_AUTOTUNE": "0", "V4_DSPARK_RESULT_BUFFER": "0",
    "V4_PREFILL_EXPERT_PIPELINE": "0", "V4_PREFILL_EXPERT_DEPTH": "2", "V4_PREFILL_EXPERT_BATCH": "0",
    "V4_CONVERSATION_CACHE_MIB": "0", "V4_CONVERSATION_CACHE_GPU_MIB": "0",
    "V4_CONVERSATION_CACHE_ENTRIES": "4", "V4_CONVERSATION_CACHE_TTL_S": "300",
    "V4_KERNELS": "tilelang", "V4_DTYPE": "bfloat16", "V4_MAX_SEQ": "8192",
    "V4_HADAMARD": "auto", "V4_MAX_BATCH": "1", "V4_CUDA_GRAPH": "whole", "V4_GRAPH_MAX": "192",
    "V4_MOE_GROUPED": "1", "V4_MOE_DECODE": "1", "V4_MOE_MULTI": "1",
    "V4_MOE_MULTI_MAX": "32", "V4_MOE_IN_GRAPH": "1", "V4_FP8_GEMV": "1",
    "V4_FP8_SHARED": "1", "V4_DSPARK_FAST": "0", "V4_DSPARK_GRAPH": "0",
    "V4_DSPARK_MOE": "1", "V4_DSPARK_BLOCK": "8", "V4_DRAFT_TOP2": "0",
    "V4_PIPELINED_SPEC": "1", "V4_SPEC_DEPTH": "16", "V4_LAZY_DRAFT": "1",
    "V4_REFILL_FLOOR": "1", "V4_DSPARK_CONF_GATE": "0", "V4_DSPARK_CONF_THRESH": "0",
    "V4_DSPARK_CONF_MIN": "1", "V4_FP8_WIRE": "0", "V4_REF_SLIM": "0",
    "V4_REF_SLIM_NOQAT": "0", "V4_FAST_VERIFY": "0", "V4_FAST_VERIFY_MAX": "16",
    "V4_KEEPWARM": "0", "V4_KEEPWARM_MS": "150", "V4_TIMING": "0",
    "V4_TIMING_EVERY": "0", "V4_RUNTIME_METRICS": "1", "V4_LEVERS_STRICT": "1",
    "V4_EXPERT_PLACEMENT": "gpu", "V4_EXPERT_CACHE_SLOTS": "0",
    "V4_EXPERT_CACHE_MIB": "0", "V4_EXPERT_CACHE_RESERVE_MIB": "2048",
}
_PROFILES = {
    "resident-whole": {"V4_CUDA_GRAPH": "whole", "V4_MOE_IN_GRAPH": "1"},
    "resident-island": {"V4_CUDA_GRAPH": "island", "V4_MOE_IN_GRAPH": "0"},
    "ram-experts": {"V4_CUDA_GRAPH": "whole", "V4_MOE_IN_GRAPH": "0",
        "V4_EXPERT_PLACEMENT": "ram", "V4_EXPERT_CACHE_SLOTS": "2", "V4_DSPARK_MOE": "0"},
}
_INVALID_NAMES = {"V4_CHUNK_PREFILL": "V4_PREFILL_QUERY_CHUNK", "V4_PREFETCH": "V4_EXPERT_PREFETCH"}


@lru_cache(maxsize=1)
def _known_environment_names():
    # Read declarations without importing v4_levers: its strict flag is itself
    # frozen at import and must see the fully expanded recipe.
    tree = ast.parse(Path(__file__).with_name("v4_levers.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Lever" and node.args:
            if isinstance(node.args[0], ast.Constant): names.add(node.args[0].value)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "NON_LEVER_ENV" for t in node.targets):
            names.update(k.value for k in node.value.keys if isinstance(k, ast.Constant))
    return frozenset(names)


def baseline_environment():
    return dict(BASELINE_ENV)


def resolve_profile(name, overrides=None):
    if name not in _PROFILES:
        raise ValueError("execution profile must be " + ", ".join(sorted(_PROFILES)))
    overrides = dict(overrides or {})
    if any(not isinstance(k, str) or not isinstance(v, str) or "\x00" in k + v for k, v in overrides.items()):
        raise ValueError("profile environment requires string keys/values")
    for old, current in _INVALID_NAMES.items():
        if old in overrides:
            raise ValueError(f"unrecognized {old}; configure {current} explicitly")
    unknown = sorted(k for k in overrides if k.startswith("V4_") and k not in _known_environment_names())
    if unknown:
        raise ValueError("unrecognized profile environment names: " + ", ".join(unknown))
    selected = {**BASELINE_ENV, **_PROFILES[name]}
    protected = {"V4_REF_SLIM", "V4_REF_SLIM_NOQAT", "V4_FAST_VERIFY", "V4_FP8_WIRE",
        "V4_CUDA_GRAPH", "V4_MOE_IN_GRAPH", "V4_MOE_GROUPED", "V4_EXPERT_PLACEMENT", "V4_DSPARK_MOE"}
    for key in protected:
        if key in overrides and overrides[key] != selected[key]:
            raise ValueError(f"{key} conflicts with the {name} numerical/placement recipe; use an explicit experimental environment")
    if "V4_EXECUTION_PROFILE" in overrides and overrides["V4_EXECUTION_PROFILE"] != name:
        raise ValueError("conflicting execution profile identity")
    selected.update({k: v for k, v in overrides.items() if k.startswith("V4_")})
    selected["V4_EXECUTION_PROFILE"] = name
    return selected


def apply_profile():
    name = os.environ.get("V4_EXECUTION_PROFILE", "")
    if not name:
        if getattr(sys, "_flyai_v4_execution_profile_binding", None) is not None:
            raise RuntimeError("execution profile cannot be disabled after it was frozen")
        return None
    observed = {k: v for k, v in os.environ.items() if k.startswith("V4_")}
    selected = resolve_profile(name, observed)
    fixed = {key: value for key, value in selected.items() if key not in {"V4_DIR", "V4_DEV"}}
    fingerprint = hashlib.sha256(json.dumps(fixed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    frozen = getattr(sys, "_flyai_v4_execution_profile_binding", None)
    loaded = any(key.rsplit(".", 1)[-1].startswith("v4_") and module is not None and
                 key.rsplit(".", 1)[-1] != "v4_execution_profiles" and
                 any(name in _known_environment_names() for name in vars(module))
                 for key, module in tuple(sys.modules.items()))
    if (frozen is None and loaded) or frozen is not None and frozen != fingerprint:
        raise RuntimeError("execution profile cannot change after model modules were imported")
    os.environ.update(selected)
    # Shared across package/flat imports; pre-injecting a complete environment
    # after constants were read cannot manufacture this initial binding.
    sys._flyai_v4_execution_profile_binding = fingerprint
    return {"name": name, "environment_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest(), "qualification": "requires independent hardware and numerical evidence"}


def stage_profile_status(stage):
    name = os.environ.get("V4_EXECUTION_PROFILE", "")
    result = {"name": name or None, "performance_qualified": False,
        "scope": "observed decode capture smoke; not all-context or speed qualification"}
    graph_mode = getattr(stage, "_graph_mode", "off")
    result["graph_mode"] = graph_mode
    if graph_mode == "whole" and getattr(stage, "_block_graphs", None) is not None:
        graphs = sys.modules.get("v4_whole_layer_graph") or sys.modules.get("engines.deepseek_v4.v4_whole_layer_graph")
        if graphs is None:
            raise RuntimeError("whole stage lacks its actually loaded graph module")
        result["coverage"] = graphs.coverage_summary(stage)
    elif graph_mode == "island" and getattr(stage, "_block_graphs", None) is not None:
        runtime = sys.modules.get("v4_stage") or sys.modules.get("engines.deepseek_v4.v4_stage")
        if runtime is None:
            raise RuntimeError("island stage lacks its actually loaded module")
        result["coverage"] = runtime.island_coverage_summary(stage)
    return result


def expanded_environment(environment):
    environment = dict(environment)
    name = environment.get("V4_EXECUTION_PROFILE", "")
    return {**environment, **resolve_profile(name, environment)} if name else environment


def transport_buffer_reservation(environment=None):
    environment = os.environ if environment is None else environment
    size = int(environment.get("V4_TRANSPORT_BUFFER_MIB", "0"))
    slots = int(environment.get("V4_TRANSPORT_BUFFER_SLOTS", "2"))
    pinned = environment.get("V4_TRANSPORT_PINNED", "0")
    if not 0 <= size <= 512 or not 1 <= slots <= 64 or pinned not in ("0", "1"):
        raise ValueError("invalid bounded socket buffer reservation")
    return {"host_reserved_bytes": size * 1024**2,
            "pinned_reserved_bytes": size * 1024**2 if pinned == "1" else 0,
            "slots": slots, "scope": "process-owned transport pool; per-stage process calibration"}


def require_profile_warmup(receipts, expected_by_node):
    """Named resident-whole requires actual signed per-layer capture observation.

    The caller must already have checked the receipt signatures and assignments.
    This smoke does not certify long contexts or permit bypassing the GPU gates.
    """
    expected = {node: name for node, name in expected_by_node.items() if name}
    seen = set()
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise ValueError("profile warmup requires validated receipt objects")
        observation = receipt.get("runtime_observation", {})
        if not isinstance(observation, dict):
            raise ValueError("profile warmup runtime observation must be an object")
        node = observation.get("node_id")
        if node not in expected:
            continue
        status = observation.get("optimizations", {}).get("execution_profile", {})
        if not isinstance(status, dict):
            raise ValueError("profile warmup capture observation must be an object")
        if status.get("name") != expected[node]:
            raise ValueError("signed warmup execution profile differs from the calibrated node")
        expected_mode = "island" if expected[node] == "resident-island" else "whole"
        if status.get("graph_mode") != expected_mode or observation.get("graph_mode") != expected_mode:
            raise ValueError("named profile warmup observed another graph execution mode")
        coverage = status.get("coverage", {})
        if not isinstance(coverage, dict):
            raise ValueError("profile warmup requires bounded capture coverage")
        lo, hi = receipt.get("layer_start"), receipt.get("layer_end")
        if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= 4096:
            raise ValueError("invalid assigned profile layer span")
        count = hi - lo
        layers = coverage.get("layers", [])
        if (not isinstance(layers, list) or len(layers) != count or
            any(not isinstance(row, dict) or type(row.get("layer")) is not int or
                type(row.get("captured_graphs")) is not int or type(row.get("eager")) is not bool
                for row in layers) or
            any(type(coverage.get(field)) is not int for field in ("captured_layers", "refused_layers", "undecided_layers"))):
            raise ValueError("malformed profile capture coverage")
        schema = "flyai-v4-island-coverage/1" if expected_mode == "island" else "flyai-v4-whole-coverage/1"
        if (coverage.get("schema") != schema or
                coverage.get("captured_layers") != count or coverage.get("refused_layers") != 0 or
                coverage.get("undecided_layers") != 0 or
                not isinstance(layers, list) or len(layers) != count or
                {row.get("layer") for row in layers} != set(range(lo, hi)) or
                any(not isinstance(row, dict) or row.get("eager") or not row.get("captured_graphs") or
                    expected_mode == "island" and row.get("captured_graphs") != 3 or
                    expected[node] == "resident-whole" and row.get("moe_in_graph") is not True for row in layers)):
            raise ValueError("named profile warmup did not observe every assigned layer's required captures")
        seen.add(node)
    if seen != set(expected):
        raise ValueError("named profile warmup lacks assigned node observations")
    return deepcopy(expected)

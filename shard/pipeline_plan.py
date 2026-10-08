"""One validated layer/route manifest for launchers and engine handshakes.

This module describes placement; it neither chooses hardware nor executes remote
commands. Runtime and resource measurements remain separate signed contracts.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

SCHEMA = "shard-pipeline-plan/1"


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def model_layers(config):
    value = config.get("num_hidden_layers", config.get("n_layers"))
    return _integer(value, "model layer count", 1)


def endpoint(value):
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ValueError("endpoint must be host:port")
    try:
        parsed = urlsplit("//" + value)
        if not parsed.hostname or parsed.port is None or not 1 <= parsed.port <= 65535:
            raise ValueError
        if parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            raise ValueError
    except ValueError:
        raise ValueError("endpoint must be host:port (IPv6 addresses use brackets)") from None
    return value


def parse_split(split, n_layers, nstages):
    """Return contiguous half-open spans, including all model layers exactly once."""
    _integer(n_layers, "n_layers", 1)
    _integer(nstages, "nstages", 1)
    if nstages > n_layers:
        raise ValueError("nstages cannot exceed model layers")
    if split is None or split == "":
        return [(i * n_layers // nstages, (i + 1) * n_layers // nstages)
                for i in range(nstages)]
    if isinstance(split, str):
        if not re.fullmatch(r"\s*[0-9]+\s*(,\s*[0-9]+\s*)*", split):
            raise ValueError("split must be comma-separated positive layer counts")
        counts = [int(part) for part in split.split(",")]
    elif isinstance(split, (list, tuple)):
        counts = list(split)
    else:
        raise ValueError("split must be a string or list of layer counts")
    if len(counts) != nstages:
        raise ValueError("split length must equal nstages")
    for count in counts:
        _integer(count, "split layer count", 1)
    if sum(counts) != n_layers:
        raise ValueError(f"split totals {sum(counts)} layers, model requires {n_layers}")
    spans, cursor = [], 0
    for count in counts:
        spans.append((cursor, cursor + count))
        cursor += count
    return spans


def validate_plan(value, config=None):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError(f"{SCHEMA} manifest required")
    # Detach caller data and reject non-finite/non-JSON configuration.
    plan = json.loads(json.dumps(value, allow_nan=False))
    if not isinstance(plan.get("ring_id"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", plan["ring_id"]):
        raise ValueError("ring_id must contain 1..128 safe identifier characters")
    if not isinstance(plan.get("cohort_id"), str) or not re.fullmatch(r"[0-9a-f]{64}", plan["cohort_id"]):
        raise ValueError("cohort_id must be a lowercase SHA-256")
    layers = _integer(plan.get("n_layers"), "n_layers", 1)
    if config is not None and model_layers(config) != layers:
        raise ValueError("deployment plan layer count differs from model config")
    stages = plan.get("stages")
    if not isinstance(stages, list) or not stages or len(stages) > 256:
        raise ValueError("1..256 stages required")
    if type(plan.get("nstages", len(stages))) is not int or plan.get("nstages", len(stages)) != len(stages):
        raise ValueError("nstages differs from stage list")
    plan["nstages"] = len(stages)
    cursor, nodes, devices = 0, set(), set()
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or stage.get("index") != index or type(stage.get("index")) is not int:
            raise ValueError("stage indices must be ordered and contiguous")
        lo, hi = stage.get("lo"), stage.get("hi")
        _integer(lo, "lo"); _integer(hi, "hi", 1)
        if lo != cursor or hi <= lo or hi > layers:
            raise ValueError("layer blocks must tile the model without gap or overlap")
        cursor = hi
        for role, expected in (("head", index == 0), ("tail", index == len(stages) - 1)):
            if type(stage.get(role)) is not bool or stage[role] != expected:
                raise ValueError("stage roles differ from execution order")
        node = stage.get("node_id")
        if not isinstance(node, str) or not node or len(node) > 512 or node in nodes:
            raise ValueError("distinct node identities required")
        nodes.add(node)
        signer = stage.get("signer_pubkey")
        if signer is not None:
            try:
                if len(base64.b64decode(signer, validate=True)) != 32:
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError("signer_pubkey must be a base64 Ed25519 public key") from None
        gpu = stage.get("gpu_uuid")
        if gpu is not None:
            if not isinstance(gpu, str) or not gpu or gpu.lower() in devices:
                raise ValueError("distinct GPU UUIDs required")
            devices.add(gpu.lower())
        endpoint(stage.get("endpoint"))
        if stage.get("context_limit") is not None:
            _integer(stage["context_limit"], "stage context_limit", 1)
        if index < len(stages) - 1:
            endpoint(stage.get("next_endpoint"))
        elif stage.get("next_endpoint") is not None:
            raise ValueError("tail cannot forward to another stage")
    if cursor != layers:
        raise ValueError("stage blocks do not cover the complete model")
    coord = plan.get("coordinator")
    if not isinstance(coord, dict):
        raise ValueError("coordinator routes required")
    endpoint(coord.get("head")); endpoint(coord.get("tail"))
    if plan.get("execution") is not None:
        cap = _integer(plan["execution"].get("max_context"), "execution max_context", 1)
        known = [s["context_limit"] for s in stages if s.get("context_limit") is not None]
        if known and cap > min(known):
            raise ValueError("execution context exceeds a selected stage capacity")
    if coord.get("signer_pubkey") is not None:
        try:
            if len(base64.b64decode(coord["signer_pubkey"], validate=True)) != 32:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("coordinator signer must be a base64 Ed25519 public key") from None
    cohort = plan.get("model_cohort")
    if cohort is not None:
        from .offers import ModelCohort
        bound = ModelCohort.from_dict(cohort)
        if bound.cohort_id != plan["cohort_id"] or bound.n_layers != layers:
            raise ValueError("model cohort differs from deployment plan")
    return plan


def load_plan(path, config=None):
    target = Path(path)
    if target.is_symlink() or target.stat().st_size > (1 << 20):
        raise ValueError("invalid deployment manifest file")
    return validate_plan(json.loads(target.read_text(encoding="utf-8-sig")), config)


def stage_spec(plan, index):
    plan = validate_plan(plan)
    _integer(index, "stage")
    if index >= len(plan["stages"]):
        raise ValueError("stage index outside deployment")
    return copy.deepcopy(plan["stages"][index])


def resolve_bounds(config, stage, nstages, split=None, lo=None, hi=None, plan=None):
    layers = model_layers(config)
    _integer(stage, "stage"); _integer(nstages, "nstages", 1)
    if stage >= nstages:
        raise ValueError("stage index outside nstages")
    explicit = lo is not None and lo != -1 or hi is not None and hi != -1
    if plan is not None:
        bound = validate_plan(plan, config)
        if bound["nstages"] != nstages:
            raise ValueError("nstages differs from deployment plan")
        spec = bound["stages"][stage]
        span = spec["lo"], spec["hi"]
        if split not in (None, "") and parse_split(split, layers, nstages)[stage] != span:
            raise ValueError("split differs from deployment plan")
        if explicit and (lo, hi) != span:
            raise ValueError("lo/hi differ from deployment plan")
        return span
    if explicit:
        if split not in (None, ""):
            raise ValueError("choose split or explicit lo/hi")
        _integer(lo, "lo"); _integer(hi, "hi", 1)
        if not lo < hi <= layers or (stage == 0 and lo != 0) or (stage == nstages - 1 and hi != layers):
            raise ValueError("explicit bounds violate model or boundary role")
        return lo, hi
    return parse_split(split, layers, nstages)[stage]


def build_plan(config, *, ring_id, cohort_id, endpoints, split=None, node_ids=None,
               gpu_uuids=None, head=None, tail=None, model_cohort=None):
    stages = len(endpoints)
    ranges = parse_split(split, model_layers(config), stages)
    if node_ids is None:
        node_ids = [f"stage-{i}" for i in range(stages)]
    if gpu_uuids is None:
        gpu_uuids = [None] * stages
    if len(node_ids) != stages or len(gpu_uuids) != stages:
        raise ValueError("node/GPU lists must match stage count")
    plan = {"schema": SCHEMA, "ring_id": ring_id, "cohort_id": cohort_id,
            "n_layers": model_layers(config), "nstages": stages,
            "coordinator": {"head": head or endpoints[0], "tail": tail or endpoints[-1]},
            "stages": [{"index": i, "node_id": node_ids[i], "gpu_uuid": gpu_uuids[i],
                        "lo": lo, "hi": hi, "head": i == 0, "tail": i == stages - 1,
                        "endpoint": endpoints[i],
                        "next_endpoint": endpoints[i + 1] if i + 1 < stages else None}
                       for i, (lo, hi) in enumerate(ranges)]}
    if model_cohort is not None:
        plan["model_cohort"] = model_cohort
    return validate_plan(plan, config)


def plan_digest(plan):
    return hashlib.sha256(json.dumps(validate_plan(plan), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig")) if args.config else None
    plan = load_plan(args.plan, config)
    print(json.dumps({"valid": True, "plan_sha256": plan_digest(plan), "plan": plan}))


if __name__ == "__main__":
    main()

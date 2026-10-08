"""Generate a strict GPT-OSS pipeline plan from verified model identity and public node metadata.

No GPU/model load, key generation, SSH or route probing. The same configuration
file can be passed to deploy_oss.py --nodes/--config; private deployment fields
are never copied into the public plan. Generation is not resource calibration.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shard.download_inventory import verify_inventory
from shard.gpt_oss_contract import validate_supported_cohort
from shard.offers import ModelCohort
from shard.pipeline_plan import build_plan, endpoint, load_plan, parse_split, plan_digest, validate_plan

GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
STAGE_FIELDS = {"node_id", "gpu_uuid", "signer_pubkey", "endpoint", "next_endpoint", "context_limit"}
COORDINATOR_FIELDS = {"node_id", "head", "tail", "signer_pubkey"}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def read_json(path):
    target = Path(path)
    if target.stat().st_size > 1024**2:
        raise ValueError("plan input exceeds 1 MiB")
    return json.loads(target.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite JSON input")))


def _public_key(value, where):
    try:
        if not isinstance(value, str):
            raise ValueError
        raw = base64.b64decode(value, validate=True)
        if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != value:
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{where} must contain a canonical base64 Ed25519 public key") from exc
    return raw


def _text(value, where):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"{where} must be a nonempty string")
    return value


def _positive_int(value, where, maximum=None):
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ValueError(f"{where} must be a positive integer" + (f" <= {maximum}" if maximum else ""))
    return value


def generate_plan(model_dir, cohort, deployment, *, ring_id, split=None, verify_files=False):
    """Return a validated public plan; supplied identities are declarations, not attestation.

    Model identity is checked against the real immutable local download inventory
    and raw config. Cheap verification keeps weight integrity explicitly unknown;
    deploy/startup still performs its complete rehash and native loading checks.
    """
    if not isinstance(cohort, (ModelCohort, dict)):
        raise ValueError("cohort must be a complete descriptor object from gpt_oss_manifest.py")
    cohort = cohort if isinstance(cohort, ModelCohort) else ModelCohort.from_dict(cohort)
    inventory = verify_inventory(model_dir, expected_checkpoint_id=cohort.checkpoint_id,
        expected_repo=cohort.model_id, verify_files=verify_files)
    config_path = Path(model_dir) / "config.json"
    raw_config = config_path.read_bytes()
    if (hashlib.sha256(raw_config).hexdigest() != cohort.config_sha256 or
            inventory["config_sha256"] != cohort.config_sha256 or
            inventory["manifest_sha256"] != cohort.manifest_sha256):
        raise ValueError("model config/download inventory differs from supplied cohort")
    config = json.loads(raw_config, object_pairs_hook=_unique)
    validate_supported_cohort(cohort, config)
    if not isinstance(deployment, dict) or not isinstance(deployment.get("nodes"), dict):
        raise ValueError("configuration requires a nodes map")
    rows, coordinator = deployment.get("stages"), deployment.get("coordinator")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
        raise ValueError("configuration requires 1..256 ordered stages")
    if not isinstance(coordinator, dict) or set(coordinator) - COORDINATOR_FIELDS:
        raise ValueError("coordinator supports only node_id/head/tail/signer_pubkey")
    _public_key(coordinator.get("signer_pubkey"), "coordinator signer_pubkey")
    endpoint(coordinator.get("head")); endpoint(coordinator.get("tail"))
    identities, devices, keys = set(), set(), set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) - STAGE_FIELDS:
            raise ValueError("stage contains unknown planning fields; runtime/resource hashes are not generated here")
        node_id = _text(row.get("node_id"), "stage node_id")
        uuid = row.get("gpu_uuid")
        if not isinstance(uuid, str) or GPU_UUID.fullmatch(uuid) is None:
            raise ValueError("stage gpu_uuid must be a complete NVIDIA GPU UUID from the actual node")
        key = _public_key(row.get("signer_pubkey"), "stage signer_pubkey")
        if node_id in identities or uuid.casefold() in devices or key in keys:
            raise ValueError("stages require distinct node IDs, GPU UUIDs and receipt signers")
        identities.add(node_id); devices.add(uuid.casefold()); keys.add(key)
        endpoint(row.get("endpoint"))
        if index < len(rows) - 1:
            endpoint(row.get("next_endpoint"))  # sender-local: never infer from another node's localhost.
        elif row.get("next_endpoint") is not None:
            raise ValueError("tail must not have a next_endpoint")
        if row.get("context_limit") is not None:
            _positive_int(row["context_limit"], "stage context_limit")
    coordinator_id = coordinator.get("node_id", rows[0]["node_id"])
    _text(coordinator_id, "coordinator node_id")
    for node_id in identities | {coordinator_id}:
        node = deployment["nodes"].get(node_id)
        if not isinstance(node, dict):
            raise ValueError("each selected stage/coordinator needs a nodes deployment entry")
        for field in ("workspace", "model", "node_key", "ssh_target"):
            _text(node.get(field), f"deployment node {field}")
        if "listen_port" in node:
            _positive_int(node["listen_port"], "node-local listen_port", 65535)
        if "max_context" in node:
            _positive_int(node["max_context"], "node max_context")
    selected_split = deployment.get("split") if split is None else split
    if split is not None and deployment.get("split") is not None and (
            parse_split(split, cohort.n_layers, len(rows)) != parse_split(deployment["split"], cohort.n_layers, len(rows))):
        raise ValueError("CLI split differs from configuration split")
    result = build_plan(config, ring_id=ring_id, cohort_id=cohort.cohort_id,
        endpoints=[row["endpoint"] for row in rows], split=selected_split,
        node_ids=[row["node_id"] for row in rows], gpu_uuids=[row["gpu_uuid"] for row in rows],
        head=coordinator["head"], tail=coordinator["tail"], model_cohort=cohort.to_dict())
    context_limits = []
    for stage, row in zip(result["stages"], rows):
        stage["signer_pubkey"] = row["signer_pubkey"]
        stage["next_endpoint"] = row.get("next_endpoint")
        node_limit = deployment["nodes"][row["node_id"]].get("max_context", 8192)
        stage["context_limit"] = min(node_limit, row.get("context_limit") or node_limit)
        context_limits.append(stage["context_limit"])
    result["coordinator"].update(node_id=coordinator_id, signer_pubkey=coordinator["signer_pubkey"])
    context_limits.append(deployment["nodes"][coordinator_id].get("max_context", 8192))
    # Configured software bounds (including deploy's existing default), never a
    # fabricated GPU-memory measurement or a resource configuration fingerprint.
    result["execution"] = {"max_context": min(context_limits)}
    return validate_plan(result, config)


def write_plan(path, plan):
    """Idempotent for the same plan; never overwrite a different plan or symlink."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or load_plan(path) != plan:
            raise ValueError("output already contains another plan; use a fresh ring ID/output path")
        return False
    data = json.dumps(plan, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        output.write(data)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local model directory with actual config and .shard-download.json")
    parser.add_argument("--cohort", required=True, help="full cohort JSON from gpt_oss_manifest.py")
    parser.add_argument("--config", "--nodes", dest="config", required=True, help="deployment JSON: nodes, ordered stages, coordinator; same file as deploy_oss")
    parser.add_argument("--ring-id", required=True)
    parser.add_argument("--split", help="positive layer counts in stage order; otherwise configuration split or even split")
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify-files", action="store_true", help="also rehash all local weight files; startup still performs its own verification")
    args = parser.parse_args(argv)
    try:
        plan = generate_plan(args.model, read_json(args.cohort), read_json(args.config),
            ring_id=args.ring_id, split=args.split, verify_files=args.verify_files)
        written = write_plan(args.out, plan)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({"plan": str(Path(args.out).resolve()), "written": written,
        "plan_sha256": plan_digest(plan), "cohort_id": plan["cohort_id"], "n_layers": plan["n_layers"],
        "nstages": plan["nstages"], "weight_payload_rehashed": args.verify_files,
        "scope": "validated configuration only; no key ownership, route, GPU fit or readiness claim"}))
    return 0


if __name__ == "__main__":
    main()

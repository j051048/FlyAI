"""Validate a concrete multi-GPU deployment against measured per-stage contracts.

This is a read-only deployment gate. It accepts co-location, sums independently
owned host pools, checks GPU/port identity, and binds planned placement to the
configuration actually sent to the runtime. No shell execution or cloud rental.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

try:
    from shard.resources import NodeResources, PlacementRequirements, evaluate_fit
except ImportError:
    from resources import NodeResources, PlacementRequirements, evaluate_fit


def config_digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def deployment_policies(bundle):
    """Admission and token visibility are independent of authenticated transport.

    The old trusted_nodes label never verified an operator. Keep old bundles
    readable, without treating that label as an assertion of verified trust.
    """
    legacy = bundle.get("trust_mode")
    if legacy not in (None, "open", "trusted_nodes", "sealed_ids"):
        raise ValueError("unsupported legacy trust_mode")
    admission = bundle.get("registration_policy", "open")
    if admission != "open":
        raise ValueError("registration_policy must be open")
    default = "sealed_ids" if legacy == "sealed_ids" else "plain_ids"
    visibility = bundle.get("token_privacy", default)
    if visibility not in ("plain_ids", "sealed_ids"):
        raise ValueError("token_privacy must be plain_ids or sealed_ids")
    if legacy == "sealed_ids" and visibility != "sealed_ids":
        raise ValueError("conflicting legacy and explicit token privacy")
    return {"registration_policy": admission, "token_privacy": visibility,
            "operator_trust_verified": False}


def _fresh(stamp, now, max_age_s):
    measured = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if measured.tzinfo is None:
        raise ValueError("measurement requires a timezone")
    age = (now - measured).total_seconds()
    return -30 <= age <= max_age_s


def check_deployment(bundle, *, now=None, max_age_s=3600):
    now = now or datetime.now(timezone.utc)
    if not math.isfinite(max_age_s) or max_age_s <= 0:
        raise ValueError("max_age_s must be finite and positive")
    errors, warnings, rows = [], [], []
    if not isinstance(bundle, dict) or bundle.get("schema") != "shard-deployment/1":
        return {"ready": False, "errors": ["unsupported deployment schema"], "stages": []}
    stages = bundle.get("stages", [])
    if not isinstance(stages, list) or not stages:
        return {"ready": False, "errors": ["nonempty concrete stage assignment required"], "stages": []}
    uuids, keys, host_ports, node_ids = set(), set(), set(), set()
    cursor, hosts = 0, {}
    for index, stage in enumerate(stages):
        try:
            for key in ("node_id", "host_id", "gpu_uuid", "signer_pubkey"):
                if not isinstance(stage.get(key), str) or not stage[key].strip():
                    raise ValueError(f"stage {index}: missing {key}")
            if stage["gpu_uuid"] in uuids or stage["signer_pubkey"] in keys or stage["node_id"] in node_ids:
                raise ValueError(f"stage {index}: GPU, signing key or node identity reused")
            uuids.add(stage["gpu_uuid"]); keys.add(stage["signer_pubkey"]); node_ids.add(stage["node_id"])
            import base64
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            Ed25519PublicKey.from_public_bytes(base64.b64decode(stage["signer_pubkey"], validate=True))
            lo, hi = stage["lo"], stage["hi"]
            if type(lo) is not int or type(hi) is not int or lo != cursor or hi <= lo:
                raise ValueError(f"stage {index}: layer ranges must tile in execution order")
            cursor = hi
            port = stage["port"]
            if type(port) is not int or not 1 <= port <= 65535:
                raise ValueError(f"stage {index}: invalid engine port")
            hp = stage["host_id"], port
            if hp in host_ports:
                raise ValueError(f"stage {index}: duplicate engine port on one host")
            host_ports.add(hp)
            req = PlacementRequirements.from_dict(stage["requirements"])
            if (req.layer_start, req.layer_end, req.model_id) != (lo, hi, bundle["model_id"]):
                raise ValueError(f"stage {index}: runtime resource contract differs from assignment")
            if req.provenance.node_id != stage["node_id"]:
                raise ValueError(f"stage {index}: calibration belongs to another node")
            if req.provenance.checkpoint_id != bundle["checkpoint_id"]:
                raise ValueError(f"stage {index}: checkpoint calibration differs")
            runtime_config = stage["runtime_config"]
            if config_digest(runtime_config) != req.provenance.runtime_config_sha256:
                raise ValueError(f"stage {index}: runtime configuration differs from measured calibration")
            if ((runtime_config.get("lo"), runtime_config.get("hi")) != (lo, hi) or
                    runtime_config.get("head") is not (index == 0) or
                    runtime_config.get("tail") is not (index == len(stages) - 1) or
                    runtime_config.get("args", {}).get("n_layers") != bundle.get("layer_count")):
                raise ValueError(f"stage {index}: measured range/model/boundary roles differ from assignment")
            if runtime_config.get("dspark") is not bool(stage.get("dspark", False)):
                raise ValueError(f"stage {index}: measured DSpark role differs from launch")
            if not _fresh(req.provenance.measured_at, now, max_age_s):
                raise ValueError(f"stage {index}: runtime resource calibration is stale")
            env = stage["env"]
            if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                for k, v in env.items()):
                raise ValueError(f"stage {index}: launch environment must be a string mapping")
            if "V4_TOKEN_PRIVACY_KEY" in env:
                raise ValueError(f"stage {index}: private token secrets must not enter deployment metadata")
            measured_env = runtime_config.get("environment")
            if not isinstance(measured_env, dict):
                raise ValueError(f"stage {index}: measured public runtime environment absent")
            # Only checkpoint location and process-local device ordinal may move.
            # Model/numerics/graph/KV/cache/privacy settings must match the calibration.
            relocatable = {"V4_DIR", "V4_DEV"}
            actual_flags = {k: v for k, v in env.items() if k.startswith("V4_") and k not in relocatable}
            measured_flags = {k: v for k, v in measured_env.items() if k.startswith("V4_") and k not in relocatable}
            if actual_flags != measured_flags:
                raise ValueError(f"stage {index}: launch flags differ from measured runtime environment")
            placement = stage.get("placement", "gpu")
            if placement not in ("gpu", "ram") or env.get("V4_EXPERT_PLACEMENT", "gpu") != placement:
                raise ValueError(f"stage {index}: planned placement was not propagated to runtime env")
            if runtime_config.get("expert_placement") != placement:
                raise ValueError(f"stage {index}: measured runtime used a different expert placement")
            cap = NodeResources(**stage["capacity"])
            if not _fresh(stage["capacity_measured_at"], now, max_age_s):
                raise ValueError(f"stage {index}: free capacity measurement is stale")
            fit = evaluate_fit(req, cap)
            if not fit["fits"]:
                errors.append(f"stage {index}: {fit['status']} resources {fit['insufficient'] + fit['unknown']}")
            rows.append({"node_id": stage["node_id"], "fit": fit})
            # A host is always the shared resource domain. VM partitions require an
            # independently verified reservation; merely changing a label is insufficient.
            h = hosts.setdefault(stage["host_id"], {"ram": 0, "pinned": 0, "ram_cap": [], "pin_cap": []})
            h["ram"] += req.host.peak_bytes
            h["pinned"] += req.host.pinned_bytes
            h["ram_cap"].append(cap.available_ram_bytes)
            h["pin_cap"].append(cap.pinnable_ram_bytes)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append(str(exc))
    if cursor != bundle.get("layer_count"):
        errors.append("assignment does not cover the complete declared model")
    if bundle.get("model_id", "").startswith("deepseek") and cursor == 43 and stages[-1].get("lo", 43) > 40:
        errors.append("DSpark tail must own target layers 40, 41 and 42")
    for host_id, row in hosts.items():
        for resource, capacities in (("ram", row["ram_cap"]), ("pinned", row["pin_cap"])):
            if row[resource] and (None in capacities or row[resource] > min(capacities)):
                errors.append(f"host {host_id}: aggregate {resource} allocations exceed measured shared budget")
        probe = bundle.get("host_io", {}).get(host_id)
        if row["pinned"]:
            if not probe or probe.get("schema") != "shard-host-io-probe/1" or probe.get("host_id") != host_id:
                errors.append(f"host {host_id}: simultaneous GPU I/O and pinned-allocation probe required")
            else:
                try:
                    try:
                        from shard.host_probe import verify_report
                    except ImportError:
                        from host_probe import verify_report
                    verify_report(probe)
                    if not _fresh(probe["measured_at"], now, max_age_s):
                        raise ValueError("stale I/O probe")
                    if (probe.get("pin_budget_verified") is not True or
                            type(probe.get("observed_pinned_allocation_bytes")) is not int or
                            probe["observed_pinned_allocation_bytes"] < row["pinned"]):
                        raise ValueError("whole-host pinned requirement was not actually allocated")
                    offered = {s["gpu_uuid"] for s in stages if s.get("host_id") == host_id}
                    observed = {s["gpu_uuid"] for s in probe["devices"]}
                    if observed != offered or probe.get("concurrent") is not True:
                        raise ValueError("I/O measurement must cover all assigned GPUs simultaneously")
                    bw = probe["aggregate_h2d_gb_s"]
                    if type(bw) not in (int, float) or not math.isfinite(bw) or bw <= 0:
                        raise ValueError("invalid measured aggregate H2D bandwidth")
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append(f"host {host_id}: {exc}")
    try:
        policies = deployment_policies(bundle)
    except ValueError as exc:
        errors.append(str(exc))
        policies = {}
    if policies.get("token_privacy") == "sealed_ids":
        if any(s.get("env", {}).get("V4_SEALED_IDS") != "1" for s in stages):
            errors.append("sealed_ids must be enabled throughout the ring")
        warnings.append("sealed token IDs do not conceal activations; hash layers and tail remain trusted")
    else:
        if any(s.get("env", {}).get("V4_SEALED_IDS") == "1" for s in stages):
            errors.append("sealed runtime requires explicit sealed_ids token privacy")
        warnings.append("participating V4 nodes can see token IDs and activations")
    warnings.append("resource and I/O evidence is operator provenance, not remote attestation; recheck at load")
    try:
        deployment_sha256 = config_digest(bundle)
    except (ValueError, TypeError) as exc:
        errors.append(f"deployment is not canonical finite JSON: {exc}")
        deployment_sha256 = None
    return {"schema": "shard-deployment-check/1", "ready": not errors, "errors": errors,
            "warnings": warnings, "stages": rows, "hosts": hosts,
            "policies": policies,
            "deployment_sha256": deployment_sha256, "checked_at": now.isoformat()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--max-age-s", type=float, default=3600)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        result = check_deployment(json.loads(Path(args.bundle).read_text(encoding="utf-8-sig")),
                                  max_age_s=args.max_age_s)
        encoded = json.dumps(result, indent=2, allow_nan=False)
        if args.out:
            Path(args.out).write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
        return 0 if result["ready"] else 2
    except (ValueError, OSError) as exc:
        print(json.dumps({"ready": False, "errors": [str(exc)]}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

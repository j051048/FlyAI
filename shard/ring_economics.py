"""Opt-in healthy-ring replacement from operator-pinned measurement artifacts.

Nothing in a contributor offer authorizes a download. Candidate definitions are
local configuration; raw signed V4 benchmark evidence is reverified, while the
formation wall clock remains an explicitly operator-observed (not attested)
measurement. Missing preparation/cache-loss costs disable economic replacement.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA = "shard-ring-economics/1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def formation_digest(row):
    # Lifecycle IDs and opt-in policies do not change the executable template.
    excluded = {"ring_id", "auto_replace", "auto_generations", "calibrated_alternatives",
                "healthy_rebalance", "economics", "replaces", "publish_aliases", "_economic_expected_plan",
                "_economic_expected_profile"}
    return digest({key: value for key, value in row.items() if key not in excluded})


def _positive(value, *, zero=False):
    return type(value) in (int, float) and math.isfinite(value) and (value >= 0 if zero else value > 0)


def options(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"enabled", "stability_window_s", "cooldown_s",
            "horizon_requests", "minimum_gain_fraction", "minimum_net_savings_s"}:
        raise ValueError("unknown healthy_rebalance policy")
    if type(value.get("enabled", False)) is not bool:
        raise ValueError("healthy_rebalance requires explicit boolean enabled")
    if not value.get("enabled", False):
        return None
    result = {"enabled": True, "stability_window_s": value.get("stability_window_s", 300),
        "cooldown_s": value.get("cooldown_s", 600), "horizon_requests": value.get("horizon_requests", 100),
        "minimum_gain_fraction": value.get("minimum_gain_fraction", .15),
        "minimum_net_savings_s": value.get("minimum_net_savings_s", 30)}
    for key in ("stability_window_s", "cooldown_s", "minimum_net_savings_s"):
        if not _positive(result[key]) or result[key] > 86400:
            raise ValueError("bounded positive rebalance timing policy required")
    if type(result["horizon_requests"]) is not int or not 1 <= result["horizon_requests"] <= 10000:
        raise ValueError("bounded explicit request horizon required")
    if not _positive(result["minimum_gain_fraction"]) or result["minimum_gain_fraction"] >= 1:
        raise ValueError("rebalance gain fraction must lie strictly between zero and one")
    return result


def benchmark_economics(report, plan, formation, measurement, *, ttl_s):
    """Export only independently valid strict-suite evidence; derive all speeds.

    This deliberately refuses old reports without full signed runtime identity.
    A benchmark can be below the speed target and still supply valid evidence.
    """
    from phase0 import v4_benchmark as bench
    from .pipeline_plan import validate_plan
    from .offers import ModelCohort
    if not _positive(ttl_s) or ttl_s > 86400:
        raise ValueError("economics TTL must be positive and at most one day")
    evaluated = bench.evaluate_report(report)
    if evaluated.get("valid_evidence") is not True:
        raise ValueError("economics requires complete independently verified live benchmark evidence")
    protocol = report["protocol"]
    if formation.get("mode", "pipelined") != protocol["run"]["mode"]:
        raise ValueError("economics formation mode differs from the measured coordinator")
    if (formation.get("round_calibration_file") or any(formation.get(name, {}).get("enabled", False)
            for name in ("speculation_policy", "round_speculation", "native_lanes"))):
        raise ValueError("fixed benchmark economics does not qualify adaptive or multi-lane coordinator policies")
    if evaluated["runtime_evidence"]["signed_observation_count"] != len(report["samples"]) * len(protocol["hardware"]):
        raise ValueError("economics requires signed runtime identity for every stage and job")
    plan = validate_plan(plan)
    cohort = ModelCohort.from_dict(formation["cohort"])
    if plan.get("model_cohort") != formation["cohort"] or plan["cohort_id"] != cohort.cohort_id:
        raise ValueError("economics plan and configured cohort differ")
    expected = {(n["node_id"], n["gpu_uuid"], n["signer_pubkey"], n["layer_start"], n["layer_end"])
                for n in protocol["hardware"]}
    actual = {(n["node_id"], n["gpu_uuid"], n["signer_pubkey"], n["lo"], n["hi"]) for n in plan["stages"]}
    if expected != actual or protocol["checkpoint"]["config_sha256"] != cohort.config_sha256:
        raise ValueError("economics plan does not describe the benchmark hardware/configuration")
    if protocol["checkpoint"].get("identity_kind") == "logical-tensor-catalogue":
        if (protocol["checkpoint"].get("checkpoint_id"), protocol["checkpoint"].get("manifest_sha256")) != (
                cohort.checkpoint_id, cohort.manifest_sha256):
            raise ValueError("economics benchmark checkpoint differs from configured cohort")
    else:
        raise ValueError("economics requires the native logical catalogue cohort")
    if any(sample["swarm_id"] != plan["ring_id"] for sample in report["samples"]):
        raise ValueError("economics benchmark was measured on another ring")
    for stage in plan["stages"]:
        identity = evaluated["runtime_evidence"]["stage_identities"][stage["node_id"]]
        if not stage.get("runtime_config_sha256") or identity["runtime_config_sha256"] != stage["runtime_config_sha256"]:
            raise ValueError("economics stage runtime is not pinned to the plan")
    if (plan["coordinator"]["head"] != formation["head"]
            or plan["coordinator"]["tail"] != formation["tail"]):
        raise ValueError("economics formation endpoints differ from the measured plan")
    ready_s, coordinator_profile = None, None
    measured_at = datetime.fromisoformat(report["created_utc"].replace("Z", "+00:00")).timestamp()
    if measurement is not None:
        if (not isinstance(measurement, dict) or measurement.get("schema") != "shard-formation-timing/1"
                or measurement.get("formation_sha256") != formation_digest(formation)
                or measurement.get("ring_id") != plan["ring_id"]
                or measurement.get("cohort_id") != cohort.cohort_id
                or measurement.get("clock") != "perf_counter"
                or not _positive(measurement.get("measured_at"))
                or measurement["measured_at"] > measured_at
                or not _positive(measurement.get("prepare_load_capture_warmup_s"))):
            raise ValueError("formation preparation timing lacks exact configuration/ring binding")
        ready_s = measurement["prepare_load_capture_warmup_s"]
        expected_profile = {"mode": protocol["run"]["mode"], "coordinator_environment": {
            name: protocol["env"][name] for name in ("V4_SPEC_DEPTH", "V4_REFILL_FLOOR", "V4_LAZY_DRAFT")}}
        coordinator_profile = measurement.get("coordinator_profile")
        if coordinator_profile != expected_profile:
            raise ValueError("formation coordinator configuration differs from frozen benchmark environment")
        measured_at = min(measured_at, measurement["measured_at"])
    warm = [sample for sample in report["samples"] if sample["phase"] == "warm"]
    # The same four-workload mix is compared. Do not drop receipt-sweep costs or
    # infer the result from the best decode-only sample.
    service_s = statistics.median(sample["measurement"]["full_service_s"] for sample in warm)
    cache_disabled = (protocol["env"].get("V4_CONVERSATION_CACHE_MIB", "0") in ("", "0")
                      and not formation.get("conversation_cache", {}).get("enabled", False)
                      and not formation.get("native_lanes", {}).get("enabled", False))
    summary = {"cohort_id": cohort.cohort_id, "formation_sha256": formation_digest(formation),
        "runtime_sha256": digest(evaluated["runtime_evidence"]["stage_identities"]),
        "source_sha256": protocol["source"]["sha256"], "checkpoint_id": cohort.checkpoint_id,
        "workload_sha256": digest({"prompts": protocol["prompts"], "run": protocol["run"]}),
        "workload": {"prompt_tokens": protocol["run"]["prompt_tokens"], "max_new": protocol["run"]["max_new"],
                     "mode": protocol["run"]["mode"], "prompt_sha256s": sorted(p["sha256"] for p in protocol["prompts"])},
        "service_s": service_s, "warm_samples": len(warm), "coordinator_profile": coordinator_profile,
        "prepare_load_capture_warmup_s": ready_s, "cache_loss_s": 0.0 if cache_disabled else None,
        "cache_loss_scope": "cache_disabled_in_frozen_protocol" if cache_disabled else "unknown",
        "measured_at": measured_at,
        "ttl_s": ttl_s, "source": "verified_strict_benchmark_and_operator_formation_clock",
        "scope": "fixed workload median forecast; not hardware attestation or an SLO guarantee"}
    return {"schema": SCHEMA, "summary": summary, "benchmark": report,
            "deployment_plan": plan, "formation_measurement": measurement}


def load_economics(reference, formation, local, *, now, cache=None):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("operator-pinned economics path and SHA256 required")
    path = Path(local(reference["path"]))
    if not path.is_file() or path.stat().st_size > 32 * 1024**2:
        raise ValueError("bounded local economics artifact required")
    raw = path.read_bytes()
    pin = hashlib.sha256(raw).hexdigest()
    if pin != reference["sha256"]:
        raise ValueError("economics artifact hash differs from its configured pin")
    key = (pin, formation_digest(formation))
    value = cache.get(key) if cache is not None else None
    if value is None:
        value = json.loads(raw)
        if value.get("schema") != SCHEMA:
            raise ValueError("unsupported economics artifact")
        expected = benchmark_economics(value["benchmark"], value["deployment_plan"], formation,
            value.get("formation_measurement"), ttl_s=value["summary"]["ttl_s"])
        if expected != value:
            raise ValueError("economics summary differs from its raw evidence")
        if cache is not None:
            if len(cache) >= 32:
                cache.pop(next(iter(cache)))
            cache[key] = value
    summary = value["summary"]
    if not 0 <= now - summary["measured_at"] <= summary["ttl_s"]:
        raise ValueError("economics benchmark evidence is stale or future dated")
    return summary


def compare_economics(incumbent, candidate, policy):
    for key in ("cohort_id", "workload_sha256", "source_sha256"):
        if incumbent[key] != candidate[key]:
            return {"eligible": False, "reason": "economic_evidence_not_comparable"}
    costs = [candidate.get("prepare_load_capture_warmup_s"), incumbent.get("cache_loss_s")]
    if any(not _positive(value, zero=True) for value in costs):
        return {"eligible": False, "reason": "replacement_cost_unknown"}
    gain = incumbent["service_s"] - candidate["service_s"]
    fraction = gain / incumbent["service_s"]
    net = gain * policy["horizon_requests"] - sum(costs)
    eligible = fraction >= policy["minimum_gain_fraction"] and net >= policy["minimum_net_savings_s"]
    return {"eligible": eligible, "reason": "economic_gain_confirmed" if eligible else "economic_gain_insufficient",
        "request_gain_s": gain, "gain_fraction": fraction, "switch_cost_s": sum(costs),
        "net_savings_s": net, "assumed_horizon_requests": policy["horizon_requests"],
        "scope": "explicit fixed-workload forecast with measured switch costs"}


def require_measured_placement(manifest, expected):
    fields = ("node_id", "gpu_uuid", "signer_pubkey", "lo", "hi", "runtime_config_sha256",
              "endpoint", "next_endpoint", "context_limit")
    selected = [tuple(stage.get(key) for key in fields) for stage in manifest["stages"]]
    measured = [tuple(stage.get(key) for key in fields) for stage in expected["stages"]]
    if (selected != measured or manifest["cohort_id"] != expected["cohort_id"]
            or manifest.get("execution") != expected.get("execution")
            or any(manifest["coordinator"].get(key) != expected["coordinator"].get(key) for key in ("head", "tail"))):
        raise ValueError("economic forecast is not valid for the actual selected placement")


def require_measured_coordinator(backend, expected):
    method = getattr(backend, "economics_profile", None)
    if expected is None or not callable(method) or method() != expected:
        raise ValueError("economic forecast differs from the live coordinator configuration")

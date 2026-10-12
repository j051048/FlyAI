"""Real Ed25519 evidence and lifecycle tests; synthetic CPU timing, no GPU claims."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import threading
from types import SimpleNamespace

import pytest

from shard.offers import ModelCohort
from shard.pipeline_plan import build_plan
from shard.ring_economics import (benchmark_economics, compare_economics, formation_digest,
                                  load_economics, options, require_measured_placement)
from shard.ring_pool import RingPool, RingState
from test_ring_pool import add_ready
from test_weight_replacement import reconciliation_service
from test_v4_benchmark import (bench, logical_checkpoint, observed_protocol, reseal_observed_protocol,
                               Clock, ObservedAdapter, report)


@pytest.fixture
def measured_pair(tmp_path):
    _, assets, _ = logical_checkpoint(tmp_path)
    base, keys = observed_protocol()
    base["checkpoint"] = bench._checkpoint_identity(bench.checkpoint_inventory(assets))
    for node in base["hardware"]:
        node["checkpoint_sha256"] = base["checkpoint"]["sha256"]
        node["config_sha256"] = base["checkpoint"]["config_sha256"]
    base = reseal_observed_protocol(base)
    checkpoint = base["checkpoint"]
    cohort = ModelCohort(base["model_id"], checkpoint["manifest_sha256"], checkpoint["checkpoint_id"],
        checkpoint["config_sha256"], "fp8-fp4", "fixture-native-v4", "strict/1", "fixture-greedy", 43)
    result = []
    for ring_id, prefill in (("old", 10), ("new", 1)):
        p = deepcopy(base)
        if ring_id == "new":
            for row in p["hardware"]:
                row["node_id"] += "-candidate"
                row["gpu_uuid"] += "-candidate"
            p = reseal_observed_protocol(p)
        clock = Clock()
        adapter = ObservedAdapter(p, keys, clock, rate=50, prefill_s=prefill, sweep_s=.1)
        adapter.swarm_id = ring_id
        adapter.artifact_verification = {**bench.verify_checkpoint(assets, p["checkpoint"]),
            "engine_source_verified": True, "authenticated_cohort_verified": True}
        raw = bench.run_suite(adapter, p, fresh_ring=True, clock=clock)
        plan = build_plan({"n_layers": 43}, ring_id=ring_id, cohort_id=cohort.cohort_id,
            endpoints=[f"127.0.0.1:{9000+i}" for i in range(4)], split=[12, 12, 12, 7],
            node_ids=[row["node_id"] for row in p["hardware"]],
            gpu_uuids=[row["gpu_uuid"] for row in p["hardware"]], model_cohort=cohort.to_dict())
        for stage, hw in zip(plan["stages"], p["hardware"]):
            stage.update(signer_pubkey=hw["signer_pubkey"], runtime_config_sha256=hw["runtime_config_sha256"])
        formation = {"ring_id": ring_id, "cohort": cohort.to_dict(), "head": plan["coordinator"]["head"],
            "tail": plan["coordinator"]["tail"], "route_ids": [s["node_id"] for s in plan["stages"]]}
        measured_at = datetime.fromisoformat(raw["created_utc"].replace("Z", "+00:00")).timestamp()
        startup = {"schema": "shard-formation-timing/1", "clock": "perf_counter", "ring_id": ring_id,
            "cohort_id": cohort.cohort_id, "formation_sha256": formation_digest(formation),
            "coordinator_profile": {"mode": p["run"]["mode"], "coordinator_environment": {
                name: p["env"][name] for name in ("V4_SPEC_DEPTH", "V4_REFILL_FLOOR", "V4_LAZY_DRAFT")}},
            "measured_at": measured_at-1, "prepare_load_capture_warmup_s": 2.0}
        artifact = benchmark_economics(raw, plan, formation, startup, ttl_s=3600)
        reference = save(tmp_path / (ring_id + ".json"), artifact)
        formation["economics"] = reference
        result.append((formation, artifact))
    return result


def save(path, artifact):
    raw = json.dumps(artifact, sort_keys=True, allow_nan=False).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def test_export_uses_whole_service_and_rechecks_signatures(measured_pair, tmp_path):
    formation, artifact = measured_pair[0]
    observation = load_economics(formation["economics"], formation, Path, now=time.time())
    assert observation["service_s"] == pytest.approx(10+.16+.001+.1)
    assert observation["warm_samples"] == 12
    assert observation["prepare_load_capture_warmup_s"] == 2
    assert observation["cache_loss_s"] == 0
    bad = deepcopy(artifact)
    bad["benchmark"]["samples"][0]["receipts"][0]["signature"] = "forged"
    reference = save(tmp_path / "forged.json", bad)
    with pytest.raises(ValueError, match="independently verified"):
        load_economics(reference, formation, Path, now=time.time())
    bad = deepcopy(artifact); bad["summary"]["service_s"] = .001
    with pytest.raises(ValueError, match="summary differs"):
        load_economics(save(tmp_path / "false-speed.json", bad), formation, Path, now=time.time())


def test_legacy_or_missing_signed_runtime_cannot_be_promoted_to_cost(measured_pair):
    formation, artifact = measured_pair[0]
    with pytest.raises(ValueError, match="signed runtime identity"):
        benchmark_economics(report(), artifact["deployment_plan"], formation, None, ttl_s=3600)
    bad = deepcopy(artifact["benchmark"])
    bad["samples"][0]["receipts"][0].pop("runtime_observation")
    with pytest.raises(ValueError, match="independently verified"):
        benchmark_economics(bad, artifact["deployment_plan"], formation, None, ttl_s=3600)


def test_cost_forecast_rejects_unknown_cache_loss_or_unmeasured_preparation(measured_pair):
    (old, a), (new, b) = measured_pair
    policy = options({"enabled": True, "horizon_requests": 10})
    assert compare_economics(a["summary"], b["summary"], policy)["eligible"]
    unmeasured = benchmark_economics(b["benchmark"], b["deployment_plan"], new, None, ttl_s=3600)
    assert compare_economics(a["summary"], unmeasured["summary"], policy)["reason"] == "replacement_cost_unknown"
    cached = {**old, "conversation_cache": {"enabled": True}}
    unknown = benchmark_economics(a["benchmark"], a["deployment_plan"], cached, None, ttl_s=3600)
    assert unknown["summary"]["cache_loss_s"] is None
    assert not compare_economics(unknown["summary"], b["summary"], policy)["eligible"]
    assert not compare_economics(a["summary"], {**b["summary"], "workload_sha256": "0"*64}, policy)["eligible"]


def test_fixed_benchmark_cannot_qualify_other_live_coordinator_recipe(measured_pair):
    from shard.ring_economics import require_measured_coordinator
    formation, artifact = measured_pair[0]
    measurement = deepcopy(artifact["formation_measurement"])
    measurement["coordinator_profile"]["coordinator_environment"]["V4_SPEC_DEPTH"] = "4"
    with pytest.raises(ValueError, match="coordinator configuration"):
        benchmark_economics(artifact["benchmark"], artifact["deployment_plan"], formation, measurement, ttl_s=3600)
    for field in ("round_speculation", "native_lanes", "speculation_policy"):
        with pytest.raises(ValueError, match="fixed benchmark"):
            benchmark_economics(artifact["benchmark"], artifact["deployment_plan"],
                {**formation, field: {"enabled": True}}, None, ttl_s=3600)
    with pytest.raises(ValueError, match="fixed benchmark"):
        benchmark_economics(artifact["benchmark"], artifact["deployment_plan"],
            {**formation, "round_calibration_file": "later-calibration.json"}, None, ttl_s=3600)
    with pytest.raises(ValueError, match="live coordinator"):
        require_measured_coordinator(SimpleNamespace(economics_profile=lambda: None), artifact["summary"]["coordinator_profile"])


def test_artifacts_expire_and_bind_formation_and_exact_placement(measured_pair, tmp_path):
    formation, artifact = measured_pair[0]
    summary = artifact["summary"]
    with pytest.raises(ValueError, match="stale"):
        load_economics(formation["economics"], formation, Path, now=summary["measured_at"] + 3601)
    with pytest.raises(ValueError, match="configuration"):
        load_economics(formation["economics"], {**formation, "spec_depth": 99}, Path, now=time.time())
    with pytest.raises(ValueError, match="hash differs"):
        load_economics({**formation["economics"], "sha256": "0"*64}, formation, Path, now=time.time())
    # Fresh report cannot launder an old initialization observation.
    stale_startup = {**artifact["formation_measurement"], "measured_at": summary["measured_at"]-7200}
    stale = benchmark_economics(artifact["benchmark"], artifact["deployment_plan"], formation, stale_startup, ttl_s=3600)
    with pytest.raises(ValueError, match="stale"):
        load_economics(save(tmp_path / "stale-startup.json", stale), formation, Path, now=time.time())
    changed = deepcopy(artifact["deployment_plan"]); changed["ring_id"] = "new-epoch"
    require_measured_placement(changed, artifact["deployment_plan"])
    changed["stages"][0]["gpu_uuid"] = "different-GPU"
    with pytest.raises(ValueError, match="actual selected placement"):
        require_measured_placement(changed, artifact["deployment_plan"])


def configured_service(measured_pair):
    (definition, a), (candidate, b) = measured_pair
    pool = RingPool()
    old = add_ready(pool, "old", cohort=a["summary"]["cohort_id"])
    old.backend.economics_profile = lambda: a["summary"]["coordinator_profile"]
    service = reconciliation_service(pool, {**definition, "calibrated_alternatives": [candidate]})
    service._economic_cache = {}
    service.local = Path
    policy = service._replacement_policies["old"]
    policy.update(economic=options({"enabled": True, "stability_window_s": 30, "cooldown_s": 60,
                                  "horizon_requests": 10}), replace_failed=False, active_definition=definition)
    for prompt in a["benchmark"]["protocol"]["prompts"]:
        old.routing_costs.note_completion({"mode": "pipelined", "prompt_ids": prompt["token_ids"]}, 512, 8)
    return service, policy, old, a, b


def test_healthy_replacement_observes_window_holds_candidate_then_switches_boundary(measured_pair):
    service, policy, old, a, b = configured_service(measured_pair)
    pool = service.pool
    held = pool.acquire("test-model", old.cohort_id, estimated_tokens=1)
    formed = []
    def form(candidate):
        formed.append(candidate)
        new = add_ready(pool, candidate["ring_id"], cohort=old.cohort_id)
        new.backend.economics_profile = lambda: b["summary"]["coordinator_profile"]
        assert new.state == RingState.READY and new.admission_held
        during = pool.acquire("test-model", old.cohort_id, estimated_tokens=1)
        assert during.ring is old
        pool.release(during)
        return {"ring_id": new.ring_id, "formation_measurement": {"prepare_load_capture_warmup_s": 2}}
    service.form_with_alternatives = form
    try:
        service._evaluate_healthy_candidate("old", policy, old)
        assert formed == [] and pool.snapshot()["reconciliation"]["old"]["state"] == "economic_observing"
        policy["stable_since"] -= 31
        service._evaluate_healthy_candidate("old", policy, old)
        assert len(formed) == 1
        assert formed[0]["_economic_expected_plan"] == b["deployment_plan"]
        assert old.state == RingState.DRAINING and held.ring is old and not old.backend.closed
        next_job = pool.acquire("test-model", old.cohort_id, estimated_tokens=1)
        assert next_job.ring.ring_id == formed[0]["ring_id"] and not next_job.ring.admission_held
        assert policy["next_attempt"] > time.monotonic() + 59
        pool.release(next_job)
    finally:
        pool.release(held); pool.shutdown()


@pytest.mark.parametrize("reason", ["setup_overrun", "workload_changed", "evidence_changed"])
def test_preparation_cannot_publish_if_final_economics_changed(measured_pair, reason):
    service, policy, old, a, b = configured_service(measured_pair)
    pool = service.pool
    def form(candidate):
        new = add_ready(pool, candidate["ring_id"], cohort=old.cohort_id)
        new.backend.economics_profile = lambda: b["summary"]["coordinator_profile"]
        if reason == "workload_changed":
            old.routing_costs.note_completion({"mode": "pipelined"}, 4096, 8)
        if reason == "evidence_changed":
            Path(policy["economic_pending"]["economics"]["path"]).write_text("{}")
        return {"ring_id": new.ring_id, "formation_measurement": {
            "prepare_load_capture_warmup_s": 1000 if reason == "setup_overrun" else 2}}
    service.form_with_alternatives = form
    try:
        service._evaluate_healthy_candidate("old", policy, old)
        policy["stable_since"] -= 31
        service._evaluate_healthy_candidate("old", policy, old)
        assert old.state == RingState.READY and not old.backend.closed
        assert pool.snapshot()["reconciliation"]["old"]["state"] == "unavailable"
        assert all(r.state == RingState.STOPPED for r in pool.rings() if r is not old)
    finally:
        pool.shutdown()


def test_unknown_workload_never_triggers_download_and_default_is_disabled(measured_pair):
    assert options(None) is None and options({"enabled": False}) is None
    service, policy, old, _, _ = configured_service(measured_pair)
    old.routing_costs.note_completion({"mode": "greedy"}, 512, 8)
    service.form_with_alternatives = lambda _: pytest.fail("unknown workload must not trigger loading")
    try:
        service._evaluate_healthy_candidate("old", policy, old)
        assert service.pool.snapshot()["reconciliation"]["old"]["reason"] == "recent_workload_not_calibrated"
    finally:
        service.pool.shutdown()


def test_cli_exports_real_raw_report_and_formation_clock(measured_pair, tmp_path, capsys):
    from phase0.v4_ring_economics import main
    formation, artifact = measured_pair[0]
    files = {"report": artifact["benchmark"], "deployment-plan": artifact["deployment_plan"],
             "formation": formation, "formation-result": {"formation_measurement": artifact["formation_measurement"]}}
    args = []
    for flag, value in files.items():
        path = tmp_path / (flag + ".json"); path.write_text(json.dumps(value))
        args += ["--" + flag, str(path)]
    out = tmp_path / "export.json"
    assert main([*args, "--out", str(out)]) == 0
    ref = json.loads(capsys.readouterr().out)
    assert ref.pop("replacement_cost_known")
    assert load_economics(ref, formation, Path, now=time.time())["service_s"] == artifact["summary"]["service_s"]


def test_real_formation_wrapper_collects_wall_clock_and_actual_plan(measured_pair, tmp_path):
    from shard.network_service import OpenNetworkService
    row, evidence = measured_pair[0]
    row = {**row, "measurements": {}, "profile": {"n_layers": 43}}
    service = OpenNetworkService.__new__(OpenNetworkService)
    service.records, service._configured_record_ids = {}, {row["ring_id"]}
    service.pool = RingPool()
    service._formation_lock = threading.RLock()
    service.profile_factory = None
    service.local = Path
    service.config = {"formation_evidence_dir": str(tmp_path / "recorded")}
    manifest = evidence["deployment_plan"]
    def form(ring_id, cohort, profile, **kwargs):
        time.sleep(.003)  # Real monotonic observation, no manually provided cost.
        return {"ring_id": ring_id, "cohort_id": ModelCohort.from_dict(cohort).cohort_id, "ready": True}
    service.controller = SimpleNamespace(form=form,
        _formations={row["ring_id"]: {"backend": SimpleNamespace(pipeline_plan=manifest)}})
    result = service.form_one(row)
    stored = json.loads(Path(result["formation_evidence_path"]).read_text())
    assert stored["deployment_plan"] == manifest
    assert stored["formation_measurement"]["prepare_load_capture_warmup_s"] > 0
    assert stored["formation_measurement"]["formation_sha256"] == formation_digest(row)
    service.pool.shutdown()

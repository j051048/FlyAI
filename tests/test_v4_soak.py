import copy
from types import SimpleNamespace

import pytest

import v4_soak as acceptance
import v4_benchmark as bench
from test_v4_benchmark import (Clock, FakeAdapter, ObservedAdapter, logical_checkpoint,
    observed_protocol, protocol, reseal_observed_protocol)


def campaign(*, backend="live_ring"):
    p, keys = protocol()
    clock = Clock()
    adapter = FakeAdapter(p, keys, clock)
    # Synthetic signed fixtures exercise verifier branches; these are not measurements.
    adapter.backend = backend
    report = acceptance.run_soak(adapter, p, cycles=1, duration_s=10,
                                 ttft_p95_s=3, token_gap_p95_s=.1, clock=clock)
    return report, p


def test_soak_reverifies_every_committed_stream_and_signed_job():
    report, p = campaign()
    result = acceptance.evaluate_soak(report, p, report["contract"])
    assert result["status"] == "passed", result
    assert result["jobs"] == 8
    assert result["summary"]["median_decode_tok_s"] == pytest.approx(50)
    assert result["summary"]["ttft_p95_s"] > 2


def test_mock_does_not_qualify_hardware_campaign():
    report, p = campaign(backend="mock")
    assert acceptance.evaluate_soak(report, p, report["contract"])["status"] == "unverified"


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(complete=False),
    lambda r: r.update(elapsed_s=1),
    lambda r: r.update(elapsed_s=r["elapsed_s"] + 3600),
    lambda r: r.update(active_observation_s=r["active_observation_s"] + 3600),
    lambda r: r["samples"][1].update(campaign_interval_s=r["samples"][0]["campaign_interval_s"]),
    lambda r: r["samples"].pop(),
    lambda r: r["samples"][0]["tokens"].__setitem__(0, 999),
    lambda r: r["samples"][0]["receipts"][0].update(job_id="forged"),
    lambda r: r["samples"][0]["measurement"].update(decode_committed_tok_s=999),
    lambda r: r["samples"][1].update(nonce=r["samples"][0]["nonce"]),
])
def test_incomplete_or_fabricated_soak_cannot_pass(mutation):
    report, p = campaign()
    contract = copy.deepcopy(report["contract"])
    mutation(report)
    assert acceptance.evaluate_soak(report, p, contract)["status"] == "failed"


def test_relaxing_saved_slo_cannot_override_independent_contract():
    report, p = campaign()
    contract = copy.deepcopy(report["contract"])
    report["contract"]["ttft_p95_s"] = 999
    assert acceptance.evaluate_soak(report, p, contract)["status"] == "failed"


def test_runtime_metrics_can_qualify_only_gpu_mode_with_real_signed_fields():
    p, keys = protocol(metrics=True)
    clock = Clock()
    adapter = FakeAdapter(p, keys, clock)
    adapter.backend = "live_ring"
    report = acceptance.run_soak(adapter, p, cycles=1, duration_s=10,
                                 ttft_p95_s=3, token_gap_p95_s=.1, clock=clock)
    assert acceptance.evaluate_soak(report, p, report["contract"])["status"] == "passed"


def test_cohosted_benchmark_requires_explicit_different_protocol():
    p, _ = protocol()
    p["hardware"][1]["host_id"] = p["hardware"][0]["host_id"]
    assert bench.protocol_errors(bench.seal_protocol(p))
    p["isolation"] = "none"
    assert bench.protocol_errors(bench.seal_protocol(p)) == []
    p["hardware"][1]["gpu_uuid"] = p["hardware"][0]["gpu_uuid"]
    assert bench.protocol_errors(bench.seal_protocol(p))


class _LifecycleEvidenceAdapter(ObservedAdapter):
    """Synthetic model/device data with genuine receipt and owner signatures.

    This adapter only exercises the saved-evidence verifier. It does not claim
    to execute a model, CUDA, socket recovery or arithmetic rollback.
    """
    def __init__(self, p, keys, clock, **kwargs):
        super().__init__(p, keys, clock, **kwargs)
        from shard.pipeline_plan import build_plan
        from shard.pipeline_session import SessionConfig, SessionRegistry
        from shard.receipt import gen_key, pub_b64
        from shard.offers import ModelCohort
        from shard.weight_artifacts import catalog_model_cohort
        cohort = ModelCohort.from_dict(catalog_model_cohort(p["checkpoint"]["global_catalog"]))
        plan = build_plan({"n_layers": 43}, ring_id=self.swarm_id, cohort_id=cohort.cohort_id,
            model_cohort=cohort.to_dict(), endpoints=[f"127.0.0.1:{29610+i}" for i in range(len(keys))],
            split=[n["layer_end"]-n["layer_start"] for n in p["hardware"]],
            node_ids=[n["node_id"] for n in p["hardware"]], gpu_uuids=[n["gpu_uuid"] for n in p["hardware"]])
        for stage, node in zip(plan["stages"], p["hardware"]):
            stage["signer_pubkey"] = node["signer_pubkey"]
        plan["coordinator"]["signer_pubkey"] = pub_b64(gen_key())
        session = SessionConfig.from_plan(plan)
        self.session_options = {"session_config": session}
        self.registry = SessionRegistry(session, keys[0])
        self.fence = 0
        self.reconnect()

    def reconnect(self):
        self.fence += 1
        self.pipe = SimpleNamespace(grant=self.registry._sign({
            "ring_id": self.swarm_id, "cohort_id": self.registry.config.plan["cohort_id"],
            "session_id": f"fixture-session-{self.fence}", "fence": self.fence,
            "boot_id": self.registry.boot_id, "boot_started_at": self.registry.boot_started_at,
            "owner": f"fixture-owner-{self.fence}", "expires_at": 1.0}))
        return dict(self.pipe.grant)

    def close(self):
        pass

    def generate(self, *args, drafter=None, ngram_k=None, ngram_order=None, **kwargs):
        result = super().generate(*args, **kwargs)
        if drafter is not None:
            result.update(rounds=2, accept_hist={"0": 2})
        return result


def native_campaign(tmp_path, kind, *, local_scope="assets", observation_mutator=None,
                    signed_observations=True):
    full, assets, partial = logical_checkpoint(tmp_path)
    local = {"full": full, "assets": assets, "partial": partial}[local_scope]
    p, keys = observed_protocol()
    p["checkpoint"] = bench._checkpoint_identity(bench.checkpoint_inventory(local))
    for node in p["hardware"]:
        node["checkpoint_sha256"] = p["checkpoint"]["sha256"]
        node["config_sha256"] = p["checkpoint"]["config_sha256"]
    p = reseal_observed_protocol(p)
    clock = Clock()
    adapter_type = ObservedAdapter if kind == "soak" else _LifecycleEvidenceAdapter
    adapter = adapter_type(p, keys, clock, observation_mutator=observation_mutator)
    adapter.artifact_verification = {**bench.verify_checkpoint(local, p["checkpoint"]),
        "engine_source_verified": True, "authenticated_cohort_verified": True}
    if not signed_observations:
        adapter.sweep = lambda nonce: FakeAdapter.sweep(adapter, nonce)
    if kind == "soak":
        report = acceptance.run_soak(adapter, p, cycles=1, duration_s=10,
            ttft_p95_s=3, token_gap_p95_s=.1, clock=clock)
        evaluate = acceptance.evaluate_soak
    else:
        report = acceptance.run_lifecycle(adapter, p, contract=acceptance.lifecycle_contract(1))
        evaluate = acceptance.evaluate_lifecycle
    return report, p, evaluate


@pytest.mark.parametrize("kind", ["soak", "lifecycle"])
@pytest.mark.parametrize("local_scope", ["full", "assets", "partial"])
def test_native_campaign_accepts_truthful_authenticated_local_weight_scope(tmp_path, kind, local_scope):
    report, p, evaluate = native_campaign(tmp_path, kind, local_scope=local_scope)
    result = evaluate(report, p, report["contract"])
    assert result["status"] == "passed", result
    assert report["artifact_verification"]["checkpoint_bytes_verified"] is (local_scope == "full")


@pytest.mark.parametrize("kind", ["soak", "lifecycle"])
@pytest.mark.parametrize("field,value", [
    ("authenticated_cohort_verified", False), ("global_catalog_verified", False),
    ("config_bytes_verified", False), ("tokenizer_bytes_verified", False),
    ("engine_source_verified", False), ("checkpoint_id", "tensor-sha256:" + "0" * 64),
    ("local_weight_scope", "unverified size-only inventory"),
])
def test_native_campaign_missing_authentication_or_identity_stays_unverified(tmp_path, kind, field, value):
    report, p, evaluate = native_campaign(tmp_path, kind)
    report["artifact_verification"][field] = value
    result = evaluate(report, p, report["contract"])
    assert result["status"] == "unverified", result
    assert result["missing_evidence"]


@pytest.mark.parametrize("kind", ["soak", "lifecycle"])
def test_native_coordinator_assets_cannot_claim_complete_local_checkpoint_bytes(tmp_path, kind):
    report, p, evaluate = native_campaign(tmp_path, kind)
    report["artifact_verification"]["checkpoint_bytes_verified"] = True
    result = evaluate(report, p, report["contract"])
    assert result["status"] == "failed", result
    assert "falsely claim complete" in ";".join(result["errors"])


@pytest.mark.parametrize("kind", ["soak", "lifecycle"])
def test_native_catalogue_authentication_still_requires_signed_runtime_for_every_stage(tmp_path, kind):
    report, p, evaluate = native_campaign(tmp_path, kind, signed_observations=False)
    result = evaluate(report, p, report["contract"])
    assert result["status"] == "unverified", result
    assert "runtime observation absent" in ";".join(result["missing_evidence"])


@pytest.mark.parametrize("kind", ["soak", "lifecycle"])
def test_native_catalogue_authentication_does_not_replace_actual_runtime_identity(tmp_path, kind):
    report, p, evaluate = native_campaign(tmp_path, kind,
        observation_mutator=lambda observation: observation.update(process_run_id="another-process"))
    result = evaluate(report, p, report["contract"])
    assert result["status"] == "failed", result
    assert "runtime node/GPU/process identity differs" in ";".join(result["errors"])


@pytest.mark.parametrize("kind", ["soak", "lifecycle"])
def test_native_catalogue_authentication_requires_complete_node_inventory(tmp_path, kind):
    report, p, evaluate = native_campaign(tmp_path, kind)
    del p["hardware"][0]["gpu_uuid"]
    p = bench.seal_protocol(p)
    report.update(protocol=p, protocol_sha256=p["sha256"])
    assert evaluate(report, p, report["contract"])["status"] in ("failed", "unverified")


@pytest.mark.parametrize("kind", ["soak", "lifecycle"])
def test_native_catalogue_authentication_cannot_replace_full_signed_stage_coverage(tmp_path, kind):
    report, p, evaluate = native_campaign(tmp_path, kind)
    row = report["samples"][0] if kind == "soak" else report["cases"][0]
    row["receipts"].pop()
    assert evaluate(report, p, report["contract"])["status"] == "failed"

import copy
from datetime import datetime, timezone

import pytest

from shard.deployment import check_deployment, config_digest
from shard.host_probe import summarize_transfers, validate_probe, verify_report
from shard.receipt import gen_key, pub_b64
from shard.resources import (CalibrationProvenance, GpuRequirements, HostRequirements,
                             PlacementRequirements)


NOW = datetime(2026, 10, 8, tzinfo=timezone.utc)
STAMP = NOW.isoformat()


def bundle():
    stages = []
    for i, (lo, hi) in enumerate(((0, 22), (22, 43))):
        config = {"expert_placement": "ram", "lo": lo, "hi": hi,
                  "head": i == 0, "tail": i == 1, "dspark": False, "args": {"n_layers": 43},
                  "environment": {"V4_EXPERT_PLACEMENT": "ram"}}
        req = PlacementRequirements("deepseek-test", lo, hi,
            GpuRequirements(resident_weights_bytes=20), HostRequirements(routed_experts_bytes=60, pinned_bytes=60),
            CalibrationProvenance("checkpoint", config_digest(config), STAMP, f"node-{i}", "measured", "fixture"))
        stages.append({"node_id": f"node-{i}", "host_id": "shared", "gpu_uuid": f"GPU-{i}",
                       "signer_pubkey": pub_b64(gen_key()), "lo": lo, "hi": hi, "port": 29000 + i,
                       "requirements": req.to_dict(), "runtime_config": config, "placement": "ram",
                       "env": {"V4_EXPERT_PLACEMENT": "ram"}, "capacity_measured_at": STAMP,
                       "capacity": {"available_vram_bytes": 100, "available_ram_bytes": 150,
                                    "pinnable_ram_bytes": 150, "available_disk_bytes": 100}})
    io = {"schema": "shard-host-io-probe/1", "host_id": "shared", "measured_at": STAMP,
          "pin_budget_verified": True, "observed_pinned_allocation_bytes": 150,
          "sample_bytes": 32, "repeats": 2, "pin_budget_requested_bytes": 150,
          "reservation": False, "host_capacity": {"available_ram_bytes": 1000},
          **summarize_transfers([{"local_index": 0, "gpu_uuid": "GPU-0", "samples_ms": [1, 1]},
                                 {"local_index": 1, "gpu_uuid": "GPU-1", "samples_ms": [1, 1]}], .01, 32, 2)}
    return {"schema": "shard-deployment/1", "model_id": "deepseek-test", "checkpoint_id": "checkpoint",
            "layer_count": 43, "trust_mode": "trusted_nodes", "stages": stages,
            "host_io": {"shared": io}}


def test_colocated_cards_are_admitted_when_shared_budget_and_probe_fit():
    result = check_deployment(bundle(), now=NOW)
    assert result["ready"], result
    assert result["hosts"]["shared"]["ram"] == 120
    assert "attestation" in result["warnings"][-1]


def test_individually_fitting_cards_cannot_double_count_shared_host_ram():
    b = bundle()
    for stage in b["stages"]:
        stage["capacity"]["available_ram_bytes"] = 100
    result = check_deployment(b, now=NOW)
    assert not result["ready"]
    assert all(row["fit"]["fits"] for row in result["stages"])
    assert any("aggregate ram" in error for error in result["errors"])


@pytest.mark.parametrize("mutation", [
    lambda b: b["stages"][1].update(gpu_uuid="GPU-0"),
    lambda b: b["stages"][1].update(port=29000),
    lambda b: b["stages"][1]["env"].update(V4_EXPERT_PLACEMENT="gpu"),
    lambda b: b["stages"][1]["env"].update(V4_KV_PLACEMENT="layer"),
    lambda b: b["stages"][1]["env"].update(V4_PREFILL_QUERY_CHUNK="512"),
    lambda b: b["stages"][0]["runtime_config"].update(lo=1),
    lambda b: b["stages"][0].update(capacity_measured_at="2026-01-01T00:00:00Z"),
    lambda b: b["host_io"]["shared"].update(pin_budget_verified=False),
    lambda b: b["host_io"]["shared"].update(devices=[{"gpu_uuid": "GPU-0"}]),
    lambda b: b["host_io"]["shared"].update(aggregate_h2d_gb_s=float("nan")),
    lambda b: b["host_io"]["shared"].update(aggregate_h2d_gb_s=999),
    lambda b: b["host_io"]["shared"].update(observed_pinned_allocation_bytes=999),
])
def test_mismatched_or_unmeasured_deployment_is_rejected(mutation):
    b = bundle()
    mutation(b)
    assert not check_deployment(b, now=NOW)["ready"]


def test_explicit_sealed_mode_requires_all_stages_and_exposes_activation_scope():
    b = bundle()
    b["trust_mode"] = "sealed_ids"
    assert not check_deployment(b, now=NOW)["ready"]
    for stage in b["stages"]:
        stage["env"]["V4_SEALED_IDS"] = "1"
        stage["runtime_config"]["environment"]["V4_SEALED_IDS"] = "1"
        stage["requirements"]["provenance"]["runtime_config_sha256"] = config_digest(stage["runtime_config"])
    result = check_deployment(b, now=NOW)
    assert result["ready"], result
    assert any("activations" in warning for warning in result["warnings"])


def test_transfer_summary_measures_concurrent_aggregate_not_sum_of_solo_bandwidths():
    result = summarize_transfers([{"gpu_uuid": "GPU-a", "samples_ms": [1, 1]},
                                 {"gpu_uuid": "GPU-b", "samples_ms": [1, 1]}], .010, 10**6, 2)
    assert result["aggregate_h2d_gb_s"] == pytest.approx(.4)
    assert result["devices"][0]["h2d_gb_s"] == pytest.approx(1)
    with pytest.raises(ValueError):
        summarize_transfers([{"gpu_uuid": "GPU-a", "samples_ms": [0]}], .1, 1024, 1)


def test_probe_refuses_unknown_ram_or_destructive_allocation_pressure():
    with pytest.raises(ValueError, match="unknown"):
        validate_probe([0], 1024, 2, 0, None)
    with pytest.raises(ValueError, match="80%"):
        validate_probe([0, 1], 1024, 2, 3000, 3000)
    with pytest.raises(ValueError, match="distinct"):
        validate_probe([0, 0], 1024, 2, 0, 4096)
    assert validate_probe([0, 1], 1024, 2, 3000, 4096) == 3000


def test_repeated_public_ip_diagnostic_does_not_reject_working_routes(monkeypatch):
    import preflight
    monkeypatch.setattr(preflight, "get_total_ram_gb", lambda: 32)
    monkeypatch.setattr(preflight, "get_disk_free_gb", lambda _: 50)
    monkeypatch.setattr(preflight, "measure_tcp_rtt_ms", lambda *args, **kw: 1)
    result = preflight.run_preflight_checks(endpoints=["198.51.100.1:29000", "198.51.100.1:29001"])
    assert result["ok"] and result["warnings"]
    monkeypatch.setattr(preflight, "measure_tcp_rtt_ms", lambda *args, **kw: None)
    with pytest.raises(RuntimeError, match="Unreachable"):
        preflight.run_preflight_checks(endpoints=["198.51.100.1:29000"])

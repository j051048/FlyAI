"""Real signature checks over explicitly synthetic benchmark/device fixtures."""
from copy import deepcopy
import pytest

from engines.deepseek_v4.v4_round_evidence import build_calibration, verify_calibration


@pytest.mark.parametrize("tools_first", [True, False])
def test_runtime_evidence_import_is_independent_of_cli_search_order(tools_first):
    import os
    from pathlib import Path
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    directories = [str(root / "phase0"), str(root / "engines/deepseek_v4")]
    if not tools_first:
        directories.reverse()
    script = ("import sys; sys.path[:0]=" + repr(directories + [str(root)]) + "; "
              "import v4_round_calibration as cli, v4_round_evidence as runtime; "
              "assert cli is not runtime; assert callable(cli.main); "
              "assert callable(runtime.verify_calibration); assert callable(runtime.CalibrationSlot); "
              "import v4_gateway; print('DISTINCT_CLI_RUNTIME_IMPORTED')")
    result = subprocess.run([sys.executable, "-c", script], cwd=root,
                            env={**os.environ, "PYTHONUTF8": "1"}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "DISTINCT_CLI_RUNTIME_IMPORTED" in result.stdout
from v4_adaptive_round import RoundPolicy, digest
from test_v4_benchmark import bench, observed_protocol, ObservedAdapter, Clock


class PrefixAdapter(ObservedAdapter):
    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        if kwargs["mode"] == "pipelined":
            exp = self.protocol["round_experiment"]
            result.update(draft_prefix_cap=exp["draft_prefix_cap"], loaded_block_limit=exp["loaded_block_size"],
                          predictive_dprev_disabled=True)
        return result


def evidence():
    base, keys = observed_protocol()
    entries = []
    for cap, rate in ((8, 40), (2, 50)):
        protocol = deepcopy(base)
        protocol["round_experiment"] = {"schema": bench.ROUND_EXPERIMENT_SCHEMA, "draft_prefix_cap": cap,
                                        "loaded_block_size": 8}
        protocol = bench.seal_protocol(protocol)
        clock = Clock()
        report = bench.run_suite(PrefixAdapter(protocol, keys, clock, rate=rate), protocol,
                                fresh_ring=True, clock=clock)
        entries.append({"protocol": protocol, "report": report})
    options = {"loaded_block_size": 8, "depth": 16, "floor": 1, "binding": {
        "lazy": True, "capabilities": {"ring_id": "test-swarm"},
        "source_sha256": digest({r["name"]: r["sha256"] for r in base["source"]["files"]}),
        "runtime_config_sha256_by_node": {row["node_id"]: row["runtime_config_sha256"] for row in base["hardware"]},
        "checkpoint_config_sha256": base["checkpoint"]["config_sha256"],
        "stages": [{"node_id": row["node_id"], "gpu_uuid": row["gpu_uuid"], "signer_pubkey": row["signer_pubkey"],
                    "lo": row["layer_start"], "hi": row["layer_end"]} for row in base["hardware"]]}}
    return entries, options


def test_only_reverified_raw_reports_can_be_promoted_to_compact_runtime_profiles():
    entries, options = evidence()
    result = bench.compare_reports(entries[0]["report"], entries[1]["report"], vary="round:draft_prefix_cap")
    assert result["status"] in ("verified_comparison", "verified_target_pass"), result
    bundle = build_calibration(entries, options)
    assert [row["draft_prefix_cap"] for row in bundle["profiles"]] == [2, 8]
    assert all(row["expert_dma_bytes"] is None for row in bundle["profiles"])
    compact = verify_calibration(bundle, options)
    assert "evidence" not in compact and len(compact["evidence_sha256"]) == 64
    policy = RoundPolicy({"enabled": True, "mode": "enforce", "calibration": compact}, **options)
    assert policy.refill(committed_tokens=2, available_drafts=8) == 2


@pytest.mark.parametrize("damage", ["signature", "profile", "protocol", "source", "cap", "ring", "mock"])
def test_profile_promotion_rejects_tampering_and_unverified_measurements(damage):
    entries, options = evidence()
    if damage == "profile":
        bundle = build_calibration(entries, options)
        bundle["profiles"][0]["elapsed_s"] /= 100
        with pytest.raises(ValueError, match="profiles differ"):
            verify_calibration(bundle, options)
        return
    if damage == "signature":
        entries[1]["report"]["samples"][0]["receipts"][0]["nonce"] = "f"*64
    elif damage == "protocol":
        entries[1]["protocol"]["run"]["max_new"] += 1
    elif damage == "source":
        options["binding"]["source_sha256"] = "a"*64
    elif damage == "cap":
        entries[1]["report"]["samples"][0]["coordinator_stats"]["draft_prefix_cap"] = 8
    elif damage == "ring":
        options["binding"]["capabilities"]["ring_id"] = "other-ring"
    elif damage == "mock":
        entries[1]["report"]["backend"] = "mock"
    with pytest.raises(ValueError):
        build_calibration(entries, options)


def test_fixed_prefix_is_a_real_frozen_comparison_variable():
    entries, options = evidence()
    result = bench.compare_reports(entries[0]["report"], entries[1]["report"])
    assert result["status"] == "unverified"
    assert result["changes"] == ["round:draft_prefix_cap"]
    with pytest.raises(ValueError, match="baseline"):
        build_calibration(entries[1:], options)


@pytest.mark.parametrize("cap", [-1, True, 9])
def test_invalid_static_experiment_never_bypasses_frozen_protocol_validation(cap):
    entries, _ = evidence()
    p = deepcopy(entries[0]["protocol"])
    p["round_experiment"]["draft_prefix_cap"] = cap
    assert bench.protocol_errors(bench.seal_protocol(p))


def test_explicit_static_cap_is_executed_on_real_fifo_sockets():
    from test_v4_pipe import _ScriptedPipeRing, TRUTH, PIPE_PROMPT, VP, _perfect_block
    for cap in (0, 1, 3):
        ring = _ScriptedPipeRing(TRUTH, _perfect_block); ring.lazy = True
        try:
            result = VP.coordinate_dspark_pipelined(ring.pipe_a, ring.ret_b, PIPE_PROMPT, 12,
                timeout=3, depth=8, floor=1, lazy=True, draft_block_limit=3, draft_prefix_cap=cap)
            assert result["tokens"] == list(range(103, 115))
            assert result["draft_prefix_cap"] == cap and result["predictive_dprev_disabled"]
            assert result["max_inflight"] <= cap+1
            assert not any(ring.dprev.values())
        finally:
            ring.close()


def _selected_service_fixture(monkeypatch):
    """Real signatures and selected metadata; GPU/artifact data are test fixtures."""
    from v4_adaptive_round import service_options
    from test_v4_gateway import Pipe, Tokenizer, GW
    import v4_observability
    base, keys = observed_protocol()
    source = {row["name"]: row["sha256"] for row in base["source"]["files"]}
    monkeypatch.setattr(v4_observability, "source_inventory", lambda: deepcopy(source))
    configs, leases, stages = {}, {}, []
    for row in base["hardware"]:
        lo, hi = row["layer_start"], row["layer_end"]
        cfg = {"lo": lo, "hi": hi, "head": lo == 0, "tail": hi == 43,
            "dspark": hi == 43, "dspark_loaded": hi == 43, "dspark_block_size": 8 if hi == 43 else None,
            "spec_depth": 16, "args": {"n_layers": 43, "max_seq_len": 8192, "dspark_block_size": 8, "n_mtp_layers": 3}}
        configs[row["node_id"]] = cfg
        leases[row["node_id"]] = {"lease_id": row["node_id"]+"-lease", "fencing_token": 1}
        row["runtime_config_sha256"] = digest(cfg)
        stages.append({name: row[name] for name in ("node_id", "gpu_uuid", "signer_pubkey", "runtime_config_sha256")} |
                      {name: cfg[name] for name in ("lo", "hi", "head", "tail")})
    plan = {"n_layers": 43, "cohort_id": "a"*64, "ring_id": "test-swarm", "stages": stages,
            "model_cohort": {"model_id": base["model_id"], "config_sha256": base["checkpoint"]["config_sha256"]}}
    options = service_options(plan, configs, leases, depth=16, floor=1, lazy=True)
    entries = []
    for cap in (8, 2):
        protocol = deepcopy(base)
        protocol["round_experiment"] = {"schema": bench.ROUND_EXPERIMENT_SCHEMA, "draft_prefix_cap": cap, "loaded_block_size": 8}
        protocol = bench.seal_protocol(protocol)
        clock = Clock()
        report = bench.run_suite(PrefixAdapter(protocol, keys, clock), protocol, fresh_ring=True, clock=clock)
        entries.append({"protocol": protocol, "report": report})
    bundle = build_calibration(entries, options)
    producer = Pipe()
    producer.V4_LAZY_DRAFT = True
    producer.verify_cohort_directory = lambda *_: {"test_fixture": True}
    assignments = {row["signer_pubkey"]: (row["lo"], row["hi"]) for row in stages}
    common = dict(vp=producer, tokenizer=Tokenizer(), pipeline_plan=plan,
                  calibrated_runtime_configs=configs, lease_fences=leases)
    return bundle, options, assignments, common


def test_gateway_admission_reverifies_full_raw_bundle_against_selected_calibrations(monkeypatch):
    from test_v4_gateway import GW
    bundle, options, assignments, common = _selected_service_fixture(monkeypatch)
    service = GW.V4RingBackend("test-fixture", "h:1", "t:2", assignments,
        round_speculation={"enabled": True, "calibration": bundle}, **common)
    try:
        assert service._round_config["calibration"] == verify_calibration(bundle, options)
        assert "evidence" not in service._round_config["calibration"]
        bundle["evidence"][0]["report"]["samples"][0]["receipts"][0]["nonce"] = "tampered"
        with pytest.raises(ValueError, match="verified raw"):
            GW.V4RingBackend("test-fixture", "h:1", "t:2", assignments,
                round_speculation={"enabled": True, "calibration": bundle}, **common)
    finally:
        service.close()


def test_local_slot_atomic_reload_retains_previous_policy_on_bad_identity_or_expiry(tmp_path, monkeypatch):
    import json
    import time
    from test_v4_gateway import GW
    from engines.deepseek_v4.v4_round_evidence import publish_slot
    bundle, _, assignments, common = _selected_service_fixture(monkeypatch)
    path = tmp_path / "private" / "round.json"
    service = GW.V4RingBackend("test-fixture", "h:1", "t:2", assignments,
        round_calibration_file=path, **common)
    # Actual request-boundary hook, isolated from an explicit fake model.
    service._execute = lambda *_: service.stats()["round_speculation_mode"]
    try:
        assert service.execute(None, None, None) == "disabled"
        assert service.stats()["round_calibration_reload"]["status"] == "waiting_for_local_file"
        publish_slot(path, {"enabled": True, "calibration": bundle})
        assert service.execute(None, None, None) == "shadow"
        approved = service._round_config
        assert service.stats()["round_calibration_reload"]["accepted_updates"] == 1
        for damage in ("identity", "expiry", "json", "checksum"):
            changed = deepcopy(bundle)
            if damage == "identity":
                changed["binding_sha256"] = "0"*64
            elif damage == "expiry":
                changed.update(created_at=time.time()-20, expires_at=time.time()-10)
            publish_slot(path, {"enabled": True, "mode": "enforce", "calibration": changed})
            if damage == "json":
                path.write_bytes(b"{unfinished")
            elif damage == "checksum":
                value = json.loads(path.read_bytes()); value["calibration_sha256"] = "0"*64
                path.write_text(json.dumps(value))
            assert service.execute(None, None, None) == "shadow"
            assert service._round_config is approved
            assert service.stats()["round_calibration_reload"]["status"] == "rejected_update"
        publish_slot(path, {"enabled": False})
        assert service.execute(None, None, None) == "disabled"
        assert service.stats()["round_calibration_reload"]["accepted_updates"] == 2
    finally:
        service.close()


def test_local_slot_never_changes_an_executing_request(tmp_path, monkeypatch):
    import threading
    from test_v4_gateway import GW
    from engines.deepseek_v4.v4_round_evidence import publish_slot
    bundle, _, assignments, common = _selected_service_fixture(monkeypatch)
    path = tmp_path / "private" / "round.json"
    publish_slot(path, {"enabled": True, "mode": "shadow", "calibration": bundle})
    service = GW.V4RingBackend("test-fixture", "h:1", "t:2", assignments,
        round_calibration_file=path, **common)
    entered, release = threading.Event(), threading.Event()
    observed = []
    def execute(*_):
        entered.set()
        assert release.wait(3)
        observed.append(service._round_config["mode"])
    service._execute = execute
    worker = threading.Thread(target=lambda: service.execute(None, None, None))
    try:
        worker.start(); assert entered.wait(3)
        publish_slot(path, {"enabled": True, "mode": "enforce", "calibration": bundle})
        assert service._round_config["mode"] == "shadow"
        release.set(); worker.join(3)
        assert not worker.is_alive() and observed == ["shadow"]
        service.execute(None, None, None)
        assert observed == ["shadow", "enforce"]
    finally:
        release.set(); worker.join(3)
        service.close()


def test_slot_rejects_oversized_and_nonlocal_input(tmp_path):
    from engines.deepseek_v4.v4_round_evidence import CalibrationSlot, MAX_SLOT_BYTES
    with pytest.raises(ValueError, match="absolute local"):
        CalibrationSlot("https://example.invalid/profile.json")
    path = tmp_path / "large.json"
    with path.open("wb") as stream:
        stream.truncate(MAX_SLOT_BYTES+1)
    path.chmod(0o600)
    called = []
    slot = CalibrationSlot(path); slot.poll(lambda cfg: called.append(cfg))
    assert not called and slot.status()["status"] == "rejected_update"


def test_operator_cli_publishes_default_shadow_from_exported_live_binding(tmp_path):
    import json
    from phase0.v4_round_calibration import main
    from engines.deepseek_v4.v4_round_evidence import publish_binding, CalibrationSlot
    entries, options = evidence()
    reports, protocols = [], []
    for index, entry in enumerate(entries):
        protocol, report = tmp_path/f"protocol-{index}.json", tmp_path/f"report-{index}.json"
        protocol.write_text(json.dumps(entry["protocol"])); report.write_text(json.dumps(entry["report"]))
        protocols.append(str(protocol)); reports.append(str(report))
    path = tmp_path/"private"/"round.json"
    publish_binding(path, options)
    assert main(["--reports", *reports, "--protocols", *protocols,
        "--binding", str(path)+".binding.json", "--publish-slot", "--out", str(path)]) == 0
    configs = []
    def install(config):
        configs.append(config)
        verify_calibration(config["calibration"], options)
        return {"mode": config["mode"]}
    slot = CalibrationSlot(path); slot.poll(install)
    assert slot.status()["status"] == "accepted" and configs[0]["mode"] == "shadow"


def test_posix_public_slot_cannot_be_published_or_loaded(tmp_path):
    import os
    from engines.deepseek_v4.v4_round_evidence import CalibrationSlot, publish_slot
    if os.name != "posix":
        pytest.skip("NTFS ACLs are operator-managed; Unix mode test")
    path = tmp_path / "public" / "round.json"
    path.parent.mkdir(mode=0o755); path.parent.chmod(0o755)
    with pytest.raises(ValueError, match="private"):
        publish_slot(path, {"enabled": False})
    slot = CalibrationSlot(path); slot.poll(lambda _: {})
    assert slot.status()["status"] == "rejected_update"

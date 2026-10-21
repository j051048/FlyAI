"""CPU correctness and synthetic-signature verifier tests; no hardware speed claim."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import time

import pytest

from engines.deepseek_v4.v4_confidence_calibration import (
    ConfidenceBudget, FIT_SCHEMA, PROBE_SCHEMA, SCHEMA, ProbeRecorder, build_calibration,
    fit_statistics, isotonic_fit, predict, verify_calibration,
)
from v4_adaptive_round import RoundPolicy, calibration_binding, digest


OPTIONS = {"binding": {"cohort": "a"*64, "runtime": "b"*64, "source": "c"*64},
           "loaded_block_size": 3, "depth": 8, "floor": 1}
PROMPT = [1, 2, 3]


def requests():
    rows = [{"position": pos, "raw_conf": score, "accepted": accepted}
            for pos in (1, 2, 3) for _ in range(32)
            for score, accepted in ((-4., False), (-1., True))]
    return [{"job_id": f"independent-request-{i}", "probe": {
        "schema": PROBE_SCHEMA, "labels": deepcopy(rows), "refills": 64, "dropped_labels": 0}}
            for i in range(8)]


def compact(*, now=None):
    """Isolated controller input, intentionally synthetic; never hardware evidence."""
    now = time.time() if now is None else now
    return {"schema": SCHEMA, "binding_sha256": calibration_binding(OPTIONS),
        "created_at": now-1, "expires_at": now+60, "statistics": fit_statistics(requests(), block_size=3),
        "costs": [{"workload": "fixture", "prompt_sha256": digest(PROMPT), "max_new": 18,
            "draft_prefix_cap": cap, "refills": 20, "wall_s": 20*(.1+cap*.05),
            "wall_per_refill_s": .1+cap*.05, "expert_dma_bytes_per_refill": None,
            "p95_ttft_s": .5, "p95_tpot_s": .1} for cap in range(4)],
        "limits": {"max_p95_ttft_s": 1., "max_p95_tpot_s": .2, "max_heldout_ece": .1},
        "gpu_qualified": True, "evidence_sha256": "e"*64,
        "verification": "raw_benchmark_reverified"}


def budget(value=None, **kwargs):
    result = ConfidenceBudget(value or compact(), binding_sha256=calibration_binding(OPTIONS), baseline=3, **kwargs)
    result.bind_request(PROMPT, 18)
    return result


def policy(mode="enforce", value=None):
    from test_v4_adaptive_round import config
    settings = config(mode=mode)
    settings["confidence_calibration"] = value or compact()
    return RoundPolicy(settings, **OPTIONS)


def test_negative_raw_logits_are_empirically_fitted_without_a_sigmoid():
    model = isotonic_fit([(-9., 0), (-5., 1), (-5., 0), (-2., 1)])
    assert predict(model, -9.) == 0.
    assert predict(model, -5.) == .5
    assert predict(model, -2.) == 1.
    assert predict(model, -10.) is None and predict(model, 0.) is None
    pooled = isotonic_fit([(-3., 1), (-2., 0), (-1., 1)])
    assert pooled["knots"] == [[-2., .5], [-1., 1.]]


def test_whole_request_holdout_is_independent_and_reliability_uses_heldout_only():
    probes = requests()
    original = fit_statistics(probes, block_size=3)
    heldout = set(original["heldout_job_ids"])
    assert not heldout.intersection(original["training_job_ids"])
    assert original["hardware_qualified"] is False
    for request in probes:
        if request["job_id"] in heldout:
            for row in request["probe"]["labels"]:
                row["accepted"] = not row["accepted"]
    changed = fit_statistics(probes, block_size=3)
    for position in ("1", "2", "3"):
        assert original["positions"][position]["model"] == changed["positions"][position]["model"]
        assert original["positions"][position]["heldout"]["ece"] == 0
        assert changed["positions"][position]["heldout"]["ece"] == 1


def test_prefix_objective_uses_calibrated_survival_and_measured_wall_including_wan():
    gate = budget(mode="enforce")
    assert gate.select([-4., -1., -1.], 3) == 0
    assert gate.select([-1., -1., -1.], 3) == 3
    first, second = gate.report()["records"]
    assert first["expected_committed"] == 1 and second["expected_committed"] == 4
    assert second["expected_rate"] == pytest.approx(4/.25)
    # Price a measured transport-dominated cap directly. Acceptance alone must
    # not keep it when its actual full traversal cost is worse.
    value = compact()
    value["costs"][3].update(wall_s=40., wall_per_refill_s=2.)
    assert budget(value).select([-1., -1., -1.], 3) == 2


@pytest.mark.parametrize("damage", [
    lambda c: c.update(binding_sha256="f"*64),
    lambda c: c.pop("verification"),
    lambda c: c.update(gpu_qualified=1),
    lambda c: c["statistics"]["heldout_job_ids"].append(c["statistics"]["training_job_ids"][0]),
    lambda c: c["statistics"]["positions"]["1"]["model"]["knots"][0].__setitem__(1, -1.),
    lambda c: c["costs"][0].update(wall_per_refill_s=0.),
    lambda c: c["costs"][0].update(expert_dma_bytes_per_refill=float("nan")),
    lambda c: c["limits"].update(max_p95_tpot_s=0.),
])
def test_invalid_or_unverified_compact_metadata_is_rejected(damage):
    value = compact(); damage(value)
    with pytest.raises(ValueError):
        budget(value)


def test_expired_unseen_unreliable_or_unqualified_inputs_keep_unchanged_cap():
    gate = budget(compact(now=10.), wall_clock=lambda: 100.)
    assert gate.select([-4., -1., -1.], 3) == 3
    gate = budget(); gate.bind_request([999], 18)
    assert gate.select([-4., -1., -1.], 3) == 3
    gate.bind_request(PROMPT, 19)
    assert gate.select([-4., -1., -1.], 3) == 3
    for key, value in (("gpu_qualified", False),):
        data = compact(); data[key] = value
        assert budget(data).select([-4., -1., -1.], 3) == 3
    data = compact(); data["statistics"]["positions"]["1"]["heldout"]["ece"] = .2
    assert budget(data).select([-4., -1., -1.], 3) == 3
    assert budget().select([-100., -1., -1.], 3) == 3
    assert budget().select([float("nan"), -1., -1.], 3) == 3
    assert budget().select([], 3) == 3


def test_p95_and_unknown_dma_cannot_pass_constraints_as_zero_cost():
    data = compact()
    for row in data["costs"]:
        row["p95_tpot_s"] = .3
    assert budget(data).select([-4., -1., -1.], 3) == 3
    gate = budget(dma_limit=100.)
    assert gate.select([-4., -1., -1.], 3) == 3
    assert gate.report()["unknown_expert_dma_is_zero"] is False
    data = compact()
    for row in data["costs"]:
        row["expert_dma_bytes_per_refill"] = 0.
    assert budget(data, dma_limit=100.).select([-4., -1., -1.], 3) == 0


def test_default_shadow_keeps_baseline_and_records_real_counterfactual_choice():
    selected = policy("shadow")
    selected.bind_request(PROMPT, 18)
    assert selected.refill(committed_tokens=2, available_drafts=3, raw_conf=[-4., -1., -1.]) == 3
    report = selected.report()
    assert report["records"][-1]["selected_cap"] == 0
    assert report["confidence"]["records"][-1]["applied_cap"] == 3


def test_bounded_probe_records_only_observed_reached_positions():
    recorder = ProbeRecorder(True, limit=2)
    recorder.refill()
    recorder.observe(1, -4., False)
    recorder.observe(2, None, True)
    recorder.observe(1, -1., True)
    recorder.observe(1, -1., False)
    report = recorder.report()
    assert report["labels"] == [{"position": 1, "raw_conf": -4., "accepted": False},
                                {"position": 1, "raw_conf": -1., "accepted": True}]
    assert report["dropped_labels"] == 1 and report["refills"] == 1


def signed_probe_evidence():
    """Actual fixture Ed25519 signatures; entirely synthetic CPU labels/timings."""
    from test_v4_round_calibration import evidence
    entries, options = evidence()
    for entry in entries:
        for sample in entry["report"]["samples"]:
            if sample["phase"] != "warm":
                continue
            stats = sample["coordinator_stats"]
            labels = [{"position": 1, "raw_conf": -1. if i < 3 else -4., "accepted": i < 3} for i in range(7)]
            stats["confidence_probe"] = {"schema": PROBE_SCHEMA, "labels": labels,
                "refills": 7, "dropped_labels": 0}
            stats["accept_by_depth"] = {"1": [3, 7]}
    return entries, options


LIMITS = {"max_p95_ttft_s": 10., "max_p95_tpot_s": 1., "max_heldout_ece": .1}


def test_raw_signed_reports_are_reverified_separately_from_calibration_statistics():
    entries, options = signed_probe_evidence()
    full = build_calibration(entries, options, limits=LIMITS)
    checked = verify_calibration(full, options)
    assert checked["verification"] == "raw_benchmark_reverified" and "evidence" not in checked
    assert checked["statistics"]["positions"]["1"]["heldout"]["ece"] == 0.
    assert checked["gpu_qualified"] is False, "signed fixture declarations do not establish executed GPU work"
    assert all(row["expert_dma_bytes_per_refill"] is None for row in checked["costs"])
    assert all(row["wall_per_refill_s"] > 0 for row in checked["costs"])
    gate = ConfidenceBudget(checked, binding_sha256=calibration_binding(options), baseline=8, mode="enforce")
    prompt = entries[0]["protocol"]["prompts"][0]["token_ids"]
    gate.bind_request(prompt, 8)
    assert gate.select([-4.]*8, 8) == 8


@pytest.mark.parametrize("damage", ["signature", "source", "lease", "fit", "cost", "probe_counts", "missing_probe"])
def test_promotion_rejects_tampering_arbitrary_telemetry_or_another_binding(damage):
    entries, options = signed_probe_evidence()
    full = build_calibration(entries, options, limits=LIMITS)
    if damage == "signature":
        full["evidence"][0]["report"]["samples"][0]["receipts"][0]["nonce"] = "f"*64
    elif damage == "source":
        options["binding"]["source_sha256"] = "f"*64
    elif damage == "lease":
        options["binding"]["capabilities"]["ring_generation"] = "new-lease-generation"
    elif damage == "fit":
        full["statistics"]["positions"]["1"]["model"]["knots"][0][1] = .9
    elif damage == "cost":
        full["costs"][0]["wall_per_refill_s"] = 1e-12
    elif damage == "probe_counts":
        full["evidence"][0]["report"]["samples"][1]["coordinator_stats"]["confidence_probe"]["refills"] = 9999
        full["evidence_sha256"] = digest(full["evidence"])
    else:
        full["evidence"][0]["report"]["samples"][1]["coordinator_stats"].pop("confidence_probe")
        full["evidence_sha256"] = digest(full["evidence"])
    with pytest.raises(ValueError):
        verify_calibration(full, options)


def test_cli_is_distinct_from_runtime_and_defaults_to_shadow(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("confidence_cli", root / "phase0/v4_confidence_calibration.py")
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    entries, options = signed_probe_evidence()
    from v4_round_evidence import BINDING_SCHEMA
    binding = {"schema": BINDING_SCHEMA, "options": options, "binding_sha256": calibration_binding(options)}
    paths = []
    for i, entry in enumerate(entries):
        report_path, protocol_path = tmp_path/f"r{i}.json", tmp_path/f"p{i}.json"
        report_path.write_text(json.dumps(entry["report"]), encoding="utf-8")
        protocol_path.write_text(json.dumps(entry["protocol"]), encoding="utf-8")
        paths.append((report_path, protocol_path))
    binding_path = tmp_path/"binding.json"; binding_path.write_text(json.dumps(binding), encoding="utf-8")
    out = tmp_path/"out.json"
    args = ["--reports", *[str(p[0]) for p in paths], "--protocols", *[str(p[1]) for p in paths],
        "--binding", str(binding_path), "--max-p95-ttft-s", "10", "--max-p95-tpot-s", "1", "--out", str(out)]
    assert cli.main(args) == 0
    assert json.loads(out.read_text())["mode"] == "shadow"
    assert cli.main(args+["--mode", "enforce"]) == 2
    assert "cannot qualify enforcement" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
@pytest.mark.parametrize("lazy", [False, True])
def test_actual_fifo_prefix_selection_changes_only_unsent_candidates(monkeypatch, mode, lazy):
    from test_v4_pipe import _ScriptedPipeRing, TRUTH, PIPE_PROMPT, VP, _perfect_block
    original = VP.recv_msg
    def receive(sock):
        reply = original(sock)
        if isinstance(reply, dict) and "draft" in reply:
            reply["conf"] = [-4.]*len(reply["draft"])
        return reply
    monkeypatch.setattr(VP, "recv_msg", receive)
    selected = policy(mode)
    ring = _ScriptedPipeRing(TRUTH, _perfect_block); ring.lazy = lazy
    try:
        result = VP.coordinate_dspark_pipelined(ring.pipe_a, ring.ret_b, PIPE_PROMPT, 18,
            depth=8, floor=1, lazy=lazy, draft_block_limit=3, round_policy=selected, timeout=5)
    finally:
        ring.close()
    assert result["tokens"] == [TRUTH[p] for p in range(3, 21)]
    assert result["depth"] == 8 and result["floor"] == 1 and result["block_len"] == 3
    assert all(len(ids) == 1 for ids, pos, _ in ring.frames if pos)
    if mode == "enforce":
        assert result["drafted"] == 0
        assert not any(ring.dprev.values())
    else:
        assert result["drafted"] > 0


def test_actual_probe_never_labels_confidence_beyond_the_first_rejected_candidate(monkeypatch):
    from test_v4_pipe import _ScriptedPipeRing, TRUTH, PIPE_PROMPT, VP
    original = VP.recv_msg
    def receive(sock):
        reply = original(sock)
        if isinstance(reply, dict) and "draft" in reply:
            reply["conf"] = [-4., 10., 999.]
        return reply
    monkeypatch.setattr(VP, "recv_msg", receive)
    ring = _ScriptedPipeRing(TRUTH, lambda _: [901, 902, 903])
    try:
        result = VP.coordinate_dspark_pipelined(ring.pipe_a, ring.ret_b, PIPE_PROMPT, 12,
            depth=8, floor=1, lazy=False, draft_block_limit=3, draft_prefix_cap=3, timeout=5)
    finally:
        ring.close()
    probe = result["confidence_probe"]
    assert probe["labels"]
    assert all(row == {"position": 1, "raw_conf": -4., "accepted": False} for row in probe["labels"])
    assert result["cancels"] > 0 and result["stale_replies"] > 0
    assert len(probe["labels"]) == result["accept_by_depth"][1][1]


@pytest.fixture(scope="module")
def real_model(tmp_path_factory):
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    from test_v4_pipe import VP, PROMPT as real_prompt, NEW
    reference = pytest.importorskip("v4_ref_cpu")
    pytest.importorskip("v4_stage"); pytest.importorskip("safetensors.torch")
    try:
        args = reference.cpu_args()
        # Seed one produces genuinely negative reached-position head logits.
        model = reference.build_oracle(args, seed=1)
        directory = str(tmp_path_factory.mktemp("confidence-real-cpu"))
        VP._write_tiny_checkpoint(directory, args, model)
        yield directory, args, VP._reference_tokens(model, real_prompt, NEW)
    finally:
        torch.set_num_threads(previous)


def test_real_cpu_model_strict_signed_ring_keeps_greedy_tokens_and_reset_barriers(real_model):
    """Real V4 weights/rollback/sockets/signatures, synthetic budget only; no speed qualification."""
    from test_v4_pipe import _Ring, VP, PROMPT as real_prompt, NEW
    directory, args, oracle = real_model
    value = compact()
    for row in value["costs"]:
        row.update(prompt_sha256=digest(real_prompt), max_new=NEW)
    # An isolated synthetic controller predicts rejection across the actual raw
    # score support. The model scores and all committed tokens remain genuine.
    for row in value["statistics"]["positions"].values():
        row["model"] = {"minimum": -1000., "maximum": 1000., "knots": [[1000., 0.]], "observations": 100}
    ring = _Ring(directory, args, VP._dspark_tiling(args, 3), receipts=True, dspark=True)
    try:
        baseline = ring.coordinate(real_prompt, NEW, nonce="cpu-confidence-baseline")
        assignments = {row["pubkey"]: (row["layer_start"], row["layer_end"]) for row in baseline["receipts"]}
        selected = policy("enforce", value)
        trimmed = ring.pipelined(real_prompt, NEW, nonce="a"*64, swarm_id="swarm", job_id="cpu-confidence-test",
            expected_by_signer=assignments, strict_job_binding=True, round_policy=selected,
            draft_block_limit=3, depth=8, floor=1, lazy=True, confidence_probe=True)
        reset = ring.pipelined(real_prompt, NEW, nonce="b"*64, swarm_id="swarm", job_id="cpu-confidence-reset",
            expected_by_signer=assignments, strict_job_binding=True, depth=8, floor=1, lazy=True,
            confidence_probe=True)
    finally:
        ring.close()
    assert baseline["tokens"] == trimmed["tokens"] == reset["tokens"] == oracle
    assert baseline["receipts_ok"] and trimmed["receipts_ok"] and reset["receipts_ok"]
    assert trimmed["drafted"] == 0 and trimmed["block_len"] == args.dspark_block_size == 3
    assert trimmed["round_speculation"]["confidence"]["records"][-1]["selected_cap"] == 0
    assert reset["confidence_probe"]["labels"], "real raw model scores must reach the probe after the next reset"
    assert any(row["raw_conf"] < 0 for row in reset["confidence_probe"]["labels"])

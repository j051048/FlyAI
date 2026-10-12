"""Controlled strategy evidence and real two-stage CPU strict lifecycle execution."""
from copy import deepcopy
import gc
import json

import pytest

import v4_benchmark as bench
import v4_soak as soak
from test_v4_benchmark import observed_protocol, observed_report, reseal_observed_protocol
from test_v4_conversation_protocol import strict_tiny_ring


def strategy(protocol, mode, *, width=4, order=3):
    result = deepcopy(protocol)
    result["run"]["mode"] = mode
    for name in ("ngram_k", "ngram_order"): result["run"].pop(name, None)
    if mode == "spec": result["run"].update(ngram_k=width, ngram_order=order)
    return bench.seal_protocol(result)


@pytest.mark.parametrize("mode", ["greedy", "spec", "dspark"])
def test_mode_comparison_freezes_every_non_strategy_variable(mode):
    p, keys = observed_protocol()
    q = strategy(p, mode)
    before, after = observed_report(p, keys), observed_report(q, keys)
    assert bench.compare_reports(before, after)["status"] == "unverified"
    result = bench.compare_reports(before, after, vary="mode")
    assert result["status"] == "verified_target_pass", result
    assert result["changes"] == ["mode"]


@pytest.mark.parametrize("mutation", ["source", "env", "driver", "network", "tokens", "length"])
def test_mode_comparison_refuses_other_changes_and_output_divergence(mutation):
    p, keys = observed_protocol(); q = strategy(p, "spec")
    if mutation == "source": q["source"]["files"][0]["sha256"] = "e" * 64
    elif mutation == "env": q["env"]["V4_SPEC_DEPTH"] = "8"
    elif mutation == "driver": q["hardware"][0]["driver"] = "changed"
    elif mutation == "network": q["network_comparison"]["route_identity_sha256"] = "b" * 64
    elif mutation == "length": q["run"]["max_new"] += 1
    q = reseal_observed_protocol(q)
    before, after = observed_report(p, keys), observed_report(q, keys)
    if mutation == "tokens":
        for sample in after["samples"]:
            sample["tokens"][0] += 1; sample["output_sha256"] = bench.digest(sample["tokens"])
    assert bench.compare_reports(before, after, vary="mode")["status"] == "unverified"


def test_ngram_width_order_are_frozen_and_change_only_with_declared_mode():
    p, keys = observed_protocol(); p = strategy(p, "spec")
    q = strategy(p, "spec", width=2, order=2)
    assert bench.compare_reports(observed_report(p, keys), observed_report(q, keys), vary="mode")["changes"] == ["mode"]
    for name, value in (("ngram_k", True), ("ngram_k", 65), ("ngram_order", 0), ("ngram_order", 17)):
        bad = deepcopy(p); bad["run"][name] = value
        assert bench.protocol_errors(bench.seal_protocol(bad))
    bad = deepcopy(p); bad["run"]["mode"] = "greedy"
    assert bench.protocol_errors(bench.seal_protocol(bad))


def test_prepare_cli_freezes_spec_parameters_and_proposer_source(tmp_path, monkeypatch):
    from engines.deepseek_v4 import v4_tokenizer
    p, _ = observed_protocol()
    class Tokenizer:
        def encode(self, text, **kwargs): return list(text.encode())
    monkeypatch.setattr(v4_tokenizer, "load_v4_tokenizer", lambda path: Tokenizer())
    real_source = bench.source_identity()
    assert "phase0/ngram_draft.py" in {row["name"] for row in real_source["files"]}
    monkeypatch.setattr(bench, "checkpoint_inventory", lambda path: p["checkpoint"])
    monkeypatch.setattr(bench, "source_identity", lambda: p["source"])
    hardware, environment, output = (tmp_path / name for name in ("hardware.json", "env.json", "protocol.json"))
    hardware.write_text(json.dumps(p["hardware"])); environment.write_text(json.dumps(p["env"]))
    assert bench.main(["prepare", "--dir", str(tmp_path), "--hardware", str(hardware),
        "--env-file", str(environment), "--mode", "spec", "--ngram-k", "7", "--ngram-order", "5",
        "--max-new", "8", "--out", str(output)]) == 0
    prepared = json.loads(output.read_text())
    assert prepared["run"]["ngram_k"] == 7 and prepared["run"]["ngram_order"] == 5
    assert prepared["run"]["mode"] == "spec" and not bench.protocol_errors(prepared)


@pytest.mark.parametrize("strict_tiny_ring", [True], indirect=True)
@pytest.mark.parametrize("buffered", [False, True])
def test_real_strict_two_stage_matrix_all_modes_cancel_reconnect_and_rollback(strict_tiny_ring, monkeypatch, buffered):
    import torch
    import v4_stage
    ring = strict_tiny_ring
    configure_receive_pool = ring.VP.configure_transport_buffers
    receive_pool_status = ring.VP.transport_buffer_status
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    rewind_spans = set()
    seek = v4_stage.Stage._seek
    def track(stage, position):
        if position < stage._pos: rewind_spans.add((stage.lo, stage.hi))
        return seek(stage, position)
    monkeypatch.setattr(v4_stage.Stage, "_seek", track)
    if buffered: configure_receive_pool(8 << 20, slots=2, pinned=False)
    p = bench.seal_protocol({"layer_count": 4, "run": {"max_new": 6, "mode": "pipelined"},
        "env": {"V4_SPEC_DEPTH": "4", "V4_LAZY_DRAFT": "1", "V4_REFILL_FLOOR": "1"},
        "prompts": [{"token_ids": ring.prompt}],
        "hardware": [{"signer_pubkey": key, "layer_start": lo, "layer_end": hi}
                     for key, (lo, hi) in ring.expected.items()]})
    # Constructor's production checkpoint/RTX5090 admission is intentionally not
    # replaced with fake evidence. This is an actual CPU model and strict sockets.
    adapter = object.__new__(bench.LiveRingAdapter)
    adapter.backend = "cpu_native_strict"
    adapter.artifact_verification = {}
    adapter.vp, adapter.protocol, adapter.timeout = ring.VP, p, 20
    adapter.swarm_id, adapter.layer_count = "lifecycle-cpu", 4
    adapter.pipe, adapter.ret = ring.pipe, ring.ret
    adapter.session_options = {"session_config": ring.coordinator}
    plan = ring.coordinator.plan
    adapter.head, adapter.tail = plan["coordinator"]["head"], plan["coordinator"]["tail"]
    adapter.retry_s = 5
    contract = soak.lifecycle_contract(2, ngram_k=2, ngram_order=2)
    try:
        report = soak.run_lifecycle(adapter, p, contract=contract)
        evaluation = soak.evaluate_lifecycle(report, p, contract)
        assert report["complete"], report.get("failure")
        assert evaluation["errors"] == [], evaluation
        assert evaluation["status"] == "unverified"  # CPU evidence never GPU qualification.
        assert evaluation["cases"] == 45
        assert rewind_spans == {(0, 1), (1, 4)}
        for action in ("reconnect", "cancel", "forced_rollback", "recovery"):
            altered = deepcopy(report)
            row = next(x for x in altered["cases"] if x["action"] == action)
            if action == "reconnect": row["new_owner"] = deepcopy(row["previous_owner"])
            elif action == "forced_rollback": row["statistics"]["accept_hist"] = {}
            else: row["tokens"][0] += 1
            assert soak.evaluate_lifecycle(altered, p, contract)["status"] == "failed"
        if buffered:
            assert receive_pool_status()["borrowed_frames"] > 2
    finally:
        try: ring.VP.send_msg(adapter.pipe, {"op": "stop"})
        except OSError: pass
        adapter.close()
        torch.set_num_threads(old_threads)
        # Stage threads need to relinquish their last received tensor views.
        if buffered:
            import time
            deadline = time.monotonic() + 5
            while receive_pool_status()["live_slots"] and time.monotonic() < deadline:
                gc.collect(); time.sleep(.01)
            configure_receive_pool(0)

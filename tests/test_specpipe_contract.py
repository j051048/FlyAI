"""Actual coordinate_pipe against an explicit deterministic oracle, never GPU evidence."""
from collections import deque
from types import SimpleNamespace
import time
import os
from pathlib import Path
import subprocess
import sys
import socket

import pytest
import torch

import specpipe as SP


class Tokenizer:
    eos_token_id = 999
    def apply_chat_template(self, *args, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]])}
    def decode(self, ids, **kwargs):
        return ",".join(str(token) for token in ids)


class Draft:
    def __init__(self, teacher, *, match=True):
        self.teacher, self.match = teacher, match
        self.pending = None
        self.accepted = []
    def request(self, ids, k):
        self.pending = (list(ids), k)
    def propose(self, ids, k):
        start = len(ids) - 3
        return self.teacher[start:start + k] if self.match else [2000] * k
    def fetch(self):
        return self.propose(*self.pending)
    def note_accepted(self, n):
        self.accepted.append(n)


class Ring:
    def __init__(self, teacher):
        self.teacher, self.replies = teacher, deque()
        self.messages = []
    def settimeout(self, timeout):
        pass
    def send(self, sock, msg):
        self.messages.append(dict(msg))
        if msg["op"] == "reset":
            self.replies.clear(); self.replies.append("ok")
        elif msg["op"] == "verify":
            if msg.get("prefill"):
                generated = max(0, msg["start"] + len(msg["token_ids"]) - 3)
                self.replies.append([self.teacher[generated]])
            else:
                index = msg["start"] - 3 + 1
                self.replies.append(self.teacher[index:index + len(msg["token_ids"])])
    def receive(self, sock):
        assert self.replies, "unexpected read without an actual queued frame"
        return self.replies.popleft()


def run(monkeypatch, *, maximum=16, K=4, depth=3, match=True, resume=None, adaptive=False, adaptive_depth=False, teacher=None, callback=None):
    teacher = teacher or list(range(65, 400))
    ring = Ring(teacher)
    monkeypatch.setattr(SP, "send_msg", ring.send)
    monkeypatch.setattr(SP, "recv_msg", ring.receive)
    result = SP.coordinate_pipe(None, ring, Tokenizer(), "prompt", K, maximum, 2, depth,
        ret_sock=ring, local_draft=Draft(teacher, match=match), resume_ids=resume,
        adaptive_pipe=adaptive, adaptive_depth=adaptive_depth, on_commit=callback)
    return result, ring


def test_full_acceptance_gain_is_K_and_commit_callback_is_capped_before_publication(monkeypatch):
    events = []
    result, ring = run(monkeypatch, maximum=9, K=4, callback=lambda event: events.append(event))
    assert result["output_ids"] == list(range(65, 74))
    assert result["rounds"] == 2 and result["toks_per_traversal"] == 4
    assert result["metrics"]["new_decode_tokens"] == 8
    assert all(len(event["out"]) <= 9 for event in events)
    assert events[0]["phase"] == "prefilled" and events[0]["out"] == [65]
    assert not ring.replies


def test_rejection_and_plain_path_produce_exactly_the_same_gold_stream(monkeypatch):
    rejected, ring = run(monkeypatch, maximum=25, K=8, depth=4, match=False)
    plain, _ = run(monkeypatch, maximum=25, K=0, depth=1)
    assert rejected["output_ids"] == plain["output_ids"] == list(range(65, 90))
    assert rejected["metrics"]["stale_frames"] > 0 and plain["toks_per_traversal"] == 1


def test_resume_and_first_prefill_token_are_excluded_from_decode_throughput(monkeypatch):
    result, _ = run(monkeypatch, maximum=12, K=4, resume=list(range(65, 70)))
    assert result["output_ids"] == list(range(65, 77))
    assert result["resume_tokens"] == 5
    assert result["metrics"]["new_tokens"] == 7
    assert result["new_decode_tokens"] == 6
    assert result["tok_s"] == pytest.approx(6 / result["decode_s"])


def test_eos_in_an_accepted_chunk_never_publishes_eos_or_later_tokens(monkeypatch):
    events = []
    teacher = list(range(65, 400)); teacher[3] = Tokenizer.eos_token_id
    result, _ = run(monkeypatch, maximum=25, K=8, teacher=teacher, callback=lambda e: events.append(e))
    assert result["output_ids"] == [65, 66, 67]
    assert all(999 not in event["out"] and len(event["out"]) <= 3 for event in events)
    assert result["new_decode_tokens"] == 2


def test_first_token_eos_or_zero_budget_commits_nothing(monkeypatch):
    teacher = [999] + list(range(65, 400))
    result, _ = run(monkeypatch, teacher=teacher)
    assert result["output_ids"] == [] and result["tok_s"] == 0
    empty, ring = run(monkeypatch, maximum=0)
    assert empty["output_ids"] == [] and ring.messages == []


def test_adaptive_low_accept_switches_to_real_plain_path_only_at_drained_boundary(monkeypatch):
    result, ring = run(monkeypatch, maximum=32, K=8, depth=4, match=False, adaptive=True)
    assert result["output_ids"] == list(range(65, 97))
    assert result["adaptive"]["verified_buckets"] == [0, 1, 2, 8]
    assert result["adaptive"]["changes"][0]["K"] == 0
    assert result["adaptive"]["changes"][0]["boundary"] == "empty_pipeline"
    assert result["metrics"]["shape_warmup_frames"] > 0
    assert result["K"] == 0 and any(len(msg.get("token_ids", [])) == 1 for msg in ring.messages if msg["op"] == "verify")


def test_live_shape_gate_declines_a_bucket_that_changes_the_actual_teacher_stream(monkeypatch):
    teacher = list(range(65, 400)); ring = Ring(teacher)
    def send(sock, msg):
        ring.send(sock, msg)
        if msg["op"] == "verify" and not msg.get("prefill") and len(msg["token_ids"]) == 9:
            reply = ring.replies.pop(); reply[0] = 888; ring.replies.append(reply)
    monkeypatch.setattr(SP, "send_msg", send); monkeypatch.setattr(SP, "recv_msg", ring.receive)
    result = SP.coordinate_pipe(None, ring, Tokenizer(), "prompt", 8, 24, 2, 3,
        ret_sock=ring, local_draft=Draft(teacher), adaptive_pipe=True)
    assert 8 in result["adaptive"]["declined_buckets"] and result["output_ids"] == list(range(65, 89))
    assert all(change["K"] != 8 for change in result["adaptive"]["changes"])


def test_late_mixed_bucket_drift_is_never_published_and_uses_original_plain_replay(monkeypatch):
    teacher = list(range(65, 400)); ring = Ring(teacher); events = []
    def send(sock, msg):
        ring.send(sock, msg)
        if (msg["op"] == "verify" and not msg.get("prefill") and len(msg["token_ids"]) == 9
                and msg["start"] >= 20):
            reply = ring.replies.pop(); reply[0] = 888; ring.replies.append(reply)
    monkeypatch.setattr(SP, "send_msg", send); monkeypatch.setattr(SP, "recv_msg", ring.receive)
    result = SP.coordinate_pipe(None, ring, Tokenizer(), "prompt", 8, 40, 2, 3,
        ret_sock=ring, local_draft=Draft(teacher), adaptive_pipe=True,
        on_commit=lambda event: events.append(list(event["out"])))
    assert result["output_ids"] == teacher[:40]
    assert result["adaptive"]["fallback"].startswith("original_request_plain_replay")
    assert all(ids == teacher[:len(ids)] for ids in events)
    assert all(len(a) <= len(b) for a, b in zip(events, events[1:]))
    assert result["new_decode_tokens"] == 39


def test_cancel_check_runs_before_the_first_reset(monkeypatch):
    ring = Ring(list(range(65, 400)))
    monkeypatch.setattr(SP, "send_msg", ring.send); monkeypatch.setattr(SP, "recv_msg", ring.receive)
    def stop():
        raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        SP.coordinate_pipe(None, ring, Tokenizer(), "", 4, 12, 1, 2, prompt_ids=[1, 2, 3], cancel_check=stop)
    assert not ring.messages


def test_same_K_adaptive_depth_preserves_static_tokens_and_reduces_real_stale_frames(monkeypatch):
    static, _ = run(monkeypatch, maximum=40, K=8, depth=4, match=False)
    adapted, ring = run(monkeypatch, maximum=40, K=8, depth=4, match=False, adaptive_depth=True)
    assert adapted["output_ids"] == static["output_ids"] == list(range(65, 105))
    assert adapted["K"] == static["K"] == 8 and adapted["depth"] == 1
    assert adapted["metrics"]["stale_frames"] < static["metrics"]["stale_frames"]
    assert adapted["metrics"]["shape_gate_s"] == 0 and adapted["metrics"]["shape_warmup_frames"] == 0
    assert adapted["adaptive"]["changes"] == [{"round": 6, "K": 8, "depth": 1, "boundary": "empty_pipeline", "budget": 48}]
    assert all(len(msg["token_ids"]) == 9 for msg in ring.messages if msg["op"] == "verify" and not msg.get("prefill"))


def test_checkout_libp2p_help_has_no_flat_module_or_PYTHONPATH_requirement(tmp_path):
    env = dict(os.environ); env.pop("PYTHONPATH", None)
    env["SHARD_TRANSPORT"] = "libp2p"; env.pop("SHARD_PSK", None)
    result = subprocess.run([sys.executable, str(Path(SP.__file__).resolve()), "--help"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0 and "--adaptive-depth" in result.stdout, result.stderr


def test_accepted_bracketed_ipv6_plan_uses_a_real_ipv6_listener_and_authenticated_drive(monkeypatch):
    from shard.pipeline_plan import build_plan
    from shard.pipeline_session import SessionConfig, hello_client
    from shard.receipt import gen_key, pub_b64
    from shard.transport import send_msg, recv_msg
    monkeypatch.setattr(SP, "_raw_send_msg", send_msg)
    monkeypatch.setattr(SP, "_raw_recv_msg", recv_msg)
    if not socket.has_ipv6:
        pytest.skip("IPv6 sockets unavailable")
    probe = socket.socket(socket.AF_INET6)
    try:
        probe.bind(("::1", 0)); port = probe.getsockname()[1]
    except OSError:
        pytest.skip("IPv6 loopback unavailable")
    finally:
        probe.close()
    node, controller = gen_key(), gen_key()
    plan = build_plan({"num_hidden_layers": 2}, ring_id="ipv6", cohort_id="a" * 64,
                      endpoints=[f"[::1]:{port}"])
    plan["stages"][0]["signer_pubkey"] = pub_b64(node)
    plan["coordinator"]["signer_pubkey"] = pub_b64(controller)
    cfg = SessionConfig.from_plan(plan, 0, caller_key=node)
    server, acceptor = SP._server({"_session_config": cfg, "_session_key": node}, port, 2)
    channel = socket.create_connection(SP._address(f"[::1]:{port}"), timeout=2)
    try:
        wrapped = hello_client(channel, SessionConfig.from_plan(plan, -1, caller_key=controller),
            0, "drive", SP._raw_send_msg, SP._raw_recv_msg, session_id="ipv6-session")
        assert wrapped.grant["session_id"] == "ipv6-session"
    finally:
        channel.close(); acceptor.close()


@pytest.mark.parametrize("value,adaptive,cap", [("auto", True, 256), ("fixed:64", False, 64), ("32", False, 32)])
def test_margin_policy_is_explicit_and_legacy_integer_is_preserved(monkeypatch, value, adaptive, cap):
    monkeypatch.setenv("NGRAM_MARGIN", value)
    draft = SP.NgramDrafter()
    assert draft.adaptive is adaptive and draft.margin_cap == cap


@pytest.mark.parametrize("value", ["-1", "fixed:-1", "fixed:bad", "unknown"])
def test_bad_margin_never_silently_changes_to_another_policy(monkeypatch, value):
    monkeypatch.setenv("NGRAM_MARGIN", value)
    with pytest.raises(ValueError):
        SP.NgramDrafter()


@pytest.mark.parametrize("coordinator,stage,limits,execution,max_ctx,legacy,accepted", [
    (False, 0, [512, 2048], None, 513, False, False),
    (False, 0, [512, 2048], None, 512, False, True),
    (False, 0, [512, 2048], None, 256, False, True),
    (False, 1, [512, 2048], None, 1024, False, True),
    (True, 0, [None, 512], None, 513, False, False),
    (True, 0, [None, 512], None, 512, False, True),
    (True, 0, [None, None], 1024, 1025, False, False),
    (False, 0, [2048, 2048], 1024, 1024, False, True),
    (False, 0, [2048, 2048], 1024, 1025, False, False),
    (True, 0, [None, None], None, 8192, False, True),
    (False, 0, [512, 2048], None, 4096, True, True),
])
def test_strict_cli_enforces_only_known_execution_and_assigned_context_limits(
        monkeypatch, tmp_path, coordinator, stage, limits, execution, max_ctx, legacy, accepted):
    """Real strict CLI contract/plan/key/config parsing; no GPU or fake calibration.

    Only download payload verification and HF config loading are isolated. This
    fixture tests context admission, not whether its synthetic weights exist.
    """
    import argparse
    import hashlib
    import json
    from transformers import AutoConfig
    from shard.pipeline_plan import build_plan
    from shard.offers import ModelCohort
    from shard.receipt import gen_key, pub_b64
    from shard import download_inventory
    from shard.gpt_oss_contract import RUNTIME_ABI, WIRE_VERSION, NUMERIC_CONTRACT, QUANTIZATION

    monkeypatch.delenv("SHARD_STAGE_LEASE_CONFIG", raising=False)
    monkeypatch.delenv("NGRAM_MARGIN", raising=False)
    config = {"model_type": "gpt_oss", "num_hidden_layers": 4,
              "quantization_config": {"quant_method": "mxfp4", "dequantize": False}}
    config_bytes = json.dumps(config).encode()
    (tmp_path / "config.json").write_bytes(config_bytes)
    cohort = {"model_id": "test/synthetic-gpt-oss", "manifest_sha256": "b" * 64,
              "checkpoint_id": "sha256:" + "b" * 64,
              "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
              "quantization": QUANTIZATION, "runtime_abi": RUNTIME_ABI,
              "wire_version": WIRE_VERSION, "numeric_contract": NUMERIC_CONTRACT, "n_layers": 4}
    keys = [gen_key(), gen_key(), gen_key()]
    plan = build_plan(config, ring_id="context-test", cohort_id=ModelCohort.from_dict(cohort).cohort_id,
        endpoints=["127.0.0.1:29501", "127.0.0.1:29502"])
    plan["model_cohort"] = cohort
    for index, row in enumerate(plan["stages"]):
        row["signer_pubkey"] = pub_b64(keys[index])
        if limits[index] is not None:
            row["context_limit"] = limits[index]
    plan["coordinator"]["signer_pubkey"] = pub_b64(keys[2])
    if execution is not None:
        plan["execution"] = {"max_context": execution}
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(to_dict=lambda: config))
    checked = []
    def verify(*args, **kwargs):
        checked.append(kwargs)
    monkeypatch.setattr(download_inventory, "verify_inventory", verify)
    monkeypatch.setattr(SP, "load_or_make_node_key", lambda path: keys[2 if coordinator else stage])
    args = SimpleNamespace(model=str(tmp_path), coordinator=coordinator, stage=stage, nstages=2,
        deployment_plan=str(plan_file), legacy_protocol=legacy, split=None, lo=-1, hi=-1,
        max_ctx=max_ctx, direct_return=False, served_head=False, next="", tail="", device="cpu",
        fast=True, attn="eager", coordinator_key="fixture-only", timeout=2, K=2, depth=2, ngram_n=2)
    parser = argparse.ArgumentParser()
    if not accepted:
        with pytest.raises(SystemExit) as refused:
            SP._cli_contract(args, parser)
        assert refused.value.code == 2
        assert not checked, "context mismatch passed into weight verification/model setup"
    else:
        lo, hi, session, guard, normalized = SP._cli_contract(args, parser)
        assert (lo, hi) == (stage * 2, stage * 2 + 2)
        assert args.max_ctx == max_ctx and normalized["stages"][stage].get("context_limit") == limits[stage]
        assert (session is None) == legacy and guard is None
        assert checked and checked[0]["verify_files"] is True

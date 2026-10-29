"""Actual stage-delta chains and real socket telemetry; CPU is not GPU evidence."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest
import torch

import v4_round_cost as C
from test_v4_adaptive_round import Clock, config, make_policy
from test_v4_pipe import tiny


@pytest.mark.parametrize("mode,expected", [("cuda_dma", 24), ("cpu_reference", 0)])
def test_exclusive_cost_counts_preserved_d2h_without_claiming_cpu_dma(mode, expected):
    base = {"mode": mode, "storage_mode": "exclusive", "dma_bytes": 8 if mode == "cuda_dma" else 0,
            "preserved_experts": 2, "exchange_bytes": 8}
    cache = SimpleNamespace(stats=lambda: {"pools": {"main:0": base}})
    stage = SimpleNamespace(_expert_cache=cache, _expert_placement="ram")
    assert C._snapshot(stage)[0] == expected
    if mode == "cuda_dma":
        del base["preserved_experts"]
        assert C._snapshot(stage)[0] is None
        base["preserved_experts"] = -1
        assert C._snapshot(stage)[0] is None


def test_stage_chain_uses_actual_counter_deltas_and_rejects_wrong_frame():
    clock = Clock()
    stages = [SimpleNamespace(_expert_placement="gpu", _block_graphs=None) for _ in range(3)]
    message = {"start_pos": 7, "epoch": 2, "round_cost_trace": "b"*32}
    for index, stage in enumerate(stages):
        C.reset(stage, message)
        token = C.begin(stage, message, clock=clock)
        clock.advance(.02)
        message["round_cost"] = C.append(stage, message, token, stage_index=index, nstages=3, clock=clock)
    cost = C.validate_reply(message["round_cost"], trace_id="b"*32, epoch=2, pos=7, expected_stages=3)
    assert cost["expert_dma_bytes"] == 0 and cost["coverage"] == "all_stages"
    assert len(cost["stages"]) == 3 and "not_additive" in cost["timing_scope"]
    assert C.validate_reply(cost, trace_id="c"*32, epoch=2, pos=7) is None
    assert C.validate_reply(cost, trace_id="b"*32, epoch=3, pos=7) is None
    assert C.validate_reply(cost, trace_id="b"*32, epoch=2, pos=7, expected_stages=4) is None
    forged = deepcopy(cost); forged["expert_dma_bytes"] = 10
    assert C.validate_reply(forged, trace_id="b"*32, epoch=2, pos=7) is None


def test_ram_missing_counters_remain_unknown_and_capture_is_not_steady_sample():
    clock = Clock(); value = config(mode="enforce", now=clock())
    policy = make_policy(value, clock)
    policy.refill(committed_tokens=1, available_drafts=3)
    policy.observe_reply_cost({"coverage": "all_stages", "expert_dma_bytes": 3, "capture_changed": True})
    clock.advance(.5); policy.refill(committed_tokens=3, available_drafts=3)
    row = [r for r in policy.report()["records"] if r["kind"] == "window"][-1]
    assert not row["learned"] and row["capture_changed"] and not policy._runtime
    stage = SimpleNamespace(_expert_placement="ram")
    message = {"start_pos": 2, "round_cost_trace": "a"*32}
    C.reset(stage, message)
    result = C.append(stage, message, C.begin(stage, message), stage_index=0, nstages=1)
    assert result["expert_dma_bytes"] is None and result["coverage"] == "partial"


def test_reexploration_is_bounded_and_only_uses_preapproved_dma_safe_caps():
    clock = Clock(); value = config(mode="enforce", now=clock(), dma=0)
    value["exploration"] = {"every_windows": 2, "max_probes": 2, "stale_windows": 2}
    policy = make_policy(value, clock)
    for i in range(16):
        policy.refill(committed_tokens=i+1, available_drafts=3)
        policy.observe_reply_cost({"coverage": "all_stages", "expert_dma_bytes": 0})
        clock.advance(.1)
    decisions = [r for r in policy.report()["records"] if r["kind"] == "decision"]
    assert sum(r["reason"] == "bounded_reexploration" for r in decisions) == 2
    assert policy.report()["exploration_probes"] == 2
    value["max_dma_bytes_per_committed_token"] = 10
    for row in value["calibration"]["profiles"]:
        row["expert_dma_bytes"] = None
    blocked = make_policy(value, clock)
    for i in range(8):
        blocked.refill(committed_tokens=i+1, available_drafts=3); clock.advance(.1)
    assert blocked.report()["exploration_probes"] == 0


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("sealed", [False, True])
def test_pipe_device_digest_byte_order_and_no_hidden_cpu_witness(monkeypatch, packed, sealed):
    import v4_pipe as P
    from v4_privacy import TokenPrivacy
    # Explicit CPU hasher fixture checks wiring/byte order, not a CUDA claim.
    class Local:
        device_digest_enabled = True
        def __init__(self): self.value = None; self.parts = []
        def publish(self, tensor, binding): self.value = tensor; return {"owned": True}
        def frame_digest(self, parts):
            self.parts.append(parts)
            from shard.receipt import ActualByteDigest
            raw = b"".join(original_bytes(x) if torch.is_tensor(x) else x for x in parts)
            return ActualByteDigest(hashlib.sha256(raw).digest())
        def receive(self, ref, binding): return self.value.clone()
        def touch(self): pass
    local = Local()
    original_bytes = P._tbytes
    h = torch.arange(64, dtype=torch.bfloat16).reshape(1, 1, 4, 16)
    ids = torch.tensor([[7]])
    privacy = TokenPrivacy(bytes(range(32)))
    binding = {"job_id": "j", "job_nonce": "a"*64, "swarm_id": "r", "start_pos": 0, "epoch": 0}
    envelope = privacy.seal(ids, binding) if sealed else None
    monkeypatch.setattr(P, "V4_FP8_WIRE", packed)
    wire_h, scales = P._pack_t(h) if packed else (h, None)
    if sealed:
        from v4_privacy import wire_payload_bytes
        expected = wire_payload_bytes(P._tbytes(wire_h), envelope, P._tbytes(scales) if packed else b"")
    else:
        expected = P._wire_bytes(wire_h, ids, scales) if packed else P._payload_bytes(h, ids)
    def no_hidden_witness(value):
        if torch.is_tensor(value) and value.ndim == 4:
            raise AssertionError("device transport must not construct a hidden CPU witness")
        return original_bytes(value)
    monkeypatch.setattr(P, "_tbytes", no_hidden_witness)
    frame, out = P._make_step_frame(h, ids, 0, object(), token_envelope=envelope,
                                  host_transport=local, host_binding={})
    assert "h" not in frame and out.digest == hashlib.sha256(expected).digest()
    _, _, incoming = P._recv_hids(frame, object(), host_transport=local, host_binding={},
        token_privacy=privacy if sealed else None, binding=binding if sealed else None, is_tail=True)
    assert incoming.digest == out.digest


def test_real_three_stage_socket_pipeline_returns_complete_frame_costs(tiny):
    from test_v4_pipe import _Ring, VP, PROMPT
    from v4_adaptive_round import RoundPolicy, CALIBRATION_SCHEMA, calibration_binding
    import time
    directory, args, reference = tiny
    options = {"binding": {"stages": [{"node": i} for i in range(3)]},
               "loaded_block_size": args.dspark_block_size, "depth": 16, "floor": 1}
    now = time.time()
    cfg = {"enabled": True, "mode": "shadow", "calibration": {"schema": CALIBRATION_SCHEMA,
        "binding_sha256": calibration_binding(options), "created_at": now-1, "expires_at": now+300,
        "profiles": [{"draft_prefix_cap": args.dspark_block_size, "committed_tokens": 4,
                      "elapsed_s": 1., "expert_dma_bytes": 0}]}}
    policy = RoundPolicy(cfg, **options)
    observed = []
    original = policy.observe_reply_cost
    def observe(value):
        if value is not None: observed.append(value)
        original(value)
    policy.observe_reply_cost = observe
    ring = _Ring(directory, args, VP._dspark_tiling(args, 3), dspark=True, receipts=True)
    try:
        result = ring.pipelined(list(PROMPT), len(reference), depth=16, floor=1,
            draft_block_limit=args.dspark_block_size, round_policy=policy)
    finally:
        ring.close()
    assert result["tokens"] == reference and result["receipts_ok"]
    assert observed and all(len(row["stages"]) == 3 for row in observed)
    assert all(row["expert_dma_bytes"] == 0 for row in observed)

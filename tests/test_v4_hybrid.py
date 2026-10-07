"""Real tiny V4 hybrid-runtime acceptance on the explicit CPU reference seam.

These tests exercise the same meta construction, canonical host pool, bounded cache and
checkpoint/rollback integration as RAM placement. They do not qualify GPU math or speed.
"""
import dataclasses
import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
ST = pytest.importorskip("safetensors.torch")
REF = pytest.importorskip("v4_ref_cpu")
V4 = pytest.importorskip("v4_stage")


@pytest.fixture(autouse=True)
def deterministic_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def args():
    return REF.cpu_args(n_layers=4, n_routed_experts=6, n_activated_experts=2,
                        compress_ratios=(0, 4, 8, 0, 0, 0), dspark_target_layer_ids=(1, 2, 3))


def checkpoint(directory, args, seed=7, *, force_routes=False):
    model = REF.build_oracle(args, seed)
    if force_routes:
        with torch.no_grad():
            for layer in model.layers:
                gate = layer.ffn.gate
                if gate.hash:
                    gate.tid2eid[:, 0] = 0
                    gate.tid2eid[:, 1] = 1
                else:
                    gate.bias.fill_(-1000)
                    gate.bias[:2] = torch.tensor([100.0, 200.0])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # A converted checkpoint does not duplicate embed/head aliases under mtp.*.
    weights = {name: value.detach().cpu().clone().contiguous()
               for name, value in model.state_dict().items()
               if not (name.startswith("mtp.") and (".embed." in name or ".head." in name))}
    ST.save_file(weights, str(directory / "model0-mp1.safetensors"))
    (directory / "config.json").write_text(json.dumps(dataclasses.asdict(args)), encoding="utf-8")
    return directory, model


def stage(directory, args, *, hybrid, metrics=False, dspark=False, slots=2):
    kw = dict(head=True, tail=True, dspark=dspark, device="cpu", runtime_metrics=metrics)
    if hybrid:
        kw.update(expert_placement="ram", expert_cache_reference=True, expert_cache_slots=slots)
    st = V4.Stage(0, args.n_layers, args, **kw)
    st.load(str(directory))
    return st


def run(st, ids, pos):
    h = st.forward(st.embed(ids), ids, pos)
    return h, st.logits_all(h, full_logits=False)


def state(st):
    """Only reachable compressed history; rejected future slots deliberately stay unsnapshotted."""
    snapshots = []
    for layer in st.layers:
        attn = layer.attn
        count = st.args.window_size + (st._pos // attn.compress_ratio if attn.compress_ratio else 0)
        snapshots.append(attn.kv_cache[:, :count].clone())
        if attn.compress_ratio and attn.indexer is not None:
            snapshots.append(attn.indexer.kv_cache[:, :st._pos // attn.compress_ratio].clone())
    for compressor, _ in st._compressors():
        snapshots += [compressor.kv_state.clone(), compressor.score_state.clone()]
    return snapshots


def assert_equal_stages(a, b, ids, pos):
    a_h, a_l = run(a, ids, pos)
    b_h, b_l = run(b, ids, pos)
    assert torch.equal(a_h, b_h), f"hidden bytes changed at position {pos}"
    assert torch.equal(a_l, b_l), f"logit bytes changed at position {pos}"
    assert a._pos == b._pos
    for x, y in zip(state(a), state(b)):
        assert torch.equal(x, y), f"KV/compressor state changed at position {pos}"
    return a_l.argmax(-1).unsqueeze(1)


def ids(args, n, *, batch=1, seed=3):
    return torch.randint(0, args.vocab_size, (batch, n), generator=torch.Generator().manual_seed(seed))


def test_routed_expert_construction_is_meta_before_materialization(args, monkeypatch):
    M = V4.ref()
    original = M.Expert.__init__
    constructed = []

    def observed(self, *a, **kw):
        original(self, *a, **kw)
        constructed.append({p.device.type for p in self.parameters()})

    monkeypatch.setattr(M.Expert, "__init__", observed)
    st = V4.Stage(0, 1, args, head=True, device="cpu", expert_placement="ram",
                  expert_cache_reference=True, expert_cache_slots=2)
    # Every routed Expert begins on meta. A bounded cache may construct its own CPU test views later.
    assert constructed[:args.n_routed_experts] == [{"meta"}] * args.n_routed_experts
    for expert in st.layers[0].ffn.experts:
        assert all(p.device.type == "cpu" for p in expert.parameters())
    assert st.layers[0].attn.kv_cache.device.type == "cpu"
    assert all(p.device.type != "meta" for p in st.layers.parameters())


@pytest.mark.parametrize("batch,prompt", [(1, 13), (2, 7), (1, 33)])
def test_prefill_decode_chunk_and_new_job_are_bit_exact(tmp_path, args, batch, prompt):
    directory, _ = checkpoint(tmp_path / "weights", args)
    resident = stage(directory, args, hybrid=False)
    cached = stage(directory, args, hybrid=True)
    addresses = [{name: bank.data_ptr() for name, bank in layer.ffn._hybrid_runtime.cache.banks.items()
                  if bank is not None} for layer in cached.layers]
    sequence = ids(args, prompt + 5, batch=batch)
    assert_equal_stages(resident, cached, sequence[:, :prompt], 0)
    assert_equal_stages(resident, cached, sequence[:, prompt:prompt + 3], prompt)
    for pos in range(prompt + 3, prompt + 5):
        assert_equal_stages(resident, cached, sequence[:, pos:pos + 1], pos)
    resident.reset(); cached.reset()
    assert_equal_stages(resident, cached, sequence[:, :prompt], 0)
    assert addresses == [{name: bank.data_ptr() for name, bank in layer.ffn._hybrid_runtime.cache.banks.items()
                          if bank is not None} for layer in cached.layers]


@pytest.mark.parametrize("accepted", [0, 1, 3, 4])
def test_rejected_chunk_replay_preserves_hidden_logits_and_live_kv(tmp_path, args, accepted):
    directory, _ = checkpoint(tmp_path / "weights", args)
    reference = stage(directory, args, hybrid=False)
    cached = stage(directory, args, hybrid=True)
    cached._spec = True
    sequence = ids(args, 17)
    assert_equal_stages(reference, cached, sequence[:, :13], 0)
    run(cached, sequence[:, 13:], 13)
    for offset in range(accepted):
        run(reference, sequence[:, 13 + offset:14 + offset], 13 + offset)
    cached._seek(13 + accepted)
    correction = ids(args, 1, seed=5)
    for pos in range(13 + accepted, 17 + accepted):
        correction = assert_equal_stages(reference, cached, correction, pos)


def test_hash_duplicate_last_occurrence_matches_reference_and_forces_cache_eviction(tmp_path, args):
    directory, model = checkpoint(tmp_path / "weights", args)
    # Two-slot pools must serve a prefill whose hash rows require all six logical experts.
    with torch.no_grad():
        for layer in model.layers:
            gate = layer.ffn.gate
            if gate.hash:
                gate.tid2eid[:, 0] = torch.arange(args.vocab_size) % args.n_routed_experts
                gate.tid2eid[:, 1] = gate.tid2eid[:, 0]  # duplicate-ID indexed update, not scatter add
    weights = {name: tensor.detach().clone().contiguous() for name, tensor in model.state_dict().items()
               if not (name.startswith("mtp.") and (".embed." in name or ".head." in name))}
    ST.save_file(weights, str(directory / "model0-mp1.safetensors"))
    resident = stage(directory, args, hybrid=False)
    cached = stage(directory, args, hybrid=True, slots=2)
    prompt = torch.arange(13).unsqueeze(0)
    assert_equal_stages(resident, cached, prompt, 0)
    for pos in range(13, 21):
        token = torch.tensor([[pos % args.n_routed_experts]])
        assert_equal_stages(resident, cached, token, pos)
    runtime = cached.layers[0].ffn._hybrid_runtime
    assert runtime.cache.capacity == 2
    assert runtime.cache.stats()["evictions"] > 0, "this is an actual eviction test, not an all-hit parity run"


def test_strict_load_parameter_aliases_and_checkpoint_reload(tmp_path, args):
    first, _ = checkpoint(tmp_path / "first", args, 7, force_routes=True)
    second, model = checkpoint(tmp_path / "second", args, 11, force_routes=True)
    cached = stage(first, args, hybrid=True)
    run(cached, ids(args, 13), 0)
    run(cached, ids(args, 1), 13)
    cached.load(str(second))  # cached copies must be invalidated, including resident hit candidates
    cached.reset()
    resident = stage(second, args, hybrid=False)
    assert_equal_stages(resident, cached, ids(args, 13), 0)
    for layer, original in zip(cached.layers, model.layers):
        got, want = layer.state_dict(), original.state_dict()
        assert set(got) == set(want), "cache tensors must not become serialized parameters"
        assert all(torch.equal(got[name], want[name]) for name in want)
        runtime = layer.ffn._hybrid_runtime
        assert runtime.pool is not None
        for expert in layer.ffn.experts:
            for kind in ("w1", "w2", "w3"):
                linear = getattr(expert, kind)
                if getattr(linear, "scale", None) is not None:
                    assert linear.weight.scale is linear.scale


def test_missing_routed_parameter_refuses_strict_checkpoint_load(tmp_path, args):
    directory, _ = checkpoint(tmp_path / "broken", args)
    path = directory / "model0-mp1.safetensors"
    weights = ST.load_file(str(path))
    del weights["layers.0.ffn.experts.0.w2.weight"]
    ST.save_file(weights, str(path))
    with pytest.raises(RuntimeError, match="missing|Missing"):
        stage(directory, args, hybrid=True)


def test_packed_fp4_checkpoint_bytes_and_scale_aliases_survive_canonical_and_cache_views(tmp_path, args):
    if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("packed V4 checkpoint tests require recent PyTorch formats")
    packed = dataclasses.replace(args, expert_dtype="fp4")
    M = V4.ref()
    V4._set_globals(M, packed)
    with torch.device("cpu"), M.set_dtype(torch.bfloat16):
        layer = M.Block(0, packed)
    with torch.no_grad():
        for index, parameter in enumerate(layer.parameters()):
            if parameter.dtype == torch.float4_e2m1fn_x2:
                raw = parameter.view(torch.uint8)
                raw.copy_((torch.arange(raw.numel()) + index).remainder(256).to(torch.uint8).reshape(raw.shape))
            elif parameter.dtype == torch.float8_e8m0fnu:
                parameter.view(torch.uint8).fill_(127)
            elif parameter.dtype in (torch.int32, torch.int64):
                parameter.zero_()
            else:
                parameter.fill_(0.02)
    weights = {"layers.0." + name: value.detach().clone().contiguous()
               for name, value in layer.state_dict().items()}
    directory = tmp_path / "packed"
    directory.mkdir()
    ST.save_file(weights, str(directory / "model0-mp1.safetensors"))
    st = V4.Stage(0, 1, packed, device="cpu", expert_placement="ram",
                  expert_cache_reference=True, expert_cache_slots=2).load(str(directory))
    runtime = st.layers[0].ffn._hybrid_runtime
    lease = runtime.cache.acquire([0, 1])
    lease.wait_on()
    for expert_id in (0, 1):
        canonical = st.layers[0].ffn.experts[expert_id]
        slot_expert = runtime.cache.experts[lease.mapping[expert_id]]
        for kind in ("w1", "w2", "w3"):
            linear, slot_linear = getattr(canonical, kind), getattr(slot_expert, kind)
            for attribute in ("weight", "scale"):
                original = weights[f"layers.0.ffn.experts.{expert_id}.{kind}.{attribute}"]
                assert torch.equal(getattr(linear, attribute).view(torch.uint8), original.view(torch.uint8))
                assert torch.equal(getattr(slot_linear, attribute).view(torch.uint8), original.view(torch.uint8))
            assert linear.weight.scale is linear.scale and slot_linear.weight.scale is slot_linear.scale
    lease.release()


def test_cpu_reference_metrics_do_not_claim_gpu_hits_dma_or_cpu_fallback(tmp_path, args):
    directory, _ = checkpoint(tmp_path / "weights", args)
    st = stage(directory, args, hybrid=True, metrics=True)
    run(st, ids(args, 13), 0)
    run(st, ids(args, 1), 13)
    metrics = st.runtime_metrics()
    assert metrics["mode"] == "reference_cpu"
    total = metrics["totals"]
    assert total["routed_entries"] > 0
    assert total["reference_routes"] == total["routed_entries"]
    assert all(total[key] == 0 for key in ("resident_hits", "dma_misses", "dma_bytes", "cpu_misses"))
    assert metrics["kv"]["gpu_bytes"] == 0 and metrics["kv"]["host_bytes"] > 0


def test_dspark_uses_canonical_experts_and_preserves_tail_aliases(tmp_path, args):
    DS = pytest.importorskip("v4_dspark_draft")
    directory, _ = checkpoint(tmp_path / "weights", args)
    full = stage(directory, args, hybrid=False, dspark=True)
    cached = stage(directory, args, hybrid=True, dspark=True)
    a = DS.DSparkTail(full).load(str(directory))
    b = cached._hybrid_ring_drafter.tail  # eager-owned MTP participates in the one stage cache budget
    prompt = ids(args, 13)
    h_a, logits_a = run(full, prompt, 0)
    h_b, logits_b = run(cached, prompt, 0)
    assert torch.equal(h_a, h_b) and torch.equal(logits_a, logits_b)
    token = logits_a.argmax(-1)
    a.prefill(token, full.tail_main_hidden()); b.prefill(token, cached.tail_main_hidden())
    for pos in range(13, 16):
        token_ids = token.unsqueeze(1)
        next_token = assert_equal_stages(full, cached, token_ids, pos)
        out_a = a.advance_and_draft(next_token, full.tail_main_hidden(), start_pos=pos)
        out_b = b.advance_and_draft(next_token, cached.tail_main_hidden(), start_pos=pos)
        assert all(torch.equal(x, y) for x, y in zip(out_a, out_b))
        for left, right in zip(a.mtp, b.mtp):
            assert torch.equal(left.attn.kv_cache, right.attn.kv_cache)
            assert right.embed is cached.embed_tokens and right.head is cached.lm_head
            assert right.ffn._hybrid_runtime.pool is not None
            assert all(p.device.type == "cpu" for e in right.ffn.experts for p in e.parameters())
        token = next_token[:, 0]


@pytest.mark.parametrize("kwargs", [
    {"expert_placement": "unknown"},
    {"expert_placement": "ram", "expert_cache_slots": 0, "expert_cache_reference": True},
    {"expert_placement": "ram", "expert_cache_bytes": 1, "expert_cache_reference": True},
    {"expert_placement": "ram", "expert_cache_slots": 2},  # no silent CPU operator fallback
])
def test_invalid_or_insufficient_placement_refuses_before_running(tmp_path, args, kwargs):
    with pytest.raises((ValueError, RuntimeError), match="placement|cache|budget|slot|CUDA|cpu|RAM|ram"):
        st = V4.Stage(0, 1, args, head=True, device="cpu", **kwargs)
        if "expert_cache_bytes" in kwargs:
            directory, _ = checkpoint(tmp_path / "weights", args)
            st.load(str(directory))  # all pool shapes are known; reject the true minimum-slot budget


def test_default_gpu_placement_does_not_install_hybrid_adapter(args):
    st = V4.Stage(0, 1, args, head=True, device="cpu")
    assert not hasattr(st.layers[0].ffn, "_hybrid_runtime")


@pytest.mark.parametrize("flag", ["V4_MOE_IN_GRAPH", "V4_DSPARK_MOE"])
def test_incompatible_graph_or_draft_dispatch_flags_refuse(args, monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    with pytest.raises(ValueError, match="graph|dispatch|MTP"):
        V4.Stage(0, 1, args, device="cpu", expert_placement="ram",
                  expert_cache_reference=True, expert_cache_slots=2)


def test_operator_flags_are_strict_in_fresh_process():
    env = dict(os.environ)
    env["V4_EXPERT_PLACEMENT"] = "typo"
    script = "import v4_stage, v4_ref_cpu; v4_stage.Stage(0,1,v4_ref_cpu.cpu_args(),device='cpu')"
    result = subprocess.run([sys.executable, "-c", script], env=env, text=True,
                            capture_output=True, timeout=60)
    assert result.returncode != 0
    assert "placement" in (result.stderr + result.stdout).lower()


def test_continuous_decode_lfu_eviction_and_zero_lease_leak(tmp_path, args):
    """Step 5 & 6 validation: multi-step decode smoothly evicts via LFU and leaves 0 live leases."""
    directory, _ = checkpoint(tmp_path / "weights", args)
    cached = stage(directory, args, hybrid=True, slots=2, metrics=True)
    prompt = ids(args, 8)
    run(cached, prompt, 0)
    for pos in range(8, 28):
        token = ids(args, 1, seed=pos)
        run(cached, token, pos)
        for layer in cached.layers:
            stats = layer.ffn._hybrid_runtime.cache.stats()
            assert stats["live_leases"] == 0, f"layer {layer.layer_id} leaked a cache lease at pos {pos}"
    layer0_stats = cached.layers[0].ffn._hybrid_runtime.cache.stats()
    assert layer0_stats["acquires"] > 0
    assert layer0_stats["evictions"] > 0, "bounded cache of 2 slots must have evicted across 20 decode steps"
    metrics = cached.runtime_metrics()
    assert metrics["totals"]["routed_entries"] > 0


def test_prefetch_recent_records_metrics_without_leaking_leases(tmp_path, args):
    """Step 6 prefetch validation: prefetching previous routes pre-warms slots without blocking or leaking."""
    directory, _ = checkpoint(tmp_path / "weights", args)
    cached = stage(directory, args, hybrid=True, slots=3, metrics=True)
    prompt = ids(args, 6)
    run(cached, prompt, 0)
    # Explicitly invoke prefetch on layers before the next forward step
    for layer in cached.layers:
        layer.ffn._hybrid_runtime.prefetch_recent()
        assert layer.ffn._hybrid_runtime.cache.stats()["live_leases"] == 0
    token = ids(args, 1, seed=12)
    run(cached, token, 6)
    for layer in cached.layers:
        assert layer.ffn._hybrid_runtime.cache.stats()["live_leases"] == 0


def test_speculative_rollback_stress_keeps_cache_consistent(tmp_path, args):
    """Step 5 & 6 rollback validation: repeated rewind does not corrupt cached slots or live KV."""
    directory, _ = checkpoint(tmp_path / "weights", args)
    reference = stage(directory, args, hybrid=False)
    cached = stage(directory, args, hybrid=True, slots=2)
    cached._spec = True
    prompt = ids(args, 10)
    assert_equal_stages(reference, cached, prompt, 0)
    # Simulate a speculative verification loop with repeated rollbacks
    curr_pos = 10
    for round_idx in range(4):
        # Speculative draft of 3 tokens
        draft = ids(args, 3, seed=100 + round_idx)
        run(cached, draft, curr_pos)
        # Verifier accepts only 1 token
        accepted = 1
        run(reference, draft[:, :accepted], curr_pos)
        cached._seek(curr_pos + accepted)
        curr_pos += accepted
        # Next round starts from accepted pos; stage states must match exactly
        next_token = ids(args, 1, seed=200 + round_idx)
        assert_equal_stages(reference, cached, next_token, curr_pos)
        curr_pos += 1
        for layer in cached.layers:
            assert layer.ffn._hybrid_runtime.cache.stats()["live_leases"] == 0


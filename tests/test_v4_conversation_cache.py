"""Real CPU V4 recurrent state, rollback and exact-prompt restore contracts."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
R = pytest.importorskip("v4_ref_cpu")
V = pytest.importorskip("v4_stage")
DS = pytest.importorskip("v4_dspark_draft")
from v4_conversation_cache import (StageConversationCache, CacheQuotas, ConversationIdentity,
    ConversationCacheError, FullRingRestoreTransaction, PrefixBinding, _buffers, _same, _metadata,
    _digest, _byte_chunks)


def identity(**kw):
    return replace(ConversationIdentity("tenant-a", "a" * 64, "bit-native/1", "tensor-sha256:" + "b" * 64,
        "c" * 64, "d" * 64, "e" * 64, "ring-1", "epoch-1", (("node-1", 2),)), **kw)


@pytest.fixture
def model():
    args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
        dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=64)
    return args, R.build_oracle(args, 7)


def stage(model, *, tier=False, draft=False):
    args, oracle = model
    st = V.Stage(0, args.n_layers, args, head=True, tail=True, dspark=draft, device="cpu",
        kv_placement="layer" if tier else "gpu", kv_reference=tier,
        kv_gpu_budget_bytes=2_000_000 if tier else 0, kv_host_budget_bytes=10_000_000 if tier else 0)
    for target, source in zip(st.layers, oracle.layers):
        target.load_state_dict(source.state_dict(), strict=True)
    st.embed_tokens.load_state_dict(oracle.embed.state_dict())
    st.norm.load_state_dict(oracle.norm.state_dict())
    st.lm_head.load_state_dict(oracle.head.state_dict())
    with torch.no_grad():
        for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            getattr(st, name).copy_(getattr(oracle, name))
    return st


def cache(st, **kw):
    quotas = kw.pop("quotas", CacheQuotas(20_000_000))
    return StageConversationCache(st, quotas, enabled=True, **kw)


def forward(st, ids, pos):
    with torch.no_grad():
        h = st.forward(st.embed(ids), ids, pos)
        logits = st.logits_all(h, full_logits=False)
    return h, logits


def capture(c, prefix, reply=None):
    return c.capture(identity(), prefix, committed_frontier=c.stage._pos, drained=True, tail_reply=reply)


@pytest.mark.parametrize("tier", [False, True])
def test_exact_repeat_restores_full_history_and_continuation_bytes(model, tier):
    st = stage(model, tier=tier)
    c = cache(st)
    ids = torch.arange(13).view(1, -1)
    _, prefill = forward(st, ids, 0)
    reply = {"token": int(prefill.argmax(-1).item())}
    pointers = {name: value.data_ptr() for name, value in _buffers(st, None).items()}
    entry = capture(c, ids, reply)
    states = {name: value.clone() for name, value in _buffers(st, None).items()}
    nxt = torch.tensor([[reply["token"]]])
    expected_h, expected_logits = forward(st, nxt, 13)
    st.reset()
    unrelated = torch.arange(7, 20).view(1, -1)
    forward(st, unrelated, 0)
    st.reset()
    ticket = c.prepare_restore(identity(), ids, entry_id=entry)
    assert st._pos == 0  # Prepare never mutates model state.
    restored = c.commit_restore(ticket)
    assert restored["tail_reply"] == reply and st._pos == 13
    assert pointers == {name: value.data_ptr() for name, value in _buffers(st, None).items()}
    for name, tensor in _buffers(st, None).items():
        assert torch.equal(tensor, states[name]), name
    actual_h, actual_logits = forward(st, nxt, 13)
    assert torch.equal(actual_h, expected_h)
    assert torch.equal(actual_logits, expected_logits)
    assert c.status()["hits"] == 1


def test_rollback_checkpoint_and_seek_survive_snapshot(model):
    st = stage(model)
    st._spec = True
    c = cache(st)
    ids = torch.arange(13).view(1, -1)
    forward(st, ids, 0)
    forward(st, torch.tensor([[20, 21, 22]]), 13)
    entry = capture(c, list(range(13)) + [20, 21, 22])
    target = torch.tensor([[31]])
    h, logits = forward(st, target, 14)  # Real rewind/replay into checkpoint.
    c.reset_current()
    c.commit_restore(c.prepare_restore(identity(), list(range(13)) + [20, 21, 22], entry_id=entry))
    assert len(st._spec_ckpts) == 1 and st._spec_ckpts.maxlen == st._spec_depth
    actual_h, actual_logits = forward(st, target, 14)
    assert torch.equal(h, actual_h) and torch.equal(logits, actual_logits)


def test_mtp_cache_last_spec_and_ring_frontier_restore(model):
    st = stage(model, draft=True)
    tail = DS.DSparkTail(st)
    for i, blk in enumerate(tail.mtp):
        sd = {k: v for k, v in model[1].mtp[i].state_dict().items() if k not in DS.ALIAS_KEYS}
        blk.load_state_dict(sd, strict=False)
    drafter = DS.RingDrafter(tail)
    drafter.pipelined = True
    c = cache(st, drafter=drafter)
    ids = torch.arange(13).view(1, -1)
    h, logits = forward(st, ids, 0)
    cur = int(logits.argmax(-1).item())
    drafter.on_chunk({"ids": ids.tolist(), "start_pos": 0}, st, {"token": cur})
    h, logits = forward(st, torch.tensor([[cur]]), 13)
    drafter.on_chunk({"ids": [[cur]], "start_pos": 13}, st,
                     {"token": int(logits.argmax(-1).item()), "tokens": [int(logits.argmax(-1).item())]})
    assert tail.last_spec is not None
    entry = capture(c, list(range(13)) + [cur])
    saved = {k: v.clone() for k, v in _buffers(st, drafter).items()}
    last_spec = tuple(t.clone() for t in tail.last_spec)
    frontier = (drafter._cfront, drafter._mfront, drafter._last)
    c.reset_current()
    assert tail._pos is None and drafter._last is None
    c.commit_restore(c.prepare_restore(identity(), list(range(13)) + [cur], entry_id=entry))
    for k, value in _buffers(st, drafter).items():
        assert torch.equal(value, saved[k])
    assert (drafter._cfront, drafter._mfront, drafter._last) == frontier
    assert all(torch.equal(a, b) for a, b in zip(tail.last_spec, last_spec))
    c.reset_current()
    replacement_wrapper = DS.RingDrafter(tail)
    c.bind_drafter(replacement_wrapper)
    c.commit_restore(c.prepare_restore(identity(), list(range(13)) + [cur], entry_id=entry))
    assert (replacement_wrapper._cfront, replacement_wrapper._mfront, replacement_wrapper._last) == frontier


@pytest.mark.parametrize("change", [dict(tenant="tenant-b"), dict(cohort_id="f" * 64),
    dict(numeric_contract="approx/1"), dict(source_id="different"), dict(config_sha256="f" * 64),
    dict(tokenizer_sha256="f" * 64), dict(template_sha256="f" * 64), dict(ring_id="ring-2"),
    dict(ring_epoch="epoch-2"), dict(lease_fences=(("node-1", 3),))])
def test_exact_identity_never_crosses_tenant_model_or_fence(model, change):
    st = stage(model)
    c = cache(st)
    forward(st, torch.tensor([[1, 2, 3]]), 0)
    capture(c, [1, 2, 3])
    assert c.prepare_restore(identity(**change), [1, 2, 3]) is None
    assert c.prepare_restore(identity(), [1, 2, 3, 4]) is None


def test_sealed_prefix_contains_no_raw_ids(model):
    st = stage(model)
    c = cache(st)
    forward(st, torch.tensor([[7, 8, 9]]), 0)
    binding = PrefixBinding(3, "1" * 64)
    entry = capture(c, binding.to_dict(), {"token": 11})
    assert c._entries[entry].prefix == binding
    assert c.prepare_restore(identity(), {"token_count": 3, "digest": "2" * 64}) is None
    c.reset_current()
    assert c.commit_restore(c.prepare_restore(identity(), binding))["frontier"] == 3


def test_storage_or_weights_changed_after_prepare_reject_before_restore(model):
    st = stage(model)
    c = cache(st)
    forward(st, torch.tensor([[1, 2, 3]]), 0)
    capture(c, [1, 2, 3])
    c.reset_current()
    ticket = c.prepare_restore(identity(), [1, 2, 3])
    st.layers[0].attn.kv_cache = st.layers[0].attn.kv_cache.clone()
    with pytest.raises(ConversationCacheError, match="storage changed"):
        c.commit_restore(ticket)
    assert st._pos == 0 and c.status()["entries"] == 0
    forward(st, torch.tensor([[1, 2, 3]]), 0)
    capture(c, [1, 2, 3])
    with torch.no_grad():
        next(st.layers.parameters()).add_(0)
    with pytest.raises(ConversationCacheError, match="changed model"):
        c.prepare_restore(identity(), [1, 2, 3])


@pytest.mark.parametrize("failure", ["miss", "commit"])
def test_all_stage_failure_invalidates_and_resets_whole_ring(model, monkeypatch, failure):
    a, b = stage(model), stage(model)
    caches = [cache(a), cache(b)]
    for c in caches:
        forward(c.stage, torch.tensor([[1, 2, 3]]), 0)
        capture(c, [1, 2, 3])
        c.reset_current()
    if failure == "miss":
        caches[1].invalidate()
    else:
        monkeypatch.setattr(caches[1], "commit_restore", lambda _: (_ for _ in ()).throw(RuntimeError("forced")))
    result = FullRingRestoreTransaction(caches).restore(identity(), [1, 2, 3])
    assert result["restored"] is False and result["fallback"] == "full_prefill"
    for c in caches:
        assert c.stage._pos == 0 and c.status()["entries"] == c.status()["prepared"] == 0
        assert all(torch.count_nonzero(t).item() == 0 for k, t in _buffers(c.stage, None).items() if "score_state" not in k)


def test_lru_ttl_prepared_protection_and_quota_preflight(model):
    st = stage(model)
    now = [0.0]
    c = cache(st, quotas=CacheQuotas(20_000_000, max_entries=1), clock=lambda: now[0])
    forward(st, torch.tensor([[1, 2, 3]]), 0)
    first = capture(c, [1, 2, 3])
    ticket = c.prepare_restore(identity(), [1, 2, 3])
    with pytest.raises(ConversationCacheError, match="held by prepared"):
        capture(c, [1, 2, 3])
    c.abort_restore(ticket)
    second = capture(c, [1, 2, 3])
    assert first != second and first not in c._entries
    now[0] = 301
    assert c.prepare_restore(identity(), [1, 2, 3]) is None
    tiny = cache(st, quotas=CacheQuotas(1))
    before = {k: v.clone() for k, v in _buffers(st, None).items()}
    with pytest.raises(ConversationCacheError, match="quota"):
        capture(tiny, [1, 2, 3])
    assert _same(before, _buffers(st, None)) and tiny.status()["entries"] == 0


def test_capture_requires_drained_committed_frontier_and_excludes_old_receipt(model):
    st = stage(model)
    c = cache(st)
    forward(st, torch.tensor([[1, 2, 3]]), 0)
    with pytest.raises(ConversationCacheError, match="drained"):
        c.capture(identity(), [1, 2, 3], committed_frontier=3, drained=False)
    with pytest.raises(ConversationCacheError, match="frontier"):
        c.capture(identity(), [1, 2, 3], committed_frontier=2, drained=True)
    with pytest.raises(ConversationCacheError, match="signatures"):
        capture(c, [1, 2, 3], {"token": 1, "receipt": "old"})
    disabled = StageConversationCache(st, CacheQuotas(0))
    assert disabled.capture(None, None, committed_frontier=0, drained=False) is None
    assert disabled.prepare_restore(None, None) is None


def test_per_request_shadow_checks_state_bytes_and_reply(model):
    st = stage(model)
    c = cache(st)
    ids = torch.tensor([[1, 2, 3]])
    _, logits = forward(st, ids, 0)
    reply = {"token": int(logits.argmax(-1).item())}
    candidate = capture(c, ids, reply)
    c.reset_current()
    forward(st, ids, 0)
    assert c.shadow_compare(candidate, identity=identity(), prefix_ids=ids, tail_reply=reply)["passed"]
    assert not c.shadow_compare(candidate, identity=identity(), prefix_ids=ids,
                               tail_reply={"token": (reply["token"] + 1) % st.args.vocab_size})["passed"]
    with torch.no_grad():
        st.layers[0].attn.kv_cache[0, 0, 0].add_(1)
    assert not c.shadow_compare(candidate, identity=identity(), prefix_ids=ids, tail_reply=reply)["passed"]
    assert st._pos == 3  # Full-prefill reference is kept even on gate failure.


def test_snapshot_content_tamper_between_prepare_and_commit_fails_closed(model):
    st = stage(model)
    c = cache(st)
    forward(st, torch.tensor([[1, 2, 3]]), 0)
    entry = capture(c, [1, 2, 3], {"token": 1})
    digest = c.entry_digest(entry)
    c.reset_current()
    ticket = c.prepare_restore(identity(), [1, 2, 3])
    assert ticket["entry_content_sha256"] == digest
    c._entries[entry].buffers["main.0.kv"][0, 0, 0].add_(1)
    with pytest.raises(ConversationCacheError, match="content was modified"):
        c.commit_restore(ticket)
    assert st._pos == 0 and c.status()["entries"] == 0


def test_entry_hash_commits_identity_reply_and_byte_state(model):
    st = stage(model)
    c = cache(st)
    forward(st, torch.tensor([[1, 2, 3]]), 0)
    entry = capture(c, [1, 2, 3], {"token": 1})
    ticket = c.prepare_restore(identity(), [1, 2, 3])
    restored = c.commit_restore(ticket)
    storage = c.storage_inventory()
    assert storage["host_tensor_bytes"] > 0
    assert storage["host_budget_charge_bytes"] >= storage["host_tensor_bytes"]
    assert storage["pinned_tensor_bytes"] == storage["gpu_tensor_bytes"] == 0
    assert restored["entry_content_sha256"] == c.entry_digest(entry)
    assert restored["restored_state_sha256"] == restored["state_digest"]
    c._entries[entry].tail_reply["token"] = 2
    with pytest.raises(ConversationCacheError, match="content was modified"):
        c.entry_digest(entry)


def test_extended_prefix_candidate_is_compared_to_actual_full_prefill(model):
    # This exercises a real differing GEMM shape/compressor branch. No assertion
    # assumes all seeds pass: the gate must reflect exact *this request* bytes.
    st = stage(model)
    c = cache(st)
    prefix, full = torch.arange(5).view(1, -1), torch.arange(13).view(1, -1)
    forward(st, prefix, 0)
    assert c.prepare_restore(identity(), full) is None
    h, logits = forward(st, full[:, 5:], 5)
    reply = {"token": int(logits.argmax(-1).item())}
    candidate = capture(c, full, reply)
    saved = c._entries[candidate]
    c.reset_current()
    h, logits = forward(st, full, 0)
    expected_reply = {"token": int(logits.argmax(-1).item())}
    expected = (_same(saved.buffers, _buffers(st, None)) and
                _same(saved.metadata, _metadata(st, None)) and _same(reply, expected_reply))
    observed = c.shadow_compare(candidate, identity=identity(), prefix_ids=full, tail_reply=expected_reply)
    assert observed["passed"] == expected
    assert observed["scope"] == "per_request_full_prefill_state_and_reply_bytes"
    assert st._pos == 13


def test_tenant_quota_evicts_only_that_tenant_when_global_space_exists(model):
    st = stage(model)
    c = cache(st, quotas=CacheQuotas(20_000_000, max_entries=3, max_tenant_entries=1))
    forward(st, torch.tensor([[1, 2, 3]]), 0)
    own = capture(c, [1, 2, 3])
    other = c.capture(identity(tenant="tenant-b"), [1, 2, 3], committed_frontier=3, drained=True)
    replacement = capture(c, [1, 2, 3])
    assert own not in c._entries and other in c._entries and replacement in c._entries


def test_prepared_ticket_expires_and_cannot_pin_capacity_forever(model):
    st = stage(model)
    now = [0.0]
    c = cache(st, quotas=CacheQuotas(20_000_000, ttl_s=1), clock=lambda: now[0])
    forward(st, torch.tensor([[1, 2, 3]]), 0)
    capture(c, [1, 2, 3])
    ticket = c.prepare_restore(identity(), [1, 2, 3])
    now[0] = 2
    assert c.status()["entries"] == c.status()["prepared"] == 0
    with pytest.raises(ConversationCacheError, match="unknown"):
        c.commit_restore(ticket)


def test_bounded_hash_and_compare_follow_logical_bytes_across_mixed_layouts():
    value = torch.arange(129 * 513, dtype=torch.float32).reshape(129, 513)
    transposed_storage = value.t().contiguous().t()
    assert not transposed_storage.is_contiguous()
    assert _same(value, transposed_storage, chunk_bytes=4096)
    assert _same(transposed_storage, value, chunk_bytes=4096)
    assert _digest(value, chunk_bytes=4096) == _digest(transposed_storage, chunk_bytes=8192)
    assert all(block.numel() <= 4096 for block in _byte_chunks(transposed_storage, chunk_bytes=4096))
    transposed_storage[0, 0] = -0.0  # torch.equal floats would overlook this.
    assert not _same(value, transposed_storage, chunk_bytes=4096)


def test_noncontiguous_long_context_does_not_materialize_row_iterator_pools():
    import tracemalloc
    value = torch.arange(200_000, dtype=torch.float32).reshape(2, 100_000).t()
    assert not value.is_contiguous()
    tracemalloc.start()
    try:
        chunks = _byte_chunks(value, chunk_bytes=4096)
        first = next(chunks)
        _, peak = tracemalloc.get_traced_memory()
        assert torch.equal(first, value[0].contiguous().view(torch.uint8))
        assert peak < 65536  # First chunk must not retain 100,000 Python indices/views.
        chunks.close()
    finally:
        tracemalloc.stop()


def test_hash_workspace_is_inside_host_quota_before_snapshot_allocation(model):
    st = stage(model)
    c = cache(st, quotas=CacheQuotas(2_000_000, max_entries=1))
    forward(st, torch.tensor([[1, 2, 3]]), 0)
    capture(c, [1, 2, 3])
    inventory = c.storage_inventory()
    reserve = c.hash_workspace_reservation()
    assert reserve["hash_workspace_host_reserved_bytes"] == 3 * reserve["hash_chunk_bytes"]
    assert reserve["hash_workspace_gpu_max_reserved_bytes"] == 0
    assert inventory["host_tensor_bytes"] < inventory["host_budget_charge_bytes"] <= c.quotas.host_bytes
    assert dict(c.snapshot_tensors())["conversation.hash_workspace"].numel() == reserve["hash_chunk_bytes"]
    from dataclasses import replace
    existing = next(iter(c._entries.values())).host_bytes
    tight = cache(st, quotas=CacheQuotas(existing))
    with pytest.raises(ConversationCacheError, match="quota"):
        capture(tight, [1, 2, 3])
    assert tight.status()["entries"] == 0 and tight._hash_workspace is None

"""Same native execution trace, expressly not full-prefill/chat parity."""
from dataclasses import replace
import pytest
import torch
from test_v4_conversation_cache import stage, cache, identity, forward
from test_v4_kv_runtime import args
import v4_ref_cpu as R
from v4_native_trace import NativeTraceStore, CONTRACT
from v4_conversation_cache import ConversationCacheError


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("tier", [False, True])
def test_continuation_from_committed_decode_frontier_matches_uninterrupted_native_trace(tier):
    a = args(); model = a, R.build_oracle(a, 7)
    uninterrupted, resumed = stage(model, tier=tier), stage(model, tier=tier)
    c = cache(resumed); store = NativeTraceStore(c)
    trace = list(range(13))
    for value in (uninterrupted, resumed): forward(value, torch.tensor([trace]), 0)
    for pos in (13, 14, 15):
        token = pos + 3; trace.append(token)
        for value in (uninterrupted, resumed): forward(value, torch.tensor([[token]]), pos)
    point = store.capture(identity(), trace, horizon=32, drained=True)
    assert c.prepare_restore(identity(), trace) is None, "decode snapshot must never masquerade as normal prefill"
    resumed.reset()
    restored = store.restore(point, identity=identity(), prefix_ids=trace, horizon=32)
    assert restored["contract"] == CONTRACT and restored["gpu_qualified"] is False
    for pos in range(16, 24):
        token = pos + 3
        want, logits = forward(uninterrupted, torch.tensor([[token]]), pos)
        got = store.advance(token)
        assert torch.equal(want, got) and torch.equal(logits, resumed.logits_all(got, full_logits=False))
    with pytest.raises(ConversationCacheError): store.restore(point, identity=replace(identity(), tenant="other"), prefix_ids=trace, horizon=32)
    with pytest.raises(ConversationCacheError): store.restore(point, identity=identity(), prefix_ids=trace, horizon=33)


def test_trace_refuses_changed_state_non_greedy_shape_and_exhausted_horizon():
    a = args(); st = stage((a, R.build_oracle(a, 7))); c = cache(st); store = NativeTraceStore(c)
    forward(st, torch.tensor([[1, 2, 3]]), 0)
    point = store.capture(identity(), [1, 2, 3], horizon=4, drained=True)
    with pytest.raises(ConversationCacheError): store.advance([1, 2])
    with pytest.raises(ConversationCacheError): store.advance(1.5)
    store.advance(4)
    with pytest.raises(ConversationCacheError, match="exhausted"): store.advance(5)
    store.restore(point, identity=identity(), prefix_ids=[1, 2, 3], horizon=4)
    st.layers[0].attn.kv_cache.add_(0)
    with pytest.raises(ConversationCacheError, match="changed outside"): store.advance(4)
    st._spec = True
    with pytest.raises(ConversationCacheError, match="greedy"): store.capture(identity(), [1, 2, 3], horizon=4, drained=True)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA continuation gate unavailable")
def test_actual_cuda_native_trace_restores_and_executes_original_single_token_math():
    import v4_stage as V
    import v4_kernels_cpu as K
    from v4_conversation_cache import StageConversationCache, CacheQuotas
    if K.backend() != "cpu": pytest.skip("BF16 trace twin uses Torch reference kernels; native FP4 release gate is separate")
    a = args(); oracle = R.build_oracle(a, 7)
    V.ref().precompute_freqs_cis.cache_clear()
    def gpu_stage():
        value = V.Stage(0, a.n_layers, a, head=True, tail=True, device="cuda")
        for target, source in zip(value.layers, oracle.layers): target.load_state_dict(source.state_dict())
        value.embed_tokens.load_state_dict(oracle.embed.state_dict()); value.norm.load_state_dict(oracle.norm.state_dict())
        value.lm_head.load_state_dict(oracle.head.state_dict())
        with torch.no_grad():
            for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"): getattr(value, name).copy_(getattr(oracle, name))
        return value
    try:
        control, resumed = gpu_stage(), gpu_stage()
        store = NativeTraceStore(StageConversationCache(resumed, CacheQuotas(32 << 20, gpu_bytes=32 << 20), enabled=True))
        prefix = list(range(13))
        for value in (control, resumed): forward(value, torch.tensor([prefix], device="cuda"), 0)
        point = store.capture(identity(), prefix, horizon=32, drained=True)
        resumed.reset(); store.restore(point, identity=identity(), prefix_ids=prefix, horizon=32)
        for pos in range(13, 24):
            wanted, logits = forward(control, torch.tensor([[pos]], device="cuda"), pos)
            got = store.advance(pos)
            assert torch.equal(wanted, got) and torch.equal(logits, resumed.logits_all(got, full_logits=False))
    finally:
        V.ref().precompute_freqs_cis.cache_clear()

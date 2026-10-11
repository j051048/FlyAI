"""Only unreachable future storage is omitted; all readable native state is exact."""
import pytest
import torch
from test_v4_conversation_cache import stage, cache, identity, forward
from test_v4_kv_runtime import args
import v4_ref_cpu as R
from v4_conversation_cache import _buffers, _size, _metadata, _byte_chunks, _digest


@pytest.mark.parametrize("tier", [False, True])
def test_smaller_snapshot_retains_all_valid_history_and_future_poison_is_rewritten(tier):
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        a = args(); model = a, R.build_oracle(a, 7)
        control, restored = stage(model, tier=tier), stage(model, tier=tier)
        c = cache(restored)
        prefix = list(range(13))
        for value in (control, restored): forward(value, torch.tensor([prefix]), 0)
        entry = c.capture(identity(), prefix, committed_frontier=13, drained=True)
        saved = c._entries[entry]
        assert _size(saved.buffers) < _size(_buffers(restored, None))
        restored.reset()
        runtime = restored._kv_runtime
        for layer in restored.layers:
            attn = layer.attn
            if not attn.compress_ratio: continue
            rows = 13 // attn.compress_ratio
            if runtime is None:
                attn.kv_cache[:, attn.window_size + rows:].fill_(float("nan"))
                if attn.indexer is not None: attn.indexer.kv_cache[:, rows:].fill_(float("nan"))
            else:
                state = runtime.entries[id(attn)]
                state["history"][:, rows:].fill_(float("nan"))
                if state["index_history"] is not None: state["index_history"][:, rows:].fill_(float("nan"))
        result = c.commit_restore(c.prepare_restore(identity(), prefix, entry_id=entry))
        assert result["state_digest"] == saved.state_digest
        for pos in range(13, 25):
            token = torch.tensor([[pos]])
            wanted, logits = forward(control, token, pos)
            got, actual = forward(restored, token, pos)
            assert torch.equal(wanted, got) and torch.equal(logits, actual)
    finally:
        torch.set_num_threads(previous)


def test_noncontiguous_history_hash_uses_large_contiguous_slabs_with_identical_bytes():
    whole = torch.arange(2 * 128 * 64, dtype=torch.float32).reshape(2, 128, 64)
    prefix = whole[:, :16]
    assert not prefix.is_contiguous()
    chunks = list(_byte_chunks(prefix, chunk_bytes=4096))
    assert len(chunks) == 2, "one contiguous slab per batch, not one driver readback per head row"
    assert _digest(prefix, chunk_bytes=4096) == _digest(prefix.contiguous(), chunk_bytes=17)

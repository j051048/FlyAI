"""Native KV transfer reductions, with full valid-prefix/reference semantics."""
import pytest
import torch
import v4_ref_cpu as R
import v4_stage as V
from test_v4_kv_runtime import args, stage, forward, assert_state_equal


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def track(runtime, monkeypatch):
    reads = []
    host = {entry[name].untyped_storage().data_ptr() for entry in runtime.entries.values()
            for name in ("history", "index_history") if entry[name] is not None}
    original = runtime._copy_host_rows
    def copy(target, source, batch):
        if source.untyped_storage().data_ptr() in host:
            reads.append((source.shape[1], batch))
        return original(target, source, batch)
    monkeypatch.setattr(runtime, "_copy_host_rows", copy)
    return reads


def test_prefill_never_loads_future_history_and_decode_loads_only_old_rows(monkeypatch):
    a = args(); oracle = R.build_oracle(a, 7)
    reference, tiered = stage(oracle, a), stage(oracle, a, layer=True)
    reads = track(tiered._kv_runtime, monkeypatch)
    token = torch.arange(7).view(1, -1)
    assert torch.equal(forward(reference, token, 0), forward(tiered, token, 0))
    assert reads == []
    for pos in range(7, 17):
        reads.clear()
        token = torch.tensor([[pos]])
        assert torch.equal(forward(reference, token, pos), forward(tiered, token, pos))
        assert reads == [(pos // 4, 1), (pos // 4, 1)] + ([(pos // 8, 1)] if pos // 8 else [])
    assert_state_equal(reference, tiered)
    assert tiered._kv_runtime.transfer_status()["new_rows_not_loaded"] > 0
    assert tiered._kv_runtime.h2d_bytes == 0, "CPU emulation must never claim DMA"


def test_unchanged_single_owner_reuses_prefix_then_history_or_workspace_mutation_invalidates(monkeypatch):
    a = args()
    M = R.load_ref(); V._set_globals(M, a)
    oracle = R.build_oracle(a, 7)
    common = dict(device="cpu", kv_gpu_budget_bytes=2_000_000, kv_host_budget_bytes=10_000_000)
    reference = V.Stage(1, 2, a, device="cpu")
    tiered = V.Stage(1, 2, a, kv_placement="layer", kv_reference=True, **common)
    for value in (reference, tiered):
        value.layers[0].load_state_dict(oracle.layers[1].state_dict())
    runtime = tiered._kv_runtime; reads = track(runtime, monkeypatch)
    torch.manual_seed(7)
    def compare(pos, count):
        h = torch.randn(1, count, a.hc_mult, a.dim, dtype=torch.bfloat16)
        token = torch.full((1, count), 3, dtype=torch.long)
        assert torch.equal(reference.forward(h, token, pos), tiered.forward(h, token, pos))
    compare(0, 13)
    reads.clear(); compare(13, 1)
    assert reads == [], "same owner/layout/history keeps its complete valid prefix"
    entry = next(iter(runtime.entries.values()))
    entry["history"].add_(0)  # Same values, new provenance version.
    reads.clear(); compare(14, 1)
    assert reads == [(3, 1), (3, 1)]
    runtime.workspace.add_(0)
    reads.clear(); compare(15, 1)  # New compressed row also changes the operand layout.
    assert reads == [(3, 1), (3, 1)]
    assert runtime.transfer_status()["prefix_rows_reused"] == 3
    runtime.invalidate_workspace(); reads.clear(); compare(16, 1)
    assert reads == [(4, 1), (4, 1)]
    tiered.reset(); assert runtime._resident_prefix is None


def test_seek_snapshot_restore_and_exception_discard_workset():
    a = args(); s = stage(R.build_oracle(a, 7), a, layer=True)
    forward(s, torch.arange(13).view(1, -1), 0)
    runtime = s._kv_runtime
    snap = s._snapshot(); s._restore(snap)
    assert runtime._resident_prefix is None
    attn = s.layers[1].attn
    with pytest.raises(RuntimeError, match="injected"):
        with runtime.layer(attn, 13, 1, 1):
            raise RuntimeError("injected")
    assert runtime._active is None and runtime._resident_prefix is None

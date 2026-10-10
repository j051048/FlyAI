"""Conditional CUDA gates; CPU CI skips, no weights/network downloads.

The second test proves a native torch graph's fixed KV pointer contract, not
whole-V4 graph numerical parity. Neither test is a throughput measurement.
"""
from collections import deque
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from engines.deepseek_v4 import v4_conversation_cache as C
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")]


def _identity():
    return C.ConversationIdentity("cuda-test-tenant", "a" * 64, "cuda-torch-reference/1", "synthetic-fixture",
        "c" * 64, "d" * 64, "e" * 64, "cuda-ring", "boot:1", (("node", 1),))


def test_cuda_actual_v4_snapshot_restores_same_continuation_and_state_bytes():
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("V4 bfloat16 reference needs an Ampere-or-newer device")
    R = pytest.importorskip("v4_ref_cpu")
    V = pytest.importorskip("v4_stage")
    StageConversationCache, CacheQuotas = C.StageConversationCache, C.CacheQuotas
    _buffers, _metadata, _same = C._buffers, C._metadata, C._same
    args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
        dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=64)
    oracle = R.build_oracle(args, 7)
    with torch.device("cuda"), torch.no_grad():
        stage = V.Stage(0, args.n_layers, args, head=True, tail=True, device="cuda", runtime_metrics=False)
        for target, source in zip(stage.layers, oracle.layers):
            target.load_state_dict(source.state_dict(), strict=True)
        stage.embed_tokens.load_state_dict(oracle.embed.state_dict(), strict=True)
        stage.norm.load_state_dict(oracle.norm.state_dict(), strict=True)
        stage.lm_head.load_state_dict(oracle.head.state_dict(), strict=True)
        for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            getattr(stage, name).copy_(getattr(oracle, name))
        cache = StageConversationCache(stage, CacheQuotas(32 << 20, gpu_bytes=32 << 20), enabled=True)
        prefix = torch.arange(13, dtype=torch.long, device="cuda").view(1, -1)
        hidden = stage.forward(stage.embed(prefix), prefix, 0)
        token = stage.logits_all(hidden, full_logits=False).argmax(-1).view(1, 1)
        entry = cache.capture(_identity(), prefix, committed_frontier=13, drained=True,
                              tail_reply={"token": int(token.item())})
        pointers = {name: value.data_ptr() for name, value in _buffers(stage, None).items()}
        expected_h = stage.forward(stage.embed(token), token, 13).clone()
        expected_logits = stage.logits_all(expected_h, full_logits=False).clone()
        expected = {name: value.cpu().clone() for name, value in _buffers(stage, None).items()}
        stage.reset()
        restored = cache.commit_restore(cache.prepare_restore(_identity(), prefix, entry_id=entry))
        assert restored["restored_state_sha256"] == restored["state_digest"]
        assert pointers == {name: value.data_ptr() for name, value in _buffers(stage, None).items()}
        actual = stage.forward(stage.embed(token), token, 13)
        assert torch.equal(actual, expected_h)
        assert torch.equal(stage.logits_all(actual, full_logits=False), expected_logits)
        assert _same(expected, _buffers(stage, None), workspace=cache._hash_workspace,
                     chunk_bytes=cache.hash_workspace_reservation()["hash_chunk_bytes"])
        assert _metadata(stage, None)["main"]["_pos"] == 14
        assert cache.storage_inventory()["host_budget_charge_bytes"] <= cache.quotas.host_bytes


class _GraphStage:
    """Minimal fixed-buffer owner for native CUDA graph memory correctness."""
    def __init__(self):
        attention = torch.nn.Module()
        attention.compress_ratio = 0
        attention.register_buffer("kv_cache", torch.arange(128, device="cuda", dtype=torch.float32).reshape(1, 8, 16))
        attention.register_buffer("freqs_cis", torch.zeros(32, 8, device="cuda"))
        layer = torch.nn.Module()
        layer.attn = attention
        self.layers = torch.nn.ModuleList([layer])
        self.args = SimpleNamespace(max_seq_len=32, vocab_size=512)
        self.device, self.lo, self.hi, self.head, self.tail = "cuda", 0, 1, True, True
        self._kv_runtime, self._dspark_capable, self._tap_ids = None, False, ()
        self._spec_depth, self._spec, self._dspark, self._replaying = 2, False, False, False
        self._pos, self._last_tap = 3, {}
        self._spec_ckpts = deque(maxlen=2)
    def _owned_modules(self):
        yield self.layers
    def reset(self):
        self.layers[0].attn.kv_cache.zero_()
        self._pos = 0
        self._last_tap = {}
        self._spec_ckpts.clear()


def test_native_cuda_graph_reads_restored_fixed_kv_without_recapture():
    StageConversationCache, CacheQuotas = C.StageConversationCache, C.CacheQuotas
    with torch.no_grad():
        stage = _GraphStage()
        cache = StageConversationCache(stage, CacheQuotas(8 << 20, gpu_bytes=8 << 20), enabled=True)
        entry = cache.capture(_identity(), [1, 2, 3], committed_frontier=3, drained=True)
        buffer = stage.layers[0].attn.kv_cache
        pointer, original = buffer.data_ptr(), buffer.clone()
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                buffer.square().sum(dim=1)
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = buffer.square().sum(dim=1)
        graph.replay()
        expected = output.clone()
        buffer.add_(7)
        graph.replay()
        assert not torch.equal(output, expected)
        stage.reset()
        cache.commit_restore(cache.prepare_restore(_identity(), [1, 2, 3], entry_id=entry))
        assert buffer.data_ptr() == pointer and torch.equal(buffer, original)
        graph.replay()  # The original graph, no capture after restoration.
        assert torch.equal(output, expected)
        assert cache.storage_inventory()["host_budget_charge_bytes"] <= cache.quotas.host_bytes

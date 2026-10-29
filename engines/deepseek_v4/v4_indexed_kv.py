"""Exact, ordered top-K materialization for the native single-token MLA path.

The native Indexer and Compressor still execute. Only sparse_attn's KV operand
is compacted, in its original index order (including invalid entries). This
does not change the selected experts, attention keys, sink or reduction order.
CUDA numerical qualification remains separate from CPU reference evidence.
"""
from types import FunctionType, MethodType
import torch


def install(attention, runtime):
    original = attention.forward.__func__
    if "sparse_attn" not in original.__globals__:
        raise ValueError("indexed KV gather requires the original native Attention.forward")
    sparse = original.__globals__["sparse_attn"]

    def selected(q, kv, sink, indexes, scale):
        active = runtime._gather_active
        if active is None or active["attention"] is not attention:
            return sparse(q, kv, sink, indexes, scale)
        if q.shape[1] != 1 or indexes.shape[:2] != q.shape[:2]:
            raise ValueError("indexed KV gather accepts native single-token queries only")
        batch, _, width = indexes.shape
        raw = indexes.detach().to("cpu", dtype=torch.long)[:, 0]
        runtime.gather_index_d2h_bytes += raw.numel() * raw.element_size() if not runtime.reference else 0
        if ((raw < -1) | (raw >= kv.shape[1])).any():
            raise ValueError("native Indexer produced an out-of-range KV position")
        win, previous = attention.window_size, active["previous"]
        # Only cold compressed rows are copied. Ring/new rows stay on device.
        cold = (raw >= win) & (raw < win + previous)
        staging = torch.zeros((batch, width, attention.head_dim), dtype=kv.dtype,
                              device="cpu", pin_memory=not runtime.reference)
        for row in range(batch):
            chosen = cold[row].nonzero().flatten()
            if chosen.numel():
                staging[row, chosen] = active["history"][row].index_select(0, raw[row, chosen] - win)
        compact = staging.to(kv.device, non_blocking=not runtime.reference)
        local_indexes = torch.where(cold.to(indexes.device), 0, indexes[:, 0].long().clamp_min(0))
        local = kv.gather(1, local_indexes.unsqueeze(-1).expand(-1, -1, kv.shape[-1]))
        compact = torch.where(cold.to(kv.device).unsqueeze(-1), compact, local)
        remapped = torch.arange(width, device=indexes.device, dtype=indexes.dtype).view(1, 1, width).expand(batch, 1, -1)
        remapped = torch.where(indexes >= 0, remapped, -1)
        amount = staging.numel() * staging.element_size()
        runtime.h2d_bytes += amount if not runtime.reference else 0
        runtime.gather_calls += 1
        runtime.gather_selected_rows += int(cold.sum())
        return sparse(q, compact, sink, remapped, scale)

    namespace = dict(original.__globals__, sparse_attn=selected)
    patched = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    patched.__kwdefaults__ = original.__kwdefaults__
    attention.forward = MethodType(patched, attention)

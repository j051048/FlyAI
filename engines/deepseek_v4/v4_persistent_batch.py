"""Persistent request slots and ragged one-token attention.

Q/KV/O projections, hyper-connections and MoE see a real request batch in one
Stage.forward. Native index selection, compressor recurrence and sparse reads
use each row's own position and storage view. No complete KV gather/scatter is
performed per token. This is an eager numerical contract, not a ragged CUDA
graph claim. A bounded write-set journal restores a failed local transaction.
"""
from contextlib import contextmanager
from types import MethodType

import torch

from v4_runtime_init import bind_attention_aliases

CONTRACT = "v4-native-persistent-ragged-greedy/3"
COMPUTE = "native_persistent_ragged_batch"


def journal_layout(stage, buffers, max_jobs):
    """Upper bound: one ring row, one compressed row, all fixed recurrence."""
    result = {}
    for name, value in buffers.items():
        rows = (2 if name.endswith(".kv") else 1) if name.endswith((".kv", ".index")) else value.shape[1]
        result[name] = torch.empty((max_jobs, rows, *value.shape[2:]), dtype=value.dtype, device=value.device)
    return result


def journal_bytes(buffers, max_jobs):
    total = 0
    for name, value in buffers.items():
        rows = (2 if name.endswith(".kv") else 1) if name.endswith((".kv", ".index")) else value.shape[1]
        total += max_jobs * rows * value[0, 0].numel() * value.element_size()
    return total


def write_set(stage, slot, position):
    """Enumerate ALL reference one-token writes, including boundary shifts."""
    for index, layer in enumerate(stage.layers):
        attn = layer.attn
        bank = slot["buffers"][f"main.{index}.kv"]
        yield f"main.{index}.kv", bank[:, position % attn.window_size:position % attn.window_size + 1], 0
        ratio = attn.compress_ratio
        if ratio and (position + 1) % ratio == 0:
            row = attn.window_size + position // ratio
            yield f"main.{index}.kv", bank[:, row:row + 1], 1
            name = f"main.{index}.index"
            if name in slot["buffers"]:
                yield name, slot["buffers"][name][:, position // ratio:position // ratio + 1], 0
        for label in ("attention", "indexer"):
            for field in ("kv_state", "score_state"):
                name = f"main.{index}.{label}.{field}"
                if name in slot["buffers"]:
                    yield name, slot["buffers"][name], 0


@contextmanager
def row_state(attn, slot, index):
    owners = [(attn, "kv_cache", f"main.{index}.kv")]
    indexer = getattr(attn, "indexer", None)
    if indexer is not None:
        owners.append((indexer, "kv_cache", f"main.{index}.index"))
    for label, owner in (("attention", attn), ("indexer", indexer)):
        compressor = getattr(owner, "compressor", None)
        if compressor is not None:
            owners.extend((compressor, field, f"main.{index}.{label}.{field}")
                          for field in ("kv_state", "score_state"))
    previous = [(owner, field, getattr(owner, field)) for owner, field, _ in owners]
    try:
        for owner, field, name in owners:
            setattr(owner, field, slot["buffers"][name])
        bind_attention_aliases(attn, rebind=True)
        yield
    finally:
        for owner, field, value in previous:
            setattr(owner, field, value)
        bind_attention_aliases(attn, rebind=True)


def attention_forward(attn, x, positions, slots, index, model):
    """Native Attention transcription, batched projections and exact row reads.

    Narrow native topk is never padded or replaced. The sparse kernel sees the
    canonical slot's entire history VIEW, so only selected rows are read by the
    kernel; no complete history tensor is copied into a batch workspace.
    """
    count, seqlen, _ = x.shape
    if count != len(positions) or seqlen != 1:
        raise ValueError("persistent attention needs one native token per request")
    rd, win, ratio = attn.rope_head_dim, attn.window_size, attn.compress_ratio
    qr = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + attn.eps)
    kv = attn.kv_norm(attn.wkv(x))
    for row, position in enumerate(positions):
        freqs = attn.freqs_cis[position:position + 1]
        model.apply_rotary_emb(q[row:row + 1, ..., -rd:], freqs)
        model.apply_rotary_emb(kv[row:row + 1, ..., -rd:], freqs)
    model.act_quant(kv[..., :-rd], 64, model.scale_fmt, model.scale_dtype, True)
    outputs = []
    for row, (position, slot) in enumerate(zip(positions, slots)):
        with row_state(attn, slot, index):
            topk = model.get_window_topk_idxs(win, 1, 1, position)
            if ratio:
                offset = 1 if position == 0 else win
                if attn.indexer is not None:
                    compressed = attn.indexer(x[row:row + 1], qr[row:row + 1], position, offset).int()
                else:
                    compressed = model.get_compress_topk_idxs(ratio, 1, 1, position, offset)
                topk = torch.cat([topk, compressed], -1)
            if position == 0:
                attn.kv_cache[:1, :1] = kv[row:row + 1]
                read = kv[row:row + 1]
                if ratio:
                    extra = attn.compressor(x[row:row + 1], position)
                    if extra is not None:
                        read = torch.cat([read, extra], 1)
            else:
                attn.kv_cache[:1, position % win] = kv[row:row + 1].squeeze(1)
                if ratio:
                    attn.compressor(x[row:row + 1], position)
                read = attn.kv_cache[:1]
            output = model.sparse_attn(q[row:row + 1], read, attn.attn_sink, topk, attn.softmax_scale)
            model.apply_rotary_emb(output[..., -rd:], attn.freqs_cis[position:position + 1], True)
            outputs.append(output)
    output = torch.cat(outputs, 0).view(count, 1, attn.n_local_groups, -1)
    weight = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
    output = torch.einsum("bsgd,grd->bsgr", output, weight)
    return attn.wo_b(output.flatten(2))


@contextmanager
def bound_attention(stage, slots, positions):
    saved = []
    try:
        for index, layer in enumerate(stage.layers):
            attn = layer.attn
            old = attn.__dict__.get("forward")
            present = "forward" in attn.__dict__
            saved.append((attn, present, old))
            def forward(owner, x, ignored_position, *extra, _index=index):
                if extra:
                    raise ValueError("persistent ragged attention refuses alternate attention arguments")
                return attention_forward(owner, x, positions, slots, _index, stage._M)
            attn.forward = MethodType(forward, attn)
        yield
    finally:
        for attn, present, old in saved:
            if present:
                attn.forward = old
            else:
                del attn.forward


def forward(run, jobs, hidden, ids, positions, *, capture_graph=True):
    from v4_batch_invariant import request_row_math
    run._check()
    if type(positions) is int:
        positions = [positions] * len(jobs)
    if (not isinstance(jobs, (list, tuple)) or not 1 <= len(jobs) <= run.max_jobs
            or len(set(jobs)) != len(jobs) or not isinstance(positions, (list, tuple))
            or len(positions) != len(jobs) or any(type(pos) is not int or not 0 <= pos < run.stage.args.max_seq_len for pos in positions)
            or type(capture_graph) is not bool or tuple(ids.shape) != (len(jobs), 1)
            or hidden.shape[:2] != ids.shape or ids.dtype != torch.long
            or ids.device != hidden.device or hidden.device != next(iter(run.buffers.values())).device):
        raise ValueError("persistent native batch requires unique bounded request rows and independent positions")
    slots = [run.active.get(job) for job in jobs]
    if any(slot is None or slot["pos"] != pos for slot, pos in zip(slots, positions)):
        raise ValueError("persistent native frame differs from its live independent frontiers")
    if run._batch_active:
        raise ValueError("persistent native batch transaction already owned")
    changes = [(row, name, target, offset) for row, (slot, pos) in enumerate(zip(slots, positions))
               for name, target, offset in write_set(run.stage, slot, pos)]
    saved_position, saved_taps = run.stage._pos, run.stage._last_tap
    run._batch_active = True
    copied = 0
    try:
        with torch.no_grad():
            for row, name, target, offset in changes:
                backup = run.journal[name][row:row + 1, offset:offset + target.shape[1]]
                backup.copy_(target)
                copied += target.numel() * target.element_size()
            try:
                run.stage._pos = positions[0]
                with bound_attention(run.stage, slots, positions), request_row_math():
                    result = run.stage.forward(hidden, ids, positions[0])
                if run.stage._pos != positions[0] + 1:
                    raise ValueError("persistent native forward did not advance exactly one token")
                if run.stage._last_tap:
                    raise ValueError("persistent native batch refuses speculative taps")
            except Exception:
                try:
                    for row, name, target, offset in changes:
                        target.copy_(run.journal[name][row:row + 1, offset:offset + target.shape[1]])
                    if ids.device.type == "cuda":
                        torch.cuda.current_stream(ids.device).synchronize()
                except Exception:
                    run._batch_fault = True
                    raise
                run.counters["state_rollback_copy_bytes"] += copied
                raise
            for slot, pos in zip(slots, positions):
                slot["pos"] = pos + 1
    finally:
        run.stage._pos, run.stage._last_tap = saved_position, saved_taps
        run._batch_active = False
        run.counters["state_journal_copy_bytes"] += copied
    run.counters["batch_forward_calls" if len(jobs) > 1 else "scalar_forward_calls"] += 1
    if len(jobs) > 1:
        run.counters["batched_request_rows"] += len(jobs)
    if len(set(positions)) > 1:
        run.counters["ragged_forward_calls"] += 1
    return result

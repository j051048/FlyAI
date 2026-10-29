"""Bounded complete committed-advance/MTP graphs; native verifier is unchanged.

The cache advances only over committed positions. Position and input tokens are
live graph inputs; native request shapes and ring/index ordering are preserved.
This is independent of request batching and does not speculate unknown taps.
"""
from copy import deepcopy
import torch

SCHEMA = "v4-committed-draft-chain/1"


def attention(attn, x, main_x, position, topk, model):
    """Native DSparkAttention decode with device-side position selection."""
    b, width, _ = x.shape
    rd = attn.rope_head_dim
    main_freq = attn.freqs_cis.index_select(0, position)
    main_kv = attn.kv_norm(attn.wkv(main_x))
    model.apply_rotary_emb(main_kv[..., -rd:], main_freq)
    model.act_quant(main_kv[..., :-rd], 64, model.scale_fmt, model.scale_dtype, True)
    offsets = torch.arange(width, device=x.device, dtype=torch.long)
    freq = attn.freqs_cis.index_select(0, position + 1 + offsets)
    q = attn.wq_b(attn.q_norm(attn.wq_a(x))).unflatten(-1, (attn.n_local_heads, attn.head_dim))
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + attn.eps)
    model.apply_rotary_emb(q[..., -rd:], freq)
    kv = attn.kv_norm(attn.wkv(x))
    model.apply_rotary_emb(kv[..., -rd:], freq)
    model.act_quant(kv[..., :-rd], 64, model.scale_fmt, model.scale_dtype, True)
    attn.kv_cache[:b].index_copy_(1, position % attn.window_size, main_kv)
    out = model.sparse_attn(q, torch.cat([attn.kv_cache[:b], kv], 1), attn.attn_sink, topk, attn.softmax_scale)
    model.apply_rotary_emb(out[..., -rd:], freq, True)
    out = out.view(b, width, attn.n_local_groups, -1)
    wa = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
    return attn.wo_b(torch.einsum("bsgd,grd->bsgr", out, wa).flatten(2))


def cache_advance(tail, hidden, position, model):
    main = tail.mtp[0].main_norm(tail.mtp[0].main_proj(hidden))
    for block in tail.mtp:
        attn = block.attn
        rd = attn.rope_head_dim
        kv = attn.kv_norm(attn.wkv(main))
        model.apply_rotary_emb(kv[..., -rd:], attn.freqs_cis.index_select(0, position))
        model.act_quant(kv[..., :-rd], 64, model.scale_fmt, model.scale_dtype, True)
        attn.kv_cache[:hidden.shape[0]].index_copy_(1, position % attn.window_size, kv)


def program(tail, ids, hidden, position, topk, model):
    """Same M=1 cache advances and one full kept forward, including Markov head."""
    for index in range(ids.shape[1] - 1):
        cache_advance(tail, hidden[:, index:index+1], position + index, model)
    pos = position + ids.shape[1] - 1
    token = ids[:, -1]
    h, main = tail.mtp[0].forward_embed(hidden[:, -1:], token)
    for block in tail.mtp:
        residual = h
        x, post, comb = block.hc_pre(h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
        x = attention(block.attn, block.attn_norm(x), main, pos, topk, model)
        h = block.hc_post(x, residual, post, comb)
        residual = h
        x, post, comb = block.hc_pre(h, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
        x = block.ffn(block.ffn_norm(x), token)
        h = block.hc_post(x, residual, post, comb)
    return tail.mtp[-1].forward_head(h, token)


class DraftChain:
    def __init__(self, tail, *, graph_bytes, max_graphs=16, reference=False):
        if (type(graph_bytes) is not int or graph_bytes <= 0 or type(max_graphs) is not int
                or not 1 <= max_graphs <= 128 or type(reference) is not bool
                or reference != (torch.device(tail.device).type == "cpu")):
            raise ValueError("draft chain needs explicit graph bounds and honest device mode")
        if tail.temperature != 0 or getattr(tail.stage, "_expert_cache", None) is not None:
            raise ValueError("draft chain requires greedy resident experts")
        self.tail, self.reference = tail, reference
        self.graph_bytes, self.max_graphs = graph_bytes, max_graphs
        self.cache = [layer.attn.kv_cache for layer in tail.mtp]
        backup_bytes = sum(t.numel() * t.element_size() for t in self.cache)
        if backup_bytes >= graph_bytes:
            raise ValueError("draft chain quota does not hold complete state backup")
        self.backup = [torch.empty_like(value) for value in self.cache]
        self.used = backup_bytes
        self.retained = 0
        self.capture_disabled = False
        self.entries, self.refused = {}, set()
        self.failed = False
        self._stamp = None
        self.counters = dict(captures=0, replays=0, reference_eager=0, fallbacks=0,
                             capture_failures=0, budget_refusals=0)

    def config(self):
        return dict(schema=SCHEMA, graph_bytes=self.graph_bytes, max_graphs=self.max_graphs,
                    coverage="committed_advance_and_full_mtp", reference=self.reference,
                    graph_qualified=False, uncommitted_taps=False)

    def status(self):
        return {**self.config(), **self.counters, "used_bytes": self.used,
                "allocator_retained_bytes": self.retained,
                "live_graphs": len(self.entries), "poisoned": self.failed}

    def tensors(self):
        for index, value in enumerate(self.backup):
            yield f"draft_chain.backup.{index}", value
        for key, entry in self.entries.items():
            for name, value in entry["inputs"].items():
                yield f"draft_chain.{key}.{name}", value
            for index, value in enumerate(entry["outputs"]):
                yield f"draft_chain.{key}.output.{index}", value

    def _check(self):
        from v4_conversation_cache import _model_stamp
        from v4_runtime_init import tensor_storage_signature
        current = (_model_stamp(self.tail.stage, self.tail), tuple(tensor_storage_signature(
            layer.attn.kv_cache) for layer in self.tail.mtp))
        if self._stamp is None:
            self._stamp = current
        if self.failed or self._stamp != current:
            raise RuntimeError("draft chain ownership/model changed or capture state is poisoned")

    def _save(self):
        for target, value in zip(self.backup, self.cache):
            target.copy_(value)

    def _restore(self):
        for target, value in zip(self.cache, self.backup):
            target.copy_(value)

    def _inputs(self, ids, hidden, pos):
        win = self.tail.args.window_size
        kept = pos + ids.shape[1] - 1
        indices = torch.cat([torch.arange(min(win, kept + 1), device=ids.device),
                             win + torch.arange(self.tail.block_size, device=ids.device)]).int()
        return dict(ids=ids.clone(), hidden=hidden.clone(),
                    position=torch.tensor([pos], dtype=torch.long, device=ids.device),
                    topk=indices.view(1, 1, -1).expand(ids.shape[0], self.tail.block_size, -1).contiguous())

    def _program(self, values):
        return program(self.tail, values["ids"], values["hidden"], values["position"], values["topk"], self.tail.stage._M)

    def _supports_capture(self):
        import v4_dspark_moe as pair
        for block in self.tail.mtp:
            ffn = block.ffn
            # nn.Linear is the explicit synthetic dense GPU probe, not native
            # packed-MoE qualification. Production uses the known pair backend.
            if isinstance(ffn, torch.nn.Linear):
                continue
            bank=getattr(ffn,"_grouped_bank",None)
            if (getattr(ffn.forward, "__func__", None) is not pair.draft_forward
                    or self.tail.block_size <= 1 or pair._WORLD_SIZE != 1
                    or ffn.gate.hash or not bank):
                return False
            block_n=pair._grouped()._BLOCK_N
            if bank["w13"].shape[1] % block_n or bank["w2"].shape[1] % block_n:
                return False
        return True

    def _capture(self, values):
        self._save()
        device = values["ids"].device
        before = torch.cuda.memory_reserved(device)
        metrics = getattr(self.tail.stage, "_runtime_metrics", None)
        work = deepcopy(metrics.work) if metrics else None
        stats = [{name: deepcopy(value) for name, value in vars(layer.ffn).items()
                  if name.startswith("_draft_")} for layer in self.tail.mtp]
        side = None
        try:
            side = torch.cuda.Stream(device=device)
            caller = torch.cuda.current_stream(device)
            side.wait_stream(caller)
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._restore(); self._program(values)
            caller.wait_stream(side)
            torch.cuda.synchronize(device)
            self._restore()
            side.wait_stream(caller)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(side):
                graph.capture_begin()
                try:
                    outputs = self._program(values)
                finally:
                    graph.capture_end()
            caller.wait_stream(side)
            size = max(sum(v.numel()*v.element_size() for v in (*values.values(), *outputs)),
                       torch.cuda.memory_reserved(device)-before)
            if self.used + self.retained + size > self.graph_bytes:
                raise MemoryError("draft chain capture exceeded its private pool reservation")
            self.used += size
            self.counters["captures"] += 1
            return dict(graph=graph, inputs=values, outputs=outputs)
        finally:
            try:
                if side is not None:
                    torch.cuda.current_stream(device).wait_stream(side)
                self._restore()
            except Exception:
                self.failed = True
                raise
            finally:
                if metrics:
                    metrics.work = work
                for layer, saved in zip(self.tail.mtp, stats):
                    for name in list(vars(layer.ffn)):
                        if name.startswith("_draft_"):
                            delattr(layer.ffn, name)
                    for name, value in saved.items():
                        setattr(layer.ffn, name, value)

    def run(self, ids, hidden, start_pos):
        self._check()
        if (not torch.is_tensor(ids) or not torch.is_tensor(hidden)
                or ids.ndim != 2 or hidden.ndim != 3 or ids.shape[0] != 1
                or ids.dtype != torch.long or hidden.shape[:2] != ids.shape
                or hidden.shape[-1] != self.tail.hidden_dim or hidden.dtype != self.tail.dtype
                or ids.device != hidden.device or hidden.device != self.cache[0].device
                or type(start_pos) is not int or start_pos < 1
                or not 1 <= ids.shape[1] <= self.tail.block_size + 1
                or start_pos + ids.shape[1] + self.tail.block_size > self.tail.args.max_seq_len):
            raise ValueError("draft chain requires a bounded live committed token run")
        if self.reference:
            values = self._inputs(ids, hidden, start_pos)
            try:
                result = self._program(values)
            except Exception:
                self.failed = True
                raise
            self.counters["reference_eager"] += 1
            return result
        if not self._supports_capture():
            self.counters["fallbacks"] += 1
            return None
        key = (ids.shape[1], min(self.tail.args.window_size, start_pos + ids.shape[1]))
        entry = self.entries.get(key)
        if entry is None:
            if self.capture_disabled or key in self.refused or len(self.entries) >= self.max_graphs:
                self.counters["budget_refusals"] += 1
                self.counters["fallbacks"] += 1
                return None
            try:
                before_reserved = torch.cuda.memory_reserved(ids.device)
                values = self._inputs(ids, hidden, start_pos)
                entry = self._capture(values)
                self.entries[key] = entry
            except (RuntimeError, MemoryError) as error:
                if self.failed:
                    raise
                self.retained += max(0, torch.cuda.memory_reserved(ids.device) - before_reserved)
                if self.used + self.retained > self.graph_bytes:
                    self.failed = True
                    raise RuntimeError("failed draft capture exceeded its retained allocator quota") from error
                if isinstance(error, MemoryError):
                    self.capture_disabled = True
                if len(self.refused) < self.max_graphs:
                    self.refused.add(key)
                self.counters["capture_failures"] += 1
                self.counters["fallbacks"] += 1
                return None
        entry["inputs"]["ids"].copy_(ids)
        entry["inputs"]["hidden"].copy_(hidden)
        entry["inputs"]["position"].fill_(start_pos)
        try:
            entry["graph"].replay()
        except RuntimeError:
            self.failed = True
            raise
        self.counters["replays"] += 1
        metrics = getattr(self.tail.stage, "_runtime_metrics", None)
        from v4_dspark_moe import draft_forward
        for layer in self.tail.mtp:
            ffn = layer.ffn
            if getattr(ffn.forward, "__func__", None) is draft_forward:
                from v4_dspark_moe import _grouped
                pairs = self.tail.block_size * ffn.n_activated_experts
                chunks = (pairs + _grouped()._BLOCK_M - 1) // _grouped()._BLOCK_M
                ffn._draft_steps = getattr(ffn, "_draft_steps", 0) + 1
                ffn._draft_pairs = getattr(ffn, "_draft_pairs", 0) + pairs
                widths = getattr(ffn, "_draft_pair_widths", {})
                ffn._draft_pair_widths = {**widths, pairs: widths.get(pairs, 0) + 1}
                ffn._draft_cuda_chunks = getattr(ffn, "_draft_cuda_chunks", 0) + chunks
                ffn._draft_cuda_grouped_gemms = getattr(ffn, "_draft_cuda_grouped_gemms", 0) + 2 * chunks
            if metrics and hasattr(ffn, "n_activated_experts"):
                metrics.resident_routes(self.tail.block_size * layer.ffn.n_activated_experts,
                                        role="draft", phase="decode")
        return tuple(value.clone() for value in entry["outputs"])

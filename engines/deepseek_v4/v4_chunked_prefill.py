"""Chunk cursor utilities and the local hybrid runtime's measured-history predictor.

ControlledPrefetcher proposes bounded expert IDs; HybridMoE schedules the copies
outside attention and keeps authoritative demand DMA as fallback. It is not an
oracle for a future Gate. ChunkedPrefillExecutor is a transport/cursor utility.
query_chunk_attention below is the actual Stage seam: it preserves full-prompt
projection, Compressor, HC and MoE shapes and chunks only independent attention
queries/index scores. It is graded against the unchanged reference before use.
"""
from __future__ import annotations

import collections
import dataclasses
import math
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch


def query_chunk_attention(attn, x, model, chunk_size):
    """Reference prefill arithmetic with bounded query/index-score temporaries.

    Full-prompt projections and the two original Compressor calls are retained.
    Every query sees the same ordered window/compressed indices and the same KV
    bytes. Output projection also retains its full-prompt shape. This therefore
    reduces index-score/sparse-attention query scratch, not every activation or
    expert weight working set. It is exclusively the start_pos==0 branch.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("prefill query chunk size must be a positive integer")
    bsz, seqlen, _ = x.size()
    win, ratio, rd = attn.window_size, attn.compress_ratio, attn.rope_head_dim
    freqs = attn.freqs_cis[:seqlen]
    if ratio:
        attn.compressor.kv_cache = attn.kv_cache[:, win:]
        attn.compressor.freqs_cis = attn.freqs_cis
        if attn.indexer is not None:
            attn.indexer.freqs_cis = attn.freqs_cis
    qr = q = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(q).unflatten(-1, (attn.n_local_heads, attn.head_dim))
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + attn.eps)
    model.apply_rotary_emb(q[..., -rd:], freqs)
    kv = attn.kv_norm(attn.wkv(x))
    model.apply_rotary_emb(kv[..., -rd:], freqs)
    model.act_quant(kv[..., :-rd], 64, model.scale_fmt, model.scale_dtype, True)
    indexer = getattr(attn, "indexer", None)
    iq = weights = None
    if indexer is not None:
        indexer.compressor.kv_cache = indexer.kv_cache
        indexer.compressor.freqs_cis = indexer.freqs_cis
        iq = indexer.wq_b(qr).unflatten(-1, (indexer.n_local_heads, indexer.head_dim))
        model.apply_rotary_emb(iq[..., -rd:], freqs)
        iq = model.rotate_activation(iq)
        model.fp4_act_quant(iq, model.fp4_block_size, True)
        indexer.compressor(x, 0)
        weights = indexer.weights_proj(x) * (indexer.softmax_scale * indexer.n_heads ** -0.5)
    if seqlen <= win:
        attn.kv_cache[:bsz, :seqlen] = kv
    else:
        cutoff = seqlen % win
        attn.kv_cache[:bsz, cutoff:win], attn.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
    if ratio:
        compressed = attn.compressor(x, 0)
        if compressed is not None:
            kv = torch.cat([kv, compressed], dim=1)
    o = torch.empty_like(q)
    for start in range(0, seqlen, chunk_size):
        end = min(seqlen, start + chunk_size)
        # Build only this query block's indices, with absolute prompt positions.
        positions = torch.arange(start, end, device=x.device)
        window = (positions[:, None] - win + 1).clamp(0) + torch.arange(min(seqlen, win), device=x.device)
        window = torch.where(window > positions[:, None], -1, window)
        indices = window.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()
        if ratio:
            n = seqlen // ratio
            if indexer is not None:
                score = torch.einsum("bshd,btd->bsht", iq[:, start:end], indexer.kv_cache[:bsz, :n])
                score = (score.relu_() * weights[:, start:end].unsqueeze(-1)).sum(dim=2)
                if model.world_size > 1:
                    raise ValueError("query-chunk prefill requires world_size=1")
                mask = torch.arange(n, device=x.device)[None, :] >= (positions + 1)[:, None] // ratio
                score += torch.where(mask, float("-inf"), 0)
                compressed_ids = score.topk(min(indexer.index_topk, n), dim=-1)[1]
                mask = compressed_ids >= (positions + 1)[None, :, None] // ratio
                compressed_ids = torch.where(mask, -1, compressed_ids + seqlen).int()
            else:
                compressed_ids = torch.arange(n, device=x.device).repeat(end - start, 1)
                mask = compressed_ids >= (positions + 1)[:, None] // ratio
                compressed_ids = torch.where(mask, -1, compressed_ids + seqlen).int().unsqueeze(0).expand(bsz, -1, -1).contiguous()
            indices = torch.cat([indices, compressed_ids], dim=-1)
        o[:, start:end].copy_(model.sparse_attn(q[:, start:end].contiguous(), kv, attn.attn_sink,
                                               indices, attn.softmax_scale))
    model.apply_rotary_emb(o[..., -rd:], freqs, True)
    o = o.view(bsz, seqlen, attn.n_local_groups, -1)
    wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    return attn.wo_b(o.flatten(2))


def query_chunk_block_cls(base_cls, model, chunk_size):
    """Per-instance Attention subclass; never modifies the vendored class."""
    base_attention = base_cls.attention_cls

    class QueryChunkAttention(base_attention):
        _prefill_query_chunks = 0

        def _gate_state(self):
            tensors = [self.kv_cache]
            indexer = getattr(self, "indexer", None)
            if indexer is not None:
                tensors.append(indexer.kv_cache)
            for owner in (self, indexer):
                compressor = getattr(owner, "compressor", None)
                if compressor is not None:
                    tensors.extend((compressor.kv_state, compressor.score_state))
            runtime = getattr(self, "_kv_runtime_owner", None)
            return [(tensor, runtime.snapshot_tensor(tensor) if runtime is not None else
                     tensor.detach().to(device="cpu", copy=True)) for tensor in tensors]

        def _put_state(self, state):
            runtime = getattr(self, "_kv_runtime_owner", None)
            for tensor, saved in state:
                if runtime is not None:
                    runtime.restore_tensor(tensor, saved)
                else:
                    tensor.copy_(saved)

        def _readback(self, tensor):
            runtime = getattr(self, "_kv_runtime_owner", None)
            return runtime.readback_tensor(tensor) if runtime is not None else tensor.detach().to("cpu")

        @staticmethod
        def _bits_equal(left, right):
            # Compare bytes, including signed zero and untouched NaN payloads
            # in unused batch rows. torch.equal is numerical for those cases.
            return (left.dtype == right.dtype and left.shape == right.shape and
                    torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))

        def forward(self, x, start_pos):
            if start_pos == 0 and x.shape[1] > chunk_size:
                # The first real input for each shape is a strict output AND
                # recurrence/storage gate, including Indexer top-k effects.
                # Gate snapshots live on CPU, never as extra device KV banks.
                # It costs a reference prefill on that first shape; it cannot
                # be used to claim a lower first-job peak. Subsequent jobs use
                # the proven query geometry. Parameter reload invalidates it.
                key = (tuple(x.shape), str(x.dtype), str(x.device),
                       tuple((id(parameter), parameter._version) for parameter in self.parameters()))
                verified = getattr(self, "_prefill_gate_verified", collections.OrderedDict())
                if key not in verified:
                    before = self._gate_state()
                    expected = super().forward(x, start_pos)
                    after = self._gate_state()
                    self._put_state(before)
                    try:
                        actual = query_chunk_attention(self, x, model, chunk_size)
                        equal = self._bits_equal(actual, expected) and all(
                            self._bits_equal(self._readback(tensor), saved) for tensor, saved in after)
                    except BaseException:
                        self._put_state(after)
                        raise
                    if not equal:
                        self._put_state(after)
                        raise RuntimeError("V4 query-chunk prefill failed reference output/state bit parity; "
                                           "disable V4_PREFILL_QUERY_CHUNK for this configuration")
                    verified[key] = None
                    while len(verified) > 32:
                        verified.popitem(last=False)
                    self._prefill_gate_verified = verified
                    self._prefill_gate_checks = getattr(self, "_prefill_gate_checks", 0) + 1
                    self._prefill_query_chunks += (x.shape[1] + chunk_size - 1) // chunk_size
                    return actual
                self._prefill_query_chunks += (x.shape[1] + chunk_size - 1) // chunk_size
                return query_chunk_attention(self, x, model, chunk_size)
            return super().forward(x, start_pos)

        def _load_from_state_dict(self, *args, **kwargs):
            self._prefill_gate_verified = collections.OrderedDict()
            return super()._load_from_state_dict(*args, **kwargs)

    class QueryChunkBlock(base_cls):
        attention_cls = QueryChunkAttention

    return QueryChunkBlock


@dataclasses.dataclass
class PrefillChunkState:
    """Carries persistent state across sequential prefill chunks of a single request."""
    request_id: str
    total_prompt_length: int
    chunk_size: int
    processed_tokens: int = 0
    chunk_index: int = 0
    is_completed: bool = False
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if type(self.total_prompt_length) is not int or self.total_prompt_length < 0:
            raise ValueError("total_prompt_length must be a nonnegative integer")
        if type(self.chunk_size) is not int or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if type(self.processed_tokens) is not int or not 0 <= self.processed_tokens <= self.total_prompt_length:
            raise ValueError("processed_tokens lies outside the prompt")
        if type(self.chunk_index) is not int or self.chunk_index < 0:
            raise ValueError("chunk_index must be a nonnegative integer")
        self.is_completed = self.processed_tokens == self.total_prompt_length

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.total_prompt_length - self.processed_tokens)

    def next_chunk_slice(self) -> Tuple[int, int]:
        """Returns (start_idx, end_idx) for the next prefill chunk."""
        if self.is_completed:
            raise RuntimeError(f"Prefill for request {self.request_id} is already completed")
        start = self.processed_tokens
        end = min(self.total_prompt_length, start + self.chunk_size)
        return start, end

    def advance(self, chunk_tokens: int):
        """Advances cursor after a chunk is successfully processed."""
        if self.is_completed:
            raise RuntimeError(f"Prefill for request {self.request_id} is already completed")
        if type(chunk_tokens) is not int or not 0 < chunk_tokens <= min(self.chunk_size, self.remaining_tokens):
            raise ValueError("chunk advance must fit the next planned slice")
        self.processed_tokens += chunk_tokens
        self.chunk_index += 1
        if self.processed_tokens >= self.total_prompt_length:
            self.is_completed = True


class ControlledPrefetcher:
    """Manages proactive H2D expert loading during speculative decode rounds.

    Combines:
      - Speculative draft token routing cues (from DSpark MTP);
      - Historical expert access heat (EMA decay);
      - Budget-capped prefetch limits to avoid evicting active leased slots.
    """

    def __init__(
        self,
        max_prefetch_slots: int = 8,
        ema_decay: float = 0.85,
        heat_threshold: float = 0.15,
        *, expert_count: Optional[int] = None, warmup_steps: int = 0,
    ):
        if type(max_prefetch_slots) is not int or max_prefetch_slots < 1:
            raise ValueError("max_prefetch_slots must be a positive integer")
        if isinstance(ema_decay, bool) or not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be finite and in [0,1)")
        if isinstance(heat_threshold, bool) or not math.isfinite(heat_threshold) or not 0 <= heat_threshold <= 1:
            raise ValueError("heat_threshold must be finite and in [0,1]")
        if expert_count is not None and (type(expert_count) is not int or expert_count < 1):
            raise ValueError("expert_count must be a positive integer")
        if type(warmup_steps) is not int or warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        self.max_prefetch_slots = max_prefetch_slots
        self.ema_decay = float(ema_decay)
        self.heat_threshold = float(heat_threshold)
        self.expert_count, self.warmup_steps = expert_count, warmup_steps
        self.observations = 0

        # expert_id -> heat score [0.0, 1.0]
        self.expert_heat: Dict[int, float] = collections.defaultdict(float)

        # Telemetry metrics
        self.prefetch_requested = 0
        self.prefetch_hits = 0
        self.prefetch_wasted = 0
        self.on_demand_fallbacks = 0
        self.last_prefetched_ids: Set[int] = set()

    def update_access_history(self, accessed_expert_ids: List[int]):
        """Decays existing heat and boosts actually accessed experts."""
        for eid in list(self.expert_heat.keys()):
            self.expert_heat[eid] *= self.ema_decay
            if self.expert_heat[eid] < 1e-4:
                del self.expert_heat[eid]

        accessed = set(eid for eid in accessed_expert_ids if self._valid(eid))
        for eid in accessed:
            self.expert_heat[eid] = self.expert_heat.get(eid, 0.0) + (1.0 - self.ema_decay)
        if accessed:
            self.observations += 1

    def _valid(self, eid):
        return type(eid) is int and eid >= 0 and (self.expert_count is None or eid < self.expert_count)

    def reset_job(self):
        """Keep measured affinity, but warm up on actual accesses in each new job."""
        self.observations = 0
        self.last_prefetched_ids.clear()
        self.prefetch_requested = self.prefetch_hits = self.prefetch_wasted = self.on_demand_fallbacks = 0

    def predict_prefetch_set(
        self,
        draft_hint_experts: Optional[List[int]] = None,
        exclude_currently_cached: Optional[Set[int]] = None,
    ) -> List[int]:
        """Calculates the optimal candidate expert IDs to prefetch into GPU slots.

        Prioritizes:
          1. Explicit draft tokens proposed by speculator (highest confidence);
          2. Top heat experts above the heat_threshold.
        """
        exclude = exclude_currently_cached or set()
        candidates: List[int] = []
        if self.observations < self.warmup_steps:
            self.last_prefetched_ids.clear()
            return candidates

        # 1. Draft hints
        if draft_hint_experts:
            for eid in draft_hint_experts:
                if not self._valid(eid) or (self.expert_count is not None and eid not in self.expert_heat):
                    continue  # bounded production policy never seeds unobserved/unknown experts
                if eid not in exclude and eid not in candidates:
                    candidates.append(eid)
                    if len(candidates) >= self.max_prefetch_slots:
                        break

        # 2. Historical heat fill
        if len(candidates) < self.max_prefetch_slots:
            sorted_by_heat = sorted(
                self.expert_heat.items(),
                key=lambda kv: (-kv[1], kv[0]),
            )
            for eid, score in sorted_by_heat:
                if score >= self.heat_threshold and eid not in exclude and eid not in candidates:
                    candidates.append(eid)
                    if len(candidates) >= self.max_prefetch_slots:
                        break

        self.last_prefetched_ids = set(candidates)
        self.prefetch_requested += len(candidates)
        return candidates

    def record_actual_execution(self, actually_needed_experts: List[int]):
        """Evaluates prefetch accuracy against the actual router decision.

        If a needed expert was not prefetched or cached, it counts as on_demand_fallback.
        """
        needed_set = set(actually_needed_experts)
        hits = len(self.last_prefetched_ids.intersection(needed_set))
        wasted = len(self.last_prefetched_ids - needed_set)

        self.prefetch_hits += hits
        self.prefetch_wasted += wasted

        fallbacks = len(needed_set - self.last_prefetched_ids)
        self.on_demand_fallbacks += fallbacks

        self.update_access_history(actually_needed_experts)

    @property
    def precision(self) -> float:
        total = self.prefetch_hits + self.prefetch_wasted
        return (self.prefetch_hits / total) if total > 0 else 1.0


class ChunkedPrefillExecutor:
    """Executes long-prompt prefills in partitioned chunks with managed expert access."""

    def __init__(self, default_chunk_size: int = 512):
        if type(default_chunk_size) is not int or default_chunk_size < 1:
            raise ValueError("default_chunk_size must be a positive integer")
        self.default_chunk_size = default_chunk_size

    def create_state(
        self,
        request_id: str,
        prompt_length: int,
        chunk_size: Optional[int] = None,
    ) -> PrefillChunkState:
        cs = chunk_size if chunk_size is not None else self.default_chunk_size
        return PrefillChunkState(
            request_id=request_id,
            total_prompt_length=prompt_length,
            chunk_size=cs,
        )

    def split_prompt_tensor(
        self,
        input_ids: torch.Tensor,
        state: PrefillChunkState,
    ) -> List[torch.Tensor]:
        """Divides prompt tensor into chunks matching the plan."""
        chunks = []
        seq_len = input_ids.shape[-1]
        if seq_len != state.total_prompt_length:
            raise ValueError("prompt tensor length disagrees with the chunk state")
        for start in range(0, seq_len, state.chunk_size):
            end = min(seq_len, start + state.chunk_size)
            chunks.append(input_ids[..., start:end])
        return chunks

    def execute_chunked(
        self,
        input_ids: torch.Tensor,
        process_chunk_fn: Callable[[torch.Tensor, int, int], Any],
        chunk_size: Optional[int] = None,
    ) -> List[Any]:
        """Iterates through all chunks sequentially, advancing state and aggregating outputs.

        process_chunk_fn signature: (chunk_tensor, start_pos, end_pos) -> chunk_output
        """
        seq_len = input_ids.shape[-1]
        state = self.create_state("req_auto", seq_len, chunk_size)
        outputs = []

        while not state.is_completed:
            start, end = state.next_chunk_slice()
            chunk_tensor = input_ids[..., start:end]
            res = process_chunk_fn(chunk_tensor, start, end)
            outputs.append(res)
            state.advance(end - start)

        return outputs

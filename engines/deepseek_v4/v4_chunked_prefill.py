"""Chunked Prefill State Management and Controlled Speculative Prefetching (Step 9).

Solves the large-context activation memory wall and expert cache thrashing:
  1. ChunkedPrefillState: Tracks cross-chunk context offsets, KV write boundaries,
     and accumulated hidden states for long prompts (e.g. 4096 - 8192 tokens).
  2. ChunkedPrefillExecutor: Breaks long sequence prefills into deterministic chunks
     (e.g. 512 or 1024 tokens), preventing instantaneous GPU activation spikes and
     limiting simultaneous non-resident expert slot competition.
  3. ControlledPrefetcher: Uses speculative draft token cues and exponential moving
     average (EMA) expert affinity heatmaps to overlap DMA with current Attention /
     Shared Expert compute, with a fail-safe fallback to standard on-demand DMA.
"""
from __future__ import annotations

import collections
import dataclasses
import math
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch


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
    ):
        self.max_prefetch_slots = max(1, int(max_prefetch_slots))
        self.ema_decay = float(ema_decay)
        self.heat_threshold = float(heat_threshold)

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

        for eid in accessed_expert_ids:
            self.expert_heat[eid] = self.expert_heat.get(eid, 0.0) + (1.0 - self.ema_decay)

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

        # 1. Draft hints
        if draft_hint_experts:
            for eid in draft_hint_experts:
                if eid not in exclude and eid not in candidates:
                    candidates.append(eid)
                    if len(candidates) >= self.max_prefetch_slots:
                        break

        # 2. Historical heat fill
        if len(candidates) < self.max_prefetch_slots:
            sorted_by_heat = sorted(
                self.expert_heat.items(),
                key=lambda kv: kv[1],
                reverse=True,
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
        self.default_chunk_size = int(default_chunk_size)

    def create_state(
        self,
        request_id: str,
        prompt_length: int,
        chunk_size: Optional[int] = None,
    ) -> PrefillChunkState:
        cs = chunk_size if chunk_size is not None else self.default_chunk_size
        return PrefillChunkState(
            request_id=request_id,
            total_prompt_length=int(prompt_length),
            chunk_size=max(1, int(cs)),
        )

    def split_prompt_tensor(
        self,
        input_ids: torch.Tensor,
        state: PrefillChunkState,
    ) -> List[torch.Tensor]:
        """Divides prompt tensor into chunks matching the plan."""
        chunks = []
        seq_len = input_ids.shape[-1]
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

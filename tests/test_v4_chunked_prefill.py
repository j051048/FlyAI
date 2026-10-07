"""Unit tests for Chunked Prefill State Management & Controlled Prefetching (Step 9)."""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.deepseek_v4.v4_chunked_prefill import (
    ChunkedPrefillExecutor,
    ControlledPrefetcher,
    PrefillChunkState,
)


def test_prefill_chunk_state_cursor_and_completion():
    state = PrefillChunkState(
        request_id="req_001",
        total_prompt_length=1250,
        chunk_size=512,
    )
    assert state.remaining_tokens == 1250
    assert not state.is_completed

    # Chunk 0: [0, 512)
    s0, e0 = state.next_chunk_slice()
    assert (s0, e0) == (0, 512)
    state.advance(e0 - s0)
    assert state.processed_tokens == 512
    assert state.chunk_index == 1
    assert state.remaining_tokens == 738

    # Chunk 1: [512, 1024)
    s1, e1 = state.next_chunk_slice()
    assert (s1, e1) == (512, 1024)
    state.advance(e1 - s1)
    assert state.processed_tokens == 1024
    assert state.chunk_index == 2
    assert state.remaining_tokens == 226

    # Chunk 2: [1024, 1250)
    s2, e2 = state.next_chunk_slice()
    assert (s2, e2) == (1024, 1250)
    state.advance(e2 - s2)
    assert state.processed_tokens == 1250
    assert state.remaining_tokens == 0
    assert state.is_completed

    with pytest.raises(RuntimeError, match="already completed"):
        state.next_chunk_slice()


@pytest.mark.parametrize("prompt_len, chunk_size", [
    (100, 32),
    (512, 128),
    (1024, 512),
    (2048, 1024),
    (4096, 512),
])
def test_chunked_prefill_lossless_tensor_reconstruction(prompt_len, chunk_size):
    executor = ChunkedPrefillExecutor(default_chunk_size=chunk_size)
    raw_tokens = torch.randint(0, 10000, (1, prompt_len), dtype=torch.int64)

    collected_chunks = []

    def mock_process(chunk: torch.Tensor, start: int, end: int):
        collected_chunks.append(chunk)
        return chunk.sum().item()

    results = executor.execute_chunked(raw_tokens, mock_process, chunk_size=chunk_size)

    # Reconstructed tokens must be bit-exact
    reconstructed = torch.cat(collected_chunks, dim=-1)
    assert torch.equal(raw_tokens, reconstructed)
    assert len(results) == math.ceil(prompt_len / chunk_size)


def test_controlled_prefetcher_prioritizes_draft_hints():
    prefetcher = ControlledPrefetcher(max_prefetch_slots=4)

    # Speculator suggests experts [12, 45, 99]
    candidates = prefetcher.predict_prefetch_set(
        draft_hint_experts=[12, 45, 99],
        exclude_currently_cached={12},  # 12 is already in cache
    )
    assert candidates == [45, 99]
    assert prefetcher.prefetch_requested == 2


def test_controlled_prefetcher_uses_historical_heat_and_decay():
    prefetcher = ControlledPrefetcher(max_prefetch_slots=3, ema_decay=0.8, heat_threshold=0.1)

    # Step 1: Expert 7 and 9 accessed heavily
    for _ in range(3):
        prefetcher.update_access_history([7, 9])

    # Predict without draft hints
    candidates = prefetcher.predict_prefetch_set()
    assert 7 in candidates and 9 in candidates

    # Step 2: New experts accessed; heat decays
    for _ in range(5):
        prefetcher.update_access_history([101, 102])

    candidates_new = prefetcher.predict_prefetch_set()
    assert 101 in candidates_new and 102 in candidates_new


def test_prefetch_evaluation_and_on_demand_fallback():
    prefetcher = ControlledPrefetcher(max_prefetch_slots=4)

    # Predict [10, 20, 30]
    prefetcher.predict_prefetch_set(draft_hint_experts=[10, 20, 30])

    # Actually router chose [20, 30, 40] (10 was false positive; 40 was misprediction)
    prefetcher.record_actual_execution([20, 30, 40])

    assert prefetcher.prefetch_hits == 2        # 20, 30 hit
    assert prefetcher.prefetch_wasted == 1      # 10 wasted
    assert prefetcher.on_demand_fallbacks == 1  # 40 on-demand fallback
    assert prefetcher.precision == 2.0 / 3.0

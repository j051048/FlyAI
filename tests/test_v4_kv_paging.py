"""Unit tests for V4-Dedicated Paged KV Cache & Dual-Tier Management (Step 10)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.deepseek_v4.v4_kv_paging import (
    KVPage,
    SlidingWindowKVCache,
    V4DualTierKVPool,
)


def test_kv_page_allocation_and_write():
    page = KVPage(page_id=1, page_size=16, n_heads=4, head_dim=32, device="cpu")
    page.allocate()
    assert page.k.shape == (16, 4, 32)
    assert page.v.shape == (16, 4, 32)

    k_sample = torch.ones((5, 4, 32), dtype=torch.bfloat16)
    v_sample = torch.full((5, 4, 32), 2.0, dtype=torch.bfloat16)
    page.write_slice(k_sample, v_sample, offset=0)

    assert page.tokens_stored == 5
    assert torch.equal(page.k[:5], k_sample)
    assert torch.equal(page.v[:5], v_sample)


def test_dual_tier_kv_pool_bounds_gpu_pages_and_spills_to_host():
    # Allow at most 2 GPU pages (each page 16 tokens -> max 32 tokens in GPU VRAM)
    pool = V4DualTierKVPool(
        n_heads=2, head_dim=16, page_size=16, max_gpu_pages=2, gpu_device="cpu"
    )

    # Allocate page 1 & page 2
    p1 = pool.new_page(on_gpu=True)
    p2 = pool.new_page(on_gpu=True)
    assert len(pool.gpu_pages) == 2
    assert set(pool.gpu_pages.keys()) == {p1.page_id, p2.page_id}

    # Allocate page 3: triggers spill of oldest page (p1) to host RAM
    p3 = pool.new_page(on_gpu=True)
    assert len(pool.gpu_pages) == 2
    assert p1.page_id not in pool.gpu_pages
    assert set(pool.gpu_pages.keys()) == {p2.page_id, p3.page_id}

    # Full history archive still preserves all 3 pages
    assert len(pool.host_archive) == 3
    assert p1.page_id in pool.host_archive

    # Requesting p1 back swaps out p2
    retrieved_p1 = pool.ensure_page_on_gpu(p1.page_id)
    assert retrieved_p1.page_id == p1.page_id
    assert len(pool.gpu_pages) == 2
    assert set(pool.gpu_pages.keys()) == {p3.page_id, p1.page_id}


def test_sliding_window_cache_append_and_rollback():
    pool = V4DualTierKVPool(
        n_heads=4, head_dim=32, page_size=16, max_gpu_pages=4, gpu_device="cpu"
    )
    cache = SlidingWindowKVCache(pool, window_size=64)

    # 1. Append 20 prompt tokens and commit
    k_prompt = torch.randn((20, 4, 32), dtype=torch.bfloat16)
    v_prompt = torch.randn((20, 4, 32), dtype=torch.bfloat16)
    cache.append(k_prompt, v_prompt)
    cache.commit()

    assert cache.committed_length == 20
    assert cache.current_length == 20
    assert len(cache.page_table) == 2  # 20 tokens takes 2 pages (page size 16)

    # 2. Drafter proposes 10 speculative tokens
    k_draft = torch.randn((10, 4, 32), dtype=torch.bfloat16)
    v_draft = torch.randn((10, 4, 32), dtype=torch.bfloat16)
    cache.append(k_draft, v_draft)

    assert cache.current_length == 30
    assert cache.committed_length == 20

    # 3. Verifier rejects draft tokens: rollback to committed
    cache.rollback_to_committed()
    assert cache.current_length == 20
    assert cache.committed_length == 20

    # Subsequent append continues cleanly from position 20
    k_next = torch.randn((5, 4, 32), dtype=torch.bfloat16)
    v_next = torch.randn((5, 4, 32), dtype=torch.bfloat16)
    cache.append(k_next, v_next)
    cache.commit()

    assert cache.current_length == 25
    assert cache.committed_length == 25

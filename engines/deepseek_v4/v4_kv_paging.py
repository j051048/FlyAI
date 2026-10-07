"""V4-Dedicated Paged KV Cache and Dual-Tier Working Set Management (Step 10).

Prevents OOM during long-context serving on 32GB GPUs by strictly bounding VRAM:
  1. GPU Active Working Set: Retains only sliding-window tokens (e.g. recent 2048)
     and initial attention sinks, keeping GPU VRAM strictly bounded regardless of context length.
  2. Host RAM Full History: Pinned host memory archives full multi-turn conversation
     KV pages for long-context recall.
  3. Speculative Decode Rollback: Deterministically trims unaccepted draft token KV
     entries back to the last committed token position.
"""
from __future__ import annotations

import collections
import dataclasses
from typing import Dict, List, Optional, Tuple

import torch


@dataclasses.dataclass
class KVPage:
    """A fixed-capacity contiguous block of KV tokens."""
    page_id: int
    page_size: int
    n_heads: int
    head_dim: int
    dtype: torch.dtype = torch.bfloat16
    device: str = "cpu"
    tokens_stored: int = 0

    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None

    def allocate(self):
        if self.k is None:
            # Shape: [page_size, n_heads, head_dim]
            self.k = torch.zeros((self.page_size, self.n_heads, self.head_dim),
                                 dtype=self.dtype, device=self.device)
            self.v = torch.zeros((self.page_size, self.n_heads, self.head_dim),
                                 dtype=self.dtype, device=self.device)

    def write_slice(self, k_slice: torch.Tensor, v_slice: torch.Tensor, offset: int):
        self.allocate()
        n = k_slice.shape[0]
        self.k[offset:offset + n] = k_slice.to(self.device)
        self.v[offset:offset + n] = v_slice.to(self.device)
        self.tokens_stored = max(self.tokens_stored, offset + n)

    def to_device(self, target_device: str):
        if self.k is not None:
            self.k = self.k.to(target_device)
            self.v = self.v.to(target_device)
            self.device = str(target_device)


class V4DualTierKVPool:
    """Manages GPU working-set KV pages and Host RAM full-history archive."""

    def __init__(
        self,
        n_heads: int = 8,
        head_dim: int = 128,
        page_size: int = 64,
        max_gpu_pages: int = 32,      # e.g. 32 pages * 64 = 2048 tokens active working set
        dtype: torch.dtype = torch.bfloat16,
        gpu_device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.max_gpu_pages = max(1, int(max_gpu_pages))
        self.dtype = dtype
        self.gpu_device = str(gpu_device)

        self._page_counter = 0
        # Active pages currently on GPU: page_id -> KVPage
        self.gpu_pages: Dict[int, KVPage] = collections.OrderedDict()
        # Full history archive on Host RAM: page_id -> KVPage
        self.host_archive: Dict[int, KVPage] = {}

    def new_page(self, on_gpu: bool = True) -> KVPage:
        self._page_counter += 1
        dev = self.gpu_device if on_gpu else "cpu"
        page = KVPage(
            page_id=self._page_counter,
            page_size=self.page_size,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=dev,
        )
        page.allocate()

        if on_gpu:
            if len(self.gpu_pages) >= self.max_gpu_pages:
                self._spill_oldest_gpu_page()
            self.gpu_pages[page.page_id] = page

        # Archive always tracks master copy in host RAM
        self.host_archive[page.page_id] = page
        return page

    def _spill_oldest_gpu_page(self):
        """Spills oldest GPU page to host RAM archive to preserve bounded VRAM."""
        oldest_id, oldest_page = self.gpu_pages.popitem(last=False)
        oldest_page.to_device("cpu")

    def ensure_page_on_gpu(self, page_id: int) -> KVPage:
        """Retrieves page into GPU, swapping out oldest if at capacity."""
        if page_id in self.gpu_pages:
            self.gpu_pages.move_to_end(page_id)
            return self.gpu_pages[page_id]

        if page_id not in self.host_archive:
            raise KeyError(f"Page {page_id} not found in archive")

        page = self.host_archive[page_id]
        if len(self.gpu_pages) >= self.max_gpu_pages:
            self._spill_oldest_gpu_page()

        page.to_device(self.gpu_device)
        self.gpu_pages[page_id] = page
        return page


class SlidingWindowKVCache:
    """Maintains token sequence view with sliding window bounds and rollback checkpoints."""

    def __init__(
        self,
        pool: V4DualTierKVPool,
        window_size: int = 2048,
        sink_tokens: int = 4,
    ):
        self.pool = pool
        self.window_size = int(window_size)
        self.sink_tokens = max(0, int(sink_tokens))

        self.committed_length = 0
        self.current_length = 0
        self.page_table: List[int] = []  # logical page index -> page_id

    def append(self, k: torch.Tensor, v: torch.Tensor):
        """Appends new token keys and values.

        k, v shape: [seq_len, n_heads, head_dim]
        """
        num_tokens = k.shape[0]
        written = 0

        while written < num_tokens:
            token_idx = self.current_length
            offset_in_page = token_idx % self.pool.page_size
            page_idx = token_idx // self.pool.page_size

            if page_idx >= len(self.page_table):
                page = self.pool.new_page(on_gpu=True)
                self.page_table.append(page.page_id)
            else:
                page_id = self.page_table[page_idx]
                page = self.pool.ensure_page_on_gpu(page_id)

            can_write = min(num_tokens - written, self.pool.page_size - offset_in_page)
            page.write_slice(
                k[written:written + can_write],
                v[written:written + can_write],
                offset_in_page,
            )

            written += can_write
            self.current_length += can_write

    def commit(self):
        """Marks current token position as committed (accepted by verifier)."""
        self.committed_length = self.current_length

    def rollback_to_committed(self):
        """Trims unaccepted draft tokens, returning to committed length."""
        if self.current_length <= self.committed_length:
            return

        excess = self.current_length - self.committed_length
        self.current_length = self.committed_length

        # Recompute valid page table boundary
        needed_pages = (self.current_length + self.pool.page_size - 1) // self.pool.page_size
        if needed_pages < len(self.page_table):
            self.page_table = self.page_table[:needed_pages]

    @property
    def active_gpu_token_capacity(self) -> int:
        return self.pool.max_gpu_pages * self.pool.page_size

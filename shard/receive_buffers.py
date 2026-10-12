"""Bounded receive storage; a borrowed tensor keeps its slot until its last view dies.

The wire and authentication are unchanged. This removes the additional CPU blob
copy, not D2H/H2D or socket traffic. Pinning is explicit and never inferred from
available RAM. Exhaustion falls back to the existing owned-frame decoder.
"""
from __future__ import annotations

import threading

import numpy as np
import torch


class _Lease:
    def __init__(self, pool, index):
        self.pool, self.index = pool, index

    def __del__(self):
        self.pool.release(self.index)


class _BorrowedArray(np.ndarray):
    def __array_finalize__(self, source):
        self.lease = getattr(source, "lease", None)


class ReceiveBufferPool:
    def __init__(self, budget_bytes, *, slots=2, pinned=False):
        if type(budget_bytes) is not int or not 1024 <= budget_bytes <= 512 * 1024**2:
            raise ValueError("receive pool budget must be 1KiB..512MiB")
        if type(slots) is not int or not 1 <= slots <= 64 or type(pinned) is not bool:
            raise ValueError("bounded slot count and explicit pinning required")
        if pinned and not torch.cuda.is_available():
            raise RuntimeError("pinned receive pool requires an available CUDA allocator")
        self.budget_bytes, self.slots, self.pinned = budget_bytes, slots, pinned
        self.lock = threading.Lock()
        self.buffers, self.busy = [], set()
        self.hits = self.fallbacks = self.high_water = self.allocation_failures = 0

    def borrow(self, size):
        if type(size) is not int or size < 0:
            raise ValueError("nonnegative receive frame size required")
        with self.lock:
            free = [i for i, value in enumerate(self.buffers) if i not in self.busy and value.numel() >= size]
            index = min(free, key=lambda i: self.buffers[i].numel()) if free else None
            allocated = sum(value.numel() for value in self.buffers)
            if index is None:
                reusable = [i for i in range(len(self.buffers)) if i not in self.busy]
                index = min(reusable, key=lambda i: self.buffers[i].numel()) if reusable else len(self.buffers)
                old = self.buffers[index].numel() if index < len(self.buffers) else 0
                capacity = max(1024, 1 << (max(1, size) - 1).bit_length())
                # A resize temporarily owns the old and new allocations. Count both.
                if index >= self.slots or allocated + capacity > self.budget_bytes:
                    self.fallbacks += 1
                    return None
                try:
                    value = torch.empty(capacity, dtype=torch.uint8, device="cpu", pin_memory=self.pinned)
                except (RuntimeError, OSError, MemoryError):
                    self.fallbacks += 1
                    self.allocation_failures += 1
                    return None  # optional optimization cannot kill a warm stage
                if old:
                    self.buffers[index] = value
                else:
                    self.buffers.append(value)
            self.busy.add(index)
            self.hits += 1
            self.high_water = max(self.high_water, len(self.busy))
            array = self.buffers[index].numpy().view(_BorrowedArray)[:size]
            array.lease = _Lease(self, index)
            return array

    def release(self, index):
        with self.lock:
            self.busy.discard(index)

    def snapshot(self):
        with self.lock:
            return {"schema": "shard-receive-buffer-pool/1", "budget_bytes": self.budget_bytes,
                    "allocated_bytes": sum(value.numel() for value in self.buffers),
                    "slots": self.slots, "live_slots": len(self.busy), "high_water_slots": self.high_water,
                    "borrowed_frames": self.hits, "fallback_frames": self.fallbacks,
                    "allocation_failures": self.allocation_failures,
                    "pinned_allocated_bytes": sum(value.numel() for value in self.buffers) if self.pinned else 0,
                    "pinned": self.pinned, "scope": "process lifetime; CPU receive copies only"}

"""Opt-in reusable, lossless host transfer for a small DSpark result tuple.

IDs retain int64 storage (never float packing); confidence retains the existing
float32 conversion. A single CUDA completion event fences reuse of both buffers.
The byte arena is not a wire format: only independently owned Python lists leave
this object. CPU operation is an explicit reference, not measured CUDA DMA.
"""
from __future__ import annotations

import time
import torch


class DraftResultBuffer:
    def __init__(self):
        self._device = self._host = self._event = None
        self._capacity = 0
        self._stats = {"calls": 0, "allocations": 0, "gpu_d2h_copies": 0,
                       "allocated_host_bytes": 0, "pack_and_submit_s": 0., "cuda_wait_s": 0.}

    def _allocate(self, device, capacity):
        # Reallocation happens only after the preceding pack's event completed.
        self._capacity = capacity
        self._confidence_offset = capacity * 8
        self._runner_offset = ((capacity * 12 + 7) // 8) * 8
        size = self._runner_offset + capacity * 8
        self._device = torch.empty(size, dtype=torch.uint8, device=device)
        self._host = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
        self._event = torch.cuda.Event()
        self._stats["allocations"] += 1
        self._stats["allocated_host_bytes"] = size

    def pack(self, draft, confidence, runner_up=None):
        if draft.ndim != 1 or confidence.ndim != 1 or draft.numel() != confidence.numel():
            raise ValueError("draft and confidence must be equal one-dimensional blocks")
        n = draft.numel()
        if not 0 <= n <= 256 or draft.dtype not in (torch.int32, torch.int64):
            raise ValueError("bounded integer draft IDs required")
        if not confidence.is_floating_point() or confidence.device != draft.device:
            raise ValueError("confidence must share the draft device")
        if runner_up is not None and (runner_up.ndim != 1 or runner_up.numel() != n
                or runner_up.device != draft.device or runner_up.dtype not in (torch.int32, torch.int64)):
            raise ValueError("runner-up IDs must align with the draft block")
        self._stats["calls"] += 1
        if draft.device.type != "cuda" or n == 0:
            result = {"draft": draft.tolist(), "conf": [round(c, 4) for c in confidence.float().tolist()]}
            if runner_up is not None:
                result["d2"] = runner_up.tolist()
            return result
        started = time.perf_counter()
        if self._device is None or self._device.device != draft.device or self._capacity < n:
            self._allocate(draft.device, n)
        self._device[:8*n].view(torch.int64).copy_(draft)
        co, ro = self._confidence_offset, self._runner_offset
        self._device[co:co+4*n].view(torch.float32).copy_(confidence)
        if runner_up is not None:
            self._device[ro:ro+8*n].view(torch.int64).copy_(runner_up)
        self._host.copy_(self._device, non_blocking=True)
        self._event.record(torch.cuda.current_stream(draft.device))
        submitted = time.perf_counter()
        self._event.synchronize()
        finished = time.perf_counter()
        self._stats["gpu_d2h_copies"] += 1
        self._stats["pack_and_submit_s"] += submitted-started
        self._stats["cuda_wait_s"] += finished-submitted
        result = {"draft": self._host[:8*n].view(torch.int64).tolist(),
                  "conf": [round(c, 4) for c in self._host[co:co+4*n].view(torch.float32).tolist()]}
        if runner_up is not None:
            result["d2"] = self._host[ro:ro+8*n].view(torch.int64).tolist()
        return result

    def stats(self):
        return dict(self._stats, mode="cuda_pinned" if self._device is not None else "cpu_reference")

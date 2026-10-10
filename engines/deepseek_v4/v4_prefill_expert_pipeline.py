"""Bounded local expert-copy lookahead, after the authoritative router has run.

The producer is a finite FIFO of cache leases, not a second router or a token
chunker. Copies use the cache's existing CUDA stream and fixed slots; the caller
keeps every original per-expert row shape and logical accumulation order. There
is no additional weight bank, pinned staging arena, CPU compute or network I/O.
"""
from collections import deque

from v4_expert_cache import ExpertCacheError


SCHEMA = "v4-prefill-expert-pipeline/1"


def validate_options(*, enabled=False, depth=2, batch_size=0):
    if type(enabled) is not bool:
        raise ValueError("prefill expert pipeline enabled must be a boolean")
    if type(depth) is not int or not 1 <= depth <= 4:
        raise ValueError("prefill expert pipeline depth must be an integer between 1 and 4")
    if type(batch_size) is not int or batch_size < 0:
        raise ValueError("prefill expert pipeline batch size must be a nonnegative integer")
    return {"enabled": enabled, "depth": depth, "batch_size": batch_size}


def plan(cache, *, enabled=False, depth=2, batch_size=0):
    requested = validate_options(enabled=enabled, depth=depth, batch_size=batch_size)
    size = batch_size or max(1, cache.capacity // depth)
    effective_depth = min(depth, cache.capacity // size)
    reason = ("disabled" if not enabled else "depth_has_no_lookahead" if depth < 2 else
              "insufficient_cache_slots" if effective_depth < 2 else None)
    active = reason is None
    return {"schema": SCHEMA, "requested": requested, "active": active, "reason": reason,
            "mode": "cpu_reference" if cache.emulation else "cuda_dma",
            "effective_batch_size": size if active else 0,
            "effective_depth": effective_depth if active else 0,
            "slot_budget": size * effective_depth if active else 0,
            "byte_budget": size * effective_depth * cache.pool.bytes_per_expert if active else 0,
            "additional_gpu_bytes": 0, "additional_pinned_bytes": 0,
            "source": "canonical_host_expert_banks", "consumer_order": "logical_expert_ascending"}


class ExpertBatchPipeline:
    """At most `depth` owned batches, including the current consumer's batch.

    Enqueueing the next copy before the current expert kernels lets the copy
    stream advance while the consumer computes. A released slot still carries
    its consumer event; cache eviction waits for that event before overwriting
    it. Canonical pinned source banks are retained by every lease/cache and may
    be reloaded only through HostExpertPool's checked reload barrier.
    """
    def __init__(self, cache, ordered_ids, *, depth=2, batch_size=0):
        self.cache = cache
        self.config = plan(cache, enabled=True, depth=depth, batch_size=batch_size)
        if not self.config["active"]:
            raise ExpertCacheError("prefill expert pipeline needs cache space for current and lookahead batches")
        self._ids = tuple(ordered_ids)
        if (any(type(eid) is not int or not 0 <= eid < cache.pool.expert_count for eid in self._ids)
                or self._ids != tuple(sorted(set(self._ids)))):
            raise ExpertCacheError("prefill expert producer needs distinct, ascending, valid logical IDs")
        self._pending = deque()
        self._next = 0
        self._yielded = None
        self.closed = False
        self._stats = {"queued_batches": 0, "consumed_batches": 0, "cancelled_batches": 0,
                       "new_copy_experts": 0, "copied_bytes": 0, "dma_bytes": 0,
                       "peak_batches": 0, "peak_slots": 0}

    def _fill(self):
        size, depth = self.config["effective_batch_size"], self.config["effective_depth"]
        while len(self._pending) < depth and self._next < len(self._ids):
            batch = self._ids[self._next:self._next + size]
            lease = self.cache.acquire(batch)
            self._pending.append((batch, lease))
            self._next += len(batch)
            self._stats["queued_batches"] += 1
            self._stats["new_copy_experts"] += len(lease.new_copy_ids)
            self._stats["copied_bytes"] += lease.copied_bytes
            self._stats["dma_bytes"] += lease.dma_bytes
            self._stats["peak_batches"] = max(self._stats["peak_batches"], len(self._pending))
            self._stats["peak_slots"] = max(self._stats["peak_slots"],
                                             sum(len(ticket.mapping) for _, ticket in self._pending))

    def __enter__(self):
        if self.closed:
            raise ExpertCacheError("prefill expert producer is closed")
        try:
            self._fill()
        except BaseException:
            self.close()
            raise
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        if self._yielded is not None:
            if not self._yielded.released:
                raise ExpertCacheError("release the current consumer lease before advancing the producer")
            self._pending.popleft()
            self._stats["consumed_batches"] += 1
            self._yielded = None
        try:
            self._fill()
        except BaseException:
            self.close()
            raise
        if not self._pending:
            raise StopIteration
        batch, lease = self._pending[0]
        self._yielded = lease
        return batch, lease

    def stats(self):
        return {**self._stats, "closed": self.closed,
                "pending_batches": len(self._pending),
                "pending_slots": sum(len(ticket.mapping) for _, ticket in self._pending)}

    def close(self):
        if self.closed:
            return
        if self._yielded is not None and self._yielded.released:
            self._pending.popleft()
            self._stats["consumed_batches"] += 1
            self._yielded = None
        for _, lease in self._pending:
            if not lease.released:
                # An unconsumed ticket has no compute reader: release on the
                # copy stream. A consumer already waited records its own readers.
                lease.release(None if lease.waited else self.cache.copy_stream)
                self._stats["cancelled_batches"] += 1
        self._pending.clear()
        self.closed = True

    def __exit__(self, *_):
        self.close()

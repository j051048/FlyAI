"""Exclusive local expert tiers: fixed GPU hot slots and a bounded RAM complement.

The default cache keeps immutable canonical RAM weights. This opt-in backend
instead owns each expert in one canonical tier. A single exchange bank preserves
an evicted expert before its unique GPU copy can be overwritten. These are local
storage transactions, not a distributed expert service or a speed qualification.
"""
from collections import OrderedDict
import threading
import uuid
import weakref

import torch

from v4_expert_cache import (HostExpertPool, FixedSlotCache, CacheLease,
    ExpertCacheError, _Slot, _CompletedEvent, _bank_specs, _allocate_banks,
    _bind_expert, _clone_module_structure, _integer)


SCHEMA = "v4-exclusive-expert-tiers/1"


class ExclusiveExpertPool(HostExpertPool):
    """Construct the actual cold complement BEFORE loading checkpoint weights.

    Capacity is explicit. Hot expert Parameters bind directly to the eventual
    cache slots, so loading never constructs an entire CPU canonical bank first.
    CPU emulation has separate real hot/cold storages, but makes no physical GPU
    or host-RAM saving claim.
    """
    storage_mode = "exclusive"

    def __init__(self, moe, *, hot_slots, device, identity, pin=True,
                 emulation=False, preserve=False):
        if getattr(moe, "_host_expert_pool", None) is not None:
            raise ExpertCacheError("MoE already belongs to a canonical expert tier owner")
        _integer(hot_slots, "hot_slots", positive=True)
        if type(pin) is not bool or type(emulation) is not bool or type(preserve) is not bool:
            raise ExpertCacheError("exclusive tier flags must be booleans")
        if emulation != (torch.device(device).type == "cpu") or pin == emulation:
            raise ExpertCacheError("exclusive tiers require CUDA/pinned or explicit CPU/unpinned emulation")
        if not emulation and not torch.cuda.is_available():
            raise ExpertCacheError("exclusive CUDA tiers require an available accelerator")
        if (not isinstance(identity, tuple) or len(identity) != 2 or identity[0] not in ("main", "draft")
                or type(identity[1]) is not int or identity[1] < 0):
            raise ExpertCacheError("exclusive pool identity requires role and global layer")
        self.moe, self.experts = moe, list(moe.experts)
        self.specs = _bank_specs(self.experts)
        self.expert_count = len(self.experts)
        if hot_slots > self.expert_count:
            raise ExpertCacheError("exclusive hot capacity exceeds expert count")
        if preserve and any(getattr(getattr(expert, projection), spec.attribute).is_meta
                            for expert in self.experts for spec in self.specs for projection in spec.projections):
            raise ExpertCacheError("meta experts have no readable weights to preserve")
        self.bytes_per_expert = sum(spec.bytes_per_expert for spec in self.specs)
        self.hot_slots, self.device = hot_slots, torch.device(device)
        self.emulation, self.pinned = emulation, pin
        self.identity = (uuid.uuid4().hex, identity[0], identity[1], str(self.device))
        self.model_identity = None
        self.epoch, self.loaded = 0, False
        self._lock = threading.RLock()
        self._caches = weakref.WeakSet()
        self._reload_depth = 0
        self._reload_failed = False
        self._reload_initial_loaded = False
        self._reload_complete_pool = False
        self._hook_contexts = {}
        self._pending = None
        self._poisoned = False
        self._generation = 0
        self._hot_owners = {index: index for index in range(hot_slots)}
        self._cold = {}
        self._exchange = None
        # No full-sized CPU bank is hidden in this compatibility field.
        self.banks = {"w13": None, "w13_s": None, "w2": None, "w2_s": None}
        self.hot_banks = _allocate_banks(self.specs, hot_slots, self.device)
        self.device = next(value.device for value in self.hot_banks.values() if value is not None)
        self.gpu_uuid = (str(getattr(torch.cuda.get_device_properties(self.device), "uuid", "")) or None
                         if not emulation else None)
        self.identity = (self.identity[0], identity[0], identity[1], str(self.device), self.gpu_uuid)
        try:
            for eid in range(hot_slots, self.expert_count):
                self._cold[eid] = _allocate_banks(self.specs, 1, "cpu", pin=pin)
            if self._cold:
                self._exchange = _allocate_banks(self.specs, 1, "cpu", pin=pin)
            if preserve:
                for eid, expert in enumerate(self.experts):
                    banks = self.hot_banks if eid < hot_slots else self._cold[eid]
                    index = eid if eid < hot_slots else 0
                    for spec in self.specs:
                        offset = 0
                        for projection, rows in zip(spec.projections, spec.rows):
                            source = getattr(getattr(expert, projection), spec.attribute)
                            banks[spec.key][index, offset:offset + rows].view(torch.uint8).copy_(
                                source.detach().view(torch.uint8))
                            offset += rows
            for eid, expert in enumerate(self.experts):
                _bind_expert(expert, self.hot_banks if eid < hot_slots else self._cold[eid],
                             self.specs, eid if eid < hot_slots else 0)
        except BaseException:
            self.hot_banks.clear(); self._cold.clear(); self._exchange = None
            raise
        self._hooks = []
        self._install_load_hooks(moe, complete_pool=True)
        for expert in self.experts:
            self._install_load_hooks(expert, complete_pool=False)
            for projection in ("w1", "w2", "w3"):
                self._install_load_hooks(getattr(expert, projection), complete_pool=False)
        # state_dict can be called through a parent, an Expert, or a Linear.
        # Every entry fences a pending H2D before exposing canonical Parameters.
        for module in (moe, *self.experts, *[getattr(e, p) for e in self.experts for p in ("w1", "w2", "w3")]):
            self._hooks.append(module.register_state_dict_pre_hook(self._save_barrier))
        moe._host_expert_pool = self
        if hasattr(moe, "_grouped_bank"):
            moe._grouped_bank = None
        if preserve:
            self.mark_loaded()

    @property
    def host_bytes(self):
        return (self.expert_count - self.hot_slots + int(self.hot_slots < self.expert_count)) * self.bytes_per_expert

    @property
    def hot_bytes(self):
        return self.hot_slots * self.bytes_per_expert

    def bind_model_identity(self, value):
        if not isinstance(value, str) or not 1 <= len(value) <= 256:
            raise ExpertCacheError("expert model identity must be a bounded nonempty string")
        with self._lock:
            if self.model_identity not in (None, value) and self.loaded:
                raise ExpertCacheError("changing a loaded expert model identity requires a new pool")
            self.model_identity = value

    def _save_barrier(self, *args):
        with self._lock:
            self.drain(wait=True)

    def begin_reload(self, *, complete_pool=True):
        if self._poisoned:
            raise ExpertCacheError("exclusive tier copy failed; restart the owned Stage before reload")
        super().begin_reload(complete_pool=complete_pool)

    def finish_reload(self, *, success=True):
        super().finish_reload(success=success)
        if self._reload_depth == 0 and self.loaded:
            for cache in tuple(self._caches):
                cache._refresh_epochs()

    def storage_tensors(self):
        for key, bank in self.hot_banks.items():
            if bank is not None:
                yield "hot." + key, bank
        for eid, banks in self._cold.items():
            for key, bank in banks.items():
                if bank is not None:
                    yield f"cold.{eid}.{key}", bank
        if self._exchange is not None:
            for key, bank in self._exchange.items():
                if bank is not None:
                    yield "exchange." + key, bank

    def drain(self, *, wait=False):
        """Finish publication only after the incoming bytes are actually ready."""
        pending = self._pending
        if pending is None:
            return True
        event = pending["ready"]
        try:
            if not event.query():
                if not wait:
                    return False
                event.synchronize()
        except BaseException:
            self._poisoned = True
            pending["cache"]._faulted = True
            raise
        cache, eid, index = pending["cache"], pending["eid"], pending["index"]
        if (pending["identity"] != self.identity or pending["epoch"] != self.epoch
                or pending["generation"] != cache._generations[index]):
            self._poisoned = True
            cache._faulted = True
            raise ExpertCacheError("exclusive migration identity/generation changed before commit")
        source = self._cold.pop(eid)
        _bind_expert(self.experts[eid], self.hot_banks, self.specs, index)
        self._exchange = source
        self._hot_owners[index] = eid
        self._pending = None
        cache._tier_stats["committed_swaps"] += 1
        return True


class _TierLease(CacheLease):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity = self.cache.pool.identity
        self.generations = {slot: self.cache._generations[slot] for slot in self.mapping.values()}

    def _validate(self):
        super()._validate()
        if self.cache.pool._poisoned:
            raise ExpertCacheError("exclusive tier copy failed; no further compute is licensed")
        if self.identity != self.cache.pool.identity or any(
                self.cache._generations[slot] != generation for slot, generation in self.generations.items()):
            raise ExpertCacheError("exclusive lease belongs to another owner/generation")

    def wait_on(self, stream=None):
        if self.cache._faulted or self.cache.pool._poisoned:
            raise ExpertCacheError("exclusive tier copy failed; no further compute is licensed")
        return super().wait_on(stream)


class ExclusiveTierCache(FixedSlotCache):
    """Cache API backed by the pool's unique GPU canonical hot banks.

    An old unique GPU expert must reach the exchange bank before overwriting
    its slot. This necessary D2H preservation barrier may block; only the new
    H2D is queued asynchronously. There is one pending exchange, never an
    unbounded queue or an unaccounted full-RAM backup.
    """
    def __init__(self, pool, capacity, *, device, emulation=False, decay_interval=256):
        if not isinstance(pool, ExclusiveExpertPool) or capacity != pool.hot_slots:
            raise ExpertCacheError("exclusive cache capacity is fixed before checkpoint load")
        if pool._poisoned:
            raise ExpertCacheError("exclusive tier copy failed; restart the owned Stage before rebuilding")
        requested = torch.device(device)
        if (requested.type != pool.device.type or requested.index is not None and requested.index != pool.device.index
                or emulation != pool.emulation):
            raise ExpertCacheError("exclusive cache owner/device must match its pool")
        if pool._caches:
            raise ExpertCacheError("exclusive canonical hot slots have exactly one cache owner")
        self.pool, self.capacity, self.emulation = pool, capacity, emulation
        self.device = pool.device
        self.mode = "cpu_reference" if emulation else "cuda_exclusive_tiers"
        self.cache_bytes, self.decay_interval = pool.hot_bytes, decay_interval
        self._lock = threading.RLock()
        self._owners = {eid: index for index, eid in pool._hot_owners.items()}
        self._slots = [_Slot(owner=pool._hot_owners.get(index), epoch=pool.epoch,
                             ready=_CompletedEvent()) for index in range(capacity)]
        self._generations = [0] * capacity
        self._frequencies, self._tick = {}, 0
        self.closed = self._faulted = False
        self._stats = dict(acquires=0, hits=0, misses=0, pending_hits=0, new_copies=0,
                           copied_bytes=0, dma_bytes=0, evictions=0, invalidations=0)
        self._tier_stats = dict(committed_swaps=0, preserved_experts=0, preservation_host_waits=0)
        self.copy_stream = None if emulation else torch.cuda.Stream(device=self.device)
        self.banks = pool.hot_banks
        experts = []
        for index in range(capacity):
            expert = _clone_module_structure(pool.experts[0])
            for projection in ("w1", "w2", "w3"):
                linear = getattr(expert, projection)
                for attr in ("weight", "scale"):
                    old = getattr(linear, attr, None)
                    if old is not None:
                        setattr(linear, attr, torch.nn.Parameter(torch.empty(0, device="meta", dtype=old.dtype),
                                                               requires_grad=old.requires_grad))
            _bind_expert(expert, self.banks, pool.specs, index)
            experts.append(expert)
        self.experts = torch.nn.ModuleList(experts)
        pool._caches.add(self)
        if pool.loaded:
            self._refresh_epochs()

    def _refresh_epochs(self):
        with self.pool._lock, self._lock:
            if self.emulation:
                ready = _CompletedEvent()
            else:
                with torch.cuda.device(self.device):
                    ready = torch.cuda.Event()
                    ready.record(torch.cuda.current_stream(self.device))
            for index, eid in self.pool._hot_owners.items():
                slot = self._slots[index]
                slot.owner, slot.epoch, slot.ready = eid, self.pool.epoch, ready
                self._owners[eid] = index

    def acquire(self, ids, *, demand=True, protect_ids=()):
        with self.pool._lock, self._lock:
            self.pool.drain(wait=False)
            lease = super().acquire(ids, demand=demand, protect_ids=protect_ids)
            return _TierLease(self, lease.mapping, lease.hit_ids, lease.miss_ids,
                lease.pending_ids, lease.new_copy_ids, lease._copies)

    def _copy_expert_unbudgeted(self, eid, index):
        pool, slot = self.pool, self._slots[index]
        pool.drain(wait=True)  # Reuse the one exchange buffer only after commit.
        if pool._poisoned or eid not in pool._cold or pool._exchange is None:
            raise ExpertCacheError("exclusive transfer needs a valid cold source and exchange bank")
        old = pool._hot_owners.get(index)
        if old is None or old == eid:
            raise ExpertCacheError("exclusive slot owner disagrees with the canonical map")
        source, exchange = pool._cold[eid], pool._exchange
        try:
            if self.emulation:
                for event in slot.use_events:
                    event.synchronize()
                if slot.ready is not None:
                    slot.ready.synchronize()
                for spec in pool.specs:
                    exchange[spec.key][0].view(torch.uint8).copy_(self.banks[spec.key][index].view(torch.uint8))
            else:
                with torch.cuda.device(self.device), torch.cuda.stream(self.copy_stream):
                    for event in slot.use_events:
                        self.copy_stream.wait_event(event)
                    if slot.ready is not None:
                        self.copy_stream.wait_event(slot.ready)
                    for spec in pool.specs:
                        exchange[spec.key][0].view(torch.uint8).copy_(
                            self.banks[spec.key][index].view(torch.uint8), non_blocking=True)
                    preserved = torch.cuda.Event()
                    preserved.record(self.copy_stream)
                preserved.synchronize()
                self._tier_stats["preservation_host_waits"] += 1
            # The old expert's only bytes are safe before its GPU slot is reused.
            pool._cold[old] = exchange
            pool._exchange = None
            pool._hot_owners.pop(index)
            _bind_expert(pool.experts[old], exchange, pool.specs, 0)
            self._tier_stats["preserved_experts"] += 1
            self._generations[index] += 1
            if self.emulation:
                start, ready = None, _CompletedEvent()
                for spec in pool.specs:
                    self.banks[spec.key][index].view(torch.uint8).copy_(source[spec.key][0].view(torch.uint8))
            else:
                with torch.cuda.device(self.device), torch.cuda.stream(self.copy_stream):
                    start, ready = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(self.copy_stream)
                    for spec in pool.specs:
                        self.banks[spec.key][index].record_stream(self.copy_stream)
                        self.banks[spec.key][index].view(torch.uint8).copy_(source[spec.key][0].view(torch.uint8), non_blocking=True)
                    ready.record(self.copy_stream)
            pool._pending = dict(cache=self, eid=eid, index=index, ready=ready,
                identity=pool.identity, epoch=pool.epoch, generation=self._generations[index])
            if self.emulation:
                pool.drain(wait=True)
            return start, ready
        except BaseException:
            pool._poisoned = self._faulted = True
            raise

    def lookup(self, eid, *, include_pending=False):
        with self.pool._lock, self._lock:
            self.pool.drain(wait=False)
            return super().lookup(eid, include_pending=include_pending)

    def cached_ids(self, *, include_pending=False):
        with self.pool._lock, self._lock:
            self.pool.drain(wait=False)
            return super().cached_ids(include_pending=include_pending)

    def _invalidate_for_reload(self):
        with self.pool._lock, self._lock:
            self._check_reload_safe()
            self.pool.drain(wait=True)
            for slot in self._slots:
                for event in slot.use_events:
                    event.synchronize()
                if slot.ready is not None:
                    slot.ready.synchronize()
                slot.use_events = []
            # Canonical hot owners cannot be discarded as ordinary cache copies.
            self._stats["invalidations"] += 1

    def stats(self):
        with self.pool._lock, self._lock:
            self.pool.drain(wait=False)
            return {**super().stats(), **self._tier_stats, "storage_mode": "exclusive",
                "identity": list(self.pool.identity), "model_identity": self.pool.model_identity,
                "cold_experts": len(self.pool._cold), "hot_experts": len(self.pool._hot_owners),
                "pending_exchange": self.pool._pending is not None,
                "exchange_bytes": self.pool.bytes_per_expert if self.pool.hot_slots < self.pool.expert_count else 0,
                "emulated_hot_cpu_bytes": self.cache_bytes if self.emulation else 0,
                "qualification": "cpu_reference_only" if self.emulation else "gpu_acceptance_pending"}

    def close(self):
        with self.pool._lock, self._lock:
            self._check_reload_safe()
            self.pool.drain(wait=True)
            if self.copy_stream is not None:
                self.copy_stream.synchronize()
            for slot in self._slots:
                for event in slot.use_events:
                    event.synchronize()
                if slot.ready is not None:
                    slot.ready.synchronize()
            # The pool owns the only hot weights, even after a cache wrapper
            # closes. A subsequent wrapper can reuse the same canonical banks.
            self.experts = torch.nn.ModuleList()
            self._owners.clear()
            self.pool._caches.discard(self)
            self.closed = True

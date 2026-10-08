"""Local, fixed-address expert storage. No transport or model math lives here.

Production host banks must be CUDA pinned. Explicit CPU emulation copies the same
bytes synchronously and never reports those copies as hardware DMA. Cache leases
protect weights until consumer events have been recorded; eviction waits on every
prior consumer stream. The adapter must call lease.wait_on() before using slots.
"""
from collections import OrderedDict
from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
import threading
import weakref

import torch


class ExpertCacheError(RuntimeError):
    pass


def _integer(value, name, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ExpertCacheError(f"{name} must be an {'positive' if positive else 'nonnegative'} integer")
    return value


@dataclass(frozen=True)
class BankSpec:
    key: str
    projections: tuple
    attribute: str
    rows: tuple
    shape: tuple  # per expert; leading expert/slot axis is added at allocation
    dtype: torch.dtype
    element_bytes: int

    @property
    def bytes_per_expert(self):
        result = self.element_bytes
        for dimension in self.shape:
            result *= dimension
        return result


def _bank_specs(experts):
    if not experts or any(expert is None for expert in experts):
        raise ExpertCacheError("local expert pools require a nonempty world-size-one expert list")
    if any(getattr(getattr(expert, projection), "bias", None) is not None
           for expert in experts for projection in ("w1", "w2", "w3")):
        raise ExpertCacheError("V4 expert slot matrices must be bias-free")
    if any(type(expert) is not type(experts[0]) or getattr(expert, "swiglu_limit", None) != getattr(experts[0], "swiglu_limit", None)
           for expert in experts):
        raise ExpertCacheError("V4 slot experts require a uniform module class and SwiGLU configuration")
    specs = []
    for key, projections in (("w13", ("w1", "w3")), ("w2", ("w2",))):
        for attribute, suffix in (("weight", ""), ("scale", "_s")):
            first = [getattr(getattr(experts[0], projection), attribute, None) for projection in projections]
            if all(tensor is None for tensor in first):
                if attribute == "weight":
                    raise ExpertCacheError("expert Linear weights are required")
                if any(getattr(getattr(expert, projection), attribute, None) is not None
                       for expert in experts for projection in projections):
                    raise ExpertCacheError("expert scale layout is not uniform")
                continue
            if any(not isinstance(tensor, torch.Tensor) or tensor.ndim < 2 for tensor in first):
                raise ExpertCacheError("expert weights/scales must be uniform matrices")
            template = first[0]
            rows = tuple(tensor.shape[0] for tensor in first)
            if any(tensor.dtype != template.dtype or tuple(tensor.shape[1:]) != tuple(template.shape[1:]) for tensor in first):
                raise ExpertCacheError("fused expert bank dtypes/shapes disagree")
            for expert in experts:
                for projection, source in zip(projections, first):
                    tensor = getattr(getattr(expert, projection), attribute, None)
                    if not isinstance(tensor, torch.Tensor) or tensor.dtype != source.dtype or tensor.shape != source.shape:
                        raise ExpertCacheError("every expert must have the same matrix/scale layout")
            shape = (sum(rows),) + tuple(template.shape[1:])
            specs.append(BankSpec(key + suffix, projections, attribute, rows, shape,
                                  template.dtype, template.element_size()))
    return tuple(specs)


def _allocate_banks(specs, count, device, pin=False):
    """Allocate packed bytes, then reinterpret; no dtype conversion occurs."""
    banks = {"w13": None, "w13_s": None, "w2": None, "w2_s": None}
    raw = None
    try:
        for spec in specs:
            raw = torch.empty(count * spec.bytes_per_expert, dtype=torch.uint8,
                              device=device, pin_memory=pin if str(device) == "cpu" else False)
            banks[spec.key] = raw.view(spec.dtype).reshape((count,) + spec.shape)
            if pin and not banks[spec.key].is_pinned():
                raise ExpertCacheError("host bank allocation did not produce pinned storage")
            raw = None
        return banks
    except BaseException:
        banks.clear()
        raw = None
        raise


def _bind_expert(expert, banks, specs, index):
    # Meta Parameters cannot change to CPU through .data. Replacing the Parameter
    # at construction retains its state_dict key and avoids a GPU weight transient.
    for spec in specs:
        offset = 0
        for projection, rows in zip(spec.projections, spec.rows):
            linear = getattr(expert, projection)
            old = getattr(linear, spec.attribute)
            view = banks[spec.key][index, offset:offset + rows]
            if old.device.type != view.device.type or old.device != view.device or old.is_meta:
                parameter = torch.nn.Parameter(view, requires_grad=old.requires_grad)
                setattr(linear, spec.attribute, parameter)
            else:
                old.data = view
            offset += rows
    for projection in ("w1", "w2", "w3"):
        linear = getattr(expert, projection)
        linear.weight.scale = getattr(linear, "scale", None)


def _clone_module_structure(module):
    """Copy module structure without cloning any backing tensor storage."""
    clone = copy.copy(module)
    clone._parameters = dict(module._parameters)
    clone._buffers = dict(module._buffers)
    clone._modules = {name: _clone_module_structure(child) if child is not None else None
                      for name, child in module._modules.items()}
    for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
                 "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks", "_state_dict_hooks"):
        setattr(clone, name, OrderedDict())
    return clone


class HostExpertPool:
    def __init__(self, moe, *, pin=True, emulation=False, preserve=False):
        if type(pin) is not bool or type(emulation) is not bool or type(preserve) is not bool:
            raise ExpertCacheError("pool flags must be booleans")
        if emulation and pin:
            raise ExpertCacheError("CPU reference emulation must explicitly request pin=False")
        if not emulation and not pin:
            raise ExpertCacheError("production routed expert banks require pinned CPU storage")
        if not emulation and not torch.cuda.is_available():
            raise ExpertCacheError("production expert pinning requires available CUDA; use explicit CPU emulation for tests")
        self.moe = moe
        self.experts = list(moe.experts)
        self.specs = _bank_specs(self.experts)
        self.expert_count = len(self.experts)
        self.bytes_per_expert = sum(spec.bytes_per_expert for spec in self.specs)
        self.host_bytes = self.expert_count * self.bytes_per_expert
        self.emulation, self.pinned = emulation, pin
        self.epoch, self.loaded = 0, False
        self._lock = threading.RLock()
        self._caches = weakref.WeakSet()
        self._reload_depth = 0
        self._reload_failed = False
        self._reload_initial_loaded = False
        self._reload_complete_pool = False
        self._hook_contexts = {}
        if preserve and any(getattr(getattr(expert, projection), spec.attribute).is_meta
                            for expert in self.experts for spec in self.specs for projection in spec.projections):
            raise ExpertCacheError("meta experts have no readable weights to preserve")
        self.banks = _allocate_banks(self.specs, self.expert_count, "cpu", pin=pin)
        try:
            if preserve:
                for spec in self.specs:
                    for index, expert in enumerate(self.experts):
                        offset = 0
                        for projection, rows in zip(spec.projections, spec.rows):
                            source = getattr(getattr(expert, projection), spec.attribute)
                            self.banks[spec.key][index, offset:offset + rows].view(torch.uint8).copy_(source.detach().view(torch.uint8))
                            offset += rows
            for index, expert in enumerate(self.experts):
                _bind_expert(expert, self.banks, self.specs, index)
        except BaseException:
            self.banks.clear()
            raise
        self._hooks = []
        self._install_load_hooks(moe, complete_pool=True)
        for expert in self.experts:
            self._install_load_hooks(expert, complete_pool=False)
            for projection in ("w1", "w2", "w3"):
                self._install_load_hooks(getattr(expert, projection), complete_pool=False)
        moe._host_expert_pool = self
        if hasattr(moe, "_grouped_bank"):
            moe._grouped_bank = None  # release any obsolete full-GPU expert bank
        if preserve:
            self.mark_loaded()

    @classmethod
    def from_moe(cls, moe, **kwargs):
        if getattr(moe, "_host_expert_pool", None) is not None:
            raise ExpertCacheError("MoE already belongs to a canonical host expert pool")
        return cls(moe, **kwargs)

    def _install_load_hooks(self, module, complete_pool):
        def before(current, state, prefix, metadata, strict, missing, unexpected, errors):
            self.begin_reload(complete_pool=complete_pool)
            self._hook_contexts.setdefault(id(current), []).append((prefix, errors))
        def after(current, incompatible):
            prefix, errors = self._hook_contexts[id(current)].pop()
            failed = bool(errors) or any(key.startswith(prefix) for key in
                                        incompatible.missing_keys + incompatible.unexpected_keys)
            self.finish_reload(success=not failed)
        self._hooks.append(module.register_load_state_dict_pre_hook(before))
        self._hooks.append(module.register_load_state_dict_post_hook(after))

    def begin_reload(self, *, complete_pool=True):
        with self._lock:
            if self._reload_depth == 0:
                caches = list(self._caches)
                # Check every cache before invalidating any, so a live lease leaves
                # both the source weights and all cache owners untouched.
                for cache in caches:
                    cache._check_reload_safe()
                for cache in caches:
                    cache._invalidate_for_reload()
                self._reload_initial_loaded = self.loaded
                self._reload_complete_pool = False
                self._reload_failed = False
                self.loaded = False
                self.epoch += 1
            self._reload_complete_pool |= complete_pool
            self._reload_depth += 1

    def finish_reload(self, *, success=True):
        with self._lock:
            if self._reload_depth <= 0:
                raise ExpertCacheError("finish_reload without begin_reload")
            self._reload_failed |= not success
            self._reload_depth -= 1
            if self._reload_depth == 0:
                self.loaded = not self._reload_failed and (self._reload_complete_pool or self._reload_initial_loaded)

    @contextmanager
    def reloading(self):
        self.begin_reload()
        try:
            yield self
        except BaseException:
            self.finish_reload(success=False)
            raise
        else:
            self.finish_reload()

    def mark_loaded(self):
        """For initial/manual fills. Use reloading() BEFORE modifying an active pool."""
        with self._lock:
            if self._reload_depth:
                raise ExpertCacheError("cannot mark_loaded inside an unfinished reload")
            self.begin_reload()
            self.finish_reload()


class _CompletedEvent:
    def query(self):
        return True

    def synchronize(self):
        return None


@dataclass
class _Slot:
    owner: int | None = None
    epoch: int = -1
    references: int = 0
    ready: object = None
    use_events: list = field(default_factory=list)
    touched: int = 0


class CacheLease:
    def __init__(self, cache, mapping, hit_ids, miss_ids, pending_ids, new_copy_ids, copies):
        self.cache = cache
        self.mapping = dict(mapping)
        self.hit_ids, self.miss_ids = tuple(hit_ids), tuple(miss_ids)
        self.pending_ids, self.new_copy_ids = tuple(pending_ids), tuple(new_copy_ids)
        self.copied_bytes = len(new_copy_ids) * cache.pool.bytes_per_expert
        self.dma_bytes = 0 if cache.emulation else self.copied_bytes
        self.ready_events = tuple(dict.fromkeys(cache._slots[index].ready for index in mapping.values()))
        self.epoch = cache.pool.epoch
        self.released = False
        self.waited = False
        self._consumer_streams = {}
        self._copies = copies
        self._wait_pairs = []

    def wait_on(self, stream=None):
        with self.cache._lock:
            self._validate()
            if self.cache.emulation:
                for event in self.ready_events:
                    event.synchronize()
            else:
                with torch.cuda.device(self.cache.device):
                    stream = stream or torch.cuda.current_stream(self.cache.device)
                    if stream.cuda_stream in self._consumer_streams:
                        return self
                    pending = [event for event in self.ready_events if not event.query()]
                    if pending:
                        wait_start, wait_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        wait_start.record(stream)
                        for event in pending:
                            stream.wait_event(event)
                        wait_end.record(stream)
                        self._wait_pairs.append((wait_start, wait_end))
                    for bank in self.cache.banks.values():
                        if bank is not None:
                            bank.record_stream(stream)
                    self._consumer_streams[stream.cuda_stream] = stream
            self.waited = True
        return self

    def _validate(self):
        if self.released:
            raise ExpertCacheError("lease has already been released")
        if self.epoch != self.cache.pool.epoch:
            raise ExpertCacheError("lease belongs to stale weights")

    def release(self, stream=None, *, done_event=None):
        self.cache._release(self, stream, done_event)

    def dma_timings(self):
        """Only completed real copy events have timings; never synchronize here."""
        if self.cache.emulation:
            return []
        records = []
        for kind, pairs in (("copy", self._copies), ("wait", self._wait_pairs)):
            for start, stop in pairs:
                records.append({"kind": kind, "bytes": self.cache.pool.bytes_per_expert if kind == "copy" else 0,
                                "start_event": start, "end_event": stop,
                                "duration_ms": start.elapsed_time(stop) if stop.query() else None})
        return records

    def __enter__(self):
        return self.wait_on()

    def __exit__(self, *_):
        self.release()


class FixedSlotCache:
    def __init__(self, pool, capacity, *, device="cuda:0", emulation=False, decay_interval=256):
        _integer(capacity, "capacity", positive=True)
        _integer(decay_interval, "decay_interval", positive=True)
        if capacity > pool.expert_count:
            raise ExpertCacheError("cache capacity exceeds the local expert count")
        if type(emulation) is not bool or emulation != pool.emulation:
            raise ExpertCacheError("host pool and cache execution modes must agree")
        self.device = torch.device(device)
        if emulation and self.device.type != "cpu" or not emulation and self.device.type != "cuda":
            raise ExpertCacheError("CPU emulation uses CPU slots; production cache uses CUDA slots")
        if not emulation and not pool.pinned:
            raise ExpertCacheError("DMA cache requires pinned canonical host banks")
        self.pool, self.capacity, self.emulation = pool, capacity, emulation
        self.mode = "cpu_reference" if emulation else "cuda_dma"
        self.cache_bytes = capacity * pool.bytes_per_expert
        self.decay_interval = decay_interval
        self._lock = threading.RLock()
        self._slots = [_Slot() for _ in range(capacity)]
        self._owners = {}
        self._frequencies = {}
        self._tick = 0
        self.closed = False
        self._faulted = False
        self._stats = {"acquires": 0, "hits": 0, "misses": 0, "pending_hits": 0,
                       "new_copies": 0, "copied_bytes": 0, "dma_bytes": 0,
                       "evictions": 0, "invalidations": 0}
        self.copy_stream = None if emulation else torch.cuda.Stream(device=self.device)
        self.banks = _allocate_banks(pool.specs, capacity, self.device)
        try:
            experts = []
            for index in range(capacity):
                expert = _clone_module_structure(pool.experts[0])
                # Always replace CPU Parameters with slot Parameters, never mutate
                # shared source Parameter objects copied by the structural clone.
                for projection in ("w1", "w2", "w3"):
                    linear = getattr(expert, projection)
                    for attribute in ("weight", "scale"):
                        parameter = getattr(linear, attribute, None)
                        if parameter is not None:
                            setattr(linear, attribute, torch.nn.Parameter(torch.empty(0, device="meta", dtype=parameter.dtype),
                                                                       requires_grad=parameter.requires_grad))
                _bind_expert(expert, self.banks, pool.specs, index)
                experts.append(expert)
            self.experts = torch.nn.ModuleList(experts)
        except BaseException:
            self.banks.clear()
            raise
        pool._caches.add(self)

    def _ids(self, ids):
        try:
            values = list(ids)
        except TypeError as exc:
            raise ExpertCacheError("expert ids must be an iterable of Python integers") from exc
        for eid in values:
            _integer(eid, "expert id")
            if eid >= self.pool.expert_count:
                raise ExpertCacheError("expert id outside the local host pool")
        return list(dict.fromkeys(values))

    def lookup(self, eid, *, include_pending=False):
        self._ids([eid])
        with self.pool._lock, self._lock:
            slot_index = self._owners.get(eid)
            if slot_index is None:
                return None
            slot = self._slots[slot_index]
            if slot.epoch != self.pool.epoch or not self.pool.loaded:
                return None
            return slot_index if include_pending or slot.ready.query() else None

    def is_resident(self, eid):
        return self.lookup(eid) is not None

    def cached_ids(self, *, include_pending=False):
        """One metadata snapshot for a prediction pass, with no per-ID allocations/locks.

        Pending slots already own a queued copy and should normally be excluded
        from optional prediction. Demand still checks readiness with its lease.
        """
        with self.pool._lock, self._lock:
            if self.closed or not self.pool.loaded:
                return set()
            return {eid for eid, index in self._owners.items()
                    if self._slots[index].epoch == self.pool.epoch
                    and (include_pending or self._slots[index].ready.query())}

    def _ensure_open(self):
        if self.closed:
            raise ExpertCacheError("cache is closed")
        if self._faulted:
            raise ExpertCacheError("cache copy failed; close and rebuild before serving")
        if not self.pool.loaded:
            raise ExpertCacheError("canonical host weights have not completed loading")
        if not self.emulation and torch.cuda.is_current_stream_capturing():
            raise ExpertCacheError("cache acquisition belongs outside CUDA graph capture")

    def acquire(self, ids, *, demand=True, protect_ids=()):
        requested = self._ids(ids)
        protected_experts = self._ids(protect_ids)
        if len(requested) > self.capacity:
            raise ExpertCacheError("the distinct expert request exceeds fixed slot capacity; split the expert batch")
        with self.pool._lock, self._lock:
            self._ensure_open()
            mapping, hits, pending, missing = {}, [], [], []
            for eid in requested:
                index = self._owners.get(eid)
                if index is not None and self._slots[index].epoch == self.pool.epoch:
                    mapping[eid] = index
                    (hits if self._slots[index].ready.query() else pending).append(eid)
                else:
                    missing.append(eid)
            protected = set(mapping.values())
            protected.update(self._owners[eid] for eid in protected_experts if eid in self._owners)
            candidates = [index for index, slot in enumerate(self._slots)
                          if slot.references == 0 and index not in protected]
            candidates.sort(key=lambda index: (self._slots[index].owner is not None,
                                               self._frequencies.get(self._slots[index].owner, 0),
                                               self._slots[index].touched, index))
            if len(candidates) < len(missing):
                raise ExpertCacheError("no evictable slots: active leases protect the cache")
            copies = []
            # Capacity failure above leaves all owners and counters unchanged.
            for eid, index in zip(missing, candidates):
                slot = self._slots[index]
                if slot.owner is not None:
                    self._owners.pop(slot.owner, None)
                    self._stats["evictions"] += 1
                try:
                    start, ready = self._copy_expert(eid, index)
                except BaseException:
                    # A partly overwritten slot must never advertise old/new weights.
                    slot.owner, slot.epoch, slot.ready = None, -1, None
                    self._faulted = True
                    for changed_eid in missing:
                        changed_index = self._owners.pop(changed_eid, None)
                        if changed_index is not None:
                            changed_slot = self._slots[changed_index]
                            changed_slot.owner, changed_slot.epoch = None, -1
                    raise
                slot.owner, slot.epoch, slot.ready = eid, self.pool.epoch, ready
                slot.use_events = []
                self._owners[eid] = index
                mapping[eid] = index
                if start is not None:
                    copies.append((start, ready))
            if demand:
                self._tick += 1
                if self._tick % self.decay_interval == 0:
                    self._frequencies = {eid: value * 0.5 for eid, value in self._frequencies.items()}
                for eid in requested:
                    self._frequencies[eid] = self._frequencies.get(eid, 0) + 1
            for eid, index in mapping.items():
                self._slots[index].references += 1
                self._slots[index].touched = self._tick
            self._stats["acquires"] += int(demand)
            self._stats["hits"] += len(hits) if demand else 0
            self._stats["misses"] += len(missing) + len(pending) if demand else 0
            self._stats["pending_hits"] += len(pending)
            self._stats["new_copies"] += len(missing)
            self._stats["copied_bytes"] += len(missing) * self.pool.bytes_per_expert
            self._stats["dma_bytes"] += 0 if self.emulation else len(missing) * self.pool.bytes_per_expert
            return CacheLease(self, mapping, hits, missing + pending, pending, missing, copies)

    def _copy_expert(self, eid, index):
        slot = self._slots[index]
        if self.emulation:
            # Cold reload/copy tests may inject an unfinished event. CPU emulation
            # still waits that dependency rather than pretending hardware overlap.
            for event in slot.use_events:
                event.synchronize()
            if slot.ready is not None:
                slot.ready.synchronize()
            for spec in self.pool.specs:
                self.banks[spec.key][index].view(torch.uint8).copy_(self.pool.banks[spec.key][eid].view(torch.uint8))
            return None, _CompletedEvent()
        with torch.cuda.device(self.device), torch.cuda.stream(self.copy_stream):
            for event in slot.use_events:
                self.copy_stream.wait_event(event)
            if slot.ready is not None:
                self.copy_stream.wait_event(slot.ready)
            start = torch.cuda.Event(enable_timing=True)
            ready = torch.cuda.Event(enable_timing=True)
            start.record(self.copy_stream)
            for spec in self.pool.specs:
                self.banks[spec.key].record_stream(self.copy_stream)
                self.banks[spec.key][index].view(torch.uint8).copy_(self.pool.banks[spec.key][eid].view(torch.uint8), non_blocking=True)
            ready.record(self.copy_stream)
            return start, ready

    def _release(self, lease, stream, done_event):
        with self._lock:
            lease._validate()
            if self.emulation:
                events = [done_event or _CompletedEvent()]
            elif done_event is not None:
                events = [done_event]
            else:
                with torch.cuda.device(self.device):
                    current = stream or torch.cuda.current_stream(self.device)
                    streams = dict(lease._consumer_streams)
                    streams[current.cuda_stream] = current
                    events = []
                    for consumer in streams.values():
                        event = torch.cuda.Event()
                        event.record(consumer)
                        events.append(event)
            for index in set(lease.mapping.values()):
                slot = self._slots[index]
                if slot.references <= 0:
                    raise ExpertCacheError("corrupt cache lease reference count")
                slot.references -= 1
                slot.use_events = [event for event in slot.use_events if not event.query()] + events
            lease.released = True

    def prefetch(self, ids):
        lease = self.acquire(ids, demand=False)
        lease.wait_on(self.copy_stream)
        lease.release(self.copy_stream)
        return lease  # completed/released ticket, not a lease licensed for later compute

    def try_prefetch(self, ids, *, max_copies=2, max_evictions=2,
                     reserve_slots=1, protect_ids=()):
        """Optional, bounded prediction: skip invalid/no-room candidates without failing a job.

        Active leases and the most recent working set remain protected. Preflight
        and acquire share the same locks, so every selected victim obeys the copy,
        eviction and spare-slot budgets atomically. CUDA/copy faults still fail
        closed; they are not recoverable prediction misses or valid weights.
        Returns a released ticket (or None), never a license to consume slots.
        """
        for value, name in ((max_copies, "max_copies"), (max_evictions, "max_evictions"),
                            (reserve_slots, "reserve_slots")):
            _integer(value, name)
        try:
            requested, protected_experts = self._ids(ids), self._ids(protect_ids)
        except ExpertCacheError:
            return None
        with self.pool._lock, self._lock:
            if self.closed or not self.pool.loaded:
                return None
            self._ensure_open()  # actual failed DMA/capture misuse must not masquerade as success
            existing = [eid for eid in requested if eid in self._owners
                        and self._slots[self._owners[eid]].epoch == self.pool.epoch]
            protected = {self._owners[eid] for eid in (*existing, *protected_experts) if eid in self._owners}
            victims = [index for index, slot in enumerate(self._slots)
                       if slot.references == 0 and index not in protected]
            victims.sort(key=lambda index: (self._slots[index].owner is not None,
                                           self._frequencies.get(self._slots[index].owner, 0),
                                           self._slots[index].touched, index))
            limit = max(0, len(victims) - reserve_slots)
            empty = sum(self._slots[index].owner is None for index in victims[:limit])
            limit = min(limit, max_copies, empty + max_evictions)
            missing = [eid for eid in requested if eid not in existing][:limit]
            chosen = (existing + missing)[:self.capacity]
            if not chosen:
                return None
            lease = self.acquire(chosen, demand=False, protect_ids=protected_experts)
            try:
                lease.wait_on(self.copy_stream)
            finally:
                lease.release(self.copy_stream)
            return lease

    def _check_reload_safe(self):
        with self._lock:
            if any(slot.references for slot in self._slots):
                raise ExpertCacheError("cannot reload weights while a cache lease is active")

    def _invalidate_for_reload(self):
        with self._lock:
            self._check_reload_safe()
            # Pinned source bytes cannot be overwritten while DMA is still reading
            # them. Reload is a cold barrier; no global/per-token synchronization.
            for slot in self._slots:
                if slot.ready is not None:
                    slot.ready.synchronize()
                slot.owner, slot.epoch, slot.ready = None, -1, None
            if self._faulted and self.copy_stream is not None:
                self.copy_stream.synchronize()
            self._owners.clear()
            self._stats["invalidations"] += 1

    def stats(self):
        with self._lock:
            return {**self._stats, "mode": self.mode, "epoch": self.pool.epoch,
                    "host_bytes": self.pool.host_bytes, "cache_bytes": self.cache_bytes,
                    "capacity": self.capacity, "loaded": self.pool.loaded,
                    "live_leases": sum(slot.references for slot in self._slots),
                    "owners": {eid: index for eid, index in self._owners.items()}}

    def close(self):
        with self.pool._lock, self._lock:
            self._check_reload_safe()
            if self.copy_stream is not None:
                self.copy_stream.synchronize()  # cold teardown/failed allocation only
            for slot in self._slots:
                for event in slot.use_events:
                    event.synchronize()
                if slot.ready is not None:
                    slot.ready.synchronize()
            self.banks.clear()
            self.experts = torch.nn.ModuleList()
            self._owners.clear()
            self.pool._caches.discard(self)
            self.closed = True


class StageBudgetManager:
    """Register every main/MTP pool before allocating fixed per-pool slot banks.

    This never guesses available GPU memory. The caller supplies an explicit byte
    budget (after resident/KV/workspace reserves) or explicit slots per pool.
    Registration freezes at allocation because changing existing capacities would
    replace pointers already captured by graphs. Build lazy MTP pools eagerly
    before this boundary, or rebuild the stage with a new complete budget.
    """
    def __init__(self, budget_bytes=None, *, slots_per_pool=None, device="cuda:0", emulation=False):
        if budget_bytes is None and slots_per_pool is None:
            raise ExpertCacheError("an explicit cache byte budget or slots_per_pool is required")
        if budget_bytes is not None:
            _integer(budget_bytes, "budget_bytes")
        if slots_per_pool is not None:
            _integer(slots_per_pool, "slots_per_pool", positive=True)
        self.budget_bytes, self.slots_per_pool = budget_bytes, slots_per_pool
        self.device, self.emulation = device, emulation
        self.pools, self.caches, self._minimums = OrderedDict(), {}, {}
        self.allocated_bytes = 0
        self._allocated = False
        self._lock = threading.RLock()

    @property
    def host_bytes(self):
        return sum(pool.host_bytes for pool in self.pools.values())

    def register(self, pool, key, *, min_slots=1):
        _integer(min_slots, "min_slots", positive=True)
        with self._lock:
            if self._allocated:
                raise ExpertCacheError("pool registration is frozen; register all main/MTP pools before cache allocation")
            if key in self.pools or any(existing is pool for existing in self.pools.values()):
                raise ExpertCacheError("duplicate cache pool/key registration")
            if pool.emulation != self.emulation or min_slots > pool.expert_count:
                raise ExpertCacheError("pool mode/minimum capacity does not fit this manager")
            self.pools[key] = pool
            self._minimums[key] = min_slots
        return key

    def _capacities(self):
        if not self.pools:
            raise ExpertCacheError("cannot allocate an empty cache manager")
        if self.slots_per_pool is not None:
            capacities = {key: self.slots_per_pool for key in self.pools}
            if any(capacities[key] < self._minimums[key] or capacities[key] > pool.expert_count for key, pool in self.pools.items()):
                raise ExpertCacheError("explicit per-pool slots violate expert count/minimum")
        else:
            capacities = dict(self._minimums)
        total = sum(capacities[key] * pool.bytes_per_expert for key, pool in self.pools.items())
        if self.budget_bytes is not None and total > self.budget_bytes:
            raise ExpertCacheError(f"cache budget {self.budget_bytes} bytes is below required minimum {total}")
        if self.slots_per_pool is None:
            remaining = self.budget_bytes - total
            order = {key: index for index, key in enumerate(self.pools)}
            while True:
                candidates = [key for key, pool in self.pools.items()
                              if capacities[key] < pool.expert_count and pool.bytes_per_expert <= remaining]
                if not candidates:
                    break
                key = min(candidates, key=lambda item: (capacities[item] * self.pools[item].bytes_per_expert, order[item]))
                capacities[key] += 1
                remaining -= self.pools[key].bytes_per_expert
        return capacities

    def allocate(self):
        with self._lock:
            if self._allocated:
                return self.caches
            capacities = self._capacities()
            created = {}
            try:
                for key, pool in self.pools.items():
                    created[key] = FixedSlotCache(pool, capacities[key], device=self.device, emulation=self.emulation)
            except BaseException:
                for cache in created.values():
                    cache.close()
                created.clear()
                raise
            self.caches = created
            self.allocated_bytes = sum(cache.cache_bytes for cache in created.values())
            self._allocated = True
            return self.caches

    def cache_for(self, key):
        if key not in self.pools:
            raise ExpertCacheError("unknown cache pool key")
        return self.allocate()[key]

    def stats(self):
        return {"mode": "cpu_reference" if self.emulation else "cuda_dma",
                "budget_bytes": self.budget_bytes, "allocated_bytes": self.allocated_bytes,
                "host_bytes": self.host_bytes, "registered_pools": len(self.pools),
                "allocated": self._allocated,
                "pools": {str(key): cache.stats() for key, cache in self.caches.items()}}

    def close(self):
        with self._lock:
            for cache in self.caches.values():
                cache.close()
            self.caches.clear()
            self.allocated_bytes = 0
            self._allocated = False


SharedCacheManager = StageBudgetManager

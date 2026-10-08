"""Local RAM expert pools behind V4's original Block and Expert mathematics.

Only storage and dispatch change. Resident attention/router/shared/HC parameters
stay on the execution device; every routed expert remains a canonical host view.
Cache slot modules are not registered under the checkpoint's MoE, and a lease
protects their fixed addresses until its last GPU consumer finishes.

CPU execution is explicit cache emulation for parity tests, never a production
fallback. GPU grouped decode reuses v4_moe_grouped's existing FP4 kernels. All
other shapes retain the reference's per-expert row shapes, logical expert order
and duplicate-index scatter, even when the routing union exceeds cache capacity.
"""
from contextlib import contextmanager, nullcontext
import threading
from types import MethodType

import torch


def _repair_scale_aliases(module):
    """to_empty replaces Parameters; quantized Linear.weight.scale must follow."""
    for child in module.modules():
        weight, scale = getattr(child, "weight", None), getattr(child, "scale", None)
        if isinstance(weight, torch.nn.Parameter) and scale is not None:
            weight.scale = scale


class _AttentionPlaceholder(torch.nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()


def hybrid_block_cls(base_cls, ref_model, manager, *, device, emulation=False,
                     role="main", metrics=None):
    """Factory for an instance-local Block/DSparkBlock subclass, with no model monkeypatch.

    The original constructor runs on META. Its attention_cls seam gets a temporary
    placeholder, so the reference's process-wide lru_cache of rotary tables never
    caches a META tensor. The original attention class is then rebuilt normally,
    initializing its real KV/state/rotary buffers exactly as the reference does.
    Register ALL main/draft pools before manager.allocate(), then bind_caches().
    """
    if role not in ("main", "draft"):
        raise ValueError("hybrid expert role must be main or draft")
    if not emulation and not str(device).startswith("cuda"):
        raise ValueError("RAM expert placement requires CUDA; CPU cache emulation must be explicit")
    if emulation and str(device) != "cpu":
        raise ValueError("cache emulation must execute on CPU")
    if int(getattr(ref_model, "world_size", 1)) != 1:
        raise ValueError("local expert caching requires V4's single-rank pipeline stage")
    attention_cls = base_cls.attention_cls

    class HybridBlock(base_cls):
        attention_cls = _AttentionPlaceholder

        def __init__(self, layer_id, args):
            from v4_expert_cache import HostExpertPool
            with torch.device("meta"):
                super().__init__(layer_id, args)
            self.attention_cls = attention_cls
            with torch.device(device):
                self.attn = self.attention_cls(layer_id, args)
            # Materialize only the resident half. Never .to_empty() the whole MoE:
            # that would allocate every routed expert on the GPU before offload.
            for name, child in self.named_children():
                if name == "attn":
                    continue
                if name == "ffn":
                    for resident in (child.gate, child.shared_experts):
                        resident.to_empty(device=device)
                        _repair_scale_aliases(resident)
                else:
                    child.to_empty(device=device)
                    _repair_scale_aliases(child)
            # HC and DSpark head scalars live directly on Block, outside children.
            for name, parameter in list(self.named_parameters(recurse=False)):
                self._parameters[name] = torch.nn.Parameter(
                    torch.empty_like(parameter, device=device), requires_grad=parameter.requires_grad)
            pool = HostExpertPool.from_moe(self.ffn, pin=not emulation,
                                          emulation=emulation, preserve=False)
            key = (role, layer_id)
            manager.register(pool, key, min_slots=1)
            runtime = HybridMoE(self.ffn, ref_model, pool, key, device=device,
                                emulation=emulation, role=role, observer=metrics)
            self.ffn._hybrid_runtime = runtime
            # Protect the canonical host parameter views from a legacy bank_layout
            # pass. Parent Stage normally skips that pass entirely in this mode.
            self.ffn._grouped_bank = pool.banks
            self.ffn.forward = MethodType(_hybrid_forward, self.ffn)

        def forward(self, *args, **kwargs):
            position = args[1] if len(args) > 1 else kwargs.get("start_pos", 0)
            runtime = self.ffn._hybrid_runtime
            phase = "prefill" if position == 0 else runtime.phase
            with runtime.phase_context(phase):
                if role == "draft" and position > 0 and runtime.phase != "replay" and not runtime._warmup_depth:
                    runtime.prefetch_for_attention()
                return super().forward(*args, **kwargs)

    HybridBlock.__name__ = f"Hybrid{base_cls.__name__}"
    return HybridBlock


def bind_caches(blocks, caches):
    """Bind manager.allocate()'s fixed slots without registering checkpoint tensors."""
    for block in blocks:
        runtime = block.ffn._hybrid_runtime
        cache = caches[runtime.key]
        if cache.pool is not runtime.pool:
            raise ValueError("expert cache belongs to another host pool")
        runtime.cache = cache


def _hybrid_forward(moe, x, input_ids):
    return moe._hybrid_runtime.forward(x, input_ids)


_hybrid_forward._v4_hybrid = True


class HybridMoE:
    """One layer's local dispatch; mathematical kernels stay in their owning modules."""

    def __init__(self, moe, model, pool, key, *, device, emulation=False,
                 role="main", observer=None):
        self.moe, self.model, self.pool, self.key = moe, model, pool, key
        self.device, self.emulation = str(device), bool(emulation)
        self.role, self.phase, self.observer = role, "decode", observer
        self.cache = None
        self.recent_ids = []
        self.last_event = None
        self.grouped_steps = 0
        self.generic_steps = 0
        self._warmup_depth = 0
        self.profiler = None
        self._forward_lock = threading.RLock()
        self._router_stream = None
        self._router_buffers = {}  # per consumer stream: host IDs and fixed physical-slot map
        self.router_overlap_enabled = False
        self.prefetch_enabled = False
        self._prefetch_policy = None
        self._prefetch_spare = 1
        self._pending_prediction = set()
        self._prefetch_counts = dict(requested=0, used=0, wasted=0, skipped=0, candidate_count=0)

    def _host(self, name):
        return self.profiler.host(f"{self.role}.{self.phase}.{name}.host") \
            if self.profiler is not None and not self._warmup_depth else nullcontext()

    def _gpu(self, name):
        return self.profiler.gpu(f"{self.role}.{self.phase}.{name}.gpu", torch, self.device) \
            if self.profiler is not None and not self._warmup_depth else nullcontext()

    def configure_prefetch(self, enabled=False, max_slots=2, ema_decay=.85,
                           heat_threshold=.2, warmup_steps=2, min_cache_spare=1):
        from v4_chunked_prefill import ControlledPrefetcher
        if type(enabled) is not bool:
            raise ValueError("prefetch enabled must be a boolean")
        if type(min_cache_spare) is not int or min_cache_spare < 0:
            raise ValueError("min_cache_spare must be a nonnegative integer")
        self._prefetch_policy = ControlledPrefetcher(
            max_prefetch_slots=max_slots, ema_decay=ema_decay, heat_threshold=heat_threshold,
            expert_count=self.pool.expert_count, warmup_steps=warmup_steps)
        self.prefetch_enabled = enabled
        self._prefetch_spare = min_cache_spare
        self.reset_job()

    def reset_job(self):
        """Fresh observations per job; canonical weights, cache and measured heat survive."""
        self._pending_prediction.clear()
        self._prefetch_counts = dict(requested=0, used=0, wasted=0, skipped=0, candidate_count=0)
        if self._prefetch_policy is not None:
            self._prefetch_policy.reset_job()
        self.last_event = None
        self.grouped_steps = self.generic_steps = 0

    def prefetch_stats(self):
        return dict(self._prefetch_counts)

    def prefetch_config(self):
        """Detached calibration metadata; no dynamic affinity or job counters."""
        policy = self._prefetch_policy
        return {"configured": policy is not None, "enabled": self.prefetch_enabled,
                "max_slots": policy.max_prefetch_slots if policy is not None else None,
                "warmup_steps": policy.warmup_steps if policy is not None else None,
                "ema_decay": policy.ema_decay if policy is not None else None,
                "heat_threshold": policy.heat_threshold if policy is not None else None,
                "min_cache_spare": self._prefetch_spare,
                "router_shared_overlap": self.router_overlap_enabled}

    def router_scratch_tensors(self):
        """Named actual storages for resource observers; never expose control dictionaries.

        Diagnostic callers inspect allocation metadata at a barrier and must not
        mutate tensor values. Cloning would hide the actual owned backing storage.
        """
        for stream, entry in tuple(self._router_buffers.items()):
            for name in ("ids", "slots_host", "slots_gpu"):
                yield f"stream_{stream}.{name}", entry[name]

    def configure_router_overlap(self, enabled=False):
        """Explicit A/B experiment, off by default: shared overlaps router IDs D2H.

        Early shared work can finish before miss DMA is submitted and lose the
        existing shared/weight-copy overlap. Production preserves acquire-before-
        shared until real phase profiles justify changing that ordering.
        """
        if type(enabled) is not bool:
            raise ValueError("router overlap must be a boolean")
        self.router_overlap_enabled = enabled

    def _prefetch_quality(self, **values):
        row = dict(requested=0, used=0, wasted=0, skipped=0, candidate_count=0)
        row.update(values)
        for key, value in row.items():
            self._prefetch_counts[key] += value
        self._event(row, kind="prefetch_quality")

    def prefetch_for_attention(self):
        """Bounded prediction on measured history, outside attention/graph execution.

        No routing oracle is claimed: wrong predictions are disposable, while the
        ordinary Gate -> demand DMA path remains authoritative. Empty/cold or fully
        protected caches are no-ops. The latest working set is never evicted here.
        """
        if not self.prefetch_enabled or self._warmup_depth or self.phase == "replay":
            return None
        self._require_cache()
        policy = self._prefetch_policy
        with self._forward_lock, self._host("prefetch_schedule"):
            # A repeated scheduling call before execution replaces an unused prediction;
            # account it as waste rather than let it disappear from quality statistics.
            if self._pending_prediction:
                self._prefetch_quality(wasted=len(self._pending_prediction))
                self._pending_prediction.clear()
            cached = self.cache.cached_ids(include_pending=True)
            candidates = policy.predict_prefetch_set(exclude_currently_cached=cached)
            ticket = self.cache.try_prefetch(
                candidates, max_copies=policy.max_prefetch_slots,
                max_evictions=policy.max_prefetch_slots, reserve_slots=self._prefetch_spare,
                protect_ids=self.recent_ids)
            requested = set(ticket.mapping) if ticket is not None else set()
            self._pending_prediction = requested
            self._prefetch_quality(candidate_count=len(candidates), requested=len(requested),
                                   skipped=len(candidates) - len(requested))
            if ticket is not None:
                self._event({"dma_bytes": 0 if self.emulation else ticket.copied_bytes,
                             "dma_wait_ms": 0.0}, ticket.dma_timings(), kind="prefetch")
            return ticket

    def _observe_actual_routes(self, ordered):
        if self._warmup_depth or self._prefetch_policy is None or not self.prefetch_enabled or self.phase == "replay":
            return
        needed = set(ordered)
        if self._pending_prediction:
            self._prefetch_quality(used=len(self._pending_prediction & needed),
                                   wasted=len(self._pending_prediction - needed))
            self._pending_prediction.clear()
        self._prefetch_policy.update_access_history(ordered)

    def _router_buffers_for(self, indices, consumer):
        """Stable per-stream pinned scratch; never shared by concurrent consumers."""
        key = consumer.cuda_stream
        rows, topk = indices.shape
        entry = self._router_buffers.get(key)
        if entry is None or entry["rows"] < rows or entry["dtype"] != indices.dtype or entry["topk"] != topk:
            capacity = max(rows, entry["rows"] * 2 if entry is not None else 1)
            entry = {"rows": capacity, "topk": topk, "dtype": indices.dtype, "consumer": consumer,
                     "ids": torch.empty((capacity, topk), dtype=indices.dtype,
                                        device="cpu", pin_memory=True),
                     # Long physical indices make all four _gather_fp ids.long()
                     # calls no-ops instead of allocating/casting the same six IDs.
                     "slots_host": torch.empty(topk, dtype=torch.long, device="cpu", pin_memory=True),
                     "slots_gpu": torch.empty(topk, dtype=torch.long, device=self.device),
                     "gate_ready": torch.cuda.Event(), "ids_ready": torch.cuda.Event()}
            if key in self._router_buffers:
                # Preserve old pinned H2D source/slot-map storage until the current
                # Gate completion proves all previous consumers on this stream done.
                entry["retired"] = self._router_buffers[key]
            self._router_buffers[key] = entry
        return entry

    def _read_routes_and_shared(self, xv, indices):
        if self.emulation:
            with self._host("router_readback"):
                return indices.detach().to("cpu").tolist(), None, None
        consumer = torch.cuda.current_stream(self.device)
        scratch = self._router_buffers_for(indices, consumer)
        if self._router_stream is None:
            self._router_stream = torch.cuda.Stream(device=self.device)
        scratch["gate_ready"].record(consumer)
        with torch.cuda.stream(self._router_stream):
            self._router_stream.wait_event(scratch["gate_ready"])
            with self._gpu("router_readback"):
                scratch["ids"][:indices.shape[0]].copy_(indices.detach(), non_blocking=True)
            scratch["ids_ready"].record(self._router_stream)
        shared = None
        if self.router_overlap_enabled:
            # Experimental ordering only. The default queues shared AFTER expert
            # acquire, so on-demand weight DMA still overlaps its computation.
            with self._gpu("shared"):
                shared = self.moe.shared_experts(xv)
        with self._host("router_readback"):
            scratch["ids_ready"].synchronize()  # one necessary host wait, not a device-wide sync
            rows = scratch["ids"][:indices.shape[0]].tolist()
            scratch.pop("retired", None)
        return rows, shared, scratch

    def _physical_slots(self, logical_ids, mapping, xv, scratch):
        if scratch is None:
            return torch.tensor([mapping[e] for e in logical_ids], dtype=torch.int32, device=xv.device)
        signature = (self.pool.epoch, tuple(mapping[eid] for eid in logical_ids))
        if scratch.get("slot_signature") == signature:
            return scratch["slots_gpu"]
        for index, slot in enumerate(signature[1]):
            scratch["slots_host"][index] = slot
        scratch["slots_gpu"].copy_(scratch["slots_host"], non_blocking=True)
        scratch["slot_signature"] = signature
        # On this consumer stream, the current Gate/IDs-ready dependency completed
        # after every previous slot-map copy. Host reuse is safe, and GPU reads/copy
        # remain FIFO. Different streams own different scratch allocations.
        return scratch["slots_gpu"]

    def set_phase(self, phase, role=None):
        if phase not in ("prefill", "decode", "replay"):
            raise ValueError("unknown hybrid work phase")
        if role is not None:
            if role not in ("main", "draft"):
                raise ValueError("unknown hybrid work role")
            self.role = role
        self.phase = phase

    def set_observer(self, observer):
        self.observer = observer

    @contextmanager
    def phase_context(self, phase):
        before = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = before

    @contextmanager
    def capture_warmup(self):
        """Graph priming can warm slots, but is not an executed model position."""
        self._warmup_depth += 1
        try:
            yield
        finally:
            self._warmup_depth -= 1

    @property
    def supports_grouped(self):
        if self.emulation or self.cache is None:
            return False
        banks = self.cache.banks
        return (banks["w13"].dtype == torch.float4_e2m1fn_x2
                and banks["w13"].shape[1] % 128 == 0
                and banks["w2"].shape[1] % 128 == 0
                and self.moe.n_activated_experts <= 32)

    def _event(self, counts, timings=(), *, kind="routes"):
        if self._warmup_depth:
            return
        event = dict(kind=kind, role=self.role, phase=self.phase,
                     layer_id=self.moe.layer_id, **counts)
        if timings and not self.emulation:
            event["dma_timings"] = timings
        self.last_event = event
        if self.observer is not None:
            self.observer(event)

    def _require_cache(self):
        if self.cache is None:
            raise RuntimeError("hybrid MoE cache not bound: register all pools, allocate and bind before serving")
        if not self.pool.loaded:
            raise RuntimeError("hybrid host expert pool is not loaded")

    def prefetch_recent(self):
        """Optional local prediction from past routes, never a claim about future router output.

        Called explicitly by a scheduling policy before attention, not automatically
        by the mathematical layer. No network read and no second router invocation.
        """
        self._require_cache()
        if not self.recent_ids:
            return
        ticket = self.cache.try_prefetch(self.recent_ids[:self.cache.capacity],
                                         max_copies=min(2, self.cache.capacity),
                                         max_evictions=min(2, self.cache.capacity), reserve_slots=0)
        if ticket is not None:
            self._event({"dma_bytes": 0 if self.emulation else ticket.copied_bytes,
                         "dma_wait_ms": 0.0}, ticket.dma_timings(), kind="prefetch")

    def _record_lease(self, lease, pairs, counts, timings):
        for eid in lease.mapping:
            n = len(pairs[eid])
            counts["routed_entries"] += n
            if self.emulation:
                counts["reference_routes"] += n
            elif eid in lease.hit_ids:
                counts["resident_hits"] += n
            else:
                counts["dma_misses"] += n
        if not self.emulation:
            counts["dma_bytes"] += lease.copied_bytes
            observed = lease.dma_timings()
            if observed:
                timings.extend(observed)

    def forward(self, x, input_ids):
        # One module normally has one serving thread. Serializing host scratch use
        # also makes direct callers safe; different CUDA streams retain independent
        # GPU/pinned buffers and the cache's consumer-event leases stay authoritative.
        with self._forward_lock:
            return self._forward(x, input_ids)

    def _forward(self, x, input_ids):
        self._require_cache()
        if not self.emulation and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("local expert cache routing must remain outside CUDA graphs")
        shape = x.shape
        xv = x.view(-1, self.moe.dim)
        with self._gpu("gate"):
            weights, indices = self.moe.gate(xv, input_ids.flatten())
        # Exactly one IDs readback per layer; the host still must resolve demand
        # misses. GPU shared work is queued while a dedicated stream copies into
        # reusable pinned IDs instead of synchronously allocating a CPU tensor.
        logical_rows, shared, scratch = self._read_routes_and_shared(xv, indices)
        with self._host("cache_schedule"):
            pairs = {}
            for row, eids in enumerate(logical_rows):
                for top, eid in enumerate(eids):
                    eid = int(eid)
                    if not 0 <= eid < self.pool.expert_count:
                        raise RuntimeError(f"router selected invalid expert {eid}")
                    pairs.setdefault(eid, []).append((row, top))
            ordered = sorted(pairs)
            self._observe_actual_routes(ordered)
            self.recent_ids = sorted(ordered, key=lambda e: (-len(pairs[e]), e))
        counts = {"routed_entries": 0, "resident_hits": 0, "dma_misses": 0,
                  "dma_bytes": 0, "dma_wait_ms": 0.0, "cpu_misses": 0, "reference_routes": 0}
        timings = []
        y = torch.zeros_like(xv, dtype=torch.float32)
        if xv.shape[0] == 1 and self.supports_grouped and len(ordered) <= self.cache.capacity and indices.shape[1] <= 32:
            with self._host("cache_schedule"):
                lease = self.cache.acquire(ordered)
            try:
                # Copies run on their own stream while the independent shared expert computes.
                if shared is None:
                    with self._gpu("shared"):
                        shared = self.moe.shared_experts(xv)
                lease.wait_on()
                from v4_moe_grouped import grouped_routed_sum
                physical = self._physical_slots(logical_rows[0], lease.mapping, xv, scratch)
                with self._gpu("routed"):
                    y = grouped_routed_sum(self.moe, self.model, xv, weights,
                                           indices[0].to(torch.int32), self.cache.banks, physical)
                if not self._warmup_depth:
                    self.grouped_steps += 1
                self._record_lease(lease, pairs, counts, timings)
            finally:
                lease.release()
        else:
            for begin in range(0, len(ordered), self.cache.capacity):
                batch = ordered[begin:begin + self.cache.capacity]
                with self._host("cache_schedule"):
                    lease = self.cache.acquire(batch)
                try:
                    if shared is None:
                        with self._gpu("shared"):
                            shared = self.moe.shared_experts(xv)
                    lease.wait_on()
                    with self._gpu("routed"):
                        for eid in batch:
                            rows, top = zip(*pairs[eid])
                            row_idx = torch.tensor(rows, dtype=torch.long, device=xv.device)
                            top_idx = torch.tensor(top, dtype=torch.long, device=xv.device)
                            expert = self.cache.experts[lease.mapping[eid]]
                            # Preserve the original Expert.forward shape and duplicate-index
                            # scatter: hash repeats keep the final write, not an extra sum.
                            y[row_idx] += expert(xv[row_idx], weights[row_idx, top_idx, None])
                    self._record_lease(lease, pairs, counts, timings)
                finally:
                    lease.release()
            if not self._warmup_depth:
                self.generic_steps += 1
        if shared is None:
            shared = self.moe.shared_experts(xv)
        y += shared  # Always last, after the logical ascending expert fold.
        self._event(counts, timings)
        return y.type_as(xv).view(shape)

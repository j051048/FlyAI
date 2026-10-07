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
from contextlib import contextmanager
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
        lease = self.cache.acquire(self.recent_ids[:self.cache.capacity], demand=False)
        try:
            # Release on the copy stream: demand will wait on the readiness events.
            stream = getattr(self.cache, "copy_stream", None)
            lease.wait_on(stream)
            timings = lease.dma_timings()
            self._event({"dma_bytes": 0 if self.emulation else lease.copied_bytes,
                         "dma_wait_ms": 0.0}, timings, kind="prefetch")
        finally:
            lease.release(getattr(self.cache, "copy_stream", None))

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
        self._require_cache()
        if not self.emulation and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("local expert cache routing must remain outside CUDA graphs")
        shape = x.shape
        xv = x.view(-1, self.moe.dim)
        weights, indices = self.moe.gate(xv, input_ids.flatten())
        # The only router D2H in this layer. All per-expert row/top indices are
        # derived from these host ids; torch.where/nonzero never adds a GPU sync.
        logical_rows = indices.detach().to("cpu").tolist()
        pairs = {}
        for row, eids in enumerate(logical_rows):
            for top, eid in enumerate(eids):
                eid = int(eid)
                if not 0 <= eid < self.pool.expert_count:
                    raise RuntimeError(f"router selected invalid expert {eid}")
                pairs.setdefault(eid, []).append((row, top))
        ordered = sorted(pairs)
        self.recent_ids = sorted(ordered, key=lambda e: (-len(pairs[e]), e))
        counts = {"routed_entries": 0, "resident_hits": 0, "dma_misses": 0,
                  "dma_bytes": 0, "dma_wait_ms": 0.0, "cpu_misses": 0, "reference_routes": 0}
        timings = []
        y = torch.zeros_like(xv, dtype=torch.float32)
        shared = None
        if xv.shape[0] == 1 and self.supports_grouped and len(ordered) <= self.cache.capacity:
            lease = self.cache.acquire(ordered)
            try:
                # Copies run on their own stream while the independent shared expert computes.
                shared = self.moe.shared_experts(xv)
                lease.wait_on()
                from v4_moe_grouped import grouped_routed_sum
                physical = torch.tensor([lease.mapping[e] for e in logical_rows[0]],
                                        dtype=torch.int32, device=xv.device)
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
                lease = self.cache.acquire(batch)
                try:
                    if shared is None:
                        shared = self.moe.shared_experts(xv)
                    lease.wait_on()
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

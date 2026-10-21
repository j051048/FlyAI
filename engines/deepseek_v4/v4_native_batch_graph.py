"""Bounded request-axis CUDA graphs for native tokenwise batches.

The capture boundary is attention and dense hyper-connections. Routed MoE is
executed eagerly on the REAL request batch between two graphs; it is never
traced through the reference's CPU routing synchronization. This is explicitly
partial coverage, not a whole-stage or whole-MoE graph claim.

Native ``topk`` has width-dependent tie order. Its original narrow scoring and
selection therefore stay eager; the graph reads the resulting ordered indices
from a static input, padded ONLY at the tail with -1. No stable-sort replacement
or widened-score topk is permitted. Indexer state advances once before replay.
The graph key is request count, context bucket and compression branch; position
and index lists are live device inputs. A finite budget limits the graph pool;
an exhausted key executes the same eager twin.
"""
from dataclasses import dataclass, field
from contextlib import nullcontext

import torch

from v4_batch_invariant import request_row_math
from v4_runtime_init import tensor_storage_signature, bind_attention_aliases
from v4_conversation_cache import _model_stamp
from v4_whole_layer_graph import (_Ref, block_pre_cs, block_post_cs, build_win_topk,
    build_comp_topk, bucket_width)

SCHEMA = "v4-native-batch-graphs/1"
COVERAGE = "attention_dense_partial"


class UnsafeCaptureState(RuntimeError):
    """The capture stream/state could not be drained; eager continuation is forbidden."""


@dataclass(frozen=True)
class BatchGraphKey:
    layer: int
    requests: int
    context_bucket: int
    read_width: int = field(compare=False)
    compress: bool
    position: int = field(compare=False)


def graph_key(layer, *, requests, position, max_jobs, horizon):
    """A coarse context key with native logical width kept as live metadata.

    Position supplies the representative capture input, but does not determine
    equality: original narrow scoring stays eager, so two positions within the
    same coarse bucket/branch safely reuse a graph without changing topk ties.
    """
    if (type(requests) is not int or not 1 <= requests <= max_jobs
            or type(position) is not int or not 0 < position < horizon):
        raise ValueError("native batch graph requires bounded decode position and request count")
    ratio = int(layer.attn.compress_ratio)
    width = (position + 1) // ratio if ratio else 0
    maximum = horizon // ratio if ratio else 0
    floor = layer.attn.indexer.index_topk if ratio and layer.attn.indexer is not None else 0
    return BatchGraphKey(int(layer.layer_id), requests,
        bucket_width(width, maximum, floor=floor) if ratio else 0, width,
        bool(ratio and (position + 1) % ratio == 0), position)


class NativeBatchGraphPool:
    """One pool owns a fixed native-lane workspace and its graph reservations.

    Call ``forward`` only while that workspace is bound to the stage. A CPU
    reference executes the identical eager program and reports ZERO captures
    and replays. Graph output is cloned before escaping, so another key/request
    cannot overwrite a previous frame that is still owned by transport.
    """
    def __init__(self, stage, workspace, *, max_jobs, max_graphs, graph_bytes,
                 reference=False):
        if (type(max_jobs) is not int or not 2 <= max_jobs <= 4
                or type(max_graphs) is not int or not 2 <= max_graphs <= 4096
                or type(graph_bytes) is not int or graph_bytes <= 0
                or type(reference) is not bool):
            raise ValueError("native batch graphs need bounded count and explicit memory reservation")
        declared_device = torch.device(stage.device)
        device = next(iter(workspace.values())).device if workspace else declared_device
        if (declared_device.type != device.type
                or (declared_device.index is not None and declared_device.index != device.index)
                or reference != (device.type == "cpu") or device.type not in ("cpu", "cuda")):
            raise ValueError("native batch graphs require CUDA or explicit CPU reference")
        if (getattr(stage, "_dspark_capable", False) or getattr(stage, "_kv_runtime", None) is not None
                or getattr(stage, "_expert_placement", "gpu") != "gpu"
                or getattr(stage, "_spec", False)
                or getattr(stage, "_block_graphs", None) is not None):
            raise ValueError("native batch graphs require independent resident native state")
        if not workspace or any(value.shape[0] != max_jobs or value.device != device
                                for value in workspace.values()):
            raise ValueError("native batch graph workspace must have its fixed request capacity")
        self.stage, self.workspace = stage, workspace
        self.max_jobs, self.max_graphs, self.graph_bytes = max_jobs, max_graphs, graph_bytes
        self.reference, self.device = reference, device
        self.R = _Ref(stage._M)
        self._signatures = {name: tensor_storage_signature(value) for name, value in workspace.items()}
        # Stage creates lane owners before loading checkpoint tensors. Freeze
        # parameters (including versions) and rotary tables at first execution,
        # after ordinary loading, rather than invalidating a legitimate load.
        self._immutable_stamp = None
        self.backup_bytes = sum(value.numel() * value.element_size() for value in workspace.values())
        if self.backup_bytes >= graph_bytes:
            raise ValueError("native batch graph reservation cannot hold the complete state backup")
        self._backup = {name: torch.empty_like(value) for name, value in workspace.items()}
        self._entries, self._refused = {}, {}
        self._used_bytes = self.backup_bytes
        self._capture_peak_bytes = self._allocator_retained_bytes = 0
        self._capture_disabled = False
        self._busy = False
        self._counts = {"captured_graphs": 0, "replays": 0, "reference_eager": 0,
                        "eager_fallbacks": 0, "capture_failures": 0,
                        "budget_refusals": 0, "position_zero_eager": 0,
                        "capture_suppressed": 0}

    def config(self):
        return {"schema": SCHEMA, "max_jobs": self.max_jobs, "max_graphs": self.max_graphs,
                "graph_bytes": self.graph_bytes, "backup_bytes": self.backup_bytes,
                "coverage": COVERAGE, "moe": "eager_real_request_batch",
                "indexer": "eager_native_exact_width",
                "indexer_eager": True,
                "selection": "native_exact_width_topk", "position_keyed": False,
                "reference": self.reference, "graph_qualified": False}

    def status(self):
        return {**self.config(), **self._counts, "live_graphs": 2 * len(self._entries),
                "used_bytes": self._used_bytes, "refused_keys": len(self._refused),
                "capture_peak_bytes": self._capture_peak_bytes,
                "allocator_retained_bytes": self._allocator_retained_bytes,
                "accounted_device_bytes": self.accounted_device_bytes,
                "qualification": "cpu_reference_only" if self.reference else "gpu_acceptance_pending"}

    @property
    def accounted_device_bytes(self):
        return 0 if self.reference else self._used_bytes + self._allocator_retained_bytes

    def tensors(self):
        for name, value in self._backup.items():
            yield f"batch_graph.backup.{name}", value
        for index, entry in enumerate(self._entries.values()):
            for name, value in entry["buffers"].items():
                yield f"batch_graph.{index}.{name}", value

    def _state_bindings(self):
        for index, layer in enumerate(self.stage.layers):
            attn = layer.attn
            yield f"main.{index}.kv", attn.kv_cache
            indexer = getattr(attn, "indexer", None)
            if indexer is not None:
                yield f"main.{index}.index", indexer.kv_cache
            for label, owner in (("attention", attn), ("indexer", indexer)):
                compressor = getattr(owner, "compressor", None)
                if compressor is not None:
                    for field in ("kv_state", "score_state"):
                        yield f"main.{index}.{label}.{field}", getattr(compressor, field)

    def _check(self, h, ids, position):
        if self._busy:
            raise ValueError("native batch graph workspace already owned")
        if (h.ndim != 4 or ids.ndim != 2 or h.shape[:2] != ids.shape or h.shape[1] != 1
                or tuple(h.shape[2:]) != (self.stage.args.hc_mult, self.stage.args.dim)
                or ids.dtype != torch.long
                or h.device != self.device or ids.device != self.device
                or h.dtype != self.stage.dtype or type(position) is not int
                or not 0 <= position < self.stage.args.max_seq_len
                or not 1 <= h.shape[0] <= self.max_jobs
                or self.stage._pos != position):
            raise ValueError("native batch graph inputs differ from the bound token frontier")
        bindings = dict(self._state_bindings())
        if (set(bindings) != set(self.workspace) or any(
                tensor_storage_signature(bindings[name]) != signature
                or tensor_storage_signature(self.workspace[name]) != signature
                for name, signature in self._signatures.items())):
            raise ValueError("native batch graph workspace addresses changed or are not bound")
        for layer in self.stage.layers:
            bind_attention_aliases(layer.attn, rebind=False)
        stamp = (_model_stamp(self.stage, None), repr(self.stage.args))
        if self._immutable_stamp is None:
            self._immutable_stamp = stamp
        elif self._immutable_stamp != stamp:
            raise ValueError("native batch graph immutable weights or rotary tables changed")

    def _snapshot(self):
        for name, value in self.workspace.items():
            self._backup[name].copy_(value)

    def _restore(self):
        for name, value in self.workspace.items():
            value.copy_(self._backup[name])

    def _inputs(self, layer, h, ids, key):
        b = key.requests
        result = {"h": torch.empty_like(h), "ids": torch.empty_like(ids),
                  "position": torch.tensor([key.position], device=self.device, dtype=torch.long),
                  "window": build_win_topk(self.R.M, layer.attn.window_size, key.position).to(self.device).expand(b, -1, -1).clone()}
        if layer.attn.compress_ratio and layer.attn.indexer is None:
            result["compressed"] = build_comp_topk(self.R.M, layer.attn.compress_ratio,
                key.position, layer.attn.window_size, key.context_bucket).to(self.device).expand(b, -1, -1).clone()
        elif getattr(layer.attn, "indexer", None) is not None:
            result["indexer_topk"] = torch.full((b, 1, layer.attn.indexer.index_topk), -1,
                                              dtype=torch.int32, device=self.device)
        return result

    def _feed_indexer(self, layer, h, buffers, key):
        """Execute the real narrow Indexer ONCE, outside every captured region.

        The few pure projections are intentionally duplicated: this preserves
        native scoring/tie order without moving a data-dependent read width
        into a reusable graph. Its selected indices retain their native order.
        """
        if getattr(layer.attn, "indexer", None) is not None:
            from v4_fp8_gemv import tuning_execution
            with tuning_execution("graph"):
                x, _, _ = layer.hc_pre(h, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
                x = layer.attn_norm(x)
                qr = layer.attn.q_norm(layer.attn.wq_a(x))
                selected = layer.attn.indexer(x, qr, key.position, layer.attn.window_size)
            buffers["indexer_topk"].fill_(-1)
            buffers["indexer_topk"][..., :selected.shape[-1]].copy_(selected)

    def _pre(self, layer, buffers, key):
        from v4_fp8_gemv import tuning_execution
        with tuning_execution("graph"):
            return block_pre_cs(self.R, layer, buffers["h"], buffers["position"], buffers["window"],
                buffers.get("compressed"), None, key.compress, key.read_width,
                batch_size=key.requests, native_topk=True,
                indexer_topk=buffers.get("indexer_topk"))

    @staticmethod
    def _post(layer, buffers):
        from v4_fp8_gemv import tuning_execution
        with tuning_execution("graph"):
            return block_post_cs(layer, buffers["moe_out"], buffers["residual"],
                                 buffers["post"], buffers["comb"])

    @staticmethod
    def _moe(layer, x, ids):
        from v4_fp8_gemv import tuning_execution
        with tuning_execution("eager"):
            return layer.ffn(x, ids)

    def _eager_block(self, layer, h, ids, key):
        # Exact-width native topk, not the old whole graph's stable-sort tie policy.
        buffers = self._inputs(layer, h, ids, key)
        buffers["h"].copy_(h); buffers["ids"].copy_(ids)
        self._feed_indexer(layer, h, buffers, key)
        x, residual, post, comb = self._pre(layer, buffers, key)
        from v4_fp8_gemv import tuning_execution
        y = self._moe(layer, x, ids)
        with tuning_execution("graph"):
            return block_post_cs(layer, y, residual, post, comb)

    def _capture(self, layer, h, ids, key):
        # Capture both regions as a transaction. Warmups AND capture execute
        # state writes, so every exit restores the complete canonical workspace.
        self._snapshot()
        before_alloc = torch.cuda.memory_allocated(self.device)
        before_reserved = torch.cuda.memory_reserved(self.device)
        buffers = {}
        side = None
        try:
            buffers = self._inputs(layer, h, ids, key)
            buffers["h"].copy_(h); buffers["ids"].copy_(ids)
            example = self._pre(layer, buffers, key)
            self._restore()
            for label, value in zip(("ffn_in", "residual", "post", "comb"), example):
                buffers[label] = torch.empty_like(value)
            buffers["moe_out"] = torch.empty_like(example[0])
            del example

            def pre():
                for label, value in zip(("ffn_in", "residual", "post", "comb"), self._pre(layer, buffers, key)):
                    buffers[label].copy_(value)

            # JITs and lazy library initialization happen before capture, with
            # fresh state for each iteration. MoE is NEVER inside either graph.
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._restore(); pre()
                    buffers["moe_out"].copy_(self._moe(layer, buffers["ffn_in"], buffers["ids"]))
                    self._post(layer, buffers)
            torch.cuda.current_stream(self.device).wait_stream(side)
            torch.cuda.synchronize(self.device)
            self._restore()
            first = torch.cuda.CUDAGraph()
            # Use a dedicated non-default stream. Direct capture_begin/end
            # also avoids the high-level graph context's global empty_cache(),
            # which must not evict another workload's allocator cache.
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                first.capture_begin()
                try:
                    pre()
                finally:
                    first.capture_end()
            torch.cuda.current_stream(self.device).wait_stream(side)
            self._restore()
            second = torch.cuda.CUDAGraph()
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                second.capture_begin()
                try:
                    out = self._post(layer, buffers)
                finally:
                    second.capture_end()
            torch.cuda.current_stream(self.device).wait_stream(side)
            # Independent pools avoid allocator aliasing of pre's handoffs
            # with post, or with another request-count/position key.
            buffers["output"] = out
            static_bytes = sum(value.numel() * value.element_size() for value in buffers.values())
            allocation = max(0, torch.cuda.memory_allocated(self.device) - before_alloc,
                             torch.cuda.memory_reserved(self.device) - before_reserved, static_bytes)
            self._capture_peak_bytes = max(self._capture_peak_bytes, self._used_bytes + allocation)
            if self._used_bytes + allocation > self.graph_bytes:
                self._capture_disabled = True
                raise MemoryError("native batch graph private pools exceed declared reservation")
            entry = {"pre": first, "post_graph": second, "buffers": buffers, "bytes": allocation}
            self._used_bytes += allocation
            self._counts["captured_graphs"] += 2
            return entry
        finally:
            try:
                if side is not None:
                    # Also join on warmup/capture exceptions. Enqueued KV writes
                    # must finish BEFORE restoring and starting an eager twin.
                    torch.cuda.current_stream(self.device).wait_stream(side)
                self._restore()
            except RuntimeError as error:
                raise UnsafeCaptureState("native batch capture stream could not restore state") from error

    def _block(self, layer, h, ids, key, force_eager):
        if self.reference or force_eager:
            if self.reference:
                self._counts["reference_eager"] += 1
            return self._eager_block(layer, h, ids, key)
        entry = self._entries.get(key)
        if entry is None and self._capture_disabled:
            self._counts["eager_fallbacks"] += 1
            return self._eager_block(layer, h, ids, key)
        if entry is None and key not in self._refused:
            # A finite dictionary is itself part of the quota. Do not remember
            # unlimited rejected positions once the graph count is exhausted.
            if len(self._entries) * 2 + 2 > self.max_graphs:
                self._counts["budget_refusals"] += 1
                self._counts["eager_fallbacks"] += 1
                return self._eager_block(layer, h, ids, key)
            try:
                before_reserved = (torch.cuda.memory_reserved(self.device)
                                   if self.device.type == "cuda" else 0)
                entry = self._capture(layer, h, ids, key)
                self._entries[key] = entry
            except UnsafeCaptureState:
                raise
            except (RuntimeError, MemoryError) as error:
                self._counts["capture_failures"] += 1
                if not self.reference and self.device.type == "cuda":
                    # Failed captures can leave ordinary allocator cache blocks.
                    # Charge their observed retained bytes; never empty another
                    # workload's global allocator to make this look smaller.
                    self._allocator_retained_bytes += max(0,
                        torch.cuda.memory_reserved(self.device) - before_reserved)
                if len(self._refused) < self.max_graphs:
                    self._refused[key] = type(error).__name__
        if entry is None:
            self._counts["eager_fallbacks"] += 1
            return self._eager_block(layer, h, ids, key)
        buffers = entry["buffers"]
        buffers["h"].copy_(h); buffers["ids"].copy_(ids)
        buffers["position"].fill_(key.position)
        buffers["window"].copy_(build_win_topk(self.R.M, layer.attn.window_size,
                                             key.position).to(self.device).expand(key.requests, -1, -1))
        if "compressed" in buffers:
            buffers["compressed"].copy_(build_comp_topk(self.R.M, layer.attn.compress_ratio,
                key.position, layer.attn.window_size, key.context_bucket).to(self.device).expand(key.requests, -1, -1))
        self._feed_indexer(layer, h, buffers, key)
        entry["pre"].replay()
        buffers["moe_out"].copy_(self._moe(layer, buffers["ffn_in"], buffers["ids"]))
        entry["post_graph"].replay()
        self._counts["replays"] += 2
        return buffers["output"]

    def _forward(self, h, ids, position, *, force_eager, capture_suppressed=False):
        self._check(h, ids, position)
        self._busy = True
        try:
            if capture_suppressed:
                self._counts["capture_suppressed"] += 1
            with torch.no_grad(), request_row_math():
                if position == 0:
                    self._counts["position_zero_eager"] += 1
                    return self.stage.forward(h, ids, position)
                for layer in self.stage.layers:
                    key = graph_key(layer, requests=h.shape[0], position=position,
                                    max_jobs=self.max_jobs, horizon=self.stage.args.max_seq_len)
                    profiler = getattr(self.stage, "_runtime_profile", None)
                    with (profiler.interval("main.decode.native_batch_layer", torch, self.device)
                          if profiler else nullcontext()):
                        h = self._block(layer, h, ids, key, force_eager)
                    metrics = getattr(self.stage, "_runtime_metrics", None)
                    if metrics is not None:
                        metrics.resident_routes(h.shape[0] * layer.ffn.n_activated_experts, phase="decode")
                    observer = getattr(self.stage, "_numeric_layer_observer", None)
                    if observer is not None:
                        observer(layer.layer_id, position, h, "native_batch_graph_eager_twin" if force_eager
                                 else "native_batch_graph_or_eager")
                self.stage._pos = position + 1
                metrics = getattr(self.stage, "_runtime_metrics", None)
                if metrics is not None:
                    metrics.sample_kv(self.stage._runtime_kv_tensors())
                return h.clone()
        finally:
            self._busy = False

    def forward(self, h, ids, pos, *, capture_allow=True):
        if type(capture_allow) is not bool:
            raise ValueError("native batch graph capture permission must be an explicit boolean")
        return self._forward(h, ids, pos, force_eager=not capture_allow,
                             capture_suppressed=not capture_allow)

    def eager_twin(self, h, ids, pos):
        return self._forward(h, ids, pos, force_eager=True)

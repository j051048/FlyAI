"""Independent resident V4 state and real equal-frontier native batch calls.

The batch dimension is a request dimension, never a sequence dimension. A
cohort runs one Stage.forward; native attention, routing and dense projections
see [requests, 1, ...]. Unequal frontiers require separate cohorts. Kernels may
round differently from scalar execution, hence the separate numerical contract.
"""
from contextlib import contextmanager
import math
import time

import torch

from v4_conversation_cache import _buffers, _signature
from v4_native_lanes import NativeLanes
from v4_batch_invariant import CONTRACT as MATH_CONTRACT, prepare_native_row_math, request_row_math

BATCH_CONTRACT = "v4-native-small-batch-greedy/2"


class DeviceNativeLanes(NativeLanes):
    contract = BATCH_CONTRACT

    def __init__(self, stage, *, max_jobs, host_bytes, device_bytes, ttl_s=300,
                 reference=False, async_depth=2, prefill_quantum=1, clock=time.monotonic,
                 state_directory=None, disk_bytes=0, graph_mode="off", graph_bytes=0, max_graphs=64):
        if type(max_jobs) is not int or not 2 <= max_jobs <= 4:
            raise ValueError("native batch count must be 2..4")
        if (type(host_bytes) is not int or host_bytes <= 0 or type(device_bytes) is not int
                or device_bytes <= 0 or type(ttl_s) not in (int, float)
                or not math.isfinite(ttl_s) or not 0 < ttl_s <= 3600
                or type(async_depth) is not int or not 1 <= async_depth <= max_jobs
                or type(prefill_quantum) is not int or not 1 <= prefill_quantum <= 8):
            raise ValueError("native batch needs explicit bounded host/device reservations and async depth")
        if type(reference) is not bool or reference != (torch.device(stage.device).type == "cpu"):
            raise ValueError("native device batch requires CUDA, or explicit CPU reference emulation")
        if torch.device(stage.device).type not in ("cpu", "cuda"):
            raise ValueError("native batch supports CUDA or CPU reference only")
        if state_directory or disk_bytes:
            raise ValueError("resident native batches do not yet support durable prefixes")
        if (getattr(stage, "_dspark_capable", False) or getattr(stage, "_kv_runtime", None) is not None
                or getattr(stage, "_expert_placement", "gpu") != "gpu"):
            raise ValueError("native batch requires resident KV/experts and no loaded MTP")
        if getattr(stage, "_graph_mode", "off") != "off" or getattr(stage, "_block_graphs", None) is not None:
            raise ValueError("native batch graphs are independent of scalar whole/island graphs; use V4_CUDA_GRAPH=0")
        if (graph_mode not in ("off", "partial") or type(graph_bytes) is not int or graph_bytes < 0
                or (graph_mode == "off") != (graph_bytes == 0) or type(max_graphs) is not int
                or not 2 <= max_graphs <= 4096):
            raise ValueError("native batch graphs require partial mode with an explicit bounded pool")
        import sys
        slim = sys.modules.get("v4_ref_slim")
        if slim and (getattr(slim, "V4_REF_SLIM", False) or getattr(slim, "V4_REF_SLIM_NOQAT", False)):
            raise ValueError("native batch refuses history-dependent reference bypass")
        self.stage, self.max_jobs, self.clock, self.ttl_s = stage, max_jobs, clock, float(ttl_s)
        self.reference, self.async_depth = reference, async_depth
        prepare_native_row_math(stage._M)
        self.graph_mode, self.graph_bytes, self.max_graphs = graph_mode, graph_bytes, max_graphs
        self.graph_host_bytes = 65536 * max_graphs if graph_mode != "off" else 0
        self.graph_pool = None
        self.prefill_quantum = prefill_quantum
        self.buffers = _buffers(stage, None)
        self.signatures = {name: _signature(value) for name, value in self.buffers.items()}
        if not self.buffers or any(value.dim() < 1 or value.shape[0] < 1 for value in self.buffers.values()):
            raise ValueError("native batch requires an explicit native state batch axis")
        self.tensor_bytes = sum(value[0:1].numel() * value.element_size() for value in self.buffers.values())
        self.metadata_bytes = 65536 + stage.args.max_seq_len * 64
        self.restore_workspace_bytes = 0
        self.hash_workspace_bytes = 2 << 20
        self.per_job_bytes = self.tensor_bytes + self.metadata_bytes
        self.batch_workspace_bytes = self.tensor_bytes * max_jobs
        self.device_reserved_bytes = self.tensor_bytes * max_jobs + self.batch_workspace_bytes + graph_bytes
        # On a reference CPU run these are real CPU tensors, but the explicit
        # emulated device quota is still enforced identically before allocation.
        self.reserved_bytes = (self.metadata_bytes * max_jobs + self.hash_workspace_bytes + self.graph_host_bytes
                               + (self.device_reserved_bytes if reference else 0))
        if self.device_reserved_bytes > device_bytes or self.reserved_bytes > host_bytes:
            raise ValueError("native batch states/workspace exceed declared device or host reservation")
        self.workspace = {name: torch.empty((max_jobs, *value.shape[1:]), dtype=value.dtype, device=value.device)
                          for name, value in self.buffers.items()}
        self.slots = [{"buffers": {name: torch.empty_like(value[0:1]) for name, value in self.buffers.items()},
                       "job": None, "cache": None, "pos": 0} for _ in range(max_jobs)]
        self.active, self.owner_epoch, self.current = {}, None, None
        self.model_stamp, self.disk_identity, self.store = None, None, None
        self.last_cache_write_error = None
        self._batch_active = False
        self._batch_fault = False
        self.counters = {"batch_forward_calls": 0, "batched_request_rows": 0,
                         "scalar_forward_calls": 0, "state_host_copy_bytes": 0}
        if graph_mode == "partial":
            from v4_native_batch_graph import NativeBatchGraphPool
            self.graph_pool = NativeBatchGraphPool(stage, self.workspace, max_jobs=max_jobs,
                max_graphs=max_graphs, graph_bytes=graph_bytes, reference=reference)

    def config(self):
        return {"enabled": True, "numerical_contract": self.contract, "max_jobs": self.max_jobs,
                "horizon": self.stage.args.max_seq_len, "per_job_state_bytes": self.per_job_bytes,
                "reserved_state_bytes": self.per_job_bytes * self.max_jobs + self.batch_workspace_bytes + self.hash_workspace_bytes
                    + self.graph_bytes + self.graph_host_bytes,
                "tensor_bytes_per_job": self.tensor_bytes, "metadata_bytes_per_job": self.metadata_bytes,
                "prefill_chunk_tokens": self.prefill_quantum, "restore_workspace_bytes": 0,
                "durable_prefix": {"enabled": False, "max_bytes": 0}, "ttl_s": self.ttl_s,
                "state_tier": "cpu_reference" if self.reference else "device",
                "compute": "native_equal_frontier_batch", "host_reserved_bytes": self.reserved_bytes,
                "device_reserved_bytes": self.device_reserved_bytes,
                "batch_workspace_bytes": self.batch_workspace_bytes, "async_depth": self.async_depth,
                "reference": self.reference, "graph_qualified": False, "hash_workspace_bytes": self.hash_workspace_bytes,
                "math_contract": MATH_CONTRACT,
                "graph_policy": {"mode": self.graph_mode, "graph_bytes": self.graph_bytes,
                    "host_metadata_bytes": self.graph_host_bytes, "max_graphs": self.max_graphs if self.graph_pool else 0,
                    "coverage": "attention_dense_partial" if self.graph_pool else "off",
                    "selection": "native_exact_width_topk"}}

    def graph_status(self):
        return self.graph_pool.status() if self.graph_pool is not None else {"enabled": False,
            "captured_graphs": 0, "replays": 0, "graph_qualified": False}

    def logits_all(self, hidden, full_logits=True):
        with request_row_math():
            return self.stage.logits_all(hidden, full_logits=full_logits)

    def _check(self):
        if self._batch_fault:
            raise RuntimeError("native batch state restoration/commit failed; restart the owned stage before reuse")
        super()._check()

    def tensors(self):
        yield from super().tensors()
        for name, value in self.workspace.items():
            yield f"batch_workspace.{name}", value
        if self.graph_pool is not None:
            yield from self.graph_pool.tensors()

    def _save(self, slot):
        self._check()
        with torch.no_grad():
            for name, value in self.buffers.items():
                slot["buffers"][name].copy_(value[0:1])
        slot["pos"] = self.stage._pos

    def activate(self, job, pos):
        self._check()
        slot = self.active.get(job)
        if slot is None or type(pos) is not int or pos != slot["pos"] or pos >= self.stage.args.max_seq_len:
            raise ValueError("native frame does not match its live lane frontier")
        with torch.no_grad():
            for name, value in self.buffers.items():
                value[0:1].copy_(slot["buffers"][name])
        self.stage._pos, self.current = pos, job
        return slot

    def _bindings(self):
        for index, layer in enumerate(self.stage.layers):
            attn = layer.attn
            yield f"main.{index}.kv", attn, "kv_cache"
            indexer = getattr(attn, "indexer", None)
            if indexer is not None:
                yield f"main.{index}.index", indexer, "kv_cache"
            for label, owner in (("attention", attn), ("indexer", indexer)):
                compressor = getattr(owner, "compressor", None)
                if compressor is not None:
                    for field in ("kv_state", "score_state"):
                        yield f"main.{index}.{label}.{field}", compressor, field

    @contextmanager
    def _bind_batch(self, rows, pos):
        self._check()
        if self._batch_active:
            raise ValueError("native batch workspace already owned")
        self._batch_active = True
        original_pos = self.stage._pos
        bindings = list(self._bindings())
        try:
            with torch.no_grad():
                for name, value in self.workspace.items():
                    for index, slot in enumerate(rows):
                        value[index:index + 1].copy_(slot["buffers"][name])
            for name, owner, field in bindings:
                setattr(owner, field, self.workspace[name])
            from v4_runtime_init import bind_attention_aliases
            for layer in self.stage.layers:
                bind_attention_aliases(layer.attn, rebind=True)
            self.stage._pos = pos
            yield
        finally:
            try:
                for name, owner, field in bindings:
                    setattr(owner, field, self.buffers[name])
                from v4_runtime_init import bind_attention_aliases
                for layer in self.stage.layers:
                    bind_attention_aliases(layer.attn, rebind=True)
            except Exception:
                # A partly restored alias set cannot be admitted in a new epoch.
                self._batch_fault = True
                raise
            finally:
                self.stage._pos = original_pos
                self._batch_active = False

    def forward_batch(self, jobs, h, ids, pos, *, capture_graph=True):
        """One actual native call, with all validation before state mutation.

        Canonical states commit only after the complete local forward succeeds.
        A failing workspace is discarded, preserving every admitted frontier.
        """
        self._check()
        if (not isinstance(jobs, (list, tuple)) or not 1 <= len(jobs) <= self.max_jobs
                or len(set(jobs)) != len(jobs) or type(pos) is not int
                or not 0 <= pos < self.stage.args.max_seq_len or type(capture_graph) is not bool
                or tuple(ids.shape) != (len(jobs), 1) or h.shape[:2] != ids.shape
                or ids.dtype != torch.long or ids.device != h.device
                or h.device != next(iter(self.buffers.values())).device):
            raise ValueError("native batch requires unique equal-frontier one-token request rows")
        rows = [self.active.get(job) for job in jobs]
        if any(slot is None or slot["pos"] != pos for slot in rows):
            raise ValueError("native batch frame differs from live independent frontiers")
        with torch.no_grad():
            with self._bind_batch(rows, pos), request_row_math():
                if self.graph_pool is None:
                    out = self.stage.forward(h, ids, pos)
                else:
                    from v4_native_batch_graph import UnsafeCaptureState
                    try:
                        out = self.graph_pool.forward(h, ids, pos, capture_allow=capture_graph)
                    except UnsafeCaptureState:
                        self._batch_fault = True
                        raise
                if self.stage._pos != pos + 1:
                    raise ValueError("native batched forward did not advance exactly one token")
                if self.graph_pool is not None and self.graph_pool.accounted_device_bytes > self.graph_bytes:
                    self._batch_fault = True
                    raise RuntimeError("native batch graph allocation exceeded the calibrated reservation")
            # Commit only after the original buffer bindings AND aliases have
            # been restored. Device copy failures poison this owner; a partial
            # commit must never be reused by another job or connection epoch.
            try:
                for index, slot in enumerate(rows):
                    for name, value in self.workspace.items():
                        slot["buffers"][name].copy_(value[index:index + 1])
            except Exception:
                self._batch_fault = True
                raise
            for slot in rows:
                slot["pos"] = pos + 1
        key = "batch_forward_calls" if len(jobs) > 1 else "scalar_forward_calls"
        self.counters[key] += 1
        if len(jobs) > 1:
            self.counters["batched_request_rows"] += len(jobs)
        return out

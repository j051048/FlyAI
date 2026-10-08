"""V4 compressed-history offload with one bounded, layer-local device workspace.

This is not an ordinary K/V sliding window. The reference's ring and both
Compressor recurrences remain resident. Its compressed Attention and Indexer
histories have canonical host storage. While a layer executes, its entire valid
compressed prefix is materialized in shared device storage and the original
reference methods run unchanged. Ratio-4 indexing and non-indexed compression
still read their full prefix: a quota therefore bounds supported context, not
attention semantics. Capacity is checked for the whole frame before mutation.

Transfers use the consumer CUDA stream and pinned host tensors. The D2H write
and subsequent H2D read are ordered on that stream; changing streams inserts an
event dependency. There is no per-token device synchronization. A host reader
or reset explicitly fences the preceding writes. CPU emulation is test-only.
"""
from contextlib import contextmanager
from dataclasses import replace
import threading

import torch


def _count(value, name, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def tiered_block_cls(base_cls, model):
    """Avoid constructing a full device KV bank, including a transient copy.

    Only Attention receives max_seq_len=0 during buffer construction. Its real
    rotary table is then initialized with the original arguments. Parameter
    names, shapes, Compressor state and all forward arithmetic are unchanged.
    This class can itself be wrapped by hybrid_block_cls's META construction.
    """
    attention_cls = base_cls.attention_cls

    class TieredAttention(attention_cls):
        def __init__(self, layer_id, args):
            super().__init__(layer_id, replace(args, max_seq_len=0))
            original, theta = ((args.original_seq_len, args.compress_rope_theta)
                               if self.compress_ratio else (0, args.rope_theta))
            self.freqs_cis = model.precompute_freqs_cis(
                args.rope_head_dim, args.max_seq_len, original, theta,
                args.rope_factor, args.beta_fast, args.beta_slow)

    class TieredBlock(base_cls):
        attention_cls = TieredAttention

    return TieredBlock


class V4KVRuntime:
    """One stage's explicitly budgeted compressed-history storage seam.

    gpu_budget_bytes includes main rings/Compressor state, one workspace, and a
    declared DSpark ring reserve. Rollback KV snapshots are on the host and
    their worst-case bytes are included in host_budget_bytes. It excludes RoPE,
    activations, attention score/output temporaries and non-KV model weights;
    resource calibration must budget those separately.
    """
    def __init__(self, layers, args, *, device, dtype, gpu_budget_bytes,
                 host_budget_bytes, spec_depth=0, draft_layers=0, reference=False,
                 prefill_gate=False):
        _count(gpu_budget_bytes, "kv_gpu_budget_bytes", positive=True)
        _count(host_budget_bytes, "kv_host_budget_bytes", positive=True)
        _count(spec_depth, "spec_depth")
        _count(draft_layers, "draft_layers")
        if type(reference) is not bool:
            raise ValueError("kv_reference must be a boolean")
        self.device, self.dtype, self.reference = torch.device(device), dtype, reference
        if reference != (self.device.type == "cpu"):
            raise ValueError("tiered KV needs CUDA, or explicit CPU reference emulation")
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("tiered KV supports CUDA or CPU reference emulation")
        self.layers, self.args = tuple(layers), args
        self.gpu_budget_bytes, self.host_budget_bytes = gpu_budget_bytes, host_budget_bytes
        self._lock = threading.RLock()
        self._active, self._event, self._stream_id = None, None, None
        element = torch.empty((), dtype=dtype, device="cpu").element_size()
        self.element_size = element
        self.fixed_bytes = 0
        self.host_history_bytes = 0
        self.entries = {}
        for layer in self.layers:
            attn = layer.attn
            if tuple(attn.kv_cache.shape) != (args.max_batch_size, args.window_size, args.head_dim):
                raise ValueError("tiered KV must be selected before full device history construction")
            self.fixed_bytes += attn.kv_cache.numel() * element
            for owner in (attn, getattr(attn, "indexer", None)):
                compressor = getattr(owner, "compressor", None)
                if compressor is not None:
                    self.fixed_bytes += sum(getattr(compressor, name).numel() * 4
                                            for name in ("kv_state", "score_state"))
            ratio = attn.compress_ratio
            if ratio:
                rows = args.max_seq_len // ratio
                self.host_history_bytes += args.max_batch_size * rows * attn.head_dim * element
                indexer = getattr(attn, "indexer", None)
                if indexer is not None:
                    self.host_history_bytes += args.max_batch_size * rows * indexer.head_dim * element
        self.draft_reserve_bytes = draft_layers * args.max_batch_size * args.window_size * args.head_dim * element
        # Python builds the incoming snapshot before deque.append evicts its
        # oldest entry, so the transient bound is W+1, even for maxlen=0.
        self.snapshot_reserve_bytes = (spec_depth + 1) * self.fixed_bytes
        maximum = max((self._elements(layer.attn, args.max_seq_len) * element
                       for layer in self.layers if layer.attn.compress_ratio), default=0)
        largest_state = max((sum(t.numel() * t.element_size()
                                for owner in (layer.attn, getattr(layer.attn, "indexer", None))
                                for compressor in (getattr(owner, "compressor", None),)
                                if compressor is not None
                                for t in (compressor.kv_state, compressor.score_state))
                             for layer in self.layers), default=0)
        # Before/after gate snapshots and one comparison readback. These are
        # temporary host tensors, so include their bound even before a job.
        gate_kv = max((max(self._elements(layer.attn, args.max_seq_len) * element,
                          layer.attn.kv_cache.numel() * element) for layer in self.layers), default=0)
        self.gate_host_reserve_bytes = 3 * (gate_kv + largest_state) if prefill_gate else 0
        required_host = self.host_history_bytes + self.snapshot_reserve_bytes + self.gate_host_reserve_bytes
        if host_budget_bytes < required_host:
            raise ValueError(f"KV host quota needs at least {required_host} bytes including rollback reserve")
        available = gpu_budget_bytes - self.fixed_bytes - self.draft_reserve_bytes
        # Every indexed attention needs a contiguous [ring, compressed] operand.
        minimum = max((args.max_batch_size * args.window_size * layer.attn.head_dim * element
                       for layer in self.layers if layer.attn.compress_ratio), default=0)
        if available < minimum:
            raise ValueError(f"KV GPU quota needs resident state plus at least {minimum} workspace bytes")
        # Avoid allocating a caller's oversized quota when the model needs less.
        self.workspace_bytes = min(available // element * element, maximum)
        if not reference:
            free, _ = torch.cuda.mem_get_info(self.device)
            if self.workspace_bytes > free:
                raise RuntimeError("KV workspace exceeds measured free VRAM")
        self.workspace = torch.empty(self.workspace_bytes // element, dtype=dtype, device=self.device)
        for layer in self.layers:
            attn, ratio = layer.attn, layer.attn.compress_ratio
            attn._kv_runtime_owner = self
            if not ratio:
                continue
            rows = args.max_seq_len // ratio
            history = torch.zeros((args.max_batch_size, rows, attn.head_dim), dtype=dtype,
                                  device="cpu", pin_memory=not reference)
            indexer = getattr(attn, "indexer", None)
            index_history = (torch.zeros((args.max_batch_size, rows, indexer.head_dim), dtype=dtype,
                                         device="cpu", pin_memory=not reference)
                             if indexer is not None else None)
            self.entries[id(attn)] = {"attention": attn, "ring": attn.kv_cache,
                                     "history": history, "index_history": index_history}
        self.reset_stats()

    def _elements(self, attn, end_pos):
        ratio = attn.compress_ratio
        if not ratio:
            return 0
        rows = end_pos // ratio
        size = self.args.max_batch_size * (attn.window_size + rows) * attn.head_dim
        if getattr(attn, "indexer", None) is not None:
            size += self.args.max_batch_size * rows * attn.indexer.head_dim
        return size

    def validate_frame(self, start_pos, tokens, batch):
        """Run before seek, checkpoints, expert acquisition or KV writes."""
        _count(start_pos, "start_pos")
        _count(tokens, "tokens", positive=True)
        _count(batch, "batch", positive=True)
        end = start_pos + tokens
        if batch > self.args.max_batch_size or end > self.args.max_seq_len:
            raise ValueError("KV frame exceeds configured batch/context capacity")
        need = max((self._elements(layer.attn, end) * self.element_size
                    for layer in self.layers), default=0)
        if need > self.workspace_bytes:
            raise ValueError(f"KV active prefix needs {need} workspace bytes, quota has {self.workspace_bytes}; "
                             "V4's Indexer/compressed attention requires its full valid prefix")

    def _ordered_stream(self):
        if self.reference:
            return None
        stream = torch.cuda.current_stream(self.device)
        identity = stream.cuda_stream
        if self._event is not None and identity != self._stream_id:
            stream.wait_event(self._event)
        self._stream_id = identity
        return stream

    def synchronize_host(self):
        """Explicit barrier only for reset, diagnostics and numerical gates."""
        if self._event is not None:
            self._event.synchronize()

    def order_current_stream(self):
        """Order seek/snapshot writes too, before the next layer is entered."""
        with self._lock:
            self._ordered_stream()

    def _record_current_stream(self, stream):
        if stream is not None:
            self._event = self._event or torch.cuda.Event()
            self._event.record(stream)

    def snapshot_tensor(self, value):
        if self.reference:
            return value.clone()
        saved = torch.empty_like(value, device="cpu", pin_memory=True)
        saved.copy_(value, non_blocking=True)
        self.d2h_bytes += value.numel() * value.element_size()
        self._record_current_stream(torch.cuda.current_stream(self.device))
        return saved

    def restore_tensor(self, target, saved):
        target.copy_(saved, non_blocking=not self.reference)
        if not self.reference:
            self.h2d_bytes += saved.numel() * saved.element_size()
            self._record_current_stream(torch.cuda.current_stream(self.device))

    def readback_tensor(self, value):
        saved = self.snapshot_tensor(value)
        self.synchronize_host()
        return saved

    def _copy_host_rows(self, target, source, batch):
        # Prefix views across batch have different row strides in archive and
        # workspace. Copy each contiguous batch row directly; otherwise a
        # generic cross-device copy may pack an extra device/host temporary.
        for row in range(batch):
            target[row].copy_(source[row], non_blocking=not self.reference)

    @contextmanager
    def layer(self, attn, start_pos, tokens, batch=None):
        batch = self.args.max_batch_size if batch is None else batch
        if not attn.compress_ratio:
            with self._lock:
                if self._active is not None:
                    raise RuntimeError("tiered KV workspace cannot have concurrent active layers")
                stream = self._ordered_stream()
                self._active = id(attn)
                succeeded = False
                try:
                    yield
                    succeeded = True
                finally:
                    self._record_current_stream(stream)
                    self._active = None
                    self.layer_calls += int(succeeded)
            return
        with self._lock:
            if self._active is not None:
                raise RuntimeError("tiered KV workspace cannot have concurrent active layers")
            entry = self.entries[id(attn)]
            stream = self._ordered_stream()
            rows = (start_pos + tokens) // attn.compress_ratio
            count = self.args.max_batch_size * (attn.window_size + rows) * attn.head_dim
            kv = self.workspace[:count].view(self.args.max_batch_size, attn.window_size + rows, attn.head_dim)
            kv[:, :attn.window_size].copy_(entry["ring"])
            if rows:
                self._copy_host_rows(kv[:, attn.window_size:], entry["history"][:, :rows], batch)
                self.h2d_bytes += batch * rows * attn.head_dim * self.element_size if not self.reference else 0
            attn.kv_cache = kv
            attn.compressor.kv_cache = kv[:, attn.window_size:]
            attn.compressor.freqs_cis = attn.freqs_cis
            indexer = getattr(attn, "indexer", None)
            if indexer is not None:
                index = self.workspace[count:count + self.args.max_batch_size * rows * indexer.head_dim].view(
                    self.args.max_batch_size, rows, indexer.head_dim)
                if rows:
                    self._copy_host_rows(index, entry["index_history"][:, :rows], batch)
                    self.h2d_bytes += batch * rows * indexer.head_dim * self.element_size if not self.reference else 0
                indexer.kv_cache = index
                indexer.freqs_cis = attn.freqs_cis
                indexer.compressor.kv_cache = index
                indexer.compressor.freqs_cis = attn.freqs_cis
            self._active = id(attn)
            succeeded = False
            try:
                yield
                succeeded = True
            finally:
                # Retain actual completed writes even on an exception; the job
                # must reset after a model failure, as with resident execution.
                entry["ring"].copy_(kv[:, :attn.window_size])
                first = 0 if start_pos == 0 else start_pos // attn.compress_ratio
                if rows > first:
                    self._copy_host_rows(entry["history"][:, first:rows],
                                         kv[:, attn.window_size + first:attn.window_size + rows], batch)
                    amount = batch * (rows - first) * attn.head_dim * self.element_size
                    if indexer is not None:
                        self._copy_host_rows(entry["index_history"][:, first:rows], index[:, first:rows], batch)
                        amount += batch * (rows - first) * indexer.head_dim * self.element_size
                    self.d2h_bytes += amount if not self.reference else 0
                attn.kv_cache = entry["ring"]
                attn.compressor.kv_cache = None
                if indexer is not None:
                    # A zero-sized view allocates no separate device history.
                    indexer.kv_cache = self.workspace[:0].view(self.args.max_batch_size, 0, indexer.head_dim)
                    indexer.compressor.kv_cache = None
                self._record_current_stream(stream)
                self._active = None
                self.layer_calls += int(succeeded)

    def reset_stats(self):
        self.h2d_bytes = self.d2h_bytes = self.layer_calls = 0

    def reset(self):
        with self._lock:
            if self._active is not None:
                raise RuntimeError("cannot reset an active KV workspace")
            self.synchronize_host()
            for entry in self.entries.values():
                attn = entry["attention"]
                attn.kv_cache = entry["ring"]
                entry["ring"].zero_()
                attn.compressor.kv_cache = None
                indexer = getattr(attn, "indexer", None)
                if indexer is not None:
                    indexer.kv_cache = self.workspace[:0].view(self.args.max_batch_size, 0, indexer.head_dim)
                    indexer.compressor.kv_cache = None
                entry["history"].zero_()
                if entry["index_history"] is not None:
                    entry["index_history"].zero_()
            self.reset_stats()

    def tensors(self):
        """Actual owned storage; callers deduplicate views and module aliases."""
        yield "workspace", self.workspace
        for i, entry in enumerate(self.entries.values()):
            for name in ("ring", "history", "index_history"):
                yield f"layer_{i}.{name}", entry[name]

    def status(self):
        lo, hi = 0, self.args.max_seq_len
        while lo < hi:
            end = (lo + hi + 1) // 2
            if max((self._elements(layer.attn, end) * self.element_size for layer in self.layers), default=0) > self.workspace_bytes:
                hi = end - 1
            else:
                lo = end
        return {"mode": "reference_cpu" if self.reference else "layer_working_set",
                "gpu_budget_bytes": self.gpu_budget_bytes, "host_budget_bytes": self.host_budget_bytes,
                "resident_main_state_bytes": self.fixed_bytes, "draft_reserve_bytes": self.draft_reserve_bytes,
                "workspace_bytes": self.workspace_bytes, "host_history_bytes": self.host_history_bytes,
                "rollback_host_reserve_bytes": self.snapshot_reserve_bytes,
                "gate_host_reserve_bytes": self.gate_host_reserve_bytes,
                "max_supported_tokens": lo, "layer_calls": self.layer_calls,
                "h2d_bytes": self.h2d_bytes, "d2h_bytes": self.d2h_bytes,
                "scope": "KV storage only; excludes RoPE, activations and kernel temporaries"}

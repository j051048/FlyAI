"""Optional, signed observations of executed work and local memory residency.

No torch import: receipt verifiers can validate this schema without loading an engine.
Routes count token/expert entries, including rejected speculation and rollback replay,
not unique experts or committed output tokens. Every path uses that same denominator.
The V4 resident engine records geometry on the host; it never copies router tensors to
the CPU or changes a captured graph to obtain a counter.
"""
from copy import deepcopy
from contextlib import contextmanager
import math
import threading
import weakref

SCHEMA = "shard-runtime-metrics/1"
ROLES = ("main", "draft")
PHASES = ("prefill", "decode", "replay")
COUNTS = ("routed_entries", "resident_hits", "dma_misses", "dma_bytes",
          "dma_wait_ms", "cpu_misses", "reference_routes")
KV_FIELDS = ("gpu_bytes", "host_bytes", "gpu_peak_bytes", "host_peak_bytes")
GPU_FIELDS = ("scope", "allocated_bytes", "reserved_bytes", "allocated_peak_bytes",
              "reserved_peak_bytes")
GPU_MODES = ("gpu_resident", "gpu_expert_cache")
_GPU_OBSERVERS = {}
_EXTERNAL_ALLOCATOR_INTERVALS = {}
_ALLOCATOR_INTERVAL_LOCK = threading.RLock()


@contextmanager
def external_allocator_interval(device, *, torch_module=None):
    """Let a calibration owner retain load+forward peaks without job resets.

    This does not reset CUDA statistics itself. During this interval per-job
    metrics omit allocator observations rather than mislabel calibration peaks.
    A later normal job reset starts the normal exclusive observation interval.
    """
    key = str(device)
    if key == "cuda" and torch_module is not None:
        key = f"cuda:{torch_module.cuda.current_device()}"
    if not key.startswith("cuda:") or not key.partition(":")[2].isdigit():
        raise ValueError("allocator interval needs an explicit CUDA device")
    with _ALLOCATOR_INTERVAL_LOCK:
        _EXTERNAL_ALLOCATOR_INTERVALS[key] = _EXTERNAL_ALLOCATOR_INTERVALS.get(key, 0) + 1
        for observer in _GPU_OBSERVERS.get(key, ()):
            observer._gpu_peak_exclusive = False
            observer.gpu_memory = None
    try:
        yield
    finally:
        with _ALLOCATOR_INTERVAL_LOCK:
            remaining = _EXTERNAL_ALLOCATOR_INTERVALS[key] - 1
            if remaining:
                _EXTERNAL_ALLOCATOR_INTERVALS[key] = remaining
            else:
                del _EXTERNAL_ALLOCATOR_INTERVALS[key]


def _counts():
    return {key: 0.0 if key == "dma_wait_ms" else 0 for key in COUNTS}


def _keys(value, keys, where):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{where}: expected keys {sorted(keys)}")


def _nonnegative(value, where, *, real=False):
    if isinstance(value, bool) or not isinstance(value, (int, float) if real else int):
        raise ValueError(f"{where}: expected a nonnegative {'number' if real else 'integer'}")
    if value < 0 or (real and not math.isfinite(value)):
        raise ValueError(f"{where}: expected a finite nonnegative value")


def validate_runtime_metrics(value):
    """Validate and detach a JSON-only snapshot. Unknown versions fail closed."""
    keys = {"schema", "mode", "work", "totals", "resident_hit_rate", "kv"}
    if isinstance(value, dict) and "gpu_memory" in value:
        keys.add("gpu_memory")
    if isinstance(value, dict) and "expert_prefetch" in value:
        keys.add("expert_prefetch")
    if isinstance(value, dict) and "expert_cache" in value:
        keys.add("expert_cache")
    if isinstance(value, dict) and "performance" in value:
        keys.add("performance")
    if isinstance(value, dict) and "prefetch_policy" in value:
        keys.add("prefetch_policy")
    if isinstance(value, dict) and "kernel_coverage" in value:
        keys.add("kernel_coverage")
    for optional in ("kv_policy", "prefill_policy"):
        if isinstance(value, dict) and optional in value:
            keys.add(optional)
    _keys(value, keys, "runtime_metrics")
    if value["schema"] != SCHEMA:
        raise ValueError(f"unknown runtime metrics schema {value['schema']!r}")
    if value["mode"] not in (*GPU_MODES, "reference_cpu"):
        raise ValueError(f"unknown runtime residency mode {value['mode']!r}")
    _keys(value["work"], ROLES, "work")
    total = _counts()
    for role in ROLES:
        _keys(value["work"][role], PHASES, f"work.{role}")
        for phase in PHASES:
            c = value["work"][role][phase]
            _keys(c, COUNTS, f"work.{role}.{phase}")
            for key in COUNTS:
                _nonnegative(c[key], key, real=key == "dma_wait_ms")
                total[key] += c[key]
            if c["routed_entries"] != sum(c[k] for k in
                    ("resident_hits", "dma_misses", "cpu_misses", "reference_routes")):
                raise ValueError("expert route counts do not share one denominator")
            if value["mode"] in GPU_MODES and c["reference_routes"]:
                raise ValueError("GPU residency cannot claim unclassified CPU reference routes")
            if value["mode"] == "reference_cpu" and any(c[k] for k in
                    ("resident_hits", "dma_misses", "dma_bytes", "dma_wait_ms", "cpu_misses")):
                raise ValueError("CPU reference execution is not a GPU cache or CPU fallback")
            if not c["dma_misses"] and (c["dma_bytes"] or c["dma_wait_ms"]):
                raise ValueError("DMA bytes/wait without a DMA miss")
    _keys(value["totals"], COUNTS, "totals")
    for key in COUNTS:
        _nonnegative(value["totals"][key], f"totals.{key}", real=key == "dma_wait_ms")
    if value["totals"] != total:
        raise ValueError("runtime totals disagree with phase counts")
    rate = total["resident_hits"] / total["routed_entries"] if total["routed_entries"] else None
    if value["resident_hit_rate"] != rate or isinstance(value["resident_hit_rate"], bool):
        raise ValueError("resident_hit_rate disagrees with the total routed entries")
    _keys(value["kv"], KV_FIELDS, "kv")
    for key in KV_FIELDS:
        _nonnegative(value["kv"][key], f"kv.{key}")
    for tier in ("gpu", "host"):
        if value["kv"][f"{tier}_peak_bytes"] < value["kv"][f"{tier}_bytes"]:
            raise ValueError("KV peak is smaller than current allocation")
    if "gpu_memory" in value:
        if value["mode"] not in GPU_MODES:
            raise ValueError("CPU reference metrics cannot claim CUDA allocator observations")
        g = value["gpu_memory"]
        _keys(g, GPU_FIELDS, "gpu_memory")
        if g["scope"] != "process":
            raise ValueError("GPU allocator observations must name their process scope")
        for key in GPU_FIELDS[1:]:
            _nonnegative(g[key], f"gpu_memory.{key}")
        if g["allocated_peak_bytes"] < g["allocated_bytes"] or \
                g["reserved_peak_bytes"] < g["reserved_bytes"] or \
                g["allocated_bytes"] > g["reserved_bytes"]:
            raise ValueError("inconsistent GPU allocator observations")
    if "expert_prefetch" in value:
        p = value["expert_prefetch"]
        _keys(p, ("dma_bytes", "dma_wait_ms"), "expert_prefetch")
        _nonnegative(p["dma_bytes"], "expert_prefetch.dma_bytes")
        _nonnegative(p["dma_wait_ms"], "expert_prefetch.dma_wait_ms", real=True)
        if value["mode"] == "reference_cpu" and any(p.values()):
            raise ValueError("CPU reference execution cannot claim prefetched DMA")
    if "expert_cache" in value:
        cache = value["expert_cache"]
        _keys(cache, ("device", "cache_bytes", "host_experts_bytes", "pinned_host_experts_bytes",
                      "slots_per_pool"), "expert_cache")
        if not isinstance(cache["device"], str) or not (
                cache["device"].startswith("cuda") if value["mode"] in GPU_MODES else cache["device"] == "cpu"):
            raise ValueError("expert cache device disagrees with runtime execution")
        for key in ("cache_bytes", "host_experts_bytes", "pinned_host_experts_bytes"):
            _nonnegative(cache[key], f"expert_cache.{key}")
        if cache["pinned_host_experts_bytes"] > cache["host_experts_bytes"]:
            raise ValueError("pinned expert bytes exceed the host pool")
        if value["mode"] == "gpu_expert_cache" and cache["pinned_host_experts_bytes"] != cache["host_experts_bytes"]:
            raise ValueError("GPU expert caching requires a fully pinned canonical host pool")
        slots = cache["slots_per_pool"]
        if not isinstance(slots, dict) or any(not isinstance(k, str) or type(v) is not int or v <= 0
                                             for k, v in slots.items()):
            raise ValueError("expert cache slots must name positive per-pool capacities")
    if "performance" in value:
        try:
            from runtime_profile import validate_profile
        except ImportError:
            from shard.runtime_profile import validate_profile
        validate_profile(value["performance"])
        if value["mode"] == "reference_cpu" and any(
                name.endswith(".gpu") for name in value["performance"]["phases"]):
            raise ValueError("CPU reference execution cannot claim GPU phase timings")
    if "prefetch_policy" in value:
        p = value["prefetch_policy"]
        _keys(p, ("requested", "used", "wasted", "skipped", "candidate_count"), "prefetch_policy")
        for key, count in p.items():
            _nonnegative(count, f"prefetch_policy.{key}")
        if p["used"] + p["wasted"] > p["requested"]:
            raise ValueError("prefetch outcomes exceed requested predictions")
    if "kernel_coverage" in value:
        k = value["kernel_coverage"]
        _keys(k, ("scope", "counters"), "kernel_coverage")
        if k["scope"] != "job_python_observed":
            raise ValueError("invalid kernel observation scope")
        expected = ("main_grouped_calls", "main_cache_generic_calls", "main_grouped_declines",
                    "draft_grouped_calls", "draft_grouped_declines", "draft_cuda_grouped_gemms",
                    "draft_cache_grouped_calls", "draft_cache_generic_calls",
                    "draft_shared_calls", "draft_shared_cuda_calls", "draft_shared_declines")
        _keys(k["counters"], expected, "kernel_coverage.counters")
        for key, count in k["counters"].items():
            _nonnegative(count, f"kernel_coverage.{key}")
        if value["mode"] == "reference_cpu" and any(
                count for key, count in k["counters"].items() if "cuda" in key):
            raise ValueError("CPU reference execution cannot claim CUDA kernel calls")
    if "kv_policy" in value:
        policy = value["kv_policy"]
        if not isinstance(policy, dict) or policy.get("mode") not in (
                "gpu_resident", "layer_working_set", "reference_cpu"):
            raise ValueError("invalid KV policy mode")
        if policy["mode"] == "gpu_resident":
            _keys(policy, ("mode",), "kv_policy")
        else:
            counts = ("gpu_budget_bytes", "host_budget_bytes", "resident_main_state_bytes",
                      "draft_reserve_bytes", "workspace_bytes", "host_history_bytes",
                      "rollback_host_reserve_bytes", "gate_host_reserve_bytes", "max_supported_tokens",
                      "layer_calls", "h2d_bytes", "d2h_bytes")
            _keys(policy, ("mode", "scope", *counts), "kv_policy")
            for key in counts:
                _nonnegative(policy[key], f"kv_policy.{key}")
            if policy["scope"] != "KV storage only; excludes RoPE, activations and kernel temporaries":
                raise ValueError("invalid KV storage scope")
            if policy["mode"] == "reference_cpu" and (policy["h2d_bytes"] or policy["d2h_bytes"]):
                raise ValueError("CPU KV policy cannot claim DMA")
            if value["mode"] == "reference_cpu" and policy["mode"] == "layer_working_set":
                raise ValueError("CPU execution cannot claim GPU KV tiering")
            if (policy["resident_main_state_bytes"] + policy["draft_reserve_bytes"] + policy["workspace_bytes"] >
                    policy["gpu_budget_bytes"] or policy["host_history_bytes"] +
                    policy["rollback_host_reserve_bytes"] + policy["gate_host_reserve_bytes"] > policy["host_budget_bytes"]):
                raise ValueError("KV policy exceeds declared tier budgets")
    if "prefill_policy" in value:
        policy = value["prefill_policy"]
        counts = ("query_chunk_tokens", "gate_checks", "executed_query_chunks")
        _keys(policy, ("mode", "scope", *counts), "prefill_policy")
        if policy["mode"] not in ("reference", "query_chunks") or policy["scope"] != (
                "query/index-score scratch; full projection, Compressor and MoE shapes retained"):
            raise ValueError("invalid prefill query policy")
        for key in counts:
            _nonnegative(policy[key], f"prefill_policy.{key}")
        if policy["mode"] == "reference" and any(policy[k] for k in counts):
            raise ValueError("reference prefill cannot claim query chunk execution")
    return deepcopy(value)


def storage_residency(tensors):
    """Actual backing-storage bytes; aliases/views count once, including unused capacity.

    `kv_cache[:, win:]` is a view, so numel() would both double-count it and omit
    backing capacity. Storage pointer + device identifies the allocation instead.
    Accessing this metadata does not read tensor values or synchronize a GPU.
    """
    seen, sizes = set(), {"gpu_bytes": 0, "host_bytes": 0}
    for tensor in tensors:
        if tensor is None:
            continue
        storage = tensor.untyped_storage()
        nbytes = storage.nbytes()
        if not nbytes:
            continue
        device = str(tensor.device)
        key = (device, storage.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        if device == "cpu":
            tier = "host_bytes"
        elif device.startswith("cuda"):
            tier = "gpu_bytes"
        else:
            raise ValueError(f"runtime metrics cannot classify KV storage on {device}")
        sizes[tier] += nbytes
    return sizes


class RuntimeMetrics:
    """One stage's observations for one job. Reset only at the job boundary.

    KV peaks are peaks of observed owned allocations at execution boundaries;
    allocator peaks come from PyTorch's measured process-wide CUDA counters.
    They are not per-stage ownership estimates or simulated hardware figures.
    """

    def __init__(self, device, *, torch_module=None, mode=None):
        self.device = str(device)
        self.mode = mode or ("gpu_resident" if self.device.startswith("cuda") else "reference_cpu")
        if self.mode not in (*GPU_MODES, "reference_cpu") or \
                (self.mode in GPU_MODES) != self.device.startswith("cuda"):
            raise ValueError("runtime metrics residency mode disagrees with its device")
        self.torch = torch_module
        self._gpu_owners = None
        if self.mode in GPU_MODES and self.torch is not None:
            # An unindexed CUDA device means the CURRENT GPU, which may be
            # nonzero in a multi-GPU process. Share the same owner set as its
            # explicit cuda:N spelling, and keep allocator reads on that GPU.
            key = self.device if ":" in self.device else f"cuda:{self.torch.cuda.current_device()}"
            self.device = key
            self._gpu_owners = _GPU_OBSERVERS.setdefault(key, weakref.WeakSet())
            self._gpu_owners.add(self)
            if len(self._gpu_owners) > 1:
                for observer in self._gpu_owners:
                    if observer is not self:
                        # Sharing at any time invalidates this job's exclusive peak interval,
                        # even if the other Stage is destroyed before receipt collection.
                        observer._gpu_peak_exclusive = False
                        observer.gpu_memory = None
        self.reset()

    def reset(self):
        self.work = {role: {phase: _counts() for phase in PHASES} for role in ROLES}
        self.kv = {key: 0 for key in KV_FIELDS}
        self.gpu_memory = None
        self.expert_prefetch = None
        self.expert_cache = None
        self.performance = None
        self.prefetch_policy = None
        self.kernel_coverage = None
        self._gpu_peak_exclusive = (self._gpu_owners is not None and len(self._gpu_owners) == 1
                                    and not _EXTERNAL_ALLOCATOR_INTERVALS.get(self.device))
        if self._gpu_peak_exclusive:
            # Host allocator metadata, no synchronize() and no per-token resets.
            self.torch.cuda.reset_peak_memory_stats(self.device)
            self.sample_gpu_memory()

    def resident_routes(self, entries, *, role="main", phase="decode"):
        _nonnegative(entries, "routed entries")
        if role not in ROLES or phase not in PHASES:
            raise ValueError("unknown runtime work role/phase")
        c = self.work[role][phase]
        c["routed_entries"] += entries
        c["resident_hits" if self.mode in GPU_MODES else "reference_routes"] += entries

    def expert_routes(self, counts, *, role="main", phase="decode"):
        """Actual demand paths from a local cache; counts share one route denominator."""
        if role not in ROLES or phase not in PHASES or set(counts) != set(COUNTS):
            raise ValueError("unknown runtime work role/phase or incomplete route counts")
        for key in COUNTS:
            _nonnegative(counts[key], key, real=key == "dma_wait_ms")
        if counts["routed_entries"] != sum(counts[k] for k in
                ("resident_hits", "dma_misses", "cpu_misses", "reference_routes")):
            raise ValueError("expert route counts do not share one denominator")
        if self.mode == "reference_cpu" and any(counts[k] for k in
                ("resident_hits", "dma_misses", "dma_bytes", "dma_wait_ms", "cpu_misses")):
            raise ValueError("CPU emulation cannot claim GPU hits or DMA")
        if self.mode in GPU_MODES and counts["reference_routes"]:
            raise ValueError("GPU caching cannot claim reference execution")
        for key, value in counts.items():
            self.work[role][phase][key] += value

    def dma_wait(self, elapsed_ms, *, role="main", phase="decode", prefetch=False):
        """CUDA consumer wait measured at the job barrier, not copy duration."""
        _nonnegative(elapsed_ms, "dma_wait_ms", real=True)
        if self.mode not in GPU_MODES:
            raise ValueError("CPU reference execution cannot measure a CUDA wait")
        if prefetch:
            self.prefetch(0, elapsed_ms)
        else:
            if role not in ROLES or phase not in PHASES:
                raise ValueError("unknown runtime work role/phase")
            self.work[role][phase]["dma_wait_ms"] += elapsed_ms

    def prefetch(self, dma_bytes, elapsed_ms=0.0):
        _nonnegative(dma_bytes, "prefetch DMA bytes")
        _nonnegative(elapsed_ms, "prefetch DMA wait", real=True)
        if self.mode not in GPU_MODES:
            raise ValueError("CPU reference execution cannot claim prefetched DMA")
        if self.expert_prefetch is None:
            self.expert_prefetch = {"dma_bytes": 0, "dma_wait_ms": 0.0}
        self.expert_prefetch["dma_bytes"] += dma_bytes
        self.expert_prefetch["dma_wait_ms"] += elapsed_ms

    def sample_kv(self, tensors):
        sizes = storage_residency(tensors)
        for tier in ("gpu", "host"):
            key = f"{tier}_bytes"
            self.kv[key] = sizes[key]
            peak = f"{tier}_peak_bytes"
            self.kv[peak] = max(self.kv[peak], sizes[key])

    def prefetch_quality(self, counts):
        fields = ("requested", "used", "wasted", "skipped", "candidate_count")
        _keys(counts, fields, "prefetch_policy")
        for key, count in counts.items():
            _nonnegative(count, f"prefetch_policy.{key}")
        if self.prefetch_policy is None:
            self.prefetch_policy = {key: 0 for key in fields}
        for key, count in counts.items():
            self.prefetch_policy[key] += count

    def sample_gpu_memory(self):
        if not self._gpu_peak_exclusive or len(self._gpu_owners) != 1:
            # Global allocator reset would corrupt another Stage's measurement in a shared
            # process. Omit process peaks for this job, rather than claim per-job figures.
            self.gpu_memory = None
            return
        cuda, d = self.torch.cuda, self.device
        self.gpu_memory = {"scope": "process", "allocated_bytes": cuda.memory_allocated(d),
                           "reserved_bytes": cuda.memory_reserved(d),
                           "allocated_peak_bytes": cuda.max_memory_allocated(d),
                           "reserved_peak_bytes": cuda.max_memory_reserved(d)}

    def snapshot(self):
        self.sample_gpu_memory()
        totals = _counts()
        for role in ROLES:
            for phase in PHASES:
                for key in COUNTS:
                    totals[key] += self.work[role][phase][key]
        body = {"schema": SCHEMA, "mode": self.mode, "work": self.work, "totals": totals,
                "resident_hit_rate": totals["resident_hits"] / totals["routed_entries"]
                if totals["routed_entries"] else None, "kv": self.kv}
        if self.gpu_memory is not None:
            body["gpu_memory"] = self.gpu_memory
        if self.expert_prefetch is not None:
            body["expert_prefetch"] = self.expert_prefetch
        if self.expert_cache is not None:
            body["expert_cache"] = self.expert_cache
        if self.performance is not None:
            body["performance"] = self.performance
        if self.prefetch_policy is not None:
            body["prefetch_policy"] = self.prefetch_policy
        if self.kernel_coverage is not None:
            body["kernel_coverage"] = self.kernel_coverage
        return validate_runtime_metrics(body)

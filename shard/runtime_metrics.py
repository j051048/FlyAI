"""Optional, signed observations of executed work and local memory residency.

No torch import: receipt verifiers can validate this schema without loading an engine.
Routes count token/expert entries, including rejected speculation and rollback replay,
not unique experts or committed output tokens. Every path uses that same denominator.
The V4 resident engine records geometry on the host; it never copies router tensors to
the CPU or changes a captured graph to obtain a counter.
"""
from copy import deepcopy
import math
import weakref

SCHEMA = "shard-runtime-metrics/1"
ROLES = ("main", "draft")
PHASES = ("prefill", "decode", "replay")
COUNTS = ("routed_entries", "resident_hits", "dma_misses", "dma_bytes",
          "dma_wait_ms", "cpu_misses", "reference_routes")
KV_FIELDS = ("gpu_bytes", "host_bytes", "gpu_peak_bytes", "host_peak_bytes")
GPU_FIELDS = ("scope", "allocated_bytes", "reserved_bytes", "allocated_peak_bytes",
              "reserved_peak_bytes")
_GPU_OBSERVERS = {}


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
    _keys(value, keys, "runtime_metrics")
    if value["schema"] != SCHEMA:
        raise ValueError(f"unknown runtime metrics schema {value['schema']!r}")
    if value["mode"] not in ("gpu_resident", "reference_cpu"):
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
            if value["mode"] == "gpu_resident" and c["reference_routes"]:
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
        if value["mode"] != "gpu_resident":
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

    def __init__(self, device, *, torch_module=None):
        self.device = str(device)
        self.mode = "gpu_resident" if self.device.startswith("cuda") else "reference_cpu"
        self.torch = torch_module
        self._gpu_owners = None
        if self.mode == "gpu_resident" and self.torch is not None:
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
        self._gpu_peak_exclusive = self._gpu_owners is not None and len(self._gpu_owners) == 1
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
        c["resident_hits" if self.mode == "gpu_resident" else "reference_routes"] += entries

    def sample_kv(self, tensors):
        sizes = storage_residency(tensors)
        for tier in ("gpu", "host"):
            key = f"{tier}_bytes"
            self.kv[key] = sizes[key]
            peak = f"{tier}_peak_bytes"
            self.kv[peak] = max(self.kv[peak], sizes[key])

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
        return validate_runtime_metrics(body)

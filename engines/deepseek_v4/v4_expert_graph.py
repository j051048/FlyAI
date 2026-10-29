"""Bounded stateless CUDA segments around a host-scheduled expert cache.

Attention/KV remains owned by WholeBlockGraphs' pre/post split. These graphs
contain only gate, shared expert or ONE fixed physical expert slot. Acquisition,
eviction and waits never enter capture. A slot graph reads its current bytes,
not the logical expert that occupied it during capture. CPU mode is an explicit
static-buffer reference for lifecycle tests and is never graph qualification.
"""
from threading import RLock

import torch


SCHEMA = "v4-expert-segment-graphs/1"


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    else:
        raise ValueError("expert graph outputs must be tensors or tensor sequences")


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    return type(value)(_clone(item) for item in value)


def _bytes(values):
    seen, total = set(), 0
    for value in values:
        storage = value.untyped_storage()
        key = (str(value.device), storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


class ExpertSegmentGraphs:
    """No LRU or resizing: fixed physical addresses until this owner is closed."""
    def __init__(self, runtime, *, max_bytes, max_graphs=32, reference=False):
        if type(max_bytes) is not int or max_bytes <= 0 or type(max_graphs) is not int or not 1 <= max_graphs <= 4096:
            raise ValueError("expert segments require positive bounded bytes and graph count")
        if type(reference) is not bool or reference != runtime.emulation:
            raise ValueError("expert segments require CUDA or explicit CPU reference")
        self.runtime, self.max_bytes, self.max_graphs = runtime, max_bytes, max_graphs
        self.reference, self.used_bytes = reference, 0
        self._entries, self._stamp = {}, None
        self._retained_failed = []
        self._lock = RLock()
        self._faulted, self._closed = False, False
        self.counters = dict(captures=0, replays=0, reference_calls=0, eager_fallbacks=0, capture_errors=0)

    def _identity(self):
        run = self.runtime
        run._require_cache()
        # Slot versions change on each legitimate migration. Pointers/shape and
        # pool epoch bind the graphs; resident gate/shared weights also bind their
        # version so a reload cannot silently replay an old structural program.
        def desc(t, version=False):
            return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device),
                    t._version if version else None)
        return (run.pool.identity if hasattr(run.pool, "identity") else id(run.pool), run.pool.epoch,
                tuple((name, desc(t)) for name, t in run.cache.banks.items() if t is not None),
                tuple((name, desc(t, True)) for prefix, module in
                      (("gate", run.moe.gate), ("shared", run.moe.shared_experts))
                      for name, t in [(prefix + "." + n, p) for n, p in
                                     (*module.named_parameters(), *module.named_buffers())]))

    @torch.no_grad()
    def run(self, key, fn, *inputs):
        with self._lock:
            if self._closed or self._faulted:
                raise RuntimeError("expert segment graph owner is closed or faulted")
            if not inputs or any(not isinstance(t, torch.Tensor) for t in inputs):
                raise ValueError("expert segments require tensor inputs")
            stamp = self._identity()
            if self._stamp is None:
                self._stamp = stamp
            if self._stamp != stamp:
                self._faulted = True
                raise RuntimeError("expert segment graph weights/slot storage changed; rebuild the Stage")
            program = (id(fn), id(type(fn).forward), id(fn.__dict__.get("forward"))) if isinstance(fn, torch.nn.Module) else (
                id(getattr(fn, "__func__", fn)), id(getattr(fn, "__self__", None)))
            signature = (key, tuple((tuple(t.shape), tuple(t.stride()), t.dtype, str(t.device)) for t in inputs))
            entry = self._entries.get(signature)
            if entry is not None and entry["program"] != program:
                self._faulted = True
                raise RuntimeError("expert segment callable changed; rebuild the Stage")
            if entry is None:
                # A shape that does not fit the bounded graph set executes the
                # same original arithmetic eagerly. No unbounded retained pools.
                minimum = sum(t.numel() * t.element_size() for t in inputs)
                if len(self._entries) >= self.max_graphs or self.used_bytes + minimum > self.max_bytes:
                    self.counters["eager_fallbacks"] += 1
                    return fn(*inputs)
                try:
                    entry = self._capture(fn, inputs)
                    entry["program"] = program
                    if self.used_bytes + entry["bytes"] > self.max_bytes:
                        self._retained_failed.append(entry)
                        self.used_bytes += entry["bytes"]
                        self._faulted = True
                        raise RuntimeError("expert segment capture exceeded declared device budget; restart Stage")
                    self.used_bytes += entry["bytes"]
                    self._entries[signature] = entry
                except BaseException:
                    self.counters["capture_errors"] += 1
                    self._faulted = True
                    raise
            if self.reference:
                for target, source in zip(entry["inputs"], inputs):
                    target.copy_(source)
                result = fn(*entry["inputs"])
                self.counters["reference_calls"] += 1
                return _clone(result)
            current = torch.cuda.current_stream(self.runtime.device)
            if entry["done"] is not None:
                current.wait_event(entry["done"])
            for target, source in zip(entry["inputs"], inputs):
                target.copy_(source)
            try:
                entry["graph"].replay()
                result = _clone(entry["out"])
                done = torch.cuda.Event()
                done.record(current)
                entry["done"] = done
                self.counters["replays"] += 1
                return result
            except BaseException:
                self._faulted = True
                raise

    def _capture(self, fn, inputs):
        if self.reference:
            static = tuple(t.clone() for t in inputs)
            out = fn(*static)
            return {"inputs": static, "out": out, "bytes": _bytes((*static, *_tensors(out)))}
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("host expert graph dispatch cannot run inside another capture")
        device = torch.device(self.runtime.device)
        current = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(current)  # includes this expert's individual ready event
        with torch.cuda.device(device), torch.cuda.stream(side), torch.no_grad():
            static = tuple(t.clone() for t in inputs)
            for _ in range(2):
                fn(*static)
        side.synchronize()  # cold first-use JIT only, never a replay synchronization
        before = torch.cuda.memory_allocated(device)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.device(device), torch.cuda.graph(graph, stream=side), torch.no_grad():
                out = fn(*static)
        except BaseException:
            side.synchronize()
            raise
        current.wait_stream(side)
        size = max(_bytes((*static, *_tensors(out))), torch.cuda.memory_allocated(device) - before)
        self.counters["captures"] += 1
        return {"inputs": static, "out": out, "graph": graph, "done": None, "bytes": size}

    def tensors(self):
        for index, entry in enumerate((*self._entries.values(), *self._retained_failed)):
            for j, tensor in enumerate((*entry["inputs"], *_tensors(entry["out"]))):
                yield f"segment.{index}.{j}", tensor

    def status(self):
        return {"schema": SCHEMA, "max_bytes": self.max_bytes, "max_graphs": self.max_graphs,
                "accounted_bytes": self.used_bytes, "entries": len(self._entries), "reference": self.reference,
                "faulted": self._faulted, "graph_qualified": False, **self.counters}

    def close(self):
        with self._lock:
            for entry in (*self._entries.values(), *self._retained_failed):
                if entry.get("done") is not None:
                    entry["done"].synchronize()
            self._entries.clear()
            self._retained_failed.clear()
            self.used_bytes = 0
            self._closed = True

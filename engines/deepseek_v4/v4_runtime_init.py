"""Explicit V4 runtime prerequisites, storage aliases and CPU-test isolation."""
from copy import deepcopy
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import types

V4_HADAMARD = os.environ.get("V4_HADAMARD", "auto")
_LOCK = threading.RLock()
_FUNCTION = None
_IDENTITY = None
_PROBES = {}
_FAILED = {}
ALIAS_SCHEMA = "v4-attention-aliases/1"


def _function_identity(function, module):
    files = set()
    for value in (module, sys.modules.get(getattr(function, "__module__", "")),
                  sys.modules.get("hadamard_transform_cuda")):
        path = getattr(value, "__file__", None)
        if path and Path(path).is_file():
            files.add(Path(path))
    hashes = [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
              for path in sorted(files)]
    try:
        code = inspect.getsource(function).encode()
    except (OSError, TypeError):
        code = repr(hashes).encode()
    try:
        version = importlib.metadata.version("fast-hadamard-transform")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"source_sha256": hashlib.sha256(code).hexdigest(), "module_files": hashes,
            "dependency_version": version}


def ensure_hadamard():
    """Select once; register only FHT, never replace TileLang's `kernel` module."""
    global _FUNCTION, _IDENTITY
    with _LOCK:
        if _FUNCTION is not None:
            if getattr(sys.modules.get("fast_hadamard_transform"), "hadamard_transform", None) is not _FUNCTION:
                raise RuntimeError("Hadamard backend changed after initialization; use a fresh process")
            return _IDENTITY["backend"]
        if V4_HADAMARD not in ("auto", "extension", "torch"):
            raise ValueError("V4_HADAMARD must be auto, extension or torch")
        module, reason = None, "requested"
        if V4_HADAMARD != "torch":
            try:
                module = importlib.import_module("fast_hadamard_transform")
            except (ImportError, OSError) as error:
                if V4_HADAMARD == "extension":
                    raise RuntimeError("requested Hadamard extension cannot be imported") from error
                reason = "extension_unavailable:" + type(error).__name__
        candidate = getattr(module, "hadamard_transform", None)
        from v4_kernels_cpu import hadamard_transform as torch_reference
        cpu_shim = module is not None and (getattr(module, "_v4_cpu_backend", False) or candidate is torch_reference)
        if V4_HADAMARD == "extension" and cpu_shim:
            raise RuntimeError("requested Hadamard extension resolved to a Torch reference shim")
        if module is None or cpu_shim or V4_HADAMARD == "torch":
            from v4_kernels_cpu import hadamard_transform
            function = hadamard_transform
            module = types.ModuleType("fast_hadamard_transform")
            module.hadamard_transform = function
            module._v4_cpu_backend = True
            module._v4_hadamard_backend = "torch"
            sys.modules["fast_hadamard_transform"] = module
            backend = "torch"
            reason = "cpu_reference" if cpu_shim else reason
        else:
            function = getattr(module, "hadamard_transform", None)
            if not callable(function):
                raise RuntimeError("Hadamard extension exposes no callable transform")
            backend = "extension"
        _FUNCTION = function
        _IDENTITY = {"requested": V4_HADAMARD, "backend": backend, "reason": reason,
                     **_function_identity(function, module)}
        return backend


def hadamard_identity():
    """Stable configuration identity; shape probes never enter this payload."""
    return deepcopy(_IDENTITY) if _IDENTITY is not None else {
        "requested": V4_HADAMARD, "backend": "uninitialized", "reason": "not_initialized",
        "source_sha256": None, "module_files": [], "dependency_version": None}


def hadamard_status():
    """Runtime observations, including actual device/shape validation."""
    return {**hadamard_identity(), "probe_count": len(_PROBES),
            "probes": list(deepcopy(_PROBES).values()), "failed_probes": list(deepcopy(_FAILED).values())}


def probe_hadamard(device, dtype, shape):
    """Exercise the actual input layout before READY/capture, not just import.

    Exact dyadic inputs make a shape/layout smoke test deterministic. This does
    not certify every trained input, or replace the real-model GPU parity gate.
    A chosen extension failure is refused, never silently switched in a live graph.
    """
    import torch
    shape = tuple(shape)
    if not shape or any(type(n) is not int or n < 1 for n in shape) or shape[-1] & (shape[-1] - 1):
        raise ValueError("Hadamard probe requires positive dimensions and power-of-two width")
    device = torch.device(device)
    key = (str(device), str(dtype), shape)
    ensure_hadamard()
    with _LOCK:
        if key in _FAILED:
            raise RuntimeError("Hadamard device/shape was already refused")
        if key in _PROBES:
            return deepcopy(_PROBES[key])
        row = {"device": str(device), "dtype": str(dtype), "shape": list(shape),
               "scope": "actual-device dyadic layout smoke; not all-input numerical proof"}
        try:
            from v4_kernels_cpu import hadamard_transform
            count = 1
            for n in shape:
                count *= n
            x = ((torch.arange(count, device=device, dtype=torch.float32).remainder(17) - 8) / 16).reshape(shape).to(dtype)
            original = x.clone()
            scale = shape[-1] ** -.5
            with torch.no_grad():
                expected = hadamard_transform(x, scale=scale)
                actual = _FUNCTION(x, scale=scale)
            if (not torch.is_tensor(actual) or actual.shape != x.shape or actual.dtype != dtype
                    or actual.device != x.device or not torch.equal(actual, expected)
                    or not torch.equal(x, original) or not torch.isfinite(actual).all().item()):
                raise RuntimeError("Hadamard device/shape numerical smoke failed")
        except Exception as error:
            _FAILED[key] = {**row, "error_type": type(error).__name__}
            raise RuntimeError("Hadamard backend refused actual device/shape before READY") from error
        _PROBES[key] = {**row, "passed": True}
        return deepcopy(_PROBES[key])


def tensor_storage_signature(tensor):
    if tensor is None:
        return None
    return (str(tensor.device), str(tensor.dtype), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.storage_offset(), tensor.untyped_storage().data_ptr())


def bind_attention_aliases(attention, *, rebind=False):
    """Bind/validate actual views without touching recurrence/cache contents.

    `rebind=True` is for a known buffer replacement. Existing graph owners must
    discard captures when the returned storage signature changes.
    """
    if not attention.compress_ratio:
        return {"schema": ALIAS_SCHEMA, "changed": False, "signature":
                (tensor_storage_signature(attention.kv_cache), tensor_storage_signature(attention.freqs_cis))}
    expected = [(attention.compressor, "kv_cache", attention.kv_cache[:, attention.window_size:]),
                (attention.compressor, "freqs_cis", attention.freqs_cis)]
    indexer = attention.indexer
    if indexer is not None:
        expected += [(indexer, "freqs_cis", attention.freqs_cis),
                     (indexer.compressor, "kv_cache", indexer.kv_cache),
                     (indexer.compressor, "freqs_cis", attention.freqs_cis)]
    changed, signatures = False, []
    for owner, name, wanted in expected:
        current = getattr(owner, name)
        signature = tensor_storage_signature(wanted)
        if current is None or tensor_storage_signature(current) != signature:
            if current is not None and not rebind:
                raise RuntimeError("V4 attention alias no longer matches its owning buffer")
            setattr(owner, name, wanted); changed = True
        signatures.append(signature)
    return {"schema": ALIAS_SCHEMA, "changed": changed, "signature": tuple(signatures)}


def prepare_cpu_selftest():
    """Select CPU before reference import; refuse cached GPU state in this process."""
    kernel = sys.modules.get("kernel")
    if kernel is not None and not getattr(kernel, "_v4_cpu_backend", False):
        raise RuntimeError("CPU selftest refuses an already loaded GPU/foreign kernel; start a fresh selftest process")
    cpu = sys.modules.get("v4_kernels_cpu")
    if cpu is not None and cpu.backend() != "cpu":
        raise RuntimeError("CPU selftest refuses a cached GPU backend; start a fresh selftest process")
    if _IDENTITY is not None and _IDENTITY["backend"] != "torch":
        raise RuntimeError("CPU selftest refuses a cached GPU Hadamard backend; start a fresh selftest process")
    os.environ["V4_KERNELS"] = "cpu"
    os.environ["V4_HADAMARD"] = "torch"
    recipe = {"V4_KERNELS": "cpu", "V4_HADAMARD": "torch", "V4_CUDA_GRAPH": "0",
              "V4_MOE_GROUPED": "0", "V4_MOE_IN_GRAPH": "0", "V4_FP8_GEMV": "0",
              "V4_FP8_SHARED": "0", "V4_DSPARK_GRAPH": "0", "V4_DSPARK_MOE": "0",
              "V4_WIRE_FUSED": "0", "V4_EXPERT_PLACEMENT": "gpu", "V4_KV_PLACEMENT": "gpu",
              "V4_PREFILL_EXPERT_PIPELINE": "0", "V4_LEVERS_STRICT": "0"}
    os.environ.update(recipe)
    if cpu is not None:
        cpu.V4_KERNELS = "cpu"
    global V4_HADAMARD
    V4_HADAMARD = "torch"
    if _IDENTITY is not None:
        _IDENTITY["requested"] = "torch"
    print("CPU_SELFTEST_RECIPE " + json.dumps(recipe, sort_keys=True), flush=True)

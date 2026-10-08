"""Opt-in CUDA compilation of the existing FP8 wire expressions, gated byte-for-byte.

There is no codec format here: output is the same e4m3 activation and BF16 row scale
used by v4_pipe. CPU/unsupported/unwarmed shapes use the reference expressions.
Compilation, numerical checks and synchronization occur only in explicit warmup(),
normally before the stage is ready. The live path never initiates compilation.

CUDA graphs are disabled in the compiler: every call owns fresh output storage,
so a queued transport frame cannot be overwritten by the next packed activation.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import warnings

import torch

SCHEMA = "v4-wire-codec/1"
# Preserve the BF16 scale roundtrip before division, division rounding, subnormals
# and signed zero. An older compiler missing these controls safely declines.
COMPILE_OPTIONS = {
    "emulate_precision_casts": True,
    "use_fast_math": False,
    "strict_signed_zero": True,
    "eager_numerics.division_rounding": True,
    "eager_numerics.disable_ftz": True,
    "triton.cudagraphs": False,
    "max_autotune": False,
    "shape_padding": False,
    "inplace_buffers": False,
    "memory_planning": False,
}


def reference_pack(t):
    """The original expressions, including FP32 amax and the rounded BF16 divisor."""
    f = t.detach().float()
    scale = (f.abs().amax(-1, keepdim=True) / 448.0).clamp(
        min=1e-8, max=torch.finfo(torch.bfloat16).max).to(torch.bfloat16)
    q = (f / scale.float()).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    return q, scale.squeeze(-1).contiguous()


def reference_unpack(q, scale):
    """The original BF16 multiplication; no FP32-sized decoded temporary."""
    return q.to(torch.bfloat16) * scale.unsqueeze(-1)


def _legacy_unpack(q, scale):
    """The wire receiver historically multiplied on CPU, including its NaN bit policy.

    The normal caller checks can_unpack() BEFORE moving compact payloads to CUDA.
    A direct GPU call that declines still preserves that CPU decoder's bytes.
    """
    if q.device.type == "cuda":
        return reference_unpack(q.cpu(), scale.cpu()).to(q.device)
    return reference_unpack(q, scale)


def _signature(*tensors):
    return tuple((str(t.device), str(t.dtype), tuple(t.shape), tuple(t.stride()),
                  bool(t.requires_grad), bool(t.is_inference())) for t in tensors)


def _bit_equal(a, b):
    # Floating equality rejects NaN==NaN and can conceal the sign of zero. The wire
    # and receipt contract is byte equality, including those special bit patterns.
    return (a.dtype == b.dtype and a.shape == b.shape and
            torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def _owns_outputs(outputs, inputs):
    """Storage metadata only; this check belongs to warmup, not the live path."""
    input_stores = {t.untyped_storage().data_ptr() for t in inputs}
    return all(t.is_contiguous() and t.untyped_storage().data_ptr() not in input_stores
               for t in outputs)


class CodecParityError(RuntimeError):
    pass


@dataclass
class _Entry:
    state: str = "warming"
    fn: object = None
    reason: str | None = None


class WireCodec:
    """One process's bounded, exact-shape compiled kernels and observable fallbacks."""

    def __init__(self, *, compiler=None, max_keys=32, max_probe_elements=1048576):
        if type(max_keys) is not int or max_keys < 1:
            raise ValueError("wire codec max_keys must be a positive integer")
        if type(max_probe_elements) is not int or max_probe_elements < 1:
            raise ValueError("wire codec max_probe_elements must be a positive integer")
        self._compiler, self.max_keys = compiler, max_keys
        self.max_probe_elements = max_probe_elements
        self._entries = {"pack": {}, "unpack": {}}
        self._warm_lock = threading.RLock()
        self._capture_warned = set()
        self._counters = {
            "warmup_calls": 0, "warmup_skips": 0,
            "compile_failures": 0, "parity_failures": 0, "runtime_failures": 0,
            "uses": {"pack": 0, "unpack": 0},
            "fallbacks": {"cpu": 0, "unsupported": 0, "unwarmed": 0,
                          "failed_key": 0, "runtime_error": 0, "capture": 0},
        }

    def _eligible(self, op, inputs):
        if op == "pack":
            (t,) = inputs
            return (t.device.type == "cuda" and t.dtype == torch.bfloat16 and t.ndim >= 1
                    and t.numel() > 0 and t.is_contiguous() and not t.requires_grad)
        q, scale = inputs
        return (q.device.type == "cuda" and scale.device == q.device
                and q.dtype == torch.float8_e4m3fn and scale.dtype == torch.bfloat16
                and q.ndim >= 1 and q.numel() > 0 and tuple(scale.shape) == tuple(q.shape[:-1])
                and q.is_contiguous() and scale.is_contiguous()
                and not q.requires_grad and not scale.requires_grad)

    def _capturing(self, inputs):
        # A capture-state query reads no tensor, performs no stream synchronization
        # and prevents a captured codec output from being reused by pending frames.
        if inputs[0].device.type != "cuda":
            return False
        with torch.cuda.device(inputs[0].device):
            return torch.cuda.is_current_stream_capturing()

    def _cuda_inputs(self, inputs):
        return inputs[0].device.type == "cuda"

    def _compile(self, fn):
        if self._compiler is not None:  # deterministic dispatcher tests inject an explicit fake
            return self._compiler(fn, dict(COMPILE_OPTIONS))
        # Deferred entirely: no Dynamo/compiler import on CPU or an unwarmed live call.
        options = set(torch._inductor.list_options())
        missing = set(COMPILE_OPTIONS) - options
        if missing:
            raise RuntimeError(f"compiler lacks exact-wire controls: {sorted(missing)}")
        return torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False,
                             options=dict(COMPILE_OPTIONS))

    def _fail(self, op, key, error, counter):
        entry = self._entries[op][key]
        if entry.state == "failed":
            return
        entry.state, entry.fn = "failed", None
        entry.reason = f"{type(error).__name__}: {str(error)[:240]}"
        self._counters[counter] += 1
        warnings.warn(f"V4 fused wire {op} declined for {key}: {entry.reason}; "
                      "using the original codec", RuntimeWarning, stacklevel=3)

    def _dispatch(self, op, inputs, reference):
        if not self._cuda_inputs(inputs):
            self._counters["fallbacks"]["cpu"] += 1
            return reference(*inputs)
        if not self._eligible(op, inputs):
            self._counters["fallbacks"]["unsupported"] += 1
            return reference(*inputs)
        key = _signature(*inputs)
        entry = self._entries[op].get(key)
        if entry is None or entry.state == "warming":
            self._counters["fallbacks"]["unwarmed"] += 1
            return reference(*inputs)
        if entry.state == "failed":
            self._counters["fallbacks"]["failed_key"] += 1
            return reference(*inputs)
        if self._capturing(inputs):
            self._counters["fallbacks"]["capture"] += 1
            tag = (op, key)
            if tag not in self._capture_warned:
                self._capture_warned.add(tag)
                warnings.warn("fused V4 wire output must remain outside CUDA capture; "
                              "using the original codec", RuntimeWarning, stacklevel=3)
            return reference(*inputs)
        try:
            # Stable grad mode avoids a new Dynamo specialization on the live path.
            with torch.inference_mode(bool(inputs[0].is_inference())), torch.no_grad():
                result = entry.fn(*inputs)
        except Exception as error:
            self._fail(op, key, error, "runtime_failures")
            self._counters["fallbacks"]["runtime_error"] += 1
            return reference(*inputs)
        self._counters["uses"][op] += 1
        return result

    def pack(self, tensor):
        return self._dispatch("pack", (tensor,), reference_pack)

    def unpack(self, q, scale):
        return self._dispatch("unpack", (q, scale), _legacy_unpack)

    def can_unpack(self, q, scale, *, device=None):
        """Metadata-only admission before compact H2D, for an already byte-verified key."""
        target = torch.device(device) if device is not None else q.device
        if target.type != "cuda" or q.requires_grad or scale.requires_grad:
            return False
        if (q.dtype != torch.float8_e4m3fn or scale.dtype != torch.bfloat16
                or tuple(scale.shape) != tuple(q.shape[:-1]) or not q.is_contiguous()
                or not scale.is_contiguous() or not q.numel()):
            return False
        if target.index is None:
            if not torch.cuda.is_available():
                return False
            target = torch.device("cuda", torch.cuda.current_device())
        key = tuple((str(target), signature[1], signature[2], signature[3], signature[4], signature[5])
                    for signature in _signature(q, scale))
        entry = self._entries["unpack"].get(key)
        return entry is not None and entry.state == "ready"

    def _probes(self, tensor):
        """Same layout, numerical stress inputs; never run by pack()/unpack()."""
        shape, device = tensor.shape, tensor.device
        generator = torch.Generator(device=device).manual_seed(713)
        random = torch.randn(shape, dtype=torch.float32, device=device, generator=generator).to(tensor.dtype)
        yield tensor.detach().clone()
        yield random
        yield torch.zeros_like(tensor)
        special = random.clone()
        flat = special.reshape(-1)
        values = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"),
                               torch.finfo(torch.bfloat16).max,
                               torch.finfo(torch.bfloat16).tiny], dtype=tensor.dtype, device=device)
        count = min(flat.numel(), values.numel())
        flat[:count] = values[:count]
        yield special
        infinities = random.clone()
        flat = infinities.reshape(-1)
        flat[0] = float("inf")
        flat[min(shape[-1], flat.numel() - 1)] = -float("inf")
        yield infinities
        # Explicit signed BF16 NaN payloads and signed subnormal patterns.
        bits = random.clone()
        raw = bits.view(torch.int16).reshape(-1)
        patterns = torch.tensor([32704, -64, 32767, -1, 1, -32767, 0, -32768],
                                dtype=torch.int16, device=device)
        count = min(raw.numel(), patterns.numel())
        raw[:count] = patterns[:count]
        yield bits
        # Different positions/streams must keep independent scales across a chunk.
        levels = random.float().reshape(-1, shape[-1])
        powers = torch.arange(levels.shape[0], dtype=torch.float32, device=device).remainder(37) - 18
        levels *= torch.exp2(powers).unsqueeze(-1)
        yield levels.reshape(shape).to(tensor.dtype)

    def _verify_pack(self, fn, probes):
        retained = []
        for probe in probes:
            expected = reference_pack(probe)
            actual = fn(probe)
            if (not isinstance(actual, tuple) or len(actual) != 2
                    or not all(_bit_equal(a, b) for a, b in zip(actual, expected))
                    or not _owns_outputs(actual, (probe,))):
                raise CodecParityError("FP8 payload/rounded BF16 scales differ, or output aliases its input")
            retained.append((actual, tuple(t.clone() for t in actual)))
        # Keep previous results alive: an allocator/capture buffer cannot overwrite a pending frame.
        for actual, snapshot in retained:
            if not all(_bit_equal(a, b) for a, b in zip(actual, snapshot)):
                raise CodecParityError("compiled pack reused storage belonging to a retained frame")

    def _verify_unpack(self, fn, probes):
        retained = []
        pairs = [reference_pack(probe) for probe in probes]
        q, scale = pairs[0]
        edge = torch.zeros_like(q).view(torch.uint8)
        # FP8 byte codes for +0/-0, +1/-1, +448/-448 and both signed NaNs.
        codes = torch.tensor([0, 128, 56, 184, 126, 254, 127, 255],
                             dtype=torch.uint8, device=q.device)
        flat = edge.reshape(-1)
        count = min(flat.numel(), codes.numel())
        flat[:count] = codes[:count]
        edge = edge.view(torch.float8_e4m3fn)
        for value in (0.0, -0.0, float("inf"), -float("inf"), float("nan"),
                      torch.finfo(torch.bfloat16).max, torch.finfo(torch.bfloat16).tiny):
            pairs.append((edge, torch.full_like(scale, value)))
        for q, scale in pairs:
            expected, actual = reference_unpack(q.cpu(), scale.cpu()), fn(q, scale)
            if not _bit_equal(actual.cpu(), expected) or not _owns_outputs((actual,), (q, scale)):
                raise CodecParityError("BF16 decoded bytes differ from the original CPU receiver, or output aliases its input")
            retained.append((actual, actual.clone()))
        for actual, snapshot in retained:
            if not _bit_equal(actual, snapshot):
                raise CodecParityError("compiled unpack reused storage belonging to a retained frame")

    def warmup(self, tensor):
        """Compile and byte-verify pack and unpack for this exact CUDA shape, before readiness.

        A failure is sticky for that key and is reported once; later jobs use the
        reference. CPU/unsupported inputs cannot initialize Dynamo or claim CUDA use.
        """
        self._counters["warmup_calls"] += 1
        if (not self._eligible("pack", (tensor,)) or self._capturing((tensor,))
                or tensor.numel() > self.max_probe_elements):
            self._counters["warmup_skips"] += 1
            return self.stats()
        with self._warm_lock, torch.inference_mode(bool(tensor.is_inference())), torch.no_grad():
            probes = list(self._probes(tensor))
            q, scale = reference_pack(tensor)
            specifications = (("pack", (tensor,), reference_pack, self._verify_pack),
                              ("unpack", (q, scale), reference_unpack, self._verify_unpack))
            for op, inputs, reference, verify in specifications:
                key = _signature(*inputs)
                if key in self._entries[op]:
                    continue
                if len(self._entries[op]) >= self.max_keys:
                    self._counters["warmup_skips"] += 1
                    continue
                entry = self._entries[op][key] = _Entry()
                try:
                    fn = self._compile(reference)
                    verify(fn, probes)
                except Exception as error:
                    self._fail(op, key, error, "parity_failures" if isinstance(error, CodecParityError)
                               else "compile_failures")
                    continue
                entry.fn, entry.state = fn, "ready"
        return self.stats()

    def stats(self):
        ready = {op: sum(e.state == "ready" for e in entries.values())
                 for op, entries in self._entries.items()}
        return {"schema": SCHEMA, "backend": "injected_test_compiler" if self._compiler is not None
                else "torch_compile_inductor",
                "ready": bool(ready["pack"] and ready["unpack"]),
                "ready_pack": ready["pack"], "ready_unpack": ready["unpack"],
                "ready_keys": sum(ready.values()),
                "compiled_calls": sum(self._counters["uses"].values()),
                "fallback_calls": sum(self._counters["fallbacks"].values()),
                "declined_keys": sum(entry.state == "failed" for entries in self._entries.values()
                                     for entry in entries.values()),
                "uses": dict(self._counters["uses"]), "fallbacks": dict(self._counters["fallbacks"]),
                **{k: v for k, v in self._counters.items() if k not in ("uses", "fallbacks")},
                "declined": {op: [entry.reason for entry in entries.values() if entry.state == "failed"]
                             for op, entries in self._entries.items()}}


_CODEC = WireCodec()


def pack(tensor):
    return _CODEC.pack(tensor)


def unpack(q, scale):
    return _CODEC.unpack(q, scale)


def warmup(tensor):
    return _CODEC.warmup(tensor)


def stats():
    return _CODEC.stats()


def can_unpack(q, scale, *, device=None):
    return _CODEC.can_unpack(q, scale, device=device)

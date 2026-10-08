"""Fail-closed native MXFP4 loading and packed expert-layout verification.

Dependency versions are prerequisites, not proof of native execution. The load
context intercepts the installed HF quantizer's fallback before conversion; the
post-load check inspects real parameter/buffer/custom Triton storage in the
selected layers. No model, GPU, Hub kernel, or network work happens on import.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import functools
import importlib
import math
import threading
import sys
from typing import Any, Optional

_NATIVE = ContextVar("flyai_native_mxfp4_load", default=False)
_LOCK = threading.RLock()


class MXFP4Error(RuntimeError):
    pass


def parse_version_tuple(v_str: str) -> tuple:
    from packaging.version import Version, InvalidVersion
    try:
        release = Version(str(v_str)).release
    except InvalidVersion as exc:
        raise MXFP4Error("unrecognized dependency version") from exc
    return tuple((list(release) + [0, 0, 0])[:3])


def _quantization(config):
    return config.get("quantization_config") if isinstance(config, dict) else getattr(config, "quantization_config", None)


def _value(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def config_uses_mxfp4(config):
    quant = _quantization(config)
    method = _value(quant, "quant_method")
    return getattr(method, "value", method) == "mxfp4"


def _require_native_config(quant):
    if quant is None:
        raise MXFP4Error("MXFP4 quantizer configuration is missing")
    dequantize = _value(quant, "dequantize", False)
    if type(dequantize) is not bool:
        raise MXFP4Error("MXFP4 dequantize setting must be boolean")
    if dequantize:
        raise MXFP4Error("native MXFP4 required: dequantization/fallback refused before weight conversion")


def verify_mxfp4_runtime_environment(enforce: bool = True) -> dict:
    """Check the installed Transformers' own kernel ABI bounds, without claiming execution."""
    errors, versions = [], {}
    modules = {}
    for name in ("transformers", "kernels"):
        try:
            modules[name] = importlib.import_module(name)
        except ImportError:
            errors.append(f"{name} package is unavailable")
        versions[name] = getattr(modules.get(name), "__version__", "missing")
    if len(modules) == 2:
        try:
            bounds = importlib.import_module("transformers.utils.import_utils")
            minimum = bounds.KERNELS_MIN_VERSION
            maximum = bounds.KERNELS_MAX_VERSION
            from packaging.version import Version
            if not Version(str(minimum)) <= Version(str(versions["kernels"])) < Version(str(maximum)):
                errors.append(f"incompatible kernels ABI: installed Transformers requires [{minimum},{maximum})")
        except (ImportError, AttributeError, ValueError, TypeError):
            errors.append("installed Transformers does not expose validated MXFP4 kernel ABI bounds")
    status = {"ok": not errors, **versions, "errors": errors, "native_execution_verified": False,
              "scope": "dependency prerequisites only; loading hooks and packed layout must also pass"}
    if errors and enforce:
        raise MXFP4Error("[mxfp4_guard] prerequisite check failed: " + "; ".join(errors))
    return status


@contextmanager
def native_mxfp4_loading(config, *, quantizer_class=None):
    """Forbid fallback at the quantizer seam before the first expanded weight.

    Installed HF currently selects Mxfp4Dequantize after validate_environment or
    before-weight hooks set dequantize=True. Wrappers are thread-context scoped,
    serialized, and restored even when loading fails. Unsupported hook APIs fail
    closed instead of pretending that a version string establishes safety.
    """
    if not config_uses_mxfp4(config):
        yield
        return
    _require_native_config(_quantization(config))
    if quantizer_class is None:
        try:
            module = importlib.import_module("transformers.quantizers.quantizer_mxfp4")
            quantizer_class = module.Mxfp4HfQuantizer
        except (ImportError, AttributeError) as exc:
            raise MXFP4Error("installed native MXFP4 quantizer adapter is unavailable") from exc
    names = ("validate_environment", "_process_model_before_weight_loading", "get_weight_conversions")
    with _LOCK:
        original = {}
        for name in names:
            method = getattr(quantizer_class, name, None)
            if not callable(method):
                raise MXFP4Error(f"unsupported MXFP4 quantizer loading seam: {name}")
            original[name] = method

        def wrap(name, method):
            @functools.wraps(method)
            def guarded(self, *args, **kwargs):
                if not _NATIVE.get():
                    return method(self, *args, **kwargs)
                _require_native_config(getattr(self, "quantization_config", None))
                result = method(self, *args, **kwargs)
                _require_native_config(getattr(self, "quantization_config", None))
                if name == "_process_model_before_weight_loading":
                    model = args[0] if args else kwargs.get("model")
                    _assert_native_expert_placeholders(model)
                if name == "get_weight_conversions":
                    for conversion in result or []:
                        for operation in getattr(conversion, "operations", []) or []:
                            if "dequant" in type(operation).__name__.lower():
                                raise MXFP4Error("expanded MXFP4 conversion operation refused before loading")
                return result
            return guarded

        token = _NATIVE.set(True)
        try:
            for name, method in original.items():
                setattr(quantizer_class, name, wrap(name, method))
            yield
            _require_native_config(_quantization(config))
        finally:
            for name, method in original.items():
                setattr(quantizer_class, name, method)
            _NATIVE.reset(token)


def _assert_native_expert_placeholders(model):
    """Catch an excluded/unconverted dense expert BEFORE checkpoint loading."""
    import torch
    if model is None or not callable(getattr(model, "named_modules", None)):
        raise MXFP4Error("MXFP4 pre-load model is not inspectable")
    found = False
    for _name, module in model.named_modules():
        if not (hasattr(module, "gate_up_proj") and hasattr(module, "down_proj")):
            continue
        found = True
        for projection in ("gate_up_proj", "down_proj"):
            value = getattr(module, projection)
            if torch.is_tensor(value):
                if value.dtype != torch.uint8:
                    raise MXFP4Error("unconverted dense expert refused before checkpoint weight loading")
            elif not _fp4_dtype(value):
                raise MXFP4Error("unknown expert placeholder refused before checkpoint weight loading")
    if not found:
        raise MXFP4Error("MXFP4 adapter installed no native expert modules")


def _positive(value, label):
    if type(value) is not int or value < 1:
        raise MXFP4Error(f"invalid expert {label}")
    return value


def _at(model, path):
    value = model
    try:
        for part in filter(None, path.split(".")):
            value = getattr(value, part)
    except AttributeError as exc:
        raise MXFP4Error("assigned decoder layer path is unavailable") from exc
    return value


def _layers(model, layer_path):
    if layer_path is not None:
        return _at(model, layer_path)
    for path in ("model.layers", "layers"):
        try:
            return _at(model, path)
        except MXFP4Error:
            pass
    raise MXFP4Error("decoder layer list is missing; native layout was not verified")


def _tensor(value):
    import torch
    if torch.is_tensor(value):
        return value, False
    storage = getattr(value, "storage", None)
    data = getattr(storage, "data", None)
    if torch.is_tensor(data):
        return data, True
    raise MXFP4Error("expert weight/scale has no inspectable tensor storage")


def _fp4_dtype(value):
    dtype = getattr(value, "dtype", None)
    # Use the already-loaded kernel's canonical datatype even when its repr is
    # an opaque Python object. Never import/download/initialize a Hub kernel.
    integration = sys.modules.get("transformers.integrations.mxfp4")
    hub = getattr(integration, "triton_kernels_hub", None)
    known = getattr(getattr(hub, "tensor", None), "FP4", None)
    return known is not None and dtype is known or "fp4" in str(dtype).lower()


def _device(tensor, expected):
    import torch
    if tensor.device.type == "meta":
        raise MXFP4Error("assigned expert weight/scale is still meta")
    if expected is not None:
        wanted = torch.device(expected)
        if tensor.device.type != wanted.type or wanted.index is not None and tensor.device.index != wanted.index:
            raise MXFP4Error("expert weight/scale was loaded on the wrong assigned device")


def _packed_projection(module, proj, experts, hidden, intermediate, expected_device):
    import torch
    raw = getattr(module, proj, None)
    if raw is None:
        raise MXFP4Error(f"missing packed expert {proj}")
    weight, wrapped = _tensor(raw)
    _device(weight, expected_device)
    if weight.dtype != torch.uint8:
        raise MXFP4Error(f"{proj} is expanded/non-MXFP4 storage ({weight.dtype})")
    logical = (experts, hidden, 2 * intermediate) if proj == "gate_up_proj" else (experts, intermediate, hidden)
    packed_bytes = math.prod(logical) // 2
    padded_bytes = experts * math.ceil(logical[1] / 512) * 512 * math.ceil(logical[2] / 512) * 512 // 2
    if wrapped:
        # HF removes the nn.Parameter and attaches a Triton FP4 Tensor with
        # physical uint8 storage. Its published logical shape is transposed.
        if not _fp4_dtype(raw) or tuple(getattr(raw, "shape", ())) != logical:
            raise MXFP4Error("custom expert tensor is not the expected logical FP4 layout")
        if not packed_bytes <= weight.numel() <= padded_bytes:
            raise MXFP4Error("packed expert storage size is outside its declared logical/padded shape")
        precision = getattr(module, proj + "_precision_config", None)
        scale = getattr(precision, "weight_scale", None)
    else:
        rows, columns = (2 * intermediate, hidden) if proj == "gate_up_proj" else (hidden, intermediate)
        allowed = {(experts, rows, columns // 32, 16), (experts, rows, columns // 2)}
        if tuple(weight.shape) not in allowed or weight.numel() != packed_bytes:
            raise MXFP4Error("raw packed expert shape differs from model dimensions")
        scale = getattr(module, proj + "_scales", None)
        if scale is None:
            precision = getattr(module, proj + "_precision_config", None)
            scale = getattr(precision, "weight_scale", None)
    if scale is None:
        raise MXFP4Error(f"{proj} has no loaded block scales")
    scales, scale_wrapped = _tensor(scale)
    _device(scales, expected_device)
    permitted = {torch.uint8}
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is not None:
        permitted.add(e8m0)
    if scales.dtype not in permitted:
        raise MXFP4Error("MXFP4 scales are not E8M0 byte storage")
    if scales.device != weight.device:
        raise MXFP4Error("expert packed weights and scales are on different devices")
    expected_scales = packed_bytes // 16  # one E8M0 scale per 32 FP4 values
    if wrapped:
        scale_shape = (experts, logical[1] // 32, logical[2])
        if tuple(getattr(scale, "shape", scales.shape)) != scale_shape:
            raise MXFP4Error("custom MXFP4 scale logical shape differs")
        if not expected_scales <= scales.numel() <= padded_bytes // 16:
            raise MXFP4Error("MXFP4 scale storage size differs")
    else:
        rows, columns = (2 * intermediate, hidden) if proj == "gate_up_proj" else (hidden, intermediate)
        if tuple(scales.shape) != (experts, rows, columns // 32) or scales.numel() != expected_scales:
            raise MXFP4Error("raw MXFP4 scale shape differs from packed weights")
    return weight.numel() + scales.numel() * scales.element_size()


def inspect_mxfp4_quantized(model: Any, *, layer_range=None, layer_path=None, expected_device=None):
    """Verify GPT-OSS expert packed layouts, selected range, parameters and buffers.

    The CPU test seam verifies storage without claiming kernel execution. Unknown
    expert layouts fail closed and need an explicit qualified adapter.
    """
    if model is None or not hasattr(model, "config") or not config_uses_mxfp4(model.config):
        raise MXFP4Error("a real MXFP4 model/config is required for layout verification")
    _require_native_config(_quantization(model.config))
    quantizer = getattr(model, "hf_quantizer", None)
    if quantizer is not None:
        _require_native_config(getattr(quantizer, "quantization_config", None))
    layers = _layers(model, layer_path)
    try:
        total = len(layers)
    except TypeError as exc:
        raise MXFP4Error("decoder layers are not an inspectable sequence") from exc
    declared_total = _value(model.config, "num_hidden_layers")
    if declared_total is not None and (type(declared_total) is not int or declared_total != total):
        raise MXFP4Error("actual decoder layer count differs from checkpoint configuration")
    lo, hi = layer_range if layer_range is not None else (0, total)
    if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= total:
        raise MXFP4Error("invalid assigned MXFP4 layer range")
    observed, storage_bytes = [], 0
    for index, layer in enumerate(layers):
        candidates = [(name, module) for name, module in layer.named_modules()
                      if name.split(".")[-1] == "experts" or
                      hasattr(module, "gate_up_proj") and hasattr(module, "down_proj")]
        if not lo <= index < hi:
            # Meta placement must not accidentally load the unused model tail.
            for _, tensor in list(layer.named_parameters()) + list(layer.named_buffers()):
                if tensor.numel() and tensor.device.type != "meta":
                    raise MXFP4Error("unassigned decoder layer was materialized")
            for _, module in candidates:
                for proj in ("gate_up_proj", "down_proj"):
                    value = getattr(module, proj, None)
                    if value is not None:
                        data, _ = _tensor(value)
                        if data.numel() and data.device.type != "meta":
                            raise MXFP4Error("unassigned custom expert tensor was materialized")
                    scale = getattr(module, proj + "_scales", None)
                    if scale is None:
                        scale = getattr(getattr(module, proj + "_precision_config", None), "weight_scale", None)
                    if scale is not None:
                        data, _ = _tensor(scale)
                        if data.numel() and data.device.type != "meta":
                            raise MXFP4Error("unassigned custom expert scales were materialized")
            continue
        if not candidates:
            raise MXFP4Error(f"assigned layer {index} has no inspectable MXFP4 experts")
        for name, module in candidates:
            experts = _positive(getattr(module, "num_experts", _value(model.config, "num_local_experts")), "count")
            hidden = _positive(getattr(module, "hidden_size", _value(model.config, "hidden_size")), "hidden size")
            intermediate = _positive(getattr(module, "intermediate_size", _value(model.config, "intermediate_size")), "intermediate size")
            if hidden % 32 or intermediate % 32:
                raise MXFP4Error("MXFP4 expert dimensions require blocks of 32")
            for field, actual in (("num_local_experts", experts), ("hidden_size", hidden), ("intermediate_size", intermediate)):
                expected = _value(model.config, field)
                if expected is not None and expected != actual:
                    raise MXFP4Error("expert dimensions differ from model configuration")
            for proj in ("gate_up_proj", "down_proj"):
                storage_bytes += _packed_projection(module, proj, experts, hidden, intermediate, expected_device)
            # Inspect registered parameters AND buffers: ordinary biases may be
            # FP32, but other expanded expert matrices may not hide beside FP4.
            for tensor_name, tensor in list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False)):
                bias_shapes = {"gate_up_proj_bias": (experts, 2 * intermediate),
                               "down_proj_bias": (experts, hidden)}
                if tensor_name in bias_shapes:
                    if tuple(tensor.shape) != bias_shapes[tensor_name] or tensor.device.type == "meta":
                        raise MXFP4Error("expert bias differs from its declared shape/device")
                    continue
                if tensor_name in ("gate_up_proj_scales", "down_proj_scales"):
                    continue
                if tensor.numel() and tensor.is_floating_point():
                    raise MXFP4Error("expanded floating-point expert weight remained registered")
            if getattr(module, "_is_dequantized", False) is True:
                raise MXFP4Error("expert module is explicitly marked dequantized")
            observed.append({"layer": index, "module": name})
    return {"packed_layout_verified": True, "native_execution_verified": False, "layer_range": [lo, hi],
            "experts": observed, "packed_weight_scale_bytes": storage_bytes,
            "scope": "loaded storage layout only; kernel output and performance require separate runtime validation"}


def assert_mxfp4_quantized(model: Any, model_id: Optional[str] = None, *, layer_range=None,
                            layer_path=None, expected_device=None) -> bool:
    report = inspect_mxfp4_quantized(model, layer_range=layer_range, layer_path=layer_path, expected_device=expected_device)
    print(f"[mxfp4_guard] packed expert layout verified for layers {report['layer_range']}; kernel execution unverified",
          flush=True)
    return True

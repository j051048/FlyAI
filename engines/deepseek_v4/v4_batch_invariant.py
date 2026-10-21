"""Request-axis numerical execution domain for native one-token batches.

The original scalar model is deliberately untouched. BLAS is allowed to choose
different reductions when the token-row count changes; this domain fixes each
projection/einsum's request-row shape while the Stage, routing, cache updates
and attention still execute one actual request-axis batch. CUDA capture records
these bounded row projections once. This is not a grouped GEMM optimization.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import sys

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

CONTRACT = "v4-native-request-row-math/1"
_ACTIVE = ContextVar("v4_request_row_math", default=False)

# CPU vector math can change approximations at SIMD-width boundaries as well
# as BLAS changing GEMM reductions. In particular a hash Gate's [1, 8]
# softplus takes a different implementation from [4, 8]. Keep the complete
# native row shape for these small-width functions; changing only linear
# fixes the original 17-token fixture but still drifts on longer trajectories.
_UNARY = {F.softplus, F.silu, torch.sigmoid, torch.Tensor.sigmoid,
          torch.sqrt, torch.Tensor.sqrt, torch.rsqrt, torch.Tensor.rsqrt,
          torch.exp, torch.Tensor.exp, torch.tanh, torch.Tensor.tanh}
_REDUCTION = {torch.sum, torch.Tensor.sum, torch.mean, torch.Tensor.mean,
              torch.amax, torch.Tensor.amax, torch.softmax, torch.Tensor.softmax,
              torch.log_softmax, torch.Tensor.log_softmax, F.softmax, F.log_softmax}
_ROW_FUNCTIONS = _UNARY | _REDUCTION


def prepare_native_row_math(module=None):
    """Install dormant kernel adapters; scalar calls pass through unchanged.

    TileLang kernels are Python entry points rather than torch-dispatch ops.
    Their activation row count must be fixed as well as F.linear's. No weight
    is dequantized or converted: each actual backend sees the original packed
    weights/scales and exactly one activation row. An absent native module is
    harmless for isolated mathematical tests.
    """
    module = module or sys.modules.get("dsv4_model")
    if module is None:
        return
    for name in ("fp4_gemm", "fp8_gemm"):
        original = getattr(module, name)
        if getattr(original, "_v4_request_row_math", False):
            continue

        @wraps(original)
        def fixed_rows(a, a_s, weight, weight_scale, *args, _original=original, **kwargs):
            if not _ACTIVE.get():
                return _original(a, a_s, weight, weight_scale, *args, **kwargs)
            rows = a.reshape(-1, a.shape[-1])
            if rows.shape[0] <= 1:
                return _original(a, a_s, weight, weight_scale, *args, **kwargs)
            scales = a_s.reshape(rows.shape[0], -1)
            return torch.cat([_original(rows[i:i + 1], scales[i:i + 1], weight,
                                       weight_scale, *args, **kwargs)
                              for i in range(rows.shape[0])], 0).reshape(*a.shape[:-1], weight.shape[0])

        fixed_rows._v4_request_row_math = True
        setattr(module, name, fixed_rows)


class RequestRowMath(TorchFunctionMode):
    """Keep row reductions independent of the number/order of other requests."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is F.linear:
            x = args[0] if args else kwargs["input"]
            weight = args[1] if len(args) > 1 else kwargs["weight"]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            if x.shape[-1] == 0:
                return func(*args, **kwargs)
            rows = x.reshape(-1, x.shape[-1])
            if rows.shape[0] > 1:
                return torch.cat([F.linear(rows[i:i + 1], weight, bias)
                                  for i in range(rows.shape[0])], 0).reshape(*x.shape[:-1], weight.shape[0])
        elif func is torch.einsum and isinstance(args[0], str):
            equation, *operands = args
            if "->" not in equation or not all(torch.is_tensor(value) for value in operands):
                return func(*args, **kwargs)
            left, right = equation.replace(" ", "").split("->")
            labels = left.split(",")
            # All native contractions use b as the request axis. Weight-only
            # operands have no b; shared weights are never replicated.
            if "b" in right and all("b" not in label or label.startswith("b") for label in labels):
                count = next((value.shape[0] for label, value in zip(labels, operands) if "b" in label), 1)
                if count > 1:
                    values = [torch.einsum(equation, *[value[i:i + 1] if "b" in label else value
                              for label, value in zip(labels, operands)], **kwargs) for i in range(count)]
                    return torch.cat(values, dim=right.index("b"))
        elif func in (torch.topk, torch.Tensor.topk):
            x = args[0]
            k = args[1] if len(args) > 1 else kwargs["k"]
            dim = args[2] if len(args) > 2 else kwargs.get("dim", -1)
            if x.ndim > 1 and x.shape[-1] > 0 and dim % x.ndim == x.ndim - 1 and kwargs.get("out") is None:
                rows = x.reshape(-1, x.shape[-1])
                if rows.shape[0] > 1:
                    largest = args[3] if len(args) > 3 else kwargs.get("largest", True)
                    sorted_ = args[4] if len(args) > 4 else kwargs.get("sorted", True)
                    values = [torch.topk(rows[i:i + 1], k, -1, largest, sorted_)
                              for i in range(rows.shape[0])]
                    shape = (*x.shape[:-1], k)
                    return torch.return_types.topk((torch.cat([v.values for v in values], 0).reshape(shape),
                                                   torch.cat([v.indices for v in values], 0).reshape(shape)))
        elif func in _ROW_FUNCTIONS and args:
            x = args[0]
            if torch.is_tensor(x) and x.ndim > 1 and x.shape[0] > 1:
                split = func in _UNARY
                if not split:
                    dim = args[1] if len(args) > 1 else kwargs.get("dim")
                    dims = (dim,) if isinstance(dim, int) else dim
                    # A true aggregate over requests must keep its semantics.
                    split = bool(dims) and all(value % x.ndim != 0 for value in dims)
                if split:
                    return torch.cat([func(x[i:i + 1], *args[1:], **kwargs)
                                      for i in range(x.shape[0])], 0)
        return func(*args, **kwargs)


@contextmanager
def request_row_math():
    """Thread-local scope; dormant adapters preserve all out-of-scope calls."""
    prepare_native_row_math()
    token = _ACTIVE.set(True)
    try:
        with RequestRowMath():
            yield
    finally:
        _ACTIVE.reset(token)

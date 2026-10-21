"""Measured context/layer cost surfaces; predictions never grant an SLO proof.

Full-span wall measurements price the stage. CUDA layer intervals provide
placement weights, not isolated layer wall latency. No unseen frame width or
larger context is extrapolated.
"""
from copy import deepcopy
import math


def _positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 3_600_000:
        raise ValueError("cost surface needs finite positive measurements")


def validate_surface(trace):
    surface = trace.get("cost_surface")
    if surface is None:
        return None
    if (trace.get("schema") != "shard-stage-trace/3" or not isinstance(surface, dict)
            or set(surface) != {"schema", "source_sha256", "samples"}
            or surface["schema"] != "shard-stage-cost-surface/1"):
        raise ValueError("invalid stage cost surface")
    sha = surface["source_sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("stage cost surface needs exact source digest")
    samples = surface["samples"]
    if not isinstance(samples, list) or not 1 <= len(samples) <= 128:
        raise ValueError("bounded stage cost samples required")
    depth = trace["layer_end"] - trace["layer_start"]
    seen = set()
    for row in samples:
        if (not isinstance(row, dict) or set(row) != {"context_tokens", "frame_tokens", "warmness",
                "frame_samples_ms", "prefill_samples_ms", "layer_weights"}):
            raise ValueError("invalid stage cost shape")
        for name, high in (("context_tokens", 1_000_000), ("frame_tokens", 256)):
            if type(row[name]) is not int or not 1 <= row[name] <= high:
                raise ValueError(f"invalid cost surface {name}")
        if row["warmness"] not in ("cold", "warm"):
            raise ValueError("invalid cost warmness")
        shape = row["context_tokens"], row["frame_tokens"], row["warmness"]
        if shape in seen:
            raise ValueError("duplicate stage cost shape")
        seen.add(shape)
        for name in ("frame_samples_ms", "prefill_samples_ms"):
            values = row[name]
            if name == "prefill_samples_ms" and values is None:
                continue
            if not isinstance(values, list) or not 5 <= len(values) <= 128:
                raise ValueError("cost shape requires 5..128 measured samples")
            for value in values:
                _positive(value)
        weights = row["layer_weights"]
        if not isinstance(weights, list) or len(weights) != depth or not 1 <= depth <= 256:
            raise ValueError("one measured cost weight per owned layer required")
        for weight in weights:
            _positive(weight)
    return surface


def shape_trace(trace, workload, *, start, layers, source_sha256=None):
    surface = validate_surface(trace)
    if surface is None:
        return None
    if source_sha256 is None:
        raise ValueError("stage cost surface requires the node runtime source binding")
    if surface["source_sha256"] != source_sha256:
        raise ValueError("stage cost source differs from node runtime")
    context = (workload or {}).get("context_tokens")
    frame = (workload or {}).get("frame_tokens", 1)
    warmness = (workload or {}).get("warmness")
    if type(context) is not int or context < 1 or warmness not in ("cold", "warm"):
        return None
    samples = [r for r in surface["samples"] if r["context_tokens"] >= context
               and r["frame_tokens"] == frame and r["warmness"] == warmness]
    if not samples:
        return None
    sample = min(samples, key=lambda r: r["context_tokens"])
    lo, hi = trace["layer_start"], trace["layer_end"]
    if start is None:
        start = lo
    if not lo <= start < start + layers <= hi:
        return None
    local = deepcopy(trace)
    local.update(context_tokens=sample["context_tokens"], frame_tokens=frame, warmness=warmness,
                 frame_ms=sum(sample["frame_samples_ms"]) / len(sample["frame_samples_ms"]))
    pf = sample["prefill_samples_ms"]
    local["prefill_ms"] = sum(pf) / len(pf) if pf else None
    local.pop("quality", None)  # No percentile from a different workload shape.
    if trace.get("shared_resource_id"):
        # No matching DMA contention sample exists for the new shape. Serialise
        # its complete predicted frame rather than reuse another context's busy
        # interval and silently claim additional shared-I/O overlap.
        local["shared_busy_ms"] = local["frame_ms"]
    weights = sample["layer_weights"]
    scale = sum(weights[start-lo:start-lo+layers]) / sum(weights)
    exact = start == lo and start + layers == hi and context == sample["context_tokens"]
    return local, scale, exact

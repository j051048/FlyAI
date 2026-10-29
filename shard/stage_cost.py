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
            or surface.get("schema") not in ("shard-stage-cost-surface/1", "shard-stage-cost-surface/2")):
        raise ValueError("invalid stage cost surface")
    role_aware = surface["schema"] == "shard-stage-cost-surface/2"
    expected = {"schema", "source_sha256", "samples"} | ({"decode_mode", "role"} if role_aware else set())
    if set(surface) != expected:
        raise ValueError("invalid stage cost surface")
    if role_aware:
        validate_role(surface["role"], surface["decode_mode"])
    sha = surface["source_sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("stage cost surface needs exact source digest")
    samples = surface["samples"]
    if not isinstance(samples, list) or not 1 <= len(samples) <= 128:
        raise ValueError("bounded stage cost samples required")
    depth = trace["layer_end"] - trace["layer_start"]
    seen = set()
    for row in samples:
        fields = {"context_tokens", "frame_tokens", "warmness", "frame_samples_ms", "prefill_samples_ms", "layer_weights"}
        if role_aware:
            fields |= {"component_samples_ms", "prefill_component_samples_ms"}
        if not isinstance(row, dict) or set(row) != fields:
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
        if role_aware:
            validate_components(row["component_samples_ms"], row["frame_samples_ms"], surface["role"])
            if row["prefill_samples_ms"] is None:
                if row["prefill_component_samples_ms"] is not None:
                    raise ValueError("prefill component costs require measured prefill samples")
            else:
                validate_components(row["prefill_component_samples_ms"], row["prefill_samples_ms"], surface["role"])
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
    if surface["schema"] == "shard-stage-cost-surface/2":
        if ((workload or {}).get("decode_mode") != surface["decode_mode"]
                or (workload or {}).get("role") != surface["role"]):
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
    if surface["schema"] == "shard-stage-cost-surface/2":
        local["_role_cost_prediction"] = price_components(sample, scale)
    return local, scale, exact


COMPONENTS = ("main", "embedding", "lm_head", "draft", "dma", "sync")
MODES = ("greedy", "spec", "dspark", "pipelined")


def validate_role(role, mode):
    if (mode not in MODES or not isinstance(role, dict) or set(role) != {"head", "tail", "draft"}
            or any(type(value) is not bool for value in role.values())
            or role["draft"] and (not role["tail"] or mode not in ("dspark", "pipelined"))):
        raise ValueError("explicit decode mode and head/tail/draft role required")
    return role


def validate_components(components, totals, role):
    if not isinstance(components, dict) or set(components) != set(COMPONENTS):
        raise ValueError("complete non-overlapping role cost components required")
    for name, values in components.items():
        if not isinstance(values, list) or len(values) != len(totals):
            raise ValueError("component intervals must cover each measured frame")
        for value in values:
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 3_600_000:
                raise ValueError("finite nonnegative component cost required")
        active = role["head"] if name == "embedding" else role["tail"] if name == "lm_head" else role["draft"] if name == "draft" else True
        if not active and any(values):
            raise ValueError("cost charged to an absent stage role")
    for index, total in enumerate(totals):
        if components["main"][index] <= 0 or not math.isclose(
                sum(components[name][index] for name in COMPONENTS), total, rel_tol=1e-7, abs_tol=1e-6):
            raise ValueError("component partitions must sum to the real measured frame")
    return components


def price_components(sample, scale):
    """Keep every measured fixed role/DMA/sync cost; scale only main layers."""
    def price(name):
        values = sample[name]
        if values is None:
            return None
        means = {key: sum(part) / len(part) for key, part in values.items()}
        fixed = sum(value for key, value in means.items() if key != "main")
        return {"main_ms": means["main"] * scale, "fixed_ms": fixed,
                "total_ms": means["main"] * scale + fixed, "components_ms": means}
    return {"frame": price("component_samples_ms"), "prefill": price("prefill_component_samples_ms")}


def surface_from_frames(frames, *, source_sha256):
    """Export explicit completion-fenced wall partitions, never nested GPU sums."""
    if not isinstance(frames, list) or not 5 <= len(frames) <= 128:
        raise ValueError("5..128 complete calibration frames required")
    first = frames[0]
    binding = (first["decode_mode"], first["role"], first["layer_start"], first["layer_end"])
    shapes = {}
    for row in frames:
        validate_role(row["role"], row["decode_mode"])
        if (row["decode_mode"], row["role"], row["layer_start"], row["layer_end"]) != binding:
            raise ValueError("role calibration cannot mix modes, roles or layer spans")
        key = row["context_tokens"], row["frame_tokens"], row["warmness"]
        shapes.setdefault(key, []).append(row)
    samples = []
    for (context, width, warmness), rows in shapes.items():
        if len(rows) < 5:
            continue
        weights = rows[0]["layer_weights"]
        if any(row["layer_weights"] != weights for row in rows):
            raise ValueError("one frozen layer weight calibration per shape required")
        samples.append(dict(context_tokens=context, frame_tokens=width, warmness=warmness,
            frame_samples_ms=[row["frame_ms"] for row in rows], prefill_samples_ms=None,
            layer_weights=list(weights),
            component_samples_ms={name: [row["components_ms"][name] for row in rows] for name in COMPONENTS},
            prefill_component_samples_ms=None))
    result = {"schema": "shard-stage-cost-surface/2", "source_sha256": source_sha256,
              "decode_mode": first["decode_mode"], "role": deepcopy(first["role"]), "samples": samples}
    validate_surface({"schema": "shard-stage-trace/3", "layer_start": first["layer_start"],
                      "layer_end": first["layer_end"], "cost_surface": result})
    return result

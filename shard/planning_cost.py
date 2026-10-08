"""Prediction-only finite-window pipeline costs from explicit observations.

Serial latency and saturated throughput are different quantities. This model
simulates FIFO stage/link/shared-resource calendars with bounded in-flight work,
feedback, fill/drain and optional rejection/replay. It neither executes a model
nor certifies a hardware SLO. Speculative rejection distribution is not recoverable
from mean acceptance alone; absent a trace, draining each round is conservative.
"""
import math
from .locality import number, timestamp


def _integer(value, name, minimum=1, maximum=1_000_000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return value


def workload_spec(value):
    value = dict(value or {})
    depth = _integer(value.get("depth", 1), "depth", maximum=256)
    block = _integer(value.get("block_tokens", depth), "block_tokens", maximum=256)
    gain = number(value.get("acceptance_gain", 1.0), "acceptance_gain", minimum=1)
    if gain > block:
        raise ValueError("acceptance_gain cannot exceed block_tokens")
    cancel = value.get("cancel_rate")
    if cancel is not None and number(cancel, "cancel_rate") > 1:
        raise ValueError("cancel_rate must be <=1")
    return {"depth": depth, "block_tokens": block, "acceptance_gain": gain,
            "generated_tokens": _integer(value.get("generated_tokens", 256), "generated_tokens"),
            "frame_tokens": _integer(value.get("frame_tokens", 1), "frame_tokens", maximum=256),
            "cancel_rate": cancel, "refill_ms": number(value.get("refill_ms", 0), "refill_ms"),
            "replay_frames_per_cancel": number(value.get("replay_frames_per_cancel", 0), "replay_frames_per_cancel")}


def stage_observation(node, layers, fallback_ms, *, now):
    trace = node.get("stage_trace")
    warning = []
    if trace is not None:
        if not isinstance(trace, dict) or trace.get("schema") != "shard-stage-trace/1":
            raise ValueError("unsupported stage trace schema")
        age = now - timestamp(trace["measured_at"])
        ttl = number(trace["ttl_s"], "trace ttl_s", minimum=1e-9)
        lo, hi = trace["layer_start"], trace["layer_end"]
        _integer(lo, "layer_start", minimum=0)
        _integer(hi, "layer_end")
        if hi <= lo:
            raise ValueError("trace layer range must be nonempty")
        frame = number(trace["frame_ms"], "frame_ms", minimum=1e-9)
        shared = number(trace.get("shared_busy_ms", 0), "shared_busy_ms")
        if shared > frame:
            raise ValueError("shared busy time must be included in frame_ms")
        queue = number(trace.get("queue_ms", 0), "queue_ms")
        prefill = trace.get("prefill_ms")
        if prefill is not None:
            prefill = number(prefill, "prefill_ms")
        if -30 <= age <= ttl:
            scale = layers / (hi - lo)
            return {"service_ms": frame * scale, "queue_ms": queue,
                    "shared_ms": shared * scale, "shared_id": trace.get("shared_resource_id"),
                    "prefill_ms": prefill * scale if prefill is not None else None,
                    "source": "fresh_stage_trace" if scale == 1 else "scaled_stage_trace",
                    "uncertainty": [] if scale == 1 else [f"{node['id']}: stage trace scaled to another block size"]}
        warning.append(f"{node['id']}: expired stage trace")
    warning.append(f"{node['id']}: scalar stage service estimate; no fresh matching stage trace")
    return {"service_ms": layers * fallback_ms, "queue_ms": 0, "shared_ms": 0,
            "shared_id": None, "prefill_ms": None, "source": "scalar_estimate", "uncertainty": warning}


def simulate_frames(stages, hops, transfer, entry_ms, return_ms, depth, frames):
    """Deterministic resource calendar, complete feedback and finite window.

    Shared busy time is inside a stage's measured service interval, not another
    additive DMA penalty. Other stages sharing the resource cannot overlap that
    interval. Link serialization excludes propagation latency.
    """
    available = [s["queue_ms"] for s in stages]
    links = [0.0] * max(0, len(stages) - 1)
    shared, replies = {}, []
    for frame in range(frames):
        arrival = (replies[frame - depth] if frame >= depth else 0.0) + entry_ms
        for i, stage in enumerate(stages):
            domain = stage["shared_id"]
            start = max(arrival, available[i], shared.get(domain, 0) if domain is not None else 0)
            if domain is not None:
                shared[domain] = start + stage["shared_ms"]
            end = start + stage["service_ms"]
            available[i] = end
            if i + 1 < len(stages):
                tx = max(end, links[i])
                links[i] = tx + transfer[i]
                arrival = links[i] + hops[i]
            else:
                replies.append(end + return_ms)
    return replies


def estimate(order, alloc, layer_ms, L, c_out, c_in, nodes, model, workload,
             *, edges=None, now, objective="pipeline"):
    w = workload_spec(workload)
    stages = [stage_observation(nodes[n], alloc[n], layer_ms[n], now=now) for n in order]
    warnings = [warning for stage in stages for warning in stage["uncertainty"]]
    payload = number(model.get("decode_bytes", 0), "decode_bytes") * w["frame_tokens"]
    hops, transfers = [], []
    for a, b in zip(order, order[1:]):
        hops.append(L[a][b])
        row = (edges or {}).get((nodes[a]["id"], nodes[b]["id"]), {})
        bw = row.get("bandwidth_mbps") or nodes[a].get("up_mbps")
        if bw is None:
            warnings.append(f"{nodes[a]['id']} -> {nodes[b]['id']}: transfer bandwidth unknown")
            transfers.append(0.0)
        else:
            transfers.append(payload * 8 / (number(bw, "bandwidth_mbps", minimum=1e-9) * 1000))
    entry, back = c_out[order[0]], c_in[order[-1]]
    serial = sum(s["service_ms"] for s in stages) + sum(hops) + sum(transfers) + entry + back
    prefill_values = [s["prefill_ms"] for s in stages]
    if any(value is None for value in prefill_values):
        warnings.append("prefill compute has no complete stage trace")
    prefill = sum(value or 0 for value in prefill_values) + sum(hops) + entry + back
    pfbytes = number(model.get("prefill_bytes", 0), "prefill_bytes")
    for i, (a, b) in enumerate(zip(order, order[1:])):
        if payload:
            prefill += transfers[i] * pfbytes / payload
    if objective == "serial":
        elapsed = serial * w["generated_tokens"] + prefill
        first = serial
    elif objective == "pipeline":
        rounds = math.ceil(w["generated_tokens"] / w["acceptance_gain"])
        # Full acceptance can continue without a cancellation drain. Otherwise
        # explicitly model complete bounded-window drain plus observed replay.
        if w["acceptance_gain"] == w["block_tokens"] and not w["cancel_rate"]:
            total = math.ceil(w["generated_tokens"] / w["frame_tokens"])
            sample = min(total, max(64, w["depth"] * 4))
            replies = simulate_frames(stages, hops, transfers, entry, back, w["depth"], sample)
            elapsed = replies[-1]
            if total > sample:
                period = min(w["depth"], sample - 1)
                cycle = (replies[-1] - replies[-1 - period]) / period
                elapsed += (total - sample) * cycle
                warnings.append("steady-state tail extrapolated from bounded resource-calendar simulation")
        else:
            replies = simulate_frames(stages, hops, transfers, entry, back, w["depth"], w["block_tokens"])
            cancel = 1.0 if w["cancel_rate"] is None else w["cancel_rate"]
            replay = cancel * w["replay_frames_per_cancel"] * serial
            elapsed = rounds * (replies[-1] + replay + w["refill_ms"])
            if w["cancel_rate"] is None:
                warnings.append("acceptance distribution absent; conservative drain-per-round estimate")
        first = replies[0]
        elapsed += prefill
    else:
        raise ValueError("objective must be serial or pipeline")
    worst = max(range(len(stages)), key=lambda i: stages[i]["service_ms"])
    return {"prediction_only": True, "objective": objective, "predicted_request_ms": elapsed,
            "predicted_serial_step_ms": serial, "predicted_first_frame_ms": first,
            "predicted_committed_tok_s": 1000 * w["generated_tokens"] / elapsed if elapsed else None,
            "finite_depth": w["depth"], "acceptance_gain": w["acceptance_gain"],
            "bottleneck": {"node_id": nodes[order[worst]]["id"], "service_ms": stages[worst]["service_ms"]},
            "stages": stages, "uncertainty": sorted(set(warnings)),
            "scope": "resource-calendar prediction; hardware acceptance requires a live verified run"}

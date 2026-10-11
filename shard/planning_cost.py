"""Prediction-only finite-window pipeline costs from explicit observations.

Serial latency and saturated throughput are different quantities. This model
simulates FIFO stage/link/shared-resource calendars with bounded in-flight work,
feedback, fill/drain and optional rejection/replay. It neither executes a model
nor certifies a hardware SLO. Speculative rejection distribution is not recoverable
from mean acceptance alone; absent a trace, draining each round is conservative.
"""
import math
from copy import deepcopy
from .locality import number, timestamp, measurement_quality

SLO_SCHEMA = "shard-planning-slo/1"
_NORMALIZERS = {"ttft_ms": 1000.0, "tpot_ms": 100.0, "request_cost": 1.0, "hourly_cost": 1.0,
    "jitter_ms": 100.0, "packet_loss_rate": 1.0, "observation_age_s": 300.0, "failure_domain_concentration": 1.0}


def slo_spec(value):
    """Opt-in scoring/constraints; component percentiles are not an E2E SLO proof."""
    if value is None:
        return None
    if (not isinstance(value, dict) or value.get("schema") != SLO_SCHEMA or
            set(value)-{"schema", "mode", "targets", "weights", "normalizers", "currency", "unknown_penalty", "failure_domain_types"}):
        raise ValueError("unsupported planning SLO schema/fields")
    mode = value.get("mode", "advisory")
    if mode not in ("advisory", "require_known"):
        raise ValueError("SLO mode must be advisory or require_known")
    targets, weights, scales = value.get("targets", {}), value.get("weights"), dict(_NORMALIZERS)
    if not isinstance(targets, dict) or set(targets)-set(scales):
        raise ValueError("unknown SLO target")
    targets = {key: number(item, key, minimum=1e-9 if key in ("ttft_ms", "tpot_ms") else 0) for key,item in targets.items()}
    weights = {key: 1.0 for key in targets} if weights is None else weights
    if not isinstance(weights, dict) or set(weights)-set(scales):
        raise ValueError("unknown SLO weight")
    weights = {key: number(item, "SLO weight") for key,item in weights.items()}
    if not targets and not any(weights.values()):
        raise ValueError("SLO needs a target or positive scoring weight")
    supplied = value.get("normalizers", {})
    if not isinstance(supplied, dict) or set(supplied)-set(scales):
        raise ValueError("unknown SLO normalizer")
    scales.update({key: number(item, "SLO normalizer", minimum=1e-9) for key,item in supplied.items()})
    for key,target in targets.items():
        if key not in supplied and target > 0:
            scales[key] = target
    for name in ("packet_loss_rate", "failure_domain_concentration"):
        if targets.get(name, 0) > 1:
            raise ValueError(f"{name} target must be <=1")
    currency = value.get("currency", "USD")
    if not isinstance(currency, str) or len(currency) != 3 or not currency.isascii() or not currency.isupper() or not currency.isalpha():
        raise ValueError("SLO uses one explicit three-letter currency; no FX conversion")
    domains = value.get("failure_domain_types", ["host"])
    if (not isinstance(domains, list) or not 1 <= len(domains) <= 16 or
            any(not isinstance(name, str) or not name or len(name) > 64 for name in domains) or len(set(domains)) != len(domains)):
        raise ValueError("bounded explicit failure domain types required")
    return {"schema": SLO_SCHEMA, "mode": mode, "targets": targets, "weights": weights, "normalizers": scales,
            "currency": currency, "unknown_penalty": number(value.get("unknown_penalty", 10), "unknown penalty", minimum=1e-9),
            "failure_domain_types": list(domains)}


def validate_failure_domains(value):
    if (not isinstance(value, dict) or len(value)>16 or any(not isinstance(k,str) or not 1<=len(k)<=64 or
            not isinstance(v,str) or not v.strip() or len(v)>256 for k,v in value.items())):
        raise ValueError("failure_domains must be a bounded explicit identity map")
    return dict(value)


def validate_pricing(value):
    if not isinstance(value,dict) or set(value)!={"schema","currency","per_hour","measured_at","ttl_s"} or value.get("schema")!="shard-node-price/1":
        raise ValueError("versioned node price with measurement/TTL required")
    currency=value["currency"]
    if not isinstance(currency,str) or len(currency)!=3 or not currency.isascii() or not currency.isalpha() or not currency.isupper():
        raise ValueError("invalid node price currency")
    number(value["per_hour"],"per_hour");timestamp(value["measured_at"]);number(value["ttl_s"],"price ttl_s",minimum=1e-9)
    return deepcopy(value)


def _failure_risk(selected, policy):
    """Structural concentration, never outage probability or inferred IP identity."""
    categories, unknown = {}, []
    for kind in policy["failure_domain_types"]:
        counts, missing = {}, []
        for node in selected:
            domains = node.get("failure_domains", {})
            domains=validate_failure_domains(domains)
            domain = domains.get(kind)
            if domain is None and kind == "host":
                domain = node.get("host_id") or node.get("memory_domain_id")
            if domain is None:
                missing.append(node["id"])
            elif not isinstance(domain, str) or not domain.strip():
                raise ValueError("invalid failure-domain identity")
            else:
                counts[domain] = counts.get(domain, 0)+1
        pairs = sum(count*(count-1)//2 for count in counts.values())
        all_pairs = len(selected)*(len(selected)-1)//2
        concentration = pairs/all_pairs if all_pairs and not missing else 0.0 if not missing else None
        categories[kind] = {"known_domain_count": len(counts), "shared_pairs": pairs,
                            "unknown_nodes": missing, "concentration": concentration}
        unknown.extend(f"{node}: {kind} failure domain unknown" for node in missing)
    values = [row["concentration"] for row in categories.values()]
    return {"concentration": max(values) if values and all(value is not None for value in values) else None,
            "domains": categories, "uncertainty": unknown,
            "scope": "structural shared-domain concentration, not failure probability; public IP is not a domain"}


def _prices(selected, now, currency):
    prices, ages, unknown = [], [], []
    for node in selected:
        row = node.get("pricing")
        if row is None:
            unknown.append(f"{node['id']}: node price unknown"); continue
        row=validate_pricing(row)
        price = number(row["per_hour"], "per_hour")
        age = now-timestamp(row["measured_at"])
        ttl = number(row["ttl_s"], "price ttl_s", minimum=1e-9)
        if row["currency"] != currency:
            unknown.append(f"{node['id']}: currency differs; no conversion assumed")
        elif not -30 <= age <= ttl:
            unknown.append(f"{node['id']}: price expired or in future")
        else:
            prices.append(price); ages.append(max(0, age))
    return {"hourly_cost": sum(prices) if len(prices) == len(selected) else None, "ages": ages,
            "uncertainty": unknown, "currency": currency,
            "scope": "advertised selected-node compute allocation; excludes cold load, idle, network, storage and billing terms"}


def _integer(value, name, minimum=1, maximum=1_000_000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return value


def workload_spec(value):
    value = dict(value or {})
    depth = _integer(value.get("depth", 1), "depth", maximum=256)
    frame_tokens = _integer(value.get("frame_tokens", 1), "frame_tokens", maximum=256)
    block = _integer(value.get("block_tokens", frame_tokens if frame_tokens > 1 else depth), "block_tokens", maximum=256)
    gain = number(value.get("acceptance_gain", 1.0), "acceptance_gain", minimum=1)
    if gain > block:
        raise ValueError("acceptance_gain cannot exceed block_tokens")
    cancel = value.get("cancel_rate")
    if cancel is not None and number(cancel, "cancel_rate") > 1:
        raise ValueError("cancel_rate must be <=1")
    if frame_tokens > 1:
        _integer(value.get("context_tokens"), "chunk context_tokens")
        if value.get("warmness") not in ("cold", "warm"):
            raise ValueError("chunk planning requires explicit cold/warm state")
        if value.get("draft_tokens") != frame_tokens - 1 or type(value.get("draft_tokens")) is not int:
            raise ValueError("verify chunk must bind frame_tokens == K+1")
        if cancel is None or "stale_frames_per_valid" not in value:
            raise ValueError("chunk planning requires observed/explicit cancel_rate and stale_frames_per_valid")
        if number(value["stale_frames_per_valid"], "stale_frames_per_valid") > depth - 1:
            raise ValueError("stale frames exceed the finite in-flight suffix")
    return {"depth": depth, "block_tokens": block, "acceptance_gain": gain,
            "generated_tokens": _integer(value.get("generated_tokens", 256), "generated_tokens"),
            "frame_tokens": frame_tokens,
            "context_tokens": value.get("context_tokens"), "warmness": value.get("warmness"),
            "stale_frames_per_valid": value.get("stale_frames_per_valid", 0),
            "cancel_rate": cancel, "refill_ms": number(value.get("refill_ms", 0), "refill_ms"),
            "replay_frames_per_cancel": number(value.get("replay_frames_per_cancel", 0), "replay_frames_per_cancel")}


def stage_observation(node, layers, fallback_ms, *, now, start=None, workload=None,
                      require_fresh_chunk=True, collect_quality=False):
    frame_tokens = (workload or {}).get("frame_tokens", 1)
    trace = node.get("stage_trace")
    warning = []
    if trace is not None:
        if not isinstance(trace, dict) or trace.get("schema") not in ("shard-stage-trace/1", "shard-stage-trace/2"):
            raise ValueError("unsupported stage trace schema")
        age = now - timestamp(trace["measured_at"])
        ttl = number(trace["ttl_s"], "trace ttl_s", minimum=1e-9)
        lo, hi = trace["layer_start"], trace["layer_end"]
        _integer(lo, "layer_start", minimum=0)
        _integer(hi, "layer_end")
        if hi <= lo:
            raise ValueError("trace layer range must be nonempty")
        frame = number(trace["frame_ms"], "frame_ms", minimum=1e-9)
        quality = measurement_quality(trace.get("quality"), "stage")
        if "frame_tokens" in trace:
            _integer(trace["frame_tokens"], "trace frame_tokens", maximum=256)
        shared = number(trace.get("shared_busy_ms", 0), "shared_busy_ms")
        if shared > frame:
            raise ValueError("shared busy time must be included in frame_ms")
        queue = number(trace.get("queue_ms", 0), "queue_ms")
        prefill = trace.get("prefill_ms")
        if prefill is not None:
            prefill = number(prefill, "prefill_ms")
        expected = {"node_id": node["id"], "gpu_uuid": node.get("gpu_uuid"),
                    "cohort_id": node.get("cohort_id"), "frame_tokens": frame_tokens,
                    "runtime_config_sha256": node.get("runtime_config_sha256")}
        if frame_tokens > 1:
            if trace.get("schema") != "shard-stage-trace/2":
                raise ValueError("verify chunk needs version-2 bound stage measurement")
            expected.update(context_tokens=(workload or {}).get("context_tokens"), warmness=(workload or {}).get("warmness"))
        bound = True
        for key, value in expected.items():
            if value is None or trace.get(key) is None:
                bound = False
                warning.append(f"{node['id']}: trace {key} binding unavailable")
            elif trace[key] != value:
                raise ValueError(f"stage trace {key} differs from node/cohort/frame configuration")
        if trace.get("shared_resource_id") is not None and (
                not isinstance(trace["shared_resource_id"], str) or not trace["shared_resource_id"].strip()):
            raise ValueError("shared resource identity must be a nonempty string")
        if shared and not isinstance(trace.get("shared_resource_id"), str):
            raise ValueError("observed shared busy time needs an explicit resource identity")
        if -30 <= age <= ttl and bound:
            scale = layers / (hi - lo)
            matches = scale == 1 and (start is None or lo == start)
            result = {"service_ms": frame * scale, "queue_ms": queue,
                    "shared_ms": shared * scale, "shared_id": trace.get("shared_resource_id"),
                    "prefill_ms": prefill * scale if prefill is not None else None,
                    "source": "fresh_stage_trace" if matches else "scaled_stage_trace",
                    "uncertainty": [] if matches else [f"{node['id']}: stage trace reused for another block/range"]}
            if collect_quality:
                result.update(quality=quality, observation_age_s=max(0,age)+quality.get("window_s",0),
                    exact_bound=matches and (workload or {}).get("context_tokens") is not None and
                    trace.get("context_tokens") == workload["context_tokens"] and
                    (workload or {}).get("warmness") is not None and trace.get("warmness") == workload["warmness"])
            return result
        if not -30 <= age <= ttl:
            warning.append(f"{node['id']}: expired stage trace")
    if frame_tokens > 1 and require_fresh_chunk:
        raise ValueError("verify chunk service cannot be derived from missing/unbound/expired measurement")
    warning.append(f"{node['id']}: scalar stage service estimate; no fresh matching stage trace")
    result = {"service_ms": layers * fallback_ms, "queue_ms": 0, "shared_ms": 0,
            "shared_id": None, "prefill_ms": None, "source": "scalar_estimate", "uncertainty": warning}
    if collect_quality:
        result.update(quality=measurement_quality(None,"stage"), observation_age_s=None, exact_bound=False)
    return result


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
             *, edges=None, now, objective="pipeline", slo=None, coordinator_id=None):
    w = workload_spec(workload)
    stages, start = [], 0
    for n in order:
        stages.append(stage_observation(nodes[n], alloc[n], layer_ms[n], now=now, start=start, workload=w))
        start += alloc[n]
    warnings = [warning for stage in stages for warning in stage["uncertainty"]]
    if not isinstance(workload, dict) or workload.get("context_tokens") is None:
        warnings.append("context/cache warmness is not bound to the stage service observation")
    else:
        context = _integer(workload["context_tokens"], "context_tokens")
        for n in order:
            trace = nodes[n].get("stage_trace")
            if trace and trace.get("context_tokens") != context:
                warnings.append(f"{nodes[n]['id']}: trace context differs/unknown; service extrapolation is unverified")
    payload = number(model.get("wire_bytes_per_token", model.get("decode_bytes", 0)), "decode_bytes") * w["frame_tokens"]
    hops, transfers, bandwidths = [], [], []
    for a, b in zip(order, order[1:]):
        hops.append(L[a][b])
        row = (edges or {}).get((nodes[a]["id"], nodes[b]["id"]), {})
        bw = row.get("bandwidth_mbps") or nodes[a].get("up_mbps")
        if bw is None:
            # Match the legacy residential selector's cautious unknown-uplink
            # floor. Missing bandwidth is not free transport or proof of LAN.
            bw = .5
            warnings.append(f"{nodes[a]['id']} -> {nodes[b]['id']}: transfer bandwidth unknown; conservative 0.5Mbps assumption")
        bandwidths.append(bw)
        transfers.append(payload * 8 / (number(bw, "bandwidth_mbps", minimum=1e-9) * 1000))
    entry, back = c_out[order[0]], c_in[order[-1]]
    serial = sum(s["service_ms"] for s in stages) + sum(hops) + sum(transfers) + entry + back
    prefill_values = [s["prefill_ms"] for s in stages]
    if any(value is None for value in prefill_values):
        warnings.append("prefill compute has no complete stage trace")
    prefill = sum(value or 0 for value in prefill_values) + sum(hops) + entry + back
    pfbytes = number(model.get("prefill_bytes", 0), "prefill_bytes")
    if slo is not None and model.get("wire_bytes_per_token") is not None and w["context_tokens"] is not None:
        pfbytes=number(model["wire_bytes_per_token"],"wire_bytes_per_token")*_integer(w["context_tokens"],"context_tokens")
    for bw in bandwidths:
        if bw is not None:
            prefill += pfbytes * 8 / (number(bw, "bandwidth_mbps", minimum=1e-9) * 1000)
    decode_needed = max(0, w["generated_tokens"] - 1)  # prefill emits the first committed token
    if objective == "serial":
        valid = math.ceil(decode_needed / w["acceptance_gain"]) if w["frame_tokens"] > 1 else decode_needed
        frames = valid
        refill = 0
        if w["frame_tokens"] > 1:
            frames += math.ceil(valid * w["stale_frames_per_valid"])
            frames += math.ceil(valid * w["cancel_rate"] * w["replay_frames_per_cancel"])
            refill = valid * w["cancel_rate"] * w["refill_ms"]
            warnings.append("chunk forecast prices full stale/replay frame work; early-abort savings not assumed")
        elapsed = serial * frames + refill + prefill
        first = serial
    elif objective == "pipeline":
        if w["frame_tokens"] > 1:
            valid_frames = math.ceil(decode_needed / w["acceptance_gain"])
            stale = math.ceil(valid_frames * w["stale_frames_per_valid"])
            replay = math.ceil(valid_frames * w["cancel_rate"] * w["replay_frames_per_cancel"])
            frames = max(1, valid_frames + stale + replay)
            sampled = min(frames, max(64, w["depth"] * 4))
            replies = simulate_frames(stages, hops, transfers, entry, back, w["depth"], sampled)
            elapsed = replies[-1]
            if frames > sampled:
                period = min(w["depth"], sampled - 1)
                elapsed += (frames - sampled) * (replies[-1] - replies[-1-period]) / period
            # Observed cancellation drains/refills are additional synchronization
            # barriers. Full suffix work is conservative when early-abort skips it.
            elapsed += valid_frames * w["cancel_rate"] * w["refill_ms"]
            first = replies[0] if decode_needed else 0
            elapsed = (elapsed if decode_needed else 0) + prefill
            warnings.append("chunk forecast prices full stale/replay frame work; early-abort savings not assumed")
        else:
            elapsed, first, token_warnings = _token_pipeline(stages, hops, transfers, entry, back, w, decode_needed, serial, prefill)
            warnings.extend(token_warnings)
    else:
        raise ValueError("objective must be serial or pipeline")
    constraints = [{"kind": "stage", "node_id": nodes[n]["id"], "service_ms": stages[i]["service_ms"]}
                   for i, n in enumerate(order)]
    constraints += [{"kind": "link_serialization", "src": nodes[a]["id"], "dst": nodes[b]["id"], "service_ms": transfers[i]}
                    for i, (a, b) in enumerate(zip(order, order[1:]))]
    domains = {}
    for stage in stages:
        if stage["shared_id"]:
            domains[stage["shared_id"]] = domains.get(stage["shared_id"], 0) + stage["shared_ms"]
    constraints += [{"kind": "shared_resource", "resource_id": domain, "service_ms": busy} for domain, busy in domains.items()]
    constraints.append({"kind": "feedback_window", "service_ms": serial / w["depth"]})
    same_host = [nodes[n].get("host_id") for n in order]
    if len({h for h in same_host if h}) < len([h for h in same_host if h]) and not domains:
        warnings.append("colocated GPU shared-I/O contention has no observed resource service trace")
    result = {"prediction_only": True, "objective": objective, "predicted_request_ms": elapsed,
            "predicted_serial_step_ms": serial, "predicted_first_frame_ms": first,
            "predicted_committed_tok_s": 1000 * w["generated_tokens"] / elapsed if elapsed else None,
            "finite_depth": w["depth"], "acceptance_gain": w["acceptance_gain"], "frame_tokens": w["frame_tokens"],
            "bottleneck": max(constraints, key=lambda row: row["service_ms"]),
            "stages": stages, "uncertainty": sorted(set(warnings)),
            "scope": "resource-calendar prediction; hardware acceptance requires a live verified run"}
    if slo is not None:
        result["slo"] = assess_slo(result, order, alloc, layer_ms, L, c_out, c_in, nodes, {**model,"prefill_bytes":pfbytes}, workload,
            edges=edges or {}, now=now, objective=objective, policy=slo_spec(slo), coordinator_id=coordinator_id)
        result["selection_score"] = result["slo"]["selection_score"]
    return result


def assess_slo(point, order, alloc, layer_ms, L, c_out, c_in, nodes, model, workload,
               *, edges, now, objective, policy, coordinator_id=None):
    """Observed-component stress scenario; never sum p95s and call it E2E p95.

    Packet loss and jitter remain measured quality penalties: transport timeout,
    TCP retry and correlated outages cannot be inferred from a loss percentage.
    """
    w = workload_spec(workload)
    selected = [nodes[n] for n in order]
    observed, cursor = [], 0
    for n in order:
        observed.append(stage_observation(nodes[n], alloc[n], layer_ms[n], now=now, start=cursor, workload=w, collect_quality=True))
        cursor += alloc[n]
    coordinator = coordinator_id if coordinator_id is not None else nodes[order[0]]["id"]
    route_pairs = [(nodes[a]["id"],nodes[b]["id"]) for a,b in zip(order,order[1:])]
    route_pairs += [(coordinator,nodes[order[0]]["id"]),(nodes[order[-1]]["id"],coordinator)]
    route_pairs = list(dict.fromkeys(pair for pair in route_pairs if pair[0] != pair[1]))
    routes = [edges.get(pair, {}) for pair in route_pairs]
    metrics = dict.fromkeys(_NORMALIZERS)
    unknown = []
    def complete(values):
        return all(value is not None for value in values)
    for key in ("jitter_ms", "packet_loss_rate"):
        values = [row.get("quality",{}).get(key) for row in routes]
        metrics[key] = max(values, default=0.0) if complete(values) else None
    ages = [row["observation_age_s"] for row in observed]
    for row in routes:
        ages.append(max(0,now-row["measured_at"])+row.get("quality",{}).get("window_s",0) if row.get("source") == "fresh_sparse" else None)
    metrics["observation_age_s"] = max(ages,default=0.0) if complete(ages) else None
    risk = _failure_risk(selected, policy)
    metrics["failure_domain_concentration"] = risk["concentration"]
    prices = _prices(selected, now, policy["currency"])
    metrics["hourly_cost"] = prices["hourly_cost"]
    # Missing serialization bandwidth cannot turn a cautious legacy point
    # assumption into a known target forecast. Zero payload needs no bandwidth.
    payload = number(model.get("wire_bytes_per_token", model.get("decode_bytes",0)), "decode_bytes")
    pfbytes = number(model.get("prefill_bytes",0), "prefill_bytes")
    prefill_geometry_known=not pfbytes or model.get("wire_bytes_per_token") is not None or model.get("prefill_context_tokens")==w["context_tokens"]
    middle_routes = [edges.get((nodes[a]["id"],nodes[b]["id"]),{}) for a,b in zip(order,order[1:])]
    decode_bw_known = not payload or all(row.get("bandwidth_mbps") is not None for row in middle_routes)
    prefill_bw_known = not pfbytes or all(row.get("bandwidth_mbps") is not None for row in middle_routes)
    exact_stages = all(row["exact_bound"] for row in observed)
    stage_service = all(row["quality"].get("frame_p95_ms") is not None for row in observed)
    stage_prefill = all(row["quality"].get("prefill_p95_ms") is not None and row["quality"].get("queue_p95_ms") is not None for row in observed)
    known_links = all(row.get("source") == "fresh_sparse" and row.get("quality",{}).get("latency_p95_ms") is not None for row in routes)
    def tail_hop(a,b):
        if a == b:return 0.0
        row=edges[a,b]
        p95=row["quality"]["latency_p95_ms"]
        if row.get("hop_policy")=="half_rtt_assumption":p95/=2
        return max(row.get("hop_ms",row["rtt_ms"]),p95)
    if exact_stages and stage_prefill and known_links and prefill_bw_known and prefill_geometry_known:
        prefill_stages=[dict(service_ms=max(row["prefill_ms"] or 0,row["quality"]["prefill_p95_ms"]),
            queue_ms=max(row["queue_ms"],row["quality"]["queue_p95_ms"]),shared_id=None,shared_ms=0) for row in observed]
        prefill_transfer=[pfbytes*8/(row["bandwidth_mbps"]*1000) if pfbytes else 0 for row in middle_routes]
        metrics["ttft_ms"]=simulate_frames(prefill_stages,[tail_hop(nodes[a]["id"],nodes[b]["id"]) for a,b in zip(order,order[1:])],
            prefill_transfer,tail_hop(coordinator,nodes[order[0]]["id"]),tail_hop(nodes[order[-1]]["id"],coordinator),1,1)[0]
    stress = None
    if exact_stages and stage_service and known_links and decode_bw_known:
        stress_nodes = {n: deepcopy(nodes[n]) for n in order}
        for n,row in zip(order,observed):
            trace = stress_nodes[n]["stage_trace"]
            trace["frame_ms"] = max(trace["frame_ms"], row["quality"]["frame_p95_ms"])
            if row["shared_id"] is not None:
                # No observed shared-I/O percentile: serialize the whole tail
                # stage interval instead of treating its mean DMA as a bound.
                trace["shared_busy_ms"] = trace["frame_ms"]
            trace["prefill_ms"] = 0  # internal decode-only calendar; no claim of measured zero prefill
            trace["queue_ms"] = 0  # queue occupancy is consumed by prefill, never charged again per decode
        stress_L = [list(row) for row in L]
        for a,b in zip(order,order[1:]):stress_L[a][b]=tail_hop(nodes[a]["id"],nodes[b]["id"])
        stress_out,stress_in=list(c_out),list(c_in)
        stress_out[order[0]]=tail_hop(coordinator,nodes[order[0]]["id"])
        stress_in[order[-1]]=tail_hop(nodes[order[-1]]["id"],coordinator)
        stress=estimate(order,alloc,layer_ms,stress_L,stress_out,stress_in,stress_nodes,{**model,"prefill_bytes":0},workload,
            edges=edges,now=now,objective=objective)
        # A known TTFT needs all prefill and queue measurements. TPOT excludes
        # TTFT, retaining decoder/feedback/drain costs of the finite window.
        prefix=sum(stress_L[a][b] for a,b in zip(order,order[1:]))+stress_out[order[0]]+stress_in[order[-1]]
        if w["generated_tokens"]>1:
            metrics["tpot_ms"]=(stress["predicted_request_ms"]-prefix)/(w["generated_tokens"]-1)
    mean_known = exact_stages and decode_bw_known and prefill_bw_known and prefill_geometry_known and all(row.get("source")=="fresh_sparse" for row in routes) and all(row["prefill_ms"] is not None for row in observed)
    if prices["hourly_cost"] is not None and mean_known:
        metrics["request_cost"]=prices["hourly_cost"]*point["predicted_request_ms"]/3_600_000
    required=set(policy["targets"])|{key for key,weight in policy["weights"].items() if weight>0}
    for name in sorted(required):
        if metrics[name] is None:unknown.append(name)
    exceeded=[name for name,target in policy["targets"].items() if metrics[name] is not None and metrics[name]>target]
    score=0.0
    for name,weight in policy["weights"].items():
        if weight:
            score+=weight*(metrics[name]/policy["normalizers"][name] if metrics[name] is not None else policy["unknown_penalty"])
    # Constraints still guide advisory ranking if the caller explicitly gave
    # zero objective weight to a constrained metric.
    for name in exceeded:score+=metrics[name]/max(policy["targets"][name],1e-9)
    for name in unknown:
        if not policy["weights"].get(name,0):score+=policy["unknown_penalty"]
    eligible=policy["mode"]!="require_known" or not unknown and not exceeded
    if not eligible:score=math.inf
    return {"schema":SLO_SCHEMA,"prediction_only":True,"acceptance_verified":False,
        "status":"unknown" if unknown else "predicted_exceeds_targets" if exceeded else "predicted_within_targets",
        "eligible":eligible,"selection_score":score,"metrics":metrics,"unknown_metrics":unknown,"exceeded_targets":exceeded,
        "policy":deepcopy(policy),"failure_risk":risk,"pricing":prices,
        "component_stress_request_ms":metrics["ttft_ms"]+metrics["tpot_ms"]*(w["generated_tokens"]-1) if metrics["ttft_ms"] is not None and metrics["tpot_ms"] is not None else None,
        "scope":"component-p95 stress scenario, not end-to-end p95 or an SLO guarantee; loss/jitter do not invent retransmit time",
        "uncertainty":sorted(set(risk["uncertainty"]+prices["uncertainty"]+point["uncertainty"]+
            (["shared-resource tail busy unknown; full-stage serialized stress assumption"] if any(row["shared_id"] is not None for row in observed) else [])))}


def _token_pipeline(stages, hops, transfers, entry, back, w, decode_needed, serial, prefill):
    warnings = []
    rounds = math.ceil(decode_needed / w["acceptance_gain"])
    # Full acceptance can continue without a cancellation drain. Otherwise
    # explicitly model complete bounded-window drain plus observed replay.
    if w["acceptance_gain"] == w["block_tokens"] and not w["cancel_rate"]:
        total = max(1, math.ceil(decode_needed / w["frame_tokens"]))
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
    first = replies[0] if decode_needed else 0.0
    if not decode_needed:
        elapsed = 0.0
    elapsed += prefill
    return elapsed, first, warnings

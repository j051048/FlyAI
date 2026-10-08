"""Candidate tiers and expiring directed link observations; no network I/O."""
from datetime import datetime, timezone
import math
import time

UNREACHABLE = 9000.0


def shortlist_candidates(nodes, model, policy=None, measurements=None, *, now=None):
    """O(N log N + E) inventory pass BEFORE dense solver adaptation, hard bounded.

    Retain small capacity covers from regional groups plus high-capacity global
    offers. This is a heuristic frontier, never proof that omitted nodes cannot
    form a ring. Callers may widen the explicit bound (maximum 256) or retry a
    region; the network service need not measure every pair in the open pool.
    """
    policy = {"mode": policy} if isinstance(policy, str) else dict(policy or {})
    bound = policy.get("max_candidates", 32)
    if type(bound) is not int or not 1 <= bound <= 256:
        raise ValueError("max_candidates must be an integer in [1,256]")
    report = {"total_nodes": len(nodes), "examined_nodes": min(len(nodes), bound),
              "max_candidates": bound, "truncated": len(nodes) > bound,
              "method": "bounded capacity/locality frontier; no global optimality claim"}
    if len(nodes) <= bound:
        return list(range(len(nodes))), report
    now = time.time() if now is None else timestamp(now)
    m = model or {}
    depth = int(m.get("n_layers", 43))
    per = float(m.get("layer_vram_mb", 1)) + float(m.get("kv_mb_per_layer", 0))
    def capacity(index):
        node = nodes[index]
        lv = (float(node["layer_vram_mb"]) + float(m.get("kv_mb_per_layer", 0))
              if node.get("layer_vram_mb") else per)
        raw = max(0, float(node["free_vram_mb"]) - float(m.get("reserve_mb", 0)))
        cap = int(raw // max(lv, 1e-9))
        if node.get("cap_layers") is not None:
            cap = min(cap, int(node["cap_layers"]))
        else:
            ceiling = int(m.get("cap_layers", depth))
            if node.get("total_vram_mb"):
                ceiling = max(0, round(ceiling * float(node["total_vram_mb"]) / 32768))
            cap = min(cap, ceiling)
        if m.get("layer_host_ram_mb"):
            ram, pin = node.get("free_ram_mb"), node.get("pinnable_ram_mb")
            cap = min(cap, int(min(ram, pin) // m["layer_host_ram_mb"])) if ram is not None and pin is not None else 0
        return max(0, cap)
    caps = {i: capacity(i) for i in range(len(nodes))}
    groups = {}
    for i, node in enumerate(nodes):
        region = node.get("region") or node.get("region_id")
        groups.setdefault(region, []).append(i)
    ordered = sorted(range(len(nodes)), key=lambda i: (-caps[i], float(nodes[i].get("layer_ms") or 1), str(nodes[i]["id"])))
    chosen = []
    def add(index):
        if index not in chosen and len(chosen) < bound:
            chosen.append(index)
    target = policy.get("region")
    regions = sorted(groups, key=lambda region: (region != target if target else False,
        -sum(caps[i] for i in groups[region]), str(region)))
    # Do not spend the entire frontier on a crowded, insufficient local group.
    # Regional covers get at most half the budget; expansion keeps global offers.
    for region in regions:
        covered = 0
        for i in sorted(groups[region], key=lambda i: (-caps[i], str(nodes[i]["id"]))):
            if len(chosen) >= max(1, bound // 2):
                break
            add(i); covered += caps[i]
            if covered >= depth:
                break
        if len(chosen) >= max(1, bound // 2):
            break
    anchor = policy.get("anchor_id")
    if anchor and measurements:
        proximity = {}
        for row in measurements.get("edges", []):
            if row.get("src") != anchor:
                continue
            try:
                age = now - timestamp(row["measured_at"])
                if -30 <= age <= number(row["ttl_s"], "ttl_s", minimum=1e-9):
                    proximity[row["dst"]] = number(row["rtt_ms"], "rtt_ms")
            except (KeyError, ValueError, TypeError):
                continue  # Full snapshot validation follows on the bounded frontier.
        for i in sorted(range(len(nodes)), key=lambda i: proximity.get(nodes[i]["id"], math.inf)):
            if nodes[i]["id"] in proximity and len(chosen) < bound * 3 // 4:
                add(i)
    # Trusted candidates can supply the two privacy boundary roles. Preserve
    # only a bounded handful, not unbounded extra cover after the hard cap.
    for i in ordered:
        if (nodes[i].get("trusted") is True or nodes[i].get("staked") is True) and len(chosen) < bound * 3 // 4:
            add(i)
    for i in ordered:
        add(i)
        if len(chosen) == bound:
            break
    report["examined_nodes"] = len(chosen)
    return chosen, report


def number(value, name, *, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def timestamp(value):
    if isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError("measurement timestamps require a timezone")
        return dt.timestamp()
    return number(value, "measured_at")


def link_snapshot(nodes, rtt=None, measurements=None, *, now=None):
    """Sparse observations are authoritative; stale/missing links cannot become zero.

    Legacy dense matrices retain their existing RTT-as-hop-delay interpretation.
    New directed rtt_ms observations use that same conservative interpretation,
    and explicitly report the assumption rather than inventing one-way precision.
    """
    now = time.time() if now is None else timestamp(now)
    ids = [node["id"] for node in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate node identity")
    n = len(ids)
    dense = [[0.0 if i == j else UNREACHABLE for j in range(n)] for i in range(n)]
    uncertainty, edges = [], {}
    if measurements is None:
        if rtt is None or len(rtt) != n or any(len(row) != n for row in rtt):
            raise ValueError("a square RTT mesh or sparse measurements is required")
        for i in range(n):
            for j in range(n):
                value = rtt[i][j]
                if value is None or value == float("inf"):
                    continue
                dense[i][j] = number(value, "rtt_ms")
                edges[ids[i], ids[j]] = {"rtt_ms": dense[i][j], "source": "legacy_dense"}
        uncertainty.append("legacy RTT mesh has no freshness proof")
    else:
        if not isinstance(measurements, dict) or measurements.get("schema") != "shard-link-measurements/1":
            raise ValueError("unsupported link measurement schema")
        rows = measurements.get("edges")
        if not isinstance(rows, list):
            raise ValueError("measurement edges must be a list")
        latest = {}
        for row in rows:
            if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k] for k in ("src", "dst")):
                raise ValueError("link endpoints must be nonempty node IDs")
            measured = timestamp(row["measured_at"])
            ttl = number(row["ttl_s"], "ttl_s", minimum=1e-9)
            latency = number(row["rtt_ms"], "rtt_ms")
            bandwidth = row.get("bandwidth_mbps")
            if bandwidth is not None:
                bandwidth = number(bandwidth, "bandwidth_mbps", minimum=1e-9)
            key = row["src"], row["dst"]
            if key not in latest or measured >= latest[key][0]:
                latest[key] = measured, ttl, latency, bandwidth
        for key, (measured, ttl, latency, bandwidth) in latest.items():
            if not -30 <= now - measured <= ttl:
                uncertainty.append(f"expired/future link {key[0]} -> {key[1]}")
                continue
            edges[key] = {"rtt_ms": latency, "bandwidth_mbps": bandwidth, "source": "fresh_sparse"}
        for i, source in enumerate(ids):
            for j, destination in enumerate(ids):
                if i != j and (source, destination) in edges:
                    dense[i][j] = edges[source, destination]["rtt_ms"]
        uncertainty.append("RTT is priced as conservative hop delay, not measured one-way latency")
    return {"rtt": dense, "edges": edges, "uncertainty": uncertainty, "now": now}


def candidate_tiers(nodes, snapshot, policy=None):
    """Search local tiers first. Unknown geography can join by measured closeness.

    With no locality metadata or explicit policy, preserve the historical global
    solve. An unknown label is never synthesized from public IP or NAT.
    """
    policy = {"mode": policy} if isinstance(policy, str) else dict(policy or {})
    mode = policy.get("mode", "prefer_local")
    if mode not in ("prefer_local", "global"):
        raise ValueError("locality mode must be prefer_local or global")
    limit = number(policy.get("max_rtt_ms", 35.0), "max_rtt_ms", minimum=1e-9)
    ids = [node["id"] for node in nodes]
    regions = {node["id"]: node.get("region") or node.get("region_id") for node in nodes}
    zones = {node["id"]: node.get("zone") or node.get("zone_id") for node in nodes}
    anchor = policy.get("anchor_id")
    requested_region, requested_zone = policy.get("region"), policy.get("zone")
    edges = snapshot["edges"]
    def close(a, b):
        if a == b:
            return True
        return all((x, y) in edges and edges[x, y]["rtt_ms"] <= limit
                   for x, y in ((a, b), (b, a)))
    fresh_mesh = any(row.get("source") == "fresh_sparse" for row in edges.values())
    if mode == "global" or (not any(regions.values()) and anchor is None and not requested_region and not policy and not fresh_mesh):
        return [{"name": "global", "pools": [ids], "legacy": not policy,
                 "reason": "global_policy" if mode == "global" else "locality_metadata_unavailable"}]
    region_names = [requested_region] if requested_region else sorted({r for r in regions.values() if r})
    tiers = []
    if anchor is not None:
        pool = [nid for nid in ids if close(anchor, nid)]
        if pool:
            tiers.append({"name": "near_anchor", "pools": [pool], "reason": "measured_low_latency"})
    if requested_zone:
        pool = [nid for nid in ids if zones[nid] == requested_zone and
                (not requested_region or regions[nid] == requested_region)]
        if pool:
            tiers.append({"name": "zone", "pools": [pool], "reason": "requested_zone"})
    pools = []
    for region in region_names:
        known = [nid for nid in ids if regions[nid] == region]
        unknown = [nid for nid in ids if not regions[nid] and any(close(nid, k) for k in known)]
        if known:
            pools.append(known + unknown)
    if pools:
        tiers.append({"name": "region", "pools": pools, "reason": "regional_candidate_pool"})
    if not tiers:
        # No labeled region: each latency neighborhood is a possible local pool.
        pools = [[nid for nid in ids if close(seed, nid)] for seed in ids]
        tiers.append({"name": "measured_neighborhood", "pools": pools, "reason": "measured_low_latency"})
    tiers.append({"name": "expanded", "pools": [ids], "reason": "no_local_feasible_plan"})
    return tiers

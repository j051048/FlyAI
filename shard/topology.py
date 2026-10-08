"""latency-optimal pipeline ordering — the heart of serving a *scattered* swarm.

c0mpute nodes may be scattered consumer GPUs or several GPUs in one miner's rig. every token
traverses coordinator -> head -> ... -> tail -> (direct return) -> coordinator, so the
per-token WAN cost is:

    entry hop  +  sum of forward hops  +  return hop
    c_out[h]   +  sum L[node_i, node_i+1]  +  c_in[t]

the visit order is free (any node can hold any contiguous block), so the cheapest
pipeline is the minimum-latency Hamiltonian loop with the coordinator as the depot.
internet RTT is asymmetric and doesn't track geography (peering), so we optimize on the
*measured* mesh, not distance.

  - optimal_loop : exact (Held-Karp) for n<=16, nearest-neighbor + 2-opt above.
  - select_and_order : pick the best k of n online nodes AND order them (exact for n<=16).

both take L (L[i][j] = ms from node i to node j, asymmetric ok), c_out (coordinator->i),
c_in (i->coordinator). pure python, no deps; run `python -m shard.topology` for a demo.
"""
from itertools import combinations, permutations

INF = float("inf")
_TRIM = 12          # max candidates fed to exhaustive Held-Karp; the network layer funnels bigger pools first


def _up(up_mbps, n):
    """Upload Mbps for node n, floored to avoid div-by-zero. An unmeasured/zero node reads as very
    slow (0.5 Mbps) -> costed OFF the critical path, then relegated. Unmeasured == assume bad uplink
    is the safe default (a lying/absent uplink can't sneak onto a load-bearing hop)."""
    v = up_mbps.get(n, 0.0) if hasattr(up_mbps, "get") else up_mbps[n]
    return max(float(v), 0.5)


def _xfer_ms(nbytes, up_mbps, n):
    """ms to upload `nbytes` from node n over its uplink (bits / bandwidth). Residential is asymmetric
    (fast down, slow up), so a hop a->b is bound by the SENDER a's UPLOAD, never the receiver's."""
    return nbytes * 8.0 / (_up(up_mbps, n) * 1000.0)


def loop_cost(order, L, c_out, c_in):
    """total per-traversal latency for a given node ordering (entry + hops + return)."""
    if not order:
        return 0.0
    cost = c_out[order[0]] + c_in[order[-1]]
    for a, b in zip(order, order[1:]):
        cost += L[a][b]
    return cost


def _held_karp(nodes, L, c_out, c_in):
    """exact min-latency loop over exactly `nodes` (indices). O(k^2 2^k), k<=~16."""
    idx = list(nodes)
    k = len(idx)
    if k == 1:
        return idx, c_out[idx[0]] + c_in[idx[0]]
    pos = {n: i for i, n in enumerate(idx)}                 # node -> bit position
    dp = [[INF] * k for _ in range(1 << k)]                 # dp[mask][j] = min cost ... ending at j
    par = [[-1] * k for _ in range(1 << k)]
    for j in range(k):
        dp[1 << j][j] = c_out[idx[j]]
    for mask in range(1 << k):
        for j in range(k):
            if dp[mask][j] == INF or not (mask >> j) & 1:
                continue
            base = dp[mask][j]
            for m in range(k):
                if (mask >> m) & 1:
                    continue
                nmask = mask | (1 << m)
                cand = base + L[idx[j]][idx[m]]
                if cand < dp[nmask][m]:
                    dp[nmask][m] = cand
                    par[nmask][m] = j
    full = (1 << k) - 1
    best, bj = INF, -1
    for j in range(k):
        c = dp[full][j] + c_in[idx[j]]
        if c < best:
            best, bj = c, j
    order, mask, j = [], full, bj                           # reconstruct
    while j != -1:
        order.append(idx[j])
        pj = par[mask][j]
        mask ^= (1 << j)
        j = pj
    order.reverse()
    return order, best


def _pin_order_oversize(nodes, L, c_out, c_in, heads, ends):
    """heuristic ends-constrained order for k>16 (outside Held-Karp reach; rings are ~5-8 so this
    is a widen-fallback safety net, not a hot path): for each trusted (head, tail) pair, greedy
    nearest-neighbor path head->...->tail over the middle, then 2-opt on the middle segment."""
    best_o, best_c = None, INF
    for h in heads:
        for t in ends:
            if t == h:
                continue
            mid, tour = set(nodes) - {h, t}, [h]
            while mid:
                nxt = min(mid, key=lambda n: L[tour[-1]][n])
                tour.append(nxt); mid.discard(nxt)
            tour.append(t)
            improved = True
            while improved:
                improved = False
                for i in range(1, len(tour) - 2):
                    for j in range(i + 1, len(tour) - 1):
                        cand = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                        if loop_cost(cand, L, c_out, c_in) + 1e-9 < loop_cost(tour, L, c_out, c_in):
                            tour, improved = cand, True
            c = loop_cost(tour, L, c_out, c_in)
            if c < best_c:
                best_o, best_c = tour, c
    return best_o, best_c


def _nn_2opt(nodes, L, c_out, c_in, rounds=4):
    """heuristic for large k: nearest-neighbor seed, then 2-opt segment reversals."""
    idx = list(nodes)
    # nearest-neighbor from the cheapest entry hop
    start = min(idx, key=lambda n: c_out[n])
    tour, rest = [start], set(idx) - {start}
    while rest:
        last = tour[-1]
        nxt = min(rest, key=lambda n: L[last][n])
        tour.append(nxt); rest.discard(nxt)
    best = loop_cost(tour, L, c_out, c_in)
    improved = True
    while improved:
        improved = False
        for i in range(len(tour) - 1):
            for j in range(i + 1, len(tour)):
                cand = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                c = loop_cost(cand, L, c_out, c_in)
                if c + 1e-9 < best:
                    tour, best, improved = cand, c, True
    return tour, best


def optimal_loop(nodes, L, c_out, c_in):
    """min-latency pipeline order over all `nodes`. exact <=16, heuristic above."""
    nodes = list(nodes)
    if len(nodes) <= 16:
        return _held_karp(nodes, L, c_out, c_in)
    return _nn_2opt(nodes, L, c_out, c_in)


def select_and_order(nodes, L, c_out, c_in, k):
    """pick the best k of n online nodes AND order them into the cheapest loop.

    exact for n<=16 (Held-Karp answer ranges over size-k subsets); above that, solve the
    full order then greedily drop the node whose removal helps most until k remain.
    """
    nodes = list(nodes)
    if k >= len(nodes):
        return optimal_loop(nodes, L, c_out, c_in)
    if len(nodes) <= 16:
        best_order, best_cost = None, INF
        for subset in combinations(nodes, k):
            order, cost = _held_karp(subset, L, c_out, c_in)
            if cost < best_cost:
                best_order, best_cost = order, cost
        return best_order, best_cost
    order, _ = _nn_2opt(nodes, L, c_out, c_in)              # greedy-drop from a good full tour
    while len(order) > k:
        drop = min(range(len(order)),
                   key=lambda i: loop_cost(order[:i] + order[i + 1:], L, c_out, c_in))
        order = order[:drop] + order[drop + 1:]
    return _nn_2opt(order, L, c_out, c_in)


# ---- health + capability-aware selection (the self-optimizer's pure core) ----
# select_and_order picks the lowest-LATENCY ring. select_ring picks the lowest predicted
# STEP-TIME ring (communication + per-stage compute), drops unhealthy/slow nodes, and
# sizes each block to the node's speed. tok/s = accept_gain / step_ms and accept_gain is ~constant
# across rings, so minimizing predicted step_ms maximizes usable tok/s. The objective is physical
# (milliseconds), never hand-tuned weights: a power-capped or far node just shows up as more ms.
# PURE: measured stats in, a RingSpec out — no probing/IO here (that's the network layer's job).


def predict_step_ms(order, layers, L, c_out, c_in, layer_ms, up_mbps=None, decode_bytes=0.0):
    """Predicted per-traversal time of one decode step (ms): WAN round-trip (depends on ORDER) +
    sum of per-stage compute (depends on the layer ASSIGNMENT) + the per-step activation UPLOAD when
    `up_mbps` is given. `layers[n]` = #layers node n holds; `layer_ms[n]` = MEASURED ms to run one
    layer for a decode step on n (a throttled GPU measures higher — don't infer it from watts).
    Decode's activation is small (a few draft tokens' hidden state) but NOT free below ~fiber: at
    20 Mbps a ~50KB bundle costs ~20ms/hop, so on residential links decode transport is real and
    each stage uploads its per-step output once around the loop. On fiber `decode_bytes/up -> ~0` and
    this reduces to the pure round-trip+compute model (transport was rightly omitted there)."""
    ms = loop_cost(order, L, c_out, c_in) + sum(layers[n] * layer_ms[n] for n in order)
    if up_mbps is not None and decode_bytes:
        ms += sum(_xfer_ms(decode_bytes, up_mbps, n) for n in order)   # every stage uploads once/step
    return ms


def predict_prefill_ms(order, layers, L, c_out, c_in, up_mbps, prefill_bytes, prefill_chunks=1,
                       prefill_layer_ms=None):
    """Predicted TTFT (ms) — the RESIDENTIAL WALL. One forward traversal of the prompt + the [S,H]
    activation UPLOAD (upload-bound, dominant) + optional prefill compute. The prompt's [S,H]
    activation (`prefill_bytes` per hop, e.g. 16k*3072*2 ~= 100MB) is pipelined across the ring as
    `prefill_chunks` (C) chunks, so each FORWARDING (non-tail) stage uploads its whole [S,H] split
    into C pieces. The pipeline makespan interpolates the two physical regimes:
        transport = ( sum_fwd(u) + (C-1)*max_fwd(u) ) / C ,   u_s = _xfer_ms(prefill_bytes, up, s)
    C=1 (a single blob per hop) -> SUM (serial: each stage waits for the whole activation);
    C large (fine chunking) -> MAX (steady-state pipeline, bounded by the slowest uplink). The engine
    runs chunked+pipelined prefill (prefill_chunk, prefill_depth), so C = ceil(S/prefill_chunk).
    The TAIL forwards nothing onward (it returns only the first token's logits, tiny) -> EXEMPT, so a
    low-upload node belongs at the TAIL. Compute optional: residential prefill is transport-dominated
    (~100MB@20Mbps = ~40s/hop vs seconds of compute); pass `prefill_layer_ms` for fiber-accurate TTFT."""
    lat = loop_cost(order, L, c_out, c_in)                       # one traversal; return = first-token logits (small)
    fwd = order[:-1]                                             # non-tail stages upload [S,H] onward
    C = max(1, int(prefill_chunks))
    if fwd:
        us = [_xfer_ms(prefill_bytes, up_mbps, n) for n in fwd]
        transport = (sum(us) + (C - 1) * max(us)) / C
    else:
        transport = 0.0
    compute = sum(layers[n] * prefill_layer_ms[n] for n in order) if prefill_layer_ms else 0.0
    return lat + transport + compute


def _boundary_nodes(order, alloc, n_layers, boundary_in, boundary_out):
    """The stages whose contiguous block intersects a LEAKY range: [0, boundary_in) near the
    embedding, [n_layers-boundary_out, n_layers) near the output — PLUS both ends of the ring
    unconditionally, because the roles themselves leak regardless of which layers they hold: the
    head is handed the raw prompt token ids to embed, and the tail computes logits and returns
    argmax token ids (during prefill that's the greedy next-token at every prompt position — a
    near-copy of the prompt). Middle stages see only the activation tensor."""
    out, lo = {order[0], order[-1]}, 0
    hi_cut = n_layers - max(0, boundary_out)
    for n in order:
        hi = lo + alloc[n]
        if lo < boundary_in or hi > hi_cut:
            out.add(n)
        lo = hi
    return out


def _relegate(order, dropped, caps, subnet, up_mbps, layer_ms, trusted=None, boundary_subnets=()):
    """Advisory off-critical-path role for every DROPPED node, derived from WHY the objective dropped
    it — NOT a fresh absolute threshold. This is the PLACEMENT half of the decided admission/placement
    framing: the "threshold" is per-role capability against the CHOSEN ring, never a velvet rope at the
    door. c0mpute makes the final placement; these are hints. Coverage is TOTAL (every dropped node
    gets a role). The only capacity split is physical: cap==0 (can't be a stage) vs cap>=1.
      weight-seeder      : cap==0 — serves weight shards from disk (the torrent fetch path); no VRAM/
                           compute/latency needs. The universal floor.
      aggregator/relay   : upload >= the ring's BEST uplink AND subnet-distinct — a fiber-class node
                           wasted as a mere stage; spend its scarce UPLOAD as a prefill fan-in / relay
                           supernode (the research's top off-ring lever). Mechanism lives in c0mpute.
      hot-standby        : subnet-twin of a chosen stage — warm PASSIVE failover for that block (co-
                           location is fine for a spare; a twin is latency-close, so failover keeps the
                           ring's step_ms — we never route a high-latency node here). Trust-aware: an
                           UNTRUSTED twin of a boundary stage is never a hot-standby (failover would
                           hand the leaky block to a stranger — the exact hole pinning closes).
      decode-only-replica: compute ring-competitive but dropped for its slow UPLOAD — decode's tiny
                           activation survives its uplink; candidate member of a decode-only ring (ring
                           formation, which needs >=k subnet-distinct peers, lives in c0mpute).
      spot-check-verifier: any other block-capable node — samples & recomputes a stage to catch
                           cheaters (latency/upload tolerant, async, bounded demand)."""
    if not order:
        return {}
    ring_subnets = {subnet[n] for n in order}
    ring_best_up = max(_up(up_mbps, n) for n in order)          # the ring's fastest uplink
    ring_worst_compute = max(layer_ms[n] for n in order)        # slowest per-layer compute the ring admitted
    roles = {}
    for n in dropped:
        if caps.get(n, 0) == 0:
            roles[n] = "weight-seeder"                          # can't hold a stage -> seed weights
        elif _up(up_mbps, n) >= ring_best_up and subnet[n] not in ring_subnets:
            roles[n] = "aggregator"                             # better-connected than the whole ring
        elif subnet[n] in ring_subnets and (trusted is None or n in trusted
                                            or subnet[n] not in boundary_subnets):
            roles[n] = "hot-standby"                            # subnet-twin of a stage -> warm failover
        elif layer_ms[n] <= ring_worst_compute:
            roles[n] = "decode-only-replica"                    # compute-fine, dropped for upload -> decode is ok
        else:
            roles[n] = "spot-check-verifier"                    # slow compute/high latency -> sampled recompute
    return roles


def _head_first(order, head, L, c_out, c_in):
    """Deployable orientation: the coordinator lives ON the head box, so the launcher needs the ring
    path to START at `head` (stage 0 is dialed locally; the head sidecar carries the coord-return to
    the tail). optimal_loop models the coordinator only through c_out/c_in and may legally end the
    path at `head` — or even place it mid-path under asymmetric matrices — which is undeployable.
    Re-solve the ordering with head PINNED FIRST: exhaustive for ring-sized k (<=5040 orders; the
    trim funnel keeps pools small and rings are ~5-8 nodes). No blind reversal: under the aware
    matrices (upload rides FORWARD edges by sender) a mirrored path does NOT cost the same."""
    if order[0] == head:
        return list(order)
    rest = [n for n in order if n != head]
    if len(rest) <= 7:
        best = min(permutations(rest), key=lambda p: loop_cost([head, *p], L, c_out, c_in))
        return [head, *best]
    cand = ([head] + rest, [head] + rest[::-1])                 # oversize fallback (unreachable via the funnel)
    return list(min(cand, key=lambda p: loop_cost(p, L, c_out, c_in)))


def node_capacity(free_vram_mb, layer_vram_mb, kv_mb_per_layer=0):
    """max contiguous layers a node can hold (weights + KV cache), >= 0."""
    per = layer_vram_mb + kv_mb_per_layer
    return int(free_vram_mb // per) if per > 0 else 0


def assign_layers(order, n_layers, caps, layer_ms, floors=None, *, groups=None, group_caps=None):
    """Size each node's contiguous block to MINIMIZE total decode-step compute — the SUM of per-stage
    times, which is exactly what predict_step_ms scores and the right model for single-traversal
    autoregressive decode (token t+1 can't enter the ring until t exits, so per-step latency is the
    sum, not a pipeline makespan). Every stage must hold >=1 layer (no empty hops), so: floor 1 per
    node, then pile the remaining layers onto the lowest-layer_ms nodes up to their VRAM `caps`.
    `floors` (optional {node: min_layers}) raises a node's floor above 1 — boundary pinning uses it
    to make a trusted end-stage hold the whole leaky boundary range instead of letting layers spill
    onto an untrusted neighbor. Returns {node: cnt} (sums to n_layers, every value >= its floor) or
    None if the subset can't satisfy floors+caps and still hold the model. (A PIPELINED throughput
    regime minimizes the max stage instead; predict_step_ms would then switch to max — the two must
    stay in lockstep.)"""
    base = {n: max(1, (floors or {}).get(n, 1)) for n in order}
    need = sum(base.values())
    if n_layers <= 0 or n_layers < need:                        # need >= floor layers per stage
        return None
    if any(base[n] > caps[n] for n in order):                   # a floor its node can't hold
        return None
    if sum(caps[n] for n in order) < n_layers:
        return None
    alloc = dict(base)
    groups = groups or {n: n for n in order}
    group_caps = group_caps or {}
    used = {}
    for n, count in alloc.items():
        group = groups[n]
        used[group] = used.get(group, 0) + count
    if any(count > group_caps.get(group, INF) for group, count in used.items()):
        return None
    rem = n_layers - need
    for n in sorted(order, key=lambda n: layer_ms[n]):          # remaining layers -> cheapest-per-layer first (min sum)
        group = groups[n]
        take = min(caps[n] - alloc[n], rem, group_caps.get(group, INF) - used[group])
        if take > 0:
            alloc[n] += take; rem -= take; used[group] += take
        if rem <= 0:
            break
    return alloc if rem == 0 else None


def assign_pipeline_layers(order, n_layers, caps, layer_ms, floors=None, *, groups=None, group_caps=None):
    """Integer water filling under the same floors, GPU and shared-RAM limits.

    This supplies a balanced candidate. The finite-window resource-calendar
    objective must still score it against the serial allocation; balance alone
    is not evidence of a throughput improvement.
    """
    alloc = {n: max(1, (floors or {}).get(n, 1)) for n in order}
    groups, group_caps = groups or {n: n for n in order}, group_caps or {}
    used = {}
    for n in order:
        if alloc[n] > caps[n]:
            return None
        used[groups[n]] = used.get(groups[n], 0) + alloc[n]
    if sum(alloc.values()) > n_layers or any(v > group_caps.get(g, INF) for g, v in used.items()):
        return None
    for _ in range(n_layers - sum(alloc.values())):
        eligible = [n for n in order if alloc[n] < caps[n] and used[groups[n]] < group_caps.get(groups[n], INF)]
        if not eligible:
            return None
        node = min(eligible, key=lambda n: ((alloc[n] + 1) * layer_ms[n], layer_ms[n]))
        alloc[node] += 1
        used[groups[node]] += 1
    return alloc


def _is_adjacent_same_host(order, host_id=None):
    """Check if any adjacent stages in a ring order reside on the same physical host.

    A policy predicate, not proof of Sybil resistance or network reachability.
    """
    if not host_id or len(order) <= 1:
        return False
    lk = len(order)
    for idx in range(lk):
        ha = host_id.get(order[idx])
        hb = host_id.get(order[(idx + 1) % lk])
        if ha is not None and hb is not None and ha != "" and ha == hb:
            return True
    return False


def _constrained_loop(nodes, L, c_out, c_in, host_id=None, require=None, allowed_tails=None):
    """Cheapest loop with permissible ends and optional adjacent-host separation.

    Rejecting the unconstrained cheapest order can miss a valid A-B-A-B order.
    Exclude conflicts inside the DP instead; no production isolation is implied.
    """
    idx = list(nodes)
    tails = set(idx) if allowed_tails is None else set(allowed_tails)
    if len(idx) == 1:
        return (idx, loop_cost(idx, L, c_out, c_in)) if idx[0] in tails else (None, INF)
    heads = [require] if require is not None else idx
    if len(idx) > 16:
        order, cost = _pin_order_oversize(idx, L, c_out, c_in, heads, tails)
        return (order, cost) if order and not _is_adjacent_same_host(order, host_id) else (None, INF)
    best_order, best_cost = None, INF
    k, full = len(idx), (1 << len(idx)) - 1
    for head in heads:
        start = idx.index(head)
        dp = [[INF] * k for _ in range(1 << k)]
        parent = [[-1] * k for _ in range(1 << k)]
        dp[1 << start][start] = c_out[head]
        for mask in range(1 << k):
            for j in range(k):
                if dp[mask][j] == INF:
                    continue
                for m in range(k):
                    if mask & (1 << m) or (host_id and host_id[idx[j]] == host_id[idx[m]]):
                        continue
                    next_mask = mask | (1 << m)
                    candidate = dp[mask][j] + L[idx[j]][idx[m]]
                    if candidate < dp[next_mask][m]:
                        dp[next_mask][m], parent[next_mask][m] = candidate, j
        for j, tail in enumerate(idx):
            if tail not in tails or (host_id and host_id[head] == host_id[tail]):
                continue
            cost = dp[full][j] + c_in[tail]
            if cost >= best_cost:
                continue
            order, mask, end = [], full, j
            while end != -1:
                order.append(idx[end])
                previous = parent[mask][end]
                mask ^= 1 << end
                end = previous
            best_order, best_cost = order[::-1], cost
    return best_order, best_cost


def select_ring(nodes, L, c_out, c_in, *, free_vram_mb, layer_ms, subnet,
                n_layers, layer_vram_mb, kv_mb_per_layer=0, slack=2, exclude=None, require=None,
                up_mbps=None, prefill_bytes=0.0, decode_bytes=0.0, decode_steps=1,
                prefill_chunks=1, prefill_layer_ms=None, relegate=True,
                trusted=None, boundary_in=0, boundary_out=0, max_stages=6,
                tail_floor=0, host_id=None, isolation="none", device_id=None,
                host_layer_caps=None, host_memory_domain=None, tail_host_layer_caps=None,
                node_layer_caps=None, tail_reserve_mb=0, objective="serial", cost_model=None):
    """The self-optimizer's pure core. From a candidate POOL, choose the subset + ring order +
    per-node layer split that MINIMIZES predicted request time, subject to:
      * VRAM feasibility — the chosen nodes must hold the whole model (+ KV),
      * production permits colocated distinct GPUs; isolation is explicit policy,
      * health — a power-capped/slow node has a high `layer_ms`, so it's dropped or given fewer
        layers automatically; no hand-tuned weights, just physical milliseconds.
    Prefers the FEWEST nodes that fit (each extra node is another full WAN round-trip — fewer,
    fatter stages win over scatter), trying sizes k_min..k_min+slack so a faster larger set can
    still win.

    UPLOAD-AWARE (opt-in via `up_mbps={node: Mbps}`): the objective becomes TOTAL REQUEST TIME
    T = prefill_ms + decode_steps * decode_step_ms, with per-node UPLOAD a first-class cost. This is
    the residential lever: a home link is asymmetric (fast down, SLOW up) and the per-hop bottleneck
    is MOVING THE ACTIVATION on the sender's uplink. Decode's activation is tiny (survives), but
    long-context PREFILL ([S,H] ~= 100MB/hop @16k) is the WALL — minutes of TTFT on a 20 Mbps cable
    uplink. So the selector (a) tails the lowest-upload node (the tail forwards nothing — see
    predict_prefill_ms), (b) drops nodes whose upload would dominate prefill, and (c) RELEGATES those
    dropped nodes to off-critical-path roles (see _relegate) instead of discarding useful capacity.
    Bytes are PRE-MULTIPLIED by the caller (workload- and dtype-agnostic core): `prefill_bytes`=S*H*
    dtype, `decode_bytes`=draft_tokens*H*dtype (fp8 wire => halve them), `prefill_chunks`=ceil(S/
    prefill_chunk) sets the SUM<->MAX pipeline regime, `prefill_layer_ms` (optional) adds prefill
    compute for fiber-accurate TTFT. Missing/absent `up_mbps` == today's pure decode-step objective
    (BYTE-IDENTICAL legacy path); when set, the spec also carries prefill_ms/request_ms/roles.

    Returns a RingSpec dict, or None if the pool can't hold the model:
      {order, blocks:{n:(lo,hi)}, layers:{n:cnt}, step_ms, tok_s_per_g, dropped, k}
      (+ prefill_ms, request_ms, roles:{dropped_node: role}  when up_mbps is given).
    `require` pins a node that MUST be in the ring (our coordinator runs on the head box, so for
    that deployment pass the head node and set c_out/c_in relative to it -> the loop becomes the
    correct head->...->tail->head cycle, and the returned order is GUARANTEED to start at `require`
    — the launcher puts the coordinator on order[0]'s box, so any other orientation is undeployable). `exclude` drops nodes outright. NOTE: `slack` is the pool
    headroom you rented (N+slack) — selection can only drop bad nodes when the pool exceeds what the
    model strictly needs. Assumes the pool is already pre-filtered to a tractable candidate set (the
    network layer funnels thousands -> ~16 via latency coordinates before calling this); if larger,
    it pre-trims to the 14 lowest-RTT usable nodes (always keeping `require`).

    BOUNDARY-LAYER PINNING (opt-in via `trusted={node,...}`) — the open-admission privacy rail. An
    untrusted stage can invert the activations it forwards back toward the prompt, and inversion is
    strongest at the BOUNDARIES: near the embedding (early layers) and near the output (late layers
    + the lm_head). When `trusted` is given, placement enforces:
      * the HEAD and TAIL stages are trusted UNCONDITIONALLY — the head is handed raw prompt token
        ids to embed, and the tail computes logits + returns argmax token ids (at prefill: the
        greedy next-token at every prompt position, a near-copy of the prompt). Those roles leak
        whatever layers they hold.
      * every stage whose block intersects [0, boundary_in) or [n_layers-boundary_out, n_layers)
        is trusted — strangers hold only deep-middle layers, where inversion decays.
    assign_layers is floored so a trusted end-stage absorbs its whole boundary range when its VRAM
    allows, instead of spilling leaky layers onto an untrusted neighbor. `trusted=None` (default)
    is the exact legacy objective; an EMPTY trusted set with pinning on is honestly infeasible.
    Trust is a CONSTRAINT, never a score: among trust-valid rings the objective is unchanged."""
    if require is not None and exclude and require in set(exclude):
        raise ValueError("`require` and `exclude` name the same node")
    if isolation not in ("none", "subnet", "host", "adjacent_host"):
        raise ValueError("isolation must be none, subnet, host or adjacent_host")
    if len(set(nodes)) != len(nodes):
        raise ValueError("duplicate node IDs cannot contribute capacity twice")
    host_id = dict(host_id or {})
    raw_subnets = dict(subnet or {})
    subnet = {n: raw_subnets.get(n) if raw_subnets.get(n) not in (None, "") else ("unknown-network", n)
              for n in nodes}
    devices = {}
    for n in nodes:
        value = (device_id or {}).get(n)
        items = [value] if isinstance(value, str) else list(value) if value is not None else []
        if any(not isinstance(item, str) or not item.strip() for item in items):
            raise ValueError("device_id must identify globally unique GPU UUID strings")
        identities = [item.strip().casefold() for item in items]
        if len(identities) != len(set(identities)):
            raise ValueError("a GPU UUID is repeated inside one device announcement")
        devices[n] = set(identities)
    memory_groups = {}
    for n in nodes:
        domain = (host_memory_domain or {}).get(n)
        if domain in (None, ""):
            domain = host_id.get(n)
        memory_groups[n] = ("node", n) if domain in (None, "") else domain
    host_layer_caps = dict(host_layer_caps or {})
    tail_host_layer_caps = dict(tail_host_layer_caps or {})
    for value in (*host_layer_caps.values(), *tail_host_layer_caps.values()):
        if type(value) is not int or value < 0:
            raise ValueError("host layer capacities must be nonnegative integers")

    def isolation_key(n):
        return subnet[n] if isolation == "subnet" else host_id[n] if isolation == "host" else ("node", n)

    def unique_devices(subset):
        seen = set()
        for n in subset:
            if seen & devices[n]:
                return False
            seen.update(devices[n])
        return True
    pin = trusted is not None
    trust = set(trusted) if pin else set()
    # clamp each window to [0, n_layers]: a window >= n_layers means "that whole end is leaky" and
    # must not overflow the layer math (an unclamped b_in > n_layers false-infeasibled every ring).
    b_in = min(max(0, int(boundary_in)), n_layers) if pin else 0
    b_out = min(max(0, int(boundary_out)), n_layers) if pin else 0
    nodes = [n for n in nodes if not exclude or n not in exclude]
    # layer_vram_mb is a scalar OR a per-node dict {node: mb}: a heterogeneous ring holds cards whose
    # per-layer footprint differs by ARCH+backend (5090 cutlass NVFP4 ~1.7GB/layer; a 4090/3090 on the
    # marlin dequant path ~4.1GB, so it holds far fewer layers). Per-node caps flow straight into every
    # downstream cap[n] check, so the selector fills a small card with few layers automatically. A scalar
    # is byte-identical to the legacy path (goldens unchanged).
    def _lv(n):
        return layer_vram_mb[n] if isinstance(layer_vram_mb, dict) else layer_vram_mb
    caps = {n: node_capacity(free_vram_mb[n], _lv(n), kv_mb_per_layer) for n in nodes}
    if node_layer_caps is not None:
        caps = {n: min(c, node_layer_caps[n]) for n, c in caps.items()}
    usable = [n for n in nodes if caps[n] > 0]
    if isolation == "subnet":
        usable = [n for n in usable if raw_subnets.get(n) not in (None, "")]
    elif isolation in ("host", "adjacent_host"):
        usable = [n for n in usable if host_id.get(n) not in (None, "")]
    if require is not None and require not in usable:
        return None                                              # the pinned coord/head can't hold a block
    if pin:
        if require is not None and require not in trust:
            return None                                          # the coord/head box sees raw tokens: must be trusted
        if not any(n in trust for n in usable):
            return None                                          # no trusted stage-capable node -> can't hold the ends

    def feasible_cap(pool):
        best = {}
        for n in pool:
            key = isolation_key(n)
            best[key] = max(best.get(key, 0), caps[n])
        memory = {}
        for n in pool:
            key = memory_groups[n]
            memory[key] = memory.get(key, 0) + caps[n]
        return min(sum(best.values()), sum(min(count, host_layer_caps.get(key, count))
                                          for key, count in memory.items()))
    if feasible_cap(usable) < n_layers:                          # feasibility on the FULL set, honoring co-location
        return None                                              # genuinely can't serve the model (no false negative)

    by_cap = sorted(usable, key=lambda n: caps[n], reverse=True)
    acc, k_min, used_sub, cover = 0, 0, set(), []
    for n in by_cap:
        if isolation_key(n) in used_sub:
            continue
        used_sub.add(isolation_key(n)); cover.append(n); k_min += 1
        # This is a lower bound on ring width. A greedy cover under shared RAM
        # limits can take three colocated offers even when two other GPUs fit.
        acc += caps[n]
        if acc >= n_layers:
            break

    if len(usable) > _TRIM:                                      # latency funnel, but never trim out nodes feasibility
        keep = sorted(usable, key=lambda n: c_out[n] + c_in[n])[:_TRIM]   # or `require` need
        must, seen, cover = set(), set(), []                     # retain a cover under the selected isolation policy
        for m in by_cap:
            if isolation_key(m) in seen:
                continue
            seen.add(isolation_key(m)); must.add(m); cover.append(m)
            if len(must) >= k_min + slack and feasible_cap(cover) >= n_layers:
                break
        if require is not None:                                  # retain a cover compatible with the mandatory head
            must.add(require)
            seen, acc, cover = {isolation_key(require)}, caps[require], [require]
            for m in by_cap:
                if isolation_key(m) in seen:
                    continue
                seen.add(isolation_key(m)); must.add(m); cover.append(m); acc = feasible_cap(cover)
                if acc >= n_layers:
                    break
        if pin:                                                  # ...and a TRUSTED cover: both ring ends (+ the
            seen, kept_t = set(), 0                              # boundary layers) must sit on trusted nodes, and
            for m in by_cap:                                     # the low-RTT `keep` may hold none -> keep the
                if m not in trust or isolation_key(m) in seen:    # preserve trusted ends under the selected policy
                    continue                                     # so pinning can't be starved into false-infeasible
                seen.add(isolation_key(m)); must.add(m); kept_t += 1
                if kept_t >= 2 + slack:
                    break
        usable = keep + [n for n in must if n not in keep]

    aware = up_mbps is not None
    D = max(0, int(decode_steps))
    if aware:
        # Effective ORDERING matrices: fold the frequency-weighted latency (each traversal happens
        # once for prefill + D times for decode) with a per-edge upload cost, so the SAME optimal_loop
        # returns a request-time-optimal order. The big [S,H] prefill term rides ONLY forward edges
        # (sender a uploads to its successor); the return hop (Ein) carries just the first token's
        # logits, so the tail's expensive forward upload vanishes -> optimal_loop naturally TAILS the
        # lowest-upload node. Compute is order-independent (summed per node), so it's added at scoring,
        # not here; hence minimizing this fold == minimizing request_ms over orders (exact for C=1,
        # a tight tail-exempting heuristic for C>1 where prefill is a MAX the loop-sum can't express).
        EL = {a: {} for a in usable}
        Eout, Ein = {}, {}
        for a in usable:
            pf_a = _xfer_ms(prefill_bytes, up_mbps, a)
            dc_a = _xfer_ms(decode_bytes, up_mbps, a)
            Eout[a] = (1 + D) * c_out[a]                         # entry: coord uploads tiny token-ids -> latency only
            Ein[a] = (1 + D) * c_in[a] + D * dc_a               # return: tail uploads decode output D times, no [S,H]
            for b in usable:
                if a != b:
                    EL[a][b] = (1 + D) * L[a][b] + pf_a + D * dc_a

    def _score(order, alloc):
        if cost_model is not None:
            predicted = cost_model(order, alloc)
            step = predict_step_ms(order, alloc, L, c_out, c_in, layer_ms,
                                   up_mbps if aware else None, decode_bytes)
            return predicted["predicted_request_ms"], step, 0.0
        step = predict_step_ms(order, alloc, L, c_out, c_in, layer_ms,
                               up_mbps if aware else None, decode_bytes)
        if not aware:
            return step, step, 0.0                              # legacy: rank == decode step; no prefill
        pf = predict_prefill_ms(order, alloc, L, c_out, c_in, up_mbps, prefill_bytes,
                                prefill_chunks, prefill_layer_ms)
        return pf + D * step, step, pf                          # rank by total request time

    def memory_limits(order):
        limits = dict(host_layer_caps)
        group = memory_groups[order[-1]]
        if group in tail_host_layer_caps:
            limits[group] = min(limits.get(group, tail_host_layer_caps[group]), tail_host_layer_caps[group])
        return limits

    def role_caps(order, subset_caps):
        result = dict(subset_caps)
        if tail_reserve_mb:
            tail = order[-1]
            result[tail] = min(result[tail], max(0, node_capacity(
                free_vram_mb[tail] - tail_reserve_mb, _lv(tail), kv_mb_per_layer)))
        return result

    def allocate(order, subset_caps, floors):
        effective = role_caps(order, subset_caps)
        args = dict(groups=memory_groups, group_caps=memory_limits(order))
        serial = assign_layers(order, n_layers, effective, layer_ms, floors, **args)
        if objective != "pipeline" or serial is None:
            return serial
        balanced = assign_pipeline_layers(order, n_layers, effective, layer_ms, floors, **args)
        if balanced is None or cost_model is None:
            return balanced or serial
        return min((serial, balanced), key=lambda alloc: _score(order, alloc)[0])

    def _pin_floors(order, subset_caps):
        """Layer floors that FORCE the boundary onto a trusted CONTIGUOUS prefix/suffix of `order`,
        or None if this order can't hold the boundary safely. The input boundary [0,b_in) must fall on
        a trusted run from the front (each front node trusted until b_in layers are covered); likewise
        [n-b_out,n) on a trusted run from the back. This is what makes the boundary guarantee hold when
        a single end node is too small (b_out > tail cap) — the spill lands on the next trusted stage by
        construction, not by luck of the greedy fill. Both ends are trusted regardless (role leak)."""
        if order[0] not in trust or order[-1] not in trust:
            return None
        if b_in + b_out >= n_layers:
            # the two windows meet/overlap -> the WHOLE model is boundary, every stage must be trusted.
            # Don't sum a front + back floor here (that double-counts the shared layers and false-
            # infeasibles even an all-trusted ring): require all-trusted, let assign_layers tile freely.
            return {} if all(n in trust for n in order) else None
        floors = {}
        need, i = b_in, 0
        while need > 0 and i < len(order):
            nd = order[i]
            if nd not in trust:
                return None                                          # untrusted node inside the input boundary
            take = min(subset_caps[nd], need)
            floors[nd] = max(floors.get(nd, 0), take)
            need -= take; i += 1
        if need > 0:
            return None                                              # trusted prefix can't cover b_in
        need, j = b_out, len(order) - 1
        while need > 0 and j >= 0:
            nd = order[j]
            if nd not in trust:
                return None                                          # untrusted node inside the output boundary
            take = min(subset_caps[nd], need)
            floors[nd] = max(floors.get(nd, 0), take)
            need -= take; j -= 1
        if need > 0:
            return None
        return floors

    def _pin_orders(subset, k):
        """Cost-ordered orders with a trusted head (require when pinned) + trusted tail. Yields ALL of
        them (cheapest first) so _search can take the cheapest that also holds the boundary safely —
        picking only the single min-latency order (then rejecting it) is the false-infeasible class:
        another order of the same subset may seat the trusted nodes where the boundary needs them.
        Exhaustive for ring-sized k (<=7 middles, 5040 orders); a heuristic single order above — rings
        are ~5-8 stages and the network funnels the pool to ~16 before calling, so the exhaustive path
        is the hot one and the k! is bounded; the fallback only guards a pathologically wide pool."""
        Lm, om, im = (EL, Eout, Ein) if aware else (L, c_out, c_in)
        heads = [require] if require is not None else [n for n in subset if n in trust]
        ends = {n for n in subset if n in trust}
        if k == 1:
            return [[h] for h in heads if h in ends]
        if k - 1 > 7:                                                # oversize fallback (funnel keeps k small)
            o = _pin_order_oversize(subset, Lm, om, im, heads, ends)[0]
            return [o] if o else []
        cand = []
        for h in heads:
            mids = [n for n in subset if n != h]
            for perm in permutations(mids):
                if perm[-1] not in ends:                             # trusted tail
                    continue
                order = [h, *perm]
                cand.append((loop_cost(order, Lm, om, im), order))
        cand.sort(key=lambda x: x[0])
        return [o for _, o in cand]

    def _search(k_lo, k_hi):
        found = None
        for k in range(k_lo, min(k_hi, len(usable)) + 1):
            for subset in combinations(usable, k):
                if require is not None and require not in subset:        # coord/head must be in the ring
                    continue
                if pin and sum(1 for n in subset if n in trust) < (1 if k == 1 else 2):
                    continue                                             # both ends must be trusted (distinct if k>1)
                if len(set(isolation_key(n) for n in subset)) < k or not unique_devices(subset):
                    continue
                subset_caps = {n: caps[n] for n in subset}

                if pin:
                    # cheapest order whose trusted prefix/suffix can hold the boundary AND whose greedy
                    # fill keeps every boundary layer trusted. Ordered by cost, so the first hit is best.
                    for order in _pin_orders(subset, k):
                        if isolation == "adjacent_host" and _is_adjacent_same_host(order, host_id):
                            continue
                        floors = _pin_floors(order, subset_caps)
                        if floors is None:
                            continue
                        if tail_floor > 0:
                            floors = dict(floors)
                            floors[order[-1]] = max(floors.get(order[-1], 0), tail_floor)
                        alloc = allocate(order, subset_caps, floors)
                        if alloc is None:
                            continue
                        if not _boundary_nodes(order, alloc, n_layers, b_in, b_out) <= trust:
                            continue                                     # a spilled boundary layer (belt-and-braces)
                        rank, step, pf = _score(order, alloc)
                        if found is None or rank < found[0]:
                            found = (rank, order, alloc, k, step, pf)
                        if cost_model is None and not tail_reserve_mb:
                            break                                        # legacy latency-ordered first valid
                    continue
                if isolation == "adjacent_host" or tail_host_layer_caps or tail_reserve_mb or cost_model is not None:
                    matrices = (EL, Eout, Ein) if aware else (L, c_out, c_in)
                    # Tail reservations belong to its shared memory domain. Try
                    # another tail when the cheapest order cannot hold its pool;
                    # rejecting the whole subset would lose a valid deployment.
                    tails = set()
                    for tail in subset:
                        trial = [n for n in subset if n != tail] + [tail]
                        floors = {tail: tail_floor} if tail_floor > 0 else None
                        if allocate(trial, subset_caps, floors) is not None:
                            tails.add(tail)
                    if cost_model is not None or tail_reserve_mb:
                        # Role budgets and finite-window costs depend on which
                        # node is tail. Compare each feasible tail, not just the
                        # unconstrained cheapest network orientation.
                        for tail in sorted(tails):
                            order, _ = _constrained_loop(subset, *matrices,
                                host_id if isolation == "adjacent_host" else None, require, {tail})
                            if order is None:
                                continue
                            floors = {tail: tail_floor} if tail_floor > 0 else None
                            alloc = allocate(order, subset_caps, floors)
                            if alloc is None:
                                continue
                            rank, step, pf = _score(order, alloc)
                            if found is None or rank < found[0]:
                                found = (rank, order, alloc, k, step, pf)
                        continue
                    order, _ = _constrained_loop(subset, *matrices,
                        host_id if isolation == "adjacent_host" else None, require, tails)
                    if order is None:
                        continue
                else:
                    order, _ = optimal_loop(subset, EL, Eout, Ein) if aware else optimal_loop(subset, L, c_out, c_in)
                if require is not None and order[0] != require:      # deployable orientation: coord box = stage 0
                    order = (_head_first(order, require, EL, Eout, Ein) if aware
                             else _head_first(order, require, L, c_out, c_in))
                if isolation == "adjacent_host" and _is_adjacent_same_host(order, host_id):
                    continue
                floors = {order[-1]: tail_floor} if tail_floor > 0 else None
                alloc = allocate(order, subset_caps, floors)
                if alloc is None:
                    continue
                rank, step, pf = _score(order, alloc)
                if found is None or rank < found[0]:                     # ties: smaller k wins (k ascending, strict <)
                    found = (rank, order, alloc, k, step, pf)
        return found

    # P1-4: Reject over-sharding. Restrict ring width to max_stages (default 6) unless model requires more.
    eff_max = max(k_min, int(max_stages) if max_stages is not None else 6)
    kmax = min(k_min + slack, eff_max)
    best = _search(k_min, kmax)
    if best is None and kmax < len(usable):                      # co-location can push the true minimum k above k_min+slack
        best = _search(kmax + 1, min(eff_max, len(usable)))      # -> widen up to eff_max first
    if best is None and eff_max < len(usable):
        best = _search(eff_max + 1, len(usable))                 # ultimate fallback if pool truly needs wider ring
    if best is None:
        return None
    rank, order, alloc, k, step, pf = best
    blocks, lo = {}, 0
    for n in order:
        blocks[n] = (lo, lo + alloc[n]); lo += alloc[n]
    dropped = [n for n in nodes if n not in order]
    spec = {"order": order, "blocks": blocks, "layers": alloc, "step_ms": round(step, 1),
            "tok_s_per_g": round(1000.0 / step, 2) if step > 0 else INF, "dropped": dropped, "k": k}
    b_nodes = _boundary_nodes(order, alloc, n_layers, b_in, b_out) if pin else set()
    if pin:
        spec["boundary"] = [n for n in order if n in b_nodes]    # trust-critical stages, ring order
    if aware:
        spec["prefill_ms"] = round(pf, 1)
        spec["request_ms"] = round(rank, 1)
        if relegate:
            spec["roles"] = _relegate(order, dropped, caps, subnet, up_mbps, layer_ms,
                                      trust if pin else None, {subnet[n] for n in b_nodes})
    return spec


# ---- demo: a scattered-US mesh, optimal loop vs naive ordering ----
if __name__ == "__main__":
    # ~ms one-way, lat/long-ish placement; internet RTT, not distance: a couple of
    # asymmetric peering quirks baked in so geography is NOT the answer.
    cities = ["WA", "OR", "CA", "TX", "KS", "IL", "GA", "NC", "VA", "NY"]
    xy = {"WA": (0, 9), "OR": (0, 7), "CA": (1, 3), "TX": (5, 1), "KS": (6, 5),
          "IL": (8, 6), "GA": (9, 2), "NC": (11, 3), "VA": (11, 5), "NY": (12, 8)}
    def base(a, b):
        (x1, y1), (x2, y2) = xy[a], xy[b]
        return 4.0 + 2.3 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5      # ms
    n = len(cities)
    L = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(cities):
        for j, b in enumerate(cities):
            if i != j:
                L[i][j] = base(a, b) + (3.0 if (i + 2 * j) % 5 == 0 else 0.0)  # asym noise
    L[2][8] = L[8][2] = 9.0          # CA<->VA: a fat peering pipe, far but fast
    c_out = [base("WA", c) for c in cities]   # coordinator near WA (entry hop)
    c_in = [base("WA", c) * 0.9 for c in cities]  # direct return, slightly cheaper path

    nodes = list(range(n))
    geo = loop_cost(nodes, L, c_out, c_in)            # input order = a clean geographic guess
    join = [4, 9, 2, 7, 0, 5, 8, 3, 6, 1]            # arbitrary join order (the real case)
    join_cost = loop_cost(join, L, c_out, c_in)
    order, cost = optimal_loop(nodes, L, c_out, c_in)
    name = lambda o: " -> ".join(cities[i] for i in o)
    print(f"arbitrary join order {join_cost:6.1f} ms   {name(join)}")
    print(f"geographic guess     {geo:6.1f} ms   {name(nodes)}")
    print(f"OPTIMAL loop         {cost:6.1f} ms   {name(order)}")
    print(f"  -> {join_cost / cost:.2f}x vs how nodes actually join, {geo / cost:.2f}x vs a hand geo-guess")
    sub_order, sub_cost = select_and_order(nodes, L, c_out, c_in, k=6)
    print(f"best 6 of {n}          {sub_cost:6.1f} ms   {name(sub_order)}")

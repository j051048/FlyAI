"""scheduler / control plane — light and replaceable.

fits the target model to the currently-joined (heterogeneous) gpus, orders them
into a pipeline preferring low-latency edges, tracks health, and reassigns blocks
when a node drops. holds no weights and no user data, so decentralizing it later
(rotating/elected) is a follow-up, not a rewrite. hosted by the c0mpute
orchestrator at first.
"""

from dataclasses import dataclass
from .node import LayerRange


@dataclass
class JoinedNode:
    node_id: str
    vram_gb: float
    rtt_ms: dict  # node_id -> measured rtt to other nodes
    ram_gb: float = 0.0              # Host RAM (GB)
    pinnable_ram_gb: float = 0.0     # Pinned RAM headroom (GB)
    h2d_gbps: float = 20.0           # Measured H2D PCIe bandwidth (GB/s)
    gpu_model: str = "rtx_5090"
    host_id: str | None = None
    gpu_uuid: str | None = None
    gpu_uuids: tuple[str, ...] = ()
    subnet: str | None = None
    public_ip: str | None = None  # transport information, never physical-host identity
    memory_domain_id: str | None = None


def _distribute(total: int, caps: list[tuple[str, int]]) -> dict[str, int]:
    """Hand out `total` layers across nodes proportional to their layer capacity, never
    exceeding a node's cap, summing to exactly `total`. Largest-remainder rounding, then the
    leftover (from capped/rounded nodes) is pushed onto the nodes with the most spare capacity
    (fat nodes first). Caller pre-checks sum(cap) >= total, so this always closes."""
    ids = [i for i, _ in caps]
    cap = dict(caps)
    capsum = sum(cap.values())
    base = {i: min(cap[i], (total * cap[i]) // capsum) for i in ids}      # floor share, capped
    assigned = sum(base.values())
    # distribute the remainder to nodes with the largest fractional share that still have room
    rem = total - assigned
    order = sorted(ids, key=lambda i: (-(total * cap[i] % capsum), -cap[i]))
    k = 0
    while rem > 0:
        i = order[k % len(order)]
        if base[i] < cap[i]:
            base[i] += 1
            rem -= 1
        k += 1
        if k > 4 * len(order) * (total + 1):                              # safety, never trips if sum(cap)>=total
            break
    return base


class Scheduler:
    def __init__(self, model: str, total_layers: int):
        self.model = model
        self.total_layers = total_layers
        self.nodes: dict[str, JoinedNode] = {}

    def register(self, node: JoinedNode) -> None:
        self.nodes[node.node_id] = node

    def deregister(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)

    def capacities(self, gb_per_layer: float, kv_gb_per_layer: float = 0.0,
                   headroom_gb: float = 2.0, boundary_gb: float = 1.0) -> dict[str, int]:
        """max layers each node's VRAM can hold: model bytes/layer + KV bytes/layer at the
        target context, minus a runtime headroom (activations/graph) and the boundary weights
        (embed/lm_head, charged once as slack on every node so the head/tail always fit)."""
        per = gb_per_layer + kv_gb_per_layer
        return {nid: max(0, int((n.vram_gb - headroom_gb - boundary_gb) / per))
                for nid, n in self.nodes.items()}

    def capacities_dual(self, gb_per_layer_vram: float, gb_per_layer_host: float = 0.0,
                        kv_gb_per_layer: float = 0.0, headroom_gb: float = 2.0,
                        boundary_gb: float = 1.0) -> dict[str, int]:
        """Dual-resource capacity: takes the minimum of VRAM and Host RAM (or pinned RAM) limits."""
        vram_caps = self.capacities(gb_per_layer_vram, kv_gb_per_layer, headroom_gb, boundary_gb)
        if gb_per_layer_host <= 0.0:
            return vram_caps
        caps = {}
        for nid, n in self.nodes.items():
            v_cap = vram_caps[nid]
            avail_ram = n.pinnable_ram_gb if n.pinnable_ram_gb > 0 else n.ram_gb
            r_cap = max(0, int(avail_ram / gb_per_layer_host)) if avail_ram > 0 else v_cap
            caps[nid] = min(v_cap, r_cap)
        return caps

    def plan(self, gb_per_layer: float | None = None, kv_gb_per_layer: float = 0.0,
             headroom_gb: float = 2.0, boundary_gb: float = 1.0,
             model_id: str | None = None, placement: str = "gpu", isolation: str = "none") -> dict:
        """ONE joint placement: pipeline order and contiguous blocks decided together."""
        from .plan import plan_ring, profile_for
        ids = list(self.nodes)
        nodes = [{
            "id": nid,
            "free_vram_mb": self.nodes[nid].vram_gb * 1024.0,
            "free_ram_mb": self.nodes[nid].ram_gb * 1024.0 if self.nodes[nid].ram_gb else None,
            "pinnable_ram_mb": self.nodes[nid].pinnable_ram_gb * 1024.0 if self.nodes[nid].pinnable_ram_gb else None,
            "h2d_gbps": self.nodes[nid].h2d_gbps,
            "subnet": self.nodes[nid].subnet,
            "host_id": self.nodes[nid].host_id,
            "gpu_uuid": self.nodes[nid].gpu_uuid,
            "gpu_uuids": self.nodes[nid].gpu_uuids,
            "public_ip": self.nodes[nid].public_ip,
            "memory_domain_id": self.nodes[nid].memory_domain_id,
        } for nid in ids]
        rtt = [[0.0 if a == b else float(self.nodes[a].rtt_ms[b]) for b in ids] for a in ids]

        if model_id is not None or self.model in ("deepseek-ai/DeepSeek-V4-Flash-0731", "v4"):
            mid = model_id or "deepseek-ai/DeepSeek-V4-Flash-0731"
            base_model = profile_for(mid)
            if placement == "ram":
                base_model = profile_for("deepseek-ai/DeepSeek-V4-Flash-0731-Dual")
            elif placement == "gpu":
                base_model = profile_for("deepseek-ai/DeepSeek-V4-Flash-0731-Resident")
            model = dict(base_model)
            if gb_per_layer is not None:
                model["layer_vram_mb"] = gb_per_layer * 1024.0
        else:
            gb_val = gb_per_layer if gb_per_layer is not None else 1.0
            model = {
                "n_layers": self.total_layers,
                "layer_vram_mb": gb_val * 1024.0,
                "kv_mb_per_layer": kv_gb_per_layer * 1024.0,
                "layer_ms_base": 0.65,
                "reserve_mb": headroom_gb * 1024.0,
                "head_reserve_mb": boundary_gb * 1024.0,
                "tail_reserve_mb": boundary_gb * 1024.0,
                "cap_layers": self.total_layers,
                "head_layer_ms_mult": 1.0,
            }
        out = plan_ring(nodes, rtt, model, isolation=isolation)
        if out is None:
            raise ValueError(f"insufficient resources: pool cannot hold {self.total_layers} layers "
                             f"under placement policy {placement!r}")
        return out

    def allocate(self, gb_per_layer: float, kv_gb_per_layer: float = 0.0,
                 headroom_gb: float = 2.0, boundary_gb: float = 1.0) -> dict[str, LayerRange]:
        """DEPRECATED as half of a pair: composing allocate() with topology() is incoherent (this
        gives the coordinator a block and orders ranges fat-first; topology() excludes it and
        orders by latency — the two orderings disagree). Use plan() for a deployable result.

        assign each node a contiguous block that fits its vram, covering the whole stack,
        FAT NODES FIRST (a 48GB card holds more layers than a 24GB one -> fewer nodes, fewer
        WAN hops). Contiguous so each node's KV-cache is a simple per-block window; the block's
        layer indices reindex 0-based on the node (pipeline.load_stage --lo/--hi).

        privacy note (docs/ARCHITECTURE.md#privacy): a later pass pins the embedding + final
        blocks to trusted/staked nodes and leaves only deep middle blocks to untrusted
        volunteers; this fit is the VRAM-feasibility layer that lands under that policy.
        """
        cap = self.capacities(gb_per_layer, kv_gb_per_layer, headroom_gb, boundary_gb)
        if sum(cap.values()) < self.total_layers:
            raise ValueError(f"insufficient VRAM: capacity {sum(cap.values())} layers "
                             f"< model {self.total_layers}")
        order = sorted(self.nodes, key=lambda nid: -self.nodes[nid].vram_gb)   # fat node first
        counts = _distribute(self.total_layers, [(nid, cap[nid]) for nid in order])
        out, cur = {}, 0
        for nid in order:
            c = counts[nid]
            if c == 0:
                continue
            out[nid] = LayerRange(cur, cur + c)
            cur += c
        return out

    def topology(self, coordinator_id: str, k: int | None = None) -> list[str]:
        """DEPRECATED as half of a pair: see allocate() — use plan() for a deployable result.

        order nodes into the cheapest pipeline loop on the measured rtt mesh.

        the coordinator is the depot (entry hop out, direct-return hop back); the stage
        order is the min-latency Hamiltonian loop through it. with k set, also selects the
        best k of the joined nodes. see shard/topology.py for the solver.
        """
        from .topology import optimal_loop, select_and_order
        ids = [nid for nid in self.nodes if nid != coordinator_id]
        coord = self.nodes[coordinator_id]
        L = [[0.0 if a == b else self.nodes[a].rtt_ms[b] for b in ids] for a in ids]
        c_out = [coord.rtt_ms[a] for a in ids]
        c_in = [self.nodes[a].rtt_ms[coordinator_id] for a in ids]
        idx = list(range(len(ids)))
        order, _ = (select_and_order(idx, L, c_out, c_in, k) if k else
                    optimal_loop(idx, L, c_out, c_in))
        return [ids[i] for i in order]

    def on_drop(self, node_id: str, gb_per_layer: float, kv_gb_per_layer: float = 0.0,
                **fit) -> dict[str, LayerRange]:
        """node dropped: remove it and re-fit the survivors into a fresh allocation. Returns the
        new assignment (the caller rebuilds the ring + re-prefills the in-flight request).
        Raises if the survivors no longer have the VRAM to hold the model -> the scheduler must
        pull a replacement from the pool before serving. Seamless MID-REQUEST KV migration (no
        re-prefill) is the genuine research item (roadmap step 7 / INTEGRATION.md §11) and is NOT
        what this does -- this is the correct, available 'heal by rebuild' floor."""
        self.deregister(node_id)
        return self.allocate(gb_per_layer, kv_gb_per_layer, **fit)

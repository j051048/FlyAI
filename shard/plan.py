"""Deployable-ring planner — the seam c0mpute's control plane calls to place a sharded swarm.

`select_ring` (topology.py) is the pure decision core: clean numbers in, a ring out. But turning what
a node ANNOUNCES (GPU, free VRAM, a CPU probe, its subnet) plus a measured RTT mesh into those clean
numbers takes engine calibration — VRAM reserves, the launch-bound per-layer time, and head placement.
That calibration lived inline in `scratchpad/ring_up.py` (throwaway SSH glue). `plan_ring` lifts it into
a tracked, tested function so the network layer can PLACE a swarm with one call instead of re-deriving
the engine's memory/compute model.

Boundary (docs/INTEGRATION.md): c0mpute owns MEASUREMENT (collecting `nodes` + `rtt`); shard owns the
ENGINE model (reserves, per-layer ms, the select_ring decision). Deps point one way: c0mpute -> shard.
`python3 -m shard.plan` reads `{nodes, rtt, model}` JSON on stdin and prints the plan on stdout, so a
TypeScript orchestrator drives the same proven planner as `ring_up` without porting its subtleties.
"""
import json
import math
import sys

from .topology import select_ring

# M2.5-on-5090 anchors (docs/M25_ENGINE.md, mirrored from ring_up.py) — the DEFAULT model profile.
# The caller (c0mpute's model catalog) SHOULD pass an explicit `model` profile; these keep the seam
# runnable for M2.5 without one.
M25_PROFILE = {
    "n_layers": 62,
    "layer_vram_mb": 2330.0,     # NVFP4 experts + bf16 attn + norms, per decoder layer — MEASURED
                                 # 2026-07-09 (capability probe, one real layer resident: 2329.5 MB;
                                 # a warm 13-layer stage read 31.5/32.6 GiB — the old 1700 estimate
                                 # under-modeled by ~35% and packed stages one allocation from OOM)
    "kv_mb_per_layer": 150.0,    # at the 40960 KV cap (B=1; batched callers scale by B*maxlen/40960)
    "layer_ms_base": 0.65,       # per-layer decode compute on an idle fast-CPU 5090 box
    "reserve_mb": 1500.0,        # CUDA context + allocator slack per box
    "head_reserve_mb": 4096.0,   # coordinator process on the head: embed + EAGLE head + its context
    "tail_reserve_mb": 1400.0,   # tail stage: final norm + lm_head (measured 1.15 GiB bf16 + slack —
                                 # a 13-layer tail OOM'd on it live while 13-layer middles warmed fine)
    "cap_layers": 12,            # 32 GB ceiling by MEASURED footprint ((32768-1500)/2480 = 12.6);
                                 # 13 ran warm but at 31.5/32.6 GiB — brim-riding, not a plan target
    "head_layer_ms_mult": 1.3,   # the head box also runs the coordinator
    # per-hop activation payloads for upload-aware placement (bf16 wire, hidden 3072 — see
    # phase0/m25_stage.py). Without these the residential objective priced every byte at ZERO and
    # upload-aware placement decayed to pure latency. Nominal request unit: a one-chunk (4096-tok)
    # prefill + 256 decode traversals of a (K+1)-token draft bundle (K=8, the measured sweet spot);
    # callers with a real workload override (fp8 wire => halve bytes, long prompts => S*H*2 and
    # prefill_chunks = ceil(S/4096)).
    "prefill_bytes": 4096 * 3072 * 2.0,   # chunk*H*2 — one [chunk,H] prefill hop (~25 MB)
    "decode_bytes": 9 * 3072 * 2.0,       # (K+1)*H*2 — one draft round's hidden bundle (~54 KB)
    "decode_steps": 256,                  # nominal decode traversals per request (~600 tok @ g~2.5)
    "prefill_chunks": 1,
}

# ── Kimi-K3 ───────────────────────────────────────────────────────────────────
# Every number below is MEASURED on real K3 weights (the G0 kernel probe, 2026-07-28, sm_120 —
# scratchpad/k3-research-20260727/g0-results.md) or read off the checkpoint's own config/shard
# headers. K3 is 93 decoder layers of a 2.78T-param MoE shipped natively in MXFP4; a layer is an
# ATOMIC 15.8 GiB stage unit, which is what makes its placement a different problem from M2.5's.
_K3_N_LAYERS = 93
_K3_HIDDEN = 7168                # text_config.hidden_size
_K3_MLA_LAYERS = 24              # gated-MLA layers (0-indexed 3,7,11,…,91,92 — a 3:1 interleave)
_K3_KDA_LAYERS = 69              # Kimi delta-attention layers: FIXED recurrent state, no paged KV
_K3_MLA_KV_BYTES_PER_TOKEN = 576 * 2.0    # kv_lora_rank 512 + qk_rope 64 elements, bf16, per layer
_K3_KDA_STATE_MB = 6.0                    # MEASURED recurrent state per KDA layer per REQUEST (fp32)
# AttnRes: a K3 layer boundary carries `prefix_sum` + a `block_residual` stack of `num_blocks`
# snapshots, i.e. (1+num_blocks) x hidden x 2B per token, not one hidden state. Blocks are appended
# at layers {0,12,24,36,48,60,72,84}, so the payload steps 2 units (28 KiB/token after L11) up to
# 9 units (126 KiB/token at L92) against a plain transformer's 1 unit (14 KiB). Summed over the 92
# layer boundaries that is 492 units => a MEAN of 492/92 = 5.348 units = 74.87 KiB/token.
_K3_BOUNDARY_UNITS_MEAN = 492 / 92
_K3_BOUNDARY_BYTES_PER_TOKEN = _K3_HIDDEN * 2.0 * _K3_BOUNDARY_UNITS_MEAN     # ~76.7 kB/token/hop
_K3_PREFILL_CHUNK = 1024         # tokens per pipelined prefill chunk — see the prefill_bytes note


def k3_kv_mb_per_layer(batch: int = 8, maxlen: int = 32768) -> float:
    """K3's per-request state, folded into the planner's ONE per-layer KV number.

    The planner (and probe.derive_layers) models a block as `layers * (layer_vram_mb +
    kv_mb_per_layer)`: one scalar, charged to every layer alike. K3's state is not one thing —
    24 MLA layers hold a paged KV cache that grows with CONTEXT (576 elements/token/layer bf16 =
    27.6 kB/token across the 24), and 69 KDA layers hold a FIXED recurrent state that does not
    grow with context at all (6.00 MiB/layer/request measured, 0.404 GiB/request across the 69).

    The honest fold is the per-layer AVERAGE at a stated (batch, maxlen), which is what this
    returns — not a per-layer vector, because there is nowhere to put one. The approximation is
    good because the MLA layers are evenly interleaved 3:1, so any contiguous block of >=4 layers
    carries close to its share: the worst realistic case (a 5-layer stage that lands 2 MLA layers
    instead of 1.3) is under-modeled by ~170 MB, against the ~5 GB of slack a 5-layer stage has
    left on a 96 GB card. State that does NOT scale this way (a 1M-token context is 27 GiB of MLA
    KV per request on its own) must be planned by calling this with the real numbers.
    """
    mla_mb = _K3_MLA_LAYERS * _K3_MLA_KV_BYTES_PER_TOKEN * maxlen * batch / (1024 * 1024)
    kda_mb = _K3_KDA_LAYERS * _K3_KDA_STATE_MB * batch          # context-independent, per request
    return (mla_mb + kda_mb) / _K3_N_LAYERS


K3_PROFILE = {
    "n_layers": _K3_N_LAYERS,
    "layer_vram_mb": 16203.0,    # 15.823 GiB — MEASURED 2026-07-28: one layer resident on a 5090
                                 # (15.823) and a 5-layer pack on a 96 GB Pro 6000 (79.113/5 =
                                 # 15.8226), agreeing to 0.4 MB. Not an independent sample: the pack
                                 # reused layer 46's weights 5x, so it is the same tensors on a
                                 # second card. Includes the Marlin-repacked
                                 # MXFP4 experts (14.643 GiB, NOT dequantized) + bf16 attention,
                                 # shared experts, latent projections and router.
                                 #
                                 # UNIFORM, and deliberately the MAXIMUM. K3's layers differ (68
                                 # KDA-MoE at 15.823 GiB, 24 MLA-MoE at 15.429, layer 0 dense at
                                 # 2.180), but nothing downstream can express that: plan_ring and
                                 # select_ring both size a block as layers*(layer_vram+kv) from one
                                 # scalar. Given one number, it must be the heaviest layer —
                                 # under-modeling VRAM is the failure mode that OOMs a stage at load
                                 # (M2.5's 1700-vs-2330 estimate packed stages one allocation from
                                 # OOM). The over-model is small and paid in the safe direction:
                                 # 93*15.823 = 1471.5 GiB planned against 1448.4 GiB real (+1.6%),
                                 # concentrated in the one stage that holds layer 0 (13.6 GiB heavy
                                 # — it may plan 5 layers where 6 would physically fit).
    "kv_mb_per_layer": k3_kv_mb_per_layer(batch=8, maxlen=32768),   # 109.9 MB — see the function.
                                 # B=8 x 32k is the phase-1 target; callers with a real workload
                                 # recompute (the MLA term scales with B*maxlen, the KDA term with
                                 # B alone — they do NOT share a scale factor, so no single
                                 # multiplier corrects this number).
    "layer_ms_base": 1.15,       # MEASURED B=1 with a whole-layer CUDA graph: 1.1186 ms on a 5090,
                                 # 1.1738 on a Pro 6000. EAGER is 1.20-1.51 — a node whose graph
                                 # capture failed must announce layer_ms=1.4, not be planned at
                                 # this. (93 layers => C = 107 ms graphed, 130 eager.)
                                 # CAVEAT, and G0 calls it the single biggest source of error in C:
                                 # the layer measured is a KDA layer, and 24 of the 93 are gated-MLA
                                 # layers that read a paged KV cache and will cost MORE at long
                                 # context. Treating all 93 as KDA-like is an approximation.
    "reserve_mb": 1500.0,        # CUDA context + allocator slack per box, as on M2.5, and it has to
                                 # cover a MEASURED gap: at the 5-layer pack the driver reported
                                 # 89.661 GiB in use against an allocator peak of 88.332 — 1.329 GiB
                                 # of context and fragmentation the allocator's own numbers never
                                 # show (the context alone is 0.546 GiB at zero allocation). The K3
                                 # load transient is NOT folded in here — see load_peak_extra_mb.
    "load_peak_extra_mb": 9440.0,  # 9.218 GiB: loading an MXFP4 layer peaks at resident + the
                                 # largest single tensor being Marlin-repacked (w13, 9.19 GiB).
                                 # MEASURED 25.041 GiB peak for a 15.823 GiB layer, IDENTICAL on
                                 # both cards — a property of the model, not of the box, which is
                                 # why the profile declares it. A node that never ran a K3 probe is
                                 # then still peak-gated, and a node that DID measure it overrides
                                 # rather than adds (plan_ring/probe.derive_layers take the node's
                                 # value when present). Docking both would subtract ~19 GB and cost
                                 # a 32 GB card its only layer — the all-5090 ring would read as
                                 # infeasible, not merely conservative.
                                 #
                                 # This is what produces the G0-proven caps by arithmetic rather
                                 # than by a hardcoded ceiling: (97249-10940)/16313 = 5.29 layers on
                                 # a 96 GB card, (32607-10940)/16313 = 1.33 on a 32 GB 5090.
    "head_reserve_mb": 4096.0,   # coordinator on the head: embed_tokens 2.188 GiB (2241 MB, untied)
                                 # + the coordinator process, tokenizer and logit buffers. Phase 1
                                 # has no drafter to hold (num_nextn_predict_layers = 0, no MTP or
                                 # EAGLE head in the checkpoint). FUTURE: the DSpark drafter is
                                 # 21.4 GB (~20.4 GiB) — adopting it raises this reserve by more
                                 # than a whole layer slot, so a phase-2 head holds ~4 layers where
                                 # it holds 5 today. On a 32 GB card this reserve is what makes the
                                 # head the tightest seat in an all-5090 ring: it needs
                                 # per_layer + reserve + load_peak + this = 31349 MB free, so a 5090
                                 # with more than ~1.2 GB already in use cannot head one.
    "tail_reserve_mb": 2500.0,   # final norm + lm_head 2.188 GiB (2241 MB, untied) + the model-level
                                 # output AttnRes proj/norm (KB) + slack. Same failure this guards on
                                 # M2.5: a tail packed to its brim OOMs loading the output head.
    "cap_layers": 5,             # G0-PROVEN on a 94.97 GiB Pro 6000: 5 layers resident 79.113 GiB /
                                 # 88.332 GiB peak, and the SIXTH layer OOMs allocating 588 MiB. A
                                 # 32 GB 5090 holds exactly ONE (25.04 GiB peak, 6.3 GiB headroom).
                                 # Note both of those already fall out of the reserve arithmetic
                                 # above: this stays a non-binding SANITY ceiling, which is the only
                                 # thing it can honestly be here. density_cap_layers scales a proven
                                 # cap LINEARLY with card size, and K3's true ceiling is not linear
                                 # in it — the repack transient is a fixed additive term, so the
                                 # real rule is floor((vram - 9.2 GiB) / 15.8 GiB). The linear rule
                                 # reads 14-15 layers on a 96 GB card, well above the budget's 5,
                                 # so it never binds and never has to be right.
    "head_layer_ms_mult": 1.3,   # carried from M2.5 (the coordinator penalty is engine-side, not
                                 # model-side) — NOT re-measured on K3.
    # per-hop payloads. K3's boundary is not a hidden state: see _K3_BOUNDARY_BYTES_PER_TOKEN.
    # predict_step_ms charges ONE decode_bytes to every stage alike, so the cost model cannot
    # express a payload that grows with depth; the mean over the 92 boundaries is the faithful
    # single number, and a ring's stage boundaries are near-evenly spaced so their mean lands on
    # it. The error is per-hop (a head stage is over-charged ~2.7x, the tail under-charged ~1.7x),
    # not in aggregate.
    "prefill_bytes": 4096 * _K3_BOUNDARY_BYTES_PER_TOKEN,   # S*(1+nb)*H*2 for a 4096-token prompt
                                 # = 299 MiB/hop, against M2.5's 25 MB. Pipelined as prefill_chunks,
                                 # and the chunk is what has to FIT: transport.MAX_FRAME is 256 MiB,
                                 # and at the deepest boundary (9 units, 126 KiB/token) a 4096-token
                                 # chunk is 504 MiB — unsendable. 2048 lands at 252 MiB, inside the
                                 # frame but with no room for framing overhead. 1024 (126 MiB
                                 # worst-case) is the largest chunk that is comfortably deployable,
                                 # so the nominal 4096-token prefill is FOUR chunks, not one.
                                 # THE ENGINE DOES NOT DEFAULT TO THIS: coordinate.py ships
                                 # --prefill-chunk 4096, which a K3 ring must override or its first
                                 # deep hop dies on "frame length ... exceeds MAX_FRAME".
    "decode_bytes": 1 * _K3_BOUNDARY_BYTES_PER_TOKEN,       # ~74.9 KiB — one token's boundary
                                 # payload. Phase 1 runs g=1: the checkpoint carries no MTP head and
                                 # no drafter is trained, so a traversal moves ONE token, not M2.5's
                                 # (K+1)-token draft bundle.
    "decode_steps": 600,         # nominal decode traversals per request. Same nominal ~600-token
                                 # response M25_PROFILE models, but at g=1 that is 600 traversals
                                 # rather than 256 — and K3 reasons on every request by default.
    "prefill_chunks": 4096 // _K3_PREFILL_CHUNK,
}

# The engine profile per catalog model_id — the seam c0mpute's control plane resolves before it
# calls plan_ring (its catalog holds a model_id and a manifest ref; the calibration is ours). Keys
# are the manifest model_id, which NAMES THE QUANT: two quantizations of one model are two entries,
# never one, because layer_vram_mb (and the weight_map behind it) differ.
# ── DeepSeek-V4-Flash-0731 ────────────────────────────────────────────────────
# 43 backbone layers, 256 routed experts (FP4, ~12.75 MiB each = 3264 MiB/layer), 6 activated experts.
# 3 MTP draft blocks, 4096 hidden dimension with four HC streams.
_V4_N_LAYERS = 43
_V4_EXPERT_BYTES = 13369344          # ~12.75 MiB per FP4 routed expert
_V4_EXPERTS_PER_LAYER = 256
_V4_ROUTED_HOST_MB = (_V4_EXPERTS_PER_LAYER * _V4_EXPERT_BYTES) / (1024 * 1024)  # 3264.0 MB

V4_ALL_RESIDENT_PROFILE = {
    "n_layers": _V4_N_LAYERS,
    "layer_vram_mb": 3614.0,         # ~350 MB resident (attn+shared+gate) + 3264 MB routed experts
    "kv_mb_per_layer": 150.0,        # baseline sliding + compressed working KV
    "layer_ms_base": 0.70,
    "reserve_mb": 2048.0,            # CUDA context + graph + allocator slack
    "head_reserve_mb": 3500.0,       # coordinator + token embedding + prefill buffer
    "tail_reserve_mb": 5500.0,       # LM Head + 3 MTP draft blocks
    "cap_layers": 8,                 # 32 GB GPU ceiling all-resident: max 8 layers
    "head_layer_ms_mult": 1.2,
    "placement": "gpu",
    "prefill_bytes": 4096 * 4096 * 4 * 2.0,
    "decode_bytes": 4096 * 4 * 2.0,  # one greedy/pipelined s=1 four-stream frame
    "decode_steps": 256,
    "prefill_chunks": 1,           # query chunks do not imply wire-prefill pipelining
}

V4_DUAL_RESOURCE_PROFILE = {
    "n_layers": _V4_N_LAYERS,
    "layer_vram_mb": 758.0,          # ~350 MB resident + 32 expert cache slots (~408 MB)
    "layer_host_ram_mb": _V4_ROUTED_HOST_MB,  # 3264.0 MB pinned host RAM per layer
    "tail_host_reserve_mb": 3 * _V4_ROUTED_HOST_MB,  # MTP pools; charged once to the tail's RAM domain
    "kv_mb_per_layer": 150.0,
    "layer_ms_base": 0.75,           # fallback compute estimate; exposed DMA priced separately
    "reserve_mb": 2048.0,
    "head_reserve_mb": 3500.0,
    "tail_reserve_mb": 5500.0,
    "cap_layers": 15,                # GPU permits up to 15 layers on 32 GB card in hybrid mode
    "head_layer_ms_mult": 1.2,
    "placement": "ram",
    "expert_slot_bytes": _V4_EXPERT_BYTES,
    "expert_count_per_layer": _V4_EXPERTS_PER_LAYER,
    "default_expert_cache_slots": 32,
    "prefill_bytes": 4096 * 4096 * 4 * 2.0,
    "decode_bytes": 4096 * 4 * 2.0,
    "decode_steps": 256,
    "prefill_chunks": 1,
}

PROFILES = {
    "nvidia/MiniMax-M2.5-NVFP4": M25_PROFILE,
    "moonshotai/Kimi-K3-MXFP4": K3_PROFILE,
    "deepseek-ai/DeepSeek-V4-Flash-0731": V4_DUAL_RESOURCE_PROFILE,
    "deepseek-ai/DeepSeek-V4-Flash-0731-Dual": V4_DUAL_RESOURCE_PROFILE,
    "deepseek-ai/DeepSeek-V4-Flash-0731-Resident": V4_ALL_RESIDENT_PROFILE,
}


def profile_for(model_id: str) -> dict:
    """The engine profile for a catalog `model_id`, as a copy the caller may override.

    Unknown ids raise rather than falling back to M2.5's calibration: planning an unknown model at
    another model's per-layer footprint is exactly the admit-then-OOM failure the measured numbers
    exist to prevent. ValueError, not KeyError — `_main` reserves KeyError for a request that is
    missing a field, and an unserviceable model_id is a bad VALUE, not an absent one."""
    try:
        return dict(PROFILES[model_id])
    except KeyError:
        raise ValueError(f"no engine profile for model_id {model_id!r} "
                         f"(known: {', '.join(sorted(PROFILES))})") from None

_SLACK = 3                       # default pool headroom for the exact subset search: k_min..k_min+3.
                                 # slack=len(nodes) made select_ring's exact search range over EVERY
                                 # k up to the pool size (combinatorial in a wide pool), defeating
                                 # the trim funnel; select_ring still widens past this on its own
                                 # when co-location forces a bigger ring, so feasibility is intact.

_UNREACHABLE = 9000.0            # RTT sentinel: treat >= this as "no usable path" when ranking centrality

_PROVEN_CAP_VRAM_MB = 32768.0    # the card size cap_layers was proven on; bigger cards scale by density


def density_cap_layers(cap_layers, total_vram_mb):
    """The proven layer DENSITY scaled to the card size — a flat cap collapsed a 96 GB card
    to the 32 GB verdict (the spec's core distinction). ONE rule, shared with probe.derive_layers.

    ROUNDED, not truncated. `_PROVEN_CAP_VRAM_MB` is the card's NOMINAL size, and no real card
    reports it: the 5090s this cap was proven on report 32103-32117 MB once ECC/driver overhead is
    taken out. Truncating then docked every one of them a full layer for a <2% shortfall against a
    marketing number, which is how a 7-node pool that physically holds 62 layers was rejected as
    "need more/fatter nodes". Rounding to nearest restores the intended ceiling on the proven card
    without the ceil() behaviour of granting a 13th layer to a card one MB over the anchor (that
    lands at ~98% VRAM, the configuration that OOM'd live).

    This is only ever a SANITY CEILING; the binding rule is the footprint arithmetic in plan_ring
    (measured layer_vram_mb + kv + reserves), which stays strictly conservative.
    """
    return max(0, int(round(int(cap_layers) * float(total_vram_mb) / _PROVEN_CAP_VRAM_MB)))


def ram_dma_overhead_ms(node, model):
    """Decode transfer cost in milliseconds; h2d_gbps is decimal GB/s.

    A measured full layer already includes its cache misses. Only an unmeasured
    layer needs a transfer estimate; optional observed exposed wait takes priority
    over the fallback miss count / overlap fraction.
    """
    if node.get("layer_ms") is not None:
        return 0.0
    def number(value):
        if isinstance(value, bool):
            raise ValueError("RAM transfer calibration must be numeric, not boolean")
        try:
            return float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("RAM transfer calibration must be numeric") from error
    def calibration(name, default):
        value = node.get(name)
        if value is None:
            value = model.get(name)
        return number(default if value is None else value)
    measured = node.get("dma_exposed_ms_per_layer")
    if measured is not None:
        value = number(measured)
        if not math.isfinite(value) or value < 0:
            raise ValueError("measured DMA exposure must be finite nonnegative milliseconds")
        return value
    bandwidth = calibration("h2d_gbps", 20.0)
    misses = calibration("expert_misses_per_layer", 1.8)
    overlap = calibration("dma_overlap_fraction", .7)
    size = number(model.get("expert_slot_bytes", 13369344))
    if not all(math.isfinite(v) for v in (bandwidth, misses, overlap, size)) or \
            bandwidth <= 0 or misses < 0 or size <= 0 or not 0 <= overlap <= 1:
        raise ValueError("invalid RAM transfer calibration (GB/s, misses, overlap, expert bytes)")
    # Bytes / bytes-per-second: multiplying by eight here would price GB/s as Gb/s.
    return misses * size / (bandwidth * 1e9) * 1000.0 * (1.0 - overlap)


def _plan_ring_core(nodes, rtt, model=None, *, slack=None, privacy=None, isolation=None,
                    head_choice=None, joint_roles=False, objective="serial", cost_model=None,
                    hard_max_stages=None, coordinator_costs=None, coordinator_on_head=True):
    """Place a deployable sharded ring from announced capabilities + a measured RTT mesh.

    nodes: [{"id": <hashable>, "free_vram_mb": float, "subnet": str,
             "cpu_factor": float=1.0,          # >=1; pyloop/0.10 + load — a slow/loaded box drafts slower
             "up_mbps": float|None,            # optional; present on ALL nodes -> upload-aware placement
             "trusted": bool=False,            # ASSIGNED by the control plane (stake/reputation), never
                                               # self-reported by the node
             # per-node MEASURED capability (the probe's cap vector; every field optional — absent
             # fields fall back to the model profile, so a homogeneous pool plans byte-identically):
             "layer_vram_mb": float|None,      # this node's per-layer footprint (arch/backend-specific:
                                               # cutlass ~2330, marlin ~4060 — a hetero ring needs both)
             "cap_layers": int|None,           # probe-verdict layer ceiling for THIS card (wins outright)
             "total_vram_mb": float|None,      # else: density-scale the proven cap to the card size
             "load_peak_extra_mb": float|None, # measured load/run transient above resident (peak
                                               # gate). ABSENT -> the model profile's own, when it
                                               # declares one: a transient that is a property of the
                                               # MODEL (K3's MXFP4 repack peaks the same 9.2 GiB on
                                               # every card) must gate a node that never probed it.
                                               # The node's own measurement WINS rather than adding,
                                               # or an honest probe would dock the same peak twice
             "layer_ms": float|None}]          # measured decode ms/layer (overrides base*cpu_factor)
    rtt:   NxN one-way ms matrix, row/col order aligned to `nodes` (rtt[i][i] ignored).
    model: profile dict (see M25_PROFILE), or a catalog model_id resolved through `profile_for`
           (see PROFILES); defaults to M2.5.
    slack: select_ring pool headroom; defaults to min(len(nodes), 3) — enough to drop weak
           boxes without letting the exact subset search range over every k up to the pool size.
    privacy: {"boundary_in": int, "boundary_out": int} — turn on BOUNDARY-LAYER PINNING: the ring's
             head/tail (they handle raw prompt / output tokens) and every stage holding a boundary
             layer must be `trusted` nodes; strangers hold only deep-middle layers. The head is the
             most central TRUSTED capable node under pinning (it runs the coordinator, which sees
             the raw prompt). None (default) = placement exactly as before.
    isolation: "none" (production default), "host", "subnet" or "adjacent_host".
               None uses the model profile's policy, falling back to "none". Host/GPU metadata
               comes from explicit announcements; a public IP does not identify a machine.

    Returns a plan dict, or None if the pool genuinely can't hold the model (with pinning: can't
    hold it SAFELY — e.g. no trusted node for an end):
      {"order":  [node_id, ...],                       # head-first, deployable
       "head":   node_id,
       "stages": [{"id", "index", "lo", "hi", "head", "tail", "layers",
                   "boundary"}...],                    # "boundary" only when privacy pinning is on
       "dropped":[node_id, ...],
       "roles":  {node_id: role},                      # only when every node carries up_mbps
       "step_ms", "tok_s_per_g", "k",
       "request_ms", "prefill_ms",                     # only when upload-aware
       "privacy": {"boundary_in", "boundary_out", "boundary_stages"}}   # only when pinning is on
    """
    if isinstance(model, str):
        model = profile_for(model)
    if isinstance(model, dict) and ("schema" in model
            or model.get("calibration_status") in {"structural", "unmeasured"}):
        raise ValueError("structural/resource evidence is not a calibrated scalar GPU placement profile; "
                         "the memory-aware planner has not been enabled")
    # A supplied profile is its own model; never inherit another model's wire
    # geometry or boundary reserves. Partial generic callers retain neutral defaults.
    generic = dict(kv_mb_per_layer=0.0, layer_ms_base=1.0, reserve_mb=0.0,
                   head_reserve_mb=0.0, tail_reserve_mb=0.0, head_layer_ms_mult=1.0,
                   prefill_bytes=0.0, decode_bytes=0.0, decode_steps=1, prefill_chunks=1)
    if model is not None and "n_layers" not in model:
        if model.get("model_id"):
            raise ValueError("an explicit model profile must declare n_layers")
        # Historical API: {tail_reserve_mb: ...} is an M2.5 override,
        # not an independently identified model descriptor.
        m = {**M25_PROFILE, **model}
    else:
        m = dict(M25_PROFILE) if model is None else {**generic, **model}
    m.setdefault("cap_layers", m["n_layers"])
    isolation = m.get("isolation", "none") if isolation is None else isolation
    if isolation not in ("none", "subnet", "host", "adjacent_host"):
        raise ValueError("isolation must be none, subnet, host or adjacent_host")
    n = len(nodes)
    if n == 0:
        return None
    ids = [nd["id"] for nd in nodes]
    if len(set(ids)) != len(ids):                            # duplicate ids collide in the output maps
        raise ValueError("duplicate node id in `nodes`")     # (order/roles/boundary_stages) -> mis-deploy
    # Network addresses do not identify physical hosts: several independent
    # miners may share one NAT IP, while one host may have several subnets.
    host_map = {i: str(nd["host_id"]) if nd.get("host_id") not in (None, "") else None
                for i, nd in enumerate(nodes)}
    memory_map = {i: str(nd["memory_domain_id"]) if nd.get("memory_domain_id") not in (None, "")
                  else host_map[i] or ("node", i) for i, nd in enumerate(nodes)}
    devices = {i: nd.get("gpu_uuids") or nd.get("gpu_uuid") or nd.get("device_id")
               for i, nd in enumerate(nodes)}
    exact_templates = any("allowed_spans" in node for node in nodes)
    allowed_spans, template_resources = None, None
    if exact_templates:
        from .topology import prepare_allowed_spans
        template_resources = {}
        for i, node in enumerate(nodes):
            template_resources[i] = node.get("resource_capacity") or {
                key: None if node.get(source) is None else int(float(node[source]) * 1024**2)
                for key, source in (("available_vram_bytes", "free_vram_mb"),
                    ("available_ram_bytes", "free_ram_mb"), ("pinnable_ram_bytes", "pinnable_ram_mb"))}
        allowed_spans = prepare_allowed_spans({i: node.get("allowed_spans", []) for i, node in enumerate(nodes)},
                                             int(m["n_layers"]), template_resources)
    layer_vram, kv = float(m["layer_vram_mb"]), float(m["kv_mb_per_layer"])
    cap_layers = int(m["cap_layers"])

    # 1) calibrate free VRAM per node: strip the per-box reserve AND the node's measured load-peak
    #    transient (the admit-then-OOM gate), then cap at the proven layer ceiling — density-scaled
    #    to the card size when the node announces one (a flat cap collapsed a 96 GB card to the
    #    32 GB verdict). The footprint is per-NODE too: a marlin card (~4.1 GB/layer) and a cutlass
    #    card (~2.3 GB) hold very different blocks, and select_ring already takes the per-node dict.
    lv = {i: float(nodes[i].get("layer_vram_mb") or layer_vram) for i in range(n)}
    per_layer = {i: lv[i] + kv for i in range(n)}

    def _node_cap(i):
        if exact_templates:
            measured = max((row["hi"] - row["lo"] for row in allowed_spans[i]), default=0)
            return min(measured, int(nodes[i]["cap_layers"])) if nodes[i].get("cap_layers") is not None else measured
        if nodes[i].get("cap_layers") is not None:
            vram_cap = int(nodes[i]["cap_layers"])
        else:
            total = float(nodes[i].get("total_vram_mb") or 0.0)
            vram_cap = density_cap_layers(cap_layers, total) if total > 0 else cap_layers
        # Dual-resource check: host RAM / pinnable RAM bottleneck
        host_ram_per_layer = float(m.get("layer_host_ram_mb") or 0.0)
        if host_ram_per_layer > 0.0:
            free_ram = nodes[i].get("free_ram_mb")
            pinnable_ram = nodes[i].get("pinnable_ram_mb")
            if m.get("placement") == "ram":
                if pinnable_ram is None or free_ram is None:
                    # Fail-closed: unmeasured pinnable memory cannot host pinned expert pool
                    return 0
                budgets = (float(free_ram), float(pinnable_ram))
                if not all(math.isfinite(v) and v >= 0 for v in budgets):
                    raise ValueError("RAM and pinned-memory budgets must be finite nonnegative MiB")
                avail_ram = min(budgets)
            else:
                avail_ram = pinnable_ram if pinnable_ram is not None else free_ram
            if avail_ram is not None:
                ram_cap = int(float(avail_ram) // host_ram_per_layer)
                return min(vram_cap, ram_cap)
        return vram_cap

    raw_free = {i: max(nodes[i]["free_vram_mb"] if exact_templates else nodes[i]["free_vram_mb"] - float(m["reserve_mb"])
                       - float(nodes[i].get("load_peak_extra_mb")
                               or m.get("load_peak_extra_mb") or 0.0), 0.0) for i in range(n)}
    node_caps = {i: _node_cap(i) for i in range(n)}
    free = (dict(raw_free) if joint_roles or exact_templates else
            {i: min(raw_free[i], node_caps[i] * per_layer[i]) for i in range(n)})
    cap_ok = [i for i in range(n) if node_caps[i] > 0 and (exact_templates or free[i] >= per_layer[i])]
    if isolation == "subnet":
        cap_ok = [i for i in cap_ok if nodes[i].get("subnet") not in (None, "")]
    elif isolation in ("host", "adjacent_host"):
        cap_ok = [i for i in cap_ok if host_map[i] is not None]
    if not cap_ok:
        return None                                          # no node can hold even one layer

    host_caps, tail_host_caps = {}, {}
    host_ram_per_layer = float(m.get("layer_host_ram_mb") or 0.0)
    if not exact_templates and m.get("placement") == "ram" and host_ram_per_layer > 0:
        budgets = {}
        for i in cap_ok:
            nd = nodes[i]
            ram, pinned = nd.get("free_ram_mb"), nd.get("pinnable_ram_mb")
            available = min(float(ram), float(pinned)) if ram is not None and pinned is not None else 0.0
            if not math.isfinite(available) or available < 0:
                raise ValueError("RAM and pinned-memory budgets must be finite nonnegative MiB")
            group = memory_map[i]
            budgets[group] = min(budgets.get(group, available), available)
        reserve = float(m.get("host_reserve_mb", 0.0))
        tail_reserve_host = float(m.get("tail_host_reserve_mb", 0.0))
        if not all(math.isfinite(v) and v >= 0 for v in (reserve, tail_reserve_host)):
            raise ValueError("host reserves must be finite nonnegative MiB")
        host_caps = {group: max(0, int((available - reserve) // host_ram_per_layer))
                     for group, available in budgets.items()}
        tail_host_caps = {group: max(0, int((available - reserve - tail_reserve_host) // host_ram_per_layer))
                          for group, available in budgets.items()}

    # 2) head = most central capable node (lowest total RTT to the rest); it runs the coordinator.
    #    Under privacy pinning the coordinator sees the raw prompt, so the head must be TRUSTED —
    #    rank centrality over trusted capable nodes only.
    pin = privacy is not None
    # STRICT bool — trust is the security boundary, so read it fail-CLOSED: only a genuine `True`
    # (JSON `true`) marks a node trusted or staked. A truthy string like "false"/"0" or an int must NOT sneak a
    # node into the trust set (a control plane that serialized the flag as a string would otherwise
    # fail OPEN — the one way a stranger could reach a boundary while the plan claims to be pinned).
    trusted = {i for i in range(n) if nodes[i].get("trusted") is True or nodes[i].get("staked") is True} if pin else None
    head_pool = [i for i in cap_ok if i in trusted] if pin else cap_ok
    if exact_templates:
        head_pool = [i for i in head_pool if any(row["head"] for row in allowed_spans[i])]
    if not head_pool:
        return None                                          # pinning on, but no trusted node can hold a block

    def centrality(i):
        # clamp each edge at the sentinel instead of OMITTING unreachable ones — omission summed a
        # fully-disconnected node to 0, which won min() and made it the mandatory head (undeployable)
        return sum(min(float(rtt[i][j]), _UNREACHABLE) for j in range(n) if j != i)

    def _connected_cap(i):
        # This is only a capacity upper bound, not the route solve. A legal
        # directed chain need not be a bidirectional star around its head.
        # The solver still checks every actual forward and return channel.
        reachable, pending = {i}, [i]
        while pending:
            source = pending.pop()
            for destination in cap_ok:
                if destination not in reachable and rtt[source][destination] < _UNREACHABLE:
                    reachable.add(destination)
                    pending.append(destination)
        capacity = {}
        for j in reachable:
            group = memory_map[j]
            count = node_caps[j] if exact_templates else min(node_caps[j], int(free[j] // per_layer[j]))
            capacity[group] = capacity.get(group, 0) + count
        return sum(min(count, host_caps.get(group, count)) for group, count in capacity.items())
    head_pool = [i for i in head_pool if _connected_cap(i) >= int(m["n_layers"])]
    if not head_pool:
        return None                              # no candidate head can REACH enough capacity to serve
    if head_choice is not None:
        if head_choice not in head_pool:
            return None
        head = head_choice
    else:
        head = min(head_pool, key=centrality)
    if not exact_templates:
        free[head] = max(free[head] - float(m["head_reserve_mb"]), 0.0)

    # 3) launch-bound per-layer time: base * the node's cpu_factor; the head pays a coordinator
    #    penalty. A node announcing a MEASURED layer_ms (the probe's graph-replayed decode number)
    #    is placed at that, not the modeled base — a box whose graph capture failed runs eager at
    #    ~4x and must be planned as what it measured, not what its GPU label suggests.
    layer_ms = {i: (float(nodes[i]["layer_ms"]) if nodes[i].get("layer_ms") is not None
                    else float(m["layer_ms_base"]) * float(nodes[i].get("cpu_factor", 1.0)))
                for i in range(n)}
    if any(not math.isfinite(value) or value < 0 for value in layer_ms.values()):
        raise ValueError("layer timings must be finite nonnegative milliseconds")
    if coordinator_on_head:
        layer_ms[head] *= float(m["head_layer_ms_mult"])
    if not exact_templates and m.get("placement") == "ram":
        for i in range(n):
            layer_ms[i] += ram_dma_overhead_ms(nodes[i], m)

    # 4) coordinator entry/return hops are measured relative to the chosen head.
    c_out = ([rtt[head][i] if i != head else 1.0 for i in range(n)] if coordinator_costs is None else coordinator_costs[0])
    c_in = ([rtt[i][head] if i != head else 1.0 for i in range(n)] if coordinator_costs is None else coordinator_costs[1])
    subnet = {i: nodes[i].get("subnet") for i in range(n)}

    # 5) upload-aware placement iff EVERY node announced an uplink (residential lever); else decode-only.
    ups = [nodes[i].get("up_mbps") for i in range(n)]
    aware = all(u is not None for u in ups)
    extra = {"isolation": isolation, "host_id": host_map, "device_id": devices,
             "host_layer_caps": host_caps, "host_memory_domain": memory_map,
             "tail_host_layer_caps": tail_host_caps}
    if exact_templates:
        extra.update(allowed_spans=allowed_spans, template_resources=template_resources, node_layer_caps=node_caps)
        if cost_model is not None:
            extra["template_cost_model"] = lambda order, alloc, rows: cost_model(order, alloc, layer_ms, c_out, c_in, rows)
    if joint_roles:
        extra.update(node_layer_caps=node_caps, tail_reserve_mb=float(m["tail_reserve_mb"]),
                     objective=objective, cost_model=(lambda order, alloc: cost_model(
                         order, alloc, layer_ms, c_out, c_in)) if cost_model else None)
        if hard_max_stages is not None:
            extra["hard_max_stages"] = hard_max_stages
    if aware:
        extra.update({"up_mbps": {i: float(ups[i]) for i in range(n)},
                 "prefill_bytes": float(m.get("prefill_bytes", 0.0)),
                 "decode_bytes": float(m.get("decode_bytes", 0.0)),
                 "decode_steps": int(m.get("decode_steps", 1)),
                 "prefill_chunks": int(m.get("prefill_chunks", 1))})

    if pin:
        extra["trusted"] = trusted
        extra["boundary_in"] = int(privacy.get("boundary_in", 0))
        extra["boundary_out"] = int(privacy.get("boundary_out", 0))

    if "max_stages" in m:
        extra["max_stages"] = int(m["max_stages"])

    tail_floor = int(m.get("tail_floor", 3 if int(m.get("n_layers", 0)) == 43 else 0))
    if tail_floor > 0:
        extra["tail_floor"] = tail_floor


    # 6) the TAIL stage also holds the final norm + lm_head (measured 1.15 GiB bf16 on
    #    M2.5 — a 13-layer tail OOM'd loading it on a 32 GB 5090, live 2026-07-09, while
    #    the same 13 layers warmed fine as a middle). The reserve applies to WHICHEVER
    #    node lands the tail, which select_ring decides — so plan, check the landed
    #    tail's block against the reserve, and if it doesn't fit, bake the reserve into
    #    that node's budget EXACTLY ONCE and re-plan (the tail may move). Convergence is
    #    checked against the ORIGINAL budget: the old loop compared against the already-
    #    docked value — re-demanding the reserve on top of itself — so a node that
    #    reappeared as tail was docked again each round (a feasible pool read as
    #    infeasible), and after 4 blind rounds the LAST spec was returned even when its
    #    tail never fit at all. The docked set is finite and only grows, so this
    #    converges in <= n rounds or honestly reports None.
    tail_reserve = float(m.get("tail_reserve_mb", 0.0))
    base_free = dict(free)                       # budgets to validate against (head reserve included)
    docked = set()                               # nodes whose budget already models the tail reserve
    spec = None
    for _ in range(n + 1):
        spec = select_ring(range(n), rtt, c_out, c_in, free_vram_mb=free, layer_ms=layer_ms,
                           subnet=subnet, n_layers=int(m["n_layers"]), layer_vram_mb=lv,
                           kv_mb_per_layer=kv, slack=min(n, _SLACK) if slack is None else int(slack),
                           require=head, **extra)
        if spec is None:
            return None
        tail_i = spec["order"][-1]
        lo, hi = spec["blocks"][tail_i]
        if exact_templates or joint_roles or tail_reserve == 0.0 or base_free[tail_i] >= (hi - lo) * per_layer[tail_i] + tail_reserve:
            break                                # the landed tail fits block + reserve in its budget
        if tail_i in docked:
            return None                          # reserve already modeled and it STILL can't fit
        docked.add(tail_i)
        free[tail_i] = max(base_free[tail_i] - tail_reserve, 0.0)
    else:
        return None                              # no tail placement converged: the reserve fits nowhere
    assert spec["order"][0] == head, "select_ring must return a head-first (deployable) order"
    # a deployable ring never traverses an unreachable (sentinel) edge: the forward hops and the
    # tail -> head coordinator return must all be measured, finite paths — if feasibility forced
    # one in, there IS no usable ring, so say so instead of shipping a dead hop
    _o = spec["order"]
    if (any(rtt[a][b] >= _UNREACHABLE for a, b in zip(_o, _o[1:]))
            or c_out[_o[0]] >= _UNREACHABLE or c_in[_o[-1]] >= _UNREACHABLE):
        return None
    # belt-and-braces: every stage's block must fit the node's ORIGINAL budget (the tail
    # including its reserve) — a violation here is a planner bug, never a deployable answer
    for i in spec["order"]:
        lo, hi = spec["blocks"][i]
        if exact_templates:
            if spec["calibrations"][i]["gpu_bytes"] > template_resources[i]["available_vram_bytes"]:
                raise RuntimeError("calibrated stage exceeds actual available GPU bytes")
            continue
        need = (hi - lo) * per_layer[i] + (tail_reserve if i == spec["order"][-1] else 0.0)
        if need > base_free[i] + 1e-6:
            raise RuntimeError(f"planned block [{lo}:{hi}) needs {need:.0f} MB on node {ids[i]!r} "
                               f"whose budget is {base_free[i]:.0f} MB")

    boundary = set(spec.get("boundary", []))
    order = [ids[i] for i in spec["order"]]
    last = len(spec["order"]) - 1
    stages = []
    for k, i in enumerate(spec["order"]):
        lo, hi = spec["blocks"][i]
        st = {"id": ids[i], "index": k, "lo": lo, "hi": hi,
              "head": k == 0, "tail": k == last, "layers": hi - lo}
        if exact_templates:
            row = spec["calibrations"][i]
            st.update(runtime_config_sha256=row["runtime_config_sha256"],
                      calibrated_resources={key: row[key] for key in ("gpu_bytes", "host_bytes", "pinned_bytes")})
        if m.get("placement") == "ram":
            st["layer_ms"] = layer_ms[i]
            st["latency_source"] = ("measured_layer" if nodes[i].get("layer_ms") is not None else
                "measured_dma" if nodes[i].get("dma_exposed_ms_per_layer") is not None else "estimated_dma")
            st["placement"] = "ram"
            st["expert_cache_slots"] = int(m.get("default_expert_cache_slots", 32))
            st["host_pinned_mb"] = (hi - lo) * float(m.get("layer_host_ram_mb", 0.0))
        if pin:
            st["boundary"] = i in boundary
        stages.append(st)
    out = {
        "order": order,
        "head": ids[head],
        "stages": stages,
        "dropped": [ids[i] for i in spec["dropped"]],
        "step_ms": spec["step_ms"],
        "tok_s_per_g": spec["tok_s_per_g"],
        "k": spec["k"],
        "isolation": isolation,
    }
    if aware:
        out["request_ms"] = spec.get("request_ms")
        out["prefill_ms"] = spec.get("prefill_ms")
        out["roles"] = {ids[int(i)]: r for i, r in spec.get("roles", {}).items()}
    if exact_templates:
        out["calibration_search"] = spec["calibration_search"]
    if pin:
        out["privacy"] = {"boundary_in": extra["boundary_in"], "boundary_out": extra["boundary_out"],
                          "boundary_stages": [ids[i] for i in spec["order"] if i in boundary]}

    # P2-2: Automatic in-region coordinator placement selection
    tail_stage_idx = spec["order"][-1]
    best_c_id = None
    min_c_rtt = float("inf")
    for idx, node in enumerate(nodes):
        c_rtt = rtt[idx][head] + rtt[tail_stage_idx][idx]
        if c_rtt < min_c_rtt:
            min_c_rtt = c_rtt
            best_c_id = ids[idx]
    out["coordinator_placement"] = {
        "preferred_host": ids[head],  # Runtime and cost model pin coordinator on the actual head.
        "min_roundtrip_ms": round(float(rtt[tail_stage_idx][head]), 2),
        "in_region": float(rtt[tail_stage_idx][head]) < 35.0,
        "advisory_alternative_host": best_c_id,
        "advisory_alternative_ms": round(min_c_rtt, 2),
    }

    # Per-node impairment and bottleneck reporting (pricing & hardware suitability tiering)
    impairments = []
    ideal_layers = math.ceil(int(m["n_layers"]) / len(stages)) if stages else 0
    for st in stages:
        idx = next(i for i in range(n) if ids[i] == st["id"])
        nd = nodes[idx]
        total_vram = float(nd.get("total_vram_mb") or 0.0)
        v_cap = int(nd.get("cap_layers")) if nd.get("cap_layers") is not None else (
            density_cap_layers(cap_layers, total_vram) if total_vram > 0 else cap_layers
        )
        host_ram_per_layer = float(m.get("layer_host_ram_mb") or 0.0)
        r_cap = v_cap
        if m.get("placement") == "ram" and host_ram_per_layer > 0.0:
            p_ram = nd.get("pinnable_ram_mb")
            r_cap = int(float(p_ram) // host_ram_per_layer) if p_ram is not None else 0

        binding = "none"
        reason = "adequate capacity"
        if r_cap < v_cap and st["layers"] <= r_cap:
            binding = "pinned_ram"
            reason = f"pinnable RAM ({nd.get('pinnable_ram_mb', 0):.0f} MB) capped layers to {st['layers']} (VRAM permitted {v_cap})"
        elif v_cap < r_cap and st["layers"] >= v_cap:
            binding = "vram"
            reason = f"GPU VRAM capped layers to {st['layers']}"

        penalty_ms = 0.0
        if binding == "pinned_ram" and st["layers"] < ideal_layers:
            penalty_ms = (ideal_layers - st["layers"]) * float(m.get("layer_ms_base", 0.75))

        impairments.append({
            "node_id": st["id"],
            "stage_index": st["index"],
            "allocated_layers": st["layers"],
            "binding_constraint": binding,
            "vram_cap": v_cap,
            "ram_cap": r_cap,
            "step_penalty_ms": round(penalty_ms, 2),
            "reason": reason,
        })
    out["impairment_report"] = impairments
    return out


def plan_ring(nodes, rtt=None, model=None, *, slack=None, privacy=None, isolation=None,
              locality=None, objective="serial", workload=None, measurements=None, now=None,
              diagnostics=None, coordinator_id=None, route_ids=None):
    """Backward-compatible entrypoint with locality tiers and prediction-only costs.

    A legacy metadata-free call retains its historical successful solve. New
    locality/sparse/trace calls jointly evaluate head roles within each tier;
    expansion happens only if every local candidate failed feasibility/SLO.
    """
    from .locality import link_snapshot, candidate_tiers, shortlist_candidates
    from .planning_cost import estimate, workload_spec
    if diagnostics is not None:
        if not isinstance(diagnostics, dict):
            raise ValueError("diagnostics must be a dictionary")
        diagnostics.clear()
    if objective not in ("serial", "pipeline"):
        raise ValueError("objective must be serial or pipeline")
    if objective == "pipeline" and workload is None:
        raise ValueError("pipeline planning requires an explicit workload/acceptance assumption")
    resolved = profile_for(model) if isinstance(model, str) else model
    if isinstance(resolved, dict) and ("schema" in resolved or
            resolved.get("calibration_status") in {"structural", "unmeasured"}):
        raise ValueError("structural/resource evidence is not a calibrated scalar placement profile")
    cohorts = {node.get("cohort_id") for node in nodes if node.get("cohort_id") is not None}
    if len(cohorts) > 1:
        raise ValueError("planner candidates must belong to one exact model cohort")
    if nodes and isinstance(resolved, dict) and resolved.get("cohort_id") is not None and cohorts != {resolved["cohort_id"]}:
        raise ValueError("model profile and candidate cohort differ")
    if not nodes:
        return None
    all_nodes = nodes
    work = workload_spec(workload)
    require_exact = (resolved or {}).get("require_exact_calibrations", False)
    if type(require_exact) is not bool:
        raise ValueError("require_exact_calibrations must be a boolean")
    floor = (resolved or {}).get("gpu_weight_storage_floor")
    if floor is not None:
        if (not isinstance(floor, dict) or set(floor) != {"version", "layer_bytes", "head_bytes", "tail_bytes"}
                or type(floor["version"]) is not int or floor["version"] != 1 or not isinstance(floor["layer_bytes"], list)
                or len(floor["layer_bytes"]) != int((resolved or {})["n_layers"])
                or any(type(v) is not int or v < 0 for v in [*floor["layer_bytes"], floor["head_bytes"], floor["tail_bytes"]])):
            raise ValueError("invalid native GPU weight storage floor")
    if require_exact or any("allowed_spans" in node for node in nodes):
        eligible = []
        for i, original in enumerate(nodes):
            node = dict(original)
            spans = list(node.get("allowed_spans", []))
            if floor is not None:
                spans = [row for row in spans if row["gpu_bytes"] >= sum(floor["layer_bytes"][row["lo"]:row["hi"]])
                         + (floor["head_bytes"] if row["head"] else 0) + (floor["tail_bytes"] if row["tail"] else 0)]
            if work["frame_tokens"] > 1:
                trace = node.get("stage_trace", {})
                spans = [row for row in spans if row.get("runtime_config_sha256") == node.get("runtime_config_sha256")
                         and (row.get("lo"), row.get("hi")) == (trace.get("layer_start"), trace.get("layer_end"))]
            if spans:
                node["allowed_spans"] = spans
                eligible.append((i, node))
        if not eligible:
            if diagnostics is not None:
                diagnostics.update(reason="no fresh fitting exact stage calibrations are available")
            return None
        if rtt is not None:
            rtt = [[rtt[i][j] for j, _ in eligible] for i, _ in eligible]
        nodes = [node for _, node in eligible]
    legacy_dense = (not require_exact and not any("allowed_spans" in node for node in nodes) and
                    coordinator_id is None and rtt is not None and measurements is None and locality is None and workload is None
                    and objective == "serial" and not any(node.get("region") or node.get("region_id")
                    or node.get("stage_trace") for node in nodes))
    if legacy_dense:
        # Existing callers already supplied a full mesh (including very wide
        # K3 pools). Preserve their golden path rather than silently truncating it.
        selected = list(range(len(nodes)))
        frontier = {"total_nodes": len(nodes), "examined_nodes": len(nodes), "max_candidates": len(nodes),
                    "truncated": False, "method": "legacy caller-supplied dense mesh"}
    else:
        selected, frontier = shortlist_candidates(nodes, resolved or M25_PROFILE, locality, measurements, now=now)
    nodes = [nodes[i] for i in selected]
    if rtt is not None and not legacy_dense:
        rtt = [[rtt[i][j] for j in selected] for i in selected]
    snapshot = link_snapshot(nodes, rtt, measurements, now=now, route_ids=route_ids)
    # A fresh measured frame already includes expert transfers. Its per-layer
    # average provides the heterogeneous allocation seed; the full stage trace
    # is still re-scored and labeled when reused for another layer range.
    from .planning_cost import stage_observation
    nodes = [dict(node) for node in nodes]
    for node in nodes:
        if node.get("stage_trace") is not None:
            observed = stage_observation(node, 1, 1.0, now=snapshot["now"], workload=work)
            if observed["source"] != "scalar_estimate":
                node["layer_ms"] = observed["service_ms"]
    matrix = snapshot["rtt"]
    def hop(source, destination):
        if source == destination:
            return 0.0
        edge = snapshot["edges"].get((source, destination), {})
        return edge.get("hop_ms", edge.get("rtt_ms", _UNREACHABLE))
    tiers = candidate_tiers(nodes, snapshot, locality)
    by_id = {node["id"]: i for i, node in enumerate(nodes)}
    attempts = []
    heads_truncated = False
    use_cost = objective == "pipeline" or workload is not None or any(node.get("stage_trace") is not None for node in nodes)
    if isinstance(locality, dict) and locality.get("min_predicted_tok_s") is not None:
        from .locality import number
        number(locality["min_predicted_tok_s"], "min_predicted_tok_s", minimum=1e-9)
        use_cost = True
    if use_cost:
        workload_spec(workload)
    for tier in tiers:
        best, rank = None, math.inf
        best_used_joint = False
        distinct_pools = {tuple(dict.fromkeys(pool)): None for pool in tier["pools"]}
        for pool in distinct_pools:
            indices = [by_id[nid] for nid in dict.fromkeys(pool)]
            subset = [nodes[i] for i in indices]
            mesh = [[matrix[i][j] for j in indices] for i in indices]
            if not subset:
                continue
            legacy = tier.get("legacy", False) and not use_cost and measurements is None and coordinator_id is None
            if legacy:
                found = _plan_ring_core(subset, mesh, resolved, slack=slack, privacy=privacy, isolation=isolation)
                if found is not None:
                    best = found
                    break
            # Bounded candidate funnel is still inside select_ring. Evaluate
            # all capable head roles; a small central head must not kill a pool.
            policy = locality if isinstance(locality, dict) else {}
            max_heads = policy.get("max_head_candidates", 12)
            if type(max_heads) is not int or not 1 <= max_heads <= 32:
                raise ValueError("max_head_candidates must be an integer in [1,32]")
            budget_profile = resolved or M25_PROFILE
            def head_capable(i):
                node = subset[i]
                if "allowed_spans" in node:
                    if not any(row.get("head") is True for row in node["allowed_spans"]):
                        return False
                    if privacy is not None and not (node.get("trusted") is True or node.get("staked") is True):
                        return False
                    effective_isolation = isolation or budget_profile.get("isolation", "none")
                    return not (effective_isolation in ("host", "adjacent_host") and not node.get("host_id") or
                                effective_isolation == "subnet" and not node.get("subnet"))
                per = float(node.get("layer_vram_mb") or budget_profile.get("layer_vram_mb", 1)) + float(budget_profile.get("kv_mb_per_layer", 0))
                need = per + float(budget_profile.get("reserve_mb", 0)) + float(budget_profile.get("head_reserve_mb", 0)) + float(node.get("load_peak_extra_mb") or budget_profile.get("load_peak_extra_mb") or 0)
                if float(node["free_vram_mb"]) < need or node.get("cap_layers") == 0:
                    return False
                host_per = float(budget_profile.get("layer_host_ram_mb", 0))
                if budget_profile.get("placement") == "ram" and host_per:
                    ram, pin = node.get("free_ram_mb"), node.get("pinnable_ram_mb")
                    if ram is None or pin is None or min(float(ram), float(pin)) < host_per + float(budget_profile.get("host_reserve_mb", 0)):
                        return False
                if privacy is not None and not (node.get("trusted") is True or node.get("staked") is True):
                    return False
                effective_isolation = isolation or budget_profile.get("isolation", "none")
                if effective_isolation in ("host", "adjacent_host") and not node.get("host_id"):
                    return False
                if effective_isolation == "subnet" and not node.get("subnet"):
                    return False
                return True
            eligible_heads = [i for i in range(len(subset)) if head_capable(i)]
            heads_truncated = heads_truncated or len(eligible_heads) > max_heads
            heads = sorted(eligible_heads, key=lambda i: (
                sum(min(float(value), _UNREACHABLE) for value in mesh[i]), -float(subset[i]["free_vram_mb"])))[:max_heads]
            hard_max = policy.get("max_stages", (resolved or {}).get("max_stages", 6))
            for head in heads:
                context = {}
                def score(order, allocation, timing, outgoing, incoming, templates=None):
                    context.update(timing=timing, outgoing=outgoing, incoming=incoming)
                    scored_nodes = dict(enumerate(subset))
                    if templates is not None:
                        scored_nodes = {i: dict(node) for i, node in enumerate(subset)}
                        for i, row in templates.items():
                            node = scored_nodes[i]
                            node["runtime_config_sha256"] = row["runtime_config_sha256"]
                            trace = node.get("stage_trace")
                            if trace and trace.get("runtime_config_sha256") != row["runtime_config_sha256"] and work["frame_tokens"] == 1:
                                node.pop("stage_trace")
                    return estimate(order, allocation, timing, mesh, outgoing, incoming,
                                    scored_nodes, resolved or M25_PROFILE, workload,
                                    edges=snapshot["edges"], now=snapshot["now"], objective=objective)
                found = _plan_ring_core(subset, mesh, resolved, slack=slack, privacy=privacy,
                    isolation=isolation, head_choice=head, joint_roles=True, objective=objective,
                    cost_model=score if use_cost else None, hard_max_stages=hard_max,
                    coordinator_on_head=coordinator_id is None or subset[head]["id"] == coordinator_id,
                    coordinator_costs=([hop(coordinator_id, node["id"]) for node in subset],
                                       [hop(node["id"], coordinator_id) for node in subset]) if coordinator_id is not None else None)
                if found is None:
                    continue
                prediction = None
                if use_cost:
                    order = [next(i for i, node in enumerate(subset) if node["id"] == nid) for nid in found["order"]]
                    allocation = {next(i for i, node in enumerate(subset) if node["id"] == stage["id"]): stage["layers"]
                                  for stage in found["stages"]}
                    templates = None
                    if "calibration_search" in found:
                        templates = {next(i for i, node in enumerate(subset) if node["id"] == stage["id"]):
                            next(row for row in next(node for node in subset if node["id"] == stage["id"])["allowed_spans"]
                                 if row["runtime_config_sha256"] == stage["runtime_config_sha256"] and (row["lo"], row["hi"]) == (stage["lo"], stage["hi"]))
                            for stage in found["stages"]}
                    prediction = score(order, allocation, context["timing"], context["outgoing"], context["incoming"], templates)
                candidate_rank = prediction["predicted_request_ms"] if prediction else found.get("request_ms", found["step_ms"])
                if policy.get("min_predicted_tok_s") is not None and (
                        not prediction or prediction["predicted_committed_tok_s"] < float(policy["min_predicted_tok_s"])):
                    continue
                if candidate_rank < rank:
                    best, rank = found, candidate_rank
                    best_used_joint = True
                    if prediction:
                        best["planning"] = prediction
        attempts.append({"tier": tier["name"], "candidate_pools": len(tier["pools"]), "feasible": best is not None})
        if best is None:
            continue
        prediction = best.setdefault("planning", {"prediction_only": True, "objective": "serial",
            "predicted_serial_step_ms": best["step_ms"], "uncertainty": ["scalar legacy calibration"],
            "scope": "planning estimate; live hardware acceptance is separate"})
        if "bottleneck" not in prediction:
            actual_profile = resolved or M25_PROFILE
            estimates = []
            for stage in best["stages"]:
                node = next(node for node in nodes if node["id"] == stage["id"])
                timing = node.get("layer_ms")
                timing = (float(timing) if timing is not None else
                          float(actual_profile.get("layer_ms_base", 1)) * float(node.get("cpu_factor", 1)))
                estimates.append({"kind": "stage_estimate", "node_id": node["id"],
                                  "service_ms": stage["layers"] * timing})
            prediction["bottleneck"] = max(estimates, key=lambda item: item["service_ms"])
        prediction["uncertainty"] = sorted(set(prediction.get("uncertainty", []) + snapshot["uncertainty"]))
        if frontier["truncated"]:
            prediction["uncertainty"].append("open pool truncated to a bounded heuristic candidate frontier")
        prediction["locality"] = {"selected_tier": tier["name"], "expansion_reason": tier["reason"],
                                  "attempts": attempts, "regions": sorted({
            node.get("region") or node.get("region_id") for node in nodes
            if node["id"] in best["order"] and (node.get("region") or node.get("region_id"))})}
        prediction["routes"] = [{"src": a, "dst": b, **snapshot["edges"].get((a, b), {})}
                                for a, b in zip(best["order"], best["order"][1:])]
        actual_coordinator = coordinator_id if coordinator_id is not None else best["head"]
        prediction["coordinator_id"] = actual_coordinator
        for a, b in ((actual_coordinator, best["order"][0]), (best["order"][-1], actual_coordinator)):
            if a != b:
                prediction["routes"].append({"src": a, "dst": b, **snapshot["edges"].get((a, b), {})})
        if coordinator_id is not None:
            total_hop = hop(actual_coordinator, best["order"][0]) + hop(best["order"][-1], actual_coordinator)
            best["coordinator_placement"].update(preferred_host=coordinator_id,
                min_roundtrip_ms=round(total_hop, 2), in_region=None, low_latency=total_hop < 35,
                latency_scope="entry and return normalized route delays; not a geography assertion")
        if tier["name"] == "expanded" and (frontier["truncated"] or heads_truncated):
            prediction["locality"]["expansion_reason"] = "bounded_local_search_exhausted"
        prediction["search"] = {"head_roles": "joint" if best_used_joint else "legacy",
            "order": "bounded latency DP per tail; pipeline ranking is heuristic",
            "allocation": "serial and integer water-fill candidates" if objective == "pipeline" else "min-sum",
            **frontier}
        prediction["search"]["heads_truncated"] = heads_truncated
        if "calibration_search" in best:
            prediction["search"]["calibrated_templates"] = best["calibration_search"]
            prediction["search"]["allocation"] = "exact measured span templates"
            if best["calibration_search"]["truncated"]:
                prediction["uncertainty"].append("exact-template search exhausted its bounded state budget")
        if heads_truncated:
            prediction["uncertainty"].append("joint head search limited to a heuristic shortlist")
        best["dropped"] = [node["id"] for node in all_nodes if node["id"] not in best["order"]]
        best["model_id"] = (model if isinstance(model, str) else (resolved or {}).get("model_id") or
                            next((key for key, value in PROFILES.items() if value == resolved), None))
        if diagnostics is not None:
            diagnostics.update(status="planned", search=prediction["search"], attempts=attempts)
        return best
    if diagnostics is not None:
        diagnostics.update(status="bounded_search_exhausted" if frontier["truncated"] or heads_truncated else "no_feasible_plan",
            reason="no plan in examined candidates under resource/locality/stage limits; widen explicit bounds or refresh observations",
            search={**frontier, "heads_truncated": heads_truncated}, attempts=attempts)
    return None


def _main() -> int:
    """`python3 -m shard.plan` — JSON in ({nodes, rtt, model?, slack?}), JSON out (the plan, or null).
    `model` is a profile dict or a catalog model_id string (PROFILES)."""
    try:
        req = json.load(sys.stdin)
    except Exception as e:  # noqa: BLE001 — a malformed request is a caller error, report it as JSON
        json.dump({"error": f"bad request json: {e}"}, sys.stdout)
        return 2
    try:
        plan = plan_ring(req["nodes"], req.get("rtt"), req.get("model"), slack=req.get("slack"),
                         privacy=req.get("privacy"), isolation=req.get("isolation"),
                         locality=req.get("locality"), objective=req.get("objective", "serial"),
                         workload=req.get("workload"), measurements=req.get("measurements"), now=req.get("now"),
                         coordinator_id=req.get("coordinator_id"), route_ids=req.get("route_ids"))
    except KeyError as e:
        json.dump({"error": f"missing field: {e}"}, sys.stdout)
        return 2
    except Exception as e:  # noqa: BLE001
        json.dump({"error": f"plan failed: {e}"}, sys.stdout)
        return 1
    json.dump(plan, sys.stdout)          # `null` when the pool can't hold the model — a valid answer
    return 0


if __name__ == "__main__":
    sys.exit(_main())

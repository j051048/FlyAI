# GLM-5.2 on consumer Blackwell (RTX 5090)

> Historical research record — reviewed 2026-10-08 against `c2ab623`.
> Original dates, measured values, projections, later addenda and historical commands are preserved.
> “Active”, “next” and implementation gaps below describe the recorded experiment, not the current backlog.
> Current placement/serving contracts: [open network](../OPEN_INFERENCE_NETWORK.md),
> [GPT-OSS operations](../GPT_OSS_PRODUCTION.md), [V4 acceptance](../V4_BENCHMARK.md),
> and [documentation index](../DOCUMENTATION_INDEX.md).
> This is the recorded GLM research topology, not an advertised current generic model backend.
> The engine/source recipe, workload and warm/cold conditions must accompany its reported speed.

*Research record. Status: feasibility de-risked end to end; serving path identified
(quantized pipeline-parallel). 2026-06-17.*

## Why GLM-5.2

744B MoE (40B active), MLA + DeepSeek Sparse Attention (DSA), a **native MTP draft**, 1M
context, MIT license. Beats GPT-5.5 on long-horizon coding at ~1/6 the cost. Serving it
across scattered consumer 5090s would prove the engine generalizes beyond gpt-oss — a
frontier open model on the swarm.

It fits the WAN spec-decode thesis *better* than gpt-oss:
- **Native MTP head** = a free trained draft (no separate 20B model; the EAGLE endpoint,
  shipped in the weights).
- **MLA** = a tiny compressed KV latent (kv_lora_rank 512) → small activations on the wire
  and a cheap fast-verify static cache.
- **MoE sparsity** (40B active) = light per-node verify compute.

## Fast-verify de-risk (real RTX 5090, sm_120)

- `glm_moe_dsa` runs eager on Blackwell; the layer block has no kernel gaps.
- **The gpt-oss CUDA-graph fast-verify lever does NOT transfer.** GLM's verify is
  memory-bandwidth-bound (reading selected-expert weights), not launch-overhead-bound, so
  graphs give ~1.0×. Dense-attn bypass (the DSA indexer is a no-op at verify lengths
  ≪ index_topk) + a sparse/grouped MoE are bit-exact, but the graph doesn't help.
- **The real levers are quantization + fused kernels.** vLLM `fused_experts` at GLM dims:
  **2.16 ms bf16 / 1.10 ms fp8** per MoE layer (NVFP4 available) → ~55–100 ms full-model
  verify projection — comparable to gpt-oss-120B's 135 ms despite 6× the params.
  (`research/glm_probe_*.py`, `research/bench_fused_moe.py`.)

## Topology — the scattered-swarm wedge, validated on real WAN

`shard/topology.py`: minimum-latency Hamiltonian loop over the **measured asymmetric** RTT
mesh (exact Held-Karp ≤16 nodes, NN + 2-opt above) + best-k-of-pool node selection, fed by
`phase0/mesh.py`. Validated on 5 cheap scattered US nodes: latencies are genuinely
asymmetric, and **node selection dropped a 210–242 ms outlier → 114 ms vs 517 ms loop =
4.53×**. The scheduler pillar works on real internet, not just synthetic data.

## 2-box WAN pipeline — our PP engine

`research/glm_stage_node.py`: a real GLM-5.2 layer block per machine with hidden-state I/O
over our TCP transport. Layer 6 (Washington 5090) ↔ layer 7 (Florida 5090), warm round
~110 ms, output sane. Multi-machine pipeline integration works.

## The full-model correctness run, and the wall

Rented 16×5090 (512 GB), `Mapika/GLM-5.2-NVFP4` (410 GB), TP=16 under vLLM 0.23. **Every
*software* wall was surmountable, in sequence:**

1. DSA sparse-MLA has **no sm_120 kernel** (`FLASHMLA_SPARSE` is Hopper / datacenter-Blackwell
   only) → patched `is_v32=False` → dense MLA (`TRITON_MLA`). Valid because dense ≡ sparse at
   decode lengths ≪ `index_topk` (2048).
2. Indexer weights have no home in dense → patched `load_weights` to skip `"indexer"` weights.
   **Model fully loaded across all 16 GPUs.**
3. flashinfer MoE workspace OOM (1.54 GB, ~200 MB short) → capped `max_num_batched_tokens`.

**Then the *hardware* wall:** `No available memory for the cache blocks`. Rank 0 holds
~29 GB (embeddings + lm_head + its weight share) on a 32 GB card; after weights + MoE
workspace there is **zero room for KV cache**. High util → workspace OOMs; low util → no KV.
No setting fits — a hard per-GPU 32 GB overflow.

**Conclusion: GLM-5.2-NVFP4 does not fit on 16×32 GB RTX 5090 under vLLM tensor-parallel.**
The overflow is a TP artifact; the model's own configs all use ≥80 GB/GPU.

## The insight, and the path

The rank-0 overflow is a **tensor-parallel** artifact — TP piles embeddings + lm_head onto
rank 0. **Pipeline parallel — our engine — spreads embeddings (stage 0), lm_head (tail), and
layers evenly, so it sidesteps the wall.** Our PP engine already runs GLM-5.2 dense + correct
on a single 5090.

The remaining work: our PP stage runner currently **dequantizes fp8→bf16** (full model
~1.5 TB — far too heavy). The unlock is **quantized PP stage execution**: run NVFP4/fp8
weights directly — `fused_experts` for the MoE at quant, dense MLA in bf16 (the NVFP4
checkpoints keep MLA/indexer in BF16 anyway) — with no bf16 dequant. That drops per-stage
memory ~2–4× so the full model fits a feasible 5090 swarm, and PP avoids the TP rank-0
overflow entirely. (Separately, stock vLLM/SGLang sm_120 DSA support is maturing — community
`vllm-sm120` builds exist — which may give a turnkey path later.)

## Status / log

- **2026-06-17** — Full de-risk above. **Decision: build quantized pipeline-parallel stage
  execution** as the c0mpute-native serving path for GLM-5.2 on consumer Blackwell; table
  stock-vLLM-TP serving (doesn't fit per-GPU on 32 GB cards) and watch the sm_120 DSA
  ecosystem. gpt-oss-120B (18–25 tok/s over WAN) remains the shipped headline meanwhile.

## NVFP4 swarm driver — validated end to end over real WAN (2026-06-18)

`research/glm_swarm_nvfp4.py`: the NVFP4 port of the PP driver (coord + multi-layer stage +
`--next` chain + TCP transport). Handles dense layers 0–2 (1-expert FusedMoE) and MoE layers
3–77 (256-expert routed + 1-expert shared). Validated **coord (Washington 5090) → stage (Texas
5090) over the open internet**: hidden states cross machines, output flows, **4.95 tok/s** for a
2-layer stage. Per-node bootstrap is just `pip install -r phase0/requirements_vmoe.txt` (all stock
PyPI, ~1 min) + `phase0/node_fetch.py` (selective per-node layer download).

**Critical kernel finding — force `VLLM_CUTLASS`, not flashinfer.** vLLM defaults the NVFP4 MoE to
`FLASHINFER_CUTLASS`, which **JIT-compiles** `fused_moe_120` (cutlass) on first forward. On fresh
nodes that compile **OOM-kills** (35 parallel `cicc`) or **deadlocks** on stale ninja locks — it
only "works" on a box that compiled it once and cached the `.so`. Fix: set
`vllm_config.kernel_config.moe_backend = "cutlass"` → the **precompiled `VLLM_CUTLASS`** kernel in
the wheel runs with **zero JIT, zero OOM**, warms in seconds. (Also: never `pkill -f glm_swarm` in a
command that launches it — self-kills the SSH shell; reap GPU procs via `nvidia-smi … | xargs kill`.)
Parallel launcher: `phase0/launch_swarm.py` (provision → bootstrap → mesh RTT → `shard/topology`
ordering → layer assignment → chained stages → coord → tok/s).

## Hardware pivot: 20× scattered 5090 → ~5–6× scattered RTX PRO 6000 (2026-06-18)

A PP WAN swarm is **hop-limited**: per-token latency ≈ (entry + Σ forward hops + return) × RTT, and
hop count = ⌈model_size / usable_VRAM_per_node⌉. For GLM-5.2-NVFP4 (~410 GB):
- **32 GB RTX 5090** → ~4 NVFP4 layers/node → **~20 nodes → ~20 hops** → naive ~0.3–1 tok/s; even
  with KV cache + spec-decode the 20-hop floor caps it ~10–13 tok/s (code). Orchestrating 20
  simultaneously-online consumer GPUs is also fragile.
- **96 GB RTX PRO 6000 Blackwell** (sm_120 — *same arch as the 5090, our NVFP4/VLLM_CUTLASS code
  runs unchanged*) → ~16 NVFP4 layers/node → **~5 nodes → ~5 hops**.

**Decision (with leyten, 2026-06-18): pivot to ~5–6 scattered RTX PRO 6000 workstations.** Why:
1. **tok/s ∝ 1/hops** → ~4× fewer hops is a ~4× win *before* spec-decode, compounding with the g×
   from GLM-5.2's native MTP draft head. Projected: naive ~3 tok/s; +KV-cache +MTP-spec-decode
   **~15 tok/s chat / ~30–40 tok/s code over WAN.**
2. **Cheaper total**: ~5 × $0.97 ≈ **$5/hr** vs ~20 × $0.45 ≈ $9/hr.
3. **Zero code changes** — RTX PRO 6000 is Blackwell sm_120; the nvfp4 cutlass path is identical.
4. **More realistic node population**: a handful of prosumer ML workstations in different cities is
   far more plausible (and stable) than 20 coordinated consumer 5090s online at once.
5. **Still the scattered thesis** — non-colocated workstations across US cities over WAN, NOT a
   datacenter rack; topology + the shortest-loop optimization still earn their keep at ~5 hops.
   vast supply: 13+ distinct US RTX PRO 6000 WS at ~$0.94–1.2/hr.

**Next: make KV cache + MTP speculative decoding as good as possible** (target 30–40 tok/s code over
WAN). KV cache = MLA compressed latent per stage (kills the O(n²) recompute + shrinks the wire to one
token/step). Spec-decode = port `phase0/specpipe.py` + `phase0/tree.py` (already built for gpt-oss)
onto GLM-5.2's native MTP head — draft K tokens coord-side, verify all in ONE chain traversal (one
WAN round-trip commits g+1 tokens). Greedy acceptance → output token-identical to plain decode (free
correctness oracle for the proof). Direct tail→coord return (vs relay-back) halves the loop.

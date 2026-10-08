# Shard roadmap

## Current implementation status (2026-10-08, `c2ab623`)

The dated phases below retain the original experiment/design history; their week
estimates, spend decisions and historical tok/s values are not current launch
authorization or universal hardware guarantees.

| Area | Current status and remaining gate |
|---|---|
| Open registration/lifecycle | Signed exact cohorts, resource templates, GPU/RAM/pin leases and shared HTTP/SSE exist; reports are not hardware attestation and readiness requires signed warmup |
| Placement | Local/region or measured low-latency frontier before expansion; directed actual routes, explicit RTT semantics/external coordinator, exact spans/roles/index/width and bounded finite-window predictions |
| V4 memory mechanisms | Opt-in local RAM experts/cache/prefetch, layer KV working set and gated query/index-score chunks are wired; CPU fixtures do not establish CUDA parity, large pin pools or speed targets |
| GPT-OSS service | Verified downloads/native packing, approved templates, authenticated sessions, shared service and committed-token metrics; see [GPT_OSS_PRODUCTION.md](GPT_OSS_PRODUCTION.md) for real recipe acceptance |
| Numerical/privacy/economics | Backend-specific numerical gates remain; receipts do not prove computation/privacy/hardware; payment/decentralized governance is external integration/research |

The V4 **4x5090 >=40 / 6x5090 >=30 committed decode tok/s** targets remain pending
independent real-cluster evidence under [V4_BENCHMARK.md](V4_BENCHMARK.md). Historical
M2.5/GPT-OSS/GLM runs do not satisfy them. Future models need real adapters and
separate compatible recipes. Start with [ARCHITECTURE.md](ARCHITECTURE.md) and
[OPEN_INFERENCE_NETWORK.md](OPEN_INFERENCE_NETWORK.md) for current architecture.

## Original June betanet phases

**North star:** Shard is the engine for a permissionless worldwide compute fabric — *BitTorrent for
VRAM/compute* — where GPUs of any kind swarm to run many models, ever bigger, and eventually training and
general compute. **MiniMax-M2.5 sharded inference is the betanet: the first rung, the proof-of-concept, not
the destination.** The phases below harden that first rung into a permissionless network; the
model-genericity and general-compute arcs are in "Beyond the betanet" at the end.

Phased so the riskiest thing is proven first and cheaply. Each phase has one goal and a hard pass/fail.

## Phase 0 — prove the transport (target: 1-2 days)

The single hardest thing: reliably serve tokens through a multi-stage split. Do it on easy mode first.

- 2 nodes, **same LAN** (or two boxes in the same datacenter), low latency.
- Split one model that genuinely needs 2 GPUs across them with our own QUIC transport moving activations.
- No speculative decoding yet. Plain token-by-token is fine, it'll be slow, that's expected.
- **Pass = a coherent completion comes out, reliably, 20 times in a row, with a measured tok/s.**

**✓ PASSED (2026-06-15).** 2-node split with per-node KV-cache. Proven on Qwen2.5-14B (29.5GB bf16, OOMs one 24GB card) split 24/24 across two 4090s: 20/20 clean completions, ~17 tok/s decode, via our own TCP transport. Code in `phase0/` (`node_kv.py` = the split node, `bench.py` = the 20x reliability harness). Caveat: nodes are on a low-latency same-host link, not WAN yet.

**Hardened (2026-06-15):** the transport is now fail-fast and instrumented — the direct answer to the opaque-transport failure mode, a black-box edge that just returns "broken pipe". Per-edge socket timeouts (a dead OR frozen node raises a `TransportError` with context — which peer, which decode step, dropped vs timed-out — never an unbounded `recv`); `os._exit` on unrecoverable failure so the process dies in ms, not after a ~10s torch/cuda teardown; per-edge health logging (round-trip latency + uplink MB). Verified by killing (SIGKILL) and freezing (SIGSTOP) the tail mid-generation — head fails cleanly both ways, no hang. This pre-validates part of Phase 1's edge-supervision goal on the easy link.

This was the thing that blocked us all day. It's the momentum milestone.

## Phase 1 — make the transport survive WAN (target: 3-5 days)

- Same 2 nodes, now on **different networks behind NAT**.
- Add hole-punching via a rendezvous (c0mpute orchestrator), relay fallback for symmetric NAT.
- Add fp8/int8 activation quantization to cut uplink bandwidth.
- Add edge supervision: kill a node mid-stream, confirm the pipeline detects it and the request fails cleanly (not a hang).
- **Pass = reliable WAN serving + an honest WAN tok/s number** (will be latency-bound and slow, that's the point, it sets the baseline spec-decode has to beat).

**◑ PARTIAL (2026-06-15).** Reliable WAN serving demonstrated over a real transatlantic link: Norway head → North Carolina tail (~115ms RTT), 2-node Qwen2.5-3B split moving activations over our own TCP transport via a direct open port. Honest WAN tok/s: plain decode **6.0 tok/s** (latency-bound, ~133ms/token round-trip) → speculative decoding **20.5 tok/s on code (3.4×)**, 9.7 on prose (1.6×) — the multiplier cashing out over real internet. Edge supervision (timeouts/fail-fast) carried from Phase 0 hardening. Remaining: NAT hole-punching + relay fallback (a direct open port stands in for now), and fp8/int8 activation quantization.

## Phase 2 — speculative decoding (target: ~1 week)

The payoff. Add the draft-verify loop.

- Small draft model on the entry node, propose K, verify K across the swarm in one traversal.
- Adaptive K based on measured latency + live acceptance rate.
- **Pass = land in the paper's regime: meaningfully more tok/s than Phase 1 on the same links, in the ~8-9 tok/s ballpark at 80ms for a small target.** This is the proof the whole approach is real.

**✓ PASSED (2026-06-15), and now proven at 120B scale over WAN (2026-06-16).** The draft-verify loop landed first on the 2-node 14B split (up to 6× tokens/traversal co-located) and over a real transatlantic 3B link (**3.4× on code, 20.5 tok/s**) — `phase0/specdec.py`. It now runs on the full **gpt-oss-120b split across four nodes / two machines over WAN** (Sweden ↔ North Carolina) via `phase0/specpipe.py` — the same draft-verify, generalized to the N-stage gpt-oss pipeline, with a gpt-oss-20b draft on its own GPU at the entry node and exact greedy acceptance. Warm steady-state: plain **4.67 tok/s → spec-decode (K=4) ~6.5 tok/s, 1.4×**, output token-for-token identical to plain. The multiplier is honest-but-modest at this scale because gpt-oss ships no draft below 20b (~62 ms/token), so the draft eats part of the round-trip it saves — the gain scales with WAN latency and inversely with draft cost. Two findings: MXFP4 kernels JIT-compile (warm is the honest number), and **fixed K beats adaptive K** here (each K change recompiles kernels for a new shape). Remaining: a lighter tokenizer-matched draft for 100B+ targets, and fp8 activation quant.

**→ Opened an active research track: [WAN-optimal speculative decoding](research/wan-speculative-decoding.md).** Instrumenting the round (draft 309 ms + verify 224 ms) showed the bottleneck at 120B is the *draft cost*, not the WAN — the verify alone implies a ~15 tok/s ceiling with a free draft. The draft is stuck at ~62 ms/tok because gpt-oss's MXFP4-MoE kernels won't `sdpa`/`compile` and the flash sink kernel is Hopper-only. Path to ~20 tok/s on consumer GPUs over WAN: a cheap draft (optimized kernels, or an EAGLE-style trained head) **then** tree speculation. See the doc for the evidence and plan.

## Build target for Phases 3+ — MiniMax-M2.5 on the scattered-5090 swarm (the betanet)

The permissionless layer is built and proven on the **betanet base model, MiniMax-M2.5** (229B-A10B MoE), across scattered consumer 5090s — the hardware a real volunteer runs, and the cheapest to iterate on Vast. M2.5 is warm-validated over libp2p with signed receipts (tool-calling, multi-turn, long context). Earlier proofs still stand and bound the engine's range: **gpt-oss-120B at ~40 tok/s on 4× RTX 4090** (consumer-card target) and **GLM-5.2 744B at ~30 tok/s on 7 GPUs** (frontier-size ceiling). Once the permissionless stack works on the betanet, it carries to other models — the serve path is being generalized behind one `ModelRuntime` interface so the network runs *any* model, not one hand-ported architecture (see [MODEL_RUNTIME.md](MODEL_RUNTIME.md)).

**Foundation already proven (Phases 0-2).** Owned authenticated + encrypted transport, fail-fast edge supervision, pipelined speculative decoding, and the split itself. gpt-oss-120B (MXFP4, 36 layers) now runs at **~40 tok/s (peak ~42), greedy/exact, over WAN on 3 scattered RTX 4090s** + a coordinator (`phase0/specpipe.py` pipelined coordinator, `phase0/launch_oss.py`) — up from a latency-bound ~18 via: async-draft pipelining, a **3-stage 12-layer ring** (4 WAN hops not 5), and **placing the layer-less coordinator in-region** (cut the ring 174→102 ms). GLM-5.2 744B runs across 7 scattered GPUs at ~30 tok/s (the frontier-size proof). What's left is turning this *hand-deployed* swarm into one that strangers can join, get paid by, can't cheat, and can't leak. That is Phases 3-6.

## Phase 3 — live & earning in c0mpute (trusted swarm) (target: ~1 week)

Goal: the swarm serves real c0mpute users and earns, registered as a single worker. No trust assumptions removed yet — this proves the integration economics before the hard crypto, and gets Shard generating revenue immediately.

- **M1 — swarm-as-worker bridge.** The coordinator speaks c0mpute's existing Socket.io worker protocol (`worker:register` → `job:new` → `job:token` → `job:complete`; reference `c0mpute-worker/src/worker.ts`). The whole swarm registers as one worker advertising the model, aggregate tok/s, and `capabilities: {distributed, longContext}`. Earnings accrue to the operator's privy_id; the existing 1-in-15 canary probes run as real swarm jobs and pass naturally.
- **Streaming bridge.** Map the pipelined / spec-decode output (tokens commit in chunks) onto per-token `job:token` events.
- **Pricing lane.** A slow-but-frontier swarm prices differently than a fast single-GPU worker; add a metered lane (c0mpute's `API_PLAN.md` already flags flat per-tier charging as wrong for this).
- **Pass = a real prompt on c0mpute.ai is served end-to-end by the 120B swarm, metered, earning recorded, withdrawable.**

## Phase 4 — self-managing swarm (still semi-trusted) (target: ~1-2 weeks)

Goal: stop hand-deploying. The swarm fits itself to whatever GPUs are present and survives churn. This is the reliability floor a permissionless network stands on — a swarm needs N nodes up at once, so one flaky GPU can't be allowed to take it down.

- **M2 — live scheduler / control plane** (`shard/scheduler.py`, hosted on the c0mpute orchestrator, which holds no weights or user data). Given the joined GPUs and their VRAM, fit the model into contiguous layer blocks, order the pipeline by measured RTT, and bring it up. Handle join (re-fit + rebuild) and leave (reassign + rebuild). Replaces the hand-run `fleet.py`.
- **M3 — fault tolerance. ◑ DEMONSTRATED (2026-06-23).** Killed a node mid-generation under load → detected ~4s, the committed tokens preserved, a pre-warmed spare spliced into its slot (only the spare + the victim's predecessor relaunch; other survivors auto-re-handshake), prompt+committed re-prefilled, generation resumed to completion (189→256 tok, continuation byte-preserved). Engine primitive `coordinate_pipe(resume_ids, resumable)` + control-plane healer `phase0/heal.py`; [receipt](receipts/fault-tolerance-20260623.json). Failover ~131s today (cold-spare reload dominated) — a HOT standby + block-only re-prefill (vs full prompt+committed) make it a <few-sec blip; that's the remaining work.
- **Pass = swap one node for a different one with zero manual editing; kill a node mid-generation under load and the request still completes.** ✅ the kill-and-complete half is met (above).

## Phase 5 — permissionless (remove trust, one assumption at a time) (target: weeks; M7 open-ended)

Goal: a stranger runs one command, their GPU joins, takes a layer block, and earns for the tokens it helped produce — without anyone having to trust them.

- **M4 — per-node identity + auth.** Kill the single shared `SHARD_PSK`. Each node authenticates with its c0mpute `cwt_` token via a keyed handshake (Noise / QUIC-TLS); the orchestrator (rendezvous) vouches for who holds which block. *Pass:* two nodes that never shared a secret form an authenticated, encrypted edge from their tokens alone; no valid token, no join.
- **M5 — NAT traversal + one-command install.** Hole-punching via the orchestrator as rendezvous, relay fallback for symmetric NAT, plus a one-line installer (auto-installs deps, checks drivers, pulls *only* the assigned block's weights, registers, serves) mirroring the existing `@c0mpute/worker` UX. *Pass:* a GPU behind a home router with no port-forwarding joins with one command.
- **M6 — per-node payment.** The orchestrator records per-node earnings for the tokens each block helped produce, paid per-token USDC on the existing `worker_earnings` rails. *Pass:* two people's GPUs in one swarm each watch their own balance tick up on the same job.
- **M7 — trustless verification (the research frontier).** The output is the *joint* product of N nodes, so detect a node that forwards plausible garbage instead of actually running its block, and pin the blame. Prototype: random redundant recompute (spot-check a block on a trusted node), per-epoch edge challenges, activation commitments — all tied to **staking with slashing** (c0mpute's staking machinery already exists). *Pass:* a node returning subtly-wrong output is caught and slashed within N jobs, without recomputing every token on a trusted node. **This is genuine research and the least certain item on the roadmap; everything above it is engineering we have proven variants of.**
- **Pass (phase) = a stranger's GPU joins via one command, takes a block, serves verified tokens, and earns — and a cheater is caught and slashed.**

## Phase 6 — privacy + hardening (ongoing)

The privacy pillar earns its word here, phase by phase, never overclaimed earlier. The leak is real: a malicious node can reconstruct a fraction of a user's tokens from the activations it processes.

- **Boundary-layer pinning** — keep the leaky embedding + final layers on staked/trusted nodes; let untrusted volunteers hold only deep middle blocks, which leak least.
- **Per-request "trusted nodes only"** routing for sensitive jobs, tied to stake.
- **Security pass** on the rendezvous + transport.
- Full detail and the privacy threat model live in [ARCHITECTURE.md](ARCHITECTURE.md).

## DeepSeek-V4 memory execution: code status versus acceptance

The original eleven-step order remains useful, but implemented tooling, connected
runtime paths and hardware acceptance are different milestones:

| Step | Actual status at this audit |
|---|---|
| 1. Frozen baseline/evidence | Benchmark protocol and validators exist; CPU/code validation does not mean the GPU baseline or speed gate passed |
| 2. Runtime observations | Optional route/DMA/KV/allocator/profile fields are validated and signed in [receipt.py](../shard/receipt.py); signed observations are not independent hardware proof |
| 3. Resource contract | Explicit GPU/host/pinned bytes, source/configuration binding, [probes](../shard/probe.py) and V4 profiles exist; scalar defaults are estimates, exact requirements need real measurement |
| 4. Separate weight tiers | Opt-in RAM canonical expert banks and GPU resident attention/router/shared/norm/boundary weights are constructed without a transient full GPU expert bank |
| 5. Fixed GPU cache | Local slot/lease/epoch guards, host budgets and conversation heat are implemented; real CUDA concurrency/pressure acceptance remains required |
| 6. Demand DMA/graph seam | Misses copy from pinned local RAM and use existing GPU arithmetic; hybrid MoE stays outside capture with supported graph islands; production CPU fallback stays off |
| 7. Placement | GPU/RAM estimates and fresh exact executable templates, role/index/width constraints and shared-memory budgets are wired; predictions are not acceptance |
| 8. Four/six-card acceptance | **Pending hardware evidence**; tooling in `phase0/v4_acceptance.py` cannot turn simulated/CPU results into 40/30 tok/s proof |
| 9. Prefetch/prefill | Controlled known-hot prefetch is wired before attention; query/index-score chunks are gated in Stage. Full projections, Compressor recurrence and MoE keep original shapes. The generic chunk executor's existence is not proof of whole-prompt activation/expert-streaming bounds |
| 10. KV tiers | `v4_kv_runtime.py` connects a shared layer working set, pinned histories and rollback snapshots. Reference valid-prefix reads remain context-dependent and add PCIe traffic; no generic sliding-window approximation |
| 11. Expansion | CPU expert math/cross-node expert replicas are not enabled production paths; broader numerical/hardware recipes have separate gates |

Use [RUNTIME_METRICS.md](RUNTIME_METRICS.md),
[RESOURCE_CONTRACT.md](RESOURCE_CONTRACT.md) and
[V4_HYBRID_RUNTIME.md](V4_HYBRID_RUNTIME.md) for concrete controls/scopes. The
inherited V4 reference is not replaced by Strata's C++ engine. The separately
requested GPT-OSS integration does not waive V4's benchmark or numerical criteria.

## Beyond the betanet — where the fabric goes

The post-betanet arcs. Named honestly as direction, not as proven work.

- **One engine, every model.** Generalize the serve path behind a single `ModelRuntime` interface
  (`shard/node.py`): inherit the per-architecture forward pass + kernels from the ecosystem
  (vLLM/Transformers), keep the moat — ring, transport, spec-decode, verification — in-house. *Pass = a
  second model (a small dense one) onboards and serves end-to-end over the ring with no engine changes.*
  Decision + plan: [MODEL_RUNTIME.md](MODEL_RUNTIME.md). **In progress.**
- **Many, ever-bigger models.** A catalog of models, each sharded across swarms sized to its needs; the
  manifest already carries `arch`, the catalog already exists c0mpute-side ([INTEGRATION.md](INTEGRATION.md) §4).
- **General compute, training included.** The permissionless rails (identity, transport, content-addressed
  weights, verification, payment) are meant to carry compute beyond inference. Training over a scattered WAN
  swarm is a **separate execution core**, not an extension of the inference engine — the rails carry over, the
  compute core does not. Years out; named as the destination, not a claim.

## The honest risk register

- **Trustless verification (M7) is unsolved at scale.** Verifying that each node in a multi-party pipeline actually did its work — without recomputing everything — is the real research frontier. It could take months, not weeks. Everything before it is engineering we have proven variants of; this is the part that might not have a clean answer.
- **A swarm is fragile in a way single-GPU isn't.** It needs N nodes up simultaneously; one drop kills the pipeline until rebuild. Phase 4's scheduler + fault tolerance have to be excellent or the worker looks unreliable to the orchestrator.
- **WAN transport across arbitrary NAT is genuinely hard.** A funded team didn't nail it. We de-risk by owning + instrumenting the layer and proving it on LAN first.
- **Spec-decode acceptance rate over real links sets the real tok/s.** If the draft is weak or the domain is hard, fewer tokens accept and the number drops. Mitigate: good draft model, adaptive K, measure honestly.
- **Pricing.** A slow-but-frontier swarm earns differently than a fast single-GPU worker; flat per-tier charging (already flagged in c0mpute's `API_PLAN.md`) misprices it. A metered lane is needed before volume.
- **Privacy vs the pillar.** The leak is real. Boundary pinning helps, full privacy is research. Do not sell what isn't true yet.

## What we need

- The **4× RTX 4090 / gpt-oss-120B** swarm on Vast as the Phase 3-5 development rig (cheap to iterate, consumer-representative).
- The c0mpute orchestrator as rendezvous + scheduler host (already running).
- A small uncensored, tokenizer-matched draft model for the target (gpt-oss-20B today; a lighter draft is the open optimization).
- c0mpute `cwt_` worker tokens + the existing payment/staking rails (already live).

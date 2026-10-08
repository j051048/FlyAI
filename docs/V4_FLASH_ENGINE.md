# DeepSeek-V4-Flash engine — current entry points and historical results

The V4-Flash engine is separate from M2.5. Its serving code lives in
`engines/deepseek_v4/`, the vendored reference in `vendor/deepseek_v4_ref/`, and the
benchmark/acceptance tools remain in `phase0/`. Reference wrappers and optional kernels
preserve the numerical acceptance contract; an optimization flag alone is not proof of parity.

## Current state at `c2ab623` (2026-10-08)

Production formation uses signed node offers, exact model cohorts and calibration, node-local
fenced leases, and a signed `shard-pipeline-plan/1`. The application HELLO pins both stage and
coordinator keys and binds roles, spans, purpose, ring and session. A head-owned grant excludes
another live coordinator; job nonce/epoch fencing and receipt validation remain separate checks.
Strict deployment connects to the actual tail endpoint and refuses `ret_relay`. Manual old
experiments require explicit `--legacy-protocol`; they do not establish the managed identity contract.

Use the managed [network service and deployment guide](V4_CLUSTER_DEPLOY_GUIDE.md) for production,
the [HTTP/service contract](V4_GATEWAY.md) for quotas, cancellation, replay and multi-ring behavior,
and the [frozen benchmark](V4_BENCHMARK.md) for acceptance. Each ring executes requests serially;
multiple independent rings can execute concurrently. This is not continuous batching.

RAM expert placement, bounded local expert caches, layer KV residency and optional local history
prefetch are implemented behind explicit flags; see [hybrid runtime](V4_HYBRID_RUNTIME.md).
They change the capacity contract from the historical all-GPU recipe below. Their new 5090 hardware
acceptance remains pending: **four distinct GPUs ≥40 committed decode tok/s; six ≥30**. The latest
**985 passed, 3 skipped** result is a selected CPU/socket suite, not whole-repository CI or GPU proof.

Co-located GPUs are allowed by the default placement policy. GPU UUIDs must be distinct, shared
host RAM/pinned budgets must be charged once, and actual reachable endpoints and measured links
must be used. An IP/NAT address is not a host identity or a latency measurement. The strict WAN
benchmark's hardware/isolation requirements are an independent protocol choice.

## Historical hardware snapshot (2026-08-02, end of day)

The numbers and conclusions below describe the old all-resident recipe and measured ring at that
date. They are retained for comparison, not a new verified pass under today's frozen protocol.

**30.15 tok/s decode median**, 512 tokens, novel prompt, six distinct EU RTX 5090s. Three
consecutive warm runs within 0.17 of each other (30.18 / 30.29 / 30.12; run 1 cold at 19.31).
Bit-identical to greedy, receipts verified. **5.6x on the session** (5.35 -> 30.15).

Ring: `Poland[0:8) -> Czechia[8:16) -> Denmark[16:24) -> Estonia[24:32) -> Poland[32:40) ->
Poland[40:43)`. Recipe: stages `V4_MOE_GROUPED=1 V4_MOE_MULTI=1 V4_CUDA_GRAPH=whole
V4_MOE_IN_GRAPH=1 V4_DSPARK_MOE=1 V4_FP8_GEMV=1 V4_FP8_SHARED=1 V4_DSPARK_BLOCK=8`,
coordinator `V4_LAZY_DRAFT=1`, depth >= cap, refill floor 1. `V4_MAX_SEQ=8192`.

### Fill-family result on that historical ring

Throughput is **useful in-flight over round-trip latency**, where useful in-flight is
`inflight_time_avg / frames_per_token`. Every fill lever raises in-flight AND raises
frames_per_token by at least as much, so the useful figure pins near **4.6** whatever you pull:

| config | tavg / f_per_tok = useful | tok/s |
|---|---|---|
| block 5, floor 1 | 4.89 / 1.234 = 3.96 | 29.2 |
| block 6, floor 1 | 5.32 / 1.277 = 4.17 | 28.3 |
| **block 8, floor 1** | **7.27 / 1.568 = 4.64** | **30.5** |
| block 10, floor 1 | 9.50 / 2.189 = 4.34 | 27.1 |
| block 8, floor 8 | 8.96 / 2.389 = 3.75 | 14.1 |

The refill floor at block 8 fills the pipe to **8.96 of a cap of 9 (99.6%) and HALVES throughput**.
`topup_disagree` goes 0 -> 1496 as it fills: a topped-up block's deep drafts condition on its own
block prefix, not on the tokens already in flight, so the added frames answer a history the ring
does not take. **The gap to the cap ceiling is not empty pipe; it is space only wrong frames fit.**

Two measured corollaries for that recipe. The draft block has an **interior optimum at 8** (cap 9) — not the trained 5, not
10; only the endpoints had ever been tested. And **depth is clamped by the cap**: at block 8, depth
9/12/16 are byte-identical (same g, tavg, f/tok, stale); below it the pipe starves (depth 4 ->
tavg 2.72 -> 17 tok/s). The rule is `depth >= block + 1`, nothing more.

L is ~152 ms of which only ~50 ms is compute; the rest is wire between countries. With the boxes
fixed in that experiment, the next proposed lever was **acceptance**. `V4_DRAFT_TOP2` measures
the rescue rate that would justify building tree speculation (see docs/V4_TREE_VERDICT.md).

### Measurement discipline on shared boxes

Identical work (same g, same tokens, `same=True`) measured a **6x spread** — block 5 read median
6.59 against best 21.39. The old investigation used best-of-N to explore attainable speed; it is
not today's acceptance estimator, because it selects favorable runs. The frozen protocol uses
the complete fixed suite's warm median, with every workload/repetition and range retained.
Verify observed levers on the PYTHON process environ, never the
bash wrapper. Long mixed runs also drift: the same control prompt read 20.61 / 13.48 / 10.62 across
one 30-minute matrix run at constant g, most likely allocator fragmentation from interleaving big
and small prefills. A fresh ring holds 30.1 steady.

### Context costs throughput, because prefill headroom comes out of KV

Prefill crosses the ring as ONE frame. At `V4_MAX_SEQ=8192` a 2048-token prefill OOM'd s0 (8 layers
plus the embedding) with 108 MiB free; at 4096 it fits but 3072 OOMs. Dropping MAX_SEQ to buy that
headroom costs ~1.5x: the same control prompt reads **30.15 at 8192 and 20.61 at 4096**. The
context x workload matrix (docs/receipts/v4-flash-matrix-20260802.json) is therefore a RELATIVE
instrument; its absolute numbers understate the engine.

### Historical binding constraint, measured per frame

| stage | on_box | detail |
|---|---|---|
| s0-s4 | 8.3 - 10.0 ms | ~1.0 ms/layer, uniform across boxes |
| **s5 tail** | **14.64 ms** | fwd 3.19 + **draft 9.16** + head/logits ~2.3 |

**The DSpark draft is 63% of the bottleneck stage and therefore of the ring's ceiling
(68.3 frames/s).** At that point the drafter received NONE of the main-model MoE work: `grouped_forward` returns on its
first line when rows != 1 and `DSparkBlock` runs `dspark_block_size` rows by construction;
`v4_moe_decode` is `s==1`-gated; `v4_moe_in_graph` captures `Stage.layers` only. Only the dispatch
fix (`V4_MOE_MULTI`) ever reached it. Make a draft CHEAPER — not rarer, not more frequent — and the
tail falls to ~5.5 ms, the bottleneck moves to Estonia at 10.04, and the ceiling goes to ~100
frames/s. That was a projected ceiling, not a measured pass. Current DSpark has its own grouped
MoE path, and the RAM adapter owns main/MTP expert pools separately; remeasure the tail rather
than carrying this bottleneck diagnosis forward unchanged.

### THE REFILL FLOOR IS NEGATIVE ON THIS RING, and the reason matters

`V4_REFILL_FLOOR` (built, correctness-proven, default 1 = shipped behaviour) refills the pipe
before it drains to the frontier. It WORKS — in-flight 4.18 -> 5.07 at floor 3, -> 6.00 at floor 5.
It still loses, because it defeats lazy drafting:

| floor | in-flight | drafts issued | g | tail on_box | measured |
|---|---|---|---|---|---|
| 1 | 4.18 | 117 | 11.13 | 14.64 (draft 9.16) | **24.3** |
| 3 | 5.07 | 411 | 9.48 | 28.00 (draft 22.86) | ~19 |
| 5 | 6.00 | 510 | 8.68 | — | ~16 |

A +21% fill gain against a **-48% ceiling loss**. Two independent models predicted +11%..+45% and
both named the drafting bill as the risk; both under-priced it. **Fill is not closed — it is gated
on cheap drafting.** Revisit the floor the moment the draft is cheap; the lever is already built.

### Bimodality: GPU idle downclock, not the code

Identical runs read 2.9 / 7.1 / 24.2 tok/s. Perfect correlation with `clocks.sm` minimum: slow runs
sat at 24-300 MHz, fast runs at 2400. Containers cannot lock clocks (`nvidia-smi -lgc` needs host
privileges), so a keepalive doing a 512x512 bf16 matmul every 100 ms holds the P-state. **Measured
throughput-neutral** (24.27 with vs 24.16 without) and it removes the cold-start cliff. Always state
the warm protocol with any number.

### Lessons from the measured workload

- **n-gram / tap-free proposers**: measured acceptance **0.012 on novel code, 0.007 on prose**
  (0.96 edit-heavy). Those workload classes must remain separate; the historical result does not
  establish a merged V4 n-gram proposer or forbid evaluating a new proposer on declared workloads.
- **Wider rings**: 10 boxes measured SLOWER than 6. In-flight caps at `block_size + 1`.
- **Box churn**: rebuilds cost 30-40 min of weight downloads each. Keep one ring, park spares.

## The model

| | |
|---|---|
| Parameters | 284B total, 13B active |
| Layers | 43 |
| Hidden | 4096 |
| Experts | 256 routed (6 active) + 1 shared |
| Precision | FP4 experts, FP8 rest |
| Licence | MIT |

Two architectural facts drive every engineering decision:

**Hyper-connections (mHC).** The inter-stage payload is `h [b, s, hc_mult=4, dim]` — four residual
streams that persist across every layer and collapse only at `hc_head`. V4 therefore sends roughly
**4x the wire bytes** of a normal pipeline-parallel model, which makes transport a first-class cost.

**DSpark.** A semi-autoregressive drafter of 3 MTP `DSparkBlock`s tapping layers 40-42, with
`dspark_block_size=5`. Because it taps the last three layers it is **tail-local by construction**:
the whole drafter must sit on the stage that owns layers 40, 41 and 42. That constraint is load
bearing — a re-tile that splits 40-42 across boxes silently breaks drafting.

## Earlier historical hardware run (6-box Nordic RTX 5090 ring, distinct machines)

| arm | tok/s | notes |
|---|---|---|
| greedy | 1.84 - 1.90 | first token 5.6 - 7.3 s warm |
| DSpark serial | 1.05 | slower than greedy: one fat block frame per round |
| **DSpark pipelined** | **3.47 - 3.68** | **bit-identical to greedy**, 6/6 receipts valid |

Pipelined speculation is **1.94x greedy** and **3.5x serial**. Losslessness is not assumed: the
bench asserts `pipelined_equals_greedy` and it holds.

Receipts were verified **out-of-process**: 6 distinct signers, full 0-43 layer coverage, every
signature valid, and a deliberately tampered receipt is correctly rejected. Note that a receipt
must be passed through `wire_receipt()` before verification — the `stage` field is a debug label
applied *after* signing, and leaving it in puts it in the preimage and fails a valid receipt.

## Earlier investigation before the 30.15 tok/s recipe

This section records the earlier slow-kernel/all-resident investigation. Its ceilings and topology
constraints were specific to those measurements, not limits of the current hybrid runtime.

**0. First, two ceilings that were wrong — do not repeat them.**
A "26 tok/s ceiling / 11-15% efficiency" figure circulated all day. It was an **artefact**: it
divided a per-frame `on_box` of 579 ms by a timed window of `s = 15`. The true `s=1` bottleneck is
**147 ms**, so this ring's ceiling is **7.3 tok/s and we measure 39% of it**. Separately, any
"ceiling = 1/max-stage-time" number silently assumes a full pipe; see §3. Quote
`min(1/max_stage, (block+1)/round_latency)` or the number is fiction.

**1. Compute, not bubbles and not transport.** A validated discrete-event model (calibrated on
disjoint evidence, RMS 7.1%, predicting g / stale replies / in-flight depth across a whole depth
sweep, worst residual 15.2%) decomposes the loss as **9% fill, 54% misspeculation waste**. The
bottleneck stage is **91% busy** and `unsent_frames` is 0 — the pipe is full of *discarded* work,
not empty. RTT is 632 ms of which only **~28 ms is wire**, so the ring is **96% compute-bound** and
fp8 wire buys almost nothing on decode (it is a prefill/TTFT lever).

**Lever prices, measured, not assumed:** `d(tok/s)/dg ~ 0.27` at g=3.7 (about **9% per unit of g**,
and g caps near 5.9 with the measured decay) against **83% for a 2x cut in per-stage tau**, which
does not cap. **Compute is worth about nine times acceptance.** Reaching 20 tok/s needs draft block
>= 12 *plus* roughly **8-16x on tau**; every ring-shape lever combined ceilings out near 4.4 tok/s.

For scale: an 8-layer stage reads ~1.2 GB of fp4 weights per token, ~0.7 ms at the card's
bandwidth, against 147 ms measured — **~200x off roofline**. At batch 1 / s=1 that is small-kernel
launch latency, not bandwidth, which is exactly the regime CUDA graphs address.

**2. The all-resident ring was VRAM-bound (6 boxes).** Six 32 GB cards hold ~45 layers against the model's 43.
An exact max-stage-minimising planner (verified against brute force on 4000 pools, zero
mismatches) prices the best possible re-tiling at only **1.151x**; with caps lifted the same boxes
reach 1.61x better, a gap no tiling can close. Straggler **ejection is infeasible**, not merely
unhelpful: every 5-box subset holds 37 layers < 43.

A wider all-resident ring looked like the fix here — it lowers max-stage-time and adds VRAM headroom in one move
— and that reasoning is **wrong**; see §3. It was acted on, a 10-box ring was provisioned, and it
came out slower. Both this section and "ceiling = 1/max-stage" argue for width, and both omit the
fill cap. Size a ring by `min(fill cap, VRAM floor)`, which for block 5 is 6 either way.

**3. Ring WIDTH is capped by the draft block — width and block size are ONE knob.** In-flight
frames saturate at **block_size + 1 = 6**, proven by depth 6 / 8 / 12 returning byte-identical
results on a 10-stage ring (`max_inflight` 6, `stale_replies` 41, g 4.92 every time). A 10-box ring
therefore leaves **four stages permanently idle** and measured *slower* than 6 boxes (pipelined
decode 4.39 vs 5.35; greedy 1.64 vs 2.28 — pure added hop latency). 6 stages is simultaneously the
fill cap and the VRAM floor, which is why 8/8/8/8/8/3 is the right topology. To use more width you
must raise the draft block first.

**4. Acceptance: historical gather-order diagnosis.** g fell **4.0 -> 3.05** serial (3.37 pipelined).
Neither named lever was at fault — both are bit-exact in isolation. The mechanism is **gather
order**: `slim_indexer_forward` returns compressed slots *ascending* while the chunked verify path's
`_chunk_indexer` returns them *score-descending*. Same position, same selected SET, different ORDER
depending on whether it arrived in a verify chunk or an s=1 frame — which changes **30-35% of
attention output elements** by ~2 bf16 ULP at the shipped shape, so the verifier rejects its own
drafter. The investigation proposed `V4_TOPK_STABLE` to canonicalise the index (a pure permutation)
and recorded 32/32 positions equal where 0/32 did before. **That environment lever is not a
registered implementation in the current tree**; `v4_levers.py` explicitly reports it as an
unimplemented advertised flag. Do not set it expecting a production fix or treat the proposal's
parity observation as current whole-recipe proof.

Two traps in that investigation worth keeping: the effect is **0.0% at topk 24/128/192 and only
appears at 256+** (shipped is 640), so a tolerance test at toy width is structurally blind; and the
tie source is **not** `relu_()` (3 exact zeros in 719,400 candidates) but **bf16 pigeonhole** —
16.8% of candidates collide. Also note ring `accept_hist` shows keys a block-5 rule cannot produce,
so ring-g chains rounds and is not the same quantity as harness-g; compare g only within one
accounting scheme, depth, and generation length.

**5. Measure at realistic length.** At 48-64 tokens, prefill is 32-48% of wall clock, so end-to-end
tok/s understates the generation rate by ~1.7x — report `decode_tok_s` too. Acceptance is also far
better on long runs (g 10.3 at 320 tokens vs 4.0 at 48), because rounds chain across blocks.

## Historical lever observations and current flag caveats

Defaults are per implementation/launcher, not uniformly OFF: `V4_MOE_DECODE` is default ON and
the launch helper supplies its own graph default. Consult `v4_levers.py` and the actual process
configuration. The speeds below are historical observations, not current acceptance results.

| lever | env | status |
|---|---|---|
| Pipelined speculation | `V4_PIPELINED_SPEC=1` | **shipped, 1.94x, lossless** |
| Grouped fp4 MoE | `V4_MOE_GROUPED=1` | Historical 3.19x eager / 11x graphed claim; CPU parity cannot establish CUDA speed, use retained GPU evidence and fresh benchmark |
| MoE bank layout | (with grouped) | **FIXED** — was an allocation-ORDER bug, peak 31.11 -> 27.98 GiB |
| Whole-layer CUDA graphs | `V4_CUDA_GRAPH=1` | **~1.0x ALONE** — needs grouped MoE to be capturable |
| DSpark fast draft | `V4_DSPARK_FAST=1` | 4.51x draft, bit-exact; **exonerated** on the g regression |
| Ref-slim decode | `V4_REF_SLIM=1` | bit-exact in isolation; **exonerated** (see gather order) |
| Stable top-k proposal | `V4_TOPK_STABLE=1` | **Not implemented/registered in the current tree; historical proposal only** |
| fp8 wire | `V4_FP8_WIRE=1` | 1.977x on the s=1 frame, but the ring is 96% compute-bound |

**The bank layout was an ORDER bug, not a leak.** Each bank was allocated *before* the experts it
replaces were released, so peak held both; the freed blocks are 256 scattered 4 MiB allocations,
the wrong shape to satisfy a 1024 MiB request. Steady-state cost of the bank is **+0 bytes** —
the bank *is* the experts. Fix is release-first (`preserve=False`). Proven with storage finalizers,
not inferred.

**Two silent holes found the same way, both worth re-checking after any refactor:** `V4_FP8_WIRE`
never reached a stage process at all (`stage_launch_cmd` hand-built the env and omitted it — now an
`ENG_ENV` allowlist, as M2.5 already had), and the then-current grouped kernel did not fire on the
drafter. Current DSpark grouped/hybrid paths and eager RAM-budget registration address that old
gap; check requested/observed lever reports rather than assuming either universal coverage or
universal failure. A lever that silently does nothing looks like one that does not work.

## Ring ops that cost us real time

- **Screen boxes on CPU load, and screen them twice.** This workload is CPU-launch-bound, so a
  contended host loses ~30x throughput with the GPU sitting at 0%. Read the load average a few
  minutes apart: the first read includes the box's own provisioning burst, and the 5/15-minute
  figures separate a transient from a co-tenant.
- **Place by role, not by index.** The tail runs the drafter, `lm_head` and sampling every round
  and carries the most VRAM; the head gates every round. Put the two idlest boxes there.
- **Validate identity and resources rather than inferring them from labels.** Two historical offers with different `machine_id`, different `host_id`
  and different geolocation labels reported *identical* load triples — the same physical host
  wearing two badges. Identical load readings are diagnostic evidence, not authoritative identity.
  Same-IP GPUs are permitted today: use distinct GPU UUIDs, one shared RAM accounting domain,
  and tested private/loopback routes where available. A failing public-IP hairpin route is a
  route failure to fix or reject, not a blanket ban on that physical placement.
- **Never combine `pkill` and a launch in one ssh command** — the launch text in the same cmdline
  matches the pattern and kills what it just started. Use separate sessions.
- **Price tail graphs for the actual recipe.** One historical graph configuration cost ~18 GB
  on the MTP/head/embedding node. Later all-resident whole-graph recipes differ, while layer-KV
  residency and dynamic RAM expert placement impose explicit graph restrictions. Use calibrated
  peak bytes and the current runtime checks instead of a universal tail graph setting.
- **The cold-start JIT cascade is SERIALIZED across stages.** A stage compiles its tilelang kernels
  when the first frame reaches it, so on a cold ring the compiles happen one after another down the
  chain — measured on a 10-box ring at roughly 1-2 min per stage, ~12-15 min to first token, once.
  It looks exactly like a hung ring; check the stage logs for advancing per-stage timestamps before
  concluding anything is stuck. Worth pre-warming each stage with a dummy forward IN PARALLEL at
  boot, which would turn a serial 15 min into a parallel 2 min and get a wide ring measuring sooner.

## Layout

| file | role |
|---|---|
| `engines/deepseek_v4/v4_kernels_cpu.py` | CPU stand-ins for fp4/fp8/sparse-attn parity; no GPU speed evidence |
| `engines/deepseek_v4/v4_ref_cpu.py` | vendored reference and tiny golden oracle |
| `engines/deepseek_v4/v4_stage.py` | contiguous layer range, placement adapters and speculative rollback |
| `engines/deepseek_v4/v4_pipe.py` | strict/legacy ring entry points, greedy/DSpark/pipelined coordination and receipts |
| `engines/deepseek_v4/v4_dspark_draft.py` | tail-local DSpark drafter and verify planning |
| `shard/plan.py` | exact tiling planner using measured profiles and resource contracts |
| `engines/deepseek_v4/v4_network_service.py` | managed open-node formation and HTTP service |
| `shard/pipeline_session.py` | signed role/session handshake and remote coordinator ownership |
| `shard/leases.py` | node-local lease ledger and work fencing |
| `phase0/v4_benchmark.py` | frozen suite runner and strict report validation |

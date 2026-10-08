# V4 production runtime optimizations

This change composes with the existing V4 engine, speculation protocol, receipt chain and
local expert placement. CPU parity and control-flow tests verify the implementation. They
do not certify CUDA numerical gates, throughput or the four/six RTX 5090 benchmark targets.

## What runs in the service path

| Area | Implementation | Observation / fallback |
|---|---|---|
| Phase timing | Bounded host statistics and sampled CUDA events, collected per job | Signed `performance` includes count, total, min/max, rolling P50/P95 and dropped samples |
| Placement timing | Whole-layer measurements take precedence over transfer estimates | RAM stages name `latency_source`; H2D bandwidth is decimal **GB/s** |
| Local expert prefetch | `Stage._run` submits a bounded prediction before each layer's attention; MTP blocks use their own instance hook | Signed `prefetch_policy`; demand DMA remains authoritative |
| DSpark grouped FP4 | Wider token/expert pair lists use existing kernels in chunks of at most 32 pairs | Block 8/top-6 uses two chunks per matrix kind; actual calls/declines are observable |
| Cache routing | Reusable pinned IDs, a dedicated readback stream and reusable per-consumer slot-index buffers | One necessary event wait remains; no device-wide synchronization is added |
| MTP shared FP8 | Each drafter's resident shared expert gets the existing w1/w3 layout and instance binding before loading | Existing numerical self-gates and reference fallback remain active |
| Wire codec | Opt-in compilation of the existing pack/unpack expressions, explicitly pre-warmed before stage readiness | Byte checks, fresh output ownership and CPU-decoder compatibility; unsupported keys use the original codec |

The normal miss sequence remains demand-copy submission -> shared expert -> readiness wait ->
routed compute. The independent readback/shared overlap experiment is an explicit instance API
`configure_router_overlap(True)` and defaults off; it can otherwise consume the shared compute
window before demand DMA begins. Compare it separately on hardware before adopting it.

The DSpark pair path applies to GPU-resident routed banks. RAM MTP dispatch keeps its leased
cache path; its **resident shared expert** benefits from the FP8 change. Wide pairs are not a
claim that a RAM drafter now executes the resident-bank grouped kernel.

## Configuration

```bash
# Per-job diagnostics; GPU intervals sample every 16 calls per named phase.
export V4_PROFILE_RUNTIME=1
export V4_PROFILE_GPU_EVERY=16

# Local expert predictions on RAM placement. Enable independently for A/B runs.
export V4_EXPERT_PLACEMENT=ram
export V4_EXPERT_PREFETCH=1
export V4_EXPERT_PREFETCH_SLOTS=2
export V4_EXPERT_PREFETCH_WARMUP=2

# Existing shared/shape-tuned FP8 operators, now also installed on MTP shared instances.
export V4_FP8_GEMV=1
export V4_FP8_SHARED=1

# Optional compilation of the existing codec, independently of FP8 wire selection.
export V4_WIRE_FUSED=1
```

All new environment settings are forwarded by the single/multi-GPU launch builders. New
profiling, prediction and compiled-wire switches default off. Fixed-slot scratch reuse and
DSpark pair chunking apply when their existing runtime paths run. `V4_FP8_WIRE` retains its
existing default: fused code does not silently enable lossy wire quantization.
Flat-file deployments must ship `shard/runtime_profile.py` alongside the existing
`runtime_metrics.py`, plus the engine's new `v4_wire_codec.py` and prefetch module. A repository
checkout resolves the shared profiling module from `shard` automatically.

GPU-resident deployments using `V4_DSPARK_MOE=1` may keep their existing block-width recipe;
the old `pairs > 32` decline has been removed. Other declines (unsupported routing/backend,
missing banks and incompatible shapes) remain visible. Enabling resident-bank flags on a RAM
cache does not change the cache's dispatch owner.

Compiled wire kernels require a compiler that exposes the requested exact-numerics controls.
Older or incompatible compiler versions explicitly decline. Compilation and parity probes
run only at startup for the single-token shape; new prefill/batch shapes use the original
implementation until explicitly warmed. There is no per-token compilation or numerical probe.
GPU unpack is used only for a key proved byte-identical to the original CPU receiver, including
exceptional values. A rejected unpack key keeps decoding on CPU even if GPU packing passed.

## Reading the evidence

`V4_PROFILE_RUNTIME=1` also enables runtime metrics. With receipts enabled, the new fields are
included in the existing signed runtime envelope. GPU samples are resolved at the job barrier;
collection does not call device-wide synchronize in the token path. Host phase time and GPU
interval time overlap and **must not be added** to estimate total latency. Use `stage.recv.host`
to inspect queue/network wait, and the remaining stage phases for dispatch, packing, logits,
drafting and sending. Layer/gate/routed/shared/readback intervals identify the local GPU work.

Percentiles cover a rolling bounded sample, not every historic interval. Count, total and
min/max cover all measured intervals; GPU measurement is sampled at the declared period.
Cold compilation/capture can appear in the first job's intervals, so compare repeated warm
jobs with a fixed prompt and allocation. `V4_TIMING` remains available for its older,
synchronizing diagnostic; leave it off when measuring overlap with the new profiler.

`kernel_coverage.scope = job_python_observed` counts forwards/declines that Python actually
observed since reset. CUDA graph replay does not rerun Python counters. These are not total
GPU launch counts or proof that every replay used a fast kernel. CPU emulation cannot report
CUDA calls. The existing graph/lever state and CUDA traces provide the complementary evidence.

`prefetch_policy` reports candidate count, requested predictions, used, wasted and skipped.
Predictions are derived from each layer's historical routes; the next layer's routing decision
is not known before its hidden input exists. Cached working sets and active leases are protected,
and skipped optional predictions do not fail a request. Actual CUDA/DMA failures remain errors.
Cache contents and long-term heat survive job reset; per-job observations and pending predictions
reset. Speculative rollback replay does not issue new predictions.

The planner accepts `layer_ms`, `dma_exposed_ms_per_layer`, `expert_misses_per_layer` and
`dma_overlap_fraction` as per-node observations/calibration. A measured `layer_ms` already includes
expert transfer and is not charged again. Without a whole-layer measurement, measured exposed
wait wins; otherwise the explicit miss/overlap fields or conservative profile defaults estimate
transfer cost. Calibration must describe the same workload and allocation being planned.

## Hardware acceptance

Run each new switch independently, then the composed configuration, on the existing V4 ring.
Cover cold/warm jobs, cache hit/miss pressure, hash routes, block 8 and wider blocks, compression
boundaries, rollback, repeated jobs and long prompts. Retain raw receipts and verify tokens,
logits/state where required, cache lifetime safety, allocator peaks and actual kernel coverage.
Report committed tokens / decode wall time, acceptance/waste, per-stage tail latency and exposed
DMA wait. The existing four-card >=40 tok/s and six-card >=30 tok/s thresholds remain the gates.

The original six-optimization change kept dynamic RAM cache management outside a whole-stage
graph and left the generic token chunk executor disconnected. The subsequent integration in
[V4_NEXT_PHASE.md](V4_NEXT_PHASE.md) adds layer-local KV storage and numerically gated query/index-score
chunks while preserving full projection/Compressor/MoE shapes. It does not enable tokenwise prefill
or a dynamic-cache whole-stage graph. The new layer-KV path explicitly rejects incompatible graph
and fast-verify modes; existing island graphs remain usable.

The original local PyTorch 2.14.1 CPU check reproduced a pre-existing offline selftest
failure on clean HEAD: reference/ring/spec/DSpark/pipeline emitted `[388] * 6`, failing
the nontrivial fingerprint gate. The next-phase fix explicitly depth-scales only the
synthetic selftest model's residual-output initialization; real checkpoints and the
token diversity/parity gates are unchanged. The full-stack CPU regression now passes
27 tests. NOQAT remains approximate and outside the recipe: independent fixed-input
logit byte checks detect its arithmetic change even when a short token stream matches.
Random drafter hits are checked against actual accepted-depth accounting rather than
assumed to be zero. Hardware acceptance remains deferred to a real RTX 5090 cluster.

The original six-optimization focused regression was **821 passed, 23 skipped**, covering cache leases and reuse, local
prediction, stage/graph composition, DSpark/FP8 paths, codec control flow, real socket receipts,
resource accounting, planning and signed observations. Skips include unavailable CUDA hardware
and environment-specific conditions. Python compilation and `git diff --check` also passed.
See [V4_NEXT_PHASE.md](V4_NEXT_PHASE.md) for subsequent integration and verification.

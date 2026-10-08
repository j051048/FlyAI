# Optional signed runtime metrics

Code audit: **2026-10-08, `c2ab623`**. This document covers V4 execution observations.
Shared committed-token/service metrics are a separate `shard-pipeline-metrics/2`
contract described in [GPT_OSS_PRODUCTION.md](GPT_OSS_PRODUCTION.md).

V4 stages can collect per-job observations with `V4_RUNTIME_METRICS=1` or the
stage CLI's `--runtime-metrics` switch. The launch helper propagates the
environment switch. Enable receipts as well to retain the observations in the
signed job evidence.

`ReceiptSigner.finalize(runtime_metrics=...)` validates and copies the snapshot
before signing it. `verify_receipt()` also validates a present snapshot. Omitting
metrics preserves the original receipt fields and signature construction.
The schema is `shard-runtime-metrics/1`; observations are signed node statements,
not an independent proof that a cache lookup or transfer occurred.

## Executed work

`work` separates `main` and `draft` work. Each has `prefill`, `decode` and `replay`
counters. `totals` is their checked sum, not a separately maintained estimate.

| Counter | Unit and meaning |
|---|---|
| `routed_entries` | Token/expert route entries, including rejected speculative work and rollback replay |
| `resident_hits` | Entries whose expert weights are already resident on the GPU |
| `dma_misses` | Entries resolved through a local host-to-GPU transfer |
| `dma_bytes` | Actual bytes transferred for those misses |
| `dma_wait_ms` | Measured waiting time for those transfers |
| `cpu_misses` | Entries resolved through a CPU expert fallback |
| `reference_routes` | CPU reference execution; not a GPU cache hit or CPU fallback |

Every phase must satisfy:

```text
routed_entries = resident_hits + dma_misses + cpu_misses + reference_routes
resident_hit_rate = totals.resident_hits / totals.routed_entries
```

The rate is `null` when no routes have executed. Repeated expert IDs are route
entries, not distinct weight transfers. Committed output token throughput uses a
different denominator; these work counters include unsuccessful speculation.

The default resident V4 mode reports `gpu_resident` with zero DMA/CPU misses.
Opt-in canonical RAM experts and GPU slots report `gpu_expert_cache` with actual
local demand hit/miss and transfer observations. CPU parity fixtures use
`reference_cpu` and cannot qualify a GPU benchmark. Production CPU fallback remains
disabled. The schema validates observations; mechanisms live in the engine/cache.

Main-layer counters are recorded outside CUDA graphs from executed batch/token
geometry, so capture/warmup does not inflate counts. Hybrid dispatch reports actual
cache outcomes; DSpark uses its executed gate/block paths, including graph-safe
accounting and skipped FFNs. Reset starts new job counters;
rollback preserves prior work and records any main-model replay separately.
Resident geometric accounting does not read router tensors back to the CPU.
Hybrid cache dispatch has its own necessary router-ID readback; it must not be
described as a zero-host-synchronization path. CUDA transfer/profile events are
resolved at the job barrier rather than synchronizing every token for telemetry.

Optional signed fields include `expert_cache`, demand-independent `expert_prefetch`,
`prefetch_policy`, sampled `performance`, Python-observed `kernel_coverage`,
`kv_policy` and `prefill_policy`. Prefetch policy records requested/used/wasted/
skipped/candidate counts; used + wasted cannot exceed requested predictions.
Unmeasured work is not manufactured from a flag being enabled. KV/query policies
declare exact scope/budgets and executed transfers/chunks; query chunks only bound
query/index-score scratch, preserving full projection/Compressor/MoE shapes.

## Memory residency

`kv` reports current and observed peak `gpu_bytes`/`host_bytes`. These are backing
storage allocations, including reserved KV capacity, indexer/compressor state,
fast-verify scratch and retained rollback KV snapshots. They are not the number
of logically occupied context tokens. Views and aliases of one allocation are
counted once; input activations and token IDs are not counted as KV.

Optional `gpu_memory` reports PyTorch allocated/reserved bytes and their measured
peaks with `scope: process`. These are not per-stage ownership figures. A job's
peak interval is available only when a single monitored Stage owns that GPU in
the process. A second observer invalidates the interval, so collecting a receipt
cannot reset another Stage's peaks or misattribute shared allocator usage.
An unindexed `cuda` device is resolved to the actual current GPU.

Unavailable allocator observations are omitted. Unknown resource capacities in
the placement contract remain unknown rather than being encoded as zero.

See [V4_BENCHMARK.md](V4_BENCHMARK.md) for committed-token throughput and evidence
checks, and [RESOURCE_CONTRACT.md](RESOURCE_CONTRACT.md) for placement calibration.

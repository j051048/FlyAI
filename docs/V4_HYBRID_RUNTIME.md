# V4 local RAM expert pool and GPU cache

The hybrid placement experiment separates routed expert storage from the resident model working
set. Attention, routers, shared experts, normalization/HC parameters and the stage's boundary
embedding/head stay on the stage device. The complete routed expert pool is stored on the local
host, in its original weight and scale format; a bounded cache contains GPU copies. This change
belongs to the node runtime. Expert cache state is not transported between ring stages.

The experiment is opt-in. The default all-resident path remains the comparison and rollback
configuration. A small or invalid budget must fail before executing a model position rather than quietly
running expert math on the CPU. Normal hybrid execution uses GPU expert kernels after local H2D
transfer. It does not enable cross-node expert lookup, a second model or production CPU fallback.
The separately integrated layer-local KV and attention-query paging are described in
[V4_NEXT_PHASE.md](V4_NEXT_PHASE.md); enabling RAM experts alone does not enable them.

## Operator switches

RAM placement uses these explicit `V4_*` settings, which must reach every owning stage process:

| Setting | Meaning |
|---|---|
| `V4_EXPERT_PLACEMENT=gpu` | Default original all-resident execution |
| `V4_EXPERT_PLACEMENT=ram` | Canonical pinned host experts and local GPU slot caches |
| `V4_EXPERT_CACHE_MIB` | One stage's aggregate cache budget, in MiB, including main and MTP pools |
| `V4_EXPERT_CACHE_SLOTS` | Alternative fixed slots per pool; choose slots or byte budget |
| `V4_EXPERT_CACHE_RESERVE_MIB` | Additional free-GPU reserve for workspace/graph/runtime growth |
| `V4_RUNTIME_METRICS=1` | Enable signed observations of the selected runtime mode |

The Stage Python API exposes the same controls with byte units:
`expert_placement`, `expert_cache_slots`, `expert_cache_bytes` and `expert_cache_reserve_bytes`.
`expert_cache_reference=True` is an explicit **CPU test constructor seam**, never an operator CLI
fallback. Production RAM placement requires CUDA and successfully pinned host allocations.

`V4_MOE_IN_GRAPH=1` and `V4_DSPARK_MOE=1` conflict with this cache's eager dispatch boundary and
are refused in RAM mode. Keep both zero. The original grouped FP4 shader can still execute from
the hybrid adapter for eligible one-token input. Whole-layer graph execution uses the attention/HC
pre/post split with eager cache lookup and expert computation between captures.

For example, an operator can freeze an **experimental 8192 MiB cache** with the following protocol
override object. This is an allocation example, not a calibrated four-card performance recipe:

```json
{
  "V4_EXPERT_PLACEMENT": "ram",
  "V4_EXPERT_CACHE_MIB": "8192",
  "V4_EXPERT_CACHE_SLOTS": "0",
  "V4_EXPERT_CACHE_RESERVE_MIB": "2048",
  "V4_MOE_IN_GRAPH": "0",
  "V4_DSPARK_MOE": "0",
  "V4_RUNTIME_METRICS": "1"
}
```

Supply this same JSON to `v4_benchmark.py inventory --env-file` and `prepare --env-file`, launch
the stages with those settings, and declare the matching environment digest in each hardware
manifest entry. Existing processes do not adopt changed coordinator environment variables.
Measure the resident, KV, workspace, graph and loading peaks before choosing a real cache budget.

## Correctness contract

Routed weights must never be constructed as a full transient CUDA expert bank and then offloaded.
Construct the block without material expert storage, materialize resident tensors on the requested
device, and materialize canonical routed tensors in the host pool before allocating the bounded
cache. Original checkpoint parameter names and expert weight/scale aliases remain valid.

Each cache slot has a stable address for its weight and scale tensors. A host expert has a stable
logical ID, independent of the slot containing its current GPU copy. Eviction must wait for every
consumer lease on that slot, including work queued on another CUDA stream. DMA completion alone
does not establish that a previous expert's computation has finished.

The arithmetic follows the original logical expert order and token/expert grouping. Hash-routed
duplicate IDs preserve the reference's indexed-update semantics, including the final occurrence of
a repeated expert at a token. Replacing that operation with a general scatter-add or sorting by
physical cache slot changes the numerical contract.

Strict checkpoint loading writes the canonical weights. A subsequent checkpoint reload invalidates
every cached copy; a resident hit cannot return the previous checkpoint's bytes. Cache allocations
are runtime state and must not become additional serialized checkpoint parameters.

## CUDA graphs and transfer boundaries

Routing determines the expert IDs after the attention dependency completes. A real cache miss
therefore triggers local pinned-host-to-GPU transfer at that point. The optional
`V4_EXPERT_PREFETCH=1` path now predicts bounded experts from local routing history
before attention. It is not knowledge of the next layer's future routing decision; wrong or late
predictions fall back to authoritative demand DMA. No expert is fetched from another node.

The cache cannot change pointer addresses captured by a graph. Attention and HC pre/post graphs
can surround an eager hybrid MoE boundary; dynamic host-cache operations stay outside capture.
Whole-layer MoE capture requires a separate proven routing/slot-address contract. A graph request
that cannot satisfy that contract must be explicitly rejected or resolved to the documented split
graph mode, with the selected mode observable in the runtime audit.

## Telemetry and resource planning

Signed runtime telemetry uses the same denominator for expert routes, resident hits and misses,
including speculative/replayed work. GPU cache hits must be counted from actual successful cache
lookups. DMA bytes/wait describe copies actually issued. A CPU test executor is reference execution,
not a GPU hit and not a production CPU fallback. Main-stack and MTP draft work remain separate.

The placement requirements remain the resource contract in [RESOURCE_CONTRACT.md](RESOURCE_CONTRACT.md):
canonical expert RAM, lockable memory, fixed GPU cache, resident parameters, KV, workspace, graph
pools and allocation peaks each need an explicit budget. On-disk safetensors sizes are not GPU
runtime-peak measurements. Tail budgets must also include its MTP expert pools and resident modules.

## Acceptance

Run the focused CPU suite before the hardware tests:

```powershell
python -m pytest tests/test_v4_hybrid.py -q
python -m pytest tests/test_v4_hybrid_gpu.py -q -m gpu
```

The CPU tests check real tiny V4 modules, checkpoint loading, parameter aliases, fixed-slot
eviction and state transitions using an explicitly CPU test cache. They do not establish CUDA
DMA behavior or throughput. GPU tests skip with a stated reason without CUDA or the necessary
original kernel dependencies. A skipped CUDA test is an unverified hardware condition.

The hardware correctness bar is `torch.equal` for original packed weights/scales and for compared
hidden states, logits and valid KV/compressor state, across prefill, decode, rejection/replay and a
new job. The original TileLang grouped-kernel comparisons must actually use that kernel, rather
than replacing it with a CPU quantizer or a mathematical approximation.

Use the frozen workload and signed-receipt procedure in [V4_BENCHMARK.md](V4_BENCHMARK.md) for the
real DeepSeek-V4-Flash deployment. Four distinct RTX 5090 GPUs must achieve at least 40 committed decode
tok/s, and six must achieve at least 30, under that fixed protocol. No toy-model parity test,
cache-unit test or successful allocation establishes these performance targets. Keep the original
2026-08-02 all-resident report as historical context. A verified before/after comparison requires
new all-resident and hybrid runs using the same complete frozen protocol; the old 30.15 figure is
not a current hybrid pass.

## Implementation status at `c2ab623` (2026-10-08)

“Implemented” here means code exists and the stated CPU/control-flow evidence is available.
It never means the new four/six-card CUDA correctness, throughput or soak campaign passed.
The previous list marked every phase Completed; that incorrectly combined software milestones,
CPU primitives and unrun hardware acceptance. The actual dependency/status distinction is:

| Step | Current status | Remaining evidence or boundary |
|---|---|---|
| 1. Freeze benchmark and acceptance | Implemented | Four >=40 / six >=30 complete-suite committed-decode medians remain unmeasured on the new recipe |
| 2. Signed route/KV/runtime observations | Integrated, CPU schema/signature tests | Actual DMA/cache/CUDA intervals require real GPU runs; CPU reference reports cannot qualify |
| 3. GPU + RAM + pinned resource contracts | Implemented | Header sizes do not prove runtime peaks; exact workload/config calibration is required |
| 4. Two weight pools at construction/load | Integrated opt-in RAM path, tiny CPU parity | Real packed FP4 pinned/H2D correctness tests must execute rather than skip |
| 5. Fixed GPU slots and consumer leases | Implemented, CPU lifecycle tests | GPU stream/event safety and measured cache pressure need hardware evidence |
| 6. Demand DMA and graph seams | Integrated, reference comparison/skip-capable GPU tests | Whole-MoE capture remains incompatible with dynamic cache; no silent CPU fallback |
| 7. Dual-resource planning | Implemented with measured budgets and role-aware costs | Shared-host reservations come from durable leases, not an estimate or probe |
| 8. Four/six GPU acceptance | **Pending** | Run actual speed, correctness, cold/warm, eviction, rollback and soak campaigns |
| 9. Prefetch and query chunking | Integrated opt-in history prefetch and attention/Indexer query chunks | Full projection/Compressor/HC/MoE shapes remain intact; generic tokenwise chunk primitives are not the serving path |
| 10. Bounded layer KV | Integrated opt-in `v4_kv_runtime.py` path | Reads the complete valid compressed prefix through bounded shared workspace; supports only calibrated context/budgets and rejects incompatible graph/fast-verify modes |
| 11. Post-speedline expansion | **Deferred** | Offline `v4_expansion.py` primitives do not establish production CPU expert fallback or network expert replicas |

No second-model migration or cross-node expert replica deployment is claimed before the speedline
stands. The core cache remains local. New identity/service contracts compose with this path:
strict `shard-pipeline-plan/1`, pinned stage/controller keys, signed owner grants and node leases.
Use [V4_CLUSTER_DEPLOY_GUIDE.md](V4_CLUSTER_DEPLOY_GUIDE.md) for actual stage/HTTP startup.
Cache and kernel flags must be part of each stage's measured runtime contract; changing only the
coordinator environment does not change a running stage or refresh its calibration.

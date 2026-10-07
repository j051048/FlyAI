# V4 local RAM expert pool and GPU cache

The hybrid placement experiment separates routed expert storage from the resident model working
set. Attention, routers, shared experts, normalization/HC parameters and the stage's boundary
embedding/head stay on the stage device. The complete routed expert pool is stored on the local
host, in its original weight and scale format; a bounded cache contains GPU copies. This change
belongs to the node runtime. Expert cache state is not transported between ring stages.

The experiment is opt-in. The default all-resident path remains the comparison and rollback
configuration. A small or invalid budget must fail before executing a model position rather than quietly
running expert math on the CPU. Normal hybrid execution uses GPU expert kernels after local H2D
transfer. It does not enable cross-node expert lookup, a second model, KV paging or CPU fallback.

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
therefore triggers local pinned-host-to-GPU transfer at that point. Earlier expert prediction and
attention-overlapped prefetch are separate future optimizations.

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
real DeepSeek-V4-Flash deployment. Four RTX 5090 nodes must achieve at least 40 committed decode
tok/s, and six must achieve at least 30, under that fixed protocol. No toy-model parity test,
cache-unit test or successful allocation establishes these performance targets. Keep the original
all-resident six-node report alongside the hybrid report and compare identical frozen recipes.

## Implementation Order & Phased Milestones

To manage engineering risk and strict dependency order, V4 hybrid placement follows an 11-step sequence.
Verifiable local expert caching is completed and validated before prefetch and KV paging are introduced.

1. **Fix Correctness Baselines & Performance Acceptance Gates**
   - Solidify the benchmark harness ([V4_BENCHMARK.md](V4_BENCHMARK.md)): fix model checkpoint, quantization formats, kernel switches, prompt sets, context lengths, generation lengths, and warm/cold run policies.
   - Record committed output tokens, per-stage wall times, speculative acceptance rates, and verified hardware topology.
   - Target thresholds preserved: 4×5090 short-context ≥ 40 tok/s, 6×5090 ≥ 30 tok/s. Throughput accounts only for committed tokens. Bit-level parity, speculative rollback, and receipt validation remain strictly enforced.

2. **Instrument Runtime Telemetry & Signed Receipts**
   - Ensure bottlenecks are measurable before altering execution paths ([RUNTIME_METRICS.md](RUNTIME_METRICS.md)).
   - Track total expert route events, cache hits, DMA transfer counts / bytes / wait latencies, CPU fallback count, and GPU / pinned RAM KV footprints (current & peak).
   - Wire telemetry collection points into [shard/receipt.py](shard/receipt.py) and V4 execution loops; sign receipts with all telemetry fields. Expert hit rate uses all route operations as denominator and distinguishes prefill, decode, and speculative replay.

3. **Establish GPU + RAM Dual-Resource Placement Contract**
   - Extend `ModelRuntime` capabilities ([RESOURCE_CONTRACT.md](RESOURCE_CONTRACT.md)) to declare resident weights, expert pool, KV cache, GPU expert cache slots, CUDA graph pools, workspace, and peak loading budgets.
   - Extend [shard/probe.py](shard/probe.py) to measure available host RAM, pinned memory limits, and effective H2D bandwidth; supply V4-specific resource profiles.
   - Budget head, tail, and intermediate stages separately: include tail's 3 MTP blocks, and mandate layers 40–42 co-located on the tail node.

4. **Refactor V4 Stage Loader with Dual Weight Pools**
   - Refactor initialization and loading in `engines/deepseek_v4/v4_stage.py`:
     - Attention, routers, shared experts, normalization / HC parameters, and boundary embedding/head reside in GPU VRAM.
     - Routed experts remain stored in host RAM in native FP4 weights and scales, pinned under explicit memory budgets.
     - GPU cache stores transient copies of these routed experts.
   - Separate allocation pools at construction time: never allocate full layers to GPU and offload afterwards. Retain all-resident mode as a baseline comparison and rollback path.

5. **Implement Fixed-Slot Local GPU Expert Cache**
   - Introduce an expert cache manager with fixed-address GPU weight slots, expert-ID-to-slot mapping, and slot lease state tracking.
   - First iteration guarantees correctness of slot loading, hit detection, eviction, and slot reuse. Follow up with per-layer quotas, frequency decay, and conversation heat adjustments.
   - Total cache capacity is strictly budgeted to preserve room for CUDA graphs, prefill activation buffers, and transport rings.

6. **Integrate On-Demand H2D DMA & Adapt CUDA Graphs**
   - Transfer cache-missed experts from pinned host RAM to pre-allocated GPU slots via DMA, executing with existing GPU math kernels.
   - Update `v4_moe_grouped.py` and `v4_whole_layer_graph.py` address bindings, graph capture boundaries, and stream synchronization events.
   - Overlap H2D transfer with shared expert or resident expert compute where possible; leave attention-overlapping prefetch as a separate follow-up.
   - Preserve logical expert accumulation order, hash-routed duplicate expert semantics, and speculative rollback behavior. Keep CPU fallback disabled by default.

7. **Wire Scheduler to Enforce Dual-Resource Placement**
   - Update `scheduler.py`, `plan.py`, and `topology.py` scoring and capacity logic to consume measured host RAM, H2D bandwidth, and GPU constraints.
   - Placements must strictly satisfy RAM, GPU, and loading peak constraints, incorporating stage compute time, cache-miss DMA costs, tail box workload, and WAN edge RTT.
   - Validate against explicit 4-stage and 6-stage tiered configurations before enabling automated placement solvers.

8. **Execute Phase 1 4-Node / 6-Node Hardware Acceptance**
   - Run end-to-end hardware acceptance over the complete execution path: compare 6-node all-resident vs. 6-node cache mode, then evaluate 4-node configurations.
   - Thoroughly cover cold cache, warm cache, frequent evictions, multi-turn dialogues, speculative rejection, and rollback.
   - Evaluate whether reduced ring hops over WAN offset added DMA latencies, adjusting cache budgets and layer assignments accordingly.

9. **Implement Chunked Prefill & Controlled Prefetching**
   - Once the on-demand cache path is proven, introduce chunked prefill state management to reduce peak activation and temporary buffer memory.
   - Large chunk prefills stream sequentially through non-resident experts; smaller chunks fetch on demand based on actual routing.
   - Implement speculative decode prefetching using historical heat or prediction heuristics, falling back gracefully to on-demand misses upon misprediction.
   - Validate numerical parity, TTFT, and cache pollution across diverse chunk sizes.

10. **Implement V4-Dedicated KV Bounds & Paging**
    - Manage sliding-window state, compressed KV, indexer read sets, compressor states, and rollback checkpoints.
    - Pinned host RAM retains full conversation history; GPU VRAM holds the immediate active working set.
    - Validate lossless storage migration, long-context integrity, and speculative rollback. Re-balance GPU budget between expert cache and KV slots based on empirical data.

11. **Post-Speedline Capabilities & Extension**
    - CPU expert computation may only be enabled after separate numerical parity and performance acceptance.
    - Secondary models, cross-node expert replication, and broader runtime generalizations proceed only after passing 4-node/6-node speed gates.
    - Activation privacy and decentralized coordination remain dedicated independent tracks.

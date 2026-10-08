# Resource contracts and memory-aware placement

Code audit: **2026-10-08, `c2ab623`**. Resource evidence, runtime mechanisms and
hardware acceptance are separate. V4 defaults to GPU-resident experts; its opt-in
RAM expert pool/cache, controlled prefetch, layer KV working set and gated prefill
query chunks are connected to Stage execution. They require explicit budgets and
do not imply the four/six RTX 5090 speed gates passed. CPU expert fallback and
cross-node expert copies are not production miss paths. See
[V4_HYBRID_RUNTIME.md](V4_HYBRID_RUNTIME.md).

`shard/resources.py` defines `shard-placement-requirements/1`. Every size is an
integer **byte count**. Negative values, booleans, floats and missing serialized
components are rejected. A requirement for a nonempty GPU stage must contain a
nonzero weight budget and measured provenance tied to the checkpoint and exact
runtime configuration. Storage estimates are a different schema and cannot be
submitted as measured requirements.

## Independent resource budgets

GPU peak budget is the sum of these explicitly declared components:

| Component | Meaning |
|---|---|
| `resident_weights_bytes` | Permanent main GPU weights; includes routed experts in resident mode, excludes the canonical RAM pool in hybrid mode |
| `kv_hot_bytes` | The GPU KV/state reservation for the measured context, batch and rollback depth |
| `expert_cache_bytes` | Actual bounded local GPU expert-slot reservation; zero when no cache is used |
| `graph_bytes` | CUDA graph allocation budget |
| `workspace_bytes` | Kernel/allocator workspace budget |
| `activation_bytes` | Prefill/decode activation budget |
| `load_peak_extra_bytes` | Additional load/repack transient above resident allocations |
| `boundary_bytes` | Embedding, output norm/head and transformer-level HC parameters assigned to this stage |
| `draft_bytes` | Additional DSpark allocations, including its expert banks and state; shared embed/head storage is counted once |
| `reserve_bytes` | Explicit remaining context/fragmentation reserve |

Host peak budget sums `routed_experts_bytes`, `kv_offload_bytes`, `prefill_bytes`,
`staging_bytes`, `draft_bytes`, `load_peak_extra_bytes` and `reserve_bytes`.
`pinned_bytes` is a **subset** of this RAM budget and is not added again. The
complete routed expert pool must fit that pinned subset. GPU-resident mode has no
canonical host expert pool. In RAM mode, account main and MTP pools and DMA/router
staging. The layer KV mode also accounts host histories, rollback/gate snapshots
and one shared device working area; its read set and workspace remain context-dependent.

`NodeResources(available_vram_bytes, available_ram_bytes, pinnable_ram_bytes)`
accepts `None` for unknown capacity. `evaluate_fit()` checks the three independent
budgets and returns `fits`, `insufficient`, or `unknown` with binding resources.
Unknown pinned capacity fails closed with an actionable allocation-verification
reason. Collect free capacities before the proposed load; a loaded process's
remaining free VRAM is not the original capacity available to its complete stage.

## Live host and transfer capability

`shard.probe.measure_host_resources()` is independent of the old M25 admission
function. It imports no torch. Linux uses `MemAvailable` and the tightest remaining
hard memory limit across the current cgroup and its ancestors. Both cgroup v1 and
v2, unlimited child groups and mounted cgroup namespace roots are handled. Missing
limit or usage evidence remains unknown instead of assuming unrestricted RAM.

Windows uses `GlobalMemoryStatusEx` for actual physical free RAM. If the process
is in a Windows job, its allocatable RAM remains unknown until the job's remaining
limit is independently established; physical free RAM is still reported separately.
Unsupported platforms report unknown fields.

Linux's `RLIMIT_MEMLOCK` and current `VmLck` are reported as OS lock-limit evidence.
They **do not establish the CUDA driver's total pinnable capacity**. Windows lock
capacity is likewise unknown. `pinnable_ram_bytes` stays `None` until the complete
pool's pinned budget has been verified for that node.

`shard.probe.measure_transfer_resources()` optionally performs an actual bounded
CPU pinned allocation and CUDA-event-timed asynchronous H2D copies. Torch is
imported only when called. It reports tested bytes and effective bytes per second.
A successful sample proves only that sample can be pinned; it does not authorize
pinning the entire expert pool. Missing dependencies/CUDA or allocation/timing
failure preserves `None` measurements, never a fabricated zero bandwidth.

The helpers themselves do not alter the legacy M25 `derive_role()` thresholds.
Their output is measurement input, not automatic permission to load or a complete
placement calibration. The current planner separately supports GPU/RAM estimates
and exact measured stage templates.

```powershell
python -m shard.probe --resources
python -m shard.probe --resources --h2d --device cuda:0 --h2d-sample-mib 16 --h2d-repeats 5
```

## V4 header inventory

`engines/deepseek_v4/v4_resources.py` can read a converted `model*-mp1.safetensors`
checkpoint without torch or loading tensor data. It validates prefix/header
lengths, duplicate keys, dtype/shape bit sizes, packed-bit alignment, contiguous
nonoverlapping offsets and complete payload coverage. FP4 `F4` records logical
shape at four bits per element; the raw tensor byte count comes from the offsets.
This follows the upstream [safetensors format implementation](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs).

The inventory distinguishes `layers.N.ffn.experts.E.*` from permanent attention,
router, shared-expert and HC weights. Quantization scale tensors are included.
Embedding, final norm/head, transformer HC parameters and `mtp.*` are classified
separately. `head.weight`, DSpark `markov_w2.weight` and confidence projection
weights have known fp32 parameter lower bounds even when stored in bf16. These
lower bounds still exclude graph, KV, workspace and transient allocation peaks.

Converted MTP checkpoints omit `embed.weight`/`head.weight` copies because they
alias the tail's actual modules. Independently stored MTP aliases are rejected.
DSpark stage inventory requires all three MTP blocks and all target layers
40/41/42 on the same tail. It also requires the tail's embedding, norm, head and
HC parameters. This is architecture validation, not a new placement algorithm.

The inventory's `metadata-sha256:` identity pins headers, filenames, file sizes
and config bytes. **It does not verify tensor payload integrity.** Continue using
the existing signed manifest/fetch validation and full checkpoint hashes in the
benchmark envelope. Changing payload bytes while keeping headers unchanged is
outside this cheap identity's integrity scope.

Example (no GPU or model download):

```powershell
python -m engines.deepseek_v4.v4_resources inventory --checkpoint C:\models\v4 --lo 40 --hi 43 --tail --dspark --output tail-storage.json
```

The structural profile only states 43 layers, DSpark targets 40/41/42, a minimum
three-layer tail and a 32768-byte bf16 boundary payload. It is **unmeasured**.
The legacy `plan_ring()` refuses schema-tagged structural, inventory, measurement
or resource-requirement dictionaries, preventing accidental merging into M25's
default calibration. `shard.plan.PROFILES` now has V4 Dual/Resident variants. Their
scalar footprint/timing defaults are planning estimates, not measured per-node
resource contracts or hardware acceptance; production formation requires fresh
fitting exact templates. Explicit model IDs never silently select M2.5/V4 for a
different model.

## Actual Stage calibration path

`measure_stage_resources(stage, draft=None, checkpoint_id=...)` inventories the
live Stage's actual module parameters/buffers, unregistered state and rollback
tensor containers. It counts physical storage once across bank views and shared
DSpark aliases, preserving actual dtypes/devices rather than checkpoint dtypes.
A provided or attached loaded drafter is included. Missing DSpark allocation
evidence is reported explicitly.

It reports process allocator allocation/reservation and free device memory. Peak
allocation/reservation is only returned when the caller explicitly owns an
isolated measurement interval; this helper never resets shared peak statistics.
Graph/workspace/activation/context decomposition remains `None` when attribution
is unavailable. Reserved allocator bytes are not labeled parameter bytes.

The following command loads the real V4 Stage on an already available local GPU,
loads DSpark when requested, and runs synthetic prefill/decode and draft forwards:

```powershell
python -m engines.deepseek_v4.v4_resources measure --checkpoint C:\models\v4 --lo 40 --hi 43 --tail --dspark --device cuda:0 --max-seq 8192 --prefill-tokens 128 --decode-tokens 32 --output tail-measured.json
```

Run in a fresh, isolated GPU process with the intended `V4_*` kernel/graph flags.
The command honors the effective runtime-metrics setting. An external allocator
measurement interval prevents job observers from resetting or mislabeling the
calibration's load/forward peak; instrumentation settings remain in the runtime
configuration fingerprint. Measurement spans loading and execution, including lazy
graph allocation, and reports the load interval separately. Nonfinite outputs
fail calibration. Middle/tail inputs in this isolated probe are synthetic; route
coverage is sampled. This is neither a WAN throughput benchmark nor evidence for
40 tok/s, full route coverage, or hybrid-cache performance.

Unavailable CUDA is an explicit error, not a CPU-based GPU calibration. No GPU
rental or weight download is performed. The command imports the same flat engine
module identity used by the live serve path; direct absolute-path invocation also
works outside the repository current directory.

The measurement report intentionally has `placement_requirements: null` until
all budgets have been explicitly calibrated. An undivided allocator peak cannot
honestly supply separate graph/workspace/KV/activation budgets as guessed zeros.
`load_calibration()` accepts only a complete `shard-placement-requirements/1`
object with `provenance.kind = "measured"`, checkpoint identity, config SHA-256,
timezone-bearing timestamp, node, method and evidence reference. Every GPU/host
component must be explicitly present. Zero is allowed only as a declared absent
component, not as a default for missing evidence.

`ModelRuntime.placement_requirements()` fails explicitly until implemented by a
backend. The concrete V4 Stage delegates to `placement_requirements_for_stage()`:
it verifies the loaded checkpoint's metadata identity, layer span, roles/config/
environment/runtime-version/engine-source fingerprint and actual current GPU
module/process allocator storage lower bound. Shared-process allocations may make
this check conservatively fail; use isolated measurement/placement evidence. A DSpark
stage needs the loaded drafter before this check can succeed. DSpark capability is
bound to the loaded configuration, not a per-job reset toggle. Existing storage
inventory or speculative hybrid estimates cannot satisfy it.

## Verification and remaining work

The CPU test suite validates independent RAM/VRAM/pinned failures, unknown
capacities, complete serialized budgets, malformed sizes/provenance, cgroup v1/v2
and namespace/ancestor bounds, corrupted safetensors headers, real packed FP4,
DSpark placement constraints, fp32 promotions, alias/storage deduplication and
the stdlib-only CLI outside repository cwd.

GPU loader/graph peaks, actual large pinned pools and H2D throughput still require
real hardware measurements. CPU coverage and optional mechanisms cannot establish
V4 4x5090 >=40 or 6x5090 >=30 committed decode tok/s. Preserve the frozen
[V4_BENCHMARK.md](V4_BENCHMARK.md) evidence gate; GPT-OSS's separately requested
backend/recipe does not count as passing it.

## Exact templates and leases

Open capability `calibrations` contain complete requirements and their actual
`runtime_config` payload. The registry exports only fresh fitting templates as
`allowed_spans`, with head/tail, actual GPU/host/pinned bytes, runtime digest and
optional stage-index/nstages constraints. The planner jointly searches these
discrete spans rather than emitting an arbitrary split for a later loader to reject.
Template GPU peaks already include their components; scalar head/tail/load reserves
are not deducted again. Native GPT-OSS storage/configuration provides an additional
physical weight lower bound, not a replacement for runtime peaks.

`LeaseResources(req.gpu.peak_bytes, req.host.peak_bytes, req.host.pinned_bytes)`
reserves per-GPU and shared memory-domain budgets. Multiple local agents sharing
RAM must use one ledger. Preparation/loading/idle residency remain reserved until
the actual process exits and cleanup is acknowledged. Lease expiry alone does not
free resident allocations. Signed requirements establish provenance of a report,
not remote hardware attestation. See
[OPEN_INFERENCE_NETWORK.md](OPEN_INFERENCE_NETWORK.md) and
[GPT_OSS_PRODUCTION.md](GPT_OSS_PRODUCTION.md).

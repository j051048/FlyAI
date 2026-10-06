# Resource evidence before memory-aware placement

This implements the resource contract and measurement foundation for the V4 experiment.
The current loader still keeps routed experts on the GPU. No RAM expert pool, GPU
expert cache, DMA miss path, paged KV, CPU expert branch or new layer allocation is
enabled by these APIs. Ring, transport and speculative decoding remain unchanged.

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
| `resident_weights_bytes` | The assigned main layers' permanent GPU weights; includes routed experts for the current resident loader |
| `kv_hot_bytes` | The GPU KV/state reservation for the measured context, batch and rollback depth |
| `expert_cache_bytes` | The future local GPU expert cache; explicitly zero for the current loader |
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
proposed complete routed expert pool must fit that pinned subset. The current
GPU-resident loader has no host expert pool; a future hybrid budget is not evidence
that a hybrid loader has been installed.

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

These helpers do not alter `derive_role()`, the existing M25 probe, scheduler
capacities or layer assignment.

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
default calibration. No V4 profile is registered in `PROFILES` yet.

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
The command forces Stage runtime receipt metrics off so calibration alone owns
the CUDA peak interval. That effective choice is included in the runtime config
fingerprint. Measurement spans loading and forward execution, including lazy
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
stage needs the loaded drafter before this check can succeed. Existing storage
inventory or speculative hybrid estimates cannot satisfy it.

## Verification and remaining work

The CPU test suite validates independent RAM/VRAM/pinned failures, unknown
capacities, complete serialized budgets, malformed sizes/provenance, cgroup v1/v2
and namespace/ancestor bounds, corrupted safetensors headers, real packed FP4,
DSpark placement constraints, fp32 promotions, alias/storage deduplication and
the stdlib-only CLI outside repository cwd.

GPU loader/graph peaks, actual large pinned pools and H2D throughput still require
real hardware measurements. Later work must build the local hybrid loader/cache,
verify exact kernels and rollback behavior, add bounded KV and prefill, and only
then enable resource-aware allocation. Keep V4 as the sole model until the
4x5090 >=40 tok/s and 6x5090 >=30 tok/s benchmark gates pass with complete evidence.

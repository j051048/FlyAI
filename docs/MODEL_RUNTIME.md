# Model execution behind a shared contract

Code audit: **2026-10-08, `c2ab623`**. The 2026-06-28 decision separated network
orchestration, placement and verification from model execution. It is a direction,
not a claim that every architecture is already supported.

## The seam

`shard/node.py` defines `ModelRuntime`: `load_shard`, `reset`, `heartbeat`,
`placement_requirements`, `embed`, `forward` and `logits`, with optional batch
methods. Default methods fail explicitly until implemented. Loading may materialize
weights into declared GPU/host tiers; the interface does not itself offload experts
or certify resource peaks.

Existing execution paths are not all subclasses of this interface:

| Path | Actual implementation |
|---|---|
| MiniMax M2.5 | Tuned stage/coordinator in `engines/minimax_m25/`, inherited vLLM quant kernels, model-specific state/drafter |
| Kimi K3 | `engines/kimi_k3/`, its own attention/residual boundary and rewind constraints |
| DeepSeek V4 | `engines/deepseek_v4/v4_stage.py`, inherited reference blocks/kernels with HC, hash-routing, Compressor/Indexer and speculative-state adaptations |
| GPT-OSS | `phase0/pipeline.py` / `specpipe.py` plus `engines/gpt_oss/network_service.py`, Transformers partial layers and native MXFP4 |
| Apple MLX | `shard/mlx_runtime.py` implements `ModelRuntime` for MLX artifacts; separate quantization/numerical cohort |

A generic `VllmRuntime` loading arbitrary registry architectures is still not
shipped. A catalog/model ID cannot create its execution, wire, state, tokenizer or
resource adapter.

## Shared contracts and backend responsibilities

Transport, signed offers/cohorts, resources, leases, planner, ring lifecycle and
HTTP/SSE are shared spine modules under `shard/`. Each model owns its forward,
reset/rewind, boundary representation, drafter inputs and numerical policy.
Manifest/config-derived key selection is supported where implemented; recognizing
a namespace alone does not establish model support.

A cohort binds checkpoint/manifest/configuration, quantization, runtime ABI, wire
ABI, numerical contract and real layer count. Stage calibration additionally binds
exact span, boundary roles and runtime configuration digest. Optional stage-index
and nstages fields constrain actual execution geometry. Explicit future-model
profiles must not silently inherit V4/M2.5 defaults.

V4 resident and RAM-expert execution are separate configurations. RAM mode constructs
canonical pinned expert banks without first allocating the whole bank on the GPU,
then uses leased fixed GPU slots and existing kernels. Cache state stays local.
Production misses execute on the GPU after H2D; CPU reference constructors are tests,
not a production fallback.

V4 KV working sets and query-chunk prefill preserve its actual reference state. They
are not generic K/V eviction or tokenwise prefill. Budgets, rollback and first-shape
output/state gates are explicit; graph modes must fit the storage mode. See
[V4_HYBRID_RUNTIME.md](V4_HYBRID_RUNTIME.md).

GPT-OSS `dtype="auto"` and the codec preserve the actual configured tensor dtype.
Wire size follows real configuration or explicit measured evidence. Native MXFP4
prerequisites and packed layout are checked. Inventory metadata cannot replace
first-load full file hashing. See [GPT_OSS_PRODUCTION.md](GPT_OSS_PRODUCTION.md).

Determinism/greedy parity is per backend and configuration. Receipts commit reported
roots; they do not make different kernels/quantizations identical. MLX community
affine-4bit and NVIDIA NVFP4 cannot silently share a cohort promising identical
native arithmetic.

## Adding a backend

1. Establish checkpoint/configuration, boundary shape and required token data.
2. Implement loading, forward/state rollback, head/tail and generation semantics.
3. Measure isolated graph/workspace/load peaks, KV, draft and host/pinned budgets,
   then publish exact executable templates.
4. Validate supported numerical modes, sessions and receipts before readiness.
5. Measure the intended hardware/workload; forecasts and CPU doubles cannot pass
   a GPU speed or quality gate.

Training needs a separate execution core. See
[RESOURCE_CONTRACT.md](RESOURCE_CONTRACT.md),
[OPEN_INFERENCE_NETWORK.md](OPEN_INFERENCE_NETWORK.md) and
[MLX_RUNTIME.md](MLX_RUNTIME.md).

# Shard architecture

Code audit: **2026-10-08, `c2ab623`**. This describes implemented interfaces and
their limits. Historical model-specific benchmark numbers are not network-wide
performance guarantees. V4's four/six RTX 5090 speed targets remain hardware gates.

## Execution and ownership

A pipeline covers a model with nonempty, contiguous half-open layer blocks:

```text
coordinator -> head/embed -> middle blocks -> tail/norm/head -> coordinator
```

The coordinator may share the head's host or run elsewhere. A single stage can
own both head and tail. Distinct GPUs may share one physical host or subnet;
neither co-location nor a public IP establishes zero-cost transport or independent
failure domains. See [COLOCATION_POLICY.md](COLOCATION_POLICY.md).

Model fit depends on checkpoint representation, KV/state, graph/workspace/load
peaks and boundary/draft roles. Pipeline parallelism alone does not make a 200 GB
resident model fit in 96 GB of VRAM.

| Part | Current code and responsibility |
|---|---|
| Model execution | `shard/node.py` defines `ModelRuntime`; M2.5, K3, V4, GPT-OSS and MLX paths have their own loading, state and numerical contracts |
| Transport | `shard/transport.py` is the tensor/message codec; Go libp2p sidecars own peer connections, encryption, NAT traversal and relays; `phase0/wire.py` remains the explicit PSK/TCP path |
| Generation | Model coordinators drive greedy/speculative verification and rollback; each backend preserves its own state and acceptance semantics |
| Placement/lifecycle | `offers.py`, `locality.py`, `plan.py`, `topology.py`, `planning_cost.py`, `leases.py`, `control_plane.py` and `ring_pool.py` under `shard/` handle compatibility, prediction, reservation and readiness |
| Request service | `shard/http_gateway.py`, `service_queue.py` and `ring_router.py` share authenticated HTTP/SSE, tenant limits, idempotency and one serial worker per ring |

The dependency direction remains control plane -> engine contracts. Model kernels
are not imported into the generic planner. [MODEL_RUNTIME.md](MODEL_RUNTIME.md)
explains the execution seam; [OPEN_INFERENCE_NETWORK.md](OPEN_INFERENCE_NETWORK.md)
describes the reference control service.

## Model-specific execution

V4's real boundary is `[batch, tokens, 4, 4096]`: **32768 bf16 bytes per token**,
plus required token-ID data. It cannot collapse to one hidden stream between
stages. Its Compressor/Indexer and rollback state are not ordinary transformer K/V.

`V4_EXPERT_PLACEMENT=gpu` remains the resident execution default. Opt-in RAM
placement uses pinned local expert banks and leased GPU slots, with GPU execution
after local H2D misses. CPU expert fallback and cross-node expert replicas are not
production miss paths. Layer KV working sets and gated prefill query chunks have
explicit budgets and state/output gates. Query chunks preserve full projection,
Compressor and MoE shapes, so they do not bound every prefill allocation. See
[V4_HYBRID_RUNTIME.md](V4_HYBRID_RUNTIME.md) and
[RESOURCE_CONTRACT.md](RESOURCE_CONTRACT.md).

GPT-OSS uses its existing Transformers/MXFP4 partial-layer runtime plus shared
authenticated sessions and leased serving. Unsupported native packing and silent
dequantization fallback fail loading. See [GPT_OSS_PRODUCTION.md](GPT_OSS_PRODUCTION.md).
MLX has a separate artifact/numerical policy; its historical slice gate is not a
generic mixed-backend proof.

## Locality, routes and prediction

New open formations search local/region or freshly measured low-latency candidates
first, then expand when examined local choices cannot serve the workload. Unknown
regions may join by measured closeness. The open frontier is bounded before dense
adaptation (default 32 candidates); exhausted truncated search is not proof of
global infeasibility. Legacy explicit dense callers retain compatibility.

Edges are directed **effective data channels**. A valid chain does not need a
bidirectional head-star. Dense/v1 RTT keeps its conservative RTT-as-hop meaning.
Version 2 distinguishes measured one-way delay from conservative RTT or explicit
RTT/2 symmetry assumptions, and records route, endpoints and actual dialer.
Production pins measured `dial_endpoint` to the actual connect address. The
coordinator dials the tail's return listener although result data flows tail ->
coordinator; external coordinator entry/return costs must be supplied.

Serial cost retains summed stage service and links. Pipeline prediction includes
finite windows, serialization, shared resources, feedback, acceptance, stale/replay
work and fill/drain; it is not sum replaced by max. Measured K+1 chunks require
`shard-stage-trace/2` binding node, GPU, cohort, runtime configuration, context and
warmness. Exact calibrated templates constrain span, head/tail and optional
stage-index/width geometry. Results remain `prediction_only`.

## Open contribution and resource lifetime

Any valid identity may register zero or more capabilities. Execution additionally
requires an exact model cohort, fresh usable observations, reachable routes, fitting
templates, committed leases and verified warmup. Unsupported devices stay registered
without becoming compatible model stages. GPU budgets are per device; host RAM and
pinned subsets aggregate by declared memory domain. Reservations survive loading
and idle residency until actual process cleanup is acknowledged. Expiry is not proof
of cleanup.

HTTP admission and locally approved node templates are distinct from open registration.
c0mpute accounts, pricing, payments and reputation are consuming network policies,
not automatically implemented by this engine checkout.

## Evidence and privacy

Signatures bind identity, jobs, nonces, coverage, activation commitments and optional
metrics. They do not attest physical hardware, prove model computation or conceal
activations from processing nodes. Signed offers/metrics retain that distinction;
separate challenges and live benchmark evidence are needed for stronger claims.

Boundary trust/pinning and V4 token-sealing options are explicit policies. Link
encryption protects transport, not endpoints. See
[V4_TRUST_BOUNDARIES.md](V4_TRUST_BOUNDARIES.md). This code/CPU-test audit does not
establish V4 4x5090 >=40 or 6x5090 >=30 committed decode tok/s.

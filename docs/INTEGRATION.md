# Integration boundaries: engine, control plane and external networks

Aligned with `c2ab623`, 2026-10-08. FlyAI is a fork of the upstream Shard engine used by c0mpute. This document describes the code in this repository; external worker releases, billing and payment deployments need their own evidence. Current operations are documented in [OPEN_INFERENCE_NETWORK.md](OPEN_INFERENCE_NETWORK.md), [GPT_OSS_PRODUCTION.md](GPT_OSS_PRODUCTION.md) and [V4_NEXT_PHASE.md](V4_NEXT_PHASE.md).

## Dependency boundary

The engine must not depend on c0mpute accounts, USDC, pricing, stake or payment APIs. A separate network may consume its public node offers, assignments, measurements and receipts. This repository now includes a reusable reference registration/control/service implementation; saying that all registration, scheduling and routing live only in c0mpute is obsolete.

| Repository component | Responsibility |
|---|---|
| `shard/offers.py` | Open signed offers, exact model cohorts, sequence and TTL validation |
| `shard/locality.py`, `planning_cost.py`, `plan.py`, `topology.py` | Sparse measured routes, locality tiers, resource-aware layer assignment and finite-workload cost predictions |
| `shard/control_plane.py` | Offer registry and identity-authenticated node resource RPC; configured controller orchestration |
| `shard/leases.py`, `leased_runtime.py` | Durable local quotas/fencing and the actual resident engine process lifetime |
| `shard/pipeline_plan.py`, `pipeline_session.py` | Strict assignment and nonce-signed drive/forward/return sessions |
| `shard/ring_pool.py`, `ring_router.py`, `service_queue.py`, `http_gateway.py` | Ready-ring routing, per-ring serial work, bounded tenant admission, HTTP/SSE and cancellation |
| Model engines/adapters | Actual loading, kernels, KV state, token rendering and runtime-specific receipts |
| External integrations | Accounts, money, invoices, reputation policy, actual settlement and deployment distribution |

The common `ModelRuntime` interface is still a migration direction, not proof that an arbitrary model runs unchanged. Model-specific kernel and state contracts remain distinct.

## Identity and transport

Nodes reuse their existing Ed25519 libp2p identity. Offers are signed by that identity; an account system may independently prove and store ownership. There is no account requirement or operator whitelist for registering a valid offer. Execution still requires supported model/runtime, measured capacity, reachable routes and a live committed lease.

The [Go sidecar](../sidecar/README.md) exposes localhost TCP tunnels, not a new Python Unix-socket/gRPC interface. It owns libp2p peer authentication, link encryption, optional QUIC, relay reservation/hole punching, and a dedicated DHT/block-fetch path. Activation traffic uses point-to-point streams. Each production ring authorizes its intended neighbors; omitting `-allow` retains the sidecar's open legacy mode and is not the recommended ring policy.

`shard.transport` handles JSON headers and raw tensor blobs over that local connection. Raw TCP through `phase0/wire.py` with a shared `SHARD_PSK` remains explicit compatibility. Neither link encryption nor an authenticated peer proves correct model execution, and local engine/sidecar endpoints must stay in their intended trust boundary.

## Model identity: two different manifests

**Signed publisher manifest / assigned block fetch.** The existing `shard.manifest` and `shard.fetch` path uses content IDs, independently pinned publisher keys, expected model/layer metadata and monotonic manifest versions. `mf1:<name>@<cid>` pins canonical bytes; the name is advisory. Peers or mirrors can provide untrusted bytes, which are checked against the trusted manifest. The pin must come from trusted distribution/configuration, not the same untrusted assignment being checked.

**GPT-OSS complete HF download inventory.** `phase0/get_model.py` resolves one immutable HF commit, verifies size plus upstream SHA256/Git-blob digest, resumes through `.part`, and installs verified files atomically. `.shard-download.json` binds repo, revision and the full file digest list; it is a local content manifest, not a publisher signature. Root loader overrides and unverified index targets are rejected, while unreferenced nested backups may remain.

The public recipe is `verify_inventory(model_dir, verify_files=True)` followed by `build_cohort_from_inventory(verified)`. Serialized JSON, a stored PASS boolean and safetensors config/index/header metadata cannot substitute for actual complete payload hashing. The resulting full `ModelCohort` binds checkpoint/config identities, quantization, runtime ABI, wire version, numerical contract and layer count; `cohort_id` hashes that exact descriptor. Offers and sessions sign their use of it, not its hardware or publisher authenticity.

The current strict GPT-OSS stage and coordinator require every file listed in the complete download inventory to be present locally and verified. A stage loads only its assigned layers into GPU memory, but its disk directory is a complete verified snapshot. Generic block-range fetching is a separate path; partial inventories are not implemented for this strict entry.

GPT-OSS supports `gpt-oss-hf/1`, `shard-pipeline-session/1`, `greedy-native-mxfp4/1` and `mxfp4`. Production load also checks actual raw config bytes and full local files. Native quantization guards prevent Transformers fallback before conversion and validate packed weights/scales afterward; their storage-layout result does not certify GPU math or speed.

## Planning, admission and resource lifetime

Search near-neighbor and same-region candidates first, expanding when they cannot fit. Compare a few strong cards with strong-plus-weak combinations using stage compute, actual payload routes, finite in-flight feedback, acceptance/discard assumptions and resource budgets. Fewer hops alone does not establish the faster plan; weak stages can set the pipeline floor.

Layer assignment uses actual supported calibrated spans and roles. Trace identity binds node/GPU/cohort/runtime, layer range, frame size, context, warmness, time and TTL. Unknown/unbound measurements cannot make a node fast. Header storage is a physical floor, not allocator/load/graph peak. All planner results remain scoped predictions until measured on the actual ring.

Distinct GPU UUIDs can share a physical host. Aggregate RAM and pin quotas use an explicit memory domain and one local ledger; a shared public IP is only a discovery clue. Optional host/subnet isolation and request-specific trust preferences do not turn registration into a trusted-operator whitelist. Unknown budgets fail closed for requirements needing them.

Node-local authenticated RPC prepares and commits a lease bound to ring, cohort, node, GPU, resources, expiry and increasing fence. Remote controllers may select locally configured executable templates; they cannot supply arbitrary argv/environment or claim a caller identity in request data. Loading and idle resident weights count as work. Expiry/revocation refuses new work, stops the owned runtime and retains allocation until the old work/process is confirmed cleaned up.

`shard.managed_launch` is a manual process helper with durable pre-spawn reservation and explicit orphan recovery. It confirms owned process termination before clearing records and protects against PID reuse. It does not reserve GPU/RAM and cannot replace the node lease contract.

## Request and service contract

The strict plan fixes full contiguous layer coverage, actual stage/coordinator endpoints and signing identities. Nonce challenge signatures bind plan, role, index, range and channel purpose. Owner/fencing metadata prevents an old coordinator or frame being relabeled into a new session. Expected return listeners and bounded first-frame/reconnect waits are part of that contract.

Strict production services are `engines/gpt_oss/network_service.py` and `engines/deepseek_v4/v4_network_service.py`. The older V4 gateway is a compatibility interface. M2.5 has its own legacy swarm-token/receipt protocol; the new session ABI is not silently imposed on it.

V4/GPT-OSS share authenticated HTTP/SSE with per-tenant quotas and job visibility. Public listening requires TLS; default listening is loopback. API keys, coordinator keys, HF tokens and sealed-ID secrets belong in protected local files. The coordinator key path must match the plan identity. Deployment sends environment through SSH stdin rather than command values.

A ring executes one job at a time; multiple ready rings can run in parallel. Idempotency, queue/history and retained SSE prefixes are in memory. Cancellation/deadlines abort owned work, and permitted same-ring retries re-prefill the committed prefix. Durable job/coordinator HA and migration to arbitrary replacement rings are not implemented. Warmup and completion require valid final receipts and live leases; streamed prefixes are provisional until final validation.

## Receipts, privacy and external settlement

Stage signatures authenticate declared activation roots, role/range, job identity and nonce. Verifiers check pinned signers, coverage and adjacency. They do not prove physical GPUs, source execution or all model operations. GPT-OSS's current production verifier requires distinct stage receipt signers; reusable host PeerId/GPU offer identities alone do not remove that receipt constraint.

`phase0/activation_proof.py` and `fraud_proof.py` are local replay/commitment helpers. A trusted adjudicator must independently recover pinned weights, backend, architecture and exact KV/position state. Their recommendations explicitly do not execute escrow or on-chain penalties. Historical payment notes describe an external integration, not automatic settlement provided here. See [PROOF.md](PROOF.md).

Inference nodes see activations. V4 sealed IDs restrict raw IDs and token hints to designated head/hash/tail recipients, while keyless middle stages forward an opaque envelope. This does not hide activations or protect from the privileged operator of a shared host. Staked boundary preference is a policy choice, not cryptographic confidentiality. GPT-OSS authentication does not provide the V4 sealed-ID mechanism.

## Evidence and remaining work

Historical [GPT-OSS 2026-06-19](receipts/gpt-oss-120b-wan-20260619.json) and [GLM 2026-06-18](receipts/glm52-nvfp4-wan-20260618.json) reports retain approximately 40 and 30 tok/s respectively. Historical [V4 2026-08-02](receipts/v4-flash-matrix-20260802.json) reports 30.15 tok/s with same-ring token comparisons. None certifies the current strict services or new cache/KV paths.

The selected 2026-10-08 local non-GPU suite reported 985 passed and 3 skipped. Next acceptance requires real native kernels, numerical/state controls, measured resource/runtime templates, actual routes, short/long-context workload matrices and sustained fault recovery. Planning predictions, signed declarations and local tests remain separate forms of evidence.

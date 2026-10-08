# Deployment readiness and evidence boundaries

Aligned with `c2ab623` on 2026-10-08. This is a repository deployment checklist, not a declaration that an external c0mpute deployment or payment service is live. The selected local non-GPU regression set reported **985 passed, 3 skipped**; it is not whole-repository CI and does not certify current GPU throughput.

## Current deployment paths

| Path | What is implemented | What still needs validation |
|---|---|---|
| GPT-OSS strict service | Full-file download identity, supported cohort, authenticated pipeline sessions, leases, multi-ring HTTP/SSE | Actual native MXFP4 GPU output/state parity, measured workload profiles, sustained hardware runs |
| V4 strict service | `v4_network_service.py`, signed assignments/sessions, resident process leases and shared authenticated HTTP | New cache/KV/graph paths and fixed four/six-card speed campaign |
| M2.5 compatibility ring | Independent engine, sidecars, swarm greetings, gateway and historical receipts | Exact installed backend/driver compatibility and any newly chosen shape or topology |
| GLM / other research scripts | Archived experimental drivers and results | Production adapters, resource and protocol contracts are not implied by a historical script |

Use [GPT-OSS production](GPT_OSS_PRODUCTION.md), [V4 current phase](V4_NEXT_PHASE.md), [open-network contract](OPEN_INFERENCE_NETWORK.md) and [M2.5 compatibility runbook](../phase0/DEPLOY_M25.md). Older V4 `v4_gateway.py` deployment/ring-pool options are compatibility interfaces, not the strict session entry.

Strict GPT-OSS nodes and the coordinator currently need the complete local file snapshot listed by `.shard-download.json`; loading only an assigned GPU layer range does not mean downloading only that range. Generic `shard.fetch` block-range propagation is separate, and this entry has no partial-inventory support.

## Before a production ring becomes READY

1. **Bind actual model bytes.** GPT-OSS downloads resolve an immutable HF revision, verify sizes and digests, then atomically write `.shard-download.json`. First production load rehashes actual files and matches the complete `ModelCohort`. Root loader overrides and unlisted index references fail; unreferenced nested backups may remain. A JSON PASS flag or header-only inventory is insufficient.
2. **Validate the installed implementation.** GPT-OSS requires `model_type=gpt_oss`, native `mxfp4`, actual layer count, ABI `gpt-oss-hf/1`, wire `shard-pipeline-session/1` and numerics `greedy-native-mxfp4/1`. The quantization guard checks prerequisites and loaded packed storage, not kernel execution or performance. Non-MXFP4 generic loading is a separate path.
3. **Bind identity, roles and routes.** Pin stage/coordinator signing identities, full contiguous layer coverage and actual drive/forward/return endpoints in the deployment plan. Strict sessions require nonce signatures and owner fencing. libp2p authenticates peers; ring authorization remains necessary for an open pool.
4. **Reserve real resources.** Use measured GPU/load/graph/workspace budgets and actual host/pinned capacity. Sum RAM/pin quotas by declared physical memory domain, never public IP. Unknown sizes and absent calibration are not zeros. A node lease covers loading, idle resident weights and work through cleanup.
5. **Complete warmup.** Successful TCP connection or a listening log is not READY. Require a real inference warmup, final signed receipts and a live committed lease. Header storage floors and finite-depth cost predictions remain predictions.
6. **Protect the serving boundary.** Shared V4/GPT-OSS HTTP requires tenant auth; public listeners require TLS. Keep node/controller/token keys in protected local files. Deployment environment travels through SSH stdin, not secret argv values.
7. **Confirm cleanup and recovery.** Expired/revoked leases reject new work while old work and resident processes retain their resources until confirmed exit. Manual launch cleanup terminates only owned processes; an unresolved durable reservation/orphan blocks restart until explicit local recovery.

## Service limits

The shared service uses a fair bounded queue and serial execution per ring; different ready rings can run concurrently. It is not continuous batching within one V4/GPT-OSS ring. Cancellation, deadlines and disconnects abort owned work. SSE reconnection preserves the published token/text prefix for the same authenticated tenant and retained job; idempotency and job history are process-local. There is no durable coordinator/job HA or guaranteed transparent recovery after service restart.

Final success requires receipt validation. A stream may expose a committed prefix before that validation finishes; those tokens are not a completed proof. Signatures bind declarations and activation roots, not physical hardware or all model operations. Local fraud helpers do not escrow assets or execute on-chain slashing. See [PROOF.md](PROOF.md) and [V4_TRUST_BOUNDARIES.md](V4_TRUST_BOUNDARIES.md).

Greedy capability is entry-specific. The GPT-OSS production numeric contract is greedy; historical speculative sampling experiments do not enable sampling in every gateway. M2.5 rejects unsupported non-greedy HTTP parameters. Encrypted transport is not activation confidentiality; boundary placement and sealed V4 IDs do not make arbitrary middle-node computation cryptographically private.

## Historical measurements, retained with their scope

| Date / original record | Historical observation |
|---|---|
| [2026-06-23 GPT-OSS async prefill](receipts/async-send-ttft-20260623.json) | Same-ring 30k TTFT 153.3→60.8 s; 110k 245.9→210.0 s. Workload/topology-specific, not a universal prefill bound |
| [2026-06-23 batch primitive](receipts/batched-verify-20260623.json) | GPT-OSS block 1.60× at B4, 2.10× at B8 with batched MoE; 1.24× at B8 with per-stream MoE. Cross-token-count quantized-kernel drift was observed |
| [2026-06-23 cold failover](receipts/fault-tolerance-20260623.json) | 189 committed tokens preserved; about 131 s failover including cold reload |
| [2026-06-23 hot failover](receipts/hot-standby-failover-20260623.json) | 423 committed tokens preserved; about 32.6 s interruption, including that harness's coordinator restart |
| [2026-06-24 libp2p A/B](receipts/libp2p-warm-ab-20260624.json) | Matched WAN RTT: libp2p 25.28 vs raw TCP 25.55 tok/s, same-engine output comparison |
| [2026-06-28 M2.5 usability](receipts/m25-usability-20260628.json) | Reported 28.7k-token prefill without OOM; short-context 20.63 and long-context 6.36 tok/s |

These results do not transfer automatically to the current strict protocols, GPU libraries, frame/context shapes, cache modes or receipt timing. Archive experiments distinguish copy/retrieval from novel generation; neither a fast copy case nor a topology-specific research limit is a universal product claim.

## Remaining hardware acceptance

Run the frozen checkpoint/source/config and same-ring greedy controls before reporting speed. Test actual kernel, graph, rollback/KV state and end-to-end token parity separately. Record short/long-context novel/copy/code cases, TTFT, actual new committed tokens, full service duration including recovery and final proofs, p50/p95 and sustained failures. `recv_wait` includes scheduling/discards and is not RTT.

V4's new-path targets remain **4×5090 ≥40 and 6×5090 ≥30 tok/s** under the frozen [benchmark protocol](V4_BENCHMARK.md). Historical six-card speed and CPU tests cannot close this hardware gate. External payment, pricing, reputation enforcement and external worker rollout require their own deployment evidence.

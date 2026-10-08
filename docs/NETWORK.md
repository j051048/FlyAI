# Open inference network: discovery, placement and serving

Code audit: **2026-10-08, `c2ab623`**. FlyAI's Shard engine can be consumed by
c0mpute or another control plane. Open contributor registration, executable model
eligibility, request authentication, payments and privacy are distinct policies.
See [OPEN_INFERENCE_NETWORK.md](OPEN_INFERENCE_NETWORK.md) for runnable contracts.

## A pipeline and its coordinator

Stage nodes own contiguous layer blocks, plus embedding on the head and output
norm/head on the tail. The coordinator drives generation and may be on the head
host or outside it. A single GPU can own the complete model if its real resource
contract fits; multiple colocated GPUs are also valid. Model execution remains
behind a backend-specific contract, not an automatic universal model loader.

```text
coordinator -> head -> middle stages -> tail -> coordinator
```

The result leg is logically tail -> coordinator, but GPT-OSS's coordinator dials
the tail's return listener. Data direction and connection reachability are not
interchangeable. Transport may use direct links, loopback/private paths, existing
libp2p sidecars or explicitly provisioned tunnels. Same public IP does not prove
same host or a usable hairpin route.

## How candidates become executable rings

1. Any valid Ed25519 identity may register capabilities, including none. Signatures
   identify the reporter; they do not attest actual GPUs or truthful performance.
2. Exact cohorts bind checkpoint/configuration, quantization, runtime/wire ABI,
   numerical policy and real layer count. Incompatible backends are not mixed.
3. Fresh resource/route observations feed a bounded frontier before any dense
   solver mesh. Prefer local/complete regional or measured low-latency candidates;
   expand when examined local choices cannot serve. Unknown region labels may join
   by observed proximity. Exhausting a truncated frontier is not global infeasibility.
4. Jointly select head/tail, directed order and layer ranges. Production uses fresh
   fitting exact resource templates, including declared stage-index/width geometry.
   Per-GPU VRAM and shared host RAM/pinned reservations must all fit.
5. Prepare/commit leases, launch only locally approved node templates, then require
   real signed warmup before READY and HTTP/SSE admission.

Distinct GPUs on one host/subnet can cooperate. Isolation `host`, `subnet` or
`adjacent_host` is optional; `none` is the production default. Public IP and keys
are not physical failure-domain evidence. [COLOCATION_POLICY.md](COLOCATION_POLICY.md)
covers grouping and the shared-memory ledger.

New registrations do not reshuffle active rings. Multiple rings can serve in
parallel with one serial worker each, shared tenant limits and version-sticky SSE.
This is not continuous batching. Draining blocks new bindings while resources stay
reserved until real process cleanup. The reference service remains centralized and
replaceable; its presence does not implement consensus or payment decentralization.

## Latency is one cost, not the only cost

Serial autoregressive traversals pay summed stage service plus actual forward and
entry/return paths. Finite-depth speculative pipelines also pay serialization,
feedback, fill/drain, shared resource contention and rejected/replayed work.
Their cost is not simply the slowest stage or the sum changed to max.

A bigger GPU can remove a stage only when actual capacity, role templates and
layer coverage allow that reduction. A 20/8/8 split and a 12/12/12 split both have
three stages. Same-region placement alone does not prove a speedup: compare bound
stage service, link bandwidth/delay, drafter acceptance and context. Three strong
GPUs can beat one strong GPU plus weak stages; measured geometry decides.

Dense/v1 RTT retains legacy conservative RTT-as-hop pricing. Version 2 records
the effective channel, actual dialer/connect address, reachability, TTL and
measured one-way or explicitly normalized RTT semantics. Production must use the
route that was measured, including external coordinator entry and return.

Single-token estimates remain supported. Measured K+1 verifier chunks require
`shard-stage-trace/2` matching node/GPU/cohort/runtime configuration, frame tokens,
context and warmness. Acceptance/cancel assumptions are explicit; all cost outputs
are `prediction_only`, never hardware speed acceptance.

## Weights and numerical evidence

The signed-manifest peer-fetch path in `shard/fetch.py` verifies supplied content
against pinned hashes and publisher trust. GPT-OSS additionally uses an immutable
download inventory binding repo/revision and file digests, with complete first-load
rehashing. Header/storage inspection alone is not payload verification.
[GPT_OSS_PRODUCTION.md](GPT_OSS_PRODUCTION.md) documents that runtime's loading gate.

Greedy/speculative numerical validity is backend/configuration-specific. Receipts
bind reported signers, jobs, nonces, layer coverage and activation commitments.
They do not alone prove honest computation, truthful hardware/metrics or
cross-quantization equivalence. Spot-checking and independent live verification
remain separate. Computing participants see activations; link encryption and
optional boundary policies do not create universal prompt confidentiality.

## Historical performance and current gates

Earlier records reported GPT-OSS around 40 tok/s and GLM around 30 tok/s under
particular deployments; M2.5's June-August measurements span very different
draftable, reasoning and batched workloads. They remain historical experiments,
not current guarantees for every card, region, model or numerical recipe. See
[M25_ENGINE.md](M25_ENGINE.md), [ROADMAP.md](ROADMAP.md) and [receipts](receipts/).

The current V4 acceptance targets remain **4x5090 >=40** and **6x5090 >=30**
committed decode tok/s under [V4_BENCHMARK.md](V4_BENCHMARK.md)'s frozen evidence
protocol. Code/CPU fixtures do not pass those GPU gates. A local multi-GPU run
must not be labeled a distinct-host WAN result. Historical recovery experiments,
including [the 2026-06-23 record](receipts/fault-tolerance-20260623.json), also do
not prove zero-cost KV migration for every backend.

Model-specific resource and execution details belong in
[RESOURCE_CONTRACT.md](RESOURCE_CONTRACT.md) and
[MODEL_RUNTIME.md](MODEL_RUNTIME.md); c0mpute pricing/settlement remains an external
integration rather than an automatic consequence of joining this repository.

# Proving a real decentralized swarm run

"You served a 120B model across consumer GPUs on different networks" is an extraordinary
claim, so it should be checkable by a skeptic — not taken on trust. This doc defines what a
*verifiable* Shard run looks like and how anyone can confirm one independently. The historical
**run record** helper is `phase0/proof_receipt.py`; archives live in `docs/receipts/`.
Aligned with `c2ab623` on 2026-10-08. Keep unsigned historical run summaries, current signed
stage receipts, benchmark measurements and local dispute helpers as distinct evidence.

## What would a fake look like?

The cheap fakes we're ruling out:
1. **One box pretending to be many** — running the whole model on a single machine and
   claiming it was distributed.
2. **Localhost, not WAN** — "distributed" processes all on one host/LAN, no real internet hop.
3. **Cherry-picked / fabricated tok/s** — a number with nothing reproducible behind it.
4. **Wrong output** — a fast pipeline that doesn't actually compute the model correctly.

The checks below retain evidence for independent reproduction. A signed declaration alone cannot exclude all fabricated hardware or computation claims.

## Evidence and verification boundaries

**1. Node identity and deployment provenance.**
Retain declared host identity, GPU UUID, signing key, reachable endpoint and software/source
versions. Distinct addresses or UUID strings are not remote hardware attestation. Independent
inventory and measurement provide provenance for a WAN claim. Co-located distinct GPUs are a
valid production assignment; their shared host budgets must be accounted once.

**2. The links are real WAN, not localhost.**
Retain measured routes and app-level RTTs (`phase0/mesh.py` for historical experiments).
Current route measurements also bind logical endpoints, actual channel/dial address, time,
TTL and latency semantics. A large RTT or geographic label alone does not prove a WAN hop;
independent topology and packet-path measurement are needed. `recv_wait` includes engine
queue/discard waits, and authenticated control RTT includes processing; neither is pure network RTT.

**3. Output parity within an explicit numerical contract.**
The current V4 benchmark compares committed token IDs against a greedy control on the same
running ring, checkpoint, wire mode and frozen prompt. This does not prove all intermediate
floats or a different single-machine backend are bit-identical. Reference/eager, kernel,
graph, state and rollback parity require separate tests with pinned hardware and numerics.

**4. Anyone can re-run the whole thing.**
The engine is open source (Apache-2.0). Retain exact commit/source hashes, complete checkpoint
identity, runtime and numerical configuration, layer→node assignment, workload and launch
commands. Independent reproduction must match those contracts and hardware/backend scope;
an arbitrary different backend or frame shape need not produce bit-identical results.

GPT-OSS's `.shard-download.json` binds immutable repo/revision and complete file sizes/digests.
First production load hashes actual files and checks the supported full `ModelCohort`. A local
inventory, a cohort hash and `packed_layout_verified=True` are not publisher signatures,
hardware attestations or native GPU execution proofs. The guard explicitly reports
`native_execution_verified=False`; details are in [GPT_OSS_PRODUCTION.md](GPT_OSS_PRODUCTION.md).

**5. Signed execution records and local dispute primitives.**
The live service signs activation roots and validates the assigned signers, complete layer
coverage, adjacent roots, job/swarm identity and fresh nonce. Hashing has a cost; receipt
signatures attest the declared bytes, not complete model execution or physical hardware.

`phase0/activation_proof.py` and `phase0/fraud_proof.py` provide optional local commitment and
arbitration helpers. They bind logical tensor bytes, input/output snapshots, stage/step,
engine identity and a challenge deadline. Same-architecture replay is required; optional
tolerance compares only the committed output snapshot. Their result explicitly reports
`onchain_executed=False`. There is no asset escrow, chain transaction or automatic slashing
in these helpers, and they are not a per-step production proof collector.

A real validator must independently load the pinned weights/backend and recover the exact
KV, Compressor and position state before replay. An arbitrary caller-supplied replay function
or self-declared architecture is not sufficient evidence. Billing and actual dispute settlement
remain the external integration's responsibility.

**6. Configurable Placement Isolation (可选的部署隔离).**
- **Co-location Allowed in Production:** Ring planning (`shard/topology.py`) defaults to
  `isolation="none"`: distinct GPUs on one mining host or subnet may serve adjacent stages.
  Explicit `host`, `subnet` and `adjacent_host` policies remain available for deployments that
  require them. A public IP does not identify a physical host; topology declarations alone do
  not prove Sybil resistance. Reachable, measured local/private routes address NAT hairpinning.
  Duplicate declared GPU UUIDs cannot contribute capacity twice, and known shared RAM budgets
  are checked once across the host's stages. See [COLOCATION_POLICY.md](COLOCATION_POLICY.md).
- **WAN Evidence:** A co-located deployment is a valid inference run, but does not establish a
  distinct-host WAN result. Such benchmark claims retain their independent inventory and
  evidence requirements.
- **Optional Boundary Policy:** A configured request/placement policy may require sensitive
  boundary roles on trusted/staked nodes. A `staked: true` declaration does not establish reputation
  or cryptographic concealment, and public offer registration is not restricted to those nodes. Legacy V4 frames carry token
  IDs to every stage. Opt-in sealed-ID mode hides IDs from keyless score-routed middle stages,
  including pipelined token hints, while activations remain visible. Head, every hash-layer
  recipient and the tail remain trusted. Co-located processes controlled by one operator share
  that operator's trust domain. See [V4_TRUST_BOUNDARIES.md](V4_TRUST_BOUNDARIES.md).

## Historical run-record example (`docs/receipts/<run_id>.json`)

This is an illustrative historical run summary, not the current signed stage receipt schema.
`phase0/proof_receipt.py` writes an unsigned envelope and reports self-consistency only.
Current service receipts must be retained in full, including their signatures and assigned keys;
the GPT-OSS benchmark stores them inside the measured result rather than converting them to this shape.

```json
{
  "run_id": "...", "utc": "...", "shard_commit": "<git sha>",
  "engine_file": "research/...", "engine_sha256": "<hash of the engine source that ran>",
  "model": "gpt-oss-120b", "quant": "mxfp4",
  "prompt": "...", "output_text": "...",
  "output_token_ids": [ ... ], "output_sha256": "<hash of token ids>",
  "tok_s_warm": 24.8, "decode": "greedy (exact)",
  "nodes": [
    {"role": "coordinator|stage|tail", "layer_range": [a, b],
     "public_ip": "x.x.x.x", "geo": "Kansas, US",
     "gpu_uuid": "GPU-...", "gpu_name": "RTX 4090"}
  ],
  "edges": [ {"from": "stage0", "to": "stage1", "rtt_ms": 41.2} ],
  "reference": {"source": "single-node decode", "tokens_match": true, "token_ids": [ ... ]}
}
```

`engine_sha256` is the hash of the exact engine source that produced the run — the precise,
commit-independent reproducibility anchor (a commit can't embed its own hash, so this is what a
skeptic checks the engine file against). `reference.token_ids` lets anyone re-run the verifier with
`--ref-tokens` and confirm the lossless check directly, rather than trusting the `tokens_match` flag.

## How to verify a receipt (skeptic's checklist)

1. **Check record scope:** distinguish unsigned/self-reported fields from assigned stage signatures;
   run `python phase0/proof_receipt.py verify RECORD --ref-tokens CONTROL_IDS` only for its documented
   historical self-consistency/reference check. Passing it is not verification of physical distribution.
2. **Authenticate the assignment:** independently pin expected stage/coordinator keys, full layer
   coverage, job identity and nonce, then verify actual stage signatures and adjacent roots.
3. **Verify model and numerical identity:** hash complete model files and actual source/config;
   reproduce the same pinned hardware/backend/shape contract with independently retained control IDs.
4. **Validate topology independently:** inspect actual GPU/host inventory and measured routes for a WAN
   claim. UUID/IP/region strings alone are declarations. Co-location is valid but is not distinct-host evidence.
5. **Reproduce performance:** preserve workload, warm/cold scope, actual committed-token counters and
   full timing, including retries and proof checks. Review p50/p95 and repeatability, not a best case alone.

## Receipts on file

- **GLM-5.2 744B NVFP4 at ~30 tok/s over WAN** — 7 GPUs in 6 US states, pipelined spec-decode
  + CUDA-graphed draft: [`receipts/glm52-nvfp4-wan-20260618.json`](receipts/glm52-nvfp4-wan-20260618.json).
- **gpt-oss-120B at ~40 tok/s over WAN** — 3 stages (12 layers each) + an in-region coordinator
  across 4 US states, pipelined spec-decode, historical 2026-06-19 record:
  [`receipts/gpt-oss-120b-wan-20260619.json`](receipts/gpt-oss-120b-wan-20260619.json).

## Scope / honesty

- A greedy record must state its exact numerical contract. A claimed `tokens_match` flag is
  not independent evidence; retain both output dumps and the control configuration. Same-ring
  committed token parity does not imply equality of all intermediate floating-point tensors.
- The separate sampling experiment supports **temperature/top-p/top-k sampling** (`phase0/specsample.py`) — not
  bit-deterministic but **seeded-reproducible**, and proven lossless *distributionally* (the committed
  token distribution equals the target's): [`receipts/sampling-lossless-20260623.json`](receipts/sampling-lossless-20260623.json).
  Its numerical/distributional control is scoped to that experiment. Current GPT-OSS production
  and the M2.5 HTTP gateway have greedy contracts; this record does not enable sampling there.
- For a **quantized** model, bit-exact reproduction across *different* engines/backends is not
  achievable in general (and not unique to Shard): batched vs single-token kernels round
  floating-point differently, so at a genuine near-tie two correct greedy decoders can pick
  different — both valid — continuations. So the proof is *within-engine* reproducibility +
  coherent correct output, not "matches your laptop's HF decode token-for-token."
- A receipt authenticates specific declarations; independent inventory, controlled replay and
  observation establish stronger execution claims. It does not alone prove distribution, correctness,
  uptime or throughput SLAs. Planner predictions, CPU tests and historical speed records do not
  certify the current strict protocol or new GPU/cache paths. The selected 2026-10-08 local
  non-GPU set (985 passed, 3 skipped) is not a whole-repository/GPU acceptance result.

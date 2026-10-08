# Proving a real decentralized swarm run

"You served a 120B model across consumer GPUs on different networks" is an extraordinary
claim, so it should be checkable by a skeptic — not taken on trust. This doc defines what a
*verifiable* Shard run looks like and how anyone can confirm one independently. Every run can
emit a **run receipt** (`phase0/proof_receipt.py`); receipts live in `docs/receipts/`.

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
The receipt records the **measured RTT of every pipeline edge** (`phase0/mesh.py`, app-level
round-trip over the live transport). Real inter-city internet is tens-to-hundreds of ms;
localhost is <1 ms. *Verify:* the edge RTTs are WAN-scale and match the geographic distances.

**3. Output parity within an explicit numerical contract.**
The current V4 benchmark compares committed token IDs against a greedy control on the same
running ring, checkpoint, wire mode and frozen prompt. This does not prove all intermediate
floats or a different single-machine backend are bit-identical. Reference/eager, kernel,
graph, state and rollback parity require separate tests with pinned hardware and numerics.

**4. Anyone can re-run the whole thing.**
The engine is open source (Apache-2.0). The receipt embeds the exact commit, model, layer→node
assignment, and launch commands. *Verify:* stand up your own nodes and reproduce — same code,
same result.

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
or self-declared architecture is not sufficient evidence. Billing and dispute settlement
remain the external control plane's responsibility.

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
- **Staked Boundary Pinning:** Sensitive input/output layers (embedding and lm_head) are pinned to
  high-reputation staked nodes (`staked: true`), providing a placement preference, not cryptographic concealment. Legacy V4 frames carry token
  IDs to every stage. Opt-in sealed-ID mode hides IDs from keyless score-routed middle stages,
  including pipelined token hints, while activations remain visible. Head, every hash-layer
  recipient and the tail remain trusted. Co-located processes controlled by one operator share
  that operator's trust domain. See [V4_TRUST_BOUNDARIES.md](V4_TRUST_BOUNDARIES.md).

## Receipt schema (`docs/receipts/<run_id>.json`)

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

1. **Distinct machines:** all `public_ip` differ and resolve to different networks/regions; all
   `gpu_uuid` differ.
2. **Real WAN:** every `edges[].rtt_ms` is WAN-scale (≫ 1 ms) and consistent with the geos.
3. **Correct output:** re-run the same `model` + `prompt` with greedy decoding anywhere; confirm
   the token ids hash to `output_sha256`.
4. **Reproduce:** check out `shard_commit`, bring up nodes, run the embedded commands.

## Receipts on file

- **GLM-5.2 744B NVFP4 at ~30 tok/s over WAN** — 7 GPUs in 6 US states, pipelined spec-decode
  + CUDA-graphed draft: [`receipts/glm52-nvfp4-wan-20260618.json`](receipts/glm52-nvfp4-wan-20260618.json).
- **gpt-oss-120B at ~40 tok/s over WAN** — 3 stages (12 layers each) + an in-region coordinator
  across 4 US states, pipelined spec-decode; the permissionless build target:
  [`receipts/gpt-oss-120b-wan-20260619.json`](receipts/gpt-oss-120b-wan-20260619.json).

## Scope / honesty

- Decoding is **greedy and deterministic**: same prompt → same tokens, every run. The receipt's
  `tokens_match` is the **lossless-optimization check** — the CUDA-graphed speculative path is
  byte-identical to the plain eager path of the *same engine* (computed from two real run dumps),
  so the speedup changes nothing about the output.
- The engine also supports **lossless temperature/top-p/top-k sampling** (`shard/specsample.py`) — not
  bit-deterministic but **seeded-reproducible**, and proven lossless *distributionally* (the committed
  token distribution equals the target's): [`receipts/sampling-lossless-20260623.json`](receipts/sampling-lossless-20260623.json).
  The greedy receipts above use the deterministic path; `temp=0` is bit-identical to it.
- For a **quantized** model, bit-exact reproduction across *different* engines/backends is not
  achievable in general (and not unique to Shard): batched vs single-token kernels round
  floating-point differently, so at a genuine near-tie two correct greedy decoders can pick
  different — both valid — continuations. So the proof is *within-engine* reproducibility +
  coherent correct output, not "matches your laptop's HF decode token-for-token."
- A receipt proves *a specific run* was real, distributed, and correct. It is not a claim of
  uptime, throughput SLAs, or that every run hits the same number — tok/s is prompt- and
  topology-dependent and reported as a range.

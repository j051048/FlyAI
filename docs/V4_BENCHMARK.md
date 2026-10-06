# Reproducible V4 baseline and acceptance

`phase0/v4_benchmark.py` freezes benchmark inputs, drives **an existing V4 ring** through its current
coordinator APIs, and independently checks the resulting JSON evidence. It does not rent machines,
download weights, start/stop stage processes or change the ring, transport, speculative acceptance
rule or CUDA kernels. Preparing and checking evidence do not import the CUDA model runtime.

This is instrumentation and an acceptance contract. No new four-card or six-card GPU measurement
is implied by these CPU tests or by the checked-in historical results.

## Fixed protocol

The protocol identifies `deepseek-ai/DeepSeek-V4-Flash-0731`, 43 target layers, FP4 routed experts,
and records:

- SHA-256 of **all weight bytes**, config, tokenizer artifacts, file sizes and the complete file list.
  A matching file size or safetensors header alone is not checkpoint verification.
- Actual runtime source file hashes, their aggregate digest and the Git revision when available.
  Working tree changes are identified by content hashes, even before a commit exists.
- Explicit V4 kernel, graph, quantization, KV reservation and speculative-decoding settings.
  `DEFAULT_ENV` is a frozen experiment recipe, not an assertion that every machine supports it.
- Four fixed workloads: Rust code, online mean/variance math, library-outage prose, and incident
  response/tool-planning. Exact token IDs and their hashes are saved; the tokenizer is never rerun
  during measurement. Deterministic context token IDs are inserted after the chat prefix to reach
  the requested total context length, preserving BOS, role markers, the task and assistant suffix.
  This is explicitly token-level padding, not a claim of whole-string re-tokenization equivalence.
- Context and fixed committed generation length (defaults: 512 prompt tokens and 512 new tokens),
  greedy temperature zero, seed zero, exactly one excluded first-case request, and three measured
  warm requests for every workload. EOS is ignored to keep the token length fixed.
- Four or six distinct single-GPU RTX 5090 hosts, actual VRAM bytes, GPU UUIDs, driver/CUDA/PyTorch/
  TileLang versions, process-run identity, signing keys and assigned layer ranges. Assignments must
  tile `[0,43)`; the tail must own layers 40, 41 and 42. Every host declares matching checkpoint,
  config, source and stage-environment digests.

A protocol is sealed with an aggregate SHA-256. A changed context, prompt, kernel setting or model
produces a different protocol. Pass `verify --protocol` to bind a report to a separately retained
protocol rather than trusting only the copy embedded in the report.

## Timing and the two speed gates

The live runner timestamps **committed-token callbacks** using `time.perf_counter()`. Drafted,
rejected, replayed and in-flight frame counts remain diagnostics and never enter the throughput
numerator. It records every callback offset, time to first token, final token time, coordinator
return time and the independent receipt-sweep duration.

```text
decode committed tokens/s = (N - 1) / (last committed callback - first committed callback)
end-to-end committed tokens/s = N / elapsed to coordinator return
TTFT = first committed callback - request start
```

Decode speed excludes prefill/TTFT. End-to-end speed includes the existing reset, prefill, decode
and coordinator drain. **Both exclude the separate receipt sweep**, which has its own duration.
There must be at least two committed tokens. Report verification recalculates the rates from the
raw times and committed counts; it does not trust `tokPerSec`, `generated` or `receiptsOk` flags.

The default acceptance statistic is the median of the **complete 12-run warm suite** (four
workloads × three repetitions). No workload or repetition can be omitted. Workload medians,
ranges, E2E speed and TTFT are also reported separately:

| Assignment | Required complete-suite warm committed decode median |
|---|---:|
| Four RTX 5090 hosts | ≥40 tok/s |
| Six RTX 5090 hosts | ≥30 tok/s |

This is not an additional requirement that the slowest workload's E2E median exceed the target.
New experimental contexts or kernels can be frozen in another explicit protocol, but reports with
changed model/prompt/context/run/kernel recipes are not given a verified before/after comparison.

## Cold requests and parity controls

The candidate runs **before** the same-workload greedy control. A control cannot accidentally warm
the first candidate request. Job resets clear sequence/KV/rollback state using the existing ring
contract; they do not claim to evict future GPU expert caches or clear compilation caches.

Only the **very first code request** can be called process/model cold. The other first-case requests
are labeled `warmup`, and all four first-case requests are excluded from the speed statistic. The
`--fresh-ring` flag is an explicit **operator assertion** that the ring's processes/models were
freshly started before the suite. The runner does not restart the user's hardware and cannot
independently prove process freshness. Without the assertion, a report remains unverified.

Each workload's candidate output token IDs must equal a greedy run on the **same running ring**
using the same frozen prompt IDs and generation length. This preserves the existing committed-token
parity criterion. It is not proof that hidden states or every intermediate float are bit-identical;
existing stage/graph/rollback parity tests remain required before changing those execution paths.

## Signed evidence

Every measured job has a fresh unpredictable nonce and unique job ID. Raw stage receipts are retained,
including optional signed `runtime_metrics`. The independent verifier strips only the existing
post-signing `stage` debug tag, then verifies all of:

1. Signature and signer pinned to the hardware assignment.
2. Exact layer coverage, with no duplicate signer, overlap, gap or zero-work receipt.
3. Per-job nonce, job ID and swarm ID, including uniqueness across the suite.
4. Adjacent output/input roots along the complete ring.
5. Complete same-ring greedy token parity and the frozen measurement protocol.

The default recipe enables `V4_RUNTIME_METRICS=1`. Each receipt must then contain valid signed GPU
telemetry; absent telemetry is unverified, and CPU-reference telemetry cannot qualify the GPU gate.
An explicit protocol with `V4_RUNTIME_METRICS=0` disables only that requirement. Raw receipt signature,
coverage, nonce, chain and token-parity checks always remain mandatory.

The hardware inventory, checkpoint declarations and fresh-process flag are operator provenance,
**not remote hardware attestation**. The local coordinator rehashes its local checkpoint and source
before opening the ring. Signed receipts prove which assigned keys declared the runtime observations;
they are not a cryptographic proof that a physical GPU, a DMA copy or a cache hit actually occurred.

`verify` returns `passed`, `failed` or `unverified`. Missing evidence never yields a speed pass, even
when a file contains impressive aggregate numbers. Non-passing reports and non-comparable reports
produce exit code 2. Partial reports are written after each completed job, so an interrupted suite
preserves raw evidence and cannot qualify through missing jobs.

## Operator workflow

Run from a repository checkout with the existing engine's dependencies and a **local** checkpoint.
The inventory and live preflight each read every weight byte; allow time for hashing large models.
No unverified size-only fast path is provided.

```powershell
python phase0/v4_benchmark.py inventory --dir D:/models/v4 --out local-identity.json
```

`local-identity.json` contains checkpoint/source/environment digests. Collect actual hardware,
software, signing identity and process-run information for each of the four/six hosts, and retain
the collected evidence. `hardware.json` is an array of objects in ring order; one object has this
shape (replace **every** placeholder with the actual deployment values):

```json
{
  "node_id": "node-0",
  "host_id": "unique-host-id",
  "gpu_uuid": "GPU-actual-uuid",
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "vram_bytes": 34359738368,
  "driver": "actual-driver-version",
  "cuda": "actual-cuda-runtime-version",
  "torch": "actual-stage-pytorch-version",
  "tilelang": "actual-stage-tilelang-version",
  "process_run_id": "identity-of-this-stage-process-start",
  "signer_pubkey": "base64-node-ed25519-public-key",
  "layer_start": 0,
  "layer_end": 12,
  "checkpoint_sha256": "checkpoint.sha256-from-local-identity",
  "config_sha256": "checkpoint.config_sha256-from-local-identity",
  "engine_source_sha256": "source.sha256-from-local-identity",
  "stage_env_sha256": "stage_env_sha256-from-local-identity"
}
```

Copying digest strings from the local file declares the nodes use that exact recipe; verify the
deployed files/settings on each host before making that declaration. `stage_env_sha256` hashes the
canonical JSON environment, not the order-dependent text of a shell command. The stage process
environment and its lever audit must match the pinned recipe, including `V4_RUNTIME_METRICS=1` and
receipt-enabled serving. Exporting coordinator settings alone does not change already-running stages.
The tool does not publish or transmit this inventory to a third party.

```powershell
python phase0/v4_benchmark.py prepare --dir D:/models/v4 --hardware hardware.json --out protocol.json
python phase0/v4_benchmark.py run --protocol protocol.json --dir D:/models/v4 --head 127.0.0.1:29610 --tail 127.0.0.1:29612 --fresh-ring --out report.json
python phase0/v4_benchmark.py verify report.json --protocol protocol.json
python phase0/v4_benchmark.py compare before.json after.json
```

`prepare` loads the local tokenizer with `local_files_only=True` and `trust_remote_code=False`, uses
the vendored V4 chat renderer and freezes the resulting IDs. It never calls the currently incomplete
message-encoding branch of the coordinator CLI. `run` directly calls `coordinate`,
`coordinate_dspark` or `coordinate_dspark_pipelined`, with receipts disabled inside generation, then
uses the existing receipt sweep after timing ends. Disconnecting it leaves the stages running.

Use `--env-file overrides.json` with **both** `inventory` and `prepare` for an explicitly different
kernel experiment; overrides are a JSON object of string-valued `V4_*` settings. An unrecorded
`V4_*` environment setting at live start is rejected. Authentication tokens such as
`SHARD_SWARM_TOKEN` stay in the existing environment path and are not placed in protocols or reports.

## Historical result and CPU verification

The existing `docs/receipts/v4-flash-matrix-20260802.json` records six scattered RTX 5090 hosts,
`MAX_SEQ=8192`, a Rust concurrency prompt, 512 generated tokens and repetitions
`19.311, 30.178, 30.287, 30.124` tok/s, with the first excluded and a reported median of **30.15**.
It lacks raw signed receipts, exact prompt IDs and checkpoint hashes. The new report carries it as
`historical_repository_claim_not_independently_verified`, not a newly verified baseline or a pass
on the new prompt suite. No corresponding four-card 40 tok/s result is checked in.

```powershell
python -m pytest tests/test_v4_benchmark.py -q
```

The tests use deterministic clocks, synthetic identities and real Ed25519 signatures. They exercise
measurement boundaries, exact four/six-card thresholds, fixed prompt preservation, raw-signature
tampering, missing evidence, signer/nonce/job/chain binding, complete-suite parity, lazy imports,
existing-coordinator dispatch and a standalone verifier outside the repository. They do not load
weights or run a GPU. Hardware acceptance still requires running the live CLI on the actual fleet.

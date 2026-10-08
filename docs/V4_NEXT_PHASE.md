# V4 runtime and service hardening

Current runtime/service entrypoints below are aligned to `c2ab623` (2026-10-08).
The requested runtime paths are implemented, while real four/six RTX 5090 acceptance
is still pending an explicitly provisioned actual cluster (owned or rented). Local CPU,
real-socket and reference-model regressions are not GPU throughput measurements.
Use a complete repository checkout for the new deployment/gateway workflow.

The subsequent open-contribution work adds signed public offers, node leases,
locality-first heterogeneous planning, multi-ring serving and version-sticky
lifecycle management. See [OPEN_INFERENCE_NETWORK.md](OPEN_INFERENCE_NETWORK.md).

## Execution paths and limits

| Area | Actual integration | Limit |
|---|---|---|
| Token confidentiality | Head seals IDs and pipelined `dnxt` once; keyless score-routed middle stages forward the unchanged envelope; hash layers and tail decrypt | Activations remain visible; privileged operators on a shared host share its trust domain |
| KV working set | Stage keeps windows/Compressor recurrences resident, archives compressed histories in pinned RAM, and reuses one bounded per-layer GPU workspace | Each layer still reads its entire valid compressed prefix; quotas limit supported context and add PCIe traffic |
| Prefill queries | Attention and Indexer queries are chunked while projection, Compressor, HC and MoE shapes remain unchanged | First real shape runs a reference-shadow output/state byte gate and still needs reference scratch; mismatch refuses the request |
| Hardware evidence | Frozen speed suite plus sustained same-prompt greedy controls, raw signatures, monotonic campaign intervals and declared SLOs | Timing and physical inventory are coordinator/operator provenance, not remote attestation |
| Online service | Shared authenticated HTTP, global tenant limits/idempotency, one serial worker per READY ring, version-sticky routing, cancellation/SSE and replay recovery | Rings execute concurrently; history is memory-only; no continuous batching, durable coordinator HA or external billing |
| Connection ownership | Signed challenge HELLO pins both identities, role/index/spans/cohort, purpose and execution plan; signed head grant binds forward/return and fences old owners | Compatible stage/controller keys and actual reachable tail endpoint are required; strict mode refuses the old return relay |
| Miner admission | Distinct GPUs may share a host; real shared RAM/pin budgets, simultaneous H2D measurements and calibrated launch settings are checked | Measurements expire and must be rechecked at load; no automatic rental or remote provisioning |

The local dispute helpers bind logical tensor bytes, both snapshots, deadline and
validator engine identity. They return local decisions with `onchain_executed=False`.
Actual model/KV replay and escrow/settlement require independent infrastructure.
See [V4_TRUST_BOUNDARIES.md](V4_TRUST_BOUNDARIES.md).

## Stage configuration

Install the checkout's common V4 dependencies with `pip install -e ".[v4]"`.
Provision the CUDA PyTorch/TileLang builds verified on your rented GPUs, and retain
their exact versions and source hashes in the protocol. CPU torch cannot run the
hardware probe or qualify a GPU result.

New memory/query paths are opt-in. Set the following explicitly on every applicable
stage **before import**; budgets below are examples to replace with calibrated values:

```bash
export V4_EXPERT_PLACEMENT=ram
export V4_KV_PLACEMENT=layer
export V4_KV_GPU_MIB=2048
export V4_KV_HOST_MIB=8192
export V4_PREFILL_QUERY_CHUNK=512
export V4_CUDA_GRAPH=island
export V4_FAST_VERIFY=0
export V4_MOE_IN_GRAPH=0
export V4_DSPARK_MOE=0
export V4_RUNTIME_METRICS=1
export V4_PROFILE_RUNTIME=1
```

`layer` KV rejects `whole` graphs and fast verify because they retain incompatible
active-storage assumptions. GPU budget covers resident KV state, the shared workspace
and draft reserve; host budget includes compressed histories, rollback snapshots and
gate state. RoPE, full-prompt activations, weights and kernel temporaries need separate
resource headroom. Reset rejects an advertised job horizon beyond KV capacity before
starting the request. Signed `kv_policy` and `prefill_policy` report actual execution.

For sealed IDs, generate a private file on a trusted machine:

```bash
python engines/deepseek_v4/v4_privacy.py keygen --out token-privacy.key
```

The command prints only the public `key_id`. Provision the file only to head/hash/tail
trust domains. Set `V4_SEALED_IDS=1` and `V4_TOKEN_PRIVACY_KEY_ID=<public-key-id>` on
all stages and the coordinator. Set `SHARD_V4_TOKEN_KEY_FILE=<private-file>` only on
trusted recipients. Secret files and transport tokens are never put in a public
deployment environment or forwarded by the launcher. Mixed versions/keys are refused.

## Shared-host onboarding

Before loading stage pools, run a bounded concurrent probe on each participating host:

```bash
python -m shard.host_probe --devices 0,1,2,3 --host-id rig-a \
  --dir /data/v4 --pin-budget-mib <whole-host-pinned-requirement> --out rig-a-io.json
```

The pin budget is really allocated. A successful 64 MiB transfer sample alone never
qualifies a larger expert pool. Allocation is bounded by available RAM and freed after
the experiment. Individual and aggregate decimal GB/s are recomputed from raw CUDA
samples and concurrent wall time; a shared PCIe link is not counted independently per
GPU. The probe is temporary evidence, not a memory reservation.

Measure each actual stage with its assigned GPU, roles, environment and workload:

```bash
python engines/deepseek_v4/v4_resources.py measure --checkpoint /data/v4 \
  --lo <start> --hi <end> --device cuda:0 --max-seq <configured-limit> \
  --prefill-tokens <tested-prompt-limit> --decode-tokens <tested-decode-length> \
  --output stage-resource-observations.json
```

Include `--head`, `--tail`, `--dspark` for its real roles. A resource observation is not
automatically a complete placement calibration: retain a measured byte budget for all
components, including peaks and reserves, as specified in [RESOURCE_CONTRACT.md](RESOURCE_CONTRACT.md).
The report now exports `runtime_config_payload` to bind that calibration to its settings.

The `shard-deployment/1` bundle contains `model_id`, `layer_count`, `checkpoint_id`,
`registration_policy`, `token_privacy`, `stages` and `host_io` keyed by host ID.
Legacy `trust_mode` values are accepted through the compatibility resolver; open registration
and plain/sealed token privacy are separate choices. Every stage contains `node_id`,
`host_id`, physical `gpu_uuid`, `signer_pubkey`, `lo`, `hi`, unique host-local `port`,
`dspark`, public `env`, full `runtime_config`, measured `requirements`, `capacity`
(`available_vram_bytes`, `available_ram_bytes`, `pinnable_ram_bytes`,
`available_disk_bytes`) and `capacity_measured_at`. `requirements` uses the existing
placement schema and matching node/checkpoint/config provenance. `host_io` embeds the
complete raw concurrent report above. Optional `coord_env` binds coordinator flags.

```bash
python -m shard.deployment deployment.json --out deployment-check.json
```

The check rejects duplicated physical GPUs/ports, changed runtime flags, stale reports,
incomplete layers or shared host overcommit. Every field is in bytes. Distinct IPs or
self-declared VM labels do not create extra RAM or independently trusted hardware.

## Strict plan, identity and resident leases

Current stage CLI requires `--deployment-plan` with full `model_cohort`, or the same plan in
its node-local `SHARD_STAGE_LEASE_CONFIG` assignment. Every stage signer and the coordinator's
signer are pinned. Manual coordinators require `--coordinator-key` or `SHARD_COORDINATOR_KEY`;
these reference an existing Python receipt-format key file and never upload its private bytes.
A plan/key mismatch or occupied head is an explicit rejection, not a delayed read or GPU fallback.
Legacy bare-op listeners require `--legacy-protocol` and are not the managed production path.

Node-local SQLite leases prepare/commit resources and enforce fencing/expiry. The resident
process handle stays occupied until the actual child exits. Multiple GPUs on one host use
one shared RAM/pin reservation domain. DRAINING stops new requests but renews leases until
admitted work finishes; only acknowledged cleanup permits reuse. New joins do not rewrite a
live ring. See [OPEN_INFERENCE_NETWORK.md](OPEN_INFERENCE_NETWORK.md) for the reference
controller, local approved templates and independent per-ring renewal.

## Authenticated serving

Create a private auth JSON file with two mappings: `keys` maps long random API keys to
tenant IDs; `tenants` maps those IDs to `max_active`, `requests_per_minute` and
`tokens_per_minute`. Token quota reserves prompt plus maximum completion per admission;
it is a resource limit, not a bill. Do not commit API keys.

```bash
python engines/deepseek_v4/v4_network_service.py --config network.json \
  --auth-file service-auth.json --host 127.0.0.1 --port 8000
```

This `shard-open-network/1` entrypoint connects signed offers, exact calibrations, approved
stage templates, leases and strict pipeline identity, then performs signed warmup before READY.
The older `v4_gateway.py --deployment/--ring-pool` CLI loaders retain compatibility but do not
inject strict plan/controller keys at this snapshot; use them only with explicitly legacy
listeners, as documented in [V4_GATEWAY.md](V4_GATEWAY.md). The managed entrypoint requires TLS
for non-loopback HTTP; the old adapter has a separate explicit trusted-proxy override.
`/health` is process liveness; `/ready` reflects verified
service readiness. Authenticated endpoints include `/v1/chat/completions`, `/v1/models`,
`/v1/jobs/<id>`, `/v1/jobs/<id>/cancel`, `/v1/jobs/<id>/receipts` and `/metrics`.

Cancellation closes active I/O. A bounded reconnect replays the original prompt and
decode mode, verifies the previously published token prefix, and emits only the unseen
suffix. New attempt replies carry job/swarm/nonce identity, and the reset barrier drains
stale replies. Numerical, replay-prefix or receipt errors fail closed. A final successful
attempt supplies one complete receipt set; receipts from different attempts are not merged.
Unsupported sampling/tool/format settings are explicitly rejected. Retained idempotency
and SSE resume are limited to this process and its bounded history.

## Vast cluster acceptance

First run [V4_BENCHMARK.md](V4_BENCHMARK.md): four-card >=40 / six-card >=30 committed
decode tok/s remain unchanged. `prepare --isolation none` allows distinct GPUs sharing
hosts; the default `host` recipe retains the separately scattered-WAN comparison. These
are different frozen protocols. Cover GPU residency versus RAM, cache pressure, long
context, rollback and cold/warm requests, retaining every raw report.

Then run sustained verification on the existing warmed ring:

```bash
python -m phase0.v4_soak run --protocol protocol.json --dir /data/v4 \
  --head 127.0.0.1:29610 --tail 127.0.0.1:29612 \
  --deployment-plan pipeline-plan.json --coordinator-key controller-receipt.key \
  --cycles 25 --duration-s 3600 --ttft-p95-s 30 --token-gap-p95-s 0.25 \
  --max-idle-gap-s 10 --out soak.json
python -m phase0.v4_soak verify soak.json --protocol protocol.json --contract soak-contract.json
```

Freeze the independent contract before running: `cycles`, `duration_s`, `ttft_p95_s`,
`token_gap_p95_s`, `max_idle_gap_s`. Duration requires complete active generation/sweep
intervals, not an editable aggregate number or idle waiting. P95 uses committed token
callbacks; incomplete campaigns, reused identities, wrong signers and changed contracts
cannot pass. The older `v4_acceptance.py` remains a CPU analytical/cache simulation and
now explicitly reports `hardware_verified=False`; it cannot pass the hardware gate.

GPU CI uses an idle dedicated sm120 runner, requires real CUDA and retains JUnit results.
It refuses busy runners and does not kill other GPU workloads. A single-runner kernel
regression is separate from the four/six-card speed and sustained-service acceptance.

## Validation history and current evidence scope

An earlier 2026-10-08 runtime-hardening selected regression recorded **1243 passed,
28 skipped**. A subsequent check of the final gateway readiness, strict benchmark
identity binding, allocator interval and related contracts completed with **101 passed**.
These overlapping suites must not be added together. Full-stack cross-configuration
CPU probes separately completed with **27 passed**. Compilation, documentation references
and `git diff --check` passed; the vendored reference files are unchanged.

The broader Windows CPU run was interrupted after failures outside this V4 regression:
the existing fetch tests require symlink privileges (`WinError 1314`), and the existing
M25 first-ack watchdog test records repeated stubbed exits. Those files are unchanged.
This is not a claim that the complete repository suite passes on Windows. GPU tests are
skipped in the local CPU-only torch environment, and real Vast speed/soak/failure testing
is explicitly deferred by the operator.

The subsequent strict-plan/session/GPT-OSS integration at `c2ab623` records **985 passed,
3 skipped** in its selected CPU/socket suite. It includes authenticated ownership, tail churn,
real tiny V4 reference token/receipt comparisons, shared serving and planner/contracts. It is
not a complete repository CI result, and skipped CUDA cases are unverified hardware evidence.
No new four-card >=40 or six-card >=30 pass is implied by any of these test counts.

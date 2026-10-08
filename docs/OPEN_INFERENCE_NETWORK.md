# Open inference network implementation

Code audit: **2026-10-08, `c2ab623`**. Control/service contracts are distinct from
GPU throughput acceptance and remote hardware attestation.

FlyAI accepts GPU contributors without an operator allowlist. Registration,
model execution eligibility, transport authentication and request privacy are
separate contracts. Distinct colocated GPUs are permitted. Public IP is
informational; region labels guide search and measured effective data channels
constrain placement.

## Implemented components

The existing GPT-OSS runtime now has the same leased serving adapter and authenticated
pipeline session contract. See [GPT-OSS deployment and measurement](GPT_OSS_PRODUCTION.md)
for immutable downloads, exact executable templates, measured chunk geometry and migration.

| Component | Implementation | Responsibility |
|---|---|---|
| Signed open offers | `shard/offers.py` | Reuse Ed25519 libp2p identity, sequence/TTL, model cohorts, optional exact stage calibrations |
| Durable node reservations | `shard/leases.py` | SQLite prepare/commit/renew/release, GPU fencing, aggregate host RAM/pin budgets |
| Local resident lifetime | `shard/leased_runtime.py` | Hold resources through process cleanup; V4/GPT-OSS frames check leases |
| Locality discovery frontier | `shard/locality.py` | Expiring sparse links, bounded candidate selection, local/region/global expansion |
| Joint placement | `shard/plan.py`, `shard/topology.py` | Joint head/tail/layer assignment and distinct serial/pipeline objectives |
| Finite-window cost prediction | `shard/planning_cost.py` | Compute/link/shared-resource calendars, feedback, acceptance and replay assumptions |
| Ring lifecycle | `shard/ring_pool.py` | Lease-bound loading, verified warmup, READY, drain, failure and same-lease recovery |
| Multiple-ring request queue | `shard/ring_router.py` | Shared tenant limits and idempotency, one serial worker per ring, model/version routing |
| Reference control service | `shard/control_plane.py` | Open offer HTTP registry, signed lease RPC, formation and independent renewal |
| V4 serving entrypoint | `engines/deepseek_v4/v4_gateway.py` | Single-ring compatibility and `--ring-pool` serving |
| Generic open service | `shard/network_service.py` | Discover agents, select exact templates, pin actual routes and launch approved local processes |
| GPT-OSS service | `engines/gpt_oss/network_service.py` | Shared lifecycle/HTTP/SSE with native MXFP4 partial-layer execution |

The engine remains independent of c0mpute accounts, reputation and payment.
The reference controller is replaceable. Its node adapters and registry contracts
can be called by another control plane. Existing libp2p sidecars still own peer
connections, encryption, NAT traversal and inference route setup.

## Model compatibility

Every offer can announce zero or more model capabilities. A contributor with no
usable inference calibration stays registered, rather than pretending to have
a measured GPU capability. A compatibility descriptor contains:

```json
{
  "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
  "manifest_sha256": "<64 lowercase hex characters>",
  "checkpoint_id": "<verified checkpoint identity>",
  "config_sha256": "<SHA256 of config.json bytes>",
  "quantization": "<actual weight representation>",
  "runtime_abi": "<compatible runtime ABI>",
  "wire_version": "<compatible activation wire ABI>",
  "numeric_contract": "<validated numerical compatibility>",
  "n_layers": 43
}
```

`ModelCohort.cohort_id` hashes the exact descriptor with domain separation.
The hash excludes stage-specific RAM/cache/graph settings. Those remain bound by
each calibration's `runtime_config_sha256`. A future catalog entry does not make
the current V4 backend implement a new architecture automatically.

## Contributor setup

Use a complete checkout and Python 3.11+. Retain the existing sidecar key outside
the repository. The offer signing CLI reads its existing Go Ed25519 key format;
it does not generate another network identity.

```sh
python -m shard.offers sign --offer unsigned-offer.json --sidecar-key node.key --out offer.json
python -m shard.offers verify --offer offer.json
python -m shard.control_plane registry --db registry.sqlite --host 127.0.0.1 --port 29100
```

POST the signed JSON to `/offers`. POST `{"cohort_id":"..."}` to `/snapshot`
to obtain currently eligible measured candidates. Any valid identity can register.
Node IDs are `PeerId/GPU-UUID`. Registration may represent several GPUs under one
host key, but a selected V4/GPT-OSS ring currently needs distinct stage receipt
signers: use stable per-GPU identities for those stages. One operator may own them
all; different keys do not attest different hardware or owners.
The signing key is never uploaded. Sequence values must increase for updates;
expired bodies release active registration capacity, while a bounded recent
sequence history protects against replay across restart.

An unsigned offer template needs `gpu_uuid`, `memory_domain_id`, `endpoints`,
`resources`, `models`, `issued_at`, `ttl_s` and `sequence`; region/zone, host ID,
public IP and `lease_endpoint` are optional. The signer adds schema, identity and
signature. Unknown resource byte counts are JSON null. `resources` contains
`available_vram_bytes`, `available_ram_bytes`, `pinnable_ram_bytes`,
`available_disk_bytes` and `measured_at` in Unix seconds.

Each model entry contains `cohort`, `profile`, `measured_at`, and optionally
`calibrations`. A calibration is `{"requirements": PlacementRequirements,
"runtime_config": measured_config}`. Node identity, model/checkpoint identity,
layer bounds, roles and configuration digest must match. Declared stage/index/
nstages fields constrain execution geometry too. Different runtime hashes for one
span are distinct templates; exact duplicates fail. Snapshot exports fresh fitting
`allowed_spans` with actual GPU/host/pinned byte budgets. Missing calibration may
stay registered without becoming executable; production formation requires exact
templates. Do not publish commands,
private token keys or `SHARD_*` provisioning in this record.

Node-local lease service:

```sh
python -m shard.control_plane node-leases --db host-leases.sqlite --sidecar-key node.key --capacity capacity.json --port 29101
```

`capacity.json` is local configuration, containing `gpu_uuid`,
`memory_domain_id`, `available_ram_bytes`, `pinnable_ram_bytes`,
`gpu_capacity_bytes` keyed by GPU UUID and `cohort_ids`. Multiple agents on the
same host must use the same SQLite ledger for shared budgets. Initial capacity
measurements and actual stage checks complement the reservation ledger; a
telemetry sample by itself is not a reservation.

Optionally configure `stages` with locally approved templates: `cohort_id`,
`lo`, `hi`, `head`, `tail`, `requirements`, `runtime_config`, `argv` and
`environment`. Only `lo`, `hi`, `stage`, `nstages` and `next` may be interpolated
in argv. Runtime paths, devices and kernel flags are local literals. Effective
`V4_*` flags must match calibration, except relocatable `V4_DIR`/`V4_DEV`.
Remote controllers select these templates; they cannot upload argv or environment.

Use the same identity for receipts, exported in the Python key format when needed:

```sh
python -m shard.offers export-receipt-key --sidecar-key node.key --out receipt.key
```

Point the local stage template's `SHARD_NODE_KEY` at that file. Preserve restricted
key-file permissions. This exports the same key, without printing its secret.

Public control listeners require both `--tls-cert` and `--tls-key`. Lease RPC
requests and responses additionally bind authenticated PeerIds, nonce, timestamp,
ownership and fencing token. Replayed or substituted replies fail verification.
TLS handshakes run inside bounded workers, rather than blocking the accept loop.
Private loopback endpoints are useful with controlled tunnels; automatic discovery
of private endpoints requires explicit local configuration.

## Locality and heterogeneous placement

```python
plan = plan_ring(
    nodes,
    model=profile,
    measurements={"schema": "shard-link-measurements/1", "edges": edges},
    locality={"mode": "prefer_local", "region": "asia", "max_candidates": 32},
    objective="pipeline",
    workload={"depth": 8, "block_tokens": 8, "acceptance_gain": 3,
              "generated_tokens": 256, "frame_tokens": 1},
    diagnostics=diagnostics,
)
```

An edge in version 1 carries `src`, `dst`, `rtt_ms`, `measured_at`, `ttl_s` and optional
`bandwidth_mbps`. The existing measurement/orchestration layer supplies these
observations. Missing and expired links do not become zero-latency edges.
Dense/v1 RTT remains conservative RTT-as-hop pricing; it is not measured one-way
delay. Local candidates are tried first. A complete regional ring is considered
before a mixed-region ring. Expansion reasons and candidate attempts are returned.
Unknown region labels can still participate through measured proximity.

Candidate reduction happens before dense solver adaptation. The default frontier
is 32 nodes, not a full N-squared mesh of the global registry. Joint head searches
are bounded too. Failure after truncation is `bounded_search_exhausted`, not proof
that the entire network cannot run the model. Callers can widen or retry another
region. Legacy callers supplying their own dense mesh retain compatibility.

Serial objectives retain summed traversal cost. Pipeline objectives simulate
finite windows and resource calendars, including propagation, serialization,
shared PCIe service, acceptance assumptions and replay. This version supports
single-token estimates and genuinely bound K+1 chunks. A chunk workload declares
`frame_tokens=K+1`, `draft_tokens=K`, context, cold/warm state, finite depth,
acceptance gain, cancel rate and stale frames per valid round. It requires fresh
`shard-stage-trace/2` matching node/GPU/cohort/runtime hash and that geometry;
missing, expired or differently shaped observations cannot become a fast scalar
chunk estimate. Stale/replay work is conservatively priced without inventing
early-abort savings. Same-K adaptive depth may lower the window; automatic GPT-OSS
formation rejects mixed-K adaptive execution against a fixed chunk calibration.

A usable `shard-stage-trace/1` observation binds node ID, GPU UUID, cohort ID,
runtime configuration hash, single-token frame geometry, layer range and TTL.
Range scaling, missing context/warmness and scalar estimates are labeled as
uncertain. `planning.prediction_only` is always true. These estimates cannot
certify a GPU performance target or manufacture speculative acceptance. Invalid
or mismatched traces fail offer admission; unbound capabilities remain registered
but ineligible, and stale/invalid persisted traces cannot poison healthy candidates.

Version 2 (`shard-link-measurements/2`) records `src`, `dst`, `route_id`, `channel`,
`src_endpoint`, `dst_endpoint`, `dialer_id`, `reachable`, `latency_ms`,
`latency_kind`, `hop_policy`, timestamp/TTL and optional bandwidth. Measured one-way
uses `measured_one_way`; RTT uses `conservative_rtt` or explicit
`half_rtt_assumption`. Distinct effective channels between the same nodes can
coexist; `route_ids` pins a selection. Missing routes stay unreachable. A directed
chain does not require a bidirectional star around its head.

Production v2 additionally requires the actual `dial_endpoint` and explicit
`coordinator_id`. `route_endpoints[route_id]` must equal that measured connect
address and the correct engine dialer. The coordinator opens the tail's return
listener even though reply data logically flows tail -> coordinator. Entry and
return measurements account for an external coordinator; no in-region placement
is inferred from a small latency value alone.

Exact templates constrain span/head/tail and optional stage-index/nstages. Their
actual peaks replace coarse reservation arithmetic without double subtraction.
GPU UUIDs remain unique and shared memory domains aggregate actual RAM/pinned
requirements. Truncated template searches report uncertainty rather than global
optimality or hardware availability proof.

## Formation and serving

`FormationController` connects the registry, planner, exact measured block
contracts, node lease adapters and ring pool. Its backend factory uses the
existing engine launcher; no remote shell string is executed.

Prepared and loading reservations are renewed independently. A slow new ring
does not hold the global controller lock or interrupt existing rings. After all
commits, node-local process runners load the assigned blocks and the actual
backend performs signed warmup before readiness.

The executable V4 reference adapter is:

```sh
python engines/deepseek_v4/v4_network_service.py --config network.json --auth-file tenants.json
```

`network.json` has schema `shard-open-network/1`, `registry_db`,
`controller_sidecar_key`, `formations`, optional `aliases` and `default_model`.
Each formation contains `ring_id`, exact `cohort`, calibrated `profile`,
`measurements` or its JSON path, coordinator `dir`, current launcher `head`/`tail`
routes, locality/workload/mode/lease timing and, for v2, `coordinator_id`, selected
`route_ids` and `route_endpoints`. Legacy v1 uses configured sidecar/`stage_next`
routes; v2 may not silently fall back to their default ports. Configure the
inference sidecars through the existing launcher before serving. The registry
can discover arbitrary signed GPU offers with compatible calibrations and usable
lease endpoints; it does not require an operator roster in this config.

`allow_private_endpoints` defaults false for automatic endpoint discovery;
enable it only for an intentionally configured local/private network. The public
reference adapter expects numeric public-IP HTTPS lease endpoints. c0mpute can
replace the adapter with its own authenticated connection management.

For already formed and separately maintained leased rings:

```sh
python engines/deepseek_v4/v4_gateway.py --ring-pool pool.json --auth-file tenants.json
```

This `shard-ring-pool/1` manifest includes `rings`, optional `aliases` and
`default_model`. Each ring records `ring_id`, `dir`, `head`, `tail`, measured
`deployment`, and `leases`. Deployment includes the full `model_cohort` and
uses `registration_policy: open`, `token_privacy: plain_ids|sealed_ids`.
Local lease specs carry ledger path/node ID/principal/lease ID/fence; remote specs
carry RPC address/node ID/controller sidecar-key path/lease ID/fence. The active
control plane must renew these leases; a static manifest is not a renewer.

The multi-ring queue shares tenant admission, fairness and idempotency globally.
Each ring has one serial worker; different rings can run concurrently. This is
not continuous batching. Every admitted request retains its selected ring,
cohort and tokenizer, including SSE reconnections. An alias switches only to an
already READY new cohort; old requests finish on their original backend.

## Drain, failure and cleanup

```text
PLANNED -> RESERVED -> LOADING -> WARMING -> READY -> DRAINING -> STOPPED
                                                  \-> FAILED -> verified recovery
```

New registrations do not reshuffle active rings. Drain blocks new bindings but
continues renewal until already bound requests finish. Physical resources are
released after backend shutdown and node-local process cleanup acknowledgement.
The process lifetime work handle includes loading and idle resident weights,
not just active requests. Remote controllers cannot end this local resident handle.

Expiry or revocation rejects new execution and stops the managed stage process.
The V4 CLI child also watches the local lease while idle. SQLite retains unknown
work after restart; absence from an in-memory process table is not proof of GPU
cleanup. Recovery requires local confirmation that the old execution stopped.

Transport retries replay the original request and verify the already published
prefix. Numeric/proof failures are not retried as transport errors. Receipt bundles
come from a complete final attempt, without mixing nonces from failed attempts.
Stopped ring history and worker threads are bounded/reclaimed during normal churn.

## Evidence and remaining deployment work

CPU tests cover real HTTP registration/lease RPC, persistent concurrency/fencing,
resident process cleanup, locality fallback, bounded 10,000-node discovery,
heterogeneous objectives, simultaneous HTTP jobs, version-sticky SSE and signed
V4 warmup/replay. Synthetic test contracts and CPU backends are explicitly labeled.

Signed offers establish who reported a capability; they are not remote hardware
attestation. Signed receipts bind participants and activation commitments; they
do not by themselves prove honest model computation. Existing local challenges
remain available; payments/anti-Sybil policy belong to the consuming network.
Open contribution does not conceal activations from computing participants.

Actual four/six RTX 5090 capacity, PCIe overlap and sustained throughput require
separate real-cluster evidence. Preserve the 4-card >=40 and 6-card >=30 valid
output tok/s acceptance targets and raw signed benchmark evidence. Future model
families require their own real backend and resource measurements.

Architectural references: [Petals routing implementation](https://github.com/bigscience-workshop/petals/blob/main/src/petals/client/routing/sequence_manager.py)
separates latency and throughput routing; [Parallax](https://arxiv.org/html/2509.26182v1)
uses regional model allocation and request-time selection. FlyAI uses its existing
ring/runtime and bounded local policies rather than claiming an exact reproduction
of either research system.

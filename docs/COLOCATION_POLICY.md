# Physical placement and co-location policy

Code audit: **2026-10-08, `c2ab623`**. Production permits colocated distinct GPUs;
historical distinct-host benchmark protocols remain separate.

Production inference allows different GPUs on the same physical host or subnet to contribute
adjacent layer ranges. The planner chooses stages using their actual resource capacity and the
measured communication mesh. Co-location by itself is not a reason to reject a feasible, faster
pipeline. A four-GPU server, a private cluster and a WAN swarm can all participate in the same
inference architecture.

This changes placement eligibility. It does not change activation encoding, transport
authentication, speculative acceptance, receipt signatures, nonce/coverage/chain verification,
boundary trust requirements or the model's layer ownership contract.

## Explicit isolation policies

An operator can request a stricter placement policy for a WAN experiment or a failure-domain
requirement. Isolation is a declared configuration rather than the default assumption that every
stage must occupy a separate residential network:

| Policy | Placement requirement | Suitable purpose |
|---|---|---|
| `none` | Allow repeated hosts and subnets; use distinct physical GPUs | Normal production inference and local multi-GPU deployments |
| `subnet` | Require different declared subnet groups for selected stages | A deliberately scattered network experiment or network-domain separation |
| `host` | Require different declared physical hosts for selected stages | Avoid sharing one machine's power, RAM, kernel and process failure domain |
| `adjacent_host` | Disallow the same declared host at adjacent cyclic ring positions, including tail/head | An explicitly requested adjacent-stage layout constraint |

Use `plan_ring(..., isolation="none")` for explicit production co-location. `plan_ring`'s
`isolation=None` resolves the model profile's `isolation` value, otherwise `"none"`.
`select_ring(..., isolation="none")` and `Scheduler.plan(..., isolation="none")` likewise default
to allowing co-location. Set `"host"`, `"subnet"` or `"adjacent_host"` explicitly when that
additional restriction is required. Unknown policy names are configuration errors.

`host` and `subnet` are different constraints. A subnet can contain many physical machines, and
one physical machine can have several interfaces or addresses. A strict policy needs the identity
and grouping evidence relevant to that policy; missing evidence does not establish independence.
Nodes without the corresponding host/subnet identity are filtered out of a strict policy; if the
remaining pool cannot form a feasible assignment, planning returns no plan. The `adjacent_host`
check preserves the cyclic tail/head comparison as an opt-in placement constraint even though the
live engine uses its existing coordinator-return channel. It does not change that channel.
An adjacent-stage restriction also does not imply that the entire deployment has distinct failure
domains. Use the global `host` policy when every selected stage must be on another host.

## GPU, host and network identities

Each GPU offer should identify its stable physical GPU UUID and its host's explicit identity.
Node metadata uses `host_id`, `gpu_uuid` (or `gpu_uuids` for an offer representing multiple devices),
and `subnet`; missing grouping fields are unknown. `public_ip` remains informational routing data.
Several offers with the same GPU UUID refer to one device and cannot supply independent capacity,
even if they use different node IDs, addresses or signing keys. Splitting one GPU's advertised
memory across several offers does not create additional VRAM.

A public IP is a routable address, not a physical host identity. Several unrelated machines can
share an address behind NAT. Conversely, one host can advertise several public/private addresses.
Do not turn `public_ip` into a host ID or infer a unique machine from a subnet label. Operator IDs
and signing keys likewise identify control/receipt attribution; they do not independently prove
distinct physical hardware or independent ownership.

Retain the provenance of the host/GPU grouping. A planner can enforce a declared grouping, while
physical attestation and Sybil resistance require additional evidence. Requiring different IPs,
subnets or neighboring hosts does not by itself supply that evidence.

## Shared resources must not be counted twice

VRAM belongs to the identified GPU. Host RAM, locked/pinnable RAM, SSD capacity and staging space
can be shared by all stage processes on a machine. Their reported free capacities are a single
host budget; they are not copied into an independent budget for every GPU offer.

For example, two distinct GPUs on a host with 128 GiB available RAM do not provide 256 GiB RAM.
If their proposed stages need 70 GiB and 65 GiB of host memory, that assignment exceeds the host's
128 GiB budget even when each offer separately advertises enough RAM for its own stage. Include
the host allocations for KV offload, expert pools, draft/MTP pools, pinned buffers and reserves in
the aggregate. A pinned subset is checked against the shared pinned limit and is not added to
total RAM again. Retained canonical expert pools are not assumed to be shared across processes.

The scalar RAM planner aggregates layer requirements by known `host_id` or declared
`memory_domain_id`. Exact-template planning sums actual host and pinned bytes
against a conservative domain capacity; the lease ledger repeats this reservation
check atomically. Supply budgets covering the allocations above;
co-location does not manufacture missing resource measurements. Shared SSD quotas and bandwidth
also need verification by the admission/deployment layer, rather than treating this isolation
policy as a complete disk or I/O resource scheduler.

`memory_domain_id` may identify a verified, separately allocated RAM partition, for example a
VM with its own memory quota. Without it, offers sharing a known `host_id` share one RAM budget.
Assigning different labels to the same unpartitioned pool does not provide independent capacity.

Offers on one host must report compatible resource evidence. The scheduler needs a conservative
single host limit or a verified allocation partition rather than summing contradictory free-memory
claims. GPU-local cache/graph/activation budgets remain per-device. Shared PCIe, CPU and memory
bandwidth can constrain aggregate service time and need their own measurements.

See [RESOURCE_CONTRACT.md](RESOURCE_CONTRACT.md) and [V4_HYBRID_RUNTIME.md](V4_HYBRID_RUNTIME.md)
for the resource components and canonical RAM expert pools.

## Communication uses reachable, measured paths

Same-host placement does not imply zero latency, infinite bandwidth, direct peer-to-peer access,
NVLink or a shared process. Use the measured RTT/bandwidth for the route the deployment can actually
reach. Loopback, private-network and existing local stage connections are valid choices when they
are reachable and preserve the existing authentication/transport contract.

Two stages using public addresses can encounter NAT hairpinning, firewall rules or an unnecessary
remote relay even while their GPUs share a host. Solve that as a route and reachability problem:
configure and measure an accessible local/private endpoint, or cost the public path that is
actually available. A blanket physical-host ban does not fix a bad route. The policy does not
invent local link costs or replace missing mesh measurements with an assumed fast path.
Edges are directed effective channels; alternative routes between one pair may
have different reachability, including same-IP hairpin failure. Version 2 records
the actual dialer/connect endpoint and one-way/RTT normalization. Legacy RTT stays
conservative RTT-as-hop. A valid chain need not form a bidirectional head-star,
and an external coordinator's entry/return legs are measured separately. See
[OPEN_INFERENCE_NETWORK.md](OPEN_INFERENCE_NETWORK.md).

## Inference placement and replica independence are separate

The stage pipeline is a cooperating computation. Each operator observes the inputs and activations
assigned to its stages under the current trust model. A host operating adjacent ranges observes
both ranges; the current protocol does not conceal those activations from that operator. Head and
tail trust/pinning rules still apply to roles that receive prompt IDs or produce output tokens.

Replication, redundant verification and high-availability recovery can require independent
operators, hosts, power domains or networks. Those requirements should be expressed for the
replicas or validators involved. A pipeline adjacency restriction alone neither proves replica
independence nor prevents one operator from registering several hosts or identities.

The existing receipts continue to bind assigned signers, jobs, nonces, contiguous layer coverage
and activation roots. Their validation remains required for co-located and scattered stages alike.
Allowing co-location does not relax these checks or convert receipts into hardware attestation.

When `box_ring_launch(..., receipts=True)` starts several GPU stages on one V4 box, it assigns
each a persistent key file `/root/.shard_node_key_gpu_<gpu-index>` and returns that path as
`key_path` in the stage specification. Register the corresponding public key for each stage;
one operator may own all of them. A single-GPU box retains the existing default key path.
`stage_launch_cmd(..., key_path=...)` can explicitly select another persistent key file.
Copying the same private key into several stage files still fails the unchanged duplicate-signer
validation.

## WAN benchmark evidence remains independent

The strict V4 WAN benchmark protocol in [V4_BENCHMARK.md](V4_BENCHMARK.md) still requires its frozen
four/six distinct-host RTX 5090 inventory and all existing identity, measurement, parity and signed
receipt evidence. Changing the production placement default does not rewrite that protocol or
retroactively verify any historical throughput claim.

A co-located run can be recorded as a local/multi-GPU experiment with its actual hardware and
network conditions. It must not be presented as a distinct-host WAN result. Likewise, explicitly
selecting an isolation policy is not evidence that a run occurred or met its speed target; retain
the actual deployment inventory, reachable-route measurements and raw receipts.

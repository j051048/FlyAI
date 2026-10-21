# Native resident request batches

`V4_NATIVE_BATCH=1` selects the separate
`v4-native-small-batch-greedy/2` numerical contract. The default remains the
legacy scalar lane path. Every selected stage and coordinator must select the
same contract in its calibrated model cohort; changing an environment variable
does not upgrade an existing calibrated deployment.

The implementation retains 2–4 independent KV, indexer and compressor states on
the stage device, plus one preallocated batch workspace. A cohort with a common
consumed position executes one actual `Stage.forward` on `[requests, 1, ...]`.
The reference attention and token-dependent hash routing see that request batch
dimension. Operator-level request-row math fixes projection, contraction,
elementwise and reduction shapes to the original scalar request shape. The
`v4-native-request-row-math/1` scope leaves the scalar oracle unchanged. This is
one stage batch, with bounded row operations inside it; it is not a loop over
complete scalar forwards and is not a fused grouped GEMM optimization.
Different consumed positions form separate cohorts; a singleton performs a
scalar native call and is counted as such. Arbitrary ragged attention is not
implemented.

Canonical slot allocations and the workspace retain their addresses across
steps. State gathers and commits are device-to-device copies, without a
per-token host KV round trip. Module state bindings and compressor aliases are
restored before a successful commit, and restoration is attempted on every failing call. A failure before commit
does not commit its workspace into canonical state. An alias-restore or
canonical device-copy failure poisons the batch runner: no affected frontier
is published, and the owned Stage must be restarted before reuse. `clear()` or
a new owner epoch cannot clear that fault. Scalar whole/island graphs
remain off. Separate optional batch graphs are described below; allocation
stability alone is not graph capture qualification.

Enablement requires `V4_NATIVE_LANES=2` (or 3 or 4), a positive
`V4_NATIVE_LANE_HOST_MIB` metadata quota, and a positive
`V4_NATIVE_LANE_DEVICE_MIB` quota covering **both** all canonical state slots and
the full batch workspace. The existing stage state is inventoried separately.
Reservations are validated before slot allocation. `V4_NATIVE_BATCH_REFERENCE=1`
is explicit CPU emulation for tests; it cannot be presented as CUDA evidence.
CUDA operation refuses the reference flag. Loaded MTP, layer KV, host-swapped
experts, durable prefixes and history-dependent reference bypasses remain
unsupported by this batch path. Original scalar whole/island graphs are
incompatible with this runner.

The host quota also reserves 2 MiB for bounded GPU-state hashing when publishing
an optional in-memory prefix. CUDA hashing is performed at publication, rather
than once per token. The resource report includes this working set separately
from canonical device state and batch workspace.

`V4_NATIVE_ASYNC_DEPTH` is bounded to 1–4 and cannot exceed the lane capacity.
The coordinator sends independent frontier cohorts before receiving the first
reply, up to this bound, allowing work to be in flight in the stage ring. Each
lane has at most one outstanding token. All submitted replies drain before
cancellation is observed or another lane transition occurs. Cancellation
suppresses output publication and the ordinary service release closes the
drained lane without retaining a prefix. Transport/proof failure closes the
session and discards local frontier/cache claims; a fresh authenticated owner
epoch clears node state. This is bounded socket pipelining, not a claim of ZOS
or communication/computation overlap measured on hardware.

Each row retains its own signed job binding, nonce, token ID, activation hash
and receipt chain. A signed batch envelope additionally binds row order.
The tail computes logits over the actual batch, then returns one greedy token
per authenticated row. Existing receipt coverage and prefix checks apply to
each request separately. The separate batch contract binds its request-row
math, resources and graph policy. Tiny CPU agreement does not qualify FP4 token
equality on a physical GPU.

`V4_NATIVE_PREFILL_QUANTUM` declares a 1–8 token prefill upper bound. Coordinator
`native_lanes.prefill_quantum` must fit every selected stage's bound. An explicit
`prefill_budget` backed by fresh bound offline measurements must select a
quantum above one; missing evidence keeps the effective quantum at one. Each
quantum executes ordinary one-token rounds, checks cancellation/deadlines after
draining, prioritizes decode cohorts, and stops each prefill request before an
additional decode output. No extra output token is hidden inside a quantum.
Quanta above one require an exact measured context/batch/quantum shape and a
common live frontier. Every request must have enough pure prefill tokens to
complete that whole quantum before its first output. Decode mixtures, ragged
frontiers and a shrinking batch fall back to one and re-plan at the next
scheduler boundary. A cancellation breaks the quantum after the drained round.

Runtime status exposes actual batched and singleton cohort counts, request row
counts, sent/received cohorts, peak in-flight cohorts and cancellations after
drain. These are execution counts, not throughput qualification. CPU tests
exercise native multi-request math through window/compression boundaries,
reordered rows, ragged rejection, slot reuse and local failure rollback. The
CUDA-marked gate requires an actual accelerator and must be run separately on
authorized idle hardware, together with full checkpoint numerical and receipt
qualification before release.

The CUDA test is
`tests/test_v4_native_batch.py::test_actual_cuda_resident_batch_preserves_independent_native_trajectories`,
parameterized for 2, 3 and 4 requests. It checks 21 consumed positions against
independent scalar CUDA stages, including generated tokens, top-5 indices,
logits/log probabilities and every KV/indexer/compressor state. The distinct
batch ABI uses explicit test tolerances (state/logit absolute 0.0625, relative
0.05; log-probability absolute 0.05, relative 0.01); greedy and top-5 token
indices must agree exactly. These tolerances describe the tiny native reference
CUDA gate and do not authorize the full FP4 checkpoint ABI. The separate B1–B4
partial-graph gate also compares bytes against an unchanged scalar oracle;
checkpoint diagnostics default to zero tolerance. A skipped gate provides no
qualification evidence.

## Optional partial batch graphs

Keep `V4_CUDA_GRAPH=0`. Select the separate path with
`V4_NATIVE_BATCH_GRAPH=partial`, an explicit `V4_NATIVE_BATCH_GRAPH_MIB` device
pool reservation, and `V4_NATIVE_BATCH_GRAPH_LIMIT` (2–4096 actual pre/post
graphs). The graph pool is additional to canonical state and batch workspace;
the host quota also covers bounded graph metadata. The default is off.

The scope is `attention_dense_partial`: dense/attention pre and dense post
segments use separate graphs, while routed MoE remains eager. The Indexer also
runs eagerly with its original exact-width native top-K selection; its ordered
indices enter a fixed graph input, padded only at the end with ignored `-1`
entries. This preserves ties instead of replacing native top-K with a stable
sort over a wider score tensor.

Keys distinguish B=1, 2, 3, 4, coarse context buckets and compression branches.
Position, hidden activations, token IDs and index lists are refreshed inputs.
The key is not one graph for every exact Indexer width. A finite graph count
and byte quota bound retained storage; there is no LRU eviction or unbounded
recapture. An exhausted key uses the identical eager twin and increments its
fallback counters. An actual byte overrun prevents committing canonical state.

Service pure-prefill requests bind `capture_graph=false` in the signed batch
envelope, so tokenwise prefill cannot populate every context bucket. The first
output-producing transition and subsequent decode may capture. This does not
fuse a full prefill or change prefill numerical
semantics. Static runtime policy and resource reservations remain constant;
actual capture, replay, Indexer and fallback observations are reported
separately. `graph_qualified=false` remains a configuration property; independent
hardware acceptance is separate evidence and must not change that frozen
policy. Requesting a mode does not qualify it.

Warmup and capture use owned non-default streams. Every exit joins the stream
and restores mutable workspace state before an eager retry. Failure to confirm
the join or restore is fatal to that step, rather than an eager fallback.
Immutable weight/rotary changes, stale aliases and workspace replacement are
rejected. Output is cloned before transport can retain it. The original
scalar whole/MoE-in-graph path remains independent and unchanged.

See [batch graph acceptance](V4_BATCH_GRAPH_ACCEPTANCE.md) for the three-way
scalar/eager-batch/partial-graph matrix and real-device gates.

`backend.export_native_prefill_samples(quota_ms=..., ttl_s=...)` groups the
bounded last 512 actual cohort timings and exports shapes with at least five
observations. Measurements cover coordinator submit through verified FIFO
drain, including network and queue time. To bootstrap time costs while running
the conservative one-token default, it can also observe consecutive spans of
2–8 real rounds. The span cost is the last verified drain minus the first actual
submit; it includes real CPU gaps between calls. It does not multiply a q1 mean
by q. Job-set/nonce digest, owner epoch, batch, consecutive frontiers and global
cohort sequence must all agree. A missing round, changed job/epoch or inserted
cohort prevents joining the span.

Both the exported artifact and its calibration config identify this scope as
`observed_consecutive_token_round_span_including_cpu_gaps`. It does not claim a
single `advance(q)` was observed. Timing decisions remain predictions: measure
the deployed behavior against the baseline before using it to claim a gain.
Export cannot establish another hardware/runtime binding or GPU qualification.

For the next independently authorized physical checkpoint test, use
`phase0/v4_batch_diagnostics.py`. It loads one owned Stage and reuses its weights
for both resident batch slots and the original scalar host-state oracle slots.
It never loads one weight copy per request or connects to a production ring.
The safetensors input contains frozen int64 `token_ids [B,N]` for B=2, 3 or 4.
A non-head stage additionally requires its original frozen
`hidden [B,N,hc_mult,dim]` in the stage dtype. These are teacher-forced inputs;
neither side feeds its own predicted token into the next step.

For example, after an operator allocates an idle device and a verified native
tail-stage artifact pack, the following command expresses explicit resource
bounds; its values must fit that actual stage and available device:

```text
python phase0/v4_batch_diagnostics.py --checkpoint CHECKPOINT --lo 40 --hi 43 --device cuda:0 --input frozen-batch.safetensors --out batch-report.json --host-mib 4096 --device-mib 24576 --execution-workspace-mib 1024 --max-tokens 256 --atol 0 --rtol 0
```

Start this command with service lane and scalar graph flags off. Checkpoint selection
is integrity verified; a header-only payload preflight precedes its one weight
load. Before allocating both state owners, the diagnostic checks original
module storage, oracle host states, resident slots/workspace, hashing, frozen
input copies, comparison/report memory and the declared execution workspace.
CUDA reports also record actual process allocator peaks. An underestimated
temporary allocation cannot become a passing quota observation.

For an independent partial-graph comparison, repeat the same frozen input and
resource bounds, adding `--batch-graph partial --graph-mib GRAPH_MIB
--graph-limit GRAPH_LIMIT` and a fresh output path. The device bound must cover
the additional graph reservation. Use `--batch-graph off` for the eager batch
baseline. These options do not turn the original scalar whole path into a
batch graph. Full commands and original scalar acceptance are in the
[batch graph acceptance guide](V4_BATCH_GRAPH_ACCEPTANCE.md).

The JSON binds verified catalog/model identity, all source hashes, runtime,
frozen inputs, span, numerical contracts, tolerances, actual GPU UUID/compute
capability/driver/Torch/CUDA identity, and comparison records. Every consumed
frontier compares readable window/compressed/indexer state and all recurrent
compressor buffers. It records first exact and tolerance failures, per-state
worst error, tail argmax/top-k logits/log probabilities and bounded comparison
digests. The `exact` fields compare original tensor bytes, including signed
zero; tolerance metrics retain their numerical meaning. Nonfinite outputs
cannot pass, while matching negative infinity is allowed only in compressor
score sentinels. CUDA quota checks include both allocated and reserved peaks.
Its `binding.batch_runtime.graph_policy` records the actual immutable
runner policy, while `batch_graphs` reports actual capture/replay/fallback
scope. A requested CUDA partial path with no observed capture cannot exit as a
passing graph diagnostic. Only a true model tail reports vocabulary predictions; other spans
explicitly leave vocabulary metrics unavailable.

Exit zero requires the specified tensor tolerance and unchanged tail greedy
and top-k IDs. Non-tail vocabulary metrics remain unavailable even if its
activation/state comparison passes. Zero tolerances test observed bitwise
agreement for these inputs. Nonzero tolerances bind a separate experimental
batch ABI; they never admit the old scalar ABI. A CPU `--fixture` run reports
fixture arithmetic and does not probe or qualify a GPU. A report always has
`release_qualified=false`; full pipeline receipt and frozen quality gates are
still required.

The batch report has its own bounded ceiling of 65536 row-position records,
with at most 16384 consumed positions per request and the loaded Stage horizon
as an additional bound. B4 no longer stops at the old aggregate 4096-position
limit. Original state, all report rows, input copies and execution/graph
workspaces still require explicit host/device quota preflight. Longer traces
can require substantial host report memory and runtime. This capacity is not a
claim that a real B4/8192 or later bucket trajectory has passed; short reset
runs cannot replace that missing recurrent history.

## Dated numerical evidence

On 2026-10-20 the `/1` CPU BF16 fixture retained greedy tokens on its frozen
17-position B4 trajectory, but first differed in top-five at position 14, row 2.
A final-layer KV difference reached 0.1796875 and exceeded the declared 0.0625
absolute/0.05 relative tolerance. That failure remains historical evidence.

On 2026-10-21 the `/2` request-row implementation fixes the same frozen
trajectory at zero absolute/relative tolerance, without altering its scalar
oracle. CPU tests additionally compare seeds 0, 7 and 23 at B2/B3/B4 over 37
positions, including window wrap, compression and discriminating Indexer
selection; output, logits/logprobs, ordered top-K and readable KV agree exactly
for those fixtures. Independent near-tie and nonfinite fault tests still fail
closed. These CPU observations do not establish actual GPU/FP4 equality, graph
capture, speed or long-context stability.

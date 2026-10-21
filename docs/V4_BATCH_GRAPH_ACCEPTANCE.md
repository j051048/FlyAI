# Native batch CUDA graph acceptance

This gate covers the separate `v4-native-small-batch-greedy/2` cohort. It does
not change the original scalar whole-layer, speculative or DSpark contracts.
Freeze the implementation and test source hashes again after this change;
older signed calibration and plans cannot qualify a new graph implementation.

Local CPU tests exercise arithmetic, dispatch, capacity and failure handling.
They cannot establish CUDA capture, FP4 correctness, GPU memory use or speed.
Keep the new graph path opt-in until the following real-device checks pass.

The supported capture scope is `attention_dense_partial`: attention and dense
pre/post segments use graphs; routed MoE and the native Indexer remain eager. `V4_NATIVE_BATCH_GRAPH`
selects this separate path. The scalar `V4_CUDA_GRAPH=whole` recipe remains a
different implementation and cannot select or qualify a batch bucket.

The signed capacity contains the immutable request-row math contract and a
`graph_policy`: mode, device pool bytes, host metadata bytes, graph-count cap,
capture scope and exact-width native top-K selection policy. Graph pools add to
the device reservation; host graph metadata adds to the host reservation.
Configuration has `graph_qualified=false`, including on a real device. Actual
capture/replay observations belong in the execution report and never mutate
the admitted runtime signature.

The Indexer selects top-K at its original exact read width outside capture.
Its selected order is preserved and padded only at the tail with ignored `-1`
indices before entering a fixed graph input. Graph keys use the request count,
coarse context bucket and compression branch, rather than one retained graph
per exact read width. Position and indices are refreshed at every replay.
The status explicitly reports `indexer=eager_native_exact_width` and
`indexer_eager=true`; this is not a whole-layer/Indexer graph.

Service rows sign `capture_graph` and cohorts separate prefill from decode even
at a common frontier. Pure prefill sends `false`, which suppresses new capture
and executes the eager twin; the first output-producing transition and later
decode may capture. Count and byte limits are finite,
without graph LRU eviction. Byte violations prevent canonical state commit.

## Independent comparison

Use the original scalar stage as the oracle. Teacher-force the same frozen
token IDs and, on a non-head stage, the same frozen input activations. Never
modify the scalar oracle, increase tolerances, ignore top-K disagreement or
truncate readable KV solely to make a failing batch report pass.

`v4-native-request-row-math/1` fixes projection/contraction reductions to the
scalar request-row shape inside the batch domain. It preserves native packed
weights and their kernels, while the out-of-domain scalar oracle stays
unchanged. It is a correctness strategy, not a grouped GEMM speed claim. Compare
performance after capture on the actual workload; correctness alone does not
show that batching is faster.

Run B=2, B=3 and B=4 separately. For each case retain the complete output,
logit/logprob and readable state comparison. Compare the window, compressed KV,
indexer KV and both recurrent compressor buffers at every consumed frontier.
Only the model tail can report actual vocabulary top-K and greedy equality.
Record the first divergent position and row, even if later tokens agree.

Capture must be checked against its own eager batch twin as well as against
the scalar oracle. Passing graph/eager equivalence alone cannot establish
scalar invariance: both batch paths could share the same arithmetic bug.

## Commands for an idle rental

These commands are for later hardware acceptance. Run them from the frozen
source checkout in an isolated environment. They do not rent, install, SSH,
fetch weights or push GitHub. Preserve the source archive and first failures.

First retain the original scalar `resident-whole` baseline and its four decode
modes using the complete [5090 guide](V4_5090_ACCEPTANCE.md). For an independently
configured eight-card rental, the existing entry points are:

```bash
python tools/v4_5090_acceptance.py plan \
  --config /data/flyai-acceptance/config.json \
  --out /data/flyai-acceptance/scalar-plan.json
python tools/v4_5090_acceptance.py run \
  --plan /data/flyai-acceptance/scalar-plan.json \
  --run-dir /data/flyai-evidence/scalar-20261021-01 \
  --confirm-gpus 0,1,2,3,4,5,6,7 --execute
```

The configured roles and exclusive GPU list must agree. This scalar campaign
does not benchmark the new batch cohort. Afterwards reserve one idle card for
the following local CUDA gates and diagnostic comparisons. Set
`IDLE_PHYSICAL_GPU`, `CHECKPOINT`, `INPUTS`, `EVIDENCE`, `HOST_MIB`, `DEVICE_MIB`,
`EXECUTION_MIB`, `GRAPH_MIB`, `GRAPH_LIMIT` and `CONTEXT` to the actual deployment.
Create a fresh evidence directory outside the source tree. Size all quotas from
the real stage inventory; adding a graph reservation cannot overcommit the
device. This example owns the tail layers 40:43 and therefore also needs frozen
original input activations in each safetensors file.

```bash
(
  set -eu
  export CUDA_VISIBLE_DEVICES="$IDLE_PHYSICAL_GPU"
  export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  export V4_CUDA_GRAPH=0 V4_NATIVE_LANES=0 V4_NATIVE_BATCH=0
  export V4_NATIVE_BATCH_REFERENCE=0 V4_NATIVE_LANE_DEVICE_MIB=0
  export V4_NATIVE_BATCH_GRAPH=off V4_NATIVE_BATCH_GRAPH_MIB=0

  python phase0/v4_release.py gpu-check \
    --junit "$EVIDENCE/gpu.junit.xml" --out "$EVIDENCE/gpu-gate.json"

  for B in 2 3 4; do
    python phase0/v4_batch_diagnostics.py \
      --checkpoint "$CHECKPOINT" --lo 40 --hi 43 --device cuda:0 \
      --input "$INPUTS/frozen-B${B}.safetensors" \
      --out "$EVIDENCE/eager-B${B}.json" \
      --host-mib "$HOST_MIB" --device-mib "$DEVICE_MIB" \
      --execution-workspace-mib "$EXECUTION_MIB" --max-tokens "$CONTEXT" \
      --batch-graph off --atol 0 --rtol 0

    python phase0/v4_batch_diagnostics.py \
      --checkpoint "$CHECKPOINT" --lo 40 --hi 43 --device cuda:0 \
      --input "$INPUTS/frozen-B${B}.safetensors" \
      --out "$EVIDENCE/partial-B${B}.json" \
      --host-mib "$HOST_MIB" --device-mib "$DEVICE_MIB" \
      --execution-workspace-mib "$EXECUTION_MIB" --max-tokens "$CONTEXT" \
      --batch-graph partial --graph-mib "$GRAPH_MIB" \
      --graph-limit "$GRAPH_LIMIT" --atol 0 --rtol 0
  done
)
```

Check every exit code and archive partial reports before proceeding. The
commands use the same frozen inputs and zero tolerances. A deliberately
different nonzero tolerance is a separately frozen experiment, not a way to
overwrite the zero-tolerance result. B1 is included in the real graph unit
gate; the checkpoint diagnostic accepts B2–B4. Repeat for the other owned
spans and deployed context capacities, retaining actual predecessor-produced
hidden inputs for non-head stages. Never fabricate tail logits for a non-tail
span.

The batch diagnostic has an independent limit of 65536 row-position records
and at most 16384 consumed positions per request, further bounded by the loaded
horizon. B4 can therefore cross the old aggregate 4096-position limit. Its
host/report memory remains explicitly reserved before allocation, in addition
to original state, frozen inputs and execution/graph workspaces. Supply real
budgets for the longer trace; this is a deliberately slow diagnostic.
The CLI separately caps its frozen safetensors input at 128 MiB plus header.
For a non-head stage, full hidden activations can reach this limit before the
row-position limit. Such a refusal is unverified coverage; do not replace the
history with independent reset runs. Larger already-owned frozen tensors need
a separately archived harness using `compare_native_batch` and its explicit
host quota, rather than bypassing the CLI limit silently.
Configuring capacity 8192 does not prove an actual B4/8192 or later bucket
trajectory has passed. Splitting the input into fresh reset runs cannot
substitute for missing recurrent history. The original scalar whole 4104-token
gate retains its own separate coverage.

In a graph report inspect `binding.batch_runtime.graph_policy`, `batch_graphs`
and `qualification.graph_capture_observed`. An actual CUDA partial request
with no observed capture/replay cannot pass as a graph diagnostic. Partial
capture with explicit eager/fallback counters is still partial; it does not
qualify full graph coverage. Every diagnostic retains
`qualification.release_qualified=false`. Signed service, strict ring, quality,
cancel/reconnect and sustained performance gates are additional requirements.

For service A/B runs, use a newly calibrated `/2` model cohort and separate
frozen runtime plans for eager and partial. Eager has `V4_NATIVE_BATCH_GRAPH=off`
and zero graph MIB. Partial has `V4_NATIVE_BATCH_GRAPH=partial`, positive
`V4_NATIVE_BATCH_GRAPH_MIB` and bounded `V4_NATIVE_BATCH_GRAPH_LIMIT`; keep
`V4_CUDA_GRAPH=0` in both. Preserve per-request outputs, receipts, TTFT, token
gaps, aggregate throughput, capture-suppressed prefill and decode replay counts.
Do not change flags inside a running process or reuse the scalar plan.

## Required CUDA matrix

Use an idle, operator-owned 5090 with the original checkpoint format. Synthetic
native-format kernel tests supplement, and cannot replace, this checkpoint
matrix. Record GPU UUID, SM version, driver, CUDA, Torch, kernel backend,
checkpoint/catalog/tokenizer hashes, recipe and all executed source hashes.

| Dimension | Required observations |
| --- | --- |
| Request count | B=1, 2, 3, 4, plus order changes and shrink/grow after cancellation |
| Freshness | Change hidden values and token IDs between successive replays; result must follow the new input |
| Position | Position zero, ordinary decode, window wrap, every owned compression ratio and indexer bucket boundary |
| State | Exact committed frontier, compressor/indexer aliases and fixed storage addresses after capture/replay |
| Ownership | Lane permutation, cancelled lane, fresh slot reuse, tenant/epoch change and injected forward failure |
| Graph dispatch | Actual captured/replayed counts, capture failures, eager fallbacks and unsupported positions |
| Resources | Declared state/graph quotas and actual allocated/reserved CUDA peaks before and after capture |
| Quality | Tail greedy and ordered top-K IDs, logits, teacher-forced logprobs and all readable KV comparisons |

Exercise both sides of compression transitions and enough tokens to wrap the
window. A tiny test's compression ratios do not qualify the checkpoint's
ratio-128 compressor or its long-context indexer buckets. Repeat the matrix at
the deployed context capacity; a short-context pass does not qualify 8192.

The dispatch report must identify its actual capture scope. A partial graph
that executes attention, compressor or MoE eagerly must retain that scope and
fallback counters; it cannot report whole-stage or whole-layer capture. B1
graphs cannot be relabelled B4. Padded inactive rows must never mutate another
request's canonical state or enter per-request receipts.

## Failure and release rules

1. Keep each failing report and its frozen input, source inventory and logs.
   A skipped, unsupported, absent or CPU-emulated case is unverified.
2. A failed capture/forward attempt must not commit workspace KV or advance any lane. Before
   retrying eagerly, restore every mutable workspace buffer and alias. Reject
   stale capture bindings after storage/model/runtime changes.
3. Confirm actual replay counters on the target path. Merely requesting graph
   mode, allocating static tensors or returning an eager result is not capture
   evidence. Do not use diagnostics or an eager warmup as a speed result.
4. Re-run the original scalar whole/rollback gate and strict single-machine
   multi-GPU ring, then cancellation/reset/reconnect and a sustained soak.
5. Measure cold and warm runs separately. Report committed tokens per request,
   aggregate request/token throughput and P50/P95 latency separately. Keep the
   same prompts, token count, sampling and compiler warmup for the A/B run.

An unsafe capture stream/join/restore, alias-restore or canonical device-copy failure is a different class of
fault: the runner is poisoned as `batch_fault`. It must not publish a frontier
for the failed transaction or accept another step. Device copies cannot be
treated as an atomic multi-request commit after an error. Restart the owned
Stage and re-establish its authenticated bindings; `clear()` or a new epoch
must not turn uncertain storage into a reusable state. Keep the failed job and
cleanup evidence before restarting only the processes owned by this run.

Archive raw JUnit, diagnostic JSON, frozen inputs, signed ring receipts,
runtime/source bindings and allocator/timeline evidence before returning the
rental. Exclude private keys and tenant prompt caches. Source publication was
explicitly authorized after local regression. Real-device release qualification
remains pending, and the new batch/graph paths remain opt-in.

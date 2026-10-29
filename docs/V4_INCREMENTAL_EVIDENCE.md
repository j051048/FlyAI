# Incremental prefix: collect native evidence before enabling reuse

Date: 2026-10-29. This guide covers the optional **resident scalar greedy**
qualification. The producer and loader reject `spec`, `dspark`, `pipelined`,
nonzero temperature, loaded MTP, service lanes and KV offload. The existing
four-mode ring acceptance remains required independently. No GPU result is
claimed by this guide or by the CPU reference tests.

## Evidence and identity

Use one frozen source bundle, checkpoint, model cohort, tokenizer/template,
tenant binding, owner epoch, lease fences, reset execution and stage assignment.
Changing any of these needs a new campaign. `max_pos` is the exact reset horizon;
it is not inferred from the current prefix. A qualification from one horizon
does not authorize another.

The campaign spec is JSON with exactly these fields:

```text
schema = "v4-incremental-prefix-campaign/1"
binding = the coordinator's actual ConversationIdentity before execution binding
source_sha256 = digest(v4_observability.source_inventory()) from the frozen bundle
execution = {max_pos, spec:false, dspark:false, pipelined:false, temp:0.0, seed}
max_context = largest tested prefix, no greater than the loaded horizon
max_suffix = largest tested append length
parent_alignment = the tested parent alignment
```

Obtain the binding and reset values from the actual coordinator. Do not invent
checkpoint/cohort/runtime identifiers or copy the CPU test fixture identifiers.

The pinned evidence JSON starts with:

```text
schema = "v4-incremental-prefix-evidence/1"
coordinator_pubkey = the public key pinned by the strict plan
stages = ordered [{stage, node_id, pubkey, lo, hi,
                  runtime_config_sha256, gpu_uuid}]
strict_canaries = []
native_cases = []
```

The stage ranges must tile the model. Every public key, GPU UUID and runtime
configuration digest comes from the actual strict deployment. The collector
verifies all signatures and assignment identities. Signatures identify who
reported measurements; they are not remote GPU computation proofs.

## Frozen per-stage inputs

Prepare a safetensors file from the same frozen target execution. It contains:

| Tensor | Type and shape | Meaning |
| --- | --- | --- |
| `token_ids` | CPU int64 `[1,N]` | Teacher-forced target tokens. |
| `other_token_ids` | CPU int64 `[1,N]` | A distinct valid negative-control sequence. |
| `hidden` | Native Stage dtype `[1,N,hc_mult,dim]` | Required only for a non-head Stage, its frozen incoming hidden states. |
| `other_hidden` | Same type/shape | Required for the non-head negative-control sequence. |

Use inputs crossing an owned compression boundary. The producer records each
owned layer's compression ratio. A Stage with no compressor has no compressed
state branch to exercise. Neither the collector nor the producer converts a
failed or uncovered compression case into a pass.

## Run the owned native cases

The following command runs in its own process on an explicitly available GPU.
It loads one assigned Stage and reuses its weights sequentially. Repeat it for
every Stage with that Stage's actual range, inputs and GPU identity. Keep the
same engine environment as its strict worker; a runtime digest mismatch is a
refusal, not a reason to edit the pin.

```bash
CUDA_VISIBLE_DEVICES="$OWNED_GPU" python phase0/v4_incremental_case_probe.py \
  --spec "$CAMPAIGN/spec.json" --evidence "$CAMPAIGN/pins.json" \
  --checkpoint "$CHECKPOINT" --input "$INPUTS/stage-$STAGE.safetensors" \
  --stage-index "$STAGE" --lo "$LO" --hi "$HI" \
  --parent-count "$PARENT" --max-seq "$HORIZON" --device cuda:0 \
  --host-mib "$HOST_MIB" --device-mib "$DEVICE_MIB" \
  --key-file "$PRIVATE_COORDINATOR_KEY" --confirm-gpu \
  --out-dir "$EVIDENCE/native-stage-$STAGE"
```

The coordinator private key is read locally for signing and is never included
in output. Store it outside the source/evidence bundle with its normal private
permissions. Output directories must be new.

The producer executes full-prefill reference versus parent/suffix execution,
compression crossing, actual native `_seek` rollback, cancellation after actual
work followed by reset/restore, independent reset, and an unrelated sequence
between a parent snapshot and its continuation. It hashes actual readable KV,
Indexer and Compressor state and last outputs. A non-tail Stage compares its
last hidden output; a tail compares native head logits and greedy token.

`report.json` contains all comparisons and the first differing state key. Only
actual CUDA/TileLang runs with matching UUID, source, checkpoint and runtime
identity can emit signed `native_cases`. Only byte-equal cases are signed. A
mismatch returns exit code 2 and leaves the campaign pending; do not loosen the
tolerance or edit hashes. CPU `--fixture --device cpu` checks the producer flow
and emits **no signed GPU cases**, even when its arithmetic comparisons pass.

These are owned numerical lifecycle probes. They do not replace the whole-ring
transport cancellation, owner-disconnect and four-mode tests.

## Collect the strict append canary

Run a strict greedy parent request followed by its appended request with
`extended_shadow` enabled. Retain the original coordinator-signed
`shadow_begin` and `shadow_compare` controls, every original stage-signed compare
ACK, and the **same appended job's** complete signed receipts. The coordinator
records bounded controls/ACKs as `incremental_canary_evidence` when a shadow
comparison finishes. The receipts come from that request's final result.

Wrap the unmodified fields in a JSON file:

```text
{
  "schema": "v4-incremental-strict-canary/1",
  "canary": {
    "begin_request": original signed shadow_begin,
    "compare_request": original signed shadow_compare,
    "acks": original ordered signed comparison ACKs,
    "receipts": same-job complete signed receipts
  }
}
```

The collector checks the same job/nonce/transaction/prefix/execution, all-stage
comparison success, the BF16 activation chain, GPU UUIDs, TileLang/CUDA versions,
source inventory digest, runtime configuration digests and pinned signers.
Unsigned summaries, CPU observations, missing stages and another job's receipts
are rejected.

## Build the loadable policy

```bash
python phase0/v4_incremental_qualification.py \
  --spec "$CAMPAIGN/spec.json" --evidence "$CAMPAIGN/pins.json" \
  --canary-report "$EVIDENCE/strict-canary.json" \
  --case-report "$EVIDENCE/native-stage-0/report.json" \
  --case-report "$EVIDENCE/native-stage-1/report.json" \
  --out-dir "$EVIDENCE/qualification"
```

Repeat `--case-report` for all assigned Stages. The collector writes
`qualification.json` only when the strict GPU canary and all six native cases
per Stage verify. Otherwise it writes only `report.json`, lists missing cases
and exits 2. Do not pass the pending report to the service as a qualification.

Pass the resulting complete JSON object as the coordinator's
`conversation_cache.incremental_qualification`, with
`incremental_shadow_every` in `1..64`. The serving loader rechecks the embedded
evidence and its content digest. It rejects a hand-written `gpu_verified:true`
or `cases:true` object without complete signed evidence.

After loading, the first eligible request still performs a fresh full-prefill
canary. At most `incremental_shadow_every - 1` eligible reuses occur between
canaries. A failed canary disables reuse for that policy. Source, tenant,
execution, horizon and geometry mismatches keep the ordinary reference path;
an EOS suffix result also uses actual prefill so a one-token result keeps a
real nonzero-work receipt.

Record reused prefix tokens, newly computed tokens, restore time, full-prefill
time, TTFT and cancellation recovery. Numerical qualification does not establish
a speed gain. Qualify or recommend this path only after the matching native and
whole-ring hardware acceptance succeeds.

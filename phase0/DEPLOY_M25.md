# MiniMax-M2.5 compatibility deployment and validation

Aligned with `c2ab623`, 2026-10-08. M2.5 remains an independent engine under `engines/minimax_m25/`; this is its legacy operator-run ring guide. It does not use the new GPT-OSS `shard-pipeline-session/1` production contract or automatically share the V4/GPT-OSS authenticated HTTP service. For those paths use [GPT-OSS production](../docs/GPT_OSS_PRODUCTION.md) and [V4 current phase](../docs/V4_NEXT_PHASE.md).

The June–August receipts are historical hardware evidence, not a new validation of current dependencies or the present fleet. The selected 2026-10-08 local suite (985 passed, 3 skipped) is non-GPU and not whole-repository CI.

## Deployment scope and process ownership

`engines/minimax_m25/m25_scatter_pipe.py` is an old dedicated-box operator harness. It assumes `/root`, configured rental-provider access and its own fixed sidecar/engine ports. **It still kills GPU compute PIDs and stops sidecar/port processes during bring-up. Do not use it on shared hosts or as the open-network lease launcher.** The safer owned-process changes to `phase0/launch_oss.py`, `launch_ngram.py` and `launch_libp2p.py` do not cover this M2.5 harness.

A manual supervisor can use `shard.managed_launch` to retain a pre-spawn reservation, stop only its own child, confirm terminate/kill completion and explicitly recover an orphan. That helper does not reserve GPU/RAM. Open-network resource lifetime must use the node-local lease ledger through loading, idle residency and confirmed runtime exit.

For a historical distinct-host WAN experiment, retain independently inspected host/GPU identity and actual edge RTTs. Different public IPs alone do not prove different physical machines; same-host distinct GPUs are valid production placements when their shared budgets are accounted by the applicable planner/lease path.

## Prepare the actual hardware and dependencies

- Inspect GPU model/UUID, driver/CUDA, the installed vLLM/FlashInfer backend and actual stage peak/KV capacity. M2.5's NVFP4/CUTLASS and non-Blackwell Marlin paths have different limitations; do not reuse one measured template as another GPU's calibration.
- Retain versions from the hardware run being reproduced. The 2026-06-26 operator pass found a CUDA-13 vLLM binary could not load on a CUDA-12.8/R570 host. That is a dated compatibility finding, not a blanket rule to install today's unpinned `pip install vllm`.
- Use bounded download/load/health deadlines, verify the files deployed and avoid blanket manual `pkill`, `fuser` or GPU-PID cleanup. A timed-out owned process must be confirmed stopped before its resources are declared free.
- Prepare current [sidecar](../sidecar/README.md), loopback engine listeners, peer addresses, relay paths if required, and intended inbound `-allow` identities. QUIC additionally needs its UDP port reachable.

## Weights and identities

The manual `m25_pull_range.py` selects safetensors shards for `[lo, hi)`, with `--head` adding embeddings/tokenizer and `--tail` adding norm/head. Its current CLI supports `--repo` and `--dir`, but does **not** expose a pinned `--revision` or produce the new `.shard-download.json`. It must not be described as having the GPT-OSS full-content download contract.

For content-verified assigned block fetching, use the existing signed publisher manifest and `shard.fetch` contract with a trusted publisher pin, expected model/layer metadata and CID-verified payloads. See [INTEGRATION.md](../docs/INTEGRATION.md). Do not replace that independent publisher pin with one supplied by an untrusted assignment.

Keep existing node/receipt keys outside source control. `SHARD_RECEIPTS=1` enables the stage receipt path. `SHARD_SWARM_TOKEN` is M2.5's shared per-ring/epoch greeting authorization: stage/return peers greet explicitly, missing or wrong tokens fail when the feature is configured; unset retains legacy behavior. It is not the strict plan-bound nonce-signature protocol and is not a replacement for sidecar peer allowlists.

## Existing dedicated-box CLI

Run from a complete checkout. The following is the existing rental-provider harness syntax; substitute your already prepared **dedicated** instances and layer ranges. `--order` entries use its `region:iid:lo:hi` format, cover all 62 layers and preserve head/tail roles.

```sh
python engines/minimax_m25/m25_scatter_pipe.py \
  --order REGION:INSTANCE:LO:HI REGION:INSTANCE:LO:HI \
  --K 8 --depth 4 --max-ctx 8192 --kv-maxlen 8192 --receipts --validate
```

`--warm-only` stops after stages/sidecars warm; `--serve` starts the old M2.5 gateway on the head instead of a one-shot coordinator job. The historical local gateway port is `127.0.0.1:18000`; tunnel it through an authenticated SSH connection. This gateway does not inherit `shard/http_gateway.py`'s tenant/TLS contract; an Internet-facing deployment needs a separately authenticated protective frontend.

```sh
ssh -N -L 8000:127.0.0.1:18000 USER@HEAD
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"minimax-m2.5","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Check `/health` separately from `/ready`: process liveness and cached/probed ring readiness do not replace a successful signed end-to-end request. `M25_GATEWAY_MOCK=1` tests HTTP behavior only; it is not inference evidence.

## Context, graphs and batching

`M25_KV_MAXLEN` defaults to 40960 in the current stage. The launcher/gateway negotiate the minimum of operator context and stage KV capacities, with speculative headroom. `M25_MAX_POS`/RoPE size alone does not make that KV budget fit. Bound prompt plus completion and verification headroom before admission; increasing a limit needs measured GPU memory, especially when `M25_BATCH>1` multiplies cache storage.

Stage `M25_CUDA_GRAPH` defaults off when directly launched; enabling it also activates static KV. Operator `eng_env()` and per-stage `graph_off` can alter the effective setting, so record the actual launch environment. Graph caches have capture-count/memory guards; unsupported new shapes can run eager. The 2026-06-28 production graph experiment and later graph/aux/batch work are distinct histories, not a permanent declaration that every graph route is slow or unvalidated.

Capture and validate the actual verify shape (`K+1`, batch/tree/context bucket), current position writes, rollback and aux state. Compare eager/graph stage tensors, state and end-to-end greedy tokens on the same hardware. Token-count-dependent quantized MoE behavior means neither batching nor changing K is automatically bit-invariant. `M25_BATCH_MOE` remains an explicit numerical/performance choice; do not use a different shape's result as the control.

The gateway serializes ring jobs under `RING_LOCK`; `M25_GW_BATCH` may combine streams into a job. This is no longer accurately described as an unconditional single-stream-only gateway, and it is not the same scheduling interface as the new multi-ring service. Content-specific K and graph capture set also affect both acceptance and memory.

## Correctness and acceptance

- Validate real tool output and required/named `tool_choice`, multi-turn recall and the intended long-context prompt. Keep prompt/tokenizer/backend identities and completed output IDs.
- The gateway is greedy: non-greedy `temperature`, `top_p` or unsupported `top_k` values are rejected, not silently accepted. Separate speculative sampling experiments do not enable it here.
- Confirm generation cap and earliest EOS on streaming/final paths, receipt signatures, assigned layer coverage and the injected job nonce. Receipt validity authenticates declarations, not all GPU math.
- Sweep K/depth on the actual workload, preserving shape-matched controls. `NGRAM_MARGIN=auto` is the current default adaptive policy; `fixed:64` or a legacy integer explicitly fixes it. There is no universally safe K/depth derived from the old `margin=256` sentence.
- Separate copy/retrieval from novel/code/long-context results; retain cold/warm scope and actual timing semantics. GPT-OSS's new committed-token accounting is not silently retrofitted to all historical M2.5 metrics.
- Test bounded transport failure and the intended healer. The engine's `resume_ids/resumable` primitive does not by itself prove transparent durable gateway or arbitrary-ring recovery.

Historical records include [2026-06-28 usability](../docs/receipts/m25-usability-20260628.json), [static MoE/graph work](../docs/receipts/m25-graph-moe-static-20260628.json) and [production graph A/B](../docs/receipts/m25-cudagraph-production-20260628.json). Their reported 28.7k prefill and short/long-context throughput are scoped to those original runs.

## Privacy boundary

libp2p authenticates and encrypts the remote links; plaintext engine/sidecar endpoints remain local trust boundaries. M2.5 middle nodes see activations, and some aux/batched paths forward `tids` for later processing. Absence of text in a simple hidden-state frame does not prove tokens cannot be inferred. Only the separate V4 sealed-ID mode provides the documented raw-ID restriction; it still exposes activations. See [PROOF.md](../docs/PROOF.md) and [V4_TRUST_BOUNDARIES.md](../docs/V4_TRUST_BOUNDARIES.md).

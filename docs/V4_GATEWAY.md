# Authenticated V4 serial service

The shared HTTP implementation is `shard/http_gateway.py`; the V4 backend lives in
`engines/deepseek_v4/v4_gateway.py`. It exposes an existing V4 ring as a bounded text-chat service.
It directly calls the current greedy, serial DSpark and pipelined DSpark coordinators. It does
not launch/rent GPU nodes, download models, provide continuous batching or claim durable HA.
The single-ring dispatcher owns one ring. `--ring-pool` now runs one serial worker
per verified leased ring, sharing tenant quotas and idempotency across the pool.
The complete open-contribution workflow, renewal ownership and rolling-version
routing are documented in [OPEN_INFERENCE_NETWORK.md](OPEN_INFERENCE_NETWORK.md).

## Current managed entrypoint (`c2ab623`)

For managed production use the open-network adapter. It forms leased rings from signed offers
and exact stage calibrations, injects a signed pipeline plan and controller key, starts approved
local stage templates, and requires successful signed warmup:

```sh
python engines/deepseek_v4/v4_network_service.py \
  --config network.json --auth-file /run/private/v4-gateway-auth.json \
  --host 127.0.0.1 --port 8000
```

`network.json` uses `shard-open-network/1`; see [OPEN_INFERENCE_NETWORK.md](OPEN_INFERENCE_NETWORK.md).
HTTP bearer keys authenticate clients. The separate controller Ed25519 key authenticates engine
HELLO and node lease RPC. Stage public keys, exact cohort, layer ranges and controller public key
are pinned in `shard-pipeline-plan/1`. Forward and return bind to one signed head owner grant;
an occupied head replies BUSY. Idle head-control pings renew ownership without reading tail replies.
Strict V4 routes must reach the actual tail listener. The old `--ret-relay` ingress is refused
in strict mode; it is not a transparent production fallback.

## Existing-ring compatibility adapter

The standalone single-ring and `--ring-pool` CLI loaders retain measured deployment checks,
but at `c2ab623` do not inject `pipeline_plan`/`coordinator_key` into their backend. They require
explicitly legacy stage listeners and are not the current strict managed deployment recipe.
`check_deployment()` still checks the resource contracts, runtime
configuration, shared RAM/pinned budgets, GPU/port identities and concurrent host-I/O evidence.
The served model/layer count, signer assignments, stage context limits, optional `coord_env`,
and sealed-token mode/key identity must match that bundle. Public bundles must not contain
swarm secrets or private key-file configuration.

Compatibility example on a privately controlled ring started with `--legacy-protocol`:

```sh
python engines/deepseek_v4/v4_gateway.py \
  --dir /models/v4 --deployment measured-deployment.json \
  --auth-file /run/private/v4-gateway-auth.json \
  --head 127.0.0.1:29610 --tail 127.0.0.1:29612 \
  --mode pipelined --max-context 8192 --warmup-timeout-s 300
```

The compatibility bundle supplies the pinned signer/layer map. Optional `--assignments` cross-checks a separately
retained map; it cannot replace or contradict the deployment gate. Stage processes must enable
receipts and support per-job reply binding before this adapter starts. Reply binding is separate
from the newer authenticated connection HELLO.

Startup performs a real two-token request through the chosen coordinator mode, with a fresh nonce
and full signature/signer/job/swarm/coverage/chain validation. It does not consume tenant quota.
Warmup failure prevents default service startup. `--skip-warmup` explicitly permits a diagnostic
listener, but readiness remains false until a complete signed request succeeds. New TCP connections
alone never establish readiness. Once verified, the same live established channels retain readiness
while idle; a nonblocking peek checks EOF/errors without consuming any pending bytes. Active
attempts own their readers, so readiness never changes their socket mode or steals replies.
Closed channels or execution faults clear the signed-success state. Readiness is evidence of a
verified connected ring, not a guarantee that every future GPU kernel will complete; deadlines
and active-I/O fault handling remain necessary.

Bind defaults to loopback. A public bind requires a TLS certificate/key or an explicit
`--allow-insecure-http` acknowledgement that a trusted TLS proxy terminates the public connection.
TLS handshakes, header/body reads and stream writes have deadlines; handler concurrency and request
body size are bounded. Bearer credentials are checked before request processing and never logged.

Authentication configuration shape:

```json
{
  "keys": {"replace-with-a-random-private-256-bit-key": "customer-a"},
  "tenants": {
    "customer-a": {
      "max_active": 4,
      "requests_per_minute": 30,
      "tokens_per_minute": 65536
    }
  }
}
```

Keep this file private. Several keys mapped to one tenant share its limits. The token quota reserves
prompt length plus maximum completion length in a rolling minute, including accepted jobs that later
fail/cancel; it is a resource admission limit, not a billing invoice. Idempotent repeats do not
consume another reservation. Queue count, queued token budget and retained job count are also bounded.
Tenant queues rotate fairly, preserving FIFO within each tenant; an already running long request
is not token-preempted, so cancellation/deadlines bound its occupancy.

## HTTP contract

All data/control endpoints require `Authorization: Bearer ...`; only `/health` is public and it
returns process liveness without prompts, tokens, tenant identities or job details.

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Text chat, JSON or SSE streaming |
| `GET /v1/models` | Served model identity |
| `GET /ready` | Verified live established ring, not fresh port availability |
| `GET /metrics` | Bounded service/ring counts and the caller's active-job count |
| `GET /v1/jobs/<id>` | Caller-owned job status and committed count |
| `POST /v1/jobs/<id>/cancel` | Cancel only a caller-owned job |
| `GET /v1/jobs/<id>/receipts` | Final caller-owned signed receipt set; unsettled jobs return 409 |

The current API supports text system/user/assistant messages, greedy sampling, streaming,
`max_tokens`/`max_completion_tokens`, `thinking`, `reasoning_effort`, `timeout_s` and `shard_mode`.
Non-greedy sampling, multiple completions, penalties, custom stop strings, constrained JSON output,
vision inputs and structured tool-call adaptation are rejected rather than silently ignored.
Thought/tool markup generated as ordinary model text is not transformed into structured OpenAI
reasoning/tool-call fields by this version.

`Idempotency-Key` identifies one logical generation within a tenant. Another payload under the same
key returns 409; the same payload returns its retained job/result without another GPU execution.
Idempotency is in-memory, bounded and currently retained for up to 600 seconds/128 completed jobs.
Process restart loses it. Private prompt/payload storage is released on terminal transitions.

Multiple rings share client limits and idempotency. `shard_cohort` selects a known exact version;
`shard_region` is a routing preference. Each admitted job remains fixed to one backend/tokenizer
through retry and SSE resume. Alias cutover requires a READY target; old jobs retain their version.
DRAINING rejects new bindings while admitted work finishes under renewed leases. Stopped workers
and bounded ring history are reclaimed. Several serial rings can execute concurrently; this is
request-level routing, not continuous batching.

SSE events use IDs `<job-id>:<committed-token-count>:<published-character-count>`. Supply that exact
`Last-Event-ID` with the same idempotency key to resume delivery. The character offset distinguishes
an incomplete UTF-8 suffix from a final replacement character at the same token position. Legacy
token-only cursors remain accepted only when unambiguous. A client may receive provisional deltas
before the final receipts settle; only the successful terminal event carries `shard.proof_verified`.
An error event followed by `[DONE]` must not be interpreted as successful settlement.

Dropping the final client cancels the job. Disconnect cleanup atomically rechecks the current
subscriber count, so a client that has already reattached is not cancelled by an older handler.
Plain TCP EOF and streaming writes detect disconnects; a non-streaming TLS request may continue
until its explicit deadline because TLS cannot use `MSG_PEEK`. The explicit cancel endpoint always
remains available. Active ring send/receive waits are interrupted by shutting down the owned
attempt's sockets; bootstrap connection waits have a separate short timeout.

## Recovery and proof scope

Only transport/timeout failures receive the configured bounded retry (default one). Numerical
divergence, invalid receipts, cancellation and deadlines are never treated as retryable churn.
Every new attempt reconnects, uses a new nonce/job-attempt identity and resets the sequence.

Recovery re-executes the original prompt, mode and generation limit. Every previously committed
token must be reproduced exactly before the unseen suffix is delivered. Concatenating the previous
completion into a new prefill is avoided because it changes prefill/decode matrix shapes. Replayed
callbacks are suppressed, so neither the SSE stream nor logical usage counts duplicate tokens.
If a replayed prefix differs, the job fails instead of conditioning on an unverifiable continuation.

A successful final attempt re-executes the complete request and supplies its complete receipt set.
Different attempts' nonces/hash chains are never merged into one fabricated proof. `verified` means
the pinned signatures, freshness, identities, coverage and wire chains checked successfully; it is
not a ZK proof of the model's arithmetic or independent hardware attestation.

Shutdown drains or cancels bounded jobs, closes active ring sockets, wakes waiters and rejects new
admission. CLI SIGTERM/SIGINT uses a separate HTTP shutdown thread to avoid `serve_forever` deadlock.

## CPU validation

```sh
python -m pytest tests/test_service_queue.py tests/test_v4_gateway.py tests/test_v4_multi_gateway.py tests/test_ring_pool.py tests/test_ring_router.py tests/test_pipeline_session.py tests/test_v4_session.py -q
```

Tests use actual HTTP/SSE handlers, deterministic fake token callbacks and real Ed25519 receipts.
They cover fairness/admission, deduplication, cancellation/deadlines while receive is blocked,
replay prefix mismatch, forged receipts, readiness warmup, tenant isolation, streaming cursors and
terminal-state races. They validate service control behavior without loading a model or claiming
four/six-card throughput.

Current four-card >=40 / six-card >=30 committed-decode acceptance remains unmeasured on the new recipe. The 2026-10-08 selected CPU/socket regression (985 passed, 3 skipped) is not a full-repository CI result or a GPU benchmark.

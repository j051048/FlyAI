# Exact state cache, text protocol and offline quality artifacts

The state cache remains explicitly enabled, node-local and exact. It is not an
answer cache. The existing tenant/cohort/source/config/tokenizer/template/ring
epoch/lease identity and all-stage prepare/commit barrier still apply. Extended
prefixes still require their own original-full-prefill shadow comparison.

Snapshots copy only readable state: the filled sliding-window ring, compressed
attention/index history through the committed frontier, every compressor
accumulator, rollback/tap metadata, and the MTP ring through its own last-written
cursor. Chunk scratch and unwritten future rows are omitted. Restore copies into
the existing allocations; compressor aliases and CUDA graph addresses stay fixed.
MTP cursors, last speculative result and ring-drafter frontiers are retained, not
reconstructed from the main cursor. Host/pinned quotas include copied tensor and
conservative Python metadata charges. Dynamic restore tensors and bounded hash
scratch are checked against the explicit GPU reserve. Snapshots remain on host;
this change does not create another persistent GPU cache. Pinned transfers own a
completion event and source/partial destination trees through both success and
failure paths. CPU poison/continuation tests cover short windows, ring wrap,
compressed history and fixed storage. Actual CUDA tests remain separate gates.

`v4_text_protocol.encode_messages_ids` uses the native renderer. Template-owned
markers retain their real token IDs; identical spelling in supported message
text uses ordinary byte tokens. Ordinary messages retain the original encoding.
The parser recognizes reasoning boundaries and DSML only from genuine marker
IDs. Strict final DSML parsing is available for diagnostics; the public gateway
continues to serve plain text, rejects tool requests, and never invents API
`tool_calls` from model text. Its native ByteLevel streaming decoder holds an
incomplete UTF-8 suffix and preserves a correctly encoded literal U+FFFD.
Unimplemented schema constraints, custom sampling and logprobs requests remain
explicitly unsupported. A schema prompt or regular expression is not an exact
tail-sampler mask and does not qualify as structured-output support.

`phase0/v4_quality.py` separates saved generation artifacts from scoring. It does
not load a model, open a ring, run a GPU or call external services. Export actual
committed tokens from an existing benchmark report:

```powershell
.venv\Scripts\python.exe phase0/v4_quality.py export-benchmark benchmark-report.json --out generation.jsonl
.venv\Scripts\python.exe phase0/v4_quality.py compare baseline.jsonl generation.jsonl --out comparison.json
```

The JSONL has a hashed header and individually hashed rows, a bound frozen dataset,
original V4 runtime ABI, source/tokenizer/config identity, sampling recipe and
hardware inventory. It is bounded to 4,096 rows, 16,384 tokens per sequence, 2 MiB
per line and 64 MiB per file. Every sample binds its case and prompt hash.
Benchmark export retains output/receipt/report identity; raw signed receipts and
the original benchmark report remain the evidence for execution. Benchmark
probabilities are recorded as unknown because that runner did not collect them.

For previously generated local outputs, `package --identity identity.json
--dataset dataset.json --records records.json --out generation.jsonl` accepts
the same schema. A dataset case is `{id, prompt_ids, expected_tokens?}`. A row is
`{sample_id, case_id, prompt_sha256, tokens, token_logprobs?, metadata?}`;
probabilities, when present, must be finite, nonpositive and cover every actual
output token. `score generation.jsonl --out score.json` uses only frozen
`expected_tokens` references and reports exact-token match rate. Missing references
are not scored. It does not execute code tests or establish semantic quality.
Imported data and hashes are not execution attestations. CPU fixtures have an
explicit fixture ABI. No path reports Pass@1 or production qualification.

The existing `phase0/v4_numeric_diagnose.py trace` runner now records each actual
next frozen teacher token and its log probability from native scalar tail logits.
The last input token has no next target and therefore no invented score. Invalid
logits keep their numerical failure flags and a null probability. Compare saved
traces with:

```powershell
.venv\Scripts\python.exe phase0/v4_quality.py compare-teacher reference-trace.json candidate-trace.json --out teacher-comparison.json
```

This retains the existing first per-layer hidden/state/logit byte divergence and
adds aligned teacher-token probability differences. Legacy traces without these
optional probabilities keep exact tensor-byte comparison; their probability
coverage remains unknown. Diagnostics include hashing/synchronization overhead
and are never throughput or full-model-quality receipts. V4.1 remains outside the
native adapter ABI.

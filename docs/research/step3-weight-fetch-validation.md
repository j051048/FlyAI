# Step 3 — content-addressed verified weight fetch: validation report

> Historical research record — reviewed 2026-10-08 against `c2ab623`.
> Original dates, measured values, projections, later addenda and historical commands are preserved.
> “Active”, “next” and implementation gaps below describe the recorded experiment, not the current backlog.
> Current placement/serving contracts: [open network](../OPEN_INFERENCE_NETWORK.md),
> [GPT-OSS operations](../GPT_OSS_PRODUCTION.md), [V4 acceptance](../V4_BENCHMARK.md),
> and [documentation index](../DOCUMENTATION_INDEX.md).
> This early signed-manifest/block-fetch validation is separate from the current local
> `.shard-download.json` full-snapshot GPT-OSS loading contract. Metadata/header checks and small-file
> tests did not hash every byte of the complete checkpoint. Resolved/verified paths are documented
> in the current guides; the original finding list and historical line numbers remain here.
> Current resolution: `shard.fetch._safe_rel` rejects lexical/symlink path escapes;
> `MirrorProvider` uses Content-Range placement and signed per-shard revisions;
> `Libp2pProvider` is implemented rather than the stub described below.
> The strict GPT-OSS full-inventory entrypoint requires all listed snapshot files,
> while generic block/range fetching remains a separate contract.

Validates the JOIN-pillar mechanism that lets a node pull ONLY its layer block's
weights from any provider and verify every byte against a signed manifest before
loading. Files under test: `shard/manifest.py`, `shard/fetch.py`,
`phase0/publish_manifest.py`, `phase0/node_pull.py`, cross-checked against
`phase0/pipeline.py:load_stage`.

Validated end-to-end **without downloading the 63 GB model** — manifest built from
HF metadata (LFS oids), per-byte verification exercised on the small config/tokenizer
shards against the live HF mirror. Model: `openai/gpt-oss-120b` (GptOssForCausalLM,
36 layers, MXFP4 safetensors).

## Verdict summary

| Step | What | Verdict |
|------|------|---------|
| 1 | Build real signed manifest from HF metadata (no weight DL) | **PASS** |
| 2 | Manifest round-trip + tamper detection (fail closed) | **PASS** (stronger than spec — see note) |
| 3 | Block selection vs `load_stage` ground truth | **PASS** (zero missing files; selectivity weaker than claimed — note) |
| 4 | Real per-byte verification + fail-closed on corruption | **PASS** |
| 5 | Integration seam (node_pull → load_stage) | documented, not implemented |

**Bugs found:** 1 security finding (path traversal via `shard["path"]`, bounded
severity), 2 minor robustness/doc nits. No way found to bypass per-byte verification
on an honest path. The trust primitive holds: a malicious mirror/peer cannot feed
corrupted weights — sha256 mismatch deletes the file and fails closed.

---

## Step 1 — build a real signed manifest (no weight download) — PASS

```
$ cd phase0 && python3 publish_manifest.py --hf openai/gpt-oss-120b \
      --key /tmp/pub.key --out /tmp/manifest.json
generated new publisher key -> /tmp/pub.key
{
  "model_id": "openai/gpt-oss-120b",
  "arch": "GptOssForCausalLM",
  "layer_count": 36,
  "shards": 22,
  "weights_shards": 15,
  "total_gb": 65.28,
  "publisher_pubkey": "TZrI0MaZsOOiNsxxRruUjBbCxeG6agRYzQHtX7j6EfI=",
  "out": "/tmp/manifest.json"
}        # EXIT=0
```

`layer_count == 36` ✓, 15 weight shards (`model-0000{0..14}-of-00014.safetensors`) ✓,
`total_gb == 65.28` (~63 expected, fine — manifest counts raw byte sizes) ✓, prints
`publisher_pubkey` ✓, exit 0 ✓. Manifest has 22 shards: 15 weights + 3 config
(`config.json`, `generation_config.json`, `model.safetensors.index.json`) + 4 tokenizer.
`tied_embeddings: false`. Built in seconds with **no LFS download** — the multi-GB
safetensors were hashed for free from HF's LFS `oid` (== sha256). Only the KB-MB
config/tokenizer files were actually fetched-and-hashed.

Environment note: `huggingface_hub` is NOT required (the publisher uses `urllib`
directly against `huggingface.co/api` + `/resolve/main/`). `cryptography 46.0.5` is
present. No `pip install` was needed.

---

## Step 2 — manifest round-trip + tamper detection — PASS (fail closed)

```
[PASS] baseline verify (correct pubkey): no-raise
[PASS] baseline verify (no expected pubkey): no-raise
[PASS] (a) flip shard sha256 hex char: ManifestError: signature ... InvalidSignature
[PASS] (b) mutate publisher_pubkey: ManifestError: signature ... InvalidSignature
[PASS] (b2) mutate model_id (signed field): ManifestError: signature ... InvalidSignature
[PASS] (b3) mutate layer_count: ManifestError: signature ... InvalidSignature
[PASS] (c) wrong expected_pubkey: ManifestError: publisher pubkey does not match the pinned key
[PASS] (extra) unsigned manifest: ManifestError: manifest is unsigned
[PASS] (extra) bad schema: ManifestError: unknown manifest schema 'evil/9'
```

**Note — stronger than the spec predicted.** The task brief expected case (a)
(flipping a shard's sha256) to *still pass* on the assumption the signature only
covers a manifest subset. It does NOT. `canonical()` (`manifest.py:75-80`) signs the
entire manifest minus the `signature` field — including the full `shards` array. So
tampering with any shard's `sha256`/`size`/`path`/`shard_id` **breaks the manifest
signature itself**, caught at `verify_manifest` before any fetch. The per-byte check
in `fetch._verify` is a second, independent guard (catches corruption *in transit*
that a valid manifest wouldn't predict). Defense in depth — good.

`verify_manifest` is fail-closed against malformed input too: garbage base64 in
`publisher_pubkey`/`signature` and short key bytes all raise `ManifestError` rather
than crashing (the broad `except ... Exception` at `manifest.py:108` is intentional
and correct here — fail closed on anything).

---

## Step 3 — block selection (THE key correctness check) — PASS

Method: reconstructed the ground truth directly from `load_stage`'s `device_map`
(`pipeline.py:95-100`) — the exact set of weight names each stage places on its
device — mapped those names to files via the signed `weight_map`, and compared
against `fetch.shards_for_block`. **Zero missing files for every stage.**

```
=== Ground-truth (load_stage device_map) vs shards_for_block — tied=False ===
  stage 0 [0:9]:  truth=9 got=9 MISSING=[] extra=0
  stage 1 [9:18]: truth=5 got=5 MISSING=[] extra=0
  stage 2 [18:27]:truth=6 got=6 MISSING=[] extra=0
  stage 3 [27:36]:truth=6 got=6 MISSING=[] extra=0
```

Split math (`block_for_stage`, `fetch.py:132-137`) is identical to `load_stage`:
`lo = stage*36//4`, `hi = (stage+1)*36//4` → `[0:9] [9:18] [18:27] [27:36]`, an
exact 9-layers-per-stage partition. No off-by-one.

### Block-selection table (stage → files fetched)

`embed_tokens`, `model.norm`, and `lm_head` all live in `model-00012`.
`tied_embeddings == false` for this model.

| stage | role | layers | weight files (`model-XXXXX-of-00014`) | weight GB | tok | cfg |
|-------|------|--------|----------------------------------------|-----------|-----|-----|
| 0 | stage | [0:9] | 01,02,04,05,06,09,10,**12**,14 | 39.08 | 4 | 3 |
| 0 | coordinator | [0:9] | 01,02,04,05,06,09,10,**12**,14 | 39.08 | 4 | 3 |
| 1 | stage | [9:18] | 06,07,08,10,11 | 22.05 | 0 | 3 |
| 1 | coordinator | [9:18] | 06,07,08,10,11 | 22.05 | 4 | 3 |
| 2 | stage | [18:27] | 00,02,03,04,08,09 | 26.79 | 0 | 3 |
| 2 | coordinator | [18:27] | 00,02,03,04,08,09 | 26.79 | 4 | 3 |
| 3 | stage | [27:36] | 00,01,02,**12**,13,14 | 26.17 | 0 | 3 |
| 3 | coordinator | [27:36] | 00,01,02,**12**,13,14 | 26.17 | 4 | 3 |

(file **12** = the boundary-tensor file. Stage 0 gets it as the head — embed_tokens.
Stage 3 gets it as the tail — model.norm + lm_head. Bold = boundary file.)

Criteria:
- **(a)** A middle stage fetches ONLY its block's files, not all 36 layers — ✓.
  Stage 1 pulls 5/15 files, stage 2 pulls 6/15. Selective at the file level.
- **(b)** Head / coordinator additionally gets embeddings (file 12) + the 4 tokenizer
  files — ✓ (`shards_for_block` adds `model.embed_tokens` when `is_head`, and
  `want_tokenizer = role=="coordinator" or is_head`).
- **(c)** Tail (stage 3) gets `model.norm` + `lm_head` (both in file 12) — ✓.
- **(d)** `tied_embeddings`: simulated `tied=True` and confirmed the **tail** also
  pulls the embed file (`is_tail and tied` branch, `fetch.py:156`). The head still
  gets it via `is_head`. Stages 1,2 do not. Correct.

Every node also pulls all 3 config shards — including `model.safetensors.index.json`,
which `from_pretrained` requires to map weight names → files. Confirmed delivered in
Step 4.

### ⚠ Finding 3A (not a bug, but a load-bearing caveat): selectivity is weak for this checkpoint

The "selective" property holds at the *file* level but is far weaker in *bytes* than
the docstrings imply ("downloads only the safetensors files those layers live in").
The gpt-oss-120b MXFP4 `weight_map` is **not laid out in contiguous layer order** —
layers are scattered across files and many layers straddle two or three files:

```
  layer  0: [model-00009]
  layer  4: [model-00004, model-00005, model-00014]   # spans 3 files
  layer 25: [model-00000, model-00004]
  layer 30: [model-00002, model-00012]
```

Consequence — per-stage download as a fraction of the full 65.25 GB:

```
  stage 0 [0:9]:   9/15 files = 39.08 GB (60%)
  stage 1 [9:18]:  5/15 files = 22.05 GB (34%)
  stage 2 [18:27]: 6/15 files = 26.79 GB (41%)
  stage 3 [27:36]: 6/15 files = 26.17 GB (40%)
```

So a 4-way split does NOT give each node ~25% (~16 GB); stage 0 pulls 60% of all
weight bytes for its 9 layers. This is a property of the published checkpoint's shard
packing, **not a fetch.py bug** — `shards_for_block` correctly pulls the minimal set
of *files* that cover the needed tensors, and `load_stage` maps the unneeded
co-resident layers to `meta` (never loaded into VRAM, just downloaded to disk). It is
real disk/bandwidth cost worth knowing for node sizing and for any future
re-shard-on-publish optimization. It does not affect correctness or the trust property.

---

## Step 4 — real per-byte verification + fail-closed — PASS

All against the **live** HF mirror `https://huggingface.co/openai/gpt-oss-120b/resolve/main/`,
small shards only (`generation_config.json` = 177 B, etc.).

**4a — honest fetch via `MirrorProvider`:**
```
chosen small shard: generation_config.json size 177 kind config
=== 4a honest fetch ===
  [PASS] fetched+verified, file exists=True, size=177
  .part leftover? False     # os.replace consumes the .part on success
```

**4b — corrupt expected sha256 → fail closed, bad file deleted:**
```
=== 4b corrupt expected sha256 ===
  after provider.fetch: file present=True              # provider does NOT verify (by design)
  [PASS] _verify raised FetchError: generation_config.json: sha256 mismatch (corrupt or tampered)
  [PASS] bad file deleted? True
=== 4b2 corrupt expected size ===
  [PASS] _verify raised on size: ... size 177 != manifest 178; deleted=True
```

**4c — full `fetch_block` end-to-end, valid signature, corruption-in-transit:**
re-signed a manifest with one wrong shard sha (so `verify_manifest` passes, isolating
the per-byte guard):
```
  [PASS] per-byte guard caught corruption-vs-signed-manifest: generation_config.json: sha256 mismatch
  [PASS] file removed: True
```

**Full `fetch_block` integration against the real mirror** (crafted a no-weights
manifest so only the 7 config+tokenizer shards download — real HF bytes, full code
path: `verify_manifest` gate → `block_for_stage` → `shards_for_block` → `_cached` →
`provider.fetch` → `_verify` → return paths):
```
[fetch] stage 0/4 layers [0:1] role=coordinator: 7 shards (0 weights), 0.03 GB
[fetch]   fetch chat_template.jinja / config.json / generation_config.json /
          model.safetensors.index.json / special_tokens_map.json /
          tokenizer.json / tokenizer_config.json
[fetch] block verified: 7 files in /tmp/fetchblock_e2e
[PASS] fetch_block returned 7 verified files
```

**Manifest-pin gate fires before any download:**
```
=== fetch_block with wrong expected_pubkey ===
  [PASS] raised ManifestError: publisher pubkey does not match the pinned key
  [PASS] no model dir created / no download: dir exists=False
```

**Resume integrity:** seeded a corrupt `.part` (wrong bytes, half size), let
`MirrorProvider` resume via HTTP Range, then verified — the full-file hash in
`_verify` caught the corrupt prefix and failed closed. `Libp2pProvider` correctly
raises `ProviderUnavailable` (step-8 stub); `node_pull --source libp2p` probes it,
catches that, and falls back to the mirror.

---

## Bugs / findings

### 🟠 Finding B1 (security, bounded severity): path traversal / absolute path via `shard["path"]`

`fetch.py:228` — `dest = os.path.join(model_dir, s["path"])` — uses the manifest's
`path` field verbatim with no traversal guard. A manifest containing a `../` or an
absolute path writes **outside `model_dir`**. Proven reachable through `fetch_block`
with a validly-signed manifest and a hash-matching payload:

```
[fetch]   fetch ../ESCAPED_AND_PERSISTED (0.00 GB)
[fetch] block verified: 1 files in /tmp/persist_sandbox/model
  escaped target = /tmp/persist_sandbox/ESCAPED_AND_PERSISTED
  PERSISTED outside model_dir (hash matched)? True
  content: b'* * * * * root /bin/evil\n'
```

`os.makedirs(os.path.dirname(dest), ...)` at `fetch.py:229` will even create the
escaped parent dirs. An absolute path (`/etc/cron.d/evil`) escapes outright because
`os.path.join` discards the prefix on an absolute second arg. The written file's
sha256 is still verified, so the attacker must supply content matching their own
manifest entry — but that is trivially under their control.

**Threat model / why severity is bounded:**
- The `path` field is *signed*, so a **mirror/peer cannot inject it** — the manifest
  signature would break (and the per-byte guard is downstream of an honest path).
  The transport-level trust property ("a malicious mirror can't feed bad weights")
  is intact.
- Exploitation requires a **malicious or compromised publisher**, OR a node run with
  `--pubkey` omitted (so it trusts the manifest's own key — `node_pull.py:54-55`
  already warns about this), OR a manifest trusted before the catalog pin is enforced.
- Real HF-derived manifests never contain traversal — confirmed: the actual
  gpt-oss-120b manifest has **NONE** (`bad paths == NONE`).

It is still a defect for a **trust primitive** that ingests a publisher-controlled
field and writes to the filesystem. Defense in depth warrants rejecting it.

**Proposed fix** (in `fetch.shards_for_block` or at the top of `fetch_block`, before
the loop) — reject any non-relative / escaping path, fail closed:

```python
def _safe_rel(model_dir: str, rel: str) -> str:
    if os.path.isabs(rel) or os.path.splitdrive(rel)[0]:
        raise FetchError(f"unsafe shard path (absolute): {rel!r}")
    dest = os.path.normpath(os.path.join(model_dir, rel))
    root = os.path.normpath(model_dir)
    if dest != root and not dest.startswith(root + os.sep):
        raise FetchError(f"unsafe shard path (escapes model_dir): {rel!r}")
    return dest
```

Then `dest = _safe_rel(model_dir, s["path"])` at `fetch.py:228`. Optionally also
reject `..`/absolute components at publish time in `publish_manifest._kind` so bad
manifests can't be signed in the first place. (Do NOT implement here — flagged only.)

### 🟡 Finding B2 (minor robustness): stale `.part` on mid-download crash + a changed same-size file

`MirrorProvider._download` (`fetch.py:78-89`) only resets a `.part` when it is
*larger* than `total` (`have > total`, line 80). If a download dies mid-stream the
`.part` persists; on the next run it resumes via Range. If the server's file changed
to a *different file of the same/larger size*, the resume math won't catch it — but
the full-file hash in `_verify` still fails closed, so this is wasted-retry
robustness, not a security gap. Acceptable as-is; noting for completeness. No fix
required.

### 🟡 Finding B3 (doc nit): docstrings overstate byte-level selectivity

`fetch.py:7-8` and the `shards_for_block` docstring describe selectivity as
"downloads only the safetensors files those layers live in" implying near-1/N
bandwidth. As Finding 3A shows, for the as-published MXFP4 layout a stage can pull
60% of all bytes. Worth a one-line caveat that byte savings depend on the
checkpoint's shard packing. Not a code bug.

---

## Step 5 — integration seam (node_pull → load_stage): what's needed to wire it

**Not implemented** (per instructions) — documented precisely.

Current launch flow (per box):
1. `phase0/stage_bootstrap.sh:7` runs `python3 /root/get_model.py
   openai/gpt-oss-120b /root/models/gpt-oss-120b` — the **unverified, whole-model**
   download (the path step 3 replaces).
2. `phase0/launchA.sh` / `launchB.sh` then start the stage:
   `python3 pipeline.py --stage N --nstages 4 --model $M ...`, which calls
   `load_stage($M, N, 4)` — and `$M` is the local `/root/models/gpt-oss-120b` dir.

The seam is one line: **replace the `get_model.py` call with `node_pull.py`**,
parameterized by that box's stage and role, before the stage launches. Each box
already knows its `--stage`/`--nstages` (they are hardcoded in `launchA.sh`/`launchB.sh`),
so the bootstrap needs the same values plus role and the manifest reference + pinned
pubkey. Concretely, per box, replace stage_bootstrap.sh line 7 with:

```bash
python3 /root/node_pull.py \
    --manifest "$MANIFEST_URL"   \  # URL or local path to the signed manifest
    --pubkey   "$PUBLISHER_PUBKEY"\  # the catalog-pinned key (or @/path/to/key)
    --model-dir /root/models/gpt-oss-120b \
    --stage "$STAGE" --nstages "$NSTAGES" \
    --role  "$ROLE"                  # "coordinator" for the head/stage-0 box, else "stage"
```

`node_pull` defaults `--base-url` to the HF resolve URL for `manifest.model_id`, so a
plain HF mirror needs nothing else; `--base-url` overrides it for a private mirror.
Requirements / gotchas for wiring:
- **Stage ↔ role consistency.** `fetch_block` derives head/tail from `stage`/`nstages`
  and pulls the tokenizer when `role=="coordinator" or stage==0`. The box that hosts
  the coordinator/driver must pass `--role coordinator`; pure middle/tail stages pass
  `--role stage`. These must match the `--stage`/`--nstages` the box later passes to
  `pipeline.py` (same split math, verified in Step 3). A box running two stages
  (e.g. launchB hosts stages 2 and 3) must run `node_pull` for **each** stage so both
  blocks land in the shared `model-dir` (the union of files; `_cached` dedupes the
  shared boundary file 12).
- **Pubkey distribution.** The pinned `PUBLISHER_PUBKEY` (printed by
  `publish_manifest.py`) must reach each box — env var or a file referenced as
  `--pubkey @/root/.shard_pub`. Omitting it triggers `node_pull`'s warning and trusts
  the manifest's own key (lab-only; a mirror could then serve a self-signed manifest).
- **Manifest hosting.** The signed `manifest.json` must be reachable by every box
  (`--manifest` accepts a URL or a local path that's been distributed). In the c0mpute
  catalog model this is the manifest pointer the catalog stores.
- **No `huggingface_hub` dependency** on the boxes — `node_pull`/`fetch` use `urllib`
  only. `cryptography` is the one hard dep (for `verify_manifest`).
- The result is a `model-dir` containing exactly the files `load_stage` will
  materialize (verified in Step 3 + Step 4), so `pipeline.py --model /root/models/...`
  works unchanged. `load_stage` re-reads `config.json` for `num_hidden_layers` /
  `tie_word_embeddings` — both are in the config shards every node pulls, so they are
  present.

---

## Conclusion

The step-3 mechanism is **sound as a trust primitive**. Manifest signing/verification
is fail-closed and covers the full manifest (including every shard hash). Block
selection is exactly correct against `load_stage` (zero missing tensors across all
stages and roles, head/tail/tied boundaries handled). Per-byte verification deletes
and rejects any corrupt or tampered shard, including corruption in transit on a
validly-signed manifest, and verified against the live HF mirror. A malicious
mirror/peer demonstrably cannot feed corrupted weights.

Two caveats and one bounded security finding qualify the result:
- **B1 (path traversal)** is the only actionable security item — a publisher-controlled
  `path` field is written without a traversal guard. It cannot be exploited by a
  mirror (signature-protected) and never occurs in real HF manifests, but a trust
  primitive should reject it. Fix proposed above.
- **3A/B3 (weak byte-selectivity)** is a checkpoint-layout reality, not a bug, but
  matters for node sizing and the docs overstate it.
- **B2** is benign retry robustness.

Recommend landing B1's guard before this is wired into the live launcher; the rest is
documentation/operational. The code is otherwise ready to replace the
`get_model.py` whole-model download per the Step 5 seam.

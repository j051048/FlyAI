# V4 token confidentiality and local replay boundaries

`V4_SEALED_IDS=1` is an opt-in, versioned wire mode. All members must negotiate
`v4-token-privacy/1` with the same public key ID during reset. A legacy member, an
unknown version, a missing descriptor, or a different key ID rejects the job.
The descriptor explicitly records `activations: visible`.

The head encrypts each new frame's exact int64 token matrix once using
ChaCha20-Poly1305 with a random 96-bit nonce. Its authenticated metadata binds a
fresh 256-bit job nonce, the epoch, position, token shape, and job/swarm/model
identifiers. The token-valued pipelined `dnxt` hint is encrypted with the IDs in
the same payload; its boolean presence and the `dprev` prediction flag are public.
Every subsequent hop forwards the identical envelope. Rejected stale frames
carry only fence metadata, without IDs, token hints, or activations. BF16 and FP8
receipt inputs include the transmitted ciphertext and metadata, plus the exact
wire activation bytes and FP8 scales; they never hash local zero placeholders.
The legacy signed receipt format remains unchanged. Its payload root commits to
the new wire representation when this mode is explicitly selected.

Provision the independent 32-byte token secret only to the head, **every stage
holding one of the first `n_hash_layers` layers**, and the tail. The native Flash
configuration has three hash layers, which may span several stages. The tail
needs originals for DSpark's verification frontier and acceptance decisions.
Ordinary score-routed middle stages have no decryption secret and supply only a
shape-correct zero matrix to the model API. Those gates do not read token IDs;
the CPU numerical regression checks equal hidden states and logits against a
plaintext four-stage ring through prefill and decode. This is not a GPU benchmark.

Generate a key locally, then provision its file through an authenticated channel
to those trusted recipients only:

```text
python engines/deepseek_v4/v4_privacy.py keygen --out token.key
```

The command exclusively creates the file and prints only its path and public
descriptor. POSIX files are mode 0600. Windows key generation removes inherited
ACLs and grants read/write to the current account SID. `read_key_file` checks
POSIX permissions; Windows operators must preserve the account-restricted ACL
when copying the file. It rejects symlinks and malformed key files. Do not reuse
`SHARD_PSK` or put this secret in a runtime configuration, log, receipt, CLI
argument, or general environment propagation list.

The serving integration reads the file from `SHARD_V4_TOKEN_KEY_FILE` on trusted
stages and uses `V4_TOKEN_PRIVACY_KEY_ID` as the public identity. The path is local
provisioning metadata, not part of the public `V4_` configuration payload.
`V4_SEALED_IDS` and the public key ID must be configured consistently across the
ring. An ordinary middle needs only the public key ID. Provisioning a key never
makes an otherwise untrusted operator trustworthy.

This mode hides token IDs from keyless middle processes. It does **not** encrypt
activations, prevent activation inversion, hide length/position/routing side
channels, prove execution, authenticate an operator's hardware, or protect from
a trusted recipient that leaks the shared key. Stages on the same physical host
are not separate security domains when their operators can read one another's
memory/files. Cross-job/epoch/position substitution is rejected; a repeated frame
with exactly the same binding still requires the transport's normal sequencing
and epoch-fencing rules. The coordinator must generate a fresh job nonce rather
than reuse an old one. Public envelope validation at a keyless middle cannot
verify the AEAD tag; trusted hash stages and the tail verify it before use.

## Activation commitments and local disputes

`phase0/activation_proof.py` hashes every logical tensor byte with dtype and
shape. It supports BF16/FP8 without NumPy dtype conversion and excludes unrelated
backing storage. GPU snapshots incur a host copy and O(bytes) hashing; this is a
local dispute helper, not a constant-cost production kernel proof. Sampled
hashes (`stride != 1`) are rejected. Chain verification checks a trusted initial
root (zero by default), a single stage, consecutive steps, and an optional final
root. Receipt bundles bind their declared stage and final endpoint. An unsigned
chain can still be fabricated wholesale and is only internally consistent.

`phase0/fraud_proof.py` requires a nonempty SHA-256 engine identity on creation
and an identical validator engine identity and GPU architecture on replay. It
binds the declared step commitment, input and output digests, deposits, challenge
identity, previous root, deadline, and numerical policy. Snapshot submission
checks **both** input and output and stores detached copies. Replay gets a copy,
and stored evidence is rehashed before use. A missing snapshot can produce a
timeout recommendation only after the deadline. Numerical tolerance must be
chosen at creation; it cannot be introduced to excuse an output afterward, and
shape mismatch, broadcasting, NaN, and infinity are not accepted by that policy.

The resulting `SLASH_DEFENDER` / `CHALLENGE_FAILED` names and amount fields are
historical local recommendations. Results explicitly carry
`settlement: local_arbitration` and `onchain_executed: false`. No transaction is
submitted, deposit moved, stake burned, or permissionless consensus reached.
The caller is a trusted local adjudicator; it must independently authenticate
receipts, code/weights/configuration, numerical mode, KV/rollback replay state,
hardware and validator identity. Matching self-reported source hashes alone
does not establish those facts. A supplied replay callback is not a proof that
the remote worker executed the same code.

`phase0/proof_receipt.py` continues to label its envelope self-reported and
unsigned. Stage commitment checks establish structure and internal consistency,
not model correctness. Independent reference token agreement is token agreement,
not a byte-level proof of every floating-point intermediate. Signed serving
receipts authenticate a signer's committed bytes and adjacency; authenticating
bytes alone does not verify the computation that produced them.

Focused CPU validation:

```text
python -m pytest tests/test_v4_privacy.py tests/test_v4_privacy_ring.py tests/test_proof_boundaries.py tests/test_fraud_proof.py -q
```

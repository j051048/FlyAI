"""Opt-in token confidentiality for a V4 ring's untrusted middle stages.

This seals token IDs, not hidden activations. Provision the independent 32-byte
secret only to the head, every stage holding a hash-routed layer, and the tail
(DSpark verification needs candidate IDs). Ordinary stages forward the exact
same envelope and use shape-correct zeros for the score-routed model API.
No transport secret is read, distributed, or reused by this module.
"""
import hashlib
import json
import re
import secrets
import struct
import os
import stat

SCHEMA = "v4-token-privacy/1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MAX_TOKENS = 1 << 20
_BINDING_FIELDS = {"job_nonce", "epoch", "start_pos", "job_id", "swarm_id", "model_id"}
_ENVELOPE_FIELDS = {"schema", "key_id", "binding", "shape", "has_next_token", "nonce", "ciphertext"}


class TokenPrivacyError(ValueError):
    pass


def read_key_file(path):
    """Read only an explicitly provisioned file; never read a transport/env secret.

    POSIX rejects group/world-readable files. On Windows, the operator must use
    an account-restricted ACL (keygen does this); chmod bits do not model ACLs.
    The file path itself must not be included in a public runtime configuration.
    """
    if stat.S_ISLNK(os.stat(path, follow_symlinks=False).st_mode):
        raise TokenPrivacyError("token key must be a regular file")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 128:
            raise TokenPrivacyError("token key must be a small regular hex file")
        if os.name != "nt" and info.st_mode & 0o077:
            raise TokenPrivacyError("token key file must have permissions 0600 or stricter")
        raw = source.read(128)
    try:
        encoded = raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise TokenPrivacyError("token key file must contain 32 bytes in hex") from exc
    return _hex(encoded, 32, "token secret")


def generate_key_file(path):
    """Create a fresh private key file exclusively; print/return public key_id only."""
    descriptor = None
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        if os.name == "nt":
            # chmod(0600) alone is ineffective for Windows confidentiality.
            import csv
            import subprocess
            who = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"],
                                 check=True, capture_output=True, text=True)
            rows = list(csv.reader(who.stdout.strip().splitlines()))
            sid = rows[0][1] if len(rows) == 1 and len(rows[0]) == 2 else ""
            if not re.fullmatch(r"S-1-(?:[0-9]+-)*[0-9]+", sid):
                raise TokenPrivacyError("cannot identify current Windows account SID for private key ACL")
            subprocess.run(["icacls", os.fspath(path), "/inheritance:r", "/grant:r", f"*{sid}:(R,W)"],
                           check=True, capture_output=True)
        secret = secrets.token_bytes(32)
        encoded = secret.hex().encode("ascii") + b"\n"
        with os.fdopen(fd, "wb") as target:
            fd = None
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        descriptor = TokenPrivacy(secret).descriptor()
    except Exception:
        if fd is not None:
            os.close(fd)
            fd = None
        os.unlink(path)
        raise
    finally:
        if fd is not None:
            os.close(fd)
    return descriptor


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise TokenPrivacyError(f"{name} must be a nonnegative integer")
    return value


def _hex(value, nbytes, name):
    if not isinstance(value, str) or len(value) != 2 * nbytes:
        raise TokenPrivacyError(f"{name} must encode {nbytes} bytes as lowercase hex")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise TokenPrivacyError(f"invalid {name}") from exc
    if raw.hex() != value:
        raise TokenPrivacyError(f"invalid {name}")
    return raw


def _binding(value):
    if not isinstance(value, dict) or set(value) - _BINDING_FIELDS:
        raise TokenPrivacyError("invalid token binding fields")
    if not isinstance(value.get("job_nonce"), str) or not _HEX64.fullmatch(value["job_nonce"]):
        raise TokenPrivacyError("job_nonce must be a fresh 32-byte lowercase hex nonce")
    result = {"job_nonce": value["job_nonce"],
              "epoch": _integer(value.get("epoch"), "epoch"),
              "start_pos": _integer(value.get("start_pos"), "start_pos")}
    for name in ("job_id", "swarm_id", "model_id"):
        if name in value:
            if not isinstance(value[name], str) or len(value[name]) > 4096:
                raise TokenPrivacyError(f"invalid {name}")
            result[name] = value[name]
    return result


def _shape(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TokenPrivacyError("token shape must be [batch, sequence]")
    b, s = (_integer(x, "shape dimension") for x in value)
    if b == 0 or s == 0 or b * s > _MAX_TOKENS:
        raise TokenPrivacyError("invalid or oversized token shape")
    return [b, s]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def validate_envelope(envelope, binding=None, shape=None):
    """Validate public metadata without a secret; AEAD verification requires open()."""
    if not isinstance(envelope, dict) or set(envelope) != _ENVELOPE_FIELDS:
        raise TokenPrivacyError("invalid sealed token envelope fields")
    if envelope["schema"] != SCHEMA:
        raise TokenPrivacyError("unsupported token privacy schema")
    _hex(envelope["key_id"], 32, "key_id")
    actual_binding = _binding(envelope["binding"])
    actual_shape = _shape(envelope["shape"])
    if type(envelope["has_next_token"]) is not bool:
        raise TokenPrivacyError("has_next_token must be boolean")
    if actual_binding != envelope["binding"] or actual_shape != envelope["shape"]:
        raise TokenPrivacyError("noncanonical sealed token metadata")
    _hex(envelope["nonce"], 12, "nonce")
    _hex(envelope["ciphertext"], (actual_shape[0] * actual_shape[1] + envelope["has_next_token"]) * 8 + 16,
         "ciphertext")
    if binding is not None and actual_binding != _binding(binding):
        raise TokenPrivacyError("sealed token binding mismatch (job, epoch, or position)")
    if shape is not None and actual_shape != _shape(shape):
        raise TokenPrivacyError("sealed token shape mismatch")
    return envelope


def wire_token_bytes(envelope):
    """The transmitted opaque envelope, never decrypted IDs or local placeholders."""
    validate_envelope(envelope)
    return b"V4-sealed-ids/1\0" + _canonical(envelope)


def wire_payload_bytes(hidden_bytes, envelope, scale_bytes=b""):
    """Identical BF16/FP8 sender/receiver receipt input before any wire decoding."""
    if not isinstance(hidden_bytes, bytes) or not isinstance(scale_bytes, bytes):
        raise TokenPrivacyError("wire tensor payloads must be bytes")
    tokens = wire_token_bytes(envelope)
    return (b"V4-sealed-wire/1\0" + struct.pack("<QQQ", len(hidden_bytes), len(scale_bytes), len(tokens))
            + hidden_bytes + scale_bytes + tokens)


class TokenPrivacy:
    """A keyed trusted recipient or a keyless opaque forwarding codec."""

    def __init__(self, secret=None, *, key_id=None):
        if secret is not None and (not isinstance(secret, bytes) or len(secret) != 32):
            raise TokenPrivacyError("token privacy needs an independently provisioned 32-byte secret")
        actual_id = hashlib.sha256(b"V4-token-secret/1\0" + secret).hexdigest() if secret is not None else None
        if key_id is not None:
            _hex(key_id, 32, "key_id")
        if actual_id is not None and key_id is not None and actual_id != key_id:
            raise TokenPrivacyError("token secret does not match negotiated key_id")
        if actual_id is None and key_id is None:
            raise TokenPrivacyError("keyless codec requires negotiated key_id")
        self.__secret = secret
        self.key_id = actual_id or key_id

    def descriptor(self):
        return {"schema": SCHEMA, "key_id": self.key_id, "activations": "visible"}

    def verify_descriptor(self, peer):
        if peer != self.descriptor():
            raise TokenPrivacyError("token privacy mode/key mismatch; mixed legacy/sealed rings are unsupported")

    def _aead(self):
        if self.__secret is None:
            raise TokenPrivacyError("this stage has no trusted token decryption secret")
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        return ChaCha20Poly1305(self.__secret)

    def _aad(self, envelope):
        return b"V4-sealed-ids-AAD/1\0" + _canonical({k: envelope[k] for k in
                                                    ("schema", "key_id", "binding", "shape", "has_next_token")})

    def seal(self, ids, binding, *, next_token=None):
        """Head-only operation, once per new frame. Forward its result unchanged."""
        aead = self._aead()
        if hasattr(ids, "detach"):
            ids = ids.detach().cpu().tolist()
        if not isinstance(ids, (list, tuple)) or not ids or not isinstance(ids[0], (list, tuple)):
            raise TokenPrivacyError("token IDs must be a rectangular integer matrix")
        shape = _shape([len(ids), len(ids[0])])
        flat = []
        for row in ids:
            if not isinstance(row, (list, tuple)) or len(row) != shape[1]:
                raise TokenPrivacyError("token IDs must be rectangular")
            for token in row:
                token = _integer(token, "token ID")
                if token > (1 << 63) - 1:
                    raise TokenPrivacyError("token ID exceeds int64")
                flat.append(token)
        if next_token is not None:
            next_token = _integer(next_token, "next token hint")
            if next_token > (1 << 63) - 1:
                raise TokenPrivacyError("next token hint exceeds int64")
            flat.append(next_token)
        nonce = secrets.token_bytes(12)
        envelope = {"schema": SCHEMA, "key_id": self.key_id, "binding": _binding(binding),
                    "shape": shape, "has_next_token": next_token is not None, "nonce": nonce.hex()}
        envelope["ciphertext"] = aead.encrypt(nonce, struct.pack(f"<{len(flat)}q", *flat),
                                               self._aad(envelope)).hex()
        return envelope

    def open(self, envelope, binding):
        return self.open_payload(envelope, binding)["ids"]

    def open_payload(self, envelope, binding):
        """Trusted tail may also consume the encrypted pipelined dnxt hint."""
        binding = _binding(binding)
        validate_envelope(envelope, binding)
        if envelope["key_id"] != self.key_id:
            raise TokenPrivacyError("sealed token key_id mismatch")
        aead = self._aead()
        try:
            raw = aead.decrypt(bytes.fromhex(envelope["nonce"]), bytes.fromhex(envelope["ciphertext"]),
                               self._aad(envelope))
        except Exception as exc:
            from cryptography.exceptions import InvalidTag
            if isinstance(exc, InvalidTag):
                raise TokenPrivacyError("sealed token authentication failed") from exc
            raise
        b, s = envelope["shape"]
        values = struct.unpack(f"<{b * s + envelope['has_next_token']}q", raw)
        if any(x < 0 for x in values):
            raise TokenPrivacyError("sealed token ID is negative")
        return {"ids": [list(values[i * s:(i + 1) * s]) for i in range(b)],
                "next_token": values[-1] if envelope["has_next_token"] else None}

    def open_next_token(self, envelope, binding):
        return self.open_payload(envelope, binding)["next_token"]

    def local_ids(self, envelope, binding, *, lo, n_hash_layers, is_tail, shape):
        """Only score-routed middle stages can use zeros; tail DSpark needs originals."""
        binding = _binding(binding)
        validate_envelope(envelope, binding, shape)
        if envelope["key_id"] != self.key_id:
            raise TokenPrivacyError("sealed token key_id mismatch")
        _integer(lo, "stage lo")
        _integer(n_hash_layers, "n_hash_layers")
        if type(is_tail) is not bool:
            raise TokenPrivacyError("is_tail must be boolean")
        if lo < n_hash_layers or lo == 0 or is_tail:
            return self.open(envelope, binding)
        b, s = _shape(shape)
        return [[0] * s for _ in range(b)]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    keygen = commands.add_parser("keygen", help="create an account-private independent key file")
    keygen.add_argument("--out", required=True)
    args = parser.parse_args()
    descriptor = generate_key_file(args.out)
    print(json.dumps({"key_file": os.path.abspath(args.out), **descriptor}, sort_keys=True))

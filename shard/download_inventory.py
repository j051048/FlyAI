"""Content-bound model download inventory, shared by downloader and runtimes.

This is a local content manifest, not a publisher signature or remote hardware
attestation. Model cohorts pin checkpoint_id/manifest_sha256; first load rehashes
the actual files. A stored boolean can never substitute for that verification.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

FILENAME = ".shard-download.json"
SCHEMA = "shard-model-download/1"
_IDENTITY = ("schema", "repo", "revision", "files")
_FILE_FIELDS = {"path", "size", "algorithm", "digest", "payload_sha256"}
_VERIFICATION_PROOF = object()


class InventoryError(ValueError):
    pass


class VerifiedInventory(dict):
    """JSON-compatible, same-process evidence of a complete file verification.

    Serialized JSON deliberately loses this evidence: a declared boolean is not
    a file verification. The snapshot is checked before consumers use it, so
    mutating this mapping cannot silently change the verified identity. This is
    a local verification result, not remote attestation or a publisher signature.
    """

    def __init__(self, value, *, _proof=None, _config=None):
        if _proof is not _VERIFICATION_PROOF:
            raise InventoryError("verified inventory must come from actual file verification")
        super().__init__(json.loads(canonical(value)))
        self._proof = _proof
        self._snapshot = hashlib.sha256(canonical(self)).digest()
        self._config = canonical(_config)


def verified_config(inventory):
    """Return the config snapshot from verify_inventory(verify_files=True).

    A copied/serialized mapping, a header-only inventory, or a changed verified
    mapping must be verified again. Runtimes still verify their files at load;
    this result records what was actually hashed at the time of verification.
    """
    if (not isinstance(inventory, VerifiedInventory)
            or getattr(inventory, "_proof", None) is not _VERIFICATION_PROOF
            or hashlib.sha256(canonical(inventory)).digest() != inventory._snapshot
            or inventory.get("payload_integrity_verified") is not True):
        raise InventoryError("actual complete local file verification required; call verify_inventory(verify_files=True)")
    return json.loads(inventory._config)


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False).encode("ascii")
    except (ValueError, TypeError) as exc:
        raise InventoryError("inventory requires canonical finite JSON") from exc


def relative_path(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or any(ord(c) < 32 for c in value)):
        raise InventoryError("unsafe inventory file path")
    if PurePosixPath(value).is_absolute() or any(p in ("", ".", "..") for p in value.split("/")):
        raise InventoryError("unsafe inventory file path")
    if value in (FILENAME, ".flyai-download.lock"):
        raise InventoryError("model file collides with inventory metadata")
    return value


def safe_file(root, relative):
    relative_path(relative)
    root = Path(root).resolve()
    path = root.joinpath(*PurePosixPath(relative).parts)
    try:
        path.resolve().relative_to(root)
    except ValueError:
        raise InventoryError("inventory file escapes model directory") from None
    current = path
    while current != root:
        if current.is_symlink():
            raise InventoryError("inventory must not follow model file symlinks")
        current = current.parent
    return path


def _hex(value, length, label):
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise InventoryError(f"invalid {label}")


def validate_body(body):
    if not isinstance(body, dict) or set(body) != set(_IDENTITY) or body["schema"] != SCHEMA:
        raise InventoryError("unsupported model inventory schema or fields")
    if not isinstance(body["repo"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?", body["repo"]):
        raise InventoryError("invalid model repository identity")
    if not isinstance(body["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", body["revision"]):
        raise InventoryError("inventory revision must be an immutable commit")
    files = body["files"]
    if not isinstance(files, list) or not 1 <= len(files) <= 10000:
        raise InventoryError("inventory requires a bounded file list")
    seen = set()
    for row in files:
        if not isinstance(row, dict) or set(row) != _FILE_FIELDS:
            raise InventoryError("invalid inventory file fields")
        relative_path(row["path"])
        identity = row["path"].casefold() if os.name == "nt" else row["path"]
        if identity in seen:
            raise InventoryError("duplicate or platform-colliding inventory file")
        seen.add(identity)
        if type(row["size"]) is not int or not 0 <= row["size"] <= (1 << 63) - 1:
            raise InventoryError("invalid inventory byte size")
        if not isinstance(row["algorithm"], str):
            raise InventoryError("unsupported upstream file digest")
        length = {"sha256": 64, "git-sha1": 40}.get(row["algorithm"])
        if length is None:
            raise InventoryError("unsupported upstream file digest")
        _hex(row["digest"], length, "upstream digest")
        _hex(row["payload_sha256"], 64, "payload SHA256")
        if row["algorithm"] == "sha256" and row["digest"] != row["payload_sha256"]:
            raise InventoryError("LFS digest differs from payload identity")
    paths = {row["path"] for row in files}
    if "config.json" not in paths or not ("model.safetensors" in paths or "model.safetensors.index.json" in paths):
        raise InventoryError("model inventory requires config.json and standard safetensors model weights")
    if "adapter_config.json" in paths:
        raise InventoryError("adapter/base-model identity needs a separate qualified inventory")
    return body


def build_inventory(repo, revision, files):
    body = validate_body({"schema": SCHEMA, "repo": repo, "revision": revision,
                          "files": sorted(files, key=lambda row: row["path"])})
    digest = hashlib.sha256(canonical(body)).hexdigest()
    return {**body, "manifest_sha256": digest, "checkpoint_id": "sha256:" + digest}


def hash_file(path, size):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
        raise InventoryError("model file size/type differs from inventory")
    sha, git = hashlib.sha256(), hashlib.sha1()
    git.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            sha.update(block); git.update(block)
    return sha.hexdigest(), git.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InventoryError("duplicate inventory JSON key")
        result[key] = value
    return result


def _verify_loader_files(root, rows):
    """A verified list must also be the files the loader actually resolves."""
    config = rows["config.json"]
    sha, git = hash_file(safe_file(root, "config.json"), config["size"])
    if (sha != config["payload_sha256"] or
            (sha if config["algorithm"] == "sha256" else git) != config["digest"]):
        raise InventoryError("model configuration differs from inventory")
    # HF's default resolver looks for root weights or the explicit weight_map.
    # Unreferenced nested original/metal backups cannot override those paths.
    # Every index target remains required below to appear in the verified list.
    for path in root.glob("*.safetensors"):
        relative = path.relative_to(root).as_posix()
        if relative not in rows:
            raise InventoryError("unlisted safetensors file could override the verified checkpoint; isolate model directory")
    controls = {"config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
                "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "vocab.json",
                "merges.txt", "tokenizer.model", "spiece.model", "adapter_config.json",
                "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json"}
    for name in controls:
        if (root / name).exists() and name not in rows:
            raise InventoryError("unlisted model/tokenizer loader file differs from inventory")
    if (root / "adapter_config.json").exists():
        raise InventoryError("unqualified adapter/base-model loading is forbidden")
    if "model.safetensors.index.json" in rows:
        row = rows["model.safetensors.index.json"]
        path = safe_file(root, row["path"])
        sha, _ = hash_file(path, row["size"])
        if sha != row["payload_sha256"]:
            raise InventoryError("model index differs from inventory")
        try:
            index = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
            weight_map = index["weight_map"]
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
            raise InventoryError("invalid verified model weight index") from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise InventoryError("nonempty safetensors weight map required")
        for name, target in weight_map.items():
            if (not isinstance(name, str) or not name or not isinstance(target, str)
                    or target not in rows or not target.endswith(".safetensors")):
                raise InventoryError("model index references an unverified weight file")


def verify_inventory(directory, expected_checkpoint_id=None, expected_repo=None, expected_revision=None,
                     verify_files=True):
    """Verify identity and optionally every byte. Return provenance with real scope."""
    if type(verify_files) is not bool:
        raise InventoryError("verify_files must be boolean")
    root = Path(directory).expanduser().resolve()
    marker = root / FILENAME
    if marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 8 * 1024 * 1024:
        raise InventoryError("verified model inventory is absent or invalid; rerun get_model.py")
    try:
        record = json.loads(marker.read_text(encoding="utf-8"), object_pairs_hook=_unique,
                            parse_constant=lambda _value: (_ for _ in ()).throw(InventoryError("nonfinite inventory")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InventoryError("cannot read verified model inventory") from exc
    if not isinstance(record, dict) or set(record) != set(_IDENTITY) | {"manifest_sha256", "checkpoint_id"}:
        raise InventoryError("invalid inventory completion fields")
    body = validate_body({key: record[key] for key in _IDENTITY})
    rebuilt = build_inventory(body["repo"], body["revision"], body["files"])
    if record != rebuilt:
        raise InventoryError("model inventory content identity mismatch")
    for expected, actual, label in ((expected_checkpoint_id, record["checkpoint_id"], "checkpoint"),
                                     (expected_repo, record["repo"], "repository"),
                                     (expected_revision, record["revision"], "revision")):
        if expected is not None and expected != actual:
            raise InventoryError(f"model inventory {label} differs from assignment")
    rows = {row["path"]: row for row in record["files"]}
    _verify_loader_files(root, rows)
    if verify_files:
        for row in record["files"]:
            path = safe_file(root, row["path"])
            sha, git = hash_file(path, row["size"])
            upstream = sha if row["algorithm"] == "sha256" else git
            if sha != row["payload_sha256"] or upstream != row["digest"]:
                raise InventoryError("model payload differs from fixed inventory")
    result = {**record, "payload_integrity_verified": verify_files,
            "config_sha256": rows["config.json"]["payload_sha256"],
            "index_sha256": rows.get("model.safetensors.index.json", {}).get("payload_sha256"),
            "scope": "actual complete file hashes" if verify_files else "inventory/config/index identity; weight payload unverified"}
    if not verify_files:
        return result
    # Bind the config used by a cohort builder to the raw bytes just verified,
    # rather than accepting a caller-supplied config or a storage-header digest.
    try:
        raw = safe_file(root, "config.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != result["config_sha256"]:
            raise InventoryError("model configuration changed during inventory verification")
        config = json.loads(raw, object_pairs_hook=_unique,
                            parse_constant=lambda _value: (_ for _ in ()).throw(InventoryError("nonfinite model configuration")))
        if not isinstance(config, dict):
            raise InventoryError("model configuration must be a JSON object")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InventoryError("cannot read verified model configuration") from exc
    return VerifiedInventory(result, _proof=_VERIFICATION_PROOF, _config=config)

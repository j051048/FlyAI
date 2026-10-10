"""Validate signed runtime declarations without importing an inference engine.

These are observations signed by a worker, not remote execution/hardware attestation.
Paths identify source files; private key paths and arbitrary environment values are
deliberately outside this schema.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath
import re
import uuid

SCHEMA = "shard-runtime-observation/1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def gpu_uuid_text(value):
    if isinstance(value, bytes) and len(value) == 16:
        return "GPU-" + str(uuid.UUID(bytes=value))
    if value is None or value == "":
        return None
    text = str(value)
    try:
        return "GPU-" + str(uuid.UUID(text))
    except ValueError:
        return text


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _text(value, name, maximum=4096, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise ValueError(f"invalid runtime observation {name}")


def _hash(value, name):
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError(f"invalid runtime observation {name}")


def validate_runtime_observation(value):
    keys = {"schema", "node_id", "gpu_uuid", "process_run_id", "runtime_config_sha256",
        "source_files", "source_sha256", "environment", "environment_sha256",
        "effective_flags", "effective_flags_sha256", "kernel_backend", "hadamard_backend",
        "graph_mode", "wire_mode", "transport", "versions", "phase"}
    if isinstance(value, dict) and "backend_identity" in value:
        keys.add("backend_identity")
    if isinstance(value, dict) and "optimizations" in value:
        keys.add("optimizations")
    if not isinstance(value, dict) or set(value) != keys or value.get("schema") != SCHEMA:
        raise ValueError("unknown runtime observation schema/fields")
    for name in ("node_id", "process_run_id", "kernel_backend", "hadamard_backend", "graph_mode", "wire_mode", "transport"):
        _text(value[name], name, maximum=256)
    _text(value["gpu_uuid"], "gpu_uuid", maximum=256, nullable=True)
    _hash(value["runtime_config_sha256"], "runtime_config_sha256")
    if value["phase"] not in ("loaded", "job_complete"):
        raise ValueError("invalid runtime observation phase")
    files = value["source_files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= 4096:
        raise ValueError("runtime source inventory must be bounded and nonempty")
    for path, sha in files.items():
        _text(path, "source path")
        pure = PurePosixPath(path)
        if pure.is_absolute() or "\\" in path or ":" in path or any(p in (".", "..") for p in path.split("/")):
            raise ValueError("runtime source path must be repository relative")
        _hash(sha, "source file SHA256")
    env = value["environment"]
    if not isinstance(env, dict) or len(env) > 256:
        raise ValueError("invalid runtime public environment")
    for name, setting in env.items():
        if not re.fullmatch(r"V4_[A-Z0-9_]+", name) or any(part in name for part in ("SECRET", "PASSWORD", "KEY_FILE", "TOKEN_FILE")):
            raise ValueError("private or invalid runtime environment name")
        if not isinstance(setting, str) or len(setting) > 4096 or "\x00" in setting:
            raise ValueError("invalid runtime environment value")
    flags = value["effective_flags"]
    if not isinstance(flags, dict) or len(flags) > 256:
        raise ValueError("invalid effective runtime flags")
    for name, row in flags.items():
        if not re.fullmatch(r"V4_[A-Z0-9_]+", name) or not isinstance(row, dict) or set(row) != {"requested", "parsed", "observed", "verdict", "reason"}:
            raise ValueError("invalid effective flag record")
        if row["requested"] is not None and (not isinstance(row["requested"], str) or len(row["requested"]) > 4096):
            raise ValueError("invalid requested runtime flag")
        for field in ("parsed", "observed", "verdict", "reason"):
            if not isinstance(row[field], str) or len(row[field]) > 4096:
                raise ValueError("invalid effective flag value")
    versions = value["versions"]
    if not isinstance(versions, dict) or set(versions) != {"python", "torch", "cuda", "tilelang"}:
        raise ValueError("invalid runtime versions")
    for name, version in versions.items():
        _text(version, name, maximum=256, nullable=True)
    if "backend_identity" in value:
        identity = value["backend_identity"]
        if not isinstance(identity, dict) or set(identity) != {"hadamard"}:
            raise ValueError("invalid backend identity")
        hadamard = identity["hadamard"]
        if not isinstance(hadamard, dict) or set(hadamard) != {"requested", "backend", "reason", "source_sha256", "module_files", "dependency_version"}:
            raise ValueError("invalid Hadamard backend identity fields")
        if hadamard["requested"] not in ("auto", "torch", "extension") or hadamard["backend"] not in ("torch", "extension"):
            raise ValueError("invalid Hadamard backend selection")
        if hadamard["backend"] != value["hadamard_backend"]:
            raise ValueError("Hadamard backend identity differs from its runtime observation")
        _text(hadamard["reason"], "Hadamard reason", maximum=256)
        _hash(hadamard["source_sha256"], "Hadamard function source SHA256")
        _text(hadamard["dependency_version"], "Hadamard dependency version", maximum=256, nullable=True)
        hadamard_files = hadamard["module_files"]
        if not isinstance(hadamard_files, list) or len(hadamard_files) > 16:
            raise ValueError("bounded Hadamard module identity required")
        for row in hadamard_files:
            if not isinstance(row, dict) or set(row) != {"name", "sha256"}:
                raise ValueError("invalid Hadamard module file identity")
            _text(row["name"], "Hadamard module filename", maximum=256)
            if "/" in row["name"] or "\\" in row["name"]:
                raise ValueError("Hadamard module identity must omit machine paths")
            _hash(row["sha256"], "Hadamard module SHA256")
    for field, body in (("source_sha256", files), ("environment_sha256", env), ("effective_flags_sha256", flags)):
        _hash(value[field], field)
        if value[field] != digest(body):
            raise ValueError(f"runtime observation {field} does not match its payload")
    # Reject non-JSON content and detach all caller-owned dictionaries before signing.
    if "optimizations" in value:
        optimizations = value["optimizations"]
        if not isinstance(optimizations, dict) or set(optimizations) != {"prefill_expert_pipeline", "conversation_cache"}:
            raise ValueError("unknown runtime optimization observations")
        if any(not isinstance(item, dict) for item in optimizations.values()) or len(json.dumps(optimizations, allow_nan=False)) > 65536:
            raise ValueError("bounded JSON runtime optimization observations required")
    json.dumps(value, allow_nan=False)
    return deepcopy(value)

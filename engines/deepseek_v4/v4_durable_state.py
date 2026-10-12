"""Private, bounded, node-local native-state checkpoints; JSON + safetensors only.

Callers hold the lane execution lock and bind the full numerical/source/lease
identity. Nothing is transferred across nodes, and a new lease is not authorized
by this storage helper. Publishing the manifest last makes interrupted writes
unusable. Hashes detect corruption, not dishonest computation by a worker.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import threading
import time
import uuid

from shard.job_journal import _private_path, _windows_acl, JournalError

SCHEMA = "v4-durable-native-state/1"
_PAYLOAD = re.compile(r"[0-9a-f]{64}-[0-9a-f]{32}\.safetensors\Z")
_MANIFEST = re.compile(r"[0-9a-f]{64}\.json\Z")
_MAX_JSON = 1 << 20


class DurableStateError(RuntimeError):
    pass


class DurableStateCapacity(DurableStateError):
    """Optional checkpoint capacity failure; never weakens integrity checks."""
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def _entry_key(entry_id):
    if not isinstance(entry_id, str) or not 1 <= len(entry_id) <= 512:
        raise DurableStateError("bounded native checkpoint identity required")
    return hashlib.sha256(entry_id.encode()).hexdigest()


class DurableStateStore:
    def __init__(self, directory, *, max_bytes, max_entries=4, ttl_s=300, clock=time.time):
        if type(max_bytes) is not int or max_bytes < 4096 or max_bytes > 2 ** 50:
            raise DurableStateError("explicit bounded checkpoint disk budget required")
        if type(max_entries) is not int or not 1 <= max_entries <= 1024:
            raise DurableStateError("bounded checkpoint entry count required")
        if type(ttl_s) not in (int, float) or not math.isfinite(ttl_s) or not 0 < ttl_s <= 86400:
            raise DurableStateError("bounded checkpoint TTL required")
        self.lock_path = _private_path(Path(directory) / ".state-lock")
        self.directory = self.lock_path.parent
        self.max_bytes, self.max_entries, self.ttl_s, self.clock = max_bytes, max_entries, ttl_s, clock
        self._lock = threading.RLock()
        self._counters = {"written": 0, "loaded": 0, "misses": 0, "corrupt": 0,
                          "write_seconds": 0.0, "read_seconds": 0.0}

    @contextmanager
    def _exclusive(self):
        with self._lock:
            if os.name == "nt":
                _windows_acl(self.directory)
                _windows_acl(self.lock_path)
            with self.lock_path.open("r+b") as stream:
                if os.name == "nt":
                    import msvcrt
                    if self.lock_path.stat().st_size == 0:
                        stream.write(b"0"); stream.flush()
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if os.name == "nt":
                        stream.seek(0)
                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _file(self, name):
        if not (_PAYLOAD.fullmatch(name) or _MANIFEST.fullmatch(name)):
            raise DurableStateError("invalid local checkpoint filename")
        path = self.directory / name
        if path.is_symlink() or path.exists() and getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
            raise DurableStateError("checkpoint cannot follow links")
        if path.exists() and os.name != "nt":
            info = path.stat()
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise DurableStateError("checkpoint files must remain private (0600)")
        elif path.exists():
            try:
                _windows_acl(path)
            except JournalError as error:
                raise DurableStateError("checkpoint file DACL must remain private") from error
        return path

    def _read(self, path):
        if path.stat().st_size > _MAX_JSON:
            raise DurableStateError("checkpoint manifest exceeds metadata budget")
        value = json.loads(path.read_bytes())
        fields = {"schema", "entry_id", "identity", "metadata", "tensors", "payload", "bytes",
                  "payload_sha256", "created_at", "expires_at", "sha256"}
        if (not isinstance(value, dict) or set(value) != fields or value["schema"] != SCHEMA or
                value["sha256"] != hashlib.sha256(canonical({k: v for k, v in value.items() if k != "sha256"})).hexdigest() or
                path.name != _entry_key(value["entry_id"]) + ".json"):
            raise DurableStateError("invalid checkpoint manifest binding")
        if (type(value["bytes"]) is not int or not 0 < value["bytes"] <= self.max_bytes or
                any(type(value[k]) not in (int, float) or not math.isfinite(value[k]) for k in ("created_at", "expires_at")) or
                not 0 < value["expires_at"] - value["created_at"] <= self.ttl_s or
                not isinstance(value["identity"], dict) or not isinstance(value["metadata"], dict)):
            raise DurableStateError("invalid checkpoint size, identity or lifetime")
        payload = self._file(value["payload"])
        if not payload.name.startswith(path.stem + "-"):
            raise DurableStateError("checkpoint payload belongs to another entry")
        return value

    def _sync_directory(self):
        if os.name != "nt":
            fd = os.open(self.directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def _remove(self, path, value=None):
        # Remove authority first. A crash cannot publish a half-removed entry.
        path.unlink(missing_ok=True)
        if value is not None:
            self._file(value["payload"]).unlink(missing_ok=True)

    def _inventory(self):
        rows, keep = [], set()
        for path in self.directory.glob("*.json"):
            if not _MANIFEST.fullmatch(path.name):
                continue
            path = self._file(path.name)
            try:
                value = self._read(path)
                if not value["created_at"] <= self.clock() < value["expires_at"]:
                    self._remove(path, value)
                    continue
                payload = self._file(value["payload"])
                if not payload.is_file() or payload.stat().st_size != value["bytes"]:
                    raise DurableStateError("missing or truncated checkpoint payload")
                keep.add(payload.name)
                rows.append((path, value))
            except (OSError, ValueError, KeyError, TypeError, DurableStateError):
                self._counters["corrupt"] += 1
                path.unlink(missing_ok=True)
        # Files from interrupted generations have no manifest authority.
        for path in self.directory.glob("*.safetensors"):
            if _PAYLOAD.fullmatch(path.name) and path.name not in keep:
                self._file(path.name).unlink(missing_ok=True)
        for path in self.directory.glob("*.tmp"):
            if re.fullmatch(r"[0-9a-f]{64}\.tmp", path.name):
                if path.is_symlink():
                    raise DurableStateError("checkpoint temporary file cannot follow a link")
                path.unlink(missing_ok=True)
        return rows

    def put(self, entry_id, identity, tensors, metadata):
        import torch
        from safetensors.torch import save_file
        started = time.perf_counter()
        if not isinstance(identity, dict) or not identity or not isinstance(metadata, dict):
            raise DurableStateError("exact runtime identity and checkpoint metadata required")
        if len(canonical({"identity": identity, "metadata": metadata})) > _MAX_JSON // 2:
            raise DurableStateError("checkpoint identity/metadata exceeds budget")
        identity, metadata = json.loads(canonical(identity)), json.loads(canonical(metadata))
        if (not isinstance(tensors, dict) or not 1 <= len(tensors) <= 4096 or any(
                not isinstance(name, str) or not 1 <= len(name) <= 256 or not torch.is_tensor(t) or
                t.device.type != "cpu" or not t.is_contiguous() for name, t in tensors.items())):
            raise DurableStateError("contiguous detached CPU state tensors required")
        raw_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        description = {name: {"shape": list(t.shape), "dtype": str(t.dtype)} for name, t in tensors.items()}
        metadata_bytes = len(canonical({"identity": identity, "metadata": metadata}))
        if len(canonical(description)) + metadata_bytes > _MAX_JSON - 4096:
            raise DurableStateError("checkpoint tensor manifest exceeds metadata budget")
        reserve = raw_bytes + len(canonical(description)) * 3 + metadata_bytes + 4096
        if reserve > self.max_bytes:
            raise DurableStateCapacity("checkpoint exceeds disk/staging quota")
        key = _entry_key(entry_id)
        with self._exclusive():
            rows = self._inventory()
            # Free space excludes existing generations. This bounds the new
            # temporary payload, without pretending to reserve the whole volume.
            if shutil.disk_usage(self.directory).free < reserve:
                raise DurableStateCapacity("insufficient actual free disk for atomic checkpoint staging")
            original = next((value for path, value in rows if path.stem == key), None)
            used = sum(value["bytes"] + path.stat().st_size for path, value in rows)
            for path, value in sorted(rows, key=lambda row: row[1]["created_at"]):
                if used + reserve <= self.max_bytes and len(rows) < self.max_entries + bool(original):
                    break
                if path.stem == key:
                    continue
                used -= value["bytes"] + path.stat().st_size
                rows.remove((path, value)); self._remove(path, value)
            if used + reserve > self.max_bytes:
                raise DurableStateCapacity("atomic checkpoint replacement exceeds peak disk quota")
            payload = self._file(key + "-" + uuid.uuid4().hex + ".safetensors")
            manifest = self._file(key + ".json")
            temporary = manifest.with_suffix(".tmp")
            try:
                fd = os.open(payload, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600); os.close(fd)
                save_file({name: tensor.detach() for name, tensor in tensors.items()}, str(payload))
                with payload.open("r+b") as stream:
                    os.fsync(stream.fileno())
                now = self.clock()
                value = {"schema": SCHEMA, "entry_id": entry_id, "identity": identity, "metadata": metadata,
                         "tensors": description, "payload": payload.name, "bytes": payload.stat().st_size,
                         "payload_sha256": _sha(payload), "created_at": now, "expires_at": now + self.ttl_s}
                value["sha256"] = hashlib.sha256(canonical(value)).hexdigest()
                if used + value["bytes"] + len(canonical(value)) > self.max_bytes:
                    raise DurableStateCapacity("checkpoint serialized footprint exceeds quota")
                fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, manifest)
                self._sync_directory()
                if original:
                    self._file(original["payload"]).unlink(missing_ok=True)
                self._counters["written"] += 1
                return {"entry_id": entry_id, "payload_sha256": value["payload_sha256"], "bytes": value["bytes"]}
            except Exception:
                # Published manifests remain valid; uncommitted payloads are not reusable.
                if not manifest.exists() or self._read(manifest)["payload"] != payload.name:
                    payload.unlink(missing_ok=True)
                raise
            finally:
                temporary.unlink(missing_ok=True)
                self._counters["write_seconds"] += time.perf_counter() - started

    def get(self, entry_id, identity):
        from safetensors.torch import load_file
        started = time.perf_counter()
        with self._exclusive():
            self._inventory()
            path = self._file(_entry_key(entry_id) + ".json")
            if not path.exists():
                self._counters["misses"] += 1
                return None
            try:
                value = self._read(path)
                if value["identity"] != json.loads(canonical(identity)):
                    self._counters["misses"] += 1
                    return None
                payload = self._file(value["payload"])
                if _sha(payload) != value["payload_sha256"]:
                    raise DurableStateError("checkpoint payload checksum differs")
                tensors = load_file(str(payload), device="cpu")
                if {name: {"shape": list(t.shape), "dtype": str(t.dtype)} for name, t in tensors.items()} != value["tensors"]:
                    raise DurableStateError("checkpoint tensor layout differs")
                self._counters["loaded"] += 1
                return tensors, value["metadata"]
            except Exception as error:
                self._counters["corrupt"] += 1
                self._remove(path, value if "value" in locals() else None)
                raise DurableStateError("native checkpoint rejected") from error
            finally:
                self._counters["read_seconds"] += time.perf_counter() - started

    def remove(self, entry_id):
        with self._exclusive():
            self._inventory()
            path = self._file(_entry_key(entry_id) + ".json")
            if path.exists():
                self._remove(path, self._read(path)); self._sync_directory()

    def status(self):
        with self._exclusive():
            rows = self._inventory()
            return {"schema": SCHEMA, "entries": len(rows), "disk_bytes": sum(
                path.stat().st_size + value["bytes"] for path, value in rows),
                "max_bytes": self.max_bytes, "max_entries": self.max_entries, **self._counters,
                "scope": "node-local exact-identity checkpoints; no cross-lease recovery or HA"}

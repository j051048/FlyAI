"""Opt-in, locally provisioned activation slots; the WAN carries no IPC handles.

The capability in a frame selects a manifest in a PRIVATE operator directory.
The manifest is authenticated with a host-local key and bound to the exact
channel, model cohort, request, owner epoch and slot generation. No pickle,
network pathname, public IP inference or remotely supplied CUDA handle is used.

CUDA IPC copies directly into the receiving worker's owned tensor. Receipt
hashing still examines real bytes on CPU: this is not an end-to-end zero-copy
claim. Every slot also has a bounded shared-memory witness for safe IPC fallback.
The explicit CPU reference backend exercises real cross-process shared memory;
it cannot qualify CUDA, peer access or speed.
"""
from collections import deque
from contextlib import nullcontext
import base64
import ctypes
from ctypes import c_int, c_size_t, c_uint, c_uint64, c_void_p
import hashlib
import hmac
import json
import math
import mmap
from multiprocessing import shared_memory
import os
from pathlib import Path
import re
import secrets
import stat
import threading
import time

import torch

SCHEMA = "v4-host-activation/1"
CONFIG_SCHEMA = "v4-host-transport-config/1"
_HEX = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_MAX_BYTES = 1 << 30
_DTYPES = {str(value): value for value in (torch.bfloat16, torch.float16, torch.float32,
    torch.float8_e4m3fn, torch.float8_e5m2)}


class HostTransportError(ValueError):
    pass


class _Unavailable(RuntimeError):
    pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _shared_bytes(name, count):
    """Read a producer-owned mapping without taking unlink ownership.

    Python <3.13's POSIX SharedMemory attachment registers with the reader's
    resource tracker. Independent serving processes would then unlink another
    worker's slot on exit. Open the POSIX descriptor read-only instead; only the
    allocating owner calls unlink. Windows mappings are handle-counted by the OS.
    """
    if os.name == "posix":
        import _posixshmem
        descriptor = _posixshmem.shm_open("/" + name, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0), mode=0o600)
        try:
            info = os.fstat(descriptor)
            if info.st_size < count or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise HostTransportError("local activation mapping size/owner/private mode differs")
            with mmap.mmap(descriptor, count, access=mmap.ACCESS_READ) as memory:
                return memory[:count]
        finally:
            os.close(descriptor)
    memory = shared_memory.SharedMemory(name=name)
    try:
        if memory.size < count:
            raise HostTransportError("local activation shared memory is smaller than its manifest")
        return bytes(memory.buf[:count])
    finally:
        memory.close()


def request_binding(owner_epoch, job, *, start_pos, frame_epoch=0):
    if not isinstance(job, dict):
        raise HostTransportError("local activation requires a complete request object")
    body = {"owner_epoch": owner_epoch, "job_id": job.get("job_id"), "nonce": job.get("nonce"),
            "swarm_id": job.get("swarm_id"), "start_pos": start_pos, "frame_epoch": frame_epoch}
    if (any(not isinstance(body[key], str) or not 1 <= len(body[key]) <= 256
            for key in ("owner_epoch", "job_id", "swarm_id"))
            or not isinstance(body["nonce"], str) or not _HEX.fullmatch(body["nonce"])
            or any(type(body[key]) is not int or not 0 <= body[key] <= 1_000_000_000
                   for key in ("start_pos", "frame_epoch"))):
        raise HostTransportError("local activation requires a live exact request/owner binding")
    return body


def stage_binding(stage, message):
    owner = getattr(stage, "_conversation_session_epoch", None)
    job = getattr(stage, "_conversation_job_binding", None)
    return request_binding(owner, job or {}, start_pos=message["start_pos"],
                           frame_epoch=message.get("epoch", 0))


def _read(path):
    # Reject symlinks even in the private directory. A frame never supplies path.
    if path.is_symlink():
        raise HostTransportError("local transport manifest cannot be a symlink")
    with path.open("rb") as handle:
        raw = handle.read(65537)
    if len(raw) > 65536:
        raise HostTransportError("local transport manifest exceeds bounded metadata")
    try:
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate local transport metadata field")
                result[key] = value
            return result
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError) as error:
        raise HostTransportError("invalid local transport metadata") from error
    if not isinstance(value, dict):
        raise HostTransportError("local transport metadata must be an object")
    return value


def _private(path, *, directory=False):
    if path.is_symlink() or (directory and not path.is_dir()) or (not directory and not path.is_file()):
        raise HostTransportError("operator transport path must be a real local file/directory")
    if os.name == "posix":
        s = path.stat()
        if s.st_uid != os.geteuid() or stat.S_IMODE(s.st_mode) & 0o077:
            raise HostTransportError("host transport paths must be owned by this user and private (0700/0600)")


def load_config(path=None):
    path = path if path is not None else os.environ.get("V4_HOST_TRANSPORT_CONFIG", "")
    if not path:
        return None
    location = Path(path).absolute()
    _private(location)
    body = _read(location)
    return _validate_config(body)


def _validate_config(body):
    fields = {"schema", "host_id", "directory", "secret", "backend", "max_bytes", "slots", "ttl_s", "channels"}
    if set(body) not in (fields, fields | {"digest_backend"}) or body["schema"] != CONFIG_SCHEMA:
        raise HostTransportError("invalid host transport configuration schema")
    if body.get("digest_backend", "cpu") not in ("cpu", "device_sha256"):
        raise HostTransportError("host digest backend must be cpu or device_sha256")
    if body.get("digest_backend") == "device_sha256" and body.get("backend") != "cuda_ipc":
        raise HostTransportError("device digest requires CUDA IPC; no reference qualification")
    if (not isinstance(body["host_id"], str) or not _ID.fullmatch(body["host_id"])
            or body["backend"] not in ("cuda_ipc", "shared_memory_reference")
            or not isinstance(body["secret"], str) or not _HEX.fullmatch(body["secret"])
            or type(body["max_bytes"]) is not int or not 4096 <= body["max_bytes"] <= _MAX_BYTES
            or type(body["slots"]) is not int or not 1 <= body["slots"] <= 64
            or body["max_bytes"] // body["slots"] < 1024
            or type(body["ttl_s"]) not in (int, float) or not math.isfinite(body["ttl_s"])
            or not 1 <= body["ttl_s"] <= 300
            or not isinstance(body["channels"], list) or not 1 <= len(body["channels"]) <= 64):
        raise HostTransportError("host transport requires explicit bounded capacity and host identity")
    directory = Path(body["directory"])
    if not directory.is_absolute():
        raise HostTransportError("host transport directory must be an absolute operator path")
    _private(directory, directory=True)
    seen = set()
    for channel in body["channels"]:
        if (not isinstance(channel, dict) or set(channel) != {"id", "cohort_id", "src_pubkey", "dst_pubkey", "src_uuid", "dst_uuid"}
                or not isinstance(channel["id"], str) or not _ID.fullmatch(channel["id"])
                or channel["id"] in seen or not isinstance(channel["cohort_id"], str)
                or not _HEX.fullmatch(channel["cohort_id"])
                or any(not isinstance(channel[key], str) or not 1 <= len(channel[key]) <= 128
                       for key in ("src_uuid", "dst_uuid"))):
            raise HostTransportError("invalid pinned host channel")
        for name in ("src_pubkey", "dst_pubkey"):
            try:
                if len(base64.b64decode(channel[name], validate=True)) != 32:
                    raise ValueError()
            except (ValueError, TypeError) as error:
                raise HostTransportError("host channels require pinned Ed25519 peers") from error
        seen.add(channel["id"])
    return body


def public_config(path=None):
    config = load_config(path)
    if config is None:
        return {"enabled": False}
    public = {key: value for key, value in config.items() if key not in ("secret", "directory")}
    # Pin the credential identity without exposing the credential or path.
    public["key_id"] = _sha(bytes.fromhex(config["secret"]))
    return {"enabled": True, "schema": CONFIG_SCHEMA, "config_sha256": _sha(_canonical(public)),
            "host_id": config["host_id"], "backend": config["backend"],
            "max_bytes": config["max_bytes"], "slots": config["slots"], "ttl_s": config["ttl_s"],
            "channels": config["channels"], "performance_qualified": False,
            **({"digest_backend": config["digest_backend"]} if "digest_backend" in config else {})}


def _write(directory, name, body, key):
    data = {**body, "mac": hmac.new(key, _canonical(body), hashlib.sha256).hexdigest()}
    temporary = directory / ("tmp-" + secrets.token_hex(16))
    try:
        with temporary.open("xb") as handle:
            if os.name == "posix":
                os.chmod(temporary, 0o600)
            handle.write(_canonical(data))
        os.replace(temporary, directory / name)
    finally:
        if temporary.exists():
            temporary.unlink()


def _authenticated(path, key):
    value = _read(path)
    mac = value.pop("mac", None)
    if not isinstance(mac, str) or not hmac.compare_digest(mac, hmac.new(key, _canonical(value), hashlib.sha256).hexdigest()):
        raise HostTransportError("local transport authentication rejected")
    return value


class _Driver:
    """Small explicit driver seam; CUDA memory is NOT a Torch allocator view."""
    class Handle(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * 64)]

    def __init__(self):
        if os.name != "posix":
            raise _Unavailable("CUDA IPC backend currently requires Linux; shared-memory fallback remains available")
        try:
            self.lib = ctypes.CDLL("libcuda.so.1")
            self.alloc = self._function("cuMemAlloc_v2", [ctypes.POINTER(c_uint64), c_size_t])
            self.free = self._function("cuMemFree_v2", [c_uint64])
            self.export = self._function("cuIpcGetMemHandle", [ctypes.POINTER(self.Handle), c_uint64])
            name = "cuIpcOpenMemHandle_v2" if hasattr(self.lib, "cuIpcOpenMemHandle_v2") else "cuIpcOpenMemHandle"
            self.open = self._function(name, [ctypes.POINTER(c_uint64), self.Handle, c_uint])
            self.close = self._function("cuIpcCloseMemHandle", [c_uint64])
            self.copy = self._function("cuMemcpyDtoDAsync_v2", [c_uint64, c_uint64, c_size_t, c_void_p])
        except (OSError, AttributeError) as error:
            raise _Unavailable("CUDA driver IPC symbols unavailable") from error

    def _function(self, name, arguments):
        function = getattr(self.lib, name)
        function.argtypes, function.restype = arguments, c_int
        return function

    @staticmethod
    def checked(code, operation):
        if code:
            raise _Unavailable(f"CUDA IPC {operation} returned status {code}")

    def allocate(self, size):
        ptr, handle = c_uint64(), self.Handle()
        self.checked(self.alloc(ctypes.byref(ptr), size), "allocate")
        try:
            self.checked(self.export(ctypes.byref(handle), ptr), "export")
        except BaseException:
            self.free(ptr)
            raise
        return ptr.value, base64.b64encode(bytes(handle)).decode()

    def receive(self, encoded, target, count, stream):
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as error:
            raise HostTransportError("invalid locally authenticated CUDA handle") from error
        if len(raw) != 64:
            raise HostTransportError("invalid locally authenticated CUDA handle length")
        handle = self.Handle.from_buffer_copy(raw)
        ptr = c_uint64()
        self.checked(self.open(ctypes.byref(ptr), handle, 1), "open")
        try:
            self.checked(self.copy(target.data_ptr(), ptr.value, count, c_void_p(stream.cuda_stream)), "copy")
            stream.synchronize()  # copy must finish before acknowledgement/close/reuse
        except BaseException:
            try:
                stream.synchronize()
            except RuntimeError as error:
                raise HostTransportError("unsafe CUDA IPC copy stream; owner cannot be acknowledged") from error
            raise
        finally:
            if self.close(ptr):
                raise HostTransportError("CUDA IPC mapping close failed; owner cannot be acknowledged")


class HostTransport:
    def __init__(self, config, *, local_pubkey, local_uuid, cohort_id, device="cpu", clock=time.time):
        config = _validate_config(json.loads(_canonical(config)))
        self.config = config
        self.directory, self.key = Path(config["directory"]), bytes.fromhex(config["secret"])
        self.local_pubkey, self.local_uuid, self.cohort_id = local_pubkey, local_uuid, cohort_id
        self.device, self.clock = torch.device(device), clock
        self.public = public_config_from_body(config)
        self.outgoing = [c for c in config["channels"] if c["src_pubkey"] == local_pubkey and c["cohort_id"] == cohort_id]
        self.incoming = [c for c in config["channels"] if c["dst_pubkey"] == local_pubkey and c["cohort_id"] == cohort_id]
        if len(self.outgoing) > 1 or len(self.incoming) > 1:
            raise HostTransportError("a pipeline worker has at most one pinned local predecessor/successor")
        for channel in self.outgoing:
            if channel["src_uuid"] != local_uuid:
                raise HostTransportError("source GPU identity differs from private host channel")
        for channel in self.incoming:
            if channel["dst_uuid"] != local_uuid:
                raise HostTransportError("destination GPU identity differs from private host channel")
        self.reference = config["backend"] == "shared_memory_reference"
        if self.reference != (self.device.type == "cpu"):
            raise HostTransportError("CPU shared-memory reference cannot impersonate the CUDA backend")
        self.driver, self.backend_reason = None, None
        if not self.reference:
            if not torch.cuda.is_available():
                raise HostTransportError("CUDA host transport requires an actual accelerator")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            from v4_observability import gpu_uuid_text
            observed = gpu_uuid_text(getattr(torch.cuda.get_device_properties(self.device), "uuid", None))
            if not observed or observed.casefold() != local_uuid.casefold():
                raise HostTransportError("actual GPU UUID differs from private host channel")
            try:
                with torch.cuda.device(self.device):
                    torch.empty(1, device=self.device)  # establish this worker's context
                    self.driver = _Driver()
            except _Unavailable as error:
                self.backend_reason = str(error)
        self.slot_bytes = config["max_bytes"] // config["slots"]
        self.device_digest_enabled = config.get("digest_backend", "cpu") == "device_sha256"
        if self.device_digest_enabled:
            if self.driver is None:
                raise HostTransportError("device SHA256 requires actual CUDA IPC; no host fallback")
            # Build and check real CUDA bytes before publishing readiness.
            from v4_device_digest import tensor_digest
            for n in (0, 55, 56, 63, 64, 65, 129):
                sample = bytes(i % 256 for i in range(n))
                if tensor_digest([sample], device=self.device) != hashlib.sha256(sample).digest():
                    raise HostTransportError("CUDA SHA256 startup known-answer test failed")
        self._slots = []
        self._seen, self._seen_set = deque(), set()
        self._closed, self._lock, self._generation = False, threading.RLock(), 0
        self._poisoned = False
        self._counts = {name: 0 for name in ("published", "received", "discarded", "ipc_copies", "shared_memory_copies",
            "ipc_fallbacks", "socket_fallbacks", "reclaimed", "quarantined", "rejected")}
        self._boot_id = secrets.token_hex(32)
        self.host_compute = None
        self.touch()

    def touch(self):
        """An idle peer expires; its first ordinary frame refreshes availability."""
        if self._closed:
            return
        for channel in self.incoming:
            body = {"schema": SCHEMA, "config_sha256": self.public["config_sha256"],
                    "channel": channel, "boot_id": self._boot_id, "ipc": self.driver is not None,
                    "expires_at": self.clock() + self.config["ttl_s"]}
            _write(self.directory, "ready-" + channel["id"] + ".json", body, self.key)

    def _peer(self):
        path = self.directory / ("ready-" + self.outgoing[0]["id"] + ".json")
        if not path.exists():
            return None
        body = _authenticated(path, self.key)
        if (set(body) != {"schema", "config_sha256", "channel", "boot_id", "ipc", "expires_at"}
                or body["schema"] != SCHEMA or body["config_sha256"] != self.public["config_sha256"]
                or body["channel"] != self.outgoing[0] or type(body["ipc"]) is not bool
                or not isinstance(body["boot_id"], str) or not _HEX.fullmatch(body["boot_id"])
                or type(body["expires_at"]) not in (float, int) or not math.isfinite(body["expires_at"])
                or not 0 < body["expires_at"] - self.clock() <= self.config["ttl_s"]):
            return None
        return body

    def _manifest(self, capability):
        return self.directory / ("slot-" + capability + ".json")

    def _ack(self, capability):
        return self.directory / ("ack-" + capability + ".json")

    def _retire(self, slot):
        if self._poisoned:
            return False
        if not slot.get("capability"):
            if slot["event"] is not None:
                slot["event"].synchronize()
                slot["event"] = None
            return True
        path = self._ack(slot["capability"])
        if not path.exists():
            return False  # expiry/cancellation NEVER permits an unacknowledged GPU overwrite
        ack = _authenticated(path, self.key)
        if ack != {"schema": SCHEMA, "capability": slot["capability"], "generation": slot["generation"],
                   "binding_sha256": slot["binding_sha256"], "sha256": slot["sha256"]}:
            raise HostTransportError("local activation acknowledgement differs from its owner")
        if slot["event"] is not None:
            # A fenced consumer may discard before the producer's copy finishes.
            slot["event"].synchronize()
            slot["event"] = None
        path.unlink()
        self._manifest(slot["capability"]).unlink(missing_ok=True)
        slot["capability"] = None
        self._counts["reclaimed"] += 1
        return True

    def _slot(self):
        for slot in self._slots:
            if self._retire(slot):
                return slot
        if len(self._slots) == self.config["slots"]:
            return None
        memory = shared_memory.SharedMemory(create=True, size=self.slot_bytes)
        slot = {"memory": memory, "capability": None, "ptr": None, "handle": None, "event": None}
        try:
            if self.driver is not None:
                with torch.cuda.device(self.device):
                    slot["ptr"], slot["handle"] = self.driver.allocate(self.slot_bytes)
        except _Unavailable:
            self._counts["ipc_fallbacks"] += 1  # real bounded shared-memory path remains usable
        except BaseException:
            memory.close(); memory.unlink()
            raise
        self._slots.append(slot)
        return slot

    def publish(self, tensor, binding, *, witness=None):
        with (self.host_compute.copy() if self.host_compute is not None else nullcontext()):
            return self._publish(tensor, binding, witness=witness)

    def frame_digest(self, parts):
        """Read actual local tensor bytes; a manifest's claimed hash is insufficient."""
        if not self.device_digest_enabled or self._closed or self._poisoned:
            raise HostTransportError("device digest is not available on this live channel")
        from v4_device_digest import tensor_digest, ActualByteDigest
        return ActualByteDigest(tensor_digest(parts, device=self.device,
            max_bytes=3 * self.slot_bytes + 65536))

    def _publish(self, tensor, binding, *, witness=None):
        """Return a small capability, or None to retain the original socket payload."""
        with self._lock:
            if self._closed or self._poisoned:
                raise HostTransportError("host transport is closed or its CUDA ownership is poisoned")
            if not self.outgoing or binding is None:
                self._counts["socket_fallbacks"] += 1
                return None
            _validate_binding(binding)
            peer = self._peer()
            if peer is None:
                self._counts["socket_fallbacks"] += 1
                return None
            if (not torch.is_tensor(tensor) or str(tensor.dtype) not in _DTYPES or tensor.ndim != 4
                    or any(int(n) < 1 for n in tensor.shape) or tensor.device != self.device):
                raise HostTransportError("local activations require owned contiguous-compatible B,S,HC,D tensors")
            count = tensor.numel() * tensor.element_size()
            if count > self.slot_bytes:
                self._counts["socket_fallbacks"] += 1
                return None
            slot = self._slot()
            if slot is None:
                self._counts["socket_fallbacks"] += 1
                return None
            if self.device_digest_enabled:
                if slot["ptr"] is None or not peer["ipc"]:
                    raise HostTransportError("device digest channel requires an IPC-capable peer")
                checksum = self.frame_digest([tensor]).digest.hex()
            else:
                if witness is None:
                    witness = tensor.detach().cpu().contiguous()
                if witness.device.type != "cpu" or witness.dtype != tensor.dtype or witness.shape != tensor.shape:
                    raise HostTransportError("receipt witness differs from local activation layout")
                raw = _bytes(witness)
                slot["memory"].buf[:count] = raw
                checksum = _sha(raw)
            handle, ready = None, None
            if slot["ptr"] is not None and peer["ipc"]:
                try:
                    with torch.cuda.device(self.device):
                        stream = torch.cuda.current_stream(self.device)
                        source = tensor.detach().contiguous()
                        self.driver.checked(self.driver.copy(slot["ptr"], source.data_ptr(), count,
                            c_void_p(stream.cuda_stream)), "publish copy")
                        source.record_stream(stream)
                        event = torch.cuda.Event(interprocess=True)
                        event.record(stream)
                        ready = base64.b64encode(event.ipc_handle()).decode()
                        slot["event"] = event  # must stay alive until receiver acknowledges
                        handle = slot["handle"]
                except BaseException:
                    # Fence a submitted copy if creation/export of its ownership
                    # event failed. Admission refusal never reaches this branch.
                    try:
                        torch.cuda.synchronize(self.device)
                    except RuntimeError as error:
                        self._poisoned = True
                        raise HostTransportError("failed local publish has an unsafe CUDA stream; allocations quarantined") from error
                    raise
            self._generation += 1
            capability = secrets.token_hex(32)
            body = {"schema": SCHEMA, "host_id": self.config["host_id"],
                "config_sha256": self.public["config_sha256"], "channel": self.outgoing[0],
                "capability": capability, "generation": self._generation,
                "receiver_boot_id": peer["boot_id"],
                "binding_sha256": _sha(_canonical(binding)), "shape": list(tensor.shape),
                "dtype": str(tensor.dtype), "nbytes": count, "sha256": checksum,
                "memory_name": slot["memory"].name, "handle": handle, "ready": ready,
                "expires_at": self.clock() + self.config["ttl_s"]}
            _write(self.directory, self._manifest(capability).name, body, self.key)
            slot.update(capability=capability, generation=self._generation,
                        binding_sha256=body["binding_sha256"], sha256=body["sha256"])
            self._counts["published"] += 1
            return {name: body[name] for name in ("schema", "capability", "generation", "binding_sha256")}

    def receive(self, reference, binding):
        with self._lock:
            try:
                with (self.host_compute.copy() if self.host_compute is not None else nullcontext()):
                    return self._receive(reference, binding)
            except (HostTransportError, OSError, RuntimeError):
                self._counts["rejected"] += 1
                raise

    def _receive(self, reference, binding):
        if self._closed or self._poisoned or not self.incoming:
            raise HostTransportError("local activation on an unconfigured pinned channel")
        _validate_binding(binding)
        if (not isinstance(reference, dict) or set(reference) != {"schema", "capability", "generation", "binding_sha256"}
                or reference["schema"] != SCHEMA or not isinstance(reference["capability"], str)
                or not _HEX.fullmatch(reference["capability"]) or type(reference["generation"]) is not int
                or not 1 <= reference["generation"] < 2 ** 63
                or reference["binding_sha256"] != _sha(_canonical(binding))):
            raise HostTransportError("local activation capability/request/owner rejected")
        capability = reference["capability"]
        if capability in self._seen_set:
            raise HostTransportError("local activation replay rejected")
        body = _authenticated(self._manifest(capability), self.key)
        fields = {"schema", "host_id", "config_sha256", "channel", "capability", "generation", "receiver_boot_id", "binding_sha256",
                  "shape", "dtype", "nbytes", "sha256", "memory_name", "handle", "ready", "expires_at"}
        if (set(body) != fields or any(body.get(k) != v for k, v in reference.items())
                or body["host_id"] != self.config["host_id"] or body["config_sha256"] != self.public["config_sha256"]
                or body["channel"] != self.incoming[0] or body["receiver_boot_id"] != self._boot_id
                or type(body["expires_at"]) not in (int, float)
                or not math.isfinite(body["expires_at"]) or not 0 < body["expires_at"] - self.clock() <= self.config["ttl_s"]
                or body["dtype"] not in _DTYPES or not isinstance(body["shape"], list) or len(body["shape"]) != 4
                or any(type(n) is not int or not 1 <= n <= 1_000_000 for n in body["shape"])
                or type(body["nbytes"]) is not int or not 0 < body["nbytes"] <= self.slot_bytes
                or not isinstance(body["sha256"], str) or not _HEX.fullmatch(body["sha256"])
                or not isinstance(body["memory_name"], str) or not _ID.fullmatch(body["memory_name"])):
            raise HostTransportError("local activation manifest identity/shape/quota/expiry rejected")
        dtype = _DTYPES[body["dtype"]]
        if math.prod(body["shape"]) * torch.empty((), dtype=dtype, device="cpu").element_size() != body["nbytes"]:
            raise HostTransportError("local activation byte geometry differs")
        result = None
        if self.driver is not None and body["handle"] is not None:
            try:
                with torch.cuda.device(self.device):
                    result = torch.empty(body["shape"], dtype=dtype, device=self.device)
                    try:
                        ready_raw = base64.b64decode(body["ready"], validate=True)
                    except (ValueError, TypeError) as error:
                        raise HostTransportError("invalid locally authenticated IPC event") from error
                    if len(ready_raw) != 64:
                        raise HostTransportError("invalid locally authenticated IPC event length")
                    try:
                        event = torch.cuda.Event.from_ipc_handle(self.device, ready_raw)
                    except RuntimeError as error:
                        raise _Unavailable("peer IPC event import unsupported") from error
                    stream = torch.cuda.current_stream(self.device)
                    stream.wait_event(event)
                    self.driver.receive(body["handle"], result, body["nbytes"], stream)
                self._counts["ipc_copies"] += 1
            except _Unavailable:
                result = None
                self._counts["ipc_fallbacks"] += 1
            except (RuntimeError, HostTransportError):
                # A failed IPC event/copy/mapping close has no safe reusable
                # GPU ownership proof. Never acknowledge or continue this owner.
                self._poisoned = True
                raise
        if result is None:
            if self.device_digest_enabled:
                raise HostTransportError("device digest IPC failed; no populated host witness exists")
            raw = _shared_bytes(body["memory_name"], body["nbytes"])
            if _sha(raw) != body["sha256"]:
                raise HostTransportError("local activation shared-memory bytes changed")
            result = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(body["shape"]).clone().to(self.device)
            self._counts["shared_memory_copies"] += 1
        # Hash actual destination bytes. A descriptor/hash alone is not receipt evidence.
        observed = self.frame_digest([result]).digest.hex() if self.device_digest_enabled else _sha(_bytes(result))
        if observed != body["sha256"]:
            raise HostTransportError("local activation destination bytes changed")
        ack = {name: body[name] for name in ("schema", "capability", "generation", "binding_sha256", "sha256")}
        _write(self.directory, self._ack(capability).name, ack, self.key)
        self._seen.append(capability); self._seen_set.add(capability)
        while len(self._seen) > 2 * self.config["slots"]:
            self._seen_set.remove(self._seen.popleft())
        self._counts["received"] += 1
        self.touch()
        return result

    def discard(self, reference, binding):
        """A fenced frame is acknowledged without reading or computing its data."""
        with self._lock:
            _validate_binding(binding)
            if (self._closed or not self.incoming or not isinstance(reference, dict)
                    or set(reference) != {"schema", "capability", "generation", "binding_sha256"}
                    or reference.get("schema") != SCHEMA or not isinstance(reference.get("capability"), str)
                    or not _HEX.fullmatch(reference["capability"])
                    or reference.get("binding_sha256") != _sha(_canonical(binding))):
                raise HostTransportError("fenced local activation does not match its request")
            capability = reference["capability"]
            if capability in self._seen_set:
                return
            body = _authenticated(self._manifest(capability), self.key)
            if (any(body.get(k) != value for k, value in reference.items())
                    or body.get("config_sha256") != self.public["config_sha256"]
                    or body.get("channel") != self.incoming[0] or body.get("receiver_boot_id") != self._boot_id):
                raise HostTransportError("fenced local activation belongs to another channel/owner")
            ack = {name: body[name] for name in ("schema", "capability", "generation", "binding_sha256", "sha256")}
            _write(self.directory, self._ack(capability).name, ack, self.key)
            self._seen.append(capability); self._seen_set.add(capability)
            while len(self._seen) > 2 * self.config["slots"]:
                self._seen_set.remove(self._seen.popleft())
            self._counts["discarded"] += 1

    def status(self):
        return {"enabled": True, "schema": SCHEMA, "config_sha256": self.public["config_sha256"],
                "requested_backend": self.config["backend"], "reference": self.reference,
                "data_path": "cuda_ipc_or_shared_memory_fallback" if self.driver else "shared_memory",
                "backend_reason": self.backend_reason, "slots": self.config["slots"],
                "slot_bytes": self.slot_bytes, "closed": self._closed,
                "poisoned": self._poisoned,
                "quarantined_slots": sum(bool(s.get("quarantined")) and s["memory"] is not None for s in self._slots),
                "live_slots": sum(bool(s.get("capability")) for s in self._slots),
                "allocated_host_bytes": sum(self.slot_bytes for s in self._slots if s["memory"] is not None),
                "allocated_device_bytes": sum(self.slot_bytes for s in self._slots if s["ptr"] is not None),
                "performance_qualified": False, "receipt_hash_path": (
                    "actual_bytes_device_sha256" if self.device_digest_enabled else "actual_bytes_cpu"),
                "host_copy_budget": "completion_fenced" if self.host_compute is not None else "disabled",
                **self._counts}

    def reservation(self):
        out = bool(self.outgoing)
        # Bounded source/destination CPU witness, hashing and clone working set,
        # including fallback, is separate from the retained shared-memory pool.
        scratch = 4 * self.slot_bytes if self.incoming or self.outgoing else 0
        # The original socket path decodes FP8 on CPU unless its fused codec
        # is active. Local GPU inputs instead may simultaneously retain FP8,
        # its bf16 conversion and bf16 decoded output (1+2+2 payload bytes).
        device_scratch = (8 if self.device_digest_enabled else 5 if self.incoming else 1 if out else 0) * self.slot_bytes if not self.reference else 0
        if self.device_digest_enabled:
            device_scratch += 131072  # bounded sidecars/headers and their concatenation
        return {"host_reserved_bytes": (self.config["max_bytes"] if out else 0) + scratch + self.config["slots"] * 65536,
                "host_scratch_reserved_bytes": scratch,
                "device_reserved_bytes": (self.config["max_bytes"] if out and not self.reference else 0) + device_scratch,
                "device_scratch_reserved_bytes": device_scratch,
                "driver_device_allocated_bytes": self.status()["allocated_device_bytes"]}

    def close(self):
        with self._lock:
            # Calling close again after the consumer drains may reclaim a
            # previously quarantined slot. Time alone never authorizes reuse.
            self._closed = True
            for channel in self.incoming:
                path = self.directory / ("ready-" + channel["id"] + ".json")
                if path.exists() and _authenticated(path, self.key).get("boot_id") == self._boot_id:
                    path.unlink()
            for slot in self._slots:
                if slot["memory"] is None:
                    continue
                safe = self._retire(slot)
                if safe:
                    if slot["ptr"] is not None:
                        with torch.cuda.device(self.device):
                            self.driver.checked(self.driver.free(slot["ptr"]), "free")
                        slot["ptr"] = None
                    slot["memory"].close(); slot["memory"].unlink()
                    slot["memory"] = None
                else:
                    # Never free or recycle memory a remote GPU might still use.
                    # The process owner/lease must retain this quarantined allocation.
                    if not slot.get("quarantined"):
                        slot["quarantined"] = True
                        self._counts["quarantined"] += 1


def _validate_binding(binding):
    if not isinstance(binding, dict) or set(binding) != {"owner_epoch", "job_id", "nonce", "swarm_id", "start_pos", "frame_epoch"}:
        raise HostTransportError("complete local activation binding required")
    request_binding(binding["owner_epoch"], binding, start_pos=binding["start_pos"], frame_epoch=binding["frame_epoch"])


def public_config_from_body(config):
    public = {key: value for key, value in config.items() if key not in ("secret", "directory")}
    public["key_id"] = _sha(bytes.fromhex(config["secret"]))
    return {"config_sha256": _sha(_canonical(public))}


def configured(stage, session_config):
    config = load_config()
    if config is None:
        return None
    if session_config is None:
        raise HostTransportError("host transport requires a strict authenticated stage plan")
    row = session_config.stage(session_config.index)
    public = row["signer_pubkey"]
    uuid = row.get("gpu_uuid")
    if not uuid:
        raise HostTransportError("host transport requires an explicit GPU identity, including CPU reference fixtures")
    peers = {c["id"]: c for c in config["channels"] if public in (c["src_pubkey"], c["dst_pubkey"])}
    if not peers:
        return None
    stages = session_config.plan["stages"]
    index = session_config.index
    for channel in peers.values():
        if channel["cohort_id"] != session_config.plan["cohort_id"]:
            raise HostTransportError("host channel cohort differs from authenticated plan")
        if channel["src_pubkey"] == public:
            expected = stages[index + 1] if index + 1 < len(stages) else None
            name = "dst"
        else:
            expected = stages[index - 1] if index > 0 else None
            name = "src"
        if (expected is None or expected["signer_pubkey"] != channel[name + "_pubkey"]
                or expected.get("gpu_uuid") != channel[name + "_uuid"]):
            raise HostTransportError("host channel does not name the exact adjacent strict stage")
    result = HostTransport(config, local_pubkey=public, local_uuid=uuid,
                           cohort_id=session_config.plan["cohort_id"], device=stage.device)
    result.host_compute = getattr(stage, "_host_compute", None)
    return result

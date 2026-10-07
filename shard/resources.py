"""Versioned, byte-unit placement evidence. This does not choose layer allocations.

Unknown capacity is None, never zero. Header sizes belong to a storage inventory;
PlacementRequirements needs measured runtime provenance before it can be admitted.
"""
from dataclasses import asdict, dataclass, fields
from datetime import datetime
import ctypes
import os
from pathlib import PurePosixPath
import re
import sys

SCHEMA = "shard-placement-requirements/1"


class ResourceError(ValueError):
    pass


def byte_count(value, name):
    if type(value) is not int or value < 0:
        raise ResourceError(f"{name} must be a nonnegative integer number of bytes")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ResourceError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class CalibrationProvenance:
    checkpoint_id: str
    runtime_config_sha256: str
    measured_at: str
    node_id: str
    method: str
    evidence: str
    kind: str = "measured"

    def __post_init__(self):
        if self.kind != "measured":
            raise ResourceError("runtime placement requires measured calibration, not storage estimates")
        for name in ("checkpoint_id", "node_id", "method", "evidence"):
            _text(getattr(self, name), name)
        if not isinstance(self.runtime_config_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.runtime_config_sha256):
            raise ResourceError("runtime_config_sha256 must be a lowercase SHA-256")
        try:
            stamp = datetime.fromisoformat(self.measured_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError) as exc:
            raise ResourceError("measured_at must be an ISO timestamp with a timezone") from exc
        if stamp.tzinfo is None:
            raise ResourceError("measured_at must include a timezone")


@dataclass(frozen=True)
class GpuRequirements:
    resident_weights_bytes: int = 0
    kv_hot_bytes: int = 0
    expert_cache_bytes: int = 0
    graph_bytes: int = 0
    workspace_bytes: int = 0
    activation_bytes: int = 0
    load_peak_extra_bytes: int = 0
    boundary_bytes: int = 0
    draft_bytes: int = 0
    reserve_bytes: int = 0

    def __post_init__(self):
        for field in fields(self):
            byte_count(getattr(self, field.name), field.name)

    @property
    def peak_bytes(self):
        return sum(asdict(self).values())


@dataclass(frozen=True)
class HostRequirements:
    routed_experts_bytes: int = 0
    kv_offload_bytes: int = 0
    prefill_bytes: int = 0
    staging_bytes: int = 0
    draft_bytes: int = 0
    load_peak_extra_bytes: int = 0
    reserve_bytes: int = 0
    # A subset of host bytes, not an additional RAM allocation.
    pinned_bytes: int = 0

    def __post_init__(self):
        for field in fields(self):
            byte_count(getattr(self, field.name), field.name)
        if self.pinned_bytes > self.peak_bytes:
            raise ResourceError("pinned_bytes cannot exceed the host allocation budget")
        if self.pinned_bytes < self.routed_experts_bytes:
            raise ResourceError("the local routed expert pool must fit the pinned-memory budget")

    @property
    def peak_bytes(self):
        return sum(value for name, value in asdict(self).items() if name != "pinned_bytes")


@dataclass(frozen=True)
class PlacementRequirements:
    model_id: str
    layer_start: int
    layer_end: int
    gpu: GpuRequirements
    host: HostRequirements
    provenance: CalibrationProvenance
    schema: str = SCHEMA

    def __post_init__(self):
        if self.schema != SCHEMA:
            raise ResourceError(f"unsupported resource schema {self.schema!r}")
        _text(self.model_id, "model_id")
        byte_count(self.layer_start, "layer_start")
        byte_count(self.layer_end, "layer_end")
        if self.layer_end <= self.layer_start:
            raise ResourceError("layer span must be nonempty")
        if not isinstance(self.gpu, GpuRequirements) or not isinstance(self.host, HostRequirements):
            raise ResourceError("gpu and host must be validated resource requirements")
        if not isinstance(self.provenance, CalibrationProvenance):
            raise ResourceError("measured calibration provenance is required")
        if self.gpu.resident_weights_bytes + self.gpu.boundary_bytes + self.gpu.draft_bytes <= 0:
            raise ResourceError("a nonempty GPU layer block needs a nonzero measured weight budget")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ResourceError("requirements must be an object")
        try:
            body = dict(value)
            if body.get("schema") != SCHEMA:
                raise ResourceError("serialized requirements must declare the supported schema")
            for group, kind in (("gpu", GpuRequirements), ("host", HostRequirements)):
                if not isinstance(body.get(group), dict) or set(body[group]) != {f.name for f in fields(kind)}:
                    raise ResourceError(f"{group} must explicitly declare every byte component; unknown is not zero")
            body["gpu"] = GpuRequirements(**body["gpu"])
            body["host"] = HostRequirements(**body["host"])
            body["provenance"] = CalibrationProvenance(**body["provenance"])
            return cls(**body)
        except (KeyError, TypeError) as exc:
            raise ResourceError(f"malformed requirements: {exc}") from exc


STORAGE_SCHEMA = "shard-storage-requirements/1"


@dataclass(frozen=True)
class StorageRequirements:
    model_id: str
    layer_start: int
    layer_end: int
    storage_bytes: int
    files: tuple[str, ...] = ()
    manifest_sha256: str = ""
    schema: str = STORAGE_SCHEMA

    def __post_init__(self):
        if self.schema != STORAGE_SCHEMA:
            raise ResourceError(f"unsupported storage schema {self.schema!r}")
        _text(self.model_id, "model_id")
        byte_count(self.layer_start, "layer_start")
        byte_count(self.layer_end, "layer_end")
        byte_count(self.storage_bytes, "storage_bytes")
        if self.layer_end <= self.layer_start:
            raise ResourceError("layer span must be nonempty")
        for f in self.files:
            _text(f, "file name")
        if self.manifest_sha256:
            if not isinstance(self.manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256):
                raise ResourceError("manifest_sha256 must be a lowercase SHA-256")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ResourceError("requirements must be an object")
        body = dict(value)
        if body.get("schema") != STORAGE_SCHEMA:
            raise ResourceError("serialized requirements must declare the supported schema")
        files = tuple(body.get("files", ()))
        body["files"] = files
        return cls(**body)


@dataclass(frozen=True)
class NodeResources:
    available_vram_bytes: int | None
    available_ram_bytes: int | None
    pinnable_ram_bytes: int | None
    available_disk_bytes: int | None = None

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                byte_count(value, field.name)


def evaluate_fit(requirements, capacity, storage=None):
    """Fail closed on missing evidence; return the independent binding resources."""
    if not isinstance(requirements, PlacementRequirements) or not isinstance(capacity, NodeResources):
        raise ResourceError("validated requirements and node resources are required")
    checks = [("vram", requirements.gpu.peak_bytes, capacity.available_vram_bytes),
              ("ram", requirements.host.peak_bytes, capacity.available_ram_bytes),
              ("pinned", requirements.host.pinned_bytes, capacity.pinnable_ram_bytes)]
    if storage is not None:
        if not isinstance(storage, StorageRequirements):
            raise ResourceError("storage requirement must be a validated StorageRequirements")
        checks.append(("disk", storage.storage_bytes, capacity.available_disk_bytes))
    insufficient, unknown = [], []
    for name, need, have in checks:
        if need == 0:
            continue
        if have is None:
            unknown.append(name)
        elif need > have:
            insufficient.append(name)
    action = None
    if "pinned" in unknown:
        action = "verify the complete expert-pool pinned allocation budget before placement"
    elif "disk" in unknown:
        action = "measure available disk space on the target volume before placement"
    elif "vram" in unknown:
        action = "measure available GPU VRAM before placement"
    elif "ram" in unknown:
        action = "measure available host RAM before placement"
    out = {"fits": not insufficient and not unknown,
           "status": "insufficient" if insufficient else "unknown" if unknown else "fits",
           "insufficient": insufficient, "unknown": unknown,
           "required_vram_bytes": requirements.gpu.peak_bytes,
           "required_ram_bytes": requirements.host.peak_bytes,
           "required_pinned_bytes": requirements.host.pinned_bytes,
           "action": action}
    if storage is not None:
        out["required_disk_bytes"] = storage.storage_bytes
    return out


def _read_optional(read_text, path):
    try:
        return read_text(path)
    except (OSError, UnicodeError):
        return None


def _mount_unescape(value):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def linux_cgroup_headroom(read_text):
    """Read each ancestor's hard memory limit, including nested v1/v2 cgroups.

    Return missing usage as unknown. A parent limit binds a child even when the
    child's memory.max says max. Mount roots matter inside cgroup namespaces.
    """
    membership = _read_optional(read_text, "/proc/self/cgroup")
    mounts = _read_optional(read_text, "/proc/self/mountinfo")
    if membership is None or mounts is None:
        return {"status": "unknown", "available_bytes": None, "limits": []}
    members = []
    for line in membership.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3:
            members.append((parts[1].split(","), parts[2]))
    limits = []
    unknown = False
    found = False
    for line in mounts.splitlines():
        before, sep, after = line.partition(" - ")
        left, right = before.split(), after.split()
        if not sep or len(left) < 5 or len(right) < 3:
            continue
        version = 2 if right[0] == "cgroup2" else 1 if right[0] == "cgroup" and "memory" in right[2].split(",") else 0
        if not version:
            continue
        matches = [path for controllers, path in members
                   if (version == 2 and controllers == [""]) or (version == 1 and "memory" in controllers)]
        if not matches:
            continue
        found = True
        root, mount = _mount_unescape(left[3]), _mount_unescape(left[4])
        group = PurePosixPath(matches[0])
        if ".." in group.parts or not group.is_absolute():
            raise ResourceError("invalid cgroup membership path")
        # Namespace root '/' maps to the mounted subtree, otherwise strip mount's root.
        try:
            relative = group.relative_to(root) if str(group) != "/" else PurePosixPath(".")
        except ValueError:
            unknown = True
            continue
        current = PurePosixPath(mount) / relative
        while True:
            limit_path = str(current / ("memory.max" if version == 2 else "memory.limit_in_bytes"))
            usage_path = str(current / ("memory.current" if version == 2 else "memory.usage_in_bytes"))
            raw_limit = _read_optional(read_text, limit_path)
            if raw_limit is None:
                unknown = True
            else:
                text = raw_limit.strip()
                try:
                    limit = None if text == "max" else int(text)
                    if limit is not None:
                        byte_count(limit, "cgroup limit")
                        if version == 1 and limit >= 2**60:
                            limit = None
                    if limit is not None:
                        raw_usage = _read_optional(read_text, usage_path)
                        usage = int(raw_usage.strip()) if raw_usage is not None else None
                        if usage is not None:
                            byte_count(usage, "cgroup usage")
                        else:
                            unknown = True
                        limits.append({"path": str(current), "limit_bytes": limit, "usage_bytes": usage,
                                       "available_bytes": max(0, limit - usage) if usage is not None else None})
                except ValueError:
                    unknown = True
            if current == PurePosixPath(mount):
                break
            current = current.parent
    available = [item["available_bytes"] for item in limits if item["available_bytes"] is not None]
    return {"status": "unknown" if unknown or not found else "measured",
            "available_bytes": min(available) if available and not unknown else None, "limits": limits}


def linux_host_resources(read_text, memlock_limit_bytes, disk_bytes=None):
    meminfo = _read_optional(read_text, "/proc/meminfo") or ""
    values = {m[1]: int(m[2]) * 1024 for m in re.finditer(r"^(\w+):\s+(\d+)\s+kB$", meminfo, re.M)}
    available = values.get("MemAvailable")
    cg = linux_cgroup_headroom(read_text)
    if cg["status"] == "unknown":
        effective = None  # Do not pretend a missing container bound is unrestricted.
    else:
        bounds = [x for x in (available, cg["available_bytes"]) if x is not None]
        effective = min(bounds) if available is not None else None
    status = _read_optional(read_text, "/proc/self/status") or ""
    match = re.search(r"^VmLck:\s+(\d+)\s+kB$", status, re.M)
    locked = int(match[1]) * 1024 if match else None
    if memlock_limit_bytes is not None:
        byte_count(memlock_limit_bytes, "memlock_limit_bytes")
    lock_headroom = max(0, memlock_limit_bytes - locked) if memlock_limit_bytes is not None and locked is not None else None
    return {"schema": "shard-host-capability/1", "platform": "linux",
            "available_ram_bytes": effective, "host_available_ram_bytes": available,
            "pinnable_ram_bytes": None, "available_disk_bytes": disk_bytes,
            "os_lock_limit_bytes": memlock_limit_bytes,
            "os_lock_headroom_bytes": lock_headroom,
            "locked_bytes": locked, "cgroup": cg,
            "pinning_note": "RLIMIT_MEMLOCK headroom is an OS budget; CUDA pinning still needs an allocation probe"}


def measure_host_resources(path="."):
    """Read live OS capacity without importing torch or claiming CUDA pinning success."""
    import shutil
    try:
        disk_bytes = shutil.disk_usage(path).free
    except (OSError, ValueError):
        disk_bytes = None

    if sys.platform.startswith("linux"):
        import resource
        limit = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
        # None here means unlimited/unknown, not zero capacity.
        limit = None if limit == resource.RLIM_INFINITY else limit
        def read_text(p):
            with open(p, encoding="utf-8") as handle:
                return handle.read()
        return linux_host_resources(read_text, limit, disk_bytes)
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ("total_phys", "available_phys", "total_page",
                "available_page", "total_virtual", "available_virtual", "available_extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        api = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
        api.argtypes, api.restype = [ctypes.POINTER(MemoryStatus)], ctypes.c_int
        if not api(ctypes.byref(status)):
            raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        in_job = ctypes.c_int()
        kernel.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        kernel.IsProcessInJob.restype = ctypes.c_int
        job_known = bool(kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(in_job)))
        # Physical free RAM does not establish process capacity under an unknown job limit.
        available = status.available_phys if job_known and not in_job.value else None
        return {"schema": "shard-host-capability/1", "platform": "windows",
                "available_ram_bytes": available, "host_available_ram_bytes": status.available_phys,
                "pinnable_ram_bytes": None, "available_disk_bytes": disk_bytes,
                "os_lock_limit_bytes": None, "locked_bytes": None,
                "windows_job": {"status": "unrestricted" if job_known and not in_job.value else "unknown",
                                "in_job": bool(in_job.value) if job_known else None},
                "cgroup": {"status": "not_applicable", "available_bytes": None, "limits": []},
                "pinning_note": "CUDA locked-memory capacity is unknown; allocation probe is required; Windows job limits are not inferred"}
    return {"schema": "shard-host-capability/1", "platform": sys.platform,
            "available_ram_bytes": None, "host_available_ram_bytes": None,
            "pinnable_ram_bytes": None, "available_disk_bytes": disk_bytes,
            "os_lock_limit_bytes": None, "locked_bytes": None,
            "cgroup": {"status": "unknown", "available_bytes": None, "limits": []}}

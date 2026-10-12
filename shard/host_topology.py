"""Operator-local host observations; never an identity or GPU transport attestation.

PCI/NUMA observations are read-only hints. A successful device copy is not proof
that the engine uses peer IPC. Only fresh, exactly bound engine frame samples
may replace the selected transport's cost, and never its actual routing.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import islice
import math
from pathlib import Path
import re
import sys

from .locality import timestamp

TRANSPORT_SCHEMA = "shard-host-transport/1"
TOPOLOGY_SCHEMA = "shard-host-topology/1"
TRANSPORTS = {"loopback_tcp", "buffered_tcp", "cuda_peer"}
_BDF = re.compile(r"^[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")


def _text(value, name, maximum=256):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be a bounded nonempty string")
    return value


def _integer(value, name, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its integer bound")
    return value


def _positive(value, name, maximum=math.inf):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError(f"{name} must be finite, positive and bounded")
    return value


def _samples(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 1000:
        raise ValueError("1..1000 raw timing samples required")
    return [_positive(item, "sample_ms", 3_600_000) for item in value]


def validate_transport_sample(row):
    """Validate a local engine sample, including its full runtime source digest.

    source_sha256 identifies the observed runtime inventory including the active
    transport implementation, not merely this collector script or model kernels.
    Validation alone does not qualify the sample: both fresh selected stage
    observations must bind the same digest in matching_transport().
    """
    required = {"schema", "scope", "transport", "host_id", "src", "dst", "src_gpu_uuid", "dst_gpu_uuid",
        "src_runtime_config_sha256", "dst_runtime_config_sha256", "cohort_id", "source_sha256",
        "route_id", "dial_endpoint", "dialer_id", "payload_bytes", "frame_tokens", "context_tokens",
        "warmness", "measured_at", "ttl_s", "samples_ms"}
    optional = {"shared_resource_id", "shared_busy_ms"}
    if not isinstance(row, dict) or required - set(row) or set(row) - required - optional:
        raise ValueError("complete versioned host transport bindings required")
    if row["schema"] != TRANSPORT_SCHEMA or row["scope"] != "engine_frame_ready_to_ready":
        raise ValueError("host transport must measure an engine frame ready-to-ready interval")
    if row["transport"] not in TRANSPORTS:
        raise ValueError("unsupported measured host transport")
    for key in required - {"schema", "scope", "payload_bytes", "frame_tokens", "context_tokens",
                            "measured_at", "ttl_s", "samples_ms"}:
        _text(row[key], key, 1024 if key == "dial_endpoint" else 256)
    for key in ("src_runtime_config_sha256", "dst_runtime_config_sha256", "source_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
            raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    if row["src"] == row["dst"] or row["src_gpu_uuid"] == row["dst_gpu_uuid"]:
        raise ValueError("distinct GPU workers required for a host edge")
    _integer(row["payload_bytes"], "payload_bytes", 1, 2**30)
    _integer(row["frame_tokens"], "frame_tokens", 1, 256)
    _integer(row["context_tokens"], "context_tokens", 1, 2**24)
    timestamp(row["measured_at"])
    _positive(row["ttl_s"], "ttl_s", 86400)
    values = _samples(row["samples_ms"])
    if ("shared_resource_id" in row) != ("shared_busy_ms" in row):
        raise ValueError("shared resource identity and measured busy time must be supplied together")
    if "shared_resource_id" in row:
        _text(row["shared_resource_id"], "shared_resource_id")
        _positive(row["shared_busy_ms"], "shared_busy_ms", sum(values) / len(values))
    return deepcopy(row)


def validate_topology_report(row):
    if not isinstance(row, dict) or row.get("schema") != TOPOLOGY_SCHEMA:
        raise ValueError("unsupported host topology schema")
    _text(row.get("host_id"), "host_id")
    timestamp(row["measured_at"])
    _positive(row["ttl_s"], "ttl_s", 86400)
    if row.get("affinity_applied") is not False:
        raise ValueError("topology observation must not claim automatic affinity")
    devices, pairs = row.get("devices"), row.get("pairs")
    if not isinstance(devices, list) or not 1 <= len(devices) <= 16 or not isinstance(pairs, list) or len(pairs) > 64:
        raise ValueError("bounded device and pair observations required")
    uuids, indices = set(), set()
    for device in devices:
        uuid = _text(device.get("gpu_uuid"), "gpu_uuid")
        index = _integer(device.get("local_index"), "local_index", 0, 1023)
        if uuid in uuids or index in indices:
            raise ValueError("duplicate physical GPU or local device index")
        uuids.add(uuid); indices.add(index)
        bdf = device.get("pci_bus_id")
        if bdf is not None and (not isinstance(bdf, str) or not _BDF.fullmatch(bdf)):
            raise ValueError("invalid PCI address")
        if device.get("numa_node") is not None:
            _integer(device["numa_node"], "numa_node", 0, 65535)
        cpus, ancestry = device.get("local_cpulist"), device.get("pci_ancestry", [])
        if cpus is not None and (not isinstance(cpus, str) or not re.fullmatch(r"[0-9,\-]{1,4096}", cpus)):
            raise ValueError("invalid observed CPU locality list")
        if not isinstance(ancestry, list) or len(ancestry) > 16 or any(
                not isinstance(part, str) or not _BDF.fullmatch(part) for part in ancestry):
            raise ValueError("invalid observed PCI ancestry")
    seen = set()
    for pair in pairs:
        key = pair.get("src_gpu_uuid"), pair.get("dst_gpu_uuid")
        if key[0] not in uuids or key[1] not in uuids or key[0] == key[1] or key in seen:
            raise ValueError("invalid or duplicate directed GPU pair")
        seen.add(key)
        if type(pair.get("peer_access")) is not bool:
            raise ValueError("explicit observed peer capability required")
        if pair.get("scope") != "device_copy_not_engine_wire" or pair.get("copy_api") != "torch.Tensor.copy_":
            raise ValueError("device-copy timing must not masquerade as engine transport")
        _integer(pair.get("sample_bytes"), "sample_bytes", 1, 16 * 1024**2)
        values = _samples(pair.get("samples_ms"))
        if "mean_gb_s" in pair:
            expected = pair["sample_bytes"] / (sum(values) / len(values) / 1000) / 1e9
            if not math.isclose(_positive(pair["mean_gb_s"], "mean_gb_s"), expected, rel_tol=1e-9):
                raise ValueError("device bandwidth differs from raw copy samples")
    return deepcopy(row)


def host_group_policy(value):
    """Only the operator's opt-in locality policy activates grouping/cost hints."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"enabled", "mode", "topology_reports", "transport_samples"}:
        raise ValueError("unknown host grouping policy fields")
    if type(value.get("enabled")) is not bool or value.get("mode", "contiguous") != "contiguous":
        raise ValueError("explicit host grouping enabled flag and contiguous mode required")
    reports, samples = value.get("topology_reports", []), value.get("transport_samples", [])
    if not isinstance(reports, list) or len(reports) > 64 or not isinstance(samples, list) or len(samples) > 1024:
        raise ValueError("bounded operator-local host evidence required")
    result = dict(enabled=value["enabled"], mode="contiguous",
        topology_reports=[validate_topology_report(row) for row in reports],
        transport_samples=[validate_transport_sample(row) for row in samples])
    return result if result["enabled"] else None


def contiguous_hosts(order, nodes):
    seen, previous = set(), None
    for index in order:
        # Unknown hosts remain eligible as independent workers, not an inferred IP group.
        host = ("host", nodes[index]["host_id"]) if nodes[index].get("host_id") else ("node", nodes[index]["id"])
        if host != previous:
            if host in seen:
                return False
            seen.add(host); previous = host
    return True


def group_placement(stages, nodes, policy, *, now, coordinator_id, routes):
    by_id = {node["id"]: node for node in nodes}
    groups, warnings = [], []
    for stage in stages:
        node = by_id[stage["id"]]
        host = node.get("host_id")
        if not groups or not host or groups[-1]["host_id"] != host:
            groups.append(dict(host_id=host, node_ids=[], gpu_uuids=[], lo=stage["lo"], hi=stage["hi"]))
        group = groups[-1]
        group["node_ids"].append(node["id"]); group["gpu_uuids"].append(node.get("gpu_uuid")); group["hi"] = stage["hi"]
        if host is None:
            warnings.append(f"{node['id']}: physical host unknown; kept as an independent worker")
    reports = policy["topology_reports"]
    for group in groups:
        match = [row for row in reports if row["host_id"] == group["host_id"] and
            -30 <= now - timestamp(row["measured_at"]) <= row["ttl_s"] and
            set(group["gpu_uuids"]) <= {device["gpu_uuid"] for device in row["devices"]}]
        group["topology"] = deepcopy(max(match, key=lambda row: timestamp(row["measured_at"]))) if match else None
        group["ingress_node"] = group["node_ids"][0]
        group["egress_node"] = group["node_ids"][-1]
        if not match:
            warnings.append(f"{group['host_id']}: no fresh GPU-bound PCI/NUMA/pair observation")
    by_node = {nid: i for i, group in enumerate(groups) for nid in group["node_ids"]}
    internal, boundaries = [], []
    for route in routes:
        a, b = route["src"], route["dst"]
        (internal if a in by_node and b in by_node and by_node[a] == by_node[b] else boundaries).append(deepcopy(route))
    return dict(enabled=True, mode="contiguous", groups=groups, internal_routes=internal,
        boundary_routes=boundaries, tail_return=dict(src=stages[-1]["id"], dst=coordinator_id),
        affinity_applied=False, uncertainty=warnings,
        scope="contiguous individually calibrated GPU workers; independent identities, leases and direct tail return retained")


def matching_transport(policy, src, dst, route, workload, payload, stages, *, now):
    """Return only a compatible fresh sample; capability reports never qualify."""
    if not policy or not src.get("host_id") or src.get("host_id") != dst.get("host_id"):
        return None
    if any(stage.get("source") != "fresh_stage_trace" for stage in stages):
        return None
    source_digests = []
    for node in (src, dst):
        trace = node.get("stage_trace", {})
        if trace.get("service_scope") != "compute_after_receive_before_send" or any(
                trace.get(key) != workload.get(key) for key in ("context_tokens", "warmness")):
            return None
        digest = trace.get("source_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return None
        # A caller with a pinned current runtime source has an additional
        # constraint. Never override it with a still-fresh trace from old code.
        if node.get("source_sha256") is not None and node["source_sha256"] != digest:
            return None
        source_digests.append(digest)
    # Version 1 describes one deployed source inventory. Different inventories
    # need a separately bound pair sample, not an assumption of compatibility.
    if source_digests[0] != source_digests[1]:
        return None
    expected = dict(host_id=src["host_id"], src=src["id"], dst=dst["id"], src_gpu_uuid=src.get("gpu_uuid"),
        dst_gpu_uuid=dst.get("gpu_uuid"), src_runtime_config_sha256=src.get("runtime_config_sha256"),
        dst_runtime_config_sha256=dst.get("runtime_config_sha256"), cohort_id=src.get("cohort_id"),
        source_sha256=source_digests[0],
        transport=route.get("channel"), route_id=route.get("route_id"), dial_endpoint=route.get("dial_endpoint"),
        dialer_id=route.get("dialer_id"), payload_bytes=payload,
        **{key: workload.get(key) for key in ("frame_tokens", "context_tokens", "warmness")})
    if dst.get("cohort_id") != expected["cohort_id"] or any(value is None for value in expected.values()):
        return None
    matches = [row for row in policy["transport_samples"] if all(row.get(k) == v for k, v in expected.items())
        and -30 <= now - timestamp(row["measured_at"]) <= row["ttl_s"]]
    return max(matches, key=lambda row: timestamp(row["measured_at"])) if matches else None


def observe_linux_topology(devices, *, sysfs_root="/sys", proc_root="/proc", platform=None):
    """Read bounded Linux PCI/NUMA metadata; no subprocess, binding or mutation."""
    if not isinstance(devices, list) or not 1 <= len(devices) <= 16:
        raise ValueError("1..16 device observations required")
    result = [dict(row, pci_bus_id=None, numa_node=None, local_cpulist=None, pci_ancestry=[]) for row in devices]
    if not (platform or sys.platform).startswith("linux"):
        return result
    def read(path):
        try:
            with Path(path).open("r", encoding="utf-8") as stream:
                return stream.read(8192).strip()
        except (OSError, UnicodeError):
            return ""
    base = Path(proc_root) / "driver/nvidia/gpus"
    uuid_to_bdf = {}
    try:
        entries = list(islice(base.iterdir(), 64))
    except OSError:
        entries = []
    for entry in entries:
        if not _BDF.fullmatch(entry.name):
            continue
        fields = dict(line.split(":", 1) for line in read(entry / "information").splitlines() if ":" in line)
        uuid = fields.get("GPU UUID", "").strip()
        if uuid:
            uuid_to_bdf[uuid] = entry.name.lower()
    for row in result:
        bdf = uuid_to_bdf.get(row["gpu_uuid"])
        if bdf is None:
            continue
        # Linux normally uses four digits; NVIDIA procfs can expose eight.
        if len(bdf.split(":")[0]) == 8 and bdf.startswith("0000"):
            bdf = bdf[4:]
        path = Path(sysfs_root) / "bus/pci/devices" / bdf
        row["pci_bus_id"] = bdf
        numa = read(path / "numa_node")
        row["numa_node"] = int(numa) if numa.isdecimal() and int(numa) <= 65535 else None
        cpus = read(path / "local_cpulist")
        row["local_cpulist"] = cpus if re.fullmatch(r"[0-9,\-]{1,4096}", cpus) else None
        try:
            row["pci_ancestry"] = [part for part in path.resolve().parts if _BDF.fullmatch(part)][-16:]
        except OSError:
            pass
    return result

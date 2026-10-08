"""GPT-OSS checkpoint metadata plus explicit measured planning calibration.

No model import, GPU probe, download or hardware-speed preset. Safetensors
storage is a lower bound, not a runtime/load/graph peak. A usable scalar profile
requires separately supplied measured budget components and timing provenance.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time

from .locality import number, timestamp


def _json(path, limit=16 * 1024**2):
    if path.stat().st_size > limit:
        raise ValueError("metadata file exceeds bounded JSON size")
    return json.loads(path.read_text(encoding="utf-8"))


def _int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be integer >= {minimum}")
    return value


def _header(path):
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("truncated safetensors file")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= 16 * 1024**2 or length > path.stat().st_size - 8:
            raise ValueError("invalid safetensors header length")
        blob = handle.read(length)
    header, tensors = json.loads(blob), {}
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    payload_size = path.stat().st_size - 8 - length
    spans = []
    bits = {"BOOL": 8, "U8": 8, "I8": 8, "U16": 16, "I16": 16, "U32": 32, "I32": 32,
            "U64": 64, "I64": 64, "F16": 16, "BF16": 16, "F32": 32, "F64": 64,
            "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8, "F4": 4}
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(value, dict) or not isinstance(value.get("dtype"), str):
            raise ValueError("malformed safetensors tensor metadata")
        shape, offsets = value.get("shape"), value.get("data_offsets")
        if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError("invalid tensor shape/offsets")
        for dim in shape:_int(dim, "tensor dimension")
        start, end = (_int(v, "tensor offset") for v in offsets)
        if end < start or end > payload_size:
            raise ValueError("out of range safetensors payload")
        if value["dtype"] not in bits or math.prod(shape) * bits[value["dtype"]] != (end - start) * 8:
            raise ValueError("tensor dtype/shape disagrees with payload byte count")
        spans.append((start, end))
        tensors[name] = {"dtype": value["dtype"], "shape": shape, "storage_bytes": end - start}
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError("overlap or holes in checkpoint tensor spans")
        cursor = end
    if cursor != payload_size:
        raise ValueError("unindexed checkpoint payload bytes")
    return tensors, hashlib.sha256(raw + blob).hexdigest()


def inspect_checkpoint(model_dir, *, model_id=None, checkpoint_id=None, verify_payload=True):
    directory = Path(model_dir).resolve()
    download = None
    if checkpoint_id is not None or (directory / ".shard-download.json").exists():
        from .download_inventory import verify_inventory
        download = verify_inventory(directory, expected_checkpoint_id=checkpoint_id,
                                    expected_repo=model_id, verify_files=verify_payload)
    config_path = directory / "config.json"
    config = _json(config_path)
    if config.get("model_type") != "gpt_oss":
        raise ValueError("GPT-OSS adapter requires model_type=gpt_oss")
    layers = _int(config.get("num_hidden_layers"), "num_hidden_layers", 1)
    hidden = _int(config.get("hidden_size"), "hidden_size", 1)
    dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8}
    declared_dtypes = [str(config[key]).removeprefix("torch.") for key in ("dtype", "torch_dtype") if config.get(key) is not None]
    if len(set(declared_dtypes)) > 1:
        raise ValueError("config dtype and torch_dtype are inconsistent")
    wire_elements = dtype_bytes.get(declared_dtypes[0]) if declared_dtypes else None
    if declared_dtypes and wire_elements is None:
        raise ValueError("unsupported GPT-OSS floating wire dtype")
    quant = config.get("quantization_config", {})
    if not isinstance(quant, dict) or quant.get("quant_method") != "mxfp4":
        raise ValueError("native GPT-OSS planning requires explicit MXFP4 config, never inferred BF16 conversion")
    index_path = directory / "model.safetensors.index.json"
    index = _json(index_path)
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("converted/local GPT-OSS checkpoint needs a nonempty safetensors weight map")
    all_tensors, files = {}, []
    for filename in sorted(set(mapping.values())):
        if not isinstance(filename, str):
            raise ValueError("weight_map files must be strings")
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory) or path.suffix != ".safetensors":
            raise ValueError("checkpoint shard resolves outside model directory")
        tensors, digest = _header(path)
        files.append({"file": filename, "header_sha256": digest, "file_bytes": path.stat().st_size})
        for name, data in tensors.items():
            if name in all_tensors:
                raise ValueError("tensor duplicated across checkpoint shards")
            if mapping.get(name) != filename:
                raise ValueError("safetensors header and weight_map disagree")
            all_tensors[name] = data
    if set(mapping) != set(all_tensors):
        raise ValueError("weight_map contains missing tensors")
    per_layer, gpu_floor = [0] * layers, [0] * layers
    boundary, unknown = {"embedding_bytes": 0, "tail_bytes": 0}, []
    gpu_boundary = {"head_bytes": 0, "tail_bytes": 0}
    for name, data in all_tensors.items():
        # dtype='auto' follows the actual config. Wider saved floating weights
        # may be cast; native packed MXFP4 blocks/scales keep their stored bytes.
        # Without a declared dtype use only a conservative 16-bit float floor.
        stored = data["storage_bytes"]
        resident_floor = (math.prod(data["shape"]) * (wire_elements or 2)
                          if data["dtype"] in ("F16", "BF16", "F32", "F64") else stored)
        match = re.match(r"^model\.layers\.(\d+)\.", name)
        if match:
            layer = int(match[1])
            if layer >= layers:
                raise ValueError("tensor layer index outside actual config")
            per_layer[layer] += data["storage_bytes"]
            gpu_floor[layer] += resident_floor
        elif name.startswith("model.embed_tokens."):
            boundary["embedding_bytes"] += data["storage_bytes"]
            gpu_boundary["head_bytes"] += resident_floor
        elif name.startswith("lm_head.") or name.startswith("model.norm."):
            boundary["tail_bytes"] += data["storage_bytes"]
            gpu_boundary["tail_bytes"] += resident_floor
        else:
            unknown.append(name)
    if any(value <= 0 for value in per_layer) or not boundary["embedding_bytes"]:
        raise ValueError("checkpoint lacks every configured layer or embedding")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if download is not None and (download["config_sha256"] != config_sha or
            download["index_sha256"] != hashlib.sha256(index_path.read_bytes()).hexdigest()):
        raise ValueError("local config/index differs from immutable download inventory")
    metadata_id = "metadata-sha256:" + hashlib.sha256(json.dumps(
        {"config_sha256": config_sha, "index": index, "files": files}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload_verified = bool(download and download["payload_integrity_verified"])
    return {"schema": "gpt-oss-storage-inventory/1", "model_id": model_id or (download or {}).get("repo") or config.get("_name_or_path") or "gpt_oss",
            "checkpoint_id": download["checkpoint_id"] if download is not None else metadata_id,
            "checkpoint_metadata_id": metadata_id,
            "config_sha256": config_sha, "n_layers": layers, "hidden_size": hidden,
            "wire_element_bytes": wire_elements,
            "quantization": "mxfp4", "config": config, "layer_storage_bytes": per_layer,
            "gpu_weight_storage_floor": {"version": 1, "layer_bytes": gpu_floor, **gpu_boundary},
            **boundary, "unknown_tensors": unknown, "total_storage_bytes": sum(d["storage_bytes"] for d in all_tensors.values()),
            "files": files, "payload_integrity_verified": payload_verified,
            "download_manifest_sha256": download["manifest_sha256"] if download else None,
            "identity_scope": "immutable download identity and complete local file digests" if payload_verified else
                              "config/index/header metadata only; actual load must verify complete file digests"}


def planning_profile(inventory, calibration, *, now=None, prompt_tokens=2048):
    """Return a generic plan_ring profile; no baked GPU capacity or tok/s.

    Calibration components are explicit upper budgets measured for the intended
    backend/workload, not auto-estimated from checkpoint size. Individual nodes
    still supply their bound stage_trace and per-node layer footprint/capacity.
    """
    if inventory.get("schema") != "gpt-oss-storage-inventory/1" or not isinstance(calibration, dict) or calibration.get("schema") != "gpt-oss-planning-calibration/1":
        raise ValueError("explicit GPT-OSS inventory and measured planning calibration required")
    if inventory.get("unknown_tensors"):
        raise ValueError("unclassified checkpoint tensors require an explicit resource mapping")
    for field in ("checkpoint_id", "config_sha256"):
        if calibration.get(field) != inventory[field]:
            raise ValueError(f"calibration {field} differs from checkpoint")
    if calibration.get("kind") != "measured" or not all(isinstance(calibration.get(k), str) and calibration[k] for k in ("node_id", "gpu_uuid", "runtime_config_sha256", "method", "evidence")):
        raise ValueError("calibration needs measured node/GPU/config/method/evidence provenance")
    if not re.fullmatch(r"[0-9a-f]{64}", calibration["runtime_config_sha256"]):
        raise ValueError("calibration runtime configuration must be a SHA-256")
    measured = timestamp(calibration["measured_at"])
    ttl = number(calibration["ttl_s"], "calibration ttl_s", minimum=1e-9)
    now = time.time() if now is None else timestamp(now)
    if not -30 <= now - measured <= ttl:
        raise ValueError("planning calibration expired")
    names = ("layer_vram_bytes", "kv_bytes_per_layer", "reserve_bytes", "head_reserve_bytes", "tail_reserve_bytes", "load_peak_extra_bytes")
    budget = {key: _int(calibration[key], key) for key in names}
    floor = inventory["gpu_weight_storage_floor"]
    if budget["layer_vram_bytes"] < max(floor["layer_bytes"]) or budget["head_reserve_bytes"] < floor["head_bytes"] or budget["tail_reserve_bytes"] < floor["tail_bytes"]:
        raise ValueError("measured runtime budget is below checkpoint storage lower bound")
    cap = _int(calibration["cap_layers"], "cap_layers", 1)
    timing = number(calibration["layer_ms"], "measured layer_ms", minimum=1e-9)
    _int(prompt_tokens, "prompt_tokens", 1)
    # The codec preserves tensor dtype. A filename/model label cannot establish it.
    wire_elements = calibration.get("wire_element_bytes", inventory["wire_element_bytes"])
    if type(wire_elements) is not int or wire_elements not in (2, 4, 8):
        raise ValueError("actual config dtype or measured wire_element_bytes is required")
    if inventory["wire_element_bytes"] is not None and wire_elements != inventory["wire_element_bytes"]:
        raise ValueError("measured wire bytes differ from actual auto dtype config")
    wire_per_token = inventory["hidden_size"] * wire_elements
    mib = 1024**2
    return {"model_id": inventory["model_id"], "n_layers": inventory["n_layers"], "placement": "gpu",
            "require_exact_calibrations": True,
            "gpu_weight_storage_floor": inventory["gpu_weight_storage_floor"],
            "layer_vram_mb": budget["layer_vram_bytes"] / mib, "kv_mb_per_layer": budget["kv_bytes_per_layer"] / mib,
            "reserve_mb": budget["reserve_bytes"] / mib, "head_reserve_mb": budget["head_reserve_bytes"] / mib,
            "tail_reserve_mb": budget["tail_reserve_bytes"] / mib, "load_peak_extra_mb": budget["load_peak_extra_bytes"] / mib,
            "cap_layers": cap, "layer_ms_base": timing, "head_layer_ms_mult": 1.0,
            "wire_bytes_per_token": wire_per_token, "decode_bytes": wire_per_token,
            "prefill_bytes": prompt_tokens * wire_per_token, "prefill_chunks": 1,
            "decode_steps": 1, "calibration_provenance": {key: calibration[key] for key in
                ("node_id", "gpu_uuid", "runtime_config_sha256", "measured_at", "ttl_s", "method", "evidence")},
            "checkpoint_id": inventory["checkpoint_id"], "config_sha256": inventory["config_sha256"],
            "payload_integrity_verified": inventory["payload_integrity_verified"],
            "scope": "measured budget/timing input to a prediction-only planner; not hardware acceptance"}

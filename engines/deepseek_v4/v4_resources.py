"""V4 storage inventory and explicit resource calibration, independent of placement.

inspect_checkpoint reads only local safetensors headers/config. No torch import,
weight download, CUDA allocation or placement occurs during this inventory.
The metadata identity is NOT an integrity hash of tensor payloads.
"""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import struct

try:  # deployed flat beside the engine files
    from resources import PlacementRequirements, ResourceError, byte_count
except ImportError:
    # A direct absolute-path CLI invocation need not have the repo cwd/PYTHONPATH.
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "shard" / "resources.py").is_file() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from shard.resources import PlacementRequirements, ResourceError, byte_count

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
MAX_HEADER_BYTES = 16 * 1024 * 1024
_BITS = {"BOOL": 8, "U8": 8, "I8": 8, "I16": 16, "U16": 16, "I32": 32,
         "U32": 32, "I64": 64, "U64": 64, "F16": 16, "BF16": 16,
         "F32": 32, "F64": 64, "C64": 64, "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6,
         "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8, "F8_E4M3FNUZ": 8, "F8_E5M2FNUZ": 8}
_LAYER = re.compile(r"^layers\.(\d+)\.")
_DRAFT = re.compile(r"^mtp\.(\d+)\.")
_EXPERT = re.compile(r"^(?:layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.")
_ALIAS = re.compile(r"^mtp\.\d+\.(embed|head)\.weight$")


def _unique_object(pairs):
    body = {}
    for key, value in pairs:
        if key in body:
            raise ResourceError(f"duplicate JSON key {key!r}")
        body[key] = value
    return body


def _json_bytes(blob, label):
    try:
        body = json.loads(blob, object_pairs_hook=_unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(ResourceError(f"nonfinite JSON {value}")))
    except (UnicodeError, ValueError) as exc:
        raise ResourceError(f"invalid JSON in {label}: {exc}") from exc
    if not isinstance(body, dict):
        raise ResourceError(f"{label} must contain a JSON object")
    return body


def _small_file(path):
    with path.open("rb") as handle:
        blob = handle.read(MAX_HEADER_BYTES + 1)
    if len(blob) > MAX_HEADER_BYTES:
        raise ResourceError(f"metadata file is too large: {path.name}")
    return blob


def read_safetensors_header(path):
    """Validate header ranges, dtypes, shapes and packed-bit alignment without loading data."""
    path = Path(path)
    file_bytes = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ResourceError(f"truncated safetensors prefix: {path.name}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_bytes <= MAX_HEADER_BYTES or header_bytes > file_bytes - 8:
            raise ResourceError(f"invalid safetensors header length: {path.name}")
        blob = handle.read(header_bytes)
    header = _json_bytes(blob, path.name)
    payload_bytes = file_bytes - 8 - header_bytes
    tensors, spans = {}, []
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise ResourceError("safetensors __metadata__ must map strings to strings")
    for name, entry in header.items():
        if not name or not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise ResourceError(f"malformed tensor metadata: {name!r}")
        dtype, shape, offsets = entry["dtype"], entry["shape"], entry["data_offsets"]
        if not isinstance(dtype, str) or dtype not in _BITS:
            raise ResourceError(f"unsupported safetensors dtype {dtype!r}")
        if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ResourceError(f"invalid shape/offsets for {name}")
        for value in shape:
            byte_count(value, f"{name} dimension")
        start, end = (byte_count(value, f"{name} offset") for value in offsets)
        if end < start or end > payload_bytes:
            raise ResourceError(f"out-of-range tensor offsets for {name}")
        bits = math.prod(shape) * _BITS[dtype]
        if bits % 8 or bits // 8 != end - start:
            raise ResourceError(f"dtype/shape byte size disagrees with offsets for {name}")
        tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": offsets,
                         "storage_bytes": end - start}
        spans.append((start, end, name))
    cursor = 0
    # Empty tensors may share endpoints, but may not introduce holes or overlap.
    for start, end, name in sorted(spans):
        if start != cursor:
            raise ResourceError(f"overlapping or noncontiguous data offsets at {name}")
        cursor = end
    if cursor != payload_bytes:
        raise ResourceError(f"unindexed safetensors payload bytes in {path.name}")
    return {"file": path.name, "file_bytes": file_bytes, "header_bytes": header_bytes,
            "header_sha256": hashlib.sha256(prefix + blob).hexdigest(), "tensors": tensors,
            "metadata": metadata}


def structural_profile(config=None):
    """Architecture facts only. Explicitly rejected by the existing scalar planner."""
    config = config or {}
    values = {"n_layers": config.get("n_layers", 43), "dim": config.get("dim", 4096),
              "hc_mult": config.get("hc_mult", 4),
              "dspark_target_layer_ids": config.get("dspark_target_layer_ids", [40, 41, 42])}
    for field in ("n_layers", "dim", "hc_mult"):
        byte_count(values[field], field)
    targets = values["dspark_target_layer_ids"]
    if not isinstance(targets, list) or any(type(value) is not int for value in targets):
        raise ResourceError("DSpark target layers must be a list of integer indices")
    if type(config.get("n_mtp_layers", 3)) is not int or config.get("n_mtp_layers", 3) != 3:
        raise ResourceError("DeepSeek-V4-Flash calibration requires all three MTP blocks")
    if values != {"n_layers": 43, "dim": 4096, "hc_mult": 4, "dspark_target_layer_ids": [40, 41, 42]}:
        raise ResourceError("resource adapter is restricted to DeepSeek-V4-Flash's 43-layer structure")
    return {"schema": "shard-model-structure/1", "calibration_status": "structural",
            "model_id": MODEL_ID, **values, "decode_bytes": 32768,
            "tail_min_layers": 3, "placement_calibration": None}


def classify_tensor(name):
    layer, draft, expert, alias = _LAYER.match(name), _DRAFT.match(name), _EXPERT.match(name), _ALIAS.match(name)
    if alias:
        return {"scope": "draft", "index": int(draft[1]), "kind": "alias",
                "alias_of": f"{alias[1]}.weight"}
    if layer or draft:
        match = layer or draft
        return {"scope": "main" if layer else "draft", "index": int(match[1]),
                "kind": "routed_expert" if expert else "resident",
                "expert_id": int(expert[2]) if expert else None}
    if name == "embed.weight":
        return {"scope": "embedding", "kind": "resident"}
    if name == "head.weight" or name == "norm.weight" or name in {"hc_head_fn", "hc_head_base", "hc_head_scale"}:
        return {"scope": "output", "kind": "resident"}
    return {"scope": "unknown", "kind": "unknown"}


def _runtime_parameter_floor(name, info):
    # ParallelHead and DSparkConfidenceHead use fp32 parameters even with bf16 checkpoint weights.
    promoted = name == "head.weight" or name.endswith(".markov_w2.weight") or name.endswith(".confidence_head.proj.weight")
    if promoted:
        return math.prod(info["shape"]) * 4
    return info["storage_bytes"]


def inspect_checkpoint(checkpoint_dir, *, expected_checkpoint_id=None):
    directory = Path(checkpoint_dir).resolve()
    config_blob = _small_file(directory / "config.json")
    config = _json_bytes(config_blob, "config.json")
    required = {"n_layers", "dim", "hc_mult", "dspark_target_layer_ids", "n_mtp_layers"}
    if not required <= set(config):
        raise ResourceError("converted V4 config must explicitly declare layer/dimension/HC/DSpark structure")
    structure = structural_profile(config)
    files = sorted(directory.glob("model*-mp*.safetensors"))
    if not files:
        raise ResourceError("V4 resource inventory requires converted model*-mp1.safetensors files")
    if any(not re.fullmatch(r"model\d+-mp1\.safetensors", path.name) for path in files):
        raise ResourceError("tensor-parallel checkpoints require a different runtime; use converted mp1 weights")
    tensors, headers = {}, []
    for path in files:
        if path.resolve().parent != directory:
            raise ResourceError("checkpoint shard resolves outside checkpoint directory")
        header = read_safetensors_header(path)
        headers.append({key: value for key, value in header.items() if key != "tensors"})
        for name, info in header["tensors"].items():
            if name in tensors:
                raise ResourceError(f"tensor appears in multiple checkpoint shards: {name}")
            classification = classify_tensor(name)
            if classification.get("scope") == "main" and classification["index"] >= structure["n_layers"]:
                raise ResourceError(f"layer index outside model structure: {name}")
            tensors[name] = {**info, **classification, "file": path.name,
                             "runtime_parameter_floor_bytes": 0 if classification["kind"] == "alias" else _runtime_parameter_floor(name, info)}
    identity_body = {"config_sha256": hashlib.sha256(config_blob).hexdigest(), "headers": headers}
    checkpoint_id = "metadata-sha256:" + hashlib.sha256(json.dumps(identity_body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if expected_checkpoint_id is not None and checkpoint_id != expected_checkpoint_id:
        raise ResourceError("checkpoint metadata identity changed")
    return {"schema": "v4-storage-inventory/1", "calibration_status": "unmeasured",
            "checkpoint_id": checkpoint_id, "identity_kind": "headers-and-config-only",
            "payload_integrity_verified": False, "structure": structure,
            "config": config, "files": headers, "tensors": tensors,
            "total_storage_bytes": sum(info["storage_bytes"] for info in tensors.values()),
            "unknown_tensors": [name for name, info in tensors.items() if info["kind"] == "unknown"],
            "runtime_peak_bytes": None}


def stage_storage_inventory(inventory, lo, hi, *, head=False, tail=False, dspark=False):
    structure = inventory["structure"]
    byte_count(lo, "lo")
    byte_count(hi, "hi")
    if not 0 <= lo < hi <= structure["n_layers"]:
        raise ResourceError("stage range is outside the model")
    if type(head) is not bool or type(tail) is not bool or type(dspark) is not bool:
        raise ResourceError("stage role flags must be booleans")
    if head and lo != 0 or tail and hi != structure["n_layers"]:
        raise ResourceError("head/tail must own the corresponding model boundary")
    if dspark and (not tail or not all(lo <= target < hi for target in structure["dspark_target_layer_ids"])):
        raise ResourceError("DSpark requires a tail owning all target layers 40..42")
    selected = {}
    for name, info in inventory["tensors"].items():
        scope = info["scope"]
        wanted = (scope == "main" and lo <= info["index"] < hi or scope == "draft" and dspark
                  or scope == "embedding" and (head or dspark) or scope == "output" and tail)
        if wanted:
            selected[name] = info
    layers = {info["index"] for info in selected.values() if info["scope"] == "main"}
    if layers != set(range(lo, hi)):
        raise ResourceError("checkpoint has no tensors for every assigned layer")
    if (head or dspark) and "embed.weight" not in selected:
        raise ResourceError("checkpoint is missing the required embedding")
    if tail and not {"head.weight", "norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"} <= set(selected):
        raise ResourceError("checkpoint is missing tail norm/head/hyper-connection parameters")
    if dspark:
        drafts = {info["index"] for info in selected.values() if info["scope"] == "draft" and info["kind"] != "alias"}
        if drafts != set(range(inventory["config"].get("n_mtp_layers", 3))):
            raise ResourceError("checkpoint is missing DSpark MTP blocks")
        # The reference registers aliases; converted checkpoints must omit copies.
        if any(info["kind"] == "alias" for info in selected.values()):
            raise ResourceError("converted MTP embed/head aliases must be omitted, not independently stored")
    routed = sum(info["storage_bytes"] for info in selected.values() if info["kind"] == "routed_expert")
    resident = sum(info["runtime_parameter_floor_bytes"] for info in selected.values() if info["kind"] == "resident")
    return {"schema": "v4-stage-storage-inventory/1", "calibration_status": "unmeasured",
            "checkpoint_id": inventory["checkpoint_id"], "lo": lo, "hi": hi,
            "head": head, "tail": tail, "dspark": dspark, "tensors": selected,
            "routed_expert_storage_bytes": routed, "resident_parameter_floor_bytes": resident,
            "total_storage_bytes": sum(info["storage_bytes"] for info in selected.values()),
            "runtime_peak_bytes": None, "aliases": {"mtp.*.embed.weight": "embed.weight", "mtp.*.head.weight": "head.weight"} if dspark else {}}


def load_calibration(path_or_dict, *, checkpoint_id=None):
    """Accept explicit measured requirements; no inventory-to-measurement conversion."""
    body = _json_bytes(_small_file(Path(path_or_dict)), "calibration") if isinstance(path_or_dict, (str, os.PathLike)) else path_or_dict
    requirements = PlacementRequirements.from_dict(body)
    if requirements.model_id != MODEL_ID:
        raise ResourceError("calibration is not for DeepSeek-V4-Flash")
    if checkpoint_id is not None and requirements.provenance.checkpoint_id != checkpoint_id:
        raise ResourceError("calibration checkpoint identity mismatch")
    return requirements


def runtime_config_identity(stage):
    args = asdict(stage.args) if is_dataclass(stage.args) else dict(vars(stage.args))
    body = {"args": args, "lo": stage.lo, "hi": stage.hi, "head": stage.head,
            "tail": stage.tail, "dspark": stage._dspark, "dtype": str(stage.dtype),
            "runtime_metrics_enabled": getattr(stage, "_runtime_metrics", None) is not None,
            "environment": {key: value for key, value in sorted(os.environ.items()) if key.startswith("V4_")}}
    import sys
    torch_module = sys.modules.get("torch")
    body["runtime_versions"] = {"torch": str(getattr(torch_module, "__version__", "unknown")),
                                "cuda": getattr(getattr(torch_module, "version", None), "cuda", None)}
    # A calibration cannot silently survive an engine implementation change.
    directory = Path(__file__).resolve().parent
    body["engine_source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in sorted(directory.glob("v4_*.py"))}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_engine_module(name):
    """Use the same flat module identity as the serve path, even outside repo cwd."""
    import importlib
    import sys
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    return importlib.import_module(name)


def placement_requirements_for_stage(stage, calibration=None, checkpoint_dir=None):
    if calibration is None:
        raise NotImplementedError("V4 placement needs measured calibration for this checkpoint/range/roles; header storage bytes are not runtime peaks")
    if not str(stage.device).startswith("cuda"):
        raise ResourceError("GPU placement calibration cannot be validated against a CPU Stage")
    requirements = load_calibration(calibration)
    directory = checkpoint_dir or getattr(stage, "_resource_checkpoint_dir", None)
    if directory is None:
        raise ResourceError("the loaded checkpoint directory is required to verify calibration identity")
    inventory = inspect_checkpoint(directory, expected_checkpoint_id=requirements.provenance.checkpoint_id)
    stage_storage_inventory(inventory, stage.lo, stage.hi, head=stage.head, tail=stage.tail, dspark=stage._dspark)
    if (requirements.layer_start, requirements.layer_end) != (stage.lo, stage.hi):
        raise ResourceError("calibration layer span mismatch")
    if requirements.provenance.runtime_config_sha256 != runtime_config_identity(stage):
        raise ResourceError("calibration runtime configuration/roles/environment mismatch")
    report = measure_stage_resources(stage, checkpoint_id=requirements.provenance.checkpoint_id)
    if report["draft_measurement_status"] == "missing":
        raise ResourceError("DSpark placement needs the actually loaded drafter; main-stage storage is insufficient")
    if report["allocator"]["allocated_bytes"] is None or report["allocator"]["reserved_bytes"] is None:
        raise ResourceError("actual CUDA allocator evidence is unavailable")
    observed = max(report["module_storage"]["gpu_bytes"], report["allocator"]["allocated_bytes"],
                   report["allocator"]["reserved_bytes"])
    if requirements.gpu.peak_bytes < observed:
        raise ResourceError("calibration budget is below actual GPU module/process allocator storage")
    if requirements.host.peak_bytes < report["module_storage"]["host_bytes"]:
        raise ResourceError("calibration budget is below actual host tensor storage")
    if requirements.host.pinned_bytes < report["module_storage"]["host_pinned_bytes"]:
        raise ResourceError("calibration budget is below actual pinned host tensor storage")
    return requirements


def measure_stage_resources(stage, draft=None, *, checkpoint_id, peak_interval_started=False):
    """Inventory actual tensor storages and optionally caller-owned CUDA peak stats.

    Parameter/buffer storage is deduplicated across expert bank views, graph views,
    and DSpark embedding/head aliases. Allocator peaks belong to the process, not
    an individual Stage. Only an isolated caller that reset stats before loading
    may mark peak_interval_started=True. This function never resets global stats.
    """
    import torch  # library inventory path is lazy and also works on CPU fixtures
    from collections import deque
    if draft is None:
        draft = getattr(stage, "_resource_drafter", None) or getattr(stage, "_runtime_draft", None)
    seen, entries = set(), []

    def record(name, tensor, kind):
        if not isinstance(tensor, torch.Tensor):
            return
        storage = tensor.untyped_storage()
        device = str(tensor.device)
        key = (device, storage.data_ptr(), storage.nbytes())
        if key in seen:
            return
        seen.add(key)
        entries.append({"name": name, "kind": kind, "device": device,
                        "dtype": str(tensor.dtype), "storage_bytes": storage.nbytes(),
                        "pinned": bool(tensor.is_pinned()) if tensor.device.type == "cpu" else False})

    modules = []
    for name in ("layers", "embed_tokens", "norm", "lm_head"):
        module = getattr(stage, name, None)
        if isinstance(module, torch.nn.Module):
            modules.append((name, module, "main" if name == "layers" else "boundary"))
    draft_module = getattr(getattr(draft, "tail", draft), "mtp", None)
    if isinstance(draft_module, torch.nn.Module):
        modules.append(("draft.mtp", draft_module, "draft"))
    for name, module, scope in modules:
        for parameter_name, tensor in module.named_parameters():
            kind = "routed_expert" if ".ffn.experts." in f".{parameter_name}" else "resident"
            record(f"{name}.{parameter_name}", tensor, f"{scope}_{kind}")
    for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        record(name, getattr(stage, name, None), "boundary_resident")
    for name, module, _ in modules:
        for buffer_name, tensor in module.named_buffers():
            record(f"{name}.{buffer_name}", tensor, "state")
    walked = set()
    def walk_state(name, obj, depth=0):
        if isinstance(obj, torch.Tensor):
            record(name, obj, "state")
            return
        if depth > 12 or id(obj) in walked:
            return
        walked.add(id(obj))
        if isinstance(obj, dict):
            children = obj.items()
        elif isinstance(obj, (list, tuple, deque)):
            children = enumerate(obj)
        elif isinstance(obj, torch.nn.Module):
            children = vars(obj).items()
        else:
            return  # Kernel modules/CUDA graph handles are not traversable tensor containers.
        for key, value in children:
            walk_state(f"{name}.{key}", value, depth + 1)
    # Catch unregistered compressor/KV buffers and rollback snapshots too.
    for key, value in vars(stage).items():
        walk_state(f"stage.{key}", value)
    if draft is not None:
        tail = getattr(draft, "tail", draft)
        for key, value in vars(tail).items():
            walk_state(f"draft.{key}", value)
    gpu_bytes = sum(entry["storage_bytes"] for entry in entries if entry["device"].startswith("cuda"))
    host_bytes = sum(entry["storage_bytes"] for entry in entries if entry["device"] == "cpu")
    totals = {}
    for entry in entries:
        key = ("gpu_" if entry["device"].startswith("cuda") else "host_") + entry["kind"] + "_bytes"
        totals[key] = totals.get(key, 0) + entry["storage_bytes"]
    allocator = {"scope": "process", "peak_interval_owned": bool(peak_interval_started),
                 "allocated_bytes": None, "reserved_bytes": None, "peak_allocated_bytes": None,
                 "peak_reserved_bytes": None, "device_free_bytes": None, "device_total_bytes": None}
    hardware = {"device": str(stage.device), "gpu_name": None, "gpu_uuid": None,
                "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda}
    if str(stage.device).startswith("cuda") and torch.cuda.is_available():
        device = torch.device(stage.device)
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        allocator.update(allocated_bytes=torch.cuda.memory_allocated(device),
                         reserved_bytes=torch.cuda.memory_reserved(device),
                         device_free_bytes=free, device_total_bytes=total)
        if peak_interval_started:
            allocator.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                             peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        props = torch.cuda.get_device_properties(device)
        hardware.update(gpu_name=props.name, gpu_uuid=str(getattr(props, "uuid", "")) or None)
    return {"schema": "v4-resource-measurement/1", "calibration_status": "measured",
            "model_id": MODEL_ID, "checkpoint_id": checkpoint_id,
            "runtime_config_sha256": runtime_config_identity(stage),
            "measured_at": datetime.now(timezone.utc).isoformat(), "node_id": socket.gethostname(),
            "lo": stage.lo, "hi": stage.hi, "head": stage.head, "tail": stage.tail,
            "dspark": bool(getattr(stage, "_dspark", False)), "hardware": hardware,
            "draft_measurement_status": "measured" if draft_module is not None else "missing" if getattr(stage, "_dspark", False) else "not_requested",
            "module_storage": {"gpu_bytes": gpu_bytes, "host_bytes": host_bytes,
                               "host_pinned_bytes": sum(entry["storage_bytes"] for entry in entries if entry["pinned"]),
                               "by_kind": totals, "storages": entries},
            "allocator": allocator,
            "unattributed_components": {"graph_bytes": None, "workspace_bytes": None,
                                         "activation_bytes": None, "cuda_context_bytes": None},
            "placement_requirements": None,
            "note": "Actual module storage and process allocator evidence, not an enabled hybrid loader or a calibrated placement profile"}


def measure_checkpoint(checkpoint_dir, lo, hi, *, head=False, tail=False, dspark=False,
                       device="cuda:0", max_seq=8192, prefill_tokens=16, decode_tokens=8):
    """Run an isolated real Stage load/prefill/decode probe; never rent/download GPUs."""
    inventory = inspect_checkpoint(checkpoint_dir)
    stage_inventory = stage_storage_inventory(inventory, lo, hi, head=head, tail=tail, dspark=dspark)
    for name, value in (("max_seq", max_seq), ("prefill_tokens", prefill_tokens), ("decode_tokens", decode_tokens)):
        byte_count(value, name)
    if prefill_tokens < 1 or decode_tokens < 1 or prefill_tokens + decode_tokens + 16 > max_seq:
        raise ResourceError("probe needs positive prefill/decode lengths and room for DSpark blocks")
    import torch
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ResourceError("V4 hardware calibration requires an available CUDA device; CPU fixtures are not GPU calibration")
    # Use the real engine's initialization. Imported only for an explicitly requested probe.
    v4_stage = _load_engine_module("v4_stage")
    args = v4_stage.config(str(Path(checkpoint_dir).resolve()))
    args.max_seq_len, args.max_batch_size = max_seq, 1
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.reset_peak_memory_stats(device)
    stage = v4_stage.Stage(lo, hi, args, head=head, tail=tail, dspark=dspark, device=device, runtime_metrics=False)
    stage.load(str(Path(checkpoint_dir).resolve()))
    drafter = None
    if dspark:
        ring_drafter = _load_engine_module("v4_dspark_draft").ring_drafter
        drafter = ring_drafter(stage, str(Path(checkpoint_dir).resolve()))
    torch.cuda.synchronize(device)
    load_stats = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                  "after_load_allocated_bytes": torch.cuda.memory_allocated(device),
                  "after_load_reserved_bytes": torch.cuda.memory_reserved(device)}
    # One interval includes load and the real forwards, including lazy graph allocation.
    with torch.inference_mode():
        stage.reset()
        if drafter is not None:
            drafter.tail.reset()
        predicted = None
        for position, count in [(0, prefill_tokens)] + [(prefill_tokens + i, 1) for i in range(decode_tokens)]:
            ids = predicted.reshape(1, 1) if predicted is not None else torch.arange(position, position + count, device=device).remainder(args.vocab_size).reshape(1, count)
            hidden = stage.embed(ids) if stage.embed_tokens is not None else torch.zeros(1, count, args.hc_mult, args.dim, device=device, dtype=stage.dtype)
            hidden = stage.forward(hidden, ids, position)
            if not torch.isfinite(hidden).all().item():
                raise ResourceError("calibration forward produced nonfinite hidden states")
            if tail:
                logits = stage.logits_all(hidden, full_logits=False)
                if not torch.isfinite(logits).all().item():
                    raise ResourceError("calibration forward produced nonfinite logits")
                predicted = logits.argmax(dim=-1).reshape(1)
            if drafter is not None:
                tap = stage.tail_main_hidden()
                if position == 0:
                    drafter.tail.prefill(predicted, tap)
                else:
                    drafter.tail.advance_and_draft(predicted.reshape(1, 1), tap, position)
    report = measure_stage_resources(stage, drafter, checkpoint_id=inventory["checkpoint_id"], peak_interval_started=True)
    report.update(storage_inventory=stage_inventory, load_interval=load_stats,
                  workload={"kind": "synthetic_resource_probe", "prefill_tokens": prefill_tokens,
                            "decode_tokens": decode_tokens, "max_seq": max_seq, "batch": 1,
                            "route_coverage": "sampled routes; not a throughput or cache-hit benchmark"})
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for operation in ("inventory", "measure"):
        command = sub.add_parser(operation)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--lo", type=int, required=True)
        command.add_argument("--hi", type=int, required=True)
        for role in ("head", "tail", "dspark"):
            command.add_argument(f"--{role}", action="store_true")
        command.add_argument("--output")
        if operation == "measure":
            command.add_argument("--device", default="cuda:0")
            command.add_argument("--max-seq", type=int, default=8192)
            command.add_argument("--prefill-tokens", type=int, default=16)
            command.add_argument("--decode-tokens", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        roles = dict(head=args.head, tail=args.tail, dspark=args.dspark)
        if args.operation == "inventory":
            result = stage_storage_inventory(inspect_checkpoint(args.checkpoint), args.lo, args.hi, **roles)
        else:
            result = measure_checkpoint(args.checkpoint, args.lo, args.hi, **roles, device=args.device,
                                        max_seq=args.max_seq, prefill_tokens=args.prefill_tokens, decode_tokens=args.decode_tokens)
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
        if args.output:
            Path(args.output).write_text(encoded + "\n", encoding="utf-8")
        else:
            print(encoded)
        return 0
    except (ResourceError, OSError, ImportError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}", "placement_requirements": None}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Packing-independent model identity and fully verified node-local weight artifacts.

The catalogue commits logical tensor bytes, shapes and dtypes. A packing commits
containers and offsets. A stage derives an exact role/span subset from the full
catalogue; possessing that subset never proves local possession of the full model.
No torch, GPU, HTTP, SSH or model execution is performed by this module.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct

GLOBAL_FILE = ".shard-model-artifacts.json"
PACK_FILE = ".shard-weight-pack.json"
STAGE_FILE = ".shard-stage-artifacts.json"
CATALOG_SCHEMA = "shard-model-artifacts/1"
PACK_SCHEMA = "shard-weight-pack/1"
STAGE_SCHEMA = "shard-stage-artifacts/1"
MAX_METADATA = 128 << 20
MAX_HEADER = 16 << 20
BITS = {"BOOL": 8, "U8": 8, "I8": 8, "I16": 16, "U16": 16,
        "I32": 32, "U32": 32, "I64": 64, "U64": 64, "F16": 16,
        "BF16": 16, "F32": 32, "F64": 64, "C64": 64, "F4": 4,
        "F6_E2M3": 6, "F6_E3M2": 6, "F8_E4M3": 8, "F8_E5M2": 8,
        "F8_E8M0": 8, "F8_E4M3FNUZ": 8, "F8_E5M2FNUZ": 8}
_PROOF = object()


class ArtifactError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError("duplicate metadata field")
        result[key] = value
    return result


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_METADATA:
        raise ArtifactError("invalid artifact metadata file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_METADATA + 1)
    if len(raw) > MAX_METADATA:
        raise ArtifactError("artifact metadata exceeds limit")
    value = json.loads(raw, object_pairs_hook=_object,
        parse_constant=lambda _v: (_ for _ in ()).throw(ArtifactError("nonfinite metadata")))
    if not isinstance(value, dict):
        raise ArtifactError("metadata must be an object")
    return value


def relative_path(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or any(ord(c) < 32 for c in value)):
        raise ArtifactError("unsafe artifact path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/")):
        raise ArtifactError("unsafe artifact path")
    return value


def safe_path(directory, relative):
    relative_path(relative)
    root = Path(directory).resolve()
    target = root.joinpath(*relative.split("/"))
    cursor = target
    while cursor != root:
        if cursor.is_symlink():
            raise ArtifactError("artifact path crosses a symlink")
        cursor = cursor.parent
    if not target.resolve().is_relative_to(root):
        raise ArtifactError("artifact escapes its directory")
    return target


def _count(value, label):
    if type(value) is not int or value < 0:
        raise ArtifactError(f"{label} must be a nonnegative integer")
    return value


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ArtifactError("expected a lowercase SHA-256")
    return value


def hash_file(path, *, chunk_bytes=1 << 20):
    value, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while part := stream.read(chunk_bytes):
            value.update(part); size += len(part)
    return value.hexdigest(), size


def fsync_directory(path):
    """Flush directory entries on POSIX; Windows retains atomic rename semantics."""
    if os.name == "nt": return
    fd = os.open(Path(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _tensor(record):
    if not isinstance(record, dict) or set(record) != {"dtype", "shape", "size", "sha256"}:
        raise ArtifactError("invalid logical tensor record")
    if record["dtype"] not in BITS or not isinstance(record["shape"], list):
        raise ArtifactError("invalid logical dtype/shape")
    for value in record["shape"]:
        _count(value, "dimension")
    bit_count = math.prod(record["shape"]) * BITS[record["dtype"]]
    if bit_count % 8 or bit_count // 8 != _count(record["size"], "tensor size"):
        raise ArtifactError("tensor shape does not describe its bytes")
    _sha(record["sha256"])


def catalog_digest(catalog):
    return digest({key: value for key, value in catalog.items()
                   if key not in ("checkpoint_id", "manifest_sha256", "source")})


def _catalog_manifest_digest(catalog):
    return digest({key: value for key, value in catalog.items()
                   if key not in ("checkpoint_id", "manifest_sha256")})


def validate_catalog(value):
    if not isinstance(value, dict) or value.get("schema") != CATALOG_SCHEMA:
        raise ArtifactError("logical model catalogue required")
    catalog = deepcopy(value)
    if not isinstance(catalog.get("model_id"), str) or not catalog["model_id"]:
        raise ArtifactError("model_id required")
    config = catalog.get("config")
    if not isinstance(config, dict) or type(config.get("n_layers")) is not int or config["n_layers"] < 1:
        raise ArtifactError("native model config with n_layers required")
    if not isinstance(catalog.get("tensors"), dict) or not catalog["tensors"]:
        raise ArtifactError("nonempty logical tensor catalogue required")
    for name, record in catalog["tensors"].items():
        if not isinstance(name, str) or not name:
            raise ArtifactError("invalid tensor name")
        _tensor(record)
    assets = catalog.get("assets")
    if not isinstance(assets, dict) or "config.json" not in assets:
        raise ArtifactError("config asset required")
    for name, record in assets.items():
        relative_path(name)
        if not isinstance(record, dict) or set(record) != {"size", "sha256"}:
            raise ArtifactError("invalid model asset record")
        _count(record["size"], "asset size"); _sha(record["sha256"])
    if catalog.get("config_sha256") != assets["config.json"]["sha256"]:
        raise ArtifactError("config identity differs from asset")
    value_digest = catalog_digest(catalog)
    if (catalog.get("checkpoint_id") != "tensor-sha256:" + value_digest
            or catalog.get("manifest_sha256") != _catalog_manifest_digest(catalog)):
        raise ArtifactError("logical model identity mismatch")
    return catalog


def make_catalog(model_id, config, tensors, assets, *, source=None, runtime_abi="deepseek-v4-native/1"):
    body = {"schema": CATALOG_SCHEMA, "model_id": model_id, "runtime_abi": runtime_abi,
            "config": deepcopy(config), "config_sha256": assets["config.json"]["sha256"],
            "tensors": deepcopy(tensors), "assets": deepcopy(assets), "source": deepcopy(source or {})}
    root = catalog_digest(body)
    body.update(checkpoint_id="tensor-sha256:" + root, manifest_sha256=_catalog_manifest_digest(body))
    return validate_catalog(body)


def validate_native_production_config(cfg):
    expected = {"n_layers": 43, "dim": 4096, "n_routed_experts": 256,
                "n_activated_experts": 6, "hc_mult": 4, "n_mtp_layers": 3,
                "dspark_target_layer_ids": [40, 41, 42], "dtype": "fp8",
                "expert_dtype": "fp4", "scale_fmt": "ue8m0"}
    if not isinstance(cfg, dict) or any(type(cfg.get(k)) is not type(v) or cfg.get(k) != v for k, v in expected.items()):
        raise ArtifactError("production cohort requires the supported native V4-Flash architecture and FP4/FP8 ABI")
    return deepcopy(cfg)


def catalog_model_cohort(catalog):
    """Build the supported production cohort from an immutable native catalogue.

    This descriptor binds identity and ABI; it does not claim GPU numerical or
    resource calibration. Tiny CPU fixtures deliberately cannot use it as a
    production model declaration.
    """
    catalog = validate_catalog(catalog)
    cfg = catalog["config"]
    validate_native_production_config(cfg)
    if catalog.get("runtime_abi") != "deepseek-v4-native/1":
        raise ArtifactError("production cohort requires the supported native V4-Flash architecture and FP4/FP8 ABI")
    from .offers import ModelCohort
    return ModelCohort.from_dict({"model_id": catalog["model_id"],
        "manifest_sha256": catalog["manifest_sha256"], "checkpoint_id": catalog["checkpoint_id"],
        "config_sha256": catalog["config_sha256"], "quantization": "fp4-fp8",
        "runtime_abi": "deepseek-v4-native/1", "wire_version": "shard-pipeline-session/1",
        "numeric_contract": "greedy-native-fp4-fp8/1", "n_layers": cfg["n_layers"]}).to_dict()


def _file_record(record):
    if not isinstance(record, dict):
        raise ArtifactError("file descriptor required")
    relative_path(record.get("path")); _count(record.get("size"), "file size"); _sha(record.get("sha256"))


def validate_pack(catalog, value, *, complete=False):
    catalog = validate_catalog(catalog)
    if not isinstance(value, dict) or value.get("schema") != PACK_SCHEMA:
        raise ArtifactError("weight packing manifest required")
    pack = deepcopy(value)
    if (pack.get("checkpoint_id"), pack.get("manifest_sha256")) != (catalog["checkpoint_id"], catalog["manifest_sha256"]):
        raise ArtifactError("packing belongs to another model")
    files = pack.get("files")
    if not isinstance(files, list) or not files:
        raise ArtifactError("packing files required")
    by_path = {}
    for record in files:
        _file_record(record)
        if record["path"] in by_path:
            raise ArtifactError("duplicate packed file")
        by_path[record["path"]] = record
    weight_map, offsets = pack.get("weight_map"), pack.get("offsets")
    if not isinstance(weight_map, dict) or not isinstance(offsets, dict) or set(weight_map) != set(offsets):
        raise ArtifactError("packing tensor maps differ")
    if not set(weight_map) <= set(catalog["tensors"]) or (complete and set(weight_map) != set(catalog["tensors"])):
        raise ArtifactError("packing tensor coverage differs from catalogue")
    spans = {}
    for name, filename in weight_map.items():
        if filename not in by_path:
            raise ArtifactError("tensor references an absent file")
        file = by_path[filename]
        size = _count(file.get("header_bytes"), "safetensors header size")
        if not 2 <= size <= MAX_HEADER:
            raise ArtifactError("invalid packed header size")
        _sha(file.get("header_sha256"))
        span = offsets[name]
        if not isinstance(span, list) or len(span) != 2:
            raise ArtifactError("invalid tensor byte range")
        start, end = (_count(x, "tensor offset") for x in span)
        if end - start != catalog["tensors"][name]["size"] or end > file["size"] - 8 - size:
            raise ArtifactError("tensor range differs from catalogue")
        spans.setdefault(filename, []).append((start, end))
    for filename, ranges in spans.items():
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                raise ArtifactError("packed tensor ranges overlap or leave holes")
            cursor = end
        if cursor + 8 + by_path[filename]["header_bytes"] != by_path[filename]["size"]:
            raise ArtifactError("packed file contains unindexed bytes")
    used = set(weight_map.values()) | set(catalog["assets"])
    if set(by_path) != used:
        raise ArtifactError("packing contains undeclared files")
    for name, asset in catalog["assets"].items():
        row = by_path.get(name)
        if row is None or any(row[field] != asset[field] for field in ("size", "sha256")):
            raise ArtifactError("packing model asset differs from catalogue")
    return pack


def _required(catalog, lo, hi, head, tail, dspark):
    layers = catalog["config"]["n_layers"]
    if (type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= layers
            or any(type(flag) is not bool for flag in (head, tail, dspark))):
        raise ArtifactError("invalid stage span/roles")
    if head and lo != 0 or tail and hi != layers:
        raise ArtifactError("boundary role differs from stage span")
    taps = catalog["config"].get("dspark_target_layer_ids", [])
    if dspark and (not tail or not all(lo <= i < hi for i in taps)):
        raise ArtifactError("DSpark tail must own every tap layer")
    names = []
    for name in catalog["tensors"]:
        layer = re.match(r"^layers\.(\d+)\.", name)
        draft = re.match(r"^mtp\.(\d+)\.", name)
        if layer and lo <= int(layer[1]) < hi or draft and dspark:
            names.append(name)
        elif name == "embed.weight" and (head or dspark):
            names.append(name)
        elif name in {"norm.weight", "head.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"} and tail:
            names.append(name)
    observed = {int(re.match(r"^layers\.(\d+)\.", n)[1]) for n in names if n.startswith("layers.")}
    if observed != set(range(lo, hi)):
        raise ArtifactError("catalogue does not cover assigned layers")
    boundary = ({"embed.weight"} if head or dspark else set()) | (
        {"norm.weight", "head.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"} if tail else set())
    if not boundary <= set(names):
        raise ArtifactError("catalogue lacks boundary parameters")
    if dspark:
        drafts = {int(re.match(r"^mtp\.(\d+)\.", n)[1]) for n in names if n.startswith("mtp.")}
        if drafts != set(range(catalog["config"].get("n_mtp_layers", 3))):
            raise ArtifactError("catalogue lacks complete MTP blocks")
        if any(re.fullmatch(r"mtp\.\d+\.(embed|head)\.weight", n) for n in names):
            raise ArtifactError("MTP embed/head must alias main boundary tensors")
    return sorted(names)


def artifact_id(stage):
    return digest({k: v for k, v in stage.items() if k != "artifact_id"})


def select_stage_artifacts(catalog, pack, lo, hi, *, head=False, tail=False, dspark=False):
    catalog, pack = validate_catalog(catalog), validate_pack(catalog, pack)
    names = _required(catalog, lo, hi, head, tail, dspark)
    if not set(names) <= set(pack["weight_map"]):
        raise ArtifactError("source packing does not contain this stage")
    paths = {pack["weight_map"][n] for n in names} | set(catalog["assets"])
    result = {"schema": STAGE_SCHEMA, "model_id": catalog["model_id"],
              "checkpoint_id": catalog["checkpoint_id"], "manifest_sha256": catalog["manifest_sha256"],
              "config_sha256": catalog["config_sha256"], "lo": lo, "hi": hi,
              "head": head, "tail": tail, "dspark": dspark, "tensor_names": names,
              "files": sorted((deepcopy(f) for f in pack["files"] if f["path"] in paths), key=lambda f: f["path"])}
    result["artifact_id"] = artifact_id(result)
    return result


class VerifiedStageArtifacts(dict):
    def __init__(self, value, *, proof=None, stats=None):
        super().__init__(value)
        self._proof = proof
        self._fingerprint = digest(value)
        self._stats = deepcopy(stats) if proof is _PROOF and stats is not None else {}
        if proof is _PROOF and (stats is None or _file_stats(value) != self._stats):
            raise ArtifactError("artifact changed before its verification witness was sealed")


class VerifiedWeightPack(VerifiedStageArtifacts):
    pass


def verified_stage_artifact_descriptor(value):
    if (not isinstance(value, VerifiedStageArtifacts) or value._proof is not _PROOF
            or value._fingerprint != digest(value)):
        raise ArtifactError("actual unchanged local payload verification required")
    if _file_stats(value) != value._stats:
        raise ArtifactError("verified artifact files changed or were replaced")
    return deepcopy(value["stage"] if "stage" in value else {k: v for k, v in value.items()
        if k not in {"catalog", "pack", "config", "verification_scope", "directory", "payload_integrity_verified"}})


def _file_stats(value):
    directory = value.get("directory")
    if not directory:
        raise ArtifactError("local verification directory required")
    names = {row["path"] for row in value["pack"]["files"]} | {GLOBAL_FILE, PACK_FILE}
    if "stage" in value: names.add(STAGE_FILE)
    result = {"__weight_files__": tuple(sorted(p.name for p in Path(directory).glob("*.safetensors")))}
    for name in names:
        try:
            info = safe_path(directory, name).stat()
        except OSError:
            raise ArtifactError("missing or unreadable artifact file: " + name) from None
        result[name] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    return result


def relocate_verified_artifacts(value, directory):
    """Retain proof across an atomic directory rename, never across a copy/rewrite."""
    if not isinstance(value, VerifiedStageArtifacts) or value._proof is not _PROOF or value._fingerprint != digest(value):
        raise ArtifactError("unchanged verified artifact required for relocation")
    body = deepcopy(dict(value)); body["directory"] = str(Path(directory).resolve())
    if _file_stats(body) != value._stats:
        raise ArtifactError("relocation changed the verified files")
    return type(value)(body, proof=_PROOF, stats=value._stats)


def _header(path):
    """Model-independent packed safetensors validation, without torch imports."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8: raise ArtifactError("truncated safetensors prefix")
        amount = struct.unpack("<Q", prefix)[0]
        if not 2 <= amount <= MAX_HEADER or amount > size - 8:
            raise ArtifactError("invalid safetensors header length")
        blob = stream.read(amount)
    header = json.loads(blob, object_pairs_hook=_object,
        parse_constant=lambda _v: (_ for _ in ()).throw(ArtifactError("nonfinite safetensors metadata")))
    if not isinstance(header, dict): raise ArtifactError("safetensors header must be an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise ArtifactError("safetensors metadata must map strings to strings")
    tensors, ranges = {}, []
    for name, row in header.items():
        if not name or not isinstance(row, dict) or set(row) != {"dtype", "shape", "data_offsets"}:
            raise ArtifactError("invalid safetensors tensor record")
        dtype, shape, offsets = row["dtype"], row["shape"], row["data_offsets"]
        if not isinstance(dtype, str) or dtype not in BITS or not isinstance(shape, list):
            raise ArtifactError("invalid safetensors dtype/shape")
        if not isinstance(offsets, list) or len(offsets) != 2: raise ArtifactError("invalid tensor offsets")
        for value in shape: _count(value, "dimension")
        start, end = (_count(value, "offset") for value in offsets)
        bits = math.prod(shape) * BITS[dtype]
        if end < start or end > size - 8 - amount or bits % 8 or bits // 8 != end - start:
            raise ArtifactError("safetensors shape/byte range mismatch")
        tensors[name] = {**row, "storage_bytes": end - start}
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor: raise ArtifactError("safetensors offsets overlap or leave holes")
        cursor = end
    if cursor != size - 8 - amount: raise ArtifactError("unindexed safetensors bytes")
    return {"file": path.name, "file_bytes": size, "header_bytes": amount,
            "header_sha256": hashlib.sha256(prefix + blob).hexdigest(), "tensors": tensors,
            "metadata": metadata}


def _verify_files(directory, catalog, pack, *, verify_files):
    observed = {}
    for file in pack["files"]:
        path = safe_path(directory, file["path"])
        before = path.stat()
        if not path.is_file() or path.stat().st_size != file["size"]:
            raise ArtifactError("missing or wrongly sized artifact file: " + file["path"])
        if verify_files and hash_file(path) != (file["sha256"], file["size"]):
            raise ArtifactError("artifact payload hash mismatch: " + file["path"])
        if file["path"] in catalog["assets"]:
            # Config/encoder assets are small but must be verified even in preview.
            if hash_file(path) != (file["sha256"], file["size"]):
                raise ArtifactError("model asset hash mismatch")
            after = path.stat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ArtifactError("model asset changed during verification")
            continue
        header = _header(path)
        if header["header_bytes"] != file["header_bytes"] or header["header_sha256"] != file["header_sha256"]:
            raise ArtifactError("packed header identity mismatch")
        for name, record in header["tensors"].items():
            if name in observed or pack["weight_map"].get(name) != file["path"]:
                raise ArtifactError("duplicate or unindexed tensor")
            expected = catalog["tensors"][name]
            if (record["dtype"], record["shape"], record["storage_bytes"], record["data_offsets"]) != (
                    expected["dtype"], expected["shape"], expected["size"], pack["offsets"][name]):
                raise ArtifactError("packed tensor metadata differs from catalogue")
            if verify_files:
                sha = hashlib.sha256()
                with path.open("rb") as stream:
                    stream.seek(8 + header["header_bytes"] + record["data_offsets"][0])
                    remaining = expected["size"]
                    while remaining:
                        chunk = stream.read(min(remaining, 1 << 20))
                        if not chunk:
                            raise ArtifactError("truncated tensor payload")
                        sha.update(chunk); remaining -= len(chunk)
                if sha.hexdigest() != expected["sha256"]:
                    raise ArtifactError("logical tensor payload mismatch")
            observed[name] = record
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ArtifactError("artifact changed during verification")
    if set(observed) != set(pack["weight_map"]):
        raise ArtifactError("local tensor coverage mismatch")
    actual_weights = {p.name for p in Path(directory).glob("*.safetensors")}
    expected_weights = {f["path"] for f in pack["files"] if f["path"] not in catalog["assets"]}
    if actual_weights != expected_weights:
        raise ArtifactError("unverified root weight file overrides artifact packing")
    config = read_json(safe_path(directory, "config.json"))
    if config != catalog["config"]:
        raise ArtifactError("native config differs from catalogue")


def _local(directory, expected_checkpoint_id, expected_manifest_sha256, complete):
    metadata_stats = {}
    for name in (GLOBAL_FILE, PACK_FILE):
        path = safe_path(directory, name)
        if not path.is_file():
            raise ArtifactError("required artifact metadata is missing: " + name)
        info = path.stat()
        metadata_stats[name] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    catalog = validate_catalog(read_json(Path(directory) / GLOBAL_FILE))
    if expected_checkpoint_id is not None and catalog["checkpoint_id"] != expected_checkpoint_id:
        raise ArtifactError("checkpoint differs from pinned model")
    if expected_manifest_sha256 is not None and catalog["manifest_sha256"] != expected_manifest_sha256:
        raise ArtifactError("catalogue differs from pinned model")
    pack = validate_pack(catalog, read_json(Path(directory) / PACK_FILE), complete=complete)
    for name, expected in metadata_stats.items():
        info = safe_path(directory, name).stat()
        if expected != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns):
            raise ArtifactError("artifact metadata changed while being parsed")
    return catalog, pack, metadata_stats


def verify_weight_pack(directory, *, expected_checkpoint_id=None, expected_manifest_sha256=None, verify_files=True):
    catalog, pack, metadata_stats = _local(directory, expected_checkpoint_id, expected_manifest_sha256, True)
    body = {"catalog": catalog, "pack": pack, "config": catalog["config"],
        "checkpoint_id": catalog["checkpoint_id"], "manifest_sha256": catalog["manifest_sha256"],
        "directory": str(Path(directory).resolve()), "payload_integrity_verified": bool(verify_files),
        "verification_scope": "complete packed payload hashes" if verify_files else "metadata preview only"}
    before = _file_stats(body)
    if any(before[name] != stamp for name, stamp in metadata_stats.items()):
        raise ArtifactError("artifact metadata changed before payload verification")
    _verify_files(directory, catalog, pack, verify_files=verify_files)
    return VerifiedWeightPack(body, proof=_PROOF if verify_files else None, stats=before)


def verify_stage_artifacts(directory, *, expected_checkpoint_id=None, expected_manifest_sha256=None,
                           lo=None, hi=None, head=None, tail=None, dspark=None, verify_files=True):
    catalog, pack, metadata_stats = _local(directory, expected_checkpoint_id, expected_manifest_sha256, False)
    info = safe_path(directory, STAGE_FILE).stat()
    metadata_stats[STAGE_FILE] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    stage = read_json(Path(directory) / STAGE_FILE)
    if stage.get("schema") != STAGE_SCHEMA or stage.get("artifact_id") != artifact_id(stage):
        raise ArtifactError("stage artifact identity mismatch")
    expected = select_stage_artifacts(catalog, pack, stage["lo"], stage["hi"],
        head=stage["head"], tail=stage["tail"], dspark=stage["dspark"])
    if expected != stage:
        raise ArtifactError("stage descriptor differs from catalogue/packing")
    for key, actual in (("lo", lo), ("hi", hi), ("head", head), ("tail", tail), ("dspark", dspark)):
        if actual is not None and (type(actual) is not type(stage[key]) or actual != stage[key]):
            raise ArtifactError("stage artifact assignment differs: " + key)
    body = {**stage, "stage": stage, "catalog": catalog, "pack": pack,
            "config": catalog["config"], "directory": str(Path(directory).resolve()),
            "payload_integrity_verified": bool(verify_files),
            "verification_scope": "assigned stage payload hashes" if verify_files else "metadata preview only"}
    before = _file_stats(body)
    if any(before[name] != stamp for name, stamp in metadata_stats.items()):
        raise ArtifactError("artifact metadata changed before payload verification")
    _verify_files(directory, catalog, pack, verify_files=verify_files)
    return VerifiedStageArtifacts(body, proof=_PROOF if verify_files else None, stats=before)


def write_metadata(directory, catalog, pack, stage=None):
    directory = Path(directory)
    for name, body in ((GLOBAL_FILE, catalog), (PACK_FILE, pack), (STAGE_FILE, stage)):
        if body is not None:
            path = safe_path(directory, name)
            with path.open("xb") as out:
                out.write(canonical(body) + b"\n"); out.flush(); os.fsync(out.fileno())
    fsync_directory(directory)


def catalogue_directory(directory, model_id, *, source=None, runtime_abi="deepseek-v4-native/1"):
    """Explicitly publish a full already-native checkpoint with streamed payload hashes."""
    directory = Path(directory)
    config = read_json(directory / "config.json")
    tensors, weight_map, offsets, files, assets = {}, {}, {}, [], {}
    for path in sorted(directory.glob("model*-mp1.safetensors")):
        header = _header(path)
        file_sha, size = hash_file(path)
        files.append({"path": path.name, "size": size, "sha256": file_sha,
            "header_bytes": header["header_bytes"], "header_sha256": header["header_sha256"]})
        with path.open("rb") as stream:
            for name, row in header["tensors"].items():
                if name in tensors:
                    raise ArtifactError("duplicate native tensor name")
                stream.seek(8 + header["header_bytes"] + row["data_offsets"][0])
                remaining, sha = row["storage_bytes"], hashlib.sha256()
                while remaining:
                    chunk = stream.read(min(remaining, 1 << 20))
                    if not chunk: raise ArtifactError("truncated native tensor")
                    sha.update(chunk); remaining -= len(chunk)
                tensors[name] = {"dtype": row["dtype"], "shape": row["shape"],
                    "size": row["storage_bytes"], "sha256": sha.hexdigest()}
                weight_map[name], offsets[name] = path.name, row["data_offsets"]
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        path = directory / name
        if path.is_file():
            sha, size = hash_file(path); assets[name] = {"size": size, "sha256": sha}
            files.append({"path": name, **assets[name]})
    catalog = make_catalog(model_id, config, tensors, assets, source=source, runtime_abi=runtime_abi)
    pack = {"schema": PACK_SCHEMA, "checkpoint_id": catalog["checkpoint_id"],
            "manifest_sha256": catalog["manifest_sha256"], "files": files,
            "weight_map": weight_map, "offsets": offsets}
    return catalog, validate_pack(catalog, pack, complete=True)


def _groups(catalog, names, maximum):
    groups, pending, size, prior = [], [], 0, None
    for name in sorted(names):
        match = re.match(r"^((?:layers|mtp)\.\d+)\.", name)
        family = match[1] if match else name
        amount = catalog["tensors"][name]["size"]
        if pending and (family != prior or size + amount > maximum):
            groups.append(pending); pending, size = [], 0
        pending.append(name); size += amount; prior = family
    if pending: groups.append(pending)
    return groups


def repack_storage_bound(catalog, pack, stage, max_file_bytes=512 << 20, chunk_bytes=1 << 20):
    """Conservative pre-write disk and CPU buffer bounds from real tensor geometry.

    Hash strings have a fixed width; output metadata uses the same descriptors as
    the writer. Existing stage output is not discounted by an unverified claim.
    The bound covers one output directory plus metadata and bounded range reads.
    """
    catalog, pack = validate_catalog(catalog), validate_pack(catalog, pack)
    expected = select_stage_artifacts(catalog, pack, stage["lo"], stage["hi"],
        head=stage["head"], tail=stage["tail"], dspark=stage["dspark"])
    if expected != stage or type(max_file_bytes) is not int or max_file_bytes < 1 or type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 16 << 20:
        raise ArtifactError("invalid stage packing budget")
    files, maps, offsets, max_header, max_file = [], {}, {}, 0, 0
    for index, names in enumerate(_groups(catalog, expected["tensor_names"], max_file_bytes)):
        header, cursor = {}, 0
        for name in names:
            row = catalog["tensors"][name]
            header[name] = {"dtype": row["dtype"], "shape": row["shape"], "data_offsets": [cursor, cursor + row["size"]]}
            cursor += row["size"]
        size = len(canonical(header)); size += -size % 8
        if size > MAX_HEADER: raise ArtifactError("packed header exceeds limit")
        filename = f"model{index:06d}-mp1.safetensors"
        files.append({"path": filename, "size": cursor + 8 + size, "sha256": "0" * 64,
            "header_bytes": size, "header_sha256": "0" * 64})
        max_header, max_file = max(max_header, size), max(max_file, cursor + 8 + size)
        for name in names:
            maps[name], offsets[name] = filename, header[name]["data_offsets"]
    files.extend({"path": name, **row} for name, row in catalog["assets"].items())
    local = {"schema": PACK_SCHEMA, "checkpoint_id": catalog["checkpoint_id"],
        "manifest_sha256": catalog["manifest_sha256"], "files": files, "weight_map": maps, "offsets": offsets}
    output_stage = select_stage_artifacts(catalog, local, stage["lo"], stage["hi"],
        head=stage["head"], tail=stage["tail"], dspark=stage["dspark"])
    metadata = sum(len(canonical(x)) + 1 for x in (catalog, local, output_stage))
    # JSON/Python objects need more than the serialized text while grouping and
    # validating; include copies and header maps rather than pretending zero RAM.
    return {"disk_peak_bytes": sum(f["size"] for f in files) + metadata,
        "final_bytes": sum(f["size"] for f in files) + metadata,
        "metadata_bytes": metadata, "largest_file_bytes": max_file,
        "host_ram_bytes": metadata * 12 + max_header * 12 + chunk_bytes * 4,
        "pinned_ram_bytes": 0, "scope": "conservative serialized geometry and bounded CPU I/O buffers"}


def repack_stage(catalog, pack, stage, destination, read_range, *, max_file_bytes=512 << 20,
                 chunk_bytes=1 << 20, cancel_check=None, progress=None):
    """Range-stream native tensors into bounded stage files, preserving exact bytes.

    `read_range(file_descriptor, absolute_offset, length)` must return precisely
    length bytes. Tensor hashes authenticate ranges without retaining a giant
    source container. A single oversized tensor has its own file. The destination
    must be an empty caller-owned staging directory; publication is a separate step.
    """
    if type(max_file_bytes) is not int or max_file_bytes < 1 or type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 16 << 20:
        raise ArtifactError("invalid bounded packing geometry")
    catalog, pack = validate_catalog(catalog), validate_pack(catalog, pack)
    expected = select_stage_artifacts(catalog, pack, stage["lo"], stage["hi"],
        head=stage["head"], tail=stage["tail"], dspark=stage["dspark"])
    if stage != expected:
        raise ArtifactError("source stage is not a derived catalogue subset")
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or any(directory.iterdir()):
        raise ArtifactError("repacking requires an empty private staging directory")
    source_files = {f["path"]: f for f in pack["files"]}
    files, maps, offsets = [], {}, {}
    transferred = 0
    def copy(out, row, offset, size, expected_sha):
        nonlocal transferred
        sha = hashlib.sha256()
        remaining = size
        while remaining:
            if cancel_check: cancel_check()
            count = min(remaining, chunk_bytes)
            data = read_range(row, offset, count)
            if not isinstance(data, bytes) or len(data) != count:
                raise ArtifactError("provider returned a wrong-sized tensor range")
            out.write(data); sha.update(data)
            offset += count; remaining -= count; transferred += count
            if progress: progress({"copied_bytes": transferred, "file": row["path"]})
        if sha.hexdigest() != expected_sha:
            raise ArtifactError("streamed logical payload failed its catalogue hash")
    for index, names in enumerate(_groups(catalog, expected["tensor_names"], max_file_bytes)):
        header, cursor = {}, 0
        for name in names:
            row = catalog["tensors"][name]
            header[name] = {"dtype": row["dtype"], "shape": row["shape"], "data_offsets": [cursor, cursor + row["size"]]}
            cursor += row["size"]
        header_blob = canonical(header)
        header_blob += b" " * (-len(header_blob) % 8)
        if len(header_blob) > MAX_HEADER: raise ArtifactError("packed header exceeds limit")
        prefix = struct.pack("<Q", len(header_blob)) + header_blob
        filename = f"model{index:06d}-mp1.safetensors"
        path = safe_path(directory, filename)
        with path.open("xb") as out:
            out.write(prefix)
            for name in names:
                source = source_files[pack["weight_map"][name]]
                start = 8 + source["header_bytes"] + pack["offsets"][name][0]
                copy(out, source, start, catalog["tensors"][name]["size"], catalog["tensors"][name]["sha256"])
                maps[name], offsets[name] = filename, header[name]["data_offsets"]
            out.flush(); os.fsync(out.fileno())
        sha, size = hash_file(path)
        files.append({"path": filename, "size": size, "sha256": sha,
            "header_bytes": len(header_blob), "header_sha256": hashlib.sha256(prefix).hexdigest()})
    for name, record in catalog["assets"].items():
        path = safe_path(directory, name); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as out:
            copy(out, source_files[name], 0, record["size"], record["sha256"])
            out.flush(); os.fsync(out.fileno())
        files.append({"path": name, **record})
    local = {"schema": PACK_SCHEMA, "checkpoint_id": catalog["checkpoint_id"],
             "manifest_sha256": catalog["manifest_sha256"], "files": files, "weight_map": maps, "offsets": offsets}
    actual_stage = select_stage_artifacts(catalog, local, stage["lo"], stage["hi"],
        head=stage["head"], tail=stage["tail"], dspark=stage["dspark"])
    write_metadata(directory, catalog, local, actual_stage)
    verify_stage_artifacts(directory, expected_checkpoint_id=catalog["checkpoint_id"],
        expected_manifest_sha256=catalog["manifest_sha256"])
    return actual_stage

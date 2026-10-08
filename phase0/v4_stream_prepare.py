"""Bounded V4 HF conversion and native stage export, without a full local output copy.

HF input files are fetched and hash-verified one at a time in an owned LRU cache.
Output files are bounded layer/role groups and are delivered immediately to local
or explicitly configured SSH destinations. Only the final immutable catalogue and
verified stage directories are published. This tool does not stop any GPU process.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from copy import deepcopy
import hashlib
import functools
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from shard import weight_artifacts as A


def native_name(name):
    if name.startswith("model."): name = name[6:]
    if name.startswith("mtp.") and ("emb" in name or name.endswith("head.weight")):
        return None
    return name.replace("self_attn", "attn").replace("mlp", "ffn").replace(
        "weight_scale_inv", "scale").replace("e_score_correction_bias", "bias")


class HFSource:
    """Verified input cache; cache deletion is confined to this owned temporary tree."""
    def __init__(self, cache, *, directory=None, repo=None, revision="main", token=False,
                 max_cache_bytes=16 << 30, downloader=None):
        if (directory is None) == (repo is None):
            raise A.ArtifactError("choose one local HF directory or remote repository")
        self.cache = Path(cache).resolve(); self.cache.mkdir(parents=True, exist_ok=True)
        self.directory = Path(directory).resolve() if directory else None
        self.repo, self.token = repo, token
        self.limit = max_cache_bytes
        if type(self.limit) is not int or self.limit < 1: raise A.ArtifactError("positive source cache limit required")
        self.live, self.verified, self.specs = OrderedDict(), {}, {}
        self._stats = {}
        self.downloader = downloader
        if repo:
            try:
                from phase0.get_model import resolve_files
            except ImportError:
                from get_model import resolve_files
            self.revision, specs = resolve_files(repo, revision, token=token)
            self.specs = {s.path: s for s in specs}
            self.names = sorted(self.specs)
        else:
            self.revision = None
            self.names = sorted(p.name for p in self.directory.iterdir() if p.is_file() and not p.is_symlink())
        self.headers, self.weight_map = {}, {}
        for name in self.names:
            if not name.endswith(".safetensors"): continue
            path = self.file(name)
            header = A._header(path)
            self.headers[name] = header
            for tensor in header["tensors"]:
                if tensor in self.weight_map: raise A.ArtifactError("duplicate HF tensor name")
                self.weight_map[tensor] = name
        if not self.weight_map: raise A.ArtifactError("HF source has no safetensors weights")

    def file(self, name):
        A.relative_path(name)
        if self.directory:
            path = A.safe_path(self.directory, name)
            if not path.is_file(): raise A.ArtifactError("required HF source file absent")
            if name not in self.verified:
                sha, size = A.hash_file(path); self.verified[name] = {"sha256": sha, "size": size}
                info = path.stat(); self._stats[name] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            info = path.stat()
            if self._stats[name] != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns):
                raise A.ArtifactError("verified HF input file changed")
            return path
        spec = self.specs[name]
        if spec.size > self.limit: raise A.ArtifactError("single HF source file exceeds bounded cache budget")
        if name in self.live:
            info = self.live[name].stat()
            if self._stats[name] != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns):
                raise A.ArtifactError("verified source cache file changed")
            self.live.move_to_end(name); return self.live[name]
        while sum(self.verified[n]["size"] for n in self.live) + spec.size > self.limit:
            old, path = self.live.popitem(last=False)
            if not path.resolve().is_relative_to(self.cache) or path.is_symlink():
                raise A.ArtifactError("source cache escaped owned temporary directory")
            path.unlink()
        if shutil.disk_usage(self.cache).free < spec.size:
            raise A.ArtifactError("insufficient source-cache disk space")
        if self.downloader is None:
            try:
                from phase0.get_model import download_file
            except ImportError:
                from get_model import download_file
            downloader = download_file
        else: downloader = self.downloader
        url = f"https://huggingface.co/{quote(self.repo, safe='/')}/resolve/{self.revision}/{quote(name, safe='/')}"
        sha = downloader(self.cache, spec, url, token=self.token)
        self.verified[name] = {"size": spec.size, "sha256": sha}
        self.live[name] = A.safe_path(self.cache, name)
        info = self.live[name].stat(); self._stats[name] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        return self.live[name]

    def info(self, name):
        file = self.weight_map[name]
        return self.headers[file]["tensors"][name]

    def read(self, name, offset, count):
        file = self.weight_map[name]; row = self.info(name)
        if offset < 0 or count < 0 or offset + count > row["storage_bytes"]:
            raise A.ArtifactError("HF tensor range out of bounds")
        path = self.file(file)
        # The local source is explicitly supplied; still detect concurrent edits.
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            stamp = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            if stamp != self._stats[file]:
                raise A.ArtifactError("verified HF source was replaced before opening")
            stream.seek(8 + self.headers[file]["header_bytes"] + row["data_offsets"][0] + offset)
            data = stream.read(count)
            after_fd = os.fstat(stream.fileno())
        after = path.stat()
        if len(data) != count or any(stamp != (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
                for item in (after_fd, after)):
            raise A.ArtifactError("HF source changed or truncated during conversion")
        return data

    def provenance(self):
        return {"kind": "HF verified files" if self.repo else "explicit local HF files",
                "repo": self.repo, "revision": self.revision,
                "files": deepcopy(self.verified),
                "converter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _wo_a_chunks(source, name, scale_name, chunk_bytes):
    """The reference wo_a FP32 scale multiplication, in complete 128-row blocks."""
    import torch
    row, scale = source.info(name), source.info(scale_name)
    rows, cols = row["shape"]
    if rows % 128 or cols % 128 or scale["shape"] != [rows // 128, cols // 128]:
        raise A.ArtifactError("wo_a weight/scale geometry differs from reference")
    dtype = {"F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
             "BF16": torch.bfloat16, "F16": torch.float16}.get(row["dtype"])
    sdtype = {"F8_E8M0": torch.float8_e8m0fnu, "F32": torch.float32,
              "BF16": torch.bfloat16}.get(scale["dtype"])
    if dtype is None or sdtype is None: raise A.ArtifactError("unsupported wo_a scale dtype")
    element = A.BITS[row["dtype"]] // 8; selement = A.BITS[scale["dtype"]] // 8
    if scale["storage_bytes"] > A.MAX_HEADER:
        raise A.ArtifactError("wo_a scale tensor exceeds the bounded CPU scale buffer")
    # Fetch the small scale once. Weight and scale may live in different large
    # source shards that cannot coexist in the LRU disk cache.
    scale_payload = source.read(scale_name, 0, scale["storage_bytes"])
    for begin in range(0, rows, 128):
        raw = source.read(name, begin * cols * element, 128 * cols * element)
        offset = begin // 128 * (cols // 128) * selement
        scale_raw = scale_payload[offset:offset + cols // 128 * selement]
        weight = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(128, cols)
        scales = torch.frombuffer(bytearray(scale_raw), dtype=sdtype).reshape(1, cols // 128)
        weight = weight.unflatten(0, (1, 128)).unflatten(-1, (-1, 128)).float() * scales[:, None, :, None].float()
        payload = weight.flatten(2, 3).flatten(0, 1).bfloat16().contiguous().view(torch.uint8).numpy().tobytes()
        for offset in range(0, len(payload), chunk_bytes): yield payload[offset:offset + chunk_bytes]


class LocalTarget:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if self.root.exists(): raise A.ArtifactError("target must be a new immutable directory")
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.staging = Path(tempfile.mkdtemp(prefix=".v4-stream-", dir=self.root.parent))
        self._owner = os.urandom(24).hex()
        (self.staging / ".stream-owner").write_text(self._owner, encoding="ascii")

    def abort(self):
        if not self.staging.exists(): return
        if (self.staging.is_symlink() or self.staging.resolve().parent != self.root.parent
                or not self.staging.name.startswith(".v4-stream-")
                or (self.staging / ".stream-owner").read_text(encoding="ascii") != self._owner):
            raise A.ArtifactError("unpublished staging ownership cannot be confirmed")
        shutil.rmtree(self.staging)

    def preflight(self, required_bytes):
        if shutil.disk_usage(self.staging).free < required_bytes:
            raise A.ArtifactError("target filesystem cannot hold its complete stage artifact")

    def put(self, name, source, size, sha):
        target = A.safe_path(self.staging, name); target.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.staging).free < size: raise A.ArtifactError("target disk cannot hold next artifact")
        with Path(source).open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20); dst.flush(); os.fsync(dst.fileno())
        if A.hash_file(target) != (sha, size): raise A.ArtifactError("target transfer hash mismatch")

    def finish(self, catalog, pack, stage):
        A.write_metadata(self.staging, catalog, pack, stage)
        proof = A.verify_stage_artifacts(self.staging) if stage else A.verify_weight_pack(self.staging)
        (self.staging / ".stream-owner").unlink()
        try:
            os.rename(self.staging, self.root)
        except BaseException:
            (self.staging / ".stream-owner").write_text(self._owner, encoding="ascii")
            raise
        A.fsync_directory(self.root.parent)
        A.relocate_verified_artifacts(proof, self.root)
        return {"directory": str(self.root), "artifact_id": stage["artifact_id"] if stage else None}


class SSHTarget:
    """Explicit SSH source-to-node upload; private staging and immutable publication."""
    def __init__(self, config):
        from phase0.deploy_oss import ssh_argv
        self.config = dict(config)
        self.config["workspace"] = config.get("workspace", "/root/FlyAI")
        self.root = config["directory"]
        if not isinstance(self.root, str) or not self.root.startswith("/"):
            raise A.ArtifactError("SSH artifact directory must be an absolute remote path")
        self.staging = self.root + ".prepare-" + os.urandom(12).hex()
        self._owner = os.urandom(24).hex()
        ssh_argv(self.config)

    def _command(self, code):
        from phase0.deploy_oss import ssh_argv
        command = "cd " + shlex.quote(self.config["workspace"]) + " && " + shlex.quote(
            self.config.get("python", "python3")) + " -c " + shlex.quote(code)
        return ssh_argv(self.config, command)

    def preflight(self, required_bytes):
        code = ("from pathlib import Path; import shutil; "
            f"target=Path({self.root!r}); assert not target.exists() and not target.is_symlink(); "
            "target.parent.mkdir(parents=True,exist_ok=True); "
            f"assert shutil.disk_usage(target.parent).free >= {required_bytes}")
        result = subprocess.run(self._command(code), capture_output=True, timeout=60)
        if result.returncode: raise A.ArtifactError("SSH target lacks stage disk budget; remote details redacted")

    def abort(self):
        code = ("from pathlib import Path; import shutil; "
            f"p=Path({self.staging!r}); target=Path({self.root!r})\n"
            "if p.exists():\n"
            " assert not p.is_symlink() and p.parent==target.parent\n"
            f" assert p.name=={Path(self.staging).name!r} and (p/'.stream-owner').read_text()=={self._owner!r}\n"
            " shutil.rmtree(p)\n")
        result = subprocess.run(self._command(code), capture_output=True, timeout=60)
        if result.returncode: raise A.ArtifactError("SSH unpublished staging cleanup was not acknowledged")

    def put(self, name, source, size, sha):
        A.relative_path(name)
        code = ("from pathlib import Path; import hashlib,os,shutil,sys; from shard.weight_artifacts import safe_path; "
            f"root=Path({self.staging!r}); root.mkdir(parents=True,exist_ok=True); "
            f"owner=root/'.stream-owner'; assert not root.is_symlink(); "
            f"assert not owner.exists() or owner.read_text()=={self._owner!r}; owner.write_text({self._owner!r}); "
            f"p=safe_path(root,{name!r}); p.parent.mkdir(parents=True,exist_ok=True); "
            f"assert shutil.disk_usage(root).free>={size}; "
            f"expected={size}; sha=hashlib.sha256(); out=p.open('xb'); total=0\n"
            "while True:\n b=sys.stdin.buffer.read(min(1048576,expected-total+1))\n if not b: break\n"
            " if total+len(b)>expected: raise ValueError('upload exceeds artifact')\n out.write(b); sha.update(b); total+=len(b)\n"
            f"out.flush(); os.fsync(out.fileno()); out.close(); assert total==expected and sha.hexdigest()=={sha!r}")
        with Path(source).open("rb") as src:
            result = subprocess.run(self._command(code), stdin=src, capture_output=True, timeout=1800)
        if result.returncode: raise A.ArtifactError("SSH artifact upload failed; remote details redacted")

    def finish(self, catalog, pack, stage):
        payload = A.canonical({"catalog": catalog, "pack": pack, "stage": stage})
        code = ("import json,os,sys; from pathlib import Path; from shard.weight_artifacts import "
            "write_metadata,verify_stage_artifacts,verify_weight_pack,fsync_directory; "
            f"p=Path({self.staging!r}); target=Path({self.root!r}); assert not target.exists() and not target.is_symlink(); "
            f"d=json.loads(sys.stdin.buffer.read({A.MAX_METADATA + 1})); "
            "write_metadata(p,d['catalog'],d['pack'],d['stage']); "
            "v=verify_stage_artifacts(p) if d['stage'] else verify_weight_pack(p); "
            "(p/'.stream-owner').unlink()\n"
            "try: os.rename(p,target)\n"
            f"except BaseException:\n (p/'.stream-owner').write_text({self._owner!r}); raise\n"
            "fsync_directory(target.parent); print('published')")
        result = subprocess.run(self._command(code), input=payload, capture_output=True, timeout=1800)
        if result.returncode: raise A.ArtifactError("SSH artifact verification/publication failed; remote details redacted")
        return {"directory": self.root, "artifact_id": stage["artifact_id"] if stage else None}


def _cleanup_on_failure(function):
    @functools.wraps(function)
    def wrapped(source, native_config, targets, **kwargs):
        try:
            return function(source, native_config, targets, **kwargs)
        except BaseException as error:
            failures = []
            for sink, _ in targets:
                try: sink.abort()
                except Exception as cleanup: failures.append(cleanup)
            if failures:
                raise A.ArtifactError("unpublished artifact cleanup was not fully acknowledged; published models are retained") from error
            raise
    return wrapped


@_cleanup_on_failure
def convert_hf(source, native_config, targets, *, model_id="deepseek-ai/DeepSeek-V4-Flash",
               max_file_bytes=512 << 20, chunk_bytes=1 << 20, scratch=None, coordinator_metadata_path=None):
    """Convert exactly the reference mp1 FP4 format, bounded by one output file.

    Targets are [(sink, stage_roles_or_None)]. All layers must be represented for
    a full logical catalogue; sinks receive only their declared layer/role files.
    wo_a uses the original FP32 multiply and BF16 cast; packed FP4 bytes are copied.
    """
    if type(max_file_bytes) is not int or max_file_bytes < 1 or type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 16 << 20:
        raise A.ArtifactError("invalid streaming geometry")
    if not targets or not isinstance(native_config, dict) or type(native_config.get("n_layers")) is not int:
        raise A.ArtifactError("native config and at least one destination required")
    planned, original, scales = {}, {}, {}
    for name in source.weight_map:
        converted = native_name(name)
        if converted is None: continue
        row = source.info(name)
        if converted.endswith("wo_a.scale"): continue
        dtype, shape, size = row["dtype"], list(row["shape"]), row["storage_bytes"]
        if converted.endswith("wo_a.weight"):
            scale = next((n for n in source.weight_map if native_name(n) == converted.replace("weight", "scale")), None)
            if scale is None: raise A.ArtifactError("wo_a scale missing")
            dtype, size = "BF16", math_product(shape) * 2; scales[converted] = scale
        elif "experts" in converted and dtype == "I8":
            if native_config.get("expert_dtype") != "fp4":
                raise A.ArtifactError("stream converter preserves native FP4; FP8 expert recoding requires another version")
            dtype = "F4"; shape[-1] *= 2
        if converted in planned: raise A.ArtifactError("HF tensor names collide after conversion")
        planned[converted] = {"dtype": dtype, "shape": shape, "size": size, "sha256": "0" * 64}
        original[converted] = name
    skeleton = {"config": native_config, "tensors": planned}
    wanted = [set(planned) if roles is None else set(A._required(skeleton, **roles)) for _, roles in targets]
    if set.union(*wanted) != set(planned):
        raise A.ArtifactError("destinations must cover the full converted catalogue")
    groups = A._groups(skeleton, list(planned), max_file_bytes)
    target_bytes = [0] * len(targets)
    max_header, largest_output = 0, 0
    for names in groups:
        header, cursor = {}, 0
        for name in names:
            row = planned[name]
            header[name] = {"dtype": row["dtype"], "shape": row["shape"], "data_offsets": [cursor, cursor + row["size"]]}
            cursor += row["size"]
        header_bytes = len(A.canonical(header)); header_bytes += -header_bytes % 8
        if header_bytes > A.MAX_HEADER: raise A.ArtifactError("output group header exceeds bound")
        max_header = max(max_header, header_bytes)
        largest_output = max(largest_output, cursor + 8 + header_bytes)
        for index, selected in enumerate(wanted):
            if selected.intersection(names): target_bytes[index] += cursor + 8 + header_bytes
    asset_bytes = len(A.canonical(native_config)) + 1
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if name in source.names: asset_bytes += source.file(name).stat().st_size
    # Full catalogue and packing metadata are written on each target. The bound
    # includes repeated tensor/offset maps and Python JSON materialization, with
    # fixed-width future digest strings; it is not a measured GPU memory claim.
    metadata_bound = len(A.canonical(skeleton)) * 8 + max_header * 4 + 65536
    required_by_volume = {}
    volume_paths = {}
    def reserve_volume(path, amount):
        path = Path(path).resolve()
        device = path.stat().st_dev
        required_by_volume[device] = required_by_volume.get(device, 0) + amount
        volume_paths[device] = path
    for index, (sink, _) in enumerate(targets):
        required = target_bytes[index] + asset_bytes + metadata_bound
        if isinstance(sink, LocalTarget):
            reserve_volume(sink.staging, required)
    scratch_path = Path(scratch or tempfile.gettempdir()).resolve()
    # One bounded output scratch copy coexists with the target upload. Assets
    # are copied into owned scratch as well; source cache can grow after preflight.
    reserve_volume(scratch_path, largest_output + asset_bytes + metadata_bound)
    if source.directory is None:
        held = sum(p.stat().st_size for p in source.live.values())
        potential = min(source.limit, sum(s.size for s in source.specs.values()))
        reserve_volume(source.cache, max(0, potential - held))
    if coordinator_metadata_path:
        parent = Path(coordinator_metadata_path).resolve().parent
        parent.mkdir(parents=True, exist_ok=True)
        reserve_volume(parent, metadata_bound + asset_bytes)
    for device, need in required_by_volume.items():
        if shutil.disk_usage(volume_paths[device]).free < need:
            raise A.ArtifactError("shared filesystem cannot hold target outputs, conversion scratch and source-cache growth")
    for index, (sink, _) in enumerate(targets):
        required = target_bytes[index] + asset_bytes + metadata_bound
        if isinstance(sink, LocalTarget): required = required_by_volume[sink.staging.stat().st_dev]
        sink.preflight(required)
    files, maps, offsets, actual = [], {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="v4-convert-stream-", dir=scratch) as temporary:
        temporary = Path(temporary)
        for index, names in enumerate(groups):
            header, cursor = {}, 0
            for name in names:
                row = planned[name]
                header[name] = {"dtype": row["dtype"], "shape": row["shape"], "data_offsets": [cursor, cursor + row["size"]]}
                cursor += row["size"]
            blob = A.canonical(header); blob += b" " * (-len(blob) % 8)
            prefix = struct.pack("<Q", len(blob)) + blob
            path = temporary / f"model{index:06d}-mp1.safetensors"
            if shutil.disk_usage(temporary).free < len(prefix) + cursor:
                raise A.ArtifactError("scratch cannot hold one bounded output file")
            with path.open("xb") as out:
                out.write(prefix)
                for name in names:
                    sha, total = hashlib.sha256(), 0
                    if name in scales:
                        blocks = _wo_a_chunks(source, original[name], scales[name], chunk_bytes)
                    else:
                        def raw_blocks(n=original[name], size=planned[name]["size"]):
                            for offset in range(0, size, chunk_bytes):
                                yield source.read(n, offset, min(chunk_bytes, size - offset))
                        blocks = raw_blocks()
                    for block in blocks:
                        out.write(block); sha.update(block); total += len(block)
                    if total != planned[name]["size"]: raise A.ArtifactError("converted tensor byte count mismatch")
                    actual[name] = {**planned[name], "sha256": sha.hexdigest()}
                    maps[name], offsets[name] = path.name, header[name]["data_offsets"]
                out.flush(); os.fsync(out.fileno())
            sha, size = A.hash_file(path)
            record = {"path": path.name, "size": size, "sha256": sha,
                "header_bytes": len(blob), "header_sha256": hashlib.sha256(prefix).hexdigest()}
            files.append(record)
            for (sink, _), selected in zip(targets, wanted):
                if selected.intersection(names): sink.put(path.name, path, size, sha)
            path.unlink()  # only this converter's acknowledged scratch file
        assets = {}
        config_path = temporary / "config.json"; config_path.write_bytes(A.canonical(native_config) + b"\n")
        asset_paths = {"config.json": config_path}
        for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            if name in source.names:
                path = temporary / name
                with source.file(name).open("rb") as src, path.open("xb") as dst:
                    shutil.copyfileobj(src, dst, chunk_bytes)
                record = source.verified[name]
                if A.hash_file(path) != (record["sha256"], record["size"]):
                    raise A.ArtifactError("model asset changed from its verified HF source")
                asset_paths[name] = path
        for name, path in asset_paths.items():
            sha, size = A.hash_file(path); assets[name] = {"size": size, "sha256": sha}
            files.append({"path": name, **assets[name]})
            for sink, _ in targets: sink.put(name, path, size, sha)
        catalog = A.make_catalog(model_id, native_config, actual, assets, source=source.provenance())
        pack = {"schema": A.PACK_SCHEMA, "checkpoint_id": catalog["checkpoint_id"],
            "manifest_sha256": catalog["manifest_sha256"], "files": files, "weight_map": maps, "offsets": offsets}
        A.validate_pack(catalog, pack, complete=True)
        outputs = []
        for sink, roles in targets:
            if roles is None: local, stage = pack, None
            else:
                stage = A.select_stage_artifacts(catalog, pack, **roles)
                paths = {f["path"] for f in stage["files"]}
                names = {n: f for n, f in maps.items() if f in paths}
                local = {**pack, "files": stage["files"], "weight_map": names,
                         "offsets": {n: offsets[n] for n in names}}
            outputs.append(sink.finish(catalog, local, stage))
    return {"checkpoint_id": catalog["checkpoint_id"], "manifest_sha256": catalog["manifest_sha256"],
            "outputs": outputs, "catalog": catalog, "pack": pack, "source_cache_limit_bytes": source.limit,
            "max_output_payload_bytes": max_file_bytes, "single_tensor_may_exceed_output_group_limit": True}


def math_product(shape):
    import math
    return math.prod(shape)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    native = sub.add_parser("catalogue", help="hash an existing full native checkpoint into stable artifact metadata")
    native.add_argument("--source", required=True); native.add_argument("--model-id", required=True)
    native.add_argument("--out", required=True, help="new metadata-only directory; no copied giant weights")
    native.add_argument("--cohort-out", help="new supported production model cohort JSON")
    convert = sub.add_parser("convert", help="stream HF weights into bounded node-local native artifacts")
    origin = convert.add_mutually_exclusive_group(required=True)
    origin.add_argument("--source"); origin.add_argument("--repo")
    convert.add_argument("--revision", default="main"); convert.add_argument("--native-config", required=True)
    convert.add_argument("--model-id", default="deepseek-ai/DeepSeek-V4-Flash")
    destination = convert.add_mutually_exclusive_group(required=True)
    destination.add_argument("--out"); destination.add_argument("--stage-targets", help="JSON list with span/roles and directory or SSH node fields")
    convert.add_argument("--scratch", required=True); convert.add_argument("--cache-gib", type=int, default=16)
    convert.add_argument("--file-mib", type=int, default=512)
    convert.add_argument("--cohort-out", help="new supported production model cohort JSON")
    convert.add_argument("--metadata-out", help="new coordinator metadata/assets-only directory, without model weights")
    auth = convert.add_mutually_exclusive_group()
    auth.add_argument("--anonymous", action="store_true"); auth.add_argument("--token-env"); auth.add_argument("--token-file")
    args = parser.parse_args(argv)
    try:
        if args.command == "catalogue":
            catalog, pack = A.catalogue_directory(args.source, args.model_id)
            out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
            A.write_metadata(out, catalog, pack)
            if args.cohort_out:
                cohort = A.catalog_model_cohort(catalog)
                with Path(args.cohort_out).open("xb") as target: target.write(A.canonical(cohort) + b"\n")
            print(json.dumps({"checkpoint_id": catalog["checkpoint_id"], "manifest_sha256": catalog["manifest_sha256"], "metadata": str(out)}))
            return
        from phase0.get_model import select_token
        token = select_token(anonymous=args.anonymous, token_env=args.token_env, token_file=args.token_file)
        config = A.read_json(args.native_config)
        if args.repo or args.cohort_out:
            A.validate_native_production_config(config)
        if args.cohort_out and Path(args.cohort_out).exists():
            raise A.ArtifactError("cohort output must be a new file")
        if args.metadata_out and (Path(args.metadata_out).exists() or Path(args.metadata_out).is_symlink()):
            raise A.ArtifactError("metadata output must be a new immutable directory")
        if args.repo and args.repo != "deepseek-ai/DeepSeek-V4-Flash":
            raise A.ArtifactError("this converter's reference ABI targets DeepSeek-V4-Flash; other models require a separate adapter")
        # Fail before fetching large HF inputs when the CPU conversion backend
        # cannot express the reference's FP8 exponent scales.
        import torch
        if not hasattr(torch, "float8_e8m0fnu") or not hasattr(torch, "float8_e4m3fn"):
            raise A.ArtifactError("the installed torch lacks native FP8 conversion types; use the verified V4 runtime build")
        if args.file_mib < 1 or args.cache_gib < 1: raise A.ArtifactError("positive streaming limits required")
        scratch = Path(args.scratch).resolve(); scratch.mkdir(parents=True, exist_ok=True)
        targets = []
        if args.out: targets = [(LocalTarget(args.out), None)]
        else:
            rows = json.loads(Path(args.stage_targets).read_text(encoding="utf-8-sig"))
            if not isinstance(rows, list): raise A.ArtifactError("stage destinations must be a JSON list")
            for row in rows:
                roles = {name: row[name] for name in ("lo", "hi", "head", "tail", "dspark")}
                targets.append((SSHTarget(row) if row.get("ssh_target") else LocalTarget(row["directory"]), roles))
        with tempfile.TemporaryDirectory(prefix="v4-hf-input-", dir=scratch) as cache:
            source = HFSource(cache, directory=args.source, repo=args.repo, revision=args.revision,
                              token=token, max_cache_bytes=args.cache_gib << 30)
            result = convert_hf(source, config, targets, model_id=args.model_id,
                                max_file_bytes=args.file_mib << 20, scratch=scratch,
                                coordinator_metadata_path=args.metadata_out)
            if args.metadata_out:
                metadata = Path(args.metadata_out); metadata.mkdir(parents=True, exist_ok=False)
                A.write_metadata(metadata, result["catalog"], result["pack"])
                (metadata / "config.json").write_bytes(A.canonical(config) + b"\n")
                for name in result["catalog"]["assets"]:
                    if name == "config.json": continue
                    with source.file(name).open("rb") as src, A.safe_path(metadata, name).open("xb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                    record = result["catalog"]["assets"][name]
                    if A.hash_file(A.safe_path(metadata, name)) != (record["sha256"], record["size"]):
                        raise A.ArtifactError("coordinator asset differs from the published model")
                (metadata / "outputs.json").write_bytes(A.canonical({"scope": "coordinator metadata/assets only",
                    "weight_payload_present": False, "outputs": result["outputs"]}) + b"\n")
                result["metadata_directory"] = str(metadata.resolve())
        if args.cohort_out:
            # Every target has the same catalogue. Keep its coordinator-only
            # descriptor beside the explicit destination metadata path below.
            catalog = result.pop("catalog")
            with Path(args.cohort_out).open("xb") as target:
                target.write(A.canonical(A.catalog_model_cohort(catalog)) + b"\n")
        else:
            result.pop("catalog", None)
        result.pop("pack", None)
        print(json.dumps(result))
    except (A.ArtifactError, ValueError, OSError, KeyError) as error:
        for sink, _ in locals().get("targets", []):
            try: sink.abort()
            except Exception: pass  # convert_hf already surfaces unacknowledged cleanup
        parser.error(str(error))


if __name__ == "__main__": main()

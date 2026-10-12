"""Bundle unpublished local acceptance sources; verify a TGZ without extracting.

Only read-only Git inventory/HEAD commands run. Payloads are current working-file
bytes, including the explicitly named untracked acceptance kit. No network,
weights, GPU evidence, private keys, process state or Git mutations are involved.
The manifest is a local content-integrity record, not a publisher signature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "flyai-v4-acceptance-source-bundle/1"
PREFIX = "FlyAI/"
MANIFEST = PREFIX + "BUNDLE_MANIFEST.json"
MAX_FILE = 32 * 1024**2
MAX_TOTAL = 256 * 1024**2
MAX_MANIFEST = 4 * 1024**2
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_EXCLUDED_DIRS = frozenset({".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules",
    "temp_test", "scratchpad", "models", "weights", "run", "runs", "private", "keys", "build", "dist",
    ".shard-processes", ".shard-deployments", ".flyai-state", ".claude", "acceptance-runs"})
_EXCLUDED_SUFFIXES = frozenset({".key", ".pem", ".p12", ".pfx", ".psk", ".safetensors", ".pt", ".pth",
    ".onnx", ".ckpt", ".bin", ".pyc", ".pyo", ".log", ".db", ".sqlite", ".sqlite3", ".exe", ".dll",
    ".so", ".dylib", ".o", ".a", ".tgz", ".gz", ".zip"})


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _safe_relative(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value
            or value.startswith("/") or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError("unsafe archive/source path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError("unsafe archive/source path")
    return value


def _exclusion(name):
    path = PurePosixPath(name)
    parts = tuple(part.lower() for part in path.parts)
    if any(part in _EXCLUDED_DIRS for part in parts[:-1]):
        return "private state, weights, build output or local cache directory"
    if len(parts) > 1 and parts[:2] == ("docs", "receipts"):
        return "historical GPU/run evidence is excluded from a source bundle"
    basename = parts[-1]
    if basename == ".env" or basename.startswith(".env.") or basename in {
            ".shard_psk", "hardware.json", "gpu-inventory.json", "signed-warmup.json", "ready.json", "loaded.json",
            "stop.json", "failure.json", "fleet_state.md", "swarm_deploy_state.md"}:
        return "secret, local deployment inventory or GPU/run evidence"
    if re.fullmatch(r"(?:accepted-)?resource-stage-[0-9]+\.json", basename):
        return "GPU resource evidence"
    if path.suffix.lower() in _EXCLUDED_SUFFIXES:
        return "private key, weights, binary, evidence or archive extension"
    return None


def _source_path(root, name):
    _safe_relative(name)
    path = root.joinpath(*PurePosixPath(name).parts)
    cursor = root
    for part in PurePosixPath(name).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("symlink sources are prohibited: " + name)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError("source resolves outside repository") from None
    return path


def _git(root, argv, *, executor=subprocess.run):
    result = executor(["git", *argv], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return result.stdout


def selected_paths(root=ROOT, *, executor=subprocess.run):
    root = Path(root).resolve()
    raw = _git(root, ["ls-files", "--cached", "-z"], executor=executor)
    tracked = {name.decode("utf-8") for name in raw.split(b"\x00") if name}
    kit = set()
    for folder, pattern in (("tools", "v4_acceptance*.py"), ("tests", "test_v4_acceptance*.py")):
        kit.update(path.relative_to(root).as_posix() for path in (root / folder).glob(pattern))
    for name in ("tools/v4_5090_acceptance.py", "tests/test_v4_5090_acceptance.py",
                 "examples/v4_5090_acceptance.json", "docs/V4_5090_ACCEPTANCE.md"):
        if (root / name).exists() or (root / name).is_symlink():
            kit.add(name)
    selected, excluded = [], []
    for name in sorted(tracked | kit):
        _safe_relative(name)
        reason = _exclusion(name)
        if reason:
            excluded.append({"path": name, "reason": reason})
            continue
        path = _source_path(root, name)
        if not path.exists():
            excluded.append({"path": name, "reason": "tracked file absent from current working tree"})
            continue
        if not path.is_file():
            raise ValueError("only regular source files may be bundled: " + name)
        if path.stat().st_size > MAX_FILE:
            if name in kit:
                raise ValueError("acceptance kit file exceeds 32 MiB: " + name)
            excluded.append({"path": name, "reason": "tracked file exceeds 32 MiB source limit"})
            continue
        if name == "BUNDLE_MANIFEST.json":
            raise ValueError("repository file conflicts with reserved bundle manifest")
        selected.append(name)
    if len({name.casefold() for name in selected}) != len(selected):
        raise ValueError("source paths collide on a case-insensitive filesystem")
    if not selected:
        raise ValueError("no eligible tracked or explicit acceptance kit source files")
    return selected, kit, excluded


def _file_record(root, name):
    path = _source_path(root, name)
    sha, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            size += len(chunk)
            if size > MAX_FILE:
                raise ValueError("source grew beyond 32 MiB during hashing")
            sha.update(chunk)
    return {"path": name, "size": size, "sha256": sha.hexdigest()}


def _core_source(root, included):
    # Mirrors phase0.v4_benchmark.source_identity's content inventory without
    # importing engines or relying on installed packages in the recipient host.
    paths = set()
    for directory in ("engines/deepseek_v4", "vendor/deepseek_v4_ref", "shard"):
        paths.update((root / directory).rglob("*.py"))
    for name in ("wire", "ngram_draft", "v4_benchmark", "v4_soak", "v4_release", "v4_profile_report",
                 "v4_ring_economics", "v4_round_calibration"):
        path = root / "phase0" / (name + ".py")
        if path.is_file():
            paths.add(path)
    files = []
    for path in sorted(paths):
        name = path.relative_to(root).as_posix()
        if name not in included:
            raise ValueError("benchmark source inventory contains an untracked/excluded file: " + name)
        files.append({"name": name, "sha256": included[name]["sha256"]})
    return files


def bundle(output, *, root=ROOT, executor=subprocess.run):
    root = Path(root).resolve()
    target = Path(output).absolute()
    if target.exists() or target.is_symlink():
        raise ValueError("bundle output must be a new file")
    names, kit, excluded = selected_paths(root, executor=executor)
    files = [_file_record(root, name) for name in names]
    if sum(row["size"] for row in files) > MAX_TOTAL:
        raise ValueError("source bundle payload exceeds 256 MiB")
    # A real embedded private-key document can never hitchhike as tracked prose.
    for name in names:
        with _source_path(root, name).open("rb") as stream:
            data = stream.read(MAX_FILE + 1)
        if re.search(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----", data):
            raise ValueError("private-key content detected in eligible source: " + name)
    head = _git(root, ["rev-parse", "HEAD"], executor=executor).decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", head):
        raise ValueError("valid base Git HEAD required")
    by_name = {row["path"]: row for row in files}
    core_files = _core_source(root, by_name)
    kit_files = [row for row in files if row["path"] in kit]
    manifest = {"schema": SCHEMA, "archive_prefix": PREFIX, "base_git_head": head,
                "core_source_sha256": _digest(core_files), "core_source_files": core_files,
                "kit_files": kit_files, "kit_files_sha256": _digest(kit_files),
                "files": files, "all_payload_sha256": _digest(files), "excluded": excluded,
                "limits": {"max_file_bytes": MAX_FILE, "max_total_bytes": MAX_TOTAL},
                "content_scope": "current working-file bytes; tracked files plus explicit acceptance kit; no GPU evidence",
                "authentication": "local integrity manifest; no publisher signature or hardware qualification"}
    manifest["manifest_sha256"] = _digest(manifest)
    encoded = _canonical(manifest) + b"\n"
    if len(encoded) > MAX_MANIFEST:
        raise ValueError("bundle manifest exceeds 4 MiB")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + secrets.token_hex(8) + ".tmp")
    try:
        with temporary.open("xb") as destination:
            with tarfile.open(fileobj=destination, mode="w:gz", format=tarfile.USTAR_FORMAT) as archive:
                for row in files:
                    path = _source_path(root, row["path"])
                    # Copy only bytes: no source uid/gid, permissions, executable
                    # secret bits, hard links, device nodes or host path metadata.
                    info = tarfile.TarInfo(PREFIX + row["path"])
                    info.size = row["size"]; info.mode = 0o644; info.mtime = 0
                    with path.open("rb") as source:
                        archive.addfile(info, source)
                import io
                info = tarfile.TarInfo(MANIFEST)
                info.size = len(encoded); info.mode = 0o644; info.mtime = 0
                archive.addfile(info, io.BytesIO(encoded))
            destination.flush(); os.fsync(destination.fileno())
        # Rehash the bytes that actually entered the archive before publication;
        # source edits between hashing and tar copy must fail closed.
        verify(temporary)
        if target.exists() or target.is_symlink():
            raise ValueError("output appeared during bundling; refusing overwrite")
        os.rename(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"bundled": True, "archive": str(target), "file_count": len(files),
            "payload_bytes": sum(row["size"] for row in files), "base_git_head": head,
            "core_source_sha256": manifest["core_source_sha256"], "kit_files_sha256": manifest["kit_files_sha256"],
            "all_payload_sha256": manifest["all_payload_sha256"], "manifest_sha256": manifest["manifest_sha256"]}


def _unique_object(pairs):
    body = {}
    for key, value in pairs:
        if key in body:
            raise ValueError("duplicate JSON manifest key")
        body[key] = value
    return body


def _manifest_errors(manifest):
    fields = {"schema", "archive_prefix", "base_git_head", "core_source_sha256", "core_source_files", "kit_files",
              "kit_files_sha256", "files", "all_payload_sha256", "excluded", "limits", "content_scope",
              "authentication", "manifest_sha256"}
    if not isinstance(manifest, dict) or set(manifest) != fields or manifest["schema"] != SCHEMA or manifest["archive_prefix"] != PREFIX:
        raise ValueError("unsupported source bundle manifest")
    if (not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", manifest["base_git_head"])
            or manifest["limits"] != {"max_file_bytes": MAX_FILE, "max_total_bytes": MAX_TOTAL}):
        raise ValueError("invalid source bundle base/limits")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest["manifest_sha256"] != _digest(body):
        raise ValueError("manifest integrity digest differs")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("nonempty payload inventory required")
    names, size = set(), 0
    for row in files:
        if (not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}
                or type(row["size"]) is not int or not 0 <= row["size"] <= MAX_FILE
                or not isinstance(row["sha256"], str) or not _HEX.fullmatch(row["sha256"])):
            raise ValueError("invalid payload record")
        name = _safe_relative(row["path"])
        if name.casefold() in names or _exclusion(name) or name == "BUNDLE_MANIFEST.json":
            raise ValueError("duplicate, private or excluded payload path")
        names.add(name.casefold()); size += row["size"]
    if size > MAX_TOTAL or files != sorted(files, key=lambda row: row["path"]):
        raise ValueError("payload inventory ordering/size invalid")
    if manifest["all_payload_sha256"] != _digest(files) or manifest["kit_files_sha256"] != _digest(manifest["kit_files"]):
        raise ValueError("payload/kit aggregate digest differs")
    by_name = {row["path"]: row for row in files}
    kit_expected = [row for row in files if (
        re.fullmatch(r"tools/v4_acceptance[^/]*\.py|tests/test_v4_acceptance[^/]*\.py", row["path"])
        or row["path"] in {"tools/v4_5090_acceptance.py", "tests/test_v4_5090_acceptance.py",
                           "examples/v4_5090_acceptance.json", "docs/V4_5090_ACCEPTANCE.md"})]
    if manifest["kit_files"] != kit_expected:
        raise ValueError("kit inventory differs from archive payload")
    core = manifest["core_source_files"]
    if not isinstance(core, list) or manifest["core_source_sha256"] != _digest(core):
        raise ValueError("benchmark source aggregate digest differs")
    core_names = set()
    for row in core:
        if (not isinstance(row, dict) or set(row) != {"name", "sha256"}
                or row["name"] in core_names or row["name"] not in by_name
                or row["sha256"] != by_name[row["name"]]["sha256"]):
            raise ValueError("benchmark source inventory differs from payload")
        core_names.add(row["name"])
    core_expected = {name for name in by_name if (
        name.endswith(".py") and name.startswith(("engines/deepseek_v4/", "vendor/deepseek_v4_ref/", "shard/"))
        or name in {"phase0/" + filename + ".py" for filename in ("wire", "ngram_draft", "v4_benchmark", "v4_soak",
            "v4_release", "v4_profile_report", "v4_ring_economics", "v4_round_calibration")})}
    if core_names != core_expected:
        raise ValueError("benchmark source inventory does not cover every bundled core module")
    return by_name


def verify(archive_path):
    """Stream-check safe regular members and every hash; never extract or execute."""
    path = Path(archive_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular local archive file required")
    records, manifest, total, seen = {}, None, 0, set()
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            name = member.name
            if (not name.startswith(PREFIX) or name == PREFIX or member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                    or member.pax_headers or member.offset_data - member.offset != tarfile.BLOCKSIZE):
                raise ValueError("archive must contain only safe regular FlyAI/ members")
            relative = _safe_relative(name[len(PREFIX):])
            if name.casefold() in seen:
                raise ValueError("duplicate archive member")
            seen.add(name.casefold())
            maximum = MAX_MANIFEST if name == MANIFEST else MAX_FILE
            if not 0 <= member.size <= maximum:
                raise ValueError("archive member exceeds allowed size")
            if name != MANIFEST and _exclusion(relative):
                raise ValueError("excluded/private/evidence member in archive")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("unreadable archive member")
            sha, count, chunks = hashlib.sha256(), 0, []
            with stream:
                while chunk := stream.read(1024**2):
                    count += len(chunk); sha.update(chunk)
                    if count > maximum:
                        raise ValueError("archive member exceeds streaming bound")
                    if name == MANIFEST:
                        chunks.append(chunk)
            if count != member.size:
                raise ValueError("truncated archive member")
            if name == MANIFEST:
                manifest = json.loads(b"".join(chunks), object_pairs_hook=_unique_object,
                                      parse_constant=lambda token: (_ for _ in ()).throw(ValueError("nonfinite manifest JSON")))
            else:
                total += count
                if total > MAX_TOTAL:
                    raise ValueError("archive payload exceeds 256 MiB")
                records[relative] = {"path": relative, "size": count, "sha256": sha.hexdigest()}
        # tar readers stop at the first zero header. Do not allow an unverified
        # second archive/member to hide after that terminator. Normal USTAR
        # record padding is zero and smaller than one 10 KiB record.
        trailing = archive.fileobj.read(tarfile.RECORDSIZE + 1)
        if len(trailing) > tarfile.RECORDSIZE or any(trailing):
            raise ValueError("extraneous data after archive terminator")
    if manifest is None:
        raise ValueError("bundle manifest is missing")
    expected = _manifest_errors(manifest)
    if set(expected) != set(records):
        raise ValueError("missing or extraneous archive payload")
    if any(expected[name] != row for name, row in records.items()):
        raise ValueError("archive payload hash/size differs from manifest")
    return {"verified": True, "schema": SCHEMA, "archive": str(path.absolute()), "file_count": len(records),
            "payload_bytes": total, "base_git_head": manifest["base_git_head"],
            "core_source_sha256": manifest["core_source_sha256"], "kit_files_sha256": manifest["kit_files_sha256"],
            "all_payload_sha256": manifest["all_payload_sha256"], "manifest_sha256": manifest["manifest_sha256"],
            "scope": "all safe members and bytes verified without extraction; local manifest is not a publisher signature"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("bundle")
    build.add_argument("--out", required=True)
    build.add_argument("--root", default=str(ROOT))
    check = commands.add_parser("verify")
    check.add_argument("--archive", required=True)
    args = parser.parse_args(argv)
    result = bundle(args.out, root=args.root) if args.command == "bundle" else verify(args.archive)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

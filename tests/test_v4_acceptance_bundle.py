"""Real local archives with mocked read-only Git inventory; no network or GPUs."""
import gzip
import io
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from tools import v4_acceptance_bundle as bundle


@pytest.fixture
def sources(tmp_path):
    root = tmp_path / "source"; root.mkdir()
    content = {"README.md": b"current unpublished source\n", "shard/receipt.py": b"SOURCE_VERSION = 2\n",
               "engines/deepseek_v4/v4_stage.py": b"STAGE_VERSION = 3\n",
               "phase0/v4_benchmark.py": b"BENCH_VERSION = 4\n", "pyproject.toml": b"[project]\nname='synthetic'\n",
               "tests/test_existing.py": b"def test_fixture(): pass\n",
               "tools/v4_acceptance_ring.py": b"UNTRACKED_KIT = True\n",
               "tests/test_v4_acceptance_ring.py": b"CPU_ONLY = True\n",
               "tools/v4_5090_acceptance.py": b"CENTRAL_ENTRYPOINT = True\n",
               "tests/test_v4_5090_acceptance.py": b"CENTRAL_ENTRYPOINT_TEST = True\n",
               "examples/v4_5090_acceptance.json": b'{"synthetic":true}\n', "docs/V4_5090_ACCEPTANCE.md": b"Acceptance instructions\n",
               "private/node.key": b"DO-NOT-BUNDLE-KEY", ".env": b"DO-NOT-BUNDLE-TOKEN", "weights/model.bin": b"NO-WEIGHTS",
               "docs/receipts/gpu-evidence.json": b"NO-GPU-EVIDENCE", "tools/unrelated_untracked.py": b"NO-UNTRACKED-SURPRISE"}
    for name, data in content.items():
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    tracked = {name for name in content if not name.startswith(("tools/", "examples/"))
               and name not in {"tests/test_v4_acceptance_ring.py", "tests/test_v4_5090_acceptance.py",
                                "docs/V4_5090_ACCEPTANCE.md"}}
    calls = []
    def git(argv, **kwargs):
        calls.append(argv)
        assert Path(kwargs["cwd"]) == root
        if argv == ["git", "ls-files", "--cached", "-z"]:
            return SimpleNamespace(stdout=b"\x00".join(name.encode() for name in sorted(tracked)) + b"\x00")
        if argv == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=b"a" * 40 + b"\n")
        raise AssertionError("only read-only Git inventory and HEAD are permitted")
    return root, tracked, git, calls


def archive_members(path):
    with tarfile.open(path, "r:gz") as archive:
        return [(member, archive.extractfile(member).read() if member.isfile() else None) for member in archive]


def write_archive(path, members):
    with tarfile.open(path, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
        for original, data in members:
            member = tarfile.TarInfo(original.name)
            member.type = original.type; member.linkname = original.linkname
            member.size = len(data) if data is not None else 0
            archive.addfile(member, io.BytesIO(data) if data is not None else None)


def test_bundle_current_tracked_and_explicit_untracked_kit_without_secrets(sources, tmp_path):
    root, _, git, calls = sources
    output = tmp_path / "portable.tgz"
    built = bundle.bundle(output, root=root, executor=git)
    checked = bundle.verify(output)
    assert built["all_payload_sha256"] == checked["all_payload_sha256"]
    assert built["core_source_sha256"] == checked["core_source_sha256"]
    assert calls == [["git", "ls-files", "--cached", "-z"], ["git", "rev-parse", "HEAD"]]
    members = archive_members(output)
    names = {member.name for member, _ in members}
    assert all(name.startswith("FlyAI/") for name in names)
    assert "FlyAI/tools/v4_acceptance_ring.py" in names and "FlyAI/docs/V4_5090_ACCEPTANCE.md" in names
    assert "FlyAI/examples/v4_5090_acceptance.json" in names
    assert "FlyAI/tools/unrelated_untracked.py" not in names
    assert not any(any(part in name for part in ("private/", "weights/", "receipts/", ".env")) for name in names)
    assert dict((member.name, data) for member, data in members)["FlyAI/shard/receipt.py"] == b"SOURCE_VERSION = 2\n"
    manifest = json.loads(dict((member.name, data) for member, data in members)[bundle.MANIFEST])
    assert manifest["base_git_head"] == "a" * 40
    assert len(manifest["core_source_files"]) == 3
    assert len(manifest["kit_files"]) == 6
    with pytest.raises(ValueError, match="new file"):
        bundle.bundle(output, root=root, executor=git)


def test_untracked_central_entrypoint_and_its_tests_are_explicitly_bundled(sources, tmp_path):
    root, tracked, git, _ = sources
    central = {"tools/v4_5090_acceptance.py", "tests/test_v4_5090_acceptance.py"}
    assert central.isdisjoint(tracked)
    (root / "tools/v4_5090_unrelated.py").write_text("UNRELATED = True\n")
    output = tmp_path / "central-entrypoint.tgz"
    bundle.bundle(output, root=root, executor=git)
    assert bundle.verify(output)["verified"] is True
    members = archive_members(output)
    names = {member.name for member, _ in members}
    assert {"FlyAI/" + name for name in central} <= names
    assert "FlyAI/tools/v4_5090_unrelated.py" not in names
    manifest = json.loads(dict((member.name, data) for member, data in members)[bundle.MANIFEST])
    assert central <= {row["path"] for row in manifest["kit_files"]}


@pytest.mark.parametrize("change", ["hash", "missing", "extra", "duplicate", "traversal", "absolute", "symlink", "hardlink", "case_collision", "manifest_hash"])
def test_verify_rejects_tampered_or_unsafe_archives(sources, tmp_path, change):
    root, _, git, _ = sources
    original = tmp_path / "original.tgz"; bundle.bundle(original, root=root, executor=git)
    members = archive_members(original)
    if change == "hash": members[0] = (members[0][0], b"changed source bytes\n")
    if change == "missing": members.pop(0)
    if change == "extra": members.append((tarfile.TarInfo("FlyAI/extraneous.py"), b"extra"))
    if change == "duplicate": members.append(members[0])
    if change == "traversal": members.append((tarfile.TarInfo("FlyAI/../escape.py"), b"escape"))
    if change == "absolute": members.append((tarfile.TarInfo("/escape.py"), b"escape"))
    if change in ("symlink", "hardlink"):
        member = tarfile.TarInfo("FlyAI/linked.py")
        member.type = tarfile.SYMTYPE if change == "symlink" else tarfile.LNKTYPE
        member.linkname = "../../escape.py"; members.append((member, None))
    if change == "case_collision":
        member = tarfile.TarInfo(members[0][0].name.upper()); members.append((member, b"duplicate"))
    if change == "manifest_hash":
        i = next(i for i, (member, _) in enumerate(members) if member.name == bundle.MANIFEST)
        manifest = json.loads(members[i][1]); manifest["base_git_head"] = "b" * 40
        members[i] = (members[i][0], json.dumps(manifest).encode())
    damaged = tmp_path / (change + ".tgz"); write_archive(damaged, members)
    with pytest.raises(ValueError): bundle.verify(damaged)


def test_verify_rejects_private_member_even_with_manifest_resealed(sources, tmp_path):
    root, _, git, _ = sources
    path = tmp_path / "source.tgz"; bundle.bundle(path, root=root, executor=git)
    members = archive_members(path)
    members.append((tarfile.TarInfo("FlyAI/private/node.key"), b"private"))
    damaged = tmp_path / "private.tgz"; write_archive(damaged, members)
    with pytest.raises(ValueError, match="private"):
        bundle.verify(damaged)


def test_verify_rejects_hidden_trailing_second_archive(sources, tmp_path):
    root, _, git, _ = sources
    path = tmp_path / "source.tgz"; bundle.bundle(path, root=root, executor=git)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo("FlyAI/hidden.py"); member.size = 6
        archive.addfile(member, io.BytesIO(b"hidden"))
    damaged = tmp_path / "trailing.tgz"
    damaged.write_bytes(gzip.compress(gzip.decompress(path.read_bytes()) + stream.getvalue()))
    with pytest.raises(ValueError, match="extraneous"):
        bundle.verify(damaged)


def test_untracked_core_source_refuses_incomplete_benchmark_identity(sources, tmp_path):
    root, _, git, _ = sources
    (root / "shard/untracked_core.py").write_text("NEW_CORE = True\n")
    with pytest.raises(ValueError, match="untracked/excluded"):
        bundle.bundle(tmp_path / "incomplete.tgz", root=root, executor=git)


def test_source_limits_and_private_content_fail_closed(sources, tmp_path, monkeypatch):
    root, _, git, _ = sources
    monkeypatch.setattr(bundle, "MAX_TOTAL", 4)
    with pytest.raises(ValueError, match="256 MiB"):
        bundle.bundle(tmp_path / "over-total.tgz", root=root, executor=git)
    monkeypatch.setattr(bundle, "MAX_TOTAL", 256 * 1024**2)
    (root / "README.md").write_bytes(b"-----BEGIN " + b"PRIVATE KEY-----\nsecret\n")
    with pytest.raises(ValueError, match="private-key content"):
        bundle.bundle(tmp_path / "private-content.tgz", root=root, executor=git)


def test_source_symlinks_refused_where_supported(sources, tmp_path):
    root, _, git, _ = sources
    target = root / "tools/v4_acceptance_link.py"
    try:
        target.symlink_to(root / "README.md")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable on this host")
    with pytest.raises(ValueError, match="symlink"):
        bundle.bundle(tmp_path / "symlink-source.tgz", root=root, executor=git)


def test_verifier_is_stdlib_standalone_without_repo_imports(sources, tmp_path):
    root, _, git, _ = sources
    path = tmp_path / "source.tgz"; bundle.bundle(path, root=root, executor=git)
    standalone = tmp_path / "verify_source_bundle.py"
    standalone.write_bytes(Path(bundle.__file__).read_bytes())
    result = subprocess.run([__import__("sys").executable, "-I", str(standalone), "verify", "--archive", str(path)],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["verified"] is True

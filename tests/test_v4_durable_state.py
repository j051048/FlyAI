import json
import os
from pathlib import Path
import pytest
import torch

from v4_durable_state import DurableStateStore, DurableStateError


IDENTITY = {"cohort": "a" * 64, "source": "b" * 64, "tenant": "tenant-a", "fence": 1}


def store(tmp_path, **options):
    return DurableStateStore(tmp_path / "private-state", max_bytes=1 << 20, **options)


def tensors():
    return {"kv": torch.arange(16).reshape(1, 4, 4), "compressor": torch.ones(2, dtype=torch.bfloat16)}


def test_reopen_preserves_exact_bytes_and_identity(tmp_path):
    first = store(tmp_path)
    first.put("entry", IDENTITY, tensors(), {"frontier": 4, "horizon": 32})
    recovered, metadata = store(tmp_path).get("entry", IDENTITY)
    assert metadata == {"frontier": 4, "horizon": 32}
    for name, value in tensors().items():
        assert torch.equal(value, recovered[name])
    assert first.get("entry", {**IDENTITY, "tenant": "tenant-b"}) is None
    assert first.get("entry", {**IDENTITY, "fence": 2}) is None


def test_payload_corruption_and_partial_generation_fail_closed(tmp_path):
    s = store(tmp_path)
    s.put("entry", IDENTITY, tensors(), {})
    payload = next(s.directory.glob("*.safetensors"))
    data = bytearray(payload.read_bytes()); data[-1] ^= 1; payload.write_bytes(data)
    with pytest.raises(DurableStateError, match="rejected"):
        s.get("entry", IDENTITY)
    assert s.get("entry", IDENTITY) is None
    orphan = s.directory / ("a" * 64 + "-" + "b" * 32 + ".safetensors")
    orphan.write_bytes(b"interrupted-before-manifest"); orphan.chmod(0o600)
    temporary = s.directory / ("a" * 64 + ".tmp")
    temporary.write_bytes(b"interrupted")
    assert s.status()["entries"] == 0 and not orphan.exists() and not temporary.exists()


def test_expiry_and_entry_capacity_are_actual_disk_bounds(tmp_path):
    now = [100.]
    s = store(tmp_path, max_entries=1, ttl_s=10, clock=lambda: now[0])
    s.put("first", IDENTITY, tensors(), {})
    s.put("second", IDENTITY, tensors(), {})
    assert s.get("first", IDENTITY) is None
    assert s.status()["entries"] == 1 and s.status()["disk_bytes"] <= 1 << 20
    now[0] = 111
    assert s.get("second", IDENTITY) is None
    assert list(s.directory.glob("*.safetensors")) == []


def test_atomic_replacement_rejects_peak_quota_preserving_old_entry(tmp_path):
    s = DurableStateStore(tmp_path / "private-state", max_bytes=16000)
    old = {"kv": torch.ones(2000)}
    s.put("entry", IDENTITY, old, {"generation": 1})
    with pytest.raises(DurableStateError, match="peak"):
        s.put("entry", IDENTITY, old, {"generation": 2})
    assert s.get("entry", IDENTITY)[1] == {"generation": 1}


def test_torn_manifest_does_not_retain_orphan_payload(tmp_path):
    s = store(tmp_path)
    s.put("entry", IDENTITY, tensors(), {})
    manifest = next(s.directory.glob("*.json")); manifest.write_bytes(b"{")
    assert s.get("entry", IDENTITY) is None
    assert list(s.directory.glob("*.safetensors")) == []


def test_actual_free_disk_is_checked_before_writing_or_evicting(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import v4_durable_state as module
    s = store(tmp_path)
    s.put("entry", IDENTITY, tensors(), {})
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(DurableStateError, match="actual free disk"):
        s.put("new-entry", IDENTITY, tensors(), {})
    assert s.get("entry", IDENTITY) is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX private mode check")
def test_public_file_and_symlink_are_rejected(tmp_path):
    s = store(tmp_path)
    s.put("entry", IDENTITY, tensors(), {})
    manifest = next(s.directory.glob("*.json")); manifest.chmod(0o644)
    with pytest.raises(DurableStateError, match="private"):
        s.get("entry", IDENTITY)
    outside = tmp_path / "outside"; outside.mkdir()
    linked = tmp_path / "link"; linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(Exception, match="symlink"):
        DurableStateStore(linked, max_bytes=8192)


@pytest.mark.skipif(os.name != "nt", reason="Windows existing-file DACL check")
def test_existing_payload_with_public_windows_acl_is_rejected(tmp_path):
    import subprocess
    s = store(tmp_path)
    s.put("entry", IDENTITY, tensors(), {})
    payload = next(s.directory.glob("*.safetensors"))
    result = subprocess.run(["icacls", str(payload), "/grant", "*S-1-1-0:(R)"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    with pytest.raises(DurableStateError, match="DACL"):
        s.get("entry", IDENTITY)

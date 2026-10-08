"""Real temporary HTTP/files; no Hub calls, subprocess downloader, or model download."""
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import pytest

from get_model import (DownloadError, DownloadSpec, download_file, fetch_model, main,
                       resolve_files, select_token, verify_file, _directory_lock)
from shard.download_inventory import verify_inventory, InventoryError, FILENAME

COMMIT = "a" * 40
FILES = {"config.json": b'{"model_type":"gpt_oss"}', "model.safetensors": b"packed-fixture-weights"}


def git_sha(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


class API:
    def __init__(self, files=None):
        self.files, self.calls = files or FILES, []

    def model_info(self, **kwargs):
        self.calls.append(kwargs)
        entries = []
        for name, data in self.files.items():
            entries.append(SimpleNamespace(rfilename=name, size=len(data), blob_id=git_sha(data),
                lfs=SimpleNamespace(size=len(data), sha256=hashlib.sha256(data).hexdigest()) if name.endswith(".safetensors") else None))
        return SimpleNamespace(sha=COMMIT, siblings=entries)


@contextmanager
def server(files=None, *, mode="normal", redirect=None):
    data, requests = files or FILES, []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            row = {"path": self.path, "range": self.headers.get("Range"), "authorization": self.headers.get("Authorization")}
            requests.append(row)
            if redirect:
                self.send_response(302); self.send_header("Location", redirect); self.end_headers(); return
            payload = data[unquote(urlsplit(self.path).path).rsplit("/", 1)[-1]]
            if mode == "corrupt_first" and len(requests) == 1:
                payload = b"x" * len(payload)
            offset = int(row["range"].split("=", 1)[1].split("-", 1)[0]) if row["range"] else 0
            if row["range"] and mode != "ignore_range":
                start = 1 if mode == "wrong_range" else offset
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
                payload = payload[start:]
            else:
                self.send_response(200)
            if mode == "oversize":
                payload += b"overflow"
            self.send_header("Content-Length", str(len(payload))); self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass
    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", requests
    finally:
        http.shutdown(); http.server_close(); worker.join(2)


def spec(data=FILES["model.safetensors"], path="model.safetensors"):
    return DownloadSpec(path, len(data), "sha256", hashlib.sha256(data).hexdigest())


def test_fixed_commit_anonymous_false_and_verified_inventory(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "invalid\u2026placeholder")
    api, logs = API(), []
    with server() as (base, seen):
        result = fetch_model("org/model", tmp_path, anonymous=True, api=api, base_url=base, log=logs.append,
                             retries=2, sleep=lambda _: None)
    assert api.calls == [{"repo_id": "org/model", "revision": "main", "token": False, "files_metadata": True}]
    assert all("/resolve/" + COMMIT + "/" in row["path"] and row["authorization"] is None for row in seen)
    checked = verify_inventory(tmp_path, expected_checkpoint_id=result["checkpoint_id"], expected_repo="org/model",
                               expected_revision=COMMIT)
    assert checked["payload_integrity_verified"] is True
    assert checked["config_sha256"] == hashlib.sha256(FILES["config.json"]).hexdigest()
    assert checked["checkpoint_id"] == "sha256:" + checked["manifest_sha256"]
    assert "auth_mode" not in json.loads((tmp_path / FILENAME).read_text())
    assert not list(tmp_path.glob("*.part"))


def test_existing_files_are_content_verified_and_corruption_repaired(tmp_path):
    for name, data in FILES.items():
        (tmp_path / name).write_bytes(data)
    (tmp_path / "model.safetensors").write_bytes(b"bad")
    with server() as (base, seen):
        fetch_model("org/model", tmp_path, anonymous=True, api=API(), base_url=base, log=lambda _: None,
                    retries=2, sleep=lambda _: None)
        assert [row["path"].rsplit("/", 1)[-1] for row in seen] == ["model.safetensors"]
        fetch_model("org/model", tmp_path, anonymous=True, api=API(), base_url=base, log=lambda _: None,
                    retries=2, sleep=lambda _: None)
        assert len(seen) == 1, "verified cache must not redownload"
    assert verify_inventory(tmp_path)["payload_integrity_verified"]


@pytest.mark.parametrize("mode", ["normal", "ignore_range"])
def test_part_resume_or_whole_response_is_placed_without_duplicate_bytes(tmp_path, mode):
    payload = FILES["model.safetensors"]
    (tmp_path / "model.safetensors.part").write_bytes(payload[:4])
    with server(mode=mode) as (base, seen):
        download_file(tmp_path, spec(), base + "/model.safetensors", retries=2, sleep=lambda _: None)
    assert seen[0]["range"] == "bytes=4-"
    assert (tmp_path / "model.safetensors").read_bytes() == payload


def test_bad_full_partial_restarts_and_same_size_http_corruption_retries(tmp_path):
    payload = FILES["model.safetensors"]
    (tmp_path / "model.safetensors.part").write_bytes(b"x" * len(payload))
    with server() as (base, seen):
        download_file(tmp_path, spec(), base + "/model.safetensors", retries=3, sleep=lambda _: None)
    assert len(seen) == 1 and seen[0]["range"] is None
    (tmp_path / "model.safetensors").unlink()
    with server(mode="corrupt_first") as (base, seen):
        download_file(tmp_path, spec(), base + "/model.safetensors", retries=2, sleep=lambda _: None)
    assert len(seen) == 2 and (tmp_path / "model.safetensors").read_bytes() == payload


@pytest.mark.parametrize("mode", ["oversize", "wrong_range"])
def test_invalid_http_body_never_becomes_final_file(tmp_path, mode):
    if mode == "wrong_range":
        (tmp_path / "model.safetensors.part").write_bytes(FILES["model.safetensors"][:4])
    with server(mode=mode) as (base, _):
        with pytest.raises(DownloadError, match="verified download failed"):
            download_file(tmp_path, spec(), base + "/model.safetensors", retries=1, sleep=lambda _: None)
    assert not (tmp_path / "model.safetensors").exists()


def test_auth_never_downgrades_and_cross_origin_redirect_drops_bearer(tmp_path, monkeypatch):
    fixture = "hf_fixture_only_not_a_real_credential"
    monkeypatch.setenv("HF_TOKEN", fixture)
    api, logs = API(), []
    with server() as (base, seen):
        result = fetch_model("org/model", tmp_path, api=api, base_url=base, retries=1, log=logs.append)
    assert api.calls[0]["token"] == fixture and result["auth_mode"] == "authenticated"
    assert all(row["authorization"] == "Bearer " + fixture for row in seen)
    assert fixture not in "\n".join(logs) and fixture not in (tmp_path / FILENAME).read_text()
    redirected_dir = tmp_path / "redirected"; redirected_dir.mkdir()
    with server() as (other, other_seen):
        with server(redirect=other + "/model.safetensors") as (first, first_seen):
            download_file(redirected_dir, spec(), first + "/model.safetensors", token=fixture, retries=1)
    assert first_seen[0]["authorization"] == "Bearer " + fixture
    assert other_seen[0]["authorization"] is None


@pytest.mark.parametrize("value", ["hf_bad\u2026secret", "hf_key\r\nAuthorization: injected", "", " key", "<placeholder>"])
def test_invalid_credentials_fail_redacted(value):
    with pytest.raises(DownloadError) as caught:
        select_token(token=value)
    assert value not in str(caught.value) or value == ""
    with pytest.raises(DownloadError):
        select_token(token_env="MISSING", environ={})
    assert select_token(environ={}) is False


def test_token_file_private_and_explicit_anonymous_conflicts(tmp_path):
    path = tmp_path / "credential"
    path.write_text("hf_fixture_not_live\n", encoding="ascii"); path.chmod(0o600)
    assert select_token(token_file=path) == "hf_fixture_not_live"
    with pytest.raises(DownloadError):
        select_token(anonymous=True, token_file=path)
    if os.name != "nt":
        path.chmod(0o644)
        with pytest.raises(DownloadError, match="0600"):
            select_token(token_file=path)


def test_metadata_failure_is_redacted_and_never_fetches_anonymously(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_fixture_secret")
    class Denied:
        def model_info(self, **kwargs):
            assert kwargs["token"] == "hf_fixture_secret"
            raise RuntimeError("HTTP error with hf_fixture_secret in third-party details")
    with pytest.raises(DownloadError) as caught:
        fetch_model("org/model", tmp_path, api=Denied())
    assert "hf_fixture_secret" not in str(caught.value) and caught.value.__cause__ is None


@pytest.mark.parametrize("path", ["../escape.json", "/absolute.json", "dir/../escape.json", "C:escape.json", "dir\\escape.json"])
def test_metadata_paths_do_not_escape_root(path):
    with pytest.raises(DownloadError):
        DownloadSpec(path, 1, "sha256", "00" * 32)


def test_symlink_destinations_are_rejected(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.write_bytes(b"untouched")
    try:
        (tmp_path / "model.safetensors").symlink_to(outside)
    except OSError:
        pytest.skip("symlink privilege unavailable")
    with pytest.raises(DownloadError):
        download_file(tmp_path, spec(), "http://127.0.0.1:1", retries=1)
    assert outside.read_bytes() == b"untouched"


def test_inventory_missing_tampered_payload_or_identity_is_not_verified(tmp_path):
    with pytest.raises(InventoryError):
        verify_inventory(tmp_path)
    with server() as (base, _):
        created = fetch_model("org/model", tmp_path, anonymous=True, api=API(), base_url=base, log=lambda _: None)
    with pytest.raises(InventoryError, match="assignment"):
        verify_inventory(tmp_path, expected_checkpoint_id="claimed_identity")
    (tmp_path / "model.safetensors").write_bytes(b"x" * len(FILES["model.safetensors"]))
    assert verify_inventory(tmp_path, verify_files=False)["payload_integrity_verified"] is False
    with pytest.raises(InventoryError, match="payload"):
        verify_inventory(tmp_path)
    marker = json.loads((tmp_path / FILENAME).read_text())
    marker["revision"] = "b" * 40
    (tmp_path / FILENAME).write_text(json.dumps(marker))
    with pytest.raises(InventoryError, match="identity"):
        verify_inventory(tmp_path, verify_files=False)


def test_cli_remains_positional_and_import_has_no_work(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr("get_model.fetch_model", lambda **kwargs: called.append(kwargs))
    assert main(["org/model", str(tmp_path), "--revision", COMMIT, "--anonymous"]) == 0
    assert called[0]["repo"] == "org/model" and called[0]["anonymous"] is True
    assert called[0]["revision"] == COMMIT


def test_verify_existing_bootstraps_identity_with_zero_payload_http(tmp_path):
    for name, data in FILES.items():
        (tmp_path / name).write_bytes(data)
    result = fetch_model("org/model", tmp_path, anonymous=True, api=API(), verify_existing=True,
                         base_url="http://127.0.0.1:1", log=lambda _: None)
    assert verify_inventory(tmp_path)["checkpoint_id"] == result["checkpoint_id"]
    (tmp_path / "model.safetensors").write_bytes(b"partial")
    with pytest.raises(DownloadError):
        fetch_model("org/model", tmp_path, anonymous=True, api=API(), verify_existing=True,
                    base_url="http://127.0.0.1:1", log=lambda _: None)


def test_unlisted_weight_override_and_index_reference_cannot_claim_verified_identity(tmp_path):
    from shard.download_inventory import build_inventory
    index = json.dumps({"weight_map": {"layers.0.weight": "model-00001.safetensors"}}).encode()
    files = {"config.json": FILES["config.json"], "model.safetensors.index.json": index,
             "model-00001.safetensors": FILES["model.safetensors"]}
    rows = []
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        rows.append({"path": name, "size": len(data), "algorithm": "sha256", "digest": sha, "payload_sha256": sha})
    inventory = build_inventory("org/model", COMMIT, rows)
    (tmp_path / FILENAME).write_text(json.dumps(inventory))
    assert verify_inventory(tmp_path)["payload_integrity_verified"]
    # HF gives this single-file path precedence over the verified sharded index.
    (tmp_path / "model.safetensors").write_bytes(b"unlisted-old-checkpoint")
    with pytest.raises(InventoryError, match="unlisted"):
        verify_inventory(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    changed = json.dumps({"weight_map": {"layers.0.weight": "unknown.safetensors"}}).encode()
    (tmp_path / "model.safetensors.index.json").write_bytes(changed)
    for row in rows:
        if row["path"] == "model.safetensors.index.json":
            row.update(size=len(changed), digest=hashlib.sha256(changed).hexdigest(), payload_sha256=hashlib.sha256(changed).hexdigest())
    (tmp_path / FILENAME).write_text(json.dumps(build_inventory("org/model", COMMIT, rows)))
    with pytest.raises(InventoryError, match="unverified weight"):
        verify_inventory(tmp_path)


def test_directory_lock_prevents_two_different_revisions_mixing(tmp_path):
    with _directory_lock(tmp_path):
        with pytest.raises(DownloadError, match="another download"):
            with _directory_lock(tmp_path):
                pass


def test_pinned_revision_and_missing_upstream_digest_fail_closed():
    with pytest.raises(DownloadError, match="pinned revision"):
        resolve_files("org/model", revision="b" * 40, api=API())
    class Missing(API):
        def model_info(self, **kwargs):
            value = super().model_info(**kwargs)
            value.siblings[1].lfs.sha256 = None
            return value
    with pytest.raises(DownloadError, match="digest"):
        resolve_files("org/model", api=Missing())

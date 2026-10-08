"""Fixed-revision, content-verified HF model download (no token in argv).

Legacy CLI: python get_model.py openai/gpt-oss-120b /root/models/gpt-oss-120b
Add --anonymous to forbid implicit/cached authentication, or --token-file/--token-env.
Importing this module does no filesystem, credential, network, or model work.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
from urllib.parse import quote, urlparse
from urllib.request import Request, HTTPRedirectHandler, build_opener

SKIP_DIR = ("original/", "metal/", "onnx/", "gguf/")
KEEP_EXT = (".safetensors", ".json", ".jinja", ".txt", ".model")
INVENTORY = ".shard-download.json"
LOCKFILE = ".flyai-download.lock"


class DownloadError(ValueError):
    pass


def _inventory_api():
    try:
        from shard import download_inventory
    except ImportError:
        try:
            import download_inventory  # flat deployment
        except ImportError:
            import sys
            root = str(Path(__file__).resolve().parent.parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            from shard import download_inventory
    return download_inventory


def verify_inventory(directory, expected_checkpoint_id=None, expected_repo=None, expected_revision=None,
                     verify_files=True):
    return _inventory_api().verify_inventory(directory, expected_checkpoint_id, expected_repo,
                                              expected_revision, verify_files)


def _credential(value):
    # RFC 6750 bearer syntax; do not assume that every future credential starts
    # with hf_. Format validation is not an authentication/provenance check.
    if (not isinstance(value, str) or not 1 <= len(value) <= 4096 or not value.isascii()
            or not re.fullmatch(r"[A-Za-z0-9._~+/\-]+=*", value)):
        raise DownloadError("invalid authentication credential; value redacted (no anonymous fallback)")
    return value


def select_token(*, anonymous=False, token=None, token_env=None, token_file=None, environ=None):
    """Return an explicit bearer string or False; never return SDK-implicit None."""
    environ = os.environ if environ is None else environ
    selected = sum(value is not None for value in (token, token_env, token_file))
    if selected > 1 or anonymous and selected:
        raise DownloadError("choose one authentication source or explicit anonymous mode")
    if anonymous:
        return False
    if token_file is not None:
        try:
            path = Path(token_file)
            info = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 4098:
                raise DownloadError("authentication file must be a small regular file")
            if os.name != "nt" and info.st_mode & 0o077:
                raise DownloadError("authentication file must have permissions 0600 or stricter")
            rows = path.read_bytes().decode("ascii").splitlines()
            if len(rows) != 1:
                raise DownloadError("authentication file must contain one credential line")
            token = rows[0]
        except (OSError, UnicodeError):
            raise DownloadError("cannot read authentication file; credential redacted") from None
    elif token is None:
        name = token_env or "HF_TOKEN"
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise DownloadError("invalid credential environment variable name")
        token = environ.get(name)
        if token is None and token_env is None:
            return False
    return _credential(token)


def _relative(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or any(ord(c) < 32 for c in value)):
        raise DownloadError("unsafe model file path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise DownloadError("unsafe model file path")
    if value in (INVENTORY, LOCKFILE):
        raise DownloadError("model file collides with download metadata")
    return value


def _target(root, relative):
    _relative(relative)
    target = root.joinpath(*PurePosixPath(relative).parts)
    try:
        target.resolve().relative_to(root)
    except ValueError:
        raise DownloadError("model file escapes download directory") from None
    for part in (target, *list(target.parents)[:len(PurePosixPath(relative).parts) - 1]):
        if part.is_symlink():
            raise DownloadError("model destination must not follow symlinks")
    return target


@dataclass(frozen=True)
class DownloadSpec:
    path: str
    size: int
    algorithm: str
    digest: str

    def __post_init__(self):
        _relative(self.path)
        if type(self.size) is not int or self.size < 0:
            raise DownloadError("download metadata lacks a nonnegative exact file size")
        length = {"sha256": 64, "git-sha1": 40}.get(self.algorithm) if isinstance(self.algorithm, str) else None
        if not length or not isinstance(self.digest, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", self.digest):
            raise DownloadError("download metadata lacks a verifiable file digest")


def _field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def resolve_files(repo, revision="main", *, token=False, api=None):
    """Pin one HF commit before building file URLs; retain LFS and Git blob hashes."""
    if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?", repo):
        raise DownloadError("invalid HF model repository ID")
    if not isinstance(revision, str) or not revision or len(revision) > 512:
        raise DownloadError("nonempty HF revision required")
    if token is not False:
        token = _credential(token)
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    try:
        info = api.model_info(repo_id=repo, revision=revision, token=token, files_metadata=True)
    except Exception:
        raise DownloadError("HF metadata request failed; verify access/revision (credential redacted)") from None
    commit = _field(info, "sha")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise DownloadError("HF metadata did not resolve an immutable commit")
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) and commit != revision:
        raise DownloadError("HF metadata commit differs from the explicitly pinned revision")
    files, seen = [], set()
    siblings = _field(info, "siblings")
    if not isinstance(siblings, (list, tuple)):
        raise DownloadError("HF file metadata is absent")
    for entry in siblings:
        name = _field(entry, "rfilename")
        _relative(name)
        if name.startswith(SKIP_DIR) or not name.endswith(KEEP_EXT):
            continue
        identity = name.casefold() if os.name == "nt" else name
        if identity in seen:
            raise DownloadError("duplicate or platform-colliding model file path")
        seen.add(identity)
        size, lfs = _field(entry, "size"), _field(entry, "lfs")
        if lfs is not None:
            digest = _field(lfs, "sha256")
            if _field(lfs, "size") != size:
                raise DownloadError("HF LFS size metadata disagrees")
            files.append(DownloadSpec(name, size, "sha256", digest))
        else:
            files.append(DownloadSpec(name, size, "git-sha1", _field(entry, "blob_id")))
    if not files:
        raise DownloadError("no verifiable model files selected")
    return commit, files


def verify_file(path, spec):
    """Size plus publisher content digest; also return SHA256 for local inventory."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != spec.size:
        raise DownloadError("model file size or type differs from fixed revision metadata")
    sha, git = hashlib.sha256(), hashlib.sha1()
    git.update(f"blob {spec.size}\0".encode("ascii"))
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            sha.update(block); git.update(block)
    if (sha.hexdigest() if spec.algorithm == "sha256" else git.hexdigest()) != spec.digest:
        raise DownloadError("model file content digest mismatch")
    return sha.hexdigest()


class _Redirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        old, new = urlparse(req.full_url), urlparse(newurl)
        if old.scheme == "https" and new.scheme != "https":
            raise DownloadError("model redirect cannot downgrade HTTPS")
        if (old.scheme, old.hostname, old.port) != (new.scheme, new.hostname, new.port):
            redirected.remove_header("Authorization")
        return redirected


def download_file(root, spec, url, *, token=False, retries=6, timeout=120.0, opener=None, sleep=time.sleep):
    """Resume only a correctly placed HTTP range, then verify before atomic replace."""
    if type(retries) is not int or not 1 <= retries <= 50 or type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise DownloadError("invalid download retry/timeout policy")
    root = Path(root).resolve()
    dest = _target(root, spec.path)
    part = _target(root, spec.path + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        try:
            return verify_file(dest, spec)
        except DownloadError:
            if not dest.is_file() or dest.is_symlink():
                raise
            dest.unlink()  # Never leave a known corrupt final file available to loaders.
    if token is not False:
        token = _credential(token)
    opener = opener or build_opener(_Redirect())
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have > spec.size:
                part.unlink(); have = 0
            if have != spec.size or not part.exists():
                headers = {"User-Agent": "FlyAI-verified-fetch/1", "Accept-Encoding": "identity"}
                if token is not False:
                    headers["Authorization"] = "Bearer " + token
                if have:
                    headers["Range"] = f"bytes={have}-"
                with opener.open(Request(url, headers=headers), timeout=timeout) as response:
                    status = response.getcode()
                    if status == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if not match or int(match[3]) != spec.size or int(match[1]) not in (0, have) or int(match[2]) < int(match[1]) or int(match[2]) >= spec.size:
                            part.unlink(missing_ok=True)
                            raise DownloadError("HTTP resume range does not match fixed file metadata")
                        offset = int(match[1])
                    elif status == 200:
                        offset = 0
                    else:
                        raise DownloadError("unexpected model download response")
                    remaining = spec.size - offset
                    with part.open("ab" if offset == have and have else "wb") as output:
                        written = 0
                        while True:
                            block = response.read(min(1 << 20, remaining - written + 1))
                            if not block:
                                break
                            if written + len(block) > remaining:
                                raise DownloadError("model response exceeds its declared size")
                            output.write(block); written += len(block)
                        output.flush(); os.fsync(output.fileno())
            if part.stat().st_size != spec.size:
                raise DownloadError("incomplete model file; verified resume required")
            try:
                sha = verify_file(part, spec)
            except DownloadError:
                part.unlink(missing_ok=True)
                raise
            os.replace(part, dest)
            return sha
        except Exception:
            # Third-party HTTP errors may contain credential/request details.
            # Never print or chain them. Auth failures do NOT switch modes.
            if attempt + 1 < retries:
                sleep(min(attempt + 1, 3))
    raise DownloadError(f"verified download failed for {spec.path}; credential/HTTP details redacted") from None


@contextmanager
def _directory_lock(root):
    lock = root / LOCKFILE
    if lock.is_symlink():
        raise DownloadError("download lock must not be a symlink")
    with lock.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"\0"); handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise DownloadError("another download owns this model directory") from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def fetch_model(repo, outdir, *, revision="main", anonymous=False, token=None, token_env=None, token_file=None,
                api=None, base_url="https://huggingface.co", retries=6, timeout=120, log=print, sleep=time.sleep,
                verify_existing=False):
    inventory_api = _inventory_api()
    selected = select_token(anonymous=anonymous, token=token, token_env=token_env, token_file=token_file)
    commit, files = resolve_files(repo, revision, token=selected, api=api)
    root = Path(outdir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    mode = "anonymous" if selected is False else "authenticated"
    rows = []
    with _directory_lock(root):
        log(f"FETCH {repo}@{commit} files={len(files)} auth={mode}")
        for spec in files:
            url = base_url.rstrip("/") + "/" + quote(repo, safe="/") + "/resolve/" + commit + "/" + quote(spec.path, safe="/")
            if verify_existing:
                sha = verify_file(_target(root, spec.path), spec)
            else:
                sha = download_file(root, spec, url, token=selected, retries=retries, timeout=timeout, sleep=sleep)
            rows.append({**asdict(spec), "payload_sha256": sha})
            log("verified " + spec.path)
        inventory = inventory_api.build_inventory(repo, commit, rows)
        path = root / INVENTORY
        if path.is_symlink() or (root / (INVENTORY + ".part")).is_symlink():
            raise DownloadError("download inventory must not follow symlinks")
        temporary = root / (INVENTORY + ".part")
        with temporary.open("w", encoding="utf-8") as target:
            json.dump(inventory, target, sort_keys=True, indent=2)
            target.flush(); os.fsync(target.fileno())
        os.replace(temporary, path)
        inventory_api.verify_inventory(root, expected_checkpoint_id=inventory["checkpoint_id"], verify_files=False)
    log("DONE " + str(root))
    # Files were verified above; the completion record binds those exact bytes.
    return {**inventory, "auth_mode": mode, "payload_integrity_verified": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo"); parser.add_argument("outdir")
    parser.add_argument("--revision", default="main")
    authentication = parser.add_mutually_exclusive_group()
    authentication.add_argument("--anonymous", action="store_true")
    authentication.add_argument("--token-env")
    authentication.add_argument("--token-file")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--verify-existing", action="store_true",
                        help="resolve immutable metadata and verify local files without downloading payloads")
    args = parser.parse_args(argv)
    try:
        fetch_model(**vars(args))
        return 0
    except (DownloadError, _inventory_api().InventoryError) as error:
        print("ERROR " + str(error), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

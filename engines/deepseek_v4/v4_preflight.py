"""Read-only V4 deployment checks; environment is applied before Torch imports.

File integrity and GPU visibility do not establish a reservation, a successful
inference session, or numerical acceptance. This module never creates keys,
loads model weights into a GPU, starts tunnels, or changes model directories.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))


def apply_environment(environment):
    if not isinstance(environment, dict) or any(not isinstance(k, str) or not isinstance(v, str)
        or "\x00" in k or "\x00" in v for k, v in environment.items()):
        raise ValueError("explicit string environment required")
    if "torch" in sys.modules:
        raise ValueError("preflight environment must be installed before importing Torch")
    for name in tuple(os.environ):
        if name.startswith("V4_"):
            os.environ.pop(name)
    os.environ.update(environment)


def _key(path, expected):
    from shard.manifest import load_key, pub_b64
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4096:
        raise ValueError("existing protected signing key required")
    actual = pub_b64(load_key(str(path)))
    if not expected or actual != expected:
        raise ValueError("local signing key differs from the pinned public plan")
    return {"pubkey": actual, "matches_plan": True}


def check_artifacts(directory, cohort, *, stage=None, dspark=False, verify_files=True):
    from shard import weight_artifacts as W
    from v4_artifact_contract import validate_native_cohort
    directory = Path(directory)
    catalog = W.validate_catalog(W.read_json(directory / W.GLOBAL_FILE))
    validate_native_cohort(cohort, catalog["config"])
    if ((catalog["checkpoint_id"], catalog["manifest_sha256"], catalog["config_sha256"], catalog["model_id"]) !=
            (cohort["checkpoint_id"], cohort["manifest_sha256"], cohort["config_sha256"], cohort["model_id"])):
        raise ValueError("native catalogue/config differs from the exact model cohort")
    if stage is not None:
        kwargs = dict(expected_checkpoint_id=cohort["checkpoint_id"], expected_manifest_sha256=cohort["manifest_sha256"],
                      verify_files=verify_files)
        if (directory / W.STAGE_FILE).is_file():
            value = W.verify_stage_artifacts(directory, lo=stage["lo"], hi=stage["hi"],
                head=stage["head"], tail=stage["tail"], dspark=dspark, **kwargs)
        else:
            value = W.verify_weight_pack(directory, **kwargs)
            W.select_stage_artifacts(catalog, value["pack"], stage["lo"], stage["hi"],
                head=stage["head"], tail=stage["tail"], dspark=dspark)
        return {"payload_integrity_verified": bool(verify_files), "scope": value["verification_scope"],
                "checkpoint_id": catalog["checkpoint_id"], "manifest_sha256": catalog["manifest_sha256"]}
    # A coordinator can hold only the pinned assets and full logical metadata.
    for name, record in catalog["assets"].items():
        if W.hash_file(W.safe_path(directory, name)) != (record["sha256"], record["size"]):
            raise ValueError("coordinator model asset hash mismatch")
    if not any(name.startswith(("tokenizer", "vocab", "merges")) for name in catalog["assets"]):
        raise ValueError("pinned tokenizer assets required")
    actual_assets = {p.name for pattern in ("tokenizer*", "vocab*", "merges*", "special_tokens_map.json", "generation_config.json")
                     for p in directory.glob(pattern) if p.is_file()}
    if not actual_assets <= set(catalog["assets"]):
        raise ValueError("unverified local tokenizer override")
    return {"payload_integrity_verified": False, "assets_verified": True,
            "scope": "coordinator catalogue and local assets only", "checkpoint_id": catalog["checkpoint_id"]}


def _versions():
    values = {"python": sys.version.split()[0]}
    for name in ("torch", "tilelang", "transformers", "safetensors", "cryptography", "psutil"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def _gpu(device, expected_uuid):
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA GPU is not available in this node environment")
    actual = torch.cuda.get_device_properties(torch.device(device))
    uuid = getattr(actual, "uuid", None)
    if isinstance(uuid, bytes) and len(uuid) == 16:
        import uuid as uuid_module
        uuid = "GPU-" + str(uuid_module.UUID(bytes=uuid))
    uuid = str(uuid) if uuid is not None else None
    if uuid is None or uuid.lower() != expected_uuid.lower():
        raise ValueError("selected CUDA device UUID is unknown or differs from the plan")
    free, total = torch.cuda.mem_get_info(torch.device(device))
    return {"gpu_uuid": uuid, "gpu_name": actual.name, "available_vram_bytes": free,
            "total_vram_bytes": total, "cuda": torch.version.cuda, "hardware_attested": False}


def _port(port, bind):
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("explicit valid listener port required")
    family = socket.AF_INET6 if ":" in bind else socket.AF_INET
    with socket.socket(family) as probe:
        probe.bind((bind, port))
    return {"port": port, "available": True, "scope": "local bind availability; not peer reachability"}


def process_observation(name, *, state_dir=".shard-processes", listener_port=None, log_tail_bytes=8192):
    """Bounded owned-PID/log inspection; never opens an inference connection."""
    from shard.managed_launch import ManagedLauncher
    manager = ManagedLauncher(state_dir)
    if type(log_tail_bytes) is not int or not 1 <= log_tail_bytes <= 1 << 20:
        raise ValueError("bounded log sample size required")
    state = manager.status(name)
    owned_pids = {state.get("pid")} - {None}
    if state.get("running"):
        try:
            import psutil
            owned_pids.update(child.pid for child in psutil.Process(state["pid"]).children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    log = manager.directory / (name + ".log")
    text = ""
    if log.is_file() and not log.is_symlink():
        with log.open("rb") as stream:
            size = log.stat().st_size
            head = stream.read(65536)
            if size > 65536:
                tail_bytes = max(65536, log_tail_bytes)
                stream.seek(max(65536, size - tail_bytes))
                head += b"\n<bounded-log-sample>\n" + stream.read(tail_bytes)
        text = head.decode("utf-8", errors="replace")
    events = []
    for line in text.splitlines():
        if line.startswith("SHARD_STAGE_EVENT "):
            try:
                row = json.loads(line[len("SHARD_STAGE_EVENT "):])
                if row.get("pid") in owned_pids:
                    events.append(row)
            except (ValueError, TypeError):
                pass
    declared = any(row.get("event") == "listening" for row in events)
    declared |= bool(re.search(r"\[s\d+\] listening(?: on)? \S+:\d+", text))
    state["listening"] = bool(state.get("running") and declared)
    state["events"] = events[-32:]
    state["log_tail"] = text[-log_tail_bytes:]
    start = text.find("Traceback (most recent call last):")
    state["first_traceback"] = text[start:start+8192] if start >= 0 else None
    state["log_sample_scope"] = "first and last 64KiB; clocks do not establish a cross-host first cause"
    if state.get("running") and listener_port is not None:
        try:
            import psutil
            values = [item for pid in owned_pids for item in psutil.Process(pid).net_connections(kind="tcp")]
            state["owned_listener"] = any(row.status == psutil.CONN_LISTEN and row.laddr.port == listener_port for row in values)
        except Exception:
            state["owned_listener"] = None
    return state


def check_node(plan, index, node, *, verify_files=True, probe_gpu=True, check_tokenizer=True,
               allow_owned_listener=False):
    from shard.pipeline_plan import validate_plan
    plan = validate_plan(plan)
    values, errors, warnings = {}, [], []
    stage = plan["stages"][index] if index is not None else None
    node_id = stage["node_id"] if stage else plan["coordinator"].get("node_id", plan["stages"][0]["node_id"])
    def record(name, action):
        try:
            values[name] = action()
        except Exception as error:
            errors.append({"check": name, "error_class": type(error).__name__, "message": str(error)[:500]})
    cohort = plan.get("model_cohort")
    dspark = node.get("dspark", True)
    if type(dspark) is not bool:
        errors.append({"check": "dspark", "message": "dspark must be an explicit boolean"})
        dspark = False
    if not cohort:
        errors.append({"check": "cohort", "message": "strict deployment requires the complete model cohort"})
    else:
        record("artifacts", lambda: check_artifacts(node["model"], cohort, stage=stage,
            dspark=bool(stage and stage["tail"] and dspark), verify_files=verify_files))
    record("signer", lambda: _key(node.get("coordinator_key", node.get("node_key")) if stage is None else node["node_key"],
        plan["coordinator"].get("signer_pubkey") if stage is None else stage.get("signer_pubkey")))
    versions = _versions(); values["versions"] = versions
    for name in ("torch", "safetensors", "cryptography", "psutil"):
        if not versions[name]: errors.append({"check": "dependencies", "message": "missing " + name})
    for name, expected in node.get("expected_versions", {}).items():
        if versions.get(name) != expected:
            errors.append({"check": "versions", "message": "installed " + name + " differs from the frozen node configuration"})
    if stage is not None:
        if os.environ.get("V4_KERNELS", "tilelang") == "tilelang" and not versions["tilelang"]:
            errors.append({"check": "dependencies", "message": "tilelang kernel backend is not installed"})
        if allow_owned_listener:
            values["listener"] = {"port": node["listen_port"], "scope": "existing owned process; checked by deployment health"}
        else:
            record("listener", lambda: _port(node["listen_port"], node.get("bind", "127.0.0.1")))
        if probe_gpu:
            record("gpu", lambda: _gpu(node.get("device", "cuda:0"), stage["gpu_uuid"]))
            def hadamard():
                import torch
                from v4_runtime_init import ensure_hadamard, probe_hadamard, hadamard_identity
                ensure_hadamard()
                probes = [probe_hadamard(node.get("device", "cuda:0"), torch.bfloat16, shape)
                          for shape in ((1, 1, 64, 128), (1, 1, 128))]
                return {**hadamard_identity(), "layout_probes": probes,
                        "scope": "actual indexer layout smoke; not all-model numerical acceptance"}
            record("hadamard", hadamard)
        else:
            warnings.append("GPU visibility/identity was not checked")
        if dspark and stage["tail"] and stage["lo"] > 40:
            errors.append({"check": "dspark", "message": "DSpark tail must own layers 40,41,42 and embedding dependencies"})
    elif check_tokenizer:
        def tokenizer():
            from v4_tokenizer import load_v4_tokenizer
            value = load_v4_tokenizer(node["model"])
            return {"loaded": True, "class": type(value).__name__}
        record("tokenizer", tokenizer)
    from shard.resources import measure_host_resources
    record("host", lambda: measure_host_resources(node["model"]))
    warnings.append("manual checks do not reserve GPU/RAM/disk or certify numerical accuracy")
    if not verify_files and stage is not None:
        warnings.append("weight payload SHA256 was not verified; preview cannot authorize startup")
    return {"schema": "shard-v4-preflight/1", "node_id": node_id, "stage": index,
            "preflight_ok": not errors, "runtime_ready": False, "checks": values,
            "errors": errors, "warnings": warnings, "scope": "read-only deployment checks",
            "remaining_checks": ["Stage.load strict tensor coverage", "signed real-model warmup", "numerical/performance acceptance"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-stdin", action="store_true", required=True)
    args = parser.parse_args(argv)
    request = json.load(sys.stdin)
    apply_environment(request["environment"])
    result = check_node(request["plan"], request.get("index"), request["node"],
        verify_files=request.get("verify_files", True), probe_gpu=request.get("probe_gpu", True),
        check_tokenizer=request.get("check_tokenizer", True),
        allow_owned_listener=request.get("allow_owned_listener", False))
    print(json.dumps(result, allow_nan=False))
    return 0 if result["preflight_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

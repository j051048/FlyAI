"""Owned cross-process host activation probe; never a planner/lease calibration.

CPU is the default and tests real shared memory. Actual CUDA IPC requires two
idle Linux devices plus --execute-gpu. No weights, rental, SSH or installation.
"""
import argparse
import base64
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import socket
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "phase0", ROOT / "engines" / "deepseek_v4"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _binding(position):
    import v4_host_transport as H
    return H.request_binding("owned-probe:1", {"job_id": "synthetic-activation", "nonce": "b" * 64,
        "swarm_id": "local-probe"}, start_pos=position)


def _worker(path, target, count, port_queue):
    import torch
    import v4_host_transport as H
    import v4_pipe as V
    torch.set_num_threads(1)
    transport = server = conn = None
    try:
        cfg = H.load_config(path); ch = cfg["channels"][0]
        transport = H.HostTransport(cfg, local_pubkey=ch["dst_pubkey"], local_uuid=ch["dst_uuid"],
                                    cohort_id=ch["cohort_id"], device=target)
        server = socket.socket(); server.bind(("127.0.0.1", 0)); server.listen(1); server.settimeout(30)
        port_queue.put({"port": server.getsockname()[1]})
        conn, _ = server.accept(); conn.settimeout(30)
        for index in range(count):
            frame = V.recv_msg(conn)
            value, ids, raw = V._recv_hids(frame, object(), device=target,
                host_transport=transport, host_binding=_binding(index))
            checksum = raw.digest.hex() if transport.device_digest_enabled else hashlib.sha256(raw).hexdigest()
            V.send_msg(conn, {"sha256": checksum, "ids": ids,
                             "shape": list(value.shape), "status": transport.status()})
    except BaseException as error:
        port_queue.put({"error": type(error).__name__ + ": " + str(error)})
        raise
    finally:
        if transport is not None: transport.close()
        if conn is not None: conn.close()
        if server is not None: server.close()


def measure(*, source_device="cpu", target_device="cpu", repeats=5, tokens=1, dim=64,
            hc=4, max_mib=16, execute_gpu=False, digest_backend="cpu"):
    if (any(type(v) is not int for v in (repeats, tokens, dim, hc, max_mib))
            or not 1 <= repeats <= 1000 or not 1 <= tokens <= 256 or not 1 <= dim <= 65536
            or not 1 <= hc <= 16 or not 1 <= max_mib <= 512
            or not isinstance(source_device, str) or not isinstance(target_device, str)
            or type(execute_gpu) is not bool):
        raise ValueError("explicit bounded local transport probe dimensions required")
    gpu = source_device != "cpu" or target_device != "cpu"
    if digest_backend not in ("cpu", "device_sha256") or digest_backend != "cpu" and not gpu:
        raise ValueError("device digest probe requires actual CUDA IPC")
    if gpu and not execute_gpu:
        raise ValueError("actual GPU copies require explicit --execute-gpu on two idle Linux devices")
    if gpu and (not source_device.startswith("cuda:") or not target_device.startswith("cuda:") or os.name != "posix"):
        raise ValueError("CUDA IPC probe requires two explicit Linux CUDA devices")
    import torch
    import v4_host_transport as H
    import v4_pipe as V
    size = tokens * dim * hc * 2
    if size > (max_mib << 20) // 2:
        raise ValueError("activation exceeds the declared two-slot probe reservation")
    if gpu:
        if not torch.cuda.is_available(): raise ValueError("actual CUDA accelerator unavailable")
        from v4_observability import gpu_uuid_text
        uuids = [gpu_uuid_text(getattr(torch.cuda.get_device_properties(d), "uuid", None))
                 for d in (source_device, target_device)]
        if not all(uuids) or uuids[0] == uuids[1]: raise ValueError("two distinct actual GPU UUIDs required")
    else:
        uuids = ["reference-0", "reference-1"]
    if digest_backend == "device_sha256":
        # Cold compilation is outside the receiver's bounded accept/read waits.
        # The spawned peer loads the same compiled extension from its cache.
        from v4_device_digest import tensor_digest
        if tensor_digest([b"owned-probe"], device=source_device) != hashlib.sha256(b"owned-probe").digest():
            raise RuntimeError("device SHA warmup failed")
    ctx = mp.get_context("spawn"); queue = ctx.Queue()
    process = source = connection = None
    samples, target_status = [], None
    # Private manifests and secret never enter the JSON result or source tree.
    with tempfile.TemporaryDirectory(prefix="flyai-host-probe-") as directory:
        folder = Path(directory)
        if os.name == "posix": os.chmod(folder, 0o700)
        cfg = {"schema": H.CONFIG_SCHEMA, "host_id": "owned-probe", "directory": str(folder.resolve()),
               "secret": secrets.token_hex(32), "backend": "cuda_ipc" if gpu else "shared_memory_reference",
               "max_bytes": max_mib << 20, "slots": 2, "ttl_s": 30,
               "channels": [{"id": "0-1", "cohort_id": "a" * 64,
                 "src_pubkey": base64.b64encode(bytes([1]) * 32).decode(),
                 "dst_pubkey": base64.b64encode(bytes([2]) * 32).decode(),
                 "src_uuid": uuids[0], "dst_uuid": uuids[1]}]}
        if digest_backend != "cpu":
            cfg["digest_backend"] = digest_backend
        path = folder / "config.json"; path.write_text(json.dumps(cfg), encoding="utf-8")
        if os.name == "posix": os.chmod(path, 0o600)
        try:
            process = ctx.Process(target=_worker, args=(str(path), target_device, repeats, queue))
            process.start(); hello = queue.get(timeout=45)
            if "error" in hello: raise RuntimeError(hello["error"])
            ch = cfg["channels"][0]
            source = H.HostTransport(cfg, local_pubkey=ch["src_pubkey"], local_uuid=ch["src_uuid"],
                                     cohort_id=ch["cohort_id"], device=source_device)
            connection = socket.create_connection(("127.0.0.1", hello["port"]), timeout=30)
            for index in range(repeats):
                value = (torch.arange(tokens * hc * dim, device=source_device) % 97).reshape(1, tokens, hc, dim).bfloat16()
                started = time.perf_counter()
                frame, raw = V._make_step_frame(value, torch.arange(tokens).reshape(1, -1), index, object(),
                    host_transport=source, host_binding=_binding(index))
                if "host_ref" not in frame: raise RuntimeError("probe did not select its real local slot backend")
                V.send_msg(connection, frame); result = V.recv_msg(connection)
                checksum = raw.digest.hex() if source.device_digest_enabled else hashlib.sha256(raw).hexdigest()
                if result["sha256"] != checksum:
                    raise RuntimeError("destination payload/receipt bytes differ")
                target_status = result["status"]
                samples.append({"owner_ready_to_verified_reply_ms": (time.perf_counter() - started) * 1000,
                                "payload_bytes": (size // 2 + tokens * hc * 2 + tokens * 8
                                    if V.V4_FP8_WIRE else size + tokens * 8)})
            process.join(15)
            if process.exitcode != 0: raise RuntimeError("owned receiver did not exit cleanly")
        finally:
            if connection is not None: connection.close()
            if process is not None and process.is_alive(): process.terminate(); process.join(10)
            if source is not None: source.close()
            queue.close(); queue.join_thread()
        copied = bool(target_status and target_status["ipc_copies"] == repeats)
        if gpu and not copied:
            raise RuntimeError("actual pair fell back to shared memory; CUDA IPC is unverified")
        return {"schema": "v4-host-transport-probe/1", "scope": "synthetic_no_lease_no_strict_session_not_planner_evidence",
                "source_device": source_device, "target_device": target_device, "gpu_uuids": uuids if gpu else None,
                "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
                "gpu_copy_observed": copied, "performance_qualified": False,
                "receipt_hash_path": target_status["receipt_hash_path"], "samples": samples,
                "source": source.status(), "target": target_status,
                "process_cleanup": {"confirmed": process.exitcode == 0,
                    "quarantined_slots": source.status()["quarantined"]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-device", default="cpu"); parser.add_argument("--target-device", default="cpu")
    parser.add_argument("--repeats", type=int, default=5); parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--dim", type=int, default=64); parser.add_argument("--hc", type=int, default=4)
    parser.add_argument("--max-mib", type=int, default=16); parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--digest-backend", choices=("cpu", "device_sha256"), default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = measure(source_device=args.source_device, target_device=args.target_device, repeats=args.repeats,
                     tokens=args.tokens, dim=args.dim, hc=args.hc, max_mib=args.max_mib, execute_gpu=args.execute_gpu,
                     digest_backend=args.digest_backend)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(Path(args.out).resolve()), "gpu_copy_observed": result["gpu_copy_observed"],
                      "performance_qualified": False}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"host transport probe refused: {error}", file=sys.stderr)
        raise SystemExit(2)

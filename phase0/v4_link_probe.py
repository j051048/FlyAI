"""Measure an explicitly configured auxiliary echo route, outside the model ring.

The listener is loopback-only; an existing SSH tunnel can reach it. Samples include
socket/route/echo overhead and are not a decomposition of model-stage receive waits.
Never point this client at an inference-stage listener.
"""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import socket
import socketserver
import struct
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shard.benchmark_metrics import percentile

MAX_PAYLOAD = 1 << 20
NONCE_BYTES = 32


def _read(sock, size, *, clean_eof=False):
    out = bytearray(size)
    offset = 0
    while offset < size:
        got = sock.recv_into(memoryview(out)[offset:])
        if not got:
            if clean_eof and not offset:
                return None
            raise ConnectionError("auxiliary echo peer closed an incomplete frame")
        offset += got
    return out


def echo_connection(sock, *, timeout=10):
    sock.settimeout(timeout)
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        header = _read(sock, 4, clean_eof=True)
        if header is None:
            return
        size = struct.unpack("!I", header)[0]
        if not NONCE_BYTES <= size <= MAX_PAYLOAD + NONCE_BYTES:
            raise ValueError("bounded auxiliary echo frame required")
        message = _read(sock, size)
        sock.sendall(header + message)


class _EchoHandler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            echo_connection(self.request)
        except (ConnectionError, OSError, ValueError):
            return


def make_server(host="127.0.0.1", port=29790):
    if host not in ("127.0.0.1", "::1"):
        raise ValueError("auxiliary echo listeners must bind loopback; use an explicit SSH route")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("invalid auxiliary listener port")
    class Server(socketserver.ThreadingTCPServer):
        address_family = socket.AF_INET6 if host == "::1" else socket.AF_INET
        allow_reuse_address = True
        daemon_threads = True
    return Server((host, port), _EchoHandler)


def measure_route(endpoint, *, samples=20, warmup=3, payload_bytes=32768, timeout=10,
                  route_id="auxiliary-test-route", transport="operator-declared"):
    for value, maximum, label in ((samples, 10000, "samples"), (payload_bytes, MAX_PAYLOAD, "payload_bytes")):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{label} must be in 1..{maximum}")
    if type(warmup) is not int or not 0 <= warmup <= 1000:
        raise ValueError("warmup must be in 0..1000")
    parsed = urlsplit("//" + endpoint)
    if not parsed.hostname or parsed.port is None or not 1 <= parsed.port <= 65535:
        raise ValueError("explicit auxiliary host:port required")
    if parsed.port in (29610, 29611, 29612):
        raise ValueError("use a separate auxiliary port, not a known V4 engine/sidecar listener")
    if not isinstance(route_id, str) or not route_id or len(route_id) > 256:
        raise ValueError("bounded route identity required")
    payload = secrets.token_bytes(payload_bytes)
    raw = []
    connected = time.perf_counter()
    with socket.create_connection((parsed.hostname, parsed.port), timeout=timeout) as sock:
        connect_s = time.perf_counter() - connected
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for index in range(warmup + samples):
            message = secrets.token_bytes(NONCE_BYTES) + payload
            header = struct.pack("!I", len(message))
            started = time.perf_counter()
            sock.sendall(header + message)
            echoed_header = _read(sock, 4)
            if struct.unpack("!I", echoed_header)[0] != len(message):
                raise ValueError("auxiliary echo returned a different frame size")
            echoed = _read(sock, len(message))
            elapsed = time.perf_counter() - started
            if echoed != message:
                raise ValueError("auxiliary route did not echo this exact fresh challenge/payload")
            if index >= warmup:
                raw.append({"rtt_s": elapsed, "sent_bytes": len(message) + 4,
                            "received_bytes": len(message) + 4})
    values = [row["rtt_s"] for row in raw]
    total = sum(values)
    return {"schema": "shard-link-probe/1", "route_id": route_id, "endpoint": endpoint,
        "transport": transport, "transport_identity_scope": "operator declaration; no remote attestation",
        "payload_bytes": payload_bytes, "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "warmup_samples": warmup, "measured_samples": samples, "connect_s": connect_s,
        "samples": raw, "p50_rtt_s": percentile(values, 50), "p95_rtt_s": percentile(values, 95),
        "effective_round_trip_bytes_per_s": sum(row["sent_bytes"] + row["received_bytes"] for row in raw) / total if total > 0 else None,
        "scope": "persistent-connection application echo RTT including route, OS and echo processing; no model/GPU work",
        "counts_inference_acceptance": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="command", required=True)
    listen = children.add_parser("listen")
    listen.add_argument("--bind", default="127.0.0.1"); listen.add_argument("--port", type=int, default=29790)
    run = children.add_parser("run")
    run.add_argument("--endpoint", required=True); run.add_argument("--out", required=True)
    run.add_argument("--bytes", type=int, default=32768, dest="payload_bytes")
    run.add_argument("--samples", type=int, default=20); run.add_argument("--warmup", type=int, default=3)
    run.add_argument("--route-id", required=True); run.add_argument("--transport", default="operator-declared")
    args = parser.parse_args(argv)
    if args.command == "listen":
        with make_server(args.bind, args.port) as server:
            print(json.dumps({"auxiliary_listener": server.server_address, "model_stage": False}), flush=True)
            server.serve_forever()
    else:
        result = measure_route(args.endpoint, samples=args.samples, warmup=args.warmup,
            payload_bytes=args.payload_bytes, route_id=args.route_id, transport=args.transport)
        Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"report": args.out, "p50_rtt_s": result["p50_rtt_s"], "p95_rtt_s": result["p95_rtt_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""inter-stage transport — the part shard owns, the wedge.

The networking lives in the libp2p sidecar (a Go daemon — see ../sidecar): identity,
encryption, NAT traversal, and the peer connections are its job. The engine keeps its
plain socket code and talks to the sidecar over localhost; the sidecar carries each
connection to/from the right ring neighbour over libp2p. So this module is just the
*message* layer the engine speaks over that socket — `send_msg`/`recv_msg`, a drop-in
for phase0/wire.py minus the ChaCha seal (libp2p already encrypts + authenticates the
link). The engine swaps transport by one import: `import wire` → `import shard.transport
as wire`.

Per the boundary law (docs/INTEGRATION.md) this is pure engine plumbing: peers and
bytes, nothing about c0mpute.

frame wire: 8-byte big-endian length + JSON header + raw tensor blobs (matches the
phase0/wire.py framing so call sites are unchanged).
"""
import json
import os
import socket
import struct
import threading

import numpy as np
import torch


def _read_exact(sock: socket.socket, n: int) -> bytearray:
    # preallocate + recv_into: no bytearray growth copies, no final bytes() copy — callers
    # (struct.unpack, _unpack) only need a buffer, not bytes
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if not r:
            try:
                peer = sock.getpeername()
            except (OSError, AttributeError):
                peer = "unknown"
            raise ConnectionError(f"peer {peer} closed the connection while reading a frame ({got}/{n} bytes)")
        got += r
    return buf


# ── message codec (engine protocol): JSON header + raw tensor blobs ──
# Ported from phase0/wire.py's _pack/_unpack — the *message* format the engine speaks
# (control fields + tensors). No ChaCha seal: encryption + identity are the sidecar's job.
_DTYPES = {str(d): d for d in (torch.float32, torch.float16, torch.bfloat16,
                               torch.float8_e4m3fn, torch.float8_e5m2,   # fp8 activation transport (M25_FP8_WIRE)
                               torch.int64, torch.int32, torch.uint8, torch.bool)}
_DSIZE = {d: torch.empty(0, dtype=d).element_size() for d in set(_DTYPES.values())}  # bytes/element, to size-check a blob

# Hard cap on one inbound frame's declared length. The 8-byte length prefix is attacker-controlled
# (a permissionless peer, or a PSK-mode dialer on a libp2p ring), so an oversized claim must be
# refused BEFORE allocating/reading the body — else any peer OOMs the stage. Env M25_MAX_FRAME
# (bytes); default 256 MiB >> the largest legit activation chunk (bf16/fp8 prefill), tiny vs a DoS.
MAX_FRAME = int(os.environ.get("M25_MAX_FRAME") or 256 * 1024 * 1024)

_RECEIVE_POOL = None
_POOL_LOCK = threading.Lock()


def configure_receive_pool(budget_bytes=0, *, slots=2, pinned=False):
    """Explicit process-wide opt-in, before receiving job frames; no protocol change."""
    global _RECEIVE_POOL
    try:
        from .receive_buffers import ReceiveBufferPool
    except ImportError:
        from receive_buffers import ReceiveBufferPool
    if type(budget_bytes) is not int or budget_bytes < 0 or type(pinned) is not bool:
        raise ValueError("explicit receive buffer budget/pinning required")
    with _POOL_LOCK:
        if _RECEIVE_POOL is not None and _RECEIVE_POOL.snapshot()["live_slots"]:
            raise RuntimeError("cannot replace a receive pool while tensor views are live")
        _RECEIVE_POOL = ReceiveBufferPool(budget_bytes, slots=slots, pinned=pinned) if budget_bytes else None
    return receive_pool_status()


def receive_pool_status():
    pool = _RECEIVE_POOL
    return pool.snapshot() if pool is not None else {"enabled": False, "scope": "process lifetime"}


def _pack_parts(obj) -> list:
    """encode one message as a list of wire segments (bytes-likes) whose concatenation IS the
    frame body — byte-identical to the historical _pack output. Kept segmented so send_msg can
    hand the pieces straight to sendmsg() with zero concatenation copies; tensor blobs are
    memoryviews over the (contiguous, CPU) tensor storage, so they aren't copied here either."""
    blobs = []

    def encode(o):
        if torch.is_tensor(o):
            t = o.detach().cpu().contiguous()
            blobs.append(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
            return {"__t__": len(blobs) - 1, "dtype": str(t.dtype), "shape": list(t.shape)}
        if isinstance(o, dict):
            return {k: encode(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [encode(v) for v in o]
        if o is None or isinstance(o, (bool, int, float, str)):
            return o
        raise TypeError(f"transport cannot encode {type(o).__name__}")

    head = json.dumps(encode(obj)).encode()
    encode = None  # break recursive closure ownership; do not retain tensor blobs until cyclic GC
    parts = [struct.pack("!I", len(head)) + head]
    for b in blobs:
        parts.append(struct.pack("!Q", len(b)))
        parts.append(b)
    return parts


def _pack(obj) -> bytes:
    return b"".join(_pack_parts(obj))


def _unpack(buf, *, copy_tensors=True):
    mv = memoryview(buf)                    # zero-copy blob slices (a bytes slice copies)
    if len(mv) < 4:
        raise ValueError("truncated frame header")
    (hlen,) = struct.unpack_from("!I", mv, 0)
    if hlen > len(mv) - 4:
        raise ValueError("header length exceeds frame")
    head = json.loads(bytes(mv[4:4 + hlen]))
    blobs, off = [], 4 + hlen
    while off < len(mv):
        if len(mv) - off < 8:
            raise ValueError("truncated blob length")
        (blen,) = struct.unpack_from("!Q", mv, off); off += 8
        if blen > len(mv) - off:
            raise ValueError("blob length exceeds frame")
        blobs.append(mv[off:off + blen]); off += blen

    def decode(node):
        if isinstance(node, dict):
            if "__t__" in node:
                dt = _DTYPES.get(node["dtype"])
                if dt is None:
                    raise ValueError(f"unknown tensor dtype {node['dtype']!r}")
                index = node["__t__"]
                if type(index) is not int or not 0 <= index < len(blobs):
                    raise ValueError("invalid tensor blob index")
                shape, blob = node["shape"], blobs[index]
                if not isinstance(shape, list) or len(shape) > 32:
                    raise ValueError("invalid tensor shape")
                numel = 1
                for d in shape:
                    if type(d) is not int or d < 0:
                        raise ValueError(f"bad tensor dim {d!r}")
                    numel *= d
                if numel * _DSIZE[dt] != len(blob):    # declared shape must match the blob EXACTLY — a lying
                    raise ValueError(                  # shape (esp. an empty blob + huge shape) must never
                        f"tensor {shape} needs {numel * _DSIZE[dt]} bytes, blob has {len(blob)}")  # drive torch.empty
                if numel == 0:
                    return torch.empty(shape, dtype=dt, device="cpu")
                array = np.frombuffer(blob, dtype=np.uint8)
                raw = torch.from_numpy(array.copy() if copy_tensors else array)
                return raw.view(dt).reshape(shape)
            return {k: decode(v) for k, v in node.items()}
        if isinstance(node, list):
            return [decode(v) for v in node]
        return node

    try:
        return decode(head)
    finally:
        decode = None  # returned tensor storages, not the codec closure, own borrowed frame leases


_IOV_CAP = 512      # segments per sendmsg() call, safely under the kernel's UIO_MAXIOV (1024)


def _sendall_vectored(sock: socket.socket, parts: list) -> None:
    """sendall for a list of bytes-like segments via scatter/gather sendmsg — the segments are
    never concatenated (the old path copied every blob into one frame buffer, then copied that
    again for the length prefix: ~2× the message in transient memcpy per hop per direction)."""
    if not hasattr(sock, "sendmsg"):        # non-POSIX / wrapped socket: fall back to one join
        sock.sendall(b"".join(parts))
        return
    bufs = [memoryview(p) for p in parts if len(p)]
    while bufs:
        sent = sock.sendmsg(bufs[:_IOV_CAP])
        while bufs and sent >= len(bufs[0]):
            sent -= len(bufs.pop(0))
        if sent:                            # partial segment: advance within it
            bufs[0] = bufs[0][sent:]


def send_msg(sock: socket.socket, obj) -> int:
    """pack + send one message over a (sidecar-tunneled) socket; returns bytes on the wire.
    drop-in for phase0/wire.send_msg minus the SHARD_PSK seal (libp2p encrypts the link)."""
    parts = _pack_parts(obj)
    n = sum(len(p) for p in parts)
    _sendall_vectored(sock, [struct.pack("!Q", n)] + parts)
    return 8 + n


def recv_msg(sock: socket.socket):
    """receive + unpack one message. drop-in for phase0/wire.recv_msg.
    A frame we can't parse into a message is a DEAD EDGE, not a process death: re-raise as
    ConnectionError so `except EDGE_ERRORS` supervision resets the connection (wire.py has carried
    this guard since the pickle removal; this production path lost it — one malformed frame, e.g. a
    PSK-mode peer dialing a libp2p-mode ring, crashed the stage and its warm weights)."""
    (n,) = struct.unpack("!Q", _read_exact(sock, 8))
    if n > MAX_FRAME:                                   # refuse an oversized length BEFORE allocating the body
        raise ConnectionError(f"frame length {n} exceeds MAX_FRAME ({MAX_FRAME})")
    with _POOL_LOCK:
        pool = _RECEIVE_POOL
        frame = pool.borrow(n) if pool is not None else None
    borrowed = frame is not None
    if borrowed:
        view, got = memoryview(frame), 0
        while got < n:
            read = sock.recv_into(view[got:], n - got)
            if not read:
                raise ConnectionError(f"peer closed while reading buffered frame ({got}/{n} bytes)")
            got += read
    else:
        frame = _read_exact(sock, n)
    try:
        return _unpack(frame, copy_tensors=not borrowed)
    except (ValueError, KeyError, IndexError, TypeError, RuntimeError, struct.error, json.JSONDecodeError) as e:
        raise ConnectionError(f"malformed frame ({type(e).__name__}: {str(e)[:80]})") from e


# No-ops so `import shard.transport as wire` is a perfect drop-in: the libp2p sidecar
# provides encryption + identity, so there is no pre-shared key to load.
def key_from_env(var: str = "SHARD_PSK") -> None:
    return None


def use_key(material) -> None:
    return None

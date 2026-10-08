"""shard: speculative decoding over the N-node pipeline (specdec.py x pipeline.py).

specdec.py proved draft-and-verify over a 2-node split. pipeline.py proved an
N-stage split big enough to hold a 120B target (gpt-oss: partial per-node loading,
sliding-window masks, MXFP4). this puts them together: a small draft proposes K
tokens locally on the head, and the *distributed* target verifies all K in ONE
traversal of the whole chain -- the same single WAN round-trip that plain decode
spends on a single token now commits several. greedy acceptance, so the output is
token-for-token identical to plain pipeline decode; the target is never made whole
on any node.

the draft runs on its own GPU on the head (a 120B stage already fills a 24GB card,
so the draft can't share it); every other node is unchanged from pipeline.py and
holds only its block of the target. the verify op carries a lazy `crop`: the prior
round's rejected tokens are rolled back from every node's cache on the next verify,
piggybacked, so a round costs exactly one round-trip end to end.

  # every node shares one swarm secret (same value on each box):
  export SHARD_PSK=$(openssl rand -hex 32)
  # tail (stage N-1)
  CUDA_VISIBLE_DEVICES=1 python specpipe.py --stage 3 --nstages 4 --model M --listen-port 29503
  # middle (stage i)
  CUDA_VISIBLE_DEVICES=0 python specpipe.py --stage 2 --nstages 4 --model M --listen-port 29502 --next H:29503
  # head (stage 0): stage block on one GPU, draft on another, drives generation
  CUDA_VISIBLE_DEVICES=0,2 python specpipe.py --stage 0 --nstages 4 --model M \
      --next 127.0.0.1:29501 --draft DRAFT --device cuda:0 --draft-device cuda:1 --adaptive
"""

import argparse, socket, time, threading, queue, os, json, hashlib, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_ROOT, "shard")) and _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
if os.environ.get("SHARD_TRANSPORT") == "libp2p":   # libp2p sidecar transport (no PSK; see node_kv)
    try:
        import transport as wire
    except ImportError:
        from shard import transport as wire
else:
    import wire
from pipeline import load_stage, run_block as _run_block
from node_kv import send_msg as _raw_send_msg, recv_msg as _raw_recv_msg, EDGE_ERRORS, TransportError
try:
    from shard.pipeline_session import (SessionConfig, SessionSocket, SessionAcceptor, SessionBusy,
        ProtocolError, hello_client, send_message, recv_message, prepare_message)
except ImportError:
    from pipeline_session import (SessionConfig, SessionSocket, SessionAcceptor, SessionBusy,
        ProtocolError, hello_client, send_message, recv_message, prepare_message)


def send_msg(sock, payload):
    return send_message(_raw_send_msg, sock, payload)


def recv_msg(sock):
    return recv_message(_raw_recv_msg, sock)


def _address(endpoint):
    from urllib.parse import urlsplit
    parsed = urlsplit("//" + endpoint)
    if not parsed.hostname or parsed.port is None:
        raise ValueError("endpoint must be host:port, bracket IPv6 addresses")
    return parsed.hostname, parsed.port


def run_block(h, parts, *args, **kwargs):
    guard = parts.get("_lease_guard")
    if guard:
        guard.assert_live()
    telemetry = parts.get("_telemetry")
    if telemetry:
        with telemetry.measure("forward", device=h.device):
            result = _run_block(h, parts, *args, **kwargs)
    else:
        result = _run_block(h, parts, *args, **kwargs)
    if guard:
        guard.assert_live()
    return result


def _guard_fast_verify(fv, parts):
    guard = parts.get("_lease_guard")
    if guard:
        import functools
        for name in ("reset", "prefill", "decode", "tree_decode", "tree_gather"):
            original = getattr(fv, name)
            def checked(*args, _original=original, **kwargs):
                guard.assert_live()
                result = _original(*args, **kwargs)
                guard.assert_live()
                return result
            setattr(fv, name, functools.wraps(original)(checked))
    telemetry = parts.get("_telemetry")
    return telemetry.wrap(fv) if telemetry else fv


def connect_ring(head, tail, *, session_config=None, timeout=600, retry_s=0):
    """Strict managed entry; no-plan direct API remains explicit legacy-compatible."""
    deadline, last = time.monotonic() + retry_s, None
    while True:
        pipe = ret = None
        try:
            pipe = socket.create_connection(_address(head), timeout=timeout)
            if session_config is not None:
                sid = __import__("secrets").token_hex(16)
                pipe = hello_client(pipe, session_config, 0, "drive", _raw_send_msg, _raw_recv_msg, session_id=sid)
            if tail:
                ret = socket.create_connection(_address(tail), timeout=timeout)
                if session_config is not None:
                    ret = hello_client(ret, session_config, session_config.plan["nstages"] - 1, "return",
                                       _raw_send_msg, _raw_recv_msg, grant=pipe.grant)
                else:
                    send_msg(ret, {"op": "hello_return"})
            return pipe, ret
        except Exception as error:
            for channel in (pipe, ret):
                if channel is not None:
                    channel.close()
            if isinstance(error, (SessionBusy, ProtocolError)) or time.monotonic() >= deadline:
                raise
            last = error
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))


def ping_ring(pipe, ret=None):
    """Idle-only head control ping; never reads/consumes the tail result stream."""
    if not isinstance(pipe, SessionSocket):
        raise ProtocolError("session ping requires strict managed channels")
    send_msg(pipe, {"op": "session_ping"})
    pong = recv_msg(pipe)
    if not isinstance(pong, dict) or pong.get("op") != "session_pong" or pong.get("session_id") != pipe.grant["session_id"]:
        raise ProtocolError("invalid owner session ping acknowledgement")
    return pong


def _forward_connect(parts, nxt, timeout):
    host, port = _address(nxt)
    cfg = parts.get("_session_config")
    deadline = time.monotonic() + timeout
    while True:
        channel = None
        try:
            channel = socket.create_connection((host, int(port)), timeout=min(5, timeout))
            wrapped = hello_client(channel, cfg, cfg.index + 1, "forward", _raw_send_msg, _raw_recv_msg) if cfg else channel
            wrapped.settimeout(timeout)
            return wrapped
        except Exception as error:
            if channel is not None:
                channel.close()
            if (isinstance(error, ProtocolError) and not isinstance(error, SessionBusy)) or time.monotonic() >= deadline:
                raise
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))


def _server(parts, port, timeout):
    cfg = parts.get("_session_config")
    endpoint = cfg.stage(cfg.index)["endpoint"] if cfg else None
    ipv6 = endpoint is not None and ":" in _address(endpoint)[0]
    srv = socket.socket(socket.AF_INET6 if ipv6 else socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("::" if ipv6 else "0.0.0.0", port)); srv.listen(32)
    acceptor = (SessionAcceptor(srv, cfg, _raw_send_msg, _raw_recv_msg,
        key=parts.get("_session_key"), timeout=timeout) if cfg else None)
    return srv, acceptor


def _accept_stage(srv, acceptor, parts, timeout):
    if acceptor is not None:
        cfg = parts["_session_config"]
        return acceptor.get("drive" if cfg.index == 0 else "forward"), ("authenticated", 0)
    conn, addr = srv.accept(); conn.settimeout(timeout)
    return conn, addr
from tree import accept_tree, gather_cache
from fastverify import FastVerify
from ngram_draft import NgramDrafter
from specsample import Sampler
import os
try:
    from receipt import ReceiptSigner, load_or_make_node_key, verify_receipt, verify_coverage
except ImportError:
    from shard.receipt import ReceiptSigner, load_or_make_node_key, verify_receipt, verify_coverage
RECEIPTS = bool(os.environ.get("SHARD_RECEIPTS")) and ReceiptSigner is not None
NODE_KEY_PATH = os.environ.get("SHARD_NODE_KEY", "/root/.shard_node_key")


def _act_digest(t):
    """Commit actual dtype/shape/bytes without a lossy float16 conversion."""
    value = t.detach().contiguous().cpu()
    metadata = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                          sort_keys=True, separators=(",", ":")).encode()
    return len(metadata).to_bytes(4, "big") + metadata + value.view(torch.uint8).numpy().tobytes()


# ---- fault-tolerance CHECKPOINT envelope (--ft-dump / --resume-file) ----
# The old dump was a bare {"output_ids": [...]}: nothing bound it to the generation it came from,
# so a stale/foreign checkpoint (other prompt, other model, other sampling settings) resumed
# silently and produced a HYBRID generation. The envelope binds every generation input plus a
# digest of the committed tokens; the resume side refuses anything that doesn't match THIS run.
CKPT_SCHEMA = "shard-ckpt-v1"


class CheckpointError(Exception):
    """A resume checkpoint is unversioned, corrupt, or bound to a different generation."""


def _sha256_text(s):
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()


def _ids_digest(ids):
    return hashlib.sha256(json.dumps([int(i) for i in ids]).encode()).hexdigest()


def checkpoint_env(*, prompt, model, tokenizer, settings):
    """The generation-input binding a checkpoint carries and a resume must match exactly.
    `job` is derived from the binding fields, so same-request heal+resume matches for free
    while any cross-job reuse (different prompt/model/tokenizer/settings) is a mismatch."""
    env = {"schema": CKPT_SCHEMA, "prompt_sha256": _sha256_text(prompt), "model": str(model),
           "tokenizer": str(tokenizer), "settings": dict(settings)}
    env["job"] = _sha256_text(json.dumps(env, sort_keys=True))
    return env


def write_checkpoint(path, env, output_ids, **extra):
    """Write the versioned checkpoint ATOMICALLY (tmp + rename): a healer polling the file can
    never read a torn dump, and a crash mid-write can't leave a truncated checkpoint behind."""
    ids = [int(i) for i in output_ids]
    d = {**env, "output_ids": ids, "ids_sha256": _ids_digest(ids), **extra}
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(d, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    return d


def load_checkpoint(path, env):
    """Load the committed tokens for a resume; REJECT (CheckpointError) anything not bound to
    this exact generation -- never splice another job's tokens into a fresh prefill."""
    d = json.load(open(path))
    if d.get("schema") != CKPT_SCHEMA:
        raise CheckpointError(f"checkpoint schema {d.get('schema')!r} != {CKPT_SCHEMA!r} "
                              f"(unversioned/foreign checkpoint refused)")
    for k in ("job", "prompt_sha256", "model", "tokenizer", "settings"):
        if d.get(k) != env[k]:
            raise CheckpointError(f"checkpoint {k} mismatch: resuming it would splice another "
                                  f"generation (checkpoint {str(d.get(k))[:60]!r}, this run {str(env[k])[:60]!r})")
    ids = d.get("output_ids")
    if not isinstance(ids, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        raise CheckpointError("checkpoint output_ids malformed (not a list of token ids)")
    if d.get("ids_sha256") != _ids_digest(ids):
        raise CheckpointError("checkpoint committed-token digest mismatch (corrupt or edited)")
    return ids


SOCK_BUF = 32 << 20   # 32MB SO_SNDBUF/SO_RCVBUF: a ~24MB prefill-chunk activation buffers in-kernel,
                      # so a stage's forward send returns without waiting for the next stage to drain it.
SYNC_SEND = bool(os.environ.get("SHARD_SYNC_SEND"))   # force the OLD synchronous forward send (A/B baseline)


class _AsyncSender:
    """Decouple a stage's compute from the synchronous inter-stage WAN send. At PREFILL the forward
    activation is ~24MB/chunk; with a plain send_msg the stage blocks until the next stage drains it
    (the socket buffer fills), so pipelined prefill collapses at long context -- one stage stalls the
    whole chain (the 1.17Ã—@110k handoff wall). This pushes the send to a background thread draining a
    FIFO queue: the compute thread enqueues and immediately processes the next chunk, so stages truly
    OVERLAP and TTFT approaches the m/(m+p-1) pipeline ceiling. FIFO order is preserved, so the tail's
    results still return to the coordinator in send order. DIRECT-return only (the stage never reads
    back on this socket). A send error is captured and re-raised on the next put(), so the existing
    `except EDGE_ERRORS` edge supervision resets the link exactly as before."""
    def __init__(self, sock, telemetry=None):
        self.sock = sock
        self.telemetry = telemetry
        self.q = queue.Queue(maxsize=64)
        self.error = None
        self.closed = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self.closed.is_set():
            try:
                obj = self.q.get(timeout=0.25)   # bounded wait: a lost wake can never strand the worker
            except queue.Empty:
                continue
            if obj is None:
                return
            if self.error is not None:
                continue                       # link already failed; drain+discard while the loop tears down
            try:
                send_msg(self.sock, obj)
            except Exception as e:             # surfaced to the serve loop on the next put() -> edge reset
                self.error = e

    def put(self, obj):
        if self.closed.is_set():
            raise RuntimeError("_AsyncSender closed")
        if self.error is not None:
            raise self.error
        started = time.perf_counter()
        self.q.put(prepare_message(self.sock, obj))
        if self.telemetry:
            self.telemetry.record("queue_wait_ms", (time.perf_counter() - started) * 1000)

    def close(self, timeout=5.0):
        """Idempotent shutdown. The EVENT stops the worker, not a droppable sentinel: the old
        put_nowait(None) silently lost the sentinel when the queue was FULL (exactly the wedged-link
        case close() runs in), stranding the daemon worker forever. Queued frames are discarded --
        close() only runs on edge teardown, where they're already dead. The join is BOUNDED so a
        send stalled inside the kernel can never wedge the serve loop's reset path."""
        self.closed.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except (OSError, AttributeError):
            pass
        try: self.q.put_nowait(None)           # fast wake if there's room; the event is the guarantee
        except queue.Full: pass
        self.t.join(timeout)


def serve_spec(parts, stage, nstages, listen_port, nxt, timeout, dev, direct=False):
    """a non-head stage under speculative decoding. the op is `verify`: run this
    block on the K+1 proposed tokens, relay forward, and (at the tail) return the
    argmax for every position. `crop` rolls this node's cache back before running.
    direct=True: forward-only -- don't relay the result back up the chain (the tail
    sends it straight to the coordinator). only non-tail stages use this path; the
    direct tail is serve_tail_direct."""
    is_tail = stage == nstages - 1
    nxt_sock = None
    if not is_tail:
        nxt_sock = _forward_connect(parts, nxt, timeout)
        print(f"[s{stage}] connected forward to stage {stage+1} at {nxt}", flush=True)
    srv, acceptor = _server(parts, listen_port, timeout)
    print(f"[s{stage}] listening on :{listen_port} (edge timeout {timeout:.0f}s)", flush=True)
    while True:
        conn, addr = _accept_stage(srv, acceptor, parts, timeout)
        print(f"[s{stage}] stage {stage-1} connected from {addr}", flush=True)
        cache = DynamicCache(); verifies = 0
        with torch.no_grad():
            while True:
                try:
                    msg = recv_msg(conn)
                    if msg["op"] == "reset":
                        cache = DynamicCache()
                        if nxt_sock:
                            send_msg(nxt_sock, msg)
                            if not direct: recv_msg(nxt_sock)     # wait downstream ack (relay only)
                        if not direct: send_msg(conn, "ok")       # ack predecessor (relay only)
                        continue
                    if msg.get("op") == "abort":
                        # P2-1 Early Abort: skip computing stale in-flight frames on divergence
                        if nxt_sock:
                            send_msg(nxt_sock, msg)
                        continue
                    if msg.get("gather") is not None:      # tree: keep last round's accepted-path KV
                        gather_cache(cache, msg["gather"], dev)
                    elif msg.get("crop") is not None:      # linear: roll back the prior round's rejects
                        cache.crop(msg["crop"])
                    par, dep = msg.get("par"), msg.get("dep")
                    if "token_ids" in msg:                 # served head: coordinator sent token ids, embed here
                        x = parts["embed"](torch.tensor([msg["token_ids"]], device=dev))
                        h = run_block(x, parts, cache, msg["start"], par=par, dep=dep)
                    else:
                        h = run_block(msg["h"].to(dev), parts, cache, msg["start"], par=par, dep=dep)
                    if is_tail:                            # relay tail (non-direct)
                        h = parts["norm"](h)
                        toks = parts["lm_head"](h).argmax(-1)[0].tolist()
                        send_msg(conn, toks)
                    else:
                        send_msg(nxt_sock, {"op": "verify", "h": h.cpu(), "start": msg["start"],
                                            "crop": msg.get("crop"), "gather": msg.get("gather"),
                                            "par": par, "dep": dep})
                        if not direct:                     # relay the result back up the chain
                            send_msg(conn, recv_msg(nxt_sock))
                    verifies += 1
                except EDGE_ERRORS as e:
                    why = "stalled" if isinstance(e, socket.timeout) else "closed"
                    print(f"[s{stage}] edge {why} after {verifies} verifies ({type(e).__name__}); resetting", flush=True)
                    try: conn.close()
                    except OSError: pass
                    break


def serve_tail_direct(parts, listen_port, timeout, dev):
    """tail with DIRECT return: the result goes straight to the coordinator, not
    relayed up the chain. two connections arrive on the listen port -- the
    predecessor (activations) and the coordinator's return channel (which sends a
    {op:hello_return} on connect). select tells them apart (only the return channel
    has a message waiting; the predecessor is idle until driven). each verify's
    result is sent on the return channel."""
    import select
    srv, acceptor = _server(parts, listen_port, timeout)
    print(f"[tail] listening on :{listen_port} (predecessor + coordinator return, edge timeout {timeout:.0f}s)", flush=True)
    while True:
        if acceptor is not None:
            pred_conn = acceptor.get("drive" if parts["_session_config"].index == 0 else "forward")
            ret_conn = acceptor.get("return")
            c1, c2 = pred_conn, ret_conn
        else:
            c1, _ = srv.accept(); c2, _ = srv.accept()
            c1.settimeout(min(timeout, 5)); c2.settimeout(min(timeout, 5))
            ready, _, _ = select.select([c1, c2], [], [], timeout)
            if not ready:
                print("[tail] no return-channel handshake; resetting", flush=True)
                c1.close(); c2.close(); continue
            ret_conn = ready[0]
            try:
                hello = recv_msg(ret_conn)
            except EDGE_ERRORS:
                c1.close(); c2.close(); continue
            if not (isinstance(hello, dict) and hello.get("op") == "hello_return"):
                print("[tail] unexpected handshake; resetting", flush=True)
                c1.close(); c2.close(); continue
            pred_conn = c2 if ret_conn is c1 else c1
        pred_conn.settimeout(timeout)
        print("[tail] predecessor + coordinator-return connected", flush=True)
        cache = DynamicCache(); verifies = 0
        with torch.no_grad():
            while True:
                try:
                    msg = recv_msg(pred_conn)
                    if msg["op"] == "reset":
                        cache = DynamicCache(); send_msg(ret_conn, "ok"); continue
                    if msg.get("gather") is not None:
                        gather_cache(cache, msg["gather"], dev)
                    elif msg.get("crop") is not None:
                        cache.crop(msg["crop"])
                    h = run_block(msg["h"].to(dev), parts, cache, msg["start"],
                                  par=msg.get("par"), dep=msg.get("dep"))
                    h = parts["norm"](h)
                    toks = parts["lm_head"](h).argmax(-1)[0].tolist()
                    send_msg(ret_conn, toks); verifies += 1
                except EDGE_ERRORS as e:
                    print(f"[tail] edge after {verifies} verifies ({type(e).__name__}); resetting", flush=True)
                    try: pred_conn.close(); ret_conn.close()
                    except OSError: pass
                    break


def serve_spec_fast(parts, stage, nstages, listen_port, nxt, timeout, dev, direct=False, max_ctx=2048):
    """serve_spec with the FAST verify: a static-cache CUDA-graph stage forward (~5x
    cheaper than eager). LINEAR spec only (the graph is a fixed K+1 shape; tree is
    variable). first verify after reset = prefill (eager, prompt-length); every later
    verify = a decode round (graphed). rollback is implicit -- a round writes at `start`
    (the committed length), overwriting the prior round's rejects."""
    is_tail = stage == nstages - 1
    nxt_sock = None; sender = None
    host = port = None
    if not is_tail:
        host, port = _address(nxt)

    def mk_fwd():                                            # (re)build the forward link (+ async sender in direct mode)
        nonlocal nxt_sock, sender, host, port
        # HOT-HEAL: on a relink, repoint the forward link to a pre-warmed SPARE *without reloading weights*.
        # The control-plane healer writes "<host>:<port>" to /root/.shard_next_<stage> on the victim's
        # PREDECESSOR; this is re-read every relink attempt, so the dropped link reconnects to the spare
        # (warm, weights already in VRAM) instead of the dead victim. Absent file => use the launch --next.
        try:
            ov = open(f"/root/.shard_next_{stage}").read().strip()
            if ov:
                host, port = _address(ov)
        except OSError:
            pass
        target = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
        s = _forward_connect(parts, target, timeout)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCK_BUF)   # buffer a full ~24MB prefill chunk in-kernel
        nxt_sock = s
        # async send: decoupled forward (no read-back in direct mode). SHARD_SYNC_SEND=1 forces the old
        # synchronous path -> the clean A/B baseline (reproduces last session's handoff-bound 193s@110k).
        sender = _AsyncSender(s, parts.get("_telemetry")) if (direct and not SYNC_SEND) else None

    def fwd_send(o):                                         # async in direct mode, synchronous otherwise
        if direct and sender is not None:
            sender.put(o)
        else:
            send_msg(nxt_sock, o)

    if not is_tail:
        mk_fwd()
        print(f"[s{stage}] connected forward to stage {stage+1} at {nxt}", flush=True)
    srv, acceptor = _server(parts, listen_port, timeout)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCK_BUF)
    fv = _guard_fast_verify(FastVerify(parts, maxlen=max_ctx, dev=dev), parts)
    sampler = Sampler(device=dev)                         # greedy until a reset sets temp>0 (tail-relay path)
    node_key = load_or_make_node_key(NODE_KEY_PATH) if RECEIPTS else None
    signer = None
    print(f"[s{stage}] listening on :{listen_port} (FAST verify, max_ctx={max_ctx}, edge timeout {timeout:.0f}s"
          f"{', signed receipts ON' if RECEIPTS else ''})", flush=True)
    while True:
        if not is_tail and nxt_sock is None:                 # forward link dropped (coordinator churn) -> rebuild it;
            for _ in range(60):                              # closing it made the NEXT stage drop its link too, so the
                try:                                         # whole ring re-handshakes fresh and a new coordinator can drive it
                    mk_fwd(); break
                except OSError: time.sleep(0.5)
            print(f"[s{stage}] forward link rebuilt -> {nxt}" if nxt_sock else f"[s{stage}] relink FAILED", flush=True)
        conn, addr = _accept_stage(srv, acceptor, parts, timeout)
        print(f"[s{stage}] stage {stage-1} connected from {addr}", flush=True)
        fv.reset(); first = True; verifies = 0
        with torch.no_grad():
            while True:
                msg = None                                    # bind before recv: a partial/corrupt frame (e.g. a
                try:                                          # peer resetting mid-message during a heal) must hit the
                    msg = recv_msg(conn)                      # 'bad msg -> reset' path below, not crash on unbound msg

                    if msg["op"] == "reset":
                        fv.reset(); first = True
                        sampler = Sampler(temp=msg.get("temp", 0.0), top_p=msg.get("top_p", 1.0),
                                          top_k=msg.get("top_k", 0), seed=msg.get("seed", 0), device=dev)
                        if RECEIPTS:                     # start this job's per-stage activation hash-chain
                            signer = ReceiptSigner(node_key, msg.get("swarm_id", "swarm"),
                                                   msg.get("job_id", "job"), parts["lo"], parts["hi"], nonce=msg.get("nonce"))
                        if nxt_sock:
                            fwd_send(msg)
                            if not direct: recv_msg(nxt_sock)
                        if not direct: send_msg(conn, "ok")
                        continue
                    if msg.get("op") == "abort":
                        # P2-1: Early abort — skip computing stale divergence frames
                        if nxt_sock:
                            fwd_send(msg)
                        continue
                    if msg["op"] == "receipt":           # job done: sign + accumulate down the ring
                        if RECEIPTS and signer is not None:
                            msg.setdefault("receipts", []).append({"stage": stage, **signer.finalize()})
                        if nxt_sock:
                            fwd_send(msg)                 # tail returns the full list to the coordinator (direct)
                            if not direct: send_msg(conn, recv_msg(nxt_sock))
                        else:
                            send_msg(conn, msg.get("receipts", []))
                        continue
                    g = msg.get("gather")
                    if g:                                  # lazy: compact prev tree's accepted path KV
                        fv.tree_gather(g[0], g[1])
                    if "token_ids" in msg:                 # served head: embed ids here
                        x = parts["embed"](torch.tensor([msg["token_ids"]], device=dev))
                    else:
                        x = msg["h"].to(dev)
                    is_pf = ("par" not in msg) and (first or msg.get("prefill"))
                    draft = msg.get("draft")               # the K proposed tokens, for the tail's sampler
                    if draft is None and "token_ids" in msg and not is_pf and "par" not in msg:
                        draft = msg["token_ids"][1:]       # head derives them: chunk = [carry] + K drafts
                    if "par" in msg:                       # TREE verify (fixed-topology graph)
                        h = fv.tree_decode(x, msg["start"], msg["par"], msg["dep"])
                    else:                                  # LINEAR verify (prefill, then graphed decode)
                        h = fv.prefill(x, msg["start"]) if is_pf else fv.decode(x, msg["start"]); first = False
                        if RECEIPTS and signer is not None:   # attest this block's input->output transform
                            signer.observe(_act_digest(x), _act_digest(h))
                    if is_tail:
                        # prefill: only the last token's logit is consumed -> avoid a [chunk x vocab] OOM
                        h = parts["norm"](h[:, -1:] if is_pf else h)
                        logits = parts["lm_head"](h)
                        if is_pf:                          # prefill: the single first new token (greedy=argmax)
                            send_msg(conn, [sampler.sample_logits(logits[0, -1])])
                        elif draft is None:                # tree / no-draft decode: per-position (greedy=argmax)
                            send_msg(conn, logits.argmax(-1)[0].tolist() if sampler.greedy
                                     else [sampler.sample_logits(logits[0, i]) for i in range(logits.shape[1])])
                        else:                              # lossless speculative sampling over the K+1 logits
                            send_msg(conn, sampler.accept(logits[0], draft))
                    else:
                        fwd = {"op": "verify", "h": h.cpu(), "start": msg["start"], "prefill": msg.get("prefill")}
                        if "par" in msg: fwd["par"] = msg["par"]; fwd["dep"] = msg["dep"]
                        if draft is not None: fwd["draft"] = draft
                        if g: fwd["gather"] = g
                        fwd_send(fwd)
                        if not direct:
                            send_msg(conn, recv_msg(nxt_sock))
                    verifies += 1
                except EDGE_ERRORS as e:
                    why = "stalled" if isinstance(e, socket.timeout) else "closed"
                    print(f"[s{stage}] edge {why} after {verifies} verifies ({type(e).__name__}); resetting", flush=True)
                    try: conn.close()
                    except OSError: pass
                    if sender is not None:                    # stop the async sender thread before dropping the link
                        sender.close(); sender = None
                    if nxt_sock is not None:                  # drop forward link -> ring re-handshakes fresh on re-accept
                        try: nxt_sock.close()
                        except OSError: pass
                        nxt_sock = None
                    break
                except Exception as e:                       # survive a bad message instead of dying
                    k = list(msg.keys()) if isinstance(msg, dict) else "?"
                    print(f"[s{stage}] bad msg after {verifies} verifies ({type(e).__name__}: {str(e)[:80]} keys={k}); resetting", flush=True)
                    try: conn.close()
                    except OSError: pass
                    if sender is not None:
                        sender.close(); sender = None
                    if nxt_sock is not None:
                        try: nxt_sock.close()
                        except OSError: pass
                        nxt_sock = None
                    break


def serve_tail_fast(parts, listen_port, timeout, dev, max_ctx=2048):
    """direct-return tail, RESILIENT to coordinator churn. the predecessor (the ring's
    forward chain) and the coordinator-return channel have INDEPENDENT lifecycles:
      - return channel drops (coordinator restart / gateway reconnect / retry) -> keep
        the predecessor + KV, re-accept a new return channel. the ring is NOT poisoned.
      - predecessor drops (an upstream stage reconnected) -> re-accept it, keep the return.
    a connection is identified by CONTENT: the coordinator sends {"op":"hello_return"} on
    its return channel; anything else is the predecessor (and its first message is queued).
    this is what lets the gateway connect, fail, retry, and restart without relaunching the
    swarm -- the c0mpute come-and-go property at the tail."""
    srv, acceptor = _server(parts, listen_port, timeout)
    fv = _guard_fast_verify(FastVerify(parts, maxlen=max_ctx, dev=dev), parts)
    sampler = Sampler(device=dev)                         # greedy until a reset sets temp>0
    node_key = load_or_make_node_key(NODE_KEY_PATH) if RECEIPTS else None
    signer = None
    print(f"[tail] listening on :{listen_port} (FAST verify, max_ctx={max_ctx}, direct return, edge timeout {timeout:.0f}s"
          f"{', signed receipts ON' if RECEIPTS else ''})", flush=True)
    pred = ret = None; pending = None; first = True

    class TailSessionFenced(ProtocolError):
        pass
    def same_owner(left, right):
        return left is not None and right is not None and all(left.get(key) == right.get(key)
            for key in ("owner", "session_id", "fence", "boot_id"))
    def ret_send(payload):
        nonlocal ret
        if acceptor is None:
            return send_msg(ret, payload)
        desired = getattr(pred, "grant", None)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with acceptor.lock:
                latest = acceptor.slots.get("return")
            if latest is not None and not latest.closed:
                if not same_owner(desired, latest.grant):
                    raise TailSessionFenced("pending result belongs to a fenced coordinator owner")
                ret = latest
                try:
                    return send_msg(ret, payload)
                except OSError:
                    # A same-owner reconnect may already be accepted. Retry the
                    # SAME ack/result; never silently drop a reset barrier.
                    ret.close(); ret = None
            time.sleep(0.01)
        raise TailSessionFenced("matching coordinator return channel did not recover before its deadline")

    def fill():                                            # block until BOTH channels are present
        nonlocal pred, ret, pending
        while pred is None or ret is None:
            if acceptor is not None:
                if pred is None:
                    pred = acceptor.get("drive" if parts["_session_config"].index == 0 else "forward")
                if ret is None:
                    ret = acceptor.get("return")
                continue
            c, _ = srv.accept()
            c.settimeout(min(timeout, 5.0))
            try:
                m = recv_msg(c)
            except EDGE_ERRORS:
                try: c.close()
                except OSError: pass
                continue
            if isinstance(m, dict) and m.get("op") == "hello_return":
                if ret is not None:
                    try: ret.close()
                    except OSError: pass
                ret = c; ret.settimeout(timeout)
                print("[tail] coordinator-return (re)connected", flush=True)
            else:                                          # predecessor: its first msg is real, queue it
                if pred is not None:
                    try: pred.close()
                    except OSError: pass
                pred = c; pred.settimeout(timeout); pending = m
                print("[tail] predecessor (re)connected", flush=True)

    with torch.no_grad():
        while True:
            fill()
            try:
                msg = pending if pending is not None else recv_msg(pred)
                pending = None
            except EDGE_ERRORS as e:                       # predecessor gone -> usually a coordinator churn, which
                print(f"[tail] predecessor edge ({type(e).__name__}); re-accepting predecessor + return", flush=True)
                old_grant = getattr(pred, "grant", None)
                try: pred.close()
                except OSError: pass
                pred = None
                if ret is not None and (acceptor is None or same_owner(old_grant, getattr(ret, "grant", None))):
                    try: ret.close()                       # fresh return channel, so the reset's 'ok' must not be sent to
                    except OSError: pass                   # the dead old one (the race that made churn recovery flaky)
                    ret = None
                continue
            try:
                if msg.get("op") == "stop":
                    if acceptor is not None:
                        acceptor.close()
                    else:
                        srv.close()
                    return
                if msg["op"] == "reset":
                    fv.reset(); first = True
                    sampler = Sampler(temp=msg.get("temp", 0.0), top_p=msg.get("top_p", 1.0),
                                      top_k=msg.get("top_k", 0), seed=msg.get("seed", 0), device=dev)
                    if RECEIPTS:
                        signer = ReceiptSigner(node_key, msg.get("swarm_id", "swarm"),
                                               msg.get("job_id", "job"), parts["lo"], parts["hi"], nonce=msg.get("nonce"))
                    ret_send("ok"); continue
                if msg.get("op") == "abort":
                    # P2-1 Early Abort: Tail skips computing LM Head for aborted round
                    continue
                if msg["op"] == "receipt":                  # job done: sign + return the full ring's receipts
                    if RECEIPTS and signer is not None:
                        msg.setdefault("receipts", []).append({"stage": "tail", **signer.finalize()})
                    ret_send(msg.get("receipts", []))
                    continue
                g = msg.get("gather")
                if g:
                    fv.tree_gather(g[0], g[1])
                x = (parts["embed"](torch.tensor([msg["token_ids"]], device=dev))
                     if "token_ids" in msg and "embed" in parts else msg["h"].to(dev))
                is_pf = ("par" not in msg) and (first or msg.get("prefill"))
                draft = msg.get("draft")                    # the K proposed tokens (for the sampler's accept)
                if draft is None and "token_ids" in msg and not is_pf and "par" not in msg:
                    draft = msg["token_ids"][1:]
                if "par" in msg:                           # TREE verify
                    h = fv.tree_decode(x, msg["start"], msg["par"], msg["dep"])
                else:
                    h = fv.prefill(x, msg["start"]) if is_pf else fv.decode(x, msg["start"]); first = False
                    if RECEIPTS and signer is not None:    # attest this block's input->output transform
                        signer.observe(_act_digest(x), _act_digest(h))
                    # prefill: the coordinator only consumes the LAST token (next-token after the chunk),
                    # so run lm_head on just that position -- a full [chunk x vocab] logit tensor is ~1.5GB
                    # at chunk 4096 and OOMs a 24GB tail at long context. decode needs all K+1 logits.
                    if is_pf:
                        h = h[:, -1:]
                logits = parts["lm_head"](parts["norm"](h))
                if is_pf:                                  # prefill: the single first new token (greedy=argmax)
                    ret_send([sampler.sample_logits(logits[0, -1])])
                elif draft is None:                        # tree / no-draft decode: per-position (greedy=argmax)
                    ret_send(logits.argmax(-1)[0].tolist() if sampler.greedy
                             else [sampler.sample_logits(logits[0, i]) for i in range(logits.shape[1])])
                else:                                      # lossless speculative SAMPLING over the K+1 logits
                    ret_send(sampler.accept(logits[0], draft))
            except TailSessionFenced as error:
                print("ERROR " + json.dumps({"code": "session_fenced", "stage": "tail",
                    "session_id": getattr(pred, "grant", {}).get("session_id"), "message": str(error)}), flush=True)
                signer = None; first = True
                # Keep the static forward connection: the new owner's reset is
                # already ordered behind old frames. Its return channel stays live.
                continue
            except EDGE_ERRORS as e:                        # return channel gone -> re-accept it, keep pred + KV
                print(f"[tail] return edge ({type(e).__name__}); dropping return channel, keeping predecessor+KV", flush=True)
                try: ret.close()
                except OSError: pass
                ret = None; continue
            except Exception as e:                          # bad message -> drop the return channel, survive
                k = list(msg.keys()) if isinstance(msg, dict) else "?"
                print(f"[tail] bad msg ({type(e).__name__}: {str(e)[:80]} keys={k}); dropping return channel", flush=True)
                try: ret.close()
                except OSError: pass
                ret = None; continue


def generate_spec(draft, parts, tok, sock, prompt, K, max_new, dev, draft_dev, timeout,
                  adaptive=False, k_min=1, k_max=12, draft_sock=None):
    """stage 0: draft proposes K tokens on its own GPU; the distributed target
    verifies [cur, d_1..d_K] in one chain traversal; greedy-accept the longest
    matching prefix. caches (draft + this node's block) crop locally; downstream
    nodes crop lazily on the next verify."""
    sock.settimeout(timeout)
    eos = tok.eos_token_id
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(dev)
    prompt_ids = enc["input_ids"][0].tolist()   # for the in-house draft service (full-prefix queries)
    head_cache, draft_cache = DynamicCache(), DynamicCache()
    pos = 0
    out = []                                    # defined before any edge can fail (prefill incl.)

    def embed(tokens):                          # python list -> this node's block output
        x = torch.tensor([tokens], device=dev)
        return run_block(parts["embed"](x), parts, head_cache, pos)

    try:
        send_msg(sock, {"op": "reset"}); recv_msg(sock)
        # ---- prefill: target processes the whole prompt; draft fills its cache ----
        h = run_block(parts["embed"](ids), parts, head_cache, 0)
        preds = (send_msg(sock, {"op": "verify", "h": h.cpu(), "start": 0}), recv_msg(sock))[1]
        cur = preds[-1]                                     # target's token for position L
        pos = ids.shape[1]
        if draft_sock is None:                              # local transformers draft fills its cache
            with torch.no_grad():
                draft(input_ids=ids.to(draft_dev), past_key_values=draft_cache, use_cache=True)

        out = [cur]
        rounds, accepted_total = 0, 0
        kc, ema_n, k_hist = K, float(K), []
        tail_crop = None                                   # lazy downstream rollback, piggybacked
        t_draft = t_verify = 0.0                            # round-budget instrumentation
        t0 = time.time()
        with torch.no_grad():
            while len(out) < max_new and cur != eos:
                # 1. draft proposes kc tokens
                td = time.time()
                if draft_sock is not None:                  # in-house vLLM draft service (full prefix; prefix-cached)
                    send_msg(draft_sock, {"ids": prompt_ids + out, "k": kc})
                    drafts = recv_msg(draft_sock)
                else:                                       # local transformers draft (incremental cache)
                    drafts, dtok = [], cur
                    for i in range(kc + 1):
                        dl = draft(input_ids=torch.tensor([[dtok]], device=draft_dev),
                                   past_key_values=draft_cache, use_cache=True).logits
                        dtok = int(dl[0, -1].argmax())
                        if i < kc:
                            drafts.append(dtok)
                t_draft += time.time() - td
                # 2. verify [cur, d_1..d_kc] in one traversal (carry the prior round's rollback)
                tv = time.time()
                h = embed([cur] + drafts)
                send_msg(sock, {"op": "verify", "h": h.cpu(), "start": pos, "crop": tail_crop})
                r = recv_msg(sock)
                t_verify += time.time() - tv
                # 3. greedy acceptance: longest prefix with d_j == r_j
                n = 0
                for j in range(kc):
                    if drafts[j] == r[j]:
                        n += 1
                    else:
                        break
                committed = drafts[:n] + [r[n]]            # n accepted + 1 correction
                out.extend(committed)
                cur = r[n]
                pos += n + 1
                rounds += 1; accepted_total += n; k_hist.append(kc)
                # 4. roll caches back to the accepted length: this node's + draft's now,
                #    downstream nodes lazily on the next verify (no extra round-trip)
                head_cache.crop(pos)
                if draft_sock is None:
                    draft_cache.crop(pos)                   # vLLM service manages its own cache
                tail_crop = pos
                # 5. adaptive K: aim a couple beyond the running acceptance (EMA of n)
                ema_n = 0.7 * ema_n + 0.3 * n
                if adaptive:
                    kc = max(k_min, min(k_max, round(ema_n) + 2))
                if eos in committed:
                    break
    except EDGE_ERRORS as e:
        raise TransportError(f"pipeline edge failed at token {len(out)} ({type(e).__name__}: {e})") from e

    dt = time.time() - t0
    if eos in out:
        out = out[:out.index(eos)]
    return {
        "text": tok.decode(out, skip_special_tokens=True),
        "n_tokens": len(out), "rounds": rounds,
        "mean_accept": accepted_total / max(rounds, 1),
        "toks_per_traversal": (accepted_total + rounds) / max(rounds, 1),
        "tok_s": len(out) / max(dt, 1e-9),
        "mean_K": (sum(k_hist) / len(k_hist)) if k_hist else K,
        "k_lo": min(k_hist) if k_hist else K, "k_hi": max(k_hist) if k_hist else K,
        "draft_ms": t_draft / max(rounds, 1) * 1000, "verify_ms": t_verify / max(rounds, 1) * 1000,
        "output_ids": out,
    }


def coordinate(draft_sock, pipe_sock, tok, prompt, K, max_new, timeout,
               adaptive=False, k_min=1, k_max=12, ret_sock=None, prefill_chunk=0):
    """the in-house coordinator (c0mpute entry node): holds NO 120B layers. it
    tokenizes, queries the in-house draft for K tokens, sends token ids into the
    swarm's stage 0 (which embeds + runs), reads back the verify, greedy-accepts.
    the whole 120B lives on the scattered swarm nodes; this node is just the entry
    point plus the managed draft. lazy crop propagates to every swarm node.
    ret_sock set => DIRECT return: send forward to stage 0, receive the verify
    result straight from the tail (1 hop) instead of relayed back up the chain."""
    pipe_sock.settimeout(timeout)
    rx = ret_sock if ret_sock is not None else pipe_sock     # where results come back (direct => tail)
    eos = tok.eos_token_id
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    prompt_ids = enc["input_ids"][0].tolist()
    out = []
    try:
        send_msg(pipe_sock, {"op": "reset"}); recv_msg(rx)
        # prefill: one shot, or CHUNKED (long prompts) so each stage's per-chunk activation
        # stays bounded (the KV cache accumulates). flex_attention keeps each chunk O(n) on Ada.
        if prefill_chunk and len(prompt_ids) > prefill_chunk:
            r = None
            for i in range(0, len(prompt_ids), prefill_chunk):
                send_msg(pipe_sock, {"op": "verify", "token_ids": prompt_ids[i:i + prefill_chunk], "start": i})
                r = recv_msg(rx)
            cur = r[-1]
        else:
            send_msg(pipe_sock, {"op": "verify", "token_ids": prompt_ids, "start": 0})   # prefill
            cur = recv_msg(rx)[-1]
        pos = len(prompt_ids)
        out = [cur]
        rounds, accepted_total = 0, 0
        kc, ema_n, k_hist = K, float(K), []
        tail_crop = None
        t_draft = t_verify = 0.0
        t0 = time.time()
        while len(out) < max_new and cur != eos:
            td = time.time()
            send_msg(draft_sock, {"ids": prompt_ids + out, "k": kc}); drafts = recv_msg(draft_sock)
            t_draft += time.time() - td
            tv = time.time()
            send_msg(pipe_sock, {"op": "verify", "token_ids": [cur] + drafts, "start": pos, "crop": tail_crop})
            r = recv_msg(rx)
            t_verify += time.time() - tv
            n = 0
            for j in range(kc):
                if drafts[j] == r[j]: n += 1
                else: break
            committed = drafts[:n] + [r[n]]
            out.extend(committed); cur = r[n]; pos += n + 1
            rounds += 1; accepted_total += n; k_hist.append(kc); tail_crop = pos
            ema_n = 0.7 * ema_n + 0.3 * n
            if adaptive:
                kc = max(k_min, min(k_max, round(ema_n) + 2))
            if eos in committed:
                break
    except EDGE_ERRORS as e:
        raise TransportError(f"pipeline edge failed at token {len(out)} ({type(e).__name__}: {e})") from e
    dt = time.time() - t0
    if eos in out:
        out = out[:out.index(eos)]
    return {
        "text": tok.decode(out, skip_special_tokens=True), "n_tokens": len(out), "rounds": rounds,
        "mean_accept": accepted_total / max(rounds, 1),
        "toks_per_traversal": (accepted_total + rounds) / max(rounds, 1),
        "tok_s": len(out) / max(dt, 1e-9),
        "mean_K": (sum(k_hist) / len(k_hist)) if k_hist else K,
        "k_lo": min(k_hist) if k_hist else K, "k_hi": max(k_hist) if k_hist else K,
        "draft_ms": t_draft / max(rounds, 1) * 1000, "verify_ms": t_verify / max(rounds, 1) * 1000,
        "output_ids": out,
    }


def coordinate_pipe(draft_sock, pipe_sock, tok, prompt, K, max_new, timeout, depth, ret_sock=None,
                    ignore_eos=False, prefill_chunk=0, draft_ctx=0, on_commit=None, reasoning=None, system=None,
                    max_ctx=0, local_draft=None, temp=0.0, top_p=1.0, top_k=0, seed=0, prefill_depth=8,
                    resume_ids=None, resumable=False, prompt_ids=None, cancel_check=None,
                    swarm_id="swarm", job_id="job", nonce=None, expected_by_signer=None,
                    strict_job_binding=False, adaptive_pipe=False, adaptive_depth=False):
    """Committed-frontier pipeline; adaptive switching requires a live greedy pilot gate.

    K=0 is a real one-token target path. Cross-K floating-point reassociation is
    model-dependent: opt-in pilots compare actual greedy streams on THIS prompt,
    and only passing buckets may run. This is a prefix gate, not a universal GPU
    numerical proof. Every switch drains old frames before warming its shape.
    """
    clock = time.perf_counter
    request_started = clock()
    if type(K) is not int or not 0 <= K <= 64 or type(depth) is not int or not 1 <= depth <= 32:
        raise ValueError("K must be in 0..64 and depth in 1..32")
    if type(max_new) is not int or max_new < 0:
        raise ValueError("max_new must be a nonnegative integer")
    if strict_job_binding and (not nonce or not expected_by_signer or not RECEIPTS):
        raise ValueError("strict jobs require a nonce, pinned assignments and SHARD_RECEIPTS=1")
    pipe_sock.settimeout(timeout)
    rx = ret_sock if ret_sock is not None else pipe_sock
    if hasattr(rx, "settimeout"):
        rx.settimeout(timeout)
    def check():
        if cancel_check:
            cancel_check()
    recv_wait = draft_wait = drain_s = 0.0
    def receive():
        nonlocal recv_wait
        check(); started = clock()
        result = recv_msg(rx)
        recv_wait += clock() - started
        check()
        return result
    def transmit(payload):
        check(); send_msg(pipe_sock, payload); check()
    if prompt_ids is None:
        options = {"reasoning_effort": reasoning} if reasoning else {}
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                             **options)["input_ids"][0].tolist()
    prompt_ids, resume_ids = list(prompt_ids), list(resume_ids or [])
    if not prompt_ids or any(type(token) is not int or token < 0 for token in prompt_ids + resume_ids):
        raise ValueError("valid nonempty tokenized prompt required")
    eos = tok.eos_token_id
    if max_ctx:
        max_new = min(max_new, max(0, max_ctx - len(prompt_ids) - max(1, K * depth + 1)))
    resume_ids = resume_ids[:max_new]
    out = list(resume_ids)
    if adaptive_depth and temp != 0:
        raise ValueError("adaptive depth requires greedy decoding")
    gate = {"enabled": bool(adaptive_pipe or adaptive_depth), "mode": "mixed_K_full_control" if adaptive_pipe else "same_shape_depth",
            "scope": "current_prompt_greedy_prefix", "verified_buckets": [], "changes": []}
    gate_s = 0.0
    eligible = [K]
    golden = None
    class ShapeMismatch(ValueError):
        pass
    if adaptive_pipe:
        if temp != 0 or resume_ids:
            raise ValueError("adaptive pipeline requires greedy decoding and original-request replay")
        started = clock()
        candidates = sorted(set([0, 1, 2, K]))
        candidates = [value for value in candidates if value <= max(K, 1)]
        # A full K=0 control is deliberately paid for before publication. Prefix
        # pilots alone cannot prove future cross-K MoE numerics. Every actual
        # publication is checked against this complete greedy stream.
        control = coordinate_pipe(draft_sock, pipe_sock, tok, prompt, 0, max_new, timeout, 1,
            ret_sock=ret_sock, prefill_chunk=prefill_chunk, local_draft=local_draft,
            prompt_ids=prompt_ids, cancel_check=cancel_check, max_ctx=max_ctx, reasoning=reasoning,
            ignore_eos=ignore_eos, swarm_id=swarm_id, job_id=job_id + "/greedy-control", nonce=nonce)
        golden = control["output_ids"]
        pilots = {0: golden[:min(16, max_new)]}
        for bucket in candidates:
            if bucket == 0:
                continue
            check()
            pilots[bucket] = coordinate_pipe(draft_sock, pipe_sock, tok, prompt, bucket, min(16, max_new), timeout, 1,
                ret_sock=ret_sock, prefill_chunk=prefill_chunk, local_draft=local_draft, prompt_ids=prompt_ids,
                cancel_check=cancel_check, max_ctx=max_ctx, reasoning=reasoning, ignore_eos=ignore_eos,
                swarm_id=swarm_id, job_id=job_id + "/shape-gate", nonce=nonce,
                strict_job_binding=False, adaptive_pipe=False)["output_ids"]
        eligible = [bucket for bucket in candidates if pilots[bucket] == pilots[0]]
        gate.update(verified_buckets=eligible, pilot_tokens=len(pilots[0]),
                    validation="full_request_greedy_control_before_publish", control_tokens=len(golden),
                    declined_buckets=[bucket for bucket in candidates if bucket not in eligible])
        gate_s = clock() - started
        K = K if K in eligible else 0
        depth = depth if K else 1
    current_k, current_depth = K, depth if K else 1
    use_local, pending_draft = local_draft is not None, False
    def draft_request(ids):
        nonlocal pending_draft
        if current_k == 0:
            return
        check()
        ids = ids[-draft_ctx:] if draft_ctx else ids
        if use_local:
            local_draft.request(ids, current_k)
        else:
            send_msg(draft_sock, {"ids": ids, "k": current_k})
        pending_draft = True
    def draft_fetch():
        nonlocal pending_draft, draft_wait
        if not pending_draft:
            return []
        check(); started = clock()
        result = local_draft.fetch() if use_local else recv_msg(draft_sock)
        draft_wait += clock() - started
        pending_draft = False
        check()
        if not isinstance(result, list) or len(result) != current_k:
            raise ValueError("drafter returned the wrong bucket length")
        return result
    first_commit = last_commit = None
    new_count = committed_decode = valid = accepted = proposed_total = stale = sent = warmups = 0
    recent_accepted = recent_proposed = 0
    prefill_s = 0.0
    inflight = []
    def publish(tokens, phase):
        nonlocal first_commit, last_commit, new_count
        available = max(0, max_new - len(out))
        tokens = list(tokens)[:available]
        finished = len(tokens) >= available
        if not ignore_eos and eos in tokens:
            tokens = tokens[:tokens.index(eos)]; finished = True
        if golden is not None and out + tokens != golden[:len(out) + len(tokens)]:
            raise ShapeMismatch("cross-bucket execution changed the canonical greedy frontier")
        if tokens:
            out.extend(tokens)
            now = clock()
            first_commit = now if first_commit is None else first_commit
            last_commit = now; new_count += len(tokens)
        if on_commit:
            on_commit({"phase": phase, "out": list(out), "dt": clock() - request_started,
                       "prompt_tokens": len(prompt_ids), "resume_tokens": len(resume_ids), "prefill_s": prefill_s})
        return len(tokens), finished
    try:
        if len(out) < max_new:
            transmit({"op": "reset", "temp": temp, "top_p": top_p, "top_k": top_k, "seed": seed,
                      "swarm_id": swarm_id, "job_id": job_id, "nonce": nonce})
            receive()
            start_pf = clock(); context = prompt_ids + resume_ids
            starts = list(range(0, len(context), prefill_chunk)) if prefill_chunk else [0]
            def prefill(index):
                stop = index + prefill_chunk if prefill_chunk else len(context)
                transmit({"op": "verify", "token_ids": context[index:stop], "start": index, "prefill": True})
            issued = min(max(1, prefill_depth), len(starts))
            for index in starts[:issued]:
                prefill(index)
            reply = None
            for _ in starts:
                reply = receive()
                if issued < len(starts):
                    prefill(starts[issued]); issued += 1
            prefill_s = clock() - start_pf
            cur = int(reply[-1]); pos = len(context)
            _, done = publish([cur], "prefilled")
            dprefix, send_pos, inflight, discard = prompt_ids + out, pos, [], 0
            if not done:
                draft_request(dprefix)
            while not done:
                while len(inflight) < current_depth:
                    proposed = draft_fetch()
                    transmit({"op": "verify", "token_ids": [dprefix[-1]] + proposed, "start": send_pos})
                    sent += 1
                    inflight.append((send_pos, proposed, current_k))
                    dprefix += proposed; send_pos += current_k
                    draft_request(dprefix)
                reply = receive(); _, proposed, bucket = inflight.pop(0)
                if discard:
                    discard -= 1; stale += 1; continue
                matched = 0
                for index in range(bucket):
                    if proposed[index] != reply[index]:
                        break
                    matched += 1
                valid += 1
                proposed_total += bucket
                # Full acceptance commits K draft tokens, never an invented K+1.
                committed = proposed if bucket and matched == bucket else proposed[:matched] + [int(reply[matched])]
                count, done = publish(committed, "decode")
                committed_decode += count; accepted += min(matched, count)
                recent_accepted += min(matched, count); recent_proposed += bucket
                if hasattr(local_draft, "note_accepted"):
                    local_draft.note_accepted(min(matched, count))
                pos += len(committed); cur = committed[-1]
                if matched != bucket or bucket == 0:
                    discard = len(inflight)
                    if discard:
                        transmit({"op": "abort", "discard": discard})
                    draft_fetch()
                    dprefix, send_pos = prompt_ids + out, pos
                    if not done:
                        draft_request(dprefix)
                if (adaptive_pipe or adaptive_depth) and valid % 6 == 0 and not done:
                    ratio = recent_accepted / max(1, recent_proposed)
                    recent_accepted = recent_proposed = 0
                    if adaptive_pipe:
                        target_k = 0 if ratio < .2 else max(eligible) if ratio > .65 else min(eligible, key=lambda k: abs(k - 1))
                        target_depth = 1 if target_k <= 1 else min(depth, 4)
                    else:
                        target_k = current_k
                        target_depth = 1 if ratio < .2 else depth if ratio >= .65 else current_depth
                    if (target_k, target_depth) != (current_k, current_depth):
                        began = clock()
                        while inflight:
                            receive(); inflight.pop(0); stale += 1
                        draft_fetch(); drain_s += clock() - began
                        gate["changes"].append({"round": valid, "K": target_k, "depth": target_depth,
                                                "boundary": "empty_pipeline", "budget": valid * current_k})
                        old_k = current_k
                        current_k, current_depth = target_k, target_depth
                        dprefix, send_pos, discard = prompt_ids + out, pos, 0
                        # Real shape warmup at the committed boundary; no token is
                        # published, and the same scratch KV is overwritten next.
                        if old_k != current_k:
                            padding = local_draft.propose(dprefix, current_k) if use_local and current_k else [cur] * current_k
                            transmit({"op": "verify", "token_ids": [cur] + padding, "start": pos})
                            receive(); warmups += 1
                        draft_request(dprefix)
            began = clock()
            draft_fetch()
            while inflight:
                receive(); inflight.pop(0); stale += 1
            drain_s += clock() - began
    except ShapeMismatch:
        # No mismatching token has reached a callback. Drain the already issued
        # packets, reset the original request, and replay the canonical plain
        # path while suppressing its already published prefix.
        while inflight:
            receive(); inflight.pop(0)
        draft_fetch()
        prefix = list(out)
        replay_seen = len(prefix)
        def replay(event):
            nonlocal replay_seen, new_count, first_commit, last_commit, committed_decode
            ids = event.get("out", [])
            if ids[:min(len(ids), len(prefix))] != prefix[:min(len(ids), len(prefix))]:
                raise ValueError("plain replay revised a published prefix")
            if ids != golden[:len(ids)]:
                raise ValueError("repeated plain path differs from the canonical greedy control")
            if len(ids) > replay_seen:
                new_count += len(ids) - replay_seen
                committed_decode += len(ids) - replay_seen
                last_commit = clock()
                if first_commit is None:
                    first_commit = last_commit
                replay_seen = len(ids)
            if on_commit and len(ids) >= len(prefix):
                on_commit(event)
        plain = coordinate_pipe(draft_sock, pipe_sock, tok, prompt, 0, max_new, timeout, 1,
            ret_sock=ret_sock, prefill_chunk=prefill_chunk, local_draft=local_draft,
            prompt_ids=prompt_ids, cancel_check=cancel_check, max_ctx=max_ctx, reasoning=reasoning,
            ignore_eos=ignore_eos, on_commit=replay, swarm_id=swarm_id, job_id=job_id, nonce=nonce,
            expected_by_signer=expected_by_signer, strict_job_binding=strict_job_binding)
        if plain["output_ids"] != golden:
            raise ValueError("repeated plain path changed the canonical greedy control")
        gate["fallback"] = "original_request_plain_replay_before_mismatch_publication"
        plain["adaptive"] = gate
        actual_decode_s = max(0, last_commit - first_commit) if first_commit is not None else 0.0
        plain["new_decode_tokens"] = max(0, new_count - 1)
        plain["decode_s"] = actual_decode_s
        plain["tok_s"] = plain["new_decode_tokens"] / actual_decode_s if actual_decode_s else 0.0
        plain["ttft_s"] = first_commit - request_started if first_commit is not None else None
        plain["request_s"] = clock() - request_started - plain["metrics"].get("receipt_sweep_s", 0.0)
        for name in ("new_decode_tokens", "decode_s", "ttft_s"):
            plain["metrics"][name] = plain[name]
        plain["metrics"]["request_s"] = plain["request_s"]
        plain["metrics"]["shape_gate_s"] = gate_s
        return plain
    except EDGE_ERRORS as error:
        check()
        if resumable:
            return {"ok": False, "error": f"{type(error).__name__}: {str(error)[:160]}",
                    "output_ids": list(out), "n_tokens": len(out), "text": tok.decode(out, skip_special_tokens=True)}
        raise TransportError(f"pipeline edge failed at token {len(out)} ({type(error).__name__}: {error})") from error
    request_s = clock() - request_started
    decode_s = max(0.0, last_commit - first_commit) if first_commit is not None else 0.0
    new_decode = max(0, new_count - 1)
    metrics = {"schema": "shard-pipeline-metrics/2", "committed_tokens": len(out), "new_tokens": new_count,
               "new_decode_tokens": new_decode, "decode_s": decode_s, "request_s": request_s,
               "ttft_s": first_commit - request_started if first_commit is not None else None,
               "prefill_s": prefill_s, "recv_wait_s": recv_wait, "draft_wait_s": draft_wait,
               "drain_s": drain_s, "shape_gate_s": gate_s, "valid_rounds": valid,
               "committed_decode_tokens": committed_decode, "stale_frames": stale,
               "sent_verify_frames": sent, "shape_warmup_frames": warmups,
               "mean_gain": committed_decode / max(valid, 1)}
    result = {"ok": True, "text": tok.decode(out, skip_special_tokens=True), "n_tokens": len(out),
              "output_ids": list(out), "rounds": valid, "mean_accept": accepted / max(valid, 1),
              "toks_per_traversal": metrics["mean_gain"], "tok_s": new_decode / decode_s if decode_s else 0.0,
              "wasted": stale, "depth": current_depth, "K": current_k,
              "draft_ms": draft_wait / max(valid, 1) * 1000, "recv_ms": recv_wait / max(valid, 1) * 1000,
              "prompt_tokens": len(prompt_ids), "resume_tokens": len(resume_ids), "adaptive": gate,
              "metrics": metrics, **{k: metrics[k] for k in ("new_decode_tokens", "decode_s", "request_s", "ttft_s", "committed_tokens", "prefill_s")}}
    if local_draft is not None:
        result["effective_config"] = {"ngram_n": getattr(local_draft, "ng", None),
            "margin_mode": "adaptive" if getattr(local_draft, "adaptive", False) else "fixed",
            "margin_cap": getattr(local_draft, "margin_cap", None),
            "effective_margin": local_draft.effective_margin(len(prompt_ids)) if hasattr(local_draft, "effective_margin") else None,
            "K": current_k, "depth": current_depth}
    if expected_by_signer is not None:
        receipt_started = clock()
        transmit({"op": "receipt", "receipts": []})
        raw = receive()
        from shard.receipt import wire_receipt, verify_coverage
        receipts = [wire_receipt(row) for row in raw]
        layers = max(span[1] for span in expected_by_signer.values())
        verify_coverage(receipts, layers, expected_by_signer=expected_by_signer, expected_nonce=nonce, check_chain=True)
        if any((row.get("swarm_id"), row.get("job_id"), row.get("nonce")) != (swarm_id, job_id, nonce) for row in receipts):
            raise ValueError("receipt belongs to another job/session")
        result.update(receipts=receipts, receipts_ok=True, proof_verified=True)
        result["metrics"]["receipt_sweep_s"] = clock() - receipt_started
    return result


def sample_distribution_test(pipe_sock, tok, prompt, timeout, ret_sock, local_draft, K, n_draws=2000,
                             temp=1.0, top_p=1.0, top_k=0, seed=0, prefill_chunk=0, reasoning=None,
                             stops=12, step=8, n_per=140):
    """On-swarm LOSSLESSNESS proof for speculative SAMPLING. At each of `stops` content positions, draw
    n_per iid samples of the next token THREE ways and compare the empirical distributions:
      PLAIN  — send a 1-token chunk (K=0 accept) -> a pure target sample from the temp/top-p dist.
      SPEC   — send [carry]+K n-gram drafts -> the speculative-sampling accept's committed first token.
      PLAIN2 — a second plain block -> calibrates the Monte-Carlo noise floor (two finite samples of the
               SAME distribution still differ by ~this much), so TV(spec,plain) is judged against it.
    Both PLAIN and SPEC must be distributed as p(next | context). Draws don't advance the KV (they
    overwrite scratch slots), so they're pipelined (depth chunks in flight) and fast; between stops we
    advance `step` tokens. We sweep positions so some land HIGH-ENTROPY (where the rejection/residual
    sampling actually does work and a distribution test is discriminating). Verdict aggregates over the
    high-entropy stops: if mean TV(spec,plain) ~ mean TV(plain,plain) noise floor, sampling is lossless
    on the real gpt-oss-120B distribution end-to-end through the WAN ring."""
    import collections, math
    pipe_sock.settimeout(timeout)
    rx = ret_sock if ret_sock is not None else pipe_sock
    ct_kw = {"reasoning_effort": reasoning} if reasoning else {}
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True, **ct_kw)
    prompt_ids = enc["input_ids"][0].tolist()
    send_msg(pipe_sock, {"op": "reset", "temp": temp, "top_p": top_p, "top_k": top_k, "seed": seed}); recv_msg(rx)
    if prefill_chunk and len(prompt_ids) > prefill_chunk:
        for i in range(0, len(prompt_ids), prefill_chunk):
            send_msg(pipe_sock, {"op": "verify", "token_ids": prompt_ids[i:i + prefill_chunk], "start": i, "prefill": True})
            cur = recv_msg(rx)[-1]
    else:
        send_msg(pipe_sock, {"op": "verify", "token_ids": prompt_ids, "start": 0, "prefill": True}); cur = recv_msg(rx)[-1]
    ctx = prompt_ids + [cur]; pos = len(prompt_ids) + 1

    def block(spec, ctx, start, depth=8):                       # n_per pipelined iid draws at a fixed position
        ds = local_draft.propose(ctx, K) if spec else []
        msg = {"op": "verify", "token_ids": ([ctx[-1]] + ds) if spec else [ctx[-1]], "start": start, "draft": ds}
        h = collections.Counter(); sent = got = 0
        while got < n_per:
            while sent < n_per and sent - got < depth:
                send_msg(pipe_sock, msg); sent += 1
            h[recv_msg(rx)[0]] += 1; got += 1
        return h

    def entropy(h):
        n = sum(h.values()); return -sum((c / n) * math.log2(c / n) for c in h.values() if c)

    def tvd(a, b):
        ks = set(a) | set(b); na, nb = sum(a.values()), sum(b.values())
        return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in ks)

    rows = []
    for _ in range(stops):
        start = pos - 1
        hp = block(False, ctx, start); hs = block(True, ctx, start); hp2 = block(False, ctx, start)
        rows.append({"entropy": round(entropy(hp), 3), "tv_spec_plain": tvd(hp, hs),
                     "tv_plain_plain": tvd(hp, hp2), "distinct": len(set(hp) | set(hs)),
                     "top": [{"id": int(k), "plain": round(hp[k] / n_per, 3), "spec": round(hs.get(k, 0) / n_per, 3),
                              "text": tok.decode([k])} for k in sorted(hp, key=lambda k: -hp[k])[:6]]})
        # advance `step` tokens (greedy-commit the plain mode's mode) to reach the next content position
        for _ in range(step):
            send_msg(pipe_sock, {"op": "verify", "token_ids": [ctx[-1]], "start": pos - 1, "draft": []})
            ctx.append(recv_msg(rx)[0]); pos += 1
    hi = [r for r in rows if r["entropy"] >= 1.0]               # discriminating (>=1 bit) positions
    agg = hi if hi else rows
    mean_spec = sum(r["tv_spec_plain"] for r in agg) / len(agg)
    mean_noise = sum(r["tv_plain_plain"] for r in agg) / len(agg)
    return {"temp": temp, "top_p": top_p, "n_per": n_per, "stops": stops, "high_entropy_stops": len(hi),
            "max_entropy": max(r["entropy"] for r in rows),
            "mean_tv_spec_vs_plain": mean_spec, "mean_tv_noise_floor": mean_noise,
            "per_stop": rows}


def coordinate_tree(draft_sock, pipe_sock, tok, prompt, tree_cfg, max_new, timeout, ret_sock=None):
    """TREE spec-decode coordinator. each round the draft returns a *tree* of
    candidate continuations rooted at cur; the swarm verifies the whole tree in one
    traversal (tree mask); accept_tree walks the target's argmaxes for the longest
    matching path. the accepted path's KV is kept on each node via a lazy gather
    (piggybacked on the next verify). exact greedy => identical to plain decode."""
    pipe_sock.settimeout(timeout)
    rx = ret_sock if ret_sock is not None else pipe_sock
    eos = tok.eos_token_id
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    prompt_ids = enc["input_ids"][0].tolist()
    out = []
    try:
        send_msg(pipe_sock, {"op": "reset"}); recv_msg(rx)
        send_msg(pipe_sock, {"op": "verify", "token_ids": prompt_ids, "start": 0})   # linear prefill
        cur = recv_msg(rx)[-1]
        pos = len(prompt_ids)
        out = [cur]
        rounds, accepted_total, m_total, gather_prev = 0, 0, 0, None
        t_draft = t_verify = 0.0
        t0 = time.time()
        while len(out) < max_new and cur != eos:
            td = time.time()
            send_msg(draft_sock, {"ids": prompt_ids + out, "tree": tree_cfg})   # ask for a tree
            tr = recv_msg(draft_sock)
            t_draft += time.time() - td
            tk, par, dep = tr["tok"], tr["par"], tr["dep"]
            children = [[] for _ in tk]
            for i, p in enumerate(par):
                if p != -1: children[p].append(i)
            tv = time.time()
            send_msg(pipe_sock, {"op": "verify", "token_ids": tk, "par": par, "dep": dep,
                                 "start": pos, "gather": gather_prev})
            targ = recv_msg(rx)                                # one argmax per tree node
            t_verify += time.time() - tv
            committed, kept = accept_tree(tk, par, {i: c for i, c in enumerate(children)}, targ)
            out.extend(committed); cur = committed[-1]
            gather_prev = list(range(pos)) + [pos + ki for ki in kept]   # keep prefix + accepted path
            pos += len(kept)
            rounds += 1; accepted_total += len(kept) - 1; m_total += len(tk)
            if eos in committed:
                break
    except EDGE_ERRORS as e:
        raise TransportError(f"pipeline edge failed at token {len(out)} ({type(e).__name__}: {e})") from e
    dt = time.time() - t0
    if eos in out:
        out = out[:out.index(eos)]
    return {
        "text": tok.decode(out, skip_special_tokens=True), "n_tokens": len(out), "rounds": rounds,
        "mean_accept": accepted_total / max(rounds, 1),
        "toks_per_traversal": len(out) / max(rounds, 1),
        "tree_nodes": m_total / max(rounds, 1),
        "tok_s": len(out) / max(dt, 1e-9),
        "draft_ms": t_draft / max(rounds, 1) * 1000, "verify_ms": t_verify / max(rounds, 1) * 1000,
    }


def coordinate_tree_fast(draft_sock, pipe_sock, tok, prompt, tree_cfg, max_new, timeout, ret_sock=None):
    """SYNC tree spec on the FAST (graphed) verify. Draft a fixed-topology tree, verify all
    its nodes in ONE traversal (FastVerify.tree_decode), accept the best root-to-leaf path,
    and compact the accepted KV (static tree_gather). gather payload is (start_prev, kept)
    for the static cache (vs the eager gather_cache's absolute-index list). exact greedy."""
    pipe_sock.settimeout(timeout)
    rx = ret_sock if ret_sock is not None else pipe_sock
    eos = tok.eos_token_id
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    prompt_ids = enc["input_ids"][0].tolist()
    out = []
    t_draft = t_verify = 0.0
    try:
        send_msg(pipe_sock, {"op": "reset"}); recv_msg(rx)
        send_msg(pipe_sock, {"op": "verify", "token_ids": prompt_ids, "start": 0})   # linear prefill
        cur = recv_msg(rx)[-1]
        pos = len(prompt_ids); out = [cur]
        rounds, accepted_total, tnodes = 0, 0, 0
        gather_prev = None
        t0 = time.time()
        while len(out) < max_new and cur != eos:
            td = time.time()
            send_msg(draft_sock, {"ids": prompt_ids + out, "tree": tree_cfg}); tr = recv_msg(draft_sock)
            t_draft += time.time() - td
            tk, par, dep = tr["tok"], tr["par"], tr["dep"]
            children = [[] for _ in tk]
            for i, p in enumerate(par):
                if p != -1: children[p].append(i)
            tv = time.time()
            send_msg(pipe_sock, {"op": "verify", "token_ids": tk, "par": par, "dep": dep,
                                 "start": pos, "gather": gather_prev})
            targ = recv_msg(rx)                                # one argmax per tree node
            t_verify += time.time() - tv
            committed, kept = accept_tree(tk, par, {i: c for i, c in enumerate(children)}, targ)
            out.extend(committed); cur = committed[-1]
            gather_prev = (pos, kept)                          # static-cache compaction for next round
            pos += len(kept)
            rounds += 1; accepted_total += len(kept) - 1; tnodes += len(tk)
            if eos in committed:
                break
    except EDGE_ERRORS as e:
        raise TransportError(f"tree edge failed at token {len(out)} ({type(e).__name__}: {e})") from e
    dt = time.time() - t0
    if eos in out:
        out = out[:out.index(eos)]
    return {"text": tok.decode(out, skip_special_tokens=True), "n_tokens": len(out), "rounds": rounds,
            "mean_accept": accepted_total / max(rounds, 1),
            "toks_per_traversal": len(out) / max(rounds, 1), "tree_nodes": tnodes / max(rounds, 1),
            "tok_s": len(out) / max(dt, 1e-9),
            "draft_ms": t_draft / max(rounds, 1) * 1000, "verify_ms": t_verify / max(rounds, 1) * 1000}


def _run_coordinator(args, session_config, plan):
    import json, hashlib
    draft_sock = pipe_sock = ret_sock = None
    try:
        tok = AutoTokenizer.from_pretrained(args.model)     # 20b tokenizer == 120b tokenizer
        local_draft = None
        if args.ngram_draft:                                # model-free drafter: no draft server/socket needed
            local_draft = NgramDrafter(ng=args.ngram_n)
            draft_sock = None
            print(f"[coord] n-gram drafter (ng={args.ngram_n}); no draft server", flush=True)
        else:
            dh, dp = args.draft_server.split(":")
            draft_sock = socket.socket(); draft_sock.connect((dh, int(dp)))
        pipe_sock, ret_sock = connect_ring(args.next, args.tail if args.direct_return else None,
            session_config=session_config, timeout=args.timeout, retry_s=args.timeout)
        print(f"[coord] in-house draft {args.draft_server} + swarm stage 0 at {args.next}; generating ...", flush=True)
        if args.tree:                                       # TREE speculation
            w, d = (int(x) for x in args.tree.split(","))
            cfg = {"width": w, "depth": d}
            for _ in range(2):                              # cold + warm
                r = coordinate_tree(draft_sock, pipe_sock, tok, args.prompt, cfg, args.max_new, args.timeout, ret_sock=ret_sock)
                print(f"[TREE w={w},d={d}] {r['tok_s']:.2f} tok/s | {r['toks_per_traversal']:.2f} tok/traversal | "
                      f"accept {r['mean_accept']:.2f}/round | {r['tree_nodes']:.0f} tree nodes | "
                      f"draft {r['draft_ms']:.0f}ms + verify {r['verify_ms']:.0f}ms/round", flush=True)
            print(f"\n[coord] === OUTPUT ===\n{r['text']}\n", flush=True)
            return
        if args.tree_fast:                                 # FAST graphed tree spec (cold + warm), sweep 'w,d;w,d'
            for spec in args.tree_fast.split(";"):
                w, d = (int(x) for x in spec.split(","))
                cfg = {"width": w, "depth": d}
                for i in range(2):
                    r = coordinate_tree_fast(draft_sock, pipe_sock, tok, args.prompt, cfg, args.max_new,
                                             args.timeout, ret_sock=ret_sock)
                    print(f"[TREE-FAST w={w},d={d} {'warm' if i else 'cold'}] {r['tok_s']:.2f} tok/s | "
                          f"{r['toks_per_traversal']:.2f} tok/trav | accept {r['mean_accept']:.2f} | "
                          f"{r['tree_nodes']:.0f} nodes | draft {r['draft_ms']:.0f}ms verify {r['verify_ms']:.0f}ms", flush=True)
            print(f"\n[coord] === OUTPUT ===\n{r['text'][:400]}\n", flush=True)
            return
        if args.compare:                                   # SYNC then PIPE in ONE process (clean warm A/B)
            depths = [int(x) for x in args.depths.split(",")]
            sync_warm = pipe_warm = None
            for K in [int(x) for x in args.ks.split(",")]:
                for i in range(2):                         # sync: cold (captures K+1 graph), then warm
                    r = coordinate(draft_sock, pipe_sock, tok, args.prompt, K, args.max_new, args.timeout, ret_sock=ret_sock)
                    if i: sync_warm = r
                    print(f"[SYNC K={K} {'warm' if i else 'cold'}] {r['tok_s']:.2f} tok/s | "
                          f"{r['toks_per_traversal']:.2f} tok/trav | accept {r['mean_accept']:.2f} | "
                          f"draft {r['draft_ms']:.0f}ms verify {r['verify_ms']:.0f}ms", flush=True)
                for d in depths:
                    for i in range(2):                     # pipe: cold, then warm, at each depth
                        r = coordinate_pipe(draft_sock, pipe_sock, tok, args.prompt, K, args.max_new,
                                            args.timeout, d, ret_sock=ret_sock)
                        if i: pipe_warm = r
                        print(f"[PIPE K={K} depth={d} {'warm' if i else 'cold'}] {r['tok_s']:.2f} tok/s | "
                              f"{r['toks_per_traversal']:.2f} tok/trav | accept {r['mean_accept']:.2f} | "
                              f"+{r['wasted']} stale | draft {r['draft_ms']:.0f}ms recv {r['recv_ms']:.0f}ms", flush=True)
            if args.dump and sync_warm and pipe_warm:      # receipt: pipe ids + the sync-vs-pipe lossless check
                import json, hashlib
                sids, pids = sync_warm["output_ids"], pipe_warm["output_ids"]
                json.dump({"prompt": args.prompt, "model": args.model,
                           "tok_s_warm": round(pipe_warm["tok_s"], 2), "n_tokens": pipe_warm["n_tokens"],
                           "output_ids": pids, "output_text": pipe_warm["text"],
                           "output_sha256": hashlib.sha256(json.dumps(pids).encode()).hexdigest(),
                           "tokens_match_sync": (sids == pids)}, open(args.dump, "w"))
                print(f"[coord] dumped receipt run -> {args.dump} | tokens_match_sync={sids == pids}", flush=True)
            print(f"\n[coord] === sample output ===\n{r['text'][:400]}\n", flush=True)
            return
        if args.sample_test:                               # on-swarm losslessness proof for sampling
            import json as _json
            r = sample_distribution_test(pipe_sock, tok, args.prompt, args.timeout, ret_sock, local_draft,
                                         args.K, temp=(args.temp if args.temp > 0 else 1.0),
                                         top_p=args.top_p, top_k=args.top_k, seed=args.seed,
                                         prefill_chunk=args.prefill_chunk, reasoning=(args.reasoning or None),
                                         n_per=args.sample_test)
            print(f"\n[SAMPLE-TEST] temp={r['temp']} top_p={r['top_p']} n_per={r['n_per']} stops={r['stops']} "
                  f"(high-entropy>=1bit: {r['high_entropy_stops']}, max_entropy={r['max_entropy']:.2f} bits)", flush=True)
            print(f"  mean TV(spec, plain)  over high-entropy stops = {r['mean_tv_spec_vs_plain']:.4f}", flush=True)
            print(f"  mean TV(plain, plain) noise floor             = {r['mean_tv_noise_floor']:.4f}", flush=True)
            verdict = "LOSSLESS — spec-sampling distribution == plain sampling within MC noise" \
                if r['mean_tv_spec_vs_plain'] <= 1.5 * r['mean_tv_noise_floor'] + 0.01 else "MISMATCH"
            print(f"  VERDICT: {verdict}", flush=True)
            he = sorted([s for s in r["per_stop"] if s["entropy"] >= 1.0], key=lambda s: -s["entropy"])[:3]
            for s in he:
                print(f"  -- stop entropy={s['entropy']:.2f}b tv(spec,plain)={s['tv_spec_plain']:.3f} "
                      f"tv(plain,plain)={s['tv_plain_plain']:.3f}", flush=True)
                for t in s["top"]:
                    print(f"       {repr(t['text'])[:14]:14} plain={t['plain']:.3f} spec={t['spec']:.3f}", flush=True)
            if args.dump:
                _json.dump({"test": "spec-sampling-losslessness", "model": args.model, "verdict": verdict, **r},
                           open(args.dump, "w"))
                print(f"[coord] dumped sample-test -> {args.dump}", flush=True)
            return
        if args.ft_dump:                                   # FAULT-TOLERANT run: resumable, dump partial on node death
            import sys as _sys
            ck_env = checkpoint_env(prompt=args.prompt, model=args.model,
                                    tokenizer=getattr(tok, "name_or_path", args.model),
                                    settings={"temp": args.temp, "top_p": args.top_p, "top_k": args.top_k,
                                              "seed": args.seed, "reasoning": args.reasoning or None})
            resume_ids = load_checkpoint(args.resume_file, ck_env) if args.resume_file else None
            r = coordinate_pipe(draft_sock, pipe_sock, tok, args.prompt, args.K, args.max_new, args.timeout,
                                args.depth, ret_sock=ret_sock, prefill_chunk=args.prefill_chunk,
                                draft_ctx=args.draft_ctx, local_draft=local_draft, reasoning=(args.reasoning or None),
                                temp=args.temp, top_p=args.top_p, top_k=args.top_k, seed=args.seed,
                                prefill_depth=args.prefill_depth, resume_ids=resume_ids, resumable=True,
                                adaptive_pipe=args.adaptive_pipe, adaptive_depth=args.adaptive_depth,
                                swarm_id=plan["ring_id"] if plan else "swarm", job_id="cli-ft-" + __import__("secrets").token_hex(16),
                                nonce=__import__("secrets").token_hex(32),
                                expected_by_signer={row["signer_pubkey"]: (row["lo"], row["hi"]) for row in plan["stages"]} if plan and RECEIPTS else None,
                                strict_job_binding=bool(plan and RECEIPTS))
            write_checkpoint(args.ft_dump, ck_env, r.get("output_ids", []),
                             ok=r.get("ok", False), n_tokens=r.get("n_tokens", 0), text=r.get("text", ""),
                             error=r.get("error"), tok_s=round(r.get("tok_s", 0.0), 2),
                             prefill_s=round(r.get("prefill_s", 0.0), 2),
                             resume_tokens=(len(resume_ids) if resume_ids else 0))
            if args.json_result:
                print("RESULT " + json.dumps(r, allow_nan=False), flush=True)
            status = "OK" if r.get("ok") else f"NODE-DEATH (committed {r.get('n_tokens', 0)} tok)"
            print(f"[coord] FT run {status} -> {args.ft_dump} | {r.get('n_tokens',0)} tok"
                  f"{' | '+r['error'] if r.get('error') else ''}", flush=True)
            print(f"\n[coord] === OUTPUT ({r.get('n_tokens',0)} tok) ===\n{r.get('text','')[:600]}\n", flush=True)
            _sys.exit(0 if r.get("ok") else 3)
        if args.pipe:                                      # PIPELINED coordinator (depth chunks in flight)
            ks = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.K]
            for kv in ks:
                r = coordinate_pipe(draft_sock, pipe_sock, tok, args.prompt, kv, args.max_new,
                                    args.timeout, args.depth, ret_sock=ret_sock, prefill_chunk=args.prefill_chunk,
                                    draft_ctx=args.draft_ctx, local_draft=local_draft,
                                    reasoning=(args.reasoning or None), temp=args.temp, top_p=args.top_p,
                                    top_k=args.top_k, seed=args.seed, prefill_depth=args.prefill_depth, adaptive_pipe=args.adaptive_pipe, adaptive_depth=args.adaptive_depth,
                    swarm_id=plan["ring_id"] if plan else "swarm", job_id="cli-" + __import__("secrets").token_hex(16),
                    nonce=__import__("secrets").token_hex(32),
                    expected_by_signer={row["signer_pubkey"]: (row["lo"],row["hi"]) for row in plan["stages"]} if plan and RECEIPTS else None,
                    strict_job_binding=bool(plan and RECEIPTS))
                print(f"[PIPE K={kv} depth={args.depth} temp={args.temp}] {r['tok_s']:.2f} tok/s | "
                      f"{r['toks_per_traversal']:.2f} tok/traversal | "
                      f"accept {r['mean_accept']:.2f} | +{r['wasted']} stale | prefill {r['prefill_s']:.1f}s | "
                      f"draft {r['draft_ms']:.0f}ms recv {r['recv_ms']:.0f}ms/round", flush=True)
            if args.json_result:
                print("RESULT " + json.dumps(r, allow_nan=False), flush=True)
            if args.dump:
                import json, hashlib
                ids = r["output_ids"]
                rec = {"prompt": args.prompt, "model": args.model, "K": ks[-1], "depth": args.depth,
                       "tok_s_warm": round(r["tok_s"], 2), "n_tokens": r["n_tokens"],
                       "prompt_tokens": r.get("prompt_tokens"), "prefill_s": round(r.get("prefill_s", 0.0), 2),
                       "prefill_depth": args.prefill_depth, "prefill_chunk": args.prefill_chunk,
                       "temp": args.temp, "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed,
                       "mean_accept": round(r["mean_accept"], 3), "toks_per_traversal": round(r["toks_per_traversal"], 3),
                       "decode": ("greedy (exact)" if args.temp <= 0 else f"sampling temp={args.temp} top_p={args.top_p}"),
                       "output_ids": ids, "output_text": r["text"],
                       "output_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
                json.dump(rec, open(args.dump, "w"))
                print(f"[coord] dumped run -> {args.dump} (sha256 {rec['output_sha256'][:16]}..)", flush=True)
            if RECEIPTS and not r.get("proof_verified"):  # PROVE: sweep the ring once for signed per-stage receipts
                rx = ret_sock if ret_sock is not None else pipe_sock
                send_msg(pipe_sock, {"op": "receipt", "receipts": []})
                recs = recv_msg(rx)
                print(f"\n[coord] === PROVE: {len(recs)} signed per-stage receipts ===", flush=True)
                ok = True
                for rr in recs:
                    body = {k: v for k, v in rr.items() if k != "stage"}
                    try:
                        verify_receipt(body)
                        print(f"  stage {rr['stage']}: layers[{rr['layer_start']}:{rr['layer_end']}] "
                              f"n={rr['n_chunks']} in_root {rr['in_root'][:12]} out_root {rr['out_root'][:12]} "
                              f"pub {rr['pubkey'][:12]} — sig VALID", flush=True)
                    except Exception as e:
                        ok = False; print(f"  stage {rr['stage']}: sig FAILED ({e})", flush=True)
                try:
                    total = args.n_layers or max(rr["layer_end"] for rr in recs)
                    if not args.n_layers:                  # derived from the receipts under test = self-referential:
                        print("  ⚠ coverage target derived from the receipts themselves (pass --n-layers to pin "
                              "the model's true depth — a layer-omitting ring passes without it)", flush=True)
                    verify_coverage([{k: v for k, v in rr.items() if k != "stage"} for rr in recs], total)
                    print(f"  coverage: blocks tile [0:{total}] no gap/overlap — every layer attested by a distinct signed node", flush=True)
                except Exception as e:
                    ok = False; print(f"  coverage FAILED: {e}", flush=True)
                print(f"[coord] PROVE verdict: {'ALL receipts valid + full layer coverage — coordinator cannot fabricate, no node paid without proving its block' if ok else 'FAILED'}", flush=True)
            print(f"\n[coord] === OUTPUT ===\n{r['text']}\n", flush=True)
            return
        ks = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.K]
        for kv in ks:
            adaptive = (kv == 0) or (not args.sweep and args.adaptive)
            r = coordinate(draft_sock, pipe_sock, tok, args.prompt, (6 if kv == 0 else kv),
                           args.max_new, args.timeout, adaptive=adaptive, ret_sock=ret_sock,
                           prefill_chunk=args.prefill_chunk)
            if args.sweep:
                print(f"[SWEEP K={kv}] {r['tok_s']:.2f} tok/s | {r['toks_per_traversal']:.2f} tok/traversal | "
                      f"accept {r['mean_accept']:.2f} | draft {r['draft_ms']:.0f}ms + verify {r['verify_ms']:.0f}ms/round", flush=True)
            else:
                print(f"\n[coord] === OUTPUT ===\n{r['text']}\n", flush=True)
                print(f"[coord] {r['n_tokens']} tok | {r['tok_s']:.2f} tok/s | {r['toks_per_traversal']:.2f} tok/traversal | "
                      f"accept {r['mean_accept']:.2f} | draft {r['draft_ms']:.0f}ms + verify {r['verify_ms']:.0f}ms/round", flush=True)
        if args.dump:                                       # sync output ids + hash (for the receipt / a transport A/B)
            import json, hashlib
            ids = r["output_ids"]
            json.dump({"prompt": args.prompt, "model": args.model, "K": ks[-1], "mode": "sync",
                       "tok_s": round(r["tok_s"], 2), "n_tokens": r["n_tokens"], "output_ids": ids,
                       "output_text": r["text"],
                       "output_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}, open(args.dump, "w"))
            print(f"[coord] dumped run -> {args.dump} (sha256 {hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]}..)", flush=True)
        return

    finally:
        for channel in (draft_sock, pipe_sock, ret_sock):
            if channel is not None:
                channel.close()


def runtime_config_payload(args, config, lo, hi):
    """The calibrated portable stage contract, independent of secrets/socket handles."""
    import importlib.metadata
    from pathlib import Path
    here = Path(__file__).resolve().parent
    files = ("specpipe.py", "pipeline.py", "fastverify.py", "mxfp4_guard.py", "ngram_draft.py", "specsample.py")
    sources = {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in files}
    for name in ("pipeline_session", "pipeline_telemetry", "pipeline_plan", "gpt_oss_contract"):
        module = __import__("shard." + name, fromlist=[name])
        sources[name + ".py"] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    versions = {}
    for name in ("torch", "transformers", "kernels", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"engine": "gpt-oss", "n_layers": int(config["num_hidden_layers"]), "nstages": args.nstages,
            "config_sha256": hashlib.sha256(json.dumps({k: v for k, v in config.items()
                if k not in ("_name_or_path", "name_or_path")}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "source_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
            "versions": {**versions, "torch_cuda": torch.version.cuda}, "quantization": config.get("quantization_config"),
            "stage": args.stage, "lo": lo, "hi": hi, "head": args.stage == 0,
            "tail": args.stage == args.nstages - 1, "fast": bool(args.fast), "served_head": bool(args.served_head),
            "attn": args.attn, "max_ctx": args.max_ctx, "direct_return": bool(args.direct_return),
            "window": os.environ.get("FV_WINDOW", "0") not in ("", "0"),
            "sync_send": SYNC_SEND}


def _cli_contract(args, parser):
    from pathlib import Path
    from shard.pipeline_plan import load_plan, resolve_bounds, validate_plan
    from transformers import AutoConfig
    guard, assignment = None, {}
    if os.environ.get("SHARD_STAGE_LEASE_CONFIG") and not args.coordinator:
        from shard.leased_runtime import load_local_lease_guard, process_lease_watchdog
        guard = load_local_lease_guard(os.environ["SHARD_STAGE_LEASE_CONFIG"])
        guard.assert_live()
        assignment = getattr(guard, "stage_assignment", {}) or {}
        process_lease_watchdog(guard)
        if args.legacy_protocol:
            parser.error("managed node leases require the authenticated protocol")
    config = AutoConfig.from_pretrained(args.model, local_files_only=bool(args.deployment_plan or assignment)).to_dict()
    plan = load_plan(args.deployment_plan, config=config) if args.deployment_plan else (
        validate_plan(assignment["deployment_plan"], config=config) if assignment.get("deployment_plan") else None)
    if plan is not None and assignment.get("deployment_plan") and validate_plan(assignment["deployment_plan"], config=config) != plan:
        parser.error("CLI plan differs from the node-local leased assignment")
    lo, hi = resolve_bounds(config, args.stage, args.nstages, split=args.split, lo=args.lo, hi=args.hi, plan=plan)
    if not args.legacy_protocol and plan is None:
        parser.error("strict protocol requires --deployment-plan or a leased assignment; --legacy-protocol is explicit compatibility")
    if plan is not None:
        args.n_layers = plan["n_layers"]
        if not args.legacy_protocol:
            args.direct_return = True
            if not args.coordinator and args.stage == 0:
                args.served_head = True
        cohort = plan.get("model_cohort")
        if not args.legacy_protocol and cohort is None:
            parser.error("strict production plan requires the complete model_cohort and verified download inventory")
        if cohort:
            if not args.legacy_protocol:
                from shard.gpt_oss_contract import validate_supported_cohort
                validate_supported_cohort(cohort, config)
            if hashlib.sha256((Path(args.model) / "config.json").read_bytes()).hexdigest() != cohort["config_sha256"]:
                parser.error("model config differs from the deployment cohort")
            from shard.download_inventory import verify_inventory
            verify_inventory(args.model, expected_checkpoint_id=cohort["checkpoint_id"],
                             expected_repo=cohort["model_id"], verify_files=True)
        if args.coordinator:
            if args.next and args.next != plan["coordinator"]["head"] or args.tail and args.tail != plan["coordinator"]["tail"]:
                parser.error("coordinator endpoints differ from plan")
            args.next, args.tail = plan["coordinator"]["head"], plan["coordinator"]["tail"]
        else:
            row = plan["stages"][args.stage]
            if args.next and args.next != (row["next_endpoint"] or ""):
                parser.error("next endpoint differs from plan")
            args.next = row["next_endpoint"] or ""
            if guard:
                if (guard.ring_id != plan["ring_id"] or guard.model_cohort_sha256 != plan["cohort_id"] or
                        guard.node_id != row["node_id"] or guard.gpu_uuid.lower() != row["gpu_uuid"].lower()):
                    parser.error("node lease identity differs from stage plan")
                if not args.device.startswith("cuda"):
                    parser.error("leased production stage requires its assigned CUDA GPU")
                actual = str(getattr(torch.cuda.get_device_properties(args.device), "uuid", ""))
                if actual.lower() != guard.gpu_uuid.lower():
                    parser.error("actual CUDA device UUID differs from lease")
    payload = runtime_config_payload(args, config, lo, hi)
    if guard and getattr(guard, "expected_runtime_config", None) != payload:
        parser.error("effective GPT-OSS runtime differs from its measured lease calibration")
    caller_key = None
    if not args.legacy_protocol:
        key_path = (args.coordinator_key or os.environ.get("SHARD_COORDINATOR_KEY")) if args.coordinator else NODE_KEY_PATH
        if not key_path:
            parser.error("strict coordinator requires --coordinator-key or SHARD_COORDINATOR_KEY")
        caller_key = load_or_make_node_key(key_path)
    session = None if args.legacy_protocol else SessionConfig.from_plan(plan, -1 if args.coordinator else args.stage,
        ttl_s=min(3600, max(30, args.timeout * 2)), caller_key=caller_key)
    print("CONFIG " + json.dumps({**payload, "protocol": "legacy" if session is None else "shard-pipeline-session/1",
        "runtime_config": payload,
        "runtime_config_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "ring_id": plan["ring_id"] if plan else None, "cohort_id": plan["cohort_id"] if plan else None,
        "K": args.K, "depth": args.depth, "ngram_n": args.ngram_n,
        "margin_policy": {"mode": "adaptive" if NgramDrafter(ng=args.ngram_n).adaptive else "fixed",
                          "cap": NgramDrafter(ng=args.ngram_n).margin_cap}, "torch": torch.__version__}, sort_keys=True), flush=True)
    return lo, hi, session, guard, plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--nstages", type=int, required=True)
    ap.add_argument("--lo", type=int, default=-1, help="explicit layer-block start (uneven VRAM-aware split; -1 = even)")
    ap.add_argument("--hi", type=int, default=-1, help="explicit layer-block end (exclusive; -1 = even split)")
    ap.add_argument("--split", default=None, help="comma-separated positive per-stage layer counts")
    ap.add_argument("--deployment-plan", help="validated shard-pipeline-plan/1 file")
    ap.add_argument("--coordinator-key", help="existing controller Ed25519 key file; never embedded in the plan")
    ap.add_argument("--legacy-protocol", action="store_true", help="explicit compatibility with older raw-op rings")
    ap.add_argument("--adaptive-pipe", action="store_true", help="greedy live shape-gated K/depth adaptation at drained boundaries")
    ap.add_argument("--adaptive-depth", action="store_true", help="fixed-K same-shape depth adjustment; no full-greedy control cost")
    ap.add_argument("--json-result", action="store_true", help="emit the complete measured/signed result as RESULT JSON")
    ap.add_argument("--coordinator", action="store_true", help="in-house entry node: draft + drive, no 120B layers")
    ap.add_argument("--served-head", action="store_true", help="stage 0 runs as a swarm serve node (embeds token ids)")
    ap.add_argument("--direct-return", action="store_true", help="tail sends results straight to the coordinator (1 hop, not relayed)")
    ap.add_argument("--tail", default="", help="coordinator: host:port of the tail, for the direct return channel")
    ap.add_argument("--tree", default="", help="coordinator: tree spec 'width,depth' (e.g. 3,6) -> tree speculation")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")        # target
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--listen-port", type=int, default=29501)
    ap.add_argument("--next", default="")
    ap.add_argument("--device", default="cuda:0")           # this stage's block
    ap.add_argument("--draft-device", default="cuda:1")     # draft (head only; its own GPU)
    ap.add_argument("--draft-server", default="", help="host:port of the in-house vLLM draft service (else local draft)")
    ap.add_argument("--ngram-draft", action="store_true", help="coordinator: model-free n-gram/prompt-lookup drafter "
                    "(no draft server, no KV) — the long-context spec-decode path; survives past 100k where the 20b draft OOMs")
    ap.add_argument("--ngram-n", type=int, default=3, help="n-gram suffix length for --ngram-draft (tune 2-4)")
    ap.add_argument("--reasoning", default="", help="gpt-oss reasoning_effort (low|medium|high); low cuts the "
                    "analysis channel so the answer starts sooner (key for copy/retrieval-heavy long-ctx tasks)")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--adaptive", action="store_true", help="tune K live from the running acceptance rate")
    ap.add_argument("--fast", action="store_true", help="serve node: static-cache CUDA-graph verify (~5x, fixed-K linear)")
    ap.add_argument("--attn", default="eager", help="attention impl for stages: eager | flex_attention "
                    "(flex = O(n) prefill on Ada, needed for long context with gpt-oss sinks)")
    ap.add_argument("--prompt-file", default="", help="read the prompt from this file (for 100k-token prompts)")
    ap.add_argument("--draft-ctx", type=int, default=0, help="window the draft query to the last N tokens "
                    "(0=full); keeps the draft fast at long context (a full-95k draft is ~800ms/round)")
    ap.add_argument("--prefill-chunk", type=int, default=0, help="chunk the prefill into this many tokens "
                    "(0=one shot); long prompts need chunking so per-chunk activations stay bounded")
    ap.add_argument("--max-ctx", type=int, default=2048, help="fast-verify static cache size (prompt+gen ceiling); "
                    "sized to the request, not a hardware limit — the KV cache is ~tens of KB/token, so a 4090 "
                    "stage holds far more than the old 2048 default. overflow fails clean (ContextOverflow), never corrupts")
    ap.add_argument("--sweep", default="", help="comma K list to measure on one load, 0=adaptive (e.g. 2,3,4,0)")
    ap.add_argument("--pipe", action="store_true", help="coordinator: PIPELINED spec-decode (depth chunks in flight; needs --direct-return)")
    ap.add_argument("--depth", type=int, default=4, help="pipelined coordinator: verify chunks in flight")
    ap.add_argument("--prefill-depth", type=int, default=8, help="pipelined coordinator: PREFILL chunks in flight "
                    "(overlap prefill across stages; >=nstages fills the pipe -> ~Nstage-fold faster TTFT)")
    ap.add_argument("--temp", type=float, default=0.0, help="sampling temperature (0=greedy/argmax, exact legacy path); "
                    ">0 enables LOSSLESS speculative sampling at the tail (temp/top-p/top-k)")
    ap.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling cutoff (with --temp>0)")
    ap.add_argument("--top-k", type=int, default=0, help="top-k sampling cutoff (0=off; with --temp>0)")
    ap.add_argument("--seed", type=int, default=0, help="tail sampler seed (reproducible sampled runs / receipts)")
    ap.add_argument("--n-layers", type=int, default=0, help="coordinator: the model's TRUE layer count for the "
                    "receipt coverage check (e.g. 36 for gpt-oss-120b); 0 derives it from the receipts "
                    "themselves, which a layer-omitting ring can game (warned loudly)")
    ap.add_argument("--sample-test", type=int, default=0, help="coordinator: draw N iid next-tokens via PLAIN vs "
                    "SPECULATIVE sampling and report TV distance (on-swarm losslessness proof); needs --ngram-draft")
    ap.add_argument("--resume-file", default="", help="coordinator: versioned checkpoint (as written by --ft-dump) "
                    "to RESUME from (re-prefill prompt+committed on a healed ring, continue) — fault tolerance; "
                    "a checkpoint from another prompt/model/settings is REFUSED (CheckpointError)")
    ap.add_argument("--ft-dump", default="", help="coordinator: run resumable + atomically write the versioned "
                    "checkpoint {schema,job,prompt_sha256,...,output_ids,ids_sha256} here on completion OR "
                    "mid-request node death (exit 3 if a node died), so the control plane can heal+resume")
    ap.add_argument("--compare", action="store_true", help="coordinator: SYNC then PIPE (cold+warm) in ONE process for a clean A/B")
    ap.add_argument("--depths", default="2,4,8", help="--compare: pipe depths to sweep (one process)")
    ap.add_argument("--ks", default="4", help="--compare: K values to sweep (one process; graph recaptures per K)")
    ap.add_argument("--tree-fast", default="", help="coordinator: FAST graphed tree spec 'w,d' (cold+warm)")
    ap.add_argument("--dump", default="", help="--pipe: write {prompt, output_ids, tok_s} JSON here (for the receipt)")
    ap.add_argument("--prompt", default="Explain decentralized computing in two sentences.")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    wire.key_from_env()                 # shared swarm key (SHARD_PSK); fail fast before the model load
    lo, hi, session_config, lease_guard, plan = _cli_contract(args, ap)
    if args.prompt_file:                # long (100k) prompts can't fit on the CLI
        args.prompt = open(args.prompt_file).read()

    if args.coordinator:
        return _run_coordinator(args, session_config, plan)

    _lo, _hi = lo, hi
    parts = load_stage(args.model, args.stage, args.nstages, device=args.device, attn=args.attn, lo=_lo, hi=_hi)
    parts["_session_config"], parts["_lease_guard"] = session_config, lease_guard
    if session_config is not None:
        parts["_session_key"] = load_or_make_node_key(NODE_KEY_PATH)
        from shard.pipeline_telemetry import StageTelemetry
        parts["_telemetry"] = StageTelemetry(node_id=session_config.descriptor()["node_id"],
            cohort_id=session_config.plan["cohort_id"], index=args.stage,
            path=os.environ.get("SHARD_STAGE_TELEMETRY_FILE"))
        parts["_telemetry"].start(endpoint=args.next or None, config=session_config,
            target_index=args.stage + 1 if args.next else None, send=_raw_send_msg, recv=_raw_recv_msg)

    if args.stage != 0 or args.served_head:                 # swarm serve node (stage 0 embeds token ids)
        is_tail = args.stage == args.nstages - 1
        if args.direct_return and is_tail:
            if args.fast:
                serve_tail_fast(parts, args.listen_port, args.timeout, args.device, max_ctx=args.max_ctx)
            else:
                serve_tail_direct(parts, args.listen_port, args.timeout, args.device)
        elif args.fast:
            serve_spec_fast(parts, args.stage, args.nstages, args.listen_port, args.next, args.timeout,
                            args.device, direct=args.direct_return, max_ctx=args.max_ctx)
        else:
            serve_spec(parts, args.stage, args.nstages, args.listen_port, args.next, args.timeout,
                       args.device, direct=args.direct_return)
        return

    draft, draft_sock = None, None
    if args.draft_server:                                   # in-house vLLM draft service
        dh, dp = args.draft_server.split(":")
        draft_sock = socket.socket(); draft_sock.connect((dh, int(dp)))
        print(f"[s0] using in-house draft service at {args.draft_server}", flush=True)
    else:
        print(f"[s0] loading draft {args.draft} on {args.draft_device} ...", flush=True)
        draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype="auto",
                                                     device_map={"": args.draft_device},
                                                     attn_implementation="eager").eval()
        print(f"[s0] draft loaded, draft_mem={torch.cuda.memory_allocated(args.draft_device)/1e9:.1f}GB", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    host, port = args.next.split(":")
    sock = socket.socket(); sock.settimeout(args.timeout); sock.connect((host, int(port)))
    print(f"[s0] connected forward to stage 1 at {args.next}; K={args.K}; generating ...", flush=True)
    if args.sweep:                          # load once, measure several K on one connection
        try:
            for kv in [int(x) for x in args.sweep.split(",")]:
                adaptive = (kv == 0)
                rr = generate_spec(draft, parts, tok, sock, args.prompt, (6 if adaptive else kv),
                                   args.max_new, args.device, args.draft_device, args.timeout,
                                   adaptive=adaptive, draft_sock=draft_sock)
                tag = f"adaptive(mean {rr['mean_K']:.1f})" if adaptive else f"K={kv}"
                print(f"[SWEEP {tag}] {rr['tok_s']:.2f} tok/s | {rr['toks_per_traversal']:.2f} tok/traversal | "
                      f"accept {rr['mean_accept']:.2f} | draft {rr['draft_ms']:.0f}ms + verify {rr['verify_ms']:.0f}ms/round "
                      f"-> async ceiling {rr['toks_per_traversal']/(max(rr['draft_ms'],rr['verify_ms'])/1000):.1f} tok/s", flush=True)
        finally:
            sock.close()
        return
    try:
        r = generate_spec(draft, parts, tok, sock, args.prompt, args.K, args.max_new,
                          args.device, args.draft_device, args.timeout, adaptive=args.adaptive,
                          draft_sock=draft_sock)
    except TransportError as e:
        print(f"\n[s0] TRANSPORT FAILURE: {e}", flush=True); raise SystemExit(2)
    finally:
        sock.close()
    kdesc = f"adaptive (mean {r['mean_K']:.1f}, {r['k_lo']}-{r['k_hi']})" if args.adaptive else f"{args.K}"
    print(f"\n[s0] === OUTPUT ===\n{r['text']}\n", flush=True)
    print(f"[s0] {r['n_tokens']} tokens in {r['rounds']} verify traversals | "
          f"mean accepted/round {r['mean_accept']:.2f} | {r['toks_per_traversal']:.2f} tokens/traversal "
          f"(vs 1.0 plain) | {r['tok_s']:.2f} tok/s | draft={args.draft.split('/')[-1]} K={kdesc} "
          f"({args.nstages}-stage pipeline)", flush=True)


if __name__ == "__main__":
    main()

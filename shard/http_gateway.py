"""Authenticated bounded HTTP/SSE serving shared by engine backends."""
from __future__ import annotations
import hashlib
import hmac
import json
import math
import re
from pathlib import Path
import select
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


def _unique_json(pairs):
    body = {}
    for key, value in pairs:
        if key in body:
            raise ValueError("duplicate JSON key")
        body[key] = value
    return body


def _nonfinite_json(value):
    raise ValueError("non-finite JSON number")


def _finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number
try:
    from shard.service_queue import ServiceQueue, TenantLimits, AdmissionError, JobCancelled, JobExpired
except ImportError:
    from service_queue import ServiceQueue, TenantLimits, AdmissionError, JobCancelled, JobExpired

class AuthRegistry:
    def __init__(self, keys, limits):
        if not keys or any(not isinstance(key, str) or len(key) < 16 for key in keys):
            raise ValueError("API authentication requires nonempty keys of at least 16 characters")
        if any(tenant not in limits for tenant in keys.values()):
            raise ValueError("every API key requires a known tenant policy")
        if any(not isinstance(tenant, str) or not tenant or len(tenant) > 128
               or not isinstance(policy, TenantLimits) for tenant, policy in limits.items()):
            raise ValueError("tenant policies require bounded nonempty string identities")
        self.limits = dict(limits)
        self._keys = [(hashlib.sha256(key.encode()).digest(), tenant) for key, tenant in keys.items()]

    @classmethod
    def from_file(cls, path):
        body = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        limits = {tenant: TenantLimits(**policy) for tenant, policy in body["tenants"].items()}
        return cls(body["keys"], limits)

    def authenticate(self, header):
        if not isinstance(header, str) or len(header) > 1024:
            return None
        fields = header.split()
        if len(fields) != 2 or fields[0].lower() != "bearer":
            return None
        supplied = hashlib.sha256(fields[1].encode()).digest()
        tenant = None
        for expected, candidate in self._keys:
            if hmac.compare_digest(supplied, expected):
                tenant = candidate
        return tenant


class _BoundedServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, handler, gateway):
        self.gateway = gateway
        self._slots = threading.BoundedSemaphore(gateway.max_connections)
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, handler)

    def process_request(self, request, address):
        if not self._slots.acquire(blocking=False):
            try:
                request.settimeout(0.2)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self._slots.release()


class Gateway:
    def __init__(self, backend=None, auth=None, *, ring_pool=None, default_model=None,
                 max_queue=32, max_queued_tokens=131072,
                 max_connections=64, max_body=1048576, request_timeout=15.0, write_timeout=15.0,
                 allow_unverified_start=False):
        if ring_pool is not None:
            if backend is not None:
                raise ValueError("provide a single backend or a ring_pool")
            try:
                from shard.ring_router import RingRouter, MultiRingQueue
            except ImportError:
                from ring_router import RingRouter, MultiRingQueue
            backend = RingRouter(ring_pool, default_model=default_model)
        if backend is None or auth is None:
            raise ValueError("backend/ring_pool and authentication required")
        self.backend, self.auth = backend, auth
        if not allow_unverified_start and not backend.ready()[0]:
            raise ValueError("signed backend warmup must succeed before service admission")
        if any(type(value) is not int or value < 1 for value in (max_connections, max_body)):
            raise ValueError("connection and body bounds must be positive integers")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in (request_timeout, write_timeout)):
            raise ValueError("HTTP read/write timeouts must be finite and positive")
        self.max_connections, self.max_body = max_connections, max_body
        self.request_timeout, self.write_timeout = request_timeout, write_timeout
        queue_cls = MultiRingQueue if ring_pool is not None else ServiceQueue
        self.jobs = queue_cls(backend, auth.limits, max_queue=max_queue, max_queued_tokens=max_queued_tokens)
        self.jobs.start()

    def submit(self, tenant, body, *, idempotency_key=None):
        if hasattr(self.jobs, "submit_request"):
            return self.jobs.submit_request(tenant, body, idempotency_key=idempotency_key)
        payload, prompt_tokens, maximum, deadline = self.backend.prepare(body)
        return self.jobs.submit(tenant, payload, prompt_tokens=prompt_tokens, max_new=maximum,
                                timeout_s=deadline, idempotency_key=idempotency_key)

    def server(self, host="127.0.0.1", port=0):
        return _BoundedServer((host, port), Handler, self)

    def shutdown(self, *, drain=False, timeout=10.0):
        return self.jobs.shutdown(drain=drain, timeout=timeout)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.gateway.request_timeout)
        if isinstance(self.connection, ssl.SSLSocket):
            self.connection.do_handshake()

    def log_message(self, *args):
        pass  # URL/auth/prompt data never enter stdlib access logs

    @property
    def gateway(self):
        return self.server.gateway

    def _json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(data)

    def _error(self, message, status=400, code="invalid_request_error"):
        self._json({"error": {"message": message, "type": code, "code": code}}, status)

    def _tenant(self):
        tenant = self.gateway.auth.authenticate(self.headers.get("Authorization"))
        if tenant is None:
            self._error("invalid API credentials", 401, "authentication_error")
        return tenant

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/health":
            return self._json({"status": "ok", "service": getattr(self.gateway.backend, "service_name", "shard-inference-gateway")})
        tenant = self._tenant()
        if tenant is None:
            return
        if path == "/ready":
            ready, reason = self.gateway.backend.ready()
            ready = bool(ready and self.gateway.jobs.metrics()["accepting"])
            return self._json({"ready": ready, "reason": reason, "mode": self.gateway.jobs.metrics()["mode"]}, 200 if ready else 503)
        if path == "/metrics":
            return self._json({"service": self.gateway.jobs.metrics(tenant), "ring": self.gateway.backend.stats()})
        if path == "/v1/models":
            models = self.gateway.backend.models() if hasattr(self.gateway.backend, "models") else [self.gateway.backend.model_id]
            return self._json({"object": "list", "data": [{"id": model,
                "object": "model", "created": 0, "owned_by": "shard"} for model in models]})
        if path.startswith("/v1/jobs/"):
            if path.endswith("/receipts"):
                job = self.gateway.jobs.get(tenant, path[len("/v1/jobs/"):-len("/receipts")])
                if job is None:
                    return self._error("job not found", 404, "not_found")
                return self._json(job.result["proof"]) if job.state == "completed" else self._error("job is not settled", 409, "not_settled")
            job = self.gateway.jobs.get(tenant, path[len("/v1/jobs/"):])
            return self._json(job.snapshot()) if job is not None else self._error("job not found", 404, "not_found")
        self._error("not found", 404, "not_found")

    def do_POST(self):
        tenant = self._tenant()
        if tenant is None:
            self.close_connection = True
            return
        path = urlsplit(self.path).path
        if path.startswith("/v1/jobs/") and path.endswith("/cancel"):
            self.close_connection = True
            job_id = path[len("/v1/jobs/"):-len("/cancel")]
            if not self.gateway.jobs.cancel(tenant, job_id):
                return self._error("job not found", 404, "not_found")
            return self._json({"id": job_id, "cancel_requested": True})
        if path != "/v1/chat/completions":
            self.close_connection = True
            return self._error("not found", 404, "not_found")
        try:
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get_all("Transfer-Encoding", []) or len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]):
                raise ValueError("ambiguous request framing")
            length = int(lengths[0])
            if not 0 < length <= self.gateway.max_body:
                self.close_connection = True
                return self._error("request body missing or too large", 413, "request_too_large")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("incomplete request body")
            body = json.loads(raw, object_pairs_hook=_unique_json, parse_constant=_nonfinite_json,
                              parse_float=_finite_json_float)
            job, _ = self.gateway.submit(tenant, body, idempotency_key=self.headers.get("Idempotency-Key"))
        except AdmissionError as error:
            return self._error(str(error), error.status, error.code)
        except (OSError, ValueError, TypeError, RecursionError):
            self.close_connection = True
            return self._error("invalid or incomplete request body", 400, "invalid_request_error")
        with job.changed:
            job.clients += 1
        try:
            if body.get("stream", False):
                return self._stream(job)
            while not job.terminal:
                if self._client_gone():
                    return
                with job.changed:
                    job.changed.wait(0.2)
            if job.state != "completed":
                return self._error(job.error or "inference failed", 408 if job.state == "expired" else 409 if job.state == "cancelled" else 502,
                                   job.error_code or "inference_failed")
            result = job.result
            self._json({"id": job.id, "object": "chat.completion", "created": job.created_unix,
                "model": self._job_model(job),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": result["text"]},
                             "finish_reason": result["finish_reason"]}],
                "usage": self._usage(job), "shard": {"proof_verified": result["proof"]["verified"],
                    "proof_scope": result["proof"]["scope"], "recovery": result["recovery"],
                    **({"route": result["route"]} if "route" in result else {})}})
        except (OSError, ValueError):
            self.close_connection = True
        finally:
            with job.changed:
                job.clients -= 1
            self.gateway.jobs.cancel_if_unobserved(tenant, job.id)

    def _client_gone(self):
        if isinstance(self.connection, ssl.SSLSocket):
            return False  # SSL does not support MSG_PEEK; writes/deadlines detect disconnects
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable and self.connection.recv(1, socket.MSG_PEEK) == b"")
        except OSError:
            return True

    def _usage(self, job):
        return {"prompt_tokens": job.prompt_tokens, "completion_tokens": len(job.tokens),
                "total_tokens": job.prompt_tokens + len(job.tokens)}

    def _job_model(self, job):
        return getattr(job, "served_model", self.gateway.backend.model_id)

    def _job_decode(self, job, tokens):
        return getattr(job, "bound_backend", self.gateway.backend).decode(tokens)

    def _stream(self, job):
        cursor = 0
        published_chars = 0
        last = self.headers.get("Last-Event-ID")
        if last:
            try:
                pieces = last.rsplit(":", 2)
                prefix, offset = pieces[:2] if len(pieces) == 3 else last.rsplit(":", 1)
                cursor = int(offset)
                if prefix != job.id or not 0 <= cursor <= len(job.checkpoint()):
                    raise ValueError
                prefix_text = self._job_decode(job, job.checkpoint()[:cursor])
                if len(pieces) == 3:
                    published_chars = int(pieces[2])
                    if not 0 <= published_chars <= len(prefix_text):
                        raise ValueError
                else:
                    if prefix_text.endswith("\ufffd"):
                        raise ValueError("ambiguous legacy cursor")
                    published_chars = len(prefix_text)
            except ValueError:
                return self._error("Last-Event-ID requires this job's token and published-character cursor", 409, "invalid_resume_cursor")
        published = self._job_decode(job, job.checkpoint()[:cursor])[:published_chars]
        self._published_chars = published_chars
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers(); self.close_connection = True
        self.connection.settimeout(self.gateway.write_timeout)
        self._chunk(job, {"role": "assistant"}, None, cursor)
        ping = time.monotonic()
        while True:
            with job.changed:
                tokens, state, result, error, error_code = (list(job.tokens), job.state,
                    job.result, job.error, job.error_code)
            terminal = state in ("completed", "failed", "cancelled", "expired")
            text = self._job_decode(job, tokens).rstrip("\ufffd") if not terminal else self._job_decode(job, tokens)
            if not text.startswith(published):
                self._sse({"error": {"code": "non_append_only_decode", "message": "tokenizer revised a published prefix"}}, len(tokens), job)
                self._sse("[DONE]")
                return
            if text != published:
                self._published_chars = len(text)
                self._chunk(job, {"content": text[len(published):]}, None, len(tokens))
                published, cursor = text, len(tokens)
            if terminal:
                if state == "completed":
                    self._chunk(job, {}, result["finish_reason"], len(tokens), usage=self._usage(job),
                        shard={"proof_verified": True, "proof_scope": result["proof"]["scope"],
                               "recovery": result["recovery"],
                               **({"route": result["route"]} if "route" in result else {})})
                else:
                    self._sse({"error": {"code": error_code, "message": error}}, len(tokens), job)
                self._sse("[DONE]")
                return
            if self._client_gone():
                return
            if time.monotonic() - ping >= 5:
                self.wfile.write(b": keepalive\n\n"); self.wfile.flush(); ping = time.monotonic()
            with job.changed:
                job.changed.wait(0.1)

    def _chunk(self, job, delta, reason, cursor, **extra):
        self._sse({"id": job.id, "object": "chat.completion.chunk", "created": job.created_unix,
                   "model": self._job_model(job),
                   "choices": [{"index": 0, "delta": delta, "finish_reason": reason}], **extra}, cursor, job)

    def _sse(self, data, cursor=None, job=None):
        prefix = f"id: {job.id}:{cursor}:{self._published_chars}\n" if job is not None else ""
        value = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, allow_nan=False)
        self.wfile.write((prefix + "data: " + value + "\n\n").encode()); self.wfile.flush()

"""Bounded, tenant-fair serial inference jobs. No model, GPU, HTTP or billing dependencies.

Admission reserves prompt+maximum completion tokens in a rolling minute. That is
a resource quota, not an invoice. Idempotency lasts only for retained in-memory jobs;
this module does not claim durable failover or continuous batching.
"""
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import math
import threading
import time
import uuid


class AdmissionError(ValueError):
    def __init__(self, message, *, status=429, code="gateway_overloaded"):
        super().__init__(message)
        self.status, self.code = status, code


class JobCancelled(RuntimeError):
    pass


class JobExpired(TimeoutError):
    pass


@dataclass(frozen=True)
class TenantLimits:
    max_active: int = 4
    requests_per_minute: int = 30
    tokens_per_minute: int = 65536

    def __post_init__(self):
        for name in ("max_active", "requests_per_minute", "tokens_per_minute"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class Job:
    tenant: str
    payload: dict
    prompt_tokens: int
    max_new: int
    deadline: float
    fingerprint: str
    idempotency_key: str | None
    created_mono: float
    id: str = field(default_factory=lambda: "chatcmpl-" + uuid.uuid4().hex)
    created_unix: int = field(default_factory=lambda: int(time.time()))
    state: str = "queued"
    tokens: list = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    error_code: str | None = None
    stop_kind: str | None = None
    finished_mono: float | None = None
    clients: int = 0
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    changed: threading.Condition = field(default_factory=threading.Condition, repr=False)

    @property
    def terminal(self):
        return self.state in ("completed", "failed", "cancelled", "expired")

    def checkpoint(self):
        with self.changed:
            return list(self.tokens)

    def check_stop(self, now=None):
        if self.stop_kind == "deadline" or (now if now is not None else time.monotonic()) >= self.deadline:
            raise JobExpired("job deadline exceeded")
        if self.cancelled.is_set():
            raise JobCancelled("job cancelled")

    def commit(self, token):
        # Called AFTER the coordinator commits. Even a racing cancellation must
        # preserve this observed prefix; dropping it would lose the resume frontier.
        if type(token) is not int or token < 0:
            raise ValueError("committed token must be a nonnegative integer")
        with self.changed:
            if self.state != "running" or len(self.tokens) >= self.max_new:
                raise RuntimeError("commit outside a running job or beyond max_new")
            self.tokens.append(token)
            self.changed.notify_all()

    def snapshot(self, *, include_tokens=False):
        with self.changed:
            body = {"id": self.id, "state": self.state, "created": self.created_unix,
                    "prompt_tokens": self.prompt_tokens, "committed_tokens": len(self.tokens),
                    "max_new": self.max_new, "cancel_requested": self.cancelled.is_set(),
                    "error": self.error, "error_code": self.error_code}
            if include_tokens:
                body["tokens"] = list(self.tokens)
            return body


class ServiceQueue:
    """One worker/one ring, with round-robin tenants and bounded retained results."""

    def __init__(self, backend, tenants, *, max_queue=32, max_queued_tokens=131072,
                 max_history=128, retention_s=600.0, clock=time.monotonic):
        for value in (max_queue, max_queued_tokens, max_history):
            if type(value) is not int or value < 1:
                raise ValueError("queue, token and history bounds must be positive integers")
        self.backend, self.tenants, self.clock = backend, dict(tenants), clock
        self.max_queue, self.max_queued_tokens, self.max_history = max_queue, max_queued_tokens, max_history
        self.retention_s = float(retention_s)
        if not math.isfinite(self.retention_s) or self.retention_s <= 0:
            raise ValueError("retention_s must be finite and positive")
        self._condition = threading.Condition()
        self._queues, self._round = {}, deque()
        self._jobs, self._idempotency = {}, {}
        self._windows = {tenant: deque() for tenant in self.tenants}
        self._active = {tenant: 0 for tenant in self.tenants}
        self._queued_count = self._queued_tokens = 0
        self._running = None
        self._accepting, self._stopping = True, False
        self._worker = self._sweeper = None
        self._counts = {name: 0 for name in ("submitted", "deduplicated", "completed", "failed",
                                            "cancelled", "expired", "rejected")}

    def start(self):
        with self._condition:
            if self._worker is not None:
                return
            self._worker = threading.Thread(target=self._run, name="inference-serial-worker", daemon=True)
            self._sweeper = threading.Thread(target=self._sweep, name="inference-deadlines", daemon=True)
            self._worker.start(); self._sweeper.start()

    def _prune(self, now):
        terminal = sorted((j for j in self._jobs.values() if j.terminal), key=lambda j: j.finished_mono)
        excess = max(0, len(terminal) - self.max_history)
        for index, job in enumerate(terminal):
            if index < excess or now - job.finished_mono >= self.retention_s:
                self._jobs.pop(job.id, None)
                if job.idempotency_key is not None:
                    self._idempotency.pop((job.tenant, job.idempotency_key), None)

    def submit(self, tenant, payload, *, prompt_tokens, max_new, timeout_s=600.0, idempotency_key=None):
        if tenant not in self.tenants:
            raise AdmissionError("unknown tenant", status=403, code="permission_denied")
        if not all(type(v) is int and v >= 0 for v in (prompt_tokens, max_new)) or max_new < 1:
            raise AdmissionError("invalid token budget", status=400, code="invalid_request_error")
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise AdmissionError("invalid deadline", status=400, code="invalid_request_error")
        if idempotency_key is not None and (not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 128):
            raise AdmissionError("idempotency key must have 1..128 characters", status=400, code="invalid_request_error")
        try:
            normalized = json.dumps({"payload": payload, "prompt_tokens": prompt_tokens, "max_new": max_new,
                                     "timeout_s": timeout_s}, sort_keys=True, separators=(",", ":"), allow_nan=False)
            payload = json.loads(json.dumps(payload, allow_nan=False))  # detach mutable caller data
        except (ValueError, TypeError):
            raise AdmissionError("request must be finite JSON data", status=400, code="invalid_request_error") from None
        fingerprint = hashlib.sha256(normalized.encode()).hexdigest()
        charge, now = prompt_tokens + max_new, self.clock()
        with self._condition:
            self._prune(now)
            prior = self._idempotency.get((tenant, idempotency_key)) if idempotency_key is not None else None
            if prior is not None:
                job = self._jobs[prior]
                if job.fingerprint != fingerprint:
                    raise AdmissionError("idempotency key names another request", status=409, code="idempotency_conflict")
                self._counts["deduplicated"] += 1
                return job, False
            if not self._accepting:
                raise AdmissionError("service is draining", status=503, code="service_unavailable")
            window = self._windows[tenant]
            while window and window[0][0] <= now - 60:
                window.popleft()
            limits = self.tenants[tenant]
            if (self._active[tenant] >= limits.max_active or len(window) >= limits.requests_per_minute
                    or sum(record[1] for record in window) + charge > limits.tokens_per_minute
                    or self._queued_count >= self.max_queue or self._queued_tokens + charge > self.max_queued_tokens):
                self._counts["rejected"] += 1
                raise AdmissionError("tenant quota or queue capacity exceeded")
            job = Job(tenant, payload, prompt_tokens, max_new, now + timeout_s,
                      fingerprint, idempotency_key, now)
            self._jobs[job.id] = job
            if idempotency_key is not None:
                self._idempotency[(tenant, idempotency_key)] = job.id
            queue = self._queues.setdefault(tenant, deque())
            if not queue:
                self._round.append(tenant)
            queue.append(job)
            self._active[tenant] += 1
            self._queued_count += 1; self._queued_tokens += charge
            window.append((now, charge))
            self._counts["submitted"] += 1
            self._condition.notify_all()
            return job, True

    def get(self, tenant, job_id):
        with self._condition:
            self._prune(self.clock())
            job = self._jobs.get(job_id)
            return job if job is not None and job.tenant == tenant else None

    def cancel(self, tenant, job_id, *, reason="cancel"):
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None or job.tenant != tenant:
                return False
            if job.terminal:
                return True
            job.stop_kind = "deadline" if reason == "deadline" else "cancel"
            job.cancelled.set()
            if job.state == "queued":
                self._finish(job, "expired" if reason == "deadline" else "cancelled",
                             error="job deadline exceeded" if reason == "deadline" else "job cancelled",
                             code="deadline_exceeded" if reason == "deadline" else "job_cancelled")
            with job.changed:
                job.changed.notify_all()
            self._condition.notify_all()
            return True

    def cancel_if_unobserved(self, tenant, job_id):
        """Atomically honor disconnect only if no idempotent/reconnecting client owns the job."""
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None or job.tenant != tenant:
                return False
            with job.changed:
                if job.clients != 0 or job.terminal:
                    return False
                return self.cancel(tenant, job_id)

    def _finish(self, job, state, *, result=None, error=None, code=None):
        if job.terminal:
            return
        if job.state == "queued":
            self._queued_count -= 1
            self._queued_tokens -= job.prompt_tokens + job.max_new
            queue = self._queues.get(job.tenant)
            if queue is not None:
                # Remove tombstones immediately: submit/cancel floods must not grow
                # an unbounded deque behind one long-running model request.
                self._queues[job.tenant] = deque(item for item in queue if item is not job)
                if not self._queues[job.tenant]:
                    self._round = deque(tenant for tenant in self._round if tenant != job.tenant)
        self._active[job.tenant] -= 1
        with job.changed:
            job.result, job.error, job.error_code = result, error, code
            job.finished_mono = self.clock()
            job.payload = {}  # release private prompt/messages on every terminal transition
            job.state = state  # publish terminal state only after its result/error exists
            job.changed.notify_all()
        self._counts[state] += 1

    def _next(self):
        while self._round:
            tenant = self._round.popleft()
            queue = self._queues[tenant]
            while queue and queue[0].terminal:
                queue.popleft()
            if not queue:
                continue
            job = queue.popleft()
            if queue:
                self._round.append(tenant)
            if self.clock() >= job.deadline:
                self._finish(job, "expired", error="job deadline exceeded", code="deadline_exceeded")
                continue
            self._queued_count -= 1
            self._queued_tokens -= job.prompt_tokens + job.max_new
            with job.changed:
                job.state = "running"
                job.changed.notify_all()
            return job
        return None

    def _run(self):
        while True:
            with self._condition:
                job = self._next()
                while job is None and not self._stopping:
                    self._condition.wait(0.2)
                    job = self._next()
                if job is None and self._stopping:
                    return
                self._running = job
            try:
                result = self.backend.execute(job, job.commit, lambda: job.check_stop(self.clock()))
                if not isinstance(result, dict) or result.get("tokens") != job.checkpoint():
                    raise RuntimeError("backend result does not match committed callback frontier")
                job.check_stop(self.clock())
                with self._condition:
                    job.check_stop(self.clock())  # cancellation may have won the terminal-state lock
                    self._finish(job, "completed", result=result)
            except JobExpired as error:
                with self._condition:
                    self._finish(job, "expired", error=str(error), code="deadline_exceeded")
            except JobCancelled as error:
                with self._condition:
                    self._finish(job, "cancelled", error=str(error), code="job_cancelled")
            except Exception as error:
                with self._condition:
                    if job.cancelled.is_set() or self.clock() >= job.deadline:
                        expired = job.stop_kind == "deadline" or self.clock() >= job.deadline
                        self._finish(job, "expired" if expired else "cancelled",
                                     error="job deadline exceeded" if expired else "job cancelled",
                                     code="deadline_exceeded" if expired else "job_cancelled")
                    else:
                        self._finish(job, "failed", error=f"{type(error).__name__}: {str(error)[:240]}",
                                     code="inference_failed")
            finally:
                with self._condition:
                    self._running = None
                    self._prune(self.clock())
                    self._condition.notify_all()

    def _sweep(self):
        while True:
            with self._condition:
                if self._stopping:
                    return
                now = self.clock()
                due = [job for job in self._jobs.values() if not job.terminal and now >= job.deadline]
            for job in due:
                self.cancel(job.tenant, job.id, reason="deadline")
            with self._condition:
                self._prune(self.clock())
                self._condition.wait(0.05)

    def metrics(self, tenant=None):
        with self._condition:
            body = {"mode": "serial_one_ring", "accepting": self._accepting,
                    "queued": self._queued_count, "queued_reserved_tokens": self._queued_tokens,
                    "running": int(self._running is not None), "retained_jobs": len(self._jobs),
                    "counts": dict(self._counts), "idempotency_retention_s": self.retention_s}
            if tenant in self._active:
                body["tenant_active"] = self._active[tenant]
            return body

    def shutdown(self, *, drain=False, timeout=10.0):
        with self._condition:
            self._accepting = False
            if not drain:
                for job in list(self._jobs.values()):
                    if not job.terminal:
                        self.cancel(job.tenant, job.id)
            self._stopping = True
            self._condition.notify_all()
        if not drain and hasattr(self.backend, "abort"):
            self.backend.abort()
        if self._worker is not None:
            self._worker.join(timeout)
        if self._worker is not None and self._worker.is_alive():
            with self._condition:
                for job in list(self._jobs.values()):
                    if not job.terminal:
                        self.cancel(job.tenant, job.id)
            if hasattr(self.backend, "abort"):
                self.backend.abort()
            self._worker.join(min(timeout, 2.0))
        if self._sweeper is not None:
            self._sweeper.join(min(timeout, 1.0))
        if hasattr(self.backend, "close"):
            self.backend.close()
        return self._worker is None or not self._worker.is_alive()

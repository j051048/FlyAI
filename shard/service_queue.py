"""Bounded, tenant-fair serial inference jobs. No model, GPU, HTTP or billing dependencies.

Admission reserves prompt+maximum completion tokens in a rolling minute. That is
a resource quota, not an invoice. Idempotency is in-memory by default. An explicitly
configured private journal adds owner-fenced original-request replay recovery,
not KV migration or continuous batching.
"""
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import math
import threading
import time
import uuid
from contextlib import nullcontext, contextmanager

try:
    from .job_journal import (JobJournal, JournalError, JournalOwnerLost, JournalBusy, JournalCapacity, JournalConflict, JournalDeadline)
except ImportError:
    try:
        from job_journal import (JobJournal, JournalError, JournalOwnerLost, JournalBusy, JournalCapacity, JournalConflict, JournalDeadline)
    except ImportError:
        # Legacy copied gateway/help entrypoints remain usable with durability
        # disabled. Configuring persistence requires the full journal module.
        class JournalError(RuntimeError):
            pass
        class JournalOwnerLost(JournalError):
            pass
        class JournalBusy(JournalError):
            pass
        class JournalCapacity(JournalError):
            pass
        class JournalConflict(JournalError):
            pass
        class JournalDeadline(JournalConflict):
            pass
        class JobJournal:
            def __init__(self, *args, **kwargs):
                raise JournalError("durable jobs require a full checkout containing shard/job_journal.py")


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
    _journal: object = field(default=None, repr=False)
    _journal_owner: object = field(default=None, repr=False)
    _durable_context_value: dict | None = field(default=None, repr=False)
    _publication_check: object = field(default=None, repr=False)
    recovered: bool = False

    @property
    def terminal(self):
        return self.state in ("completed", "failed", "cancelled", "expired")

    def checkpoint(self):
        self.assert_publishable()
        with self.changed:
            return list(self.tokens)

    def assert_publishable(self):
        if self._publication_check is not None:
            self._publication_check()
        elif self._journal is not None:
            self._journal.assert_owner(self._journal_owner)

    def publish_guard(self):
        return self._publication_guard() if self._journal is not None else nullcontext()

    @contextmanager
    def _publication_guard(self):
        self.assert_publishable()
        with self._journal.publish_guard(self._journal_owner):
            self.assert_publishable()
            yield

    def check_stop(self, now=None):
        self.assert_publishable()
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
            if self._journal is not None:
                self._journal.append_token(self._journal_owner, self.id, len(self.tokens), token)
            self.tokens.append(token)
            self.changed.notify_all()

    def snapshot(self, *, include_tokens=False):
        self.assert_publishable()
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
                 max_history=128, retention_s=600.0, clock=time.monotonic,
                 journal=None, journal_owner_ttl_s=30.0, recovery_resolver=None):
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
        if journal is not None and not isinstance(journal, JobJournal):
            raise ValueError("journal must be a configured JobJournal")
        if recovery_resolver is not None and not callable(recovery_resolver):
            raise ValueError("recovery_resolver must be callable")
        if type(journal_owner_ttl_s) not in (int, float) or not math.isfinite(journal_owner_ttl_s) or journal_owner_ttl_s < 0.2:
            raise ValueError("journal owner TTL must be finite and at least 0.2 seconds")
        self.journal, self.recovery_resolver = journal, recovery_resolver
        self._journal_ttl = float(journal_owner_ttl_s)
        self._journal_owner, self._journal_thread = None, None
        self._journal_stop, self._journal_failed = threading.Event(), None
        self._recovering = set()
        self._recovery_claiming = set()
        self._preserve_pending = False
        if journal is not None:
            self._journal_owner = journal.claim_owner(ttl_s=self._journal_ttl)
            self._journal_thread = threading.Thread(target=self._renew_journal, name="job-journal-owner", daemon=True)
            self._journal_thread.start()
            try:
                self._restore_journal()
            except Exception:
                self._journal_stop.set()
                self._journal_thread.join(1)
                journal.release_owner(self._journal_owner)
                raise

    def assert_owner(self):
        if self.journal is not None:
            if self._journal_failed is not None:
                raise JournalOwnerLost("local durable coordinator was fenced")
            self.journal.assert_owner(self._journal_owner)

    def publish_guard(self):
        return self._publication_guard() if self.journal is not None else nullcontext()

    @contextmanager
    def _publication_guard(self):
        self.assert_owner()
        with self.journal.publish_guard(self._journal_owner):
            self.assert_owner()
            yield

    def _renew_journal(self):
        while not self._journal_stop.wait(max(0.05, self._journal_ttl / 3)):
            try:
                self._journal_owner = self.journal.renew(self._journal_owner, ttl_s=self._journal_ttl)
            except JournalBusy as error:
                if self.journal.clock() < self._journal_owner.expires_at:
                    continue
                self._journal_fenced(error)
                return
            except JournalError as error:
                self._journal_fenced(error)
                return

    def _journal_fenced(self, error):
        with self._condition:
            self._journal_failed = type(error).__name__
            self._accepting, self._stopping = False, True
            for job in self._jobs.values():
                if not job.terminal:
                    job.cancelled.set()
                    with job.changed:
                        job.changed.notify_all()
            self._condition.notify_all()
        if hasattr(self.backend, "abort"):
            self.backend.abort()

    def _durable_context(self, job):
        return json.loads(json.dumps(job._durable_context_value, allow_nan=False)) if job._durable_context_value is not None else None

    def _attach_journal(self, job, context):
        job._journal, job._journal_owner = self.journal, self._journal_owner
        job._durable_context_value = context
        job._publication_check = self.assert_owner

    def _restore_journal(self):
        rows, wall = self.journal.claim_jobs(self._journal_owner)
        now = self.clock()
        for row in rows:
            if row["tenant"] not in self.tenants:
                if row["state"] not in ("completed", "failed", "cancelled", "expired"):
                    self.journal.finish(self._journal_owner, row["id"], "failed",
                                        error="tenant authorization was removed", code="permission_denied")
                continue
            terminal = row["state"] in ("completed", "failed", "cancelled", "expired")
            job = Job(row["tenant"], {} if terminal else row["payload"], row["prompt"], row["maximum"],
                now + max(0, row["deadline"] - wall), row["fingerprint"], row["idem"], now,
                id=row["id"], created_unix=int(row["created"]), state=row["state"] if terminal else "recovering",
                tokens=row["tokens"], result=row["result"], error=row["error"], error_code=row["error_code"],
                stop_kind=row["cancel_kind"], finished_mono=(now - max(0, wall - row["finished"])) if terminal else None,
                recovered=not terminal)
            self._attach_journal(job, row["context"] if not terminal else {
                k: v for k, v in row["context"].items() if k != "original_request"})
            job._durable_record = row if not terminal else None
            self._jobs[job.id] = job
            if job.idempotency_key is not None:
                self._idempotency[(job.tenant, job.idempotency_key)] = job.id
            if not terminal:
                self._active[job.tenant] += 1
                self._queued_count += 1
                self._queued_tokens += job.prompt_tokens + job.max_new
                self._recovering.add(job.id)
                if row["cancel_kind"] or now >= job.deadline:
                    expired = row["cancel_kind"] == "deadline" or now >= job.deadline
                    self._finish(job, "expired" if expired else "cancelled",
                        error="job deadline exceeded" if expired else "job cancelled",
                        code="deadline_exceeded" if expired else "job_cancelled")
        records, wall = self.journal.rate_records(self._journal_owner)
        for record in records:
            if record["tenant"] in self._windows:
                self._windows[record["tenant"]].append((now - max(0, wall - record["created"]), record["charge"]))

    def recover_pending(self):
        """Resolve same-cohort backends without extending the original deadline."""
        if self.journal is None:
            return 0
        self.assert_owner()
        recovered = 0
        for job_id in sorted(tuple(self._recovering), key=lambda ident: (
                (self._jobs[ident]._durable_record or {}).get("arrival", 0), ident)):
            with self._condition:
                job = self._jobs.get(job_id)
                if job is None or job.terminal:
                    self._recovering.discard(job_id)
                    continue
                if self.clock() >= job.deadline:
                    self.cancel(job.tenant, job.id, reason="deadline")
                    continue
                if job_id in self._recovery_claiming:
                    continue
                self._recovery_claiming.add(job_id)
                record = json.loads(json.dumps(job._durable_record, allow_nan=False))
            resolved, accepted = None, False
            try:
                if self.recovery_resolver is not None:
                    resolved = self.recovery_resolver(record)
                else:
                    context = record["context"]
                    if (getattr(self.backend, "model_id", None), getattr(self.backend, "cohort_id", None)) != (context["model_id"], context["cohort_id"]):
                        continue
                    resolved = {"payload": record["payload"], "prompt_tokens": record["prompt"], "max_new": record["maximum"], "bind": {},
                                "replay_contract": getattr(self.backend, "prefix_replay_contract", None)}
                if resolved is None:
                    continue
                if (not isinstance(resolved, dict) or resolved.get("payload") != record["payload"]
                        or resolved.get("prompt_tokens") != record["prompt"] or resolved.get("max_new") != record["maximum"]
                        or not isinstance(resolved.get("bind", {}), dict)
                        or (resolved.get("cleanup") is not None and not callable(resolved["cleanup"]))):
                    raise JournalConflict("recovery must reprepare the exact original payload and budget")
                bindings = resolved.get("bind", {})
                if set(bindings) - {"ring_binding", "bound_backend", "served_model", "served_cohort", "request_fingerprint", "conversation_session_id"}:
                    raise JournalError("unexpected recovered binding attribute")
                if bindings.get("served_cohort", record["context"]["cohort_id"]) != record["context"]["cohort_id"]:
                    raise JournalConflict("recovery backend belongs to another model cohort")
                replay = resolved.get("replay_contract", getattr(bindings.get("bound_backend", self.backend), "prefix_replay_contract", None))
                backend = bindings.get("bound_backend", self.backend)
                supports = getattr(backend, "supports_prefix_replay", False) is True
                if record["tokens"] and replay != "original-request-prefix-check/1" and not supports:
                    with self._condition:
                        self._finish(job, "failed", error="backend cannot safely replay the committed prefix", code="recovery_contract_missing")
                    continue
                with self._condition:
                    if job.terminal or job_id not in self._recovering or self._stopping:
                        continue
                    self.assert_owner()
                    for key, value in bindings.items():
                        setattr(job, key, value)
                    job.state = "queued"
                    queue = self._queues.setdefault(job.tenant, deque())
                    if not queue:
                        self._round.append(job.tenant)
                    queue.append(job)
                    self._recovering.remove(job_id)
                    accepted = True
                    recovered += 1
                    self._condition.notify_all()
            finally:
                with self._condition:
                    self._recovery_claiming.discard(job_id)
                if not accepted and isinstance(resolved, dict) and callable(resolved.get("cleanup")):
                    resolved["cleanup"]()
        return recovered

    def durable_lookup(self, tenant, idempotency_key, *, request_fingerprint=None):
        if tenant not in self.tenants:
            raise AdmissionError("unknown tenant", status=403, code="permission_denied")
        with self._condition:
            self.assert_owner()
            job_id = self._idempotency.get((tenant, idempotency_key))
            job = self._jobs.get(job_id)
            if job is not None and request_fingerprint is not None:
                actual = (job._durable_context_value or {}).get("request_fingerprint", job.fingerprint)
                if actual != request_fingerprint:
                    raise AdmissionError("idempotency key names another request", status=409, code="idempotency_conflict")
            return job

    def _mark_running(self, job):
        if self.journal is not None:
            self.journal.running(self._journal_owner, job.id)
        with job.changed:
            job.state = "running"
            job.changed.notify_all()

    def start(self):
        self.recover_pending()
        with self._condition:
            if self._worker is not None:
                return
            self._worker = threading.Thread(target=self._run, name="inference-serial-worker", daemon=True)
            self._sweeper = threading.Thread(target=self._sweep, name="inference-deadlines", daemon=True)
            self._worker.start(); self._sweeper.start()

    def _prune(self, now):
        if self.journal is not None:
            self.journal.prune(self._journal_owner, max_terminal=self.max_history)
        terminal = sorted((j for j in self._jobs.values() if j.terminal), key=lambda j: j.finished_mono)
        excess = max(0, len(terminal) - self.max_history)
        for index, job in enumerate(terminal):
            if index < excess or now - job.finished_mono >= self.retention_s:
                self._jobs.pop(job.id, None)
                if job.idempotency_key is not None:
                    self._idempotency.pop((job.tenant, job.idempotency_key), None)

    def submit(self, tenant, payload, *, prompt_tokens, max_new, timeout_s=600.0, idempotency_key=None,
               durable_context=None):
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
        deadline_unix = self.journal.clock() + timeout_s if self.journal is not None else None
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
            if self.journal is not None:
                context = durable_context or {"model_id": getattr(self.backend, "model_id", None),
                    "cohort_id": getattr(self.backend, "cohort_id", None), "original_request": payload}
                context = json.loads(json.dumps(context, allow_nan=False))
                try:
                    record, fresh = self.journal.admit(self._journal_owner, {"id": job.id, "tenant": tenant,
                        "payload": payload, "context": context, "prompt_tokens": prompt_tokens,
                        "max_new": max_new, "timeout_s": timeout_s, "fingerprint": fingerprint,
                        "idempotency_key": idempotency_key, "deadline_unix": deadline_unix})
                except JournalDeadline as error:
                    raise AdmissionError(str(error), status=408, code="deadline_exceeded") from error
                except JournalConflict as error:
                    raise AdmissionError(str(error), status=409, code="idempotency_conflict") from error
                except JournalCapacity as error:
                    raise AdmissionError(str(error), status=429, code="journal_capacity") from error
                if not fresh:
                    raise JournalConflict("durable row not restored in the current queue")
                self._attach_journal(job, context)
                job.created_unix = int(record["created"])
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
            self.assert_owner()
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
            if self.journal is not None:
                self.journal.request_cancel(self._journal_owner, job.id, "deadline" if reason == "deadline" else "cancel")
            job.stop_kind = "deadline" if reason == "deadline" else "cancel"
            job.cancelled.set()
            if job.state in ("queued", "recovering"):
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
        if self._preserve_pending:
            return
        if self.journal is not None:
            try:
                self.journal.finish(self._journal_owner, job.id, state, result=result, error=error, code=code)
            except JournalDeadline:
                state, result, error, code = "expired", None, "job deadline exceeded", "deadline_exceeded"
                self.journal.finish(self._journal_owner, job.id, state, error=error, code=code)
            except JournalConflict:
                row = self.journal.get(self._journal_owner, tenant=job.tenant, job_id=job.id)
                if state != "completed" or row is None or row["cancel_kind"] not in ("cancel", "deadline"):
                    raise
                expired = row["cancel_kind"] == "deadline"
                state, result = ("expired" if expired else "cancelled"), None
                error, code = ("job deadline exceeded", "deadline_exceeded") if expired else ("job cancelled", "job_cancelled")
                self.journal.finish(self._journal_owner, job.id, state, error=error, code=code)
            except JournalOwnerLost as lost:
                self._journal_fenced(lost)
                return
        self._recovering.discard(job.id)
        if job.state in ("queued", "recovering"):
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
            job._durable_record = None
            if job._durable_context_value is not None:
                job._durable_context_value = {k: v for k, v in job._durable_context_value.items() if k != "original_request"}
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
            try:
                self._mark_running(job)
            except JournalConflict as error:
                self._finish(job, "failed", error=str(error), code="durable_start_rejected")
                continue
            except JournalError as error:
                self._journal_fenced(error)
                return None
            self._queued_count -= 1
            self._queued_tokens -= job.prompt_tokens + job.max_new
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
                    if isinstance(error, JournalError):
                        self._journal_fenced(error)
                        continue
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
            if self.journal is not None and self._recovering:
                try:
                    self.recover_pending()
                except JournalError as error:
                    self._journal_fenced(error)
                    return
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
            body["durable_journal"] = self.journal is not None
            if self.journal is not None:
                body["durable_owner_fence"] = self._journal_owner.fence
                body["recovering"] = len(self._recovering)
                body["journal_failure"] = self._journal_failed
                body["journal_io"] = self.journal.metrics()
            return body

    def shutdown(self, *, drain=False, timeout=10.0, preserve_pending=False):
        if type(preserve_pending) is not bool or (preserve_pending and self.journal is None):
            raise ValueError("preserve_pending requires an explicitly configured journal")
        self._preserve_pending = preserve_pending or self._journal_failed is not None
        with self._condition:
            self._accepting = False
            if self._preserve_pending:
                for job in self._jobs.values():
                    if not job.terminal:
                        job.cancelled.set()
            elif not drain:
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
                if not self._preserve_pending:
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
        stopped = self._worker is None or not self._worker.is_alive()
        if stopped and self.journal is not None:
            self._journal_stop.set()
            if self._journal_thread is not None:
                self._journal_thread.join(min(timeout, 1))
            try:
                self.journal.release_owner(self._journal_owner)
            except JournalOwnerLost:
                pass
            self.journal.checkpoint()
        return stopped

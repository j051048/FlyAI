"""Global tenant admission with one serial worker per independently leased ring.

This is request-level concurrency across rings, not continuous batching. Jobs are
bound at admission; alias changes never move an existing job or its tokenizer.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict

try:
    from shard.service_queue import ServiceQueue, AdmissionError, JobCancelled, JobExpired
    from shard.ring_pool import RingUnavailable, RingState
except ImportError:
    from service_queue import ServiceQueue, AdmissionError, JobCancelled, JobExpired
    from ring_pool import RingUnavailable, RingState


class RingRouter:
    def __init__(self, pool, *, default_model=None):
        self.pool = pool
        models = pool.models()
        self.model_id = default_model or (models[0] if len(models) == 1 else None)
        if self.model_id is None:
            raise ValueError("multiple served models require an explicit default_model")
        pool.resolve(self.model_id)

    def ready(self):
        return self.pool.ready()

    def stats(self):
        return self.pool.snapshot()

    def models(self):
        return self.pool.models()

    def abort(self):
        for ring in self.pool.rings():
            ring.backend.abort()


class SessionAffinity:
    """Tenant/cohort scoped, bounded hints for *new* ring admissions."""
    def __init__(self, *, max_entries=1024, ttl_s=300.0, clock=time.monotonic):
        import math
        if type(max_entries) is not int or max_entries < 1 or type(ttl_s) not in (int, float) or not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ValueError("bounded affinity capacity and finite positive TTL required")
        self.max_entries, self.ttl_s, self.clock = max_entries, ttl_s, clock
        self._entries, self._lock = OrderedDict(), threading.RLock()

    def _prune(self):
        for key, (_, expires) in tuple(self._entries.items()):
            if self.clock() >= expires:
                self._entries.pop(key)

    def preferred(self, tenant, cohort_id, session_id):
        with self._lock:
            self._prune()
            key = (tenant, cohort_id, session_id)
            value = self._entries.get(key)
            if value is None:
                return None
            self._entries.move_to_end(key)
            return value[0]

    def remember(self, tenant, cohort_id, session_id, ring_id):
        if not all(isinstance(v, str) and v for v in (tenant, cohort_id, session_id, ring_id)):
            raise ValueError("complete affinity identity required")
        with self._lock:
            self._prune()
            key = (tenant, cohort_id, session_id)
            self._entries[key] = (ring_id, self.clock() + self.ttl_s)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)


class MultiRingQueue(ServiceQueue):
    """Shares quota/idempotency/history globally; serial execution is per ring."""
    def __init__(self, router, tenants, *, session_affinity=False, affinity_max_entries=1024,
                 affinity_ttl_s=300.0, batching=None, scheduling=None, **options):
        if type(session_affinity) is not bool:
            raise ValueError("session_affinity must be an explicit boolean")
        self.router = router
        if options.get("journal") is not None:
            if options.get("recovery_resolver") is not None:
                raise ValueError("multi-ring recovery uses the pinned cohort resolver")
            options["recovery_resolver"] = self._resolve_recovered_job
        from .continuous_batch import batching_options, backend_contract
        self._batching = batching_options(batching)
        if self._batching["enabled"]:
            for ring in router.pool.rings():
                if ring.state == RingState.READY:
                    backend_contract(ring)
        scheduling = scheduling or {}
        if not isinstance(scheduling, dict) or set(scheduling) - {"policy", "aging_s"}:
            raise ValueError("unknown scheduling options")
        import math
        self._scheduling = scheduling.get("policy", "fifo")
        self._aging_s = scheduling.get("aging_s", 10.0)
        if self._scheduling not in ("fifo", "shortest_with_aging") or type(self._aging_s) not in (int, float) or not math.isfinite(self._aging_s) or self._aging_s <= 0:
            raise ValueError("invalid finite scheduling policy")
        affinity = (SessionAffinity(max_entries=affinity_max_entries, ttl_s=affinity_ttl_s,
            clock=options.get("clock", time.monotonic)) if session_affinity else None)
        super().__init__(router, tenants, **options)
        self._ring_workers = {}
        self._running_by_ring = {}
        self._session_affinity = affinity
        for job in self._jobs.values():
            context = self._durable_context(job) or {}
            if context:
                job.request_fingerprint = context.get("request_fingerprint", job.fingerprint)
                job.served_model, job.served_cohort = context.get("model_id"), context.get("cohort_id")
                if job.terminal:
                    self._bind_terminal_decoder(job)

    def _bind_terminal_decoder(self, job):
        for ring in self.router.pool.rings():
            if ring.model_id == job.served_model and ring.cohort_id == job.served_cohort:
                job.bound_backend = ring.backend
                return

    def _resolve_recovered_job(self, record):
        context = record["context"]
        try:
            model, cohort = self.router.pool.resolve(context["model_id"], context["cohort_id"])
            binding = self.router.pool.acquire(model, cohort, estimated_tokens=record["prompt"] + record["maximum"])
        except RingUnavailable:
            return None
        try:
            body = dict(context["original_request"], model=model)
            for name in ("shard_cohort", "shard_region", "shard_session_id"):
                body.pop(name, None)
            payload, prompt, maximum, _ = binding.backend.prepare(body)
            if payload != record["payload"] or prompt != record["prompt"] or maximum != record["maximum"]:
                raise ValueError("recovery preparation differs from the persisted request")
            return {"payload": payload, "prompt_tokens": prompt, "max_new": maximum,
                "bind": {"ring_binding": binding, "bound_backend": binding.backend,
                    "served_model": model, "served_cohort": cohort,
                    "request_fingerprint": context["request_fingerprint"],
                    "conversation_session_id": context.get("session_id")},
                "cleanup": lambda: self.router.pool.release(binding)}
        except Exception:
            self.router.pool.release(binding)
            raise

    def _journal_fenced(self, error):
        # The replacement owner retains the durable pending row. This process
        # can only stop its local execution and release verified local bindings.
        self._preserve_pending = True
        super()._journal_fenced(error)

    def start(self):
        self.recover_pending()
        with self._condition:
            if self._worker is not None:
                return
            self._worker = threading.Thread(target=self._dispatch, name="ring-worker-discovery", daemon=True)
            self._sweeper = threading.Thread(target=self._sweep, name="multi-ring-deadlines", daemon=True)
            self._worker.start(); self._sweeper.start()

    def _dispatch(self):
        while True:
            with self._condition:
                if self._stopping:
                    return
                for ring_id, worker in tuple(self._ring_workers.items()):
                    if not worker.is_alive():
                        self._ring_workers.pop(ring_id, None)
                for ring in self.router.pool.rings():
                    if ring.state == RingState.FAILED and not ring.bound_jobs:
                        continue
                    if ring.state != RingState.STOPPED and ring.ring_id not in self._ring_workers:
                        worker = threading.Thread(target=self._run_ring, args=(ring,),
                                                  name="ring-" + ring.ring_id, daemon=True)
                        self._ring_workers[ring.ring_id] = worker
                        worker.start()
                self._condition.wait(0.1)

    def submit_request(self, tenant, body, *, idempotency_key=None):
        if tenant not in self.tenants:
            raise AdmissionError("unknown tenant", status=403, code="permission_denied")
        if idempotency_key is not None and (not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 128):
            raise AdmissionError("idempotency key must have 1..128 characters", status=400, code="invalid_request_error")
        if not isinstance(body, dict):
            raise AdmissionError("request must be a JSON object", status=400, code="invalid_request_error")
        try:
            request = json.loads(json.dumps(body, allow_nan=False))
            fingerprint_body = {key: value for key, value in request.items() if key != "stream"}
            raw_fingerprint = hashlib.sha256(json.dumps(fingerprint_body, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        except (ValueError, TypeError):
            raise AdmissionError("request must be finite JSON data", status=400, code="invalid_request_error") from None
        with self._condition:
            self._prune(self.clock())
            prior = self._idempotency.get((tenant, idempotency_key)) if idempotency_key is not None else None
            if prior is not None:
                job = self._jobs[prior]
                if job.request_fingerprint != raw_fingerprint:
                    raise AdmissionError("idempotency key names another request", status=409, code="idempotency_conflict")
                self._counts["deduplicated"] += 1
                if job.terminal:
                    self._bind_terminal_decoder(job)
                return job, False
            model = request.get("model", self.router.model_id)
            cohort, region = request.get("shard_cohort"), request.get("shard_region")
            session_id = request.get("shard_session_id")
            if session_id is not None and (not isinstance(session_id, str) or not 1 <= len(session_id) <= 128):
                raise AdmissionError("session id must have 1..128 characters", status=400, code="invalid_request_error")
            if not isinstance(model, str) or (cohort is not None and not isinstance(cohort, str)) or (
                    region is not None and (not isinstance(region, str) or not region)):
                raise AdmissionError("invalid model/cohort/region", status=400, code="invalid_request_error")
            try:
                model_id, cohort_id = self.router.pool.resolve(model, cohort)
                maximum = request.get("max_tokens", request.get("max_completion_tokens", 512))
                if type(maximum) is not int or not 1 <= maximum <= 4096:
                    raise AdmissionError("invalid completion token budget", status=400, code="invalid_request_error")
                preference = (self._session_affinity.preferred(tenant, cohort_id, session_id)
                    if self._session_affinity is not None and session_id is not None else None)
                acquire_options = {"estimated_tokens": maximum, "region": region}
                if preference is not None:
                    acquire_options["preferred_ring_id"] = preference
                binding = self.router.pool.acquire(model_id, cohort_id, **acquire_options)
            except RingUnavailable as error:
                raise AdmissionError(str(error), status=503, code="no_ready_ring") from error
            try:
                prepared_body = dict(request, model=model_id)
                prepared_body.pop("shard_cohort", None); prepared_body.pop("shard_region", None)
                prepared_body.pop("shard_session_id", None)
                payload, count, maximum, timeout_s = binding.backend.prepare(prepared_body)
                if self._batching["enabled"]:
                    from .continuous_batch import backend_contract
                    contract = backend_contract(binding.ring)
                    if count > contract["max_prompt_tokens"] or maximum > contract["max_new_tokens"] or count + maximum > contract["max_context_tokens"]:
                        raise AdmissionError("prompt exceeds qualified batch state capacity", status=400, code="context_length_exceeded")
                self.router.pool.update_estimate(binding, count + maximum)
                job, fresh = super().submit(tenant, payload, prompt_tokens=count, max_new=maximum,
                    timeout_s=timeout_s, idempotency_key=idempotency_key,
                    durable_context={"model_id": model_id, "cohort_id": cohort_id,
                        "original_request": request, "request_fingerprint": raw_fingerprint,
                        "session_id": session_id})
                # The worker cannot see the newly enqueued job until this shared
                # admission lock is released. Backend/tokenizer remain fixed forever.
                job.ring_binding = binding
                job.served_model = model
                job.served_cohort = cohort_id
                job.bound_backend = binding.backend
                job.request_fingerprint = raw_fingerprint
                job.conversation_session_id = session_id
                if self._session_affinity is not None and session_id is not None:
                    self._session_affinity.remember(tenant, cohort_id, session_id, binding.ring.ring_id)
                self._condition.notify_all()
                return job, fresh
            except Exception:
                self.router.pool.release(binding)
                raise

    def _next_for_ring(self, ring):
        for _ in range(len(self._round)):
            tenant = self._round.popleft()
            queue = self._queues[tenant]
            candidates = [item for item in queue if not item.terminal and item.ring_binding.ring is ring]
            job = candidates[0] if candidates else None
            if candidates and self._scheduling == "shortest_with_aging":
                aged = [item for item in candidates if self.clock() - item.created_mono >= self._aging_s]
                job = min(aged, key=lambda item: item.created_mono) if aged else min(candidates,
                    key=lambda item: (item.prompt_tokens + item.max_new, item.created_mono))
            if job is None:
                if queue:
                    self._round.append(tenant)
                continue
            queue.remove(job)
            if queue:
                self._round.append(tenant)
            if self.clock() >= job.deadline:
                self._finish(job, "expired", error="job deadline exceeded", code="deadline_exceeded")
                continue
            from .job_journal import JournalConflict, JournalOwnerLost
            try:
                self._mark_running(job)
            except JournalConflict as error:
                self._finish(job, "failed", error=str(error), code="recovery_attempt_limit")
                continue
            except JournalOwnerLost as error:
                self._journal_fenced(error)
                return None
            self._queued_count -= 1
            self._queued_tokens -= job.prompt_tokens + job.max_new
            return job
        return None

    def _finish(self, job, state, **options):
        already_terminal = job.terminal
        try:
            super()._finish(job, state, **options)
        finally:
            if not already_terminal and hasattr(job, "ring_binding"):
                elapsed = self.clock() - job.execution_started if hasattr(job, "execution_started") else None
                self.router.pool.release(job.ring_binding, elapsed_s=elapsed,
                    tokens=job.prompt_tokens + len(job.tokens) if job.state == "completed" else None)

    def _run_ring(self, ring):
        if self._batching["enabled"]:
            from .continuous_batch import ContinuousRingWorker
            return ContinuousRingWorker(self, ring, self._batching).run()
        ring_id = ring.ring_id
        while True:
            with self._condition:
                job = self._next_for_ring(ring)
                while job is None and not self._stopping:
                    if ring.state == RingState.STOPPED:
                        return
                    self._condition.wait(0.1)
                    job = self._next_for_ring(ring)
                if job is None and self._stopping:
                    return
                self._running_by_ring[ring_id] = job
                job.execution_started = self.clock()
            try:
                result = self.router.pool.execute(job.ring_binding, job, job.commit,
                                                  lambda: job.check_stop(self.clock()))
                if not isinstance(result, dict) or result.get("tokens") != job.checkpoint():
                    raise RuntimeError("backend result does not match committed callback frontier")
                with self._condition:
                    job.check_stop(self.clock())
                    self._finish(job, "completed", result=result)
            except (JobExpired, JobCancelled) as error:
                with self._condition:
                    expired = isinstance(error, JobExpired)
                    self._finish(job, "expired" if expired else "cancelled", error=str(error),
                                 code="deadline_exceeded" if expired else "job_cancelled")
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
                    self._running_by_ring.pop(ring_id, None)
                    from .job_journal import JournalOwnerLost
                    try:
                        self._prune(self.clock())
                    except JournalOwnerLost as error:
                        self._journal_fenced(error)
                    self._condition.notify_all()

    def metrics(self, tenant=None):
        with self._condition:
            body = super().metrics(tenant)
            body.update(mode="parallel_serial_rings", running=len(self._running_by_ring),
                        ring_workers=len(self._ring_workers))
            if self._batching["enabled"]:
                body["mode"] = "qualified_token_batches"
                body["running"] = sum(len(value) if isinstance(value, tuple) else 1 for value in self._running_by_ring.values())
            body["scheduling_policy"] = self._scheduling
            return body

    def shutdown(self, *, drain=False, timeout=10.0, preserve_pending=False):
        preserve_pending = preserve_pending or (self.journal is not None and self._journal_failed is not None)
        if type(preserve_pending) is not bool or (preserve_pending and self.journal is None):
            raise ValueError("preserve_pending requires an explicitly configured journal")
        self._preserve_pending = preserve_pending
        end = time.monotonic() + timeout
        with self._condition:
            self._accepting = False
            self._stopping = True
            if preserve_pending:
                for job in self._jobs.values():
                    if not job.terminal:
                        job.cancelled.set()
            elif not drain:
                for job in list(self._jobs.values()):
                    if not job.terminal and not preserve_pending:
                        self.cancel(job.tenant, job.id)
            self._condition.notify_all()
        self.router.pool.shutdown()  # no new bindings; retained jobs keep their rings
        if not drain:
            for ring_id in tuple(self._running_by_ring):
                self._ring_backend(ring_id).abort()
        if self._worker:
            self._worker.join(max(0.0, end - time.monotonic()))
        for worker in tuple(self._ring_workers.values()):
            worker.join(max(0.0, end - time.monotonic()))
        live = [worker for worker in self._ring_workers.values() if worker.is_alive()]
        if live:
            with self._condition:
                for job in list(self._jobs.values()):
                    if not job.terminal:
                        self.cancel(job.tenant, job.id)
            for ring_id in tuple(self._running_by_ring):
                self._ring_backend(ring_id).abort()
            for worker in live:
                worker.join(min(timeout, 2.0))
        if self._sweeper:
            self._sweeper.join(min(timeout, 1.0))
        stopped = not any(worker.is_alive() for worker in self._ring_workers.values())
        if stopped and preserve_pending:
            for job in self._jobs.values():
                if hasattr(job, "ring_binding"):
                    self.router.pool.release(job.ring_binding)
        if stopped and self.journal is not None:
            self._journal_stop.set()
            if self._journal_thread is not None:
                self._journal_thread.join(min(timeout, 1))
            from .job_journal import JournalOwnerLost
            try:
                self.journal.release_owner(self._journal_owner)
            except JournalOwnerLost:
                pass
            self.journal.checkpoint()
        return stopped

    def _ring_backend(self, ring_id):
        return next(r.backend for r in self.router.pool.rings() if r.ring_id == ring_id)

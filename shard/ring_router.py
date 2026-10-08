"""Global tenant admission with one serial worker per independently leased ring.

This is request-level concurrency across rings, not continuous batching. Jobs are
bound at admission; alias changes never move an existing job or its tokenizer.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time

try:
    from shard.service_queue import ServiceQueue, AdmissionError, JobCancelled, JobExpired
    from shard.ring_pool import RingUnavailable
except ImportError:
    from service_queue import ServiceQueue, AdmissionError, JobCancelled, JobExpired
    from ring_pool import RingUnavailable


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


class MultiRingQueue(ServiceQueue):
    """Shares quota/idempotency/history globally; serial execution is per ring."""
    def __init__(self, router, tenants, **options):
        super().__init__(router, tenants, **options)
        self.router = router
        self._ring_workers = {}
        self._running_by_ring = {}

    def start(self):
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
                for ring in self.router.pool.rings():
                    if ring.ring_id not in self._ring_workers:
                        worker = threading.Thread(target=self._run_ring, args=(ring.ring_id,),
                                                  name="ring-" + ring.ring_id, daemon=True)
                        self._ring_workers[ring.ring_id] = worker
                        worker.start()
                self._condition.wait(0.1)

    def submit_request(self, tenant, body, *, idempotency_key=None):
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
                return job, False
            model = request.get("model", self.router.model_id)
            cohort, region = request.get("shard_cohort"), request.get("shard_region")
            if not isinstance(model, str) or (cohort is not None and not isinstance(cohort, str)) or (
                    region is not None and (not isinstance(region, str) or not region)):
                raise AdmissionError("invalid model/cohort/region", status=400, code="invalid_request_error")
            try:
                model_id, cohort_id = self.router.pool.resolve(model, cohort)
                maximum = request.get("max_tokens", request.get("max_completion_tokens", 512))
                if type(maximum) is not int or not 1 <= maximum <= 4096:
                    raise AdmissionError("invalid completion token budget", status=400, code="invalid_request_error")
                binding = self.router.pool.acquire(model_id, cohort_id, estimated_tokens=maximum, region=region)
            except RingUnavailable as error:
                raise AdmissionError(str(error), status=503, code="no_ready_ring") from error
            try:
                prepared_body = dict(request, model=model_id)
                prepared_body.pop("shard_cohort", None); prepared_body.pop("shard_region", None)
                payload, count, maximum, timeout_s = binding.backend.prepare(prepared_body)
                self.router.pool.update_estimate(binding, count + maximum)
                job, fresh = super().submit(tenant, payload, prompt_tokens=count, max_new=maximum,
                                           timeout_s=timeout_s, idempotency_key=idempotency_key)
                # The worker cannot see the newly enqueued job until this shared
                # admission lock is released. Backend/tokenizer remain fixed forever.
                job.ring_binding = binding
                job.served_model = model
                job.served_cohort = cohort_id
                job.bound_backend = binding.backend
                job.request_fingerprint = raw_fingerprint
                self._condition.notify_all()
                return job, fresh
            except Exception:
                self.router.pool.release(binding)
                raise

    def _next_for_ring(self, ring_id):
        for _ in range(len(self._round)):
            tenant = self._round.popleft()
            queue = self._queues[tenant]
            job = next((item for item in queue if not item.terminal and
                        item.ring_binding.ring.ring_id == ring_id), None)
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
            self._queued_count -= 1
            self._queued_tokens -= job.prompt_tokens + job.max_new
            with job.changed:
                job.state = "running"
                job.changed.notify_all()
            return job
        return None

    def _finish(self, job, state, **options):
        already_terminal = job.terminal
        super()._finish(job, state, **options)
        if not already_terminal and hasattr(job, "ring_binding"):
            elapsed = self.clock() - job.execution_started if hasattr(job, "execution_started") else None
            self.router.pool.release(job.ring_binding, elapsed_s=elapsed,
                                     tokens=job.prompt_tokens + len(job.tokens) if state == "completed" else None)

    def _run_ring(self, ring_id):
        while True:
            with self._condition:
                job = self._next_for_ring(ring_id)
                while job is None and not self._stopping:
                    self._condition.wait(0.1)
                    job = self._next_for_ring(ring_id)
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
                    self._prune(self.clock())
                    self._condition.notify_all()

    def metrics(self, tenant=None):
        with self._condition:
            body = super().metrics(tenant)
            body.update(mode="parallel_serial_rings", running=len(self._running_by_ring),
                        ring_workers=len(self._ring_workers))
            return body

    def shutdown(self, *, drain=False, timeout=10.0):
        end = time.monotonic() + timeout
        with self._condition:
            self._accepting = False
            self._stopping = True
            if not drain:
                for job in list(self._jobs.values()):
                    if not job.terminal:
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
        return not any(worker.is_alive() for worker in self._ring_workers.values())

    def _ring_backend(self, ring_id):
        return next(r.backend for r in self.router.pool.rings() if r.ring_id == ring_id)

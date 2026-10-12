"""Bounded token-step scheduling for explicitly qualified, state-isolated backends.

This scheduler never turns a serial model into a batched model. A backend must
implement the contract and bind its digest to verified warmup before admission.
"""
import hashlib
import json

from .ring_pool import RingUnavailable, RingState
from .service_queue import JobCancelled, JobExpired


def contract_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def backend_contract(ring):
    getter = getattr(ring.backend, "batch_contract", None)
    if not callable(getter) or not callable(getattr(ring.backend, "advance_batch", None)) or not callable(
            getattr(ring.backend, "release_batch_job", None)):
        raise ValueError("backend lacks qualified independent-state token batching")
    value = getter()
    required = {"schema", "model_id", "cohort_id", "source_sha256", "runtime_config_sha256",
        "numerical_contract", "max_jobs", "max_step_tokens", "max_prompt_tokens", "max_new_tokens", "max_context_tokens",
        "per_job_state_bytes", "reserved_state_bytes", "prefill_chunk_tokens", "cancellation"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != "shard-token-batch/1":
        raise ValueError("invalid qualified token-batch contract")
    if value["model_id"] != ring.model_id or value["cohort_id"] != ring.cohort_id:
        raise ValueError("batch contract belongs to another model cohort")
    if value["cancellation"] != "per_job" or not isinstance(value["numerical_contract"], str) or not value["numerical_contract"]:
        raise ValueError("batching requires per-job cancellation and a numerical contract")
    for name in ("source_sha256", "runtime_config_sha256"):
        if not isinstance(value[name], str) or len(value[name]) != 64 or any(c not in "0123456789abcdef" for c in value[name]):
            raise ValueError("batch source/configuration digest required")
    for name in ("max_jobs", "max_step_tokens", "max_prompt_tokens", "max_new_tokens", "max_context_tokens", "per_job_state_bytes",
                 "reserved_state_bytes", "prefill_chunk_tokens"):
        if type(value[name]) is not int or not 1 <= value[name] < 2 ** 63:
            raise ValueError("batch capacity must be bounded positive integers")
    if value["max_jobs"] > 32 or value["max_step_tokens"] > 64 or value["prefill_chunk_tokens"] > value["max_prompt_tokens"]:
        raise ValueError("batch step/capacity exceeds scheduler limits")
    if value["per_job_state_bytes"] * value["max_jobs"] > value["reserved_state_bytes"]:
        raise ValueError("batch live states exceed calibrated reservation")
    if value["max_prompt_tokens"] >= value["max_context_tokens"] or value["max_new_tokens"] > value["max_context_tokens"]:
        raise ValueError("batch token bounds exceed calibrated context capacity")
    if getattr(ring, "batch_contract_sha256", None) != contract_digest(value):
        raise ValueError("batch contract was not bound by the verified warmup")
    return json.loads(json.dumps(value, allow_nan=False))


def batching_options(value):
    if value is None:
        return {"enabled": False, "max_jobs": 4, "max_step_tokens": 1}
    if not isinstance(value, dict) or set(value) - {"enabled", "max_jobs", "max_step_tokens"}:
        raise ValueError("unknown batching options")
    enabled = value.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("batching enabled must be boolean")
    result = {"enabled": enabled, "max_jobs": value.get("max_jobs", 4),
              "max_step_tokens": value.get("max_step_tokens", 1)}
    for name, bound in (("max_jobs", 32), ("max_step_tokens", 64)):
        if type(result[name]) is not int or not 1 <= result[name] <= bound:
            raise ValueError("invalid bounded batching options")
    return result


class ContinuousRingWorker:
    def __init__(self, queue, ring, options):
        self.queue, self.ring, self.options = queue, ring, options
        self.active = {}
        self.stalled = {}
        self.replay_allowance = {}

    def _finish(self, job, state, **kwargs):
        cleanup_error = None
        try:
            self.ring.backend.release_batch_job(job.id)
        except Exception as error:
            cleanup_error = error
            self.queue.router.pool.fail(self.ring.ring_id, "batch_state_cleanup_unconfirmed")
            state, kwargs = "failed", {"error": "batch state cleanup unconfirmed", "code": "inference_failed"}
        finally:
            self.active.pop(job.id, None)
            self.stalled.pop(job.id, None)
            self.replay_allowance.pop(job.id, None)
        with self.queue._condition:
            self.queue._running_by_ring[self.ring.ring_id] = tuple(self.active.values())
            self.queue._finish(job, state, **kwargs)
            self.queue._condition.notify_all()
        if cleanup_error is not None:
            raise cleanup_error

    def _stop(self, job, error):
        expired = isinstance(error, JobExpired)
        self._finish(job, "expired" if expired else "cancelled", error=str(error),
                     code="deadline_exceeded" if expired else "job_cancelled")

    def run(self):
        queue, ring = self.queue, self.ring
        try:
            while True:
                contract = backend_contract(ring)
                capacity = min(self.options["max_jobs"], contract["max_jobs"])
                step_tokens = min(self.options["max_step_tokens"], contract["max_step_tokens"])
                with queue._condition:
                    while len(self.active) < capacity:
                        job = queue._next_for_ring(ring)
                        if job is None:
                            break
                        job.execution_started = queue.clock()
                        self.active[job.id] = job
                        # A recovered backend must recompute/check its already
                        # committed prefix without emitting duplicate tokens.
                        # Freeze this allowance at admission; fresh output must
                        # never enlarge the number of tolerated silent steps.
                        self.replay_allowance[job.id] = len(job.checkpoint())
                    queue._running_by_ring[ring.ring_id] = tuple(self.active.values())
                    if not self.active:
                        if queue._stopping or ring.state in (RingState.STOPPED, RingState.FAILED):
                            return
                        queue._condition.wait(.05)
                        continue
                for job in tuple(self.active.values()):
                    try:
                        job.check_stop(queue.clock())
                    except (JobCancelled, JobExpired) as error:
                        self._stop(job, error)
                if not self.active:
                    continue
                jobs = list(self.active.values())
                before = {job.id: len(job.tokens) for job in jobs}
                results = queue.router.pool.execute_batch_step(
                    [job.ring_binding for job in jobs], jobs, max_step_tokens=step_tokens)
                for job in jobs:
                    value = results[job.id]
                    try:
                        job.check_stop(queue.clock())
                    except (JobCancelled, JobExpired) as error:
                        self._stop(job, error)
                        continue
                    if isinstance(value, (JobCancelled, JobExpired)):
                        raise RuntimeError("backend reported cancellation without a cancelled job")
                    if isinstance(value, Exception):
                        self._finish(job, "failed", error=type(value).__name__, code="inference_failed")
                    elif value is not None:
                        self._finish(job, "completed", result=value)
                    elif len(job.tokens) == before[job.id]:
                        self.stalled[job.id] = self.stalled.get(job.id, 0) + 1
                        limit = ((job.prompt_tokens + contract["prefill_chunk_tokens"] - 1) // contract["prefill_chunk_tokens"]
                                 + self.replay_allowance[job.id] + 1)
                        if self.stalled[job.id] > limit:
                            raise RuntimeError("batch backend exhausted its bounded prefill without progress")
                    else:
                        self.stalled[job.id] = 0
                # New jobs join between bounded steps; existing per-job state stays in
                # the backend. There is no busy sleep or mandatory batching delay.
        except Exception as error:
            queue.router.pool.fail(ring.ring_id, "batch_execution_failed")
            for job in tuple(self.active.values()):
                try:
                    self._finish(job, "failed", error=f"{type(error).__name__}: {str(error)[:160]}",
                                 code="inference_failed")
                except Exception:
                    pass
            with queue._condition:
                while True:
                    job = queue._next_for_ring(ring)
                    if job is None:
                        break
                    queue._finish(job, "failed", error="qualified batch ring failed", code="inference_failed")
        finally:
            with queue._condition:
                queue._running_by_ring.pop(ring.ring_id, None)
                queue._condition.notify_all()

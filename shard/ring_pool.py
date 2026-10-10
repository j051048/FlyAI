"""Lease-bound, signed-warmed inference rings; no discovery or identity authority.

Joining adds a new lifecycle entry and never rewrites an executing ring. A binding
owns one backend until completion, including its internal original-prompt retries.
The control plane owns lease renewal/release; this pool enforces node guards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
import threading
import time


class RingState(str, Enum):
    PLANNED = "PLANNED"
    RESERVED = "RESERVED"
    LOADING = "LOADING"
    WARMING = "WARMING"
    READY = "READY"
    DRAINING = "DRAINING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class RingUnavailable(RuntimeError):
    pass


def _field(value, name):
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


@dataclass
class Ring:
    ring_id: str
    backend: object
    model_id: str
    cohort_id: str
    gpu_uuids: tuple
    region: str | None = None
    state: RingState = RingState.PLANNED
    guards: tuple = ()
    bound_jobs: int = 0
    reserved_seconds: float = 0.0
    seconds_per_token: float = 0.1  # replaced by measured warmup before READY
    signed_warmup: bool = False
    local_work: int = 0
    lease_expires_at: float = 0.0
    reason: str = "planned"
    on_stopped: object = None
    replacement_ring_id: str | None = None
    stopping: bool = False
    guard_stop: object = field(default_factory=threading.Event, repr=False)
    guard_thread: object = field(default=None, repr=False)


@dataclass
class RingBinding:
    ring: Ring
    estimated_seconds: float
    released: bool = False

    @property
    def backend(self):
        return self.ring.backend


class RingPool:
    def __init__(self, *, max_rings=256, max_history=256):
        if any(type(value) is not int or value < 1 for value in (max_rings, max_history)):
            raise ValueError("ring capacity and history must be positive integers")
        self._lock = threading.RLock()
        self._rings = {}
        self._aliases = {}
        self._accepting = True
        self.max_rings = max_rings
        self.max_history = max_history
        self._reconciliation = {}

    def record_reconciliation(self, policy_id, status):
        """Bounded public lifecycle state; contains no filesystem/provider secrets."""
        with self._lock:
            self._reconciliation[policy_id] = dict(status)
            while len(self._reconciliation) > self.max_rings:
                self._reconciliation.pop(next(iter(self._reconciliation)))

    def add(self, ring_id, backend, *, model_id, cohort_id, gpu_uuids, region=None, on_stopped=None):
        if not isinstance(ring_id, str) or not ring_id or len(ring_id) > 128:
            raise ValueError("bounded nonempty ring_id required")
        if not isinstance(cohort_id, str) or not re.fullmatch(r"[0-9a-f]{64}", cohort_id):
            raise ValueError("cohort_id must be a lowercase SHA256")
        if not isinstance(model_id, str) or not model_id or backend.model_id != model_id:
            raise ValueError("backend model differs from declared model")
        ids = tuple(gpu_uuids)
        if not ids or any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
            raise ValueError("distinct physical GPU UUIDs required")
        if region is not None and (not isinstance(region, str) or not region):
            raise ValueError("region must be a nonempty string or unknown")
        with self._lock:
            self._prune_history()
            live = sum(r.state != RingState.STOPPED for r in self._rings.values())
            if not self._accepting or ring_id in self._rings or live >= self.max_rings:
                raise ValueError("pool is closed, ring_id already registered, or live ring capacity reached")
            if any(r.backend is backend and r.state != RingState.STOPPED for r in self._rings.values()):
                raise ValueError("one backend connection owner cannot belong to multiple rings")
            ring = Ring(ring_id, backend, model_id, cohort_id, ids, region, on_stopped=on_stopped)
            self._rings[ring_id] = ring
            return ring

    def _prune_history(self):
        stopped = [r.ring_id for r in self._rings.values() if r.state == RingState.STOPPED]
        for ring_id in stopped[:max(0, len(stopped) - self.max_history)]:
            # Jobs/bindings retain their own object and tokenizer. Removing lookup
            # history cannot mutate or redirect a completed client's SSE resume.
            self._rings.pop(ring_id, None)

    def _assert_guards(self, ring):
        if not ring.guards:
            raise RingUnavailable("committed node lease guards required")
        covered = []
        expirations = []
        for guard in ring.guards:
            lease = guard.assert_live()
            if (_field(lease, "ring_id") != ring.ring_id or
                    _field(lease, "model_cohort_sha256") != ring.cohort_id):
                raise RingUnavailable("lease belongs to another ring/model cohort")
            covered.append(guard.gpu_uuid)
            expiry = _field(lease, "expires_at")
            if type(expiry) not in (float, int) or not math.isfinite(expiry) or expiry <= time.time():
                raise RingUnavailable("lease has expired or lacks a finite expiration")
            expirations.append(expiry)
        if len(covered) != len(set(covered)) or set(covered) != set(ring.gpu_uuids):
            raise RingUnavailable("lease guards must cover each physical GPU exactly once")
        ring.lease_expires_at = min(expirations)

    def reserve(self, ring_id, lease_guards):
        with self._lock:
            ring = self._rings[ring_id]
            if ring.state != RingState.PLANNED:
                raise RingUnavailable("only a planned ring can reserve resources")
            ring.guards = tuple(lease_guards)
            try:
                for other in self._rings.values():
                    if other is ring or other.state == RingState.PLANNED:
                        continue
                    # FAILED may still own work/leases. Only a fully stopped ring
                    # may surrender this pool's duplicate-GPU exclusion.
                    if other.state != RingState.STOPPED and set(other.gpu_uuids) & set(ring.gpu_uuids):
                        raise RingUnavailable("physical GPU already belongs to another ring")
            except Exception:
                ring.guards = ()
                raise
            ring.state, ring.reason = RingState.RESERVED, "verifying_committed_leases"
        try:
            self._assert_guards(ring)  # remote RPC never holds the global pool lock
            with self._lock:
                if ring.state != RingState.RESERVED:
                    raise RingUnavailable("reservation was drained during verification")
                ring.reason = "committed_leases"
                self._start_guard_monitor(ring)
            return ring
        except Exception:
            with self._lock:
                if ring.state == RingState.RESERVED:
                    ring.state, ring.reason, ring.guards = RingState.PLANNED, "reservation_rejected", ()
            raise

    def mark_loading(self, ring_id):
        with self._lock:
            ring = self._rings[ring_id]
            if ring.state != RingState.RESERVED:
                raise RingUnavailable("loading requires committed reservations")
        self._assert_guards(ring)
        with self._lock:
            if ring.state != RingState.RESERVED:
                raise RingUnavailable("reservation changed during loading verification")
            ring.state, ring.reason = RingState.LOADING, "loading"

    def _begin(self, ring, work_id):
        works = []
        try:
            self._assert_guards(ring)
            for guard in ring.guards:
                works.append((guard, guard.begin_work(work_id)))
            return works
        except Exception:
            cleanup_errors = []
            for guard, work in reversed(works):
                try:
                    guard.end_work(work)
                except Exception as error:
                    cleanup_errors.append(error)
            if cleanup_errors:
                self.fail(ring.ring_id, "partial_lease_work_cleanup_failed")
            else:
                self._finished_local(ring)
                self.fail(ring.ring_id, "lease_work_begin_failed")
            raise

    def _end(self, ring, works):
        errors = []
        for guard, work in reversed(works):
            try:
                guard.end_work(work)
            except Exception as error:
                errors.append(error)
        if errors:
            # Failed cleanup keeps the ring unavailable; never advertise recycled
            # capacity whose node-local active-work record is still outstanding.
            self.fail(ring.ring_id, "lease_work_cleanup_failed")
            raise errors[0]
        self._finished_local(ring)

    def _finished_local(self, ring):
        with self._lock:
            ring.local_work -= 1
            if ring.state == RingState.DRAINING and ring.bound_jobs == 0 and ring.local_work == 0:
                self._stop(ring)

    def warmup(self, ring_id, timeout_s=300.0):
        with self._lock:
            ring = self._rings[ring_id]
            if ring.state not in (RingState.LOADING, RingState.WARMING) or ring.bound_jobs:
                raise RingUnavailable("warmup requires an idle loaded ring")
            ring.state, ring.reason, ring.signed_warmup = RingState.WARMING, "signed_warmup", False
            ring.local_work += 1
        works = self._begin(ring, "warmup-" + str(time.time_ns()))
        done, invalid = threading.Event(), []
        watcher = threading.Thread(target=self._watch_guards, args=(ring, done, invalid), daemon=True)
        watcher.start()
        started = time.monotonic()
        try:
            proof = ring.backend.warmup(timeout_s=timeout_s)
            if invalid:
                raise invalid[0]
            self._assert_guards(ring)
            with self._lock:
                if proof.get("proof_verified") is not True or not ring.backend.ready()[0]:
                    raise RingUnavailable("signed warmup/readiness verification failed")
                if ring.state != RingState.WARMING:
                    raise RingUnavailable("ring was drained during warmup")
                ring.signed_warmup = True
                count = proof.get("committed_tokens")
                if type(count) is not int or count < 1:
                    raise RingUnavailable("warmup must report actual committed token count")
                ring.seconds_per_token = max(1e-6, (time.monotonic() - started) / count)
                ring.state, ring.reason = RingState.READY, "signed_warmup_verified"
            return proof
        except Exception as error:
            self.fail(ring_id, type(error).__name__)
            raise
        finally:
            done.set(); watcher.join(0.2)
            self._end(ring, works)

    def recover(self, ring_id, timeout_s=300.0):
        """Explicit idle rewarm under the SAME valid leases, never move old jobs."""
        with self._lock:
            ring = self._rings[ring_id]
            if (ring.state not in (RingState.FAILED, RingState.WARMING) or
                    ring.bound_jobs or ring.local_work or not self._accepting):
                raise RingUnavailable("recovery requires an idle failed/unverified ring")
        self._assert_guards(ring)
        with self._lock:
            if ring.bound_jobs or ring.local_work or ring.state not in (RingState.FAILED, RingState.WARMING):
                raise RingUnavailable("ring changed during recovery verification")
            ring.state, ring.reason, ring.signed_warmup = RingState.WARMING, "recovering", False
            self._start_guard_monitor(ring)
        return self.warmup(ring_id, timeout_s=timeout_s)

    def _start_guard_monitor(self, ring):
        if ring.guard_thread is not None and ring.guard_thread.is_alive():
            return
        ring.guard_stop = threading.Event()
        def monitor():
            interval = min(getattr(guard, "watch_interval_s", 0.05) for guard in ring.guards)
            interval = max(0.01, min(5.0, interval))
            while not ring.guard_stop.wait(interval):
                with self._lock:
                    if ring.state == RingState.STOPPED:
                        return
                    if ring.local_work:
                        continue  # execution has its own owner/watchdog
                try:
                    self._assert_guards(ring)
                except Exception as error:
                    self.fail(ring.ring_id, type(error).__name__)
                    ring.backend.abort()
                    # Keep the same monitor alive for explicit idle recovery;
                    # replacing a thread that is just exiting can lose renewal
                    # observation. Only STOPPED terminates this monitor.
                    continue
        ring.guard_thread = threading.Thread(target=monitor, name="ring-guard-" + ring.ring_id, daemon=True)
        ring.guard_thread.start()

    def _watch_guards(self, ring, done, invalid):
        interval = min(getattr(guard, "watch_interval_s", 0.05) for guard in ring.guards)
        interval = max(0.01, min(5.0, interval))
        while not done.wait(max(0.0, min(interval, ring.lease_expires_at - time.time()))):
            try:
                if time.time() >= ring.lease_expires_at:
                    raise RingUnavailable("bound ring lease expired")
                self._assert_guards(ring)
            except Exception as error:
                invalid.append(error)
                ring.backend.abort()  # actively interrupt a blocked send/recv
                return

    def _eligible(self, ring):
        if ring.state != RingState.READY or not ring.signed_warmup:
            return False
        try:
            if ring.lease_expires_at <= time.time():
                raise RingUnavailable("cached committed lease has expired")
            if not ring.backend.ready()[0]:
                # An established connection losing proof cannot accept new jobs.
                ring.state, ring.reason = RingState.WARMING, "connection_reverification_required"
                ring.signed_warmup = False
                return False
        except Exception as error:
            ring.state, ring.reason, ring.signed_warmup = RingState.FAILED, type(error).__name__, False
            return False
        return True

    def set_alias(self, alias, model_id, cohort_id):
        if not isinstance(alias, str) or not alias:
            raise ValueError("nonempty model alias required")
        with self._lock:
            if not any(r.model_id == model_id and r.cohort_id == cohort_id and self._eligible(r)
                       for r in self._rings.values()):
                raise RingUnavailable("alias may switch only to an already READY cohort")
            self._aliases[alias] = (model_id, cohort_id)

    def publish_replacement(self, ring_id, retired_ring_ids, *, aliases=()):
        """Publish a warmed SAME-cohort ring at request boundaries.

        All validation precedes mutation. Previously issued bindings keep their
        backend/tokenizer/leases, including queued work and prefix replay.
        """
        retired = tuple(retired_ring_ids)
        names = tuple(aliases)
        if len(set(retired)) != len(retired) or ring_id in retired:
            raise ValueError("distinct old/new replacement rings required")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("nonempty model aliases required")
        with self._lock:
            if not self._accepting or ring_id not in self._rings:
                raise RingUnavailable("replacement pool is closed or ring is unknown")
            new = self._rings[ring_id]
            if not self._eligible(new):
                raise RingUnavailable("replacement must finish signed warmup before publication")
            old = [self._rings[key] for key in retired]
            if any((ring.model_id, ring.cohort_id) != (new.model_id, new.cohort_id) for ring in old):
                raise RingUnavailable("automatic replacement must preserve exact model cohort")
            if any(name in self._aliases and self._aliases[name] != (new.model_id, new.cohort_id) for name in names):
                raise RingUnavailable("replacement cannot silently change an existing alias cohort")
            for name in names:
                self._aliases[name] = (new.model_id, new.cohort_id)
            for ring in old:
                ring.replacement_ring_id = new.ring_id
                if ring.state != RingState.STOPPED:
                    ring.state, ring.reason = RingState.DRAINING, "replaced_at_request_boundary"
            # DRAINING is already visible to admission before cleanup callbacks.
            for ring in old:
                if ring.state != RingState.STOPPED and not ring.bound_jobs and not ring.local_work:
                    try:
                        self._stop(ring)
                    except Exception:
                        # Publication is already complete. Keep the warmed new
                        # route and retain old capacity until cleanup is proven.
                        ring.reason = "replacement_cleanup_unconfirmed"
            return {"ring_id": new.ring_id, "retired_ring_ids": list(retired),
                    "cohort_id": new.cohort_id, "published": True}

    def resolve(self, model, cohort=None):
        with self._lock:
            if model in self._aliases:
                target = self._aliases[model]
                if cohort is not None and cohort != target[1]:
                    raise RingUnavailable("requested cohort differs from model alias")
                return target
            cohorts = {r.cohort_id for r in self._rings.values() if r.model_id == model}
            if cohort is not None and cohort in cohorts:
                return model, cohort
            if cohort is None and len(cohorts) == 1:
                return model, next(iter(cohorts))
            raise RingUnavailable("model requires a known unambiguous cohort or ready alias")

    def acquire(self, model_id, cohort_id, *, estimated_tokens, region=None, preferred_ring_id=None):
        if preferred_ring_id is not None and (not isinstance(preferred_ring_id, str) or not preferred_ring_id):
            raise ValueError("preferred_ring_id must be a nonempty ring identity")
        with self._lock:
            if not self._accepting:
                raise RingUnavailable("ring pool is draining")
            candidates = [r for r in self._rings.values() if r.model_id == model_id and
                          r.cohort_id == cohort_id and self._eligible(r)]
            if not candidates:
                raise RingUnavailable("no READY leased ring for this model cohort")
            # Region is a preference, not an invented RTT. Compare estimated work
            # within the closest declared region; unknown/cross-region is fallback.
            preferred = [r for r in candidates if r.ring_id == preferred_ring_id]
            local = [r for r in candidates if region is not None and r.region == region]
            # Affinity is an admission preference under current cohort/lease
            # checks. It never revives a draining ring or redirects a held job.
            candidates = preferred or local or candidates
            ring = min(candidates, key=lambda r: (r.reserved_seconds + estimated_tokens * r.seconds_per_token,
                                                 r.bound_jobs, r.ring_id))
            cost = estimated_tokens * ring.seconds_per_token
            ring.bound_jobs += 1; ring.reserved_seconds += cost
            return RingBinding(ring, cost)

    def release(self, binding, *, elapsed_s=None, tokens=None):
        with self._lock:
            if binding.released:
                return
            binding.released = True
            ring = binding.ring
            ring.bound_jobs -= 1
            ring.reserved_seconds = max(0.0, ring.reserved_seconds - binding.estimated_seconds)
            if elapsed_s is not None and tokens and math.isfinite(elapsed_s) and elapsed_s > 0:
                ring.seconds_per_token = 0.8 * ring.seconds_per_token + 0.2 * elapsed_s / tokens
            if ring.state == RingState.DRAINING and ring.bound_jobs == 0 and ring.local_work == 0:
                self._stop(ring)

    def update_estimate(self, binding, token_work):
        with self._lock:
            ring = binding.ring
            cost = token_work * ring.seconds_per_token
            ring.reserved_seconds += cost - binding.estimated_seconds
            binding.estimated_seconds = cost

    def execute(self, binding, job, emit, cancel_check):
        ring = binding.ring
        with self._lock:
            if binding.released or ring.state in (RingState.FAILED, RingState.STOPPED):
                raise RingUnavailable("bound ring is no longer executable")
            ring.local_work += 1
        works = self._begin(ring, job.id)
        done, invalid = threading.Event(), []
        watcher = threading.Thread(target=self._watch_guards, args=(ring, done, invalid), daemon=True)
        watcher.start()
        def checked():
            cancel_check()
            if invalid:
                raise invalid[0]
            # Node guards enforce fencing at execution. The async watcher checks
            # revocation; a local expiry check keeps per-token hooks free of RPC.
            if time.time() >= ring.lease_expires_at:
                raise RingUnavailable("bound ring lease expired")
        try:
            result = ring.backend.execute(job, emit, checked)
            checked()
            if result.get("proof", {}).get("verified") is not True or result["proof"].get("scope") != "complete_final_attempt":
                raise RingUnavailable("complete final-attempt signed proof required")
            self._assert_guards(ring)
            with self._lock:
                if not ring.backend.ready()[0]:
                    raise RingUnavailable("successful result has no live verified ring")
                # A previously bound request may reconnect/reverify this SAME ring.
                if ring.state == RingState.WARMING:
                    ring.state, ring.reason = RingState.READY, "signed_job_reverified"
                ring.signed_warmup = True
            result["route"] = {"ring_id": ring.ring_id, "model_id": ring.model_id,
                               "cohort_id": ring.cohort_id, "region": ring.region}
            return result
        except Exception as error:
            if isinstance(error, RingUnavailable) or not ring.backend.ready()[0]:
                self.fail(ring.ring_id, type(error).__name__)
            raise
        finally:
            done.set(); watcher.join(0.2)
            self._end(ring, works)

    def fail(self, ring_id, reason="failed"):
        with self._lock:
            ring = self._rings[ring_id]
            if ring.state not in (RingState.DRAINING, RingState.STOPPED):
                ring.state, ring.reason, ring.signed_warmup = RingState.FAILED, str(reason), False

    def _stop(self, ring):
        if ring.replacement_ring_id is not None:
            if ring.stopping:
                return
            ring.stopping = True
            ring.guard_stop.set()
            # Remote stop/cleanup acknowledgements must not block admission to
            # the newly warmed ring or hold the queue's request boundary lock.
            def cleanup():
                try:
                    ring.backend.close()
                except Exception:
                    with self._lock:
                        ring.reason, ring.stopping = "replacement_cleanup_unconfirmed", False
                    return
                with self._lock:
                    ring.state, ring.reason, ring.signed_warmup = RingState.STOPPED, "drained", False
                    ring.stopping = False
                    self._prune_history()
                if ring.on_stopped is not None:
                    ring.on_stopped(ring.ring_id)
            threading.Thread(target=cleanup, daemon=True, name="replacement-cleanup-" + ring.ring_id).start()
            return
        ring.guard_stop.set()
        ring.backend.close()
        ring.state, ring.reason, ring.signed_warmup = RingState.STOPPED, "drained", False
        if ring.on_stopped is not None:
            ring.on_stopped(ring.ring_id)
        self._prune_history()

    def drain(self, ring_id):
        with self._lock:
            ring = self._rings[ring_id]
            if ring.state == RingState.STOPPED:
                return
            ring.state, ring.reason = RingState.DRAINING, "draining"
            if ring.bound_jobs == 0 and ring.local_work == 0:
                self._stop(ring)

    def rings(self):
        with self._lock:
            return tuple(self._rings.values())

    def ready(self):
        with self._lock:
            ready = self._accepting and any(self._eligible(r) for r in self._rings.values())
            return bool(ready), "ready_rings" if ready else "no_ready_leased_rings"

    def models(self):
        with self._lock:
            return sorted(set(self._aliases) | {r.model_id for r in self._rings.values()})

    def snapshot(self):
        with self._lock:
            for ring in self._rings.values():
                self._eligible(ring)
            return {"mode": "parallel_serial_rings", "accepting": self._accepting,
                    **({"reconciliation": {key: dict(value) for key, value in self._reconciliation.items()}}
                       if self._reconciliation else {}),
                    "aliases": {name: {"model_id": value[0], "cohort_id": value[1]}
                                for name, value in self._aliases.items()},
                    "rings": [{"ring_id": r.ring_id, "model_id": r.model_id, "cohort_id": r.cohort_id,
                               "region": r.region, "state": r.state.value, "bound_jobs": r.bound_jobs,
                               "estimated_backlog_s": r.reserved_seconds, "reason": r.reason,
                               **({"replacement_ring_id": r.replacement_ring_id}
                                  if r.replacement_ring_id is not None else {})}
                              for r in self._rings.values()]}

    refresh = snapshot

    def shutdown(self):
        with self._lock:
            self._accepting = False
            for ring in tuple(self._rings.values()):
                self.drain(ring.ring_id)

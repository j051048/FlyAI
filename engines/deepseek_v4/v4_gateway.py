"""Authenticated, bounded serial OpenAI-style V4 service over an existing ring.

No continuous batching or durable coordinator HA is claimed. Recovery reconnects,
replays the ORIGINAL request, checks every previously committed token, and emits only
the unseen suffix. A successful final attempt has a complete signed receipt set.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import select
import socket
import signal
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent if HERE.name == "deepseek_v4" and HERE.parent.name == "engines" else HERE
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from shard.service_queue import ServiceQueue, TenantLimits, AdmissionError, JobCancelled, JobExpired
except ImportError:  # flat deployment next to the other engine files
    from service_queue import ServiceQueue, TenantLimits, AdmissionError, JobCancelled, JobExpired


class ReplayMismatch(RuntimeError):
    pass


class ReceiptValidationError(RuntimeError):
    pass




try:
    from shard.http_gateway import AuthRegistry, Gateway, Handler, _BoundedServer
except ImportError:
    from http_gateway import AuthRegistry, Gateway, Handler, _BoundedServer


class V4RingBackend:
    """One serial ring's connections. Numeric/receipt failures are never retried."""
    service_name = "v4-serial-gateway"
    def __init__(self, directory, head, tail, assignments, *, mode="pipelined", max_retries=1,
                 timeout=60.0, max_context=8192, swarm_id="v4-service", vp=None, tokenizer=None, pipeline_plan=None, coordinator_key=None):
        if mode not in ("greedy", "dspark", "pipelined"):
            raise ValueError("invalid V4 service mode")
        if type(max_retries) is not int or not 0 <= max_retries <= 3:
            raise ValueError("max_retries must be an integer in 0..3")
        self.directory, self.head, self.tail = str(directory), head, tail
        self.mode, self.max_retries, self.timeout, self.max_context = mode, max_retries, float(timeout), max_context
        if not math.isfinite(self.timeout) or self.timeout <= 0 or type(max_context) is not int or max_context < 1:
            raise ValueError("I/O timeout and context must be positive")
        self.swarm_id = swarm_id
        self.pipeline_plan = pipeline_plan
        self.coordinator_key = coordinator_key
        if vp is None:
            for path in (Path(__file__).resolve().parent, ROOT / "vendor" / "deepseek_v4_ref" / "encoding"):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            vp = importlib.import_module("v4_pipe")
        self.vp = vp
        self.artifact_verification = None
        if pipeline_plan is not None and pipeline_plan.get("model_cohort"):
            self.artifact_verification = vp.verify_cohort_directory(directory, pipeline_plan["model_cohort"])
        self.model_id = (pipeline_plan["model_cohort"]["model_id"] if pipeline_plan is not None and pipeline_plan.get("model_cohort")
                         else vp.V4_MODEL_ID)
        self.layers = int(json.loads((Path(directory) / "config.json").read_text(encoding="utf-8-sig"))['n_layers']) if tokenizer is None else vp.N_LAYERS
        self.assignments = {str(key): tuple(span) for key, span in assignments.items()}
        self._check_assignments()
        methods = (vp.coordinate, vp.coordinate_dspark, vp.coordinate_dspark_pipelined)
        required = {"cancel_check", "expected_by_signer", "strict_job_binding"}
        if any(not required <= set(inspect.signature(method).parameters) for method in methods):
            raise ValueError("V4 coordinator lacks required cancellation/receipt/job-binding capability")
        if tokenizer is None:
            from v4_tokenizer import load_v4_tokenizer
            tokenizer = load_v4_tokenizer(directory)
        self.tokenizer = tokenizer
        eos = tokenizer.eos_token_id
        self.eos_ids = tuple(eos) if isinstance(eos, (tuple, list)) else (() if eos is None else (eos,))
        self._lock = threading.RLock()
        self._io_lock = threading.RLock()
        self._idle_stop = threading.Event()
        self._channels = None
        self._attempt_owner = None
        self._last_ok = 0.0
        self._stats = {"attempts": 0, "reconnects": 0, "replayed_tokens": 0, "receipt_failures": 0}
        self._idle_thread = None
        if pipeline_plan is not None:
            self._idle_thread = threading.Thread(target=self._idle_keepalive, daemon=True, name="v4-session-keepalive")
            self._idle_thread.start()

    def _check_assignments(self):
        if not self.assignments:
            raise ValueError("production V4 service requires pinned signer assignments")
        cursor = 0
        for lo, hi in sorted(self.assignments.values()):
            if type(lo) is not int or type(hi) is not int or lo != cursor or hi <= lo or hi > self.layers:
                raise ValueError("signer assignments must exactly tile the model")
            cursor = hi
        if cursor != self.layers:
            raise ValueError("incomplete signer assignments")

    def prepare(self, body):
        if not isinstance(body, dict):
            raise AdmissionError("request must be a JSON object", status=400, code="invalid_request_error")
        if body.get("model", self.model_id) != self.model_id:
            raise AdmissionError("requested model is not served", status=404, code="model_not_found")
        if body.get("tools") or body.get("tool_choice"):
            raise AdmissionError("this V4 gateway currently serves text chat; tool-call adaptation is unavailable",
                                 status=400, code="unsupported_feature")
        for key, default in (("temperature", 0.0), ("top_p", 1.0)):
            value = body.get(key, default)
            if type(value) not in (int, float) or not math.isfinite(value) or value != default:
                raise AdmissionError("V4 service currently requires greedy decoding", status=400, code="unsupported_sampling")
        for key, default in (("n", 1), ("seed", 0), ("presence_penalty", 0), ("frequency_penalty", 0)):
            if type(body.get(key, default)) is bool or body.get(key, default) != default:
                raise AdmissionError(f"{key} is unsupported by this V4 service", status=400, code="unsupported_feature")
        if body.get("stop") or body.get("top_k") or body.get("logprobs") or body.get("top_logprobs") or body.get("response_format", {"type": "text"}) not in (None, {"type": "text"}):
            raise AdmissionError("custom stop/top_k/constrained formats are unsupported", status=400, code="unsupported_feature")
        if any(key in body and type(body[key]) is not bool for key in ("stream", "thinking")):
            raise AdmissionError("stream/thinking must be boolean", status=400, code="invalid_request_error")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or any(not isinstance(m, dict)
                or m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str) for m in messages):
            raise AdmissionError("messages require text system/user/assistant entries", status=400, code="invalid_request_error")
        maximum = body.get("max_tokens", body.get("max_completion_tokens", 512))
        if "max_tokens" in body and "max_completion_tokens" in body and body["max_tokens"] != body["max_completion_tokens"]:
            raise AdmissionError("completion token limits conflict", status=400, code="invalid_request_error")
        if type(maximum) is not int or not 1 <= maximum <= 4096:
            raise AdmissionError("max_tokens must be an integer in 1..4096", status=400, code="invalid_request_error")
        mode = body.get("shard_mode", self.mode)
        if mode not in ("greedy", "dspark", "pipelined"):
            raise AdmissionError("invalid shard_mode", status=400, code="invalid_request_error")
        effort = body.get("reasoning_effort")
        ids = self.vp._encode_prompt(self.tokenizer, {"messages": messages,
            "thinking": bool(body.get("thinking", False)), "reasoningEffort": effort})
        if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
            raise AdmissionError("V4 prompt encoding failed", status=400, code="invalid_request_error")
        if len(ids) + maximum + (64 if mode != "greedy" else 0) > self.max_context:
            raise AdmissionError("prompt, completion and speculative margin exceed context", status=400, code="context_length_exceeded")
        timeout_s = body.get("timeout_s", 600.0)
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 3600:
            raise AdmissionError("timeout_s must be finite in (0,3600]", status=400, code="invalid_request_error")
        return {"prompt_ids": ids, "mode": mode, "max_new": maximum}, len(ids), maximum, timeout_s

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _close_channels(self, owner=None):
        with self._lock:
            if owner is not None and owner is not self._attempt_owner:
                return
            channels, self._channels = self._channels, None
            self._last_ok = 0.0
            if channels:
                for channel in channels:
                    try:
                        channel.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    try:
                        channel.close()
                    except OSError:
                        pass

    def abort(self):
        self._close_channels()

    def close(self):
        self._idle_stop.set()
        self.abort()
        if self._idle_thread is not None:
            self._idle_thread.join(.2)

    def _idle_keepalive(self):
        while not self._idle_stop.wait(20):
            if self._channels is None or not self._io_lock.acquire(False):
                continue
            try:
                from shard.pipeline_session import SessionSocket, send_message, recv_message
                pipe = self._channels[0]
                if isinstance(pipe, SessionSocket):
                    pipe.settimeout(min(2, self.timeout))
                    send_message(self.vp._raw_send_msg, pipe, {"op": "session_ping"})
                    result = recv_message(self.vp._raw_recv_msg, pipe)
                    if result.get("op") != "session_pong":
                        raise ConnectionError("invalid owner session ping")
            except (OSError, ValueError):
                self.abort()
            finally:
                self._io_lock.release()

    def _watch_attempt(self, job, owner, done):
        while not done.wait(0.05):
            if job.cancelled.is_set() or time.monotonic() >= job.deadline:
                self._close_channels(owner)
                return

    def _verify_receipts(self, result, nonce, attempt_id):
        try:
            from shard.receipt import verify_coverage, wire_receipt
        except ImportError:
            from receipt import verify_coverage, wire_receipt
        receipts = [wire_receipt(receipt) for receipt in (result.get("receipts") or [])]
        try:
            verify_coverage(receipts, self.layers, expected_by_signer=self.assignments,
                            expected_nonce=nonce, check_chain=True)
            if any(receipt.get("job_id") != attempt_id or receipt.get("swarm_id") != self.swarm_id for receipt in receipts):
                raise ValueError("signed receipt belongs to another job/swarm")
        except Exception as error:
            self._stats["receipt_failures"] += 1
            raise ReceiptValidationError("signed assignment/nonce/job/swarm/coverage/chain validation failed") from error
        return receipts

    def execute(self, job, emit, cancel_check):
        with self._io_lock:
            return self._execute(job, emit, cancel_check)

    def _execute(self, job, emit, cancel_check):
        failures = []
        for attempt in range(self.max_retries + 1):
            cancel_check()
            owner, done = object(), threading.Event()
            with self._lock:
                self._attempt_owner = owner
            monitor = threading.Thread(target=self._watch_attempt, args=(job, owner, done),
                                       name="v4-attempt-cancel", daemon=True)
            monitor.start()
            prefix, seen = job.checkpoint(), 0
            nonce = __import__("secrets").token_hex(32)
            attempt_id = f"{job.id}/attempt-{attempt + 1}"
            try:
                if self._channels is None:
                    remaining = job.deadline - time.monotonic()
                    # connect_ring does not expose its in-progress sockets: bound bootstrap
                    # waits separately; established send/recv are actively interrupted below.
                    session_options = {}
                    if self.pipeline_plan is not None:
                        from shard.pipeline_session import SessionConfig
                        session_options["session_config"] = SessionConfig.from_plan(self.pipeline_plan, index=-1, ttl_s=3600,
                                                                                    caller_key=self.coordinator_key)
                    channels = self.vp.connect_ring(self.head, self.tail, **session_options, timeout=max(0.05, min(0.5, self.timeout, remaining)),
                        token=self.vp.SWARM_TOKEN, retry_s=max(0.05, min(0.5, remaining)))
                    with self._lock:
                        self._channels = channels
                cancel_check()
                for channel in self._channels:
                    channel.settimeout(max(0.05, min(self.timeout, job.deadline - time.monotonic())))
                self._stats["attempts"] += 1

                def committed(token):
                    nonlocal seen
                    token = int(token)
                    if seen < len(prefix):
                        if token != prefix[seen]:
                            raise ReplayMismatch("recomputed committed prefix differs; refusing unsafe resume")
                        self._stats["replayed_tokens"] += 1
                    else:
                        emit(token)
                    seen += 1

                method = {"greedy": self.vp.coordinate, "dspark": self.vp.coordinate_dspark,
                          "pipelined": self.vp.coordinate_dspark_pipelined}[job.payload["mode"]]
                kwargs = dict(eos_ids=self.eos_ids, nonce=nonce, swarm_id=self.swarm_id, job_id=attempt_id,
                    layer_count=self.layers, receipts=True, timeout=max(0.05, min(self.timeout, job.deadline - time.monotonic())),
                    on_token=committed, cancel_check=cancel_check, expected_by_signer=self.assignments,
                    strict_job_binding=True)
                if job.payload["mode"] == "greedy":
                    kwargs.update(temp=0.0, seed=0)
                result = method(*self._channels, job.payload["prompt_ids"], job.max_new, **kwargs)
                cancel_check()
                if not result.get("ok") or result.get("tokens") != job.checkpoint() or seen != len(job.checkpoint()):
                    raise ReplayMismatch("coordinator result differs from the committed callback frontier")
                receipts = self._verify_receipts(result, nonce, attempt_id)
                self._last_ok = time.monotonic()
                return {"tokens": job.checkpoint(), "text": self.decode(job.checkpoint()),
                    "finish_reason": "length" if len(job.tokens) >= job.max_new else "stop",
                    "proof": {"verified": True, "scope": "complete_final_attempt", "attempt_id": attempt_id,
                              "nonce": nonce, "swarm_id": self.swarm_id, "receipts": receipts},
                    "recovery": {"attempts": attempt + 1, "replayed_committed_tokens": len(prefix),
                                 "strategy": "original_request_replay_with_prefix_check", "transport_failures": failures}}
            except (OSError, EOFError) as error:
                self._close_channels(owner)
                cancel_check()  # cancellation/deadline never masquerades as retryable transport churn
                failures.append(type(error).__name__)
                if attempt >= self.max_retries:
                    raise
                self._stats["reconnects"] += 1
            except Exception:
                self._close_channels(owner)
                raise
            finally:
                done.set()
                monitor.join(0.2)
                with self._lock:
                    if self._attempt_owner is owner:
                        self._attempt_owner = None
        raise RuntimeError("unreachable retry state")

    def ready(self):
        with self._lock:
            channels = self._channels
            if not self._last_ok or channels is None:
                return False, "signed_warmup_required"
            try:
                if any(channel.fileno() < 0 for channel in channels):
                    raise ConnectionError("established channel closed")
                if self._attempt_owner is not None:
                    # An active reader owns these streams. Never change its mode or
                    # race it with a peek; execute()/the monitor clears proof on fault.
                    return True, "verified_ring_in_use"
                for channel in channels:
                    readable, _, errors = select.select([channel], [], [channel], 0)
                    if errors:
                        raise ConnectionError("established channel error")
                    if not readable:
                        continue
                    old_timeout = channel.gettimeout()
                    try:
                        # Safe only while idle under this lock: execute cannot acquire
                        # ownership while the readiness check temporarily changes mode.
                        channel.setblocking(False)
                        try:
                            peek = channel.recv(1, socket.MSG_PEEK)
                        except BlockingIOError:
                            continue
                        if peek == b"":
                            raise ConnectionError("established channel EOF")
                    finally:
                        channel.settimeout(old_timeout)
                return True, "verified_established_ring"
            except (OSError, ValueError, AttributeError):
                self._close_channels()
                return False, "established_ring_closed"

    def warmup(self, timeout_s=300.0):
        """A real signed two-token request, isolated from tenant quota/accounting."""
        payload, count, maximum, timeout_s = self.prepare({"model": self.model_id,
            "messages": [{"role": "user", "content": "Reply briefly with a greeting."}],
            "max_tokens": 2, "shard_mode": self.mode, "timeout_s": timeout_s})
        now = time.monotonic()
        try:
            from shard.service_queue import Job
        except ImportError:
            from service_queue import Job
        job = Job("__startup_warmup__", payload, count, maximum, now + timeout_s,
                  "startup-warmup", None, now, state="running")
        result = self.execute(job, job.commit, job.check_stop)
        return {"ready": True, "committed_tokens": len(result["tokens"]),
                "proof_verified": result["proof"]["verified"]}

    def stats(self):
        return dict(self._stats)








def validate_deployment_bundle(bundle, *, assignments=None, env=None, checker=None):
    """Bind service identities/security mode to the measured concrete deployment gate."""
    if checker is None:
        try:
            from shard.deployment import check_deployment
        except ImportError:
            from deployment import check_deployment
        checker = check_deployment
    checked = checker(bundle)
    if checked.get("ready") is not True:
        raise ValueError("deployment is not ready: " + "; ".join(checked.get("errors", [])))
    expected = {stage["signer_pubkey"]: (stage["lo"], stage["hi"]) for stage in bundle["stages"]}
    if assignments is not None and {str(key): tuple(span) for key, span in assignments.items()} != expected:
        raise ValueError("service signer assignments differ from deployment")
    current = os.environ if env is None else env
    declared = bundle.get("coord_env", {})
    if not isinstance(declared, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in declared.items()):
        raise ValueError("coord_env must be a string-valued environment map")
    for key, value in declared.items():
        if key == "SHARD_SWARM_TOKEN" or key.endswith("_KEY_FILE"):
            raise ValueError("secret configuration must not be embedded in a public deployment bundle")
        if current.get(key) != value:
            raise ValueError(f"coordinator environment differs for {key}")
    try:
        from shard.deployment import deployment_policies
    except ImportError:
        from deployment import deployment_policies
    sealed = deployment_policies(bundle)["token_privacy"] == "sealed_ids"
    if (current.get("V4_SEALED_IDS", "0") == "1") != sealed:
        raise ValueError("coordinator token privacy mode differs from deployment")
    if sealed:
        key_id = current.get("V4_TOKEN_PRIVACY_KEY_ID")
        declared_keys = {stage["env"].get("V4_TOKEN_PRIVACY_KEY_ID") for stage in bundle["stages"]}
        if not key_id or declared_keys != {key_id}:
            raise ValueError("sealed token key identity differs across coordinator/deployment")
    return expected


def _manifest_guard(spec, *, ring_id, cohort_id, base):
    """Local reference ledgers or existing-identity authenticated node RPC."""
    if "rpc_address" in spec:
        try:
            from shard.control_plane import remote_lease_guard
        except ImportError:
            from control_plane import remote_lease_guard
        spec = dict(spec)
        key = Path(spec["sidecar_key"])
        spec["sidecar_key"] = str(key if key.is_absolute() else base / key)
        return remote_lease_guard(spec, ring_id=ring_id, cohort_id=cohort_id)
    try:
        from shard.leases import LeaseLedger
    except ImportError:
        from leases import LeaseLedger
    path = Path(spec["ledger"])
    principal = spec["principal"]
    if not isinstance(principal, str) or not principal:
        raise ValueError("local ledger requires a configured controller principal")
    ledger = LeaseLedger(str(path if path.is_absolute() else base / path), node_id=spec["node_id"],
        authorize=lambda identity, action, binding: principal if identity == principal else None)
    # Local principal is controlled service configuration, not remote client input.
    return ledger.guard(spec["lease_id"], spec["fencing_token"], principal=spec["principal"],
                        ring_id=ring_id, model_cohort_sha256=cohort_id)


def load_ring_pool(path, *, backend_factory=V4RingBackend, guard_loader=None,
                   checker=None, warmup_timeout_s=300.0, env=None):
    """Create real warmed rings from a measured, versioned, leased manifest.

Injection arguments are for CPU integration tests/control-plane adapters. JSON
readiness claims are never accepted; every backend actually executes warmup.
"""
    try:
        from shard.ring_pool import RingPool
        from shard.offers import ModelCohort
        from shard.resources import PlacementRequirements
    except ImportError:
        from ring_pool import RingPool
        from offers import ModelCohort
        from resources import PlacementRequirements
    path = Path(path).resolve()
    body = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(body, dict) or body.get("schema") != "shard-ring-pool/1" or not isinstance(body.get("rings"), list) or not body["rings"]:
        raise ValueError("nonempty shard-ring-pool/1 manifest required")
    base, pool = path.parent, RingPool()
    def local_path(value):
        value = Path(value)
        return value if value.is_absolute() else base / value
    try:
        for record in body["rings"]:
            declaration = record["deployment"]
            bundle = declaration if isinstance(declaration, dict) else json.loads(local_path(declaration).read_text(encoding="utf-8-sig"))
            assignments = validate_deployment_bundle(bundle, checker=checker, env=env)
            cohort = ModelCohort.from_dict(bundle["model_cohort"])
            if ((cohort.model_id, cohort.checkpoint_id, cohort.n_layers) !=
                    (bundle["model_id"], bundle["checkpoint_id"], bundle["layer_count"])):
                raise ValueError("model cohort differs from measured deployment identity")
            if record.get("cohort_id", cohort.cohort_id) != cohort.cohort_id:
                raise ValueError("manifest cohort differs from its exact descriptor")
            directory = local_path(record["dir"])
            if hashlib.sha256((directory / "config.json").read_bytes()).hexdigest() != cohort.config_sha256:
                raise ValueError("local checkpoint config differs from model cohort")
            caps = [stage["runtime_config"]["args"]["max_seq_len"] for stage in bundle["stages"]]
            if not caps or any(type(cap) is not int or cap < 1 for cap in caps):
                raise ValueError("measured stage context limits are required")
            context = record.get("max_context", min(caps))
            if any(type(cap) is not int or cap < 1 for cap in caps) or type(context) is not int or not 1 <= context <= min(caps):
                raise ValueError("service context must fit every measured stage")
            ring_id = record["ring_id"]
            backend = backend_factory(str(directory), record["head"], record["tail"], assignments,
                mode=record.get("mode", "pipelined"), max_retries=record.get("retries", 1),
                timeout=record.get("io_timeout", 60.0), max_context=context, swarm_id=ring_id)
            if backend.model_id != cohort.model_id or backend.layers != cohort.n_layers:
                backend.close()
                raise ValueError("served backend differs from declared model cohort")
            backend.model_cohort = cohort.to_dict()
            try:
                pool.add(ring_id, backend, model_id=cohort.model_id, cohort_id=cohort.cohort_id,
                         gpu_uuids=[stage["gpu_uuid"] for stage in bundle["stages"]], region=record.get("region"))
            except Exception:
                backend.close()
                raise
            guards = [(guard_loader(spec, ring_id=ring_id, cohort_id=cohort.cohort_id) if guard_loader else
                       _manifest_guard(spec, ring_id=ring_id, cohort_id=cohort.cohort_id, base=base))
                      for spec in record["leases"]]
            by_gpu = {guard.gpu_uuid: guard for guard in guards}
            for stage in bundle["stages"]:
                guard = by_gpu[stage["gpu_uuid"]]
                req = PlacementRequirements.from_dict(stage["requirements"])
                if (guard.node_id != stage["node_id"] or guard.memory_domain_id != stage.get("memory_domain_id", stage["host_id"]) or
                        guard.resources.vram_bytes < req.gpu.peak_bytes or
                        guard.resources.ram_bytes < req.host.peak_bytes or
                        guard.resources.pinned_bytes < req.host.pinned_bytes):
                    raise ValueError("node lease identity/budget differs from measured stage requirements")
            pool.reserve(ring_id, guards)
            pool.mark_loading(ring_id)
            pool.warmup(ring_id, timeout_s=warmup_timeout_s)
        aliases = body.get("aliases", {})
        if not isinstance(aliases, dict):
            raise ValueError("aliases must be a model/cohort mapping")
        for alias, target in aliases.items():
            pool.set_alias(alias, target["model_id"], target["cohort_id"])
        return pool, body.get("default_model")
    except Exception:
        pool.shutdown()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir")
    parser.add_argument("--head", default="127.0.0.1:29610")
    parser.add_argument("--tail", default="127.0.0.1:29612")
    parser.add_argument("--auth-file", required=True)
    parser.add_argument("--deployment", help="concrete measured shard-deployment/1 bundle (single ring)")
    parser.add_argument("--ring-pool", help="shard-ring-pool/1 JSON: measured deployments, cohorts, node leases, aliases")
    parser.add_argument("--assignments", help="optional map to cross-check against deployment identities")
    parser.add_argument("--mode", choices=("greedy", "dspark", "pipelined"), default="pipelined")
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--io-timeout", type=float, default=60.0)
    parser.add_argument("--warmup-timeout-s", type=float, default=300.0)
    parser.add_argument("--skip-warmup", action="store_true", help="diagnostic start only; readiness remains false until a signed job succeeds")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    parser.add_argument("--allow-insecure-http", action="store_true", help="explicitly permit non-loopback HTTP behind a trusted TLS proxy")
    args = parser.parse_args(argv)
    if args.ring_pool:
        if args.dir or args.deployment or args.assignments or args.skip_warmup:
            parser.error("--ring-pool owns per-ring dir/deployment/assignments and requires signed warmup")
    elif not args.dir or not args.deployment:
        parser.error("single-ring serving requires --dir and --deployment")
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error("provide both TLS certificate and key")
    if args.bind not in ("127.0.0.1", "::1", "localhost") and not args.tls_cert and not args.allow_insecure_http:
        parser.error("non-loopback serving requires TLS or explicit trusted-proxy HTTP opt-in")
    auth = AuthRegistry.from_file(args.auth_file)
    if args.ring_pool:
        pool, default_model = load_ring_pool(args.ring_pool, warmup_timeout_s=args.warmup_timeout_s)
        try:
            gateway = Gateway(auth=auth, ring_pool=pool, default_model=default_model)
        except Exception:
            pool.shutdown()
            raise
        backend = gateway.backend
    else:
        bundle = json.loads(Path(args.deployment).read_text(encoding="utf-8-sig"))
        supplied = json.loads(Path(args.assignments).read_text(encoding="utf-8-sig")) if args.assignments else None
        assignments = validate_deployment_bundle(bundle, assignments=supplied)
        context_caps = [stage.get("runtime_config", {}).get("args", {}).get("max_seq_len") for stage in bundle["stages"]]
        if any(type(cap) is not int or cap < 1 for cap in context_caps) or args.max_context > min(context_caps):
            parser.error("service context must fit every deployment stage's measured max_seq_len")
        backend = V4RingBackend(args.dir, args.head, args.tail,
            assignments, mode=args.mode,
            max_retries=args.retries, timeout=args.io_timeout, max_context=args.max_context)
        if backend.model_id != bundle["model_id"] or backend.layers != bundle["layer_count"]:
            backend.close()
            parser.error("deployment model/layer count differs from the served checkpoint")
        if not args.skip_warmup:
            try:
                backend.warmup(args.warmup_timeout_s)
            except Exception as error:
                backend.close()
                raise SystemExit(f"signed V4 startup warmup failed ({type(error).__name__})") from error
        gateway = Gateway(backend, auth, allow_unverified_start=args.skip_warmup)
    server = gateway.server(args.bind, args.port)
    if args.tls_cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(args.tls_cert, args.tls_key)
        server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
    print("SHARD_GATEWAY_READY " + json.dumps({"port": server.server_port, "mode": gateway.jobs.metrics()["mode"], "authenticated": True,
                                             "ready": backend.ready()[0]}), flush=True)
    previous_signals = {}
    def stop_server(signum, frame):
        # HTTPServer.shutdown must run on another thread than serve_forever.
        threading.Thread(target=server.shutdown, name="gateway-shutdown", daemon=True).start()
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_signals[sig] = signal.signal(sig, stop_server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        gateway.shutdown(); server.server_close()
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

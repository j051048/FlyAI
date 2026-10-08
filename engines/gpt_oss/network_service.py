"""Serve the existing GPT-OSS runtime through shared offers, leases and ring queues."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import secrets
import select
import socket
import ssl
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shard.http_gateway import AuthRegistry, Gateway
from shard.network_service import OpenNetworkService
from shard.pipeline_plan import validate_plan
from shard.pipeline_session import ProtocolError
from shard.receipt import ReceiptError, verify_coverage, verify_receipt, wire_receipt
from shard.service_queue import AdmissionError, Job


class GPTOSSRingBackend:
    """Serialized greedy requests with original-prompt replay and complete receipts."""
    service_name = "gpt-oss-inference-gateway"

    def __init__(self, directory, plan, *, tokenizer=None, runtime=None, K=4, depth=2,
                 ngram_n=3, max_context=8192, timeout=60, max_retries=1,
                 prefill_chunk=2048, adaptive=False, coordinator_key=None, adaptive_depth=False):
        self.plan = validate_plan(plan)
        if any(not stage.get("signer_pubkey") for stage in self.plan["stages"]):
            raise ValueError("production GPT-OSS service requires pinned stage signers")
        self.assignments = {s["signer_pubkey"]: (s["lo"], s["hi"]) for s in self.plan["stages"]}
        if len(self.assignments) != self.plan["nstages"]:
            raise ValueError("distinct stage signers required by this receipt verifier")
        self.directory = str(directory)
        self.layers = self.plan["n_layers"]
        self.model_cohort = self.plan.get("model_cohort")
        if not self.model_cohort:
            raise ValueError("production service requires the full model cohort")
        self.model_id = self.model_cohort["model_id"]
        self.head, self.tail = self.plan["coordinator"]["head"], self.plan["coordinator"]["tail"]
        for name, value, limit in (("K", K, 64), ("depth", depth, 32), ("ngram_n", ngram_n, 16)):
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"invalid {name}")
        if type(max_context) is not int or max_context < 1 or type(prefill_chunk) is not int or prefill_chunk < 0:
            raise ValueError("invalid context/prefill chunk")
        if not math.isfinite(timeout) or timeout <= 0 or type(max_retries) is not int or not 0 <= max_retries <= 3:
            raise ValueError("invalid timeout/retry budget")
        self.K, self.depth, self.ngram_n = K, depth, ngram_n
        if self.plan.get("execution"):
            max_context = min(max_context, self.plan["execution"]["max_context"])
        self.max_context, self.timeout, self.max_retries = max_context, float(timeout), max_retries
        self.prefill_chunk, self.adaptive = prefill_chunk, bool(adaptive)
        self.adaptive_depth = bool(adaptive_depth)
        if coordinator_key is None:
            from shard.manifest import load_key
            key_path = os.environ.get("SHARD_COORDINATOR_KEY")
            if not key_path:
                raise ValueError("coordinator signing key required for the strict inference handshake")
            coordinator_key = load_key(key_path)
        self.coordinator_key = coordinator_key
        if runtime is None:
            from shard.download_inventory import verify_inventory
            from shard.gpt_oss_contract import validate_supported_cohort
            validate_supported_cohort(self.model_cohort,
                json.loads((Path(directory) / "config.json").read_text(encoding="utf-8-sig")))
            verified = verify_inventory(directory, expected_checkpoint_id=self.model_cohort["checkpoint_id"],
                                        expected_repo=self.model_id, verify_files=True)
            if verified["manifest_sha256"] != self.model_cohort["manifest_sha256"]:
                raise ValueError("model payload inventory differs from the cohort manifest")
            if verified["config_sha256"] != self.model_cohort["config_sha256"]:
                raise ValueError("actual model configuration differs from the cohort")
            for path in (ROOT / "shard", ROOT / "phase0"):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            runtime = importlib.import_module("specpipe")
            if not runtime.RECEIPTS:
                raise ValueError("SHARD_RECEIPTS=1 is required by the production GPT-OSS coordinator")
        self.runtime = runtime
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        self.tokenizer = tokenizer
        self._channels = None
        self._attempt_owner = None
        self._lock = threading.RLock()
        self._io_lock = threading.RLock()
        self._closed = threading.Event()
        self._last_ok = 0.0
        self._stats = {"attempts": 0, "reconnects": 0, "replayed_tokens": 0, "receipt_failures": 0}
        self._idle_thread = threading.Thread(target=self._idle_keepalive, daemon=True, name="gptoss-session-keepalive")
        self._idle_thread.start()

    def prepare(self, body):
        if not isinstance(body, dict):
            raise AdmissionError("request must be an object", status=400)
        if body.get("model", self.model_id) != self.model_id:
            raise AdmissionError("requested model is not served", status=404, code="model_not_found")
        for key, default in (("temperature", 0.0), ("top_p", 1.0), ("n", 1),
                             ("presence_penalty", 0), ("frequency_penalty", 0), ("seed", 0)):
            value = body.get(key, default)
            if type(value) not in (int, float) or not math.isfinite(value) or value != default:
                raise AdmissionError("this service requires greedy decoding", status=400, code="unsupported_sampling")
        if any(body.get(k) for k in ("tools", "tool_choice", "stop", "logprobs", "top_logprobs", "top_k")):
            raise AdmissionError("unsupported request feature", status=400, code="unsupported_feature")
        if body.get("response_format", {"type": "text"}) not in (None, {"type": "text"}):
            raise AdmissionError("only text response format is supported", status=400, code="unsupported_feature")
        if "stream" in body and type(body["stream"]) is not bool:
            raise AdmissionError("stream must be boolean", status=400)
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) or
            m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str) for m in messages):
            raise AdmissionError("text chat messages required", status=400)
        effort = body.get("reasoning_effort", "low")
        if effort not in ("low", "medium", "high"):
            raise AdmissionError("invalid reasoning effort", status=400)
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 reasoning_effort=effort)
        if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
            raise AdmissionError("prompt encoding failed", status=400)
        maximum = body.get("max_tokens", body.get("max_completion_tokens", 512))
        if body.get("max_completion_tokens", maximum) != maximum:
            raise AdmissionError("completion token budgets conflict", status=400)
        if type(maximum) is not int or not 1 <= maximum <= 4096:
            raise AdmissionError("max_tokens must be in 1..4096", status=400)
        if len(ids) + maximum + self.K * self.depth + 1 > self.max_context:
            raise AdmissionError("request and speculative window exceed context", status=400, code="context_length_exceeded")
        deadline = body.get("timeout_s", 600.0)
        if type(deadline) not in (int, float) or not math.isfinite(deadline) or not 0 < deadline <= 3600:
            raise AdmissionError("timeout_s must be in (0,3600]", status=400)
        return {"prompt_ids": ids, "reasoning_effort": effort}, len(ids), maximum, float(deadline)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def abort(self, owner=None):
        with self._lock:
            if owner is not None and owner is not self._attempt_owner:
                return
            channels, self._channels = self._channels, None
        if channels:
            for channel in channels:
                try:
                    channel.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                channel.close()

    def close(self):
        self._closed.set()
        self.abort()
        self._idle_thread.join(.2)

    def _idle_keepalive(self):
        while not self._closed.wait(20):
            ping = getattr(self.runtime, "ping_ring", None)
            if ping is None or self._channels is None or not self._io_lock.acquire(False):
                continue
            try:
                if self._channels:
                    for channel in self._channels: channel.settimeout(min(2, self.timeout))
                    ping(*self._channels)
            except (OSError, ValueError):
                self.abort()
            finally:
                self._io_lock.release()

    def _proof(self, result, nonce, attempt_id):
        receipts = [wire_receipt(r) for r in result.get("receipts", [])]
        if len(receipts) != len(self.assignments):
            raise ReceiptError("complete stage receipts required")
        seen = set()
        for receipt in receipts:
            key = receipt.get("pubkey")
            if key in seen or key not in self.assignments:
                raise ReceiptError("unexpected/repeated stage signer")
            seen.add(key)
            verify_receipt(receipt, key)
            if (receipt.get("layer_start"), receipt.get("layer_end")) != self.assignments[key]:
                raise ReceiptError("receipt differs from assigned layer block")
            if (receipt.get("swarm_id"), receipt.get("job_id"), receipt.get("nonce")) != (self.plan["ring_id"], attempt_id, nonce):
                raise ReceiptError("stale or cross-job receipt")
            if receipt.get("n_chunks", 0) < 1:
                raise ReceiptError("empty stage receipt")
        verify_coverage(receipts, self.layers)
        return {"verified": True, "scope": "complete_final_attempt", "receipts": receipts,
                "nonce": nonce, "attempt_id": attempt_id, "swarm_id": self.plan["ring_id"]}

    def execute(self, job, emit, cancel_check):
        with self._io_lock:
            return self._execute(job, emit, cancel_check)

    def _execute(self, job, emit, cancel_check):
        started, initial_tokens = time.perf_counter(), len(job.checkpoint())
        first_emission = last_emission = None
        failures = []
        for attempt in range(self.max_retries + 1):
            cancel_check()
            done = threading.Event()
            owner = object()
            with self._lock:
                self._attempt_owner = owner
            def watch(_done=done, _owner=owner):
                while not _done.wait(.1):
                    try:
                        cancel_check()
                    except Exception:
                        self.abort(_owner)
                        return
            watcher = threading.Thread(target=watch, daemon=True, name="gptoss-cancel")
            watcher.start()
            prefix, seen = job.checkpoint(), 0
            nonce, attempt_id = secrets.token_hex(32), f"{job.id}/attempt-{attempt + 1}"
            def observe(event):
                nonlocal seen, first_emission, last_emission
                cancel_check()
                ids = event.get("out")
                if ids is None:
                    return
                if ids[:min(len(ids), len(prefix))] != prefix[:min(len(ids), len(prefix))]:
                    raise ValueError("replayed generation differs from published prefix")
                if len(ids) < seen:
                    raise ValueError("generation revised its committed frontier")
                for index in range(seen, len(ids)):
                    if index >= len(prefix):
                        emit(int(ids[index]))
                        last_emission = time.perf_counter()
                        if first_emission is None:
                            first_emission = last_emission
                seen = len(ids)
            try:
                from shard.pipeline_session import SessionConfig
                if self._channels is None:
                    cfg = SessionConfig.from_plan(self.plan, index=-1, ttl_s=3600, caller_key=self.coordinator_key)
                    channels = self.runtime.connect_ring(self.head, self.tail, session_config=cfg,
                        timeout=min(.5, self.timeout), retry_s=min(.5, max(.05, job.deadline - time.monotonic())))
                    with self._lock:
                        self._channels = channels
                for channel in self._channels:
                    channel.settimeout(max(.05, min(self.timeout, job.deadline - time.monotonic())))
                self._stats["attempts"] += 1
                drafter = self.runtime.NgramDrafter(ng=self.ngram_n)
                result = self.runtime.coordinate_pipe(None, self._channels[0], self.tokenizer, "", self.K,
                    job.max_new, self.timeout, self.depth, ret_sock=self._channels[1],
                    local_draft=drafter, prefill_chunk=self.prefill_chunk, max_ctx=self.max_context,
                    prompt_ids=job.payload["prompt_ids"], cancel_check=cancel_check, on_commit=observe,
                    swarm_id=self.plan["ring_id"], job_id=attempt_id, nonce=nonce,
                    expected_by_signer=self.assignments, strict_job_binding=True,
                    adaptive_pipe=self.adaptive, adaptive_depth=self.adaptive_depth)
                observe({"out": result["output_ids"]})
                cancel_check()
                if not result.get("ok") or result["output_ids"] != job.checkpoint():
                    raise ValueError("coordinator result differs from committed frontier")
                proof = self._proof(result, nonce, attempt_id)
                self._last_ok = time.monotonic()
                engine_metrics = result.get("metrics", {})
                new_tokens = len(job.checkpoint()) - initial_tokens
                metrics = {**engine_metrics, "schema": "shard-pipeline-metrics/2",
                    "committed_tokens": len(job.checkpoint()), "new_tokens": new_tokens,
                    "new_decode_tokens": max(0, new_tokens - 1),
                    "decode_s": last_emission - first_emission if first_emission is not None else 0,
                    "ttft_s": first_emission - started if first_emission is not None else None,
                    "request_s": time.perf_counter() - started,
                    "timing_scope": "gateway_commit_and_complete_receipt_verification",
                    "engine_metrics": engine_metrics}
                self._stats["replayed_tokens"] += len(prefix)
                return {"tokens": job.checkpoint(), "text": self.decode(job.checkpoint()),
                    "finish_reason": "length" if len(job.tokens) >= job.max_new else "stop", "proof": proof,
                    "metrics": metrics,
                    "recovery": {"attempts": attempt + 1, "replayed_committed_tokens": len(prefix),
                                 "strategy": "original_request_replay_with_prefix_check", "transport_failures": failures}}
            except ReceiptError:
                self._stats["receipt_failures"] += 1
                self.abort()
                raise
            except (OSError, EOFError, self.runtime.TransportError) as error:
                self.abort()
                if isinstance(error, ProtocolError) or isinstance(error.__cause__, ProtocolError):
                    raise
                cancel_check()
                failures.append(type(error).__name__)
                if attempt >= self.max_retries:
                    raise
                self._stats["reconnects"] += 1
            except BaseException:
                self.abort()
                raise
            finally:
                done.set()
                watcher.join(.2)
                with self._lock:
                    if self._attempt_owner is owner:
                        self._attempt_owner = None
        raise RuntimeError("unreachable retry state")

    def ready(self):
        channels = self._channels
        if not self._last_ok or channels is None:
            return False, "not_warmed"
        for channel in channels:
            try:
                ready, _, _ = select.select([channel], [], [], 0)
                if ready and channel.recv(1, socket.MSG_PEEK) == b"":
                    return False, "closed_inference_channel"
            except (OSError, ValueError):
                return False, "closed_inference_channel"
        return True, "signed_warmup_verified"

    def stats(self):
        return dict(self._stats)

    def warmup(self, timeout_s=300):
        payload, count, maximum, _ = self.prepare({"messages": [{"role": "user", "content": "Reply with a greeting."}], "max_tokens": 2})
        now = time.monotonic()
        job = Job("__warmup__", payload, count, maximum, now + timeout_s, "startup-warmup", None, now, state="running")
        result = self.execute(job, job.commit, job.check_stop)
        return {"ready": True, "committed_tokens": len(result["tokens"]), "proof_verified": result["proof"]["verified"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--auth-file", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tls-cert"); parser.add_argument("--tls-key")
    args = parser.parse_args(argv)
    if bool(args.tls_cert) != bool(args.tls_key) or (args.host not in {"127.0.0.1", "localhost", "::1"} and not args.tls_cert):
        parser.error("public listeners require TLS certificate and key")
    def factory(directory, plan, cohort, row, contracts):
        return GPTOSSRingBackend(directory, plan, K=row.get("K", 4), depth=row.get("depth", 2),
            ngram_n=row.get("ngram_n", 3), max_context=row.get("max_context", 8192),
            timeout=row.get("io_timeout_s", 60), prefill_chunk=row.get("prefill_chunk", 2048), adaptive=row.get("adaptive_pipe", False),
            coordinator_key=service.key, adaptive_depth=row.get("adaptive_depth", False))
    def profile(directory, row):
        from shard.gpt_oss_planning import inspect_checkpoint, planning_profile
        from shard.offers import ModelCohort
        cohort = ModelCohort.from_dict(row["cohort"])
        if row.get("adaptive_pipe", False):
            raise ValueError("a single measured chunk profile cannot price mixed-K control/pilot work; use the benchmark experiment or provide multi-shape planning")
        inventory = inspect_checkpoint(directory, model_id=cohort.model_id, checkpoint_id=cohort.checkpoint_id, verify_payload=False)
        from shard.gpt_oss_contract import validate_supported_cohort
        validate_supported_cohort(cohort, inventory["config"])
        calibration = row["planning_calibration"]
        if isinstance(calibration, str):
            target = Path(calibration)
            if not target.is_absolute(): target = Path(args.config).resolve().parent / target
            calibration = json.loads(target.read_text(encoding="utf-8-sig"))
        workload = row.get("workload", {})
        if (workload.get("frame_tokens") != row.get("K", 4) + 1 or
                workload.get("draft_tokens") != row.get("K", 4) or workload.get("depth") != row.get("depth", 2)):
            raise ValueError("placement workload must match the actual K/depth verification geometry")
        return planning_profile(inventory, calibration, prompt_tokens=workload.get("context_tokens", 2048))
    service = OpenNetworkService(args.config, factory, profile_factory=profile)
    gateway = server = None
    try:
        service.form_all()
        gateway = Gateway(auth=AuthRegistry.from_file(args.auth_file), ring_pool=service.pool,
                          default_model=service.config.get("default_model"))
        server = gateway.server(args.host, args.port)
        if args.tls_cert:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(args.tls_cert, args.tls_key)
            server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
        print(json.dumps({"ready": True, "listen": server.server_address, "rings": service.pool.snapshot()}), flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server: server.server_close()
        if gateway: gateway.shutdown()
        service.close()


if __name__ == "__main__":
    main()

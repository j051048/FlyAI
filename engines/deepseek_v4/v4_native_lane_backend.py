"""Coordinator adapter for strict, native, tokenwise state-isolated lanes."""
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import hmac
import math
import secrets
import time

from v4_conversation_protocol import _sign, _verify, canonical
from v4_native_lane_protocol import SCHEMA, DOMAIN, ACK_DOMAIN, runtime_binding
from v4_native_lanes import CONTRACT
from shard.continuous_batch import contract_digest
from shard.execution_proof import verify_prefix_restore
from shard.pipeline_session import SessionConfig
from shard.service_queue import JobCancelled, JobExpired


def _validate_capacity(row, horizon):
    """Signed observations are untrusted capacity inputs, not allocation orders."""
    fields = {"enabled", "numerical_contract", "max_jobs", "horizon", "per_job_state_bytes",
        "reserved_state_bytes", "tensor_bytes_per_job", "metadata_bytes_per_job", "prefill_chunk_tokens",
        "restore_workspace_bytes", "durable_prefix", "ttl_s", "state_tier", "compute"}
    if (not isinstance(row, dict) or set(row) != fields or row["enabled"] is not True
            or row["numerical_contract"] != CONTRACT or type(row["max_jobs"]) is not int
            or not 2 <= row["max_jobs"] <= 4 or type(row["horizon"]) is not int or row["horizon"] != horizon
            or type(row["ttl_s"]) not in (int, float) or not math.isfinite(row["ttl_s"]) or not 0 < row["ttl_s"] <= 3600
            or type(row["prefill_chunk_tokens"]) is not int or row["prefill_chunk_tokens"] != 1
            or row["state_tier"] != "pageable_host" or row["compute"] != "serial_scalar_lanes"):
        raise ValueError("native calibrated capacity must be 2..4 finite bounded scalar lanes")
    for name in ("per_job_state_bytes", "reserved_state_bytes", "tensor_bytes_per_job", "metadata_bytes_per_job"):
        if type(row[name]) is not int or not 0 < row[name] <= 2 ** 50:
            raise ValueError("native state reservations must be bounded positive bytes")
    if (type(row["restore_workspace_bytes"]) is not int or not 0 <= row["restore_workspace_bytes"] <= 2 ** 50
            or row["per_job_state_bytes"] != row["tensor_bytes_per_job"] + row["metadata_bytes_per_job"]
            or row["reserved_state_bytes"] != row["per_job_state_bytes"] * row["max_jobs"] + row["restore_workspace_bytes"]):
        raise ValueError("native state capacity does not cover every lane and its restore workspace")
    disk = row["durable_prefix"]
    if (not isinstance(disk, dict) or set(disk) != {"enabled", "max_bytes"} or type(disk["enabled"]) is not bool
            or type(disk["max_bytes"]) is not int or not 0 <= disk["max_bytes"] <= 2 ** 50
            or (disk["enabled"] and (disk["max_bytes"] < row["tensor_bytes_per_job"] + (1 << 20)
                or row["restore_workspace_bytes"] < row["tensor_bytes_per_job"] + (2 << 20)))
            or (not disk["enabled"] and (disk["max_bytes"] != 0 or row["restore_workspace_bytes"] != 0))):
        raise ValueError("native durable capacity/workspace binding is invalid")


class NativeLaneBackend:
    def __init__(self, backend, options):
        self.backend = b = backend
        if (not isinstance(options, dict) or set(options) != {"enabled", "numerical_contract"}
                or options["enabled"] is not True or options["numerical_contract"] != CONTRACT):
            raise ValueError("native_lanes must explicitly select v4-native-tokenwise-greedy/1")
        if b.mode != "greedy" or b._policy is not None or b._conversation_owner is not None:
            raise ValueError("native lane service is a separate greedy cohort, without legacy cache/policy")
        if not b.pipeline_plan or b.pipeline_plan.get("model_cohort", {}).get("numeric_contract") != CONTRACT:
            raise ValueError("native lane service needs an explicitly distinct numerical model cohort")
        from v4_request_features import validate_selected_bindings
        validate_selected_bindings(b.pipeline_plan, b._runtime_configs, b._lease_fences)
        configs = [b._runtime_configs[row["node_id"]].get("native_lanes", {}) for row in b.pipeline_plan["stages"]]
        if any(row.get("enabled") is not True or row.get("numerical_contract") != CONTRACT
               or row.get("horizon") != b.max_context for row in configs):
            raise ValueError("every calibrated stage must provide the same native numerical contract/horizon")
        for row in configs:
            _validate_capacity(row, b.max_context)
        self.configs = configs
        self.runtime_bindings = [runtime_binding(b._runtime_configs[row["node_id"]]) for row in b.pipeline_plan["stages"]]
        self.max_jobs = min(row["max_jobs"] for row in configs)
        self.ttl_s = min(row["ttl_s"] for row in configs)
        self.runtime_digest = hashlib.sha256(canonical(b._runtime_configs)).hexdigest()
        source = {node: config.get("engine_source_sha256") for node, config in b._runtime_configs.items()}
        if any(not value for value in source.values()):
            raise ValueError("native lanes require calibrated implementation source hashes")
        self.contract = {"schema": "shard-token-batch/1", "model_id": b.model_id,
            "cohort_id": b.pipeline_plan["cohort_id"], "source_sha256": hashlib.sha256(canonical(source)).hexdigest(),
            "runtime_config_sha256": self.runtime_digest, "numerical_contract": CONTRACT,
            "max_jobs": self.max_jobs, "max_step_tokens": 1,
            "max_prompt_tokens": b.max_context - 1, "max_new_tokens": min(4096, b.max_context - 1),
            "max_context_tokens": b.max_context, "prefill_chunk_tokens": 1,
            "per_job_state_bytes": sum(row["per_job_state_bytes"] for row in configs),
            "reserved_state_bytes": sum(row["reserved_state_bytes"] for row in configs), "cancellation": "per_job"}
        self.jobs, self.cache = {}, OrderedDict()
        self.epoch = None
        self.last_cache_write_failures = []

    def identity(self):
        return {"cohort_id": self.contract["cohort_id"], "runtime_config_sha256": self.runtime_digest}

    @contextmanager
    def _io(self):
        b, owner = self.backend, object()
        with b._io_lock:
            with b._lock:
                if b._attempt_owner is not None:
                    raise ValueError("native socket ownership overlaps another coordinator attempt")
                b._attempt_owner = owner
            try:
                yield
            finally:
                with b._lock:
                    if b._attempt_owner is owner:
                        b._attempt_owner = None

    def _identity(self, tenant):
        b = self.backend
        return hmac.new(b._feature_secret, canonical([tenant, self.identity(), b._lease_fences,
                        self.epoch, CONTRACT, b.max_context]), hashlib.sha256).hexdigest()

    def _prefix(self, identity, ids):
        return {"token_count": len(ids), "digest": hmac.new(self.backend._feature_secret,
                canonical([identity, ids]), hashlib.sha256).hexdigest()}

    def _connect(self, job):
        b = self.backend
        if b._channels is None:
            options = {"session_config": SessionConfig.from_plan(b.pipeline_plan, index=-1,
                        ttl_s=3600, caller_key=b.coordinator_key)}
            remaining = job.deadline - time.monotonic()
            b._channels = b.vp.connect_ring(b.head, b.tail, **options,
                timeout=max(.05, min(.5, b.timeout, remaining)), token=b.vp.SWARM_TOKEN,
                retry_s=max(.05, min(.5, remaining)))
        grant = getattr(b._channels[0], "grant", None)
        if not grant:
            raise ValueError("native lanes require a live strict session owner grant")
        epoch = f"{grant['boot_id']}:{grant['fence']}"
        if self.epoch is not None and epoch != self.epoch:
            if self.jobs:
                raise ValueError("native owner session changed while lanes are active")
            self.cache.clear()
        self.epoch = epoch

    def _call(self, state, action, parameters, *, allow_failure=False):
        b = self.backend
        request = {"schema": SCHEMA, "contract": CONTRACT, "action": action,
            "request_id": secrets.token_hex(16), "job": state["job_binding"],
            "identity_sha256": state["identity"], "owner_epoch": self.epoch,
            "cohort_id": self.contract["cohort_id"], "horizon": b.max_context, "parameters": parameters}
        signed = _sign(b.coordinator_key, DOMAIN, request)
        remaining = state["deadline"] - time.monotonic()
        if remaining <= 0 and action != "close":
            raise JobExpired("native request deadline expired before the next state transition")
        timeout = min(b.timeout, max(.05, remaining)) if action != "close" else min(b.timeout, 5.0)
        for channel in b._channels:
            channel.settimeout(timeout)
        frame = {"op": "native_lane", "request": signed, "acks": []}
        if action == "open":
            # The existing strict forward session accepts a new owner only at
            # a reset barrier. Keep that transport fence: this specially marked
            # reset is additionally checked as a signed native OPEN at each node.
            frame.update(op="reset", native_lane_open=True)
        b.vp.send_msg(b._channels[0], frame)
        reply = b.vp.recv_msg(b._channels[1])
        if not isinstance(reply, dict) or reply.get("op") != "native_lane_ack":
            raise ValueError("unexpected native lane reply")
        rows = reply.get("acks")
        expected = b.pipeline_plan["stages"]
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise ValueError("native acknowledgement omits assigned stages")
        values, digest = [], hashlib.sha256(canonical(request)).hexdigest()
        for index, (row, assignment) in enumerate(zip(rows, expected)):
            ack = _verify(assignment["signer_pubkey"], ACK_DOMAIN, row)
            if (ack.get("stage") != index or ack.get("pubkey") != assignment["signer_pubkey"]
                    or ack.get("request_id") != request["request_id"] or ack.get("request_sha256") != digest
                    or ack.get("job") != request["job"] or ack.get("contract") != CONTRACT
                    or type(ack.get("ok")) is not bool):
                raise ValueError("native acknowledgement identity or transaction changed")
            values.append(ack)
        if not allow_failure and any(not row["ok"] for row in values):
            raise ValueError("native stage rejected operation: " + str([row["result"] for row in values if not row["ok"]])[:512])
        return values

    def _open(self, job):
        payload = job.payload
        if (not isinstance(payload, dict) or set(payload) != {"prompt_ids", "mode", "max_new", "numerical_contract"}
                or payload["mode"] != "greedy" or payload["numerical_contract"] != CONTRACT
                or type(payload["max_new"]) is not int or payload["max_new"] != job.max_new
                or not 1 <= job.max_new <= self.contract["max_new_tokens"]
                or not isinstance(payload["prompt_ids"], list) or len(payload["prompt_ids"]) != job.prompt_tokens
                or not 1 <= job.prompt_tokens <= self.contract["max_prompt_tokens"]
                or any(type(token) is not int or token < 0 for token in payload["prompt_ids"])
                or job.prompt_tokens + job.max_new > self.backend.max_context):
            raise ValueError("native lane request differs from its admitted numerical/token contract")
        self._connect(job)
        if len(self.jobs) >= self.max_jobs:
            raise ValueError("native job count exceeds calibrated slots")
        identity = self._identity(job.tenant)
        now = time.monotonic()
        prompt = list(job.payload["prompt_ids"])
        for key, row in tuple(self.cache.items()):
            if row["expires"] <= now:
                self.cache.pop(key)
        candidates = [row for row in self.cache.values() if row["identity"] == identity
                      and len(row["ids"]) < len(prompt) and prompt[:len(row["ids"])] == row["ids"]]
        parent = max(candidates, key=lambda row: len(row["ids"]), default=None) if not job.tokens else None
        # A restored prefix slot becomes this request's mutable lane. It cannot
        # remain advertised as an immutable reusable parent after opening.
        if parent:
            for cached_id, row in tuple(self.cache.items()):
                if row is parent:
                    self.cache.pop(cached_id)
                    break
        elif len(self.jobs) + len(self.cache) >= self.max_jobs and self.cache:
            self.cache.popitem(last=False)
        binding = {"job_id": job.id + "/attempt-1", "nonce": secrets.token_hex(32), "swarm_id": self.backend.swarm_id}
        restore = None
        if parent:
            restore = {"transaction_id": secrets.token_hex(16), "prefix": self._prefix(identity, parent["ids"]),
                       "identity_sha256": identity, "stages": deepcopy(parent["stages"])}
        state = {"job_binding": binding, "identity": identity, "restore": restore,
                 "pos": len(parent["ids"]) if parent else 0, "consumed": list(parent["ids"]) if parent else [],
                 "generated": [], "prior": job.checkpoint(), "started": time.perf_counter(),
                 "prompt": prompt, "restore_s": None, "first_s": None, "finished": False,
                 "deadline": job.deadline}
        values = self._call(state, "open", {"restore": restore} if restore else {}, allow_failure=True)
        if restore and any(not row["ok"] for row in values):
            self._call(state, "close", {"retain": False, "prefix": None})
            self.cache.clear()  # a partial cache cohort can never claim a hit
            state.update(restore=None, pos=0, consumed=[])
            values = self._call(state, "open", {})
        if any(not row["ok"] for row in values):
            self._call(state, "close", {"retain": False, "prefix": None})
            raise ValueError("native stage admission failed")
        for row, configured, expected_runtime in zip(values, self.configs, self.runtime_bindings):
            if (row["result"].get("capability") != configured or row["result"].get("pos") != state["pos"]
                    or row["result"].get("runtime_binding_sha256") != expected_runtime):
                self._call(state, "close", {"retain": False, "prefix": None})
                raise ValueError("loaded native capability differs from selected calibration")
        if state["restore"]:
            state["restore_s"] = time.perf_counter() - state["started"]
            state["restore_tiers"] = [row["result"].get("restore_tier") for row in values]
        self.jobs[job.id] = state
        return state

    def advance(self, jobs, *, emit, cancel_check, max_step_tokens):
        b = self.backend
        if type(max_step_tokens) is not int or max_step_tokens != 1 or len(jobs) > self.max_jobs:
            raise ValueError("native scalar lanes advance at most one token per job")
        results = {}
        with self._io():
            for job in jobs:
                try:
                    cancel_check[job.id]()
                    state = self.jobs.get(job.id) or self._open(job)
                    pos = state["pos"]
                    token = state["prompt"][pos] if pos < len(state["prompt"]) else state["generated"][-1]
                    rows = self._call(state, "step", {"pos": pos, "token": token})
                    if any(row["result"].get("pos") != pos + 1 for row in rows):
                        raise ValueError("native lane frontier differs across the ring")
                    state["pos"] += 1
                    state["consumed"].append(token)
                    results[job.id] = None
                    if state["pos"] >= len(state["prompt"]):
                        predicted = rows[-1]["result"].get("token")
                        if type(predicted) is not int or predicted < 0:
                            raise ValueError("native tail omitted its greedy token")
                        index = len(state["generated"])
                        if index < len(state["prior"]):
                            if predicted != state["prior"][index]:
                                raise ValueError("native original-request replay differs from committed prefix")
                        else:
                            emit[job.id](predicted)
                        state["generated"].append(predicted)
                        if state["first_s"] is None:
                            state["first_s"] = time.perf_counter() - state["started"]
                        if len(state["generated"]) >= job.max_new or predicted in b.eos_ids:
                            results[job.id] = self._complete(job, state)
                    cancel_check[job.id]()
                except (JobCancelled, JobExpired) as stopped:
                    if job.id in self.jobs:
                        self.jobs[job.id]["finished"] = False
                    results[job.id] = stopped
                except Exception:
                    b._close_channels()
                    raise
        return results

    def _complete(self, job, state):
        b = self.backend
        receipts = [row["result"]["receipt"] for row in self._call(state, "receipt", {})]
        if state["generated"] != job.checkpoint():
            raise ValueError("native returned tokens differ from committed output")
        binding = state["job_binding"]
        from shard.receipt import verify_coverage
        verify_coverage(receipts, b.layers, expected_by_signer=b.assignments,
                        expected_nonce=binding["nonce"], check_chain=True)
        if any(row.get("job_id") != binding["job_id"] or row.get("swarm_id") != b.swarm_id for row in receipts):
            raise ValueError("native receipt belongs to another job")
        expected_chunks = state["pos"] - (state["restore"]["prefix"]["token_count"] if state["restore"] else 0)
        if any(row.get("n_chunks") != expected_chunks for row in receipts):
            raise ValueError("native receipt does not cover each newly consumed token")
        if state["restore"]:
            verify_prefix_restore(receipts, state["restore"], b.assignments)
        elif any("conversation_restore" in row for row in receipts):
            raise ValueError("unexpected restored native prefix")
        response = {"tokens": job.checkpoint(), "text": b.decode(job.tokens),
            "numerical_contract": CONTRACT,
            "finish_reason": "length" if len(job.tokens) == job.max_new else "stop",
            "proof": {"verified": True, "scope": "fresh_suffix_with_signed_prefix_restore" if state["restore"] else "complete_final_attempt",
                      "attempt_id": binding["job_id"], "nonce": binding["nonce"], "swarm_id": b.swarm_id, "receipts": receipts},
            "recovery": {"attempts": 1, "replayed_committed_tokens": len(state["prior"]),
                         "strategy": "native_tokenwise_original_request_prefix_check"},
            "optimizations": {"native_lanes": {"numerical_contract": CONTRACT, "compute": "serial_scalar_lanes",
                "gpu_qualification": "requires_separate_native_gpu_release_gate",
                "restore_tiers": state.get("restore_tiers", []),
                "restored_prefix_tokens": state["restore"]["prefix"]["token_count"] if state["restore"] else 0}}}
        if state["restore"]:
            response["proof"]["prefix_restore"] = state["restore"]
        response["proof"]["validation_binding"] = b._proof_binding(job, response)
        state["finished"] = True
        b._last_ok = time.monotonic()
        return response

    def release(self, job_id):
        b = self.backend
        with self._io():
            state = self.jobs.get(job_id)
            if state is None:
                return
            if b._channels is None:
                raise ValueError("native state cleanup cannot be confirmed on a closed session")
            retain = state["finished"]
            prefix = self._prefix(state["identity"], state["consumed"]) if retain else None
            rows = self._call(state, "close", {"retain": retain, "prefix": prefix})
            if any(row["result"].get("released") is not True for row in rows):
                raise ValueError("native lane cleanup incomplete")
            self.jobs.pop(job_id)
            self.last_cache_write_failures = [{"stage": row["stage"], "reason": row["result"]["cache_write_error"]}
                for row in rows if row["result"].get("cache_write_error")]
            if retain:
                stages = []
                complete_cache = True
                for row in rows:
                    entry = row["result"]["cache"]
                    if entry is None:
                        complete_cache = False
                        continue
                    if entry["identity"] != state["identity"] or entry["prefix"] != prefix:
                        raise ValueError("native cached state differs from committed close boundary")
                    stages.append({"stage": row["stage"], "pubkey": row["pubkey"],
                                   "entry_id": entry["entry_id"], "state_digest": entry["state_digest"]})
                if not complete_cache:
                    self.cache.pop(job_id, None)
                    return
                self.cache[job_id] = {"identity": state["identity"], "ids": state["consumed"],
                    "stages": stages, "expires": time.monotonic() + self.ttl_s}
                while len(self.cache) > self.max_jobs:
                    self.cache.popitem(last=False)

    def execute(self, job, emit, cancel_check):
        try:
            while True:
                result = self.advance([job], emit={job.id: emit}, cancel_check={job.id: cancel_check}, max_step_tokens=1)[job.id]
                if isinstance(result, Exception):
                    raise result
                if result is not None:
                    return result
        finally:
            self.release(job.id)

    def status(self):
        return {"enabled": True, "numerical_contract": CONTRACT, "compute": "serial_scalar_lanes",
                "max_jobs": self.max_jobs, "active_jobs": len(self.jobs), "prefix_entries": len(self.cache),
                "supported_modes": ["greedy"], "mtp": False, "grouped_request_gemm": False,
                "per_job_runtime_metrics": "unavailable_shared_stage_counters",
                "last_cache_write_failures": deepcopy(self.last_cache_write_failures),
                "gpu_qualification": "requires_separate_native_gpu_release_gate",
                "prefix_policy": "exact_consumed_frontier_new_tokenwise_suffix_no_shadow"}

    def warmup(self, timeout_s=300):
        b = self.backend
        from shard.service_queue import Job
        prepared = b.prepare({"model": b.model_id, "messages": [{"role": "user", "content": "Hi"}],
                              "max_tokens": 2, "shard_mode": "greedy", "timeout_s": timeout_s})
        payload, count, maximum, timeout = prepared
        now = time.monotonic()
        # Multiple real native lanes must share the same live strict ring. Each
        # completion has independent signer/nonce/coverage checks above.
        jobs = [Job("__native_warmup__" + str(i), deepcopy(payload), count, maximum,
                    now + timeout, "native-warmup-" + str(i), None, now, state="running")
                for i in range(self.max_jobs)]
        pending = list(jobs)
        try:
            while pending:
                results = self.advance(pending, emit={j.id: j.commit for j in pending},
                    cancel_check={j.id: j.check_stop for j in pending}, max_step_tokens=1)
                for job in tuple(pending):
                    result = results[job.id]
                    if isinstance(result, Exception):
                        raise result
                    if result is not None:
                        b.verify_result_proof(job, result)
                        b._verify_warmup_profile(result)
                        self.release(job.id)
                        pending.remove(job)
            return {"ready": True, "proof_verified": True,
                    "committed_tokens": sum(len(j.tokens) for j in jobs),
                    "batch_contract_sha256": contract_digest(self.contract)}
        finally:
            for job in jobs:
                self.release(job.id)


def install(backend, options):
    if options is None or options == {"enabled": False}:
        if (backend.pipeline_plan or {}).get("model_cohort", {}).get("numeric_contract") == CONTRACT:
            raise ValueError("native tokenwise cohort requires explicit native_lanes configuration")
        return None
    client = NativeLaneBackend(backend, options)
    # Only an opted-in backend advertises the capability. A serial backend must
    # not accidentally acquire batch_contract merely by importing this module.
    backend.batch_contract = lambda: deepcopy(client.contract)
    backend.advance_batch = client.advance
    backend.release_batch_job = client.release
    return client

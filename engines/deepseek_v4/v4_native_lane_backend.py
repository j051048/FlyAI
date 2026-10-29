"""Coordinator adapter for strict, native, tokenwise state-isolated lanes."""
from collections import OrderedDict, deque
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
from v4_native_batch import BATCH_CONTRACT, PERSISTENT_BATCH_CONTRACT
from v4_batch_invariant import CONTRACT as MATH_CONTRACT
from shard.continuous_batch import contract_digest
from shard.execution_proof import verify_prefix_restore
from shard.pipeline_session import SessionConfig
from shard.service_queue import JobCancelled, JobExpired


def _validate_capacity(row, horizon):
    """Signed observations are untrusted capacity inputs, not allocation orders."""
    fields = {"enabled", "numerical_contract", "max_jobs", "horizon", "per_job_state_bytes",
        "reserved_state_bytes", "tensor_bytes_per_job", "metadata_bytes_per_job", "prefill_chunk_tokens",
        "restore_workspace_bytes", "durable_prefix", "ttl_s", "state_tier", "compute"}
    persistent = isinstance(row, dict) and row.get("numerical_contract") == PERSISTENT_BATCH_CONTRACT
    batch = isinstance(row, dict) and row.get("numerical_contract") in (BATCH_CONTRACT, PERSISTENT_BATCH_CONTRACT)
    if persistent:
        fields.add("persistent_policy")
    if batch:
        fields |= {"host_reserved_bytes", "device_reserved_bytes", "batch_workspace_bytes", "async_depth",
                   "reference", "graph_qualified", "hash_workspace_bytes", "math_contract", "graph_policy"}
    if (not isinstance(row, dict) or set(row) != fields or row["enabled"] is not True
            or row["numerical_contract"] not in (CONTRACT, BATCH_CONTRACT, PERSISTENT_BATCH_CONTRACT) or type(row["max_jobs"]) is not int
            or not 2 <= row["max_jobs"] <= 4 or type(row["horizon"]) is not int or row["horizon"] != horizon
            or type(row["ttl_s"]) not in (int, float) or not math.isfinite(row["ttl_s"]) or not 0 < row["ttl_s"] <= 3600
            or type(row["prefill_chunk_tokens"]) is not int
            or not 1 <= row["prefill_chunk_tokens"] <= (8 if batch else 1)
            or (not batch and (row["state_tier"] != "pageable_host" or row["compute"] != "serial_scalar_lanes"))
            or (batch and (row["state_tier"] not in ("device", "cpu_reference")
                          or row["compute"] != ("native_persistent_ragged_batch" if persistent else "native_equal_frontier_batch")))):
        raise ValueError("native calibrated capacity must be 2..4 finite bounded scalar lanes")
    if batch:
        graph = row["graph_policy"]
        if (not isinstance(graph, dict) or set(graph) != {"mode", "graph_bytes", "host_metadata_bytes",
                "max_graphs", "coverage", "selection"} or graph["mode"] not in ("off", "partial")
                or any(type(graph[k]) is not int or not 0 <= graph[k] <= 2 ** 50
                       for k in ("graph_bytes", "host_metadata_bytes", "max_graphs"))
                or type(row["batch_workspace_bytes"]) is not int or row["batch_workspace_bytes"] <= 0
                or type(row["hash_workspace_bytes"]) is not int or row["hash_workspace_bytes"] != 2 << 20
                or graph["selection"] != "native_exact_width_topk"
                or (graph["mode"] == "off" and any(graph[k] != 0 for k in
                    ("graph_bytes", "host_metadata_bytes", "max_graphs")))
                or (graph["mode"] == "partial" and (not 2 <= graph["max_graphs"] <= 4096
                    or graph["graph_bytes"] <= row["batch_workspace_bytes"]
                    or graph["host_metadata_bytes"] != graph["max_graphs"] * 65536))
                or graph["coverage"] != ("attention_dense_partial" if graph["mode"] == "partial" else "off")
                or row["math_contract"] != MATH_CONTRACT):
            raise ValueError("native graph policy/math contract differs from bounded partial capture")
        if persistent:
            policy = row["persistent_policy"]
            if (not isinstance(policy, dict) or policy != {
                    "schema": "v4-persistent-request-slots/1", "positions": "per_request",
                    "attention": "batched_projections_native_row_reads", "state": "direct_slot_views",
                    "transaction": "bounded_native_write_set", "journal_bytes": row["batch_workspace_bytes"],
                    "full_state_copy_per_step": False} or graph["mode"] != "off"):
                raise ValueError("persistent native state policy must bind its exact write journal and eager arithmetic")
    for name in ("per_job_state_bytes", "reserved_state_bytes", "tensor_bytes_per_job", "metadata_bytes_per_job"):
        if type(row[name]) is not int or not 0 < row[name] <= 2 ** 50:
            raise ValueError("native state reservations must be bounded positive bytes")
    if (type(row["restore_workspace_bytes"]) is not int or not 0 <= row["restore_workspace_bytes"] <= 2 ** 50
            or row["per_job_state_bytes"] != row["tensor_bytes_per_job"] + row["metadata_bytes_per_job"]
            or row["reserved_state_bytes"] != row["per_job_state_bytes"] * row["max_jobs"] +
                (row["batch_workspace_bytes"] + row["hash_workspace_bytes"] +
                 row["graph_policy"]["graph_bytes"] + row["graph_policy"]["host_metadata_bytes"]
                 if batch else row["restore_workspace_bytes"])):
        raise ValueError("native state capacity does not cover every lane and its restore workspace")
    disk = row["durable_prefix"]
    if (not isinstance(disk, dict) or set(disk) != {"enabled", "max_bytes"} or type(disk["enabled"]) is not bool
            or type(disk["max_bytes"]) is not int or not 0 <= disk["max_bytes"] <= 2 ** 50
            or (disk["enabled"] and (disk["max_bytes"] < row["tensor_bytes_per_job"] + (1 << 20)
                or row["restore_workspace_bytes"] < row["tensor_bytes_per_job"] + (2 << 20)))
            or (not disk["enabled"] and (disk["max_bytes"] != 0 or row["restore_workspace_bytes"] != 0))):
        raise ValueError("native durable capacity/workspace binding is invalid")
    if batch:
        if (any(type(row[k]) is not int or not 0 < row[k] <= 2 ** 50 for k in
                ("host_reserved_bytes", "device_reserved_bytes", "batch_workspace_bytes", "async_depth"))
                or not 1 <= row["async_depth"] <= row["max_jobs"]
                or type(row["reference"]) is not bool or row["reference"] != (row["state_tier"] == "cpu_reference")
                or row["graph_qualified"] is not False or disk["enabled"]
                or (not persistent and row["batch_workspace_bytes"] != row["tensor_bytes_per_job"] * row["max_jobs"])
                or (persistent and row["batch_workspace_bytes"] > row["tensor_bytes_per_job"] * row["max_jobs"])
                or row["device_reserved_bytes"] != row["tensor_bytes_per_job"] * row["max_jobs"] + row["batch_workspace_bytes"] + graph["graph_bytes"]
                or type(row["hash_workspace_bytes"]) is not int or row["hash_workspace_bytes"] != 2 << 20
                or row["host_reserved_bytes"] != row["metadata_bytes_per_job"] * row["max_jobs"] +
                    row["hash_workspace_bytes"] + graph["host_metadata_bytes"] +
                    (row["device_reserved_bytes"] if row["reference"] else 0)):
            raise ValueError("native device batch reservation/reference/capture binding is invalid")


class NativeLaneBackend:
    def __init__(self, backend, options):
        self.backend = b = backend
        if (not isinstance(options, dict) or set(options) - {"enabled", "numerical_contract", "prefill_quantum", "prefill_budget"}
                or not {"enabled", "numerical_contract"}.issubset(options)
                or options["enabled"] is not True or options["numerical_contract"] not in (CONTRACT, BATCH_CONTRACT, PERSISTENT_BATCH_CONTRACT)):
            raise ValueError("native_lanes must explicitly select its native numerical contract")
        self.numerical_contract = options["numerical_contract"]
        self.is_persistent = self.numerical_contract == PERSISTENT_BATCH_CONTRACT
        self.is_batch = self.numerical_contract in (BATCH_CONTRACT, PERSISTENT_BATCH_CONTRACT)
        self.compute = "native_persistent_ragged_batch" if self.is_persistent else "native_equal_frontier_batch" if self.is_batch else "serial_scalar_lanes"
        if b.mode != "greedy" or b._policy is not None or b._conversation_owner is not None:
            raise ValueError("native lane service is a separate greedy cohort, without legacy cache/policy")
        if not b.pipeline_plan or b.pipeline_plan.get("model_cohort", {}).get("numeric_contract") != self.numerical_contract:
            raise ValueError("native lane service needs an explicitly distinct numerical model cohort")
        from v4_request_features import validate_selected_bindings
        validate_selected_bindings(b.pipeline_plan, b._runtime_configs, b._lease_fences)
        configs = [b._runtime_configs[row["node_id"]].get("native_lanes", {}) for row in b.pipeline_plan["stages"]]
        if any(row.get("enabled") is not True or row.get("numerical_contract") != self.numerical_contract
               or row.get("horizon") != b.max_context for row in configs):
            raise ValueError("every calibrated stage must provide the same native numerical contract/horizon")
        for row in configs:
            _validate_capacity(row, b.max_context)
        self.configs = configs
        self.runtime_bindings = [runtime_binding(b._runtime_configs[row["node_id"]]) for row in b.pipeline_plan["stages"]]
        self.max_jobs = min(row["max_jobs"] for row in configs)
        self.async_depth = min(row.get("async_depth", 1) for row in configs)
        self.prefill_quantum = options.get("prefill_quantum", 1)
        if (type(self.prefill_quantum) is not int or not 1 <= self.prefill_quantum <=
                min(row["prefill_chunk_tokens"] for row in configs)
                or not self.is_batch and (self.prefill_quantum != 1 or options.get("prefill_budget") is not None)):
            raise ValueError("native prefill quantum exceeds calibrated tokenwise capability")
        self.pipeline_stats = {"sent_cohorts": 0, "received_cohorts": 0, "peak_inflight_cohorts": 0,
                               "batch_forward_cohorts": 0, "scalar_forward_cohorts": 0,
                               "batch_request_rows": 0, "cancelled_after_drain": 0}
        self.cohort_timings = deque(maxlen=512)
        self.ttl_s = min(row["ttl_s"] for row in configs)
        self.runtime_digest = hashlib.sha256(canonical(b._runtime_configs)).hexdigest()
        source = {node: config.get("engine_source_sha256") for node, config in b._runtime_configs.items()}
        if any(not value for value in source.values()):
            raise ValueError("native lanes require calibrated implementation source hashes")
        self.contract = {"schema": "shard-token-batch/1", "model_id": b.model_id,
            "cohort_id": b.pipeline_plan["cohort_id"], "source_sha256": hashlib.sha256(canonical(source)).hexdigest(),
            "runtime_config_sha256": self.runtime_digest, "numerical_contract": self.numerical_contract,
            "max_jobs": self.max_jobs, "max_step_tokens": 1,
            "max_prompt_tokens": b.max_context - 1, "max_new_tokens": min(4096, b.max_context - 1),
            "max_context_tokens": b.max_context, "prefill_chunk_tokens": self.prefill_quantum,
            "per_job_state_bytes": sum(row["per_job_state_bytes"] for row in configs),
            "reserved_state_bytes": sum(row["reserved_state_bytes"] for row in configs), "cancellation": "per_job"}
        self.jobs, self.cache = {}, OrderedDict()
        self.prefill_budget = None
        if options.get("prefill_budget") is not None:
            from v4_prefill_budget import TimeBudget
            binding = {key: self.contract[key] for key in ("numerical_contract", "cohort_id", "source_sha256",
                       "runtime_config_sha256", "max_context_tokens", "max_jobs")}
            self.prefill_budget = TimeBudget(options["prefill_budget"], binding)
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
                        self.epoch, self.numerical_contract, b.max_context]), hashlib.sha256).hexdigest()

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
        request = {"schema": SCHEMA, "contract": self.numerical_contract, "action": action,
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
                    or ack.get("job") != request["job"] or ack.get("contract") != self.numerical_contract
                    or type(ack.get("ok")) is not bool):
                raise ValueError("native acknowledgement identity or transaction changed")
            values.append(ack)
        if not allow_failure and any(not row["ok"] for row in values):
            raise ValueError("native stage rejected operation: " + str([row["result"] for row in values if not row["ok"]])[:512])
        return values

    def _open(self, job):
        payload = job.payload
        if (not isinstance(payload, dict) or set(payload) != {"prompt_ids", "mode", "max_new", "numerical_contract"}
                or payload["mode"] != "greedy" or payload["numerical_contract"] != self.numerical_contract
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
        if self.is_batch:
            positions = [self.jobs.get(job.id, {}).get("pos", 0) for job in jobs]
            remaining = [len(self.jobs.get(job.id, {}).get("prompt", job.payload.get("prompt_ids", [])))
                         - 1 - pos for job, pos in zip(jobs, positions)]
            # A measured [context, batch, quantum] is applicable only while the
            # complete request batch keeps exactly that shape for every round.
            # Decode, ragged positions and a short prompt use baseline one.
            limit = min([self.prefill_quantum, *remaining]) if remaining else 1
            eligible = bool(jobs) and len(set(positions)) == 1 and limit > 1
            quantum = self.prefill_budget.choose(candidates=list(range(1, limit + 1)),
                context=positions[0], batch_jobs=len(jobs)) if self.prefill_budget is not None and eligible else 1
            # Measurement is the only authority for a quantum above one.
            if type(quantum) is not int or not 1 <= quantum <= self.prefill_quantum:
                raise ValueError("measured native prefill quantum exceeds the admitted bound")
            pending, results = list(jobs), {}
            for _ in range(quantum):
                current = self._advance_device_batch(pending, emit=emit, cancel_check=cancel_check)
                results.update(current)
                pending = [job for job in pending if current[job.id] is None and job.id in self.jobs
                           and self.jobs[job.id]["pos"] < len(self.jobs[job.id]["prompt"]) - 1]
                if len(pending) != len(jobs):
                    # A cancellation/deadline changes the shape. Re-plan at the
                    # next scheduler boundary, without using the old timing.
                    break
            return results
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
            "numerical_contract": self.numerical_contract,
            "finish_reason": "length" if len(job.tokens) == job.max_new else "stop",
            "proof": {"verified": True, "scope": "fresh_suffix_with_signed_prefix_restore" if state["restore"] else "complete_final_attempt",
                      "attempt_id": binding["job_id"], "nonce": binding["nonce"], "swarm_id": b.swarm_id, "receipts": receipts},
            "recovery": {"attempts": 1, "replayed_committed_tokens": len(state["prior"]),
                         "strategy": "native_tokenwise_original_request_prefix_check"},
            "optimizations": {"native_lanes": {"numerical_contract": self.numerical_contract, "compute": self.compute,
                "gpu_qualification": "requires_separate_native_gpu_release_gate",
                "restore_tiers": state.get("restore_tiers", []),
                "restored_prefix_tokens": state["restore"]["prefix"]["token_count"] if state["restore"] else 0}}}
        if state["restore"]:
            response["proof"]["prefix_restore"] = state["restore"]
        response["proof"]["validation_binding"] = b._proof_binding(job, response)
        state["finished"] = True
        b._last_ok = time.monotonic()
        return response

    def _advance_device_batch(self, jobs, *, emit, cancel_check):
        """Send disjoint frontier cohorts ahead, drain, then publish each job.

        The first tail reply is not awaited until up to async_depth real forward
        frames have entered the ring. One job has at most one outstanding token.
        Cancellation after submit conserves the consumed frontier and suppresses
        output publication; release subsequently closes that exact drained lane.
        """
        from v4_native_pipeline import drain_pipeline
        b, results, ready = self.backend, {}, []
        if len({job.id for job in jobs}) != len(jobs):
            raise ValueError("native batch cannot repeat an admitted job")
        with self._io():
            try:
                for job in jobs:
                    try:
                        cancel_check[job.id]()
                        state = self.jobs.get(job.id) or self._open(job)
                        if state["finished"]:
                            raise ValueError("completed native lane must be released before advancing")
                        pos = state["pos"]
                        token = state["prompt"][pos] if pos < len(state["prompt"]) else state["generated"][-1]
                        ready.append((job, state, token))
                    except (JobCancelled, JobExpired) as stopped:
                        if job.id in self.jobs:
                            self.jobs[job.id]["finished"] = False
                        results[job.id] = stopped
                launchable = []
                for row in ready:
                    job, state, _ = row
                    try:
                        cancel_check[job.id]()
                        launchable.append(row)
                    except (JobCancelled, JobExpired) as stopped:
                        state["finished"] = False
                        results[job.id] = stopped
                cohorts = OrderedDict()
                for row in launchable:
                    state = row[1]
                    # Only decode/first-output cohorts may populate the graph
                    # pool. Prompt length can differ even at the same frontier.
                    capture_graph = state["pos"] >= len(state["prompt"]) - 1
                    cohorts.setdefault((None if self.is_persistent else state["pos"], capture_graph), []).append(row)
                requests = []
                ordered = sorted(cohorts.items(), key=lambda item: not item[0][1])
                for (pos, capture_graph), rows in ordered:
                    steps = [self._step_request(state, "step", {"pos": state["pos"], "token": token,
                                                              "capture_graph": capture_graph})
                             for _, state, token in rows]
                    request = self._step_request(rows[0][1], "batch_step",
                        {"steps": [_sign(b.coordinator_key, DOMAIN, step) for step in steps]})
                    requests.append((request, rows))
                submitted = {}
                def send(item):
                    request, rows = item
                    remaining = min(state["deadline"] for _, state, _ in rows) - time.monotonic()
                    # An expired request is detected before submit; after submit
                    # all replies drain before the next cancellation inspection.
                    if remaining <= 0:
                        raise JobExpired("native batch expired before socket submission")
                    for channel in b._channels:
                        channel.settimeout(min(b.timeout, max(.05, remaining)))
                    submitted[request["request_id"]] = time.perf_counter()
                    b.vp.send_msg(b._channels[0], {"op": "native_lane", "request":
                        _sign(b.coordinator_key, DOMAIN, request), "acks": []})
                def receive(item):
                    request, rows = item
                    reply = self._batch_reply(request, b.vp.recv_msg(b._channels[1]))
                    end = time.perf_counter()
                    self.cohort_timings.append({"context_tokens": rows[0][1]["pos"],
                        "batch_jobs": len(rows), "quantum_tokens": 1,
                        "elapsed_ms": max(.000001, (end - submitted[request["request_id"]]) * 1000),
                        "submitted_at": submitted[request["request_id"]], "drained_at": end,
                        "cohort_sequence": self.pipeline_stats["received_cohorts"] + 1,
                        "owner_epoch": self.epoch, "job_set_sha256": hashlib.sha256(canonical(sorted(
                            [state["job_binding"] for _, state, _ in rows], key=lambda value: value["job_id"]))).hexdigest(),
                        "prefill": all(state["pos"] < len(state["prompt"]) - 1 for _, state, _ in rows),
                        "measured_at": time.time()})
                    return rows, reply
                received = drain_pipeline(requests, depth=self.async_depth, send=send,
                                          receive=receive, stats=self.pipeline_stats)
                for rows, stage_rows in received:
                    count = len(rows)
                    self.pipeline_stats["batch_forward_cohorts" if count > 1 else "scalar_forward_cohorts"] += 1
                    if count > 1:
                        self.pipeline_stats["batch_request_rows"] += count
                    for index, (job, state, token) in enumerate(rows):
                        pos = state["pos"]
                        for stage_row in stage_rows:
                            step = stage_row["result"]["steps"][index]
                            if step.get("job") != state["job_binding"] or step.get("pos") != pos + 1:
                                raise ValueError("native batch lane order/frontier differs across the ring")
                        state["pos"] += 1
                        state["consumed"].append(token)
                        results[job.id] = None
                        try:
                            cancel_check[job.id]()
                            if state["pos"] >= len(state["prompt"]):
                                predicted = stage_rows[-1]["result"]["steps"][index].get("token")
                                if type(predicted) is not int or predicted < 0:
                                    raise ValueError("native batch tail omitted a greedy token")
                                offset = len(state["generated"])
                                if offset < len(state["prior"]):
                                    if predicted != state["prior"][offset]:
                                        raise ValueError("native batch replay differs from committed prefix")
                                else:
                                    emit[job.id](predicted)
                                state["generated"].append(predicted)
                                if state["first_s"] is None:
                                    state["first_s"] = time.perf_counter() - state["started"]
                                if len(state["generated"]) >= job.max_new or predicted in b.eos_ids:
                                    results[job.id] = self._complete(job, state)
                            cancel_check[job.id]()
                        except (JobCancelled, JobExpired) as stopped:
                            state["finished"] = False
                            results[job.id] = stopped
                            self.pipeline_stats["cancelled_after_drain"] += 1
            except Exception:
                # A transport/proof failure makes remote cleanup unconfirmable.
                # Discard every local frontier; the next owner epoch clears node
                # states before admission instead of reusing a partial cohort.
                b._close_channels()
                self.jobs.clear()
                self.cache.clear()
                self.epoch = None
                raise
        return results

    def _step_request(self, state, action, parameters):
        return {"schema": SCHEMA, "contract": self.numerical_contract, "action": action,
            "request_id": secrets.token_hex(16), "job": state["job_binding"],
            "identity_sha256": state["identity"], "owner_epoch": self.epoch,
            "cohort_id": self.contract["cohort_id"], "horizon": self.backend.max_context,
            "parameters": parameters}

    def _batch_reply(self, request, reply):
        if not isinstance(reply, dict) or reply.get("op") != "native_lane_ack":
            raise ValueError("unexpected native batch reply")
        rows = reply.get("acks")
        expected = self.backend.pipeline_plan["stages"]
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise ValueError("native batch acknowledgement omits assigned stages")
        values, digest = [], hashlib.sha256(canonical(request)).hexdigest()
        for index, (signed, assignment) in enumerate(zip(rows, expected)):
            ack = _verify(assignment["signer_pubkey"], ACK_DOMAIN, signed)
            if (ack.get("stage") != index or ack.get("pubkey") != assignment["signer_pubkey"]
                    or ack.get("request_id") != request["request_id"] or ack.get("request_sha256") != digest
                    or ack.get("job") != request["job"] or ack.get("contract") != self.numerical_contract
                    or ack.get("ok") is not True):
                raise ValueError("native batch acknowledgement identity/transaction or stage forward failed")
            result = ack.get("result", {})
            count = len(request["parameters"]["steps"])
            capture_graph = request["parameters"]["steps"][0]["parameters"]["capture_graph"]
            if (result.get("actual_batch_size") != count or not isinstance(result.get("steps"), list)
                    or len(result["steps"]) != count or result.get("compute") != self.compute
                    or result.get("graph_capture_allowed") is not capture_graph):
                raise ValueError("native batch reply does not cover every actual request row")
            values.append(ack)
        return values

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
        return {"enabled": True, "numerical_contract": self.numerical_contract, "compute": self.compute,
                "max_jobs": self.max_jobs, "active_jobs": len(self.jobs), "prefix_entries": len(self.cache),
                "supported_modes": ["greedy"], "mtp": False, "grouped_request_gemm": False,
                "request_axis_batch": self.is_batch, "math_contract": MATH_CONTRACT if self.is_batch else None,
                "batching_geometry": "independent_frontier_rows" if self.is_persistent else "equal_frontier_subgroups" if self.is_batch else None,
                "graph_qualified": False, "async_depth": self.async_depth,
                "prefill_quantum_tokens": self.prefill_quantum,
                "pipeline": deepcopy(self.pipeline_stats),
                "prefill_budget": self.prefill_budget.status() if self.prefill_budget is not None else None,
                "cohort_timing_samples": len(self.cohort_timings),
                "cohort_timing_scope": "coordinator_submit_to_verified_drain_including_network_and_queue",
                "per_job_runtime_metrics": "unavailable_shared_stage_counters",
                "last_cache_write_failures": deepcopy(self.last_cache_write_failures),
                "gpu_qualification": "requires_separate_native_gpu_release_gate",
                "prefix_policy": "exact_consumed_frontier_new_tokenwise_suffix_no_shadow"}

    def export_prefill_samples(self, *, quota_ms, ttl_s=300):
        """Bounded actual adjacent token-round wall spans, without extrapolation.

        At least five observations of one exact shape are required before that
        shape is exported as a calibration input. Consecutive q1 observations
        may establish q-span elapsed cost including real CPU gaps; they do not
        establish a single advance(q), another runtime or physical GPU gate.
        """
        from v4_prefill_budget import SCHEMA as BUDGET_SCHEMA, validate_config
        from v4_native_pipeline import observed_token_spans
        grouped = OrderedDict()
        measured_at = None
        now = time.time()
        eligible = [sample for sample in self.cohort_timings if 0 <= now - sample["measured_at"] <= ttl_s]
        for sample in observed_token_spans(eligible):
            shape = tuple(sample[k] for k in ("context_tokens", "batch_jobs", "quantum_tokens"))
            grouped.setdefault(shape, []).append(sample["elapsed_ms"])
            measured_at = min(measured_at or sample["measured_at"], sample["measured_at"])
        samples = [{"context_tokens": shape[0], "batch_jobs": shape[1], "quantum_tokens": shape[2],
                    "elapsed_ms": values[-128:]} for shape, values in grouped.items() if len(values) >= 5]
        dropped_shapes = max(0, len(samples) - 512)
        samples = samples[-512:]
        binding = {key: self.contract[key] for key in ("numerical_contract", "cohort_id", "source_sha256",
                   "runtime_config_sha256", "max_context_tokens", "max_jobs")}
        if not samples:
            return {"ready": False, "binding": binding, "reason": "need_five_actual_samples_per_exact_shape",
                    "retained_samples": len(self.cohort_timings), "max_samples": self.cohort_timings.maxlen}
        return {"ready": True, "scope": "observed_consecutive_token_round_span_including_cpu_gaps",
                "single_advance_quantum_observed": False,
                "dropped_shapes": dropped_shapes,
                "gpu_qualified": False, "calibration": validate_config({"schema": BUDGET_SCHEMA,
                    "measurement_scope": "observed_consecutive_token_round_span_including_cpu_gaps",
                    "binding": binding, "measured_at": measured_at, "ttl_s": ttl_s,
                    "quota_ms": quota_ms, "samples": samples})}

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
        if (backend.pipeline_plan or {}).get("model_cohort", {}).get("numeric_contract") in (CONTRACT, BATCH_CONTRACT, PERSISTENT_BATCH_CONTRACT):
            raise ValueError("native tokenwise cohort requires explicit native_lanes configuration")
        return None
    client = NativeLaneBackend(backend, options)
    # Only an opted-in backend advertises the capability. A serial backend must
    # not accidentally acquire batch_contract merely by importing this module.
    backend.batch_contract = lambda: deepcopy(client.contract)
    backend.advance_batch = client.advance
    backend.release_batch_job = client.release
    backend.export_native_prefill_samples = client.export_prefill_samples
    return client

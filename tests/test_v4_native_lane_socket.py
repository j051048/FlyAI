"""Real native V4 weights/math, strict TCP sessions and signed per-lane receipts.

Only the tiny model's dimensions and tokenizer are test fixtures; transport,
state copies, native forwards, cancellation and receipt validation are real.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import http.client
import json
import threading
import time

import pytest
import torch

import v4_ref_cpu as R
import v4_stage as V
import v4_pipe as VP
from v4_gateway import V4RingBackend
from v4_native_lanes import CONTRACT
from v4_resources import runtime_config_payload
from shard.offers import ModelCohort
from shard.pipeline_plan import build_plan
from shard.pipeline_session import SessionConfig
from shard.receipt import gen_key, pub_b64, verify_coverage
from shard.service_queue import Job, JobCancelled
from shard import weight_artifacts as ART


class TinyTokenizer:
    eos_token_id = None
    def decode(self, tokens, **kwargs): return " ".join(map(str, tokens))


@pytest.fixture
def ring(tmp_path, monkeypatch, request):
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    monkeypatch.setenv("V4_NATIVE_LANES", "2")
    monkeypatch.setenv("V4_NATIVE_LANE_HOST_MIB", "32")
    if getattr(request, "param", None) == "durable":
        import v4_native_lanes as native
        # Each production stage owns its private configured path. Threads in
        # this tiny fixture share one process environment, so assign paths by
        # the real layer span without changing any storage/protocol operation.
        def configured(st):
            options = native.configured_options()
            options.update(state_directory=tmp_path / f"private-{st.lo}-{st.hi}", disk_bytes=16 << 20)
            return native.NativeLanes(st, **options)
        monkeypatch.setattr(native, "configured_runner", configured)
        close = native.NativeLanes.close
        def evict_memory(runner, *a, **kw):
            entry = close(runner, *a, **kw)
            if entry is not None and not runner.active:
                runner.clear()  # A real RAM eviction: subsequent open must read disk.
            return entry
        monkeypatch.setattr(native.NativeLanes, "close", evict_memory)
    args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
        dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=64)
    oracle = R.build_oracle(args, 7)
    VP._write_tiny_checkpoint(str(tmp_path), args, oracle)
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    catalog, pack = ART.catalogue_directory(tmp_path, VP.V4_MODEL_ID, source={"fixture": "native CPU tokenwise"})
    ART.write_metadata(tmp_path, catalog, pack)
    cohort = ModelCohort(catalog["model_id"], catalog["manifest_sha256"], catalog["checkpoint_id"],
        catalog["config_sha256"], "cpu-bf16-test", catalog["runtime_abi"], "shard-pipeline-session/1", CONTRACT, 4)
    # Keep actual hash/coverage verification. Only the production 43-layer/FP4
    # dimensional check is inapplicable to this tiny native mathematical test.
    verify = VP.verify_cohort_directory
    monkeypatch.setattr(VP, "verify_cohort_directory", lambda *a, **kw: verify(*a, **kw, production=False))
    keys = [gen_key(), gen_key(), gen_key()]
    monkeypatch.setattr(VP, "load_or_make_node_key", lambda path: keys[int(str(path)[-1])])
    monkeypatch.setattr(VP, "_encode_prompt", lambda tokenizer, body: [int(x) for m in body["messages"] for x in m["content"].split()])
    ports = VP._free_ports(2)
    plan = build_plan({"n_layers": 4}, ring_id="native-cpu-ring", cohort_id=cohort.cohort_id,
        endpoints=[f"127.0.0.1:{p}" for p in ports], split="2,2", model_cohort=cohort.to_dict())
    configs, leases = {}, {}
    for index, row in enumerate(plan["stages"]):
        row["signer_pubkey"] = pub_b64(keys[index])
        st = V.Stage(row["lo"], row["hi"], args, head=row["head"], tail=row["tail"], device="cpu").load(tmp_path)
        cfg = runtime_config_payload(st)
        row["runtime_config_sha256"] = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        configs[row["node_id"]] = cfg
        leases[row["node_id"]] = {"lease_id": "cpu-lease-" + str(index), "fencing_token": 1}
    plan["coordinator"]["signer_pubkey"] = pub_b64(keys[2])
    sessions = [SessionConfig.from_plan(plan, i, caller_key=keys[i]) for i in range(2)]
    events, errors, threads = [threading.Event(), threading.Event()], [], []
    def serve(index):
        row = plan["stages"][index]
        try:
            VP.serve_stage(index, 2, row["lo"], row["hi"], ports[index], nxt=row["next_endpoint"],
                ckpt_dir=str(tmp_path), args=args, device="cpu", receipts=True, key_path="key" + str(index),
                ready=events[index], session_config=sessions[index], timeout=8)
        except Exception as error:
            errors.append(error); events[index].set()
    for index in (1, 0):
        t = threading.Thread(target=serve, args=(index,), daemon=True); t.start(); threads.append(t)
    assert all(event.wait(20) for event in events) and not errors, errors
    adapter = V4RingBackend(tmp_path, plan["coordinator"]["head"], plan["coordinator"]["tail"],
        {row["signer_pubkey"]: (row["lo"], row["hi"]) for row in plan["stages"]},
        mode="greedy", vp=VP, tokenizer=TinyTokenizer(), timeout=8, max_context=64,
        swarm_id=plan["ring_id"], pipeline_plan=plan, coordinator_key=keys[2],
        calibrated_runtime_configs=configs, lease_fences=leases,
        native_lanes={"enabled": True, "numerical_contract": CONTRACT})
    # Preserve the oracle's exact native layers/weights for tokenwise controls.
    baseline = V.Stage(0, 4, args, head=True, tail=True, device="cpu").load(tmp_path)
    try:
        yield adapter, baseline
    finally:
        if adapter._channels:
            try: VP.send_msg(adapter._channels[0], {"op": "stop"})
            except OSError: pass
        adapter.close()
        for t in threads: t.join(10)
        torch.set_num_threads(old_threads)
        assert not errors and not any(t.is_alive() for t in threads), errors


def job(prompt, count=4, tenant="a"):
    now = time.monotonic()
    return Job(tenant, {"prompt_ids": list(prompt), "mode": "greedy", "max_new": count,
                       "numerical_contract": CONTRACT}, len(prompt), count, now + 40,
               "native-test", None, now, state="running")


def reference(st, prompt, maximum):
    st.reset()
    ids, generated = list(prompt), []
    with torch.no_grad():
        for pos in range(len(prompt) + maximum - 1):
            token = ids[pos] if pos < len(prompt) else generated[-1]
            value = torch.tensor([[token]])
            h = st.forward(st.embed(value), value, pos)
            if pos >= len(prompt) - 1:
                generated.append(int(st.logits_all(h, full_logits=False).argmax()))
    return generated


def test_real_strict_socket_interleaves_native_jobs_and_restores_normal_growing_prefix(ring):
    backend, control = ring
    first, second = job(range(1, 14)), job(range(31, 39), 3, "b")
    wanted = [reference(control, j.payload["prompt_ids"], j.max_new) for j in (first, second)]
    active, results = [first, second], {}
    while active:
        stepped = backend.advance_batch(active, emit={j.id: j.commit for j in active},
            cancel_check={j.id: j.check_stop for j in active}, max_step_tokens=1)
        for request in tuple(active):
            if stepped[request.id] is not None:
                results[request.id] = stepped[request.id]
                backend.verify_result_proof(request, stepped[request.id])
                backend.release_batch_job(request.id)
                active.remove(request)
    assert [first.tokens, second.tokens] == wanted
    assert results[first.id]["proof"]["scope"] == "complete_final_attempt"
    # Ordinary new chat prompt with a matching token prefix: reuse the consumed
    # native state, process the remaining new input only, no full-prefill shadow.
    longer = first.payload["prompt_ids"] + first.tokens + [21, 22]
    third = job(longer, 4)
    expected = reference(control, longer, 4)
    result = backend.execute(third, third.commit, third.check_stop)
    assert third.tokens == expected
    assert result["proof"]["scope"] == "fresh_suffix_with_signed_prefix_restore"
    restored = result["proof"]["prefix_restore"]["prefix"]["token_count"]
    assert restored == len(first.payload["prompt_ids"]) + first.max_new - 1
    assert backend.verify_result_proof(third, result)
    for receipt in result["proof"]["receipts"]:
        assert receipt["n_chunks"] == len(longer) + third.max_new - 1 - restored
        assert receipt["runtime_observation"]["phase"] == "job_complete"
        assert "runtime_metrics" not in receipt  # Shared stage counters are not per-lane evidence.
    assert not backend._native_runner.jobs
    assert backend._native_runner.status()["per_job_runtime_metrics"] == "unavailable_shared_stage_counters"


def test_real_native_warmup_binds_batch_contract_and_cancel_releases_only_one_lane(ring, monkeypatch):
    backend, control = ring
    # warmup uses the real prepare path; a tiny numeric renderer is intentional.
    original = backend.vp._encode_prompt
    backend.vp._encode_prompt = lambda *a, **kw: [1, 2]
    try: proof = backend.warmup(timeout_s=20)
    finally: backend.vp._encode_prompt = original
    assert proof["proof_verified"] and proof["batch_contract_sha256"]
    receive = backend.vp.recv_msg
    def checked_receive(channel):
        if backend._channels is not None and channel is backend._channels[1]:
            assert backend.ready() == (True, "verified_ring_in_use")
        return receive(channel)
    monkeypatch.setattr(backend.vp, "recv_msg", checked_receive)
    cancelled, survivor = job([3, 4, 5], 4), job([9, 10, 11], 4, "b")
    jobs = [cancelled, survivor]
    backend.advance_batch(jobs, emit={j.id: j.commit for j in jobs},
                          cancel_check={j.id: j.check_stop for j in jobs}, max_step_tokens=1)
    cancelled.cancelled.set()
    outcomes = backend.advance_batch(jobs, emit={j.id: j.commit for j in jobs},
                                    cancel_check={j.id: j.check_stop for j in jobs}, max_step_tokens=1)
    assert isinstance(outcomes[cancelled.id], JobCancelled)
    backend.release_batch_job(cancelled.id)
    result = backend.execute(survivor, survivor.commit, survivor.check_stop)
    assert result["tokens"] == reference(control, survivor.payload["prompt_ids"], survivor.max_new)
    assert backend.ready()[0]


@pytest.mark.parametrize("ring", ["durable"], indirect=True)
def test_strict_socket_restores_persisted_native_frontiers_after_ram_eviction(ring):
    backend, control = ring
    first = job(list(range(1, 14)), 3)
    initial = backend.execute(first, first.commit, first.check_stop)
    assert backend.verify_result_proof(first, initial)
    prompt = first.payload["prompt_ids"] + first.tokens + [19, 20]
    followup = job(prompt, 4)
    result = backend.execute(followup, followup.commit, followup.check_stop)
    assert result["tokens"] == reference(control, prompt, 4)
    assert result["proof"]["scope"] == "fresh_suffix_with_signed_prefix_restore"
    assert result["optimizations"]["native_lanes"]["restore_tiers"] == ["disk", "disk"]
    assert backend.verify_result_proof(followup, result)


@pytest.mark.parametrize("ring", ["durable"], indirect=True)
@pytest.mark.parametrize("failure", ["capacity", "enospc"])
def test_native_optional_disk_capacity_failure_releases_lane_and_continues_cold(ring, monkeypatch, failure):
    import errno
    from pathlib import Path
    import v4_durable_state as durable
    from v4_native_lanes import NativeLanes
    backend, control = ring
    original_usage = durable.shutil.disk_usage
    original_put = durable.DurableStateStore.put
    def usage(path):
        value = original_usage(path)
        return value._replace(free=0) if Path(path).name == "private-0-2" else value
    def put(store, *a, **kw):
        if store.directory.name == "private-0-2":
            raise OSError(errno.ENOSPC, "test disk filled after admission")
        return original_put(store, *a, **kw)
    if failure == "capacity": monkeypatch.setattr(durable.shutil, "disk_usage", usage)
    else: monkeypatch.setattr(durable.DurableStateStore, "put", put)
    close, released = NativeLanes.close, []
    def checked_close(runner, key, **kwargs):
        entry = close(runner, key, **kwargs)
        assert key not in runner.active and runner.current != key
        released.append((runner.stage.lo, entry, runner.last_cache_write_error))
        return entry
    monkeypatch.setattr(NativeLanes, "close", checked_close)
    call, acknowledgements = backend._native_runner._call, []
    def checked_call(state, action, params, **kwargs):
        rows = call(state, action, params, **kwargs)  # Every assigned stage signature is verified here.
        if action == "close": acknowledgements.append(rows)
        return rows
    monkeypatch.setattr(backend._native_runner, "_call", checked_call)
    first = job(range(1, 10), 3)
    result = backend.execute(first, first.commit, first.check_stop)
    assert first.tokens == reference(control, first.payload["prompt_ids"], first.max_new)
    assert backend.verify_result_proof(first, result)
    assert backend.ready()[0] and not backend._native_runner.jobs and not backend._native_runner.cache
    reason = "durable_capacity" if failure == "capacity" else "disk_full"
    assert backend._native_runner.status()["last_cache_write_failures"] == [{"stage": 0, "reason": reason}]
    assert acknowledgements[-1][0]["result"] == {"released": True, "cache": None, "cache_write_error": reason}
    assert any(lo == 0 and entry is None and error == reason for lo, entry, error in released)
    prompt = first.payload["prompt_ids"] + first.tokens + [20, 21]
    followup = job(prompt, 3)
    resumed = backend.execute(followup, followup.commit, followup.check_stop)
    assert followup.tokens == reference(control, prompt, followup.max_new)
    assert resumed["proof"]["scope"] == "complete_final_attempt"
    assert resumed["optimizations"]["native_lanes"]["restored_prefix_tokens"] == 0
    assert backend.verify_result_proof(followup, resumed) and backend.ready()[0]


def test_native_receipt_refuses_source_change_after_actual_forward(ring, monkeypatch):
    import v4_observability as observations
    backend, _ = ring
    request = job([3], 2)
    backend.advance_batch([request], emit={request.id: request.commit},
        cancel_check={request.id: request.check_stop}, max_step_tokens=1)
    assert len(request.tokens) == 1
    runner = backend._native_runner
    source = observations.source_inventory
    changed = source()
    changed["engines/deepseek_v4/v4_stage.py"] = "f" * 64
    monkeypatch.setattr(observations, "source_inventory", lambda: dict(changed))
    try:
        with runner._io(), pytest.raises(ValueError, match="source files changed"):
            runner._call(runner.jobs[request.id], "receipt", {})
        assert not runner.jobs[request.id]["finished"]
    finally:
        monkeypatch.setattr(observations, "source_inventory", source)
        backend.release_batch_job(request.id)
    assert not runner.jobs and not runner.cache


def test_real_native_new_owner_reset_reuses_static_forward_links_but_refuses_old_prefix(ring):
    backend, control = ring
    first = job(list(range(1, 10)), 3)
    backend.execute(first, first.commit, first.check_stop)
    old_epoch = backend._native_runner.epoch
    assert backend._native_runner.cache
    backend.abort()  # The head/tail and their original forward socket keep running.
    next_prompt = first.payload["prompt_ids"] + first.tokens + [12, 13]
    request = job(next_prompt, 3)
    result = backend.execute(request, request.commit, request.check_stop)
    assert backend._native_runner.epoch != old_epoch
    assert result["proof"]["scope"] == "complete_final_attempt"
    assert result["optimizations"]["native_lanes"]["restored_prefix_tokens"] == 0
    assert request.tokens == reference(control, next_prompt, 3)
    assert backend.verify_result_proof(request, result)


def test_sqlite_recovered_continuous_job_replays_real_native_prefix_without_duplicate_publication(ring, tmp_path):
    from shard.job_journal import JobJournal
    from shard.ring_pool import RingPool
    from shard.ring_router import RingRouter, MultiRingQueue
    from shard.service_queue import TenantLimits
    from test_ring_pool import add_ready
    old, control = ring
    prompt = [3, 4]
    expected = reference(control, prompt, 10)
    def pooled(backend):
        pool = RingPool()
        encode = backend.vp._encode_prompt
        backend.vp._encode_prompt = lambda *a, **kw: [1, 2]
        try:
            assigned = add_ready(pool, backend.swarm_id, cohort=backend.pipeline_plan["cohort_id"], backend=backend)
        finally:
            backend.vp._encode_prompt = encode
        service = MultiRingQueue(RingRouter(pool, default_model=backend.model_id), {"a": TenantLimits()},
            batching={"enabled": True, "max_jobs": 2, "max_step_tokens": 1},
            journal=JobJournal(tmp_path / "journal-private" / "jobs.sqlite"))
        return pool, assigned, service
    first_pool, first_ring, first_queue = pooled(old)
    request, _ = first_queue.submit_request("a", {"model": old.model_id,
        "messages": [{"role": "user", "content": "3 4"}], "max_tokens": 10,
        "shard_mode": "greedy", "timeout_s": 60}, idempotency_key="native-resume")
    # Drive actual scheduler admission and native steps until six tokens have
    # reached the real SQLite publication frontier, then preserve that job.
    assert first_queue._next_for_ring(first_ring) is request
    while len(request.tokens) < 6:
        assert first_pool.execute_batch_step([request.ring_binding], [request], max_step_tokens=1)[request.id] is None
    assert request.tokens == expected[:6]
    old.release_batch_job(request.id)
    first_queue.shutdown(preserve_pending=True)
    replacement = V4RingBackend(old.directory, old.head, old.tail, old.assignments,
        mode="greedy", vp=old.vp, tokenizer=old.tokenizer, timeout=8, max_context=64,
        swarm_id=old.swarm_id, pipeline_plan=old.pipeline_plan, coordinator_key=old.coordinator_key,
        calibrated_runtime_configs=old._runtime_configs, lease_fences=old._lease_fences,
        native_lanes={"enabled": True, "numerical_contract": CONTRACT})
    second_queue = None
    try:
        _, _, second_queue = pooled(replacement)
        recovered = second_queue.durable_lookup("a", "native-resume")
        assert recovered.recovered and recovered.tokens == expected[:6]
        second_queue.start()
        end = time.monotonic() + 20
        with recovered.changed:
            while not recovered.terminal and time.monotonic() < end:
                recovered.changed.wait(.05)
        assert recovered.state == "completed", recovered.snapshot()
        assert recovered.tokens == expected
        assert recovered.result["recovery"]["replayed_committed_tokens"] == 6
        assert recovered.result["proof"]["scope"] == "complete_final_attempt"
        stored = second_queue.journal.get(second_queue._journal_owner, tenant="a", job_id=recovered.id)
        assert stored["tokens"] == expected
    finally:
        if replacement._channels: VP.send_msg(replacement._channels[0], {"op": "stop"})
        if second_queue is not None: second_queue.shutdown(drain=True)
        else: replacement.close()


@pytest.mark.parametrize("ring,cache_full", [(None, False), ("durable", True)], indirect=["ring"])
def test_http_continuous_queue_runs_real_native_lanes_and_new_chat_incremental_prefix(ring, monkeypatch, cache_full):
    from shard.http_gateway import AuthRegistry, Gateway
    from shard.ring_pool import RingPool
    from shard.service_queue import TenantLimits
    from test_ring_pool import add_ready
    backend, control = ring
    pool = RingPool()
    encoder = backend.vp._encode_prompt
    backend.vp._encode_prompt = lambda *a, **kw: [1, 2]
    try:
        add_ready(pool, backend.swarm_id, cohort=backend.pipeline_plan["cohort_id"], backend=backend)
    finally:
        backend.vp._encode_prompt = encoder
    if cache_full:
        from pathlib import Path
        import v4_durable_state as durable
        usage = durable.shutil.disk_usage
        def full(path):
            value = usage(path)
            return value._replace(free=0) if Path(path).name == "private-0-2" else value
        monkeypatch.setattr(durable.shutil, "disk_usage", full)
    auth = AuthRegistry({"native-api-key-a-12345": "a", "native-api-key-b-12345": "b"},
                        {"a": TenantLimits(), "b": TenantLimits()})
    gateway = Gateway(auth=auth, ring_pool=pool, default_model=backend.model_id,
                      batching={"enabled": True, "max_jobs": 2, "max_step_tokens": 1})
    server = gateway.server()
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    entered, queued = threading.Event(), threading.Event()
    original, observed = backend.advance_batch, []
    def advance(jobs, **kwargs):
        observed.append([request.id for request in jobs])
        if not entered.is_set():
            entered.set()
            assert queued.wait(5)
        return original(jobs, **kwargs)
    monkeypatch.setattr(backend, "advance_batch", advance)
    submit = gateway.submit
    def admitted(tenant, *a, **kw):
        result = submit(tenant, *a, **kw)
        if tenant == "b": queued.set()
        return result
    monkeypatch.setattr(gateway, "submit", admitted)
    def chat(prompt, key="a", maximum=4):
        connection = http.client.HTTPConnection(*server.server_address, timeout=15)
        body = {"model": backend.model_id, "messages": [{"role": "user", "content": " ".join(map(str, prompt))}],
                "max_tokens": maximum, "shard_mode": "greedy"}
        connection.request("POST", "/v1/chat/completions", json.dumps(body),
                           {"Content-Type": "application/json", "Authorization": "Bearer native-api-key-" + key + "-12345"})
        response = connection.getresponse(); raw = response.read(); connection.close()
        assert response.status == 200, raw
        return json.loads(raw)
    output, failures = {}, []
    first_prompt, other_prompt = list(range(1, 14)), [31, 32, 33]
    def first():
        try: output["first"] = chat(first_prompt)
        except Exception as error: failures.append(error)
    worker = threading.Thread(target=first); worker.start()
    try:
        assert entered.wait(5)
        other = chat(other_prompt, "b", 2)
        worker.join(15); assert not worker.is_alive() and not failures, failures
        first = output["first"]
        tokens = list(map(int, first["choices"][0]["message"]["content"].split()))
        assert tokens == reference(control, first_prompt, 4)
        assert list(map(int, other["choices"][0]["message"]["content"].split())) == reference(control, other_prompt, 2)
        assert any(len(step) == 2 for step in observed)
        assert first["shard"]["numerical_contract"] == CONTRACT
        longer = first_prompt + tokens + [20, 21]
        continued = chat(longer)
        assert continued["shard"]["proof_scope"] == ("complete_final_attempt" if cache_full
                                                      else "fresh_suffix_with_signed_prefix_restore")
        assert list(map(int, continued["choices"][0]["message"]["content"].split())) == reference(control, longer, 4)
        assert gateway.jobs.metrics()["mode"] == "qualified_token_batches"
        if cache_full:
            assert backend._native_runner.status()["last_cache_write_failures"] == [
                {"stage": 0, "reason": "durable_capacity"}]
    finally:
        queued.set(); worker.join(15)
        server.shutdown(); server.server_close(); thread.join(3)
        if backend._channels: VP.send_msg(backend._channels[0], {"op": "stop"})
        gateway.shutdown(drain=True)

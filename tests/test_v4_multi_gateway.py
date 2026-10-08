"""Actual multi-ring HTTP, signed V4 adapter retries and manifest admission."""
from datetime import datetime, timezone
import copy
import hashlib
import json
import threading
import time

import pytest

import v4_gateway as GW
from shard.ring_pool import RingPool, RingState, RingUnavailable
from shard.leases import LeaseLedger, LeaseRequest, LeaseResources
from shard.offers import ModelCohort
from shard.deployment import config_digest
from shard.resources import CalibrationProvenance, GpuRequirements, HostRequirements, PlacementRequirements
from shard.service_queue import TenantLimits
from test_ring_pool import add_ready, Guard, COHORT, OTHER
from test_v4_gateway import Pipe, Tokenizer, backend, request, chat, KEY_A, KEY_B


class ReleasePipe(Pipe):
    def __init__(self, offset=0):
        super().__init__()
        self.hold = False
        self.release = threading.Event()
        self.offset = offset
    def coordinate(self, pipe, ret, prompt_ids, maximum, *, cancel_check, expected_by_signer,
                   strict_job_binding=False, **kw):
        if self.hold:
            self.started.set()
            while not self.release.wait(0.005):
                cancel_check()
        callback = kw["on_token"]
        kw["on_token"] = lambda token: callback(token + self.offset)
        result = super().coordinate(pipe, ret, prompt_ids, maximum, cancel_check=cancel_check,
            expected_by_signer=expected_by_signer, strict_job_binding=strict_job_binding, **kw)
        result["tokens"] = [token + self.offset for token in result["tokens"]]
        return result
    coordinate_dspark = coordinate
    coordinate_dspark_pipelined = coordinate


@pytest.fixture
def multi():
    pool = RingPool()
    pipes = [ReleasePipe(), ReleasePipe(offset=10)]
    for index, pipe in enumerate(pipes):
        add_ready(pool, str(index), backend=backend(pipe), region="east")
        pipe.started.clear()
    pool.set_alias("default", Pipe.V4_MODEL_ID, COHORT)
    auth = GW.AuthRegistry({KEY_A: "a", KEY_B: "b"},
                          {"a": TenantLimits(4, 100, 10000), "b": TenantLimits(4, 100, 10000)})
    gateway = GW.Gateway(auth=auth, ring_pool=pool, default_model="default")
    server = gateway.server()
    serving = threading.Thread(target=server.serve_forever, daemon=True); serving.start()
    try:
        yield server.server_address, gateway, pool, pipes
    finally:
        server.shutdown(); server.server_close(); gateway.shutdown(); serving.join(1)


def multi_chat(**options):
    return chat(model="default", **options)


def test_actual_http_runs_two_independent_signed_rings_concurrently(multi):
    address, gateway, pool, pipes = multi
    results = []
    for pipe in pipes:
        pipe.hold = True
    threads = [threading.Thread(target=lambda key=key: results.append(request(address, "POST",
        "/v1/chat/completions", multi_chat(), key=key))) for key in (KEY_A, KEY_B)]
    for thread in threads:
        thread.start()
    assert all(pipe.started.wait(1) for pipe in pipes)
    status, data = request(address, "GET", "/metrics")
    assert status == 200 and json.loads(data)["service"]["running"] == 2
    assert json.loads(data)["service"]["mode"] == "parallel_serial_rings"
    for pipe in pipes:
        pipe.release.set()
    for thread in threads:
        thread.join(3); assert not thread.is_alive()
    assert len(results) == 2 and all(status == 200 for status, data in results)
    responses = [json.loads(data) for status, data in results]
    assert {r["choices"][0]["message"]["content"] for r in responses} == {"ABCD", "KLMN"}
    assert {r["shard"]["route"]["ring_id"] for r in responses} == {"0", "1"}
    assert all(r["shard"]["proof_verified"] for r in responses)
    assert request(address, "GET", "/ready")[0] == 200


def test_alias_switch_only_ready_new_version_keeps_old_http_job_and_idempotency(multi):
    address, gateway, pool, pipes = multi
    for pipe in pipes:
        pipe.hold = True
    results = []
    old = threading.Thread(target=lambda: results.append(request(address, "POST", "/v1/chat/completions",
        multi_chat(), headers={"Idempotency-Key": "version-cutover"})))
    old.start()
    assert pipes[0].started.wait(1)
    with pytest.raises(RingUnavailable):
        pool.set_alias("default", Pipe.V4_MODEL_ID, OTHER)
    new_pipe = ReleasePipe(offset=20)
    add_ready(pool, "new", cohort=OTHER, backend=backend(new_pipe), region="east")
    pool.set_alias("default", Pipe.V4_MODEL_ID, OTHER)
    following = json.loads(request(address, "POST", "/v1/chat/completions", multi_chat(), key=KEY_B)[1])
    assert following["choices"][0]["message"]["content"] == "UVWX"
    assert following["shard"]["route"]["cohort_id"] == OTHER
    pipes[0].release.set(); old.join(3)
    original = json.loads(results[0][1])
    assert original["shard"]["route"]["cohort_id"] == COHORT
    status, data = request(address, "POST", "/v1/chat/completions", multi_chat(stream=True),
                          headers={"Idempotency-Key": "version-cutover", "Last-Event-ID": original["id"] + ":4:4"})
    assert status == 200 and "ABCD" not in data.decode()
    assert new_pipe.calls == 2  # its warmup plus only the new request
    assert gateway.jobs.metrics()["counts"]["deduplicated"] == 1


def test_real_adapter_recovery_never_merges_attempt_signatures_or_moves_ring(multi):
    address, gateway, pool, pipes = multi
    # Force one eligible ring, then fail after two committed callbacks.
    pool.drain("1")
    pipes[0].fail_first = True; pipes[0].calls = 0; pipes[0].recorded.clear()
    response = json.loads(request(address, "POST", "/v1/chat/completions", multi_chat())[1])
    assert response["choices"][0]["message"]["content"] == "ABCD"
    assert response["shard"]["recovery"]["attempts"] == 2
    status, data = request(address, "GET", "/v1/jobs/" + response["id"] + "/receipts")
    proof = json.loads(data)
    assert status == 200 and len(proof["receipts"]) == 2
    assert {receipt["nonce"] for receipt in proof["receipts"]} == {proof["nonce"]}
    assert all(receipt["job_id"] == proof["attempt_id"] for receipt in proof["receipts"])
    assert pipes[0].recorded[0][3] != pipes[0].recorded[1][3]
    assert response["shard"]["route"]["ring_id"] == "0"


def test_dead_connection_not_routed_and_unready_cohort_returns_http_503(multi):
    address, gateway, pool, pipes = multi
    pool.rings()[0].backend._channels[1].peer.close()
    response = json.loads(request(address, "POST", "/v1/chat/completions", multi_chat())[1])
    assert response["shard"]["route"]["ring_id"] == "1"
    assert pool.rings()[0].state == RingState.WARMING
    status, data = request(address, "POST", "/v1/chat/completions", multi_chat(shard_cohort=OTHER))
    assert status == 503 and json.loads(data)["error"]["code"] == "no_ready_ring"


def manifest(tmp_path):
    directory = tmp_path / "model"; directory.mkdir()
    config = b'{"n_layers":2}'
    (directory / "config.json").write_bytes(config)
    pipe = ReleasePipe()
    cohort = ModelCohort(Pipe.V4_MODEL_ID, "e" * 64, "checkpoint", hashlib.sha256(config).hexdigest(),
                        "bf16", "v4-test/1", "shard-wire/1", "bitwise-test", 2)
    stages, leases = [], []
    stamp = datetime.now(timezone.utc).isoformat()
    for index, signer in enumerate(pipe.assignments):
        gpu, node, host = "GPU-" + str(index), "node-" + str(index), "host-" + str(index)
        runtime = {"args": {"n_layers": 2, "max_seq_len": 1024}, "lo": index, "hi": index + 1,
                   "head": index == 0, "tail": index == 1, "dspark": False,
                   "expert_placement": "gpu", "environment": {}}
        req = PlacementRequirements(Pipe.V4_MODEL_ID, index, index + 1,
            GpuRequirements(resident_weights_bytes=20), HostRequirements(reserve_bytes=5),
            CalibrationProvenance("checkpoint", config_digest(runtime), stamp, node, "measured", "CPU fake runtime fixture"))
        stages.append({"node_id": node, "host_id": host, "gpu_uuid": gpu, "signer_pubkey": signer,
                       "lo": index, "hi": index + 1, "port": 29000 + index, "requirements": req.to_dict(),
                       "runtime_config": runtime, "env": {}, "capacity_measured_at": stamp,
                       "capacity": {"available_vram_bytes": 100, "available_ram_bytes": 100,
                                    "pinnable_ram_bytes": 100, "available_disk_bytes": 100}})
        ledger_path = tmp_path / (host + ".sqlite")
        ledger = LeaseLedger(ledger_path, node_id=node, authorize=lambda principal, action, binding: principal)
        ledger.register_capacity(host, available_ram_bytes=100, pinnable_ram_bytes=50, gpu_capacity_bytes={gpu: 100})
        req = LeaseRequest("manifest-ring", cohort.cohort_id, node, gpu, host, LeaseResources(50, 50, 10), 300)
        lease = ledger.prepare(req, principal="local-controller", idempotency_key="initial")
        ledger.commit(lease.lease_id, lease.fencing_token, principal="local-controller")
        leases.append({"ledger": str(ledger_path), "node_id": node, "principal": "local-controller",
                       "lease_id": lease.lease_id, "fencing_token": lease.fencing_token})
    bundle = {"schema": "shard-deployment/1", "model_id": Pipe.V4_MODEL_ID, "checkpoint_id": "checkpoint",
              "layer_count": 2, "stages": stages, "model_cohort": cohort.to_dict(),
              "registration_policy": "open", "token_privacy": "plain_ids"}
    body = {"schema": "shard-ring-pool/1", "default_model": "default",
            "aliases": {"default": {"model_id": cohort.model_id, "cohort_id": cohort.cohort_id}},
            "rings": [{"ring_id": "manifest-ring", "dir": "model", "head": "head:1", "tail": "tail:2",
                       "deployment": bundle, "leases": leases, "region": "east"}]}
    path = tmp_path / "rings.json"; path.write_text(json.dumps(body))
    factory = lambda directory, head, tail, assignments, **options: GW.V4RingBackend(
        directory, head, tail, assignments, vp=pipe, tokenizer=Tokenizer(), **options)
    return path, body, factory, pipe


def test_manifest_loader_executes_signed_warmup_uses_actual_local_ledger_and_ready_alias(tmp_path):
    path, body, factory, pipe = manifest(tmp_path)
    pool, default = GW.load_ring_pool(path, backend_factory=factory, env={})
    assert default == "default" and pool.ready()[0] and pipe.calls == 1
    assert pool.rings()[0].guards[0].ledger.get(body["rings"][0]["leases"][0]["lease_id"],
        principal="local-controller").active_work == 0
    pool.shutdown()


@pytest.mark.parametrize("mutation,match", [
    (lambda b: b["rings"][0]["deployment"]["model_cohort"].update(checkpoint_id="wrong"), "cohort"),
    (lambda b: b["rings"][0].update(cohort_id=OTHER), "cohort"),
    (lambda b: b["rings"][0].update(max_context=2048), "context"),
    (lambda b: b["rings"][0]["leases"][0].update(fencing_token=999), "guard"),
    (lambda b: b["rings"][0]["deployment"]["stages"][0].update(gpu_uuid="different"), None),
])
def test_manifest_version_context_and_guard_claims_fail_before_readiness(tmp_path, mutation, match):
    path, body, factory, pipe = manifest(tmp_path)
    mutation(body); path.write_text(json.dumps(body))
    with pytest.raises((ValueError, KeyError), match=match):
        GW.load_ring_pool(path, backend_factory=factory, env={})
    assert pipe.calls == 0


def test_manifest_json_ready_claim_does_not_replace_signed_warmup(tmp_path):
    path, body, factory, pipe = manifest(tmp_path)
    body["rings"][0]["ready"] = True; path.write_text(json.dumps(body))
    pipe.bad_receipt = True
    with pytest.raises(GW.ReceiptValidationError):
        GW.load_ring_pool(path, backend_factory=factory, env={})
    assert pipe.calls == 1

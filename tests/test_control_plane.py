from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from urllib.request import Request, urlopen

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from shard.control_plane import (ControlError, LeaseRPCClient, NodeLeaseAgent, ReplayCache,
                                 RemoteLeaseGuard, _identity, _signed, _verify, control_server)
from shard.leases import LeaseLedger, LeaseRequest, LeaseResources
from shard.offers import OfferRegistry


@contextmanager
def serving(dispatch):
    server = control_server(("127.0.0.1", 0), dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def node(tmp_path, clock=None):
    key = Ed25519PrivateKey.generate()
    node_id = _identity(key)[0] + "/GPU-0"
    options = {} if clock is None else {"clock": clock}
    ledger = LeaseLedger(tmp_path / "leases.sqlite", node_id=node_id,
                         authorize=lambda p, a, r: p, **options)
    ledger.register_capacity("host", available_ram_bytes=1000, pinnable_ram_bytes=500,
                             gpu_capacity_bytes={"GPU-0": 1000})
    return ledger, NodeLeaseAgent(ledger, key, cohorts={"1" * 64}, **options)


def request(node_id, *, cohort="1" * 64, ring="ring", ttl=120):
    return LeaseRequest(ring, cohort, node_id, "GPU-0", "host", LeaseResources(100, 100, 50), ttl)


def test_actual_http_signed_lease_lifecycle_and_owner_binding(tmp_path):
    ledger, agent = node(tmp_path)
    def dispatch(path, body):
        assert path == "/lease"
        return agent.dispatch(body)
    with serving(dispatch) as address:
        owner = LeaseRPCClient(address, ledger.node_id, Ed25519PrivateKey.generate())
        stranger = LeaseRPCClient(address, ledger.node_id, Ed25519PrivateKey.generate())
        lease = owner.prepare(request(ledger.node_id), idempotency_key="allocation-1")
        with pytest.raises(ControlError, match="controller"):
            stranger.operation("commit", lease)
        lease = owner.operation("commit", lease)
        guard = RemoteLeaseGuard(owner, lease["lease_id"], lease["fencing_token"], ring_id="ring", cohort_id="1" * 64)
        assert guard.gpu_uuid == "GPU-0"
        work = guard.begin_work()
        draining = owner.operation("release", lease)
        assert draining["state"] == "draining" and draining["active_work"] == 1
        with pytest.raises(ControlError):
            guard.assert_live()
        guard.end_work(work)
        # A new controller can acquire the GPU only after real work has ended.
        fresh = stranger.prepare(request(ledger.node_id, ring="next"), idempotency_key="new-allocation")
        assert fresh["fencing_token"] > lease["fencing_token"]


def test_cohort_mismatch_never_reserves_node(tmp_path):
    ledger, agent = node(tmp_path)
    owner = Ed25519PrivateKey.generate()
    bad = _signed("request", {"action": "prepare", "body": {
        "request": request(ledger.node_id, cohort="2" * 64).to_dict(), "idempotency_key": "x"}}, owner)
    answer = agent.dispatch(bad)
    assert answer["payload"]["ok"] is False
    assert "cohort" in answer["payload"]["error"]


def test_rpc_nonce_replay_is_rejected_across_agent_restart(tmp_path):
    ledger, agent = node(tmp_path)
    path = tmp_path / "replays.sqlite"
    agent.replay = ReplayCache(path)
    envelope = _signed("request", {"action": "prepare", "body": {
        "request": request(ledger.node_id).to_dict(), "idempotency_key": "x"}}, Ed25519PrivateKey.generate())
    assert agent.dispatch(envelope)["payload"]["ok"]
    agent.replay.close()
    agent.replay = ReplayCache(path)
    with pytest.raises(ControlError, match="replayed"):
        agent.dispatch(envelope)


def test_forged_principal_or_node_reply_is_not_accepted(tmp_path):
    ledger, agent = node(tmp_path)
    envelope = _signed("request", {"action": "prepare", "body": {
        "request": request(ledger.node_id).to_dict(), "idempotency_key": "x"}}, Ed25519PrivateKey.generate())
    envelope["peer_id"] = agent.peer_id
    with pytest.raises(ControlError, match="signature"):
        agent.dispatch(envelope)
    forged = _signed("response", {"ok": True}, Ed25519PrivateKey.generate())
    with pytest.raises(ControlError, match="signature"):
        _verify(forged, "response", expected_peer=agent.peer_id)


def test_real_http_registry_accepts_unrelated_identities(tmp_path):
    from test_node_offers import offer, cohort
    from shard.offers import model_cohort_id
    registry = OfferRegistry(tmp_path / "offers.sqlite", clock=lambda: 1000)
    def dispatch(path, body):
        if path == "/offers":
            return registry.announce(body)
        return {"nodes": registry.snapshot(body["cohort_id"])}
    with serving(dispatch) as address:
        for _ in range(2):
            data = offer()
            with urlopen(Request(address + "/offers", json.dumps(data).encode(), method="POST"), timeout=2) as response:
                assert json.load(response)["node_id"] == data["node_id"]
        with urlopen(Request(address + "/snapshot", json.dumps({"cohort_id": model_cohort_id(cohort())}).encode(), method="POST"), timeout=2) as response:
            assert len(json.load(response)["nodes"]) == 2


def test_concurrent_authenticated_controllers_cannot_double_book_gpu(tmp_path):
    ledger, agent = node(tmp_path)
    with serving(lambda p, b: agent.dispatch(b)) as address:
        def reserve(i):
            client = LeaseRPCClient(address, ledger.node_id, Ed25519PrivateKey.generate())
            try:
                client.prepare(request(ledger.node_id, ring=f"ring-{i}"), idempotency_key="x")
                return True
            except ControlError:
                return False
        with ThreadPoolExecutor(max_workers=8) as pool:
            assert sum(pool.map(reserve, range(8))) == 1


def test_public_control_binding_and_remote_plain_http_are_rejected():
    with pytest.raises(ControlError, match="TLS"):
        control_server(("0.0.0.0", 0), lambda p, b: {})
    with pytest.raises(ControlError, match="HTTPS"):
        LeaseRPCClient("http://198.51.100.1:1234", "peer/GPU-0", Ed25519PrivateKey.generate())


def test_expired_signature_cannot_renew_even_if_request_is_well_formed():
    key = Ed25519PrivateKey.generate()
    body = _signed("request", {"action": "renew", "body": {}}, key, clock=lambda: 1000)
    with pytest.raises(ControlError, match="expired"):
        _verify(body, "request", clock=lambda: 1031)

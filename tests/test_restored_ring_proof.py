"""Actual V4 gateway -> leased RingPool seam, with signed CPU producer receipts."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from shard.execution_proof import verify_prefix_restore, RESTORE_STATEMENT
from shard.receipt import ReceiptSigner, ReceiptError, pub_b64
from shard.ring_pool import RingPool, RingState, RingUnavailable
from test_ring_pool import add_ready
from test_v4_gateway import Pipe, backend, job


class CacheOwner:
    def __init__(self):
        self.requests = 0
        self.entries = {}

    def request(self, identity, prefix, *, max_new):
        self.requests += 1
        return SimpleNamespace(identity=identity, prefix=prefix, restore_binding=None, key="cpu-test-entry",
            record={"hit": self.requests > 1, "enabled": True})


class RestoringPipe(Pipe):
    tamper = None

    def coordinate(self, *args, cancel_check, expected_by_signer, strict_job_binding=False, **kwargs):
        result = super().coordinate(*args, cancel_check=cancel_check,
            expected_by_signer=expected_by_signer, strict_job_binding=strict_job_binding, **kwargs)
        request = kwargs.get("conversation")
        if request is None or not request.record["hit"]:
            return result
        binding = {"transaction_id": "current-commit-barrier", "prefix": request.prefix,
            "identity_sha256": hashlib.sha256(b"current-cache-identity").hexdigest(),
            "stages": [{"stage": i, "pubkey": pub_b64(key), "entry_id": f"entry-{i}",
                         "state_digest": str(i + 3) * 64} for i, key in enumerate(self.keys)]}
        request.restore_binding = deepcopy(binding)
        receipts = []
        for i, key in enumerate(self.keys):
            signer = ReceiptSigner(key, kwargs["swarm_id"], kwargs["job_id"], i, i + 1,
                                   nonce=kwargs["nonce"])
            signer.observe(f"suffix-{i}".encode(), f"suffix-{i + 1}".encode())
            note = {"schema": "v4-conversation-restore/1", "transaction_id": binding["transaction_id"],
                "prefix": binding["prefix"], "identity_sha256": binding["identity_sha256"],
                "entry_id": f"entry-{i}", "state_digest": str(i + 3) * 64, "scope": RESTORE_STATEMENT}
            if self.tamper == "entry" and i == 0:
                note["entry_id"] = "another-entry"
            if self.tamper == "state" and i == 0:
                note["state_digest"] = "f" * 64
            receipts.append(signer.finalize(conversation_restore=note))
        result["receipts"] = receipts
        return result

    coordinate_dspark = coordinate
    coordinate_dspark_pipelined = coordinate


def setup():
    producer = RestoringPipe()
    adapter = backend(producer)
    pool = RingPool()
    ring = add_ready(pool, backend=adapter)
    adapter._channels[0].grant = {"boot_id": "cpu-test-boot", "fence": 1}
    adapter._conversation_owner = CacheOwner()
    adapter._conversation_base = {"config_sha256": "a" * 64}
    return producer, adapter, pool, ring


def execute(pool, ring):
    request = job()
    binding = pool.acquire(ring.model_id, ring.cohort_id, estimated_tokens=request.max_new)
    try:
        return request, pool.execute(binding, request, request.commit, request.check_stop)
    finally:
        pool.release(binding)


def test_second_exact_repeat_is_accepted_by_pool_with_its_actual_scope():
    _, adapter, pool, ring = setup()
    try:
        first, miss = execute(pool, ring)
        second, hit = execute(pool, ring)
        assert miss["proof"]["scope"] == "complete_final_attempt"
        assert hit["proof"]["scope"] == "fresh_suffix_with_signed_prefix_restore"
        assert first.tokens == second.tokens
        assert ring.state == RingState.READY
        assert adapter.verify_result_proof(second, hit) is True
        modified = deepcopy(hit)
        modified["proof"]["scope"] = "complete_final_attempt"
        with pytest.raises(Exception, match="differs"):
            adapter.verify_result_proof(second, modified)
        with pytest.raises(Exception, match="differs"):
            adapter.verify_result_proof(job(), hit)
    finally:
        pool.shutdown()


@pytest.mark.parametrize("tamper", ["entry", "state"])
def test_valid_signatures_cannot_substitute_another_restored_state(tamper):
    producer, _, pool, ring = setup()
    try:
        execute(pool, ring)
        producer.tamper = tamper
        with pytest.raises(Exception, match="signed assignment"):
            execute(pool, ring)
    finally:
        pool.shutdown()


def test_restore_scope_without_a_backend_evidence_verifier_fails_closed():
    from test_ring_pool import Backend, make_job
    class UnverifiedRestore(Backend):
        def execute(self, *args):
            result = super().execute(*args)
            result["proof"]["scope"] = "fresh_suffix_with_signed_prefix_restore"
            return result
    pool = RingPool()
    ring = add_ready(pool, backend=UnverifiedRestore())
    binding = pool.acquire(ring.model_id, ring.cohort_id, estimated_tokens=2)
    request = make_job()
    try:
        with pytest.raises(RingUnavailable, match="verifier required"):
            pool.execute(binding, request, request.commit, request.check_stop)
        assert ring.state == RingState.FAILED
    finally:
        pool.release(binding)
        pool.shutdown()


def test_prefix_restore_requires_all_fresh_participants():
    with pytest.raises(ReceiptError):
        verify_prefix_restore([], {"transaction_id": "tx", "prefix": {},
            "identity_sha256": "a" * 64, "stages": []}, {"assigned": (0, 1)})

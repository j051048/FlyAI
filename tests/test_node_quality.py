"""Pinned local audit authority, actual Ed25519 and bounded feedback state."""
from copy import deepcopy

import pytest

from shard.node_quality import NodeQuality, replay_report
from shard.receipt import ReceiptSigner, gen_key, pub_b64


def receipt(key, job):
    signer = ReceiptSigner(key, "ring", job, 0, 1, nonce=job)
    signer.observe(b"original-input", b"declared-output")
    return signer.finalize()


def case(quality, node, job="job"):
    return quality.issue("node", "a" * 64, receipt(node, job), expected_pubkey=pub_b64(node),
        numeric_contract="same-runtime-bit-exact/1", input_sha256="b" * 64,
        execution_identity_sha256="c" * 64)


def report(validator, issued, *, bad=False, evidence="cpu_reference"):
    return replay_report(validator, issued, reference_bytes=b"actual-reference", candidate_bytes=b"bad" if bad else b"actual-reference",
        input_sha256="b" * 64, execution_identity_sha256="c" * 64,
        evidence=evidence, elapsed_ms=12)


def test_gpu_policy_does_not_accept_cpu_numeric_tests_as_gpu_evidence():
    node, validator = gen_key(), gen_key()
    quality = NodeQuality([pub_b64(validator)])
    issued = case(quality, node)
    with pytest.raises(ValueError, match="backend"):
        quality.ingest(report(validator, issued))
    assert len(quality.pending) == 1 and quality.view("node", "a" * 64)["observed_jobs"] == 0


def test_two_distinct_verified_replays_quarantine_only_the_same_cohort():
    now = [100.0]
    node, validator = gen_key(), gen_key()
    quality = NodeQuality([pub_b64(validator)], required_evidence="cpu_reference", clock=lambda: now[0], quarantine_s=10)
    first = report(validator, case(quality, node, "first"), bad=True)
    assert not quality.ingest(first)["quarantined"]
    with pytest.raises(ValueError, match="replayed"):
        quality.ingest(first)
    with pytest.raises(ValueError, match="repeated"):
        case(quality, node, "first")
    second = report(validator, case(quality, node, "second"), bad=True)
    assert quality.ingest(second)["quarantined"]
    assert not quality.permits("node", "a" * 64) and quality.permits("node", "d" * 64)
    now[0] = 111
    assert quality.permits("node", "a" * 64)


def test_untrusted_valid_signature_cannot_update_reputation():
    node, validator, attacker = gen_key(), gen_key(), gen_key()
    quality = NodeQuality([pub_b64(validator)], required_evidence="cpu_reference")
    issued = case(quality, node)
    with pytest.raises(ValueError, match="trusted"):
        quality.ingest(report(attacker, issued, bad=True))
    assert quality.ingest(report(validator, issued))["matched"] == 1


def test_tampered_input_runtime_or_verdict_is_rejected():
    node, validator = gen_key(), gen_key()
    quality = NodeQuality([pub_b64(validator)], required_evidence="cpu_reference")
    issued = case(quality, node)
    result = report(validator, issued)
    result["verdict"] = "mismatch"
    with pytest.raises(Exception):
        quality.ingest(result)
    wrong = replay_report(validator, issued, reference_bytes=b"a", candidate_bytes=b"a",
        input_sha256="d" * 64, execution_identity_sha256="c" * 64)
    with pytest.raises(ValueError, match="another input"):
        quality.ingest(wrong)
    assert quality.ingest(report(validator, issued))["matched"] == 1


def test_expiry_pending_capacity_and_history_are_bounded():
    now = [100.0]
    node, validator = gen_key(), gen_key()
    quality = NodeQuality([pub_b64(validator)], required_evidence="cpu_reference", max_pending=1,
        max_events=2, history_s=20, case_ttl_s=5, clock=lambda: now[0])
    issued = case(quality, node, "expired")
    with pytest.raises(ValueError, match="capacity"):
        case(quality, node, "overflow")
    now[0] = 106
    with pytest.raises(ValueError, match="expired"):
        quality.ingest(report(validator, issued))
    for index in range(4):
        quality.ingest(report(validator, case(quality, node, str(index))))
    assert quality.view("node", "a" * 64)["observed_jobs"] == 2
    now[0] = 127
    assert quality.view("node", "a" * 64)["observed_jobs"] == 0


def test_audit_can_only_target_the_pinned_receipt_signer():
    node, validator, other = gen_key(), gen_key(), gen_key()
    quality = NodeQuality([pub_b64(validator)])
    with pytest.raises(Exception, match="signer"):
        quality.issue("node", "a" * 64, receipt(other, "other"), expected_pubkey=pub_b64(node),
            numeric_contract="bit-exact/1", input_sha256="b" * 64, execution_identity_sha256="c" * 64)

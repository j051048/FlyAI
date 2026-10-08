"""Attack and real-tensor regression cases for local dispute boundaries."""
import copy
import hashlib

import pytest

from phase0.activation_proof import (hash_activation, compute_step_commitment, StageCommitmentTracker,
                                    verify_commitment_chain)
from phase0.fraud_proof import (create_challenge, submit_dispute_snapshot, adjudicate_challenge,
                               FraudProofError, _check_numerical_tolerance)
from phase0.proof_receipt import verify_activation_commitments

ENGINE = hashlib.sha256(b"local-replay-source").hexdigest()


def challenge_for(inp, out, **overrides):
    options = dict(challenge_id="challenge", run_id="run", stage_idx=1, step=7,
                   challenger_id="challenger", defender_id="defender", challenger_deposit=100,
                   defender_deposit=300, declared_in_hash=hash_activation(inp),
                   declared_out_hash=hash_activation(out), defender_arch="sm120", engine_sha256=ENGINE,
                   now=100.0, challenge_window_s=10.0)
    options.update(overrides)
    options.setdefault("declared_commitment", compute_step_commitment(options["stage_idx"], options["step"],
                                                                    options["declared_in_hash"], options["declared_out_hash"]))
    return create_challenge(**options)


def adjudicate(challenge, replay=lambda inp: [2.0], **kw):
    return adjudicate_challenge(challenge, replay, "sm120", ENGINE, now=105.0, **kw)


def test_tensor_hash_uses_logical_view_bytes_and_binds_dtype_shape():
    torch = pytest.importorskip("torch")
    base = torch.arange(20, dtype=torch.bfloat16)
    view = base[5:9]
    expected = view.clone()
    assert hash_activation(view) == hash_activation(expected)
    base[:5] = -10
    base[9:] = 33
    assert hash_activation(view) == hash_activation(expected), "unrelated backing storage must not be hashed"
    assert hash_activation(expected) != hash_activation(expected.reshape(2, 2))
    assert hash_activation(expected) != hash_activation(expected.view(torch.int16))
    assert hash_activation(torch.tensor(7.0, dtype=torch.bfloat16))
    fp8 = torch.arange(4, dtype=torch.float32).to(torch.float8_e4m3fn)
    assert hash_activation(fp8) == hash_activation(fp8.clone())
    assert hash_activation(base[::2]) == hash_activation(base[::2].contiguous())


def test_hash_domains_and_sampling_are_not_interchangeable():
    np = pytest.importorskip("numpy")
    assert hash_activation([1, 2]) != hash_activation(b"[1, 2]")
    assert hash_activation(None) != hash_activation(b"<empty>")
    assert hash_activation(np.arange(4).reshape(2, 2)) != hash_activation(np.arange(4))
    with pytest.raises(ValueError, match="every byte"):
        hash_activation(b"abcdef", stride=2)
    with pytest.raises(ValueError, match="object arrays"):
        hash_activation(np.array([object()], dtype=object))


def test_chain_anchor_stage_order_endpoint_and_detached_history():
    inp, out = hash_activation([1]), hash_activation([2])
    tracker = StageCommitmentTracker(1)
    tracker.record_step(7, inp, out)
    tracker.record_step(8, out, inp)
    records = tracker.history()
    assert verify_commitment_chain(records, expected_stage=1, expected_final=tracker.current_commitment)[0]
    assert not verify_commitment_chain(records[1:])[0], "an unauthenticated prefix root must not be trusted"
    assert verify_commitment_chain(records[1:], initial_commitment=records[0]["commitment"], expected_stage=1)[0]
    records[0]["in_hash"] = out
    assert tracker.history()[0]["in_hash"] == inp
    assert not verify_commitment_chain(tracker.history(), expected_stage=2)[0]
    assert not verify_commitment_chain(tracker.history(), expected_final="00" * 32)[0]
    bad = tracker.history()
    bad[1]["step"] = 10
    bad[1]["commitment"] = compute_step_commitment(1, 10, out, inp, bad[0]["commitment"])
    assert "nonconsecutive" in verify_commitment_chain(bad)[1]
    with pytest.raises(ValueError, match="consecutive"):
        tracker.record_step(10, inp, out)


@pytest.mark.parametrize("value", [None, "", "claimed", 9, "aa" * 31, "AA" * 32])
def test_engine_identity_required_at_creation_and_replay(value):
    with pytest.raises(FraudProofError, match="engine_sha256"):
        challenge_for([1.0], [2.0], engine_sha256=value)
    challenge = challenge_for([1.0], [2.0])
    submit_dispute_snapshot(challenge, [1.0], [2.0], now=102.0)
    with pytest.raises(FraudProofError, match="validator_engine_sha256"):
        adjudicate_challenge(challenge, lambda inp: [2.0], "sm120", value, now=105.0)
    assert challenge.status == "SNAPSHOT_SUBMITTED"


def test_wrong_source_and_declared_commitment_fail_closed():
    with pytest.raises(FraudProofError, match="does not bind"):
        challenge_for([1.0], [2.0], declared_commitment="11" * 32)
    challenge = challenge_for([1.0], [2.0])
    submit_dispute_snapshot(challenge, [1.0], [2.0], now=102.0)
    with pytest.raises(FraudProofError, match="engine source mismatch"):
        adjudicate_challenge(challenge, lambda inp: [2.0], "sm120", "22" * 32, now=105.0)
    assert challenge.status == "SNAPSHOT_SUBMITTED"


def test_deadline_enforced_for_submit_and_timeout():
    challenge = challenge_for([1.0], [2.0])
    for now in (99.0, 111.0):
        with pytest.raises(FraudProofError, match="outside challenge window"):
            submit_dispute_snapshot(challenge, [1.0], [2.0], now=now)
    with pytest.raises(FraudProofError, match="has not expired"):
        adjudicate_challenge(challenge, lambda inp: [2.0], "sm120", ENGINE, now=110.0)
    result = adjudicate_challenge(challenge, lambda inp: [2.0], "sm120", ENGINE, now=110.01)
    assert result["verdict"] == "SLASH_DEFENDER"
    assert result["settlement"] == "local_arbitration"
    assert result["onchain_executed"] is False
    boundary = challenge_for([1.0], [2.0])
    assert submit_dispute_snapshot(boundary, [1.0], [2.0], now=110.0)


def test_forged_output_snapshot_cannot_make_tolerance_honest():
    challenge = challenge_for([1.0], [0.0], numerical_tolerance=0.01)
    assert not submit_dispute_snapshot(challenge, [1.0], [2.0], now=102.0)
    assert "output hash" in challenge.resolution_reason
    assert challenge.status == "RESOLVED_DEFENDER_SLASHED"


def test_snapshot_and_replay_input_copies_block_mutation():
    inp, out = [1.0], [2.0]
    challenge = challenge_for(inp, out)
    assert submit_dispute_snapshot(challenge, inp, out, now=102.0)
    inp[0], out[0] = 99, 99
    def destructive_replay(snapshot):
        value = snapshot[0] * 2
        snapshot[0] = 123
        return [value]
    verdict = adjudicate(challenge, destructive_replay)
    assert verdict["verdict"] == "CHALLENGE_FAILED"
    assert challenge.snapshot_input == [1.0]
    assert challenge.snapshot_output == [2.0]
    other = challenge_for([1.0], [2.0])
    submit_dispute_snapshot(other, [1.0], [2.0], now=102.0)
    other.snapshot_output[0] = 3
    with pytest.raises(FraudProofError, match="snapshot changed"):
        adjudicate(other)


@pytest.mark.parametrize("field,value", [("deadline_at", 999), ("engine_sha256", "11" * 32),
                                         ("declared_out_hash", "22" * 32), ("step", 8)])
def test_challenge_identity_and_deadline_mutation_rejected(field, value):
    challenge = challenge_for([1.0], [2.0])
    setattr(challenge, field, value)
    with pytest.raises(FraudProofError, match="changed after creation"):
        submit_dispute_snapshot(challenge, [1.0], [2.0], now=102.0)


def test_tolerance_policy_fixed_and_shape_finite_checks():
    challenge = challenge_for([1.0], [2.0])
    submit_dispute_snapshot(challenge, [1.0], [2.0], now=102.0)
    with pytest.raises(FraudProofError, match="fixed at challenge creation"):
        adjudicate(challenge, tolerance=1.0)
    tolerant = challenge_for([1.0], [2.0], numerical_tolerance=0.001)
    submit_dispute_snapshot(tolerant, [1.0], [2.0], now=102.0)
    result = adjudicate(tolerant, lambda inp: [2.0001])
    assert "predeclared numerical tolerance" in result["reason"]
    assert "exactly" not in result["reason"]
    assert not _check_numerical_tolerance([2], [[2]], 0.01)
    assert not _check_numerical_tolerance([float("nan")], [2], 1.0)
    assert not _check_numerical_tolerance([], [], 1.0)


def test_receipt_bundles_bind_stage_and_final_root_and_reject_empty_mixed():
    tracker = StageCommitmentTracker(2)
    tracker.record_step(0, hash_activation([1]), hash_activation([2]))
    bundle = {"stage_idx": 2, "chain": tracker.history(), "final_commitment": tracker.current_commitment}
    assert verify_activation_commitments([bundle])[0]
    assert verify_activation_commitments({"2": tracker.history()})[0]
    for bad in ([], {}, "text", [bundle, tracker.history()[0]], [bundle, bundle],
                [{**bundle, "stage_idx": 3}], [{**bundle, "final_commitment": "11" * 32}],
                [{**bundle, "chain": []}], {"02": tracker.history()}):
        assert not verify_activation_commitments(bad)[0]

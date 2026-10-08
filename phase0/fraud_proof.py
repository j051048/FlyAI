"""Local same-architecture dispute replay; no chain transaction or slashing occurs.

The trusted local adjudicator supplies a replay implementation and source identity.
Snapshots bind both committed input and output. This does not prove the validator's
identity, checkpoint, KV history, or the truth of a remotely supplied engine hash.
"""
from dataclasses import dataclass, field
import copy
import hashlib
import json
import math
import time
from typing import Any, Callable, Optional

from .activation_proof import hash_activation, compute_step_commitment, validate_sha256, ZERO_COMMITMENT


class FraudProofError(ValueError):
    pass


@dataclass
class ChallengeState:
    challenge_id: str
    run_id: str
    stage_idx: int
    step: int
    challenger_id: str
    defender_id: str
    challenger_deposit: int
    defender_deposit: int
    declared_commitment: str
    declared_in_hash: str
    declared_out_hash: str
    defender_arch: str
    engine_sha256: str
    created_at: float
    deadline_at: float
    prev_commitment: str = ZERO_COMMITMENT
    numerical_tolerance: Optional[float] = None
    snapshot_input: Optional[Any] = field(default=None, repr=False)
    snapshot_output: Optional[Any] = field(default=None, repr=False)
    submitted_at: Optional[float] = None
    status: str = "OPEN"
    resolution_reason: Optional[str] = None
    # Historical names retained: these are local recommended amounts, not transfers.
    slashed_amount: int = 0
    reward_amount: int = 0
    _binding_hash: str = field(default="", repr=False)


_BOUND_FIELDS = ("challenge_id", "run_id", "stage_idx", "step", "challenger_id", "defender_id",
                 "challenger_deposit", "defender_deposit", "declared_commitment", "declared_in_hash",
                 "declared_out_hash", "defender_arch", "engine_sha256", "created_at", "deadline_at",
                 "prev_commitment", "numerical_tolerance")


def _identity_binding(challenge):
    return hashlib.sha256(json.dumps({name: getattr(challenge, name) for name in _BOUND_FIELDS},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def _check_binding(challenge):
    try:
        intact = challenge._binding_hash and _identity_binding(challenge) == challenge._binding_hash
    except (AttributeError, TypeError, ValueError) as exc:
        raise FraudProofError("malformed challenge identity") from exc
    if not intact:
        raise FraudProofError("challenge identity, commitments, or deadline changed after creation")


def _time(value=None):
    value = time.time() if value is None else value
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise FraudProofError("challenge clock must be a finite timestamp")
    return float(value)


def _clone(value):
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    return copy.deepcopy(value)


def create_challenge(challenge_id: str, run_id: str, stage_idx: int, step: int,
                     challenger_id: str, defender_id: str, challenger_deposit: int,
                     defender_deposit: int, declared_commitment: str, declared_in_hash: str,
                     declared_out_hash: str, defender_arch: str = "sm120", engine_sha256: str = "",
                     *, prev_commitment: str = ZERO_COMMITMENT, challenge_window_s: float = 60.0,
                     now: Optional[float] = None, numerical_tolerance: Optional[float] = None) -> ChallengeState:
    """Create a bound local challenge. A verified engine identity is mandatory."""
    for name, value in (("challenger_deposit", challenger_deposit), ("defender_deposit", defender_deposit)):
        if type(value) is not int or value <= 0:
            raise FraudProofError(f"{name} must be a positive integer amount")
    for name, value in (("challenge_id", challenge_id), ("run_id", run_id), ("challenger_id", challenger_id),
                        ("defender_id", defender_id), ("defender_arch", defender_arch)):
        if not isinstance(value, str) or not value.strip():
            raise FraudProofError(f"{name} must be a nonempty identity")
    try:
        validate_sha256(engine_sha256, "engine_sha256")
        expected = compute_step_commitment(stage_idx, step, declared_in_hash, declared_out_hash, prev_commitment)
        validate_sha256(declared_commitment, "declared_commitment")
    except ValueError as exc:
        raise FraudProofError(str(exc)) from exc
    if expected != declared_commitment:
        raise FraudProofError("declared commitment does not bind stage/step/input/output/previous root")
    if challenge_window_s is None:
        raise FraudProofError("challenge_window_s must be a finite positive duration")
    window = _time(challenge_window_s)
    if window <= 0:
        raise FraudProofError("challenge_window_s must be positive")
    created = _time(now)
    if numerical_tolerance is not None:
        numerical_tolerance = _time(numerical_tolerance)
        if numerical_tolerance < 0:
            raise FraudProofError("numerical_tolerance must be nonnegative")
    deadline = created + window
    if not math.isfinite(deadline):
        raise FraudProofError("challenge deadline must be finite")
    challenge = ChallengeState(challenge_id, run_id, stage_idx, step, challenger_id, defender_id,
                               challenger_deposit, defender_deposit, declared_commitment, declared_in_hash,
                               declared_out_hash, defender_arch, engine_sha256, created, deadline,
                               prev_commitment, numerical_tolerance)
    challenge._binding_hash = _identity_binding(challenge)
    return challenge


def _resolve(challenge, honest, reason):
    challenge.status = "RESOLVED_DEFENDER_HONEST" if honest else "RESOLVED_DEFENDER_SLASHED"
    challenge.resolution_reason = reason
    challenge.slashed_amount = challenge.challenger_deposit if honest else challenge.defender_deposit
    challenge.reward_amount = (challenge.defender_deposit + challenge.challenger_deposit if honest
                               else challenge.challenger_deposit + challenge.defender_deposit // 2)
    return {"verdict": "CHALLENGE_FAILED" if honest else "SLASH_DEFENDER", "reason": reason,
            "slashed_node": challenge.challenger_id if honest else challenge.defender_id,
            "slashed_amount": challenge.slashed_amount,
            "winner": challenge.defender_id if honest else challenge.challenger_id,
            "settlement": "local_arbitration", "onchain_executed": False}


def submit_dispute_snapshot(challenge: ChallengeState, input_activation: Any,
                            output_activation: Any, *, now: Optional[float] = None) -> bool:
    """Accept immutable copies of both committed snapshots within the bound window."""
    _check_binding(challenge)
    if challenge.status != "OPEN":
        raise FraudProofError(f"cannot submit snapshot in state {challenge.status}")
    submitted = _time(now)
    if submitted < challenge.created_at or submitted > challenge.deadline_at:
        raise FraudProofError("snapshot submission outside challenge window")
    snapshot_input, snapshot_output = _clone(input_activation), _clone(output_activation)
    for name, actual, expected in (("input", hash_activation(snapshot_input), challenge.declared_in_hash),
                                    ("output", hash_activation(snapshot_output), challenge.declared_out_hash)):
        if actual != expected:
            _resolve(challenge, False, f"submitted snapshot {name} hash mismatches declared {name} commitment")
            return False
    challenge.snapshot_input, challenge.snapshot_output = snapshot_input, snapshot_output
    challenge.submitted_at = submitted
    challenge.status = "SNAPSHOT_SUBMITTED"
    return True


def adjudicate_challenge(challenge: ChallengeState, replay_stage_fn: Callable[[Any], Any],
                         validator_arch: str, validator_engine_sha256: Optional[str] = None,
                         tolerance: Optional[float] = None, *, now: Optional[float] = None) -> dict:
    """Return a local settlement recommendation; fail closed on replay identity."""
    _check_binding(challenge)
    if validator_arch != challenge.defender_arch:
        raise FraudProofError(f"architectural mismatch: validator is {validator_arch!r} but defender is "
                              f"{challenge.defender_arch!r}. Replay requires same GPU micro-architecture.")
    try:
        validate_sha256(validator_engine_sha256, "validator_engine_sha256")
    except ValueError as exc:
        raise FraudProofError(str(exc)) from exc
    if validator_engine_sha256 != challenge.engine_sha256:
        raise FraudProofError("engine source mismatch between validator and defended execution")
    if tolerance is not None and tolerance != challenge.numerical_tolerance:
        raise FraudProofError("numerical tolerance must be fixed at challenge creation")
    current = _time(now)
    if current < challenge.created_at:
        raise FraudProofError("adjudication clock precedes challenge creation")
    if challenge.status == "OPEN":
        if current <= challenge.deadline_at:
            raise FraudProofError("challenge window has not expired; defender can still submit")
        return _resolve(challenge, False, "defender failed to submit dispute snapshot within window")
    if challenge.status != "SNAPSHOT_SUBMITTED":
        raise FraudProofError(f"challenge already resolved with status {challenge.status}")
    if challenge.submitted_at is None or not challenge.created_at <= challenge.submitted_at <= challenge.deadline_at:
        raise FraudProofError("snapshot lacks a valid submission timestamp")
    if current < challenge.submitted_at:
        raise FraudProofError("adjudication precedes snapshot submission")
    # State is local Python data: detect accidental or hostile mutation before replay.
    if (hash_activation(challenge.snapshot_input) != challenge.declared_in_hash or
            hash_activation(challenge.snapshot_output) != challenge.declared_out_hash):
        raise FraudProofError("stored dispute snapshot changed after submission")
    replayed_out = replay_stage_fn(_clone(challenge.snapshot_input))
    replayed_hash = hash_activation(replayed_out)
    if replayed_hash == challenge.declared_out_hash:
        return _resolve(challenge, True, "replayed execution output matches declared commitment exactly")
    if (challenge.numerical_tolerance is not None and
            _check_numerical_tolerance(replayed_out, challenge.snapshot_output, challenge.numerical_tolerance)):
        return _resolve(challenge, True, "replayed execution output satisfies the predeclared numerical tolerance")
    return _resolve(challenge, False, f"replay output hash ({replayed_hash[:12]}) does not match "
                    f"defender declared hash ({challenge.declared_out_hash[:12]})")


def _check_numerical_tolerance(a: Any, b: Any, tolerance: float) -> bool:
    """No broadcasting, nonfinite values, or empty mismatched arrays are accepted."""
    try:
        import numpy as np
        def array(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().float().numpy()
            return np.asarray(value, dtype=np.float64)
        arr_a, arr_b = array(a), array(b)
        if arr_a.shape != arr_b.shape or arr_a.size == 0:
            return False
        if not np.all(np.isfinite(arr_a)) or not np.all(np.isfinite(arr_b)):
            return False
        relative = np.abs(arr_a - arr_b) / (np.maximum(np.abs(arr_a), np.abs(arr_b)) + 1e-8)
        return bool(np.max(relative) <= tolerance)
    except (ValueError, TypeError, RuntimeError):
        return False

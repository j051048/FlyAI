"""Same-arch single-stage interactive fraud proof protocol (Phase 2 Web3 hardening).

Enables dispute challenge and single-stage replay verification without re-running
the entire 43-layer / 120B model or relying on heavy ZK-ML/TEE proofs.
"""
from dataclasses import dataclass, field
import hashlib
import time
from typing import Any, Callable, Optional, Union

from .activation_proof import hash_activation, compute_step_commitment


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
    created_at: float = field(default_factory=time.time)
    snapshot_input: Optional[Any] = None
    snapshot_output: Optional[Any] = None
    status: str = "OPEN"  # OPEN, SNAPSHOT_SUBMITTED, RESOLVED_DEFENDER_HONEST, RESOLVED_DEFENDER_SLASHED, EXPIRED
    resolution_reason: Optional[str] = None
    slashed_amount: int = 0
    reward_amount: int = 0


def create_challenge(
    challenge_id: str,
    run_id: str,
    stage_idx: int,
    step: int,
    challenger_id: str,
    defender_id: str,
    challenger_deposit: int,
    defender_deposit: int,
    declared_commitment: str,
    declared_in_hash: str,
    declared_out_hash: str,
    defender_arch: str = "sm120",
    engine_sha256: str = "",
) -> ChallengeState:
    """Initialize an interactive challenge for a disputed stage computation step."""
    if challenger_deposit <= 0 or defender_deposit <= 0:
        raise FraudProofError("deposits must be positive integer amounts")

    return ChallengeState(
        challenge_id=str(challenge_id),
        run_id=str(run_id),
        stage_idx=int(stage_idx),
        step=int(step),
        challenger_id=str(challenger_id),
        defender_id=str(defender_id),
        challenger_deposit=challenger_deposit,
        defender_deposit=defender_deposit,
        declared_commitment=declared_commitment,
        declared_in_hash=declared_in_hash,
        declared_out_hash=declared_out_hash,
        defender_arch=defender_arch,
        engine_sha256=engine_sha256,
    )


def submit_dispute_snapshot(
    challenge: ChallengeState,
    input_activation: Any,
    output_activation: Any,
) -> bool:
    """Defender submits raw tensor snapshot for the disputed step within challenge window."""
    if challenge.status != "OPEN":
        raise FraudProofError(f"cannot submit snapshot in state {challenge.status}")

    # Check input activation integrity against declared hash
    actual_in_hash = hash_activation(input_activation)
    if actual_in_hash != challenge.declared_in_hash:
        challenge.status = "RESOLVED_DEFENDER_SLASHED"
        challenge.resolution_reason = (
            f"submitted snapshot input hash ({actual_in_hash[:12]}) "
            f"mismatches declared input commitment ({challenge.declared_in_hash[:12]})"
        )
        challenge.slashed_amount = challenge.defender_deposit
        challenge.reward_amount = challenge.challenger_deposit + (challenge.defender_deposit // 2)
        return False

    challenge.snapshot_input = input_activation
    challenge.snapshot_output = output_activation
    challenge.status = "SNAPSHOT_SUBMITTED"
    return True


def adjudicate_challenge(
    challenge: ChallengeState,
    replay_stage_fn: Callable[[Any], Any],
    validator_arch: str,
    validator_engine_sha256: Optional[str] = None,
    tolerance: Optional[float] = None,
) -> dict:
    """Execute single-stage replay verification and adjudicate deposits.

    Enforces same-arch validator matching to prevent false slashing from floating-point
    non-associativity across GPU generations.
    """
    if challenge.status == "OPEN":
        # Defender failed to provide snapshot -> default slash
        challenge.status = "RESOLVED_DEFENDER_SLASHED"
        challenge.resolution_reason = "defender failed to submit dispute snapshot within window"
        challenge.slashed_amount = challenge.defender_deposit
        challenge.reward_amount = challenge.challenger_deposit + (challenge.defender_deposit // 2)
        return {
            "verdict": "SLASH_DEFENDER",
            "reason": challenge.resolution_reason,
            "slashed_node": challenge.defender_id,
            "slashed_amount": challenge.slashed_amount,
            "winner": challenge.challenger_id,
        }

    if challenge.status != "SNAPSHOT_SUBMITTED":
        raise FraudProofError(f"challenge already resolved with status {challenge.status}")

    # Same-arch enforcement
    if validator_arch != challenge.defender_arch:
        raise FraudProofError(
            f"architectural mismatch: validator is {validator_arch!r} but defender is "
            f"{challenge.defender_arch!r}. Replay requires same GPU micro-architecture."
        )

    if challenge.engine_sha256 and validator_engine_sha256:
        if validator_engine_sha256 != challenge.engine_sha256:
            raise FraudProofError("engine source mismatch between validator and defended execution")

    # Replay execution of single stage
    replayed_out = replay_stage_fn(challenge.snapshot_input)
    replayed_out_hash = hash_activation(replayed_out)

    is_honest = False
    if replayed_out_hash == challenge.declared_out_hash:
        is_honest = True
    elif tolerance is not None:
        # Check bounded numerical divergence if tolerance specified
        is_honest = _check_numerical_tolerance(replayed_out, challenge.snapshot_output, tolerance)

    if is_honest:
        # Defender was honest; challenger loses deposit
        challenge.status = "RESOLVED_DEFENDER_HONEST"
        challenge.resolution_reason = "replayed execution output matches declared commitment bit-for-bit"
        challenge.slashed_amount = challenge.challenger_deposit
        challenge.reward_amount = challenge.defender_deposit + challenge.challenger_deposit
        return {
            "verdict": "CHALLENGE_FAILED",
            "reason": challenge.resolution_reason,
            "slashed_node": challenge.challenger_id,
            "slashed_amount": challenge.slashed_amount,
            "winner": challenge.defender_id,
        }
    else:
        # Defender committed fraud
        challenge.status = "RESOLVED_DEFENDER_SLASHED"
        challenge.resolution_reason = (
            f"replay output hash ({replayed_out_hash[:12]}) does not match "
            f"defender declared hash ({challenge.declared_out_hash[:12]})"
        )
        challenge.slashed_amount = challenge.defender_deposit
        challenge.reward_amount = challenge.challenger_deposit + (challenge.defender_deposit // 2)
        return {
            "verdict": "SLASH_DEFENDER",
            "reason": challenge.resolution_reason,
            "slashed_node": challenge.defender_id,
            "slashed_amount": challenge.slashed_amount,
            "winner": challenge.challenger_id,
        }


def _check_numerical_tolerance(a: Any, b: Any, tolerance: float) -> bool:
    """Helper to check if two numerical arrays/lists are within relative tolerance."""
    try:
        import numpy as np
        arr_a = np.asarray(a, dtype=np.float32)
        arr_b = np.asarray(b, dtype=np.float32)
        diff = np.abs(arr_a - arr_b)
        denom = np.maximum(np.abs(arr_a), np.abs(arr_b)) + 1e-8
        rel_diff = np.max(diff / denom)
        return float(rel_diff) <= tolerance
    except Exception:
        return False

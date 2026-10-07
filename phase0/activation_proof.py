"""Lightweight activation commitments and verification seam (Phase 1 Web3 hardening).

Provides deterministic, constant-overhead commitments for pipeline stages
without persisting full tensor activations across all decode steps.
"""
import hashlib
import json
from typing import Any, Sequence, Optional


def hash_activation(tensor_or_data: Any, stride: int = 1) -> str:
    """Compute deterministic SHA-256 digest of an activation tensor or payload.

    Operates on raw underlying bytes. No full float deserialization required.
    """
    if tensor_or_data is None:
        return hashlib.sha256(b"<empty>").hexdigest()

    # Support PyTorch tensor if present
    if hasattr(tensor_or_data, "detach") and hasattr(tensor_or_data, "cpu"):
        t = tensor_or_data.detach().cpu().contiguous()
        # View raw bytes
        if hasattr(t, "untyped_storage"):
            raw_bytes = t.untyped_storage().tobytes()
        else:
            raw_bytes = t.numpy().tobytes()
        if stride > 1 and len(raw_bytes) > stride:
            raw_bytes = raw_bytes[::stride]
        return hashlib.sha256(raw_bytes).hexdigest()

    # Support NumPy ndarray
    if hasattr(tensor_or_data, "tobytes"):
        raw_bytes = tensor_or_data.tobytes()
        if stride > 1 and len(raw_bytes) > stride:
            raw_bytes = raw_bytes[::stride]
        return hashlib.sha256(raw_bytes).hexdigest()

    # Raw bytes or bytearray
    if isinstance(tensor_or_data, (bytes, bytearray)):
        raw = bytes(tensor_or_data[::stride]) if stride > 1 else bytes(tensor_or_data)
        return hashlib.sha256(raw).hexdigest()

    # Fallback JSON serialization for Python primitives
    serialized = json.dumps(tensor_or_data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def compute_step_commitment(
    stage_idx: int,
    step: int,
    in_hash: str,
    out_hash: str,
    prev_commitment: Optional[str] = None,
) -> str:
    """Compute cryptographically linked commitment for stage i at step t.

    commitment_t = SHA256(stage_idx || step || in_hash || out_hash || prev_commitment)
    """
    prev_str = prev_commitment or "0" * 64
    payload = f"{stage_idx}:{step}:{in_hash}:{out_hash}:{prev_str}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class StageCommitmentTracker:
    """Tracks step-by-step activation commitments for a pipeline stage."""

    def __init__(self, stage_idx: int, initial_commitment: Optional[str] = None):
        self.stage_idx = int(stage_idx)
        self._current_commitment = initial_commitment or ("0" * 64)
        self._records = []

    @property
    def current_commitment(self) -> str:
        return self._current_commitment

    def record_step(self, step: int, in_hash: str, out_hash: str) -> str:
        """Advance the commitment chain by one forward computation step."""
        comm = compute_step_commitment(
            stage_idx=self.stage_idx,
            step=step,
            in_hash=in_hash,
            out_hash=out_hash,
            prev_commitment=self._current_commitment,
        )
        self._records.append({
            "stage_idx": self.stage_idx,
            "step": step,
            "in_hash": in_hash,
            "out_hash": out_hash,
            "prev_commitment": self._current_commitment,
            "commitment": comm,
        })
        self._current_commitment = comm
        return comm

    def history(self) -> list[dict]:
        return list(self._records)


def verify_commitment_chain(records: Sequence[dict]) -> tuple[bool, Optional[str]]:
    """Verify that an ordered list of step commitment records is cryptographically valid.

    Returns:
        (is_valid, error_reason)
    """
    if not records:
        return True, None

    expected_prev = records[0].get("prev_commitment", "0" * 64)
    for idx, r in enumerate(records):
        stage_idx = r.get("stage_idx")
        step = r.get("step")
        in_h = r.get("in_hash")
        out_h = r.get("out_hash")
        prev = r.get("prev_commitment")
        comm = r.get("commitment")

        if prev != expected_prev:
            return False, f"record {idx} broken link: expected prev {expected_prev}, got {prev}"

        computed = compute_step_commitment(stage_idx, step, in_h, out_h, prev)
        if computed != comm:
            return False, f"record {idx} invalid commitment: computed {computed}, declared {comm}"

        expected_prev = comm

    return True, None

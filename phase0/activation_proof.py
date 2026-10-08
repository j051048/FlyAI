"""Lightweight activation commitments and verification seam (Phase 1 Web3 hardening).

Hashes every logical activation byte; GPU hashing copies to CPU and costs O(bytes).
These local consistency commitments are separate from signed serving receipts.
"""
import hashlib
import json
import re
import copy
from typing import Any, Sequence, Optional


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ZERO_COMMITMENT = "0" * 64


def validate_sha256(value, name="hash"):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def hash_activation(tensor_or_data: Any, stride: int = 1) -> str:
    """Compute deterministic SHA-256 digest of an activation tensor or payload.

    Tensor commitments bind dtype, shape, and logical contiguous bytes (not the
    whole backing storage). Byte sampling is forbidden for binding commitments.
    """
    if type(stride) is not int or stride != 1:
        raise ValueError("activation commitments require every byte (stride=1)")
    if tensor_or_data is None:
        return hashlib.sha256(b"activation-none/1\0").hexdigest()

    # Support PyTorch tensor if present
    if hasattr(tensor_or_data, "detach") and hasattr(tensor_or_data, "cpu"):
        t = tensor_or_data.detach().cpu().contiguous()
        import torch
        # uint8 avoids numpy's missing bfloat16/float8 support. reshape removes
        # scalar view restrictions and contiguous excludes unrelated view bytes.
        raw_bytes = t.reshape(-1).view(torch.uint8).numpy().tobytes()
        metadata = json.dumps([str(t.dtype), list(t.shape)], separators=(",", ":")).encode()
        return hashlib.sha256(b"activation-torch/1\0" + metadata + b"\0" + raw_bytes).hexdigest()

    # Support NumPy ndarray
    if hasattr(tensor_or_data, "tobytes") and hasattr(tensor_or_data, "dtype") and hasattr(tensor_or_data, "shape"):
        if getattr(tensor_or_data.dtype, "hasobject", False):
            raise ValueError("object arrays have no deterministic activation byte representation")
        metadata = json.dumps([str(tensor_or_data.dtype), list(tensor_or_data.shape)], separators=(",", ":")).encode()
        raw_bytes = tensor_or_data.tobytes(order="C")
        return hashlib.sha256(b"activation-numpy/1\0" + metadata + b"\0" + raw_bytes).hexdigest()

    # Raw bytes or bytearray
    if isinstance(tensor_or_data, (bytes, bytearray)):
        raw = bytes(tensor_or_data)
        return hashlib.sha256(b"activation-bytes/1\0" + raw).hexdigest()

    # Fallback JSON serialization for Python primitives
    serialized = json.dumps(tensor_or_data, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(b"activation-json/1\0" + serialized).hexdigest()


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
    _nonnegative_integer(stage_idx, "stage_idx")
    _nonnegative_integer(step, "step")
    validate_sha256(in_hash, "in_hash")
    validate_sha256(out_hash, "out_hash")
    prev_str = ZERO_COMMITMENT if prev_commitment is None else validate_sha256(prev_commitment, "prev_commitment")
    payload = f"{stage_idx}:{step}:{in_hash}:{out_hash}:{prev_str}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class StageCommitmentTracker:
    """Tracks step-by-step activation commitments for a pipeline stage."""

    def __init__(self, stage_idx: int, initial_commitment: Optional[str] = None):
        self.stage_idx = _nonnegative_integer(stage_idx, "stage_idx")
        self._current_commitment = (ZERO_COMMITMENT if initial_commitment is None
                                    else validate_sha256(initial_commitment, "initial_commitment"))
        self._records = []

    @property
    def current_commitment(self) -> str:
        return self._current_commitment

    def record_step(self, step: int, in_hash: str, out_hash: str) -> str:
        """Advance the commitment chain by one forward computation step."""
        if self._records and step != self._records[-1]["step"] + 1:
            raise ValueError("commitment steps must be consecutive")
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
        return copy.deepcopy(self._records)


def verify_commitment_chain(records: Sequence[dict], *, initial_commitment=ZERO_COMMITMENT,
                            expected_stage=None, expected_final=None) -> tuple[bool, Optional[str]]:
    """Verify that an ordered list of step commitment records is cryptographically valid.

    Returns:
        (is_valid, error_reason)
    """
    try:
        expected_prev = validate_sha256(initial_commitment, "initial_commitment")
        if expected_stage is not None:
            _nonnegative_integer(expected_stage, "expected_stage")
        if expected_final is not None:
            validate_sha256(expected_final, "expected_final")
    except ValueError as exc:
        return False, str(exc)
    if not isinstance(records, (list, tuple)):
        return False, "records must be a list of commitment records"
    if not records:
        return (True, None) if expected_final in (None, expected_prev) else (False, "empty chain final commitment mismatch")
    chain_stage = expected_stage
    previous_step = None
    for idx, r in enumerate(records):
        if not isinstance(r, dict):
            return False, f"record {idx} must be an object"
        stage_idx = r.get("stage_idx")
        step = r.get("step")
        in_h = r.get("in_hash")
        out_h = r.get("out_hash")
        prev = r.get("prev_commitment")
        comm = r.get("commitment")

        try:
            _nonnegative_integer(stage_idx, "stage_idx")
            _nonnegative_integer(step, "step")
            validate_sha256(comm, "commitment")
            computed = compute_step_commitment(stage_idx, step, in_h, out_h, prev)
        except ValueError as exc:
            return False, f"record {idx} invalid: {exc}"
        if chain_stage is None:
            chain_stage = stage_idx
        if stage_idx != chain_stage:
            return False, f"record {idx} stage mismatch"
        if previous_step is not None and step != previous_step + 1:
            return False, f"record {idx} nonconsecutive step"
        if prev != expected_prev:
            return False, f"record {idx} broken link: expected prev {expected_prev}, got {prev}"

        if computed != comm:
            return False, f"record {idx} invalid commitment: computed {computed}, declared {comm}"

        expected_prev = comm
        previous_step = step

    if expected_final is not None and expected_prev != expected_final:
        return False, "final commitment mismatch"

    return True, None

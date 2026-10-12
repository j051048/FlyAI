"""Replay receipt-bound native greedy transcripts on a dedicated trusted replica.

Every input/output leaf is checked against the signed roots before re-execution.
This checks the assigned stage, not MTP, sampling or the whole network. The local
validator and supplied execution identity remain trust assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import threading
import time

import torch

from shard.node_quality import canonical, replay_report
from shard.receipt import verify_receipt, wire_receipt
from v4_conversation_cache import _model_stamp
from v4_observability import source_inventory
from v4_resources import runtime_config_identity


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


@dataclass(frozen=True)
class ReplayChunk:
    position: int
    hidden: torch.Tensor
    token_ids: torch.Tensor
    output_bytes: bytes

    def input_bytes(self):
        return tensor_bytes(self.hidden) + tensor_bytes(self.token_ids)


class NativeReplayAuditor:
    def __init__(self, stage, validator_key, *, cohort_id, numeric_contract,
                 max_chunks=4096, max_input_bytes=64 << 20, max_output_bytes=16 << 20,
                 fixture_reference=False, clock=time.time):
        if getattr(stage, "_dspark_capable", False) or getattr(stage, "_native_lanes", None) is not None:
            raise ValueError("auditor needs a dedicated non-MTP stage, not a live serving lane")
        if (not isinstance(cohort_id, str) or len(cohort_id) != 64 or
                any(c not in "0123456789abcdef" for c in cohort_id) or
                not isinstance(numeric_contract, str) or not 1 <= len(numeric_contract) <= 256):
            raise ValueError("pinned cohort and numerical contract required")
        for value, limit in ((max_chunks, 16384), (max_input_bytes, 128 << 20), (max_output_bytes, 16 << 20)):
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError("bounded replay work required")
        proof = getattr(stage, "_artifact_verification", None)
        self._require_raw_wire()
        if not fixture_reference and stage.dtype != torch.bfloat16:
            raise ValueError("production native replay v1 requires BF16 stage and wire dtype")
        if fixture_reference:
            if str(stage.device) != "cpu":
                raise ValueError("fixture reference cannot claim GPU audit evidence")
        elif not proof or not getattr(stage, "_resource_checkpoint_dir", None):
            raise ValueError("auditor must load fully hash-verified native stage artifacts")
        self.stage, self.key, self.clock = stage, validator_key, clock
        self.numeric_contract, self.cohort_id = numeric_contract, cohort_id
        self.max_chunks, self.max_input_bytes, self.max_output_bytes = max_chunks, max_input_bytes, max_output_bytes
        self.fixture_reference = fixture_reference
        self._lock = threading.Lock()
        self._stamp = _model_stamp(stage, None)
        self.identity = {"cohort_id": cohort_id, "numeric_contract": numeric_contract,
            "source_sha256": hashlib.sha256(canonical(source_inventory())).hexdigest(),
            "runtime_config_sha256": runtime_config_identity(stage), "layers": [stage.lo, stage.hi],
            "horizon": stage.args.max_seq_len,
            "checkpoint_id": proof["checkpoint_id"] if proof else "explicit-cpu-fixture",
            "manifest_sha256": proof["manifest_sha256"] if proof else "explicit-cpu-fixture"}
        self.execution_identity_sha256 = hashlib.sha256(canonical(self.identity)).hexdigest()

    @staticmethod
    def _require_raw_wire(receipt=None):
        import v4_pipe
        if v4_pipe.V4_FP8_WIRE or v4_pipe.V4_SEALED_IDS or (
                receipt and receipt.get("runtime_observation", {}).get("wire_mode", "bf16") != "bf16"):
            raise ValueError("native replay v1 requires raw BF16 wire; FP8/sealed payloads need a separate contract")

    def validate_cohort(self, cohort):
        """Check actual verified payload identity before a quality case is issued."""
        from shard.weight_artifacts import verified_stage_artifact_descriptor, catalog_model_cohort
        proof = getattr(self.stage, "_artifact_verification", None)
        if self.fixture_reference or not proof:
            raise ValueError("production quality audit requires a verified model cohort replica")
        verified_stage_artifact_descriptor(proof)
        actual = catalog_model_cohort(proof["catalog"])
        fields = ("model_id", "checkpoint_id", "manifest_sha256", "config_sha256", "quantization", "runtime_abi", "n_layers")
        if (cohort.cohort_id != self.cohort_id or cohort.numeric_contract != self.numeric_contract or
                any(actual[name] != getattr(cohort, name) for name in fields) or
                self.stage.hi > cohort.n_layers):
            raise ValueError("trusted replay artifact/configuration differs from the selected model cohort")

    def _validate_chunks(self, chunks):
        if not isinstance(chunks, (tuple, list)) or not 1 <= len(chunks) <= self.max_chunks:
            raise ValueError("bounded original greedy transcript required")
        position, in_bytes, out_bytes = 0, 0, 0
        for chunk in chunks:
            if (not isinstance(chunk, ReplayChunk) or type(chunk.position) is not int or chunk.position != position or
                    not torch.is_tensor(chunk.hidden) or chunk.hidden.device.type != "cpu" or
                    chunk.hidden.dtype != self.stage.dtype or chunk.hidden.ndim != 4 or
                    chunk.hidden.shape[0] != 1 or chunk.hidden.shape[2:] != (self.stage.args.hc_mult, self.stage.args.dim) or
                    not torch.is_tensor(chunk.token_ids) or chunk.token_ids.device.type != "cpu" or
                    chunk.token_ids.dtype != torch.long or tuple(chunk.token_ids.shape) != tuple(chunk.hidden.shape[:2]) or
                    not isinstance(chunk.output_bytes, bytes) or not chunk.output_bytes):
                raise ValueError("original contiguous greedy positions, hidden states and routing IDs required")
            size = chunk.hidden.shape[1]
            if size < 1 or position and size != 1:
                raise ValueError("greedy replay supports one initial prefill then single-token steps")
            if bool((chunk.token_ids < 0).any()) or bool((chunk.token_ids >= self.stage.args.vocab_size).any()):
                raise ValueError("routing token IDs exceed pinned model vocabulary")
            position += size
            in_bytes += chunk.hidden.numel() * chunk.hidden.element_size() + chunk.token_ids.numel() * 8
            if len(chunk.output_bytes) != chunk.hidden.numel() * chunk.hidden.element_size() + chunk.token_ids.numel() * 8:
                raise ValueError("signed output size differs from the actual native forward layout")
            out_bytes += len(chunk.output_bytes)
            if position > self.stage.args.max_seq_len or in_bytes > self.max_input_bytes or out_bytes > self.max_output_bytes:
                raise ValueError("greedy transcript exceeds replay resource budget")

    def input_identity(self, chunks):
        self._validate_chunks(chunks)
        body = {"horizon": self.stage.args.max_seq_len, "chunks": [
            {"position": chunk.position, "shape": list(chunk.hidden.shape),
             "input_sha256": hashlib.sha256(chunk.input_bytes()).hexdigest()} for chunk in chunks]}
        return hashlib.sha256(canonical(body)).hexdigest()

    def run(self, case, receipt, chunks):
        with self._lock:
            self._validate_chunks(chunks)
            # Freeze external buffers before validating their signed commitments.
            # Concurrent mutation can no longer change the input after validation.
            frozen = [ReplayChunk(c.position, c.hidden.detach().clone(), c.token_ids.detach().clone(), c.output_bytes)
                      for c in chunks]
            return self._run(json.loads(canonical(case)), json.loads(canonical(receipt)), frozen)

    def _run(self, case, receipt, chunks):
        self._validate_chunks(chunks)
        receipt = wire_receipt(receipt)
        self._require_raw_wire(receipt)
        verify_receipt(receipt, case["expected_pubkey"])
        if (hashlib.sha256(canonical(receipt)).hexdigest() != case["receipt_sha256"] or
                case["cohort_id"] != self.cohort_id or case["numeric_contract"] != self.numeric_contract or
                case["execution_identity_sha256"] != self.execution_identity_sha256 or
                case["input_sha256"] != self.input_identity(chunks) or
                not case["issued_at"] <= self.clock() < case["expires_at"] or
                [receipt["layer_start"], receipt["layer_end"]] != self.identity["layers"] or
                receipt["n_chunks"] != len(chunks)):
            raise ValueError("replay case differs from original receipt, input or pinned execution")
        in_root = hashlib.sha256(b"".join(hashlib.sha256(chunk.input_bytes()).digest() for chunk in chunks)).hexdigest()
        out_root = hashlib.sha256(b"".join(hashlib.sha256(chunk.output_bytes).digest() for chunk in chunks)).hexdigest()
        if in_root != receipt["in_root"] or out_root != receipt["out_root"]:
            raise ValueError("replay transcript is not bound to the signed activation roots")
        if (_model_stamp(self.stage, None) != self._stamp or runtime_config_identity(self.stage) != self.identity["runtime_config_sha256"] or
                hashlib.sha256(canonical(source_inventory())).hexdigest() != self.identity["source_sha256"]):
            raise ValueError("trusted replay weights, source or runtime changed")
        if not self.fixture_reference:
            from v4_stage import verify_checkpoint_artifacts
            verify_checkpoint_artifacts(self.stage._resource_checkpoint_dir, required=True,
                expected_checkpoint_id=self.identity["checkpoint_id"], expected_manifest_sha256=self.identity["manifest_sha256"],
                lo=self.stage.lo, hi=self.stage.hi, head=self.stage.head, tail=self.stage.tail, dspark=False)
        started = time.perf_counter()
        original, replayed = [], []
        try:
            self.stage.reset()
            with torch.no_grad():
                for chunk in chunks:
                    if self.clock() >= case["expires_at"]:
                        raise ValueError("replay challenge expired during execution")
                    ids = chunk.token_ids.to(self.stage.device)
                    hidden = self.stage.embed(ids) if self.stage.head else chunk.hidden.to(self.stage.device)
                    if self.stage.head and tensor_bytes(hidden) != tensor_bytes(chunk.hidden):
                        raise ValueError("signed head embedding differs from the pinned token inputs")
                    output = self.stage.forward(hidden, ids, chunk.position)
                    original.append(chunk.output_bytes)
                    replayed.append(tensor_bytes(output) + tensor_bytes(ids))
            return replay_report(self.key, case, reference_bytes=b"".join(original), candidate_bytes=b"".join(replayed),
                input_sha256=case["input_sha256"], execution_identity_sha256=self.execution_identity_sha256,
                evidence="gpu_replay" if not self.fixture_reference and str(self.stage.device).startswith("cuda") else "cpu_reference",
                elapsed_ms=(time.perf_counter() - started) * 1000)
        finally:
            # This dedicated replica cannot leak a previous audited request's state.
            self.stage.reset()

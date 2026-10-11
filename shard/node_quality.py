"""Bounded coordinator-issued replay audits and locally trusted quality feedback.

Node offers cannot update these scores. A pinned replica validator must compare
the assigned numerical contract, then sign a fresh issued case. This is a local
audit policy, not permissionless consensus, hardware attestation or settlement.
"""
import base64
from collections import OrderedDict, deque
import hashlib
import json
import math
import secrets
import threading
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from .receipt import verify_receipt, wire_receipt, pub_b64

SCHEMA = "shard-replay-audit/1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def replay_report(key, case, *, reference_bytes, candidate_bytes, input_sha256,
                  execution_identity_sha256, evidence="cpu_reference", elapsed_ms=0):
    """Trusted validator boundary; byte equality never silently adds a tolerance.

    The caller must replay the receipt-bound original input/state with the pinned
    model/source/configuration. Supplying arbitrary equal buffers is not a model
    proof. The report binds that validator's declaration and comparison cost.
    """
    if not isinstance(reference_bytes, bytes) or not isinstance(candidate_bytes, bytes):
        raise ValueError("detached logical tensor bytes required")
    if not reference_bytes or len(reference_bytes) > 16 << 20 or len(candidate_bytes) > 16 << 20:
        raise ValueError("bounded nonempty replay outputs required")
    if evidence not in ("cpu_reference", "gpu_replay"):
        raise ValueError("explicit replay backend evidence required")
    for digest in (input_sha256, execution_identity_sha256):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("replay input/runtime identity digest required")
    if type(elapsed_ms) not in (int, float) or not math.isfinite(elapsed_ms) or elapsed_ms < 0:
        raise ValueError("finite replay cost required")
    body = {"schema": SCHEMA, "case": case, "input_sha256": input_sha256,
        "execution_identity_sha256": execution_identity_sha256, "evidence": evidence,
        "verdict": "match" if reference_bytes == candidate_bytes else "mismatch",
        "checked_bytes": max(len(reference_bytes), len(candidate_bytes)), "elapsed_ms": elapsed_ms,
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(), "validator_pubkey": pub_b64(key)}
    body["signature"] = base64.b64encode(key.sign(b"shard-replay-audit/1\0" + canonical(body))).decode()
    return body


class NodeQuality:
    def __init__(self, validator_pubkeys, *, max_nodes=1024, max_pending=64,
                 max_events=4096, case_ttl_s=60, history_s=3600, failure_threshold=2,
                 quarantine_s=300, required_evidence="gpu_replay", clock=time.time):
        self.validators = frozenset(validator_pubkeys)
        if not self.validators or len(self.validators) > 64:
            raise ValueError("1..64 locally trusted replay validator identities required")
        for key in self.validators:
            Ed25519PublicKey.from_public_bytes(base64.b64decode(key, validate=True))
        for name, value in (("max_nodes", max_nodes), ("max_pending", max_pending),
                            ("max_events", max_events), ("failure_threshold", failure_threshold)):
            if type(value) is not int or not 1 <= value <= 100000:
                raise ValueError("bounded quality capacities required: " + name)
        for value in (case_ttl_s, history_s, quarantine_s):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("finite positive quality TTLs required")
        if required_evidence not in ("gpu_replay", "cpu_reference"):
            raise ValueError("unknown required replay evidence")
        self.max_nodes, self.max_pending, self.max_events = max_nodes, max_pending, max_events
        self.case_ttl_s, self.history_s, self.threshold, self.quarantine_s = case_ttl_s, history_s, failure_threshold, quarantine_s
        self.required_evidence, self.clock = required_evidence, clock
        self.pending, self.events, self.quarantine = OrderedDict(), deque(), OrderedDict()
        self._lock = threading.RLock()

    def _prune(self):
        now = self.clock()
        for key, case in tuple(self.pending.items()):
            if now > case["expires_at"]:
                self.pending.pop(key)
        while self.events and now - self.events[0][0] > self.history_s:
            self.events.popleft()
        for key, until in tuple(self.quarantine.items()):
            if until <= now:
                self.quarantine.pop(key)

    def issue(self, node_id, cohort_id, receipt, *, expected_pubkey, numeric_contract,
              input_sha256, execution_identity_sha256):
        receipt = wire_receipt(receipt)
        verify_receipt(receipt, expected_pubkey)
        if type(receipt.get("n_chunks")) is not int or receipt["n_chunks"] < 1:
            raise ValueError("a replay audit needs a nonempty actual signed job")
        lo, hi = receipt.get("layer_start"), receipt.get("layer_end")
        if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi:
            raise ValueError("replay receipt has invalid assigned layer range")
        for name in ("in_root", "out_root"):
            value = receipt.get(name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("replay receipt has an invalid activation commitment")
        if not isinstance(node_id, str) or not 1 <= len(node_id) <= 512 or not isinstance(numeric_contract, str) or not 1 <= len(numeric_contract) <= 256:
            raise ValueError("bounded node/numerical contract required")
        for value in (cohort_id, input_sha256, execution_identity_sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("cohort/input/execution identity must be SHA256")
        case = {"case_id": secrets.token_hex(16), "nonce": secrets.token_hex(32),
            "node_id": node_id, "cohort_id": cohort_id, "receipt_sha256": _digest(receipt),
            "expected_pubkey": expected_pubkey, "numeric_contract": numeric_contract,
            "input_sha256": input_sha256, "execution_identity_sha256": execution_identity_sha256,
            "issued_at": self.clock(), "expires_at": self.clock() + self.case_ttl_s}
        with self._lock:
            self._prune()
            if len(self.pending) >= self.max_pending:
                raise ValueError("pending audit capacity exhausted")
            if any(row[2] == case["receipt_sha256"] for row in self.events) or any(
                    other["receipt_sha256"] == case["receipt_sha256"] for other in self.pending.values()):
                raise ValueError("the same signed job cannot produce repeated quality strikes")
            self.pending[case["case_id"]] = case
            return json.loads(canonical(case))

    def ingest(self, report):
        fields = {"schema", "case", "input_sha256", "execution_identity_sha256", "evidence",
            "verdict", "checked_bytes", "elapsed_ms", "reference_sha256", "candidate_sha256",
            "validator_pubkey", "signature"}
        if not isinstance(report, dict) or set(report) != fields or report["schema"] != SCHEMA:
            raise ValueError("invalid replay audit report")
        pub = report["validator_pubkey"]
        if pub not in self.validators:
            raise ValueError("replay auditor is not locally trusted")
        body = {k: v for k, v in report.items() if k != "signature"}
        Ed25519PublicKey.from_public_bytes(base64.b64decode(pub, validate=True)).verify(
            base64.b64decode(report["signature"], validate=True), b"shard-replay-audit/1\0" + canonical(body))
        if report["evidence"] != self.required_evidence or report["verdict"] not in ("match", "mismatch"):
            raise ValueError("unqualified replay backend or verdict")
        if type(report["checked_bytes"]) is not int or not 1 <= report["checked_bytes"] <= 16 << 20:
            raise ValueError("bounded replay comparison required")
        if type(report["elapsed_ms"]) not in (int, float) or not math.isfinite(report["elapsed_ms"]) or report["elapsed_ms"] < 0:
            raise ValueError("invalid declared replay cost")
        for name in ("reference_sha256", "candidate_sha256"):
            value = report[name]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("replay content digest required")
        if (report["reference_sha256"] == report["candidate_sha256"]) != (report["verdict"] == "match"):
            raise ValueError("replay verdict contradicts compared content")
        case = report["case"]
        if not isinstance(case, dict):
            raise ValueError("issued replay case required")
        with self._lock:
            self._prune()
            pending = self.pending.get(case.get("case_id"))
            if pending is None or pending != case:
                raise ValueError("expired, replayed or substituted audit case")
            if report["input_sha256"] != case["input_sha256"] or report["execution_identity_sha256"] != case["execution_identity_sha256"]:
                raise ValueError("validator replayed another input/runtime")
            self.pending.pop(case["case_id"])
            node = (case["node_id"], case["cohort_id"])
            self.events.append((self.clock(), node, case["receipt_sha256"], report["verdict"], report["elapsed_ms"]))
            while len(self.events) > self.max_events:
                self.events.popleft()
            nodes = list(dict.fromkeys(row[1] for row in self.events))
            while len(nodes) > self.max_nodes:
                oldest = nodes.pop(0)
                self.events = deque(row for row in self.events if row[1] != oldest)
                self.quarantine.pop(oldest, None)
            if sum(row[1] == node and row[3] == "mismatch" for row in self.events) >= self.threshold:
                self.quarantine[node] = self.clock() + self.quarantine_s
            while len(self.quarantine) > self.max_nodes:
                self.quarantine.popitem(last=False)
            return self.view(*node)

    def view(self, node_id, cohort_id):
        with self._lock:
            self._prune()
            node = (node_id, cohort_id)
            events = [row for row in self.events if row[1] == node]
            return {"schema": "shard-node-quality/1", "observed_jobs": len(events),
                "matched": sum(row[3] == "match" for row in events),
                "mismatched": sum(row[3] == "mismatch" for row in events),
                "declared_replay_cost_ms": sum(row[4] for row in events),
                "quarantined": node in self.quarantine,
                "evidence": self.required_evidence, "authority": "local_pinned_replay_validator"}

    def permits(self, node_id, cohort_id):
        return not self.view(node_id, cohort_id)["quarantined"]

"""Bounded signed evidence for operator-local incremental qualification.

Signatures bind declared measurements to pinned operators and workers; they do
not prove an untrusted GPU executed arithmetic. Native case producers must hash
their actual reference/candidate state and outputs, never turn a test label into
a boolean qualification. Missing cases produce a pending campaign.
"""
import base64
import hashlib
import json

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

EVIDENCE_SCHEMA = "v4-incremental-prefix-evidence/1"
CASE_SCHEMA = "v4-incremental-native-case/1"
CASE_DOMAIN = b"v4-incremental-native-case/1\0"
MAX_BYTES = 4 << 20
CASES = ("append", "compression_boundary", "rollback", "cancel", "reset", "tenant_isolation")
OPERATIONS = {
    "append": ("prefill", "snapshot", "append"),
    "compression_boundary": ("prefill", "snapshot", "append", "compression_boundary"),
    "rollback": ("prefill", "snapshot", "speculate", "rollback", "append"),
    "cancel": ("prefill", "snapshot", "cancel", "reset", "restore", "append"),
    "reset": ("prefill", "reset", "prefill"),
    "tenant_isolation": ("tenant_a", "snapshot", "tenant_b", "restore_tenant_a", "append"),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("content SHA256 must be lowercase hexadecimal")
    return value


def public_key(value):
    if not isinstance(value, str):
        raise ValueError("pinned public key required")
    raw = base64.b64decode(value, validate=True)
    if len(raw) != 32:
        raise ValueError("Ed25519 public key must contain 32 bytes")
    return Ed25519PublicKey.from_public_bytes(raw)


def signed(value, key, domain):
    if not isinstance(value, dict) or not isinstance(value.get("signature"), str):
        raise ValueError("signed numerical evidence required")
    body = {k: v for k, v in value.items() if k != "signature"}
    public_key(key).verify(base64.b64decode(value["signature"], validate=True), domain + canonical(body))
    return body


def _stage_rows(value, binding):
    if not isinstance(value, list) or not 1 <= len(value) <= 16:
        raise ValueError("bounded complete pinned stage assignment required")
    cursor, keys, nodes = 0, set(), set()
    for index, row in enumerate(value):
        if (not isinstance(row, dict) or set(row) != {"stage", "node_id", "pubkey", "lo", "hi", "runtime_config_sha256", "gpu_uuid"}
                or type(row["stage"]) is not int or row["stage"] != index
                or type(row["lo"]) is not int or row["lo"] != cursor or type(row["hi"]) is not int or row["hi"] <= cursor
                or not isinstance(row["node_id"], str) or not 1 <= len(row["node_id"]) <= 256
                or row["node_id"] in nodes or row["pubkey"] in keys
                or not isinstance(row["gpu_uuid"], str) or not 1 <= len(row["gpu_uuid"]) <= 128):
            raise ValueError("stage assignment must tile the model and pin each node/key/GPU")
        public_key(row["pubkey"])
        sha(row["runtime_config_sha256"])
        cursor = row["hi"]
        nodes.add(row["node_id"]); keys.add(row["pubkey"])
    if nodes != {row[0] for row in binding["lease_fences"]}:
        raise ValueError("evidence assignment differs from the qualified live lease nodes")
    return value


def _gpu_observation(observed, stage, report):
    if (not isinstance(observed, dict) or observed.get("gpu_uuid") != stage["gpu_uuid"]
            or observed.get("kernel_backend") != "tilelang"
            or observed.get("source_sha256") != report["source_sha256"]
            or observed.get("node_id") != stage["node_id"] or observed.get("phase") != "job_complete"
            or observed.get("wire_mode") != "bf16"
            or not observed.get("versions", {}).get("cuda") or not observed.get("versions", {}).get("tilelang")
            or observed.get("runtime_config_sha256") != stage["runtime_config_sha256"]):
        raise ValueError("actual GPU runtime observation differs from the pinned qualification")


def _strict_canary(value, evidence, report):
    required = {"begin_request", "compare_request", "acks", "receipts"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("complete signed strict append canary required")
    coordinator, stages = evidence["coordinator_pubkey"], evidence["stages"]
    domain = b"v4-conversation-control/1\0"
    begin = signed(value["begin_request"], coordinator, domain)
    compare = signed(value["compare_request"], coordinator, domain)
    if begin.get("action") != "shadow_begin" or compare.get("action") != "shadow_compare":
        raise ValueError("strict canary needs actual begin and comparison operations")
    for request in (begin, compare):
        if (request.get("schema") != "v4-conversation-control/1" or request.get("execution") != report["execution"]
                or not isinstance(request.get("job"), dict) or set(request["job"]) != {"job_id", "nonce", "swarm_id"}
                or request["job"]["swarm_id"] != report["binding"]["ring_id"]
                or not isinstance(request.get("transaction_id"), str) or not 1 <= len(request["transaction_id"]) <= 128):
            raise ValueError("canary arithmetic, job or transaction changed")
        sha(request["job"]["nonce"])
        identity = dict(report["binding"])
        identity["config_sha256"] = digest([identity["config_sha256"], report["execution"]])
        if request.get("identity") != identity:
            raise ValueError("canary identity differs from the exact reset-bound model/runtime")
    if any(begin.get(k) != compare.get(k) for k in ("job", "transaction_id", "identity", "prefix", "execution")):
        raise ValueError("comparison is not the same strict suffix transaction")
    geometry = begin.get("parameters")
    if (not isinstance(geometry, dict) or set(geometry) != {"parent_count", "full_prefix_count"}
            or any(type(geometry[k]) is not int for k in geometry)
            or not 0 < geometry["parent_count"] < geometry["full_prefix_count"] <= report["max_context"]
            or geometry["full_prefix_count"] - geometry["parent_count"] > report["max_suffix"]
            or geometry["parent_count"] % report["parent_alignment"]
            or compare.get("prefix", {}).get("token_count") != geometry["full_prefix_count"]):
        raise ValueError("canary geometry exceeds the qualified suffix/alignment")
    acks, receipts = value["acks"], value["receipts"]
    if not isinstance(acks, list) or len(acks) != len(stages) or not isinstance(receipts, list) or len(receipts) != len(stages):
        raise ValueError("strict canary omits an assigned stage")
    try:
        from shard.receipt import verify_coverage, wire_receipt
    except ImportError:
        from receipt import verify_coverage, wire_receipt
    receipts = [wire_receipt(row) for row in receipts]
    verify_coverage(receipts, stages[-1]["hi"], expected_by_signer={row["pubkey"]: (row["lo"], row["hi"]) for row in stages},
                    expected_nonce=compare["job"]["nonce"], check_chain=True)
    by_key = {row["pubkey"]: row for row in receipts}
    for stage, raw in zip(stages, acks):
        ack = signed(raw, stage["pubkey"], b"v4-conversation-ack/1\0")
        if (ack.get("schema") != "v4-conversation-ack/1" or ack.get("stage") != stage["stage"]
                or ack.get("action") != "shadow_compare" or ack.get("ok") is not True
                or ack.get("shadow_passed") is not True
                or ack.get("job") != compare["job"] or ack.get("transaction_id") != compare["transaction_id"]
                or ack.get("prefix") != compare["prefix"] or ack.get("identity_sha256") != digest(compare["identity"])
                or ack.get("execution_sha256") != digest(compare["execution"])
                or ack.get("parameters_sha256") != digest(compare.get("parameters", {}))
                or ack.get("shadow_scope") != "per_request_full_prefill_state_and_reply_bytes"):
            raise ValueError("strict numerical comparison ACK is incomplete, failed or belongs to another transaction")
        receipt = by_key[stage["pubkey"]]
        if any(receipt.get(k) != compare["job"][k] for k in ("job_id", "nonce", "swarm_id")):
            raise ValueError("runtime receipt belongs to another canary job")
        _gpu_observation(receipt.get("runtime_observation"), stage, report)
    return geometry


def _native_case(raw, evidence, report):
    value = signed(raw, evidence["coordinator_pubkey"], CASE_DOMAIN)
    fields = {"schema", "case", "stage", "binding_sha256", "source_sha256", "execution_sha256",
              "gpu_uuid", "kernel_backend", "device", "inputs_sha256", "reference", "candidate",
              "parent_count", "full_count", "operation_sequence", "compression_ratios"}
    if (set(value) != fields or value["schema"] != CASE_SCHEMA or value["case"] not in CASES
            or type(value["stage"]) is not int or not 0 <= value["stage"] < len(evidence["stages"])
            or value["binding_sha256"] != digest(report["binding"])
            or value["source_sha256"] != report["source_sha256"]
            or value["execution_sha256"] != digest(report["execution"])
            or value["kernel_backend"] != "tilelang" or not isinstance(value["device"], str)
            or not value["device"].startswith("cuda:")
            or value["gpu_uuid"] != evidence["stages"][value["stage"]]["gpu_uuid"]
            or value["operation_sequence"] != list(OPERATIONS[value["case"]])):
        raise ValueError("native case lacks its measured operations and exact device/model/arithmetic identity")
    sha(value["inputs_sha256"])
    stage = evidence["stages"][value["stage"]]
    ratios = value["compression_ratios"]
    if (not isinstance(ratios, list) or len(ratios) != stage["hi"] - stage["lo"]
            or any(type(r) is not int or r < 0 or r > 1_048_576 for r in ratios)):
        raise ValueError("native case must describe its actual owned compression geometry")
    if (type(value["parent_count"]) is not int or type(value["full_count"]) is not int
            or not 0 < value["parent_count"] < value["full_count"] <= report["max_context"]
            or value["full_count"] - value["parent_count"] > report["max_suffix"]
            or value["parent_count"] % report["parent_alignment"]):
        raise ValueError("native case geometry is outside the declared qualification")
    if value["case"] == "compression_boundary" and any(ratios) and not any(
            value["parent_count"] // r < value["full_count"] // r for r in ratios if r):
        raise ValueError("native compression case crosses no owned compression boundary")
    for name in ("reference", "candidate"):
        if not isinstance(value[name], dict) or set(value[name]) != {"state_sha256", "logits_sha256", "tokens_sha256"}:
            raise ValueError("actual raw native state/logit/token hashes required")
        for item in value[name].values(): sha(item)
    if value["reference"] != value["candidate"]:
        raise ValueError("native candidate and original reference byte digests differ")
    return value["stage"], value["case"]


def inspect_evidence(report, *, require_complete=False):
    evidence = report.get("evidence")
    if len(canonical(evidence)) > MAX_BYTES:
        raise ValueError("incremental evidence exceeds its bounded metadata quota")
    if (not isinstance(evidence, dict) or set(evidence) != {"schema", "coordinator_pubkey", "stages", "strict_canaries", "native_cases"}
            or evidence["schema"] != EVIDENCE_SCHEMA):
        raise ValueError("complete signed incremental evidence envelope required")
    public_key(evidence["coordinator_pubkey"])
    _stage_rows(evidence["stages"], report["binding"])
    if digest(evidence) != report["evidence_sha256"]:
        raise ValueError("incremental evidence content digest differs")
    canaries, native = evidence["strict_canaries"], evidence["native_cases"]
    if not isinstance(canaries, list) or len(canaries) > 16 or not isinstance(native, list) or len(native) > 96:
        raise ValueError("incremental campaign sample count exceeds its bound")
    for canary in canaries: _strict_canary(canary, evidence, report)
    seen = set()
    for record in native:
        key = _native_case(record, evidence, report)
        if key in seen:
            raise ValueError("duplicate native case cannot add qualification coverage")
        seen.add(key)
    pending = [f"stage:{row['stage']}:{case}" for row in evidence["stages"] for case in CASES if (row["stage"], case) not in seen]
    if not canaries:
        pending.insert(0, "strict_gpu_append_canary")
    if require_complete and pending:
        raise ValueError("incremental GPU qualification pending: " + ", ".join(pending))
    return {"ready": not pending, "pending": pending, "native_cases": len(seen), "strict_canaries": len(canaries),
            "evidence_sha256": digest(evidence), "scope": "signed operator measurements; not remote computation proof"}

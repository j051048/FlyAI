"""Synthetic signed protocol fixtures test verification, never GPU arithmetic."""
from copy import deepcopy
import hashlib

import pytest

from shard.receipt import ReceiptSigner, gen_key, pub_b64
from v4_conversation_protocol import _sign, execution_binding
from v4_incremental_evidence import (EVIDENCE_SCHEMA, CASE_SCHEMA, CASE_DOMAIN, CASES, OPERATIONS,
                                     inspect_evidence, digest)
from v4_incremental_prefix import IncrementalQualification, SCHEMA


def signed_fixture(report):
    """Build explicitly simulated worker/GPU declarations with genuine signatures."""
    from v4_observability import source_inventory
    caller = gen_key()
    evidence = {"schema": EVIDENCE_SCHEMA, "coordinator_pubkey": pub_b64(caller),
                "stages": [], "strict_canaries": [], "native_cases": []}
    sources = {"fixture.py": "a" * 64}
    if digest(sources) != report["source_sha256"]:
        sources = source_inventory()
    assert digest(sources) == report["source_sha256"]
    identity = dict(report["binding"], config_sha256=digest([report["binding"]["config_sha256"], report["execution"]]))
    job = {"job_id": "explicit-synthetic-fixture", "nonce": "d" * 64, "swarm_id": report["binding"]["ring_id"]}
    prefix = {"token_count": 16, "digest": "e" * 64}
    begin = {"schema": "v4-conversation-control/1", "action": "shadow_begin", "transaction_id": "fixture-tx",
             "identity": identity, "prefix": prefix, "entries": {}, "job": job,
             "execution": report["execution"], "parameters": {"parent_count": 8, "full_prefix_count": 16}}
    compare = {**begin, "action": "shadow_compare", "parameters": {}}
    acks, receipts = [], []
    for index, (node, _) in enumerate(report["binding"]["lease_fences"]):
        key = gen_key()
        row = {"stage": index, "node_id": node, "pubkey": pub_b64(key), "lo": index, "hi": index + 1,
               "runtime_config_sha256": "c" * 64, "gpu_uuid": f"GPU-explicit-unit-fixture-{index}"}
        evidence["stages"].append(row)
        ack = {"schema": "v4-conversation-ack/1", "action": "shadow_compare", "transaction_id": "fixture-tx",
            "stage": index, "job": job, "prefix": prefix, "ok": True, "shadow_passed": True,
            "shadow_scope": "per_request_full_prefill_state_and_reply_bytes",
            "identity_sha256": digest(identity), "execution_sha256": digest(report["execution"]),
            "parameters_sha256": digest({})}
        acks.append(_sign(key, b"v4-conversation-ack/1\0", ack))
        observed = {"schema": "shard-runtime-observation/1", "node_id": node,
            "gpu_uuid": row["gpu_uuid"], "process_run_id": "explicit-synthetic-process", "runtime_config_sha256": "c" * 64,
            "source_files": sources, "source_sha256": digest(sources), "environment": {}, "environment_sha256": digest({}),
            "effective_flags": {}, "effective_flags_sha256": digest({}), "kernel_backend": "tilelang", "hadamard_backend": "torch",
            "graph_mode": "off", "wire_mode": "bf16", "transport": "explicit-synthetic-test",
            "versions": {"python": "fixture", "torch": "fixture", "cuda": "fixture", "tilelang": "fixture"}, "phase": "job_complete"}
        signer = ReceiptSigner(key, job["swarm_id"], job["job_id"], index, index + 1, nonce=job["nonce"])
        signer.observe(b"same-chain-fixture", b"same-chain-fixture")
        receipts.append(signer.finalize(runtime_observation=observed))
        hashes = {name: hashlib.sha256((name + "-explicit-unit-bytes").encode()).hexdigest()
                  for name in ("state_sha256", "logits_sha256", "tokens_sha256")}
        for case in CASES:
            record = {"schema": CASE_SCHEMA, "case": case, "stage": index,
                "binding_sha256": digest(report["binding"]), "source_sha256": report["source_sha256"],
                "execution_sha256": digest(report["execution"]), "gpu_uuid": row["gpu_uuid"], "kernel_backend": "tilelang",
                "device": "cuda:0", "inputs_sha256": "f" * 64, "reference": hashes, "candidate": hashes,
                "parent_count": 8, "full_count": 16, "operation_sequence": list(OPERATIONS[case]), "compression_ratios": [4]}
            evidence["native_cases"].append(_sign(caller, CASE_DOMAIN, record))
    evidence["strict_canaries"].append({"begin_request": _sign(caller, b"v4-conversation-control/1\0", begin),
        "compare_request": _sign(caller, b"v4-conversation-control/1\0", compare), "acks": acks, "receipts": receipts})
    return evidence


def report_fixture():
    from test_v4_conversation_protocol import numerical_identity
    result = {"schema": SCHEMA, "binding": numerical_identity(), "source_sha256": digest({"fixture.py": "a" * 64}),
              "execution": execution_binding({"max_pos": 64}), "gpu_verified": True,
              "cases": {name: True for name in CASES}, "max_context": 64, "max_suffix": 16, "parent_alignment": 4}
    result["evidence"] = signed_fixture(result)
    result["evidence_sha256"] = digest(result["evidence"])
    return result


def test_complete_signed_synthetic_evidence_verifies_but_missing_case_remains_pending():
    report = report_fixture()
    assert inspect_evidence(report, require_complete=True)["ready"]
    IncrementalQualification(report)
    report["evidence"]["native_cases"].pop()
    report["evidence_sha256"] = digest(report["evidence"])
    status = inspect_evidence(report)
    assert not status["ready"] and "stage:0:tenant_isolation" in status["pending"]
    with pytest.raises(ValueError, match="pending"):
        IncrementalQualification(report)


@pytest.mark.parametrize("changed", ["receipt", "ack", "case", "assignment", "source", "digest", "coordinator"])
def test_tampered_signed_case_ack_receipt_or_assignment_is_rejected(changed):
    report = report_fixture()
    evidence = report["evidence"]
    if changed == "receipt": evidence["strict_canaries"][0]["receipts"][0]["nonce"] = "1" * 64
    if changed == "ack": evidence["strict_canaries"][0]["acks"][0]["shadow_passed"] = False
    if changed == "case": evidence["native_cases"][0]["candidate"] = {**evidence["native_cases"][0]["candidate"], "state_sha256": "1" * 64}
    if changed == "assignment": evidence["stages"][0]["gpu_uuid"] = "another-gpu"
    if changed == "source": report["source_sha256"] = "1" * 64
    if changed == "coordinator": evidence["coordinator_pubkey"] = pub_b64(gen_key())
    if changed != "digest": report["evidence_sha256"] = digest(evidence)
    else: report["evidence_sha256"] = "1" * 64
    with pytest.raises(Exception):
        inspect_evidence(report, require_complete=True)


def test_plain_pass_booleans_without_raw_signed_evidence_never_qualify():
    report = report_fixture()
    report.pop("evidence")
    with pytest.raises(ValueError):
        IncrementalQualification(report)


def test_pending_collector_never_writes_a_loadable_qualification(tmp_path):
    from phase0.v4_incremental_qualification import collect, main
    import json
    report = report_fixture()
    evidence = deepcopy(report["evidence"])
    evidence["native_cases"] = []
    spec = {k: report[k] for k in ("binding", "source_sha256", "execution", "max_context", "max_suffix", "parent_alignment")}
    spec["schema"] = "v4-incremental-prefix-campaign/1"
    result = collect(spec, evidence)
    assert result["qualification"] is None and len(result["pending"]) == 6
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (tmp_path / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    assert main(["--spec", str(tmp_path / "spec.json"), "--evidence", str(tmp_path / "evidence.json"),
                 "--out-dir", str(tmp_path / "out")]) == 2
    assert (tmp_path / "out" / "report.json").exists()
    assert not (tmp_path / "out" / "qualification.json").exists()

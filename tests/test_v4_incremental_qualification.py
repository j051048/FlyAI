"""Policy guards and real CPU state canaries; synthetic metadata is not GPU proof."""
from copy import deepcopy

import pytest

import v4_conversation_protocol as P
from v4_incremental_prefix import IncrementalQualification, SCHEMA, digest
from test_v4_conversation_protocol import (NumericalWire, numerical_identity, numerical_prefill,
                                          strict_tiny_ring, actual_job)
from v4_conversation_cache import PrefixBinding, _buffers, _metadata, _same


def qualification(binding=None, *, source=None, execution=None):
    """Untrusted local policy fixture: never exported as hardware evidence."""
    value = {"schema": SCHEMA, "binding": binding or numerical_identity(), "source_sha256": source or digest({"fixture.py": "a" * 64}),
        "gpu_verified": True, "max_context": 64,
        "max_suffix": 16, "parent_alignment": 1,
        "execution": execution or P.execution_binding({"max_pos": 64}),
        "cases": {name: True for name in ("append", "compression_boundary", "rollback", "cancel", "reset", "tenant_isolation")}}
    from test_v4_incremental_evidence import signed_fixture
    value["evidence"] = signed_fixture(value)
    value["evidence_sha256"] = digest(value["evidence"])
    return value


@pytest.mark.parametrize("key,value", [("spec", True), ("dspark", True), ("pipelined", True),
    ("temp", .1), ("seed", 2), ("max_pos", 63)])
def test_incremental_canary_qualification_never_crosses_reset_execution(key, value):
    report = qualification()
    policy = IncrementalQualification(report)
    policy.canary(True)
    changed = {**report["execution"], key: value}
    assert not policy.eligible(report["binding"], source_sha256=report["source_sha256"],
                               execution=changed, parent=8, full=16)


def test_incremental_periodic_canary_and_failure_remain_sticky():
    policy = IncrementalQualification(qualification(), shadow_every=3)
    assert not policy.reuse()
    policy.canary(True)
    assert policy.reuse() and policy.reuse() and not policy.reuse()
    policy.canary(False)
    assert not policy.reuse()
    policy.canary(True)
    assert not policy.reuse() and policy.poisoned


@pytest.mark.parametrize("change", ["source", "tenant", "parent", "full", "suffix", "alignment"])
def test_incremental_identity_and_geometry_cannot_be_broadened(change):
    report = qualification()
    report["parent_alignment"] = 4
    policy = IncrementalQualification(report)
    source, binding, parent, full = report["source_sha256"], deepcopy(report["binding"]), 8, 16
    if change == "source": source = "c" * 64
    if change == "tenant": binding["tenant"] = "other"
    if change == "parent": parent = 0
    if change == "full": full = 65
    if change == "suffix": full = 30
    if change == "alignment": parent = 7
    assert not policy.eligible(binding, source_sha256=source, execution=report["execution"], parent=parent, full=full)


def test_real_cpu_append_canary_uses_actual_state_and_never_skips_after_failed_comparison():
    from v4_observability import source_inventory
    wire, reference = NumericalWire(), NumericalWire()
    report = qualification(source=digest(source_inventory()))
    owner = P.ConversationCoordinator(enabled=True, extended_shadow=True, incremental_qualification=report,
                                      incremental_shadow_every=2)
    numerical_prefill(wire, owner, list(range(5)), 1, maximum=59)
    reply, first = numerical_prefill(wire, owner, list(range(13)), 2, maximum=51)
    assert owner.incremental.checked == 1
    assert first.record["reference_shadow"] is True and not first.record["hit"]
    passed = first.record["all_stage_shadow_passed"]
    assert owner.incremental.poisoned is (not passed)
    observed, following = numerical_prefill(wire, owner, list(range(21)), 3, maximum=43)
    expected, _ = numerical_prefill(reference, P.ConversationCoordinator(), list(range(21)), 3, maximum=43)
    assert observed["token"] == expected["token"]
    assert _same(_buffers(wire.stage, None), _buffers(reference.stage, None))
    assert _same(_metadata(wire.stage, None), _metadata(reference.stage, None))
    assert following.record["hit"] is passed
    assert following.record["reference_shadow"] is (not passed)
    assert wire.normal_prefills == (2 if passed else 3)


def test_real_strict_ring_qualification_canary_keeps_signed_reference_on_mismatch(strict_tiny_ring):
    from v4_observability import source_inventory
    ring = strict_tiny_ring
    report = qualification(ring.identity, source=digest(source_inventory()),
                           execution=P.execution_binding({"max_pos": 64}))
    ring.owner = P.ConversationCoordinator(enabled=True, extended_shadow=True, incremental_qualification=report)
    actual_job(ring, 21, max_new=64 - len(ring.prompt))
    ring.prompt += list(range(10, 18))
    ring.prefix = PrefixBinding(len(ring.prompt), "6" * 64).to_dict()
    result, request = actual_job(ring, 22, max_new=64 - len(ring.prompt))
    assert result["receipts_ok"] is True and not request.record["hit"]
    assert ring.owner.incremental.checked == 1
    assert ring.owner.incremental.poisoned is (not request.record["all_stage_shadow_passed"])
    assert all("conversation_restore" not in row for row in result["receipts"])


def test_qualified_suffix_eos_runs_original_prefill_instead_of_empty_receipt():
    from v4_observability import source_inventory
    wire, baseline = NumericalWire(), NumericalWire()
    report = qualification(source=digest(source_inventory()))
    owner = P.ConversationCoordinator(enabled=True, extended_shadow=True, incremental_qualification=report)
    numerical_prefill(wire, owner, list(range(5)), 31, maximum=59)
    # Synthetic prior-policy state targets the EOS fallback branch; the actual
    # native suffix/model prediction below is real CPU computation, not a GPU
    # qualification or an assertion that arbitrary prefixes are equivalent.
    owner.incremental.canary(True)
    expected, _ = numerical_prefill(baseline, P.ConversationCoordinator(), list(range(13)), 32, maximum=51)
    observed, request = numerical_prefill(wire, owner, list(range(13)), 32, maximum=51,
                                         eos_ids=(expected["token"],))
    assert observed["token"] == expected["token"]
    assert request.record["hit"] is False
    assert request.restore_binding is None
    assert any(message["request"]["action"] == "shadow_abort" for message in wire.control_messages)
    assert wire.normal_prefills == 2 and wire.stage._conversation_shadow is None

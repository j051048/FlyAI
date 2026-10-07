"""Tests for Fraud-Proof Inference & Web3 Hardening (Stages 1-4).

Covers:
1. Stage activation commitments & cryptographic hash chain verification.
2. Run receipt integration with stage-level commitment envelope.
3. Interactive single-stage fraud proof adjudication (catching cheating nodes, protecting honest nodes).
4. Same-microarchitecture enforcement (preventing false slashes across GPU generations).
5. Anti-Sybil & Anti-Hairpin topology ring constraints (no adjacent stages on same physical host).
6. Staked boundary pinning integration in plan_ring.
"""
import argparse
import copy
import hashlib
import json
import pytest

from phase0.activation_proof import (
    hash_activation,
    compute_step_commitment,
    StageCommitmentTracker,
    verify_commitment_chain,
)
from phase0.fraud_proof import (
    create_challenge,
    submit_dispute_snapshot,
    adjudicate_challenge,
    FraudProofError,
)
from phase0.proof_receipt import build, verify
from shard.plan import plan_ring
from shard.topology import select_ring, _is_adjacent_same_host


# ---------------------------------------------------------------------------
# Stage 1: Activation commitments & receipt verification
# ---------------------------------------------------------------------------

def test_activation_commitment_chain_and_receipt_verify(tmp_path):
    """Verify hash_activation, StageCommitmentTracker, and proof_receipt binding."""
    # 1. Deterministic hashing
    data1 = b"tensor_activation_step_0"
    data2 = b"tensor_activation_step_1"
    h1 = hash_activation(data1)
    h2 = hash_activation(data2)
    assert len(h1) == 64
    assert h1 != h2
    assert hash_activation(data1) == h1  # Pure & deterministic

    # 2. Stage tracker & hash chain
    tracker0 = StageCommitmentTracker(stage_idx=0)
    comm0 = tracker0.record_step(step=0, in_hash="in0_hash", out_hash="out0_hash")
    comm1 = tracker0.record_step(step=1, in_hash="in1_hash", out_hash="out1_hash")
    assert comm0 != comm1
    records0 = tracker0.history()
    assert len(records0) == 2

    # Verify chain integrity
    ok, err = verify_commitment_chain(records0)
    assert ok is True
    assert err is None

    # Corrupted chain detection
    corrupted_records = copy.deepcopy(records0)
    corrupted_records[1]["out_hash"] = "tampered_hash"
    ok_bad, err_bad = verify_commitment_chain(corrupted_records)
    assert ok_bad is False
    assert "computed" in err_bad

    # 3. Receipt envelope binding
    tracker1 = StageCommitmentTracker(stage_idx=1)
    tracker1.record_step(step=0, in_hash="out0_hash", out_hash="final_hash")
    records1 = tracker1.history()

    stage_commitments = [
        {"stage_idx": 0, "final_commitment": tracker0.current_commitment, "chain": records0},
        {"stage_idx": 1, "final_commitment": tracker1.current_commitment, "chain": records1},
    ]

    nodes = [
        {"role": "coordinator", "public_ip": "1.1.1.1", "geo": "US", "gpu_uuid": "GPU-0", "gpu_name": "4090"},
        {"role": "stage0", "public_ip": "2.2.2.2", "geo": "DE", "gpu_uuid": "GPU-1", "gpu_name": "5090"},
    ]
    edges = [{"from": "coordinator", "to": "stage0", "rtt_ms": 25.0}]
    run_data = {
        "prompt": "Who proves the work?",
        "output_text": "The code proves the work.",
        "output_token_ids": [101, 202, 303],
        "tok_s_warm": 38.5,
        "stage_commitments": stage_commitments,
    }

    nodes_file = tmp_path / "nodes.json"
    edges_file = tmp_path / "edges.json"
    run_file = tmp_path / "run.json"
    receipt_out = tmp_path / "receipt.json"

    json.dump(nodes, open(nodes_file, "w"))
    json.dump(edges, open(edges_file, "w"))
    json.dump(run_data, open(run_file, "w"))

    args = argparse.Namespace(
        nodes=str(nodes_file),
        edges=str(edges_file),
        run=str(run_file),
        model="gpt-oss-120b",
        quant="mxfp4",
        out=str(receipt_out),
        run_id="test_web3_run_001",
        utc="",
        assignments=None,
        stage_receipts=None,
        stage_commitments=None,
    )
    build(args)

    receipt = json.load(open(receipt_out))
    assert "stage_commitments" in receipt["envelope"]
    assert len(receipt["envelope"]["stage_commitments"]) == 2

    # Verification passes
    ref_file = tmp_path / "ref.json"
    json.dump([101, 202, 303], open(ref_file, "w"))

    v_args = argparse.Namespace(receipt=str(receipt_out), ref_tokens=str(ref_file))
    with pytest.raises(SystemExit) as exc:
        verify(v_args)
    assert exc.value.code == 0

    # Tampering stage commitment inside receipt causes verify to fail
    receipt_tampered = copy.deepcopy(receipt)
    receipt_tampered["envelope"]["stage_commitments"][0]["chain"][0]["out_hash"] = "tampered_hex"
    tampered_file = tmp_path / "receipt_tampered.json"
    json.dump(receipt_tampered, open(tampered_file, "w"))

    v_bad = argparse.Namespace(receipt=str(tampered_file), ref_tokens=str(ref_file))
    with pytest.raises(SystemExit) as exc_bad:
        verify(v_bad)
    assert exc_bad.value.code == 1


# ---------------------------------------------------------------------------
# Stage 2: Fraud proof adjudication & same-arch enforcement
# ---------------------------------------------------------------------------

def test_fraud_proof_catches_cheating_node_and_slashes():
    """Test that a malicious node presenting falsified activations is caught and slashed."""
    input_tensor = [1.0, 2.0, 3.0, 4.0]
    expected_output = [2.0, 4.0, 6.0, 8.0]
    declared_fake_output = [0.0, 0.0, 0.0, 0.0]

    in_hash = hash_activation(input_tensor)
    fake_out_hash = hash_activation(declared_fake_output)
    commitment = compute_step_commitment(stage_idx=1, step=5, in_hash=in_hash, out_hash=fake_out_hash)

    challenge = create_challenge(
        challenge_id="ch_001",
        run_id="run_fraud_01",
        stage_idx=1,
        step=5,
        challenger_id="node_challenger",
        defender_id="node_cheater",
        challenger_deposit=1000,
        defender_deposit=5000,
        declared_commitment=commitment,
        declared_in_hash=in_hash,
        declared_out_hash=fake_out_hash,
        defender_arch="sm120",
    )

    # Defender submits disputed raw snapshot
    submitted = submit_dispute_snapshot(challenge, input_tensor, declared_fake_output)
    assert submitted is True
    assert challenge.status == "SNAPSHOT_SUBMITTED"

    # Stage replay computation (real math: x * 2)
    def replay_stage(inp):
        return [x * 2.0 for x in inp]

    verdict = adjudicate_challenge(
        challenge,
        replay_stage_fn=replay_stage,
        validator_arch="sm120",
    )

    assert verdict["verdict"] == "SLASH_DEFENDER"
    assert verdict["slashed_node"] == "node_cheater"
    assert verdict["slashed_amount"] == 5000
    assert verdict["winner"] == "node_challenger"
    assert challenge.status == "RESOLVED_DEFENDER_SLASHED"


def test_fraud_proof_catches_unsubmitted_snapshot():
    """Test that a defender who abandons the dispute without snapshot is slashed immediately."""
    challenge = create_challenge(
        challenge_id="ch_002",
        run_id="run_fraud_02",
        stage_idx=2,
        step=10,
        challenger_id="node_challenger",
        defender_id="node_offline",
        challenger_deposit=1000,
        defender_deposit=5000,
        declared_commitment="dummy_comm",
        declared_in_hash="dummy_in",
        declared_out_hash="dummy_out",
        defender_arch="sm120",
    )
    assert challenge.status == "OPEN"

    # Adjudication without snapshot submission
    verdict = adjudicate_challenge(
        challenge,
        replay_stage_fn=lambda x: x,
        validator_arch="sm120",
    )
    assert verdict["verdict"] == "SLASH_DEFENDER"
    assert verdict["slashed_node"] == "node_offline"
    assert verdict["slashed_amount"] == 5000
    assert "failed to submit dispute snapshot" in verdict["reason"]


def test_fraud_proof_protects_honest_node_from_malicious_challenge():
    """Test that an honest node is defended and the malicious challenger loses deposit."""
    input_tensor = [10.0, 20.0, 30.0]
    honest_output = [11.0, 21.0, 31.0]

    in_hash = hash_activation(input_tensor)
    out_hash = hash_activation(honest_output)
    commitment = compute_step_commitment(stage_idx=0, step=1, in_hash=in_hash, out_hash=out_hash)

    challenge = create_challenge(
        challenge_id="ch_003",
        run_id="run_honest_01",
        stage_idx=0,
        step=1,
        challenger_id="bad_challenger",
        defender_id="honest_worker",
        challenger_deposit=2000,
        defender_deposit=5000,
        declared_commitment=commitment,
        declared_in_hash=in_hash,
        declared_out_hash=out_hash,
        defender_arch="sm120",
    )

    submit_dispute_snapshot(challenge, input_tensor, honest_output)

    def replay_stage(inp):
        return [x + 1.0 for x in inp]

    verdict = adjudicate_challenge(
        challenge,
        replay_stage_fn=replay_stage,
        validator_arch="sm120",
    )

    assert verdict["verdict"] == "CHALLENGE_FAILED"
    assert verdict["slashed_node"] == "bad_challenger"
    assert verdict["slashed_amount"] == 2000
    assert verdict["winner"] == "honest_worker"
    assert challenge.status == "RESOLVED_DEFENDER_HONEST"


def test_same_arch_enforcement_prevents_cross_gpu_misjudgment():
    """Test that cross-architecture adjudication is strictly blocked."""
    challenge = create_challenge(
        challenge_id="ch_004",
        run_id="run_arch_01",
        stage_idx=1,
        step=2,
        challenger_id="challenger",
        defender_id="defender",
        challenger_deposit=1000,
        defender_deposit=5000,
        declared_commitment="comm",
        declared_in_hash=hash_activation([1.0]),
        declared_out_hash=hash_activation([2.0]),
        defender_arch="sm120",  # Blackwell
    )
    submit_dispute_snapshot(challenge, [1.0], [2.0])

    # Validator runs on Ada Lovelace sm89 -> must reject adjudication
    with pytest.raises(FraudProofError, match="architectural mismatch"):
        adjudicate_challenge(
            challenge,
            replay_stage_fn=lambda x: [x[0] * 2.0],
            validator_arch="sm89",
        )


# ---------------------------------------------------------------------------
# Stage 3: Anti-Sybil & Anti-Hairpin topology ring constraints
# ---------------------------------------------------------------------------

def test_anti_hairpin_anti_sybil_helper():
    """Unit test the _is_adjacent_same_host helper function."""
    host_id = {0: "host_A", 1: "host_A", 2: "host_B", 3: "host_C"}
    
    # 0 and 1 are adjacent -> True (invalid)
    assert _is_adjacent_same_host([0, 1, 2, 3], host_id) is True
    # 0 and 1 are head and tail (adjacent in ring) -> True (invalid)
    assert _is_adjacent_same_host([0, 2, 3, 1], host_id) is True
    # 0 and 1 separated by other hosts -> False (valid)
    assert _is_adjacent_same_host([0, 2, 1, 3], host_id) is False


def test_select_ring_anti_hairpin_rejects_adjacent_same_host():
    """Verify that select_ring never places two nodes from the same host adjacent in ring."""
    # 4 nodes: node 0 and 1 share host "box_A", 2 and 3 have distinct hosts
    n = 4
    latencies = [
        [0.0, 1.0, 20.0, 20.0],
        [1.0, 0.0, 20.0, 20.0],
        [20.0, 20.0, 0.0, 20.0],
        [20.0, 20.0, 20.0, 0.0],
    ]
    c_out = [20.0] * n
    c_in = [20.0] * n
    free_vram = {i: 32000.0 for i in range(n)}
    layer_ms = {i: 10.0 for i in range(n)}
    subnet = {i: f"sub_{i}" for i in range(n)}
    host_id = {0: "box_A", 1: "box_A", 2: "box_B", 3: "box_C"}

    # Model requiring 3 nodes
    ring = select_ring(
        range(n),
        latencies,
        c_out,
        c_in,
        free_vram_mb=free_vram,
        layer_ms=layer_ms,
        subnet=subnet,
        host_id=host_id,
        isolation="adjacent_host",
        n_layers=30,
        layer_vram_mb=800.0,
        kv_mb_per_layer=50.0,
        slack=1,
    )

    assert ring is not None
    order = ring["order"]
    assert _is_adjacent_same_host(order, host_id) is False, f"Ring {order} violated host isolation!"


def test_staked_nodes_automatically_admitted_to_boundary_pinning():
    """Verify plan_ring automatically admits staked Web3 nodes into trusted boundary pinning."""
    # 4 nodes: node 0 and node 3 are staked (Web3 deposit), node 1 and 2 are permissionless
    nodes = [
        {"id": "n0", "public_ip": "1.1.1.1", "subnet": "s0", "gpu_name": "RTX 4090", "free_vram_mb": 24000.0, "staked": True},
        {"id": "n1", "public_ip": "2.2.2.2", "subnet": "s1", "gpu_name": "RTX 3090", "free_vram_mb": 24000.0, "staked": False},
        {"id": "n2", "public_ip": "3.3.3.3", "subnet": "s2", "gpu_name": "RTX 3090", "free_vram_mb": 24000.0, "staked": False},
        {"id": "n3", "public_ip": "4.4.4.4", "subnet": "s3", "gpu_name": "RTX 4090", "free_vram_mb": 24000.0, "staked": True},
    ]
    # Synthetic flat latency matrix
    rtt = [[0.0 if i == j else 20.0 for j in range(len(nodes))] for i in range(len(nodes))]
    toy_model = {
        "n_layers": 20,
        "layer_vram_mb": 800.0,
        "kv_mb_per_layer": 50.0,
        "cap_layers": 10,
        "reserve_mb": 1000.0,
        "head_reserve_mb": 1000.0,
        "tail_reserve_mb": 1000.0,
    }

    # Plan with privacy boundary pinning (requires trusted/staked ends)
    plan = plan_ring(
        nodes,
        rtt,
        model=toy_model,
        privacy={"boundary_in": 1, "boundary_out": 1},
    )

    assert plan is not None
    order = plan["order"]
    head_node = order[0]
    tail_node = order[-1]

    # Staked nodes must hold the sensitive boundary stages (head and tail)
    node_map = {n["id"]: n for n in nodes}
    assert node_map[head_node]["staked"] is True
    assert node_map[tail_node]["staked"] is True

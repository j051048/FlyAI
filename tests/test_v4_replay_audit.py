from dataclasses import replace
import hashlib
import pytest
import torch

from test_v4_conversation_cache import stage
import v4_ref_cpu as R
from v4_replay_audit import NativeReplayAuditor, ReplayChunk, tensor_bytes
from shard.node_quality import NodeQuality
from shard.receipt import ReceiptSigner, gen_key, pub_b64


@pytest.fixture
def setup():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
                      dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=64)
    model = args, R.build_oracle(args, 7)
    original, replica = stage(model), stage(model)
    node, validator = gen_key(), gen_key()
    auditor = NativeReplayAuditor(replica, validator, cohort_id="a" * 64,
        numeric_contract="v4-native-tokenwise-greedy/1", fixture_reference=True)
    quality = NodeQuality([pub_b64(validator)], required_evidence="cpu_reference")
    chunks = []
    signer = ReceiptSigner(node, "ring", "job", 0, 4, nonce="fresh-nonce")
    with torch.no_grad():
        for pos in range(17):
            ids = torch.tensor([[pos + 1]])
            hidden = original.embed(ids)
            output = original.forward(hidden, ids, pos)
            encoded = tensor_bytes(output) + tensor_bytes(ids)
            chunk = ReplayChunk(pos, hidden.clone(), ids.clone(), encoded)
            signer.observe(chunk.input_bytes(), chunk.output_bytes)
            chunks.append(chunk)
    receipt = signer.finalize()
    def issue(receipt=receipt, chunks=chunks):
        return quality.issue("node", "a" * 64, receipt, expected_pubkey=pub_b64(node),
            numeric_contract="v4-native-tokenwise-greedy/1", input_sha256=auditor.input_identity(chunks),
            execution_identity_sha256=auditor.execution_identity_sha256)
    yield auditor, quality, chunks, receipt, issue, node
    torch.set_num_threads(old)


def test_auditor_reexecutes_original_compressed_native_stage_before_quality_update(setup):
    auditor, quality, chunks, receipt, issue, _ = setup
    result = auditor.run(issue(), receipt, chunks)
    assert result["verdict"] == "match" and result["elapsed_ms"] > 0
    assert quality.ingest(result)["matched"] == 1
    assert auditor.stage._pos == 0


def test_real_replay_detects_wrong_signed_output(setup):
    auditor, quality, chunks, receipt, issue, node = setup
    changed = bytearray(chunks[-1].output_bytes); changed[0] ^= 1
    bad = [*chunks[:-1], replace(chunks[-1], output_bytes=bytes(changed))]
    signer = ReceiptSigner(node, "ring", "different-job", 0, 4, nonce="different-nonce")
    for chunk in bad: signer.observe(chunk.input_bytes(), chunk.output_bytes)
    wrong = signer.finalize()
    result = auditor.run(issue(wrong, bad), wrong, bad)
    assert result["verdict"] == "mismatch"
    assert quality.ingest(result)["mismatched"] == 1


def test_unbound_transcript_and_out_of_order_positions_rejected_before_compute(setup):
    auditor, _, chunks, receipt, issue, _ = setup
    case = issue()
    changed = bytearray(chunks[-1].output_bytes); changed[0] ^= 1
    bad = [*chunks[:-1], replace(chunks[-1], output_bytes=bytes(changed))]
    with pytest.raises(ValueError, match="signed activation"):
        auditor.run(case, receipt, bad)
    with pytest.raises(ValueError, match="positions"):
        auditor.run(case, receipt, [*chunks[:-1], replace(chunks[-1], position=30)])
    assert auditor.stage._pos == 0


def test_weight_mutation_and_expired_case_cannot_be_blessed(setup):
    auditor, _, chunks, receipt, issue, _ = setup
    case = issue()
    with torch.no_grad():
        next(auditor.stage.layers.parameters()).add_(1)
    with pytest.raises(ValueError, match="changed"):
        auditor.run(case, receipt, chunks)
    expired = {**case, "expires_at": 0}
    with pytest.raises(ValueError, match="pinned execution"):
        auditor.run(expired, receipt, chunks)


def test_production_auditor_refuses_unverified_weights(setup):
    auditor, _, _, _, _, _ = setup
    with pytest.raises(ValueError, match="hash-verified"):
        NativeReplayAuditor(auditor.stage, gen_key(), cohort_id="a" * 64, numeric_contract="native/1")


def test_output_layout_budget_and_lossy_wire_are_rejected_before_replay(setup, monkeypatch):
    import v4_pipe
    auditor, _, chunks, receipt, issue, _ = setup
    with pytest.raises(ValueError, match="output size"):
        auditor.input_identity([replace(chunks[0], output_bytes=b"too-short")])
    case = issue()
    monkeypatch.setattr(v4_pipe, "V4_FP8_WIRE", True)
    with pytest.raises(ValueError, match="raw BF16"):
        auditor.run(case, receipt, chunks)


def test_head_embedding_is_recomputed_from_original_token_ids(setup):
    auditor, _, chunks, _, issue, node = setup
    hidden = chunks[0].hidden.clone(); hidden.add_(1)
    wrong = [replace(chunks[0], hidden=hidden), *chunks[1:]]
    signer = ReceiptSigner(node, "ring", "wrong-head", 0, 4, nonce="head-input")
    for chunk in wrong: signer.observe(chunk.input_bytes(), chunk.output_bytes)
    receipt = signer.finalize()
    case = issue(receipt, wrong)
    with pytest.raises(ValueError, match="head embedding"):
        auditor.run(case, receipt, wrong)


def test_fixture_replica_cannot_be_promoted_to_production_quality(setup):
    auditor, _, _, _, _, _ = setup
    with pytest.raises(ValueError, match="verified model cohort"):
        auditor.validate_cohort(object())


def test_production_replay_refuses_non_bf16_stage_dtype(setup):
    auditor, _, _, _, _, _ = setup
    auditor.stage.dtype = torch.float32
    with pytest.raises(ValueError, match="BF16 stage"):
        NativeReplayAuditor(auditor.stage, gen_key(), cohort_id="a" * 64, numeric_contract="native/1")


def test_actual_verified_catalogue_must_match_cohort_before_audit_issue(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    from shard import weight_artifacts as artifacts
    from shard.offers import ModelCohort
    from test_v4_benchmark import logical_checkpoint
    # Real full payload verification of small identity-only files; no fake GPU
    # model is executed or admitted by this test.
    full, _, _ = logical_checkpoint(tmp_path)
    proof = artifacts.verify_weight_pack(full, verify_files=True)
    original = ModelCohort.from_dict(artifacts.catalog_model_cohort(proof["catalog"]))
    auditor = NativeReplayAuditor.__new__(NativeReplayAuditor)
    auditor.fixture_reference = False
    auditor.stage = SimpleNamespace(_artifact_verification=proof, hi=43)
    auditor.cohort_id, auditor.numeric_contract = original.cohort_id, original.numeric_contract
    auditor.validate_cohort(original)
    wrong = replace(original, config_sha256="f" * 64)
    auditor.cohort_id = wrong.cohort_id  # Re-labelling a wrong replica is insufficient.
    with pytest.raises(ValueError, match="artifact/configuration"):
        auditor.validate_cohort(wrong)

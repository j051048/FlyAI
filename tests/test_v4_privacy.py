"""Opt-in sealed token protocol: exact trusted IDs, opaque middle, attack cases."""
import copy
import hashlib
import json
import os

import pytest

from v4_privacy import (TokenPrivacy, TokenPrivacyError, validate_envelope, wire_token_bytes,
                        wire_payload_bytes, generate_key_file, read_key_file)

SECRET = bytes(range(32))
BINDING = {"job_nonce": "ab" * 32, "epoch": 4, "start_pos": 13,
           "swarm_id": "fixture-ring", "job_id": "request", "model_id": "V4"}


def test_opaque_forwarding_and_trusted_hash_tail_ids():
    trusted = TokenPrivacy(SECRET)
    middle = TokenPrivacy(key_id=trusted.key_id)
    ids = [[41, 999, 0], [7, 10, 11]]
    envelope = trusted.seal(ids, BINDING)
    forwarded = copy.deepcopy(envelope)
    assert "ids" not in envelope
    assert trusted.descriptor()["activations"] == "visible"
    assert trusted.open(forwarded, BINDING) == ids
    assert middle.local_ids(forwarded, BINDING, lo=3, n_hash_layers=3,
                            is_tail=False, shape=(2, 3)) == [[0] * 3 for _ in range(2)]
    for lo, tail in ((0, False), (1, False), (2, False), (40, True)):
        assert trusted.local_ids(envelope, BINDING, lo=lo, n_hash_layers=3,
                                 is_tail=tail, shape=(2, 3)) == ids
        with pytest.raises(TokenPrivacyError, match="no trusted"):
            middle.local_ids(envelope, BINDING, lo=lo, n_hash_layers=3,
                              is_tail=tail, shape=(2, 3))
    assert wire_token_bytes(envelope) == wire_token_bytes(forwarded)
    for scales in (b"", b"fp8-scale-bytes"):
        outgoing = wire_payload_bytes(b"hidden-wire-bytes", envelope, scales)
        incoming = wire_payload_bytes(b"hidden-wire-bytes", forwarded, scales)
        assert hashlib.sha256(outgoing).digest() == hashlib.sha256(incoming).digest()


@pytest.mark.parametrize("field,value", [("job_nonce", "cd" * 32), ("epoch", 5),
                                         ("start_pos", 14), ("job_id", "other")])
def test_replay_binding_rejected(field, value):
    codec = TokenPrivacy(SECRET)
    envelope = codec.seal([[1]], BINDING)
    replay_binding = {**BINDING, field: value}
    with pytest.raises(TokenPrivacyError, match="binding mismatch"):
        codec.open(envelope, replay_binding)
    forged = copy.deepcopy(envelope)
    forged["binding"] = replay_binding
    with pytest.raises(TokenPrivacyError, match="authentication failed"):
        codec.open(forged, replay_binding)


@pytest.mark.parametrize("field", ["ciphertext", "nonce"])
def test_ciphertext_and_nonce_authentication(field):
    codec = TokenPrivacy(SECRET)
    envelope = codec.seal([[21, 42]], BINDING)
    tampered = copy.deepcopy(envelope)
    original = tampered[field]
    tampered[field] = ("0" if original[0] != "0" else "1") + original[1:]
    with pytest.raises(TokenPrivacyError, match="authentication failed"):
        codec.open(tampered, BINDING)
    assert wire_token_bytes(envelope) != wire_token_bytes(tampered)


def test_shape_key_version_and_mixed_mode_fail_closed():
    codec = TokenPrivacy(SECRET)
    envelope = codec.seal([[21, 42]], BINDING)
    with pytest.raises(TokenPrivacyError, match="shape mismatch"):
        codec.local_ids(envelope, BINDING, lo=3, n_hash_layers=3, is_tail=False, shape=(2, 1))
    with pytest.raises(TokenPrivacyError, match="key_id mismatch"):
        TokenPrivacy(b"x" * 32).open(envelope, BINDING)
    with pytest.raises(TokenPrivacyError, match="unsupported"):
        validate_envelope({**envelope, "schema": "v4-token-privacy/999"})
    for peer in (None, {}, {**codec.descriptor(), "key_id": "00" * 32}):
        with pytest.raises(TokenPrivacyError, match="mixed legacy"):
            codec.verify_descriptor(peer)
    codec.verify_descriptor(codec.descriptor())
    with pytest.raises(TokenPrivacyError, match="does not match"):
        TokenPrivacy(SECRET, key_id="00" * 32)


@pytest.mark.parametrize("ids", [[[True]], [[1.0]], [[-1]], [[2 ** 63]], [[1], [2, 3]], [], [1, 2]])
def test_invalid_ids_do_not_coerce(ids):
    with pytest.raises(TokenPrivacyError):
        TokenPrivacy(SECRET).seal(ids, BINDING)


def test_once_sealed_forward_unchanged_and_domain_lengths():
    codec = TokenPrivacy(SECRET)
    first, second = codec.seal([[7]], BINDING), codec.seal([[7]], BINDING)
    assert first["nonce"] != second["nonce"]
    assert first["ciphertext"] != second["ciphertext"]
    assert wire_payload_bytes(b"ab", first, b"c") != wire_payload_bytes(b"a", first, b"bc")
    with pytest.raises(TokenPrivacyError):
        validate_envelope({**first, "ids": [[7]]})


def test_next_token_hint_is_encrypted_and_authenticated_with_ids():
    codec = TokenPrivacy(SECRET)
    envelope = codec.seal([[7]], BINDING, next_token=129279)
    assert "dnxt" not in envelope and "next_token" not in envelope
    assert envelope["has_next_token"] is True
    assert codec.open(envelope, BINDING) == [[7]]
    assert codec.open_next_token(envelope, BINDING) == 129279
    assert codec.open_payload(envelope, BINDING) == {"ids": [[7]], "next_token": 129279}
    without = codec.seal([[7]], BINDING)
    assert codec.open_next_token(without, BINDING) is None
    with pytest.raises(TokenPrivacyError):
        codec.open({**envelope, "has_next_token": False}, BINDING)
    with pytest.raises(TokenPrivacyError):
        codec.seal([[7]], BINDING, next_token=True)


def test_key_file_generation_is_exclusive_and_public_descriptor(tmp_path):
    path = tmp_path / "token.key"
    descriptor = generate_key_file(path)
    secret = read_key_file(path)
    assert len(secret) == 32
    assert descriptor == TokenPrivacy(secret).descriptor()
    assert secret.hex() not in json.dumps(descriptor)
    with pytest.raises(FileExistsError):
        generate_key_file(path)
    assert read_key_file(path) == secret
    if os.name != "nt":
        assert path.stat().st_mode & 0o077 == 0
        path.chmod(0o644)
        with pytest.raises(TokenPrivacyError, match="0600"):
            read_key_file(path)


def test_score_routed_middle_zero_ids_preserve_real_v4_ring_numerics():
    torch = pytest.importorskip("torch")
    import v4_ref_cpu as REFCPU
    from test_v4_stage import stage_from_oracle
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        args = REFCPU.cpu_args()
        oracle = REFCPU.build_oracle(args, 7)
        # Hash layers split across two trusted stages; only deep middle is keyless.
        splits = [(0, 1), (1, 3), (3, 6), (6, args.n_layers)]
        plain = [stage_from_oracle(oracle, args, lo, hi) for lo, hi in splits]
        sealed = [stage_from_oracle(oracle, args, lo, hi) for lo, hi in splits]
        trusted = TokenPrivacy(SECRET)
        opaque = TokenPrivacy(key_id=trusted.key_id)
        ids = torch.randint(0, args.vocab_size, (1, 13), generator=torch.Generator().manual_seed(2))
        position = 0
        for _ in range(4):
            binding = {**BINDING, "epoch": 0, "start_pos": position}
            envelope = trusted.seal(ids, binding)
            h_plain, h_sealed = plain[0].embed(ids), sealed[0].embed(ids)
            for i, ((lo, _hi), st_plain, st_sealed) in enumerate(zip(splits, plain, sealed)):
                codec = opaque if i == 2 else trusted
                local = codec.local_ids(envelope, binding, lo=lo, n_hash_layers=args.n_hash_layers,
                                        is_tail=i == 3, shape=tuple(ids.shape))
                local = torch.tensor(local, dtype=torch.int64)
                h_plain = st_plain.forward(h_plain, ids, position)
                h_sealed = st_sealed.forward(h_sealed, local, position)
                assert torch.equal(h_plain, h_sealed), f"stage {i} changed model numerics"
            plain_logits = plain[-1].logits_all(h_plain, full_logits=False)
            sealed_logits = sealed[-1].logits_all(h_sealed, full_logits=False)
            assert torch.equal(plain_logits, sealed_logits)
            position += ids.shape[1]
            ids = plain_logits.argmax(dim=-1).unsqueeze(1)
    finally:
        torch.set_num_threads(old_threads)

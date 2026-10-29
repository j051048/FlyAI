"""SHA-256 byte compatibility; GPU tests never accept a CPU fallback."""
import hashlib
import json

import pytest
import torch

import v4_device_digest as D
from shard.receipt import ActualByteDigest, ReceiptSigner, gen_key, verify_receipt


@pytest.mark.parametrize("length", [0, 1, 55, 56, 63, 64, 65, 127, 128, 129, 4097])
def test_reference_padding_boundaries_equal_standard_sha256(length):
    raw = bytes(index % 251 for index in range(length))
    value = torch.tensor(list(raw), dtype=torch.uint8)
    assert D.tensor_digest([value], max_bytes=max(1, length), reference=True) == hashlib.sha256(raw).digest()


def test_multi_tensor_headers_dtype_and_noncontiguous_bytes():
    value = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6).t()
    ids = torch.tensor([1, 7, 9], dtype=torch.int64)
    header = b'activation-v1\x00'
    raw = header + value.contiguous().view(torch.uint8).numpy().tobytes() + ids.view(torch.uint8).numpy().tobytes()
    assert D.tensor_digest([header, value, ids], reference=True) == hashlib.sha256(raw).digest()
    with pytest.raises(ValueError, match="bound"):
        D.tensor_digest([header, value, ids], max_bytes=len(raw) - 1, reference=True)


def test_receipt_prehash_is_identical_to_real_byte_observation():
    key = gen_key()
    direct = ReceiptSigner(key, "test-swarm", "test-job", 3, 7, nonce="frozen")
    hashed = ReceiptSigner(key, "test-swarm", "test-job", 3, 7, nonce="frozen")
    for index in range(7):
        inp = bytes([index]) * (55 + index)
        out = bytes([index + 1]) * (64 + index)
        direct.observe(inp, out)
        hashed.observe(ActualByteDigest(D.tensor_digest([inp], reference=True)),
                       ActualByteDigest(D.tensor_digest([out], reference=True)))
    receipt = hashed.finalize()
    assert receipt == direct.finalize()
    verify_receipt(receipt)
    # A wire dict is not the process-local digest capability.
    claimed = {"digest": "00" * 32}
    with pytest.raises((TypeError, ValueError)):
        hashed.observe(claimed, b"out")
    assert json.loads(json.dumps(receipt)) == receipt


def test_digest_rejects_claimed_hash_and_wrong_reference_inputs():
    with pytest.raises(TypeError):
        D.tensor_digest([{"sha256": "a" * 64}], reference=True)
    with pytest.raises(ValueError):
        D.tensor_digest([], reference=True)
    with pytest.raises(ValueError, match="CUDA"):
        D.tensor_digest([torch.arange(8)])
    with pytest.raises(ValueError):
        ActualByteDigest(b"bad-length")


@pytest.mark.gpu
@pytest.mark.parametrize("length", [0, 1, 55, 56, 63, 64, 65, 127, 128, 129, 4097, 16384])
def test_cuda_sha_known_answers_read_actual_tensor_bytes(length):
    if not torch.cuda.is_available():
        pytest.skip("owned CUDA accelerator required")
    raw = bytes(index % 251 for index in range(length))
    value = torch.tensor(list(raw), dtype=torch.uint8, device="cuda:0")
    actual = D.tensor_digest([value], device="cuda:0", max_bytes=max(1, length))
    assert actual == hashlib.sha256(raw).digest()
    if length:
        value[0] ^= 1
        assert D.tensor_digest([value], device="cuda:0", max_bytes=max(1, length)) != actual


@pytest.mark.gpu
def test_cuda_multi_tensor_receipt_roots_match_plain_actual_bytes():
    if not torch.cuda.is_available():
        pytest.skip("owned CUDA accelerator required")
    a = torch.arange(128, device="cuda:0", dtype=torch.bfloat16).reshape(16, 8).t()
    b = torch.arange(11, device="cuda:0", dtype=torch.int64)
    header = b"strict-activation-1"
    expected = header + a.contiguous().cpu().view(torch.uint8).numpy().tobytes() + b.cpu().view(torch.uint8).numpy().tobytes()
    actual = D.tensor_digest([header, a, b], device="cuda:0")
    assert actual == hashlib.sha256(expected).digest()
    key = gen_key()
    old = ReceiptSigner(key, "s", "j", 0, 1, nonce="same")
    new = ReceiptSigner(key, "s", "j", 0, 1, nonce="same")
    old.observe(expected, expected)
    new.observe(ActualByteDigest(actual), ActualByteDigest(actual))
    assert old.finalize() == new.finalize()

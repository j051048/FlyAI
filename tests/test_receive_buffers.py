import gc
import socket
import struct
import threading

import pytest
import torch

from shard import transport as W
from shard.receive_buffers import ReceiveBufferPool


@pytest.fixture(autouse=True)
def reset_pool():
    W.configure_receive_pool(0)
    yield
    gc.collect()
    W.configure_receive_pool(0)


def exchange(value):
    left, right = socket.socketpair()
    left.settimeout(5); right.settimeout(5)
    thread = threading.Thread(target=W.send_msg, args=(left, value))
    thread.start()
    try:
        result = W.recv_msg(right)
        thread.join(5)
        assert not thread.is_alive()
        return result
    finally:
        left.close(); right.close()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32,
    torch.float8_e4m3fn, torch.int64, torch.bool])
def test_real_socket_preserves_bytes_and_owns_last_tensor_view(dtype):
    W.configure_receive_pool(8192, slots=1)
    original = torch.arange(20).to(dtype).reshape(2, 10)
    message = exchange({"h": original, "ids": torch.tensor([[1, 2]], dtype=torch.int64)})
    view = message["h"].reshape(-1)[3:9]
    expected = view.clone()
    del message
    gc.collect()
    assert W.receive_pool_status()["live_slots"] == 1
    # The held *view*, even after its original tensor dies, must prevent reuse.
    other = exchange({"h": torch.full((20,), 77.0)})
    assert torch.equal(view.view(torch.uint8), expected.view(torch.uint8))
    assert W.receive_pool_status()["fallback_frames"] == 1
    del view, other
    gc.collect()
    assert W.receive_pool_status()["live_slots"] == 0
    next_message = exchange({"h": original})
    assert W.receive_pool_status()["borrowed_frames"] == 2
    assert torch.equal(next_message["h"].view(torch.uint8), original.view(torch.uint8))
    del next_message


def test_buffer_views_are_pooled_but_frames_have_independent_storage():
    W.configure_receive_pool(8192, slots=2)
    first = exchange({"h": torch.arange(40)})
    pointer = first["h"].data_ptr()
    second = exchange({"h": torch.arange(40) + 10})
    assert first["h"].data_ptr() != second["h"].data_ptr()
    assert torch.equal(first["h"], torch.arange(40))
    del first
    gc.collect()
    third = exchange({"h": torch.arange(40) + 20})
    assert third["h"].data_ptr() == pointer
    del second, third


def test_reconfigure_cannot_overwrite_outstanding_tensor():
    W.configure_receive_pool(8192)
    result = exchange({"h": torch.ones(32)})
    with pytest.raises(RuntimeError, match="views are live"):
        W.configure_receive_pool(0)
    del result


def test_control_frame_releases_pool_and_binary_encoding_is_unchanged():
    W.configure_receive_pool(8192)
    assert exchange({"op": "reset", "job_id": "new", "nonce": "a" * 64})["op"] == "reset"
    gc.collect()
    assert W.receive_pool_status()["live_slots"] == 0
    value = {"h": torch.randn(2, 4).bfloat16(), "ids": torch.tensor([[2, 3]])}
    result = exchange(value)
    assert W._pack(value) == W._pack(result)
    del result


def test_budget_includes_resize_peak_and_busy_slots():
    pool = ReceiveBufferPool(2048, slots=2)
    first = pool.borrow(100)
    second = pool.borrow(900)
    assert pool.borrow(100) is None
    del first, second
    gc.collect()
    assert pool.borrow(1800) is None  # old 2KiB + replacement would exceed budget
    assert pool.snapshot()["allocated_bytes"] <= 2048


@pytest.mark.parametrize("body", [b"", struct.pack("!I", 999),
    struct.pack("!I", 2) + b"{}" + b"short",
    struct.pack("!I", 2) + b"{}" + struct.pack("!Q", 99)])
def test_malformed_buffer_does_not_keep_a_pool_lease(body):
    W.configure_receive_pool(8192)
    left, right = socket.socketpair()
    left.sendall(struct.pack("!Q", len(body)) + body)
    with pytest.raises(ConnectionError, match="malformed"):
        W.recv_msg(right)
    left.close(); right.close()
    gc.collect()
    assert W.receive_pool_status()["live_slots"] == 0


def test_pool_exhaustion_is_bounded_owned_frame_fallback():
    W.configure_receive_pool(1024, slots=1)
    result = exchange({"h": torch.arange(8192)})
    assert result["h"].shape == (8192,)
    assert W.receive_pool_status()["allocated_bytes"] == 0
    assert W.receive_pool_status()["fallback_frames"] == 1


def test_normal_codec_path_releases_last_view_without_running_cyclic_gc():
    W.configure_receive_pool(8192, slots=1)
    previous = gc.isenabled()
    gc.disable()
    try:
        for _ in range(10):
            received = exchange({"h": torch.arange(20)})
            assert W._pack(received) == W._pack({"h": torch.arange(20)})
            del received
            assert W.receive_pool_status()["live_slots"] == 0
        assert W.receive_pool_status()["fallback_frames"] == 0
    finally:
        if previous: gc.enable()


def test_optional_pool_allocator_failure_uses_original_owned_decoder(monkeypatch):
    W.configure_receive_pool(8192)
    original = torch.arange(20)
    real_empty = torch.empty
    def fail_pool(size, *args, **kwargs):
        if size == 1024 and kwargs.get("dtype") == torch.uint8:
            raise RuntimeError("allocator exhausted")
        return real_empty(size, *args, **kwargs)
    monkeypatch.setattr(torch, "empty", fail_pool)
    received = exchange({"h": original})
    assert torch.equal(received["h"], original)
    assert W.receive_pool_status()["allocation_failures"] == 1
    assert W.receive_pool_status()["allocated_bytes"] == 0


def test_pool_and_empty_wire_tensor_ignore_engine_default_device():
    # Actual serving sets default_device to CUDA. Meta provides a CPU-only
    # regression for accidentally allocating the wire pool on that default.
    pool = ReceiveBufferPool(8192)
    body = W._pack({"empty": torch.empty(0, device="cpu")})
    with torch.device("meta"):
        lease = pool.borrow(100)
        assert lease.dtype.name == "uint8"
        received = W._unpack(body)
        assert received["empty"].device.type == "cpu"
    del lease


@pytest.mark.gpu
def test_pinned_pool_tensor_stays_pinned_until_cuda_copy_finishes():
    if not torch.cuda.is_available():
        pytest.skip("actual CUDA pinned allocator unavailable")
    W.configure_receive_pool(8192, pinned=True)
    received = exchange({"h": torch.arange(40).bfloat16()})
    assert received["h"].is_pinned()
    result = received["h"].to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(result.cpu(), received["h"])
    del received

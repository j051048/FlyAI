import pytest

from phase0.v4_local_transport_bench import measure


@pytest.mark.parametrize("buffer_mib", [0, 2])
def test_real_loopback_boundary_bytes_and_diagnostic_scope(buffer_mib):
    result = measure(tokens=3, dim=64, repeats=4, warmup=1, buffer_mib=buffer_mib)
    assert len(result["samples"]) == 4
    assert result["performance_qualified"] is False
    assert result["gpu_evidence"] == "none; CPU fixture"
    assert result["transport"] == ("buffered_tcp" if buffer_mib else "loopback_tcp")
    assert all(row["payload_ready_to_ready_ms"] > 0 and row["wire_bytes"] > 0 for row in result["samples"])
    if buffer_mib:
        assert result["receive_pool"]["borrowed_frames"] == 5
        assert result["receive_pool"]["fallback_frames"] == 0
        assert result["receive_pool"]["live_slots"] == 0


def test_probe_refuses_unbounded_geometry():
    with pytest.raises(ValueError, match="bounded"):
        measure(tokens=100000)


def test_receiver_failure_preserves_root_cause_and_releases_borrowed_buffers(monkeypatch):
    import v4_pipe
    def fail(*args, **kwargs):
        raise ValueError("synthetic boundary decoder failure")
    monkeypatch.setattr(v4_pipe, "_recv_hids", fail)
    with pytest.raises(RuntimeError, match="synthetic boundary decoder failure"):
        measure(tokens=1, dim=16, repeats=1, warmup=0, buffer_mib=2)
    assert v4_pipe.transport_buffer_status().get("enabled") is False


@pytest.mark.gpu
def test_two_cuda_devices_use_actual_boundary_copies():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two dedicated CUDA devices required")
    result = measure(source_device="cuda:0", target_device="cuda:1", tokens=1, repeats=3, warmup=1, buffer_mib=2, pinned=True)
    assert result["gpu_evidence"] == "actual CUDA copies only"

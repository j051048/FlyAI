import pytest
import torch

from v4_draft_results import DraftResultBuffer


def test_cpu_reference_preserves_integer_ids_and_rounded_float_contract():
    buffer = DraftResultBuffer()
    ids = torch.tensor([2**40+3, 2**24+1, 0], dtype=torch.int64)
    confidence = torch.tensor([.123456789, .99999999, -0.0], dtype=torch.float64)
    second = ids+1
    result = buffer.pack(ids, confidence, second)
    assert result == {"draft": ids.tolist(), "conf": [round(c, 4) for c in confidence.float().tolist()], "d2": second.tolist()}
    ids.zero_(); confidence.zero_()
    assert result["draft"][0] == 2**40+3
    assert buffer.stats()["gpu_d2h_copies"] == 0 and buffer.stats()["allocations"] == 0


@pytest.mark.parametrize("bad", [torch.zeros(2, dtype=torch.int64), torch.zeros((1, 3), dtype=torch.int64)])
def test_malformed_result_is_not_silently_truncated(bad):
    with pytest.raises(ValueError):
        DraftResultBuffer().pack(bad, torch.ones(3))


def test_live_drafter_result_buffer_keeps_same_incremental_acceptance(monkeypatch):
    from test_v4_dspark import DS, _pipelined_drafter, _pframe
    monkeypatch.setattr(DS, "V4_DSPARK_RESULT_BUFFER", False)
    first, _ = _pipelined_drafter()
    reference = first.on_chunk(*_pframe([99], 4, tokens=[63]))
    monkeypatch.setattr(DS, "V4_DSPARK_RESULT_BUFFER", True)
    second, _ = _pipelined_drafter()
    assert second.on_chunk(*_pframe([99], 4, tokens=[63])) == reference
    assert second.result_buffer_stats()["calls"] == 1
    assert second.on_chunk(*_pframe([900], 5, tokens=[64])) == {"acc": False}
    assert second.result_buffer_stats()["calls"] == 1


@pytest.mark.gpu
def test_cuda_one_copy_reuses_buffers_without_changing_result_bytes():
    if not torch.cuda.is_available():
        pytest.skip("actual CUDA required")
    buffer = DraftResultBuffer()
    ids = torch.tensor([2**40+3, 2**24+1, 53], device="cuda", dtype=torch.int64)
    confidence = torch.tensor([.12345, .98765, .4], device="cuda", dtype=torch.float32)
    reference = {"draft": ids.tolist(), "conf": [round(c, 4) for c in confidence.float().tolist()], "d2": (ids+1).tolist()}
    assert buffer.pack(ids, confidence, ids+1) == reference
    pointers = buffer._device.data_ptr(), buffer._host.data_ptr()
    saved = buffer.pack(ids, confidence, ids+1)
    for _ in range(5):
        ids.add_(1); confidence.mul_(.9)
        got = buffer.pack(ids, confidence)
        assert got == {"draft": ids.tolist(), "conf": [round(c, 4) for c in confidence.tolist()]}
    assert saved == reference and pointers == (buffer._device.data_ptr(), buffer._host.data_ptr())
    assert buffer.stats()["allocations"] == 1 and buffer.stats()["gpu_d2h_copies"] == 7

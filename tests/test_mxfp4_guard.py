import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase0"))
from mxfp4_guard import parse_version_tuple, verify_mxfp4_runtime_environment, assert_mxfp4_quantized


def test_parse_version_tuple():
    assert parse_version_tuple("5.19.0") == (5, 19, 0)
    assert parse_version_tuple("0.17.2") == (0, 17, 2)
    assert parse_version_tuple("0.14.1") == (0, 14, 1)


def test_verify_mxfp4_version_mismatch_fails_loud(monkeypatch):
    """P0-2: transformers 5.19.0 with kernels 0.14.1 MUST fail loud with descriptive RuntimeError."""
    fake_tf = MagicMock(__version__="5.19.0")
    fake_kernels = MagicMock(__version__="0.14.1")

    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "kernels", fake_kernels)

    with pytest.raises(RuntimeError) as exc_info:
        verify_mxfp4_runtime_environment(enforce=True)

    err = str(exc_info.value)
    assert "Incompatible kernel ABI" in err
    assert "transformers 5.19.0 requires kernels in [0.17.0, 0.18.0)" in err
    assert "silent fallback to full bf16 dequantization" in err


def test_verify_mxfp4_version_match_passes(monkeypatch):
    """transformers 5.19.0 with kernels 0.17.1 passes cleanly."""
    fake_tf = MagicMock(__version__="5.19.0")
    fake_kernels = MagicMock(__version__="0.17.1")

    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "kernels", fake_kernels)

    res = verify_mxfp4_runtime_environment(enforce=True)
    assert res["ok"] is True
    assert res["transformers"] == "5.19.0"
    assert res["kernels"] == "0.17.1"


def test_assert_mxfp4_quantized_detects_dequantization():
    """Detects layers marked with dequantized flag."""
    model = MagicMock()
    param1 = MagicMock(_is_dequantized=True)
    model.named_parameters.return_value = [("layers.0.moe.experts.0.weight", param1)]

    with pytest.raises(RuntimeError, match="Detected silent dequantization"):
        assert_mxfp4_quantized(model)

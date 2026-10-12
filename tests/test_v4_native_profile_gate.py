import pytest

from test_v4_native_lane_socket import ring


def test_native_warmup_cannot_bypass_named_profile_capture_check(ring, monkeypatch):
    backend, *_ = ring
    monkeypatch.setattr(backend.vp, "_encode_prompt", lambda *args: [1, 2, 3])
    called = []
    original = backend._verify_warmup_profile
    def reject(result):
        assert result["proof"]["verified"] is True
        assert result["proof"]["receipts"]
        called.append(result)
        raise ValueError("native required capture smoke absent")
    backend._verify_warmup_profile = reject
    with pytest.raises(ValueError, match="capture smoke"):
        backend.warmup()
    assert called
    assert backend.ready()[0] is False
    assert backend.stats()["native_lanes"]["active_jobs"] == 0
    # Requalify through a fresh signed native OPEN/reset, so the fixture can
    # stop its owned stages rather than leaving the intentionally closed owner.
    backend._verify_warmup_profile = original
    assert backend.warmup()["proof_verified"] is True
    assert backend.ready()[0] is True

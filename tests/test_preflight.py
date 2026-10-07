import os
import sys
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase0"))
from preflight import detect_hairpin_hazard, run_preflight_checks


def test_detect_hairpin_hazard():
    """P0-3: Detect multiple distinct stages hosted on same public IP."""
    # Stages 1 and 2 on the same public IP -> hairpin hazard!
    endpoints = [
        "142.250.190.46:29600",
        "198.51.100.22:29601",
        "198.51.100.22:29602",
        "104.244.42.1:29600"
    ]
    hazards = detect_hairpin_hazard(endpoints)
    assert len(hazards) == 1
    assert "NAT Hairpin Hazard" in hazards[0]
    assert "198.51.100.22" in hazards[0]


def test_detect_hairpin_hazard_loopback_ignored():
    """Localhost/loopback stages should not trigger hairpin hazard."""
    endpoints = [
        "127.0.0.1:29610",
        "127.0.0.1:29621",
        "127.0.0.1:29622"
    ]
    assert detect_hairpin_hazard(endpoints) == []


def test_run_preflight_insufficient_ram_rejected():
    """Reject boxes with small RAM (e.g. 31GB RAM Vast boxes)."""
    with patch("preflight.get_total_ram_gb", return_value=31.2):
        with patch("preflight.get_disk_free_gb", return_value=500.0):
            with pytest.raises(RuntimeError) as exc_info:
                run_preflight_checks(min_ram_gb=64.0, enforce=True)
            assert "Insufficient Host RAM" in str(exc_info.value)
            assert "31.2GB" in str(exc_info.value)


def test_run_preflight_insufficient_disk_rejected():
    """Reject boxes with insufficient disk space (e.g. 141GB)."""
    with patch("preflight.get_total_ram_gb", return_value=64.0):
        with patch("preflight.get_disk_free_gb", return_value=141.0):
            with pytest.raises(RuntimeError) as exc_info:
                run_preflight_checks(min_disk_gb=300.0, enforce=True)
            assert "Insufficient Disk Space" in str(exc_info.value)
            assert "141.0GB" in str(exc_info.value)


def test_run_preflight_passed():
    """Accept healthy nodes."""
    with patch("preflight.get_total_ram_gb", return_value=64.0):
        with patch("preflight.get_disk_free_gb", return_value=350.0):
            res = run_preflight_checks(min_ram_gb=64.0, min_disk_gb=300.0, enforce=True)
            assert res["ok"] is True
            assert res["ram_gb"] == 64.0
            assert res["disk_free_gb"] == 350.0

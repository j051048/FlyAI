"""Shared test setup: repo root + phase0 importable no matter where pytest runs from.

Collection discipline (audit M13): `pytest --collect-only` must do no network,
subprocess, GPU, or long-import work — anything expensive belongs in a fixture.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "phase0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def pytest_collection_modifyitems(config, items):
    """Normalize test markers.

    Automatically tags CUDA/GPU-bound tests with @pytest.mark.gpu so they can be
    selected by `pytest -m gpu` on self-hosted runners or excluded on CPU-only CI.
    """
    import pytest
    gpu_marker = pytest.mark.gpu

    for item in items:
        # Check module name, test name, and skipif reasons for GPU/CUDA indications
        mod_name = item.module.__name__ if hasattr(item, "module") and item.module else ""
        text = f"{mod_name} {item.name}".lower()

        needs_gpu = False
        if any(kw in text for kw in ("cuda", "sm120", "nvfp4", "mxfp4_guard")):
            needs_gpu = True

        for mark in item.iter_markers(name="skipif"):
            reason = str(mark.kwargs.get("reason", "")).lower()
            if "cuda" in reason or "gpu" in reason:
                needs_gpu = True
                break

        if needs_gpu and not item.get_closest_marker("gpu"):
            item.add_marker(gpu_marker)

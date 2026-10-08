"""Calibration binds installed capabilities and the actual probe launch config."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
R = pytest.importorskip("v4_ref_cpu")
V = pytest.importorskip("v4_stage")
resources = pytest.importorskip("v4_resources")


def test_dspark_job_switch_cannot_change_loaded_capability_or_calibration_digest():
    args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
                     dspark_target_layer_ids=(1, 2, 3))
    stage = V.Stage(0, 4, args, head=True, tail=True, dspark=True, device="cpu",
                    runtime_metrics=False, runtime_profile=False)
    initial = resources.runtime_config_identity(stage)
    stage._dspark = False  # Exactly what a greedy job's serving reset does.
    stage.reset()
    assert stage._dspark_capable is True
    assert resources.runtime_config_payload(stage)["dspark"] is True
    assert resources.runtime_config_identity(stage) == initial
    report = resources.measure_stage_resources(stage, checkpoint_id="fixture")
    assert report["dspark"] is True
    assert report["draft_measurement_status"] == "missing"  # Capability must budget actual MTP load.
    stage._dspark = True
    assert resources.runtime_config_identity(stage) == initial


def test_cpu_stage_honors_resolved_metrics_environment(monkeypatch):
    monkeypatch.setattr(V, "V4_RUNTIME_METRICS", True)
    args = R.cpu_args()
    stage = V.Stage(0, 1, args, device="cpu", runtime_profile=False)
    assert resources.runtime_config_payload(stage)["runtime_metrics_enabled"] is True


def test_gpu_probe_constructor_does_not_override_real_launch_instrumentation(monkeypatch, tmp_path):
    # Stop at constructor dispatch. No GPU, fake capacity or measured result is
    # produced; this pins propagation before any model/probe execution.
    class ConstructorReached(Exception):
        pass
    received = {}
    def constructor(*args, **kwargs):
        received.update(kwargs)
        raise ConstructorReached
    args = SimpleNamespace(max_seq_len=1, max_batch_size=1)
    module = SimpleNamespace(config=lambda path: args, Stage=constructor)
    monkeypatch.setattr(resources, "inspect_checkpoint", lambda path: {"checkpoint_id": "fixture"})
    monkeypatch.setattr(resources, "stage_storage_inventory", lambda *a, **kw: {})
    monkeypatch.setattr(resources, "_load_engine_module", lambda name: module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch, "set_default_device", lambda device: None)
    monkeypatch.setattr(torch, "set_default_dtype", lambda dtype: None)
    with pytest.raises(ConstructorReached):
        resources.measure_checkpoint(tmp_path, 0, 1, device="cuda:0", max_seq=128,
                                     prefill_tokens=4, decode_tokens=2)
    assert "runtime_metrics" not in received
    assert "runtime_profile" not in received
    assert received["device"] == "cuda:0"

from copy import deepcopy
import json
import pytest

from v4_kernel_tuning import KernelSelector


def selector(tmp_path, **kwargs):
    return KernelSelector({"policy": "graph", "gpu_uuid": "gpu-test", "source_sha256": "a" * 64},
                          tmp_path / "tuning", **kwargs)


def test_exact_gate_and_worst_workload_control_selection(tmp_path):
    s = selector(tmp_path)
    times = {"reference": {"1": 10., "8": 10.}, "wrong": {"1": .1, "8": .1},
             "regression": {"1": 1., "8": 11.}, "safe": {"1": 8., "8": 9.}}
    assert s.select({"N": 4096}, list(times), probe=lambda n: n != "wrong", measure=times.get) == "safe"
    assert s.last["refused"] == ["wrong"]


def test_cached_choice_is_reprobed_and_environment_partitioned(tmp_path):
    times = {"reference": {"1": 10.}, "safe": {"1": 8.}}
    s = selector(tmp_path)
    assert s.select("op", list(times), probe=lambda n: True, measure=times.get) == "safe"
    seen = []
    fresh = selector(tmp_path)
    assert fresh.select("op", list(times), probe=lambda n: seen.append(n) or True,
                        measure=lambda n: pytest.fail("compatible cache should reuse timing")) == "safe"
    assert seen == ["safe"] and fresh.last["current_process_reprobed"]
    broken = selector(tmp_path)
    assert broken.select("op", list(times), probe=lambda n: n == "reference", measure=times.get) == "reference"
    changed = KernelSelector({"policy": "eager", "gpu_uuid": "gpu-test", "source_sha256": "a" * 64},
                             tmp_path / "tuning")
    assert changed._path("op") != s._path("op")


def test_stale_or_modified_cache_cannot_authorize_candidate(tmp_path):
    now = [100.]
    s = selector(tmp_path, clock=lambda: now[0], ttl_s=10)
    times = {"reference": {"1": 10.}, "safe": {"1": 8.}}
    s.select("op", list(times), probe=lambda n: True, measure=times.get)
    path = s._path("op")
    value = json.loads(path.read_text()); value["choice"] = "reference"
    path.write_text(json.dumps(value))
    called = []
    s.select("op", list(times), probe=lambda n: True, measure=lambda n: called.append(n) or times[n])
    assert called == list(times)
    now[0] = 111
    assert s._read("op", list(times)) is None


def test_no_measured_gain_keeps_reference(tmp_path):
    s = selector(tmp_path)
    assert s.select("op", ["reference", "candidate"], probe=lambda n: True,
                    measure=lambda n: {"1": 1.}) == "reference"


def test_measurement_failure_keeps_reference_and_missing_reference_fails(tmp_path):
    s = selector(tmp_path)
    assert s.select("op", ["reference", "candidate"], probe=lambda n: True,
                    measure=lambda n: {"1": 1. if n == "reference" else float("nan")}) == "reference"
    with pytest.raises(RuntimeError, match="reference"):
        s.select("other", ["reference"], probe=lambda n: False, measure=lambda n: {"1": 1.})


def test_cache_does_not_cross_source_or_policy_identity(tmp_path):
    s = selector(tmp_path)
    other = KernelSelector({**s.identity, "source_sha256": "b" * 64}, tmp_path / "tuning")
    assert s._path("op") != other._path("op")
    with pytest.raises(ValueError):
        KernelSelector({"policy": "guess"})


@pytest.mark.gpu
def test_actual_fp8_tuning_reprobes_original_kernel_and_graph(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import textwrap
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    pytest.importorskip("tilelang")
    script = textwrap.dedent("""
        import torch, v4_ref_cpu, v4_fp8_gemv as fp8, v4_kernels_cpu
        assert v4_kernels_cpu.backend() == 'tilelang'
        mod = v4_ref_cpu.load_ref()
        torch.set_default_dtype(torch.bfloat16)
        fp8.configure_tuning_execution('graph')
        fp8._REF_FP8_GEMM = mod.fp8_gemm
        choice = fp8._measured_tile(256,256,'float32')
        for m in (1,8,32):
            a,a_s,w,w_s = fp8._probe_seeded(256,256,'float32',m)
            reference = fp8._REF_FP8_GEMM(a,a_s,w,w_s,torch.float32)
            if choice is not None:
                assert torch.equal(reference,fp8._run_tiled(a,a_s,w,w_s,torch.float32,choice))
        record = fp8._TUNING_RECORDS[(256,256,'float32')]
        assert record['identity']['policy']=='graph' and record['current_process_reprobed']
        assert 'reference' in record['timings_ms']
        print('ORIGINAL_FP8_GRAPH_TUNING_VERIFIED')
    """)
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "V4_KERNELS": "tilelang", "V4_FP8_GEMV": "0", "V4_FP8_SHARED": "0",
           "V4_FP8_AUTOTUNE": "graph", "V4_HADAMARD": "torch",
           "V4_KERNEL_TUNING_DIR": str(tmp_path / "timing"),
           "PYTHONPATH": os.pathsep.join((str(root), str(root / "phase0"), str(root / "engines" / "deepseek_v4")))}
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, (result.stdout + result.stderr)[-12000:]
    assert "ORIGINAL_FP8_GRAPH_TUNING_VERIFIED" in result.stdout


def test_fp8_tuning_declines_wrong_serving_policy_before_probes(monkeypatch):
    import v4_fp8_gemv as fp8
    monkeypatch.setattr(fp8, "V4_FP8_AUTOTUNE", "graph")
    with fp8.tuning_execution("eager"):
        with pytest.raises(ValueError, match="actual serving"):
            fp8._measured_tile(256, 256, "float32")


def test_fp8_timing_cache_is_partitioned_by_output_dtype(tmp_path):
    a = KernelSelector({"policy": "graph", "output_dtype": "torch.bfloat16"}, tmp_path / "timings")
    b = KernelSelector({"policy": "graph", "output_dtype": "torch.float32"}, tmp_path / "timings")
    assert a._path("op") != b._path("op")

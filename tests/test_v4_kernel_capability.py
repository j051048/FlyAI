"""CPU tests exercise reports, isolation and refusal, never qualify GPU math."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from engines.deepseek_v4 import v4_native_capability as C


def options(device="cuda:0"):
    return C._options(device, False, 512 << 20, 20)


def fixture_report(status="pass", uuid="GPU-fixture-only"):
    identity = {"gpu_uuid": uuid, "gpu_name": "schema fixture", "capability": [12, 0],
        "driver": "fixture", "cuda": "fixture", "torch": "fixture", "tilelang": "fixture", "source_sha256": "a" * 64}
    cases = [{"name": name, "status": "pass", "executed_gpu": True, "graph_replay": False} for name in C.CASES]
    return C._report(options(), status, "schema fixture; not GPU execution", identity=identity, cases=cases)


def test_actual_cpu_subprocess_never_reports_gpu_qualification(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "phase0/v4_kernel_capability.py"),
        "--device", "cpu", "--timeout-s", "20"], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PYTHONUTF8": "1"}, timeout=30)
    assert result.returncode == 1, result.stderr
    assert result.stdout, result.stderr
    report = C.validate_report(json.loads(result.stdout))
    assert report["status"] == "gpu_unavailable" and report["qualification"] == "none"
    assert not report["executed_gpu"] and not report["runtime_ready"] and not report["cases"]


@pytest.mark.parametrize("cli_first", [False, True])
def test_fresh_process_import_order_keeps_cli_and_runtime_distinct(tmp_path, cli_first):
    root = Path(__file__).resolve().parents[1]
    # Deliberately put phase0 before the engine, as aggregate pytest collection
    # and script entry points can do. Neither import order may shadow runtime.
    code = """
import importlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path[:0] = [str(root / 'phase0'), str(root / 'engines/deepseek_v4'), str(root)]
names = ['v4_kernel_capability', 'engines.deepseek_v4.v4_native_capability']
if sys.argv[2] == 'False': names.reverse()
for name in names: importlib.import_module(name)
cli = sys.modules['v4_kernel_capability']
runtime = sys.modules['engines.deepseek_v4.v4_native_capability']
assert cli.main is runtime.main
assert pathlib.Path(cli.__file__) == root / 'phase0/v4_kernel_capability.py'
assert pathlib.Path(runtime.__file__) == root / 'engines/deepseek_v4/v4_native_capability.py'
assert pathlib.Path(runtime._worker_command()[1]) == pathlib.Path(runtime.__file__)
report = runtime._run_worker(runtime._options('cpu', False, 512 << 20, 20))
runtime.validate_report(report)
assert report['status'] == 'gpu_unavailable', report
assert report['qualification'] == 'none' and report['process_cleanup']['confirmed'], report
print(json.dumps({'runtime': pathlib.Path(runtime.__file__).name, 'status': report['status']}))
"""
    result = subprocess.run([sys.executable, "-c", code, str(root), str(cli_first)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONUTF8": "1"})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"runtime": "v4_native_capability.py", "status": "gpu_unavailable"}


@pytest.mark.parametrize("mutation", ["cpu", "coverage", "unexecuted", "ready", "graph", "identity", "bad_capability", "time"])
def test_report_validator_refuses_incomplete_or_wrong_claims(mutation):
    report = fixture_report()
    C.validate_report(report)
    if mutation == "cpu": report["options"]["device"] = "cpu"
    elif mutation == "coverage": report["cases"].pop()
    elif mutation == "unexecuted": report["cases"][0]["executed_gpu"] = False
    elif mutation == "ready": report["runtime_ready"] = True
    elif mutation == "graph": report["options"]["graph"] = True
    elif mutation == "identity": report["identity"]["driver"] = "changed"
    elif mutation == "bad_capability":
        report["identity"]["capability"] = [True, 0]
        report["identity_sha256"] = C._digest(report["identity"])
    else: report["measured_at"] = float("nan")
    with pytest.raises(ValueError): C.validate_report(report)


@pytest.mark.parametrize("error,status", [(ImportError("tilelang"), "unverified"),
    (RuntimeError("CUDA out of memory"), "unverified"),
    (RuntimeError("no kernel image is available for execution"), "unsupported"),
    (RuntimeError("ptxas requires .target sm_100a"), "unsupported"),
    (AttributeError("module 'torch' has no attribute 'float4_e2m1fn_x2'"), "unverified"),
    (AttributeError("module 'tilelang.language' has no attribute 'symbolic'"), "unverified"),
    (AttributeError("type object 'PassConfigKey' has no attribute 'TL_DISABLE_TMA_LOWER'"), "unverified"),
    (AttributeError("module 'kernel' has no attribute 'fp4_gemm'"), "fail"),
    (AssertionError("numerical mismatch"), "fail"), (RuntimeError("unknown failure"), "fail")])
def test_failure_classification_keeps_unknown_and_numeric_errors_failed(error, status):
    assert C._classify(error) == status


def test_cached_pass_never_skips_current_execution_and_identity_cannot_cross(monkeypatch, tmp_path):
    current = fixture_report()
    calls = []
    def worker(config):
        calls.append(config)
        return deepcopy(current)
    monkeypatch.setattr(C, "_run_worker", worker)
    first = C.probe_native_capability(cache_dir=tmp_path, timeout_s=20)
    assert not first["previous_matching_report"]
    second = C.probe_native_capability(cache_dir=tmp_path, timeout_s=20)
    assert second["previous_matching_report"] and len(calls) == 2
    current["status"], current["qualification"] = "fail", "none"
    current["reason"] = "current execution failed"
    assert C.probe_native_capability(cache_dir=tmp_path, timeout_s=20)["status"] == "fail"
    current = fixture_report(uuid="GPU-another-fixture")
    assert not C.probe_native_capability(cache_dir=tmp_path, timeout_s=20)["previous_matching_report"]
    current["identity"]["source_sha256"] = "b" * 64
    current["identity_sha256"] = C._digest(current["identity"])
    assert not C.probe_native_capability(cache_dir=tmp_path, timeout_s=20)["previous_matching_report"]
    assert len(calls) == 5


def test_preflight_probe_is_explicit_and_cpu_failure_never_sets_ready(tmp_path, monkeypatch):
    from test_v4_preflight import package
    from engines.deepseek_v4 import v4_preflight as P
    directory, plan = package(tmp_path)
    node = {"model": str(directory), "node_key": str(tmp_path / "absent"),
            "listen_port": 29610, "dspark": False, "device": "cpu"}
    def forbidden(*a, **kw): raise AssertionError("default preflight must not launch native kernels")
    monkeypatch.setattr(C, "probe_native_capability", forbidden)
    legacy = P.check_node(plan, 0, node, probe_gpu=False, check_tokenizer=False, allow_owned_listener=True)
    assert "native_kernel_probe" not in legacy["checks"] and legacy["scope"] == "read-only deployment checks"
    monkeypatch.setattr(C, "probe_native_capability", lambda *a, **kw:
        C._report(options("cpu"), "gpu_unavailable", "no actual CUDA execution"))
    active = P.check_node(plan, 0, node, probe_gpu=False, check_tokenizer=False, allow_owned_listener=True,
        native_kernel_probe={"enabled": True})
    assert active["checks"]["native_kernel_probe"]["status"] == "gpu_unavailable"
    assert not active["preflight_ok"] and not active["runtime_ready"]
    assert any(row["check"] == "native_kernel_probe" for row in active["errors"])
    assert "opt-in synthetic GPU" in active["scope"]


@pytest.mark.parametrize("kwargs", [{"max_probe_bytes": 128 << 20}, {"max_probe_bytes": True},
    {"timeout_s": float("nan")}, {"graph": "yes"}, {"device": "cuda;unsafe"}])
def test_invalid_unbounded_probe_options_never_launch(kwargs, monkeypatch):
    monkeypatch.setattr(C, "_run_worker", lambda _: pytest.fail("must validate before launching"))
    with pytest.raises(ValueError): C.probe_native_capability(**kwargs)


@pytest.mark.parametrize("parent_exits", [False, True])
def test_actual_owned_compiler_child_is_cleaned_on_timeout_or_parent_exit(tmp_path, monkeypatch, parent_exits):
    import psutil
    child_file = tmp_path / "owned-child.json"
    code = ("import json,os,psutil,subprocess,sys,time; sys.stdin.buffer.read(); "
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],"
        "creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)); "
        "p=psutil.Process(child.pid); "
        "open(sys.argv[1],'w').write(json.dumps({'pid':p.pid,'birth':p.create_time()})); "
        + ("sys.exit(0)" if parent_exits else "time.sleep(120)"))
    monkeypatch.setattr(C, "_worker_command", lambda: [sys.executable, "-c", code, str(child_file)])
    started = time.monotonic()
    try:
        report = C._run_worker(C._options("cpu", False, 512 << 20, 2))
        assert time.monotonic() - started < 2 + C._CLEANUP_TIMEOUT_S + 3
        assert child_file.exists(), report
        row = json.loads(child_file.read_text())
        try:
            child = psutil.Process(row["pid"])
            assert child.create_time() != row["birth"] or child.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
        assert report["process_cleanup"]["confirmed"], report
        assert report["status"] == ("fail" if parent_exits else "unverified")
        assert report["qualification"] == "none" and not report["runtime_ready"]
    finally:
        # Only test-created birth-identified PIDs; also clean up on assertion failures.
        if child_file.exists():
            row = json.loads(child_file.read_text())
            try:
                child = psutil.Process(row["pid"])
                if child.create_time() == row["birth"]:
                    targets = child.children(recursive=True) + [child]
                    for process in targets:
                        try: process.kill()
                        except psutil.NoSuchProcess: pass
                    psutil.wait_procs(targets, timeout=2)
            except psutil.NoSuchProcess:
                pass


def test_failed_compile_is_attempted_not_completed_execution():
    report = C._report(options(), "unsupported", "ISA rejected", cases=[
        {"name": "fp4_quant", "status": "unsupported", "executed_gpu": None}])
    C.validate_report(report)
    assert report["gpu_execution_attempted"] and not report["executed_gpu"]
    assert report["qualification"] == "none"


def test_malformed_worker_output_becomes_failed_report(monkeypatch):
    code = "import sys;sys.stdin.buffer.read();print(" + repr(C.MARKER + '{"schema":"bad"}') + ")"
    monkeypatch.setattr(C, "_worker_command", lambda: [sys.executable, "-c", code])
    result = C._run_worker(options())
    assert result["status"] == "fail" and result["qualification"] == "none"
    assert "invalid isolated probe report" in result["reason"]


def test_unconfirmed_child_cleanup_cannot_preserve_an_otherwise_passing_report(monkeypatch):
    monkeypatch.setattr(C, "_run_isolated", lambda *args: {"timeout": False, "returncode": 0,
        "tail": C.MARKER + json.dumps(fixture_report()), "cleanup": {
            "confirmed": False, "errors": ["owned compiler still active"]}})
    report = C._run_worker(options())
    assert report["status"] == "unverified" and report["qualification"] == "none"
    assert not report["runtime_ready"] and not report["process_cleanup"]["confirmed"]


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("FLYAI_RUN_NATIVE_KERNEL_CAPABILITY") != "1",
                    reason="explicit idle-GPU native kernel qualification only")
def test_actual_native_gpu_kernel_smoke_and_graph_replay():
    report = C.probe_native_capability(graph=True, timeout_s=900)
    assert report["status"] == "pass", report
    assert report["qualification"] == "native_kernel_smoke_only" and not report["runtime_ready"]

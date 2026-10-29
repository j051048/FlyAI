"""Frozen local/component checks and strict A/B evidence for six V4 improvements.

Nothing runs a GPU without an explicit execute flag and idle-device assignment.
Component CUDA tests are not native-model or whole-product qualification. The
existing four-mode/whole-rollback 5090 campaign remains an independent obligation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import v4_5090_acceptance as A

SCHEMA = "flyai-v4-six-acceptance-plan/1"
OBLIGATIONS = ("all_stage_round_cost", "local_ipc_actual_bytes", "production_shape_routing",
               "persistent_ragged_batch", "ram_expert_segments", "incremental_prefix_kv")
PATTERNS = ("test_v4_adaptive_round*.py", "test_v4_host_transport*.py", "test_v4_device_digest.py",
    "test_v4_expert_segments.py", "test_v4_expert_tiers*.py", "test_v4_native_batch*.py",
    "test_v4_batch*.py", "test_v4_conversation*.py", "test_v4_kv*.py", "test_ring_economics.py",
    "test_cost_aware_routes.py", "test_v4_six_acceptance.py", "test_v4_persistent_batch*.py",
    "test_v4_round_cost.py", "test_production_routing.py", "test_v4_incremental_qualification.py",
    "test_v4_indexed_kv.py", "test_v4_six_stage.py", "test_v4_incremental_case_probe.py")
GPU_FILES = ("tests/test_v4_device_digest.py", "tests/test_v4_expert_segments.py",
             "tests/test_v4_expert_tiers.py", "tests/test_v4_persistent_batch.py", "tests/test_v4_indexed_kv.py")


def _local_files():
    return sorted({path.relative_to(ROOT).as_posix() for pattern in PATTERNS
                   for path in (ROOT / "tests").glob(pattern)})


def source_identity():
    paths = set()
    for folder, pattern in (("engines/deepseek_v4", "*.py"), ("vendor/deepseek_v4_ref", "*.py"), ("shard", "*.py"),
                            ("phase0", "v4*.py"), ("tools", "v4*.py")):
        paths.update((ROOT / folder).rglob(pattern))
    paths.update(ROOT / name for name in _local_files())
    if any(path.is_symlink() for path in paths):
        raise ValueError("source identity cannot traverse symlinks")
    rows = [{"name": path.relative_to(ROOT).as_posix(), "sha256": A.file_hash(path)}
            for path in sorted(paths)]
    return {"files": rows, "sha256": A.digest(rows)}


def prepare(out):
    path = Path(out)
    if path.exists():
        raise ValueError("use a fresh plan file")
    value = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_identity(), "local_files": _local_files(), "gpu_files": list(GPU_FILES),
        "obligations": list(OBLIGATIONS), "native_gpu_verified": False,
        "github_push_allowed": False}
    value["sha256"] = A.digest(value)
    A.write(path, value)
    return value


def require_plan(path):
    value = A.read(path)
    if (value.get("schema") != SCHEMA or value.get("sha256") != A.digest(
            {k: v for k, v in value.items() if k != "sha256"})
            or value.get("source") != source_identity() or value.get("local_files") != _local_files()
            or value.get("gpu_files") != list(GPU_FILES) or value.get("obligations") != list(OBLIGATIONS)
            or value.get("native_gpu_verified") is not False or value.get("github_push_allowed") is not False):
        raise ValueError("plan/source changed or qualification was fabricated; create a new plan")
    return value


def junit_summary(path):
    suites = ET.parse(path).getroot()
    cases = list(suites.iter("testcase"))
    if not cases:
        raise ValueError("JUnit has no executed test cases")
    return {"tests": len(cases), "failures": sum(c.find("failure") is not None for c in cases),
            "errors": sum(c.find("error") is not None for c in cases),
            "skipped": sum(c.find("skipped") is not None for c in cases)}


def _fresh_directory(directory):
    path = Path(directory).resolve()
    if path.exists():
        raise ValueError("use a fresh evidence directory")
    path.mkdir(parents=True)
    return path


def checks(plan_path, directory, *, python=sys.executable, gpu=False, execute=False, devices=""):
    plan = require_plan(plan_path)
    if gpu and (not execute or not devices or any(not part.isdigit() for part in devices.split(","))
                or len(set(devices.split(","))) != len(devices.split(","))):
        raise ValueError("GPU checks require --execute-gpu and explicit distinct idle --confirm-gpus")
    directory = _fresh_directory(directory)
    runner = A.Runner(directory, python)
    files = list(GPU_FILES) if gpu else plan["local_files"]
    junit = directory / "checks.junit.xml"
    argv = [python, "-m", "pytest", *files, "-m", "gpu" if gpu else "not gpu", "-q",
            "--junitxml", str(junit), "--basetemp", str(directory / "pytest-temp")]
    env = {"PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = devices
    report = {"schema": "flyai-v4-six-checks/1", "plan_sha256": plan["sha256"],
        "kind": "gpu_components" if gpu else "local_cpu", "argv": argv,
        "native_gpu_verified": False, "github_push_allowed": False,
        "obligations": {name: "pending_native_model_ab" for name in OBLIGATIONS}}
    try:
        runner.command("six-checks", argv, 7200 if gpu else 3600, environment=env)
        report["junit"] = junit_summary(junit)
        if report["junit"]["failures"] or report["junit"]["errors"] or (gpu and report["junit"]["skipped"]):
            raise ValueError("required component case failed or GPU case skipped")
        require_plan(plan_path)
        report["status"] = "gpu_component_pass" if gpu else "local_pass"
    except BaseException as error:
        report.update(status="failed", error_class=type(error).__name__, message=str(error))
        raise
    finally:
        report["commands"] = runner.records
        report["source_after"] = source_identity()["sha256"]
        report["files"] = [{"name": p.relative_to(directory).as_posix(), "sha256": A.file_hash(p),
                            "size": p.stat().st_size} for p in directory.rglob("*")
                           if p.is_file() and not p.is_symlink() and "private" not in p.relative_to(directory).parts
                           and "pytest-temp" not in p.relative_to(directory).parts]
        A.write(directory / "report.json", report)
    return report


def strict_ab(plan_path, before, after, directory, *, vary, before_protocol=None,
              after_protocol=None, python=sys.executable):
    """Reverify real raw receipts before comparing one predeclared variable."""
    plan = require_plan(plan_path)
    directory = _fresh_directory(directory)
    before, after = Path(before).resolve(), Path(after).resolve()
    if before == after or not before.is_file() or not after.is_file():
        raise ValueError("two distinct existing native benchmark reports required")
    if (before_protocol is None or after_protocol is None
            or not Path(before_protocol).is_file() or not Path(after_protocol).is_file()):
        raise ValueError("independently frozen before/after protocols required")
    protocols = {"before": Path(before_protocol).resolve(), "after": Path(after_protocol).resolve()}
    runner = A.Runner(directory, python)
    for label, path in (("before", before), ("after", after)):
        runner.command("verify-" + label, [python, "phase0/v4_benchmark.py", "verify", str(path),
                                           "--protocol", str(protocols[label])], 600)
    runner.command("compare", [python, "phase0/v4_benchmark.py", "compare", str(before), str(after),
                               "--vary", vary], 600)
    require_plan(plan_path)
    result = {"schema": "flyai-v4-six-strict-ab/1", "plan_sha256": plan["sha256"],
        "before": {"name": before.name, "sha256": A.file_hash(before)},
        "after": {"name": after.name, "sha256": A.file_hash(after)}, "vary": vary,
        "protocols": {name: {"name": path.name, "sha256": A.file_hash(path)} for name, path in protocols.items()},
        "commands": runner.records, "status": "strict_comparison_completed",
        "note": "Inspect raw compare output and actual feature counters; exit zero alone does not establish a speed gain.",
        "native_gpu_verified": False, "github_push_allowed": False}
    A.write(directory / "report.json", result)
    return result


def digest_probe(directory, *, device="cpu", execute=False, repeats=20, sizes=(32768, 8388608)):
    if (type(repeats) is not int or not 5 <= repeats <= 200 or not sizes
            or any(type(size) is not int or not 1 <= size <= 64 << 20 for size in sizes)):
        raise ValueError("bounded repetitions/sizes required")
    import torch
    sys.path.insert(0, str(ROOT / "engines/deepseek_v4"))
    from v4_device_digest import tensor_digest
    gpu = torch.device(device).type == "cuda"
    if gpu and (not execute or not torch.cuda.is_available()):
        raise ValueError("actual digest GPU probe needs --execute-gpu and available owned CUDA")
    if not gpu and device != "cpu":
        raise ValueError("CPU reference or explicit CUDA device required")
    directory = _fresh_directory(directory)
    observations = []
    for size in sizes:
        tensor = torch.arange(size, dtype=torch.int64).remainder_(251).to(torch.uint8).to(device)
        # Compile/warm before timing, retaining first-use cost separately.
        start = time.perf_counter()
        first = tensor_digest([tensor], device=device, max_bytes=size, reference=not gpu)
        warmup_s = time.perf_counter() - start
        samples = []
        for index in range(repeats):
            pair = {}
            for arm in (("cpu", "device") if index % 2 == 0 else ("device", "cpu")):
                if gpu:
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                digest = (hashlib.sha256(tensor.cpu().numpy().tobytes()).digest() if arm == "cpu" else
                          tensor_digest([tensor], device=device, max_bytes=size, reference=not gpu))
                pair[arm + "_ms"] = (time.perf_counter() - start) * 1000
                if digest != first:
                    raise RuntimeError("actual byte SHA changed between A/B arms")
            samples.append(pair)
        observations.append({"payload_bytes": size, "sha256": first.hex(), "first_call_s": warmup_s, "samples": samples})
    result = {"schema": "flyai-v4-device-digest-probe/1", "device": device,
        "gpu_observed": gpu, "scope": "isolated_real_bytes_hash_not_full_ipc_or_native_ring",
        "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
        "gpu_uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "")) if gpu else None,
        "observations": observations, "performance_qualified": False, "github_push_allowed": False}
    A.write(directory / "report.json", result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--out", required=True)
    for name in ("local", "gpu-components"):
        cmd = sub.add_parser(name); cmd.add_argument("--plan", required=True); cmd.add_argument("--out-dir", required=True)
        cmd.add_argument("--python", default=sys.executable)
        if name == "gpu-components":
            cmd.add_argument("--execute-gpu", action="store_true"); cmd.add_argument("--confirm-gpus", required=True)
    ab = sub.add_parser("strict-ab"); ab.add_argument("--plan", required=True)
    ab.add_argument("--before", required=True); ab.add_argument("--after", required=True)
    ab.add_argument("--before-protocol", required=True); ab.add_argument("--after-protocol", required=True)
    ab.add_argument("--vary", required=True); ab.add_argument("--out-dir", required=True)
    ab.add_argument("--python", default=sys.executable)
    dg = sub.add_parser("digest-probe"); dg.add_argument("--out-dir", required=True)
    dg.add_argument("--device", default="cpu"); dg.add_argument("--execute-gpu", action="store_true")
    dg.add_argument("--repeats", type=int, default=20)
    dg.add_argument("--sizes", default="32768,8388608")
    args = p.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.out)
    elif args.command in ("local", "gpu-components"):
        result = checks(args.plan, args.out_dir, python=args.python, gpu=args.command == "gpu-components",
            execute=getattr(args, "execute_gpu", False), devices=getattr(args, "confirm_gpus", ""))
    elif args.command == "strict-ab":
        result = strict_ab(args.plan, args.before, args.after, args.out_dir, vary=args.vary,
            before_protocol=args.before_protocol, after_protocol=args.after_protocol, python=args.python)
    else:
        result = digest_probe(args.out_dir, device=args.device, execute=args.execute_gpu,
            repeats=args.repeats, sizes=tuple(int(x) for x in args.sizes.split(",")))
    print(json.dumps({key: result[key] for key in ("schema", "status", "sha256", "gpu_observed",
        "native_gpu_verified", "github_push_allowed") if key in result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""V4 release evidence gate; no rental, deployment, model fetch or remote actions.

`gpu-check` explicitly runs small local CUDA acceptance tests on an operator-owned
idle SM120 device. `freeze`/`verify` only read local reports. A CPU run cannot
produce a CUDA pass, and skipped CUDA tests do not qualify a release. Signed node
reports and local GPU probes remain operator evidence, not remote attestation.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from phase0 import v4_benchmark as B, v4_soak as S

GPU_FILES = ("tests/test_v4_runtime_init.py", "tests/test_v4_hybrid_gpu.py",
    "tests/test_v4_conversation_cache_gpu.py", "tests/test_v4_prefill_expert_pipeline.py",
    "tests/test_v4_native_trace.py", "tests/test_v4_kv_runtime.py",
    "tests/test_v4_kernel_tuning.py", "tests/test_v4_draft_results.py", "tests/test_v4_native_lanes.py",
    "tests/test_v4_numeric_diagnostics.py", "tests/test_v4_island_coverage.py",
    "tests/test_v4_kernel_capability.py", "tests/test_receive_buffers.py")
REQUIRED_CASES = ("test_actual_cuda_hadamard_smoke_and_graph_capture",
    "test_original_tilelang_grouped_fp4_matches_full_resident_reference",
    "test_cuda_actual_v4_snapshot_restores_same_continuation_and_state_bytes",
    "test_native_cuda_graph_reads_restored_fixed_kv_without_recapture",
    "test_original_tilelang_fp4_prefill_pipeline_preserves_exact_expert_math",
    "test_actual_cuda_native_trace_restores_and_executes_original_single_token_math",
    "test_actual_fp8_tuning_reprobes_original_kernel_and_graph",
    "test_cuda_one_copy_reuses_buffers_without_changing_result_bytes",
    "test_actual_cuda_native_lane_interleaving_preserves_original_v4_math",
    "test_cuda_actual_whole_graph_matches_its_eager_twin_and_rejection",
    "test_actual_island_capture_reports_three_per_layer_after_decode",
    "test_actual_native_gpu_kernel_smoke_and_graph_replay",
    "test_pinned_pool_tensor_stays_pinned_until_cuda_copy_finishes")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _junit(xml):
    root = ET.fromstring(xml)
    cases = list(root.iter("testcase"))
    return {"cases": [case.get("name", "") for case in cases],
            "failures": sum(bool(case.findall("failure") or case.findall("error")) for case in cases),
            "skips": sum(bool(case.findall("skipped")) for case in cases)}


def cuda_probe():
    import torch  # lazy: read-only CLI does not need CUDA/model dependencies
    if not torch.cuda.is_available():
        return {"available": False, "scope": "local torch CUDA probe"}
    p = torch.cuda.get_device_properties(0)
    try: tilelang = importlib.metadata.version("tilelang")
    except importlib.metadata.PackageNotFoundError: tilelang = None
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "--id=0"],
            capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError): driver = None
    from shard.runtime_observation import gpu_uuid_text
    gpu_uuid = gpu_uuid_text(getattr(p, "uuid", None))
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True).stdout
        occupied = [line for line in apps.splitlines() if line.strip() and "," in line
                    and line.split(",")[0].strip() == gpu_uuid and line.split(",")[-1].strip() != str(os.getpid())]
        idle = bool(gpu_uuid) and not occupied
    except (OSError, subprocess.SubprocessError): idle = None
    return {"available": True, "name": p.name, "uuid": gpu_uuid, "idle": idle,
            "capability": list(torch.cuda.get_device_capability(0)), "vram_bytes": p.total_memory,
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "tilelang": tilelang, "driver": driver,
            "scope": "local torch CUDA probe; not remote hardware attestation"}


def run_gpu_gate(junit_path, *, timeout_s=1800):
    if type(timeout_s) not in (int, float) or not 0 < timeout_s <= 7200:
        raise ValueError("bounded positive GPU-test timeout required")
    report = {"schema": "flyai-v4-gpu-gate/1", "hardware": cuda_probe(),
        "source": B.source_identity(),
        "test_sources": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in GPU_FILES},
        "status": "unverified", "hardware_attested": False}
    if (not report["hardware"]["available"] or report["hardware"].get("capability") != [12, 0]
            or report["hardware"].get("idle") is not True):
        report["reason"] = "idle local SM120 CUDA device required; CPU/busy/unsupported/unreadable GPU is unverified"
        return report
    path = Path(junit_path).resolve()
    if path.exists(): raise ValueError("use a fresh JUnit artifact path; stale evidence cannot be reused")
    env = {**{k:v for k,v in os.environ.items() if not k.startswith("V4_")
              and k != "FLYAI_NATIVE_PROBE_PROCESS_TOKEN"},
           "V4_KERNELS": "cpu", "V4_HADAMARD": "torch", "V4_CUDA_GRAPH": "0",
           "V4_EXPERT_PLACEMENT": "gpu", "V4_KV_PLACEMENT": "gpu", "V4_CONVERSATION_CACHE_MIB": "0",
           "V4_PREFILL_EXPERT_PIPELINE": "0", "V4_LEVERS_STRICT": "0", "V4_REF_SLIM": "0",
           "V4_REF_SLIM_NOQAT": "0", "V4_MOE_IN_GRAPH": "0", "V4_DSPARK_MOE": "0"}
    env.update(V4_FP8_AUTOTUNE="0", V4_DSPARK_RESULT_BUFFER="0", V4_NATIVE_LANES="0")
    started = time.perf_counter()
    from engines.deepseek_v4.v4_native_capability import _run_isolated
    # Nothing imports CUDA before the owned process group/job is established.
    # The same bounded tree supervisor also contains native compiler children.
    worker = ("import json,os,sys,runpy; data=json.load(sys.stdin); "
              "os.chdir(data['cwd']); "
              "[os.environ.pop(k) for k in list(os.environ) if k.startswith('V4_')]; "
              "os.environ.update(data['environment']); sys.argv=data['argv']; "
              "runpy.run_module('pytest',run_name='__main__')")
    execution = _run_isolated([sys.executable, "-c", worker], json.dumps({"cwd": str(ROOT), "environment": env,
        "argv": ["pytest", "-m", "gpu", "-q", *GPU_FILES, "--junitxml", str(path)]}), timeout_s)
    report["exit_code"] = execution["returncode"]
    report["log_tail"] = execution["tail"][-32000:]
    report["process_cleanup"] = execution["cleanup"]
    if execution["timeout"]:
        report.update(status="failed", reason="bounded CUDA gate timed out; owned compiler tree cleanup checked")
    report["elapsed_s"] = time.perf_counter() - started
    if path.is_file():
        raw = path.read_bytes()
        if len(raw) > 8 << 20: raise ValueError("JUnit artifact exceeds bounded report size")
        report["junit"] = raw.decode("utf-8")
        report["junit_sha256"] = hashlib.sha256(raw).hexdigest()
    report["source_unchanged"] = (report["source"]["sha256"] == B.source_identity()["sha256"] and
        report["test_sources"] == {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in GPU_FILES})
    verdict = evaluate_gpu_gate(report)
    report["status"], report["reason"] = verdict["status"], "; ".join(verdict["errors"] + verdict["missing_evidence"])
    return report


def evaluate_gpu_gate(report):
    errors, missing = [], []
    if not isinstance(report, dict) or report.get("schema") != "flyai-v4-gpu-gate/1":
        return {"status": "failed", "errors": ["unknown GPU gate schema"], "missing_evidence": []}
    hardware = report.get("hardware", {})
    if report.get("process_cleanup", {}).get("confirmed") is not True:
        missing.append("owned GPU/compile process tree cleanup unconfirmed")
    if hardware.get("idle") is not True: missing.append("dedicated idle CUDA device evidence absent")
    capability = hardware.get("capability")
    if (hardware.get("available") is not True or not isinstance(capability, list) or len(capability) != 2
            or any(type(value) is not int for value in capability) or capability != [12, 0]):
        missing.append("actual local SM120 CUDA evidence absent")
    if report.get("source_unchanged") is not True: missing.append("stable executed source inventory absent")
    if set(report.get("test_sources", {})) != set(GPU_FILES): missing.append("complete frozen GPU test-source inventory absent")
    if not isinstance(report.get("source"), dict) or not report["source"].get("files"):
        missing.append("executed engine source absent")
    xml = report.get("junit")
    if not isinstance(xml, str): missing.append("raw GPU JUnit absent")
    elif len(xml) > 8 << 20: errors.append("raw JUnit exceeds bounded evidence size")
    else:
        try:
            if hashlib.sha256(xml.encode()).hexdigest() != report.get("junit_sha256"):
                errors.append("JUnit byte digest differs")
            result = _junit(xml)
            if result["failures"] or report.get("exit_code") != 0: errors.append("actual GPU tests failed")
            if result["skips"]: missing.append("skipped GPU cases do not qualify")
            if any(not any(case.startswith(name) for case in result["cases"]) for name in REQUIRED_CASES):
                missing.append("mandatory native FP4/state/graph/continuation CUDA cases absent")
        except (ET.ParseError, TypeError, ValueError): errors.append("invalid raw JUnit")
    return {"status": "failed" if errors else "unverified" if missing else "passed",
            "errors": errors, "missing_evidence": missing}


def freeze_contract(four, six, soak_protocol, soak_contract):
    for count, protocol in ((4, four), (6, six)):
        B.require_protocol(protocol)
        if len(protocol["hardware"]) != count: raise ValueError("release needs independently frozen four/six-card protocols")
        if protocol.get("evidence_contract") != {"coordinator_diagnostics_required": True, "runtime_observation_required": True}:
            raise ValueError("release requires current strict signed-runtime evidence contracts")
    B.require_protocol(soak_protocol)
    if soak_protocol.get("evidence_contract") != {"coordinator_diagnostics_required": True, "runtime_observation_required": True}:
        raise ValueError("sustained release needs current signed-runtime evidence")
    if any(any(protocol["checkpoint"].get(name) != four["checkpoint"].get(name)
               for name in ("sha256", "config_sha256", "tokenizer_sha256", "quantization")) or
           protocol["source"]["sha256"] != four["source"]["sha256"] or
           protocol["prompts"] != four["prompts"] for protocol in (six, soak_protocol)):
        raise ValueError("release protocols differ in checkpoint/source/frozen workload")
    if (not isinstance(soak_contract, dict) or type(soak_contract.get("cycles")) is not int or soak_contract["cycles"] < 1
            or set(soak_contract) != {"cycles", "duration_s", "ttft_p95_s", "token_gap_p95_s", "max_idle_gap_s"}):
        raise ValueError("independent complete soak contract required")
    for name in set(soak_contract) - {"cycles"}: S._finite_positive(soak_contract[name], name)
    return {"schema": "flyai-v4-release-contract/1", "targets": {"4": 40.0, "6": 30.0},
            "four_protocol": four, "six_protocol": six, "soak_protocol": soak_protocol,
            "soak_contract": soak_contract, "required_gpu_cases": list(REQUIRED_CASES)}


def evaluate_release(report, contract):
    try:
        frozen = freeze_contract(contract["four_protocol"], contract["six_protocol"],
                                 contract["soak_protocol"], contract["soak_contract"])
        if contract != frozen or report.get("schema") != "flyai-v4-release/1" or report.get("contract_sha256") != digest(contract):
            raise ValueError("release differs from independently frozen contract")
        verdicts = {"gpu": evaluate_gpu_gate(report["gpu_gate"]),
            "four": B.evaluate_report(report["four"], expected_protocol=contract["four_protocol"]),
            "six": B.evaluate_report(report["six"], expected_protocol=contract["six_protocol"]),
            "soak": S.evaluate_soak(report["soak"], contract["soak_protocol"], contract["soak_contract"])}
        statuses = [value["status"] for value in verdicts.values()]
        gpu = report["gpu_gate"]
        compatible = (gpu.get("source", {}).get("sha256") == contract["four_protocol"]["source"]["sha256"] and all(
            all(node.get(field) == gpu.get("hardware", {}).get(field) for field in ("torch", "cuda", "tilelang", "driver"))
            and node.get("gpu_name") == gpu.get("hardware", {}).get("name")
            for protocol in (contract["four_protocol"], contract["six_protocol"], contract["soak_protocol"])
            for node in protocol["hardware"]))
        if not compatible:
            verdicts["gpu_environment"] = {"status": "unverified", "reason": "CUDA gate source/device/dependency/driver differs from deployed frozen nodes"}
            statuses.append("unverified")
        status = "passed" if all(value == "passed" for value in statuses) else "failed" if any(value in ("failed", "valid_baseline") for value in statuses) else "unverified"
        return {"status": status, "checks": verdicts, "hardware_attested": False,
                "scope": "strict frozen 4/6 warm speed + sustained ring + local CUDA gates; operator evidence"}
    except (ValueError, KeyError, TypeError) as error:
        return {"status": "failed", "errors": [str(error)], "hardware_attested": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gpu = commands.add_parser("gpu-check"); gpu.add_argument("--junit", required=True); gpu.add_argument("--out", required=True)
    gpu.add_argument("--timeout-s", type=float, default=1800)
    freeze = commands.add_parser("freeze")
    for name in ("four-protocol", "six-protocol", "soak-protocol", "soak-contract", "out"): freeze.add_argument("--" + name, required=True)
    verify = commands.add_parser("verify")
    for name in ("contract", "four", "six", "soak", "gpu-gate", "out"): verify.add_argument("--" + name, required=True)
    options = parser.parse_args(argv)
    def read(name): return json.loads(Path(getattr(options, name)).read_text(encoding="utf-8-sig"))
    if options.command == "gpu-check": result = run_gpu_gate(options.junit, timeout_s=options.timeout_s)
    elif options.command == "freeze": result = freeze_contract(read("four_protocol"), read("six_protocol"), read("soak_protocol"), read("soak_contract"))
    else:
        contract = read("contract")
        bundle = {"schema": "flyai-v4-release/1", "contract_sha256": digest(contract),
                  "four": read("four"), "six": read("six"), "soak": read("soak"), "gpu_gate": read("gpu_gate")}
        result = {"report": bundle, "verification": evaluate_release(bundle, contract)}
    Path(options.out).write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    status = result.get("verification", result).get("status", "passed")
    return 0 if status == "passed" else 1


if __name__ == "__main__": raise SystemExit(main())

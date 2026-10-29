"""Verify a signed incremental campaign; pending campaigns never enable reuse.

This collector does not run GPUs or invent native-case measurements. A strict
append canary alone cannot qualify rollback, cancellation, reset and isolation.
Each absent native case is listed explicitly and no loadable policy is written.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "engines" / "deepseek_v4"):
    sys.path.insert(0, str(directory))

from v4_incremental_evidence import inspect_evidence, digest, CASES, MAX_BYTES
from v4_incremental_prefix import SCHEMA, IncrementalQualification


def collect(spec, evidence):
    fields = {"schema", "binding", "source_sha256", "execution", "max_context", "max_suffix", "parent_alignment"}
    if not isinstance(spec, dict) or set(spec) != fields or spec["schema"] != "v4-incremental-prefix-campaign/1":
        raise ValueError("explicit incremental campaign binding/source/execution/geometry required")
    value = {key: item for key, item in spec.items() if key != "schema"}
    value.update(schema=SCHEMA, gpu_verified=True, cases={case: True for case in CASES},
                 evidence=evidence, evidence_sha256=digest(evidence))
    status = inspect_evidence(value)
    if status["ready"]:
        IncrementalQualification(value)  # Verify the exact serving loader contract.
    return {"schema": "v4-incremental-prefix-campaign-result/1", **status,
            "qualification": value if status["ready"] else None}


def read(path):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("campaign metadata exceeds 4 MiB")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--case-report", action="append", default=[],
                        help="append a real per-stage native GPU case-probe report; repeat for each assigned stage")
    parser.add_argument("--canary-report", action="append", default=[],
                        help="append a v4-incremental-strict-canary/1 report containing its signed canary")
    args = parser.parse_args(argv)
    evidence = read(args.evidence)
    for path in args.case_report:
        body = read(path)
        if (not isinstance(body, dict) or body.get("schema") != "v4-incremental-native-case-probe/1"
                or body.get("gpu_verified") is not True or body.get("reference") is not False
                or not isinstance(body.get("native_cases"), list) or len(body["native_cases"]) > 6):
            raise ValueError("a real native GPU case-probe report is required; CPU fixtures cannot qualify")
        evidence["native_cases"].extend(body["native_cases"])
    for path in args.canary_report:
        body = read(path)
        if not isinstance(body, dict) or set(body) != {"schema", "canary"} or body["schema"] != "v4-incremental-strict-canary/1":
            raise ValueError("strict signed canary report schema differs")
        evidence["strict_canaries"].append(body["canary"])
    result = collect(read(args.spec), evidence)
    directory = Path(args.out_dir)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "report.json").write_text(json.dumps({k: v for k, v in result.items() if k != "qualification"},
        sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if result["qualification"] is not None:
        (directory / "qualification.json").write_text(json.dumps(result["qualification"], sort_keys=True,
            indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "qualification"}, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

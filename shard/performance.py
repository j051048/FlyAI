"""Validate and aggregate actual committed-token measurements, never predictions."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

SCHEMA = "shard-pipeline-metrics/2"


def validate_measurement(result):
    metrics = result.get("metrics", {})
    if metrics.get("schema") != SCHEMA:
        raise ValueError("versioned committed-token measurements required")
    for key in ("committed_tokens", "new_tokens", "new_decode_tokens"):
        if type(metrics.get(key)) is not int or metrics[key] < 0:
            raise ValueError("nonnegative measured token counts required")
    tokens = result.get("output_ids", result.get("tokens"))
    if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens) or len(tokens) != metrics["committed_tokens"]:
        raise ValueError("measurement token count differs from committed output")
    if not metrics["new_decode_tokens"] <= metrics["new_tokens"] <= metrics["committed_tokens"]:
        raise ValueError("new/resume token accounting is inconsistent")
    for key in ("decode_s", "request_s", "ttft_s"):
        if key == "ttft_s" and metrics.get(key) is None and not metrics["new_tokens"]:
            continue
        if type(metrics.get(key)) not in (int, float) or not math.isfinite(metrics[key]) or metrics[key] < 0:
            raise ValueError("finite measured durations required")
    if metrics["new_decode_tokens"] and metrics["decode_s"] <= 0:
        raise ValueError("decode tokens require a positive measured interval")
    if metrics["ttft_s"] is not None and metrics["ttft_s"] > metrics["request_s"]:
        raise ValueError("first token cannot occur after request completion")
    return {**metrics, "decode_tok_s": metrics["new_decode_tokens"] / metrics["decode_s"] if metrics["decode_s"] else None,
            "request_tok_s": metrics["new_tokens"] / metrics["request_s"] if metrics["request_s"] else None}


def percentile(values, fraction):
    values = sorted(values)
    if not values: return None
    index = (len(values) - 1) * fraction
    lo = int(index); hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summarize_runs(records):
    groups = {}
    for row in records:
        if not row.get("result", {}).get("proof", {}).get("verified", row.get("result", {}).get("proof_verified", False)):
            raise ValueError("benchmark output requires a successful complete receipt verification")
        measured = validate_measurement(row["result"])
        key = (row["cohort_id"], row["configuration_sha256"], row["workload"], row.get("warmness", "uncontrolled"))
        groups.setdefault(key, []).append(measured)
    output = []
    for (cohort, config, workload, warmness), measurements in groups.items():
        row = {"cohort_id": cohort, "configuration_sha256": config, "workload": workload,
               "warmness": warmness, "runs": len(measurements)}
        for name in ("decode_tok_s", "request_tok_s", "ttft_s", "request_s"):
            values = [m[name] for m in measurements if m[name] is not None]
            row[name] = {"p50": percentile(values, .5), "p95": percentile(values, .95), "samples": len(values)}
        output.append(row)
    return {"schema": "shard-performance-summary/1", "prediction_only": False, "groups": output,
            "scope": "runtime measurements; source reports and hardware must be verified separately"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args(argv)
    records = []
    for path in args.inputs:
        body = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        records.extend(body["runs"] if isinstance(body, dict) else body)
    print(json.dumps(summarize_runs(records), allow_nan=False))


if __name__ == "__main__": main()

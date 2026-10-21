"""Bounded generation artifacts and separate offline quality scoring.

Content hashes detect accidental mutation; they are not execution attestations.
These artifacts never establish Pass@1, model quality, hardware qualification or
speed. Keep the original benchmark signatures/numeric traces for those scopes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

SCHEMA = "flyai-v4-quality-generation/1"
MAX_ROWS = 4096
MAX_TOKENS = 16384
MAX_LINE_BYTES = 2 << 20
MAX_FILE_BYTES = 64 << 20
NATIVE_ABI = "deepseek-v4-native/1"
FIXTURE_ABI = "v4-cpu-reference-fixture/1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _ids(value, *, empty=False):
    return (isinstance(value, list) and (empty or bool(value)) and len(value) <= MAX_TOKENS
            and all(type(v) is int and 0 <= v < 1 << 31 for v in value))


def validate_dataset(dataset):
    if not isinstance(dataset, list) or not 1 <= len(dataset) <= MAX_ROWS:
        raise ValueError("bounded nonempty quality dataset required")
    names = set()
    for case in dataset:
        if (not isinstance(case, dict) or set(case) - {"id", "prompt_ids", "expected_tokens"}
                or not isinstance(case.get("id"), str) or not 1 <= len(case["id"]) <= 256
                or case["id"] in names or not _ids(case.get("prompt_ids"))
                or "expected_tokens" in case and not _ids(case["expected_tokens"], empty=True)):
            raise ValueError("dataset needs unique case IDs, frozen prompt IDs and optional expected tokens")
        names.add(case["id"])
    if len(canonical(dataset)) > MAX_FILE_BYTES // 2:
        raise ValueError("quality dataset byte budget exceeded")
    return dataset


def _seal(record):
    return dict(record, sha256=digest(record))


def _checked(record):
    if not isinstance(record, dict) or record.get("sha256") != digest({k: v for k, v in record.items() if k != "sha256"}):
        raise ValueError("quality record content hash differs")
    return record


def generation_artifact(identity, dataset, records, *, origin):
    """Package actual outputs; mandatory binding includes dataset and execution recipe.

    `origin` is descriptive provenance. Neither its spelling nor operator-owned
    hardware declarations grant trusted execution or production qualification.
    """
    validate_dataset(dataset)
    if (not isinstance(identity, dict) or set(identity) != {"artifact", "source_sha256", "tokenizer_sha256",
            "config_sha256", "sampling", "hardware"} or
            any(not _sha(identity[k]) for k in ("source_sha256", "tokenizer_sha256", "config_sha256")) or
            not isinstance(identity["artifact"], dict) or identity["artifact"].get("runtime_abi") not in (NATIVE_ABI, FIXTURE_ABI)
            or not isinstance(identity["sampling"], dict) or not identity["sampling"]
            or not isinstance(identity["hardware"], (dict, list)) or not identity["hardware"]
            or not isinstance(origin, str) or not 1 <= len(origin) <= 128):
        raise ValueError("quality identity requires original V4 ABI plus source/tokenizer/config/sampling/hardware bindings")
    if len(canonical(identity)) > MAX_LINE_BYTES // 2:
        raise ValueError("quality identity byte budget exceeded")
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_ROWS:
        raise ValueError("bounded generation rows required")
    cases = {case["id"]: case for case in dataset}
    seen, rows = set(), []
    for row in records:
        if (not isinstance(row, dict) or set(row) - {"sample_id", "case_id", "prompt_sha256", "tokens", "token_logprobs", "metadata"}
                or not isinstance(row.get("sample_id"), str) or not 1 <= len(row["sample_id"]) <= 512
                or row["sample_id"] in seen or row.get("case_id") not in cases
                or row.get("prompt_sha256") != digest(cases[row["case_id"]]["prompt_ids"])
                or not _ids(row.get("tokens"), empty=True)):
            raise ValueError("generation row differs from its frozen case/prompt identity")
        seen.add(row["sample_id"])
        probabilities = row.get("token_logprobs")
        if probabilities is not None and (not isinstance(probabilities, list) or len(probabilities) != len(row["tokens"])
                or any(type(p) not in (int, float) or not math.isfinite(p) or p > 0 for p in probabilities)):
            raise ValueError("token log probabilities must match every actual output token")
        value = _seal(dict(row, schema="flyai-v4-quality-sample/1"))
        if len(canonical(value)) > MAX_LINE_BYTES:
            raise ValueError("quality sample line exceeds byte budget")
        rows.append(value)
    header = _seal({"schema": SCHEMA, "kind": "generation", "identity": identity,
        "identity_sha256": digest(identity), "dataset": dataset, "dataset_sha256": digest(dataset),
        "origin": origin, "records": len(rows), "records_sha256": digest(rows),
        "production_qualified": False, "pass_at_1": None,
        "scope": "saved generation outputs; hash integrity is not execution attestation"})
    if len(canonical(header)) > MAX_LINE_BYTES:
        raise ValueError("quality dataset/header exceeds line budget")
    return header, rows


def validate_artifact(header, rows):
    _checked(header)
    for row in rows:
        _checked(row)
        if row.get("schema") != "flyai-v4-quality-sample/1":
            raise ValueError("unsupported quality sample schema")
    if (header.get("schema") != SCHEMA or header.get("kind") != "generation"
            or header.get("production_qualified") is not False or header.get("pass_at_1") is not None
            or header.get("records") != len(rows) or header.get("records_sha256") != digest(rows)
            or header.get("identity_sha256") != digest(header.get("identity"))
            or header.get("dataset_sha256") != digest(header.get("dataset"))):
        raise ValueError("quality generation header/bindings differ")
    # Reuse semantic validation, including bounds; a recomputed hash alone must
    # not smuggle a V4.1 ABI, NaN probability or mismatched prompt into scoring.
    rebuilt = generation_artifact(header["identity"], header["dataset"],
        [{k: v for k, v in row.items() if k not in ("schema", "sha256")} for row in rows], origin=header["origin"])
    if rebuilt != (header, rows):
        raise ValueError("unsupported quality artifact fields")
    return header, rows


def write_jsonl(path, header, rows):
    validate_artifact(header, rows)
    destination = Path(path).resolve()
    fd, temporary = tempfile.mkstemp(prefix=".v4-quality-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            size = 0
            for row in (header, *rows):
                line = canonical(row) + b"\n"; size += len(line)
                if len(line) > MAX_LINE_BYTES or size > MAX_FILE_BYTES:
                    raise ValueError("quality JSONL exceeds line/file byte budget")
                stream.write(line)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_jsonl(path):
    rows, size = [], 0
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate quality JSON field")
            result[key] = value
        return result
    with Path(path).open("rb") as stream:
        while True:
            line = stream.readline(MAX_LINE_BYTES + 1)
            if not line:
                break
            size += len(line)
            if len(line) > MAX_LINE_BYTES or size > MAX_FILE_BYTES or len(rows) > MAX_ROWS:
                raise ValueError("quality JSONL exceeds bounded input budget")
            rows.append(json.loads(line, object_pairs_hook=unique,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite quality JSON"))))
    if len(rows) < 2:
        raise ValueError("quality JSONL needs its header and generation rows")
    return validate_artifact(rows[0], rows[1:])


def score_exact(header, rows):
    """Offline exact token references only, never a code-evaluation Pass@1 claim."""
    validate_artifact(header, rows)
    cases = {case["id"]: case for case in header["dataset"]}
    scored = [row for row in rows if "expected_tokens" in cases[row["case_id"]]]
    if not scored:
        raise ValueError("frozen dataset has no expected-token scoring references")
    matched = sum(row["tokens"] == cases[row["case_id"]]["expected_tokens"] for row in scored)
    return _seal({"schema": "flyai-v4-quality-score/1", "generation_sha256": header["sha256"],
        "dataset_sha256": header["dataset_sha256"], "identity_sha256": header["identity_sha256"],
        "scorer": "exact-token-reference/1", "scored_samples": len(scored), "matched_samples": matched,
        "exact_token_match_rate": matched / len(scored), "unscored_samples": len(rows) - len(scored),
        "pass_at_1": None, "production_qualified": False,
        "scope": "exact token references; no executable tests, semantic quality or Pass@1 evaluation"})


def compare_generations(left, right):
    a, x = validate_artifact(*left); b, y = validate_artifact(*right)
    if a["dataset_sha256"] != b["dataset_sha256"]:
        raise ValueError("generation comparisons require the same frozen dataset")
    for key in ("artifact", "tokenizer_sha256", "config_sha256"):
        if a["identity"][key] != b["identity"][key]:
            raise ValueError("generation model/tokenizer/config identity differs")
    pairs = ({row["sample_id"]: row for row in values} for values in (x, y))
    before, after = pairs
    if before.keys() != after.keys():
        raise ValueError("generation sample sets differ")
    differences = []
    for name, row in before.items():
        other = after[name]
        if row["case_id"] != other["case_id"] or row["prompt_sha256"] != other["prompt_sha256"]:
            raise ValueError("sample alignment differs")
        if row["tokens"] != other["tokens"] or row.get("token_logprobs") != other.get("token_logprobs"):
            differences.append(name)
    return _seal({"schema": "flyai-v4-quality-generation-comparison/1", "left_sha256": a["sha256"],
        "right_sha256": b["sha256"], "exact_match": not differences, "different_samples": differences,
        "same_source_sampling_hardware": all(a["identity"][k] == b["identity"][k]
                                              for k in ("source_sha256", "sampling", "hardware")),
        "pass_at_1": None, "production_qualified": False, "speed_evidence": False})


def from_benchmark(report):
    """Export actual committed tokens from the existing benchmark, without rerunning it."""
    try:
        from phase0 import v4_benchmark as benchmark
    except ImportError:
        import v4_benchmark as benchmark
    protocol = report.get("protocol")
    benchmark.require_protocol(protocol)
    if report.get("schema") != benchmark.REPORT_SCHEMA or report.get("protocol_sha256") != protocol["sha256"]:
        raise ValueError("benchmark report/protocol binding differs")
    dataset = [{"id": row["workload"], "prompt_ids": row["token_ids"]} for row in protocol["prompts"]]
    identity = {"artifact": {"model_id": protocol["model_id"], "runtime_abi": NATIVE_ABI,
                             "checkpoint": protocol["checkpoint"]},
        "source_sha256": protocol["source"]["sha256"],
        "config_sha256": protocol["checkpoint"]["config_sha256"],
        "tokenizer_sha256": protocol["checkpoint"]["tokenizer_sha256"],
        "sampling": protocol["run"], "hardware": protocol["hardware"]}
    records = []
    for row in report.get("samples", []):
        if row.get("output_sha256") != benchmark.digest(row.get("tokens")):
            raise ValueError("benchmark output tokens differ from their saved digest")
        records.append({"sample_id": f"{row['workload']}:{row['phase']}:{row['rep']}:{row['mode']}",
            "case_id": row["workload"], "prompt_sha256": row["prompt_sha256"], "tokens": row["tokens"],
            "token_logprobs": None, "metadata": {"mode": row["mode"], "job_id": row["job_id"],
                "benchmark_report_sha256": digest(report), "receipts_sha256": digest(row.get("receipts", [])),
                "original_backend": report.get("backend"),
                "benchmark_protocol_sha256": protocol["sha256"]}})
    return generation_artifact(identity, dataset, records, origin="existing-benchmark-committed-output")


def compare_teacher_forced(left, right):
    """Preserve the existing strict per-layer byte comparison and add logprob diagnostics."""
    try:
        from engines.deepseek_v4.v4_numeric_diagnostics import compare_traces
    except ModuleNotFoundError:
        from v4_numeric_diagnostics import compare_traces
    result = compare_traces(left, right)
    pairs = []
    for trace in (left, right):
        pairs.append({(row["position"], row["tokens"], row["layer"]): row
                      for row in trace["records"] if "teacher_token" in row})
    a, b = pairs
    deltas = []
    aligned = a.keys() == b.keys()
    if aligned:
        for key, row in a.items():
            other = b[key]
            if row["teacher_token"] != other["teacher_token"]:
                raise ValueError("teacher-forced targets differ")
            if row["next_token_logprob"] is not None and other["next_token_logprob"] is not None:
                deltas.append(abs(row["next_token_logprob"] - other["next_token_logprob"]))
    result["teacher_forced"] = {"aligned_targets": aligned, "compared_token_logprobs": len(deltas),
        "max_absolute_logprob_difference": max(deltas) if deltas else None,
        "scope": "same frozen next-token targets; absent legacy logprobs are unknown"}
    result.update(production_qualified=False, pass_at_1=None)
    return _seal(result)

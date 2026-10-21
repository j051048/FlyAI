"""Offline V4 generation packaging/scoring; no network, model load, or GPU use.

Export committed benchmark results or package previously generated local rows,
then score/compare the saved JSONL in a separate invocation. Neither operation
claims executable-test Pass@1, speed or production hardware qualification.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from engines.deepseek_v4 import v4_quality as Q


def read_json(path):
    with Path(path).open("rb") as stream:
        data = stream.read(Q.MAX_FILE_BYTES + 1)
    if len(data) > Q.MAX_FILE_BYTES:
        raise ValueError("quality input exceeds file byte budget")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate quality JSON field")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite quality JSON")))


def write_report(path, report):
    destination = Path(path).resolve()
    payload = Q.canonical(report) + b"\n"
    if len(payload) > Q.MAX_FILE_BYTES:
        raise ValueError("quality report byte budget exceeded")
    fd, temporary = tempfile.mkstemp(prefix=".v4-quality-report-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export-benchmark", help="actual committed outputs of an existing V4 benchmark report")
    export.add_argument("report"); export.add_argument("--out", required=True)
    package = sub.add_parser("package", help="package existing local outputs; imports are not execution attestations")
    package.add_argument("--identity", required=True); package.add_argument("--dataset", required=True)
    package.add_argument("--records", required=True); package.add_argument("--out", required=True)
    score = sub.add_parser("score", help="exact frozen token references; never executable-test Pass@1")
    score.add_argument("generation"); score.add_argument("--out", required=True)
    compare = sub.add_parser("compare", help="same dataset/model/tokenizer/config; retain both source/hardware identities")
    compare.add_argument("left"); compare.add_argument("right"); compare.add_argument("--out", required=True)
    teacher = sub.add_parser("compare-teacher", help="existing native numerical traces with optional next-token logprobs")
    teacher.add_argument("left"); teacher.add_argument("right"); teacher.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "export-benchmark":
        artifact = Q.from_benchmark(read_json(args.report)); Q.write_jsonl(args.out, *artifact)
        result = {"generation_sha256": artifact[0]["sha256"], "records": len(artifact[1])}
    elif args.command == "package":
        artifact = Q.generation_artifact(read_json(args.identity), read_json(args.dataset), read_json(args.records),
                                         origin="imported-local-generation-records")
        Q.write_jsonl(args.out, *artifact)
        result = {"generation_sha256": artifact[0]["sha256"], "records": len(artifact[1])}
    elif args.command == "score":
        result = Q.score_exact(*Q.read_jsonl(args.generation)); write_report(args.out, result)
    elif args.command == "compare":
        result = Q.compare_generations(Q.read_jsonl(args.left), Q.read_jsonl(args.right)); write_report(args.out, result)
    else:
        result = Q.compare_teacher_forced(read_json(args.left), read_json(args.right)); write_report(args.out, result)
    print(json.dumps({"out": str(Path(args.out).resolve()), "production_qualified": False,
                      "pass_at_1": None, **{k: v for k, v in result.items()
                          if k in ("generation_sha256", "records", "exact_match", "exact_token_match_rate")}}))
    return 1 if result.get("exact_match") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())

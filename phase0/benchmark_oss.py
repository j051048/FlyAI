"""Repeat fixed GPT-OSS requests and save raw verified measurements plus percentiles."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from engines.gpt_oss.network_service import GPTOSSRingBackend
from shard.deployment import config_digest
from shard.performance import summarize_runs
from shard.pipeline_plan import load_plan
from shard.service_queue import Job


def run_cases(backend, cases, *, repeats=5, warmup=True):
    if type(repeats) is not int or repeats < 1 or not isinstance(cases, list) or not cases:
        raise ValueError("positive repeats and nonempty benchmark cases required")
    if warmup: backend.warmup()
    config = {"K": backend.K, "depth": backend.depth, "ngram_n": backend.ngram_n,
              "max_context": backend.max_context, "adaptive": backend.adaptive,
              "adaptive_depth": backend.adaptive_depth,
              "prefill_chunk": backend.prefill_chunk, "plan": backend.plan}
    config["coordinator_sources"] = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT/"phase0/specpipe.py", ROOT/"engines/gpt_oss/network_service.py", ROOT/"shard/pipeline_session.py")}
    digest = config_digest(config)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    version = git.stdout.strip() if git.returncode == 0 else None
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True, capture_output=True)
    records = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str) or case.get("workload") not in ("copy", "novel", "code", "long_context"):
            raise ValueError("each case needs a name and explicit workload category")
        body = case["request"]
        payload, count, maximum, timeout = backend.prepare(body)
        request_hash = hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
        for repeat in range(repeats):
            now = time.monotonic()
            job = Job("__benchmark__", payload, count, maximum, now + timeout, "benchmark", None, now, state="running")
            result = backend.execute(job, job.commit, job.check_stop)
            records.append({"case": case["name"], "workload": case["workload"], "repeat": repeat,
                "warmness": "warm" if warmup else "uncontrolled", "request_sha256": request_hash,
                "configuration_sha256": digest, "cohort_id": backend.plan["cohort_id"], "code_sha": version,
                "code_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
                "configuration": config, "result": result})
    return {"schema": "shard-benchmark-runs/1", "runs": records, "summary": summarize_runs(records)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--cases", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--K", type=int, default=4); parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--ngram-n", type=int, default=3)
    parser.add_argument("--max-ctx", type=int, default=8192)
    parser.add_argument("--adaptive-pipe", action="store_true")
    parser.add_argument("--adaptive-depth", action="store_true")
    args = parser.parse_args(argv)
    backend = GPTOSSRingBackend(args.model, load_plan(args.plan), K=args.K, depth=args.depth,
                               ngram_n=args.ngram_n, max_context=args.max_ctx, adaptive=args.adaptive_pipe,
                               adaptive_depth=args.adaptive_depth)
    try:
        result = run_cases(backend, json.loads(Path(args.cases).read_text(encoding="utf-8-sig")), repeats=args.repeats)
        path = Path(args.out); part = path.with_name(path.name + ".part")
        part.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        part.replace(path)
        print(json.dumps(result["summary"], allow_nan=False))
    finally: backend.close()


if __name__ == "__main__": main()

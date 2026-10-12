"""Export reproducible ring-planner economics from strict raw V4 benchmarks.

This command reads local evidence only. It does not launch nodes or synthesize
setup durations. Save OpenNetworkService.form_one/form_all's returned formation
measurement alongside the benchmark to make preparation cost known.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shard.ring_economics import benchmark_economics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="complete raw strict v4_benchmark report")
    parser.add_argument("--deployment-plan", help="optional when formation-result contains its actual deployment_plan")
    parser.add_argument("--formation", required=True, help="one configured formation JSON object")
    parser.add_argument("--formation-result", help="saved real form_one result with measured startup duration")
    parser.add_argument("--ttl-s", type=float, default=3600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        formed = read(args.formation_result) if args.formation_result else {}
        measured = formed.get("formation_measurement")
        plan = read(args.deployment_plan) if args.deployment_plan else formed.get("deployment_plan")
        if plan is None:
            raise ValueError("actual deployment plan is required")
        value = benchmark_economics(read(args.report), plan, read(args.formation),
                                   measured, ttl_s=args.ttl_s)
        raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        print(json.dumps({"path": str(destination), "sha256": hashlib.sha256(raw).hexdigest(),
            "replacement_cost_known": value["summary"]["prepare_load_capture_warmup_s"] is not None
                and value["summary"]["cache_loss_s"] is not None}, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps({"status": "unverified", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

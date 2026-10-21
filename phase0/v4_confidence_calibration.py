"""Fit held-out raw V4 confidence and reverify separate signed workload costs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "engines" / "deepseek_v4"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--protocols", nargs="+", required=True)
    parser.add_argument("--binding", required=True, help="running service round slot's exported .binding.json")
    parser.add_argument("--max-p95-ttft-s", type=float, required=True)
    parser.add_argument("--max-p95-tpot-s", type=float, required=True)
    parser.add_argument("--max-heldout-ece", type=float, default=.1)
    parser.add_argument("--ttl-s", type=float, default=900.)
    parser.add_argument("--mode", choices=("shadow", "enforce"), default="shadow")
    parser.add_argument("--publish-slot", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    try:
        from engines.deepseek_v4.v4_confidence_calibration import build_calibration, verify_calibration
        from engines.deepseek_v4.v4_round_evidence import build_calibration as build_round, read_binding, publish_slot
        from phase0.v4_benchmark import _write
        if len(args.reports) != len(args.protocols):
            raise ValueError("one independent frozen protocol per raw report required")
        evidence = [{"protocol": read(p), "report": read(r)} for p, r in zip(args.protocols, args.reports)]
        options = read_binding(read(args.binding))
        confidence = build_calibration(evidence, options, limits={
            "max_p95_ttft_s": args.max_p95_ttft_s, "max_p95_tpot_s": args.max_p95_tpot_s,
            "max_heldout_ece": args.max_heldout_ece}, ttl_s=args.ttl_s)
        compact = verify_calibration(confidence, options)
        if args.mode == "enforce" and not compact["gpu_qualified"]:
            raise ValueError("CPU/missing stage metrics can calibrate statistics but cannot qualify enforcement")
        config = {"enabled": True, "mode": args.mode,
            "calibration": build_round(evidence, options, now=confidence["created_at"], ttl_s=args.ttl_s),
            "confidence_calibration": confidence}
        if args.publish_slot:
            publish_slot(args.out, config)
        else:
            _write(args.out, config)
        print(json.dumps({"status": "reverified_local_confidence_calibration", "mode": args.mode,
            "gpu_qualified": compact["gpu_qualified"], "published_slot": args.publish_slot,
            "out": str(Path(args.out).resolve())}))
        return 0
    except (ValueError, TypeError, KeyError, OSError, ImportError) as error:
        print(json.dumps({"status": "unverified", "error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

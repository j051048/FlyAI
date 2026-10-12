"""Verify completed static-prefix experiments; emit a default-shadow local policy."""
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
    parser.add_argument("--binding", help="the running service's round_calibration_file.binding.json")
    parser.add_argument("--plan")
    parser.add_argument("--node-configs")
    parser.add_argument("--lease-fences")
    parser.add_argument("--ttl-s", type=float, default=900.)
    parser.add_argument("--mode", choices=("shadow", "enforce"), default="shadow")
    parser.add_argument("--publish-slot", action="store_true",
                        help="atomically publish --out to the running service's private round_calibration_file")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    try:
        if len(args.reports) != len(args.protocols):
            raise ValueError("one independent protocol is required per report")
        from engines.deepseek_v4.v4_round_evidence import build_calibration, publish_slot, read_binding
        from engines.deepseek_v4.v4_adaptive_round import service_options
        from shard.pipeline_plan import validate_plan
        entries = [{"protocol": read(p), "report": read(r)} for p,r in zip(args.protocols, args.reports)]
        env = entries[0]["protocol"]["env"]
        explicit = (args.plan, args.node_configs, args.lease_fences)
        if args.binding:
            if any(explicit):
                raise ValueError("use the exported live binding or all three explicit selected-contract files")
            options = read_binding(read(args.binding))
        elif all(explicit):
            options = service_options(validate_plan(read(args.plan)), read(args.node_configs), read(args.lease_fences),
                depth=int(env["V4_SPEC_DEPTH"]), floor=int(env["V4_REFILL_FLOOR"]), lazy=env["V4_LAZY_DRAFT"] not in ("", "0"))
        else:
            raise ValueError("provide --binding or --plan/--node-configs/--lease-fences")
        calibration = build_calibration(entries, options, ttl_s=args.ttl_s)
        from phase0.v4_benchmark import _write
        config = {"enabled": True, "mode": args.mode, "calibration": calibration}
        if args.publish_slot:
            publish_slot(args.out, config)
        else:
            _write(args.out, config)
        print(json.dumps({"status": "verified_local_calibration", "mode": args.mode,
                          "published_slot": args.publish_slot, "out": str(Path(args.out).resolve())}))
        return 0
    except (KeyError, ValueError, OSError, ImportError, TypeError) as error:
        print(json.dumps({"status": "unverified", "error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

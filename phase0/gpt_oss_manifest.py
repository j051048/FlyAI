"""Build a GPT-OSS cohort from verified local bytes, optionally a measured profile."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, help="full model_cohort JSON")
    parser.add_argument("--calibration")
    parser.add_argument("--profile-out")
    args = parser.parse_args(argv)
    if bool(args.calibration) != bool(args.profile_out):
        parser.error("provide both calibration and profile-out")
    from shard.download_inventory import verify_inventory
    from shard.gpt_oss_contract import build_cohort_from_inventory
    verified = verify_inventory(args.model, verify_files=True)
    cohort = build_cohort_from_inventory(verified)
    Path(args.out).write_text(json.dumps(cohort.to_dict(), indent=2), encoding="utf-8")
    if args.calibration:
        from shard.gpt_oss_planning import inspect_checkpoint, planning_profile
        inventory = inspect_checkpoint(args.model, model_id=cohort.model_id, checkpoint_id=cohort.checkpoint_id, verify_payload=False)
        profile = planning_profile(inventory, json.loads(Path(args.calibration).read_text(encoding="utf-8-sig")))
        Path(args.profile_out).write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(json.dumps({"cohort_id": cohort.cohort_id, "checkpoint_id": cohort.checkpoint_id}))


if __name__ == "__main__": main()

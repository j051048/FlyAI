"""Record a bounded native-stage teacher-forced trace, or compare two saved traces.

Each `trace` invocation owns a new process. This is important for import-time numerical recipes.
Input safetensors: `token_ids` CPU int64 [1,total], plus `hidden` [1,total,hc_mult,dim] for non-head.
No tokenizer is called; frozen IDs/activations are the comparison inputs. This never launches a ring.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "phase0", ROOT / "engines" / "deepseek_v4"):
    sys.path.insert(0, str(path))

ENV_KEYS = {"V4_CUDA_GRAPH", "V4_MOE_IN_GRAPH", "V4_MOE_GROUPED", "V4_MOE_DECODE", "V4_MOE_MULTI",
            "V4_REF_SLIM", "V4_REF_SLIM_NOQAT", "V4_FAST_VERIFY", "V4_FAST_VERIFY_MAX", "V4_GRAPH_MAX",
            "V4_FP8_GEMV", "V4_FP8_SHARED", "V4_FP8_AUTOTUNE", "V4_KERNELS", "V4_DTYPE", "V4_HADAMARD",
            "V4_EXECUTION_PROFILE", "V4_MAX_SEQ", "V4_MOE_MULTI_MAX", "V4_PREFILL_QUERY_CHUNK"}


def read_json(path, limit=32 << 20):
    with open(path, "rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError("diagnostic JSON exceeds byte limit")
    return json.loads(value)


def write_json(path, value):
    destination = Path(path).resolve()
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
    if len(payload) > 32 << 20:
        raise ValueError("diagnostic report exceeds byte limit")
    fd, temp = tempfile.mkstemp(prefix=".numeric-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, destination)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    trace = sub.add_parser("trace", help="owned stage only; GPU use must be explicitly scheduled by the operator")
    source = trace.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="verified native weight pack or stage artifacts")
    source.add_argument("--fixture", action="store_true", help="small CPU reference weights, not production evidence")
    trace.add_argument("--seed", type=int, default=7)
    trace.add_argument("--input", required=True, help="frozen safetensors token_ids and optional hidden")
    trace.add_argument("--prompt-tokens", type=int, required=True)
    trace.add_argument("--max-tokens", type=int, default=256, help="explicit bounded total input budget, at most 16384")
    trace.add_argument("--threads", type=int, default=1, help="fixed Torch CPU thread count (1..32)")
    trace.add_argument("--lo", type=int, default=0)
    trace.add_argument("--hi", type=int, required=True)
    trace.add_argument("--device", default="cpu")
    trace.add_argument("--execution", choices=("configured", "reference", "whole_eager"), default="configured")
    trace.add_argument("--env-file", help="bounded JSON of approved numerical V4 flags; applied before engine import")
    trace.add_argument("--rejection", help="JSON {position,width,accepted} for an actual speculative rewind")
    trace.add_argument("--out", required=True)
    compare = sub.add_parser("compare", help="first per-layer exact divergence; never a speed qualification")
    compare.add_argument("left"); compare.add_argument("right"); compare.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "compare":
        from v4_numeric_diagnostics import compare_traces
        result = compare_traces(read_json(args.left), read_json(args.right))
        write_json(args.out, result)
        print(json.dumps({"exact_match": result["exact_match"], "out": str(Path(args.out).resolve())}))
        return 0 if result["exact_match"] else 1
    if args.env_file:
        env = read_json(args.env_file, 16384)
        if (not isinstance(env, dict) or set(env) - ENV_KEYS or
                any(not isinstance(v, str) or len(v) > 128 for v in env.values())):
            raise ValueError("unsupported diagnostic numerical recipe")
        os.environ.update(env)
    if args.fixture:
        if args.device != "cpu":
            raise ValueError("fixture CLI cannot claim GPU or production checkpoint evidence")
        os.environ["V4_KERNELS"] = "cpu"
    if Path(args.input).stat().st_size > (128 << 20) + 65536:
        raise ValueError("frozen diagnostic input exceeds byte budget")
    from safetensors.torch import load_file
    import torch
    import v4_stage as V
    from v4_numeric_diagnostics import trace_stage
    data = load_file(args.input, device="cpu")
    if set(data) not in ({"token_ids"}, {"token_ids", "hidden"}):
        raise ValueError("input must contain token_ids and optional frozen hidden only")
    if not 1 <= args.threads <= 32 or not 2 <= args.max_tokens <= 16384:
        raise ValueError("invalid diagnostic token/thread budget")
    torch.set_num_threads(args.threads)
    if args.fixture:
        import v4_ref_cpu as R
        model_args = R.cpu_args(max_seq_len=args.max_tokens)
        oracle = R.build_oracle(model_args, args.seed)
    else:
        model_args = V.config(args.checkpoint)
        V.verify_checkpoint_artifacts(args.checkpoint, required=True, lo=args.lo, hi=args.hi,
                                      head=args.lo == 0, tail=args.hi == model_args.n_layers, dspark=False)
    if not 0 <= args.lo < args.hi <= model_args.n_layers:
        raise ValueError("invalid diagnostic stage range")
    stage = V.Stage(args.lo, args.hi, model_args, head=args.lo == 0, tail=args.hi == model_args.n_layers,
                    device=args.device)
    if args.fixture:
        for i, layer in enumerate(stage.layers, args.lo):
            layer.load_state_dict(oracle.layers[i].state_dict(), strict=True)
        if stage.head:
            stage.embed_tokens.load_state_dict(oracle.embed.state_dict(), strict=True)
        if stage.tail:
            stage.norm.load_state_dict(oracle.norm.state_dict(), strict=True)
            stage.lm_head.load_state_dict(oracle.head.state_dict(), strict=True)
            with torch.no_grad():
                for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                    getattr(stage, name).copy_(getattr(oracle, name))
    else:
        stage.load(args.checkpoint)
    result = trace_stage(stage, data["token_ids"], hidden=data.get("hidden"), prompt_tokens=args.prompt_tokens,
                         execution=args.execution, max_tokens=args.max_tokens,
                         rejection=json.loads(args.rejection) if args.rejection else None)
    write_json(args.out, result)
    print(json.dumps({"sha256": result["sha256"], "evidence": result["evidence"], "records": len(result["records"])}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"numeric diagnostic refused: {error}", file=sys.stderr)
        raise SystemExit(2)

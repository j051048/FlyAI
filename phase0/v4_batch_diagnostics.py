"""Owned-stage batch diagnostic; one native weight load and explicit byte quotas.

Input safetensors contains CPU int64 token_ids [2..4,total] and optional frozen
hidden [2..4,total,hc_mult,dim]. This command never connects to a remote host.
"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "phase0", ROOT / "engines" / "deepseek_v4"):
    sys.path.insert(0, str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--fixture", action="store_true", help="CPU fixture only; never hardware/model evidence")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lo", type=int, default=0)
    parser.add_argument("--hi", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"), default="cpu")
    parser.add_argument("--host-mib", type=int, required=True)
    parser.add_argument("--device-mib", type=int, required=True, help="total original stage + resident state + execution workspace bound")
    parser.add_argument("--execution-workspace-mib", type=int, required=True)
    parser.add_argument("--batch-graph", choices=("off", "partial"), default="off")
    parser.add_argument("--graph-mib", type=int, default=0, help="explicit separate graph pool reservation")
    parser.add_argument("--graph-limit", type=int, default=64, help="maximum actual pre/post graphs")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--atol", type=float, default=0.)
    parser.add_argument("--rtol", type=float, default=0.)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--env-file", help="local bounded numerical recipe applied before engine import")
    args = parser.parse_args(argv)
    if (args.fixture and args.device != "cpu" or not 1 <= args.threads <= 32
            or any(not 1 <= value <= 1 << 30 for value in (args.host_mib, args.device_mib, args.execution_workspace_mib))
            or not 1 <= args.max_tokens <= 16384 or not 0 <= args.graph_mib <= 1 << 30
            or (args.batch_graph == "off") != (args.graph_mib == 0)
            or not 2 <= args.graph_limit <= 4096):
        raise ValueError("invalid explicit diagnostic resource/reference bound")
    from v4_numeric_diagnose import read_json, write_json, ENV_KEYS
    if args.env_file:
        env = read_json(args.env_file, 16384)
        if (not isinstance(env, dict) or set(env) - ENV_KEYS
                or any(not isinstance(v, str) or len(v) > 128 for v in env.values())):
            raise ValueError("unsupported diagnostic numerical recipe")
        os.environ.update(env)
    if any(os.environ.get(key, "0") not in ("", "0") for key in
           ("V4_NATIVE_LANES", "V4_NATIVE_BATCH", "V4_NATIVE_BATCH_REFERENCE", "V4_NATIVE_LANE_DEVICE_MIB")):
        raise ValueError("owned diagnostic Stage must start without an installed service lane runner")
    if os.environ.get("V4_CUDA_GRAPH", "0") not in ("", "0", "off"):
        raise ValueError("native batch diagnostic refuses existing scalar graph recipes")
    if os.environ.get("V4_NATIVE_BATCH_GRAPH", "off") != "off" or os.environ.get("V4_NATIVE_BATCH_GRAPH_MIB", "0") != "0":
        raise ValueError("owned diagnostic graph policy must be selected by CLI, outside service environment")
    if args.fixture:
        os.environ["V4_KERNELS"] = "cpu"
    if Path(args.input).stat().st_size > (128 << 20) + 65536:
        raise ValueError("frozen batch diagnostic inputs exceed 128MiB")
    from safetensors.torch import load_file
    import torch
    import v4_stage as V
    from v4_batch_diagnostics import compare_native_batch, comparison_passed
    data = load_file(args.input, device="cpu")
    if set(data) not in ({"token_ids"}, {"token_ids", "hidden"}):
        raise ValueError("frozen input must contain token_ids and optional hidden only")
    torch.set_num_threads(args.threads)
    if args.fixture:
        import v4_ref_cpu as R
        model_args = R.cpu_args(max_seq_len=args.max_tokens)
    else:
        model_args = V.config(args.checkpoint)
        model_args.max_batch_size = 1
        model_args.max_seq_len = args.max_tokens
        V.verify_checkpoint_artifacts(args.checkpoint, required=True, lo=args.lo, hi=args.hi,
                                     head=args.lo == 0, tail=args.hi == model_args.n_layers, dspark=False)
        # Header-only preflight refuses a checkpoint that cannot fit even its
        # selected tensor payload and the explicit execution workspace. The
        # instantiated-storage check below additionally covers real state.
        from v4_resources import inspect_checkpoint, stage_storage_inventory
        inventory = stage_storage_inventory(inspect_checkpoint(args.checkpoint), args.lo, args.hi,
            head=args.lo == 0, tail=args.hi == model_args.n_layers)
        payload = sum(value["storage_bytes"] for value in inventory["tensors"].values())
        if args.device.startswith("cuda") and payload + ((args.execution_workspace_mib + args.graph_mib) << 20) > args.device_mib << 20:
            raise ValueError("native checkpoint payload/graph/execution workspace exceeds declared device quota before loading")
    if not 0 <= args.lo < args.hi <= model_args.n_layers:
        raise ValueError("invalid owned diagnostic stage range")
    previous_device = torch.get_default_device()
    cache_names = ("precompute_freqs_cis", "get_window_topk_idxs", "get_compress_topk_idxs")
    for name in cache_names:
        getattr(V.ref(), name).cache_clear()
    try:
        # Native rotary/index-list helpers use the ambient default device.
        # Inputs remain frozen CPU tensors, while this owned execution interval
        # gives both the unchanged scalar oracle and the batch the same device.
        torch.set_default_device(args.device)
        return _run_owned(args, model_args, data, V,
                          compare_native_batch, comparison_passed, write_json)
    finally:
        for name in cache_names:
            getattr(V.ref(), name).cache_clear()
        torch.set_default_device(previous_device)


def _run_owned(args, model_args, data, V, compare_native_batch, comparison_passed, write_json):
    import torch
    if args.fixture:
        import v4_ref_cpu as R
        with torch.device("cpu"):
            oracle = R.build_oracle(model_args, args.seed)
    stage = V.Stage(args.lo, args.hi, model_args, head=args.lo == 0,
                    tail=args.hi == model_args.n_layers, device=args.device)
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
    if args.fixture:
        del oracle  # The comparison retains only its one owned Stage weight set.
    report = compare_native_batch(stage, data["token_ids"], hidden=data.get("hidden"),
        host_bytes=args.host_mib << 20, device_bytes=args.device_mib << 20,
        execution_workspace_bytes=args.execution_workspace_mib << 20,
        max_tokens=args.max_tokens, atol=args.atol, rtol=args.rtol,
        topk=args.topk, reference=args.device == "cpu", graph_mode=args.batch_graph,
        graph_bytes=args.graph_mib << 20, max_graphs=args.graph_limit)
    write_json(args.out, report)
    import json
    print(json.dumps({"out": str(Path(args.out).resolve()), "sha256": report["sha256"],
                      "evidence": report["evidence"], "qualification": report["qualification"]}))
    good = comparison_passed(report) and report["allocator"]["within_device_quota"] is not False
    return 0 if good else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"batch diagnostic refused: {error}", file=sys.stderr)
        raise SystemExit(2)

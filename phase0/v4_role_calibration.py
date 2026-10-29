"""Owned Stage role-cost calibration with explicit completion-fenced wall partitions.

Frozen safetensors contains token_ids [1, context+frame_tokens] and, for a
non-head stage, matching hidden activations. No network, download or GPU rental.
This measures one full-commit teacher-forced frame, not request throughput.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "engines" / "deepseek_v4"):
    sys.path.insert(0, str(directory))


def calibrate(stage, token_ids, *, hidden=None, context_tokens, frame_tokens=1,
              decode_mode="greedy", repeats=5, warmup=1, drafter=None):
    import torch
    from shard.runtime_profile import RuntimeProfiler
    from shard.stage_cost import MODES, surface_from_frames
    from v4_resources import runtime_config_payload, runtime_config_identity
    from v4_batch_diagnostics import _hardware
    from v4_numeric_diagnostics import tensor_identity
    if (decode_mode not in MODES or type(repeats) is not int or not 5 <= repeats <= 16
            or type(warmup) is not int or not 1 <= warmup <= 4
            or type(context_tokens) is not int or context_tokens < 1
            or type(frame_tokens) is not int or not 1 <= frame_tokens <= 32
            or decode_mode == "greedy" and frame_tokens != 1
            or not torch.is_tensor(token_ids) or token_ids.device.type != "cpu"
            or token_ids.dtype != torch.long or tuple(token_ids.shape) != (1, context_tokens + frame_tokens)
            or context_tokens + frame_tokens >= stage.args.max_seq_len
            or bool((token_ids < 0).any()) or bool((token_ids >= stage.args.vocab_size).any())):
        raise ValueError("bounded frozen teacher-forced role-calibration geometry required")
    if getattr(stage, "_native_lanes", None) is not None or getattr(stage, "_numeric_layer_observer", None) is not None:
        raise ValueError("role calibration requires an idle independently owned Stage")
    if hidden is None and not stage.head:
        raise ValueError("non-head calibration needs frozen predecessor hidden inputs")
    if hidden is not None and (hidden.device.type != "cpu" or hidden.dtype != stage.dtype
            or tuple(hidden.shape) != (1, context_tokens + frame_tokens, stage.args.hc_mult, stage.args.dim)
            or not bool(torch.isfinite(hidden).all())):
        raise ValueError("frozen hidden shape/dtype/finite binding differs from the owned Stage")
    needs_draft = stage.tail and decode_mode in ("dspark", "pipelined")
    if (drafter is not None) != needs_draft:
        raise ValueError("tail draft-mode calibration requires its actually loaded MTP")
    if drafter is not None and context_tokens + frame_tokens + drafter.block_size + 1 > stage.args.max_seq_len:
        raise ValueError("role calibration leaves insufficient MTP draft horizon")
    role = {"head": stage.head, "tail": stage.tail, "draft": needs_draft}
    fence = lambda: torch.cuda.synchronize(stage.device) if str(stage.device).startswith("cuda") else None
    profiler = RuntimeProfiler(role_cost_capacity=16)
    ids = token_ids.to(stage.device)
    frozen = hidden.to(stage.device) if hidden is not None else None
    metadata = dict(decode_mode=decode_mode, role=role, layer_start=stage.lo, layer_end=stage.hi,
                    context_tokens=context_tokens, frame_tokens=frame_tokens, warmness="warm")

    def prepare():
        stage.reset()
        stage._spec, stage._dspark = False, needs_draft
        if drafter is not None:
            drafter.reset()
        with torch.no_grad():
            prompt = ids[:, :context_tokens]
            h = stage.embed(prompt) if frozen is None else frozen[:, :context_tokens]
            out = stage.forward(h, prompt, 0)
            if drafter is not None:
                pred = stage.logits_all(out, full_logits=False).argmax(-1)
                drafter.prefill(pred, stage.tail_main_hidden())
        stage._spec = decode_mode != "greedy"
        fence()

    output_digest = hashlib.sha256()
    def execute(weights):
        prepare()
        current = ids[:, context_tokens:]
        with torch.no_grad(), profiler.cost_frame(**metadata, layer_weights=weights, completion=fence) as frame:
            if stage.head:
                with frame.phase("embedding"):
                    h = stage.embed(current)
            else:
                h = frozen[:, context_tokens:]
            with frame.phase("main"):
                out = stage.forward(h, current, context_tokens)
            predicted = None
            if stage.tail:
                with frame.phase("lm_head"):
                    # Keep the actual verifier's per-position output-head shape.
                    predicted = torch.stack([stage.logits_all(out[:, i:i+1], full_logits=False).argmax(-1)
                                             for i in range(frame_tokens)], dim=1)
            with frame.phase("sync"):
                stage.commit(context_tokens + frame_tokens)
            if drafter is not None:
                with frame.phase("draft"):
                    drafter.advance_and_draft(predicted, stage.tail_main_hidden(), context_tokens)
        output_digest.update(out.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())

    try:
        for _ in range(warmup):
            execute([1.] * len(stage.layers))
        profiler.reset()
        # Independently measure placement weights. Per-layer fences are not
        # present in the full-frame measurements exported below.
        prepare()
        weights, previous = [], [time.perf_counter()]
        def observed(layer, pos, out, path):
            fence(); now = time.perf_counter()
            weights.append(max(1e-9, (now - previous[0]) * 1000)); previous[0] = now
        stage._numeric_layer_observer = observed
        current = ids[:, context_tokens:]
        h = stage.embed(current) if stage.head else frozen[:, context_tokens:]
        fence(); previous[0] = time.perf_counter()
        with torch.no_grad():
            stage.forward(h, current, context_tokens)
        stage._numeric_layer_observer = None
        if len(weights) != len(stage.layers):
            # Chunk fallback may notify once per token. Sum the measured layer
            # contributions in execution order, without changing frame timing.
            if len(weights) != len(stage.layers) * frame_tokens:
                raise ValueError("layer-weight probe did not cover the owned span")
            weights = [sum(weights[i::len(stage.layers)]) for i in range(len(stage.layers))]
        for _ in range(repeats):
            execute(weights)
        frames = profiler.snapshot()["role_cost_frames"]["frames"]
        runtime = runtime_config_payload(stage)
        source = hashlib.sha256(json.dumps(runtime["engine_source_sha256"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        surface = surface_from_frames(frames, source_sha256=source)
        first = surface["samples"][0]
        artifact = (getattr(stage, "_artifact_verification", None) or {}).get("catalog", {})
        hardware = _hardware(stage, reference=not str(stage.device).startswith("cuda"))
        return {"schema": "flyai-role-calibration/1", "cost_surface": surface,
                "runtime_config_sha256": runtime_config_identity(stage), "runtime_config": runtime,
                "hardware": hardware, "checkpoint_id": artifact.get("checkpoint_id"),
                "model_id": artifact.get("model_id"), "layer_start": stage.lo, "layer_end": stage.hi,
                "measured_at": min(frame["measured_at"] for frame in frames),
                "frame_ms": sum(first["frame_samples_ms"]) / repeats,
                "frames": frames, "inputs": {"token_ids": tensor_identity(token_ids, max_bytes=128 << 20),
                    "hidden": tensor_identity(hidden, max_bytes=128 << 20) if hidden is not None else None},
                "output_sha256": output_digest.hexdigest(), "release_qualified": False,
                "measurement_scope": "owned_completion_fenced_full_commit_teacher_forced_stage_frame",
                "evidence": "verified_native_cuda_stage_calibration" if artifact and hardware["identity_complete"] else "unqualified_fixture_or_hardware_identity"}
    finally:
        stage._numeric_layer_observer = None
        stage.reset()
        if drafter is not None:
            drafter.reset()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint"); src.add_argument("--fixture", action="store_true")
    parser.add_argument("--input", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--lo", type=int, default=0); parser.add_argument("--hi", type=int, required=True)
    parser.add_argument("--context", type=int, required=True); parser.add_argument("--frame-tokens", type=int, default=1)
    parser.add_argument("--mode", choices=("greedy", "spec", "dspark", "pipelined"), default="greedy")
    parser.add_argument("--max-seq", type=int, default=8192); parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--host-mib", type=int, required=True)
    parser.add_argument("--device-mib", type=int, required=True); parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.fixture and args.device != "cpu" or min(args.host_mib, args.device_mib) <= 0 or not 8 <= args.max_seq <= 16384:
        raise ValueError("invalid explicit owned calibration resource bounds")
    if any(os.environ.get(name, "0") not in ("", "0") for name in ("V4_NATIVE_LANES", "V4_NATIVE_BATCH")):
        raise ValueError("owned calibration cannot run with service lanes installed")
    if Path(args.input).stat().st_size > min(128 << 20, args.host_mib << 19):
        raise ValueError("frozen calibration input exceeds its host preflight limit")
    if args.fixture:
        os.environ["V4_KERNELS"] = "cpu"
    import torch
    torch.set_num_threads(1)
    from safetensors.torch import load_file
    import v4_stage as V
    from v4_dspark_draft import DSparkTail, ALIAS_KEYS
    data = load_file(args.input, device="cpu")
    if set(data) not in ({"token_ids"}, {"token_ids", "hidden"}):
        raise ValueError("frozen inputs may contain token_ids and hidden only")
    if args.fixture:
        import v4_ref_cpu as R
        model_args = R.cpu_args(n_layers=4, dspark_target_layer_ids=(1,2,3), max_seq_len=args.max_seq,
                                compress_ratios=(0,4,8,0,0,0))
    else:
        model_args = V.config(args.checkpoint)
        model_args.max_batch_size, model_args.max_seq_len = 1, args.max_seq
    if not 0 <= args.lo < args.hi <= model_args.n_layers:
        raise ValueError("invalid owned Stage range")
    draft = args.mode in ("dspark", "pipelined") and args.hi == model_args.n_layers
    if not args.fixture:
        V.verify_checkpoint_artifacts(args.checkpoint, required=True, lo=args.lo, hi=args.hi,
            head=args.lo == 0, tail=args.hi == model_args.n_layers, dspark=draft)
        from v4_resources import inspect_checkpoint, stage_storage_inventory
        inventory = stage_storage_inventory(inspect_checkpoint(args.checkpoint), args.lo, args.hi,
            head=args.lo == 0, tail=args.hi == model_args.n_layers, dspark=draft)
        # In particular the shared output head is instantiated in float32 even
        # when the on-disk checkpoint stores bf16. Header payload alone is not a
        # safe lower bound on the constructed parameter allocation.
        floor = inventory["resident_parameter_floor_bytes"] + inventory["routed_expert_storage_bytes"]
        if floor > args.device_mib << 20:
            raise ValueError("checkpoint parameter floor exceeds owned device budget before loading")
    old = torch.get_default_device()
    cache_names = ("precompute_freqs_cis", "get_window_topk_idxs", "get_compress_topk_idxs", "get_dspark_topk_idxs")
    for name in cache_names: getattr(V.ref(), name).cache_clear()
    try:
        torch.set_default_device(args.device)
        stage = V.Stage(args.lo, args.hi, model_args, head=args.lo == 0,
                        tail=args.hi == model_args.n_layers, dspark=draft, device=args.device)
        dr = DSparkTail(stage, temperature=0.) if draft else None
        if args.fixture:
            with torch.device("cpu"): oracle = R.build_oracle(model_args, args.seed)
            for i, layer in enumerate(stage.layers, args.lo): layer.load_state_dict(oracle.layers[i].state_dict())
            if stage.embed_tokens is not None: stage.embed_tokens.load_state_dict(oracle.embed.state_dict())
            if stage.tail:
                stage.norm.load_state_dict(oracle.norm.state_dict()); stage.lm_head.load_state_dict(oracle.head.state_dict())
                with torch.no_grad():
                    for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"): getattr(stage, name).copy_(getattr(oracle, name))
            if dr:
                for block, source in zip(dr.mtp, oracle.mtp):
                    block.load_state_dict({k:v for k,v in source.state_dict().items() if k not in ALIAS_KEYS}, strict=False)
            del oracle
        else:
            stage.load(args.checkpoint)
            if dr: dr.load(args.checkpoint)
        from v4_resources import measure_stage_resources
        measured = measure_stage_resources(stage, draft=dr, checkpoint_id="owned-role-calibration")
        storage = measured["module_storage"]
        if storage["host_bytes"] > args.host_mib << 20 or storage["gpu_bytes"] > args.device_mib << 20:
            raise ValueError("instantiated original Stage/MTP storage exceeds the owned quota")
        result = calibrate(stage, data["token_ids"], hidden=data.get("hidden"), context_tokens=args.context,
            frame_tokens=args.frame_tokens, decode_mode=args.mode, repeats=args.repeats, drafter=dr)
        Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"out": str(Path(args.out).resolve()), "evidence": result["evidence"], "release_qualified": False}))
        return 0
    finally:
        for name in cache_names: getattr(V.ref(), name).cache_clear()
        torch.set_default_device(old)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print("role calibration refused: " + str(error), file=sys.stderr)
        raise SystemExit(2)

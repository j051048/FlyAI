"""Owned native-MTP scalar/full-chain A/B, with one model allocation at a time.

Frozen safetensors fields: prefill_ids [1], prompt_taps [1,P,H],
committed_ids [1,N], main_taps [1,N,H], run_lengths [rounds] int64.
Taps and committed tokens must come from the same frozen target execution.
The probe validates byte parity; it does not prove that provenance or speed.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "engines" / "deepseek_v4"):
    sys.path.insert(0, str(directory))

SCHEMA = "flyai-v4-native-draft-chain-probe/1"
FIELDS = {"prefill_ids", "prompt_taps", "committed_ids", "main_taps", "run_lengths"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            value.update(data)
    return value.hexdigest()


def _sources():
    from v4_observability import source_inventory
    return {**source_inventory(), "phase0/v4_draft_chain_probe.py": _file_hash(__file__)}


def _validate(data, model_args, dtype, *, max_seq, host_bytes):
    import torch
    if set(data) != FIELDS or any(not torch.is_tensor(value) or value.device.type != "cpu" for value in data.values()):
        raise ValueError("exact frozen CPU draft-chain fields required")
    pre, prompt, ids, taps, runs = (data[k] for k in
        ("prefill_ids", "prompt_taps", "committed_ids", "main_taps", "run_lengths"))
    hidden = len(model_args.dspark_target_layer_ids) * model_args.dim
    if (pre.dtype != torch.long or tuple(pre.shape) != (1,) or ids.dtype != torch.long or ids.ndim != 2
            or ids.shape[0] != 1 or not 2 <= ids.shape[1] <= 16384
            or prompt.dtype != dtype or prompt.ndim != 3 or prompt.shape[0] != 1
            or prompt.shape[2] != hidden or not 1 <= prompt.shape[1] <= 16384
            or taps.dtype != dtype or tuple(taps.shape) != (1, ids.shape[1], hidden)
            or runs.dtype != torch.long or runs.ndim != 1 or not 2 <= runs.numel() <= 128
            or bool((runs < 1).any()) or bool((runs > model_args.dspark_block_size + 1).any())
            or int(runs.sum()) != ids.shape[1]
            or prompt.shape[1] + ids.shape[1] + model_args.dspark_block_size + 1 > max_seq):
        raise ValueError("frozen draft-chain geometry or horizon differs from native args")
    if (bool((pre < 0).any()) or bool((pre >= model_args.vocab_size).any())
            or bool((ids < 0).any()) or bool((ids >= model_args.vocab_size).any())
            or not bool(torch.isfinite(prompt).all()) or not bool(torch.isfinite(taps).all())):
        raise ValueError("frozen draft-chain input contains invalid tokens or nonfinite taps")
    if sum(value.numel() * value.element_size() for value in data.values()) > host_bytes // 4:
        raise ValueError("frozen draft-chain tensors exceed host input allowance")
    return [int(value) for value in runs.tolist()]


def _environment(args, arm):
    env = {key: value for key, value in os.environ.items() if not key.startswith("V4_")}
    # This experiment deliberately owns its recipe. No ambient graph, adapter,
    # service lane, wire or expert-storage flag may silently contaminate an arm.
    env.update(V4_KERNELS="cpu" if args.fixture else "tilelang", V4_DEV=args.device,
               V4_DTYPE="bfloat16", V4_CUDA_GRAPH="0", V4_DSPARK_GRAPH="0",
               V4_DSPARK_FAST="1" if arm == "chain" else "0",
               V4_DSPARK_MOE="1" if arm == "chain" and not args.fixture else "0",
               V4_DSPARK_CHAIN_GRAPH="1" if arm == "chain" else "0",
               V4_DSPARK_CHAIN_MIB=str(args.graph_mib if arm == "chain" else 0),
               V4_DSPARK_CHAIN_REFERENCE="1" if arm == "chain" and args.fixture else "0",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1")
    return env


def _child_args(args, arm):
    source = ["--fixture"] if args.fixture else ["--checkpoint", str(Path(args.checkpoint).resolve())]
    values = {"input": str(Path(args.input).resolve()), "out-dir": str(Path(args.out_dir).resolve()),
        "lo": args.lo, "hi": args.hi, "device": args.device, "host-mib": args.host_mib,
        "device-mib": args.device_mib, "graph-mib": args.graph_mib, "max-seq": args.max_seq,
        "seed": args.seed, "arm": arm}
    command = [*source, *(value for name, setting in values.items() for value in ("--"+name, str(setting)))]
    return [*command, "--execute-gpu"] if args.execute_gpu else command


def _load_args(args):
    import v4_stage as V
    if args.fixture:
        import v4_ref_cpu as R
        model_args = R.cpu_args(max_batch_size=1, max_seq_len=args.max_seq)
    else:
        model_args = V.config(args.checkpoint)
        model_args.max_batch_size, model_args.max_seq_len = 1, args.max_seq
    if args.hi != model_args.n_layers or not 0 <= args.lo < args.hi:
        raise ValueError("draft-chain probe owns a valid tail span")
    if any(not args.lo <= layer < args.hi for layer in model_args.dspark_target_layer_ids):
        raise ValueError("tail span must hold every native target tap layer")
    return model_args


def _arm(args):
    import torch
    from safetensors.torch import load_file, save_file
    import v4_stage as V
    import v4_dspark_draft as D
    import v4_dspark_fast as F
    import v4_dspark_moe as P
    from v4_numeric_diagnostics import tensor_identity
    from v4_resources import measure_stage_resources
    from v4_batch_diagnostics import _hardware
    import v4_kernels_cpu as K
    source = _sources()
    model_args = _load_args(args)
    dtype = torch.bfloat16
    data = load_file(args.input, device="cpu")
    lengths = _validate(data, model_args, dtype, max_seq=args.max_seq, host_bytes=args.host_mib << 20)
    identity = None
    if not args.fixture:
        proof = V.verify_checkpoint_artifacts(args.checkpoint, required=True, lo=args.lo,
            hi=args.hi, head=args.lo == 0, tail=True, dspark=True)
        identity = {key: proof[key] for key in ("checkpoint_id", "manifest_sha256")}
        from v4_resources import inspect_checkpoint, stage_storage_inventory
        inventory = stage_storage_inventory(inspect_checkpoint(args.checkpoint), args.lo, args.hi,
            head=args.lo == 0, tail=True, dspark=True)
        floor = inventory["resident_parameter_floor_bytes"] + inventory["routed_expert_storage_bytes"]
        if floor + (args.graph_mib << 20) > args.device_mib << 20:
            raise ValueError("native parameters and graph reservation exceed device preflight quota")
    if args.execute_gpu:
        if not torch.cuda.is_available():
            raise ValueError("explicit GPU probe needs available CUDA")
        torch.cuda.set_device(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    torch.set_num_threads(1)
    previous = torch.get_default_device()
    caches = ("precompute_freqs_cis", "get_window_topk_idxs", "get_compress_topk_idxs", "get_dspark_topk_idxs")
    for name in caches:
        getattr(V.ref(), name).cache_clear()
    try:
        torch.set_default_device(args.device)
        stage = V.Stage(args.lo, args.hi, model_args, head=args.lo == 0, tail=True,
                        dspark=True, device=args.device)
        tail = D.DSparkTail(stage, temperature=0.)
        # Native packed MoE installs BEFORE loading, retaining one weight bank.
        installed = P.install_drafter(tail) if args.arm == "chain" else 0
        if args.fixture:
            import v4_ref_cpu as R
            with torch.device("cpu"):
                oracle = R.build_oracle(model_args, args.seed)
            for index, layer in enumerate(stage.layers, args.lo):
                layer.load_state_dict(oracle.layers[index].state_dict())
            stage.embed_tokens.load_state_dict(oracle.embed.state_dict())
            stage.norm.load_state_dict(oracle.norm.state_dict())
            stage.lm_head.load_state_dict(oracle.head.state_dict())
            with torch.no_grad():
                for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                    getattr(stage, name).copy_(getattr(oracle, name))
            for target, source_block in zip(tail.mtp, oracle.mtp):
                target.load_state_dict({key: value for key, value in source_block.state_dict().items()
                    if key not in D.ALIAS_KEYS}, strict=False)
            del oracle
            identity = {"fixture": True, "seed": args.seed, "args_sha256": _digest(asdict(model_args))}
        else:
            stage.load(args.checkpoint)
            tail.load(args.checkpoint)
        if args.arm == "chain":
            F.install(D)
        measured = measure_stage_resources(stage, draft=tail, checkpoint_id="owned-native-chain-probe")
        storage = measured["module_storage"]
        if storage["host_bytes"] + sum(value.numel()*value.element_size() for value in data.values()) > args.host_mib << 20:
            raise ValueError("loaded probe/input storage exceeds owned host quota")
        if storage["gpu_bytes"] > args.device_mib << 20:
            raise ValueError("loaded probe storage exceeds owned device quota")
        outputs, descriptors, output_bytes = {}, {}, 0

        def record(name, value):
            nonlocal output_bytes
            size = value.numel()*value.element_size()
            output_bytes += size
            if output_bytes > (args.host_mib << 20) // 4:
                raise ValueError("probe raw output/KV archive exceeds its host allowance")
            descriptors[name] = tensor_identity(value, max_bytes=(args.host_mib << 20) // 4)
            if descriptors[name]["nan"] or descriptors[name]["infinity"]:
                raise ValueError("native draft-chain produced nonfinite output or KV")
            outputs[name] = value.detach().cpu().clone().contiguous()

        with torch.no_grad():
            # A second independent replay verifies graph inputs reset/re-read.
            for repeat in range(2):
                tail.reset()
                tail.prefill(data["prefill_ids"].to(args.device), data["prompt_taps"].to(args.device))
                offset, position = 0, data["prompt_taps"].shape[1]
                for index, length in enumerate(lengths):
                    tail.advance_and_draft(data["committed_ids"][:, offset:offset+length].to(args.device),
                        data["main_taps"][:, offset:offset+length].to(args.device), start_pos=position)
                    for name, value in zip(("ids", "logits", "confidence"), tail.last_spec):
                        record(f"r{repeat}.step{index}.{name}", value)
                    for layer, block in enumerate(tail.mtp):
                        record(f"r{repeat}.step{index}.kv{layer}", block.attn.kv_cache)
                    offset, position = offset + length, position + length
        status = tail._draft_chain.status() if tail._draft_chain is not None else {"enabled": False}
        peak = 0
        if args.execute_gpu:
            torch.cuda.synchronize(args.device)
            peak = int(torch.cuda.max_memory_reserved(args.device))
            if peak > args.device_mib << 20:
                raise ValueError("native draft-chain allocator peak exceeds owned device quota")
            if args.arm == "chain" and (installed != len(tail.mtp) or status["captures"] < 1
                    or status["replays"] <= status["captures"] or status["capture_failures"]
                    or status["fallbacks"] or status["poisoned"]):
                raise ValueError("native packed-MTP graph did not capture and reuse every requested path")
        if _sources() != source:
            raise ValueError("probe source files changed during arm execution")
        result = {"schema": SCHEMA, "arm": args.arm, "sources": source, "model": identity,
            "args_sha256": _digest(asdict(model_args)), "input_sha256": _file_hash(args.input),
            "descriptors": descriptors, "chain": status, "pair_blocks_installed": installed,
            "pair_coverage": P.coverage_details(tail), "kernel_backend": K.backend(),
            "hardware": _hardware(stage, reference=args.fixture), "allocator_peak_bytes": peak,
            "release_qualified": False, "teacher_provenance_verified": False}
        out = Path(args.out_dir)
        save_file(outputs, str(out / (args.arm + ".safetensors")))
        result["output_archive_sha256"] = _file_hash(out / (args.arm + ".safetensors"))
        (out / (args.arm + ".json")).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        return result
    finally:
        for name in caches:
            getattr(V.ref(), name).cache_clear()
        torch.set_default_device(previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint"); source.add_argument("--fixture", action="store_true")
    parser.add_argument("--input", required=True); parser.add_argument("--out-dir", required=True)
    parser.add_argument("--lo", type=int, default=0); parser.add_argument("--hi", type=int, required=True)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--host-mib", type=int, required=True); parser.add_argument("--device-mib", type=int, required=True)
    parser.add_argument("--graph-mib", type=int, required=True); parser.add_argument("--max-seq", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--arm", choices=("scalar", "chain"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (not 1 <= args.host_mib <= 65536 or not 1 <= args.device_mib <= 1048576
            or not 1 <= args.graph_mib < args.device_mib or not 8 <= args.max_seq <= 16384
            or args.fixture and (args.device != "cpu" or args.execute_gpu)
            or not args.fixture and (not args.execute_gpu or not args.device.startswith("cuda:"))):
        raise ValueError("explicit independent CPU fixture or owned --execute-gpu resource bounds required")
    if Path(args.input).stat().st_size > min(128 << 20, (args.host_mib << 20) // 4):
        raise ValueError("frozen input file exceeds declared host bound before loading")
    if args.arm:
        # Replacing ambient env happens before any engine/torch imports in child.
        owned = _environment(args, args.arm)
        os.environ.clear(); os.environ.update(owned)
        _arm(args)
        return 0
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    source = _sources()
    frozen_input = _file_hash(args.input)
    for arm in ("scalar", "chain"):
        with (out / (arm+".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *_child_args(args, arm)],
                cwd=ROOT, env=_environment(args, arm), stdout=log, stderr=subprocess.STDOUT,
                timeout=7200, check=False)
        if result.returncode:
            raise ValueError(f"draft-chain {arm} arm failed; inspect its local log")
    control = json.loads((out/"scalar.json").read_text(encoding="utf-8"))
    actual = json.loads((out/"chain.json").read_text(encoding="utf-8"))
    if (_sources() != source or control["sources"] != source or actual["sources"] != source
            or _file_hash(args.input) != frozen_input
            or control["input_sha256"] != frozen_input or actual["input_sha256"] != frozen_input
            or control["model"] != actual["model"] or control["args_sha256"] != actual["args_sha256"]
            or control["output_archive_sha256"] != _file_hash(out/"scalar.safetensors")
            or actual["output_archive_sha256"] != _file_hash(out/"chain.safetensors")):
        raise ValueError("source/model/config/input changed between independent probe arms")
    compared = set(control["descriptors"]) | set(actual["descriptors"])
    mismatched = [name for name in sorted(compared)
        if control["descriptors"].get(name) != actual["descriptors"].get(name)]
    gpu = bool(args.execute_gpu and not mismatched and control["kernel_backend"] == actual["kernel_backend"] == "tilelang"
        and control["hardware"]["identity_complete"] and actual["hardware"]["identity_complete"]
        and control["hardware"]["gpu_uuid"] == actual["hardware"]["gpu_uuid"])
    report = {"schema": SCHEMA, "byte_parity": not mismatched, "mismatched": mismatched,
        "tensors_compared": len(compared), "sources": source, "model": control["model"],
        "input_sha256": frozen_input, "chain": actual["chain"], "gpu_qualified": gpu,
        "output_archives": {arm: value["output_archive_sha256"] for arm, value in (("scalar", control), ("chain", actual))},
        "release_qualified": False, "teacher_provenance_verified": False,
        "evidence": "native_checkpoint_cuda_mtp_chain_byte_parity" if gpu else "unqualified_cpu_reference_or_hardware",
        "measurement_scope": "two_sequential_independent_model_processes; no throughput claim"}
    (out/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("byte_parity", "tensors_compared", "gpu_qualified", "release_qualified")}))
    if mismatched or args.execute_gpu and not gpu:
        raise ValueError("native draft-chain parity or hardware identity gate failed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print("draft-chain probe refused: "+str(error), file=sys.stderr)
        raise SystemExit(2)

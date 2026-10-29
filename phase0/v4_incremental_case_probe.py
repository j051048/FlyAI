"""Owned-stage native incremental cases, actual byte hashes, optional GPU signing.

One Stage/weight allocation is reused sequentially. Input safetensors contain
token_ids[1,N] int64 and, for a non-head stage, hidden[1,N,hc,dim] in its native
dtype. other_token_ids/other_hidden are required for the isolation/cancel arms.
CPU fixture runs never emit signed GPU cases. A mismatch is a pending result.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "engines" / "deepseek_v4"):
    sys.path.insert(0, str(directory))

from v4_incremental_evidence import CASE_SCHEMA, CASE_DOMAIN, CASES, OPERATIONS, digest


def compare_cases(stage, data, *, parent_count, host_bytes):
    """Execute actual native reference/candidate math; no claimed pass inputs."""
    import torch
    from v4_conversation_cache import _buffers
    from v4_numeric_diagnostics import tensor_identity
    from shard.service_queue import JobCancelled
    ids = data.get("token_ids")
    other = data.get("other_token_ids")
    if (not torch.is_tensor(ids) or ids.dtype != torch.long or ids.ndim != 2 or ids.shape[0] != 1
            or not 0 < parent_count < ids.shape[1] <= min(16384, stage.args.max_seq_len)
            or not torch.is_tensor(other) or other.dtype != torch.long or other.shape != ids.shape
            or bool((ids < 0).any()) or bool((ids >= stage.args.vocab_size).any())
            or bool((other < 0).any()) or bool((other >= stage.args.vocab_size).any()) or torch.equal(ids, other)):
        raise ValueError("two distinct frozen native int64 token sequences and bounded parent required")
    if set(data) != ({"token_ids", "other_token_ids"} if stage.head else
                     {"token_ids", "other_token_ids", "hidden", "other_hidden"}):
        raise ValueError("frozen input fields must match the owned head/non-head role")
    if not stage.head:
        for field in ("hidden", "other_hidden"):
            value = data[field]
            if (not torch.is_tensor(value) or tuple(value.shape) != (1, ids.shape[1], stage.args.hc_mult, stage.args.dim)
                    or value.dtype != stage.dtype or not bool(torch.isfinite(value).all())):
                raise ValueError("frozen native hidden input shape/dtype/finite contract differs")
    full = ids.shape[1]
    total_state = sum(value.numel() * value.element_size() for value in _buffers(stage, None).values())
    input_bytes = sum(value.numel() * value.element_size() for value in data.values())
    if total_state + input_bytes + (8 << 20) > host_bytes:
        raise ValueError("actual native snapshot, frozen inputs and hash workspace exceed explicit host quota")
    if (getattr(stage, "_dspark_capable", False) or getattr(stage, "_native_lanes", None)
            or getattr(stage, "_kv_runtime", None) or getattr(stage, "_expert_placement", "gpu") != "gpu"):
        raise ValueError("incremental case probe currently qualifies scalar resident-KV greedy stages without MTP")
    if stage._spec_depth < min(full - parent_count + 1, 3):
        raise ValueError("actual rollback depth is too small for the owned probe")
    inputs_sha = digest({name: tensor_identity(value, max_bytes=max(1, input_bytes)) for name, value in sorted(data.items())})

    def run(start, stop, *, unrelated=False):
        source = other if unrelated else ids
        hidden_key = "other_hidden" if unrelated else "hidden"
        value = source[:, start:stop].to(stage.device)
        hidden = stage.embed(value) if stage.head else data[hidden_key][:, start:stop].to(stage.device)
        return stage.forward(hidden, value, start)

    def summarize(hidden):
        output = stage.logits_all(hidden, full_logits=False) if stage.tail else hidden[:, -1:]
        token = output.argmax(-1) if stage.tail else torch.empty(0, dtype=torch.long, device=stage.device)
        states = {name: tensor_identity(value, max_bytes=max(1, total_state))
                  for name, value in _buffers(stage, None, frontier=full).items()}
        output_identity = tensor_identity(output, max_bytes=128 << 20)
        if output_identity["nan"] or output_identity["infinity"] or any(value["nan"] for value in states.values()):
            raise ValueError("native case produced nonfinite output or NaN persistent state")
        result = {"state_sha256": digest(states),
                  "logits_sha256": digest(output_identity),
                  "tokens_sha256": digest(tensor_identity(token, max_bytes=128 << 20))}
        return result, states

    def fresh():
        stage._spec = stage._dspark = stage._replaying = False
        stage.reset()

    def snapshot():
        return {name: value.detach().to("cpu", copy=True)
                for name, value in _buffers(stage, None, frontier=parent_count).items()}

    def restore(saved):
        fresh()
        for name, target in _buffers(stage, None, frontier=parent_count).items():
            target.copy_(saved[name])
        stage._pos = parent_count

    result = []
    with torch.no_grad():
        for name in CASES:
            fresh()
            reference, reference_states = summarize(run(0, full))
            fresh()
            if name == "reset":
                run(0, full, unrelated=True)
                fresh()
                hidden = run(0, full)
            else:
                run(0, parent_count)
                saved = snapshot()
                if name == "rollback":
                    stage._spec = True
                    wrong_end = min(full, parent_count + 2)
                    for position in range(parent_count, wrong_end):
                        run(position, position + 1, unrelated=True)
                    stage._seek(parent_count)
                    stage._spec = False
                elif name == "cancel":
                    try:
                        run(parent_count, parent_count + 1, unrelated=True)
                        raise JobCancelled("owned numeric probe cancellation after actual native work")
                    except JobCancelled:
                        restore(saved)
                elif name == "tenant_isolation":
                    fresh()
                    run(0, full, unrelated=True)
                    restore(saved)
                # Append keeps all original one-token recurrence/selection
                # semantics. The full-prefill oracle above is independent.
                for position in range(parent_count, full):
                    hidden = run(position, position + 1)
                del saved
            candidate, candidate_states = summarize(hidden)
            ratios = [layer.attn.compress_ratio for layer in stage.layers]
            active_ratios = [r for r in ratios if r]
            covered = name != "compression_boundary" or not active_ratios or any(parent_count // r < full // r for r in active_ratios)
            result.append({"case": name, "reference": reference, "candidate": candidate,
                "passed": covered and reference == candidate, "covered": covered,
                "first_state_difference": next((key for key in reference_states if reference_states[key] != candidate_states[key]), None),
                "inputs_sha256": inputs_sha, "parent_count": parent_count, "full_count": full,
                "operation_sequence": list(OPERATIONS[name]),
                "compression_ratios": ratios,
                "output_kind": "native_head_logits" if stage.tail else "native_stage_hidden_last_token"})
        fresh()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--input", required=True)
    parser.add_argument("--stage-index", type=int, required=True)
    parser.add_argument("--lo", type=int, required=True)
    parser.add_argument("--hi", type=int, required=True)
    parser.add_argument("--parent-count", type=int, required=True)
    parser.add_argument("--max-seq", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host-mib", type=int, required=True)
    parser.add_argument("--device-mib", type=int, required=True)
    parser.add_argument("--key-file")
    parser.add_argument("--confirm-gpu", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    if (args.host_mib < 16 or args.device_mib < 1 or not 16 <= args.max_seq <= 1_048_576
            or args.fixture != args.device.startswith("cpu") or not args.fixture and
            (not args.confirm_gpu or not args.device.startswith("cuda:") or not args.checkpoint or not args.key_file)):
        raise ValueError("owned CPU fixture or confirmed CUDA/checkpoint/key with explicit budgets required")
    if Path(args.out_dir).exists():
        raise ValueError("probe output directory must be new")
    if Path(args.input).stat().st_size > min(128 << 20, args.host_mib << 19):
        raise ValueError("frozen input exceeds explicit host preflight limit")
    from phase0.v4_incremental_qualification import read
    spec, evidence = read(args.spec), read(args.evidence)
    if not 0 <= args.stage_index < len(evidence.get("stages", [])):
        raise ValueError("owned stage assignment missing")
    assignment = evidence["stages"][args.stage_index]
    if (assignment["lo"], assignment["hi"]) != (args.lo, args.hi):
        raise ValueError("owned stage range differs from the pinned campaign")
    execution = spec["execution"]
    if any(execution.get(k) for k in ("spec", "dspark", "pipelined", "temp")) or execution.get("max_pos") != args.max_seq:
        raise ValueError("case producer currently qualifies exact greedy resident-KV reset contracts only")
    if (type(spec.get("max_context")) is not int or spec["max_context"] > args.max_seq
            or type(spec.get("max_suffix")) is not int or spec["max_suffix"] < 1
            or type(spec.get("parent_alignment")) is not int or spec["parent_alignment"] < 1
            or args.parent_count % spec["parent_alignment"]):
        raise ValueError("owned probe geometry differs from the campaign before model allocation")
    if args.fixture:
        os.environ["V4_KERNELS"] = "cpu"
    import torch
    from safetensors.torch import load_file
    import v4_stage as V
    if args.fixture:
        import v4_ref_cpu as R
        model_args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
                                dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=args.max_seq)
    else:
        V.verify_checkpoint_artifacts(args.checkpoint, required=True, lo=args.lo, hi=args.hi,
            head=args.lo == 0, tail=args.hi == V.config(args.checkpoint).n_layers, dspark=False)
        from v4_resources import inspect_checkpoint, stage_storage_inventory
        model_args = V.config(args.checkpoint)
        model_args.max_seq_len, model_args.max_batch_size = args.max_seq, 1
        inv = stage_storage_inventory(inspect_checkpoint(args.checkpoint), args.lo, args.hi,
                    head=args.lo == 0, tail=args.hi == model_args.n_layers, dspark=False)
        if inv["resident_parameter_floor_bytes"] + inv["routed_expert_storage_bytes"] > args.device_mib << 20:
            raise ValueError("verified native model parameter floor exceeds owned device budget")
    old_device = torch.get_default_device()
    cache_names = ("precompute_freqs_cis", "get_window_topk_idxs", "get_compress_topk_idxs", "get_dspark_topk_idxs")
    for name in cache_names: getattr(V.ref(), name).cache_clear()
    try:
        torch.set_default_device(args.device)
        stage = V.Stage(args.lo, args.hi, model_args, head=args.lo == 0, tail=args.hi == model_args.n_layers, device=args.device)
        if args.fixture:
            with torch.device("cpu"): oracle = R.build_oracle(model_args, 7)
            for index, layer in enumerate(stage.layers, args.lo): layer.load_state_dict(oracle.layers[index].state_dict())
            if stage.head: stage.embed_tokens.load_state_dict(oracle.embed.state_dict())
            if stage.tail:
                stage.norm.load_state_dict(oracle.norm.state_dict()); stage.lm_head.load_state_dict(oracle.head.state_dict())
                with torch.no_grad():
                    for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"): getattr(stage, name).copy_(getattr(oracle, name))
            del oracle
        else:
            stage.load(args.checkpoint)
        from v4_resources import runtime_config_identity, measure_stage_resources
        measured = measure_stage_resources(stage, checkpoint_id="incremental-owned-case-probe")
        storage = measured["module_storage"]
        if storage["gpu_bytes"] > args.device_mib << 20 or storage["host_bytes"] > args.host_mib << 20:
            raise ValueError("actual native Stage storage exceeds explicit owned quotas")
        from v4_observability import source_inventory, gpu_uuid_text
        import v4_kernels_cpu
        source = digest(source_inventory())
        backend = v4_kernels_cpu.backend()
        gpu_uuid = gpu_uuid_text(getattr(torch.cuda.get_device_properties(args.device), "uuid", None)) if not args.fixture else None
        gpu = (not args.fixture and torch.cuda.is_available() and backend == "tilelang" and gpu_uuid is not None)
        if not args.fixture and (not gpu or gpu_uuid != assignment["gpu_uuid"]
                or source != spec["source_sha256"] or runtime_config_identity(stage) != assignment["runtime_config_sha256"]
                or stage._artifact_verification["checkpoint_id"] != spec["binding"]["source_id"]):
            raise ValueError("actual native GPU/source/checkpoint/runtime differs from pinned campaign")
        data = load_file(args.input, device="cpu")
        count = data.get("token_ids").shape[1] if torch.is_tensor(data.get("token_ids")) and data["token_ids"].ndim == 2 else 0
        if count > spec["max_context"] or count - args.parent_count > spec["max_suffix"]:
            raise ValueError("actual frozen suffix exceeds the pinned campaign geometry")
        if gpu:
            torch.cuda.reset_peak_memory_stats(args.device)
        comparisons = compare_cases(stage, data, parent_count=args.parent_count, host_bytes=args.host_mib << 20)
        allocator = None
        if gpu:
            torch.cuda.current_stream(args.device).synchronize()
            allocator = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device)}
            if max(allocator.values()) > args.device_mib << 20:
                raise ValueError("actual native probe allocator peaks exceed the owned CUDA quota")
        records = []
        if gpu:
            from shard.manifest import load_key, pub_b64
            from v4_conversation_protocol import _sign
            key = load_key(args.key_file)
            if pub_b64(key) != evidence["coordinator_pubkey"]:
                raise ValueError("case signing key is not the campaign's pinned coordinator")
            for item in comparisons:
                if item["passed"]:
                    record = {k: deepcopy(item[k]) for k in ("case", "reference", "candidate", "inputs_sha256", "parent_count", "full_count", "operation_sequence", "compression_ratios")}
                    record.update(schema=CASE_SCHEMA, stage=args.stage_index, binding_sha256=digest(spec["binding"]),
                        source_sha256=source, execution_sha256=digest(execution), gpu_uuid=gpu_uuid,
                        kernel_backend=backend, device=args.device)
                    records.append(_sign(key, CASE_DOMAIN, record))
        report = {"schema": "v4-incremental-native-case-probe/1", "gpu_verified": gpu,
            "source_sha256": source, "gpu_uuid": gpu_uuid, "kernel_backend": backend,
            "reference": args.fixture, "comparisons": comparisons, "native_cases": records, "allocator": allocator,
            "ready": gpu and len(records) == len(CASES), "performance_qualified": False}
        directory = Path(args.out_dir)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k not in ("comparisons", "native_cases")}, sort_keys=True))
        return 0 if report["ready"] or args.fixture else 2
    finally:
        for name in cache_names: getattr(V.ref(), name).cache_clear()
        torch.set_default_device(old_device)


if __name__ == "__main__":
    raise SystemExit(main())

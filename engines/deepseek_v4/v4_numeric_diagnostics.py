"""Bounded, teacher-forced stage numerics; diagnostic evidence, never a speed receipt.

Run separate processes for recipes that rebind model globals (REF_SLIM, kernels, dtype).
Only owned, idle diagnostic stages may be passed here: traces deliberately reset their state.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json

import torch

SCHEMA = "flyai-v4-numeric-trace/1"
MAX_TOKENS = 16384
MAX_RECORDS = 16384
MAX_TENSOR_BYTES = 128 << 20


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def tensor_identity(tensor, *, max_bytes=MAX_TENSOR_BYTES):
    """Hash raw bytes in bounded host chunks, including dtype/shape; no full KV CPU clone."""
    if tensor.numel() * tensor.element_size() > max_bytes:
        raise ValueError("diagnostic tensor exceeds byte budget")
    h = hashlib.sha256()
    invalid = {"nan": False, "infinity": False}
    def consume(value):
        if value.numel() * value.element_size() <= 1 << 20:
            cpu = value.detach().cpu().contiguous()
            h.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
            if cpu.is_floating_point():
                floating = cpu.float()
                invalid["nan"] |= bool(torch.isnan(floating).any())
                invalid["infinity"] |= bool(torch.isinf(floating).any())
        elif value.ndim:
            for row in value.unbind(0):
                consume(row)
        else:
            raise ValueError("diagnostic scalar exceeds transfer budget")
    consume(tensor)
    return {"sha256": h.hexdigest(), "shape": list(tensor.shape), "dtype": str(tensor.dtype), **invalid}


def layer_state(layer, position, *, max_bytes=MAX_TENSOR_BYTES):
    """Only readable KV and compressor state: discarded future slots are deliberately excluded."""
    a = layer.attn
    win = a.window_size
    values = {"window": a.kv_cache[:, :min(position, win)]}
    if a.compress_ratio:
        values["compressed"] = a.kv_cache[:, win:win + position // a.compress_ratio]
        values.update({"compressor." + k: getattr(a.compressor, k) for k in ("kv_state", "score_state")})
        if a.indexer is not None:
            values["indexer.cache"] = a.indexer.kv_cache[:, :position // a.compress_ratio]
            values.update({"indexer." + k: getattr(a.indexer.compressor, k)
                           for k in ("kv_state", "score_state")})
    return {name: tensor_identity(value, max_bytes=max_bytes) for name, value in values.items()}


@contextmanager
def execution_path(stage, mode):
    if mode not in ("configured", "reference", "whole_eager"):
        raise ValueError("diagnostic execution must be configured, reference or whole_eager")
    old = stage._block_graphs, stage._graph_mode, getattr(stage, "_numeric_eager_twin", False)
    try:
        if mode == "reference":
            stage._block_graphs = None
        elif mode == "whole_eager":
            if stage._block_graphs is None or stage._graph_mode != "whole":
                from v4_whole_layer_graph import WholeBlockGraphs
                stage._block_graphs = [WholeBlockGraphs(layer, stage, moe_mode="eager") for layer in stage.layers]
            stage._graph_mode = "whole"
            stage._numeric_eager_twin = True
        yield
    finally:
        stage._block_graphs, stage._graph_mode, stage._numeric_eager_twin = old


def _model_identity(stage):
    proof = getattr(stage, "_artifact_verification", None)
    if proof is not None:
        try:
            from shard.weight_artifacts import verified_stage_artifact_descriptor
        except ImportError:
            from weight_artifacts import verified_stage_artifact_descriptor
        verified_stage_artifact_descriptor(proof)
        model = {k: proof[k] for k in ("checkpoint_id", "manifest_sha256")}
        model["config_sha256"] = proof["catalog"]["config_sha256"]
        model["verified_artifacts"] = True
    else:
        # CPU fixtures are real arithmetic, not hardware/model evidence.
        if str(stage.device) != "cpu":
            raise ValueError("GPU diagnostics require verified native checkpoint artifacts")
        weights = []
        for index, module in enumerate(stage._owned_modules()):
            for name, value in module.state_dict().items():
                weights.append([index, name, tensor_identity(value)])
        if stage.tail:
            weights += [["tail", name, tensor_identity(getattr(stage, name))]
                        for name in ("hc_head_fn", "hc_head_base", "hc_head_scale")]
        model = {"fixture_weights_sha256": digest(weights), "verified_artifacts": False}
    from v4_observability import source_inventory
    return {"model": model, "source_sha256": digest(source_inventory()),
            "args_sha256": digest(vars(stage.args)), "layers": [stage.lo, stage.hi]}


def trace_stage(stage, token_ids, *, prompt_tokens, hidden=None, execution="configured",
                rejection=None, max_tokens=256, max_tensor_bytes=MAX_TENSOR_BYTES):
    """Record exact per-layer outputs/state for one frozen input, optionally rejecting a suffix.

    `rejection={position, width, accepted}` sends a wrong speculative suffix once, rewinds into
    that chunk, then resumes the ORIGINAL teacher tokens. This exercises actual Stage._seek.
    A non-head stage requires its original frozen input hidden states for every token.
    The report includes hashes, not full activations. Latencies here include diagnostics.
    """
    if (type(max_tokens) is not int or not 2 <= max_tokens <= MAX_TOKENS or
            type(max_tensor_bytes) is not int or not 1 <= max_tensor_bytes <= MAX_TENSOR_BYTES):
        raise ValueError("invalid diagnostic bounds")
    if (getattr(stage, "_native_lanes", None) is not None or getattr(stage, "_dspark_capable", False)
            or getattr(stage, "_kv_runtime", None) is not None
            or stage._pos or stage._spec_ckpts or getattr(stage, "_numeric_layer_observer", None)):
        raise ValueError("diagnostics require an idle dedicated non-MTP stage with resident KV")
    if (not torch.is_tensor(token_ids) or token_ids.device.type != "cpu" or token_ids.dtype != torch.long
            or token_ids.ndim != 2 or token_ids.shape[0] != 1):
        raise ValueError("frozen CPU int64 token_ids [1,total] required")
    total = token_ids.shape[1]
    if (type(prompt_tokens) is not int or not 1 <= prompt_tokens < total <= min(max_tokens, stage.args.max_seq_len)
            or bool((token_ids < 0).any()) or bool((token_ids >= stage.args.vocab_size).any())
            or (total - prompt_tokens + 1) * (len(stage.layers) + 1) > MAX_RECORDS):
        raise ValueError("diagnostic input exceeds token, vocabulary or record budget")
    if hidden is not None:
        if (hidden.device.type != "cpu" or hidden.dtype != stage.dtype or
                tuple(hidden.shape) != (1, total, stage.args.hc_mult, stage.args.dim)):
            raise ValueError("frozen hidden input differs from stage shape/dtype")
        tensor_identity(hidden, max_bytes=max_tensor_bytes)
    elif not stage.head:
        raise ValueError("non-head diagnostics require frozen hidden inputs")
    reject = None
    if rejection is not None:
        if not isinstance(rejection, dict) or set(rejection) != {"position", "width", "accepted"}:
            raise ValueError("invalid rejection plan")
        p, w, a = (rejection[k] for k in ("position", "width", "accepted"))
        if (any(type(v) is not int for v in (p, w, a)) or
                not prompt_tokens <= p < p + w <= total or not 0 < a < w):
            raise ValueError("rejection must restore an accepted prefix inside a bounded chunk")
        if stage._fast:
            raise ValueError("rollback diagnostic requires fixed scalar verification (FAST_VERIFY=0)")
        reject = p, w, a
    # Detach caller inputs before execution; no mutable concurrent producer is trusted.
    ids = token_ids.clone()
    frozen = hidden.clone() if hidden is not None else None
    identity = _model_identity(stage)
    inputs = {"ids": tensor_identity(ids), "hidden": tensor_identity(frozen) if frozen is not None else None,
              "prompt_tokens": prompt_tokens}
    records, paths = [], []
    stop_recording_at = None
    by_id = dict(zip(range(stage.lo, stage.hi), stage.layers))
    def observe(layer, pos, value, path):
        if stage._replaying or stop_recording_at is not None and pos >= stop_recording_at:
            return
        end = pos + value.shape[1]
        records.append({"position": pos, "tokens": value.shape[1], "layer": layer,
                        "hidden": tensor_identity(value, max_bytes=max_tensor_bytes),
                        "state": layer_state(by_id[layer], end, max_bytes=max_tensor_bytes)})
        paths.append(path)
    def forward(pos, width=1, wrong_after=None):
        chunk = ids[:, pos:pos+width].clone()
        if wrong_after is not None:
            chunk[:, wrong_after:] = (chunk[:, wrong_after:] + 1) % stage.args.vocab_size
        h = stage.embed(chunk) if frozen is None else frozen[:, pos:pos+width].to(stage.device)
        return stage.forward(h, chunk, pos)
    old_spec = stage._spec
    import v4_ref_slim
    old_horizon = v4_ref_slim._JOB_MAX_POS
    stage._numeric_layer_observer = observe
    try:
        v4_ref_slim.set_job_max_pos(total)
        with execution_path(stage, execution), torch.no_grad():
            recipe = _recipe(stage, v4_ref_slim)
            stage.reset()
            stage._spec = bool(reject)
            h = forward(0, prompt_tokens)
            if stage.tail:
                records.append(_logits_record(stage, h, 0, prompt_tokens, max_tensor_bytes))
            pos = prompt_tokens
            while pos < total:
                if reject is not None and pos == reject[0]:
                    p, width, accepted = reject
                    stop_recording_at = p + accepted
                    h = forward(pos, width, wrong_after=accepted)
                    stop_recording_at = None
                    if stage.tail:
                        for i in range(accepted):
                            records.append(_logits_record(stage, h[:, i:i+1], pos+i, 1, max_tensor_bytes))
                    pos += accepted
                else:
                    h = forward(pos)
                    if stage.tail:
                        records.append(_logits_record(stage, h, pos, 1, max_tensor_bytes))
                    pos += 1
                # Keep only the just-completed chunk needed by the forced rejection.
                if reject is None or pos != reject[0] + reject[2]:
                    stage.commit(pos)
            graph = [{"layer": stage.lo+i, "captured_variants": len(getattr(bg, "_graphs", {})),
                      "fallback": bool(getattr(bg, "eager", False)),
                      "moe_in_graph": getattr(bg, "moe_in_graph", None)}
                     for i, bg in enumerate(stage._block_graphs or [])]
            if recipe != _recipe(stage, v4_ref_slim):
                raise ValueError("shared numerical recipe changed during trace; isolate recipes in separate processes")
        records.sort(key=lambda row: (row["position"], row["layer"]))
        if identity != _model_identity(stage):
            raise ValueError("model/source changed during numerical trace")
        report = {"schema": SCHEMA, "identity": identity, "inputs": inputs, "records": records,
                  "execution": execution, "rejection": rejection, "paths": sorted(set(paths)), "graphs": graph,
                  "evidence": "gpu_execution" if str(stage.device).startswith("cuda") else "cpu_reference",
                  "speed_evidence": False, "numerical_contract": "teacher_forced_exact_bytes",
                  "scope": "local_stage_hidden_valid_state_and_optional_scalar_tail_logits"}
        report["recipe"] = recipe
        report["sha256"] = digest(report)
        return report
    finally:
        stage._numeric_layer_observer = None
        stage._spec = old_spec
        stage.reset()
        v4_ref_slim.set_job_max_pos(old_horizon)


def _recipe(stage, slim):
    return {"graph_mode": stage._graph_mode, "fast_verify": stage._fast,
        "dtype": str(stage.dtype), "device": str(stage.device), "threads": torch.get_num_threads(),
        "default_dtype": str(torch.get_default_dtype()),
        "ref_slim_installed": bool(getattr(stage._M.Indexer.forward, "_v4_ref_slim", False)),
        "ref_slim_active": slim._ACTIVE, "ref_slim_max_position": slim._JOB_MAX_POS,
        "noqat_installed": bool(getattr(stage._M.act_quant, "_v4_ref_slim", False)),
        "noqat_active": slim._ACTIVE_NOQAT,
        "whole_eager_twin": bool(getattr(stage, "_numeric_eager_twin", False)),
        "kernel_policy_scope": "numerical_twin_of_capture" if getattr(stage, "_numeric_eager_twin", False)
            else "configured_execution",
        "performance_evidence": False}


def _logits_record(stage, hidden, pos, width, limit):
    value = stage.logits_all(hidden, full_logits=False)
    return {"position": pos, "tokens": width, "layer": stage.hi,
            "logits": tensor_identity(value, max_bytes=limit), "argmax": int(value.reshape(-1, value.shape[-1])[-1].argmax())}


def compare_traces(left, right):
    for trace in (left, right):
        if (not isinstance(trace, dict) or trace.get("schema") != SCHEMA or
                trace.get("sha256") != digest({k:v for k,v in trace.items() if k != "sha256"}) or
                not isinstance(trace.get("records"), list) or len(trace["records"]) > MAX_RECORDS):
            raise ValueError("invalid or modified bounded numeric trace")
    if left["identity"] != right["identity"] or left["inputs"] != right["inputs"]:
        raise ValueError("numeric traces have different model/source/configuration/input identities")
    first = None
    for index in range(max(len(left["records"]), len(right["records"]))):
        a = left["records"][index] if index < len(left["records"]) else None
        b = right["records"][index] if index < len(right["records"]) else None
        invalid = any(row is not None and (
            any(row[k]["nan"] or row[k]["infinity"] for k in ("hidden", "logits") if k in row) or
            any(value["nan"] for value in row.get("state", {}).values())) for row in (a, b))
        if a != b or invalid:
            first = {"record": index, "left": a, "right": b}
            if invalid:
                first["invalid_numeric"] = True
            break
    return {"schema": "flyai-v4-numeric-comparison/1", "exact_match": first is None,
            "first_difference": first, "left_sha256": left["sha256"], "right_sha256": right["sha256"],
            "evidence": [left["evidence"], right["evidence"]], "speed_evidence": False,
            "scope": left["scope"]}


def compare_stages(left, right, token_ids, **options):
    """Same-process helper for recipes with identical module globals; CLI traces isolate recipes."""
    left_options = dict(options.pop("left_options", {}))
    right_options = dict(options.pop("right_options", {}))
    return compare_traces(trace_stage(left, token_ids, **options, **left_options),
                          trace_stage(right, token_ids, **options, **right_options))

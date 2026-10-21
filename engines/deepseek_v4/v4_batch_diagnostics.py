"""One loaded native checkpoint, batched device state versus scalar host oracles.

An owned idle Stage is reused; model weights are never replicated. Inputs are
frozen and teacher forced. This diagnostic is intentionally slow and never a
latency or release qualification by itself.
"""
from collections import OrderedDict
import hashlib
import math

import torch

from v4_conversation_cache import _buffers
from v4_native_batch import DeviceNativeLanes, BATCH_CONTRACT
from v4_batch_invariant import CONTRACT as ROW_MATH_CONTRACT
from v4_native_lanes import NativeLanes, CONTRACT
from v4_numeric_diagnostics import _model_identity, digest, tensor_identity, MAX_RECORDS

SCHEMA = "flyai-v4-native-batch-diagnostic/1"
MAX_BATCH_RECORDS = 4 * MAX_RECORDS  # 2..4 requests, each at most 16384 positions.


def _finite_values(value):
    return [float(v) if math.isfinite(float(v)) else None for v in value.tolist()]


def _storage(stage):
    values = []
    for module in stage._owned_modules():
        values += list(module.parameters()) + list(module.buffers())
    values += [getattr(layer.attn, "freqs_cis", None) for layer in stage.layers]
    values += [getattr(stage, name, None) for name in ("hc_head_fn", "hc_head_base", "hc_head_scale")]
    unique = {}
    for value in values:
        if torch.is_tensor(value):
            key = str(value.device), value.untyped_storage().data_ptr()
            unique[key] = value.untyped_storage().nbytes()
    return {tier: sum(size for (device, _), size in unique.items() if device.startswith(tier))
            for tier in ("cpu", "cuda")}


def _compare(left, right, *, atol, rtol, allow_negative_infinity=False):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("batch diagnostic compared incompatible native tensors")
    exact, within, finite, maximum, squares, count = True, True, True, 0., 0., 0
    a, b = left.detach().reshape(-1), right.detach().reshape(-1)
    # Two float32 CPU rows plus two source copies stay within a fixed 4MiB
    # comparison reservation. No entire KV tensor is copied to host.
    for offset in range(0, a.numel(), 65536):
        source_x = a[offset:offset + 65536].cpu().contiguous()
        source_y = b[offset:offset + 65536].cpu().contiguous()
        x, y = source_x.float(), source_y.float()
        same = x == y
        valid = torch.isfinite(x) & torch.isfinite(y)
        accepted = valid | (same & torch.isneginf(x) & allow_negative_infinity)
        finite &= bool(accepted.all())
        delta = torch.where(valid, (x - y).abs(), torch.where(same, 0., float("inf")))
        exact &= bool(accepted.all()) and torch.equal(source_x.view(torch.uint8), source_y.view(torch.uint8))
        within &= bool((accepted & (same | valid & (delta <= atol + rtol * y.abs()))).all())
        if delta.numel():
            maximum = max(maximum, float(delta.max()))
            squares += float(delta.double().square().sum())
            count += delta.numel()
    return {"exact": exact, "within_tolerance": within, "finite_or_matching_infinity": finite,
            "max_abs": maximum if math.isfinite(maximum) else None,
            "rmse": math.sqrt(squares / max(1, count)) if math.isfinite(squares) else None,
            "elements": count}


def _within_device_quota(allocator, quota):
    """A process retains reserved blocks even when its live tensors fit."""
    peaks = [allocator["peak_allocated_bytes"], allocator["peak_reserved_bytes"]]
    return max(peaks) <= quota if all(value is not None for value in peaks) else None


def comparison_passed(report):
    value = report["qualification"]
    policy = report.get("binding", {}).get("batch_runtime", {}).get("graph_policy", {})
    if (policy.get("mode") == "partial" and report.get("binding", {}).get("reference") is False
            and value.get("graph_capture_observed") is not True):
        return False
    return (value["new_batch_abi_within_tolerance"] and value["greedy_equal"] is not False
            and value["topk_equal"] is not False)


def _hardware(stage, reference):
    row = {"execution_device": str(stage.device), "torch": str(torch.__version__),
           "cuda_build": torch.version.cuda, "cuda_available": None, "gpu_uuid": None,
           "gpu_name": None, "compute_capability": None, "driver_version": None,
           "identity_complete": False, "probe": "not_requested_cpu_reference"}
    if reference:
        return row
    row["cuda_available"] = bool(torch.cuda.is_available())
    if not row["cuda_available"]:
        raise ValueError("native CUDA batch diagnostic requires an actual available CUDA accelerator")
    props = torch.cuda.get_device_properties(stage.device)
    from v4_observability import gpu_uuid_text
    row.update(gpu_uuid=gpu_uuid_text(getattr(props, "uuid", None)), gpu_name=props.name,
               compute_capability=[props.major, props.minor], probe="actual_cuda_properties_and_driver_api")
    try:
        import ctypes
        import os
        driver = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        value = ctypes.c_int()
        if driver.cuDriverGetVersion(ctypes.byref(value)) == 0:
            row["driver_version"] = value.value
    except (OSError, AttributeError):
        row["driver_version"] = None
    row["identity_complete"] = bool(row["gpu_uuid"] and row["driver_version"])
    return row


def _valid_states(stage, values, frontier):
    for index, layer in enumerate(stage.layers):
        attn = layer.attn
        base = f"main.{index}"
        yield base + ".window", values[base + ".kv"][:, :min(frontier, attn.window_size)]
        if attn.compress_ratio:
            yield base + ".compressed", values[base + ".kv"][:, attn.window_size:attn.window_size + frontier // attn.compress_ratio]
            if attn.indexer is not None:
                yield base + ".index", values[base + ".index"][:, :frontier // attn.compress_ratio]
            for label, owner in (("attention", attn), ("indexer", attn.indexer)):
                if getattr(owner, "compressor", None) is not None:
                    for field in ("kv_state", "score_state"):
                        name = f"{base}.{label}.{field}"
                        yield name, values[name]


def compare_native_batch(stage, token_ids, *, host_bytes, device_bytes, execution_workspace_bytes,
                         hidden=None, atol=0., rtol=0., topk=5, max_tokens=256,
                         reference=False, graph_mode="off", graph_bytes=0, max_graphs=64):
    """Compare a real request batch with scalar original native execution.

    CUDA requires verified original checkpoint artifacts. CPU reference reports
    fixture arithmetic only. The byte quotas cover original allocations, both
    state owners and explicitly reserved diagnostic/execution working memory.
    """
    if (type(host_bytes) is not int or host_bytes <= 0 or type(device_bytes) is not int or device_bytes <= 0
            or type(execution_workspace_bytes) is not int or execution_workspace_bytes <= 0
            or type(reference) is not bool or reference != (torch.device(stage.device).type == "cpu")
            or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in (atol, rtol))
            or type(topk) is not int or not 1 <= topk <= 32
            or type(max_tokens) is not int or not 1 <= max_tokens <= 16384
            or graph_mode not in ("off", "partial") or type(graph_bytes) is not int or graph_bytes < 0
            or (graph_mode == "off") != (graph_bytes == 0)
            or type(max_graphs) is not int or not 2 <= max_graphs <= 4096):
        raise ValueError("explicit bounded batch diagnostic resources/numerical tolerances required")
    if (getattr(stage, "_native_lanes", None) is not None or stage._pos or stage._spec or stage._dspark
            or stage._spec_ckpts or stage._last_tap or getattr(stage, "_numeric_layer_observer", None)):
        raise ValueError("batch diagnostics require an idle dedicated stage")
    if (not torch.is_tensor(token_ids) or token_ids.device.type != "cpu" or token_ids.dtype != torch.long
            or token_ids.ndim != 2 or not 2 <= token_ids.shape[0] <= 4
            or not 1 <= token_ids.shape[1] <= min(max_tokens, stage.args.max_seq_len)
            or bool((token_ids < 0).any()) or bool((token_ids >= stage.args.vocab_size).any())):
        raise ValueError("frozen CPU int64 [2..4,total] IDs within the stage horizon required")
    count, total = token_ids.shape
    if count * total > MAX_BATCH_RECORDS:
        raise ValueError("batch diagnostic per-request records exceed bounded report capacity")
    if hidden is not None:
        if (not torch.is_tensor(hidden) or hidden.device.type != "cpu" or hidden.dtype != stage.dtype
                or tuple(hidden.shape) != (count, total, stage.args.hc_mult, stage.args.dim)
                or not bool(torch.isfinite(hidden).all())):
            raise ValueError("frozen non-head hidden input shape/dtype/finite contract differs")
    elif not stage.head:
        raise ValueError("non-head batch diagnostics require frozen hidden inputs")
    buffers = _buffers(stage, None)
    scalar_tensor = sum(value.numel() * value.element_size() for value in buffers.values())
    row_tensor = sum(value[0:1].numel() * value.element_size() for value in buffers.values())
    metadata = 65536 + stage.args.max_seq_len * 64
    scalar_host = count * (scalar_tensor + metadata)
    resident_state = 2 * count * row_tensor
    graph_host = 65536 * max_graphs if graph_mode == "partial" else 0
    batch_host = metadata * count + (2 << 20) + graph_host + (resident_state + graph_bytes if reference else 0)
    input_bytes = token_ids.numel() * token_ids.element_size() + (hidden.numel() * hidden.element_size() if hidden is not None else 0)
    original = _storage(stage)
    report_reserve = count * total * (4096 + topk * 512) + count * len(buffers) * 1024
    diagnostic_host = (8 << 20) + 2 * input_bytes + report_reserve
    host_required = original["cpu"] + scalar_host + batch_host + diagnostic_host + (execution_workspace_bytes if reference else 0)
    device_required = original["cuda"] + resident_state + graph_bytes + execution_workspace_bytes
    if host_required > host_bytes or device_required > device_bytes:
        raise ValueError("batch diagnostic original allocations, states and workspaces exceed explicit quota")
    identity = _model_identity(stage)
    hardware = _hardware(stage, reference)
    ids, frozen = token_ids.clone(), hidden.clone() if hidden is not None else None
    inputs = {"token_ids": tensor_identity(ids, max_bytes=max(1, input_bytes)),
              "hidden": tensor_identity(frozen, max_bytes=max(1, input_bytes)) if frozen is not None else None}
    from v4_resources import runtime_config_payload
    runtime = runtime_config_payload(stage)
    binding = {"identity": identity, "runtime_config_sha256": digest(runtime), "inputs": inputs,
               "batch_contract": BATCH_CONTRACT, "scalar_contract": CONTRACT,
               "batch_math_contract": ROW_MATH_CONTRACT,
               "batch_size": count, "tokens": total, "topk": topk,
               "tolerance": {"atol": atol, "rtol": rtol}, "reference": reference,
               "hardware": hardware}
    if not reference:
        # This diagnostic owns an idle process interval. Record actual allocator
        # peaks as well as declared reservations; a reservation is not a measured
        # upper bound on a kernel's temporary allocations.
        process_allocated = torch.cuda.memory_allocated(stage.device)
        if process_allocated + resident_state + graph_bytes + execution_workspace_bytes > device_bytes:
            raise ValueError("actual CUDA process allocations plus diagnostic state exceed declared device quota")
        torch.cuda.reset_peak_memory_stats(stage.device)
    batch = DeviceNativeLanes(stage, max_jobs=count, host_bytes=batch_host,
                              device_bytes=resident_state + graph_bytes, reference=reference,
                              graph_mode=graph_mode, graph_bytes=graph_bytes, max_graphs=max_graphs)
    # Bind the actual immutable runner policy as well as the idle scalar Stage.
    # Runtime counters stay outside this digest because they change each step.
    binding["batch_runtime"] = batch.config()
    oracle = NativeLanes(stage, max_jobs=count, host_bytes=scalar_host)
    records, states, output_hash = [], OrderedDict(), hashlib.sha256()
    first = {"output": None, "state": None, "greedy": None, "topk": None, "tolerance": None}
    exact = within = greedy_equal = topk_equal = True
    try:
        for index in range(count):
            batch.open(f"batch-{index}", identity=str(index) * 64, horizon=stage.args.max_seq_len)
            oracle.open(f"scalar-{index}", identity=str(index) * 64, horizon=stage.args.max_seq_len)
        for pos in range(total):
            current = ids[:, pos:pos + 1].to(stage.device)
            with torch.no_grad():
                h = stage.embed(current) if frozen is None else frozen[:, pos:pos + 1].to(stage.device)
                batched = batch.forward_batch([f"batch-{i}" for i in range(count)], h, current, pos)
                # Only the new request batch enters the fixed-row execution
                # domain. The independent legacy scalar oracle below retains
                # its original native dispatch and reductions.
                batched_logits = batch.logits_all(batched, full_logits=False) if stage.tail else None
            for index in range(count):
                oracle.activate(f"scalar-{index}", pos)
                with torch.no_grad():
                    one = stage.embed(current[index:index + 1]) if frozen is None else frozen[index:index + 1, pos:pos + 1].to(stage.device)
                    scalar = stage.forward(one, current[index:index + 1], pos)
                    logits = stage.logits_all(scalar, full_logits=False) if stage.tail else None
                oracle.finish_step(f"scalar-{index}")
                comparison = _compare(batched[index:index + 1], scalar, atol=atol, rtol=rtol)
                row = {"position": pos, "request_row": index, "output": comparison, "vocab": None}
                exact &= comparison["exact"]
                within &= comparison["within_tolerance"]
                if not comparison["exact"] and first["output"] is None:
                    first["output"] = [pos, index]
                if not comparison["within_tolerance"] and first["tolerance"] is None:
                    first["tolerance"] = {"position": pos, "request_row": index, "scope": "output"}
                if stage.tail:
                    a, b = batched_logits[index:index + 1].reshape(-1), logits.reshape(-1)
                    k = min(topk, a.numel())
                    a_top, b_top = a.topk(k), b.topk(k)
                    a_logp, b_logp = a.log_softmax(-1), b.log_softmax(-1)
                    same_greedy = int(a.argmax()) == int(b.argmax())
                    same_topk = torch.equal(a_top.indices, b_top.indices)
                    row["vocab"] = {"batch_argmax": int(a.argmax()), "scalar_argmax": int(b.argmax()),
                        "batch_topk": a_top.indices.tolist(), "scalar_topk": b_top.indices.tolist(),
                        "batch_topk_logits": _finite_values(a_top.values), "scalar_topk_logits": _finite_values(b_top.values),
                        "batch_topk_logprobs": _finite_values(a_logp[a_top.indices]),
                        "scalar_topk_logprobs": _finite_values(b_logp[b_top.indices]),
                        "logits": _compare(a, b, atol=atol, rtol=rtol),
                        "logprobs": _compare(a_logp, b_logp, atol=atol, rtol=rtol)}
                    for name in ("logits", "logprobs"):
                        exact &= row["vocab"][name]["exact"]
                        within &= row["vocab"][name]["within_tolerance"]
                        if not row["vocab"][name]["within_tolerance"] and first["tolerance"] is None:
                            first["tolerance"] = {"position": pos, "request_row": index, "scope": name}
                    greedy_equal &= same_greedy
                    topk_equal &= same_topk
                    if not same_greedy and first["greedy"] is None:
                        first["greedy"] = [pos, index]
                    if not same_topk and first["topk"] is None:
                        first["topk"] = [pos, index]
                left = dict(_valid_states(stage, batch.active[f"batch-{index}"]["buffers"], pos + 1))
                right = dict(_valid_states(stage, {name: value[0:1] for name, value in oracle.active[f"scalar-{index}"]["buffers"].items()}, pos + 1))
                for name in left:
                    value = _compare(left[name], right[name], atol=atol, rtol=rtol,
                                     allow_negative_infinity=name.endswith("score_state"))
                    key = f"row.{index}.{name}"
                    summary = states.setdefault(key, {"exact": True, "within_tolerance": True,
                        "max_abs": 0., "first_divergent_position": None, "frontiers_compared": 0,
                        "global_layer": stage.lo + int(name.split(".")[1]), "nonfinite_comparisons": 0})
                    summary["exact"] &= value["exact"]
                    summary["within_tolerance"] &= value["within_tolerance"]
                    summary["max_abs"] = max(summary["max_abs"], value["max_abs"] or 0.)
                    summary["frontiers_compared"] += 1
                    summary["nonfinite_comparisons"] += not value["finite_or_matching_infinity"]
                    exact &= value["exact"]
                    within &= value["within_tolerance"]
                    if not value["within_tolerance"] and first["tolerance"] is None:
                        first["tolerance"] = {"position": pos, "request_row": index,
                                              "scope": "state", "buffer": name,
                                              "global_layer": summary["global_layer"]}
                    if not value["exact"]:
                        if summary["first_divergent_position"] is None:
                            summary["first_divergent_position"] = pos
                        if first["state"] is None:
                            first["state"] = {"position": pos, "request_row": index, "buffer": name}
                output_hash.update(digest(row).encode())
                records.append(row)
        allocator = {"scope": "owned_process_diagnostic_interval", "peak_allocated_bytes": None,
                     "peak_reserved_bytes": None, "within_device_quota": None}
        if not reference:
            torch.cuda.synchronize(stage.device)
            allocator.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(stage.device),
                             peak_reserved_bytes=torch.cuda.max_memory_reserved(stage.device))
            allocator["within_device_quota"] = _within_device_quota(allocator, device_bytes)
        artifact_cuda = not reference and identity["model"]["verified_artifacts"] and hardware["identity_complete"]
        report = {"schema": SCHEMA, "binding": binding, "binding_sha256": digest(binding),
            "evidence": ("verified_native_cuda_stage_diagnostic" if artifact_cuda else
                         "cpu_fixture_arithmetic_only" if reference else "native_cuda_identity_incomplete_unqualified"),
            "qualification": {"hardware_claim": artifact_cuda, "release_qualified": False,
                "old_scalar_abi_bitwise_invariant": exact,
                "new_batch_abi_within_tolerance": within,
                "greedy_equal": greedy_equal if stage.tail else None,
                "topk_equal": topk_equal if stage.tail else None,
                "graph_capture_observed": bool(not reference and batch.graph_pool is not None
                    and batch.graph_status()["captured_graphs"] > 0 and batch.graph_status()["replays"] > 0),
                "scope": "only_this_frozen_teacher_forced_stage_trajectory;_no_old_abi_admission",
                "vocab_metrics": "actual_tail_vocabulary" if stage.tail else "unavailable_non_tail_stage"},
            "resources": {"weights_loaded_once": True, "original_storage_bytes": original,
                "host_reserved_bytes": host_required, "device_reserved_bytes": device_required,
                "host_quota_bytes": host_bytes, "device_quota_bytes": device_bytes,
                "scalar_oracle_host_bytes": scalar_host, "resident_slot_and_batch_workspace_bytes": resident_state,
                "diagnostic_host_workspace_bytes": diagnostic_host,
                "bounded_report_metadata_bytes": report_reserve,
                "execution_workspace_reserved_bytes": execution_workspace_bytes},
            "allocator": allocator,
            "batch_graphs": batch.graph_status(),
            "first_divergence": first, "records": records, "states": states,
            "execution": dict(batch.counters), "output_comparison_sha256": output_hash.hexdigest()}
        report["sha256"] = digest(report)
        return report
    finally:
        batch.clear()
        oracle.clear()
        stage.reset()

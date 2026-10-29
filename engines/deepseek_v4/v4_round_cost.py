"""Bounded per-frame stage deltas for coordinator cost decisions.

These are local observations, not execution proofs. Dispatch wall intervals can
contain GPU waits and overlap; they are never summed into a GPU critical path.
No CUDA synchronization is added to the serving path.
"""
from __future__ import annotations

import math
import time

SCHEMA = "v4-frame-cost/1"
MAX_STAGES = 64


def _integer(value, limit=1 << 60):
    return type(value) is int and 0 <= value <= limit


def _snapshot(stage):
    cache = getattr(stage, "_expert_cache", None)
    dma = None
    if cache is None:
        if getattr(stage, "_expert_placement", "gpu") == "gpu":
            dma = 0
    else:
        pools = cache.stats().get("pools", {})
        values = []
        for row in pools.values():
            value = row.get("dma_bytes")
            if row.get("storage_mode") == "exclusive":
                # The existing cache counter measures submitted H2D. An
                # exclusive eviction also preserves its only GPU copy in RAM.
                # Its completed preservation count is independent of H2D.
                if row.get("mode") == "cuda_dma":
                    preserved, size = row.get("preserved_experts"), row.get("exchange_bytes")
                    value = value + preserved * size if all(_integer(x) for x in (value, preserved, size)) else None
                elif row.get("mode") == "cpu_reference":
                    value = 0 if value == 0 else None
                else:
                    value = None
            values.append(value)
        if all(_integer(x) for x in values):
            dma = sum(values)
    graphs = getattr(stage, "_block_graphs", None) or []
    captured = sum(len(getattr(graph, "_graphs", {})) +
        sum(getattr(item, "graph", None) is not None for item in (getattr(graph, "g", None) or [])
            if item is not None) for graph in graphs)
    tail = getattr(stage, "_resource_drafter", None)
    chain = getattr(tail, "_draft_chain", None)
    captured += len(getattr(chain, "entries", {}))
    segments = [getattr(getattr(block.ffn, "_hybrid_runtime", None), "segment_graphs", None)
                for block in getattr(stage, "_hybrid_blocks", [])]
    changes = (captured, tuple(getattr(graph, "fallback_reason", None) for graph in graphs),
        tuple((segment.counters.get("captures", 0), segment.counters.get("capture_errors", 0))
              for segment in segments if segment is not None))
    return dma, changes


def reset(stage, msg):
    trace = msg.get("round_cost_trace")
    if trace is not None and (not isinstance(trace, str) or len(trace) != 32 or
            any(c not in "0123456789abcdef" for c in trace)):
        raise ValueError("bounded frame-cost trace required")
    stage._round_cost_trace = trace


def begin(stage, msg, *, clock=time.perf_counter):
    trace = getattr(stage, "_round_cost_trace", None)
    if trace is None:
        return None
    if msg.get("round_cost_trace") != trace:
        raise ValueError("frame-cost trace differs from current job")
    return clock(), _snapshot(stage)


def append(stage, msg, token, *, stage_index, nstages, clock=time.perf_counter):
    if token is None:
        return None
    if not _integer(stage_index, MAX_STAGES-1) or not 1 <= nstages <= MAX_STAGES:
        raise ValueError("bounded actual stage geometry required")
    trace = getattr(stage, "_round_cost_trace", None)
    expected = {"trace_id": trace, "epoch": msg.get("epoch", 0), "pos": msg.get("start_pos")}
    if not all(_integer(expected[key]) for key in ("epoch", "pos")):
        raise ValueError("invalid frame-cost identity")
    prior = msg.get("round_cost")
    if stage_index == 0:
        if prior is not None:
            raise ValueError("head refuses upstream stage-cost injection")
        rows = []
    else:
        if not isinstance(prior, dict) or any(prior.get(k) != v for k, v in expected.items()):
            raise ValueError("frame-cost identity differs between stages")
        rows = prior.get("stages")
        if (prior.get("schema") != SCHEMA or not isinstance(rows, list) or len(rows) != stage_index or
                any(not isinstance(row, dict) for row in rows) or
                [row.get("stage") for row in rows] != list(range(stage_index))):
            raise ValueError("frame-cost stages missing or duplicated")
        rows = list(rows)
    before_dma, before_graphs = token[1]
    after_dma, after_graphs = _snapshot(stage)
    elapsed = clock() - token[0]
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid local dispatch interval")
    dma = (after_dma-before_dma if before_dma is not None and after_dma is not None
           and after_dma >= before_dma else None)
    rows.append({"stage": stage_index, "expert_dma_bytes": dma,
                 "dispatch_wall_s": elapsed, "capture_changed": after_graphs != before_graphs})
    complete = len(rows) == nstages and all(_integer(row.get("expert_dma_bytes")) for row in rows)
    return {"schema": SCHEMA, **expected, "stages": rows,
            "coverage": "all_stages" if complete else "partial",
            "expert_dma_bytes": sum(row["expert_dma_bytes"] for row in rows) if complete else None,
            "capture_changed": any(row.get("capture_changed", False) for row in rows),
            "timing_scope": "local_dispatch_intervals_not_additive_gpu_time"}


def validate_reply(cost, *, trace_id, epoch, pos, expected_stages=None, max_stages=MAX_STAGES):
    """Coordinator checks the returned chain before using any byte total."""
    if (not isinstance(cost, dict) or cost.get("schema") != SCHEMA or
            any(cost.get(k) != v for k, v in {"trace_id": trace_id, "epoch": epoch, "pos": pos}.items())):
        return None
    rows = cost.get("stages")
    if (not isinstance(rows, list) or not 1 <= len(rows) <= max_stages or
            (expected_stages is not None and len(rows) != expected_stages) or
            [row.get("stage") for row in rows if isinstance(row, dict)] != list(range(len(rows)))):
        return None
    for row in rows:
        if (not isinstance(row.get("capture_changed"), bool) or
                type(row.get("dispatch_wall_s")) not in (float, int) or
                not math.isfinite(row["dispatch_wall_s"]) or row["dispatch_wall_s"] < 0):
            return None
    values = [row.get("expert_dma_bytes") for row in rows]
    if cost.get("coverage") != "all_stages" or not all(_integer(x) for x in values):
        return None
    if sum(values) != cost.get("expert_dma_bytes"):
        return None
    return cost

"""Bounded FIFO ring requests with real work in flight before the first receive.

Requests here belong to disjoint lanes. Dependent steps for one lane must never
be submitted together. Cancellation is inspected by the owner after run drains
every reply, so a close barrier cannot overtake an already forwarded step.
"""
from collections import deque


def drain_pipeline(items, *, depth, send, receive, stats):
    if type(depth) is not int or not 1 <= depth <= 4 or len(items) > 4:
        raise ValueError("native pipeline requires 1..4 bounded independent cohorts")
    waiting, results, cursor = deque(), [], 0
    while cursor < len(items) or waiting:
        while cursor < len(items) and len(waiting) < depth:
            item = items[cursor]
            send(item)
            waiting.append(item)
            cursor += 1
            stats["sent_cohorts"] += 1
            stats["peak_inflight_cohorts"] = max(stats["peak_inflight_cohorts"], len(waiting))
        item = waiting.popleft()
        results.append(receive(item))
        stats["received_cohorts"] += 1
    return results


def observed_token_spans(samples, *, max_quantum=8):
    """Actual adjacent token-round spans, including elapsed coordinator gaps.

    A span is never a q1 mean multiplied by q. Its endpoints belong to one
    unchanged authenticated job set, owner epoch, batch and consecutive
    frontiers, with no intervening cohort. It is observational cost data, not
    evidence that a single advance(q) or a physical GPU gate has passed.
    """
    if len(samples) > 512 or type(max_quantum) is not int or not 1 <= max_quantum <= 8:
        raise ValueError("bounded cohort samples/quanta required")
    result = []
    for start, first in enumerate(samples):
        if not first["prefill"]:
            continue
        previous = None
        for width in range(1, min(max_quantum, len(samples) - start) + 1):
            last = samples[start + width - 1]
            if (not last["prefill"] or last["job_set_sha256"] != first["job_set_sha256"]
                    or last["owner_epoch"] != first["owner_epoch"] or last["batch_jobs"] != first["batch_jobs"]
                    or last["context_tokens"] != first["context_tokens"] + width - 1
                    or last["cohort_sequence"] != first["cohort_sequence"] + width - 1
                    or last["drained_at"] < last["submitted_at"]
                    or previous is not None and last["submitted_at"] < previous["drained_at"]):
                break
            result.append({"context_tokens": first["context_tokens"], "batch_jobs": first["batch_jobs"],
                           "quantum_tokens": width, "elapsed_ms": max(.000001,
                               (last["drained_at"] - first["submitted_at"]) * 1000),
                           "measured_at": first["measured_at"]})
            previous = last
    return result

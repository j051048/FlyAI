from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
import pytest
from shard.runtime_profile import RuntimeProfiler, validate_profile
from phase0.v4_profile_report import chrome_trace


def test_nested_host_spans_bounded_and_independent_from_full_totals():
    ticks = iter([10, 10.001, 10.003, 10.005, 10.006, 10.009])
    p = RuntimeProfiler(clock=lambda: next(ticks), timeline_capacity=2)
    with p.host("outer.host"):
        with p.host("inner.host"):
            pass
    with p.host("send.host"):
        pass
    r = p.snapshot()
    assert r["timeline"]["dropped"] == 1
    assert [s["name"] for s in r["timeline"]["spans"]] == ["outer.host", "send.host"]
    assert r["phases"]["inner.host"]["total_ms"] == pytest.approx(2)
    assert r["timeline"]["spans"][1]["start_ms"] == pytest.approx(6)
    p.reset()
    assert p.snapshot()["timeline"]["spans"] == []


def test_gpu_timeline_queries_only_until_explicit_snapshot_and_retains_epoch():
    calls, stamp = [], iter([0, 2, 5, 7, 11])
    class Event:
        def record(self, stream): self.t = next(stamp)
        def query(self): return False
        def synchronize(self): calls.append("sync")
        def elapsed_time(self, other): return other.t - self.t
    stream = SimpleNamespace(cuda_stream=13)
    cuda = SimpleNamespace(is_current_stream_capturing=lambda: False,
        device=lambda d: nullcontext(), current_stream=lambda d: stream, Event=lambda **k: Event())
    p = RuntimeProfiler(gpu_sample_every=1, timeline_capacity=4)
    for _ in range(2):
        with p.gpu("work.gpu", SimpleNamespace(cuda=cuda), "cuda:0"): pass
    assert calls == []
    snap = p.snapshot(wait=False)
    assert snap["timeline"]["spans"] == []
    spans = p.snapshot()["timeline"]["spans"]
    assert calls == ["sync", "sync"]
    assert [(s["start_ms"], s["duration_ms"]) for s in spans] == [(2, 3), (7, 4)]
    assert {s["clock"] for s in spans} == {"cuda_event"}


@pytest.mark.parametrize("bad", [float("nan"), -1, True])
def test_invalid_raw_timestamps_refused(bad):
    p = RuntimeProfiler(timeline_capacity=1)
    with p.host("x.host"): pass
    snap = deepcopy(p.snapshot())
    snap["timeline"]["spans"][0]["start_ms"] = bad
    with pytest.raises(ValueError): validate_profile(snap)


def test_trace_never_aligns_cpu_cuda_or_distinct_receipt_epochs():
    span = dict(name="x", clock="host_monotonic", track="thread:1", start_ms=2, duration_ms=3)
    rows = [dict(node_id="n", job_id="j", receipt_sha256=s,
        timeline=dict(schema="shard-runtime-timeline/1", capacity=2, dropped=0,
                      spans=[span, dict(span, clock="cuda_event")])) for s in ("a", "b")]
    trace = chrome_trace(dict(stage_profiles=rows, protocol_sha256="p", report_sha256="r"))
    assert len({e["pid"] for e in trace["traceEvents"]}) == 4
    assert trace["traceEvents"][0]["dur"] == 3000


def test_signed_timeline_detaches_and_cpu_cannot_claim_cuda_clock():
    from shard.runtime_metrics import RuntimeMetrics, validate_runtime_metrics
    from shard.receipt import ReceiptSigner, ReceiptError, gen_key, verify_receipt
    p = RuntimeProfiler(timeline_capacity=2)
    with p.host("decode.host"): pass
    metrics = RuntimeMetrics("cpu").snapshot()
    metrics["performance"] = p.snapshot()
    signed = ReceiptSigner(gen_key(), "s", "j", 0, 1).finalize(runtime_metrics=metrics)
    verify_receipt(signed)
    metrics["performance"]["timeline"]["spans"][0]["start_ms"] += 1
    verify_receipt(signed)
    changed = deepcopy(signed)
    changed["runtime_metrics"]["performance"]["timeline"]["spans"][0]["start_ms"] += 1
    with pytest.raises(ReceiptError, match="signature"): verify_receipt(changed)
    metrics["performance"]["timeline"]["spans"][0]["clock"] = "cuda_event"
    with pytest.raises(ValueError, match="CPU reference"): validate_runtime_metrics(metrics)

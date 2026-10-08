from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest

from shard.runtime_metrics import RuntimeMetrics, validate_runtime_metrics
from shard.runtime_profile import RuntimeProfiler, validate_profile
from shard.plan import ram_dma_overhead_ms
from shard.receipt import ReceiptSigner, ReceiptError, gen_key, verify_receipt


def test_profile_is_bounded_but_keeps_full_totals_and_tail_latency():
    p = RuntimeProfiler(sample_capacity=3)
    for ms in (1., 2., 3., 4., 100.):
        p.record("stage.send.host", ms)
    row = p.snapshot()["phases"]["stage.send.host"]
    assert row == dict(count=5, total_ms=110., min_ms=1., max_ms=100.,
                       p50_ms=4., p95_ms=100., samples=3)
    p.reset()
    assert p.snapshot()["phases"] == {}


def test_host_profile_preserves_an_exception_and_records_its_time():
    clocks = iter([1., 1.025])
    p = RuntimeProfiler(clock=lambda: next(clocks))
    with pytest.raises(RuntimeError, match="operator"):
        with p.host("main.decode.cache.host"):
            raise RuntimeError("operator")
    assert p.snapshot()["phases"]["main.decode.cache.host"]["total_ms"] == pytest.approx(25.)


def test_cuda_sampling_never_synchronizes_in_the_token_path():
    syncs = []
    class Event:
        def record(self, stream):
            pass
        def query(self):
            return False
        def synchronize(self):
            syncs.append(self)
        def elapsed_time(self, end):
            return 2.
    cuda = SimpleNamespace(is_current_stream_capturing=lambda: False,
        device=lambda d: nullcontext(), current_stream=lambda d: object(), Event=lambda **k: Event())
    p = RuntimeProfiler(gpu_sample_every=2, max_pending=1)
    for _ in range(5):
        with p.gpu("main.decode.layer.gpu", SimpleNamespace(cuda=cuda), "cuda:0"):
            pass
    assert syncs == [] and p.dropped == 2
    snap = p.snapshot()
    assert len(syncs) == 1
    assert snap["phases"]["main.decode.layer.gpu"]["count"] == 1
    assert snap["dropped_gpu_samples"] == 2


def test_cpu_gpu_timing_context_never_touches_cuda():
    p = RuntimeProfiler()
    with p.gpu("main.decode.layer.gpu", object(), "cpu"):
        pass
    assert p.snapshot()["phases"] == {}


def test_profile_and_prefetch_quality_are_validated_in_receipt_metrics():
    p, m = RuntimeProfiler(), RuntimeMetrics("cpu")
    p.record("stage.out.host", 3.)
    m.performance = p.snapshot()
    m.prefetch_quality(dict(requested=2, used=1, wasted=1, skipped=3, candidate_count=5))
    value = m.snapshot()
    assert value["prefetch_policy"]["used"] == 1
    bad = deepcopy(value)
    bad["performance"]["phases"]["stage.out.host"]["p95_ms"] = float("nan")
    with pytest.raises(ValueError):
        validate_runtime_metrics(bad)
    bad = deepcopy(value)
    bad["prefetch_policy"]["used"] = 3
    with pytest.raises(ValueError, match="prefetch outcomes"):
        validate_runtime_metrics(bad)
    bad = deepcopy(value)
    bad["performance"]["phases"]["main.decode.layer.gpu"] = bad["performance"]["phases"].pop("stage.out.host")
    with pytest.raises(ValueError, match="CPU reference"):
        validate_runtime_metrics(bad)


def test_new_observations_are_cryptographically_bound_and_detached():
    m, p = RuntimeMetrics("cpu"), RuntimeProfiler()
    p.record("stage.out.host", 2.)
    m.performance = p.snapshot()
    m.prefetch_quality(dict(requested=2, used=1, wasted=0, skipped=0, candidate_count=2))
    value = m.snapshot()
    receipt = ReceiptSigner(gen_key(), "s", "job", 0, 1).finalize(runtime_metrics=value)
    verify_receipt(receipt)
    value["performance"]["phases"]["stage.out.host"]["max_ms"] = 99.
    verify_receipt(receipt)  # Signed values are detached from the observer.
    changed = deepcopy(receipt)
    changed["runtime_metrics"]["prefetch_policy"]["used"] = 0  # Valid schema, altered observation.
    with pytest.raises(ReceiptError, match="signature"):
        verify_receipt(changed)


@pytest.mark.parametrize("mutate", [
    lambda v: v.update(sample_capacity=9000),
    lambda v: v.update(gpu_sample_every=True),
    lambda v: v.update(dropped_gpu_samples=-1),
    lambda v: v["phases"]["x"].update(total_ms=999.),
    lambda v: v["phases"]["x"].update(samples=99),
])
def test_invalid_profile_claims_fail_closed(mutate):
    p = RuntimeProfiler()
    p.record("x", 1.)
    value = p.snapshot()
    mutate(value)
    with pytest.raises(ValueError):
        validate_profile(value)


def test_ram_estimate_has_correct_bandwidth_units_and_measured_times_win():
    model = dict(expert_slot_bytes=24_000_000)
    node = dict(h2d_gbps=24., expert_misses_per_layer=1., dma_overlap_fraction=0.)
    assert ram_dma_overhead_ms(node, model) == 1.  # 24 MB / 24 GB/s = 1 ms, not 8 ms.
    assert ram_dma_overhead_ms(dict(node, dma_overlap_fraction=.75), model) == .25
    assert ram_dma_overhead_ms(dict(node, dma_exposed_ms_per_layer=.125), model) == .125
    assert ram_dma_overhead_ms(dict(node, layer_ms=2., dma_exposed_ms_per_layer=.125), model) == 0.


@pytest.mark.parametrize("fields", [dict(h2d_gbps=0), dict(expert_misses_per_layer=-1),
    dict(dma_overlap_fraction=1.01), dict(dma_exposed_ms_per_layer=float("nan"))])
def test_invalid_transfer_calibrations_are_rejected(fields):
    with pytest.raises(ValueError):
        ram_dma_overhead_ms(fields, {})

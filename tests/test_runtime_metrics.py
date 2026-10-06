"""Signed telemetry validation, common denominators and backing-storage accounting."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from shard.receipt import ReceiptError, ReceiptSigner, gen_key, verify_receipt, wire_receipt
from shard.runtime_metrics import RuntimeMetrics, storage_residency, validate_runtime_metrics


def observed(device="cpu"):
    metrics = RuntimeMetrics(device)
    metrics.resident_routes(12, phase="prefill")
    metrics.resident_routes(6, phase="decode")
    metrics.resident_routes(3, phase="replay")
    metrics.resident_routes(2, role="draft")
    return metrics


def test_common_denominator_counts_work_including_replay_and_draft():
    value = observed("cuda:0").snapshot()
    assert value["totals"]["routed_entries"] == value["totals"]["resident_hits"] == 23
    assert value["resident_hit_rate"] == 1.0
    assert value["work"]["main"]["replay"]["routed_entries"] == 3
    assert value["work"]["draft"]["decode"]["routed_entries"] == 2
    assert "gpu_memory" not in value  # no allocator was measured
    assert value["totals"]["dma_misses"] == value["totals"]["cpu_misses"] == 0


def test_cpu_reference_never_claims_gpu_hits_or_cpu_fallback():
    value = observed().snapshot()
    assert value["mode"] == "reference_cpu"
    assert value["totals"]["reference_routes"] == 23
    assert value["totals"]["resident_hits"] == value["totals"]["cpu_misses"] == 0
    assert value["resident_hit_rate"] == 0.0


def test_reset_and_snapshots_are_detached():
    metrics = observed()
    first = metrics.snapshot()
    metrics.reset()
    assert metrics.snapshot()["totals"]["routed_entries"] == 0
    assert metrics.snapshot()["resident_hit_rate"] is None
    assert first["totals"]["routed_entries"] == 23
    first["work"]["main"]["decode"]["routed_entries"] = 999
    assert metrics.work["main"]["decode"]["routed_entries"] == 0


def test_storage_accounting_uses_backing_allocations_and_deduplicates_views():
    class Storage:
        def __init__(self, pointer, size):
            self.pointer, self.size = pointer, size
        def data_ptr(self):
            return self.pointer
        def nbytes(self):
            return self.size
    def tensor(storage, device):
        return SimpleNamespace(device=device, untyped_storage=lambda: storage)
    cpu, gpu = Storage(10, 4096), Storage(10, 8192)
    view = tensor(cpu, "cpu")
    values = [view, tensor(cpu, "cpu"), tensor(gpu, "cuda:0"), None]
    assert storage_residency(values) == {"host_bytes": 4096, "gpu_bytes": 8192}
    metrics = RuntimeMetrics("cuda:0")
    metrics.sample_kv(values)
    metrics.sample_kv([view])
    assert metrics.snapshot()["kv"] == {"host_bytes": 4096, "gpu_bytes": 0,
                                         "host_peak_bytes": 4096, "gpu_peak_bytes": 8192}


def test_optional_metrics_are_signed_and_legacy_shape_is_unchanged():
    signer = ReceiptSigner(gen_key(), "s", "j", 0, 1)
    plain = signer.finalize()
    value = observed("cuda:0").snapshot()
    signed = signer.finalize(runtime_metrics=value)
    assert set(signed) == set(plain) | {"runtime_metrics"}
    verify_receipt(plain)
    verify_receipt(signed)
    assert wire_receipt({"stage": "tail", **signed}) == signed
    value["mode"] = "mutated"
    verify_receipt(signed)  # detached before signing
    tampered = deepcopy(signed)
    tampered["runtime_metrics"]["kv"]["host_bytes"] += 1
    tampered["runtime_metrics"]["kv"]["host_peak_bytes"] += 1
    with pytest.raises(ReceiptError, match="signature"):
        verify_receipt(tampered)


@pytest.mark.parametrize("mutate", [
    lambda v: v.update(schema="shard-runtime-metrics/2"),
    lambda v: v.update(extra="unsigned extension"),
    lambda v: v.update(resident_hit_rate=float("nan")),
    lambda v: v["totals"].update(routed_entries=True),
    lambda v: v["work"]["main"]["decode"].update(dma_wait_ms=float("inf")),
    lambda v: v["kv"].update(gpu_bytes=-1),
    lambda v: v["work"]["main"]["decode"].update(resident_hits=1),
])
def test_malformed_metrics_fail_before_signing_and_on_verification(mutate):
    bad = observed().snapshot()
    mutate(bad)
    signer = ReceiptSigner(gen_key(), "s", "j", 0, 1)
    with pytest.raises(ValueError):
        signer.finalize(runtime_metrics=bad)
    plain = signer.finalize()
    plain["runtime_metrics"] = bad
    with pytest.raises(ReceiptError, match="runtime metrics"):
        verify_receipt(plain)


def test_shared_gpu_process_omits_peaks_without_resetting_another_stage():
    class Cuda:
        resets = 0
        def reset_peak_memory_stats(self, d):
            self.resets += 1
        def memory_allocated(self, d):
            return 10
        def memory_reserved(self, d):
            return 20
        max_memory_allocated = memory_allocated
        max_memory_reserved = memory_reserved
    cuda = Cuda()
    fake = SimpleNamespace(cuda=cuda)
    one = RuntimeMetrics("cuda:17", torch_module=fake)
    assert one.snapshot()["gpu_memory"]["allocated_peak_bytes"] == 10
    two = RuntimeMetrics("cuda:17", torch_module=fake)
    assert cuda.resets == 1
    assert "gpu_memory" not in two.snapshot()
    assert "gpu_memory" not in one.snapshot()
    del two
    import gc
    gc.collect()
    assert "gpu_memory" not in one.snapshot()  # the shared job cannot regain a peak interval
    one.reset()
    assert cuda.resets == 2  # the next job is exclusive again


def test_unindexed_cuda_uses_current_device_for_shared_peak_ownership():
    class Cuda:
        def __init__(self):
            self.resets = []

        def current_device(self):
            return 19

        def reset_peak_memory_stats(self, device):
            self.resets.append(device)

        def memory_allocated(self, device):
            assert device == "cuda:19"
            return 10

        def memory_reserved(self, device):
            assert device == "cuda:19"
            return 20

        max_memory_allocated = memory_allocated
        max_memory_reserved = memory_reserved

    cuda = Cuda()
    fake = SimpleNamespace(cuda=cuda)
    implicit = RuntimeMetrics("cuda", torch_module=fake)
    assert implicit.snapshot()["gpu_memory"]["allocated_bytes"] == 10
    explicit = RuntimeMetrics("cuda:19", torch_module=fake)
    assert cuda.resets == ["cuda:19"]
    assert "gpu_memory" not in implicit.snapshot()
    assert "gpu_memory" not in explicit.snapshot()

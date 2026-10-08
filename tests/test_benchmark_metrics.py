"""Raw coordinator accounting is testable without CUDA or a model checkpoint."""
from copy import deepcopy
import math

import pytest

from shard.benchmark_metrics import MetricsContractError, make_coordinator_diagnostics, percentile, validate_coordinator_diagnostics


def counters():
    return {"accepted_predictions": 3, "proposed_predictions": 8, "cancel_events": 1,
        "speculation_cycles": 2, "frames_enqueued": 10, "frames_sent": 9,
        "replies_received": 9, "frames_judged": 7, "stale_replies": 1,
        "drained_replies": 1, "unsent_frames": 1}


def diagnostics(mode="pipelined"):
    return make_coordinator_diagnostics(mode, committed_tokens=8, counters=counters(),
        timing={"first_token_s": 2, "last_token_s": 2.14, "request_elapsed_s": 2.2, "receipt_sweep_s": 4},
        inflight_intervals=[{"duration_s": .01, "level": 1}, {"duration_s": .1, "level": 4}])


def test_measured_g_excludes_prefill_and_time_weighted_inflight_is_not_event_mean():
    value = diagnostics()
    assert value["derived"]["g_cycle"] == 3.5  # legacy total-output/cycles would be 4
    assert value["derived"]["g_frame"] == pytest.approx(7 / 9)
    assert value["derived"]["acceptance_ratio"] == 3 / 8
    assert value["derived"]["frame_waste_ratio"] == 2 / 9
    assert value["derived"]["inflight_time_avg"] == pytest.approx(.41 / .11)
    assert value["derived"]["inflight_time_avg"] != 2.5
    assert value["inflight_intervals"][0]["duration_s"] == .01
    assert value["timing"]["drain_s"] == pytest.approx(.06)
    assert value["timing"]["full_service_s"] == 6.2
    assert validate_coordinator_diagnostics(value) == value


@pytest.mark.parametrize("mode", ["greedy", "spec", "dspark"])
def test_serial_frame_unit_does_not_claim_single_token_network_occupancy(mode):
    value = diagnostics(mode)
    assert value["definitions"]["frame_unit"] == "decode_verification_traversal_may_contain_multiple_tokens"
    assert "not physical network occupancy" in value["definitions"]["inflight_level"]


def test_unknown_is_not_a_zero_measurement_and_percentiles_keep_raw_samples():
    value = make_coordinator_diagnostics("greedy", committed_tokens=1)
    assert value["counts"]["proposed_predictions"] is None
    assert value["derived"]["g_cycle"] is None
    assert value["derived"]["inflight_time_avg"] is None
    assert value["inflight_intervals"] is None
    assert percentile([], 95) is None
    assert percentile([1, 9, 3], 50) == 3
    assert percentile([1, 9, 3], 95) == pytest.approx(8.4)


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(accepted_predictions=9),
    lambda c: c.update(proposed_predictions=2),
    lambda c: c.update(frames_enqueued=9),
    lambda c: c.update(replies_received=8),
    lambda c: c.update(stale_replies=10, frames_judged=0, replies_received=None),
    lambda c: c.update(cancel_events=True),
    lambda c: c.update(frames_sent=-1),
])
def test_invalid_or_inconsistent_accounting_is_rejected(mutation):
    counts = counters(); mutation(counts)
    with pytest.raises(MetricsContractError):
        make_coordinator_diagnostics("pipelined", committed_tokens=8, counters=counts)


@pytest.mark.parametrize("intervals", [
    [{"duration_s": math.inf, "level": 1}],
    [{"duration_s": .1, "level": True}],
    [{"duration_s": .1, "level": 1, "event": "unbounded extra data"}],
    [{"duration_s": 1e308, "level": 10}],
])
def test_nonfinite_or_noncontract_inflight_data_is_rejected(intervals):
    with pytest.raises(MetricsContractError):
        make_coordinator_diagnostics("pipelined", committed_tokens=8, inflight_intervals=intervals)


def test_recomputed_diagnostic_and_clock_boundaries_cannot_be_overridden():
    value = diagnostics()
    changed = deepcopy(value); changed["derived"]["g_cycle"] = 40
    with pytest.raises(MetricsContractError, match="raw evidence"):
        validate_coordinator_diagnostics(changed)
    with pytest.raises(MetricsContractError, match="drain boundary"):
        make_coordinator_diagnostics("pipelined", committed_tokens=8,
            timing={"first_token_s": 1, "last_token_s": 2, "request_elapsed_s": 3, "drain_s": 0})

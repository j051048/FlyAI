"""Independent ragged positions cross an actual strict TCP native ring."""
import pytest

from test_v4_native_lane_socket import ring, job, reference
from test_v4_native_batch_socket import advance
from v4_native_batch import PERSISTENT_BATCH_CONTRACT


def request(prompt, count=3, tenant="a"):
    value = job(prompt, count, tenant)
    value.payload["numerical_contract"] = PERSISTENT_BATCH_CONTRACT
    return value


@pytest.mark.parametrize("ring", ["persistent"], indirect=True)
def test_real_strict_ring_ragged_batches_keep_independent_positions_outputs_and_receipts(ring):
    backend, baseline = ring
    first = request(range(1, 13))
    second = request(range(30, 39), tenant="b")
    expected = {j.id: reference(baseline, j.payload["prompt_ids"], j.max_new) for j in (first, second)}
    for _ in range(4):
        assert advance(backend, [first])[first.id] is None
    pending, results = [first, second], {}
    while pending:
        current = advance(backend, list(reversed(pending)))
        for value in tuple(pending):
            result = current[value.id]
            if result is not None:
                assert not isinstance(result, Exception)
                backend.verify_result_proof(value, result)
                assert result["tokens"] == expected[value.id]
                assert result["numerical_contract"] == PERSISTENT_BATCH_CONTRACT
                results[value.id] = result
                backend.release_batch_job(value.id)
                pending.remove(value)
    status = backend._native_runner.status()
    assert status["compute"] == "native_persistent_ragged_batch"
    assert status["batching_geometry"] == "independent_frontier_rows"
    assert status["pipeline"]["batch_forward_cohorts"] > 0
    assert results[first.id]["proof"]["nonce"] != results[second.id]["proof"]["nonce"]
    for result in results.values():
        for receipt in result["proof"]["receipts"]:
            observation = receipt["runtime_observation"]["optimizations"]["native_batch"]
            assert observation["counters"]["state_full_copy_bytes"] == 0
            assert observation["counters"]["ragged_forward_calls"] > 0

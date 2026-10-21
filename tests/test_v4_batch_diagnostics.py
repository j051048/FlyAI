"""One weight allocation, real batched/scalar native fixture diagnostics."""
import pytest
import torch

import v4_ref_cpu as R
import v4_stage as V
from test_v4_conversation_cache import stage
from v4_batch_diagnostics import compare_native_batch, comparison_passed


@pytest.fixture
def model():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
        dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=64)
    yield args, R.build_oracle(args, 7)
    torch.set_num_threads(old)


def run(st, ids, **kwargs):
    return compare_native_batch(st, ids, host_bytes=128 << 20, device_bytes=64 << 20,
        execution_workspace_bytes=8 << 20, reference=True, max_tokens=64, **kwargs)


@pytest.mark.parametrize("count", [2, 3, 4])
def test_single_weight_set_batched_vs_host_scalar_oracles_compare_all_frontiers_and_tail(model, count):
    st = stage(model)
    pointers = [value.data_ptr() for module in st._owned_modules() for value in module.parameters()]
    ids = torch.arange(count * 17).reshape(count, 17) + 1
    report = run(st, ids)
    assert report["evidence"] == "cpu_fixture_arithmetic_only"
    assert report["qualification"]["release_qualified"] is False
    assert report["qualification"]["hardware_claim"] is False
    # This is the original frozen failing B4 trajectory. The new request-row
    # mathematical domain fixes projections while the oracle remains the
    # untouched scalar native path. No relaxed comparison is required.
    assert report["qualification"]["new_batch_abi_within_tolerance"]
    assert report["qualification"]["old_scalar_abi_bitwise_invariant"]
    assert report["qualification"]["topk_equal"]
    assert all(value["exact"] for value in report["states"].values())
    assert report["qualification"]["greedy_equal"]
    assert report["qualification"]["vocab_metrics"] == "actual_tail_vocabulary"
    assert len(report["records"]) == count * 17
    assert all(value["frontiers_compared"] == 17 for value in report["states"].values())
    assert any("score_state" in name for name in report["states"])
    assert any(name.endswith(".index") for name in report["states"])
    assert report["execution"]["batch_forward_calls"] == 17
    assert report["execution"]["scalar_forward_calls"] == 0
    assert report["resources"]["weights_loaded_once"]
    assert report["resources"]["original_storage_bytes"]["cpu"] > 0
    assert pointers == [value.data_ptr() for module in st._owned_modules() for value in module.parameters()]
    assert st._pos == 0 and st._native_lanes is None


def test_non_head_requires_frozen_hidden_and_non_tail_never_fabricates_vocabulary_metrics(model):
    args, oracle = model
    st = V.Stage(1, 3, args, head=False, tail=False, device="cpu")
    for target, source in zip(st.layers, oracle.layers[1:3]):
        target.load_state_dict(source.state_dict())
    ids = torch.arange(2 * 13).reshape(2, 13)
    with pytest.raises(ValueError, match="frozen hidden"):
        run(st, ids)
    frozen = torch.linspace(-1, 1, 2 * 13 * args.hc_mult * args.dim).reshape(
        2, 13, args.hc_mult, args.dim).to(st.dtype)
    report = run(st, ids, hidden=frozen, atol=.0625, rtol=.05)
    assert report["qualification"]["greedy_equal"] is None
    assert report["qualification"]["vocab_metrics"] == "unavailable_non_tail_stage"
    assert report["binding"]["inputs"]["hidden"]["sha256"]
    assert all(row["vocab"] is None for row in report["records"])


def test_real_batch_output_divergence_is_reported_and_quotas_precede_new_allocation(model, monkeypatch):
    st = stage(model)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="quota"):
        compare_native_batch(st, ids, host_bytes=1, device_bytes=1,
            execution_workspace_bytes=1, reference=True)
    actual = st.forward
    def perturb(h, token_ids, pos):
        value = actual(h, token_ids, pos)
        return value + 1 if token_ids.shape[0] > 1 else value
    monkeypatch.setattr(st, "forward", perturb)
    report = run(st, ids)
    assert not report["qualification"]["old_scalar_abi_bitwise_invariant"]
    assert not report["qualification"]["new_batch_abi_within_tolerance"]
    assert report["first_divergence"]["output"] == [0, 0]
    assert report["resources"]["resident_slot_and_batch_workspace_bytes"] > 0


def test_long_b4_trace_has_its_own_bounded_report_quota_before_allocating(model, monkeypatch):
    st = stage(model)
    # Geometry/preflight only: no long-context state is constructed or executed.
    # The batch must admit >16384 aggregate records to reach real context bucket
    # boundaries, then reject the explicitly insufficient report memory quota.
    monkeypatch.setattr(st.args, "max_seq_len", 8192)
    ids = torch.zeros((4, 4097), dtype=torch.long)
    with pytest.raises(ValueError, match="states and workspaces exceed explicit quota"):
        compare_native_batch(st, ids, host_bytes=1, device_bytes=1,
            execution_workspace_bytes=1, max_tokens=8192, reference=True)


@pytest.mark.parametrize("graph_mode", ["off", "partial"])
def test_fixture_cli_writes_frozen_input_bound_report_without_hardware_claim(tmp_path, graph_mode):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    from safetensors.torch import save_file
    frozen = tmp_path / "frozen.safetensors"
    output = tmp_path / "batch.json"
    save_file({"token_ids": torch.tensor([[1, 2, 3, 4, 5], [21, 22, 23, 24, 25]])}, str(frozen))
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "V4_KERNELS": "cpu", "V4_NATIVE_LANES": "0", "V4_NATIVE_BATCH": "0",
           "V4_NATIVE_BATCH_REFERENCE": "0", "V4_NATIVE_LANE_DEVICE_MIB": "0", "V4_CUDA_GRAPH": "0",
           "V4_NATIVE_BATCH_GRAPH": "off", "V4_NATIVE_BATCH_GRAPH_MIB": "0"}
    result = subprocess.run([sys.executable, str(root / "phase0" / "v4_batch_diagnostics.py"),
        "--fixture", "--input", str(frozen), "--out", str(output), "--hi", "8", "--max-tokens", "16",
        "--host-mib", "256", "--device-mib", "128", "--execution-workspace-mib", "8",
        "--batch-graph", graph_mode, "--graph-mib", "8" if graph_mode == "partial" else "0",
        "--graph-limit", "16"], cwd=root, env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["qualification"]["hardware_claim"] is False
    assert report["binding"]["inputs"]["token_ids"]["shape"] == [2, 5]
    assert report["resources"]["weights_loaded_once"]
    assert report["binding"]["batch_runtime"]["graph_policy"]["mode"] == graph_mode
    assert report["binding"]["tolerance"] == {"atol": 0., "rtol": 0.}
    assert not report["qualification"]["graph_capture_observed"]


def test_near_tie_greedy_and_topk_changes_cannot_pass_even_within_tensor_tolerance(model, monkeypatch):
    st = stage(model)
    def tied_logits(h, full_logits=True):
        values = torch.zeros((h.shape[0], 1, st.args.vocab_size), dtype=torch.float32)
        values[..., 0] = 1.0001 if h.shape[0] == 1 else 1.
        values[..., 1] = 1. if h.shape[0] == 1 else 1.0001
        return values
    monkeypatch.setattr(st, "logits_all", tied_logits)
    report = run(st, torch.tensor([[1, 2], [4, 5]]), atol=.1, rtol=.1)
    assert report["qualification"]["new_batch_abi_within_tolerance"]
    assert report["qualification"]["greedy_equal"] is False
    assert report["qualification"]["topk_equal"] is False
    assert comparison_passed(report) is False
    assert report["first_divergence"]["greedy"] == [0, 0]


def test_matching_nonfinite_stage_output_is_never_a_tolerance_pass(model, monkeypatch):
    st = stage(model)
    st.tail = False
    original = st.forward
    def nonfinite(h, ids, pos):
        return torch.full_like(original(h, ids, pos), float("inf"))
    monkeypatch.setattr(st, "forward", nonfinite)
    report = run(st, torch.tensor([[1, 2], [4, 5]]), atol=.1, rtol=.1)
    assert not report["qualification"]["new_batch_abi_within_tolerance"]
    assert comparison_passed(report) is False
    assert report["first_divergence"]["tolerance"]["scope"] == "output"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bitwise_label_distinguishes_signed_zero_even_after_chunk_boundary(dtype):
    from engines.deepseek_v4.v4_batch_diagnostics import _compare
    left = torch.zeros(65537, dtype=dtype)
    right = left.clone()
    right[-1] = -0.
    value = _compare(left, right, atol=0., rtol=0.)
    assert not value["exact"] and value["within_tolerance"] and value["finite_or_matching_infinity"]
    assert value["max_abs"] == 0.
    assert _compare(left, left.clone(), atol=0., rtol=0.)["exact"]


@pytest.mark.parametrize("number,allow,accepted", [(float("nan"), False, False),
    (float("inf"), False, False), (float("-inf"), False, False), (float("-inf"), True, True)])
def test_nonfinite_bits_only_qualify_the_explicit_compressor_score_sentinel(number, allow, accepted):
    from engines.deepseek_v4.v4_batch_diagnostics import _compare
    value = torch.tensor([number])
    result = _compare(value, value.clone(), atol=0., rtol=0., allow_negative_infinity=allow)
    assert result["exact"] is accepted and result["within_tolerance"] is accepted


def test_allocator_retained_blocks_cannot_pass_a_live_tensor_only_budget():
    from engines.deepseek_v4.v4_batch_diagnostics import _within_device_quota
    interval = {"peak_allocated_bytes": 64 << 20, "peak_reserved_bytes": 128 << 20}
    assert _within_device_quota(interval, 96 << 20) is False
    assert _within_device_quota(interval, 128 << 20) is True
    assert _within_device_quota({"peak_allocated_bytes": None, "peak_reserved_bytes": None}, 128 << 20) is None

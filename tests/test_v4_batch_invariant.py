"""Frozen native request batches against independently unchanged scalar math."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import v4_ref_cpu as R
from test_v4_conversation_cache import stage, forward
from v4_batch_diagnostics import compare_native_batch, comparison_passed
from v4_batch_invariant import prepare_native_row_math, request_row_math, CONTRACT
from v4_conversation_cache import _buffers
from v4_native_batch import DeviceNativeLanes
import v4_kernels_cpu as K


def model(seed=7):
    args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
        dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=64)
    return args, R.build_oracle(args, seed)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_first_drift_is_fp32_hyperconnection_projection_shape_not_topk_tie():
    st = stage(model())
    ids = torch.tensor([[1], [18], [35], [52]])
    with torch.no_grad():
        x = st.embed(ids).flatten(2).float()
        weight = st.layers[0].hc_attn_fn
        untouched = F.linear(x, weight)
        scalar = torch.cat([F.linear(x[i:i + 1], weight) for i in range(4)])
        assert not torch.equal(untouched, scalar)
        assert float((untouched - scalar).abs().max()) > 0
        with request_row_math():
            fixed = F.linear(x, weight)
        assert torch.equal(fixed, scalar)
        # The old batched/scalar operation remains unchanged outside the scope.
        assert torch.equal(F.linear(x, weight), untouched)


def test_small_hash_gate_softplus_keeps_scalar_simd_boundary_and_unchanged_outer_scope():
    x = torch.randn(4, 8, generator=torch.Generator().manual_seed(7))
    native = F.softplus(x)
    scalar = torch.cat([F.softplus(x[i:i + 1]) for i in range(4)])
    # This was the second drift, exposed after fixing only GEMM on a longer
    # 37-token trajectory. It must not be hidden by a tolerance or new seed.
    assert not torch.equal(native, scalar)
    with request_row_math():
        actual = F.softplus(x)
    assert torch.equal(actual, scalar)
    assert torch.equal(F.softplus(x), native)


@pytest.mark.parametrize("seed", [0, 7, 23])
@pytest.mark.parametrize("count", [2, 3, 4])
def test_real_native_batch_is_zero_tolerance_equal_across_window_compression_and_indexer(seed, count):
    st = stage(model(seed))
    # 37 crosses window 8, overlapping compressor 4, ordinary compressor 8,
    # and the first discriminating indexer selection at 8 * 4 = 32.
    ids = (torch.arange(count * 37).reshape(count, 37) * 3 + 1) % st.args.vocab_size
    report = compare_native_batch(st, ids, host_bytes=128 << 20, device_bytes=64 << 20,
        execution_workspace_bytes=8 << 20, reference=True, max_tokens=64)
    assert report["binding"]["tolerance"] == {"atol": 0., "rtol": 0.}
    assert report["binding"]["batch_math_contract"] == CONTRACT
    assert report["qualification"]["old_scalar_abi_bitwise_invariant"]
    assert comparison_passed(report)
    assert report["first_divergence"] == dict.fromkeys(("output", "state", "greedy", "topk", "tolerance"))
    assert all(value["exact"] for value in report["states"].values())
    assert report["execution"]["batch_forward_calls"] == 37
    assert report["execution"]["scalar_forward_calls"] == 0
    assert report["qualification"]["hardware_claim"] is False


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_one_stage_call_request_permutation_and_b1_preserve_exact_native_state(count, monkeypatch):
    weights = model(11)
    run = DeviceNativeLanes(stage(weights), max_jobs=4, host_bytes=32 << 20,
        device_bytes=32 << 20, reference=True)
    controls = [stage(weights) for _ in range(count)]
    for row in range(count):
        run.open(str(row), identity=str(row) * 64, horizon=64)
    actual, calls = run.stage.forward, []
    def observe(h, ids, pos):
        calls.append(tuple(ids.shape))
        return actual(h, ids, pos)
    monkeypatch.setattr(run.stage, "forward", observe)
    for pos in range(37):
        order = list(reversed(range(count))) if pos % 2 else list(range(count))
        ids = torch.tensor([[(row * 41 + pos + 1) % weights[0].vocab_size] for row in order])
        with torch.no_grad():
            got = run.forward_batch([str(row) for row in order], run.stage.embed(ids), ids, pos)
            logits = run.logits_all(got, full_logits=False)
        for index, row in enumerate(order):
            expected, want_logits = forward(controls[row], ids[index:index + 1], pos)
            assert torch.equal(got[index:index + 1], expected)
            assert torch.equal(logits[index:index + 1], want_logits)
            for name, value in _buffers(controls[row], None).items():
                assert torch.equal(run.active[str(row)]["buffers"][name], value[0:1]), (pos, row, name)
    assert calls == [(count, 1)] * 37


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_projected_rows_and_tied_topk_are_permutation_invariant_without_new_tie_policy(dtype):
    generator = torch.Generator().manual_seed(37)
    x = torch.randn(4, 1, 256, generator=generator).to(dtype)
    weight = torch.randn(13, 256, generator=generator).to(dtype)
    expected = torch.cat([F.linear(x[i:i + 1], weight) for i in range(4)])
    permutation = torch.tensor([3, 1, 0, 2])
    with request_row_math():
        got = F.linear(x, weight)
        shuffled = F.linear(x[permutation], weight)
        ties = torch.tensor([[1., 1., 1., 0.], [0., 2., 2., 2.], [0., 0., 0., 0.], [3., 3., 1., 0.]])
        top = ties.topk(3)
    assert torch.equal(got, expected)
    assert torch.equal(shuffled, expected[permutation])
    assert torch.equal(top.indices, torch.cat([ties[i:i + 1].topk(3).indices for i in range(4)]))


def test_empty_indexer_topk_before_first_compression_retains_native_empty_shape():
    empty = torch.empty(4, 1, 0)
    with request_row_math():
        result = empty.topk(0)
    assert result.values.shape == result.indices.shape == (4, 1, 0)


def test_native_quantized_kernel_adapters_keep_original_weights_and_are_dormant_thread_local():
    calls = []
    def kernel(a, scale, weight, weight_scale, output_dtype=torch.float32):
        calls.append((a.shape[0], weight.data_ptr(), weight_scale.data_ptr()))
        return (a.float() @ weight.float().T).to(output_dtype)
    native = SimpleNamespace(fp4_gemm=kernel, fp8_gemm=kernel)
    prepare_native_row_math(native)
    wrapper = native.fp4_gemm
    prepare_native_row_math(native)
    assert native.fp4_gemm is wrapper
    x, scales = torch.randn(4, 128), torch.ones(4, 1)
    weight, weight_scale = torch.randn(16, 128), torch.ones(16, 1)
    native.fp4_gemm(x, scales, weight, weight_scale)
    assert calls == [(4, weight.data_ptr(), weight_scale.data_ptr())]
    calls.clear()
    with request_row_math():
        got = native.fp4_gemm(x, scales, weight, weight_scale)
        # A concurrently executing scalar/baseline thread remains out of scope.
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(native.fp8_gemm, x, scales, weight, weight_scale).result()
    assert [row[0] for row in calls] == [1, 1, 1, 1, 4]
    assert all(row[1:] == (weight.data_ptr(), weight_scale.data_ptr()) for row in calls)
    expected = torch.cat([kernel(x[i:i + 1], scales[i:i + 1], weight, weight_scale) for i in range(4)])
    assert torch.equal(got, expected)


@pytest.mark.parametrize("kind", ["fp4", "fp8"])
def test_actual_cpu_packed_quantized_gemms_preserve_scalar_row_math_and_weights(kind):
    generator = torch.Generator().manual_seed(19)
    activation = torch.randn(4, 128, generator=generator)
    weight = torch.randn(128, 128, generator=generator)
    a, scales = K.act_quant(activation)
    if kind == "fp4":
        packed, weight_scales = K.fp4_act_quant(weight)
        original = K.fp4_gemm
    else:
        packed = weight.to(torch.float8_e4m3fn)
        weight_scales = torch.ones(1, 1)
        original = K.fp8_gemm
    before = packed.view(torch.uint8).clone()
    native = SimpleNamespace(fp4_gemm=K.fp4_gemm, fp8_gemm=K.fp8_gemm)
    prepare_native_row_math(native)
    expected = torch.cat([original(a[i:i + 1], scales[i:i + 1], packed, weight_scales) for i in range(4)])
    with request_row_math():
        actual = getattr(native, kind + "_gemm")(a, scales, packed, weight_scales)
    assert torch.equal(actual, expected)
    assert torch.equal(packed.view(torch.uint8), before)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_projection_cannot_be_fabricated_into_finite_evidence(bad):
    x, weight = torch.ones(4, 16), torch.ones(4, 16)
    x[2, 0] = bad
    with request_row_math():
        value = F.linear(x, weight)
    assert not torch.isfinite(value[2]).all()
    assert torch.isfinite(value[[0, 1, 3]]).all()

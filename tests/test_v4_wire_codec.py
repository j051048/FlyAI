"""Wire bytes are the acceptance bar; CPU dispatcher tests are not a fusion benchmark."""
import json
import warnings

import pytest

torch = pytest.importorskip("torch")
CODEC = pytest.importorskip("v4_wire_codec")


def equal_bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def activation(shape=(1, 2, 4, 64), dtype=torch.bfloat16, *, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(91)
    return torch.randn(shape, generator=generator, dtype=torch.float32, device=device).to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_reference_and_cpu_dispatch_are_identical_to_original_pipe(dtype, monkeypatch):
    VP = pytest.importorskip("v4_pipe")
    monkeypatch.setattr(VP, "V4_WIRE_FUSED", False, raising=False)
    t = activation(dtype=dtype)
    original_q, original_scale = VP._pack_t(t)
    q, scale = CODEC.pack(t)
    equal_bits(q, original_q); equal_bits(scale, original_scale)
    equal_bits(CODEC.unpack(q, scale), VP._unpack_t(original_q, original_scale))
    ids = torch.tensor([[13, 17]])
    assert VP._wire_bytes(q, ids, scale) == VP._wire_bytes(original_q, ids, original_scale)


def test_zero_fp16_minimum_scale_is_fp32_clamped_and_not_zero():
    t = torch.zeros(1, 1, 4, 64, dtype=torch.float16)
    q, scale = CODEC.reference_pack(t)
    assert bool((scale > 0).all())
    assert bool(torch.isfinite(q.float()).all())
    assert bool((q.float() == 0).all())
    assert scale.dtype == torch.bfloat16
    equal_bits(CODEC.reference_unpack(q, scale), torch.zeros_like(t, dtype=torch.bfloat16))


def test_scales_are_per_position_and_stream_and_use_the_bf16_rounded_divisor():
    t = activation((2, 3, 4, 65))
    factors = torch.exp2(torch.arange(24).reshape(2, 3, 4, 1) - 12)
    t = (t.float() * factors).bfloat16()
    q, scale = CODEC.reference_pack(t)
    assert scale.shape == t.shape[:-1]
    reconstructed = (t.float() / scale.float().unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    equal_bits(q, reconstructed)
    for position in range(3):
        one_q, one_scale = CODEC.reference_pack(t[:, position:position + 1])
        equal_bits(one_q, q[:, position:position + 1])
        equal_bits(one_scale, scale[:, position:position + 1])


def test_nan_infinity_signed_zero_and_payload_bits_follow_original_codec(monkeypatch):
    VP = pytest.importorskip("v4_pipe")
    monkeypatch.setattr(VP, "V4_WIRE_FUSED", False, raising=False)
    t = activation((1, 2, 4, 64))
    t[0, 0, 0, 0] = float("inf")
    t[0, 0, 1, 0] = -float("inf")
    t[0, 0, 2].zero_(); t[0, 0, 2, 0] = -0.0
    t[0, 0, 3].view(torch.int16)[0] = -64  # a signed BF16 NaN payload
    q, scale = CODEC.reference_pack(t)
    old_q, old_scale = VP._pack_t(t)
    equal_bits(q, old_q); equal_bits(scale, old_scale)
    equal_bits(CODEC.reference_unpack(q, scale), VP._unpack_t(old_q, old_scale))
    assert q.view(torch.uint8)[0, 0, 2, 0].item() == 128
    assert bool(torch.isfinite(q[0, 0, 0].float()).all())
    assert bool(torch.isnan(q[0, 0, 3].float()).all())


@pytest.mark.parametrize("scale_value", [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1e-8])
def test_decoder_keeps_bf16_cpu_product_bytes_for_special_wire_values(scale_value):
    codes = torch.tensor([0, 128, 56, 184, 126, 254, 127, 255], dtype=torch.uint8)
    q = codes.view(torch.float8_e4m3fn).reshape(1, 1, 1, 8)
    scale = torch.full((1, 1, 1), scale_value, dtype=torch.bfloat16)
    equal_bits(CODEC.unpack(q, scale), q.to(torch.bfloat16) * scale.unsqueeze(-1))


def test_cpu_warmup_and_forward_do_not_initialize_a_compiler(monkeypatch):
    codec = CODEC.WireCodec()
    monkeypatch.setattr(torch, "compile", lambda *a, **kw: pytest.fail("CPU codec must not compile"))
    t = activation()
    state = codec.warmup(t)
    assert not state["ready"] and state["warmup_skips"] == 1
    q, scale = codec.pack(t)
    codec.unpack(q, scale)
    assert codec.stats()["compiled_calls"] == 0 and codec.stats()["fallbacks"]["cpu"] == 2
    json.dumps(codec.stats(), allow_nan=False)


def fake_cuda_dispatcher(monkeypatch, compiler, **kwargs):
    """Explicit CPU simulation of dispatcher states; never invokes CUDA or torch.compile."""
    codec = CODEC.WireCodec(compiler=compiler, **kwargs)
    monkeypatch.setattr(codec, "_cuda_inputs", lambda inputs: True)
    monkeypatch.setattr(codec, "_eligible", lambda op, inputs: True)
    monkeypatch.setattr(codec, "_capturing", lambda inputs: False)
    return codec


def test_unwarmed_key_runs_reference_without_invoking_compiler(monkeypatch):
    codec = fake_cuda_dispatcher(monkeypatch, lambda *a: pytest.fail("hot path cannot compile"))
    t = activation()
    q, scale = codec.pack(t)
    old_q, old_scale = CODEC.reference_pack(t)
    equal_bits(q, old_q); equal_bits(scale, old_scale)
    codec.unpack(q, scale)
    assert codec.stats()["fallbacks"]["unwarmed"] == 2


def test_verified_dispatcher_calls_compiled_function_and_keeps_each_frame_owned(monkeypatch):
    options_seen, calls = [], {"reference_pack": 0, "reference_unpack": 0}

    def compiler(fn, options):
        options_seen.append(options)

        def compiled(*inputs):
            calls[fn.__name__] += 1
            return fn(*inputs)
        return compiled

    codec = fake_cuda_dispatcher(monkeypatch, compiler)
    t = activation()
    state = codec.warmup(t)
    assert state["ready_keys"] == 2 and state["ready"] is True
    before = dict(calls)
    q, scale = codec.pack(t)
    saved_q, saved_scale = q.clone(), scale.clone()
    decoded = codec.unpack(q, scale)
    saved_decoded = decoded.clone()
    codec.pack(t * 2); codec.unpack(q, scale * 2)
    equal_bits(q, saved_q); equal_bits(scale, saved_scale); equal_bits(decoded, saved_decoded)
    assert calls["reference_pack"] == before["reference_pack"] + 2
    assert calls["reference_unpack"] == before["reference_unpack"] + 2
    assert codec.stats()["compiled_calls"] == 4
    assert all(o["emulate_precision_casts"] is True and o["triton.cudagraphs"] is False
               and o["eager_numerics.division_rounding"] is True for o in options_seen)


def test_compile_failure_is_sticky_reported_once_per_key_and_reference_survives(monkeypatch):
    attempts = []

    def broken(fn, options):
        attempts.append(fn.__name__)
        raise RuntimeError("compiler unavailable")

    codec = fake_cuda_dispatcher(monkeypatch, broken)
    t = activation()
    with pytest.warns(RuntimeWarning) as caught:
        codec.warmup(t)
    assert len(caught) == 2 and codec.stats()["compile_failures"] == 2
    with warnings.catch_warnings(record=True) as caught:
        codec.warmup(t); codec.pack(t); codec.pack(t)
    assert not caught and len(attempts) == 2
    assert codec.stats()["declined_keys"] == 2 and codec.stats()["fallbacks"]["failed_key"] == 2


def test_parity_gate_rejects_one_changed_wire_byte(monkeypatch):
    def wrong(fn, options):
        if fn is CODEC.reference_unpack:
            return fn

        def corrupt(t):
            q, scale = fn(t)
            q.view(torch.uint8).reshape(-1)[0] ^= 1
            return q, scale
        return corrupt

    codec = fake_cuda_dispatcher(monkeypatch, wrong)
    t = activation()
    with pytest.warns(RuntimeWarning, match="CodecParityError"):
        codec.warmup(t)
    assert codec.stats()["parity_failures"] == 1 and codec.stats()["ready_pack"] == 0
    q, scale = codec.pack(t)
    old_q, old_scale = CODEC.reference_pack(t)
    equal_bits(q, old_q); equal_bits(scale, old_scale)


def test_parity_gate_rejects_shared_output_that_overwrites_a_pending_frame(monkeypatch):
    def shared(fn, options):
        buffers = []

        def compiled(*inputs):
            result = fn(*inputs)
            values = result if isinstance(result, tuple) else (result,)
            if not buffers:
                buffers.extend(t.clone() for t in values)
            for buffer, value in zip(buffers, values):
                buffer.copy_(value)
            return tuple(buffers) if isinstance(result, tuple) else buffers[0]
        return compiled

    codec = fake_cuda_dispatcher(monkeypatch, shared)
    with pytest.warns(RuntimeWarning, match="retained frame"):
        codec.warmup(activation())
    assert codec.stats()["ready_keys"] == 0 and codec.stats()["parity_failures"] == 2


def test_runtime_failure_disables_only_failed_key_without_retry_or_hot_probe(monkeypatch):
    broken = {"value": False}

    def compiler(fn, options):
        def compiled(*inputs):
            if broken["value"] and fn is CODEC.reference_pack:
                raise RuntimeError("launch failed")
            return fn(*inputs)
        return compiled

    codec = fake_cuda_dispatcher(monkeypatch, compiler)
    t = activation()
    codec.warmup(t)
    broken["value"] = True
    with pytest.warns(RuntimeWarning, match="launch failed"):
        q, scale = codec.pack(t)
    with warnings.catch_warnings(record=True) as caught:
        codec.pack(t)
    assert not caught
    codec.unpack(q, scale)
    stats = codec.stats()
    assert stats["runtime_failures"] == 1 and stats["ready_unpack"] == 1 and stats["ready_pack"] == 0


def test_capture_fallback_is_observable_once_and_does_not_disable_verified_key(monkeypatch):
    codec = fake_cuda_dispatcher(monkeypatch, lambda fn, options: fn)
    t = activation()
    codec.warmup(t)
    monkeypatch.setattr(codec, "_capturing", lambda inputs: True)
    with pytest.warns(RuntimeWarning, match="outside CUDA capture"):
        codec.pack(t)
    with warnings.catch_warnings(record=True) as caught:
        codec.pack(t)
    assert not caught and codec.stats()["fallbacks"]["capture"] == 2
    monkeypatch.setattr(codec, "_capturing", lambda inputs: False)
    codec.pack(t)
    assert codec.stats()["uses"]["pack"] == 1


def test_cache_bound_and_large_warmup_bound_do_not_trigger_live_compilation(monkeypatch):
    codec = fake_cuda_dispatcher(monkeypatch, lambda fn, options: fn, max_keys=1, max_probe_elements=1024)
    codec.warmup(activation((1, 1, 4, 64)))
    codec.warmup(activation((1, 2, 4, 64)))
    codec.pack(activation((1, 2, 4, 64)))
    assert codec.stats()["ready_keys"] == 2 and codec.stats()["fallbacks"]["unwarmed"] == 1
    assert codec.warmup(activation((1, 5, 4, 64)))["warmup_skips"] == 3


@pytest.mark.parametrize("inference_input", [False, True])
def test_warmup_and_live_call_use_same_grad_and_inference_specialization(monkeypatch, inference_input):
    observed = []

    def compiler(fn, options):
        def compiled(*inputs):
            observed.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled(), inputs[0].is_inference()))
            return fn(*inputs)
        return compiled

    with torch.inference_mode(inference_input):
        t = activation()
    codec = fake_cuda_dispatcher(monkeypatch, compiler)
    codec.warmup(t)
    codec.pack(t)
    assert all(state == (False, inference_input, inference_input) for state in observed)


def test_receiver_admission_is_exact_shape_metadata_without_h2d(monkeypatch):
    codec = CODEC.WireCodec()
    q, scale = CODEC.reference_pack(activation())
    signatures = CODEC._signature(q, scale)
    virtual = tuple(("cuda:0",) + signature[1:] for signature in signatures)
    codec._entries["unpack"][virtual] = CODEC._Entry(state="ready", fn=lambda *a: None)
    monkeypatch.setattr(torch.Tensor, "to", lambda *a, **kw: pytest.fail("admission cannot copy a tensor"))
    assert codec.can_unpack(q, scale, device="cuda:0") is True
    assert codec.can_unpack(q[:, :1], scale[:, :1], device="cuda:0") is False
    assert codec.can_unpack(q, scale, device="cpu") is False


@pytest.mark.gpu
@pytest.mark.hardware
@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled CUDA codec parity requires a GPU")
@pytest.mark.parametrize("shape", [(1, 1, 4, 4096), (1, 9, 4, 256)])
def test_actual_cuda_compilation_and_forward_are_byte_exact_to_old_cpu_receiver(shape):
    codec = CODEC.WireCodec()
    t = activation(shape, device="cuda:0")
    state = codec.warmup(t)
    if state["compile_failures"]:
        pytest.skip("CUDA compiler/dependency/precision controls unavailable: " + str(state["declined"]))
    assert state["ready_pack"] == 1, state  # a pack mismatch cannot qualify as a fused encoder
    if not state["ready_unpack"]:
        # CPU and CUDA can choose different NaN payloads. Such a key must explicitly decline
        # GPU decoding and preserve the original CPU product, while pack can remain accelerated.
        assert state["declined"]["unpack"] and state["parity_failures"] == 1, state
        assert "CPU receiver" in state["declined"]["unpack"][0]
    for probe in codec._probes(t):
        original_q, original_scale = CODEC.reference_pack(probe)
        q, scale = codec.pack(probe)
        equal_bits(q, original_q); equal_bits(scale, original_scale)
        assert codec.can_unpack(q.cpu(), scale.cpu(), device=probe.device) == bool(state["ready_unpack"])
        equal_bits(codec.unpack(q, scale).cpu(), CODEC.reference_unpack(original_q.cpu(), original_scale.cpu()))
    saved_q, saved_scale = codec.pack(t)
    q_snapshot, scale_snapshot = saved_q.clone(), saved_scale.clone()
    codec.pack(t * 2)
    equal_bits(saved_q, q_snapshot); equal_bits(saved_scale, scale_snapshot)
    assert codec.stats()["compiled_calls"] > 0

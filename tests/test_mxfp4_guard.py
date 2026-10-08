"""Real packed-layout fixtures and pre-conversion fallback interception, CPU only."""
from contextlib import contextmanager
import sys
from types import SimpleNamespace, ModuleType
import threading

import pytest
torch = pytest.importorskip("torch")

from mxfp4_guard import (MXFP4Error, parse_version_tuple, verify_mxfp4_runtime_environment,
    config_uses_mxfp4, native_mxfp4_loading, assert_mxfp4_quantized, inspect_mxfp4_quantized)


def cfg(**changes):
    values = dict(quantization_config={"quant_method": "mxfp4", "dequantize": False},
                  num_local_experts=2, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  tie_word_embeddings=False, layer_types=None, sliding_window=0)
    values.update(changes)
    return SimpleNamespace(**values)


class PackedExperts(torch.nn.Module):
    def __init__(self, device="cpu", *, buffers=False):
        super().__init__()
        self.num_experts, self.hidden_size, self.intermediate_size = 2, 32, 64
        for name, shape in (("gate_up_proj", (2, 128, 1, 16)), ("down_proj", (2, 32, 2, 16))):
            value = torch.zeros(shape, dtype=torch.uint8, device=device)
            if buffers:
                self.register_buffer(name, value)
            else:
                self.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
        self.register_buffer("gate_up_proj_scales", torch.zeros(2, 128, 1, dtype=torch.uint8, device=device))
        self.register_buffer("down_proj_scales", torch.zeros(2, 32, 2, dtype=torch.uint8, device=device))
        self.gate_up_proj_bias = torch.nn.Parameter(torch.zeros(2, 128, device=device), requires_grad=False)
        self.down_proj_bias = torch.nn.Parameter(torch.zeros(2, 32, device=device), requires_grad=False)


class Layer(torch.nn.Module):
    def __init__(self, device="cpu", buffers=False):
        super().__init__()
        self.experts = PackedExperts(device, buffers=buffers)
        self.self_attn = SimpleNamespace(layer_idx=0)


class Model(torch.nn.Module):
    def __init__(self, devices=("cpu", "cpu"), *, buffers=False):
        super().__init__()
        self.config = cfg()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Layer(device, buffers) for device in devices])
        self.model.rotary_emb = torch.nn.Identity()
        self.model.embed_tokens = torch.nn.Embedding(8, 32)
        self.model.norm = torch.nn.LayerNorm(32)
        self.lm_head = torch.nn.Linear(32, 8)


class FakeQuantizer:
    def __init__(self, phase=None):
        self.phase = phase
        self.quantization_config = SimpleNamespace(quant_method="mxfp4", dequantize=False)

    def validate_environment(self, *args, **kwargs):
        if self.phase == "validate":
            self.quantization_config.dequantize = True

    def _process_model_before_weight_loading(self, model, **kwargs):
        if self.phase == "before":
            self.quantization_config.dequantize = True

    def get_weight_conversions(self):
        if self.phase == "conversion":
            class Mxfp4Dequantize:
                pass
            return [SimpleNamespace(operations=[Mxfp4Dequantize()])]
        return [SimpleNamespace(operations=[])]


def test_parse_version_is_not_digit_concatenation():
    assert parse_version_tuple("5.19.0") == (5, 19, 0)
    assert parse_version_tuple("5.19.0.dev123") == (5, 19, 0)
    assert parse_version_tuple("2.11.0+cu130") == (2, 11, 0)


def dependencies(monkeypatch, kernel_version):
    tf, kernels, bounds = ModuleType("transformers"), ModuleType("kernels"), ModuleType("transformers.utils.import_utils")
    tf.__version__, kernels.__version__ = "5.19.0", kernel_version
    bounds.KERNELS_MIN_VERSION, bounds.KERNELS_MAX_VERSION = "0.17.0", "0.18.0"
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "kernels", kernels)
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", bounds)


def test_dependency_bounds_prerequisite_is_not_native_execution_pass(monkeypatch):
    dependencies(monkeypatch, "0.17.2")
    result = verify_mxfp4_runtime_environment()
    assert result["ok"] and result["native_execution_verified"] is False
    assert "prerequisites" in result["scope"]


def test_dependency_mismatch_and_unknown_abi_fail(monkeypatch):
    dependencies(monkeypatch, "0.14.1")
    with pytest.raises(MXFP4Error, match="ABI"):
        verify_mxfp4_runtime_environment()
    assert verify_mxfp4_runtime_environment(enforce=False)["ok"] is False
    dependencies(monkeypatch, "0.17.2")
    del sys.modules["transformers.utils.import_utils"].KERNELS_MAX_VERSION
    with pytest.raises(MXFP4Error, match="bounds"):
        verify_mxfp4_runtime_environment()


@pytest.mark.parametrize("phase", ["validate", "before", "conversion"])
def test_hf_fallback_rejected_before_first_weight_allocation_and_hooks_restored(phase):
    original = {name: getattr(FakeQuantizer, name) for name in
                ("validate_environment", "_process_model_before_weight_loading", "get_weight_conversions")}
    allocations = []
    quantizer = FakeQuantizer(phase)
    with pytest.raises(MXFP4Error):
        with native_mxfp4_loading(cfg(), quantizer_class=FakeQuantizer):
            quantizer.validate_environment()
            quantizer._process_model_before_weight_loading(Model(devices=("meta", "meta")))
            quantizer.get_weight_conversions()
            allocations.append("expanded weights would load here")
    assert allocations == []
    assert all(getattr(FakeQuantizer, name) is value for name, value in original.items())


def test_excluded_dense_expert_rejected_at_before_weight_hook():
    model = Model(devices=("meta", "meta"))
    model.model.layers[0].experts.gate_up_proj = torch.nn.Parameter(torch.zeros(2, 128, 32, device="meta"), requires_grad=False)
    with pytest.raises(MXFP4Error, match="unconverted"):
        with native_mxfp4_loading(cfg(), quantizer_class=FakeQuantizer):
            FakeQuantizer()._process_model_before_weight_loading(model)


def test_native_context_success_restoration_and_non_mxfp4_compatibility():
    before = FakeQuantizer.validate_environment
    with native_mxfp4_loading(cfg(), quantizer_class=FakeQuantizer):
        quantizer = FakeQuantizer()
        quantizer.validate_environment()
        quantizer._process_model_before_weight_loading(Model(devices=("meta", "meta")))
        quantizer.get_weight_conversions()
    assert FakeQuantizer.validate_environment is before
    with native_mxfp4_loading(cfg(quantization_config=None), quantizer_class=object):
        pass
    assert not config_uses_mxfp4(cfg(quantization_config={"quant_method": "bitsandbytes"}))
    assert config_uses_mxfp4({"quantization_config": {"quant_method": "mxfp4"}})


def test_explicit_dequantization_and_unsupported_loader_hooks_refused():
    with pytest.raises(MXFP4Error, match="before weight"):
        with native_mxfp4_loading(cfg(quantization_config={"quant_method": "mxfp4", "dequantize": True}),
                                 quantizer_class=FakeQuantizer):
            pytest.fail("load body cannot run")
    with pytest.raises(MXFP4Error, match="seam"):
        with native_mxfp4_loading(cfg(), quantizer_class=object):
            pass


@pytest.mark.parametrize("buffers", [False, True])
def test_real_uint8_parameters_or_buffers_and_scales_are_verified(buffers):
    model = Model(buffers=buffers)
    report = inspect_mxfp4_quantized(model, layer_range=(0, 2), layer_path="model.layers", expected_device="cpu")
    assert report["packed_layout_verified"] is True and report["native_execution_verified"] is False
    assert len(report["experts"]) == 2
    assert report["packed_weight_scale_bytes"] == 2 * (4096 + 256 + 2048 + 128)
    assert assert_mxfp4_quantized(model)


def make_wrapped(module):
    for projection, logical, scale_shape, byte_count in (
        ("gate_up_proj", (2, 32, 128), (2, 1, 128), 4096),
        ("down_proj", (2, 64, 32), (2, 2, 32), 2048),
    ):
        del module._parameters[projection]
        del module._buffers[projection + "_scales"]
        setattr(module, projection, SimpleNamespace(dtype="fp4", shape=logical,
            storage=SimpleNamespace(data=torch.zeros(byte_count, dtype=torch.uint8))))
        scale = SimpleNamespace(shape=scale_shape, storage=SimpleNamespace(
            data=torch.zeros(byte_count // 16, dtype=torch.uint8)))
        setattr(module, projection + "_precision_config", SimpleNamespace(weight_scale=scale))


def test_actual_hf_triton_storage_structure_after_parameter_removal():
    model = Model()
    for layer in model.model.layers:
        make_wrapped(layer.experts)
    assert inspect_mxfp4_quantized(model)["packed_layout_verified"]


def test_opaque_native_fp4_enum_uses_already_loaded_hub_identity(monkeypatch):
    marker = object()
    integration = ModuleType("transformers.integrations.mxfp4")
    integration.triton_kernels_hub = SimpleNamespace(tensor=SimpleNamespace(FP4=marker))
    monkeypatch.setitem(sys.modules, "transformers.integrations.mxfp4", integration)
    model = Model()
    for layer in model.model.layers:
        make_wrapped(layer.experts)
        layer.experts.gate_up_proj.dtype = marker
        layer.experts.down_proj.dtype = marker
    assert inspect_mxfp4_quantized(model)["packed_layout_verified"]


@pytest.mark.parametrize("corruption", ["bf16", "scale_dtype", "scale_shape", "missing_scale", "extra_dense", "fake_bias", "meta"])
def test_native_claim_does_not_hide_expanded_or_malformed_storage(corruption):
    model = Model()
    expert = model.model.layers[0].experts
    if corruption == "bf16":
        expert.gate_up_proj = torch.nn.Parameter(torch.zeros(2, 128, 32, dtype=torch.bfloat16), requires_grad=False)
    elif corruption == "scale_dtype":
        expert.gate_up_proj_scales = expert.gate_up_proj_scales.float()
    elif corruption == "scale_shape":
        expert.gate_up_proj_scales = torch.zeros(2, 127, 1, dtype=torch.uint8)
    elif corruption == "missing_scale":
        del expert.gate_up_proj_scales
    elif corruption == "extra_dense":
        expert.register_buffer("expanded_extra_weight", torch.zeros(2, 128, 32, dtype=torch.bfloat16))
    elif corruption == "fake_bias":
        expert.register_parameter("fake_bias_weight", torch.nn.Parameter(torch.zeros(2, 128, 32), requires_grad=False))
    else:
        model.model.layers[0] = Layer("meta")
    with pytest.raises(MXFP4Error):
        inspect_mxfp4_quantized(model)


def test_assigned_range_requires_outside_layers_meta_and_matching_devices():
    model = Model(devices=("cpu", "meta"))
    assert inspect_mxfp4_quantized(model, layer_range=(0, 1), expected_device="cpu")
    with pytest.raises(MXFP4Error, match="wrong assigned device"):
        inspect_mxfp4_quantized(model, layer_range=(0, 1), expected_device="cuda:0")
    with pytest.raises(MXFP4Error, match="unassigned"):
        inspect_mxfp4_quantized(Model(), layer_range=(0, 1))
    with pytest.raises(MXFP4Error, match="range"):
        inspect_mxfp4_quantized(model, layer_range=(1, 3))


def test_missing_model_empty_experts_and_dequantized_quantizer_cannot_pass():
    with pytest.raises(MXFP4Error, match="real MXFP4"):
        assert_mxfp4_quantized(None)
    model = Model()
    model.model.layers[0].experts = torch.nn.Identity()
    with pytest.raises(MXFP4Error, match="missing packed"):
        inspect_mxfp4_quantized(model)
    model = Model()
    model.hf_quantizer = FakeQuantizer()
    model.hf_quantizer.quantization_config.dequantize = True
    with pytest.raises(MXFP4Error, match="fallback"):
        inspect_mxfp4_quantized(model)


@pytest.mark.parametrize("native", [False, True])
def test_pipeline_uses_checkpoint_config_not_directory_name(monkeypatch, native):
    import pipeline
    import transformers
    import mxfp4_guard as guard
    model = Model(devices=("cpu", "meta"))
    model.config = cfg(quantization_config={"quant_method": "mxfp4", "dequantize": False} if native else None)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: model.config)
    calls, phases = [], []
    def load(_name, **kwargs):
        calls.append(kwargs)
        return model
    monkeypatch.setattr(pipeline.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(pipeline.torch.cuda, "memory_allocated", lambda *_: 0)
    monkeypatch.setattr(guard, "verify_mxfp4_runtime_environment", lambda **kw: phases.append("prerequisite"))
    @contextmanager
    def context(config):
        phases.append("before")
        yield
        phases.append("after")
    monkeypatch.setattr(guard, "native_mxfp4_loading", context)
    parts = pipeline.load_stage("/renamed/local/model" if native else "/local/gpt-oss-nonquant",
                                0, 2, device="cpu", lo=0, hi=1)
    assert len(parts["layers"]) == 1 and calls[0]["config"] is model.config
    assert phases == (["prerequisite", "before", "after"] if native else [])

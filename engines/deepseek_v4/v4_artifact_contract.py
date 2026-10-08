"""Supported production V4 native artifact contract; no Torch/model imports.

Tiny CPU modules can exercise the same byte loader without claiming this real
model/quantization contract. V4.1's different graph/config is not a drop-in.
"""
MODEL_IDS = frozenset(("deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash-0731"))
RUNTIME_ABI = "deepseek-v4-native/1"
WIRE_VERSION = "shard-pipeline-session/1"
NUMERIC_CONTRACT = "greedy-native-fp4-fp8/1"
QUANTIZATION = "fp4-fp8"


def validate_native_config(config, *, model_id, runtime_abi):
    if model_id not in MODEL_IDS or runtime_abi != RUNTIME_ABI:
        raise ValueError("unsupported V4 model/runtime ABI; V4.1 requires a separate adapter")
    expected = {"n_layers": 43, "dim": 4096, "n_routed_experts": 256,
                "n_activated_experts": 6, "hc_mult": 4, "n_mtp_layers": 3,
                "dtype": "fp8", "expert_dtype": "fp4", "scale_fmt": "ue8m0"}
    if any(type(config.get(name)) is not type(value) or config.get(name) != value
           for name, value in expected.items()):
        raise ValueError("unsupported V4 native dimensions/quantization; require the original FP8/FP4 Flash configuration")
    if config.get("dspark_target_layer_ids") != [40, 41, 42]:
        raise ValueError("unsupported V4 DSpark dependencies; require tap layers 40..42")
    return config


def validate_native_cohort(cohort, config):
    supported = {"runtime_abi": RUNTIME_ABI, "wire_version": WIRE_VERSION,
                 "numeric_contract": NUMERIC_CONTRACT, "quantization": QUANTIZATION}
    if any(cohort.get(name) != value for name, value in supported.items()):
        raise ValueError("unsupported V4 cohort ABI/wire/numeric/quantization declaration")
    validate_native_config(config, model_id=cohort.get("model_id"), runtime_abi=cohort["runtime_abi"])
    if type(cohort.get("n_layers")) is not int or cohort["n_layers"] != config["n_layers"]:
        raise ValueError("V4 cohort layer count differs from native configuration")
    return dict(cohort)

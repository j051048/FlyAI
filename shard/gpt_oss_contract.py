"""The GPT-OSS service's supported implementation contract and cohort recipe.

Validation imports only the standard library. These fields describe software
compatibility; they do not attest hardware, kernels, speed or numeric parity.
Callers must additionally bind config_sha256 to their actual raw config file.
"""
import re

RUNTIME_ABI = "gpt-oss-hf/1"
WIRE_VERSION = "shard-pipeline-session/1"
NUMERIC_CONTRACT = "greedy-native-mxfp4/1"
QUANTIZATION = "mxfp4"

_FIELDS = {"model_id", "manifest_sha256", "checkpoint_id", "config_sha256",
           "quantization", "runtime_abi", "wire_version", "numeric_contract", "n_layers"}
_SUPPORTED = {"runtime_abi": RUNTIME_ABI, "wire_version": WIRE_VERSION,
              "numeric_contract": NUMERIC_CONTRACT, "quantization": QUANTIZATION}


class GPTOSSContractError(ValueError):
    pass


def _mapping(value, label):
    if not isinstance(value, dict):
        method = getattr(value, "to_dict", None)
        value = method() if callable(method) else None
    if not isinstance(value, dict):
        raise GPTOSSContractError(f"{label} must be a descriptor object")
    return value


def _config_layers(config):
    config = _mapping(config, "config")
    if config.get("model_type") != "gpt_oss":
        raise GPTOSSContractError("supported runtime requires actual model_type=gpt_oss")
    quant = config.get("quantization_config")
    if not isinstance(quant, dict) or quant.get("quant_method") != QUANTIZATION:
        raise GPTOSSContractError("supported runtime requires explicit native MXFP4 config")
    if "dequantize" in quant and quant["dequantize"] is not False:
        raise GPTOSSContractError("native MXFP4 contract forbids dequantization")
    layers = config.get("num_hidden_layers")
    if type(layers) is not int or not 1 <= layers <= 100000:
        raise GPTOSSContractError("actual config requires a positive integer num_hidden_layers")
    return layers


def validate_supported_cohort(cohort, config):
    """Return an exact supported descriptor or raise GPTOSSContractError.

    This checks actual config semantics and supported protocol/implementation
    versions. Raw config hash, checkpoint payload hashes and source checks are
    enforced separately by the caller before it loads a production stage.
    """
    descriptor = _mapping(cohort, "cohort")
    if set(descriptor) != _FIELDS:
        raise GPTOSSContractError("exact GPT-OSS model cohort descriptor required")
    for field in _FIELDS - {"n_layers"}:
        value = descriptor[field]
        if not isinstance(value, str) or not value.strip() or len(value) > 512:
            raise GPTOSSContractError(f"invalid cohort {field}")
        if field.endswith("sha256") and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise GPTOSSContractError(f"cohort {field} must be a lowercase SHA256")
    for field, expected in _SUPPORTED.items():
        if descriptor[field] != expected:
            raise GPTOSSContractError(f"unsupported GPT-OSS {field}; supported value is {expected}")
    layers = _config_layers(config)
    if type(descriptor["n_layers"]) is not int or descriptor["n_layers"] != layers:
        raise GPTOSSContractError("cohort n_layers differs from actual model config")
    return dict(descriptor)


def build_cohort_from_inventory(inventory):
    """Build ModelCohort from actual complete local download verification.

    Public recipe::

        checked = verify_inventory(model_dir, verify_files=True)
        cohort = build_cohort_from_inventory(checked)

    JSON manifests/declared PASS and config/index/header-only inventories are
    not verification evidence. The cohort's checkpoint identity covers the
    complete immutable file list, sizes and digests, not just storage metadata.
    """
    from .download_inventory import InventoryError, verified_config
    try:
        config = verified_config(inventory)
    except InventoryError as exc:
        raise GPTOSSContractError(str(exc)) from exc
    descriptor = {"model_id": inventory["repo"], "manifest_sha256": inventory["manifest_sha256"],
                  "checkpoint_id": inventory["checkpoint_id"], "config_sha256": inventory["config_sha256"],
                  **_SUPPORTED, "n_layers": _config_layers(config)}
    validate_supported_cohort(descriptor, config)
    # ModelCohort's identity/signing module imports cryptography. Defer that
    # dependency until constructing a cohort; compatibility checks stay stdlib.
    from .offers import ModelCohort
    return ModelCohort.from_dict(descriptor)

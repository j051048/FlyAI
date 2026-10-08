from copy import deepcopy
import pytest

from shard.receipt import ReceiptError, ReceiptSigner, gen_key, verify_receipt
from shard.runtime_observation import SCHEMA, digest, validate_runtime_observation


def observation():
    files = {"engines/deepseek_v4/v4_pipe.py": "a" * 64}
    environment = {"V4_HADAMARD": "torch", "V4_MAX_SEQ": "8192"}
    flags = {"V4_HADAMARD": {"requested": "torch", "parsed": "torch", "observed": "torch",
                            "verdict": "OK", "reason": ""}}
    return {"schema": SCHEMA, "node_id": "node-1", "gpu_uuid": None, "process_run_id": "run-1",
        "runtime_config_sha256": "b" * 64, "source_files": files, "source_sha256": digest(files),
        "environment": environment, "environment_sha256": digest(environment),
        "effective_flags": flags, "effective_flags_sha256": digest(flags),
        "kernel_backend": "cpu", "hadamard_backend": "torch", "graph_mode": "off", "wire_mode": "bf16",
        "transport": "external route declared by deployment", "versions": {"python": "3.11", "torch": "test", "cuda": None, "tilelang": None},
        "phase": "job_complete"}


def test_gpu_uuid_byte_and_text_views_keep_the_same_identity():
    import uuid
    from shard.runtime_observation import gpu_uuid_text
    value = uuid.UUID("12345678-1234-1234-1234-123456789abc")
    assert gpu_uuid_text(value.bytes) == gpu_uuid_text(str(value)) == "GPU-" + str(value)
    assert gpu_uuid_text(None) is None


def test_observation_is_detached_and_covered_by_receipt_signature():
    value = observation()
    signer = ReceiptSigner(gen_key(), "swarm", "job", 0, 1, nonce="nonce")
    signer.observe(b"input", b"output")
    receipt = signer.finalize(runtime_observation=value)
    verify_receipt(receipt)
    value["environment"]["V4_HADAMARD"] = "extension"
    assert receipt["runtime_observation"]["environment"]["V4_HADAMARD"] == "torch"
    tampered = deepcopy(receipt)
    tampered["runtime_observation"]["kernel_backend"] = "tilelang"
    with pytest.raises(ReceiptError, match="signature"):
        verify_receipt(tampered)


@pytest.mark.parametrize("mutation", [
    lambda x: x["source_files"].update({"../secret.py": "c" * 64}),
    lambda x: x["environment"].update({"V4_PRIVATE_KEY_FILE": "/private/key"}),
    lambda x: x["environment"].update({"V4_MAX_SEQ": "4096"}),
    lambda x: x["effective_flags"]["V4_HADAMARD"].update(verdict="MISMATCH"),
    lambda x: x.update(source_sha256="c" * 64),
    lambda x: x.update(phase="ready"),
])
def test_invalid_or_tampered_observation_is_rejected(mutation):
    value = observation()
    mutation(value)
    with pytest.raises(ValueError):
        validate_runtime_observation(value)


def test_old_receipts_stay_valid_without_optional_observation():
    signer = ReceiptSigner(gen_key(), "s", "j", 0, 1)
    signer.observe(b"i", b"o")
    receipt = signer.finalize()
    assert "runtime_observation" not in receipt
    verify_receipt(receipt)


def test_loaded_source_is_pinned_and_runtime_audit_does_not_expose_unknown_values(monkeypatch):
    from pathlib import Path
    import sys
    from types import SimpleNamespace
    directory = Path(__file__).resolve().parents[1] / "engines" / "deepseek_v4"
    monkeypatch.syspath_prepend(str(directory))
    import v4_observability
    import v4_resources
    import v4_runtime_init
    import v4_kernels_cpu
    source = {"engines/deepseek_v4/v4_stage.py": "a" * 64}
    monkeypatch.setattr(v4_observability, "source_inventory", lambda: dict(source))
    monkeypatch.setattr(v4_resources, "runtime_config_identity", lambda _: "b" * 64)
    monkeypatch.setattr(v4_runtime_init, "hadamard_identity", lambda: {"requested": "auto", "backend": "torch", "reason": "test",
        "source_sha256": "d" * 64, "module_files": [], "dependency_version": None})
    monkeypatch.setattr(v4_kernels_cpu, "backend", lambda: "cpu")
    monkeypatch.setenv("V4_UNKNOWN_SECRET", "must-never-appear")
    stage = SimpleNamespace(device="cpu", lo=0, hi=1, _block_graphs=None)
    loaded = v4_observability.runtime_observation(stage)
    final = v4_observability.runtime_observation(stage, phase="job_complete")
    assert loaded["process_run_id"] == final["process_run_id"]
    assert "must-never-appear" not in __import__("json").dumps(final)
    source["engines/deepseek_v4/v4_stage.py"] = "c" * 64
    with pytest.raises(RuntimeError, match="changed since"):
        v4_observability.runtime_observation(stage, phase="job_complete")

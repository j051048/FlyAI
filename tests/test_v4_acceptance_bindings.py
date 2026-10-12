"""Offline genuine hash/signature fixtures with synthetic devices and weights."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tools import v4_acceptance_bindings as bindings
from tools import v4_acceptance_numeric as numeric
from tools import v4_acceptance_ring as ring
from phase0 import v4_benchmark as bench
from shard.receipt import ReceiptSigner, gen_key, pub_b64
from test_v4_benchmark import observed_protocol, reseal_observed_protocol, signed_observation
from test_v4_acceptance_numeric import frozen, metadata_checkpoint


@pytest.fixture
def evidence(tmp_path):
    from shard import weight_artifacts as artifacts
    from shard.pipeline_plan import build_plan
    from shard.offers import model_cohort_id
    template, _ = metadata_checkpoint(tmp_path)
    checkpoint = tmp_path / "native-checkpoint"
    checkpoint.mkdir()
    for name in ("config.json", "model0-mp1.safetensors"):
        shutil.copyfile(template / name, checkpoint / name)
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    catalog, pack = artifacts.catalogue_directory(checkpoint, bench.MODEL_ID)
    artifacts.write_metadata(checkpoint, catalog, pack)
    inventory = bench._checkpoint_identity(bench.checkpoint_inventory(checkpoint))
    p, keys = observed_protocol()
    p["checkpoint"], p["source"] = inventory, ring._source_identity()
    p["env"] = ring.canonical_environment({"V4_EXECUTION_PROFILE": "resident-whole",
        "V4_MAX_SEQ": "8192", "V4_PREFILL_QUERY_CHUNK": "128", "V4_RUNTIME_METRICS": "0"})
    p["isolation"] = "none"
    for i, node in enumerate(p["hardware"]):
        node.update(host_id="synthetic-one-host", gpu_uuid=f"GPU-{i+1:08x}-0000-0000-0000-000000000000",
            checkpoint_sha256=inventory["sha256"], config_sha256=inventory["config_sha256"])
    p = reseal_observed_protocol(p)
    assert not bench.protocol_errors(p)
    cfg = {"ring_id": "binding-test", "environment": p["env"], "stages": [
        {"node_id": n["node_id"], "lo": n["layer_start"], "hi": n["layer_end"], "gpu_index": i, "port": 29610+i}
        for i, n in enumerate(p["hardware"])]}
    cohort = artifacts.catalog_model_cohort(catalog)
    plan = build_plan({"n_layers": 43}, ring_id=cfg["ring_id"], cohort_id=model_cohort_id(cohort), model_cohort=cohort,
        endpoints=[f"127.0.0.1:{n['port']}" for n in cfg["stages"]],
        split=[n["hi"]-n["lo"] for n in cfg["stages"]], node_ids=[n["node_id"] for n in cfg["stages"]],
        gpu_uuids=[n["gpu_uuid"] for n in p["hardware"]])
    for stage, node in zip(plan["stages"], p["hardware"]):
        stage.update(signer_pubkey=node["signer_pubkey"], context_limit=8192)
    plan["coordinator"].update(node_id=cfg["ring_id"]+"-coordinator", signer_pubkey=pub_b64(gen_key()))
    plan["execution"] = {"max_context": 8192}
    ids = frozen(tmp_path / "original.safetensors", [11] * 4104)
    np = numeric.prepare(numeric.parser().parse_args(["prepare", "--checkpoint", str(checkpoint),
        "--input", str(ids), "--device", "cuda:0", "--out-dir", str(tmp_path / "numeric")]))
    campaign = {"schema": "flyai-v4-5090-campaign-plan/1", "config": {"ring": cfg}, "source": p["source"],
        "numeric_max_tokens": 8192, "numeric_input_tokens": 4104, "kit": [
            {"name": name, "sha256": np["source_files"][name]} for name in
            ("tools/v4_acceptance_numeric.py", "phase0/v4_numeric_diagnose.py")]}
    campaign["sha256"] = bench.digest(campaign)
    loaded = {"schema": "flyai-v4-acceptance-ring-loaded/1", "cold_open": True, "request_count": 0,
              "ready": False, "ring_id": cfg["ring_id"], "observations": []}
    for node in p["hardware"]:
        observation = signed_observation(p, node)
        observation["phase"] = "loaded"
        loaded["observations"].append(observation)
    return {"protocol": p, "keys": keys, "plan": plan, "config": cfg, "numeric": np,
            "campaign": campaign, "checkpoint": inventory, "input": ids, "loaded": loaded}


def numeric_errors(value):
    return bindings.numeric_binding_errors(value["numeric"], value["campaign"], value["checkpoint"], value["input"])


def test_numeric_binding_matches_actual_derived_head_input_and_frozen_sources(evidence):
    assert [evidence["numeric"]["lo"], evidence["numeric"]["hi"]] == [0, 4]
    assert numeric_errors(evidence) == []


@pytest.mark.parametrize("change", ["checkpoint", "manifest", "config", "fixture", "device", "prefix", "roles",
    "capacity", "arguments", "input", "recipe", "source", "missing_source", "extra_source"])
def test_rehashed_numeric_plan_cannot_borrow_another_run_or_relax_campaign(evidence, change):
    plan = evidence["numeric"]
    if change in ("checkpoint", "manifest", "config"):
        key = {"checkpoint": "checkpoint_id", "manifest": "manifest_sha256", "config": "config_sha256"}[change]
        plan["artifact_preview"][key] = "tensor-sha256:"+"0"*64 if change == "checkpoint" else "0"*64
    elif change == "fixture": plan["fixture"] = True
    elif change == "device": plan["device"] = "cpu"
    elif change == "prefix": plan["hi"] = 3
    elif change == "roles": plan["roles"]["dspark"] = True
    elif change == "capacity": plan["limits"]["tokens"] = 4096
    elif change == "arguments":
        plan["runtime_model_args"]["index_topk"] = 256
        plan["runtime_model_args_sha256"] = numeric.digest(plan["runtime_model_args"])
    elif change == "input": plan["input"]["sha256"] = "0"*64
    elif change == "recipe":
        plan["environment"]["V4_MOE_IN_GRAPH"] = "0"
        plan["environment_sha256"] = numeric.digest(plan["environment"])
    elif change == "source": plan["source_files"]["engines/deepseek_v4/v4_stage.py"] = "0"*64
    elif change == "missing_source": del plan["source_files"]["phase0/v4_numeric_diagnose.py"]
    else: plan["source_files"]["tools/unfrozen.py"] = "0"*64
    plan["source_sha256"] = numeric.digest(plan["source_files"])
    plan["sha256"] = numeric.digest({k: v for k, v in plan.items() if k != "sha256"})
    assert numeric_errors(evidence)


def test_numeric_binding_rehashes_original_bytes_and_supports_relocated_input(evidence, tmp_path):
    portable = tmp_path / "relocated.safetensors"
    portable.write_bytes(evidence["input"].read_bytes())
    evidence["input"] = portable
    assert numeric_errors(evidence) == []
    raw = bytearray(portable.read_bytes()); raw[-8:] = (12).to_bytes(8, "little")
    portable.write_bytes(raw)
    assert numeric_errors(evidence)


def test_deployment_binding_requires_same_strict_assignment_and_loopback_routes(evidence):
    assert bindings.deployment_binding_errors(evidence["plan"], evidence["protocol"], evidence["config"]) == []


@pytest.mark.parametrize("change", ["ring", "gpu", "signer", "node", "endpoint", "next", "return", "capacity", "cohort", "coordinator"])
def test_valid_separate_deployment_cannot_qualify_frozen_campaign(evidence, change):
    plan = evidence["plan"]
    if change == "ring": plan["ring_id"] = "another-ring"
    elif change == "gpu": plan["stages"][0]["gpu_uuid"] = "GPU-another-device"
    elif change == "signer": plan["stages"][0]["signer_pubkey"] = pub_b64(gen_key())
    elif change == "node": plan["stages"][0]["node_id"] = "another-node"
    elif change == "endpoint": plan["stages"][0]["endpoint"] = "127.0.0.1:30710"
    elif change == "next": plan["stages"][0]["next_endpoint"] = "127.0.0.1:30711"
    elif change == "return": plan["coordinator"]["tail"] = "127.0.0.1:30713"
    elif change == "capacity": plan["execution"]["max_context"] = 4096
    elif change == "cohort": plan["model_cohort"]["config_sha256"] = "0"*64
    else: plan["coordinator"]["signer_pubkey"] = plan["stages"][0]["signer_pubkey"]
    assert bindings.deployment_binding_errors(plan, evidence["protocol"], evidence["config"])


def test_cold_binding_checks_new_pinned_processes_without_mutating_prior_ids(evidence):
    prior = {"unrelated-old-process"}
    original = deepcopy(evidence["loaded"])
    assert bindings.cold_binding_errors(evidence["loaded"], evidence["protocol"], prior) == []
    assert prior == {"unrelated-old-process"} and evidence["loaded"] == original
    prior.add(evidence["protocol"]["hardware"][0]["process_run_id"])
    assert bindings.cold_binding_errors(evidence["loaded"], evidence["protocol"], prior)


@pytest.mark.parametrize("change", ["work", "boolean_count", "warm", "missing", "process", "gpu", "runtime", "environment", "phase"])
def test_unsigned_cold_record_cannot_claim_another_or_already_warm_worker(evidence, change):
    loaded = evidence["loaded"]
    if change == "work": loaded["request_count"] = 1
    elif change == "boolean_count": loaded["request_count"] = False
    elif change == "warm": loaded["ready"] = True
    elif change == "missing": loaded["observations"].pop()
    elif change == "process": loaded["observations"][0]["process_run_id"] = "another-process"
    elif change == "gpu": loaded["observations"][0]["gpu_uuid"] = "GPU-another-device"
    elif change == "runtime": loaded["observations"][0]["runtime_config_sha256"] = "0"*64
    elif change == "environment": loaded["observations"][0]["environment"]["V4_PREFILL_QUERY_CHUNK"] = "0"
    else: loaded["observations"][0]["phase"] = "job_complete"
    assert bindings.cold_binding_errors(loaded, evidence["protocol"], set())


def warmup(evidence, *, observation_mutator=None):
    p, plan = evidence["protocol"], evidence["plan"]
    nonce, job = "a"*64, "actual-ed25519-synthetic-warmup"
    receipts = []
    for i, (node, key) in enumerate(zip(p["hardware"], evidence["keys"])):
        signer = ReceiptSigner(key, plan["ring_id"], job, node["layer_start"], node["layer_end"], nonce)
        signer.observe(f"boundary-{i}".encode(), f"boundary-{i+1}".encode())
        observation = signed_observation(p, node)
        observation["optimizations"] = {"prefill_expert_pipeline": {}, "conversation_cache": {},
            "execution_profile": {"name": "resident-whole", "graph_mode": "whole", "coverage": {
            "schema": "flyai-v4-whole-coverage/1", "captured_layers": node["layer_end"]-node["layer_start"],
            "refused_layers": 0, "undecided_layers": 0, "layers": [
                {"layer": layer, "captured_graphs": 1, "eager": False, "moe_in_graph": True}
                for layer in range(node["layer_start"], node["layer_end"])]}}}
        if observation_mutator:
            observation_mutator(observation)
        receipts.append(signer.finalize(runtime_observation=observation))
    return {"verified": True, "nonce": nonce, "job_id": job, "receipts": receipts}


def test_warmup_binding_reverifies_actual_ed25519_and_capture_coverage(evidence):
    proof = warmup(evidence)
    assert bindings.warmup_binding_errors(proof, evidence["protocol"], evidence["plan"]) == []


@pytest.mark.parametrize("change", ["job", "nonce", "signature", "missing", "runtime", "capture"])
def test_saved_warmup_pass_cannot_override_raw_signature_runtime_or_capture(evidence, change):
    def mutate(observation):
        if change == "runtime": observation["process_run_id"] = "another-process"
        if change == "capture": observation["optimizations"]["execution_profile"]["coverage"]["layers"][0]["eager"] = True
    proof = warmup(evidence, observation_mutator=mutate)
    if change == "job": proof["job_id"] = "another-job"
    elif change == "nonce": proof["nonce"] = "b"*64
    elif change == "signature": proof["receipts"][0]["job_id"] = "unsigned-change"
    elif change == "missing": proof["receipts"].pop()
    assert bindings.warmup_binding_errors(proof, evidence["protocol"], evidence["plan"])


@pytest.mark.parametrize("function,arguments", [
    (bindings.numeric_binding_errors, ({}, {}, {}, "missing.safetensors")),
    (bindings.deployment_binding_errors, ({}, {}, {})),
    (bindings.cold_binding_errors, ({}, {}, set())),
    (bindings.warmup_binding_errors, ({}, {}, {})),
])
def test_missing_binding_evidence_returns_errors_without_raising(function, arguments):
    assert function(*arguments)


def test_binding_module_import_is_stdlib_only_and_does_not_load_models():
    root = Path(__file__).resolve().parents[1]
    code = "import sys; from tools import v4_acceptance_bindings; assert 'torch' not in sys.modules; assert 'v4_stage' not in sys.modules"
    result = subprocess.run([sys.executable, "-S", "-c", code], cwd=root, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr

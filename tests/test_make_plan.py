"""Actual file identities and declared routes/keys; no GPU or model execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from make_plan import generate_plan, main, read_json, write_plan
from shard.download_inventory import build_inventory, verify_inventory
from shard.gpt_oss_contract import build_cohort_from_inventory
from shard.pipeline_plan import load_plan
from shard.receipt import gen_key, pub_b64

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(tmp_path):
    directory = tmp_path / "model with spaces"; directory.mkdir()
    config = dict(model_type="gpt_oss", num_hidden_layers=36, hidden_size=3072, torch_dtype="bfloat16",
                  quantization_config={"quant_method": "mxfp4", "dequantize": False})
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    # Synthetic bytes for the inventory/plan contract, never a native model PASS.
    (directory / "model.safetensors").write_bytes(b"synthetic-weight-identity-fixture")
    files = []
    for path in sorted(directory.iterdir()):
        payload = path.read_bytes(); digest = hashlib.sha256(payload).hexdigest()
        files.append(dict(path=path.name, size=len(payload), algorithm="sha256", digest=digest, payload_sha256=digest))
    record = build_inventory("openai/gpt-oss-plan-fixture", "a"*40, files)
    (directory / ".shard-download.json").write_text(json.dumps(record), encoding="utf-8")
    cohort = build_cohort_from_inventory(verify_inventory(directory, verify_files=True)).to_dict()
    keys = [pub_b64(gen_key()) for _ in range(4)]
    ids = ["N1-head", "N0-hub", "N2-tail"]
    deployment = {"nodes": {name: dict(ssh_target=name, workspace="/root/FlyAI", model="/root/models/oss",
        node_key=f"/root/private/{name}.key", max_context=2048, listen_port=29501,
        environment={"HF_TOKEN": "private-test-environment"}) for name in ids},
        "stages": [dict(node_id=name, gpu_uuid=f"GPU-{i+1:08x}-1111-2222-3333-444444444444",
            signer_pubkey=keys[i], endpoint=f"198.51.100.1:{40001+i}",
            next_endpoint=f"127.0.0.1:{29502+i}" if i < 2 else None) for i, name in enumerate(ids)],
        "coordinator": dict(node_id="N0-hub", signer_pubkey=keys[1], head="127.0.0.1:39601", tail="127.0.0.1:39603"),
        "tunnels": []}
    return directory, cohort, deployment


def generate(inputs, **kwargs):
    directory, cohort, deployment = inputs
    return generate_plan(directory, cohort, deployment, ring_id="real-shape-fixture", split="12,12,12", **kwargs)


def test_actual_model_and_public_identities_make_a_complete_directly_deployable_plan(inputs, tmp_path):
    directory, cohort, config = inputs
    plan = generate(inputs)
    assert plan["model_cohort"] == cohort and plan["n_layers"] == 36
    assert [(s["lo"], s["hi"]) for s in plan["stages"]] == [(0,12),(12,24),(24,36)]
    assert plan["coordinator"]["node_id"] == "N0-hub"  # middle host, not head.
    assert plan["coordinator"]["signer_pubkey"] == plan["stages"][1]["signer_pubkey"]
    assert plan["stages"][0]["next_endpoint"] == "127.0.0.1:29502"
    assert plan["stages"][0]["endpoint"] == "198.51.100.1:40001"
    assert plan["execution"]["max_context"] == 2048
    public = json.dumps(plan)
    assert "private-test-environment" not in public and "/root/private/" not in public
    assert "runtime_config_sha256" not in public and "calibrations" not in public
    assert config["stages"][0].get("context_limit") is None  # Input unchanged.
    path = tmp_path / "ring.json"; assert write_plan(path, plan)
    assert load_plan(path, json.loads((directory / "config.json").read_text())) == plan
    from deploy_oss import RingDeployment
    deploy = RingDeployment(plan, config["nodes"], state_dir=tmp_path / "processes")
    command, _ = deploy.stage_command(plan["stages"][0], config["nodes"]["N1-head"])
    assert command[command.index("--listen-port")+1] == "29501"  # Not mapped advertised port 40001.
    assert deploy.coordinator_id == "N0-hub"


def test_independent_cpu_coordinator_has_no_gpu_uuid_requirement(inputs):
    _, _, config = inputs
    config["nodes"]["driver"] = dict(ssh_target="driver", workspace="/root/FlyAI", model="/models/oss",
        node_key="/keys/node.key", coordinator_key="/keys/controller.key", max_context=1024)
    config["coordinator"].update(node_id="driver", signer_pubkey=pub_b64(gen_key()))
    result = generate(inputs)
    assert result["coordinator"]["node_id"] == "driver" and result["execution"]["max_context"] == 1024


def test_software_context_is_the_minimum_of_all_declared_and_configured_limits(inputs, tmp_path):
    _, _, config = inputs
    config["stages"][0]["context_limit"] = 1024
    config["nodes"]["N2-tail"]["max_context"] = 512
    plan = generate(inputs)
    assert [row["context_limit"] for row in plan["stages"]] == [1024,2048,512]
    assert plan["execution"] == {"max_context":512}
    from deploy_oss import RingDeployment
    deploy = RingDeployment(plan, config["nodes"], state_dir=tmp_path / "state")
    command, _ = deploy.stage_command(plan["stages"][0], config["nodes"]["N1-head"])
    assert command[command.index("--max-ctx")+1] == "512"


def test_existing_deploy_default_is_an_explicit_software_limit_not_a_gpu_budget(inputs):
    _, _, config = inputs
    for row in config["nodes"].values():row.pop("max_context")
    result = generate(inputs)
    assert result["execution"]["max_context"] == 8192
    assert "resource_capacity" not in result


@pytest.mark.parametrize("field,value", [("gpu_uuid", None),("gpu_uuid","GPU-0"),("signer_pubkey",None),
    ("signer_pubkey","not-base64"),("next_endpoint",None),("context_limit",True),("context_limit",0)])
def test_incomplete_or_false_stage_metadata_is_refused(inputs, field, value):
    inputs[2]["stages"][0][field] = value
    with pytest.raises(ValueError):generate(inputs)


@pytest.mark.parametrize("field", ["gpu_uuid","signer_pubkey","node_id"])
def test_duplicate_devices_signers_or_node_ids_are_not_independent_capacity(inputs, field):
    inputs[2]["stages"][1][field] = inputs[2]["stages"][0][field]
    with pytest.raises(ValueError, match="distinct"):generate(inputs)


def test_coordinator_cannot_point_to_an_unconfigured_machine(inputs):
    inputs[2]["coordinator"]["node_id"] = "missing"
    with pytest.raises(ValueError, match="deployment entry"):generate(inputs)


def test_tail_cannot_forward_or_accept_an_unverified_runtime_hash(inputs):
    inputs[2]["stages"][-1]["next_endpoint"] = "localhost:29501"
    with pytest.raises(ValueError, match="tail"):generate(inputs)
    inputs[2]["stages"][-1]["next_endpoint"] = None
    inputs[2]["stages"][0]["runtime_config_sha256"] = "a"*64
    with pytest.raises(ValueError, match="hashes"):generate(inputs)


@pytest.mark.parametrize("field,value", [("n_layers",35),("config_sha256","b"*64),
    ("checkpoint_id","sha256:"+"c"*64),("runtime_abi","other-runtime/1"),("manifest_sha256","d"*64)])
def test_supplied_cohort_cannot_override_real_model_identity_or_contract(inputs, field, value):
    inputs[1][field] = value
    with pytest.raises(ValueError):generate(inputs)


def test_metadata_plan_does_not_claim_payload_rehash_and_optional_full_hash_rejects_corruption(inputs):
    path = inputs[0] / "model.safetensors"
    payload = bytearray(path.read_bytes()); payload[-1] ^= 1; path.write_bytes(payload)
    assert generate(inputs)["n_layers"] == 36  # Marker/config binding only, explicitly documented.
    with pytest.raises(ValueError, match="payload"):generate(inputs, verify_files=True)


def test_split_must_cover_the_actual_model_and_cannot_conflict_with_config(inputs):
    directory, cohort, config = inputs
    with pytest.raises(ValueError):generate_plan(directory,cohort,config,ring_id="r",split="12,12,11")
    config["split"] = [20,8,8]
    with pytest.raises(ValueError,match="differs"):generate(inputs)


def test_cli_no_repo_cwd_and_nodes_alias_produce_the_same_plan(inputs, tmp_path):
    directory, cohort, config = inputs
    cp, np, out = tmp_path/"cohort.json", tmp_path/"nodes.json", tmp_path/"ring.json"
    cp.write_text(json.dumps(cohort)); np.write_text(json.dumps(config))
    command = [sys.executable,str(ROOT/"phase0/make_plan.py"),"--model",str(directory),"--cohort",str(cp),
        "--nodes",str(np),"--ring-id","cli-ring","--split","12,12,12","--out",str(out)]
    result = subprocess.run(command,cwd=tmp_path,capture_output=True,text=True,timeout=15)
    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    assert status["weight_payload_rehashed"] is False and status["nstages"] == 3
    assert "readiness claim" in status["scope"]
    old = out.read_bytes()
    result = subprocess.run(command,cwd=tmp_path,capture_output=True,text=True,timeout=15)
    assert result.returncode == 0 and json.loads(result.stdout)["written"] is False and out.read_bytes() == old


def test_output_cannot_clobber_another_plan(inputs,tmp_path):
    plan = generate(inputs); target = tmp_path/"ring.json"; write_plan(target,plan)
    different = deepcopy(plan); different["ring_id"] = "different"
    with pytest.raises(ValueError,match="another plan"):write_plan(target,different)
    assert load_plan(target) == plan


def test_json_duplicate_fields_and_nondict_cohort_are_refused(inputs,tmp_path):
    path=tmp_path/"bad.json"; path.write_text('{"nodes":{},"nodes":{}}')
    with pytest.raises(ValueError,match="duplicate"):read_json(path)
    with pytest.raises(ValueError,match="descriptor"):generate_plan(inputs[0],[],inputs[2],ring_id="r")


def test_generation_import_and_help_do_not_require_torch_or_transformers(tmp_path):
    script = str(ROOT/"phase0/make_plan.py")
    code = "import runpy,sys; runpy.run_path(sys.argv[1]); assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    result = subprocess.run([sys.executable,"-c",code,script],cwd=tmp_path,capture_output=True,text=True,timeout=15)
    assert result.returncode == 0, result.stderr

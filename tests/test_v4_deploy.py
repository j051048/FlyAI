"""Strict V4 command/route construction and real receipt verification; no SSH/GPU."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from phase0.v4_deploy import V4Deployment, generate_tunnels, tunnel_argv, check_tunnels
from shard.offers import ModelCohort
from shard.manifest import pub_b64
from shard.pipeline_plan import build_plan
from shard.receipt import ReceiptSigner, ReceiptError


@pytest.fixture
def fixture():
    keys = [Ed25519PrivateKey.generate() for _ in range(3)]
    cohort = ModelCohort("deepseek-ai/DeepSeek-V4-Flash", "a" * 64, "tensor-sha256:" + "b" * 64,
        "c" * 64, "fp4-fp8", "deepseek-v4-native/1", "shard-pipeline-session/1", "greedy-native-fp4-fp8/1", 43)
    nodes = {name: {"ssh_target": "node-" + name.lower(), "ssh_port": 2222,
                   "workspace": "/srv/Fly AI's checkout", "model": "/srv/V4 weights", "node_key": "/srv/node.key",
                   "listen_port": 29610, "max_context": 8192} for name in ("A", "B", "F")}
    plan = build_plan({"n_layers": 43}, ring_id="v4-test", cohort_id=cohort.cohort_id,
        endpoints=["127.0.0.1:29610"] * 3, split="20,20,3", node_ids=["A", "B", "F"],
        gpu_uuids=[f"GPU-{index+1:08x}-0000-0000-0000-000000000000" for index in range(3)],
        head="127.0.0.1:29610", tail="127.0.0.1:29612", model_cohort=cohort.to_dict())
    plan["stages"][0]["next_endpoint"] = "127.0.0.1:30001"
    plan["stages"][1]["next_endpoint"] = "127.0.0.1:30002"
    for row, key in zip(plan["stages"], keys): row["signer_pubkey"] = pub_b64(key)
    plan["coordinator"].update(node_id="A", signer_pubkey=pub_b64(keys[0]))
    return plan, {"nodes": nodes}, keys


def test_generated_star_routes_live_at_actual_callers_and_keep_tail_identity(fixture):
    plan, config, _ = fixture
    routes = generate_tunnels(plan, config["nodes"], hub="A")
    assert [(r["caller"], r["target"], r["bind_port"], r["remote_port"]) for r in routes] == [
        ("A", "B", 30001, 29610), ("B", "F", 30002, 29610), ("A", "F", 29612, 29610)]
    assert routes[1]["via_hub"] == "A" and "via_hub" not in routes[0]
    argv = tunnel_argv(routes[1], config["nodes"])
    assert argv[argv.index("-J") + 1] == "node-a:2222"
    assert argv[argv.index("-L") + 1] == "127.0.0.1:30002:127.0.0.1:29610"
    assert argv[-1] == "node-f" and "-N" in argv
    assert check_tunnels(plan, config["nodes"], routes)["checked"]
    # An opaque TCP tunnel is not the prohibited multi-GPU application relay.
    assert "ret_relay" not in json.dumps(routes)


@pytest.mark.parametrize("change", ["wrong_target", "wrong_remote", "duplicate", "listener_collision"])
def test_bad_forwarding_maps_fail_before_any_remote_call(fixture, change):
    plan, config, _ = fixture
    routes = generate_tunnels(plan, config["nodes"], hub="A")
    if change == "wrong_target": routes[1]["target"] = "A"
    if change == "wrong_remote": routes[1]["remote_port"] = 22222
    if change == "duplicate": routes.append(dict(routes[0]))
    if change == "listener_collision": routes[0]["bind_port"] = 29610
    with pytest.raises(ValueError): check_tunnels(plan, config["nodes"], routes)


def test_commands_preserve_container_listener_and_explicit_plan_and_no_legacy(fixture, tmp_path):
    plan, config, _ = fixture
    plan["stages"][0]["endpoint"] = "127.0.0.1:40001"
    deploy = V4Deployment(plan, config, state_dir=tmp_path)
    argv, path = deploy.stage_command(plan["stages"][0], config["nodes"]["A"])
    assert argv[argv.index("--port") + 1] == "29610"
    assert argv[argv.index("--next") + 1] == "127.0.0.1:30001"
    assert "--legacy-protocol" not in argv and "--receipts" in argv
    assert path.startswith("/srv/Fly AI's checkout/")
    tail, _ = deploy.stage_command(plan["stages"][-1], config["nodes"]["F"])
    assert "--dspark" in tail and "--next" not in tail


class FakeSSH:
    def __init__(self, plan, keys, *, missing_receipts=False, start_failure=None):
        self.plan, self.keys = plan, keys
        self.missing_receipts, self.start_failure = missing_receipts, start_failure
        self.calls, self.starts, self.stops = [], [], []
        self.job = None
    def __call__(self, argv, **kwargs):
        code = shlex.split(argv[-1])[-1]
        tree = ast.parse(code)
        self.calls.append((argv, code, kwargs))
        if "check_node(**data)" in code:
            data = json.loads(kwargs["input"])
            result = {"preflight_ok": True, "node_id": "fixture", "stage": data["index"], "errors": []}
        elif ".start(" in code:
            call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == "start")
            name, command = ast.literal_eval(call.args[0]), ast.literal_eval(call.args[1])
            if self.start_failure and name.endswith(self.start_failure):
                return SimpleNamespace(returncode=1, stdout="", stderr="start failed")
            self.starts.append((name, command, argv[-2]))
            if ".warmup." in name:
                parsed = ast.parse(command[-1])
                dumps = next(n for n in ast.walk(parsed) if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Attribute) and n.func.attr == "dumps")
                self.job = ast.literal_eval(dumps.args[0])
            result = {"running": True, "reused": False, "pid": 123}
        elif "process_observation" in code:
            if ".warmup." in code:
                receipts = []
                for row, key in zip(self.plan["stages"], self.keys):
                    signer = ReceiptSigner(key, self.plan["ring_id"], self.job["jobId"], row["lo"], row["hi"], self.job["nonce"])
                    signer.observe(b"fixture", b"fixture"); receipts.append(signer.finalize())
                final = {"ok": True, "jobId": self.job["jobId"], "tokensGenerated": 2, "tokenIds": [7, 8],
                         "receipts": [] if self.missing_receipts else receipts}
                result = {"running": False, "log_tail": "SHARD_JOB_DONE " + json.dumps(final)}
            else:
                result = {"running": True, "listening": True, "log_tail": "[s0] listening 127.0.0.1:29610"}
        elif ".stop(" in code:
            self.stops.append(code); result = {"stopped": True}
        else:
            result = {"ok": True}
        return SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr="")


def test_tail_first_launch_uses_stdin_environment_and_real_signature_gate(fixture, tmp_path):
    plan, config, keys = fixture
    config["environment"] = {"V4_REFILL_FLOOR": "1", "PRIVATE_TEST_ENV": "private-fixture-value"}
    fake = FakeSSH(plan, keys)
    deploy = V4Deployment(plan, config, state_dir=tmp_path, executor=fake)
    result = deploy.deploy(routes=generate_tunnels(plan, config["nodes"], hub="A"))
    stages = [name for name, _, _ in fake.starts if ".stage" in name]
    assert stages == ["v4-test.stage2", "v4-test.stage1", "v4-test.stage0"]
    assert result["ready"] and result["resource_leases"] is False
    assert result["signed_warmup"]["verified"] and len(result["signed_warmup"]["receipts"]) == 3
    assert fake.job["ignoreEOS"] is True
    assert any("node-b" == host for name, _, host in fake.starts if ".tunnel" in name)
    assert all("private-fixture-value" not in " ".join(argv) for argv, _, _ in fake.calls)
    assert any("private-fixture-value" in kwargs.get("input", "") for _, _, kwargs in fake.calls)
    assert all("--legacy-protocol" not in command for _, command, _ in fake.starts)


def test_listen_without_signed_receipts_never_reports_ready_and_rolls_back_owned_only(fixture, tmp_path):
    plan, config, keys = fixture
    fake = FakeSSH(plan, keys, missing_receipts=True)
    deploy = V4Deployment(plan, config, state_dir=tmp_path, executor=fake)
    with pytest.raises(ReceiptError): deploy.deploy(routes=())
    assert len([line for line in fake.stops if ".stage" in line]) == 3
    assert deploy.failure_diagnostics["original_error_class"]
    assert all("kill -9" not in line and "fuser" not in line for line in fake.stops)


def test_failure_does_not_kill_preexisting_or_other_stage_processes(fixture, tmp_path):
    plan, config, keys = fixture
    fake = FakeSSH(plan, keys, start_failure="stage1")
    deploy = V4Deployment(plan, config, state_dir=tmp_path, executor=fake)
    with pytest.raises(RuntimeError): deploy.deploy()
    assert len([line for line in fake.stops if ".stage" in line]) == 1
    assert "stage2" in next(line for line in fake.stops if ".stage" in line)


def test_remote_first_cause_survives_without_secret_or_command_echo(fixture, tmp_path):
    plan, config, _ = fixture
    config["environment"] = {"PRIVATE_TOKEN": "fixture-sensitive-value"}
    def failure(argv, **kwargs):
        return SimpleNamespace(returncode=255, stdout="", stderr="Permission denied (publickey). fixture-sensitive-value " + argv[-1])
    report = V4Deployment(plan, config, state_dir=tmp_path, executor=failure).preflight()
    text = json.dumps(report)
    assert "Permission denied" in text and "fixture-sensitive-value" not in text
    assert "<remote helper>" in text and not report["preflight_ok"]


def test_independent_cpu_coordinator_needs_no_listener_or_gpu_fields(fixture, tmp_path):
    plan, config, keys = fixture
    plan["coordinator"]["node_id"] = "entry"
    config["nodes"]["entry"] = {"ssh_target": "entry-host", "workspace": "/entry/Full Checkout", "model": "/entry/assets",
        "node_key": "/entry/key", "coordinator_key": "/entry/controller", "max_context": 1024}
    fake = FakeSSH(plan, keys)
    deploy = V4Deployment(plan, config, state_dir=tmp_path, executor=fake)
    assert deploy.max_context == 1024
    assert deploy.deploy()["ready"]
    assert any(host == "entry-host" for name, _, host in fake.starts if ".warmup." in name)


@pytest.mark.parametrize("change", ["production", "gpu", "keys", "listener", "boolean"])
def test_unbound_or_invalid_manual_configuration_rejected_before_ssh(fixture, tmp_path, change):
    plan, config, _ = fixture
    if change == "production": config["mode"] = "production"
    if change == "gpu": plan["stages"][0]["gpu_uuid"] = "GPU-0"
    if change == "keys": plan["stages"][1]["signer_pubkey"] = plan["stages"][0]["signer_pubkey"]
    if change == "listener": config["nodes"]["A"]["listen_port"] = True
    if change == "boolean": config["nodes"]["F"]["dspark"] = "false"
    with pytest.raises(ValueError): V4Deployment(plan, config, state_dir=tmp_path, executor=lambda *_: pytest.fail("no remote call"))


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_invalid_timeout_cannot_enter_a_poll_loop(fixture, tmp_path, timeout):
    plan, config, _ = fixture
    deploy = V4Deployment(plan, config, state_dir=tmp_path)
    with pytest.raises(ValueError): deploy.deploy(readiness_timeout=timeout)

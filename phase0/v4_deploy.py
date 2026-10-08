"""Strict manual V4 deployment with caller-local SSH routes and owned processes.

This operator utility does not acquire production GPU leases. Production serving
uses the shared OpenNetworkService/NodeLeaseAgent path; LISTEN is only a startup
milestone. READY requires an authenticated coordinator pass and verified receipts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shlex
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase0.deploy_oss import RingDeployment, ssh_argv, python_command
from phase0.make_plan import read_json, write_plan
from shard.pipeline_plan import build_plan, validate_plan, load_plan
from shard.receipt import verify_coverage

UUID = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def _port(value):
    if type(value) is not int or not 1 <= value <= 65535:
        raise ValueError("an explicit listener/forward port in 1..65535 is required")
    return value


def generate_plan(metadata, cohort, config, *, ring_id, split=None):
    from engines.deepseek_v4.v4_preflight import check_artifacts
    check_artifacts(metadata, cohort)
    rows = config.get("stages")
    if not isinstance(rows, list) or not rows:
        raise ValueError("ordered stage identities and caller-local routes are required")
    nodes = config.get("nodes")
    if not isinstance(nodes, dict):
        raise ValueError("nodes must be a deployment mapping")
    coordinator = config.get("coordinator")
    if not isinstance(coordinator, dict) or set(coordinator) - {"node_id", "head", "tail", "signer_pubkey"}:
        raise ValueError("explicit coordinator identity and caller-local head/tail routes required")
    allowed = {"node_id", "gpu_uuid", "signer_pubkey", "endpoint", "next_endpoint", "context_limit"}
    if any(not isinstance(row, dict) or set(row) - allowed for row in rows):
        raise ValueError("stage declaration cannot invent runtime hashes or execution commands")
    model = json.loads((Path(metadata) / "config.json").read_text(encoding="utf-8-sig"))
    value = build_plan(model, ring_id=ring_id, cohort_id=__import__("shard.offers", fromlist=["model_cohort_id"]).model_cohort_id(cohort),
        endpoints=[row["endpoint"] for row in rows], split=split or config.get("split"),
        node_ids=[row["node_id"] for row in rows], gpu_uuids=[row["gpu_uuid"] for row in rows],
        head=coordinator["head"], tail=coordinator["tail"], model_cohort=cohort)
    for index, row in enumerate(rows):
        stage = value["stages"][index]
        stage["signer_pubkey"] = row["signer_pubkey"]
        if index < len(rows) - 1 and not row.get("next_endpoint"):
            raise ValueError("next_endpoint must be explicit in the sending node's namespace")
        stage["next_endpoint"] = row.get("next_endpoint")
        cap = nodes[row["node_id"]].get("max_context", 8192)
        if type(cap) is not int or cap < 1:
            raise ValueError("positive node software context ceiling required")
        declared = row.get("context_limit", cap)
        if type(declared) is not int or declared < 1:
            raise ValueError("positive stage software context ceiling required")
        stage["context_limit"] = min(cap, declared)
    value["coordinator"].update(coordinator)
    coord_id = coordinator.get("node_id", rows[0]["node_id"])
    if coord_id not in nodes:
        raise ValueError("coordinator node_id must exist in the deployment map")
    coord_cap = nodes[coord_id].get("max_context", 8192)
    if type(coord_cap) is not int or coord_cap < 1:
        raise ValueError("positive coordinator context ceiling required")
    value["execution"] = {"max_context": min(coord_cap, *(s["context_limit"] for s in value["stages"]))}
    validate_plan(value)
    _identities(value, nodes)
    return value


def _identities(plan, nodes):
    from engines.deepseek_v4.v4_artifact_contract import MODEL_IDS, RUNTIME_ABI, WIRE_VERSION, NUMERIC_CONTRACT, QUANTIZATION
    from shard.offers import ModelCohort
    cohort = ModelCohort.from_dict(plan.get("model_cohort"))
    if (cohort.model_id not in MODEL_IDS or (cohort.runtime_abi, cohort.wire_version, cohort.numeric_contract, cohort.quantization) !=
            (RUNTIME_ABI, WIRE_VERSION, NUMERIC_CONTRACT, QUANTIZATION) or cohort.n_layers != 43):
        raise ValueError("full supported native V4 Flash cohort required")
    keys = set()
    for stage in plan["stages"]:
        if not UUID.fullmatch(stage.get("gpu_uuid", "")):
            raise ValueError("actual full NVIDIA GPU UUID required")
        key = stage.get("signer_pubkey")
        if not key or key in keys:
            raise ValueError("distinct pinned stage public signing keys required")
        keys.add(key)
        _port(nodes[stage["node_id"]].get("listen_port"))
    if not plan["coordinator"].get("signer_pubkey"):
        raise ValueError("pinned coordinator public signing key required")


def _jump(node):
    value = node["ssh_target"]
    if not re.fullmatch(r"[A-Za-z0-9_.@:\[\]-]+", value) or value.startswith("-"):
        raise ValueError("safe SSH jump alias/user@host required")
    return value + (":" + str(_port(node["ssh_port"])) if node.get("ssh_port") else "")


def tunnel_argv(route, nodes):
    caller, target = route["caller"], route["target"]
    if caller not in nodes or target not in nodes:
        raise ValueError("tunnel caller and destination must be declared nodes")
    if route.get("bind_host", "127.0.0.1") not in ("127.0.0.1", "::1"):
        raise ValueError("deployment tunnels bind loopback only")
    host = route.get("remote_host", "127.0.0.1")
    if host not in ("127.0.0.1", "::1"):
        raise ValueError("tunnel target must reach the destination's local listener")
    bind = route.get("bind_host", "127.0.0.1")
    spec = f"{('['+bind+']') if ':' in bind else bind}:{_port(route['bind_port'])}:{('['+host+']') if ':' in host else host}:{_port(route['remote_port'])}"
    # target.ssh_key belongs to control->target, not necessarily caller->target.
    target_spec = {key: nodes[target][key] for key in ("ssh_target", "ssh_port") if key in nodes[target]}
    if route.get("caller_ssh_key"):
        target_spec["ssh_key"] = route["caller_ssh_key"]
    argv = ssh_argv(target_spec, forward=spec)
    if route.get("via_hub"):
        hub = route["via_hub"]
        if hub not in nodes:
            raise ValueError("tunnel jump must reference a declared hub")
        argv[-1:-1] = ["-J", _jump(nodes[hub])]
    return argv


def generate_tunnels(plan, nodes, *, hub=None):
    """Generate forwarding at actual callers, not at the control laptop."""
    plan = validate_plan(plan)
    if hub is not None and hub not in nodes:
        raise ValueError("declared SSH hub required")
    routes = []
    coord = plan["coordinator"].get("node_id", plan["stages"][0]["node_id"])
    legs = [(a["node_id"], b["node_id"], a["next_endpoint"], "forward")
            for a, b in zip(plan["stages"], plan["stages"][1:])]
    legs += [(coord, plan["stages"][0]["node_id"], plan["coordinator"]["head"], "drive"),
             (coord, plan["stages"][-1]["node_id"], plan["coordinator"]["tail"], "return")]
    seen = {}
    for caller, target, address, purpose in legs:
        dial = urlsplit("//" + address)
        if dial.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("SSH tunnel generation requires caller-local loopback endpoints")
        remote = _port(nodes[target].get("listen_port"))
        if caller == target and dial.port == remote:
            continue
        route = {"caller": caller, "target": target, "bind_host": "::1" if dial.hostname == "::1" else "127.0.0.1",
                 "bind_port": dial.port, "remote_host": "127.0.0.1", "remote_port": remote,
                 "purpose": purpose}
        if hub and caller != hub and target != hub:
            route["via_hub"] = hub
        key = caller, route["bind_host"], route["bind_port"]
        if key in seen:
            if seen[key]["target"] != target or seen[key]["remote_port"] != remote:
                raise ValueError("conflicting caller-local forwarding ports")
            continue
        seen[key] = route; routes.append(route)
        tunnel_argv(route, nodes)
    return routes


def check_tunnels(plan, nodes, routes):
    """Check exact data-leg mappings without dialing any engine or SSH host."""
    expected = generate_tunnels(plan, nodes)
    by_bind = {}
    for row in routes:
        tunnel_argv(row, nodes)
        key = row["caller"], row.get("bind_host", "127.0.0.1"), row["bind_port"]
        if key in by_bind:
            raise ValueError("one caller-local port cannot belong to two tunnel processes")
        if row["bind_port"] == nodes[row["caller"]].get("listen_port"):
            raise ValueError("SSH forward would collide with the caller's own engine listener")
        by_bind[key] = row
    for row in expected:
        actual = by_bind.get((row["caller"], row["bind_host"], row["bind_port"]))
        if actual is None or (actual["target"], actual["remote_port"]) != (row["target"], row["remote_port"]):
            raise ValueError("tunnels must cover every exact caller-local forward/drive/return leg")
    return {"checked": True, "routes": len(routes), "scope": "static exact-leg mappings; not authentication/reachability"}


class V4Deployment(RingDeployment):
    def __init__(self, plan, config, *, state_dir, executor=subprocess.run):
        if not isinstance(config, dict) or not isinstance(config.get("nodes"), dict):
            raise ValueError("deployment configuration needs a nodes map")
        if config.get("mode", "manual_strict_experiment") != "manual_strict_experiment":
            raise ValueError("production leased deployment must use OpenNetworkService/NodeLeaseAgent")
        self.config = deepcopy(config)
        super().__init__(plan, self.config["nodes"], state_dir=state_dir, executor=executor)
        _identities(self.plan, self.nodes)
        self.remote_tunnels = []
        self.failure_diagnostics = None
        self.scope = "manual_strict_experiment_without_resource_leases"
        # Freeze common defaults and overlay explicit choices before any import
        # in every spawned stage/coordinator. No Torch import on the controller.
        from phase0.v4_benchmark import DEFAULT_ENV
        self.common_environment = {**DEFAULT_ENV, **self.config.get("environment", {})}
        if any(not isinstance(k, str) or not isinstance(v, str) for k,v in self.common_environment.items()):
            raise ValueError("common environment must be a string mapping")
        for node in self.nodes.values():
            env = node.get("environment", {})
            if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k,v in env.items()):
                raise ValueError("node environment must be a string mapping")
            if type(node.get("dspark", True)) is not bool:
                raise ValueError("dspark must be an explicit boolean")
            if node.get("bind", "127.0.0.1") not in ("127.0.0.1", "::1"):
                raise ValueError("manual SSH deployment engines bind loopback only")
        seq = [int(self.common_environment["V4_MAX_SEQ"])]
        seq += [int(n.get("environment", {}).get("V4_MAX_SEQ", self.common_environment["V4_MAX_SEQ"])) for n in self.nodes.values()]
        self.max_context = min(self.max_context, *seq)
        if self.max_context < 1:
            raise ValueError("positive common context ceiling required")

    def environment(self, node, *, coordinator=False):
        env = {**self.common_environment, **node.get("environment", {}),
               "V4_MAX_SEQ": str(self.max_context), "SHARD_RECEIPTS": "1"}
        env["SHARD_NODE_KEY"] = node["node_key"]
        if coordinator:
            env["SHARD_COORDINATOR_KEY"] = node.get("coordinator_key", node["node_key"])
        return env

    def _sanitize(self, message):
        value = str(message)
        dictionaries = [self.common_environment, *(row.get("environment", {}) for row in self.nodes.values())]
        secrets_to_hide = [v for values in dictionaries for key,v in values.items()
                          if any(tag in key.upper() for tag in ("TOKEN", "KEY", "SECRET", "PASSWORD", "PSK", "CREDENTIAL"))]
        secrets_to_hide += [row.get(key) for row in self.nodes.values() for key in ("node_key", "coordinator_key", "ssh_key")]
        for secret in sorted((s for s in secrets_to_hide if isinstance(s,str) and s), key=len, reverse=True):
            value = value.replace(secret, "<redacted>")
        return value[-8192:]

    def _remote(self, node, code, timeout=60, payload=None):
        command = python_command(node, code)
        try:
            result = self.executor(ssh_argv(node, command), capture_output=True, text=True, timeout=timeout,
                **({"input": json.dumps(payload)} if payload is not None else {}))
        except subprocess.TimeoutExpired:
            raise TimeoutError("SSH node operation timed out; no readiness inferred") from None
        if result.returncode:
            diagnostic = str(result.stderr or "").replace(command, "<remote helper>").replace(code, "<remote helper>")
            raise RuntimeError(f"node operation failed (exit {result.returncode}): " + self._sanitize(diagnostic))
        return result.stdout

    def stage_command(self, stage, node):
        path = self._plan_path(node)
        argv = [node.get("python", "python3"), "engines/deepseek_v4/v4_pipe.py", "stage",
            "--deployment-plan", path, "--stage", str(stage["index"]), "--nstages", str(self.plan["nstages"]),
            "--lo", str(stage["lo"]), "--hi", str(stage["hi"]), "--port", str(_port(node["listen_port"])),
            "--dir", node["model"], "--device", node.get("device", "cuda:0"),
            "--bind", node.get("bind", "127.0.0.1"), "--receipts", "--timeout", str(int(node.get("io_timeout_s", 900)))]
        if stage["next_endpoint"]: argv += ["--next", stage["next_endpoint"]]
        if stage["tail"] and node.get("dspark", True): argv.append("--dspark")
        return argv, path

    def _plan_path(self, node):
        return str(PurePosixPath(node["workspace"]) / ".shard-deployments" / self.plan["ring_id"] / "plan.json")

    def _owned_name(self, suffix):
        name = self.plan["ring_id"] + "." + suffix
        if len(name) > 128:
            name = self.plan["ring_id"][:80] + "." + hashlib.sha256(self.plan["ring_id"].encode()).hexdigest()[:12] + "." + suffix
        return name

    def _name(self, stage):
        return self._owned_name("stage" + str(stage["index"]))

    def _json_remote(self, node, code, *, payload=None, timeout=60):
        raw = self._remote(node, code, timeout=timeout, payload=payload)
        return json.loads(raw.strip().splitlines()[-1])

    def _status(self, stage):
        node, name = self.nodes[stage["node_id"]], self._name(stage)
        code = ("import json; from engines.deepseek_v4.v4_preflight import process_observation; "
                f"print(json.dumps(process_observation({name!r},listener_port={node['listen_port']!r})))")
        return self._json_remote(node, code)

    def preflight(self, *, verify_files=True, probe_gpu=True, check_tokenizer=True):
        reports = []
        targets = [(stage["index"], self.nodes[stage["node_id"]]) for stage in self.plan["stages"]]
        targets.append((None, self.nodes[self.coordinator_id]))
        for index, node in targets:
            try:
                owned = self._status(self.plan["stages"][index]).get("running", False) if index is not None else False
                code = ("import json,sys; data=json.load(sys.stdin); "
                    "from engines.deepseek_v4.v4_preflight import apply_environment,check_node; "
                    "apply_environment(data.pop('environment')); print(json.dumps(check_node(**data)))")
                reports.append(self._json_remote(node, code, timeout=self.config.get("preflight_timeout_s", 3600), payload={
                    "plan": self.plan, "index": index, "node": node, "environment": self.environment(node, coordinator=index is None),
                    "verify_files": verify_files, "probe_gpu": probe_gpu, "check_tokenizer": check_tokenizer,
                    "allow_owned_listener": bool(owned)}))
            except Exception as error:
                reports.append({"node_id": self.plan["stages"][index]["node_id"] if index is not None else self.coordinator_id,
                    "stage": index, "preflight_ok": False, "errors": [{"check": "node_operation", "error_class": type(error).__name__,
                                                                        "message": self._sanitize(error)}]})
        return {"schema": "shard-v4-deployment-preflight/1", "preflight_ok": all(row.get("preflight_ok") for row in reports),
                "runtime_ready": False, "scope": self.scope, "nodes": reports}

    def health(self):
        rows = []
        for stage in self.plan["stages"]:
            try: rows.append({"node_id": stage["node_id"], "stage": stage["index"], **self._status(stage)})
            except Exception as error: rows.append({"node_id": stage["node_id"], "stage": stage["index"], "running": None,
                                                   "error_class": type(error).__name__, "message": self._sanitize(error)})
        causes = [row for row in rows if row.get("first_traceback")]
        return {"schema": "shard-v4-deployment-health/1", "ring_id": self.plan["ring_id"], "nodes": rows,
                "first_local_tracebacks": causes, "runtime_ready": False,
                "scope": "owned process/listener observations; no cross-host first-cause clock or signed warmup claim"}

    def _install_plan(self, node):
        path = self._plan_path(node)
        code = ("import json,sys; from pathlib import Path; data=json.load(sys.stdin); "
                f"p=Path({path!r}); p.parent.mkdir(parents=True,exist_ok=True); "
                "assert not p.is_symlink(), 'plan path is a symlink'; "
                "assert not p.exists() or json.loads(p.read_text())==data['plan'], 'ring ID already binds another plan'; "
                "p.write_text(json.dumps(data['plan']),encoding='utf-8')")
        self._remote(node, code, payload={"plan": self.plan})
        return path

    def start_tunnels(self, routes):
        if not routes:
            return  # Explicit direct/previously provisioned routes stay valid.
        check_tunnels(self.plan, self.nodes, routes)
        for index, route in enumerate(routes):
            node = self.nodes[route["caller"]]
            name = self._owned_name(f"tunnel{index}")
            argv = tunnel_argv(route, self.nodes)
            code = ("import json; from shard.managed_launch import ManagedLauncher; "
                    f"print(json.dumps(ManagedLauncher('.shard-processes').start({name!r},{argv!r})))")
            state = self._json_remote(node, code)
            if not state.get("running"): raise RuntimeError("caller-local SSH tunnel exited")
            if not state.get("reused"): self.remote_tunnels.append((node, name))

    def warmup(self, *, timeout=600):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("positive finite warmup timeout required")
        node = self.nodes[self.coordinator_id]
        path = self._install_plan(node)
        nonce, job_id = secrets.token_hex(32), "deployment-warmup-" + secrets.token_hex(8)
        tail = self.nodes[self.plan["stages"][-1]["node_id"]]
        job = {"jobId": job_id, "swarmId": self.plan["ring_id"], "nonce": nonce, "maxNew": 2,
            "ignoreEOS": True,
            "messages": [{"role": "user", "content": "Reply with a short greeting."}],
            "dspark": tail.get("dspark", True), "pipelined": tail.get("dspark", True)}
        argv = [node.get("python", "python3"), "engines/deepseek_v4/v4_pipe.py", "coord",
            "--deployment-plan", path, "--coordinator-key", node.get("coordinator_key", node["node_key"]),
            "--dir", node["model"], "--receipts", "--timeout", str(int(timeout))]
        name = self._owned_name("warmup." + secrets.token_hex(4))
        runner = ("import subprocess,json,sys; " + f"r=subprocess.run({argv!r},input=json.dumps({job!r})+'\\n',text=True); "
                  "sys.exit(r.returncode)")
        code = ("import json,os,sys; from shard.managed_launch import ManagedLauncher; data=json.load(sys.stdin); "
                "[os.environ.pop(k) for k in tuple(os.environ) if k.startswith('V4_')]; "
                f"print(json.dumps(ManagedLauncher('.shard-processes').start({name!r},"
                f"[{node.get('python','python3')!r},'-c',{runner!r}],environment=data['environment'])))")
        state = self._json_remote(node, code, payload={"environment": self.environment(node, coordinator=True)})
        created = not state.get("reused")
        primary_error = None
        try:
            deadline = time.monotonic() + timeout
            while True:
                code = ("import json; from engines.deepseek_v4.v4_preflight import process_observation; "
                        f"print(json.dumps(process_observation({name!r},log_tail_bytes=1048576)))")
                observed = self._json_remote(node, code)
                self.failure_diagnostics = {"coordinator": observed}
                lines = observed.get("log_tail", "").splitlines()
                records = [json.loads(line[len("SHARD_JOB_DONE "):]) for line in lines if line.startswith("SHARD_JOB_DONE ")]
                if records:
                    break
                if not observed.get("running"):
                    self.failure_diagnostics = {"coordinator": observed}
                    raise RuntimeError("strict coordinator warmup exited; inspect owned log/first traceback")
                if time.monotonic() >= deadline:
                    self.failure_diagnostics = {"coordinator": observed}
                    raise TimeoutError("strict signed warmup timed out")
                time.sleep(.1)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if created:
                try:
                    self._remote(node, "from shard.managed_launch import ManagedLauncher; " +
                                 f"ManagedLauncher('.shard-processes').stop({name!r})")
                except Exception as error:
                    if primary_error is None:
                        raise
                    primary_error.add_note("Owned warmup cleanup unconfirmed: " + type(error).__name__)
        final = records[-1]
        expected = {stage["signer_pubkey"]: (stage["lo"], stage["hi"]) for stage in self.plan["stages"]}
        tokens = final.get("tokenIds")
        if (final.get("jobId") != job_id or final.get("ok") is not True or final.get("tokensGenerated") != 2
                or not isinstance(tokens, list) or len(tokens) != 2
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError("warmup did not commit the complete bound two-token request")
        receipts = final.get("receipts", [])
        verify_coverage(receipts, 43, expected_by_signer=expected, expected_nonce=nonce, check_chain=True)
        if any(r.get("job_id") != job_id or r.get("swarm_id") != self.plan["ring_id"] for r in receipts):
            raise ValueError("warmup receipt belongs to another job or ring")
        self.failure_diagnostics = None
        return {"verified": True, "nonce": nonce, "job_id": job_id, "tokens_generated": final["tokensGenerated"],
                "receipts": receipts, "scope": "strict handshake and complete signed warmup; not numerical/performance acceptance"}

    def deploy(self, *, routes=(), readiness_timeout=600, verify_files=True, probe_gpu=True):
        if type(readiness_timeout) not in (int, float) or not math.isfinite(readiness_timeout) or readiness_timeout <= 0:
            raise ValueError("positive finite startup timeout required")
        if not verify_files or not probe_gpu:
            raise ValueError("startup cannot use metadata-only or GPU-unchecked preflight")
        report = self.preflight(verify_files=True, probe_gpu=True)
        if not report["preflight_ok"]:
            return {"ready": False, "scope": self.scope, "preflight": report}
        try:
            self.start_tunnels(routes)
            for stage in reversed(self.plan["stages"]):
                node, name = self.nodes[stage["node_id"]], self._name(stage)
                _, path = self.stage_command(stage, node)
                self._install_plan(node)
                argv, _ = self.stage_command(stage, node)
                code = ("import json,sys; from shard.managed_launch import ManagedLauncher; data=json.load(sys.stdin); "
                    "import os; [os.environ.pop(k) for k in tuple(os.environ) if k.startswith('V4_')]; "
                    f"print(json.dumps(ManagedLauncher('.shard-processes').start({name!r},{argv!r},environment=data['environment'])))")
                state = self._json_remote(node, code, payload={"environment": self.environment(node)})
                if not state.get("running"): raise RuntimeError("stage failed before listening")
                if not state.get("reused"): self.started.append((node, name))
                deadline = time.monotonic() + readiness_timeout
                while True:
                    status = self._status(stage)
                    if not status.get("running"): raise RuntimeError("stage exited; inspect first local traceback")
                    if status.get("listening"): break
                    if time.monotonic() >= deadline: raise TimeoutError("owned stage startup timed out")
                    time.sleep(.25)
            proof = self.warmup(timeout=readiness_timeout)
            return {"ready": True, "ring_id": self.plan["ring_id"], "scope": self.scope,
                    "resource_leases": False, "signed_warmup": proof, "preflight": report}
        except BaseException as error:
            self.failure_diagnostics = {"original_error_class": type(error).__name__, "original_error": str(error),
                                        "health": self.health(), **(self.failure_diagnostics or {})}
            try:
                self.rollback()
            except Exception as cleanup_error:
                error.add_note("Owned cleanup failed: " + type(cleanup_error).__name__ + "; original traceback retained")
                self.failure_diagnostics["cleanup_error_class"] = type(cleanup_error).__name__
            self.failure_diagnostics = json.loads(self._sanitize_json(self.failure_diagnostics))
            path = self.local.directory / (self._owned_name("failure." + secrets.token_hex(4)) + ".json")
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(self.failure_diagnostics, stream, allow_nan=False)
                error.add_note("Protected deployment diagnostics: " + str(path))
            except OSError as diagnostic_error:
                error.add_note("Diagnostic write failed: " + type(diagnostic_error).__name__ + "; original traceback retained")
            raise

    def _sanitize_json(self, value):
        def walk(item):
            if isinstance(item, dict): return {key: walk(val) for key,val in item.items()}
            if isinstance(item, list): return [walk(val) for val in item]
            return self._sanitize(item) if isinstance(item,str) else item
        return json.dumps(walk(value), allow_nan=False)

    def rollback(self):
        errors = []
        try: super().rollback()
        except Exception as error: errors.append(error)
        for node, name in reversed(self.remote_tunnels):
            try: self._remote(node, "from shard.managed_launch import ManagedLauncher; " +
                              f"ManagedLauncher('.shard-processes').stop({name!r})")
            except Exception as error: errors.append(error)
        if errors: raise RuntimeError("owned V4 cleanup was not fully acknowledged") from errors[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--metadata", required=True); generate.add_argument("--cohort", required=True)
    generate.add_argument("--config", required=True); generate.add_argument("--ring-id", required=True)
    generate.add_argument("--split"); generate.add_argument("--out", required=True)
    for name in ("preflight", "health", "start", "stop", "tunnels"):
        child = commands.add_parser(name); child.add_argument("--plan", required=True); child.add_argument("--config", required=True)
        child.add_argument("--state-dir", default=".shard-processes")
        if name == "preflight": child.add_argument("--preview", action="store_true")
        if name == "tunnels": child.add_argument("--hub"); child.add_argument("--out")
        if name == "start": child.add_argument("--startup-timeout", type=int, default=900); child.add_argument("--routes-file")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    if args.command == "generate":
        plan = generate_plan(args.metadata, read_json(args.cohort), config, ring_id=args.ring_id, split=args.split)
        write_plan(args.out, plan); print(json.dumps({"plan": args.out, "ring_id": plan["ring_id"], "scope": "manual strict plan; no resource calibration invented"})); return 0
    deployment = V4Deployment(load_plan(args.plan), config, state_dir=args.state_dir)
    if args.command == "tunnels":
        routes = generate_tunnels(deployment.plan, deployment.nodes, hub=args.hub)
        result = {"routes": routes, "commands": [{"caller": row["caller"], "argv": tunnel_argv(row, deployment.nodes)} for row in routes],
                  "scope": "static caller-local routing configuration; authentication/reachability not yet tested"}
        if args.out: Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command == "preflight": result = deployment.preflight(verify_files=not args.preview, probe_gpu=not args.preview)
    elif args.command == "health": result = deployment.health()
    elif args.command == "start":
        routes = read_json(args.routes_file)["routes"] if args.routes_file else config.get("tunnel_routes", [])
        result = deployment.deploy(routes=routes, readiness_timeout=args.startup_timeout)
    else:
        deployment.started = [(deployment.nodes[stage["node_id"]], deployment._name(stage)) for stage in deployment.plan["stages"]]
        deployment.remote_tunnels = [(deployment.nodes[row["caller"]], deployment._owned_name(f"tunnel{i}"))
                                     for i, row in enumerate(config.get("tunnel_routes", []))]
        deployment.rollback(); result = {"stopped": True, "scope": "only recorded owned process identities"}
    print(json.dumps(result, allow_nan=False))
    return 1 if result.get("ready") is False or result.get("preflight_ok") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())

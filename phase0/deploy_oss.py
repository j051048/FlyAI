"""Deploy a validated GPT-OSS ring using owned processes and optional SSH -L routes.

Full checkouts (including shard/) are required on nodes. Existing network
sidecars remain responsible for peer discovery and inference connectivity.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shard.managed_launch import ManagedLauncher
from shard.pipeline_plan import load_plan


def ssh_argv(node, command=None, *, forward=None):
    target = node.get("ssh_target")
    if not isinstance(target, str) or not target or target.startswith("-") or any(c.isspace() for c in target):
        raise ValueError("SSH target must be an alias or user@host")
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=15", "-o", "TCPKeepAlive=yes"]
    if node.get("ssh_port") is not None:
        if type(node["ssh_port"]) is not int or not 1 <= node["ssh_port"] <= 65535:
            raise ValueError("invalid SSH port")
        argv += ["-p", str(node["ssh_port"])]
    if node.get("ssh_key"):
        argv += ["-i", str(node["ssh_key"])]
    if forward is not None:
        if not isinstance(forward, str) or not forward or any(c.isspace() for c in forward):
            raise ValueError("invalid SSH forward")
        argv += ["-N", "-L", forward]
    argv.append(target)
    if command is not None:
        argv.append(command)
    return argv


def python_command(node, code):
    return "cd " + shlex.quote(node["workspace"]) + " && " + shlex.quote(node.get("python", "python3")) + " -c " + shlex.quote(code)


class RingDeployment:
    def __init__(self, plan, nodes, *, state_dir, executor=subprocess.run):
        from shard.pipeline_plan import validate_plan
        self.plan = validate_plan(plan)
        self.nodes, self.executor = dict(nodes), executor
        self.coordinator_id = self.plan["coordinator"].get("node_id", self.plan["stages"][0]["node_id"])
        if not isinstance(self.coordinator_id, str) or self.coordinator_id not in self.nodes:
            raise ValueError("coordinator requires an SSH deployment entry")
        selected_ids = {stage["node_id"] for stage in self.plan["stages"]} | {self.coordinator_id}
        for node_id in selected_ids:
            if node_id not in self.nodes:
                raise ValueError("every selected node requires an SSH deployment entry")
            node = self.nodes[node_id]
            for name in ("workspace", "model", "node_key"):
                if not isinstance(node.get(name), str) or not node[name]:
                    raise ValueError(f"node requires local {name}")
            ssh_argv(node)
        # These are configured software ceilings, not measured GPU capacity.
        caps = [self._node_context(self.nodes[node_id]) for node_id in selected_ids]
        caps += [stage["context_limit"] for stage in self.plan["stages"] if stage.get("context_limit") is not None]
        if self.plan.get("execution"):
            caps.append(self.plan["execution"]["max_context"])
        self.max_context = min(caps)
        self.local = ManagedLauncher(state_dir)
        self.started, self.tunnels = [], []

    def _remote(self, node, code, timeout=60, payload=None):
        result = self.executor(ssh_argv(node, python_command(node, code)),
                               capture_output=True, text=True, timeout=timeout,
                               **({"input": json.dumps(payload)} if payload is not None else {}))
        if result.returncode:
            # Avoid echoing a command/environment or credentials in an error.
            raise RuntimeError(f"node operation failed (exit {result.returncode})")
        return result.stdout

    def _name(self, stage):
        return f"{self.plan['ring_id']}.stage{stage['index']}"

    @staticmethod
    def _node_context(node):
        value = node.get("max_context", 8192)
        if type(value) is not int or value < 1:
            raise ValueError("node max_context must be a positive integer")
        return value

    def stage_command(self, stage, node):
        plan_file = str(PurePosixPath(node["workspace"]) / ".shard-deployments" / self.plan["ring_id"] / "plan.json")
        # The advertised/dial port may be a NAT mapping or a caller-local SSH
        # forward. The engine binds the container's actual listener port.
        port = node.get("listen_port", urlsplit("//" + stage["endpoint"]).port)
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("listen_port must be an integer in 1..65535")
        argv = [node.get("python", "python3"), "phase0/specpipe.py", "--deployment-plan", plan_file,
                "--stage", str(stage["index"]), "--nstages", str(self.plan["nstages"]),
                "--model", node["model"], "--device", node.get("device", "cuda:0"),
                "--listen-port", str(port), "--fast", "--direct-return",
                "--max-ctx", str(min(self._node_context(node), self.max_context)),
                "--timeout", str(node.get("io_timeout_s", 600))]
        if stage["head"]: argv.append("--served-head")
        if stage["next_endpoint"]: argv += ["--next", stage["next_endpoint"]]
        return argv, plan_file

    def deploy(self, *, tunnels=(), warmup=True, readiness_timeout=600):
        try:
            for index, tunnel in enumerate(tunnels):
                name = f"{self.plan['ring_id']}.tunnel{index}"
                result = self.local.start(name, ssh_argv(tunnel, forward=tunnel["forward"]))
                if not result["reused"]: self.tunnels.append(name)
            for stage in reversed(self.plan["stages"]):
                node = self.nodes[stage["node_id"]]
                argv, plan_file = self.stage_command(stage, node)
                env = {**node.get("environment", {}), "SHARD_TRANSPORT": node.get("transport", "libp2p"),
                       "SHARD_RECEIPTS": "1", "SHARD_NODE_KEY": node["node_key"]}
                name = self._name(stage)
                code = ("from pathlib import Path; import json,sys; from shard.managed_launch import ManagedLauncher; "
                        "data=json.load(sys.stdin); "
                        f"p=Path({plan_file!r}); p.parent.mkdir(parents=True,exist_ok=True); "
                        "assert not p.is_symlink(), 'plan cannot be a symlink'; "
                        "assert not p.exists() or json.loads(p.read_text())==data['plan'], 'existing plan differs; use a fresh ring ID'; "
                        "p.write_text(json.dumps(data['plan']),encoding='utf-8'); "
                        f"m=ManagedLauncher('.shard-processes'); print(json.dumps(m.start({name!r},{argv!r},environment=data['environment'])))")
                result = json.loads(self._remote(node, code, payload={"plan":self.plan,"environment":env}).strip().splitlines()[-1])
                if not result.get("running"):
                    raise RuntimeError("stage process failed to start")
                if not result.get("reused"):
                    self.started.append((node, name))
                until = time.monotonic() + readiness_timeout
                while True:
                    code = ("import json; from pathlib import Path; from shard.managed_launch import ManagedLauncher; "
                            f"m=ManagedLauncher('.shard-processes'); s=m.status({name!r}); "
                            f"p=Path('.shard-processes')/{(name+'.log')!r}; "
                            "s['listening']=p.exists() and 'listening on' in p.read_text(errors='replace')[-8192:]; print(json.dumps(s))")
                    state = json.loads(self._remote(node, code).strip().splitlines()[-1])
                    if not state["running"]:
                        raise RuntimeError("stage exited before readiness")
                    if state["listening"]: break
                    if time.monotonic() >= until: raise TimeoutError("stage readiness timed out")
                    time.sleep(.5)
            # Deploy the entry manifest even when warmup is explicitly skipped:
            # an independent coordinator needs it for the later strict run.
            head = self.plan["stages"][0]
            node = self.nodes[self.coordinator_id]
            _, path = self.stage_command(head, node)
            install_plan = ("import json,sys; from pathlib import Path; data=json.load(sys.stdin); "
                            f"p=Path({path!r}); p.parent.mkdir(parents=True,exist_ok=True); "
                            "assert not p.is_symlink(), 'plan cannot be a symlink'; "
                            "assert not p.exists() or json.loads(p.read_text())==data['plan'], 'existing plan differs; use a fresh ring ID'; "
                            "p.write_text(json.dumps(data['plan']),encoding='utf-8')")
            self._remote(node, install_plan, payload={"plan": self.plan})
            if warmup:
                argv = [node.get("python", "python3"), "phase0/specpipe.py", "--coordinator",
                    "--deployment-plan", path, "--nstages", str(self.plan["nstages"]), "--model", node["model"],
                    "--coordinator-key", node.get("coordinator_key", node["node_key"]),
                    "--next", self.plan["coordinator"]["head"], "--tail", self.plan["coordinator"]["tail"],
                    "--direct-return", "--ngram-draft", "--pipe", "--K", "1", "--depth", "1",
                    "--max-new", "2", "--max-ctx", str(self.max_context),
                    "--timeout", str(node.get("io_timeout_s", 600)),
                    "--prompt", "Reply with a greeting.", "--json-result"]
                env = {**node.get("environment", {}), "SHARD_TRANSPORT": node.get("transport", "libp2p"), "SHARD_RECEIPTS": "1"}
                code = ("import subprocess,os,sys,json; data=json.load(sys.stdin); "
                        f"r=subprocess.run({argv!r},env={{**os.environ,**data['environment']}}); sys.exit(r.returncode)")
                raw = self._remote(node, code, timeout=readiness_timeout, payload={"plan":self.plan, "environment":env})
                results = [json.loads(line[len("RESULT "):]) for line in raw.splitlines() if line.startswith("RESULT ")]
                if not results or not results[-1].get("proof_verified") or not results[-1].get("output_ids"):
                    raise RuntimeError("end-to-end warmup did not produce verified stage receipts")
            return {"deployed": True, "ring_id": self.plan["ring_id"], "signed_warmup": bool(warmup)}
        except BaseException:
            self.rollback()
            raise

    def rollback(self):
        errors = []
        for node, name in reversed(self.started):
            try:
                self._remote(node, "from shard.managed_launch import ManagedLauncher; "
                                   f"ManagedLauncher('.shard-processes').stop({name!r})")
            except Exception as error: errors.append(error)
        for name in reversed(self.tunnels):
            try: self.local.stop(name)
            except Exception as error: errors.append(error)
        if errors:
            raise RuntimeError("owned deployment cleanup was not fully acknowledged") from errors[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--nodes", "--config", dest="nodes", required=True,
                        help="JSON nodes map and optional tunnels list (also used by make_plan.py)")
    parser.add_argument("--state-dir", default=".shard-processes")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.nodes).read_text(encoding="utf-8-sig"))
    deployment = RingDeployment(load_plan(args.plan), config["nodes"], state_dir=args.state_dir)
    print(json.dumps(deployment.deploy(tunnels=config.get("tunnels", []), warmup=not args.no_warmup)))


if __name__ == "__main__": main()

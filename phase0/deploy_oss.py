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
        for stage in self.plan["stages"]:
            if stage["node_id"] not in self.nodes:
                raise ValueError("every selected node requires an SSH deployment entry")
            node = self.nodes[stage["node_id"]]
            for name in ("workspace", "model", "node_key"):
                if not isinstance(node.get(name), str) or not node[name]:
                    raise ValueError(f"node requires local {name}")
            ssh_argv(node)
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

    def stage_command(self, stage, node):
        plan_file = str(PurePosixPath(node["workspace"]) / ".shard-deployments" / self.plan["ring_id"] / "plan.json")
        port = urlsplit("//" + stage["endpoint"]).port
        argv = [node.get("python", "python3"), "phase0/specpipe.py", "--deployment-plan", plan_file,
                "--stage", str(stage["index"]), "--nstages", str(self.plan["nstages"]),
                "--model", node["model"], "--device", node.get("device", "cuda:0"),
                "--listen-port", str(port), "--fast", "--direct-return",
                "--max-ctx", str(node.get("max_context", 8192)), "--timeout", str(node.get("io_timeout_s", 600))]
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
            if warmup:
                head = self.plan["stages"][0]
                node = self.nodes[head["node_id"]]
                _, path = self.stage_command(head, node)
                argv = [node.get("python", "python3"), "phase0/specpipe.py", "--coordinator",
                    "--deployment-plan", path, "--nstages", str(self.plan["nstages"]), "--model", node["model"],
                    "--coordinator-key", node.get("coordinator_key", node["node_key"]),
                    "--next", self.plan["coordinator"]["head"], "--tail", self.plan["coordinator"]["tail"],
                    "--direct-return", "--ngram-draft", "--pipe", "--K", "1", "--depth", "1",
                    "--max-new", "2", "--max-ctx", str(node.get("max_context", 8192)), "--prompt", "Reply with a greeting.", "--json-result"]
                env = {**node.get("environment", {}), "SHARD_TRANSPORT": node.get("transport", "libp2p"), "SHARD_RECEIPTS": "1"}
                code = ("import subprocess,os,sys,json; data=json.load(sys.stdin); "
                        f"r=subprocess.run({argv!r},env={{**os.environ,**data['environment']}}); sys.exit(r.returncode)")
                raw = self._remote(node, code, timeout=readiness_timeout, payload={"environment":env})
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
    parser.add_argument("--nodes", required=True, help="JSON nodes map and optional tunnels list")
    parser.add_argument("--state-dir", default=".shard-processes")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.nodes).read_text(encoding="utf-8-sig"))
    deployment = RingDeployment(load_plan(args.plan), config["nodes"], state_dir=args.state_dir)
    print(json.dumps(deployment.deploy(tunnels=config.get("tunnels", []), warmup=not args.no_warmup)))


if __name__ == "__main__": main()

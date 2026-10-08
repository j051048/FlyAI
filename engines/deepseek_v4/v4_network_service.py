"""V4 adapter: discover signed contributors, form leased rings and serve requests."""
from __future__ import annotations
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import ssl
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shard.control_plane import (ControlError, FormationController, LeaseRPCClient,
                                 ManagedRingBackend, _json_loads, load_sidecar_key)
from shard.offers import OfferRegistry, model_cohort_id

def serve_open_config(config_path, *, auth_file, bind="127.0.0.1", port=8000,
                      tls_cert=None, tls_key=None):
    """Executable reference adapter for the current V4 engine and leased nodes.

    Inference routes/sidecars are configured by the existing launcher. Arbitrary
    GPU identities are discovered from signed offers, not an operator allowlist.
    Calibrations are advertised by nodes; no new node IDs need adding to config.
    """
    from engines.deepseek_v4.v4_gateway import V4RingBackend, AuthRegistry, Gateway
    from .ring_pool import RingPool
    from .resources import PlacementRequirements
    path = Path(config_path).resolve()
    body = _json_loads(path.read_bytes())
    if body.get("schema") != "shard-open-network/1":
        raise ControlError("shard-open-network/1 configuration required")
    def local(value):
        target = Path(value)
        return target if target.is_absolute() else path.parent / target
    key = load_sidecar_key(local(body["controller_sidecar_key"]))
    registry, pool = OfferRegistry(local(body["registry_db"])), RingPool()
    records = {row["ring_id"]: row for row in body["formations"]}
    if len(records) != len(body["formations"]):
        raise ControlError("duplicate formation ID")
    def agent_factory(offer):
        address = offer["lease_endpoint"]
        parsed = urlparse(address)
        if not body.get("allow_private_endpoints", False):
            # Automatic untrusted endpoints cannot probe coordinator-local services.
            # Operator-configured private networks can explicitly enable them.
            try:
                ip = ipaddress.ip_address(parsed.hostname)
            except ValueError:
                raise ControlError("automatic public lease endpoints require a numeric public IP")
            if not ip.is_global:
                raise ControlError("private lease endpoint discovery is disabled")
        return LeaseRPCClient(address, offer["node_id"], key, timeout_s=body.get("rpc_timeout_s", 10))
    def requirements(stage, offer):
        for capability in offer["models"]:
            if model_cohort_id(capability["cohort"]) != stage["cohort_id"]:
                continue
            for row in capability.get("calibrations", []):
                req = PlacementRequirements.from_dict(row["requirements"])
                if (req.layer_start, req.layer_end) == (stage["lo"], stage["hi"]):
                    return req
        raise ControlError("selected node needs an exact calibration for this layer block")
    holder = {}
    def backend_factory(plan, cohort, contracts):
        row = records[plan["ring_id"]]
        directory = local(row["dir"])
        if hashlib.sha256((directory / "config.json").read_bytes()).hexdigest() != cohort.config_sha256:
            raise ControlError("coordinator model config differs from cohort")
        assignments = {}
        selected = []
        leases = {client.node_id: (client, lease) for client, lease in holder["controller"]._formations[plan["ring_id"]]["leases"]}
        for stage in plan["stages"]:
            offer = registry.get(stage["id"])
            assignments[offer["public_key"]] = [stage["lo"], stage["hi"]]
            client, lease = leases[stage["id"]]
            assignment = {"ring_id": plan["ring_id"], "cohort_id": cohort.cohort_id,
                "node_id": stage["id"], "gpu_uuid": offer["gpu_uuid"], "lo": stage["lo"], "hi": stage["hi"],
                "head": stage["head"], "tail": stage["tail"], "stage": stage["index"], "nstages": len(plan["stages"]),
                "next": row.get("stage_next", {}).get(stage["id"], "127.0.0.1:29611")}
            selected.append((client, lease, assignment))
        backend = V4RingBackend(str(directory), row["head"], row["tail"], assignments,
                               mode=row.get("mode", "pipelined"), swarm_id=plan["ring_id"],
                               max_context=row.get("max_context", 8192), timeout=row.get("io_timeout_s", 60))
        if backend.model_id != cohort.model_id or backend.layers != cohort.n_layers:
            backend.close()
            raise ControlError("current engine does not implement the selected model cohort")
        backend.model_cohort = cohort.to_dict()
        return ManagedRingBackend(backend, selected)
    controller = FormationController(registry, pool, {}, requirements=requirements,
        backend_factory=backend_factory, agent_factory=agent_factory)
    holder["controller"] = controller
    gateway = server = None
    try:
        for row in records.values():
            observation = row["measurements"]
            if isinstance(observation, str):
                observation = _json_loads(local(observation).read_bytes())
            controller.form(row["ring_id"], row["cohort"], row["profile"], measurements=observation,
                            locality=row.get("locality"), objective=row.get("objective", "pipeline"),
                            workload=row.get("workload"), ttl_s=row.get("lease_ttl_s", 120),
                            warmup_timeout_s=row.get("warmup_timeout_s", 300))
        for alias, target in body.get("aliases", {}).items():
            pool.set_alias(alias, target["model_id"], target["cohort_id"])
        gateway = Gateway(auth=AuthRegistry.from_file(auth_file), ring_pool=pool, default_model=body.get("default_model"))
        if bind not in {"127.0.0.1", "localhost", "::1"} and not (tls_cert and tls_key):
            raise ControlError("public inference listener requires TLS")
        server = gateway.server(bind, port)
        if tls_cert and tls_key:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(tls_cert, tls_key)
            server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
        print(json.dumps({"ready": True, "listen": server.server_address, "rings": pool.snapshot()}), flush=True)
        server.serve_forever()
    finally:
        if server is not None:
            server.server_close()
        if gateway is not None:
            gateway.shutdown()
        controller.close()
        pool.shutdown()
        registry.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--auth-file', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--tls-cert')
    parser.add_argument('--tls-key')
    args = parser.parse_args(argv)
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error('provide both TLS certificate and key')
    try:
        serve_open_config(args.config, auth_file=args.auth_file, bind=args.host,
                          port=args.port, tls_cert=args.tls_cert, tls_key=args.tls_key)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()

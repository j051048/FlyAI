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
    from engines.deepseek_v4.v4_gateway import V4RingBackend
    from shard.http_gateway import AuthRegistry, Gateway
    from shard.network_service import OpenNetworkService

    def factory(directory, manifest, cohort, row, contracts):
        assignments = {stage["signer_pubkey"]: [stage["lo"], stage["hi"]]
                       for stage in manifest["stages"]}
        return V4RingBackend(str(directory), manifest["coordinator"]["head"],
            manifest["coordinator"]["tail"], assignments,
            mode=row.get("mode", "pipelined"), swarm_id=manifest["ring_id"],
            max_context=row.get("max_context", 8192), timeout=row.get("io_timeout_s", 60),
            pipeline_plan=manifest, coordinator_key=service.key)

    service = OpenNetworkService(config_path, factory)
    gateway = server = None
    try:
        service.form_all()
        gateway = Gateway(auth=AuthRegistry.from_file(auth_file), ring_pool=service.pool,
                          default_model=service.config.get("default_model"))
        if bind not in {"127.0.0.1", "localhost", "::1"} and not (tls_cert and tls_key):
            raise ControlError("public inference listener requires TLS")
        server = gateway.server(bind, port)
        if tls_cert and tls_key:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(tls_cert, tls_key)
            server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
        print(json.dumps({"ready": True, "listen": server.server_address, "rings": service.pool.snapshot()}), flush=True)
        server.serve_forever()
    finally:
        if server is not None:
            server.server_close()
        if gateway is not None:
            gateway.shutdown()
        service.close()


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

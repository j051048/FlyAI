"""Engine-independent assembly of registered contributors into leased services."""
from __future__ import annotations

import hashlib
import ipaddress
from pathlib import Path
from urllib.parse import urlparse

from .control_plane import (ControlError, FormationController, LeaseRPCClient,
                            ManagedRingBackend, _json_loads, load_sidecar_key)
from .offers import OfferRegistry, model_cohort_id
from .pipeline_plan import build_plan, validate_plan
from .resources import PlacementRequirements
from .ring_pool import RingPool


def pipeline_assignment(plan, cohort, offers, formation):
    """Bind actual selected identities and configured sidecar routes to one plan."""
    stages = plan["stages"]
    endpoints = [formation.get("stage_endpoints", {}).get(s["id"], "127.0.0.1:29610")
                 for s in stages]
    value = build_plan({"n_layers": cohort.n_layers}, ring_id=plan["ring_id"],
        cohort_id=cohort.cohort_id, endpoints=endpoints,
        split=[s["hi"] - s["lo"] for s in stages],
        node_ids=[s["id"] for s in stages], gpu_uuids=[o["gpu_uuid"] for o in offers],
        head=formation["head"], tail=formation["tail"], model_cohort=cohort.to_dict())
    for index, offer in enumerate(offers):
        value["stages"][index]["signer_pubkey"] = offer["public_key"]
        if stages[index].get("runtime_config_sha256") is not None:
            value["stages"][index]["runtime_config_sha256"] = stages[index]["runtime_config_sha256"]
        if index < len(stages) - 1:
            value["stages"][index]["next_endpoint"] = formation.get("stage_next", {}).get(
                stages[index]["id"], "127.0.0.1:29611")
    value["selected_routes"] = plan.get("planning", {}).get("routes", [])
    route_endpoints = formation.get("route_endpoints", {})
    for route in value["selected_routes"]:
        if not route.get("route_id"):
            continue  # v1 keeps its documented operator-provisioned routes
        connect = route_endpoints.get(route["route_id"])
        if not connect:
            raise ControlError("selected measured route needs a configured engine connect endpoint")
        from .pipeline_plan import endpoint
        endpoint(connect)
        if route.get("dial_endpoint") != connect:
            raise ControlError("engine connect endpoint differs from the measured route")
        coordinator = plan.get("planning", {}).get("coordinator_id") or stages[0]["id"]
        for index, stage in enumerate(value["stages"][:-1]):
            if (route["src"], route["dst"]) == (stage["node_id"], value["stages"][index + 1]["node_id"]):
                if route.get("dialer_id") != stage["node_id"]:
                    raise ControlError("forward route has a different measured engine dialer")
                override = formation.get("stage_next", {}).get(stage["node_id"])
                if override is not None and override != connect:
                    raise ControlError("stage route differs from selected measurement")
                stage["next_endpoint"] = connect
        if route["src"] == coordinator and route["dst"] == stages[0]["id"]:
            if route.get("dialer_id") != coordinator:
                raise ControlError("coordinator entry route has a different dialer")
            if connect != formation["head"]:
                raise ControlError("coordinator head route differs from selected measurement")
        # Return is dialled by the coordinator to the tail's return listener,
        # regardless of the tail->coordinator logical data direction.
        if route["src"] == stages[-1]["id"] and route["dst"] == coordinator:
            if route.get("dialer_id") != coordinator:
                raise ControlError("return route must describe the coordinator's actual tail dial")
            if connect != formation["tail"]:
                raise ControlError("coordinator return route differs from selected measurement")
    return validate_plan(value)


class OpenNetworkService:
    """Provide an engine factory; registry/leases/formation stay shared.

    The factory receives (directory, manifest, cohort, formation, contracts).
    It must return a real backend with signed warmup, not a readiness flag.
    """
    def __init__(self, config_path, backend_factory, *, profile_factory=None,
                 registry=None, pool=None, agent_factory=None):
        self.path = Path(config_path).resolve()
        self.config = _json_loads(self.path.read_bytes())
        if self.config.get("schema") != "shard-open-network/1":
            raise ControlError("shard-open-network/1 configuration required")
        self.key = load_sidecar_key(self.local(self.config["controller_sidecar_key"]))
        self.registry = registry or OfferRegistry(self.local(self.config["registry_db"]))
        self.pool = pool or RingPool()
        self.records = {row["ring_id"]: row for row in self.config["formations"]}
        if len(self.records) != len(self.config["formations"]):
            raise ControlError("duplicate formation ID")
        self.backend_factory, self.profile_factory = backend_factory, profile_factory
        self.controller = FormationController(self.registry, self.pool, {},
            requirements=self.requirements, backend_factory=self.backend,
            agent_factory=agent_factory or self.agent)

    def local(self, value):
        target = Path(value)
        return target if target.is_absolute() else self.path.parent / target

    def agent(self, offer):
        address = offer["lease_endpoint"]
        parsed = urlparse(address)
        if not self.config.get("allow_private_endpoints", False):
            try:
                ip = ipaddress.ip_address(parsed.hostname)
            except (ValueError, TypeError):
                raise ControlError("automatic public lease endpoints require a numeric public IP") from None
            if not ip.is_global:
                raise ControlError("private lease endpoint discovery is disabled")
        return LeaseRPCClient(address, offer["node_id"], self.key,
                              timeout_s=self.config.get("rpc_timeout_s", 10))

    @staticmethod
    def requirements(stage, offer):
        matching = []
        for capability in offer["models"]:
            if model_cohort_id(capability["cohort"]) != stage["cohort_id"]:
                continue
            for row in capability.get("calibrations", []):
                req = PlacementRequirements.from_dict(row["requirements"])
                config = row["runtime_config"]
                if ((req.layer_start, req.layer_end) == (stage["lo"], stage["hi"])
                        and type(config.get("head")) is bool and type(config.get("tail")) is bool
                        and config["head"] == stage["head"] and config["tail"] == stage["tail"]
                        and (stage.get("runtime_config_sha256") is None or
                             req.provenance.runtime_config_sha256 == stage["runtime_config_sha256"])):
                    matching.append(req)
        if len(matching) != 1:
            raise ControlError("selected node needs one exact calibration for this layer block and boundary roles")
        return matching[0]

    def backend(self, plan, cohort, contracts):
        row = self.records[plan["ring_id"]]
        directory = self.local(row["dir"])
        if hashlib.sha256((directory / "config.json").read_bytes()).hexdigest() != cohort.config_sha256:
            raise ControlError("coordinator model config differs from cohort")
        offers = [self.registry.get(s["id"]) for s in plan["stages"]]
        manifest = pipeline_assignment(plan, cohort, offers, row)
        from .manifest import pub_b64
        manifest["coordinator"]["signer_pubkey"] = pub_b64(self.key)
        context_limits = []
        for stage, offer, slot in zip(plan["stages"], offers, manifest["stages"]):
            configs = [item["runtime_config"] for capability in offer["models"]
                       if model_cohort_id(capability["cohort"]) == cohort.cohort_id
                       for item in capability.get("calibrations", [])
                       if item["requirements"]["provenance"]["runtime_config_sha256"] == stage.get("runtime_config_sha256")]
            if len(configs) != 1:
                raise ControlError("selected measured stage configuration is ambiguous")
            cfg = configs[0]
            context = cfg.get("max_ctx", cfg.get("args", {}).get("max_seq_len"))
            if type(context) is not int or context < 1:
                raise ControlError("selected stage calibration needs its actual context capacity")
            context_limits.append(context)
            slot["context_limit"] = context
        requested_context = row.get("max_context", 8192)
        if type(requested_context) is not int or requested_context < 1:
            raise ControlError("invalid service context ceiling")
        row = {**row, "max_context": min(requested_context, *context_limits)}
        manifest["execution"] = {"max_context": row["max_context"]}
        leases = {client.node_id: (client, lease) for client, lease in
                  self.controller._formations[plan["ring_id"]]["leases"]}
        selected = []
        for stage, offer, slot in zip(plan["stages"], offers, manifest["stages"]):
            client, lease = leases[stage["id"]]
            assignment = {"ring_id": plan["ring_id"], "cohort_id": cohort.cohort_id,
                "node_id": stage["id"], "gpu_uuid": offer["gpu_uuid"],
                "lo": stage["lo"], "hi": stage["hi"], "head": stage["head"], "tail": stage["tail"],
                "stage": stage["index"], "nstages": len(plan["stages"]),
                "next": slot["next_endpoint"], "deployment_plan": manifest}
            if stage.get("runtime_config_sha256") is not None:
                assignment["runtime_config_sha256"] = stage["runtime_config_sha256"]
            selected.append((client, lease, assignment))
        backend = self.backend_factory(directory, manifest, cohort, row, contracts)
        if backend.model_id != cohort.model_id or backend.layers != cohort.n_layers:
            backend.close()
            raise ControlError("engine does not implement the selected model cohort")
        backend.model_cohort = cohort.to_dict()
        return ManagedRingBackend(backend, selected)

    def form_all(self):
        results = []
        for row in self.records.values():
            observation = row["measurements"]
            if isinstance(observation, str):
                observation = _json_loads(self.local(observation).read_bytes())
            if observation.get("schema") == "shard-link-measurements/2" and not row.get("coordinator_id"):
                raise ControlError("measured production routes require an explicit coordinator identity")
            profile = row.get("profile")
            if self.profile_factory is not None:
                profile = self.profile_factory(self.local(row["dir"]), row)
            profile = {**profile, "require_exact_calibrations": True}
            extra = {"coordinator_id": row["coordinator_id"]} if row.get("coordinator_id") else {}
            if row.get("route_ids") is not None:
                extra["route_ids"] = row["route_ids"]
            results.append(self.controller.form(row["ring_id"], row["cohort"], profile,
                measurements=observation, locality=row.get("locality"),
                objective=row.get("objective", "pipeline"), workload=row.get("workload"),
                ttl_s=row.get("lease_ttl_s", 120), warmup_timeout_s=row.get("warmup_timeout_s", 300), **extra))
        for alias, target in self.config.get("aliases", {}).items():
            self.pool.set_alias(alias, target["model_id"], target["cohort_id"])
        return results

    def close(self):
        self.controller.close()
        self.pool.shutdown()
        self.registry.close()

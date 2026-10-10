"""Engine-independent assembly of registered contributors into leased services."""
from __future__ import annotations

import hashlib
import ipaddress
import math
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlparse

from .control_plane import (ControlError, FormationController, LeaseRPCClient,
                            ManagedRingBackend, _json_loads, load_sidecar_key)
from .offers import OfferRegistry, model_cohort_id
from .pipeline_plan import build_plan, validate_plan
from .resources import PlacementRequirements
from .ring_pool import RingPool


class PreparedManagedRingBackend(ManagedRingBackend):
    """Finish authenticated node preparation before starting any stage process."""
    def __init__(self, backend, nodes, *, prepare_timeout_s=3600, poll_s=.1):
        super().__init__(backend, nodes)
        if (type(prepare_timeout_s) not in (int, float) or not math.isfinite(prepare_timeout_s)
                or prepare_timeout_s <= 0):
            raise ControlError("finite positive weight preparation timeout required")
        self.prepare_timeout_s, self.poll_s = float(prepare_timeout_s), poll_s
        self.preparations = {}

    def load(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        deadline = time.monotonic() + self.prepare_timeout_s
        cancelled = threading.Event()
        def prepare(record):
            client, lease, assignment = record
            if assignment.get("weight_artifacts") is None:
                return
            state = client.operation("prepare_stage", lease, assignment=assignment)
            job = state["job_id"]
            while state.get("state") != "ready":
                if cancelled.is_set():
                    raise ControlError("another selected node failed preparation")
                if state.get("state") == "failed":
                    if state.get("error_code") == "resource_conflict":
                        from .leases import LeaseConflict
                        raise LeaseConflict("node stage preparation lacks resources: " + str(state.get("error")))
                    raise ControlError("node stage preparation failed: " + str(state.get("error")))
                if time.monotonic() >= deadline:
                    raise ControlError("node stage preparation deadline exceeded")
                time.sleep(min(self.poll_s, max(0, deadline - time.monotonic())))
                state = client.operation("prepare_status", lease, assignment=assignment, job_id=job)
            expected = assignment["weight_artifacts"]
            if (state.get("payload_integrity_verified") is not True
                    or state.get("source_artifact_id") != expected["artifact_id"]
                    or state.get("checkpoint_id") != expected["checkpoint_id"]
                    or state.get("manifest_sha256") != expected["manifest_sha256"]):
                raise ControlError("node prepared weights differ from the pinned assignment")
            if (assignment.get("preparation_mode") is not None
                    and state.get("preparation_mode") != assignment["preparation_mode"]):
                raise ControlError("node used a different strategy from the planned preparation contract")
            self.preparations[client.node_id] = state
        try:
            with ThreadPoolExecutor(max_workers=min(16, len(self.nodes))) as workers:
                futures = [workers.submit(prepare, record) for record in self.nodes]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except BaseException:
                        cancelled.set()
                        raise
            super().load()
        except BaseException:
            self.close()
            raise

    def stats(self):
        return {**super().stats(), "weight_preparations": dict(self.preparations)}


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
        self._formation_lock = threading.RLock()
        self._replacement_policies = {}
        self._configured_record_ids = set(self.records)
        self._generation_sequence = 0
        self._reconcile_stop = threading.Event()
        self._reconcile_thread = None
        self._reconcile_interval = self.config.get("reconcile_interval_s", 1)
        self._reconcile_cooldown = self.config.get("reconcile_cooldown_s", 10)
        for value in (self._reconcile_interval, self._reconcile_cooldown):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ControlError("positive finite reconciliation intervals required")
        for row in self.records.values():
            enabled = row.get("auto_replace", bool(row.get("calibrated_alternatives")))
            if type(enabled) is not bool:
                raise ControlError("auto_replace must be an explicit boolean")
            if type(row.get("auto_generations", True)) is not bool:
                raise ControlError("auto_generations must be an explicit boolean")
            if enabled:
                self._replacement_policies[row["ring_id"]] = {
                    "definition": row, "active_ring_id": None, "attempted": set(),
                    "running": False, "next_attempt": 0.0}
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
        runtime_configs = {}
        for stage, offer, slot in zip(plan["stages"], offers, manifest["stages"]):
            configs = [item["runtime_config"] for capability in offer["models"]
                       if model_cohort_id(capability["cohort"]) == cohort.cohort_id
                       for item in capability.get("calibrations", [])
                       if item["requirements"]["provenance"]["runtime_config_sha256"] == stage.get("runtime_config_sha256")]
            if len(configs) != 1:
                raise ControlError("selected measured stage configuration is ambiguous")
            cfg = configs[0]
            runtime_configs[stage["id"]] = cfg
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
        artifact_source = row.get("weight_artifacts")
        if artifact_source is not None:
            from .weight_artifacts import validate_catalog, validate_pack
            if not isinstance(artifact_source, dict) or set(artifact_source) != {"catalog", "pack"}:
                raise ControlError("local catalog and pack metadata paths required")
            catalog = validate_catalog(_json_loads(self.local(artifact_source["catalog"]).read_bytes()))
            pack = validate_pack(catalog, _json_loads(self.local(artifact_source["pack"]).read_bytes()))
            if (catalog["checkpoint_id"] != cohort.checkpoint_id or catalog["manifest_sha256"] != cohort.manifest_sha256):
                raise ControlError("weight catalog differs from formation cohort")
        for stage, offer, slot in zip(plan["stages"], offers, manifest["stages"]):
            client, lease = leases[stage["id"]]
            assignment = {"ring_id": plan["ring_id"], "cohort_id": cohort.cohort_id,
                "node_id": stage["id"], "gpu_uuid": offer["gpu_uuid"],
                "lo": stage["lo"], "hi": stage["hi"], "head": stage["head"], "tail": stage["tail"],
                "stage": stage["index"], "nstages": len(plan["stages"]),
                "next": slot["next_endpoint"], "deployment_plan": manifest}
            if stage.get("runtime_config_sha256") is not None:
                assignment["runtime_config_sha256"] = stage["runtime_config_sha256"]
            stage_config = runtime_configs[stage["id"]]
            if type(stage_config.get("dspark", False)) is not bool:
                raise ControlError("calibrated draft role must be boolean")
            measured_artifact = stage.get("weight_artifacts")
            if measured_artifact is not None:
                if (not isinstance(measured_artifact, dict)
                        or set(measured_artifact) != {"artifact_id", "checkpoint_id", "manifest_sha256"}
                        or measured_artifact["checkpoint_id"] != cohort.checkpoint_id
                        or measured_artifact["manifest_sha256"] != cohort.manifest_sha256):
                    raise ControlError("planned preparation artifact differs from formation cohort")
                assignment["weight_artifacts"] = dict(measured_artifact)
            if artifact_source is not None:
                from .weight_artifacts import select_stage_artifacts
                item = select_stage_artifacts(catalog, pack, stage["lo"], stage["hi"],
                    head=stage["head"], tail=stage["tail"], dspark=stage_config.get("dspark", False))
                derived = {key: item[key] for key in ("artifact_id", "checkpoint_id", "manifest_sha256")}
                if measured_artifact is not None and derived != measured_artifact:
                    raise ControlError("planned source artifact differs from the controller packing")
                assignment["weight_artifacts"] = derived
            if assignment.get("weight_artifacts") is not None and stage_config.get("dspark", False):
                assignment["dspark"] = True
            if stage.get("preparation_mode") is not None:
                if assignment.get("weight_artifacts") is None or stage["preparation_mode"] not in ("fetch", "range_repack"):
                    raise ControlError("planned preparation needs pinned artifacts and a supported mode")
                assignment["preparation_mode"] = stage["preparation_mode"]
            selected.append((client, lease, assignment))
        feature_row = {**row, "verified_runtime_configs": runtime_configs,
            "verified_lease_fences": {node_id: {"lease_id": lease["lease_id"], "fencing_token": lease["fencing_token"]}
                                      for node_id, (_, lease) in leases.items()}}
        backend = self.backend_factory(directory, manifest, cohort, feature_row, contracts)
        if backend.model_id != cohort.model_id or backend.layers != cohort.n_layers:
            backend.close()
            raise ControlError("engine does not implement the selected model cohort")
        backend.model_cohort = cohort.to_dict()
        return PreparedManagedRingBackend(backend, selected,
            prepare_timeout_s=row.get("weight_prepare_timeout_s", 3600))

    def form_one(self, row):
        """Form one locally configured candidate; all execution budgets are measured."""
        with self._formation_lock:
            prior = self.records.get(row["ring_id"])
            if prior is not None and prior != row:
                raise ControlError("formation ID cannot be reused for another configuration")
            self.records[row["ring_id"]] = row
            self._prune_records(preserve=row["ring_id"])
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
        return self.controller.form(row["ring_id"], row["cohort"], profile,
            measurements=observation, locality=row.get("locality"),
            objective=row.get("objective", "pipeline"), workload=row.get("workload"),
            ttl_s=row.get("lease_ttl_s", 120), warmup_timeout_s=row.get("warmup_timeout_s", 300), **extra)

    def form_with_alternatives(self, row):
        """Capacity failures can use explicitly configured, same-cohort alternatives.

        No layer speed/capacity is synthesized. Each alternative goes through the
        ordinary planner, exact template matching, reservations and signed warmup.
        """
        from .leases import LeaseConflict
        from .offers import ModelCohort
        target = ModelCohort.from_dict(row["cohort"]).cohort_id
        alternatives = row.get("calibrated_alternatives", [])
        if not isinstance(alternatives, list) or len(alternatives) > 16:
            raise ControlError("at most 16 explicit calibrated alternatives required")
        candidates = [row, *alternatives]
        if len({candidate["ring_id"] for candidate in candidates}) != len(candidates):
            raise ControlError("alternative formation IDs must be distinct")
        for candidate in candidates:
            if ModelCohort.from_dict(candidate["cohort"]).cohort_id != target:
                raise ControlError("capacity fallback must retain the exact model cohort")
        failures = []
        for candidate in candidates:
            try:
                result = self.form_one(candidate)
                result["capacity_fallback"] = failures
                return result
            except LeaseConflict as error:
                failures.append({"ring_id": candidate["ring_id"], "reason": str(error)[:300]})
        raise LeaseConflict("all explicitly calibrated formation alternatives lack reservable resources")

    def replace(self, row, retired_ring_ids, *, aliases=()):
        """Prepare/load/warm a new ring, then switch admission without moving jobs."""
        result = self.form_with_alternatives(row)
        try:
            publication = self.pool.publish_replacement(result["ring_id"], retired_ring_ids, aliases=aliases)
        except BaseException:
            self.controller.stop(result["ring_id"])
            raise
        return {**result, "replacement": publication}

    def form_all(self):
        results = []
        for row in tuple(self.records.values()):
            if row.get("replaces") is not None:
                results.append(self.replace(row, row["replaces"], aliases=row.get("publish_aliases", [])))
            else:
                results.append(self.form_with_alternatives(row))
            policy = self._replacement_policies.get(row["ring_id"])
            if policy is not None:
                result = results[-1]
                policy["active_ring_id"] = result["ring_id"]
                policy["attempted"].update(item["ring_id"] for item in result.get("capacity_fallback", []))
                policy["attempted"].add(result["ring_id"])
                self.pool.record_reconciliation(row["ring_id"], {
                    "state": "armed", "active_ring_id": result["ring_id"]})
        for alias, target in self.config.get("aliases", {}).items():
            self.pool.set_alias(alias, target["model_id"], target["cohort_id"])
        self.start_reconciliation()
        return results

    def start_reconciliation(self):
        """Automatically replace failed routes using bounded configured candidates."""
        with self._formation_lock:
            if not self._replacement_policies or self._reconcile_thread is not None:
                return
            def maintain():
                while not self._reconcile_stop.wait(self._reconcile_interval):
                    self.reconcile_once()
            self._reconcile_thread = threading.Thread(target=maintain, daemon=True,
                                                       name="network-ring-reconciliation")
            self._reconcile_thread.start()

    def reconcile_once(self):
        """Launch independent workers; monitoring never waits for fetch/warmup RPC."""
        from .ring_pool import RingState
        snapshot = {ring.ring_id: ring for ring in self.pool.rings()}
        with self._formation_lock:
            if self._reconcile_stop.is_set():
                return
            for policy_id, policy in self._replacement_policies.items():
                old = snapshot.get(policy["active_ring_id"])
                if (old is None or old.state not in (RingState.FAILED, RingState.DRAINING, RingState.STOPPED)
                        or policy["running"] or time.monotonic() < policy["next_attempt"]):
                    continue
                candidates = [row for row in policy["definition"].get("calibrated_alternatives", [])
                              if row["ring_id"] not in policy["attempted"]]
                if not candidates:
                    templates = policy["definition"].get("calibrated_alternatives", [])
                    if not templates or not policy["definition"].get("auto_generations", True):
                        self.pool.record_reconciliation(policy_id, {"state": "unavailable",
                            "reason": "configured_calibrated_candidates_exhausted", "active_ring_id": old.ring_id})
                        continue
                    previous = snapshot.get(policy.get("last_attempt_id"))
                    if previous is not None and previous is not old and previous.state != RingState.STOPPED:
                        self.pool.record_reconciliation(policy_id, {"state": "unavailable",
                            "reason": "failed_attempt_cleanup_unconfirmed", "active_ring_id": old.ring_id})
                        policy["next_attempt"] = time.monotonic() + self._reconcile_cooldown
                        continue
                    cursor = policy.get("template_cursor", 0)
                    template = templates[cursor % len(templates)]
                    policy["template_cursor"] = cursor + 1
                    self._generation_sequence += 1
                    suffix = f".r{self._generation_sequence:x}.{secrets.token_hex(4)}"
                    # Only the lifecycle epoch changes. Signed calibration,
                    # source artifacts, model, workload and routes remain exact.
                    candidate = {**template, "ring_id": template["ring_id"][:128 - len(suffix)] + suffix,
                                 "calibrated_alternatives": []}
                else:
                    candidate = {**candidates[0], "calibrated_alternatives": []}
                    policy["attempted"].add(candidate["ring_id"])
                policy["last_attempt_id"] = candidate["ring_id"]
                policy["running"] = True
                self.pool.record_reconciliation(policy_id, {"state": "preparing",
                    "active_ring_id": old.ring_id, "candidate_ring_id": candidate["ring_id"]})
                threading.Thread(target=self._reconcile_candidate, args=(policy_id, policy, old, candidate),
                    daemon=True, name="replace-ring-" + candidate["ring_id"]).start()

    def _reconcile_candidate(self, policy_id, policy, old, candidate):
        from .leases import LeaseConflict
        result, published = None, False
        try:
            result = self.form_with_alternatives(candidate)
            if self._reconcile_stop.is_set():
                self.controller.stop(result["ring_id"])
                return
            aliases = [name for name, target in self.pool.snapshot()["aliases"].items()
                       if (target["model_id"], target["cohort_id"]) == (old.model_id, old.cohort_id)]
            self.pool.publish_replacement(result["ring_id"], [old.ring_id], aliases=aliases)
            published = True
            with self._formation_lock:
                policy["active_ring_id"] = result["ring_id"]
            self.pool.record_reconciliation(policy_id, {"state": "published",
                "active_ring_id": result["ring_id"], "retired_ring_id": old.ring_id})
        except Exception as error:
            if result is not None and not published:
                try:
                    self.controller.stop(result["ring_id"])
                except Exception:
                    pass  # retained pool/ledger ownership blocks unconfirmed reuse
            # Categorical public status contains no provider credentials/paths.
            self.pool.record_reconciliation(policy_id, {"state": "unavailable",
                "reason": "capacity_unavailable" if isinstance(error, LeaseConflict) else "replacement_failed",
                "error_class": type(error).__name__, "active_ring_id": old.ring_id,
                "candidate_ring_id": candidate["ring_id"]})
        finally:
            with self._formation_lock:
                policy["running"] = False
                policy["next_attempt"] = time.monotonic() + self._reconcile_cooldown

    def _prune_records(self, *, preserve=None):
        """Keep approved definitions and live generations, bound old runtime rows."""
        limit = len(self._configured_record_ids) + self.pool.max_rings + self.pool.max_history
        if len(self.records) <= limit:
            return
        from .ring_pool import RingState
        live = {ring.ring_id for ring in self.pool.rings() if ring.state != RingState.STOPPED}
        lock = getattr(self.controller, "_lock", None)
        if lock is not None:
            with lock:
                live.update(self.controller._formations)
        live.update(policy.get("active_ring_id") for policy in self._replacement_policies.values())
        for key in tuple(self.records):
            if len(self.records) <= limit:
                break
            if key != preserve and key not in live and key not in self._configured_record_ids:
                self.records.pop(key, None)

    def close(self):
        self._reconcile_stop.set()
        if self._reconcile_thread is not None:
            self._reconcile_thread.join(1)
        self.controller.close()
        self.pool.shutdown()
        self.registry.close()

"""Host grouping is a strict per-GPU placement constraint, not VRAM pooling."""
import hashlib
import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from shard.host_topology import (contiguous_hosts, host_group_policy, observe_linux_topology,
    validate_topology_report, validate_transport_sample)
from shard.plan import plan_ring
from shard.planning_cost import estimate, simulate_frames
from test_calibrated_span_planning import nodes_with_spans, span, add_calibration
from test_planning_locality import mesh, profile, pool


def policy(**kwargs):
    return dict(enabled=True, mode="contiguous", **kwargs)


def test_opt_in_selects_contiguous_calibrated_gpu_spans_instead_of_cheapest_host_bounce():
    nodes = nodes_with_spans([[span(0, 1, depth=4)],
        [span(1, 2, depth=4), span(2, 3, depth=4)],
        [span(1, 2, depth=4), span(2, 3, depth=4)], [span(3, 4, depth=4)]])
    for node, host in zip(nodes, ("A", "B", "A", "B")):
        node["host_id"] = host
    L = mesh(4, 20)
    L[0][1] = L[1][2] = L[2][3] = 1
    original = plan_ring(nodes, L, profile(4), locality={"mode": "global"})
    grouped = plan_ring(nodes, L, profile(4), locality={"mode": "global", "host_groups": policy()})
    assert original["order"] == ["n0", "n1", "n2", "n3"]
    assert grouped["order"] == ["n0", "n2", "n1", "n3"]
    groups = grouped["planning"]["host_groups"]
    assert [g["node_ids"] for g in groups["groups"]] == [["n0", "n2"], ["n1", "n3"]]
    assert [(g["lo"], g["hi"]) for g in groups["groups"]] == [(0, 2), (2, 4)]
    assert groups["tail_return"] == {"src": "n3", "dst": "n0"}
    assert groups["affinity_applied"] is False
    assert len(grouped["stages"]) == 4 and len({uuid for group in groups["groups"] for uuid in group["gpu_uuids"]}) == 4


def test_grouping_requires_exact_calibration_and_does_not_guess_a_host_from_public_ip():
    nodes = pool(["A", "A"])
    assert plan_ring(nodes, mesh(2), profile(), locality={"host_groups": policy()}) is None
    for node in nodes:
        node.pop("host_id")
        node["public_ip"] = "198.51.100.9"
    assert contiguous_hosts([0, 1], dict(enumerate(nodes)))
    assert host_group_policy({"enabled": False}) is None


def test_five_gpu_candidate_cannot_hide_nine_layer_peak_by_pooling_host_vram():
    five = nodes_with_spans([[span(lo, hi, depth=43, gpu=1100 if hi - lo == 9 else 900, index=i, nstages=5)]
        for i, (lo, hi) in enumerate(zip((0, 9, 18, 27, 35), (9, 18, 27, 35, 43)))])
    for node in five:
        node["host_id"] = "one-host"
    assert plan_ring(five, mesh(5), profile(43), locality={"host_groups": policy()}) is None
    six = nodes_with_spans([[span(lo, hi, depth=43, gpu=900, index=i, nstages=6)]
        for i, (lo, hi) in enumerate(zip((0, 8, 16, 24, 32, 40), (8, 16, 24, 32, 40, 43)))])
    for node in six:
        node["host_id"] = "one-host"
    result = plan_ring(six, mesh(6), profile(43), locality={"host_groups": policy()})
    assert result is not None
    assert len(result["planning"]["host_groups"]["groups"]) == 1
    assert result["stages"][-1]["tail"] is True and result["stages"][-1]["lo"] == 40


def sample(**updates):
    return dict(schema="shard-host-transport/1", scope="engine_frame_ready_to_ready", transport="buffered_tcp",
        host_id="host", src="n0", dst="n1", src_gpu_uuid="GPU-0", dst_gpu_uuid="GPU-1",
        src_runtime_config_sha256="a" * 64, dst_runtime_config_sha256="a" * 64,
        cohort_id="c" * 64, source_sha256="f" * 64, route_id="ab", dial_endpoint="127.0.0.1:30002",
        dialer_id="n0", payload_bytes=32, frame_tokens=1, context_tokens=128, warmness="warm",
        measured_at=100, ttl_s=10, samples_ms=[1.0, 2.0], **updates)


def transport_fixture():
    nodes = nodes_with_spans([[span(0, 1, depth=2)], [span(1, 2, depth=2)]])
    for i, node in enumerate(nodes):
        node.update(host_id="host", cohort_id="c" * 64, runtime_config_sha256="a" * 64)
        node["stage_trace"] = dict(schema="shard-stage-trace/1", node_id=node["id"], gpu_uuid=node["gpu_uuid"],
            cohort_id=node["cohort_id"], runtime_config_sha256="a" * 64, layer_start=i, layer_end=i + 1,
            frame_tokens=1, context_tokens=128, warmness="warm", frame_ms=5, prefill_ms=7,
            service_scope="compute_after_receive_before_send", source_sha256="f" * 64, measured_at=100, ttl_s=10)
    edges = {("n0", "n1"): dict(channel="buffered_tcp", route_id="ab", dial_endpoint="127.0.0.1:30002",
        dialer_id="n0", bandwidth_mbps=1000)}
    w = dict(context_tokens=128, warmness="warm", generated_tokens=10, depth=2)
    def forecast(evidence=None, *, slo=None):
        return estimate([0, 1], {0: 1, 1: 1}, [5, 5], mesh(2, 10), [0, 0], [0, 0],
            dict(enumerate(nodes)), profile(2), w, now=100, edges=edges, objective="pipeline",
            host_groups=host_group_policy(policy(transport_samples=[] if evidence is None else [evidence])), slo=slo)
    return nodes, edges, w, forecast


def test_only_selected_backend_exact_scope_runtime_route_and_shape_can_change_transport_cost():
    nodes, edges, w, forecast = transport_fixture()
    baseline = forecast()
    actual = forecast(sample())
    assert actual["predicted_request_ms"] < baseline["predicted_request_ms"]
    assert actual["host_transport_costs"][0]["source"] == "fresh_bound_engine_frame"
    for field, value in (("transport", "loopback_tcp"), ("measured_at", 80), ("src_gpu_uuid", "GPU-other"),
                         ("src_runtime_config_sha256", "b" * 64), ("payload_bytes", 64),
                         ("route_id", "other"), ("dial_endpoint", "127.0.0.1:30004"),
                         ("cohort_id", "d" * 64), ("source_sha256", "e" * 64),
                         ("context_tokens", 129), ("warmness", "cold")):
        altered = sample(); altered[field] = value
        assert forecast(altered)["predicted_request_ms"] == baseline["predicted_request_ms"]
    nodes[0]["stage_trace"].pop("service_scope")
    assert forecast(sample())["predicted_request_ms"] == baseline["predicted_request_ms"]


@pytest.mark.parametrize("endpoint", [0, 1])
@pytest.mark.parametrize("change", ["missing", "changed", "malformed", "pinned_current_source_changed"])
def test_source_only_change_cannot_reuse_transport_saving(endpoint, change):
    nodes, _, _, forecast = transport_fixture()
    original = forecast(sample())
    baseline = forecast()
    assert original["predicted_request_ms"] < baseline["predicted_request_ms"]
    node = nodes[endpoint]
    if change == "missing":
        del node["stage_trace"]["source_sha256"]
    elif change == "pinned_current_source_changed":
        node["source_sha256"] = "b" * 64
    else:
        node["stage_trace"]["source_sha256"] = "b" * 64 if change == "changed" else "unknown"
    actual = forecast(sample())
    assert actual["predicted_request_ms"] == baseline["predicted_request_ms"]
    assert actual["host_transport_costs"][0]["source"] == "route_hop_and_bandwidth_estimate"
    assert any("backend/route/frame/source" in note for note in actual["uncertainty"])


def test_mean_engine_transport_measurement_does_not_inherit_a_different_scope_slo_tail():
    nodes, edges, _, forecast = transport_fixture()
    for node in nodes:
        node["stage_trace"]["quality"] = dict(schema="shard-stage-quality/1", sample_count=20, window_s=1,
            frame_p95_ms=6, prefill_p95_ms=8, queue_p95_ms=0)
    quality = dict(schema="shard-link-quality/1", sample_count=20, window_s=1,
        latency_p95_ms=10, jitter_ms=0, packet_loss_rate=0)
    edges["n0", "n1"].update(quality=quality, measured_at=100, ttl_s=10, source="fresh_sparse", rtt_ms=10)
    edges["n1", "n0"] = dict(quality=quality, measured_at=100, ttl_s=10, source="fresh_sparse", rtt_ms=10)
    objective = dict(schema="shard-planning-slo/1", mode="require_known", targets=dict(tpot_ms=100))
    assert forecast(slo=objective)["slo"]["eligible"]
    observed = forecast(sample(), slo=objective)
    assert observed["host_transport_costs"][0]["source"] == "fresh_bound_engine_frame"
    assert observed["slo"]["metrics"]["tpot_ms"] is None and not observed["slo"]["eligible"]


def test_shared_transport_busy_time_occupies_same_resource_calendar_without_double_charging():
    stages = [dict(queue_ms=0, shared_id=None, shared_ms=0, service_ms=1) for _ in range(3)]
    independent = simulate_frames(stages, [0, 0], [10, 10], 0, 0, 4, 8)
    shared = simulate_frames(stages, [0, 0], [10, 10], 0, 0, 4, 8,
        link_shared=[("host-memory", 10), ("host-memory", 10)])
    assert shared[0] == independent[0] == 23
    assert shared[-1] > independent[-1]
    row = sample(shared_resource_id="host-memory", shared_busy_ms=1)
    assert validate_transport_sample(row) == row
    row["shared_busy_ms"] = 10
    with pytest.raises(ValueError):
        validate_transport_sample(row)


def test_device_copy_capability_cannot_be_an_engine_transport_sample():
    row = sample(); row["scope"] = "device_copy_not_engine_wire"
    with pytest.raises(ValueError):
        validate_transport_sample(row)
    row = sample(); row["samples_ms"] = [float("nan")]
    with pytest.raises(ValueError):
        validate_transport_sample(row)


@pytest.mark.parametrize("kwargs", [dict(devices=list(range(17))), dict(devices=[0, 0]),
    dict(sample_bytes=17 * 1024**2), dict(repeats=101), dict(max_pairs=65), dict(ttl_s=float("nan"))])
def test_pair_probe_refuses_unbounded_allocation_before_importing_or_running_cuda(kwargs):
    from shard.host_probe import measure_topology
    parameters = dict(devices=[0, 1], host_id="host")
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        measure_topology(**parameters)


def test_stale_or_foreign_gpu_topology_does_not_claim_observed_placement():
    nodes = nodes_with_spans([[span(0, 3)], [span(3, 6)]])
    for node in nodes:
        node["host_id"] = "host"
    report = dict(schema="shard-host-topology/1", host_id="host", measured_at=100, ttl_s=10,
        affinity_applied=False, devices=[dict(local_index=i, gpu_uuid=f"GPU-{i}") for i in range(2)], pairs=[])
    def select(value):
        return plan_ring(nodes, mesh(2), profile(), now=100,
            locality={"host_groups": policy(topology_reports=[value])})["planning"]["host_groups"]["groups"][0]
    assert select(report)["topology"] is not None
    report["measured_at"] = 80
    assert select(report)["topology"] is None
    report["measured_at"] = 100; report["devices"][1]["gpu_uuid"] = "GPU-other"
    assert select(report)["topology"] is None


def test_linux_sysfs_observation_reports_only_seen_numa_and_never_applies_affinity(monkeypatch):
    # PCI colons are valid Linux filenames and invalid Windows filenames.
    # A read-only virtual sysfs keeps this parser regression cross-platform.
    from pathlib import PurePosixPath
    from io import StringIO
    from shard import host_topology
    files = {"/proc/driver/nvidia/gpus/00000000:41:00.0/information": "Model: Fixture\nGPU UUID: GPU-0\n",
        "/sys/bus/pci/devices/0000:41:00.0/numa_node": "1",
        "/sys/bus/pci/devices/0000:41:00.0/local_cpulist": "8-15,32-39"}
    class ReadPath(PurePosixPath):
        def open(self, *args, **kwargs):
            if str(self) not in files:
                raise FileNotFoundError(str(self))
            return StringIO(files[str(self)])
        def iterdir(self):
            return iter([ReadPath("/proc/driver/nvidia/gpus/00000000:41:00.0")])
        def resolve(self):
            return self
    monkeypatch.setattr(host_topology, "Path", ReadPath)
    rows = observe_linux_topology([dict(local_index=0, gpu_uuid="GPU-0"), dict(local_index=1, gpu_uuid="GPU-1")],
        platform="linux")
    assert rows[0]["numa_node"] == 1 and rows[0]["pci_bus_id"] == "0000:41:00.0"
    assert rows[0]["local_cpulist"] == "8-15,32-39"
    assert rows[1]["numa_node"] is None and rows[1]["pci_bus_id"] is None
    report = dict(schema="shard-host-topology/1", host_id="host", measured_at=100, ttl_s=10,
        affinity_applied=False, devices=rows, pairs=[])
    assert validate_topology_report(report)["affinity_applied"] is False
    report["affinity_applied"] = True
    with pytest.raises(ValueError):
        validate_topology_report(report)


def test_strict_formation_keeps_per_gpu_leases_signers_and_actual_tail_return(tmp_path):
    """Real registry/SQLite/assignment/receipt verification; CPU fixture arithmetic."""
    from shard.control_plane import FormationController, LocalLeaseClient
    from shard.leases import LeaseLedger
    from shard.network_service import pipeline_assignment
    from shard.offers import OfferRegistry, ModelCohort, sign_offer
    from shard.resources import PlacementRequirements
    from shard.ring_pool import RingPool
    from shard.receipt import ReceiptSigner, verify_coverage
    from test_node_offers import offer, cohort
    from test_ring_pool import Backend

    now, registry, rings = time.time(), OfferRegistry(), RingPool()
    model = ModelCohort.from_dict(cohort())
    offers, keys, agents = [], {}, {}
    for i in range(4):
        key = Ed25519PrivateKey.generate()
        body = offer(key, gpu=f"GPU-{i}", now=now)
        host = "host-a" if i < 2 else "host-b"
        body.update(host_id=host, memory_domain_id=host)
        add_calibration(body, i * 3, (i + 1) * 3, stage=i, nstages=4, measured_at=now)
        body = sign_offer(body, key)
        registry.announce(body)
        offers.append(body); keys[body["node_id"]] = key
        ledger = LeaseLedger(tmp_path / f"{host}.sqlite", node_id=body["node_id"], authorize=lambda p, a, b: p)
        ledger.register_capacity(host, available_ram_bytes=2**32, pinnable_ram_bytes=2**31,
            gpu_capacity_bytes={f"GPU-{j}": 2**30 for j in range(4) if (j < 2) == (i < 2)})
        agents[body["node_id"]] = LocalLeaseClient(ledger, "operator")
    ids = [body["node_id"] for body in offers]
    edge_pairs = list(zip(ids, ids[1:])) + [("coordinator", ids[0]), (ids[-1], "coordinator")]
    edges, endpoints = [], {}
    for i, (a, b) in enumerate(edge_pairs):
        route, endpoint = f"route-{i}", f"127.0.0.1:{31000 + i}"
        endpoints[route] = endpoint
        edges.append(dict(src=a, dst=b, route_id=route, channel="loopback_tcp" if i in (0, 2) else "sidecar",
            src_endpoint="source", dst_endpoint="destination", dialer_id="coordinator" if b == "coordinator" else a,
            dial_endpoint=endpoint, reachable=True, latency_kind="one_way", hop_policy="measured_one_way",
            latency_ms=1, bandwidth_mbps=1000, measured_at=now, ttl_s=120))
    formation = dict(head=endpoints["route-3"], tail=endpoints["route-4"], route_endpoints=endpoints)
    observed = {}

    def requirements(stage, body):
        row = next(row for row in body["models"][0]["calibrations"]
            if row["requirements"]["provenance"]["runtime_config_sha256"] == stage["runtime_config_sha256"])
        return PlacementRequirements.from_dict(row["requirements"])

    def factory(plan, selected_cohort, contracts):
        assigned = [registry.get(stage["id"]) for stage in plan["stages"]]
        observed["assignment"] = pipeline_assignment(plan, selected_cohort, assigned, formation)
        backend = Backend(); backend.model_id = selected_cohort.model_id
        def warmup(timeout_s=300):
            nonce, value, receipts, expected = "fresh-host-group-warmup", b"cpu-fixture", [], {}
            for stage, body in zip(plan["stages"], assigned):
                output = hashlib.sha256(value + str(stage["lo"]).encode()).digest()
                signer = ReceiptSigner(keys[stage["id"]], plan["ring_id"], "warmup", stage["lo"], stage["hi"], nonce)
                signer.observe(value, output); receipts.append(signer.finalize())
                expected[body["public_key"]] = (stage["lo"], stage["hi"])
                value = output
            verify_coverage(receipts, 12, expected_by_signer=expected, expected_nonce=nonce, check_chain=True)
            backend.healthy = True
            observed["receipts"] = receipts
            return dict(proof_verified=True, committed_tokens=1)
        backend.warmup = warmup
        return backend

    controller = FormationController(registry, rings, agents, requirements=requirements, backend_factory=factory)
    try:
        result = controller.form("host-groups", model, {**profile(12), "require_exact_calibrations": True},
            locality={"mode": "global", "host_groups": policy()},
            measurements=dict(schema="shard-link-measurements/2", edges=edges), coordinator_id="coordinator")
        assert result["ready"] and len(result["plan"]["stages"]) == 4
        assert [g["node_ids"] for g in result["plan"]["planning"]["host_groups"]["groups"]] == [ids[:2], ids[2:]]
        assert len(controller._formations["host-groups"]["leases"]) == 4
        assert len({r["pubkey"] for r in observed["receipts"]}) == 4
        assigned = observed["assignment"]
        assert assigned["coordinator"]["tail"] == formation["tail"]
        assert assigned["stages"][-1]["node_id"] == ids[-1]
        assert [s["next_endpoint"] for s in assigned["stages"][:-1]] == [endpoints[f"route-{i}"] for i in range(3)]
    finally:
        controller.close()

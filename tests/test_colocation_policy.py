"""Mining permits co-location by default; explicit isolation does not duplicate hardware.

Pure planning tests: profile feasibility is not a throughput or hardware validation.
Host identity comes from announced host metadata, never inferred from public IP/NAT.
"""
from itertools import permutations
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from shard.plan import V4_ALL_RESIDENT_PROFILE, V4_DUAL_RESOURCE_PROFILE, plan_ring
from shard.scheduler import JoinedNode, Scheduler
from shard.topology import loop_cost, select_ring

ROOT = Path(__file__).resolve().parents[1]


def mesh(count, *, cap=1, layers=None, **kwargs):
    nodes = list(range(count))
    latency = [[0.0 if a == b else 1.0 for b in nodes] for a in nodes]
    options = dict(free_vram_mb={n: cap * 1000 for n in nodes},
                   layer_ms={n: 1.0 for n in nodes}, subnet={n: "same-network" for n in nodes},
                   n_layers=count if layers is None else layers, layer_vram_mb=1000,
                   slack=count, max_stages=max(6, count))
    options.update(kwargs)
    return nodes, latency, [1.0] * count, [1.0] * count, options


def select(count, *, cap=1, layers=None, **kwargs):
    nodes, latency, outgoing, incoming, options = mesh(count, cap=cap, layers=layers, **kwargs)
    return select_ring(nodes, latency, outgoing, incoming, **options)


def assert_tiling(result, layers):
    assert result is not None
    if "stages" in result:
        entries = [(s["lo"], s["hi"]) for s in result["stages"]]
    else:
        entries = [result["blocks"][n] for n in result["order"]]
    cursor = 0
    for lo, hi in entries:
        assert lo == cursor and hi > lo
        cursor = hi
    assert cursor == layers


@pytest.mark.parametrize("count", [4, 6])
def test_default_allows_all_gpus_on_one_host_and_subnet(count):
    result = select(count, host_id={n: "one-mining-host" for n in range(count)},
                    device_id={n: f"GPU-{n}" for n in range(count)})
    assert_tiling(result, count)
    assert set(result["order"]) == set(range(count))


def test_none_accepts_missing_identity_without_claiming_diversity():
    result = select(4, subnet={n: None for n in range(4)},
                    host_id={n: None for n in range(4)}, device_id={n: None for n in range(4)})
    assert_tiling(result, 4)


@pytest.mark.parametrize("policy", ["host", "subnet", "adjacent_host"])
def test_isolation_is_explicit_and_missing_required_metadata_fails_closed(policy):
    result = select(3, isolation=policy, subnet={n: None for n in range(3)},
                    host_id={n: None for n in range(3)})
    assert result is None


def test_subnet_and_host_are_different_opt_in_policies():
    hosts = {0: "physical-a", 1: "physical-b", 2: "physical-c"}
    assert_tiling(select(3, isolation="host", host_id=hosts), 3)
    assert select(3, isolation="subnet", host_id=hosts) is None
    # Different network labels do not turn one physical host into several hosts.
    assert select(3, isolation="host", host_id={n: "one-box" for n in range(3)},
                  subnet={n: f"network-{n}" for n in range(3)}) is None


def test_adjacent_host_searches_a_valid_order_instead_of_rejecting_the_cheapest():
    hosts = {0: "A", 1: "A", 2: "B", 3: "B"}
    nodes, latency, outgoing, incoming, options = mesh(4, require=0, host_id=hosts,
                                                     isolation="adjacent_host")
    for a in nodes:
        for b in nodes:
            if a != b:
                latency[a][b] = 50.0
    for a, b in ((0, 1), (1, 2), (2, 3)):
        latency[a][b] = 1.0
    outgoing[:] = [0.0, 100.0, 100.0, 100.0]
    incoming[:] = [100.0, 100.0, 100.0, 0.0]
    cheapest = [0, 1, 2, 3]  # A,A,B,B violates adjacency, but A,B,A,B is feasible.
    result = select_ring(nodes, latency, outgoing, incoming, **options)
    assert_tiling(result, 4)
    order = result["order"]
    assert order[0] == 0
    assert all(hosts[a] != hosts[b] for a, b in zip(order, order[1:] + order[:1]))
    valid = [[0, *tail] for tail in permutations((1, 2, 3))
             if all(hosts[a] != hosts[b] for a, b in
                    zip([0, *tail], [*tail, 0]))]
    assert loop_cost(order, latency, outgoing, incoming) == min(
        loop_cost(o, latency, outgoing, incoming) for o in valid)
    assert loop_cost(order, latency, outgoing, incoming) > loop_cost(
        cheapest, latency, outgoing, incoming)


def test_adjacent_host_includes_the_closing_edge():
    # Every linear A,B,A order appears safe, but its closing edge is A -> A.
    assert select(3, isolation="adjacent_host", host_id={0: "A", 1: "B", 2: "A"}) is None
    assert_tiling(select(3, isolation="none", host_id={0: "A", 1: "B", 2: "A"}), 3)


def test_host_policy_does_not_allow_nonadjacent_repeated_hosts():
    hosts = {0: "A", 1: "A", 2: "B", 3: "B"}
    assert_tiling(select(4, isolation="adjacent_host", host_id=hosts), 4)
    assert select(4, isolation="host", host_id=hosts) is None


@pytest.mark.parametrize("identities", [
    {0: "GPU-duplicate", 1: "GPU-duplicate"},
    {0: ("GPU-A", "GPU-B"), 1: ("GPU-B", "GPU-C")},
])
def test_one_physical_gpu_is_never_credited_twice(identities):
    assert select(2, isolation="none", host_id={0: "box", 1: "box"},
                  device_id=identities) is None


def test_distinct_multi_gpu_device_sets_can_share_a_host():
    result = select(2, host_id={0: "box", 1: "box"},
                    device_id={0: ("GPU-A", "GPU-B"), 1: ("GPU-C", "GPU-D")})
    assert_tiling(result, 2)


def test_duplicate_gpu_candidates_do_not_hide_a_distinct_alternative():
    result = select(3, layers=2, device_id={0: "GPU-A", 1: "GPU-A", 2: "GPU-B"})
    assert_tiling(result, 2)
    assert 2 in result["order"]
    assert len(set({0: "GPU-A", 1: "GPU-A", 2: "GPU-B"}[n] for n in result["order"])) == 2


def test_known_host_ram_capacity_is_shared_between_gpu_nodes():
    assert select(2, cap=4, layers=6, host_id={0: "box", 1: "box"},
                  host_layer_caps={"box": 5}, isolation="none") is None


def test_shared_host_budget_finds_an_alternative_layer_assignment():
    # Greedy unconstrained fill produces A=4, B=1. A=3, B=2 is valid and must be found.
    result = select(3, cap=2, layers=5, host_id={0: "A", 1: "A", 2: "B"},
                    host_layer_caps={"A": 3, "B": 2},
                    layer_ms={0: 0.1, 1: 0.1, 2: 10.0}, isolation="none")
    assert_tiling(result, 5)
    assert set(result["order"]) == {0, 1, 2}
    assert result["layers"][0] + result["layers"][1] == 3
    assert result["layers"][2] == 2


def test_tail_host_budget_searches_a_more_expensive_valid_order():
    nodes, _, outgoing, incoming, options = mesh(
        3, cap=2, layers=6, require=0, slack=0,
        host_id={0: "A", 1: "A", 2: "B"},
        host_layer_caps={"A": 4, "B": 2}, tail_host_layer_caps={"A": 3, "B": 2})
    latency = [[0.0, 10.0, 1.0], [1.0, 0.0, 10.0], [1.0, 1.0, 0.0]]
    # [0,2,1] is cheapest but tails host A, which then has room for only 3
    # layers. All three GPUs must carry 2, so [0,1,2] is the feasible order.
    result = select_ring(nodes, latency, outgoing, incoming, **options)
    assert_tiling(result, 6)
    assert result["order"] == [0, 1, 2]
    assert result["layers"] == {0: 2, 1: 2, 2: 2}


def test_saturated_host_cover_does_not_overestimate_minimum_ring_width():
    result = select(3, cap=4, layers=8, require=0, slack=0,
                    host_id={0: "A", 1: "A", 2: "B"},
                    host_layer_caps={"A": 6, "B": 4})
    # Walking static caps in node order sees A=4, then A=6, then B and
    # mistakenly calls three stages the minimum. GPU 0 + GPU 2 already fit 8.
    assert_tiling(result, 8)
    assert result["k"] == 2
    assert result["order"] == [0, 2]
    assert result["layers"] == {0: 4, 2: 4}


def test_co_location_does_not_relax_trusted_boundaries_pinned_head_or_tail_floor():
    result = select(4, cap=2, layers=8, host_id={n: "box" for n in range(4)},
                    trusted={0, 3}, boundary_in=2, boundary_out=2, tail_floor=2,
                    require=0, isolation="none")
    assert_tiling(result, 8)
    assert result["order"][0] == 0 and result["order"][-1] == 3
    assert set(result["boundary"]) <= {0, 3}
    assert result["layers"][result["order"][-1]] >= 2
    assert select(4, cap=2, layers=8, trusted={1, 3}, boundary_in=2,
                  boundary_out=2, require=0, isolation="none") is None


def test_tail_floor_remains_binding_when_every_node_is_colocated():
    assert select(4, cap=1, layers=4, host_id={n: "box" for n in range(4)},
                  tail_floor=2, isolation="none") is None


def test_latency_funnel_keeps_a_colocated_capacity_cover():
    result = select(14, cap=1, layers=14, host_id={n: "box" for n in range(14)},
                    device_id={n: f"GPU-{n}" for n in range(14)}, slack=0,
                    max_stages=16, isolation="none")
    assert_tiling(result, 14)
    assert len(result["order"]) == 14


def profile(layers=6, *, ram=False):
    model = {"n_layers": layers, "layer_vram_mb": 1024.0, "kv_mb_per_layer": 0.0,
             "layer_ms_base": 1.0, "reserve_mb": 0.0, "head_reserve_mb": 0.0,
             "tail_reserve_mb": 0.0, "cap_layers": 4, "head_layer_ms_mult": 1.0,
             "tail_floor": 0, "max_stages": 6, "placement": "ram" if ram else "gpu"}
    if ram:
        model["layer_host_ram_mb"] = 1024.0
    return model


def pool(count, *, host="one-host", ram_mb=192 * 1024, gpu_mb=32 * 1024):
    nodes = [{"id": f"node-{n}", "subnet": "one-subnet", "host_id": host,
              "gpu_uuid": f"GPU-{n}", "public_ip": "203.0.113.10",
              "free_vram_mb": gpu_mb, "free_ram_mb": ram_mb,
              "pinnable_ram_mb": ram_mb, "h2d_gbps": 24.0} for n in range(count)]
    latency = [[0.0 if a == b else 1.0 for b in range(count)] for a in range(count)]
    return nodes, latency


@pytest.mark.parametrize("count,model", [(4, V4_DUAL_RESOURCE_PROFILE),
                                         (6, V4_ALL_RESIDENT_PROFILE),
                                         (6, V4_DUAL_RESOURCE_PROFILE)])
def test_v4_5090_profile_can_plan_one_box_of_distinct_gpus(count, model):
    nodes, latency = pool(count)
    result = plan_ring(nodes, latency, model=model)
    assert_tiling(result, 43)
    assert result["k"] <= count


def test_same_nat_ip_does_not_define_physical_host_identity():
    nodes, latency = pool(2, gpu_mb=4 * 1024)
    nodes[0]["host_id"], nodes[1]["host_id"] = "host-A", "host-B"
    assert_tiling(plan_ring(nodes, latency, model=profile(), isolation="host"), 6)


def test_public_ip_cannot_supply_missing_host_metadata_for_strict_policy():
    nodes, latency = pool(2, gpu_mb=4 * 1024)
    for n, node in enumerate(nodes):
        node.pop("host_id")
        node["public_ip"] = f"203.0.113.{n + 1}"
    assert_tiling(plan_ring(nodes, latency, model=profile(), isolation="none"), 6)
    assert plan_ring(nodes, latency, model=profile(), isolation="host") is None


def test_nat_ip_is_not_used_to_merge_unknown_physical_ram_pools():
    nodes, latency = pool(2, gpu_mb=4 * 1024, ram_mb=5 * 1024)
    for node in nodes:
        node.pop("host_id")
    # A shared public/NAT address is not evidence these miners share system RAM.
    assert_tiling(plan_ring(nodes, latency, model=profile(ram=True), isolation="none"), 6)


def test_profile_policy_is_opt_in_and_explicit_argument_wins():
    nodes, latency = pool(2, gpu_mb=4 * 1024)
    assert_tiling(plan_ring(nodes, latency, model=profile()), 6)
    strict = dict(profile(), isolation="subnet")
    assert plan_ring(nodes, latency, model=strict) is None
    assert_tiling(plan_ring(nodes, latency, model=strict, isolation="none"), 6)


def test_plan_ring_never_allocates_each_gpu_the_whole_host_ram_pool():
    nodes, latency = pool(2, gpu_mb=4 * 1024, ram_mb=5 * 1024)
    assert plan_ring(nodes, latency, model=profile(ram=True)) is None
    nodes[1]["host_id"] = "other-host"
    assert_tiling(plan_ring(nodes, latency, model=profile(ram=True)), 6)


def test_shared_host_ram_uses_the_lower_known_free_and_pinnable_budget():
    nodes, latency = pool(2, gpu_mb=4 * 1024, ram_mb=8 * 1024)
    nodes[1]["pinnable_ram_mb"] = 5 * 1024  # conservative shared-host report
    assert plan_ring(nodes, latency, model=profile(ram=True)) is None


def test_strict_host_policy_chooses_a_known_host_head_when_the_central_node_is_unknown():
    nodes, _ = pool(3, gpu_mb=4 * 1024, ram_mb=5 * 1024)
    nodes[0].pop("host_id")
    nodes[1]["host_id"], nodes[2]["host_id"] = "A", "B"
    # Node 0 is the RTT center, but cannot be the mandatory head of a strict
    # host-isolated ring. The two identified hosts can still cover all layers.
    latency = [[0.0, 1.0, 1.0], [1.0, 0.0, 10.0], [1.0, 10.0, 0.0]]
    result = plan_ring(nodes, latency, model=profile(ram=True), isolation="host")
    assert_tiling(result, 6)
    assert result["head"] != "node-0"
    assert set(result["order"]) == {"node-1", "node-2"}


@pytest.mark.parametrize("missing", ["free_ram_mb", "pinnable_ram_mb"])
def test_explicit_gpu_layer_cap_cannot_bypass_unknown_ram_or_pinning(missing):
    nodes, latency = pool(2, gpu_mb=4 * 1024, ram_mb=5 * 1024)
    for index, node in enumerate(nodes):
        node["host_id"] = f"host-{index}"
        node["cap_layers"] = 99  # an announced GPU ceiling is not host-memory evidence
        node.pop(missing)
    assert plan_ring(nodes, latency, model=profile(ram=True)) is None


def test_nonfinite_pinnable_ram_report_is_rejected():
    nodes, latency = pool(2, gpu_mb=4 * 1024, ram_mb=5 * 1024)
    nodes[0]["pinnable_ram_mb"] = float("nan")
    with pytest.raises(ValueError):
        plan_ring(nodes, latency, model=profile(ram=True))


def test_plan_ring_detects_overlapping_aggregate_gpu_uuids():
    nodes, latency = pool(2, gpu_mb=4 * 1024)
    for node in nodes:
        node.pop("gpu_uuid")
    nodes[0]["gpu_uuids"] = ["GPU-A", "GPU-B"]
    nodes[1]["gpu_uuids"] = ["GPU-B", "GPU-C"]
    assert plan_ring(nodes, latency, model=profile()) is None
    nodes[1]["gpu_uuids"] = ["GPU-C", "GPU-D"]
    assert_tiling(plan_ring(nodes, latency, model=profile()), 6)


def test_json_cli_forwards_policy_and_default_permits_colocation():
    nodes, latency = pool(2, gpu_mb=4 * 1024)
    request = {"nodes": nodes, "rtt": latency, "model": profile()}
    def invoke(extra):
        completed = subprocess.run([sys.executable, "-m", "shard.plan"],
                                   input=json.dumps({**request, **extra}), text=True,
                                   capture_output=True, cwd=ROOT, timeout=15)
        assert completed.returncode == 0, completed.stderr + completed.stdout
        return json.loads(completed.stdout)
    assert_tiling(invoke({}), 6)
    assert invoke({"isolation": "host"}) is None
    assert_tiling(invoke({"model": dict(profile(), isolation="host"), "isolation": "none"}), 6)


def test_scheduler_facade_keeps_announced_host_and_gpu_identity():
    scheduler = Scheduler("toy", 6)
    for n in range(2):
        scheduler.register(JoinedNode(f"node-{n}", 4.0,
                           {f"node-{1 - n}": 1.0}, host_id="box", gpu_uuid=f"GPU-{n}",
                           subnet="same-subnet", public_ip="203.0.113.10"))
    assert_tiling(scheduler.plan(gb_per_layer=1.0, headroom_gb=0.0, boundary_gb=0.0), 6)
    with pytest.raises(ValueError):
        scheduler.plan(gb_per_layer=1.0, headroom_gb=0.0, boundary_gb=0.0, isolation="host")
    scheduler.nodes["node-1"].gpu_uuid = "GPU-0"
    with pytest.raises(ValueError):
        scheduler.plan(gb_per_layer=1.0, headroom_gb=0.0, boundary_gb=0.0, isolation="none")


def test_scheduler_ram_facade_aggregates_known_host_budget():
    scheduler = Scheduler("v4", 43)
    for n in range(4):
        scheduler.register(JoinedNode(f"node-{n}", 32.0,
                           {f"node-{j}": 1.0 for j in range(4) if j != n},
                           ram_gb=64.0, pinnable_ram_gb=60.0, host_id="one-host",
                           gpu_uuid=f"GPU-{n}", subnet="one-subnet"))
    with pytest.raises(ValueError):
        scheduler.plan(placement="ram")  # 60 GiB shared is not four 60 GiB pools.
    for node in scheduler.nodes.values():
        node.ram_gb = node.pinnable_ram_gb = 192.0
    assert_tiling(scheduler.plan(placement="ram"), 43)


@pytest.mark.parametrize("policy", ["typo", "gpu", True])
def test_unknown_policy_is_rejected_instead_of_silently_restricting_mining(policy):
    with pytest.raises(ValueError, match="isolation"):
        select(2, isolation=policy)


def _pipe_launch(monkeypatch):
    # Import on use: collecting pure planner tests must not load torch or an engine.
    pytest.importorskip("torch")
    import v4_pipe
    monkeypatch.delenv("V4_CUDA_GRAPH", raising=False)
    return v4_pipe


def _launch_key(command):
    assignments = [part for part in shlex.split(command) if part.startswith("SHARD_NODE_KEY=")]
    assert len(assignments) <= 1, "a signer path must not be shadowed by another assignment"
    return assignments[0].split("=", 1)[1] if assignments else None


@pytest.mark.parametrize("gpus", [2, 4, 6])
def test_multigpu_box_launch_has_distinct_stable_receipt_signer_paths(monkeypatch, gpus):
    pipe = _pipe_launch(monkeypatch)
    boxes = [{"id": "mining-box", "lo": 0, "hi": 2 * gpus}]
    first = pipe.box_ring_launch(boxes, gpus=gpus, receipts=True)
    again = pipe.box_ring_launch(boxes, gpus=gpus, receipts=True)
    assert len(first["stages"]) == gpus
    paths = [stage["key_path"] for stage in first["stages"]]
    assert all(paths) and len(set(paths)) == gpus
    assert paths == [stage["key_path"] for stage in again["stages"]]
    for stage in first["stages"]:
        assert _launch_key(stage["cmd"]) == stage["key_path"]
        assert "SHARD_RECEIPTS=1" in shlex.split(stage["cmd"])
        assert f"CUDA_VISIBLE_DEVICES={stage['gpu']}" in shlex.split(stage["cmd"])


def test_multi_box_signer_paths_are_unique_within_each_host(monkeypatch):
    pipe = _pipe_launch(monkeypatch)
    boxes = [{"id": "box-a", "lo": 0, "hi": 4}, {"id": "box-b", "lo": 4, "hi": 8}]
    launched = pipe.box_ring_launch(boxes, gpus={"box-a": 2, "box-b": 2}, receipts=True)
    for host in ("box-a", "box-b"):
        stages = [stage for stage in launched["stages"] if stage["id"] == host]
        assert len(stages) == 2
        assert len({stage["key_path"] for stage in stages}) == 2
        assert all(_launch_key(stage["cmd"]) == stage["key_path"] for stage in stages)
    # Equal path strings on different hosts refer to independent host filesystems.
    assert {stage["key_path"] for stage in launched["stages"] if stage["id"] == "box-a"} == {
        stage["key_path"] for stage in launched["stages"] if stage["id"] == "box-b"}


def test_single_gpu_boxes_preserve_the_default_signing_key_path(monkeypatch):
    pipe = _pipe_launch(monkeypatch)
    boxes = [{"id": "box-a", "lo": 0, "hi": 4}, {"id": "box-b", "lo": 4, "hi": 8}]
    launched = pipe.box_ring_launch(boxes, gpus=1, receipts=True)
    for stage in launched["stages"]:
        assert stage["key_path"] is None
        assert _launch_key(stage["cmd"]) is None
        assert "SHARD_RECEIPTS=1" in shlex.split(stage["cmd"])
    assert _launch_key(pipe.stage_launch_cmd(0, 2, 0, 4, receipts=True)) is None


def test_multigpu_launch_without_receipts_does_not_override_signing_key(monkeypatch):
    pipe = _pipe_launch(monkeypatch)
    stages = pipe.box_ring_launch([{"id": "box", "lo": 0, "hi": 8}],
                                 gpus=4, receipts=False)["stages"]
    assert all(stage["key_path"] is None and _launch_key(stage["cmd"]) is None for stage in stages)


@pytest.mark.parametrize("key_path", [
    "/root/miner keys/node signing key",
    "/root/node's key; $(touch /tmp/unwanted) `echo unwanted` & key",
    '/root/key "double quotes" $HOME \\ escaped',
])
def test_explicit_signer_key_path_is_one_shell_quoted_environment_value(monkeypatch, key_path):
    pipe = _pipe_launch(monkeypatch)
    command = pipe.stage_launch_cmd(0, 2, 0, 4, receipts=True, key_path=key_path)
    assert _launch_key(command) == key_path
    # Checking the literal single-quote escaping matters: a double-quoted $(...)
    # also looks like one token to shlex, but bash would execute the substitution.
    assert f"SHARD_NODE_KEY={shlex.quote(key_path)} " in command
    words = shlex.split(command)
    assert words.count("setsid") == words.count("bash") == words.count("-c") == 1
    inner = words[words.index("-c") + 1]
    assert inner.startswith("python3 /root/v4_pipe.py stage ")
    assert key_path not in inner  # it is inherited as data, never interpolated into the inner command

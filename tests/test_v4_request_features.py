"""Calibrated policy admission is pure metadata logic; no GPU/model claims."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from engines.deepseek_v4.v4_request_features import RequestFeatureError, build_policy, capabilities_digest, validate_selected_bindings
from shard.speculation_policy import SpeculationRecipe


COHORT = "a" * 64
BASELINE = SpeculationRecipe("original", "pipelined", 16, 1, True)


def node_configs():
    args = {"n_layers": 43, "max_seq_len": 8192, "dspark_block_size": 8, "n_mtp_layers": 3}
    return {
        "node-a": {"lo": 0, "hi": 21, "head": True, "tail": False, "dspark": False,
            "spec_depth": 16, "dspark_loaded": False, "dspark_block_size": None,
            "args": dict(args), "environment": {"V4_SPEC_DEPTH": "128"}},
        "node-b": {"lo": 21, "hi": 43, "head": False, "tail": True, "dspark": True,
            "spec_depth": 16, "dspark_loaded": True, "dspark_block_size": 8,
            "args": dict(args), "environment": {"V4_SPEC_DEPTH": "128", "V4_DSPARK_BLOCK": "8"}},
    }


def fences():
    return {"node-a": {"lease_id": "lease-a", "fencing_token": 3},
            "node-b": {"lease_id": "lease-b", "fencing_token": 9}}


def config():
    return {"enabled": True, "baseline_id": "baseline", "recipes": [
        {"recipe_id": "baseline", "mode": "pipelined", "depth": 16, "floor": 1, "lazy": True},
        {"recipe_id": "greedy", "mode": "greedy"},
        {"recipe_id": "p8f1", "mode": "pipelined", "depth": 8, "floor": 1, "lazy": True},
        {"recipe_id": "p9f2", "mode": "pipelined", "depth": 9, "floor": 2, "lazy": True},
    ]}


def build(cfg=None, nodes=None, leases=None, baseline=BASELINE):
    return build_policy(cfg if cfg is not None else config(), baseline, nodes if nodes is not None else node_configs(), COHORT,
        lease_fences=leases if leases is not None else fences(), ring_id="ring-1", clock=lambda: 100.0)


def selected_plan():
    import hashlib
    return {"n_layers": 43, "ring_id": "ring-1", "cohort_id": COHORT, "stages": [
        dict(node_id=node, runtime_config_sha256=hashlib.sha256(json.dumps(cfg,
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            **{key: cfg[key] for key in ("lo", "hi", "head", "tail")})
        for node, cfg in node_configs().items()]}


@pytest.mark.parametrize("mismatch", ["nodes", "leases", "span", "role", "depth", "hash", "missing_hash"])
def test_feature_admission_rejects_another_selected_ring_or_assignment(mismatch):
    plan, nodes, leases = selected_plan(), node_configs(), fences()
    if mismatch == "nodes":
        nodes["other-node"] = nodes.pop("node-a")
    elif mismatch == "leases":
        leases["other-node"] = leases.pop("node-a")
    elif mismatch == "span":
        plan["stages"][0]["hi"] -= 1
    elif mismatch == "role":
        plan["stages"][0]["head"] = False
    elif mismatch == "depth":
        plan["n_layers"] = 44
    elif mismatch == "hash":
        plan["stages"][0]["runtime_config_sha256"] = "0" * 64
    else:
        plan["stages"][0].pop("runtime_config_sha256")
    with pytest.raises(RequestFeatureError):
        validate_selected_bindings(plan, nodes, leases)


def test_feature_admission_accepts_the_actual_selected_calibration_hashes():
    import hashlib
    plan, nodes = selected_plan(), node_configs()
    for stage in plan["stages"]:
        stage["runtime_config_sha256"] = hashlib.sha256(json.dumps(nodes[stage["node_id"]],
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    validate_selected_bindings(plan, nodes, fences())


def test_lookahead_is_bounded_by_loaded_block_and_window_not_refill_floor():
    nodes = node_configs()
    for cfg in nodes.values():
        cfg["spec_depth"] = 256
    base = SpeculationRecipe("original", "pipelined", 256, 256, True)
    cfg = {"enabled": True, "baseline_id": "wide", "recipes": [
        {"recipe_id": "wide", "mode": "pipelined", "depth": 256, "floor": 256, "lazy": True}]}
    assert build(cfg, nodes, baseline=base)  # B8 still limits actual lookahead to 8.
    nodes["node-b"]["dspark_block_size"] = 128
    with pytest.raises(RequestFeatureError, match="lookahead.*context margin"):
        build(cfg, nodes, baseline=base)
    cfg["recipes"][0].update(depth=9, floor=1)
    assert build(cfg, nodes, baseline=SpeculationRecipe("original", "pipelined", 9, 1, True))


def test_missing_or_disabled_policy_is_exactly_the_previous_backend_path():
    # No capability guesses, no imports of a model and no even-disabled DSpark
    # baseline validation; legacy serial mode remains unchanged when disabled.
    for cfg in (None, {}, {"enabled": False}, {"enabled": False, "recipes": "ignored disabled configuration"}):
        assert build_policy(cfg, {"mode": "dspark"}, None, None) is None


def test_exact_baseline_and_approved_request_choices_preserve_loaded_and_trained_widths():
    nodes = node_configs(); before = deepcopy(nodes)
    nodes["node-b"]["args"]["dspark_block_size"] = 5  # already-calibrated B8 override of trained B5
    p = build(nodes=nodes)
    ctx = p.context(tenant_id="authenticated-tenant", context_tokens=512, max_new_tokens=512, workload="code")
    decision = p.choose(ctx)
    assert decision.recipe.to_dict() == {**BASELINE.to_dict(), "recipe_id": "baseline"}
    assert p.binding()["loaded_block_size"] == 8 and p.binding()["trained_block_size"] == 5
    assert before["node-b"]["args"]["dspark_block_size"] == 8
    assert nodes["node-b"]["args"]["dspark_block_size"] == 5
    assert p.binding()["rollback_capacity"] == 16


@pytest.mark.parametrize("field,changed", [("mode", "greedy"), ("depth", 9), ("floor", 2), ("lazy", False)])
def test_baseline_cannot_silently_replace_backend_original_configuration(field, changed):
    cfg = config(); cfg["recipes"][0][field] = changed
    with pytest.raises(RequestFeatureError):
        build(cfg)


@pytest.mark.parametrize("node,field", [("node-a", "spec_depth"), ("node-b", "dspark_loaded"), ("node-b", "dspark_block_size")])
def test_declared_environment_and_draft_flag_cannot_fill_missing_actual_calibration(node, field):
    nodes = node_configs(); nodes[node].pop(field)
    with pytest.raises(RequestFeatureError):
        build(nodes=nodes)


def test_all_stages_rollback_capacity_applies_even_if_tail_or_environment_allows_more():
    nodes = node_configs(); nodes["node-a"]["spec_depth"] = 8
    with pytest.raises(RequestFeatureError, match="actual rollback capacity"):
        build(nodes=nodes)
    cfg = config(); cfg["recipes"] = [cfg["recipes"][2]]; cfg["baseline_id"] = "p8f1"
    assert build(cfg, nodes, baseline=SpeculationRecipe("original", "pipelined", 8, 1, True))


def test_draft_must_be_loaded_and_only_on_actual_tail():
    nodes = node_configs(); nodes["node-b"].update(dspark_loaded=False, dspark_block_size=None)
    with pytest.raises(RequestFeatureError, match="loaded tail MTP"):
        build(nodes=nodes)
    nodes = node_configs(); nodes["node-a"].update(dspark=True, dspark_loaded=True, dspark_block_size=8)
    with pytest.raises(RequestFeatureError, match="calibrated tail"):
        build(nodes=nodes)


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(recipes=[]),
    lambda x: x.update(recipes=x["recipes"]*5),
    lambda x: x.update(enabled="1"),
    lambda x: x.update(V4_DSPARK_BLOCK=16),
    lambda x: x["recipes"][2].update(V4_DSPARK_BLOCK=16),
    lambda x: x["recipes"][2].update(block_size=16),
    lambda x: x.update(learning={"approved_recipes": []}),
    lambda x: x.update(baseline_id=[]),
])
def test_illegal_policy_fields_and_unbounded_or_model_mutating_recipes_are_rejected(mutation):
    cfg = config(); mutation(cfg)
    with pytest.raises(RequestFeatureError):
        build(cfg)


def test_up_to_sixteen_existing_request_recipes_are_supported_without_model_changes():
    cfg = {"enabled": True, "recipes": [
        {"recipe_id": f"w{w}", "mode": "pipelined", "depth": w, "floor": 1, "lazy": True} for w in range(2, 18)],
        "baseline_id": "w16"}
    nodes = node_configs()
    for node in nodes.values():
        node["spec_depth"] = 17
    p = build(cfg, nodes)
    assert len(p.recipes) == 16
    cfg["recipes"].append({"recipe_id": "greedy", "mode": "greedy"})
    with pytest.raises(RequestFeatureError, match="sixteen"):
        build(cfg, nodes)


@pytest.mark.parametrize("mutation", [
    lambda x: x.pop("node-a"),
    lambda x: x["node-a"].update(fencing_token=0),
    lambda x: x["node-a"].update(fencing_token=True),
    lambda x: x["node-a"].update(lease_id=""),
    lambda x: x["node-a"].update(confirm_verified=True),
])
def test_lease_generation_requires_exact_local_verified_node_bindings(mutation):
    leases = fences(); mutation(leases)
    with pytest.raises(RequestFeatureError):
        build(leases=leases)


def test_enabled_policy_cannot_be_constructed_from_missing_calibration_or_lease_inputs():
    for nodes, leases in ((None, fences()), (node_configs(), None)):
        with pytest.raises(RequestFeatureError):
            build_policy(config(), BASELINE, nodes, COHORT, lease_fences=leases, ring_id="ring-1")


@pytest.mark.parametrize("mutation", [
    lambda x: x["node-a"].update(hi=20),
    lambda x: x["node-b"].update(lo=20),
    lambda x: x["node-b"].update(tail=False),
    lambda x: x["node-a"]["args"].update(n_layers=42),
    lambda x: x["node-a"].update(cohort_id="e"*64),
    lambda x: x["node-a"].update(spec_depth=float("nan")),
    lambda x: x["node-a"].update(source=float("nan")),
])
def test_configuration_role_geometry_cohort_and_nonfinite_inputs_fail_closed(mutation):
    nodes = node_configs(); mutation(nodes)
    with pytest.raises(RequestFeatureError):
        build(nodes=nodes)


def test_full_configuration_and_lease_change_digest_and_policy_scope_are_bound_without_secrets():
    nodes = node_configs(); leases = fences()
    nodes["node-a"]["environment"]["V4_DIR"] = "/private/model/path"
    p = build(nodes=nodes, leases=leases)
    assert p.binding()["capabilities_sha256"] == capabilities_digest(nodes, COHORT, lease_fences=leases, ring_id="ring-1")
    raw = json.dumps(p.configuration())
    assert "/private/model/path" not in raw and "lease-a" not in raw
    nodes["node-a"]["environment"]["V4_DIR"] = "/another/location"
    assert capabilities_digest(nodes, COHORT, lease_fences=leases, ring_id="ring-1") != p.binding()["capabilities_sha256"]
    leases["node-a"]["fencing_token"] += 1
    new = build(nodes=nodes, leases=leases)
    assert new.binding()["ring_generation"] != p.binding()["ring_generation"]
    original_ctx = p.context(tenant_id="trusted-tenant", context_tokens=512, max_new_tokens=512)
    for field, value in (("cohort_id", "b"*64), ("ring_id", "another"), ("ring_generation", "another"), ("runtime_config_sha256", "c"*64)):
        with pytest.raises(RequestFeatureError, match="scope differs"):
            p.choose(replace(original_ctx, **{field: value}))


def test_capability_snapshot_and_public_binding_cannot_be_mutated_by_callers():
    nodes = node_configs(); leases = fences(); p = build(nodes=nodes, leases=leases)
    snapshot = p.binding()
    nodes["node-a"]["spec_depth"] = 0; leases["node-a"]["fencing_token"] = 99
    assert p.binding() == snapshot
    snapshot["rollback_capacity"] = 999
    assert p.binding()["rollback_capacity"] == 16


def test_helper_import_does_not_load_torch_or_the_engine():
    root = Path(__file__).resolve().parents[1]
    code = "import sys; import engines.deepseek_v4.v4_request_features; assert 'torch' not in sys.modules; assert 'v4_pipe' not in sys.modules"
    proc = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_lease_ids_are_node_local_namespaces_but_full_node_fences_still_bind_generation():
    leases = fences(); leases["node-a"]["lease_id"] = leases["node-b"]["lease_id"]
    assert build(leases=leases)


def test_near_context_boundary_only_approved_greedy_is_selected_without_training():
    p = build()
    ctx = p.context(tenant_id="trusted-tenant", context_tokens=8100, max_new_tokens=64)
    selected = p.choose(ctx)
    assert selected.recipe.mode == "greedy" and selected.reason == "context_capacity_greedy"
    assert not selected._learning and not p._states
    with pytest.raises(RequestFeatureError, match="context capacity"):
        p.choose(p.context(tenant_id="trusted-tenant", context_tokens=8192, max_new_tokens=1))
    cfg = config(); cfg["recipes"] = [cfg["recipes"][0]]
    no_greedy = build(cfg)
    with pytest.raises(RequestFeatureError, match="no greedy recipe"):
        no_greedy.choose(no_greedy.context(tenant_id="trusted-tenant", context_tokens=8100, max_new_tokens=64))

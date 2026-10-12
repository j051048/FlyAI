import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from engines.deepseek_v4 import v4_execution_profiles as P


def test_named_whole_is_same_frozen_baseline_and_budget_overrides_preserved():
    from phase0.v4_benchmark import DEFAULT_ENV
    assert P.baseline_environment() == DEFAULT_ENV
    environment = P.resolve_profile("resident-whole", {"V4_MAX_SEQ": "4096", "V4_GRAPH_MAX": "100"})
    assert environment["V4_CUDA_GRAPH"] == "whole"
    assert environment["V4_MOE_IN_GRAPH"] == "1"
    assert environment["V4_REF_SLIM"] == "0"
    assert environment["V4_MAX_SEQ"] == "4096"
    assert P.resolve_profile("ram-experts")["V4_MOE_IN_GRAPH"] == "0"


@pytest.mark.parametrize("overrides", [{"V4_REF_SLIM": "1"}, {"V4_FAST_VERIFY": "1"},
    {"V4_CUDA_GRAPH": "island"}, {"V4_EXPERT_PLACEMENT": "ram"}, {"V4_PREFETCH": "1"},
    {"V4_CHUNK_PREFILL": "1"}, {"V4_UNKNOWN": "1"}, {"V4_EXECUTION_PROFILE": "resident-island"}])
def test_named_profile_refuses_changed_math_unknown_names_and_placement(overrides):
    with pytest.raises(ValueError):
        P.resolve_profile("resident-whole", overrides)


def test_profile_expansion_retains_private_non_v4_launch_environment():
    expanded = P.expanded_environment({"V4_EXECUTION_PROFILE": "resident-island", "SHARD_NODE_KEY": "private/path"})
    assert expanded["SHARD_NODE_KEY"] == "private/path"
    assert expanded["V4_CUDA_GRAPH"] == "island"


def test_fresh_stage_import_expands_recipe_before_stage_constants():
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("V4_")}
    env.update(V4_EXECUTION_PROFILE="resident-whole", V4_KERNELS="cpu", PYTHONUTF8="1")
    code = "import sys; sys.path.insert(0,'engines/deepseek_v4'); import v4_stage as V; print(V.V4_CUDA_GRAPH, V.V4_EXPERT_PLACEMENT)"
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("whole gpu")


def receipt():
    return {"layer_start": 0, "layer_end": 2, "runtime_observation": {"node_id": "worker", "graph_mode": "whole",
        "optimizations": {"execution_profile": {"name": "resident-whole", "graph_mode": "whole", "coverage": {
            "schema": "flyai-v4-whole-coverage/1", "captured_layers": 2, "refused_layers": 0, "undecided_layers": 0,
            "layers": [{"layer": index, "captured_graphs": 1, "eager": False, "moe_in_graph": True} for index in range(2)]}}}}}


def test_warmup_checks_actual_assigned_capture_not_requested_flag():
    row = receipt()
    assert P.require_profile_warmup([row], {"worker": "resident-whole"}) == {"worker": "resident-whole"}
    for change in ("missing", "armed", "fallback", "duplicate"):
        broken = copy.deepcopy(row)
        coverage = broken["runtime_observation"]["optimizations"]["execution_profile"]["coverage"]
        if change == "missing":
            broken["runtime_observation"]["optimizations"] = {}
        elif change == "armed": coverage["captured_layers"] = 0
        elif change == "fallback": coverage["layers"][0]["eager"] = True
        else: coverage["layers"][1]["layer"] = 0
        with pytest.raises(ValueError):
            P.require_profile_warmup([broken], {"worker": "resident-whole"})
    with pytest.raises(ValueError, match="lacks"):
        P.require_profile_warmup([], {"worker": "resident-whole"})


def test_receive_reservation_counts_pin_pool_once_and_rejects_unbounded():
    assert P.transport_buffer_reservation({"V4_TRANSPORT_BUFFER_MIB": "8", "V4_TRANSPORT_PINNED": "1"}) == {
        "host_reserved_bytes": 8 << 20, "pinned_reserved_bytes": 8 << 20, "slots": 2,
        "scope": "process-owned transport pool; per-stage process calibration"}
    with pytest.raises(ValueError):
        P.transport_buffer_reservation({"V4_TRANSPORT_BUFFER_MIB": "513"})


def test_preflight_named_profile_expands_before_torch_import():
    root = Path(__file__).resolve().parents[1]
    code = "import sys,os;sys.path.insert(0,'engines/deepseek_v4');from v4_preflight import apply_environment;apply_environment({'V4_EXECUTION_PROFILE':'ram-experts'});print(os.environ['V4_MOE_IN_GRAPH'], 'torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0 False"


@pytest.mark.parametrize("module", ["v4_pipe", "v4_dspark_draft", "v4_conversation_protocol"])
def test_complete_late_environment_cannot_bless_stale_imported_constants(module):
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("V4_")}
    env.update(V4_KERNELS="cpu", PYTHONUTF8="1")
    code = ("import sys,os;sys.path.insert(0,'engines/deepseek_v4');"
            f"import {module};from v4_execution_profiles import resolve_profile,apply_profile;"
            "os.environ.update(resolve_profile('resident-whole',{'V4_KERNELS':'cpu'}));apply_profile()")
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "cannot change after model modules" in result.stderr


def test_profile_expansion_does_not_freeze_lever_strict_flag_too_early():
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("V4_")}
    env.update(V4_EXECUTION_PROFILE="resident-whole", V4_KERNELS="cpu", PYTHONUTF8="1")
    result = subprocess.run([sys.executable, "-c", "import sys;sys.path.insert(0,'engines/deepseek_v4');import v4_pipe,v4_levers;print(v4_levers.V4_LEVERS_STRICT)"],
        cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("True")


@pytest.mark.parametrize("profile", ["resident-island", "ram-experts"])
def test_other_named_profiles_cannot_bless_armed_or_fallback_layers(profile):
    row = receipt()
    status = row["runtime_observation"]["optimizations"]["execution_profile"]
    status["name"] = profile
    if profile == "resident-island":
        row["runtime_observation"]["graph_mode"] = status["graph_mode"] = "island"
        status["coverage"]["schema"] = "flyai-v4-island-coverage/1"
        for layer in status["coverage"]["layers"]: layer["captured_graphs"] = 3
    else:
        for layer in status["coverage"]["layers"]: layer["moe_in_graph"] = False
    P.require_profile_warmup([row], {"worker": profile})
    status["coverage"]["layers"][0]["eager"] = True
    with pytest.raises(ValueError, match="required captures"):
        P.require_profile_warmup([row], {"worker": profile})


@pytest.mark.parametrize("mutation", ["unhashable_layer", "boolean_count", "bad_mode_type", "nonobject_coverage"])
def test_malformed_signed_capture_metadata_is_a_bounded_validation_error(mutation):
    row = receipt()
    status = row["runtime_observation"]["optimizations"]["execution_profile"]
    if mutation == "unhashable_layer": status["coverage"]["layers"][0]["layer"] = []
    elif mutation == "boolean_count": status["coverage"]["captured_layers"] = True
    elif mutation == "bad_mode_type": status["coverage"]["layers"][0]["eager"] = "false"
    else: status["coverage"] = []
    with pytest.raises(ValueError):
        P.require_profile_warmup([row], {"worker": "resident-whole"})

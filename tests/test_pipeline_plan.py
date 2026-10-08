import copy
import pytest

from shard.pipeline_plan import build_plan, parse_split, resolve_bounds, validate_plan


def manifest():
    return build_plan({"num_hidden_layers": 36}, ring_id="test", cohort_id="a" * 64,
                      endpoints=["127.0.0.1:30001", "127.0.0.1:30002", "127.0.0.1:30003"],
                      split="20,8,8", gpu_uuids=["GPU-a", "GPU-b", "GPU-c"])


@pytest.mark.parametrize("split", ["20,8,7", "20,8", "20,-1,17", "20,0,16", [20, True, 15]])
def test_invalid_split_before_execution(split):
    with pytest.raises(ValueError):
        parse_split(split, 36, 3)


def test_heterogeneous_split_and_model_binding():
    value = manifest()
    assert [(s["lo"], s["hi"]) for s in value["stages"]] == [(0, 20), (20, 28), (28, 36)]
    assert resolve_bounds({"num_hidden_layers": 36}, 1, 3, plan=value) == (20, 28)
    with pytest.raises(ValueError):
        validate_plan(value, {"num_hidden_layers": 43})
    with pytest.raises(ValueError):
        resolve_bounds({"num_hidden_layers": 36}, 1, 3, lo=20, hi=29, plan=value)


@pytest.mark.parametrize("kind", ["gap", "overlap", "role", "gpu", "node", "endpoint", "tail"])
def test_manifest_rejects_bad_concrete_assignment(kind):
    value = copy.deepcopy(manifest())
    if kind == "gap": value["stages"][1]["lo"] += 1
    if kind == "overlap": value["stages"][1]["lo"] -= 1
    if kind == "role": value["stages"][1]["head"] = True
    if kind == "gpu": value["stages"][1]["gpu_uuid"] = "gpu-A"
    if kind == "node": value["stages"][1]["node_id"] = "stage-0"
    if kind == "endpoint": value["stages"][1]["endpoint"] = "localhost:123; rm"
    if kind == "tail": value["stages"][-1]["next_endpoint"] = "localhost:123"
    with pytest.raises(ValueError): validate_plan(value)


def test_ipv6_routes_and_no_same_ip_exclusion():
    value = build_plan({"num_hidden_layers": 2}, ring_id="same-ip", cohort_id="b" * 64,
                       endpoints=["[::1]:30001", "[::1]:30002"])
    assert validate_plan(value)["nstages"] == 2


def test_explicit_escape_hatch_still_checks_boundary_roles():
    with pytest.raises(ValueError):
        resolve_bounds({"num_hidden_layers": 36}, 0, 3, lo=1, hi=12)
    with pytest.raises(ValueError):
        resolve_bounds({"num_hidden_layers": 36}, 1, 3, split="12,12,12", lo=12, hi=24)

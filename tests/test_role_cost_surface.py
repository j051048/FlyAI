"""Fixed role work cannot disappear when a planner selects fewer main layers."""
from copy import deepcopy
import pytest

from shard.planning_cost import stage_observation
from shard.stage_cost import validate_surface
from test_stage_cost_surface import node
from test_stage_cost_offer import surface_offer
from shard.offers import OfferRegistry, OfferError, sign_offer, model_cohort_id
from test_node_offers import cohort


ROLE = {"head": False, "tail": True, "draft": True}


def role_node():
    n = node()
    surface = n["stage_trace"]["cost_surface"]
    surface.update(schema="shard-stage-cost-surface/2", decode_mode="dspark", role=deepcopy(ROLE))
    sample = surface["samples"][0]
    sample["component_samples_ms"] = {k: [v]*5 for k,v in
        dict(main=20., embedding=0., lm_head=5., draft=10., dma=2., sync=3.).items()}
    sample["prefill_component_samples_ms"] = deepcopy(sample["component_samples_ms"])
    sample["prefill_component_samples_ms"]["main"] = [80.]*5
    return n


def workload(**changes):
    return dict(dict(frame_tokens=1, context_tokens=4096, warmness="warm", decode_mode="dspark", role=deepcopy(ROLE)), **changes)


def test_head_mtp_dma_and_sync_fixed_costs_survive_main_layer_resizing():
    n = role_node()
    early = stage_observation(n, 2, 1, now=150, start=0, workload=workload())
    late = stage_observation(n, 2, 1, now=150, start=2, workload=workload())
    full = stage_observation(n, 4, 1, now=150, start=0, workload=workload(), collect_quality=True)
    assert early["service_ms"] == 26. and late["service_ms"] == 34.
    assert early["prefill_ms"] == 44. and late["prefill_ms"] == 76.
    assert early["role_cost"]["frame"]["fixed_ms"] == late["role_cost"]["frame"]["fixed_ms"] == 20.
    assert full["service_ms"] == 40 and full["exact_bound"]


@pytest.mark.parametrize("change", [{"decode_mode": "greedy"}, {"role": {"head":False,"tail":False,"draft":False}},
    {"decode_mode": None}, {"role": None}])
def test_unknown_or_changed_mode_and_role_never_reuses_a_measured_role_price(change):
    result = stage_observation(role_node(), 2, 1, now=150, start=0, workload=workload(**change))
    assert result["source"] == "scalar_estimate" and result["service_ms"] == 2.
    assert any("role/decode/context" in warning for warning in result["uncertainty"])


@pytest.mark.parametrize("corrupt", ["sum", "nan", "missing", "foreign_role"])
def test_role_partition_must_be_complete_and_consistent_with_executed_roles(corrupt):
    n = role_node(); surface = n["stage_trace"]["cost_surface"]
    sample = surface["samples"][0]
    if corrupt == "sum": sample["component_samples_ms"]["main"][0] += 1
    elif corrupt == "nan": sample["component_samples_ms"]["sync"][0] = float("nan")
    elif corrupt == "missing": sample["component_samples_ms"].pop("draft")
    else: surface["role"]["draft"] = False
    with pytest.raises(ValueError): validate_surface(n["stage_trace"])


def role_offer():
    body, key = surface_offer()
    trace = body["models"][0]["profile"]["stage_trace"]
    trace["context_tokens"] = 4096
    s = trace["cost_surface"]
    s.update(schema="shard-stage-cost-surface/2", decode_mode="greedy", role={"head":True,"tail":False,"draft":False})
    row = s["samples"][0]; row["prefill_samples_ms"] = None
    row["prefill_component_samples_ms"] = None
    row["component_samples_ms"] = {k:[v]*5 for k,v in
        dict(main=21., embedding=10., lm_head=0., draft=0., dma=6., sync=5.).items()}
    return body, key


def test_signed_offer_admits_role_costs_bound_to_actual_calibrated_runtime():
    body, key = role_offer()
    registry = OfferRegistry(clock=lambda:1000)
    try:
        registry.announce(sign_offer(body,key))
        n, = registry.snapshot(model_cohort_id(cohort()))
        w = dict(frame_tokens=1,context_tokens=4096,warmness="warm",decode_mode="greedy",role={"head":True,"tail":False,"draft":False})
        early = stage_observation(n,2,1,now=1001,start=0,workload=w)
        assert early["service_ms"] == 24 and early["role_cost"]["frame"]["fixed_ms"] == 21.
    finally:
        registry.close()


def test_re_signed_head_role_cannot_override_calibrated_runtime_roles():
    body,key = role_offer()
    s = body["models"][0]["profile"]["stage_trace"]["cost_surface"]
    s["role"]["tail"] = True
    registry = OfferRegistry(clock=lambda:1000)
    try:
        with pytest.raises(OfferError,match="stage trace"):
            registry.announce(sign_offer(body,key))
    finally:
        registry.close()

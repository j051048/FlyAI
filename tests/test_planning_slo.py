"""Tail/quality/risk planning contracts; simulations are not hardware SLO proofs."""
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from shard.locality import link_snapshot, measurement_quality
from shard.plan import plan_ring
from shard.planning_cost import estimate, slo_spec
from test_planning_locality import pool, mesh, profile
from test_route_chunk_planning import route, observations


def slo(metric="tpot_ms", *, target=100, mode="require_known", **options):
    return {"schema":"shard-planning-slo/1", "mode":mode,"targets":{metric:target}, **options}


def quality(*, latency=2, jitter=1, loss=0, window=1):
    return {"schema":"shard-link-quality/1","sample_count":100,"window_s":window,
            "latency_p95_ms":latency,"jitter_ms":jitter,"packet_loss_rate":loss}


def bound_nodes():
    nodes=pool(["A","A"],caps=[3,3])
    for i,node in enumerate(nodes):
        node.update(cohort_id="cohort",runtime_config_sha256="a"*64)
        node["stage_trace"]={"schema":"shard-stage-trace/2","node_id":node["id"],"gpu_uuid":node["gpu_uuid"],
            "cohort_id":"cohort","runtime_config_sha256":"a"*64,"frame_tokens":1,"layer_start":i*3,"layer_end":i*3+3,
            "context_tokens":512,"warmness":"warm","frame_ms":3,"prefill_ms":10,"queue_ms":0,"measured_at":100,"ttl_s":30,
            "quality":{"schema":"shard-stage-quality/1","sample_count":100,"window_s":1,
                "frame_p95_ms":4,"prefill_p95_ms":12,"queue_p95_ms":0}}
        node["pricing"]={"schema":"shard-node-price/1","currency":"USD","per_hour":1,"measured_at":100,"ttl_s":30}
    return nodes


def links(nodes, *, latency=2, p95=3, jitter=1, loss=0):
    return observations(*(dict(route(a["id"],b["id"],latency),quality=quality(latency=p95,jitter=jitter,loss=loss))
        for a in nodes for b in nodes if a is not b))


def workload(**kw):
    return {"context_tokens":512,"warmness":"warm","generated_tokens":20,"depth":1,"block_tokens":1,
        "acceptance_gain":1,"cancel_rate":0,**kw}


def forecast(nodes=None, *, policy=None, measurements=None):
    nodes=bound_nodes() if nodes is None else nodes
    snapshot=link_snapshot(nodes,measurements=measurements or links(nodes),now=100)
    return estimate([0,1],{0:3,1:3},{0:1,1:1},snapshot["rtt"],[0,2],[0,2],dict(enumerate(nodes)),
        {"decode_bytes":0,"prefill_bytes":0},workload(),edges=snapshot["edges"],now=100,objective="pipeline",slo=policy or slo())


def test_observed_component_p95_affects_tpot_and_reports_stress_scope_not_slo_certification():
    baseline=forecast(); slow=forecast(measurements=links(bound_nodes(),p95=30))
    assert slow["slo"]["metrics"]["tpot_ms"]>baseline["slo"]["metrics"]["tpot_ms"]
    assert baseline["slo"]["status"]=="predicted_within_targets"
    assert baseline["slo"]["acceptance_verified"] is False
    assert "not end-to-end p95" in baseline["slo"]["scope"]
    assert baseline["predicted_request_ms"]==slow["predicted_request_ms"]  # original measured-mean calendar not overwritten


def test_ttft_and_tpot_objectives_can_require_only_their_own_observations_and_no_queue_double_count():
    nodes=bound_nodes()
    for node in nodes:node["stage_trace"]["quality"].pop("frame_p95_ms")
    value=forecast(nodes,policy=slo("ttft_ms"))
    assert value["slo"]["eligible"] and value["slo"]["metrics"]["tpot_ms"] is None
    nodes=bound_nodes()
    for node in nodes:node["stage_trace"]["quality"]["queue_p95_ms"]=100
    queued=forecast(nodes)
    assert queued["slo"]["metrics"]["ttft_ms"]>forecast()["slo"]["metrics"]["ttft_ms"]
    assert queued["slo"]["metrics"]["tpot_ms"]==forecast()["slo"]["metrics"]["tpot_ms"]


def test_missing_stage_percentile_is_unknown_not_zero_and_advisory_can_return_uncertain_plan():
    nodes=bound_nodes()
    for node in nodes:node["stage_trace"].pop("quality")
    value=forecast(nodes)
    assert value["slo"]["metrics"]["tpot_ms"] is None and not value["slo"]["eligible"]
    relaxed=forecast(nodes,policy=slo(mode="advisory"))
    assert relaxed["slo"]["eligible"] and relaxed["slo"]["status"]=="unknown"


def test_foreign_context_and_scaled_range_cannot_claim_known_tail_target():
    nodes=bound_nodes();nodes[0]["stage_trace"]["context_tokens"]=1024
    assert forecast(nodes)["slo"]["metrics"]["tpot_ms"] is None
    nodes=bound_nodes();nodes[0]["stage_trace"].update(layer_end=2)
    assert forecast(nodes)["slo"]["metrics"]["tpot_ms"] is None


def test_route_selection_can_prefer_lower_tail_loss_and_jitter_without_rewriting_mean_latency():
    nodes=pool(["A","A"])
    measured=observations(dict(route("n0","n1",1,route_id="fast-mean"),quality=quality(latency=100,jitter=50,loss=.4)),
        dict(route("n0","n1",2,route_id="stable"),quality=quality(latency=3,jitter=1,loss=0)))
    assert link_snapshot(nodes,measurements=measured,now=100)["edges"]["n0","n1"]["route_id"]=="fast-mean"
    policy=slo_spec(slo(mode="advisory",weights={"tpot_ms":1,"packet_loss_rate":2,"jitter_ms":1}))
    ranked=link_snapshot(nodes,measurements=measured,now=100,quality_policy=policy)
    assert ranked["edges"]["n0","n1"]["route_id"]=="stable" and ranked["rtt"][0][1]==2
    pinned=link_snapshot(nodes,measurements=measured,now=100,quality_policy=policy,
        route_ids=[{"src":"n0","dst":"n1","route_id":"fast-mean"}])
    assert pinned["edges"]["n0","n1"]["route_id"]=="fast-mean"


def test_directed_return_quality_is_part_of_budget_and_dense_rtt_is_not_fresh_evidence():
    nodes=bound_nodes(); measured=links(nodes,loss=.4)
    value=forecast(policy=slo("packet_loss_rate",target=.1),measurements=measured)
    assert not value["slo"]["eligible"] and "packet_loss_rate" in value["slo"]["exceeded_targets"]
    data=links(nodes);data["edges"][1].pop("quality")
    value=forecast(policy=slo("packet_loss_rate",target=.1),measurements=data)
    assert value["slo"]["metrics"]["packet_loss_rate"] is None


def test_freshness_includes_statistical_window_and_keeps_expired_routes_unreachable():
    nodes=bound_nodes(); measured=links(nodes)
    for row in measured["edges"]:row["quality"]["window_s"]=60
    value=forecast(policy=slo("observation_age_s",target=30),measurements=measured)
    assert not value["slo"]["eligible"] and value["slo"]["metrics"]["observation_age_s"]==60
    assert link_snapshot(nodes,measurements=measured,now=111)["rtt"][0][1]==9000  # route ttl remains 10


def test_route_freshness_ranking_uses_oldest_sample_not_only_latest_endpoint_timestamp():
    nodes=bound_nodes();measured=links(nodes)
    measured["edges"][0].update(route_id="old-window",quality=quality(window=60))
    measured["edges"].append(dict(route("n0","n1",2,route_id="fresh-window"),
        measured_at=99,quality=quality(window=1)))
    result=plan_ring(nodes,None,{**profile(),"decode_bytes":0,"prefill_bytes":0},
        measurements=measured,now=100,workload=workload(),locality="global",
        slo=slo("observation_age_s",target=30))
    assert result is not None
    assert result["planning"]["routes"][0]["route_id"]=="fresh-window"
    assert result["planning"]["slo"]["metrics"]["observation_age_s"]==2


@pytest.mark.parametrize("first_quality", [None, quality(jitter=5)])
def test_require_known_route_cannot_hide_admissible_channel_behind_cheap_unknown_or_exceeded_quality(first_quality):
    nodes=bound_nodes();rows=links(nodes)["edges"]
    rows[0].update(route_id="cheap",latency_ms=0.1)
    if first_quality is None:rows[0].pop("quality")
    else:rows[0]["quality"]=first_quality
    rows.append(dict(route("n0","n1",300,route_id="observed"),quality=quality(latency=300,jitter=0)))
    result=plan_ring(nodes,None,{**profile(),"decode_bytes":0,"prefill_bytes":0},
        measurements=observations(*rows),now=100,workload=workload(),locality="global",
        slo=slo("jitter_ms",target=1,unknown_penalty=0.01))
    assert result is not None
    assert next(row for row in result["planning"]["routes"] if (row["src"],row["dst"])==("n0","n1"))["route_id"]=="observed"


def test_advertised_hourly_cost_requires_all_nodes_same_currency_and_fresh_prices():
    nodes=bound_nodes();value=forecast(nodes,policy=slo("hourly_cost",target=3))
    assert value["slo"]["metrics"]["hourly_cost"]==2 and value["slo"]["eligible"]
    for change in (None,"EUR","expired"):
        bad=deepcopy(nodes)
        if change is None:bad[0].pop("pricing")
        elif change=="expired":bad[0]["pricing"]["measured_at"]=50
        else:bad[0]["pricing"]["currency"]=change
        value=forecast(bad,policy=slo("hourly_cost",target=3))
        assert value["slo"]["metrics"]["hourly_cost"] is None and not value["slo"]["eligible"]
    assert "excludes cold load" in value["slo"]["pricing"]["scope"]


def test_same_ip_does_not_manufacture_fault_domain_and_colocated_gpus_remain_legal():
    nodes=bound_nodes()
    for node in nodes:node.update(public_ip="198.51.100.1",host_id="host-shared")
    value=forecast(nodes,policy=slo("failure_domain_concentration",target=1))
    assert value["slo"]["metrics"]["failure_domain_concentration"]==1 and value["slo"]["eligible"]
    for node in nodes:node.pop("host_id")
    value=forecast(nodes,policy=slo("failure_domain_concentration",target=1))
    assert value["slo"]["metrics"]["failure_domain_concentration"] is None
    for i,node in enumerate(nodes):node["failure_domains"]={"host":f"explicit-{i}","provider":"shared-provider"}
    value=forecast(nodes,policy=slo("failure_domain_concentration",target=.5,failure_domain_types=["host","provider"]))
    assert value["slo"]["metrics"]["failure_domain_concentration"]==1


def test_shared_fault_risk_changes_node_selection_instead_of_counting_cards():
    nodes=pool(["A"]*4,caps=[3]*4,speed=[1,1,2,2])
    for i,node in enumerate(nodes):node["host_id"]="shared" if i<2 else f"host-{i}"
    original=plan_ring(nodes,mesh(4,1),profile())
    assert set(original["order"])=={"n0","n1"}
    selected=plan_ring(nodes,mesh(4,1),profile(),slo=slo("failure_domain_concentration",target=0),now=100)
    assert set(selected["order"])!={"n0","n1"} and selected["planning"]["slo"]["eligible"]
    assert selected["planning"]["prediction_only"]


def test_require_known_rejects_no_evidence_with_actionable_diagnostics_not_false_slo_pass():
    diagnostics={}
    assert plan_ring(pool(["A","A"]),mesh(2),profile(),slo=slo(),diagnostics=diagnostics,now=100) is None
    assert "tpot_ms" in diagnostics["slo_rejections"]["unknown_metrics"]


def test_slo_none_preserves_existing_plan_exactly_and_cli_uses_same_contract():
    nodes=pool(["A","A"])
    assert plan_ring(nodes,mesh(2),profile())==plan_ring(nodes,mesh(2),profile(),slo=None)
    request={"nodes":nodes,"rtt":mesh(2),"model":profile(),"now":100,"slo":slo("failure_domain_concentration",target=0)}
    expected=plan_ring(nodes,mesh(2),profile(),now=100,slo=request["slo"])
    proc=subprocess.run([sys.executable,"-m","shard.plan"],input=json.dumps(request),text=True,capture_output=True)
    assert proc.returncode==0,proc.stderr+proc.stdout
    assert json.loads(proc.stdout)==expected


@pytest.mark.parametrize("value", [
    {"schema":"unknown"},slo(target=-1),slo(weights={"tpot_ms":float("nan")}),slo(normalizers={"tpot_ms":0}),
    slo(currency="USD/EUR"),slo(mode="guaranteed"),slo(failure_domain_types=[{}]),slo("packet_loss_rate",target=2),
])
def test_invalid_slo_contracts_fail_before_planning(value):
    with pytest.raises(ValueError):slo_spec(value)


@pytest.mark.parametrize("change", [
    {"sample_count":True},{"sample_count":1},{"window_s":0},{"packet_loss_rate":1.1},
    {"jitter_ms":-1},{"latency_p95_ms":float("inf")},{"unknown":1},
])
def test_invalid_observed_link_quality_cannot_enter_even_default_planning(change):
    value=quality();value.update(change)
    with pytest.raises(ValueError):measurement_quality(value,"link")


def test_signed_pricing_and_fault_domains_reach_snapshot_and_tampering_fails():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from shard.offers import OfferRegistry, OfferError, sign_offer, validate_offer, model_cohort_id
    from test_node_offers import offer, cohort
    key=Ed25519PrivateKey.generate();body=offer(key)
    body.update(pricing={"schema":"shard-node-price/1","currency":"USD","per_hour":1,"measured_at":1000,"ttl_s":30},
        failure_domains={"host":"declared-host","provider":"declared-provider"})
    signed=sign_offer(body,key);registry=OfferRegistry(clock=lambda:1000)
    registry.announce(signed)
    node=registry.snapshot(model_cohort_id(cohort()))[0]
    assert node["pricing"]==signed["pricing"] and node["failure_domains"]==signed["failure_domains"]
    node["pricing"]["per_hour"]=999
    assert registry.snapshot(model_cohort_id(cohort()))[0]["pricing"]["per_hour"]==1
    changed=deepcopy(signed);changed["pricing"]["per_hour"]=.001
    with pytest.raises(OfferError):validate_offer(changed,now=1000)
    changed=deepcopy(signed);changed["failure_domains"]["host"]="another-host"
    with pytest.raises(OfferError):validate_offer(changed,now=1000)


@pytest.mark.parametrize("field,value",[("pricing",{"per_hour":-1}), ("failure_domains",{"host":False})])
def test_invalid_signed_metadata_is_rejected_without_registration_whitelist(field,value):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from shard.offers import OfferError, sign_offer, validate_offer
    from test_node_offers import offer
    key=Ed25519PrivateKey.generate();body=offer(key);body[field]=value
    with pytest.raises(OfferError):validate_offer(sign_offer(body,key),now=1000)


def test_plan_prices_actual_request_time_and_keeps_dimensionless_score_out_of_request_ms():
    nodes=bound_nodes();p={**profile(),"decode_bytes":0,"prefill_bytes":0}
    result=plan_ring(nodes,None,p,measurements=links(nodes),now=100,workload=workload(),
        slo=slo("request_cost",target=1),locality="global")
    assert result is not None
    assert result["request_ms"]==result["planning"]["predicted_request_ms"]
    assert result["selection_score"]!=result["request_ms"]
    assert result["planning"]["slo"]["metrics"]["request_cost"]==pytest.approx(2*result["request_ms"]/3_600_000)


def test_quality_changes_real_plan_request_objective_while_default_keeps_fast_mean_route():
    nodes=bound_nodes();p={**profile(),"decode_bytes":0,"prefill_bytes":0}
    measured=links(nodes)
    # Two effective channels for the same physical direction.
    measured["edges"][0].update(route_id="fast-mean",latency_ms=1,quality=quality(latency=100))
    measured["edges"].append(dict(route("n0","n1",2,route_id="stable"),quality=quality(latency=3)))
    plain=plan_ring(nodes,None,p,measurements=measured,now=100,workload=workload(),locality="global")
    chosen=plan_ring(nodes,None,p,measurements=measured,now=100,workload=workload(),slo=slo(),locality="global")
    assert plain["planning"]["routes"][0]["route_id"]=="fast-mean"
    assert chosen["planning"]["routes"][0]["route_id"]=="stable"
    assert chosen["planning"]["slo"]["prediction_only"]


def test_prefill_serialization_uses_bound_context_geometry_only_for_explicit_slo():
    nodes=bound_nodes();snapshot=link_snapshot(nodes,measurements=links(nodes),now=100)
    args=([0,1],{0:3,1:3},{0:1,1:1},snapshot["rtt"],[0,2],[0,2],dict(enumerate(nodes)))
    p={"decode_bytes":0,"prefill_bytes":100_000,"wire_bytes_per_token":32}
    plain=estimate(*args,p,workload(),edges=snapshot["edges"],now=100)
    selected=estimate(*args,p,workload(),edges=snapshot["edges"],now=100,slo=slo("ttft_ms"))
    assert selected["slo"]["metrics"]["ttft_ms"]==pytest.approx(24+6+32*512*8/1_000_000)
    assert plain["predicted_request_ms"]-selected["predicted_request_ms"]==pytest.approx((100_000-32*512)*8/1_000_000)


def test_fixed_unbound_prefill_bytes_do_not_claim_known_ttft_or_request_cost():
    nodes=bound_nodes();snapshot=link_snapshot(nodes,measurements=links(nodes),now=100)
    args=([0,1],{0:3,1:3},{0:1,1:1},snapshot["rtt"],[0,2],[0,2],dict(enumerate(nodes)))
    p={"decode_bytes":0,"prefill_bytes":100_000}
    unknown=estimate(*args,p,workload(),edges=snapshot["edges"],now=100,slo=slo("ttft_ms"))
    assert unknown["slo"]["metrics"]["ttft_ms"] is None and not unknown["slo"]["eligible"]
    assert unknown["slo"]["metrics"]["request_cost"] is None
    bound=estimate(*args,{**p,"prefill_context_tokens":512},workload(),edges=snapshot["edges"],now=100,slo=slo("ttft_ms"))
    assert bound["slo"]["metrics"]["ttft_ms"] is not None

"""Actual local tensor accounting, immutable configuration and signed declarations."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch

from v4_draft_chain import DraftChain
from v4_resources import runtime_config_identity,measure_stage_resources
from v4_dual_runtime import observation,reservation
import v4_ref_cpu as R
from test_v4_dspark_fast import _build


def test_new_private_graph_state_is_inventoried_without_counter_identity_drift():
    args=R.cpu_args()
    stage,tail=_build(R.build_oracle(args,7),args)
    stage._runtime_draft=tail
    before=measure_stage_resources(stage,checkpoint_id="fixture")
    tail._draft_chain=DraftChain(tail,graph_bytes=16 << 20,reference=True)
    identity=runtime_config_identity(stage)
    tail._draft_chain.counters["reference_eager"]+=1
    assert runtime_config_identity(stage)==identity
    after=measure_stage_resources(stage,checkpoint_id="fixture")
    assert after["module_storage"]["host_bytes"]-before["module_storage"]["host_bytes"]==tail._draft_chain.used
    assert after["local_execution_reservation"]["driver_device_allocated_bytes"]==0
    status=observation(stage)
    assert status["draft_chain"]["reference_eager"]==1 and not status["performance_qualified"]


def test_external_ipc_driver_and_shared_memory_are_reserved_outside_torch_inventory():
    transport=SimpleNamespace(reservation=lambda:{"host_reserved_bytes":4096,"device_reserved_bytes":8192,
                                                  "driver_device_allocated_bytes":2048})
    stage=SimpleNamespace(device="cuda:0",_host_transport=transport)
    value=reservation(stage)
    assert value["remaining_host_reserved_bytes"]==4096
    assert value["remaining_device_reserved_bytes"]==6144
    assert value["driver_device_allocated_bytes"]==2048


def test_signed_local_scope_rejects_cpu_claiming_graph_replays_or_gpu_copies():
    from test_runtime_observation import observation as baseline
    from shard.runtime_observation import validate_runtime_observation
    stage=SimpleNamespace(device="cpu")
    value=baseline()
    value["optimizations"]={"prefill_expert_pipeline":{},"conversation_cache":{},"local_execution":observation(stage)}
    assert validate_runtime_observation(value)["optimizations"]["local_execution"]["performance_qualified"] is False
    for name,counts in (("draft_chain",{"reference":True,"replays":1}),("host_transport",{"ipc_copies":1})):
        invalid=deepcopy(value);invalid["optimizations"]["local_execution"][name]=counts
        with pytest.raises(ValueError,match="CPU reference"):
            validate_runtime_observation(invalid)


def test_private_configuration_locations_are_never_part_of_signed_public_declarations(monkeypatch):
    import json
    import v4_host_transport,v4_draft_adapter,v4_observability
    args=R.cpu_args()
    stage,_=_build(R.build_oracle(args,7),args)
    paths={key:"/private/operator/"+key for key in v4_observability.PRIVATE_CONFIG_PATHS}
    for name,value in paths.items(): monkeypatch.setenv(name,value)
    monkeypatch.setattr(v4_host_transport,"public_config",lambda:{"enabled":True,"config_sha256":"a"*64})
    monkeypatch.setattr(v4_draft_adapter,"public_config",lambda:{"enabled":True,"manifest_sha256":"b"*64})
    from v4_resources import runtime_config_payload
    public=json.dumps({"runtime":runtime_config_payload(stage),"observation":v4_observability.runtime_observation(stage)})
    assert all(path not in public for path in paths.values())
    assert all(name not in v4_observability.public_environment() for name in paths)

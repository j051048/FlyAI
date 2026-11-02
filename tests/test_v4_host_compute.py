"""Production helper retains uncertain copies and exposes a stable private-free policy."""
from copy import deepcopy
from types import SimpleNamespace
import json
import pytest

from v4_host_compute import HostComputeSession, configure
from shard.host_compute_budget import observed_cpu_ids, HostBudgetError


def policy(path):
    if len(observed_cpu_ids()) < 2:
        pytest.skip("a compute worker plus control CPU is required")
    return dict(schema="v4-host-compute-config/1",host_id="rig",budget_path=str(path),
        cpu_capacity=2,control_reserve=1,copy_capacity=1,workers=1,copies=1,cpu_ids=[],
        strict_affinity=False,ttl_s=300,wait_s=0)


def test_copy_budget_remains_active_until_real_completion_callback(tmp_path):
    p=policy(tmp_path/"host.db")
    session=HostComputeSession(SimpleNamespace(device="cpu"),p)
    seen=[]
    def complete():
        seen.append(session.budget.status()["active_copies"])
    config=deepcopy(session.config())
    with session.copy(completion=complete):
        assert session.budget.status()["active_copies"]==1
    assert seen==[1] and session.budget.status()["active_copies"]==0
    assert session.config()==config and "budget_path" not in json.dumps(config)
    assert str(tmp_path) not in json.dumps(config)
    session.close()


def test_failed_completion_stays_charged_until_explicit_owned_process_cleanup(tmp_path):
    session=HostComputeSession(SimpleNamespace(device="cpu"),policy(tmp_path/"host.db"))
    def failed(): raise RuntimeError("copy completion unknown")
    with pytest.raises(RuntimeError,match="completion unknown"):
        with session.copy(completion=failed): pass
    assert session.budget.status()["active_copies"]==1 and session.status()["faulted"]
    with pytest.raises(HostBudgetError,match="uncertain"):
        with session.compute(): pass
    with pytest.raises(HostBudgetError): session.close()
    session.close(confirmed_stopped=True)
    assert session.budget.status()["active_permits"]==0


def test_default_off_and_configured_stage_reuses_one_owner(tmp_path,monkeypatch):
    import torch
    st=SimpleNamespace(device="cpu")
    monkeypatch.delenv("V4_HOST_COMPUTE_CONFIG",raising=False)
    assert configure(st) is None
    cfg=tmp_path/"host.json"; cfg.write_text(json.dumps(policy(tmp_path/"host.db")))
    monkeypatch.setenv("V4_HOST_COMPUTE_CONFIG",str(cfg))
    first=configure(st)
    assert configure(st) is first
    # Declared workers are charged as a whole team: the owning process must have
    # been configured for them before serving, exactly as a launched worker is.
    # This test is the operator, so it pins the intra-op pool it declares.
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with first.compute(): assert first.budget.status()["active_workers"]==1
    finally:
        torch.set_num_threads(previous)
    first.close()
    with pytest.raises(HostBudgetError,match="closed"):
        with first.compute(): pass


def test_cpu_team_charged_as_a_whole_and_excess_torch_threads_refused(tmp_path,monkeypatch):
    import torch
    if len(observed_cpu_ids()) < 3:
        pytest.skip("two compute workers plus a control CPU required")
    p=policy(tmp_path/"team.db")
    p.update(cpu_capacity=3,workers=2)
    session=HostComputeSession(SimpleNamespace(device="cpu"),p)
    monkeypatch.setattr(torch,"get_num_threads",lambda:2)
    with session.compute():
        assert session.budget.status()["active_workers"]==2
    assert session.budget.status()["active_workers"]==0
    monkeypatch.setattr(torch,"get_num_threads",lambda:3)
    with pytest.raises(HostBudgetError,match="intra-op"):
        with session.compute(): pass
    assert session.budget.status()["active_workers"]==0
    session.close()

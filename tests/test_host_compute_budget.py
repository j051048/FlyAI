"""Real SQLite coordination, owner lifetime and calling-thread restoration."""
from concurrent.futures import ThreadPoolExecutor
import os
import pytest

from shard.host_compute_budget import HostComputeBudget, HostBudgetBusy, HostBudgetError


def budget(path, **kwargs):
    return HostComputeBudget(path, host_id="rig", cpu_capacity=4, control_reserve=1,
                             copy_capacity=1, allowed_cpu_ids=[0,1,2,3], **kwargs)


def test_two_worker_instances_share_cpu_and_copy_admission_without_reclaiming_busy_work(tmp_path):
    first, second = budget(tmp_path/"host.db"), budget(tmp_path/"host.db")
    assert first.config()["budget_id"] == second.config()["budget_id"]
    assert budget(tmp_path/"independent.db").config()["budget_id"] != first.config()["budget_id"]
    a = first.register("gpu-a","boot:1",workers=2)
    b = second.register("gpu-b","boot:1",workers=2)
    with a.work(slots=2):
        with pytest.raises(HostBudgetBusy):
            with b.work(slots=2): pass
        with b.work(slots=1):
            assert first.status()["active_workers"] == 3
        with a.work("copy"):
            with pytest.raises(HostBudgetBusy):
                with b.work("copy"): pass
    assert first.status()["active_permits"] == 0
    assert first.status()["worker_capacity"] == 3
    a.close(); b.close()


def test_owner_epoch_expiry_and_crash_do_not_silently_release_active_permits(tmp_path):
    now=[100.]
    db=budget(tmp_path/"host.db",clock=lambda:now[0])
    a=db.register("gpu-a","boot:1",ttl_s=2)
    token,_,_=db._acquire("gpu-a","boot:1","compute",1)
    now[0]=103
    assert db.status()["active_workers"]==1
    with pytest.raises(HostBudgetError):
        with a.work(): pass
    with pytest.raises(HostBudgetBusy): db.register("gpu-a","boot:2")
    with pytest.raises(HostBudgetBusy): a.close()
    a.close(confirmed_stopped=True)
    b=db.register("gpu-a","boot:2")
    with pytest.raises(HostBudgetError): db._release(token,"gpu-a","boot:1")
    with b.work(): assert db.status()["active_workers"]==1
    assert db.status()["active_workers"]==0


def test_thread_affinity_is_scoped_restored_and_permits_release_on_exception(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(os,"sched_getaffinity",lambda pid:{0,1,2,3},raising=False)
    monkeypatch.setattr(os,"sched_setaffinity",lambda pid,cpus:calls.append((pid,set(cpus))),raising=False)
    db=budget(tmp_path/"host.db"); owner=db.register("gpu","e",cpu_ids=[1,2])
    with pytest.raises(RuntimeError,match="work failed"):
        with owner.work():
            assert db.affinity_status=="calling_thread_bound"
            raise RuntimeError("work failed")
    assert calls==[(0,{1,2}),(0,{0,1,2,3})]
    assert db.status()["active_permits"]==0


def test_unknown_affinity_is_explicit_fallback_or_strict_refusal(tmp_path,monkeypatch):
    monkeypatch.delattr(os,"sched_getaffinity",raising=False)
    monkeypatch.delattr(os,"sched_setaffinity",raising=False)
    db=budget(tmp_path/"host.db")
    optional=db.register("optional","e",cpu_ids=[1])
    with optional.work(): assert db.affinity_status=="unsupported_os"
    strict=db.register("strict","e",cpu_ids=[1],strict_affinity=True)
    with pytest.raises(HostBudgetError,match="unsupported"):
        with strict.work(): pass
    assert db.status()["active_permits"]==0


def test_later_worker_cannot_raise_frozen_host_capacities(tmp_path):
    budget(tmp_path/"host.db")
    with pytest.raises(HostBudgetError,match="geometry changed"):
        HostComputeBudget(tmp_path/"host.db",host_id="rig",cpu_capacity=4,control_reserve=0,
                          copy_capacity=1,allowed_cpu_ids=[0,1,2,3])

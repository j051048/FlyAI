"""Optional host-wide scheduling for one owned Stage.

The local configuration contains a private SQLite path. Public configuration
contains only its policy digest and resource geometry. Copy mode deliberately
waits for completion before releasing admission; it does not claim more overlap.
"""
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid

from shard.host_compute_budget import HostComputeBudget, HostBudgetError


def configured_policy(path):
    path = Path(path)
    if path.stat().st_size > 16384:
        raise ValueError("host compute configuration exceeds 16KiB")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema", "host_id", "budget_path", "cpu_capacity", "control_reserve", "copy_capacity"}
    optional = {"workers", "copies", "cpu_ids", "strict_affinity", "ttl_s", "wait_s"}
    if not isinstance(value, dict) or set(value)-required-optional or not required <= set(value) or value["schema"] != "v4-host-compute-config/1":
        raise ValueError("invalid host compute configuration")
    if not isinstance(value["budget_path"], str) or not Path(value["budget_path"]).is_absolute():
        raise ValueError("host budget database requires an absolute local path")
    value = {"workers":1,"copies":1,"cpu_ids":[],"strict_affinity":False,"ttl_s":300,"wait_s":0,**value}
    if type(value["wait_s"]) not in (int,float) or not 0 <= value["wait_s"] <= 30:
        raise ValueError("host budget wait must be 0..30 seconds")
    return value


def configure(stage):
    existing = getattr(stage, "_host_compute", None)
    if existing is not None:
        return existing
    name = os.environ.get("V4_HOST_COMPUTE_CONFIG", "")
    if not name:
        return None
    session = HostComputeSession(stage, configured_policy(name))
    stage._host_compute = session
    return session


class HostComputeSession:
    def __init__(self, stage, policy):
        self.stage, self.policy = stage, dict(policy)
        self.budget = HostComputeBudget(policy["budget_path"], host_id=policy["host_id"],
            cpu_capacity=policy["cpu_capacity"], control_reserve=policy["control_reserve"],
            copy_capacity=policy["copy_capacity"])
        self.owner_id = "stage:" + uuid.uuid4().hex
        self.owner = None
        self._fault = False
        self._closed = False
        self._lock = threading.RLock()
        self._register()

    def _register(self):
        p = self.policy
        self.owner = self.budget.register(self.owner_id, uuid.uuid4().hex, workers=p["workers"], copies=p["copies"],
            ttl_s=p["ttl_s"],cpu_ids=p["cpu_ids"],strict_affinity=p["strict_affinity"])

    def _ready(self):
        with self._lock:
            if self._closed:
                raise HostBudgetError("host compute session is closed")
            if self._fault:
                raise HostBudgetError("host copy completion is uncertain; stop the owned Stage before cleanup")
            try:
                self.budget.renew(self.owner.owner,self.owner.epoch,ttl_s=self.policy["ttl_s"])
            except HostBudgetError:
                # Idle expiry creates a fresh local epoch only if all old work
                # drained; register refuses active permits, including expired ones.
                self._register()
            return self.owner

    def config(self):
        public = {k:v for k,v in self.policy.items() if k not in ("budget_path",)}
        encoded = json.dumps(public,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        return {"enabled":True,"policy_sha256":hashlib.sha256(encoded).hexdigest(),
            "budget":self.budget.config(),"workers":self.policy["workers"],"copies":self.policy["copies"],
            "copy_completion":"synchronize_before_release", "affinity":"calling_thread_only",
            "cpu_charge":"declared_torch_intraop_workers"}

    def status(self):
        return {**self.config(),"observed":self.budget.status(),"faulted":self._fault,
                "closed":self._closed,"gpu_qualified":False}

    @contextmanager
    def compute(self):
        owner = self._ready()
        # Torch's intra-op pool is process-wide. Do not mutate it during a
        # serving call: operators set OMP/MKL/torch threads before launching each
        # worker. Charge the whole declared team and refuse oversubscription.
        import torch
        workers = self.policy["workers"]
        if torch.get_num_threads() > workers:
            raise HostBudgetError("Torch intra-op threads exceed declared host workers; configure the owned process before serving")
        with owner.work("compute",slots=workers,timeout_s=self.policy["wait_s"]):
            yield

    def _completion(self):
        if str(self.stage.device).startswith("cuda"):
            import torch
            torch.cuda.synchronize(self.stage.device)

    @contextmanager
    def copy(self, completion=None):
        if completion is not None and not callable(completion):
            raise ValueError("copy completion must be callable")
        owner = self._ready()
        # Copy has no CPU affinity side effect, and no generator-owned permit:
        # an uncertain event MUST remain charged even if this context is GC'd.
        import time
        from shard.host_compute_budget import HostBudgetBusy
        deadline = time.monotonic()+self.policy["wait_s"]
        while True:
            try:
                token,_,_ = self.budget._acquire(owner.owner,owner.epoch,"copy",1)
                break
            except HostBudgetBusy:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(min(.005,max(0.,deadline-time.monotonic())))
        try:
            yield
        finally:
            try:
                (completion or self._completion)()
            except BaseException:
                self._fault = True
                raise
            else:
                self.budget._release(token,owner.owner,owner.epoch)

    def close(self, *, confirmed_stopped=False):
        with self._lock:
            if self._fault and not confirmed_stopped:
                raise HostBudgetError("uncertain copy permit retained until confirmed stopped")
            self.owner.close(confirmed_stopped=confirmed_stopped)
            self._closed = True

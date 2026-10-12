"""Bounded, state-isolated native tokenwise lanes.

This is time slicing of the ORIGINAL Stage.forward, not a batched GEMM. The
numerical reference consumes every prompt/decode token one at a time from the
empty state, with one fixed horizon. It is deliberately a separate contract
from ordinary full-prompt prefill. State copies preserve every native buffer
and the addresses used by CUDA graphs. No speculative state is shared.
"""
from copy import deepcopy
import errno
import os
import time
import uuid

import torch

from v4_conversation_cache import _buffers, _digest, _signature, _model_stamp

CONTRACT = "v4-native-tokenwise-greedy/1"


def configured_options():
    count = int(os.environ.get("V4_NATIVE_LANES", "0"))
    budget = int(os.environ.get("V4_NATIVE_LANE_HOST_MIB", "0")) << 20
    ttl = float(os.environ.get("V4_NATIVE_LANE_CACHE_TTL_S", "300"))
    directory = os.environ.get("V4_NATIVE_LANE_STATE_DIR", "")
    disk = int(os.environ.get("V4_NATIVE_LANE_DISK_MIB", "0")) << 20
    if count not in (0, 2, 3, 4) or budget < 0 or not 0 < ttl <= 3600:
        raise ValueError("native lanes need 0 or 2..4 slots, bounded host RAM and cache TTL")
    if count and not budget:
        raise ValueError("native lane host reservation is required")
    if bool(directory) != bool(disk) or disk < 0 or (directory and not count):
        raise ValueError("durable native prefixes require enabled lanes, a private directory and disk budget")
    return {"max_jobs": count, "host_bytes": budget, "ttl_s": ttl,
            "state_directory": directory or None, "disk_bytes": disk}


class NativeLanes:
    def __init__(self, stage, *, max_jobs, host_bytes, ttl_s=300, clock=time.monotonic,
                 state_directory=None, disk_bytes=0):
        if type(max_jobs) is not int or not 2 <= max_jobs <= 4:
            raise ValueError("native lane count must be 2..4")
        if type(host_bytes) is not int or host_bytes <= 0 or not 0 < ttl_s <= 3600:
            raise ValueError("native lane memory and TTL must be bounded")
        if getattr(stage, "_dspark_capable", False) or getattr(stage, "_kv_runtime", None) is not None:
            raise ValueError("native lanes v1 require resident KV and no loaded MTP")
        import sys
        slim = sys.modules.get("v4_ref_slim")
        if slim and (getattr(slim, "V4_REF_SLIM", False) or getattr(slim, "V4_REF_SLIM_NOQAT", False)):
            raise ValueError("native lanes refuse history-dependent reference bypass")
        if getattr(stage, "_expert_placement", "gpu") != "gpu":
            raise ValueError("native lanes v1 qualify resident experts independently of host swapping")
        self.stage, self.max_jobs, self.clock, self.ttl_s = stage, max_jobs, clock, float(ttl_s)
        self.buffers = _buffers(stage, None)
        self.signatures = {name: _signature(value) for name, value in self.buffers.items()}
        self.tensor_bytes = sum(value.numel() * value.element_size() for value in self.buffers.values())
        # Bounded dictionary/identity/receipt state reservation, in addition to
        # full token-capacity tensors. Never charge only the current frontier.
        self.metadata_bytes = 65536 + stage.args.max_seq_len * 64
        # A disk load maps one complete checkpoint before copying into a slot.
        # Reserve that additional resident working set instead of assuming mmap
        # has zero RAM cost. Hashing contiguous CPU state uses bounded views.
        self.restore_workspace_bytes = self.tensor_bytes + (2 << 20) if state_directory else 0
        self.per_job_bytes = self.tensor_bytes + self.metadata_bytes
        self.reserved_bytes = self.per_job_bytes * max_jobs + self.restore_workspace_bytes
        if self.reserved_bytes > host_bytes:
            raise ValueError("native lane states exceed declared host reservation")
        if state_directory and (type(disk_bytes) is not int or disk_bytes < self.tensor_bytes + (1 << 20)):
            raise ValueError("native durable budget cannot hold one complete state and its manifest")
        self.slots = []
        for _ in range(max_jobs):
            values = {name: torch.empty_like(value, device="cpu", pin_memory=False, memory_format=torch.contiguous_format)
                      for name, value in self.buffers.items()}
            self.slots.append({"buffers": values, "job": None, "cache": None, "pos": 0})
        self.active = {}
        self.owner_epoch = None
        self.current = None
        self.model_stamp = None
        self.disk_identity = None
        self.store = None
        self.last_cache_write_error = None
        if state_directory:
            from v4_durable_state import DurableStateStore
            self.store = DurableStateStore(state_directory, max_bytes=disk_bytes,
                                            max_entries=max_jobs, ttl_s=ttl_s)

    def config(self):
        return {"enabled": True, "numerical_contract": CONTRACT, "max_jobs": self.max_jobs,
                "horizon": self.stage.args.max_seq_len, "per_job_state_bytes": self.per_job_bytes,
                "reserved_state_bytes": self.reserved_bytes, "tensor_bytes_per_job": self.tensor_bytes,
                "metadata_bytes_per_job": self.metadata_bytes, "prefill_chunk_tokens": 1,
                "restore_workspace_bytes": self.restore_workspace_bytes,
                "durable_prefix": {"enabled": self.store is not None,
                                   "max_bytes": self.store.max_bytes if self.store else 0},
                "ttl_s": self.ttl_s, "state_tier": "pageable_host", "compute": "serial_scalar_lanes"}

    def tensors(self):
        for index, slot in enumerate(self.slots):
            for name, value in slot["buffers"].items():
                yield f"lane.{index}.{name}", value

    def _check(self):
        st = self.stage
        if st._spec or st._dspark or st._replaying or st._spec_ckpts or st._last_tap:
            raise ValueError("native lanes refuse speculative/replay/tap state")
        current = _buffers(st, None, synchronize=False)
        if {k: _signature(v) for k, v in current.items()} != self.signatures:
            raise ValueError("native state storage changed after lane reservation")

    def bind_epoch(self, epoch):
        if not isinstance(epoch, str) or not epoch or len(epoch) > 256:
            raise ValueError("strict owner epoch required")
        if self.owner_epoch != epoch:
            self.clear()
            self.owner_epoch = epoch

    def bind_storage_identity(self, identity):
        """Authenticated model/source/runtime/lease identity, before any disk IO."""
        if not isinstance(identity, dict) or not identity or self.owner_epoch is None:
            raise ValueError("durable native state needs a live exact runtime identity")
        binding = {"runtime": deepcopy(identity), "owner_epoch": self.owner_epoch,
                   "contract": CONTRACT, "horizon": self.stage.args.max_seq_len}
        if self.disk_identity is not None and self.disk_identity != binding and self.active:
            raise ValueError("durable native identity changed while jobs are active")
        self.disk_identity = binding

    def _disk_key(self, tenant_identity):
        if self.disk_identity is None:
            raise ValueError("durable prefix runtime identity was not bound")
        return {**self.disk_identity, "tenant_identity": tenant_identity}

    def _idle(self):
        empty = next((row for row in self.slots if row["job"] is None and row["cache"] is None), None)
        if empty is not None:
            return empty
        idle = [row for row in self.slots if row["job"] is None]
        if not idle:
            raise ValueError("all native state slots are active")
        return min(idle, key=lambda row: row["cache"]["expires"])

    def _disk_restore(self, entry_id, identity):
        if self.store is None:
            return None
        restored = self.store.get(entry_id, self._disk_key(identity))
        if restored is None:
            return None
        tensors, metadata = restored
        fields = {"entry_id", "identity", "prefix", "state_digest", "pos", "expires_at"}
        if (set(metadata) != fields or metadata["entry_id"] != entry_id or metadata["identity"] != identity
                or type(metadata["pos"]) is not int or not 0 < metadata["pos"] <= self.stage.args.max_seq_len
                or not isinstance(metadata["prefix"], dict) or set(metadata["prefix"]) != {"token_count", "digest"}
                or metadata["prefix"].get("token_count") != metadata["pos"]
                or not isinstance(metadata["prefix"]["digest"], str) or len(metadata["prefix"]["digest"]) != 64
                or any(c not in "0123456789abcdef" for c in metadata["prefix"]["digest"])
                or type(metadata["expires_at"]) not in (float, int)
                or not 0 < metadata["expires_at"] - time.time() <= self.ttl_s
                or set(tensors) != set(self.buffers)
                or any(t.shape != self.buffers[k].shape or t.dtype != self.buffers[k].dtype
                       for k, t in tensors.items()) or _digest(tensors) != metadata["state_digest"]):
            self.store.remove(entry_id)
            raise ValueError("durable native state layout/frontier/content changed")
        slot = self._idle()
        for name, value in tensors.items():
            slot["buffers"][name].copy_(value)
        cached = {k: deepcopy(v) for k, v in metadata.items() if k not in ("pos", "expires_at")}
        cached["expires"] = self.clock() + max(0, metadata["expires_at"] - time.time())
        slot.update(pos=metadata["pos"], cache=cached)
        return slot

    def clear(self):
        self.active.clear()
        for slot in self.slots:
            slot.update(job=None, cache=None, pos=0)
        self.current = None

    def open(self, job, *, identity, horizon, restore_entry=None):
        self._check()
        stamp = _model_stamp(self.stage, None)
        if self.model_stamp is None:
            self.model_stamp = stamp
        elif stamp != self.model_stamp:
            raise ValueError("native immutable model/rotary state changed after admission")
        if not isinstance(job, str) or not 1 <= len(job) <= 256 or job in self.active:
            raise ValueError("duplicate or invalid lane job")
        if horizon != self.stage.args.max_seq_len:
            raise ValueError("native lane horizon must equal its calibrated fixed capacity")
        if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("native lane tenant/cohort identity digest required")
        now = self.clock()
        restored = None
        restore_tier = None
        if restore_entry is not None:
            slot = next((row for row in self.slots if row["job"] is None and row["cache"]
                         and row["cache"]["entry_id"] == restore_entry), None)
            restore_tier = "memory" if slot is not None else "disk"
            if slot is None:
                slot = self._disk_restore(restore_entry, identity)
            if slot is None or slot["cache"]["identity"] != identity or slot["cache"]["expires"] <= now:
                raise ValueError("native prefix expired, evicted or belongs to another identity")
            restored = dict(slot["cache"])
            if _digest(slot["buffers"]) != restored["state_digest"]:
                slot["cache"] = None
                raise ValueError("native prefix state content changed")
        else:
            slot = self._idle()
            self.stage._spec = self.stage._dspark = self.stage._replaying = False
            self.stage.reset()
            slot["pos"] = 0
            self._save(slot)
        slot.update(job=job, identity=identity, cache=None, restore_tier=restore_tier)
        self.active[job] = slot
        self.current = None
        return restored

    def _save(self, slot):
        self._check()
        with torch.no_grad():
            for name, value in self.buffers.items():
                slot["buffers"][name].copy_(value, non_blocking=False)
        slot["pos"] = self.stage._pos

    def activate(self, job, pos):
        self._check()
        slot = self.active.get(job)
        if slot is None or type(pos) is not int or pos != slot["pos"] or pos >= self.stage.args.max_seq_len:
            raise ValueError("native frame does not match its live lane frontier")
        with torch.no_grad():
            for name, value in self.buffers.items():
                value.copy_(slot["buffers"][name], non_blocking=False)
        self.stage._pos = pos
        self.stage._spec = self.stage._dspark = self.stage._replaying = False
        self.current = job
        return slot

    def finish_step(self, job):
        slot = self.active.get(job)
        if slot is None or self.current != job or self.stage._pos != slot["pos"] + 1:
            raise ValueError("native forward did not advance exactly one token")
        self._save(slot)

    def close(self, job, *, retain=False, prefix=None):
        self.last_cache_write_error = None
        slot = self.active.get(job)
        if slot is None:
            return None
        if retain and (not isinstance(prefix, dict) or set(prefix) != {"token_count", "digest"}
                       or prefix["token_count"] != slot["pos"] or slot["pos"] < 1
                       or not isinstance(prefix["digest"], str) or len(prefix["digest"]) != 64
                       or any(c not in "0123456789abcdef" for c in prefix["digest"])):
            raise ValueError("native cache publication must match the consumed frontier")
        result = None
        if retain:
            result = {"entry_id": uuid.uuid4().hex, "identity": slot["identity"],
                      "prefix": dict(prefix), "state_digest": _digest(slot["buffers"]),
                      "expires": self.clock() + self.ttl_s}
            if self.store is not None:
                metadata = {k: deepcopy(v) for k, v in result.items() if k != "expires"}
                metadata["pos"] = slot["pos"]
                metadata["expires_at"] = time.time() + self.ttl_s
                from v4_durable_state import DurableStateCapacity
                try:
                    self.store.put(result["entry_id"], self._disk_key(slot["identity"]), slot["buffers"], metadata)
                except (DurableStateCapacity, OSError) as error:
                    if not isinstance(error, DurableStateCapacity) and error.errno != errno.ENOSPC:
                        raise
                    # A verified completion does not depend on its optional
                    # cache fitting. Publish no partial cross-stage frontier;
                    # release this lane normally. Integrity/permission failures
                    # remain fatal and are intentionally not caught here.
                    self.last_cache_write_error = ("durable_capacity" if isinstance(error, DurableStateCapacity)
                                                   else "disk_full")
                    result = None
        self.active.pop(job)
        slot.update(job=None, cache=result)
        if self.current == job:
            self.current = None
        return result


def configured_runner(stage):
    options = configured_options()
    return NativeLanes(stage, **options) if options["max_jobs"] else None

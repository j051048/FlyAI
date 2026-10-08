"""Keep a node lease busy for the entire resident engine process lifetime.

Only the locally configured runner chooses argv. A network peer cannot supply
commands. Expiry stops this runner's own process; its work handle is released
only after wait() confirms process exit and the OS has reclaimed GPU allocations.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import subprocess
import secrets
import tempfile
import string
import threading
import time


def local_lease_guard(spec):
    from .leases import LeaseLedger
    required = {"ledger", "node_id", "principal", "lease_id", "fencing_token", "ring_id", "cohort_id"}
    if not isinstance(spec, dict) or not required <= set(spec) or set(spec) - required - {"runtime_config", "assignment"}:
        raise ValueError("exact local stage lease configuration required")
    # This file is created by the local runner, never accepted as a remote principal.
    principal = spec["principal"]
    ledger = LeaseLedger(spec["ledger"], node_id=spec["node_id"],
                         authorize=lambda p, a, b: principal if p == principal else None)
    guard = ledger.guard(spec["lease_id"], spec["fencing_token"], principal=principal,
                         ring_id=spec["ring_id"], model_cohort_sha256=spec["cohort_id"])
    guard.expected_runtime_config = spec.get("runtime_config")
    guard.stage_assignment = spec.get("assignment")
    return guard


def load_local_lease_guard(path):
    p = Path(path)
    if p.is_symlink() or p.stat().st_size > 262144:
        raise ValueError("invalid local lease configuration file")
    return local_lease_guard(json.loads(p.read_text(encoding="utf-8")))


def install_stage_lease_checks(stage, guard):
    """One node-local check at frame/API boundaries, never a remote RPC per layer."""
    guard.assert_live()
    if getattr(stage, "_lease_guard", None) is not None:
        raise ValueError("stage already has an execution lease")
    for name in ("reset", "embed", "forward", "logits_all", "commit", "load"):
        original = getattr(stage, name, None)
        if original is None:
            continue
        def checked(*args, _original=original, **kwargs):
            guard.assert_live()
            result = _original(*args, **kwargs)
            guard.assert_live()
            return result
        setattr(stage, name, functools.wraps(original)(checked))
    stage._lease_guard = guard
    return stage


def process_lease_watchdog(guard, *, interval_s=0.25, exit_process=os._exit):
    """CLI child safety net, including periods blocked waiting for network input."""
    stop = threading.Event()
    def watch():
        while not stop.wait(interval_s):
            try:
                guard.assert_live()
            except Exception:
                exit_process(75)
                return
    thread = threading.Thread(target=watch, daemon=True, name="stage-lease-fence")
    thread.start()
    return stop


class LeasedProcessRunner:
    def __init__(self, guard, *, ledger_path, command_factory, environment=None, stop_timeout_s=5,
                 runtime_config=None):
        self.guard, self.ledger_path = guard, str(ledger_path)
        self.command_factory, self.environment = command_factory, dict(environment or {})
        self.stop_timeout_s = stop_timeout_s
        self.runtime_config = runtime_config
        self._lock = threading.RLock()
        self._process = self._work = self._monitor = self._config = None
        self._stop = threading.Event()
        self._error = None

    def start(self, assignment):
        with self._lock:
            if self._process is not None:
                raise RuntimeError("stage process already started")
            self.guard.assert_live()
            lease = self.guard.assert_live()
            if (assignment.get("ring_id") != lease.ring_id or assignment.get("cohort_id") != lease.model_cohort_sha256
                    or assignment.get("node_id") != lease.node_id or assignment.get("gpu_uuid") != lease.gpu_uuid):
                raise ValueError("stage assignment differs from node lease")
            command = self.command_factory(dict(assignment))
            if not isinstance(command, (list, tuple)) or not command or any(not isinstance(v, str) for v in command):
                raise ValueError("local command factory must return argv strings")
            work = self.guard.begin_work("resident-stage-" + lease.lease_id + "-" + secrets.token_hex(8))
            try:
                fd, config = tempfile.mkstemp(prefix="shard-stage-lease-", suffix=".json")
                spec = {"ledger": self.ledger_path, "node_id": lease.node_id,
                        "principal": self.guard.principal, "lease_id": lease.lease_id,
                        "fencing_token": lease.fencing_token, "ring_id": lease.ring_id,
                        "cohort_id": lease.model_cohort_sha256, "runtime_config": self.runtime_config,
                        "assignment": dict(assignment)}
                with os.fdopen(fd, "w", encoding="utf-8") as out:
                    json.dump(spec, out)
                env = {**os.environ, **self.environment, "SHARD_STAGE_LEASE_CONFIG": config,
                       "SHARD_STAGE_TELEMETRY_FILE": config + ".telemetry.json"}
                options = {"env": env, "stdin": subprocess.DEVNULL}
                if os.name == "nt":
                    options["creationflags"] = subprocess.CREATE_NO_WINDOW
                process = subprocess.Popen(list(command), **options)
            except BaseException:
                self.guard.end_work(work)
                if "config" in locals():
                    Path(config).unlink(missing_ok=True)
                raise
            self._process, self._work, self._config = process, work, config
            self._monitor = threading.Thread(target=self._watch, daemon=True, name="resident-stage-lease")
            self._monitor.start()
            return {"pid": process.pid, "lease_id": lease.lease_id, "resident_work": work.to_dict()}

    def _terminate(self):
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(self.stop_timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(self.stop_timeout_s)

    def _watch(self):
        try:
            while self._process.poll() is None:
                if self._stop.wait(0.25):
                    self._terminate()
                    break
                try:
                    self.guard.assert_live()
                except Exception as exc:
                    self._error = type(exc).__name__
                    self._terminate()
                    break
            # wait(), rather than a vanished readiness flag, acknowledges GPU cleanup.
            self._process.wait()
            self.guard.end_work(self._work)
            self._work = None
            Path(self._config).unlink(missing_ok=True)
            Path(self._config + ".telemetry.json").unlink(missing_ok=True)
        except Exception as exc:
            # Cleanup failure deliberately keeps the durable reservation occupied.
            self._error = type(exc).__name__

    def stop(self):
        self._stop.set()
        monitor = self._monitor
        if monitor is not None:
            monitor.join(self.stop_timeout_s * 2 + 2)
            if monitor.is_alive() or self._work is not None:
                raise RuntimeError("stage cleanup was not acknowledged; lease remains occupied")

    def status(self):
        with self._lock:
            telemetry = None
            if self._config is not None:
                path = Path(self._config + ".telemetry.json")
                if path.exists() and not path.is_symlink() and path.stat().st_size <= 65536:
                    try:
                        body = json.loads(path.read_text(encoding="utf-8"))
                        if body.get("schema") == "shard-stage-telemetry/1": telemetry = body
                    except (OSError, ValueError): pass
            return {"pid": None if self._process is None else self._process.pid,
                    "running": self._process is not None and self._process.poll() is None,
                    "resident_work_held": self._work is not None, "error": self._error, "telemetry": telemetry}


def configured_stage_factory(ledger_path, stages):
    """Build runners ONLY from locally measured and configured stage templates.

    Each template has cohort_id, lo/hi, head/tail, requirements, runtime_config,
    argv and optional environment. Remote assignments select an existing
    template and fill numeric stage coordinates; they never supply argv/env.
    """
    from .deployment import config_digest
    from .resources import PlacementRequirements
    from datetime import datetime, timezone
    configured = list(stages)
    def factory(assignment, guard):
        for name in ("lo", "hi", "stage", "nstages"):
            if type(assignment.get(name)) is not int or assignment[name] < 0:
                raise ValueError("integer stage assignment required")
        if not 0 <= assignment["stage"] < assignment["nstages"] or assignment["lo"] >= assignment["hi"]:
            raise ValueError("invalid stage coordinates")
        if (type(assignment.get("head")) is not bool or type(assignment.get("tail")) is not bool
                or assignment["head"] != (assignment["stage"] == 0)
                or assignment["tail"] != (assignment["stage"] == assignment["nstages"] - 1)):
            raise ValueError("stage roles differ from execution order")
        matching = [row for row in configured if all(row.get(k) == assignment.get(k)
                    for k in ("cohort_id", "lo", "hi", "head", "tail")) and
                    (assignment.get("runtime_config_sha256") is None or
                     config_digest(row["runtime_config"]) == assignment["runtime_config_sha256"])]
        if len(matching) != 1:
            raise ValueError("no unique locally calibrated stage template")
        row = matching[0]
        req = PlacementRequirements.from_dict(row["requirements"])
        cfg = row["runtime_config"]
        if (req.provenance.node_id != guard.node_id or (req.layer_start, req.layer_end) != (assignment["lo"], assignment["hi"])
                or req.provenance.runtime_config_sha256 != config_digest(cfg)
                or any(cfg.get(k) != assignment[k] for k in ("lo", "hi", "head", "tail"))):
            raise ValueError("local template differs from measured runtime contract")
        env = {**os.environ, **row.get("environment", {})}
        def v4_flags(values):
            return {k: v for k, v in values.items() if k.startswith("V4_") and k not in {"V4_DIR", "V4_DEV"}}
        if v4_flags(env) != v4_flags(cfg.get("environment", {})):
            raise ValueError("effective stage environment differs from calibration")
        if cfg.get("engine") == "gpt-oss":
            def oss_flags(values):
                return {k: v for k, v in values.items() if k.startswith("FV_")}
            if oss_flags(env) != oss_flags(cfg.get("environment", {})):
                raise ValueError("effective GPT-OSS stage environment differs from calibration")
        allowed_placeholders = {"lo", "hi", "stage", "nstages", "next"}
        for arg in row["argv"]:
            for _, field, fmt, conversion in string.Formatter().parse(arg):
                if field is not None and (field not in allowed_placeholders or fmt or conversion):
                    raise ValueError("stage argv may interpolate only validated execution coordinates")
        measured = datetime.fromisoformat(req.provenance.measured_at.replace("Z", "+00:00"))
        if not -30 <= (datetime.now(timezone.utc) - measured).total_seconds() <= 300:
            raise ValueError("local stage calibration is stale")
        if (guard.resources.vram_bytes < req.gpu.peak_bytes or guard.resources.ram_bytes < req.host.peak_bytes
                or guard.resources.pinned_bytes < req.host.pinned_bytes):
            raise ValueError("lease does not reserve the actual stage peak")
        def command(local_assignment):
            # Values remain argv elements; no shell interprets these strings.
            return [arg.format_map(local_assignment) for arg in row["argv"]]
        return LeasedProcessRunner(guard, ledger_path=ledger_path, command_factory=command,
                                   environment=row.get("environment"), runtime_config=cfg)
    return factory

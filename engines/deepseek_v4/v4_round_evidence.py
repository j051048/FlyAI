"""Promote frozen, independently reverified prefix experiments into local policy.

Raw worker signatures bind stage observations, not hardware honesty or coordinator
wall clocks. This is an operator-owned calibration artefact, never an attestation.
"""
from __future__ import annotations

from copy import deepcopy
import math
import json
import os
from pathlib import Path
import stat
import tempfile
import time

from v4_adaptive_round import CALIBRATION_SCHEMA, calibration_binding, digest

MEASUREMENT_SCOPE = "warm_committed_decode_and_drain; DMA includes full-job prefetch as a conservative upper bound"
SLOT_SCHEMA = "v4-round-calibration-slot/1"
BINDING_SCHEMA = "v4-round-calibration-binding/1"
MAX_SLOT_BYTES = 32 * 1024 * 1024


def _profiles(evidence, options):
    from phase0 import v4_benchmark as bench
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 16:
        raise ValueError("one to sixteen independent frozen prefix experiments required")
    profiles, previous = [], None
    binding = options["binding"]
    expected_stages = binding["stages"]
    node_hashes = binding["runtime_config_sha256_by_node"]
    for entry in evidence:
        if not isinstance(entry, dict) or set(entry) != {"protocol", "report"}:
            raise ValueError("each calibration experiment needs its independent protocol and raw report")
        protocol, report = entry["protocol"], entry["report"]
        evaluation = bench.evaluate_report(report, protocol)
        if not evaluation.get("valid_evidence"):
            raise ValueError("round calibration requires verified raw benchmark evidence")
        if protocol.get("evidence_contract") != {
                "coordinator_diagnostics_required": True, "runtime_observation_required": True}:
            raise ValueError("round calibration requires complete signed runtime and work observations")
        experiment = protocol.get("round_experiment", {})
        if experiment.get("schema") != bench.ROUND_EXPERIMENT_SCHEMA:
            raise ValueError("round calibration requires frozen static prefix experiments")
        if (experiment["loaded_block_size"], int(protocol["env"]["V4_SPEC_DEPTH"]),
                int(protocol["env"]["V4_REFILL_FLOOR"]), protocol["env"]["V4_LAZY_DRAFT"] not in ("", "0")) != (
                options["loaded_block_size"], options["depth"], options["floor"], binding["lazy"]):
            raise ValueError("prefix experiment differs from the selected fixed runtime")
        stages = [{"node_id": row["node_id"], "gpu_uuid": row["gpu_uuid"],
            "signer_pubkey": row["signer_pubkey"], "lo": row["layer_start"], "hi": row["layer_end"]}
            for row in protocol["hardware"]]
        if stages != expected_stages or protocol["checkpoint"]["config_sha256"] != binding["checkpoint_config_sha256"]:
            raise ValueError("prefix experiment belongs to another selected model or GPU assignment")
        cap = experiment["draft_prefix_cap"]
        if any(row["draft_prefix_cap"] == cap for row in profiles):
            raise ValueError("duplicate prefix experiment")
        if previous is not None:
            compared = bench.compare_reports(previous, report, vary="round:draft_prefix_cap")
            if compared["status"] not in ("verified_comparison", "verified_target_pass"):
                raise ValueError("prefix experiments differ beyond the one declared prefix cap")
        # Even one experiment must carry the independent transport comparison
        # contract; it may not become a seed for an unidentified deployment.
        if not protocol.get("network_comparison"):
            raise ValueError("prefix experiment must freeze its network route")
        for sample in report["samples"]:
            if sample["swarm_id"] != binding["capabilities"]["ring_id"]:
                raise ValueError("prefix observation belongs to another ring")
            receipts = sample["receipts"]
            if len(receipts) != len(expected_stages):
                raise ValueError("every measured job requires every selected stage")
            for receipt in receipts:
                observed = receipt.get("runtime_observation")
                if (not isinstance(observed, dict) or observed["source_sha256"] != binding["source_sha256"]
                        or observed["runtime_config_sha256"] != node_hashes.get(observed["node_id"])):
                    raise ValueError("signed source/runtime differs from the local policy binding")
        tokens = 0; elapsed = 0.; dma = 0; dma_complete = True
        for sample in report["samples"]:
            if sample["phase"] != "warm":
                continue
            tokens += len(sample["tokens"])-1
            measurement = sample["measurement"]
            elapsed += measurement["elapsed_s"]-measurement["first_token_s"]
            for receipt in sample["receipts"]:
                metrics = receipt.get("runtime_metrics")
                if not isinstance(metrics, dict):
                    dma_complete = False
                    continue
                from shard.runtime_metrics import validate_runtime_metrics
                validate_runtime_metrics(metrics)
                if (metrics["mode"] == "reference_cpu" or
                        (metrics["mode"] == "gpu_expert_cache" and "expert_prefetch" not in metrics)):
                    # An older cache report without prefetch accounting cannot
                    # certify the full transfer bill. CPU fixtures cannot do so
                    # for a GPU deployment either.
                    dma_complete = False
                dma += sum(metrics["work"][role][phase]["dma_bytes"]
                           for role in ("main", "draft") for phase in ("decode", "replay"))
                # Prefetch is only per-job today; include it conservatively
                # instead of inventing a decode-only decomposition.
                dma += metrics.get("expert_prefetch", {}).get("dma_bytes", 0)
        if tokens < 2 or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("invalid measured prefix work/time")
        profiles.append({"draft_prefix_cap": cap, "committed_tokens": tokens,
                         "elapsed_s": elapsed, "expert_dma_bytes": dma if dma_complete else None})
        previous = report
    if min(options["loaded_block_size"], options["depth"]-1) not in {row["draft_prefix_cap"] for row in profiles}:
        raise ValueError("measured unchanged prefix baseline is required")
    return sorted(profiles, key=lambda row: row["draft_prefix_cap"])


def build_calibration(evidence, options, *, now=None, ttl_s=900.):
    now = time.time() if now is None else now
    if (type(now) not in (int, float) or not math.isfinite(now) or now < 0 or
            type(ttl_s) not in (int, float) or not math.isfinite(ttl_s) or not 0 < ttl_s <= 86400):
        raise ValueError("finite calibration time and lifetime of at most one day required")
    return {"schema": CALIBRATION_SCHEMA, "binding_sha256": calibration_binding(options),
        "created_at": now, "expires_at": now+ttl_s, "profiles": _profiles(evidence, options),
        "measurement_scope": MEASUREMENT_SCOPE, "evidence": deepcopy(evidence)}


def verify_calibration(bundle, options):
    if not isinstance(bundle, dict) or set(bundle) != {"schema", "binding_sha256", "created_at", "expires_at",
                                                     "profiles", "measurement_scope", "evidence"}:
        raise ValueError("full round calibration evidence bundle required for service admission")
    if bundle["schema"] != CALIBRATION_SCHEMA or bundle["binding_sha256"] != calibration_binding(options):
        raise ValueError("round calibration bundle belongs to another runtime")
    if bundle["measurement_scope"] != MEASUREMENT_SCOPE or bundle["profiles"] != _profiles(bundle["evidence"], options):
        raise ValueError("round profiles differ from independently verified measurements")
    # A compact detached record avoids parsing/hashing multi-megabyte signed
    # reports in the token/request hot path. The full bundle remains on disk.
    return {key: deepcopy(bundle[key]) for key in ("schema", "binding_sha256", "created_at", "expires_at", "profiles")} | {
        "evidence_sha256": digest(bundle["evidence"])}


def _private(path, *, directory=False):
    details = path.lstat()
    if not (stat.S_ISDIR(details.st_mode) if directory else stat.S_ISREG(details.st_mode)):
        raise ValueError("calibration slot must be an ordinary local file in an ordinary directory")
    if os.name == "posix" and (details.st_uid != os.geteuid() or details.st_mode & 0o077):
        raise ValueError("calibration slot directory/file must be owned by this user and private (0700/0600)")
    return details


def _publish_private(path, value):
    """Operator-only atomic publication. Never follows a URL or contacts a worker.

    Unix uses owner-only permissions. On Windows the operator's directory ACL is
    authoritative; chmod is not represented as a replacement for an NTFS ACL.
    """
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _private(path.parent, directory=True)
    if path.exists() or path.is_symlink():
        _private(path)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(payload) > MAX_SLOT_BYTES:
        raise ValueError("calibration slot exceeds its 32 MiB budget")
    fd, temporary = tempfile.mkstemp(prefix=".round-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.chmod(temporary, 0o600)
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return len(payload)


def publish_slot(path, config):
    if not isinstance(config, dict) or type(config.get("enabled")) is not bool:
        raise ValueError("explicit local round configuration required")
    calibration = config.get("calibration")
    if config.get("enabled") and not isinstance(calibration, dict):
        raise ValueError("enabled round configuration needs full calibration evidence")
    if not config.get("enabled") and config != {"enabled": False}:
        raise ValueError("disabled slot permits only enabled=false")
    envelope = {"schema": SLOT_SCHEMA, "config": config,
                "calibration_sha256": digest(calibration) if calibration is not None else None}
    size = _publish_private(path, envelope)
    return {"schema": SLOT_SCHEMA, "calibration_sha256": envelope["calibration_sha256"], "bytes": size}


def publish_binding(slot_path, options):
    """Export the selected live binding, not lease credentials or fabricated rates."""
    value = {"schema": BINDING_SCHEMA, "options": options, "binding_sha256": calibration_binding(options)}
    _publish_private(str(slot_path)+".binding.json", value)


def read_binding(value):
    if (not isinstance(value, dict) or set(value) != {"schema", "options", "binding_sha256"}
            or value["schema"] != BINDING_SCHEMA or value["binding_sha256"] != calibration_binding(value["options"])):
        raise ValueError("invalid selected round calibration binding")
    return deepcopy(value["options"])


class CalibrationSlot:
    """Read only changed private files at request boundaries; retain approved state on error."""
    def __init__(self, path):
        if not isinstance(path, (str, Path)) or "://" in str(path) or not Path(path).is_absolute():
            raise ValueError("round calibration slot requires an absolute local path")
        self.path = Path(path)
        self._seen = None
        self._status = {"status": "waiting_for_local_file", "accepted_updates": 0}

    def poll(self, install):
        try:
            _private(self.path.parent, directory=True)
            details = _private(self.path)
            fingerprint = (details.st_ino, details.st_size, details.st_mtime_ns)
            if fingerprint == self._seen:
                return
            self._seen = fingerprint
            if not 0 < details.st_size <= MAX_SLOT_BYTES:
                raise ValueError("calibration slot exceeds its 32 MiB budget")
            with self.path.open("rb") as stream:
                payload = stream.read(MAX_SLOT_BYTES+1)
            if len(payload) > MAX_SLOT_BYTES:
                raise ValueError("calibration slot grew beyond its size budget")
            value = json.loads(payload)
            if not isinstance(value, dict) or set(value) != {"schema", "config", "calibration_sha256"} or value["schema"] != SLOT_SCHEMA:
                raise ValueError("invalid round calibration slot envelope")
            config = value["config"]
            if not isinstance(config, dict):
                raise ValueError("round slot requires a local config object")
            expected = digest(config["calibration"]) if "calibration" in config else None
            if expected != value["calibration_sha256"]:
                raise ValueError("round slot calibration checksum differs")
            accepted = install(config)
            self._status = {"status": "accepted", "accepted_updates": self._status["accepted_updates"]+1,
                            "calibration_sha256": expected, **accepted}
        except FileNotFoundError:
            self._status.update(status="waiting_for_local_file", reason="local slot is absent; prior approved policy retained")
            self._seen = None
        except (ValueError, TypeError, KeyError, OSError) as error:
            reason = f"local slot unavailable ({type(error).__name__})" if isinstance(error, OSError) else str(error)[:240]
            self._status.update(status="rejected_update", reason=reason)

    def status(self):
        return dict(self._status)

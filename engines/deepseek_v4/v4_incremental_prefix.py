"""Operator-local, bounded incremental-prefix qualification and canary policy.

This is a numerical qualification, not a remote computation proof. Production
reuse requires actual GPU evidence, an exact cohort/source/runtime identity, a
fresh full-prefill canary in this process, and periodic shadow checks. A failing
canary disables reuse until a new policy is installed.
"""
from copy import deepcopy
import hashlib
import json

SCHEMA = "v4-incremental-prefix-qualification/1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class IncrementalQualification:
    def __init__(self, report, *, shadow_every=8):
        fields = {"schema", "binding", "source_sha256", "gpu_verified", "cases", "evidence_sha256",
                  "max_context", "max_suffix", "parent_alignment", "execution", "evidence"}
        if (not isinstance(report, dict) or set(report) != fields or report["schema"] != SCHEMA
                or report["gpu_verified"] is not True or not isinstance(report["binding"], dict)
                or not report["binding"] or type(shadow_every) is not int or not 1 <= shadow_every <= 64):
            raise ValueError("complete local GPU incremental qualification required")
        for key in ("source_sha256", "evidence_sha256"):
            if not isinstance(report[key], str) or len(report[key]) != 64 or any(c not in "0123456789abcdef" for c in report[key]):
                raise ValueError("qualification requires content-addressed source and actual evidence")
        for key in ("max_context", "max_suffix", "parent_alignment"):
            if type(report[key]) is not int or not 1 <= report[key] <= 1_048_576:
                raise ValueError("incremental qualification geometry must be bounded")
        cases = report["cases"]
        required = {"append", "compression_boundary", "rollback", "cancel", "reset", "tenant_isolation"}
        if (not isinstance(cases, dict) or set(cases) != required
                or any(value is not True for value in cases.values())):
            raise ValueError("incremental qualification must pass all native state/isolation cases")
        self.report = deepcopy(report)
        from v4_conversation_protocol import execution_binding
        if (not isinstance(report["execution"], dict)
                or execution_binding(report["execution"]) != report["execution"]
                or any(report["execution"].get(k) for k in ("spec", "dspark", "pipelined", "temp"))):
            raise ValueError("incremental qualification currently supports its exact resident scalar greedy reset execution only")
        from v4_incremental_evidence import inspect_evidence
        inspect_evidence(report, require_complete=True)
        self.sha256 = digest(report)
        self.shadow_every = shadow_every
        self.checked = self.reused = 0
        self.poisoned = False

    def eligible(self, identity, *, source_sha256, execution, parent, full):
        return (not self.poisoned and identity == self.report["binding"]
            and source_sha256 == self.report["source_sha256"]
            and execution == self.report["execution"]
            and type(parent) is int and type(full) is int
            and 0 < parent < full <= self.report["max_context"]
            and full - parent <= self.report["max_suffix"]
            and parent % self.report["parent_alignment"] == 0)

    def reuse(self):
        if self.poisoned or not self.checked or self.reused >= self.shadow_every - 1:
            return False
        self.reused += 1
        return True

    def canary(self, passed):
        if type(passed) is not bool:
            raise ValueError("real all-stage comparison required")
        self.checked += 1
        self.reused = 0
        self.poisoned |= not passed

    def status(self):
        return {"schema": SCHEMA, "qualification_sha256": self.sha256,
                "canaries": self.checked, "since_canary": self.reused,
                "poisoned": self.poisoned, "shadow_every": self.shadow_every,
                "scope": "operator-local empirical numerical qualification; not execution proof"}

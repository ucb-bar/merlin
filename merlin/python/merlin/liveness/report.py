"""Shared result types for the liveness oracle: a severity-ranked :class:`Finding` and the aggregate
:class:`LivenessReport`. Kept dependency-free so both the static linter and the dynamic model emit the
same shape and a caller can merge them.
"""
from __future__ import annotations

import dataclasses
import enum


class Severity(str, enum.Enum):
    """Ordered by how badly it breaks a silicon run. ``UNKNOWN`` is not benign — it means a precondition
    could NOT be derived (fail-closed), so it ranks above ``WARN``: an underived check is a gap the caller
    must see, not a pass."""

    FAULT = "fault"      # will fault / cannot run or build on silicon
    STALL = "stall"      # may hang: back-pressure, missing drain, capacity deadlock
    UNKNOWN = "unknown"  # a precondition could not be derived (surfaced, never dropped)
    WARN = "warn"        # risky but likely benign
    INFO = "info"        # informational (peaks, headroom)

    @property
    def rank(self) -> int:
        return {"fault": 4, "stall": 3, "unknown": 2, "warn": 1, "info": 0}[self.value]


# Verdict is the worst severity present, mapped to a single word.
_VERDICT = {4: "fault", 3: "stall", 2: "unknown", 1: "warn", 0: "ok"}


@dataclasses.dataclass
class Finding:
    """One precondition/liveness observation. ``rule`` is a stable slug; ``derived_from`` records the
    provenance of the fact the rule used (so a reader can see the check was DERIVED, not assumed)."""

    rule: str
    severity: Severity
    message: str
    where: str | None = None          # pc / addr / instruction index / symbol / tensor
    derived_from: str | None = None   # provenance of the fact used (fail-closed audit trail)
    evidence: dict | None = None
    fix_hint: str | None = None

    def to_dict(self) -> dict:
        d = {
            "rule": self.rule,
            "severity": self.severity.value,
            "message": self.message,
        }
        for k in ("where", "derived_from", "fix_hint"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.evidence:
            d["evidence"] = self.evidence
        return d


@dataclasses.dataclass
class LivenessReport:
    """Aggregate of every finding for one program against one target, plus dynamic resource peaks."""

    target: str
    program: str | None
    findings: list[Finding] = dataclasses.field(default_factory=list)
    resource_peaks: dict = dataclasses.field(default_factory=dict)

    def add(self, f: Finding | None) -> None:
        if f is not None:
            self.findings.append(f)

    def extend(self, fs) -> None:
        for f in fs:
            self.add(f)

    @property
    def worst_rank(self) -> int:
        return max((f.severity.rank for f in self.findings), default=0)

    @property
    def verdict(self) -> str:
        """``fault`` | ``stall`` | ``unknown`` | ``warn`` | ``ok`` — the worst severity present."""
        return _VERDICT[self.worst_rank]

    def by_severity(self, sev: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity == sev]

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "program": self.program,
            "verdict": self.verdict,
            "findings": [f.to_dict() for f in self.findings],
            "resource_peaks": self.resource_peaks,
            "counts": {
                sev.value: len(self.by_severity(sev))
                for sev in Severity
                if self.by_severity(sev)
            },
        }

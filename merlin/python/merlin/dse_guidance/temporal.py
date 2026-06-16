"""Parse and validate temporal / multi-rate workload metadata.

A flat capture collapses a VLA workload to a single pass. Reality is multi-rate::

    backbone once
    for k in K:                       # denoise / action-head steps
        action_head_step(...)
    emit H actions
    execute actions at control_rate_hz

This module reads the small ``temporal_workload_metadata`` wrapper (see the schema of the
same name), validates it, and derives the replan deadline. The headline timing budget the
guidance reasons about is::

    t_backbone + K * t_head_step  <=  H / control_rate_hz

The deadline ``replan_deadline_ms = 1000 * H / control_rate_hz`` is derived here; if the file
also states it, the stated value is checked against the derived one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from merlin.common import schemas
from merlin.common.yaml import load_yaml

_SCHEMA = "temporal_workload_metadata"
# Tolerance (ms) for a stated replan_deadline_ms vs the value derived from H / control_rate.
_DEADLINE_TOL_MS = 1.0


# Region roles. A flat capture collapses these; the multi-rate view distinguishes them, and the
# triage reasons about each separately (the backbone runs once, only the head repeats K times).
ROLE_BACKBONE = "backbone_once"
ROLE_REPEATED_HEAD = "repeated_head"
ROLE_LOOP_INVARIANT = "loop_invariant_state"
ROLE_LOOP_CARRIED = "loop_carried_state"
ROLE_CONTROL = "control_loop"

# Map the (existing) cadence vocabulary onto roles when an explicit role is not given.
_CADENCE_ROLE = {
    "once_per_replan": ROLE_BACKBONE,
    "K_times_per_replan": ROLE_REPEATED_HEAD,
    "control_loop": ROLE_CONTROL,
}


@dataclass
class Region:
    name: str
    cadence: str | None = None
    role: str | None = None
    invocation_count: int | None = None
    loop_trip_count: int | None = None
    loop_invariant_state: list[str] = field(default_factory=list)
    loop_carried_state: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)
    consumes: list[str] = field(default_factory=list)


@dataclass
class TemporalMetadata:
    workload: str
    K: int
    H: int
    control_rate_hz: float
    replan_deadline_ms: float
    regions: list[Region]
    cls: str | None = None
    warnings: list[str] = field(default_factory=list)

    def region(self, name: str) -> Region | None:
        return next((r for r in self.regions if r.name == name), None)

    def loop_invariant_state(self) -> set[str]:
        """Union of loop-invariant state across the repeated-head regions."""
        out: set[str] = set()
        for r in self.regions:
            if r.role == ROLE_REPEATED_HEAD or (r.loop_trip_count or 0) > 1:
                out.update(r.loop_invariant_state)
        # State explicitly produced once and consumed by the head also counts (prefix/KV).
        for r in self.regions:
            if r.role == ROLE_LOOP_INVARIANT:
                out.update(r.produces or [r.name])
        return out

    def repeated_head_regions(self) -> list[Region]:
        """Regions that run K times per replan (the action/denoise head)."""
        return [r for r in self.regions if r.role == ROLE_REPEATED_HEAD]

    def backbone_regions(self) -> list[Region]:
        """Regions that run once per replan (the vision/LM backbone)."""
        return [r for r in self.regions if r.role == ROLE_BACKBONE]

    def has_repeated_head(self) -> bool:
        """True iff a bounded repeated-head loop (invocation_count / K > 1) is present."""
        for r in self.repeated_head_regions():
            if (r.invocation_count or r.loop_trip_count or self.K) > 1:
                return True
        # Fall back to K when no region roles were given (single-region workloads).
        return not self.regions and int(self.K) > 1

    def has_k_loop(self) -> bool:
        """True iff the workload exposes a bounded K-step loop (K > 1)."""
        return int(self.K) > 1


def derived_deadline_ms(H: float, control_rate_hz: float) -> float:
    """``replan_deadline_ms = 1000 * H / control_rate_hz`` (the action chunk's wall budget)."""
    if not control_rate_hz:
        raise ValueError("control_rate_hz must be non-zero to derive a replan deadline")
    return 1000.0 * float(H) / float(control_rate_hz)


def parse(doc: dict) -> TemporalMetadata:
    """Validate and normalize a temporal metadata mapping into a :class:`TemporalMetadata`."""
    schemas.validate_or_raise(doc, _SCHEMA)
    timing = doc.get("timing") or {}
    if "K" not in timing or "H" not in timing or "control_rate_hz" not in timing:
        raise ValueError("temporal metadata 'timing' requires K, H, and control_rate_hz")

    K = int(timing["K"])
    H = int(timing["H"])
    control_rate_hz = float(timing["control_rate_hz"])
    derived = derived_deadline_ms(H, control_rate_hz)

    warnings: list[str] = []
    deadline = derived
    if timing.get("replan_deadline_ms") is not None:
        stated = float(timing["replan_deadline_ms"])
        if abs(stated - derived) > _DEADLINE_TOL_MS:
            warnings.append(
                f"stated replan_deadline_ms={stated} disagrees with derived "
                f"1000*H/control_rate_hz={derived:.4f}; using derived value"
            )
        # Always trust the derived value (it is the equation of record).
        deadline = derived

    regions = []
    for i, r in enumerate(doc.get("regions") or []):
        cadence = r.get("cadence")
        role = r.get("role") or _CADENCE_ROLE.get(cadence)
        regions.append(Region(
            name=str(r.get("name", f"region_{i}")),
            cadence=cadence,
            role=role,
            invocation_count=(int(r["invocation_count"]) if r.get("invocation_count") is not None
                              else None),
            loop_trip_count=(int(r["loop_trip_count"]) if r.get("loop_trip_count") is not None
                             else None),
            loop_invariant_state=list(r.get("loop_invariant_state") or []),
            loop_carried_state=list(r.get("loop_carried_state") or []),
            produces=list(r.get("produces") or []),
            consumes=list(r.get("consumes") or []),
        ))

    return TemporalMetadata(
        workload=str(doc["workload"]),
        K=K, H=H, control_rate_hz=control_rate_hz,
        replan_deadline_ms=deadline,
        regions=regions,
        cls=doc.get("class"),
        warnings=warnings,
    )


def load(path) -> TemporalMetadata:
    """Load and parse temporal metadata from a YAML file."""
    return parse(load_yaml(path))

"""Target-neutral whole-program implementation plan.

``DispatchProgram`` says *what* the model computes.  ``GlobalPlan`` records one compatible choice
for *how* every node is implemented, including the representation changes between choices.  The
types in this module deliberately contain no target vocabulary: a target adapter enumerates legal
alternatives and supplies their evidence-backed costs, while the shared planner checks coverage and
chooses a compatible set.

The distinction between regions and transitions is load-bearing.  A per-op selector can choose two
individually cheap kernels whose intervening pack, host/device crossing, spill, or reconfiguration is
more expensive than both.  Keeping the edge work explicit lets a whole-model optimizer price that
case instead of hiding it in whichever endpoint happened to be measured first.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .dispatch_program import DispatchProgram

GLOBAL_PLAN_SCHEMA = "global_plan_v1"


@dataclass(frozen=True)
class CycleInterval:
    """A conservative cycle interval, or an explicit refusal.

    ``None`` is not read as zero.  Both endpoints are present for a resolved estimate because the
    planner may prune only with a lower bound and promotes only against the conservative upper end.
    """

    lo: float | None
    hi: float | None
    provenance: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.lo is None) != (self.hi is None):
            raise ValueError("a cycle interval must provide both endpoints or neither")
        if self.lo is None:
            if not self.missing:
                raise ValueError("an unresolved cycle interval must say what is missing")
            return
        if isinstance(self.lo, bool) or isinstance(self.hi, bool):
            raise TypeError("cycle endpoints must be numbers, not booleans")
        lo, hi = float(self.lo), float(self.hi)
        if lo < 0 or hi < lo:
            raise ValueError(f"invalid cycle interval [{lo}, {hi}]")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @property
    def resolved(self) -> bool:
        return self.lo is not None

    @classmethod
    def point(cls, cycles: float, provenance: str = "") -> "CycleInterval":
        prov = (provenance,) if provenance else ()
        return cls(float(cycles), float(cycles), provenance=prov)

    @classmethod
    def unknown(cls, *missing: str) -> "CycleInterval":
        return cls(None, None, missing=tuple(str(item) for item in missing if str(item).strip()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "lo": self.lo,
            "hi": self.hi,
            "resolved": self.resolved,
            "provenance": list(self.provenance),
            "missing": list(self.missing),
        }


@dataclass(frozen=True)
class ValueRepresentation:
    """The properties of one buffer that can force work at a region boundary."""

    placement: str
    layout: str
    dtype: str
    encoding: str = "plain"
    quantization: str = "none"
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name, value in (("placement", self.placement), ("layout", self.layout),
                            ("dtype", self.dtype), ("encoding", self.encoding),
                            ("quantization", self.quantization)):
            if not str(value).strip():
                raise ValueError(f"a value representation must name its {name}")
        if tuple(sorted(self.attributes)) != self.attributes:
            raise ValueError("representation attributes must be sorted for stable content addressing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "placement": self.placement,
            "layout": self.layout,
            "dtype": self.dtype,
            "encoding": self.encoding,
            "quantization": self.quantization,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class BufferRepresentation:
    buffer: str
    representation: ValueRepresentation

    def to_dict(self) -> dict[str, Any]:
        return {"buffer": self.buffer, "representation": self.representation.to_dict()}


@dataclass(frozen=True)
class PlanDemand:
    """One exact full-shape resource demand retained for the generalized roofline."""

    resource: str
    amount: float
    unit: str
    basis: str = "moved"
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.resource.strip() or not self.unit.strip():
            raise ValueError("a plan demand must name its resource and unit")
        if isinstance(self.amount, bool) or not isinstance(self.amount, (int, float)):
            raise TypeError("a plan demand amount must be numeric")
        if float(self.amount) < 0:
            raise ValueError("a plan demand amount must be non-negative")
        if self.basis not in ("moved", "algorithmic"):
            raise ValueError("a plan demand basis must be 'moved' or 'algorithmic'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "amount": float(self.amount),
            "unit": self.unit,
            "basis": self.basis,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class ResourceOccupancy:
    """Measured or derived busy time on one resource for one plan component.

    This is deliberately separate from :class:`PlanDemand`.  A byte count is work; it is not busy
    time until a measured/derived resource rate and its fixed terms price it.  Keeping both lets the
    planner report movement volume without pretending that bytes alone establish latency hiding.
    """

    resource: str
    cycles: CycleInterval
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.resource.strip():
            raise ValueError("resource occupancy must name its resource")
        if not self.cycles.resolved:
            raise ValueError(
                f"occupancy for {self.resource!r} is unresolved: {self.cycles.missing}")

    def to_dict(self) -> dict[str, Any]:
        return {"resource": self.resource, "cycles": self.cycles.to_dict(),
                "provenance": self.provenance}


@dataclass(frozen=True)
class RegionAlternative:
    """One legal implementation of one or more whole-program nodes."""

    id: str
    nodes: tuple[int, ...]
    implementation: str
    placement: str
    cycles: CycleInterval
    inputs: tuple[BufferRepresentation, ...] = ()
    outputs: tuple[BufferRepresentation, ...] = ()
    demands: tuple[PlanDemand, ...] = ()
    occupancy: tuple[ResourceOccupancy, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    resident_inputs: tuple[str, ...] = ()
    resident_outputs: tuple[str, ...] = ()
    scratch_bytes: int = 0
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.implementation.strip() or not self.placement.strip():
            raise ValueError("a region alternative must name its id, implementation, and placement")
        if not self.nodes or tuple(sorted(set(self.nodes))) != self.nodes or self.nodes[0] < 0:
            raise ValueError(f"alternative {self.id!r} must cover sorted, unique node indices")
        # Structural transformations and their emission proofs do not require a timing oracle.
        # Keep uncalibrated alternatives expressible; the cycle-ranking planner refuses them.
        if self.scratch_bytes < 0:
            raise ValueError("scratch_bytes must be non-negative")
        for side, reps in (("input", self.inputs), ("output", self.outputs)):
            names = [item.buffer for item in reps]
            if len(names) != len(set(names)):
                raise ValueError(f"alternative {self.id!r} repeats a {side} buffer representation")
        if tuple(sorted(self.metadata)) != self.metadata:
            raise ValueError("alternative metadata must be sorted for stable content addressing")
        resources = [item.resource for item in self.occupancy]
        if len(resources) != len(set(resources)):
            raise ValueError(f"alternative {self.id!r} repeats a resource occupancy")
        if self.cycles.resolved and any(
                float(item.cycles.hi) > float(self.cycles.hi) for item in self.occupancy):
            raise ValueError(
                f"alternative {self.id!r} has resource occupancy longer than its region latency")

    def input_representation(self, buffer: str) -> ValueRepresentation | None:
        return next((item.representation for item in self.inputs if item.buffer == buffer), None)

    def output_representation(self, buffer: str) -> ValueRepresentation | None:
        return next((item.representation for item in self.outputs if item.buffer == buffer), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "nodes": list(self.nodes),
            "implementation": self.implementation,
            "placement": self.placement,
            "cycles": self.cycles.to_dict(),
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": [item.to_dict() for item in self.outputs],
            "demands": [item.to_dict() for item in self.demands],
            "occupancy": [item.to_dict() for item in self.occupancy],
            "required_capabilities": list(self.required_capabilities),
            "resident_inputs": list(self.resident_inputs),
            "resident_outputs": list(self.resident_outputs),
            "scratch_bytes": self.scratch_bytes,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TransitionAlternative:
    """Explicit work needed to connect two selected region implementations."""

    id: str
    kind: str
    buffer: str
    producer: str | None
    consumer: str | None
    source: ValueRepresentation
    destination: ValueRepresentation
    cycles: CycleInterval
    demands: tuple[PlanDemand, ...] = ()
    occupancy: tuple[ResourceOccupancy, ...] = ()
    materializes: bool = True
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.kind.strip() or not self.buffer.strip():
            raise ValueError("a transition must name its id, kind, and buffer")
        if not self.cycles.resolved:
            raise ValueError(f"transition {self.id!r} has an unresolved cost: {self.cycles.missing}")
        if tuple(sorted(self.metadata)) != self.metadata:
            raise ValueError("transition metadata must be sorted for stable content addressing")
        resources = [item.resource for item in self.occupancy]
        if len(resources) != len(set(resources)):
            raise ValueError(f"transition {self.id!r} repeats a resource occupancy")
        if any(float(item.cycles.hi) > float(self.cycles.hi) for item in self.occupancy):
            raise ValueError(
                f"transition {self.id!r} has resource occupancy longer than its latency")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "buffer": self.buffer,
            "producer": self.producer,
            "consumer": self.consumer,
            "source": self.source.to_dict(),
            "destination": self.destination.to_dict(),
            "cycles": self.cycles.to_dict(),
            "demands": [item.to_dict() for item in self.demands],
            "occupancy": [item.to_dict() for item in self.occupancy],
            "materializes": self.materializes,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GlobalPlan:
    """One complete, compatible implementation of a ``DispatchProgram``."""

    selected: tuple[RegionAlternative, ...]
    transitions: tuple[TransitionAlternative, ...]
    cycles: CycleInterval
    arena_bytes: int = 0
    arena_offsets: tuple[tuple[str, int], ...] = ()
    notes: tuple[str, ...] = ()
    schema: str = GLOBAL_PLAN_SCHEMA

    @property
    def digest(self) -> str:
        body = json.dumps(self.to_dict(include_digest=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    @property
    def required_capabilities(self) -> tuple[str, ...]:
        return tuple(sorted({cap for alternative in self.selected
                             for cap in alternative.required_capabilities}))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        out = {
            "schema": self.schema,
            "selected": [item.to_dict() for item in self.selected],
            "transitions": [item.to_dict() for item in self.transitions],
            "cycles": self.cycles.to_dict(),
            "arena_bytes": self.arena_bytes,
            "arena_offsets": dict(self.arena_offsets),
            "required_capabilities": list(self.required_capabilities),
            "notes": list(self.notes),
        }
        if include_digest:
            out["digest"] = self.digest
        return out


def verify_global_plan(program: DispatchProgram, plan: GlobalPlan) -> list[str]:
    """Return every structural problem; an empty list means the plan covers the DAG exactly."""

    problems: list[str] = []
    owners: dict[int, str] = {}
    ids: set[str] = set()
    for alternative in plan.selected:
        if alternative.id in ids:
            problems.append(f"duplicate selected alternative id {alternative.id!r}")
        ids.add(alternative.id)
        for index in alternative.nodes:
            if index >= len(program.nodes):
                problems.append(f"alternative {alternative.id!r} references absent node {index}")
            elif index in owners:
                problems.append(
                    f"node {index} is covered by both {owners[index]!r} and {alternative.id!r}")
            else:
                owners[index] = alternative.id
        available = set(program.buffers)
        for rep in (*alternative.inputs, *alternative.outputs):
            if rep.buffer not in available:
                problems.append(
                    f"alternative {alternative.id!r} names absent buffer {rep.buffer!r}")
    missing = sorted(set(range(len(program.nodes))) - set(owners))
    if missing:
        problems.append(f"plan leaves node(s) uncovered: {missing}")

    transition_ids: set[str] = set()
    for transition in plan.transitions:
        if transition.id in transition_ids:
            problems.append(f"duplicate transition id {transition.id!r}")
        transition_ids.add(transition.id)
        if transition.buffer not in program.buffers:
            problems.append(f"transition {transition.id!r} names absent buffer {transition.buffer!r}")
        if transition.producer is not None and transition.producer not in ids:
            problems.append(f"transition {transition.id!r} has unknown producer {transition.producer!r}")
        if transition.consumer is not None and transition.consumer not in ids:
            problems.append(f"transition {transition.id!r} has unknown consumer {transition.consumer!r}")

    if plan.arena_bytes < 0:
        problems.append("plan arena_bytes is negative")
    offsets = dict(plan.arena_offsets)
    if len(offsets) != len(plan.arena_offsets):
        problems.append("plan repeats an arena offset")
    for buffer, offset in plan.arena_offsets:
        if buffer not in program.buffers:
            problems.append(f"arena offset names absent buffer {buffer!r}")
        if offset < 0 or offset > plan.arena_bytes:
            problems.append(f"arena offset for {buffer!r} is outside the arena")
    return problems

"""From an emitted instruction to the compiler seam that can change it.

"Where did this assembly come from" has two readings and only one of them is useful. The literal
reading — which pass emitted this byte — needs per-stage IR snapshots or debug locations threaded end
to end, and even when you have it the answer ("instruction selection") names a stage nobody can edit.
The useful reading is: *this instruction is here because of a decision, and the decision has an owner.*

So the chain is built out of things already declared, and every link is checkable:

    instruction --(endpoint role table)--> ROLE
                --(role-to-facet map)-----> CCA AXIS
                --(regions.cca_axes)------> COMPILER REGION
                --(region.edit_points)----> the seam that can change it

The last link is what makes it provenance rather than trivia. A divergence traced to a role whose
region has no forkable edit point is a divergence nobody can act on, and saying so is more useful than
routing it to a seam that does not exist. That is the same discipline the expert trace applies to the
OTHER side: their steps are stamped unmodifiable because we cannot edit their compiler.

⚠️ This does NOT claim an instruction-level attribution to a pass. It claims a decision-level
attribution to a seam, and the difference is stated because a reader who assumed the former would
believe the tool can localize a codegen bug, which it cannot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["RoleProvenance", "provenance_of_role", "provenance_table", "unowned_roles"]

#: role -> the CCA axes that role feeds. Derived from what each facet MEANS, and checked against the
#: live registry: an axis named here that the contract does not classify is a hard error rather than a
#: silently dead row (see :func:`check_axes_exist`).
ROLE_AXES: dict[str, tuple[str, ...]] = {
    "accumulate": ("compute.contraction_form", "compute.accumulator_resident",
                   "spatial.accumulator_resident", "compute.widening"),
    "operand_load": ("memory.access_pattern", "layout.operand_major", "memory.onchip_resident"),
    "weight_load": ("spatial.dataflow", "layout.prepack_required"),
    "broadcast": ("memory.a_broadcast_vf", "compute.register_block"),
    "readout": ("compute.accumulator_resident", "compute.epilogue"),
    "commit": ("compute.epilogue", "memory.capacity_fit"),
    "config": ("vector.lmul", "vector.sew", "vector.vl_strategy", "vector.tail",
               "dispatch.config_fraction", "dispatch.descriptor_reuse"),
    "loop_descriptor": ("dispatch.loop_offloaded", "dispatch.n_dispatches"),
    "sync": ("simt.barriers_in_loop", "dispatch.dma_overlap"),
    "dma": ("memory.dma_pattern", "dispatch.dma_overlap", "dispatch.double_buffered_banks"),
    "elementwise": ("compute.activation_vectorization", "coverage.non_contraction_op_fraction"),
    "move": ("memory.a_broadcast_vf", "compute.register_block"),
    # Control flow is the ENVELOPE: the code around the loop, whose cost is real (measured: a per-tile
    # runtime call was ~77% of everything retired) and whose axes are the envelope metrics.
    "control": ("envelope.calls_in_loop", "envelope.runtime_calls"),
}


@dataclass(frozen=True)
class RoleProvenance:
    """Why an instruction of this role is in our output, and who can change that."""

    role: str
    axes: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    #: (seam, file, forkable) for every edit point of every governing region.
    edit_points: tuple[tuple[str, str, bool], ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def actionable(self) -> bool:
        """Is there a seam that can be forked TODAY? A region whose only edit point is a stated GAP
        governs the axis without offering a way to change it, and a divergence routed there is a
        finding rather than a task."""
        return any(forkable for _s, _f, forkable in self.edit_points)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "axes": list(self.axes), "regions": list(self.regions),
                "edit_points": [{"seam": s, "file": f, "forkable_now": k}
                                for s, f, k in self.edit_points],
                "actionable": self.actionable, "notes": list(self.notes)}


def provenance_of_role(role: str) -> RoleProvenance:
    """The full chain for one role, read from the live region taxonomy."""
    from merlin.kernels import regions as _regions
    from merlin.kernels import roles as _roles

    if not _roles.is_role(role):
        raise KeyError(f"unknown instruction role {role!r}; known: {sorted(_roles.ROLES)}")
    axes = ROLE_AXES.get(role, ())
    regs: list[str] = []
    eps: list[tuple[str, str, bool]] = []
    for key, region in _regions.REGIONS.items():
        if not set(axes) & set(region.cca_axes):
            continue
        regs.append(key)
        for ep in region.edit_points:
            # The attribute is `forkable_now`. Reading a non-existent name with a True default made
            # every declared GAP report as actionable -- a seam nobody can fork looked like a task,
            # which is the exact inversion this chain exists to prevent. No default: a missing
            # attribute is a bug in this reader, not a fork-ready seam.
            row = (ep.seam, ep.file, bool(ep.forkable_now))
            if row not in eps:
                eps.append(row)
    notes: list[str] = []
    if axes and not regs:
        notes.append(f"role {role!r} feeds {list(axes)} and NO region governs any of them: an "
                     f"instruction of this role cannot be traced to an owner")
    if regs and not any(k for _s, _f, k in eps):
        notes.append(f"every edit point governing {role!r} is a stated GAP: the axis is owned but "
                     f"nothing can be forked today, so a divergence here is a finding, not a task")
    return RoleProvenance(role=role, axes=tuple(axes), regions=tuple(sorted(regs)),
                          edit_points=tuple(eps), notes=tuple(notes))


def provenance_table() -> dict[str, dict]:
    """The whole chain, for every role in the vocabulary."""
    from merlin.kernels import roles as _roles
    return {r: provenance_of_role(r).to_dict() for r in sorted(_roles.ROLES)}


def unowned_roles() -> tuple[str, ...]:
    """Roles no compiler region governs — instructions we can read but cannot attribute to an owner."""
    from merlin.kernels import roles as _roles
    return tuple(r for r in sorted(_roles.ROLES) if not provenance_of_role(r).regions)


def check_axes_exist() -> list[str]:
    """Every axis named above must be a real, classified CCA field.

    A dead row here reads as coverage: the chain would report an axis for a role and the axis would
    resolve to nothing downstream, which looks like provenance and is not.
    """
    from merlin.kernels.cca_contract import FIELD_REGISTRY
    problems = []
    for role, axes in sorted(ROLE_AXES.items()):
        for axis in axes:
            if axis not in FIELD_REGISTRY:
                problems.append(f"role {role!r} names axis {axis!r}, which the CCA contract does not "
                                f"classify — a dead row that reads as coverage")
    return problems


# ---------------------------------------------------------------------------------------------
# What to try next, derived from what the assembly actually shows
# ---------------------------------------------------------------------------------------------
#
# Derived, never hand-listed. A hand-written list of "optimizations to try" goes stale the moment the
# compiler changes and cannot say WHY any entry is on it; these come from the observed role histogram,
# so each carries the count that put it there and disappears when the shape does.


@dataclass(frozen=True)
class Opportunity:
    """One candidate optimization, with the observation that justifies it and the seam that does it."""

    axis: str
    observation: str                 # the measured shape in the assembly
    change: str                      # what to try
    seam: str = ""
    forkable_now: bool = False
    #: forkable | seam_is_a_gap | metric_not_a_lever | ungoverned — see :func:`_seam_for`.
    status: str = "ungoverned"
    confidence: str = "medium"       # how directly the assembly supports it

    def to_dict(self) -> dict[str, Any]:
        return {"axis": self.axis, "observation": self.observation, "change": self.change,
                "seam": self.seam, "forkable_now": self.forkable_now, "status": self.status,
                "confidence": self.confidence}


def _seam_for(axis: str) -> tuple[str, bool, str]:
    """``(seam, forkable_now, status)`` for an axis. Three states, not two.

    "no region governs this" and "the governing region's only seam is a declared gap" are different
    problems with different fixes, and reporting both as ``forkable=False`` makes an opportunity
    unactionable for reasons the reader cannot distinguish. An axis classified METRIC is a third case
    again: it DIAGNOSES rather than changes, so an opportunity that names one is pointing at a
    thermometer and calling it a dial.
    """
    from merlin.kernels import regions as _regions
    from merlin.kernels.cca_contract import FIELD_REGISTRY

    spec = FIELD_REGISTRY.get(axis)
    for region in _regions.REGIONS.values():
        if axis in region.cca_axes and region.edit_points:
            ep = next((e for e in region.edit_points if e.forkable_now), region.edit_points[0])
            return ep.seam, bool(ep.forkable_now), ("forkable" if ep.forkable_now else "seam_is_a_gap")
    if spec is not None and spec.classification == "METRIC":
        return "", False, "metric_not_a_lever"
    return "", False, "ungoverned"


#: Semantic families whose regions are contractions. A role histogram alone CANNOT distinguish an
#: unfused matmul from an activation kernel — both are "loads and arithmetic with no accumulate" — so
#: the rule that proposes a fused multiply-accumulate needs the region's family, and says so when it
#: does not have it rather than proposing a contraction rewrite for an activation.
_CONTRACTION_FAMILIES = frozenset({"contraction", "attention"})


def opportunities(hist: dict, *, engine: str = "", total: int = 0,
                  family: str | None = None) -> list[Opportunity]:
    """Candidate optimizations implied by one stream's role histogram.

    Every rule below is a SHAPE in the assembly, not a heuristic about what is usually good: a stream
    that shuffles operands more than it multiplies, a contraction that never drains its accumulator, a
    reduction re-configured on every step. Each is visible only because the instructions carry roles,
    which is the whole reason the role vocabulary exists.
    """
    out: list[Opportunity] = []
    g = hist.get
    acc, mul_like = g("accumulate", 0), g("elementwise", 0)
    shuffle = g("broadcast", 0) + g("move", 0)

    def _add(axis, observation, change, confidence="medium"):
        seam, forkable, status = _seam_for(axis)
        out.append(Opportunity(axis=axis, observation=observation, change=change, seam=seam,
                               forkable_now=forkable, status=status, confidence=confidence))

    if acc == 0 and mul_like > 0 and family in _CONTRACTION_FAMILIES:
        _add("compute.contraction_form",
             f"a {family} region with {mul_like} elementwise op(s) and ZERO multiply-accumulate: the "
             f"arithmetic is a multiply followed by an add, not a fused MAC",
             "select the fused multiply-accumulate form so each step advances the partial sum in one "
             "instruction instead of two", confidence="high")
    elif acc == 0 and mul_like > 0 and family is None and engine != "vector":
        # No family given, so this MIGHT be an unfused contraction or might be an activation kernel
        # doing exactly what it should. Reported at low confidence with the ambiguity named, never as
        # a recommendation -- proposing a contraction rewrite for an activation is worse than silence.
        _add("compute.contraction_form",
             f"{mul_like} elementwise op(s) and ZERO multiply-accumulate, and the region's family is "
             f"UNKNOWN: this is either an unfused contraction or an elementwise kernel behaving "
             f"correctly, and a role histogram cannot tell them apart",
             "establish the region's semantic family, then re-ask — if it contracts, fuse the "
             "multiply-add", confidence="low")
    offloaded = g("loop_descriptor", 0) > 0
    if acc and not g("readout", 0) and not offloaded:
        _add("compute.accumulator_resident",
             f"{acc} accumulate(s) and NO readout: the accumulator is never drained in this stream",
             "check the epilogue actually extracts the result — an accumulate-without-extraction is "
             "the documented way a kernel audits clean while computing nothing usable",
             confidence="high")
    if shuffle and acc and shuffle > 2 * acc:
        _add("compute.register_block",
             f"{shuffle} operand shuffle(s) (broadcast+move) against {acc} accumulate(s): the stream "
             f"spends more instructions moving operands than multiplying them",
             "raise the register block so one loaded panel feeds several accumulators, instead of "
             "rebuilding the operand vector per step", confidence="high")
    if shuffle and acc == 0 and shuffle > 8:
        # Cites the metric as the OBSERVATION and names the lever as the change. a_broadcast_vf is
        # classified METRIC: it diagnoses the shape, it is not a dial anyone can turn.
        _add("compute.register_block",
             f"{shuffle} operand shuffle(s) and no accumulate at all (memory.a_broadcast_vf shape): "
             f"this stream is a data ladder rebuilding operands rather than computing",
             "broadcast the scalar operand into the MAC and block the register tile so one loaded "
             "panel feeds several accumulators", confidence="medium")
    cfg = g("config", 0)
    if cfg and acc and cfg > acc:
        _add("dispatch.descriptor_reuse",
             f"{cfg} configuration instruction(s) against {acc} accumulate(s): endpoint state is being "
             f"re-set more often than it is used",
             "hoist the configuration out of the loop so the state is set once and inherited",
             confidence="high")
    if engine == "spatial" and acc and not offloaded:
        _add("dispatch.loop_offloaded",
             f"{acc} accumulate(s) issued command-by-command with no loop descriptor",
             "hand the loop nest to the endpoint's own sequencer instead of issuing every step",
             confidence="medium")
    if engine == "simt" and g("sync", 0) and acc:
        _add("simt.barriers_in_loop",
             f"{g('sync', 0)} barrier/fence(s) alongside {acc} accumulate(s)",
             "hoist barriers out of the reduction loop — one inside says the engine cannot hold its "
             "state across the reduction", confidence="medium")
    if total and g("control", 0) > total * 0.25:
        _add("envelope.calls_in_loop",
             f"{g('control', 0)} of {total} instructions are control flow ({g('control', 0)/total:.0%})",
             "the envelope dominates: check for a per-tile runtime call or an unfused loop nest — "
             "measured elsewhere, a per-tile copy was ~77% of everything retired", confidence="medium")
    return out

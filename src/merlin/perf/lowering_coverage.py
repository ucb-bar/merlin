"""Where every operation actually landed, and -- when it landed off the accelerator -- why.

WHY THIS EXISTS. A compiler that silently falls back to the host is indistinguishable, from its own
output, from one that had no accelerator at all. Measured on a whole-model ResNet-50: 119 operations
ran as host regions -- 50 quantize, 49 min/max clamp, 17 residual adds, 1 max-pool, 1 reduction --
every one of which the vendor library performs on the accelerator, and the emitted program said
nothing about it. The compiler was not asked to justify a single one. On a language model the same
compiler placed 5,558 regions on the host and declined the whole program only because one of them
exceeded a straight-line budget; had that budget been larger it would have shipped quietly.

A fallback is sometimes physically necessary -- an ISA with no vector-map opcode cannot host a
standalone element-wise map, and a data-dependent gather has nowhere else to go. The defect is not
that fallbacks exist. It is that they are INVISIBLE and UNJUSTIFIED: nothing counts them, nothing
states a reason, and nothing distinguishes "this hardware cannot express it" from "this compiler
did not try".

So this module makes every placement explicit and every off-accelerator placement carry a reason.
:func:`census` reports coverage as a number. :func:`unjustified` is the one that matters under a
no-fallback policy: it returns the operations that went to the host with NO stated reason, which is
the population a strict build should refuse to emit.

:func:`offload_verdict` asks the harder question the census cannot: not "was each fallback stated"
but "did the target run the part of this model it CAN run". It is separate because it needs one
fact a placement does not carry -- whether a unit for that class of work exists at all -- and
because getting that fact from the wrong side of a refusal is how a zero-offload gate came to be
unable to fire. See the comment above it.

Target-neutral: operation names, placements and reasons are opaque strings supplied by the caller
from its own lowering. This module never names a target, an operation, or a lane.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

from .gate_phase import PHASE_REPORT, PHASES, STATUS_INCOMPLETE, blocks

__all__ = [
    "Placement",
    "ACCELERATOR",
    "HOST",
    "census",
    "unjustified",
    "coverage_gate",
    "NO_UNIT",
    "REFUSED_ON_PROPERTY",
    "ELIGIBILITY_UNKNOWN",
    "OFF_ACCELERATOR_CAUSES",
    "STATUS_OFFLOADED",
    "STATUS_ZERO_OFFLOAD",
    "ZeroOffloadError",
    "off_accelerator_cause",
    "offload_verdict",
    "require_offload",
]

#: The operation is executed by the accelerator (including as a stage fused into another op's
#: readout, which on some targets is the only way an element-wise stage can reach it at all).
ACCELERATOR = "accelerator"
#: The operation is executed by the host processor.
HOST = "host"


class Placement(NamedTuple):
    """One operation's landing site.

    `reason` is required for a host placement and ignored for an accelerator one. `family` and
    `dtype` are opaque labels used only for grouping, so a caller may pass whatever taxonomy its
    own capability manifest uses.
    """

    operation: str
    placement: str
    family: str | None = None
    dtype: str | None = None
    reason: str | None = None


def _rows(items: Iterable[Placement | Sequence[Any]]) -> list[Placement]:
    out: list[Placement] = []
    for row in items:
        out.append(row if isinstance(row, Placement) else Placement(*row))
    return out


def unjustified(placements: Iterable[Placement | Sequence[Any]]) -> list[Placement]:
    """Host placements with no stated reason -- what a no-fallback policy must refuse.

    A blank or whitespace-only reason counts as absent: "" is not a justification, and treating it
    as one is how an unexplained fallback survives review.
    """
    return [row for row in _rows(placements) if row.placement == HOST and not str(row.reason or "").strip()]


def census(placements: Iterable[Placement | Sequence[Any]]) -> dict[str, Any]:
    """Coverage as a number, plus the reason census for everything that fell back."""
    rows = _rows(placements)
    accel = [r for r in rows if r.placement == ACCELERATOR]
    host = [r for r in rows if r.placement == HOST]
    other = [r for r in rows if r.placement not in (ACCELERATOR, HOST)]
    reasons = Counter(str(r.reason).strip() for r in host if str(r.reason or "").strip())
    by_family: dict[str, dict[str, int]] = {}
    for row in rows:
        entry = by_family.setdefault(str(row.family), {ACCELERATOR: 0, HOST: 0})
        if row.placement in entry:
            entry[row.placement] += 1
    return {
        "schema": "lowering_coverage_v1",
        "operations": len(rows),
        "on_accelerator": len(accel),
        "on_host": len(host),
        # A placement this module does not recognise is surfaced, never bucketed into one of the
        # two known lanes -- guessing would make coverage look better than it is.
        "unclassified": [r.operation for r in other],
        "coverage": round(len(accel) / len(rows), 6) if rows else None,
        "host_reasons": [
            {"reason": reason, "operations": count}
            for reason, count in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "unjustified_host_operations": [r.operation for r in unjustified(rows)],
        "by_family": by_family,
    }


def coverage_gate(placements: Iterable[Placement | Sequence[Any]], *, allow_fallback: bool) -> dict[str, Any]:
    """Verdict under a declared fallback policy.

    With ``allow_fallback=False`` a build is admitted only if every host placement carries a
    reason. The policy deliberately does NOT require zero host placements: some are physically
    forced, and a rule demanding zero would be unsatisfiable and therefore ignored. What it demands
    is that each one be stated, so a reader can tell a hardware limit from an unfinished compiler.
    """
    summary = census(placements)
    blocking = summary["unjustified_host_operations"]
    admitted = bool(allow_fallback) or not blocking
    return {
        **summary,
        "allow_fallback": bool(allow_fallback),
        "admitted": admitted,
        "blocking_operations": [] if allow_fallback else blocking,
        "licence": (
            "a fallback is not forbidden -- some are forced by the ISA -- but under "
            "allow_fallback=False every one must NAME its reason, so that 'the hardware "
            "cannot express this' is distinguishable from 'the compiler did not try'."
        ),
    }


# --------------------------------------------------------------------------------------------
# Did the target run the part of this model it CAN run?
#
# Coverage alone cannot answer that, and a gate built on coverage alone cannot fail. The rule
# everyone writes first is "eligible work and none of it on a unit is a defect", with the sound
# exemption "a model with NO eligible work passes -- the absence of an accelerator's work is only a
# defect when there was some to do". The exemption is right and the derivation of `eligible` is
# where it goes wrong.
#
# MEASURED, on three separate ResNet-50 captures, byte-identically: 231 regions, 4.09e9 MACs, ZERO
# on the accelerator, 160 contractions refused on one clause -- the operand precision read off the
# module -- and the gate did not fire. `eligible` had been computed DOWNSTREAM of that refusal, so
# the same refusal that put 100% of the work on the host also made that work ineligible, and the
# exemption was manufactured by the very failure the gate exists to catch. The report read
# `coverage 0.000, admitted True`, which is a sentence that gets cited.
#
# So the population splits three ways, and collapsing any two of them is how the gate dies:
#
#   NO_UNIT               the target has no unit for this CLASS of work. Genuinely nothing to do
#                         here, and the only thing that may exempt a model.
#   REFUSED_ON_PROPERTY   a unit for this class EXISTS and refused this instance for a property --
#                         its precision, its layout, the granularity of its scales. This is the gap,
#                         and it is exactly what the ResNet-50 census was folding into "not
#                         eligible".
#   ELIGIBILITY_UNKNOWN   nobody decided. A capture whose precision is not yet expressed in the form
#                         the backend will place from is the live case: the module says fp32 because
#                         integer preparation has not run, and reading 0.000 coverage off it is not
#                         a conservative measurement, it is a wrong one. UNKNOWN propagates to the
#                         whole verdict and makes it `incomplete`, never a pass and never a zero.
# --------------------------------------------------------------------------------------------

#: The target declares no unit for this class of work. The ONLY cause that may exempt a model.
NO_UNIT = "no_unit_for_this_class"
#: A unit for this class exists and refused this instance for a property it carries.
REFUSED_ON_PROPERTY = "refused_on_property"
#: Nobody could decide. Propagates to the verdict; never folded into either of the others.
ELIGIBILITY_UNKNOWN = "eligibility_unknown"

#: The complete vocabulary. A cause outside it is treated as :data:`ELIGIBILITY_UNKNOWN`, so a new
#: refusal kind cannot quietly become an exemption by not being listed.
OFF_ACCELERATOR_CAUSES = (NO_UNIT, REFUSED_ON_PROPERTY, ELIGIBILITY_UNKNOWN)

#: Some addressable work reached a unit.
STATUS_OFFLOADED = "offloaded"
#: Work a unit could have taken, and none of it reached one. The decided failure.
STATUS_ZERO_OFFLOAD = "zero_offload"

#: The decided failures of this gate. `incomplete` is not among them on purpose.
FAILING_OFFLOAD_STATUSES = (STATUS_ZERO_OFFLOAD,)


class ZeroOffloadError(RuntimeError):
    """The target could have run part of this model and the compiler put none of it there."""


def off_accelerator_cause(*, class_has_unit: bool | None, precision_stated: bool = True) -> str:
    """Which of the three causes a host placement is, from the two facts that decide it.

    ONE function, because the classification is where the gate dies and two callers deriving it
    separately will derive it differently. Both inputs are supplied by the caller from its own
    capability model; this module knows no families, no formats and no units.

    ``class_has_unit`` -- does the target declare a unit for this CLASS of work? ``None`` means
    nobody established it, and that is not the same as ``False``: an underivable capability read as
    "no unit" is an exemption granted by not looking.

    ``precision_stated`` -- was the operand precision this decision rests on the one the BACKEND
    WILL PLACE FROM? A capture whose quantization is still expressed as quantize/dequantize pairs
    around wide tensors carries the pre-preparation type, and integer preparation consumes those
    pairs later in the pipeline. A refusal decided on that type is not a refusal of the program the
    backend compiles, so it decides nothing: measured on one model, every contraction was refused on
    a precision clause while the router -- which is TOLD the precision rather than reading it --
    placed 54 of them. Two consumers of one module, one inferring and one told, and nothing held
    them to the same answer. So an unstated precision is UNKNOWN, never a property refusal and never
    an exemption.
    """
    if class_has_unit is None:
        return ELIGIBILITY_UNKNOWN
    if not class_has_unit:
        return NO_UNIT
    return REFUSED_ON_PROPERTY if precision_stated else ELIGIBILITY_UNKNOWN


def _cause(row: Mapping[str, Any]) -> str:
    cause = row.get("off_accelerator_cause")
    return cause if cause in OFF_ACCELERATOR_CAUSES else ELIGIBILITY_UNKNOWN


def _work(row: Mapping[str, Any]) -> int:
    """This region's weight. MACs when the extents are known, otherwise the region itself.

    Falling back to 1 rather than 0 keeps a region with unknown extents in the population: a model
    whose every extent is unknown would otherwise weigh nothing and pass by having been unreadable.
    """
    macs = row.get("macs")
    if isinstance(macs, bool) or not isinstance(macs, int) or macs < 0:
        return 1
    return macs


def offload_verdict(regions: Iterable[Mapping[str, Any]], *, phase: str = PHASE_REPORT) -> dict[str, Any]:
    """Whether the work a unit COULD have taken reached one, phased.

    Each region supplies ``placement`` and, when it is not on the accelerator, an
    ``off_accelerator_cause`` from :data:`OFF_ACCELERATOR_CAUSES`, plus an optional ``macs``.
    An absent or unrecognised cause is :data:`ELIGIBILITY_UNKNOWN`: a gate must not be able to
    acquire an exemption by being told nothing.
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    rows = list(regions)
    on_unit = [r for r in rows if r.get("placement") == ACCELERATOR]
    off = [r for r in rows if r.get("placement") != ACCELERATOR]
    by_cause: Counter = Counter(_cause(r) for r in off)
    work_by_cause: dict[str, int] = {}
    for row in off:
        cause = _cause(row)
        work_by_cause[cause] = work_by_cause.get(cause, 0) + _work(row)

    on_unit_work = sum(_work(r) for r in on_unit)
    refused_work = work_by_cause.get(REFUSED_ON_PROPERTY, 0)
    unknown_work = work_by_cause.get(ELIGIBILITY_UNKNOWN, 0)
    # Addressable = the target HAS a unit for this class. Work on a unit proves it, and work a unit
    # refused for a property proves it just as well -- refusing an instance is something only a unit
    # that exists can do. This is the line the ResNet-50 census had in the wrong place.
    addressable_work = on_unit_work + refused_work

    def verdict(status: str, reason: str) -> dict[str, Any]:
        return {
            "schema": "offload_verdict_v1",
            "phase": phase,
            "status": status,
            "reason": reason,
            "regions": len(rows),
            "on_accelerator": len(on_unit),
            "off_accelerator_by_cause": dict(sorted(by_cause.items())),
            "work_on_accelerator": on_unit_work,
            "work_refused_on_property": refused_work,
            "work_addressable": addressable_work,
            "work_eligibility_unknown": unknown_work,
            # None, not 0.0, with an empty denominator: a ratio nobody could form is not a measured
            # zero, and a reader must be able to tell a model with no acceleratable work from a
            # compiler that accelerated none of it.
            "offload_of_addressable": (round(on_unit_work / addressable_work, 6) if addressable_work else None),
            "blocking": blocks(phase, status, failing=FAILING_OFFLOAD_STATUSES),
            "admitted": status == STATUS_OFFLOADED,
        }

    if unknown_work:
        return verdict(
            STATUS_INCOMPLETE,
            (
                f"{by_cause[ELIGIBILITY_UNKNOWN]} region(s) carrying {unknown_work} unit(s) of work are "
                f"off the accelerator for a cause nobody decided. Reporting the offload ratio over the "
                f"rest would present a coverage figure as a measurement when the population it is over "
                f"is not established -- and an unknown that reads as 0% offload is a sentence that gets "
                f"cited"
            ),
        )
    if not addressable_work:
        return verdict(
            STATUS_OFFLOADED,
            (
                "no region belongs to a class this target declares a unit for, so there was no offload "
                "to miss. This is the only exemption, and it rests on there being no unit -- never on a "
                "unit having refused the work"
            ),
        )
    if not on_unit_work:
        return verdict(
            STATUS_ZERO_OFFLOAD,
            (
                f"{refused_work} unit(s) of work belong to classes this target HAS a unit for, every "
                f"one of them was refused on a property rather than for want of a unit, and nothing "
                f"reached the accelerator"
            ),
        )
    return verdict(STATUS_OFFLOADED, f"{on_unit_work} of {addressable_work} addressable unit(s) of work reached a unit")


def require_offload(verdict: Mapping[str, Any]) -> None:
    """Raise :class:`ZeroOffloadError` for a verdict that BLOCKS.

    Blocking is the verdict's own decision, so the phase is honoured here rather than re-derived:
    at ``report`` this raises for nothing, which is the point of landing there first.
    """
    if verdict.get("blocking"):
        raise ZeroOffloadError(str(verdict.get("reason")))

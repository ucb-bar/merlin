"""Will the target's readout actually APPLY the epilogue this program declares?

THE DEFECT, MEASURED. A capsule declared ``COMMIT output_dtype='i32' epilogue=['relu']`` and its
device output came back with 126 of 256 values negative, ``min = -85`` -- exactly the raw accumulator.
The compiler's intent was right (the command-buffer numeric floor and trace both passed); the
hardware simply discarded the activation, because on that target the full-width readout writes the
accumulator unmodified while only the narrowing readout applies scale and activation. The grade's own
diagnosis named the shape of the cause without knowing it: *"some field the command buffer cannot
carry (a config scale, an accumulate/dataflow bit, a readout dtype)"*.

It went unnoticed for the worst possible reason: the sibling capsule with the SAME declared epilogue
passes, because the default stimulus is non-negative, so the accumulator is never negative and
``max(0, x)`` is the identity on every value that program can produce. A declared-but-discarded
activation is invisible unless the data can tell.

WHY THIS IS NOT A FACT ABOUT ONE TARGET. Every accelerator with more than one readout width has this
shape: a readout that requantizes applies the epilogue, a readout that dumps the accumulator does
not, and which is which is a property of that datapath. So the RULE is stated here and the
CAPABILITY is supplied by the caller from the target's own declaration -- exactly as
``counter_engine_kinds`` and the row pitch are. Nothing in this module names a target, a dtype width
or an opcode, and a readout the target does not describe is UNKNOWN rather than assumed applicable:
assuming would reproduce the silent-discard defect on the next target instead of catching it.

WHAT A CALLER DOES WITH THE VERDICT. :data:`REFUSING_STATUSES` mirrors
:mod:`merlin.perf.lowering_obligation` so a caller decides whether a status is fatal for its tier. A
``discarded`` verdict means the emitted program computes something other than what it declares, which
is a correctness defect and not a performance one -- but flipping a long-passing capsule to failing
is a corpus decision, so this module reports and the caller chooses.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ReadoutCapability", "StageVerdict", "Assessment", "assess", "STATUSES",
           "REFUSING_STATUSES"]

#: Every verdict this module can reach.
STATUSES: tuple[str, ...] = (
    "applied",     # every declared stage is applied by the readout the program selected
    "discarded",   # the readout does NOT apply a declared stage: the program computes something else
    "unknown",     # the target described no readout matching this program's; refuse, never assume
    "not_applicable",  # the program declares no epilogue, so there is nothing to apply
)

#: Statuses a caller should treat as fatal. ``unknown`` is here on purpose: a readout nobody
#: described is the state in which the original defect was invisible.
REFUSING_STATUSES: frozenset[str] = frozenset({"discarded", "unknown"})


@dataclass(frozen=True)
class ReadoutCapability:
    """One readout a target offers, and which epilogue stages it APPLIES.

    ``selector`` is the value a command's declared output dtype takes for this readout. It is
    compared as data, never parsed for a width: a target whose readouts are distinguished some other
    way declares that value here and nothing in this module has to learn how it is spelled.

    ``applies`` is the set of epilogue stages the readout genuinely performs. Stages the readout
    ignores are simply absent -- and being absent is what produces a ``discarded`` verdict, so a
    target that has not enumerated its stages gets the refusal rather than a pass.
    """

    selector: str
    applies: frozenset[str]
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"selector": self.selector, "applies": sorted(self.applies),
                "evidence": self.evidence}


@dataclass(frozen=True)
class StageVerdict:
    """One declared stage on one command, and whether the selected readout applies it."""

    command_index: int
    opcode: str
    readout: str
    stage: str
    applied: bool
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"command_index": self.command_index, "opcode": self.opcode,
                "readout": self.readout, "stage": self.stage, "applied": self.applied,
                "why": self.why}


@dataclass
class Assessment:
    """The verdict for one program, plus every stage that informed it."""

    status: str
    detail: str = ""
    stages: list[StageVerdict] = field(default_factory=list)
    readouts_declared: tuple[str, ...] = ()

    @property
    def refusing(self) -> bool:
        return self.status in REFUSING_STATUSES

    @property
    def discarded(self) -> tuple[StageVerdict, ...]:
        return tuple(v for v in self.stages if not v.applied)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "merlin_epilogue_applicability_v1", "status": self.status,
                "detail": self.detail, "refusing": self.refusing,
                "readouts_declared": list(self.readouts_declared),
                "n_discarded": len(self.discarded),
                "stages": [v.to_dict() for v in self.stages]}


def _epilogue_of(command: Mapping[str, Any]) -> tuple[str, ...]:
    stages = (command.get("attributes") or {}).get("epilogue")
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        return ()
    return tuple(str(s) for s in stages if str(s))


def _readout_of(command: Mapping[str, Any]) -> str | None:
    value = (command.get("attributes") or {}).get("output_dtype")
    return str(value) if isinstance(value, str) and value else None


def assess(command_buffer: Mapping[str, Any],
           readouts: Sequence[ReadoutCapability]) -> Assessment:
    """Whether every epilogue stage this program declares is applied by the readout it selected.

    ``readouts`` is the target's own declaration. Required and never defaulted: a program whose
    readout is undescribed gets ``unknown``, because the alternative -- assuming a readout applies
    whatever is asked of it -- is precisely how a discarded activation stays invisible.
    """
    by_selector = {r.selector: r for r in readouts}
    declared = tuple(sorted(by_selector))
    commands = command_buffer.get("commands")
    if not isinstance(commands, Sequence) or isinstance(commands, (str, bytes)):
        return Assessment(status="unknown", detail="the command buffer declares no command sequence",
                          readouts_declared=declared)

    stages: list[StageVerdict] = []
    unknown_readouts: set[str] = set()
    for index, command in enumerate(commands):
        if not isinstance(command, Mapping):
            continue
        epilogue = _epilogue_of(command)
        if not epilogue:
            continue
        opcode = str(command.get("opcode") or "")
        readout = _readout_of(command)
        if readout is None:
            unknown_readouts.add("<undeclared>")
            stages.extend(StageVerdict(index, opcode, "<undeclared>", stage, False,
                                       "the command declares epilogue stages but no readout, so "
                                       "which readout would apply them is UNKNOWN")
                          for stage in epilogue)
            continue
        capability = by_selector.get(readout)
        if capability is None:
            unknown_readouts.add(readout)
            stages.extend(StageVerdict(index, opcode, readout, stage, False,
                                       f"the target describes no readout {readout!r} "
                                       f"(it declares {list(declared)}), so whether it applies this "
                                       f"stage is UNKNOWN and is refused rather than assumed")
                          for stage in epilogue)
            continue
        for stage in epilogue:
            applied = stage in capability.applies
            stages.append(StageVerdict(
                index, opcode, readout, stage, applied,
                "" if applied else
                (f"readout {readout!r} does not apply {stage!r} (it applies "
                 f"{sorted(capability.applies)}){': ' + capability.evidence if capability.evidence else ''}"
                 f" -- the emitted program therefore computes something other than what it declares")))

    if not stages:
        return Assessment(status="not_applicable",
                          detail="the program declares no epilogue stage, so none can be discarded",
                          readouts_declared=declared)
    if unknown_readouts:
        return Assessment(
            status="unknown", stages=stages, readouts_declared=declared,
            detail=(f"readout(s) {sorted(unknown_readouts)} are not described by the target, so "
                    f"whether the declared stages are applied cannot be established"))
    dropped = [v for v in stages if not v.applied]
    if dropped:
        first = dropped[0]
        return Assessment(
            status="discarded", stages=stages, readouts_declared=declared,
            detail=(f"{len(dropped)} declared epilogue stage(s) are not applied by the readout the "
                    f"program selected; first at command {first.command_index} "
                    f"({first.opcode}): {first.why}"))
    return Assessment(status="applied", stages=stages, readouts_declared=declared,
                      detail=f"all {len(stages)} declared stage(s) are applied by their readout")

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
:mod:`merlin.analysis.perf.lowering_obligation` so a caller decides whether a status is fatal for its tier. A
``discarded`` verdict means the emitted program computes something other than what it declares, which
is a correctness defect and not a performance one -- but flipping a long-passing capsule to failing
is a corpus decision, so this module reports and the caller chooses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ReadoutCapability",
    "StageRoute",
    "StageVerdict",
    "Assessment",
    "assess",
    "selectors_applying",
    "STATUSES",
    "REFUSING_STATUSES",
]

#: Every verdict this module can reach.
STATUSES: tuple[str, ...] = (
    "applied",  # every declared stage is applied by its readout or a witnessed stage route
    "discarded",  # the readout does NOT apply a declared stage: the program computes something else
    "unknown",  # the target described no readout matching this program's; refuse, never assume
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
        return {"selector": self.selector, "applies": sorted(self.applies), "evidence": self.evidence}


@dataclass(frozen=True)
class StageRoute:
    """A non-readout stage applied on a particular composition path.

    The target declares the command-buffer producer and consumer spellings. A route never changes
    what a readout applies: it only licenses the stated stage when the program carries its operand
    and the nearest writer of the committed accumulator is an eligible producer.
    """

    stage: str
    site: str
    composed_with: str
    readouts: frozenset[str]
    producer_opcodes: frozenset[str]
    consumer_opcodes: frozenset[str]
    operand_attribute: str
    operand_role: str
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [self.stage],
            "site": self.site,
            "composed_with": self.composed_with,
            "readouts": sorted(self.readouts),
            "producer_opcodes": sorted(self.producer_opcodes),
            "consumer_opcodes": sorted(self.consumer_opcodes),
            "operand_attribute": self.operand_attribute,
            "operand_role": self.operand_role,
            "evidence": self.evidence,
        }


def _route_for(routes: Sequence[StageRoute], stage: str, selector: str, composition: str) -> StageRoute | None:
    return next(
        (
            route
            for route in routes
            if route.stage == stage
            and route.composed_with == composition
            and selector in route.readouts
            and route.site == "accumulator_seed"
        ),
        None,
    )


def _route_witness(
    route: StageRoute, command: Mapping[str, Any], earlier: Sequence[Any], tensors: Mapping[str, Any]
) -> bool:
    if str(command.get("opcode") or "") not in route.consumer_opcodes:
        return False
    if not _epilogue_of(command) or _epilogue_of(command)[0] != route.stage:
        return False  # a seed precedes all compute/readout stages, so stage order is observable
    attrs = command.get("attributes") or {}
    operands = command.get("operands") or {}
    if not isinstance(attrs, Mapping) or not isinstance(operands, Mapping):
        return False
    name = attrs.get(route.operand_attribute)
    tensor = tensors.get(name) if isinstance(name, str) else None
    if not isinstance(tensor, Mapping) or tensor.get("role") != route.operand_role:
        return False
    if str(command.get("opcode") or "") in route.producer_opcodes:
        return True  # a fused operation both produces and reads out its accumulator
    source = operands.get("src")
    if not isinstance(source, str) or not source:
        return False
    # A later operand sum writing the same accumulator supersedes an earlier contraction. Looking
    # for *any* contraction would falsely grant its bias route to that sum.
    for previous in reversed(earlier):
        if not isinstance(previous, Mapping):
            continue
        previous_operands = previous.get("operands") or {}
        if isinstance(previous_operands, Mapping) and previous_operands.get("dst") == source:
            return str(previous.get("opcode") or "") in route.producer_opcodes
    return False


@dataclass(frozen=True)
class StageVerdict:
    """One declared stage on one command, and whether its selected application path applies it."""

    command_index: int
    opcode: str
    readout: str
    stage: str
    applied: bool
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_index": self.command_index,
            "opcode": self.opcode,
            "readout": self.readout,
            "stage": self.stage,
            "applied": self.applied,
            "why": self.why,
        }


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
        return {
            "schema": "merlin_epilogue_applicability_v1",
            "status": self.status,
            "detail": self.detail,
            "refusing": self.refusing,
            "readouts_declared": list(self.readouts_declared),
            "n_discarded": len(self.discarded),
            "stages": [v.to_dict() for v in self.stages],
        }


def _epilogue_of(command: Mapping[str, Any]) -> tuple[str, ...]:
    stages = (command.get("attributes") or {}).get("epilogue")
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        return ()
    return tuple(str(s) for s in stages if str(s))


def _readout_of(command: Mapping[str, Any]) -> str | None:
    value = (command.get("attributes") or {}).get("output_dtype")
    return str(value) if isinstance(value, str) and value else None


def selectors_applying(
    readouts: Sequence[ReadoutCapability],
    stages: Sequence[str],
    *,
    routes: Sequence[StageRoute] = (),
    composition: str | None = None,
) -> tuple[str, ...]:
    """The readout selectors that apply EVERY stage in ``stages``, in the target's declaration order.

    The inverse of :func:`assess`, and the one a capsule GENERATOR needs. ``assess`` answers "the
    program chose this readout -- does it apply what the program declared?", a question only askable
    once a program exists. A capsule writer has the opposite problem: it knows the stages it means to
    fuse and must CHOOSE the readout, and choosing by anything other than the target's own declaration
    is how a capsule comes to declare a computation the hardware cannot perform.

    Empty result means no declared readout applies all of them, and the caller must fail closed. That
    is NOT the same as an empty ``readouts``, which means the target described nothing and the caller
    has no basis to choose at all. Callers distinguish the two, because assuming in either direction is
    the failure this module exists to prevent.

    Order is the target's own: when more than one readout qualifies, the first one it declared wins, so
    the tie-break lives with the target rather than in a rule that cannot know which is preferable.
    """
    ordered = tuple(str(s) for s in stages if str(s))
    want = set(ordered)
    return tuple(
        r.selector
        for r in readouts
        if all(
            stage in r.applies
            or (composition is not None and ordered[0] == stage and _route_for(routes, stage, r.selector, composition))
            for stage in want
        )
    )


def assess(
    command_buffer: Mapping[str, Any],
    readouts: Sequence[ReadoutCapability],
    *,
    routes: Sequence[StageRoute] = (),
) -> Assessment:
    """Whether every epilogue stage this program declares is applied by a witnessed path.

    ``readouts`` is the target's own declaration. Required and never defaulted: a program whose
    readout is undescribed gets ``unknown``, because the alternative -- assuming a readout applies
    whatever is asked of it -- is precisely how a discarded activation stays invisible.
    """
    by_selector = {r.selector: r for r in readouts}
    declared = tuple(sorted(by_selector))
    commands = command_buffer.get("commands")
    if not isinstance(commands, Sequence) or isinstance(commands, (str, bytes)):
        return Assessment(
            status="unknown", detail="the command buffer declares no command sequence", readouts_declared=declared
        )

    stages: list[StageVerdict] = []
    tensors = command_buffer.get("tensors") or {}
    if not isinstance(tensors, Mapping):
        tensors = {}
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
            stages.extend(
                StageVerdict(
                    index,
                    opcode,
                    "<undeclared>",
                    stage,
                    False,
                    "the command declares epilogue stages but no readout, so which readout would apply them is UNKNOWN",
                )
                for stage in epilogue
            )
            continue
        capability = by_selector.get(readout)
        if capability is None:
            unknown_readouts.add(readout)
            stages.extend(
                StageVerdict(
                    index,
                    opcode,
                    readout,
                    stage,
                    False,
                    f"the target describes no readout {readout!r} "
                    f"(it declares {list(declared)}), so whether it applies this "
                    f"stage is UNKNOWN and is refused rather than assumed",
                )
                for stage in epilogue
            )
            continue
        for stage in epilogue:
            route = next(
                (
                    r
                    for r in routes
                    if r.stage == stage
                    and r.site == "accumulator_seed"
                    and readout in r.readouts
                    and _route_witness(r, command, commands[:index], tensors)
                ),
                None,
            )
            applied = stage in capability.applies or route is not None
            stages.append(
                StageVerdict(
                    index,
                    opcode,
                    readout,
                    stage,
                    applied,
                    (f"{route.site} route: {route.evidence}" if route is not None else "")
                    if applied
                    else (
                        f"readout {readout!r} does not apply {stage!r} (it applies "
                        f"{sorted(capability.applies)}){': ' + capability.evidence if capability.evidence else ''}"
                        f" -- the emitted program therefore computes something other than what it declares"
                    ),
                )
            )

    if not stages:
        return Assessment(
            status="not_applicable",
            detail="the program declares no epilogue stage, so none can be discarded",
            readouts_declared=declared,
        )
    if unknown_readouts:
        return Assessment(
            status="unknown",
            stages=stages,
            readouts_declared=declared,
            detail=(
                f"readout(s) {sorted(unknown_readouts)} are not described by the target, so "
                f"whether the declared stages are applied cannot be established"
            ),
        )
    dropped = [v for v in stages if not v.applied]
    if dropped:
        first = dropped[0]
        return Assessment(
            status="discarded",
            stages=stages,
            readouts_declared=declared,
            detail=(
                f"{len(dropped)} declared epilogue stage(s) are not applied by the readout or a "
                f"witnessed route; first at command {first.command_index} "
                f"({first.opcode}): {first.why}"
            ),
        )
    return Assessment(
        status="applied",
        stages=stages,
        readouts_declared=declared,
        detail=f"all {len(stages)} declared stage(s) are applied by their readout or a witnessed route",
    )


# Preserve historical object lookup; implementation and completion state live here.
for _compat_definition in (
    ReadoutCapability,
    StageRoute,
    _route_for,
    _route_witness,
    StageVerdict,
    Assessment,
    _epilogue_of,
    _readout_of,
    selectors_applying,
    assess,
):
    _compat_definition.__module__ = "merlin.verification.epilogue_applicability"
del _compat_definition

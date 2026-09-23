"""Whether a kernel's synchronisation is sound ON A GIVEN MACHINE -- which is a different question per
archetype, and that is the whole point of this file.

The same kernel is correct on one machine and wrong on another, with no change to a single instruction.
Where the hardware tracks dependencies, a consumer reading a producer's result needs nothing from the
compiler: reordering cannot change the answer, so an absent wait is not a defect. Where the compiler is
responsible for separation, the identical omission returns wrong data, silently, at full speed.

So this check takes a :class:`~merlin.sched.mach.Machine` and asks the question that machine actually
poses. A checker that asked one question would be wrong on half the corpus -- and wrong in the dangerous
direction on the half where it passed everything.

WHAT IT REFUSES TO DECIDE. A machine whose hazard model was never derived gets neither answer. Treating
an unknown model as interlocked accepts every illegal schedule; treating it as explicit reports defects
that may not exist. It is reported as undecidable, which is the third state the rest of this repo keeps
and the reason ``Machine.hazard_resolution`` may be ``None`` but never defaulted.
"""

from __future__ import annotations

from dataclasses import dataclass

from merlin.sched.ir import Kernel, Loop

__all__ = ["SyncReport", "check_sync"]


@dataclass(frozen=True)
class SyncReport:
    """What could be established about a kernel's synchronisation on one machine.

    ``decidable`` is false when the machine declares no hazard model. ``problems`` is then empty, and
    that emptiness means "not asked", NOT "nothing wrong" -- a caller that prints one without the other
    turns a refusal into a clean bill of health.
    """

    decidable: bool
    archetype: str | None
    problems: tuple[str, ...]
    checked: int
    reason: str = ""
    #: Classes of check this report did NOT run, each naming the declaration it needed. Distinct from
    #: ``decidable``, which is about the hazard model alone: a machine can have a hazard model and still
    #: leave a whole class of question unasked because it declares none of the inputs that class reads.
    #: Empty problems plus a non-empty ``unchecked`` is a partial answer, and a caller that prints only
    #: ``ok`` cannot tell it from a complete one.
    unchecked: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.decidable and not self.problems


def _calls(body) -> list:
    out = []
    for stmt in body:
        out.extend(_calls(stmt.body) if isinstance(stmt, Loop) else [stmt])
    return out


def _wrong_kind_of_wait(calls, machine) -> tuple[list[str], tuple[str, ...]]:
    """A wait that does not order COMPLETION cannot discharge a dependence on a result.

    This is the check a single ``fence`` concept makes unstatable. One target ships a primitive that
    reads like a fence and orders only ISSUE -- the producer was accepted, nothing more -- and another
    that orders visibility within a memory scope while saying nothing about whether a unit finished.
    Using either where a result is needed runs the consumer against data that does not exist yet, at
    full speed and with no error anywhere.

    Wrong on BOTH archetypes, so it is checked before the archetype split: hardware that tracks its own
    dependencies still cannot know that the compiler intended a backpressure poll to mean completion.
    """
    syncs = {s.instr: s for s in getattr(machine, "syncs", ())}
    if not syncs:
        # Measured 2026-09-18: NO target populates `Machine.syncs` -- `mach.derive` never sets the
        # field -- so this check returns immediately on every one of them. Returning an empty problem
        # list says "nothing wrong with the waits" when the truth is "no target declared what its waits
        # order, so the question was never asked". The caller is told which.
        return [], (
            f"{getattr(machine, 'target', '?')} declares no synchronisation instructions "
            "(Machine.syncs is empty), so whether a wait orders completion or only issue was not "
            "checked. A wait that orders issue runs the consumer against data that does not exist yet, "
            "which is precisely what this class of check exists to catch",
        )
    problems: list[str] = []
    for call in calls:
        declared = syncs.get(call.instr)
        if declared is None or declared.discharges_dependence:
            continue
        if call.awaits:
            problems.append(
                f"{call.instr}: awaits {list(call.awaits)}, but {machine.target} declares it orders "
                f"{declared.orders!r}, not completion. A wait that does not establish that the producer "
                "FINISHED cannot discharge a dependence on its result -- the consumer runs against data "
                "that does not exist yet, at full speed and with nothing reported."
            )
    return problems, ()


def check_sync(kernel: Kernel, machine) -> SyncReport:
    """Problems with ``kernel``'s synchronisation on ``machine``.

    On an EXPLICIT machine the compiler owns separation, so every consumer of an asynchronous producer
    must await its token, and a token left un-awaited at the end is work whose completion nothing
    established. On an INTERLOCKED machine neither is a defect -- but a token awaited on a unit that
    cannot produce one, or a call placed on a unit the machine does not have, is wrong on both.
    """
    hazards = getattr(machine, "hazard_resolution", None)
    calls = _calls(kernel.body)
    units = {u.name for u in getattr(machine, "units", ())}

    problems: list[str] = []
    for call in calls:
        if call.unit is not None and units and call.unit not in units:
            problems.append(
                f"{call.instr}: placed on unit {call.unit!r}, which {machine.target} does not have ({sorted(units)})"
            )

    if hazards is None:
        return SyncReport(
            decidable=False,
            archetype=None,
            problems=tuple(problems),
            checked=len(calls),
            reason=(
                f"{getattr(machine, 'target', '?')} declares no hazard model, so whether an absent wait "
                "is a defect cannot be answered. Assuming hardware tracking would accept every illegal "
                "schedule; assuming compiler responsibility would report defects that may not exist."
            ),
        )

    wait_problems, unchecked = _wrong_kind_of_wait(calls, machine)
    problems.extend(wait_problems)

    if hazards == "explicit":
        produced: dict[str, object] = {}
        awaited: set[str] = set()
        # A wait between two calls on ONE unit is redundant -- an in-order queue already separates them
        # -- and is deliberately NOT reported. Telling a caller to remove a wait is the one edit that is
        # unsafe on this archetype, so the check never suggests it.
        for call in calls:
            awaited.update(call.awaits)
            if call.produces:
                produced[call.produces] = call
        for token, producer in produced.items():
            if token not in awaited:
                problems.append(
                    f"{producer.instr}: produces token {token!r} that nothing awaits. On a machine where "
                    "the compiler separates hazards, work whose completion nothing established is a "
                    "wrong answer at full speed, not a missed optimisation."
                )

    return SyncReport(
        decidable=True,
        archetype=hazards,
        problems=tuple(problems),
        checked=len(calls),
        unchecked=unchecked,
        reason=(
            "hardware tracks dependencies, so an absent wait is not a defect here and a correctness "
            "check cannot refute a reordering; the falsifier for a scheduling change is measured "
            "occupancy"
            if hazards == "interlocked"
            else "the compiler separates hazards, so an absent wait returns wrong data"
        ),
    )

"""What a schedule costs on a machine -- and, more often, why that cannot be answered exactly.

A cost model's job is not to produce a number. It is to produce a number WITH ITS KIND, because the
three cases are genuinely different and collapsing them is how a schedule gets ranked by a figure no
arrangement of it can achieve:

``exact``        every instruction's cost on its unit was derived, and none of those costs is a floor.
``lower_bound``  every cost was derived, but at least one is a FLOOR measured on an idle machine. Real
                 contention can only make the schedule slower, so the number bounds it from below and
                 must never be reported as a prediction.
``undecidable``  some instruction has no derived cost on the unit it runs on. NOT a zero, and not the
                 sum of the ones that are known: the missing terms are unbounded, so a partial sum is
                 smaller than the truth by an unknown amount and reads as the schedule being fast.

The last is the one that matters. A cost model that skipped underived terms would rank a schedule made
of instructions nobody has measured as the cheapest one available, which is precisely backwards.

WHAT IS AND IS NOT MODELLED. Issue is serial per queue -- units sharing an ordered command stream take
each other's issue slots -- while EXECUTION overlaps unless two units share a resource. That split is
the whole reason ``Unit`` separates ``queue`` from ``executes_on``: two units behind one in-order port
still overlap, and on one target that overlap is the entire performance lever. What is deliberately NOT
modelled here is anything that would need a data-dependent duration, a bank arbiter or a memory system:
this prices ISSUE and DECLARED LATENCY, and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from merlin.sched.ir import Kernel, Loop

__all__ = ["Bound", "KINDS", "price"]

#: Ordered weakest-to-strongest claim. A caller comparing two schedules may only compare bounds of the
#: same kind; a ``lower_bound`` beating an ``exact`` says nothing.
KINDS: tuple[str, ...] = ("undecidable", "lower_bound", "exact")


@dataclass(frozen=True)
class Bound:
    """A schedule's cost, and what kind of claim the number is.

    ``cycles`` is ``None`` when ``kind`` is ``undecidable``. It is deliberately not a partial sum: a
    caller that received one would have a number smaller than the truth by an unknown amount, which is
    worse than having none, because a number gets compared.
    """

    kind: str
    cycles: int | None
    #: ``(instruction, unit)`` pairs whose cost the machine does not declare.
    missing: tuple[tuple[str, str], ...] = ()
    #: ``(instruction, unit)`` pairs whose declared cost is a floor measured on an idle machine.
    floors: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"cost kind {self.kind!r} not in {list(KINDS)}")
        if (self.cycles is None) != (self.kind == "undecidable"):
            raise ValueError(f"kind {self.kind!r} with cycles={self.cycles!r}: a decidable cost has a number")

    @property
    def comparable_to(self) -> str:
        """What this bound may be compared against: only another of the same kind."""
        return self.kind


def _calls(body) -> list:
    out = []
    for stmt in body:
        if isinstance(stmt, Loop):
            out.extend(_calls(stmt.body) * stmt.extent)
        else:
            out.append(stmt)
    return out


def price(kernel: Kernel, machine) -> Bound:
    """Price ``kernel`` on ``machine``, reporting the KIND of claim the number is.

    Issue is summed per queue, because units sharing an ordered command stream take each other's slots.
    Execution is not summed: a unit's result latency extends the finish time of the work that awaits it,
    and units on different resources overlap. The result is the later of "everything issued" and "the
    last awaited result arrived", which is a lower bound on any real machine and exact only where every
    declared cost is exact.
    """
    calls = _calls(kernel.body)
    if not calls:
        return Bound(kind="exact", cycles=0, notes=("the kernel contains no calls",))

    missing: list[tuple[str, str]] = []
    floors: list[tuple[str, str]] = []
    issue_by_queue: dict[str, int] = {}
    finish_by_token: dict[str, int] = {}
    now = 0

    for call in calls:
        unit = call.unit
        if unit is None:
            missing.append((call.instr, "<unplaced>"))
            continue
        cost = machine.latency(call.instr, unit)
        if cost is None or cost.issue is None:
            missing.append((call.instr, unit))
            continue
        if cost.contended:
            floors.append((call.instr, unit))
        try:
            queue = machine.unit(unit).queue
        except Exception:  # noqa: BLE001 - a unit the machine lacks is the sync check's finding, not a cost
            missing.append((call.instr, unit))
            continue
        # A consumer cannot issue before every result it awaits has arrived.
        ready = max((finish_by_token.get(t, 0) for t in call.awaits), default=0)
        start = max(issue_by_queue.get(queue, 0), ready)
        issue_by_queue[queue] = start + cost.issue
        now = max(now, issue_by_queue[queue])
        if call.produces:
            if cost.result is None:
                missing.append((call.instr, unit))
                continue
            finish_by_token[call.produces] = start + cost.result
            now = max(now, finish_by_token[call.produces])

    if missing:
        return Bound(
            kind="undecidable",
            cycles=None,
            missing=tuple(sorted(set(missing))),
            floors=tuple(sorted(set(floors))),
            notes=(
                "some instruction has no derived cost on the unit it runs on. The missing terms are "
                "unbounded, so a sum of the known ones would be smaller than the truth by an unknown "
                "amount -- and would rank a schedule of unmeasured instructions as the cheapest.",
            ),
        )
    if floors:
        return Bound(
            kind="lower_bound",
            cycles=now,
            floors=tuple(sorted(set(floors))),
            notes=(
                "at least one declared cost is a floor measured on an idle machine. Contention can only "
                "make this slower, so the number bounds the schedule from below and is not a prediction.",
            ),
        )
    return Bound(kind="exact", cycles=now)

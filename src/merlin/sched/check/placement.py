"""Whether a kernel's on-chip placement is sound on a given machine.

Two failures this is written against, both measured on shipped code, both silent.

A DESTINATION WRITTEN ON TOP OF A LIVE OPERAND. A kernel carried a constant destination offset that was
correct for one tile size and wrong for every larger one, so the result was written over the operand it
was still reading. Measured: zero correct elements out of 16,384, with no error reported anywhere. The
program ran, the harness completed, the numbers were wrong.

A BANK SHARED BETWEEN TWO ENGINES. Where a scratchpad arbitrates in favour of one engine, any bank the
two share starves the other -- so bank assignment decides whether a stage overlaps at all. That is a
fact about the BANK, and it is invisible from an address.

Neither is a question about the archetype: writing over a live operand returns wrong data whether or not
hardware tracks dependencies, and an arbiter starves regardless. So unlike :mod:`~merlin.sched.check.sync`
this asks one question of every machine -- what it needs from the machine is the memory's geometry, not
its hazard model.
"""

from __future__ import annotations

from dataclasses import dataclass

from merlin.sched.ir import Kernel, Loop

__all__ = ["PlacementReport", "check_placement"]


@dataclass(frozen=True)
class PlacementReport:
    problems: tuple[str, ...]
    #: Stages examined. A report with no problems over zero stages is not a placed kernel, it is an
    #: unplaced one, and a caller printing only the verdict cannot tell the difference.
    checked: int

    @property
    def ok(self) -> bool:
        return not self.problems


def _calls(body) -> list:
    out = []
    for stmt in body:
        out.extend(_calls(stmt.body) if isinstance(stmt, Loop) else [stmt])
    return out


def _occupied(stage) -> int:
    """Rows a staging actually holds: its span times its number of rotating copies.

    A depth of n means n copies laid end to end from ``row``, so pricing only ``rows`` would approve a
    double-buffered stage in a store that can hold one copy -- and would report no overlap between two
    stagings whose later copies sit on top of each other.
    """
    return stage.rows * stage.depth


def _overlaps(a, b) -> bool:
    return a.row < b.row + _occupied(b) and b.row < a.row + _occupied(a)


def check_placement(kernel: Kernel, machine) -> PlacementReport:
    """Problems with where ``kernel`` puts its operands on chip."""
    calls = _calls(kernel.body)
    memories = {m.name: m for m in getattr(machine, "memories", ())}
    problems: list[str] = []
    checked = 0

    for call in calls:
        for stage in call.stages:
            checked += 1
            memory = memories.get(stage.memory)
            if memory is None:
                problems.append(
                    f"{call.instr}: staged into {stage.memory!r}, which {machine.target} does not have "
                    f"({sorted(memories)})"
                )
                continue
            if stage.rows < 1 or stage.row < 0:
                problems.append(f"{call.instr}: stage of {stage.rows} rows at {stage.row} in {memory.name}")
                continue
            if stage.depth < 1:
                problems.append(f"{call.instr}: stage in {memory.name} declares {stage.depth} copies")
                continue
            if memory.rows is not None and stage.row + _occupied(stage) > memory.rows:
                copies = f" x{stage.depth} copies" if stage.depth != 1 else ""
                problems.append(
                    f"{call.instr}: stage {memory.name}[{stage.row}:{stage.row + stage.rows}]{copies} "
                    f"runs past the {memory.rows} rows that store has"
                )
            if stage.bank is not None:
                if memory.banks is None:
                    problems.append(
                        f"{call.instr}: names bank {stage.bank} of {memory.name}, whose bank count was "
                        "never derived -- a bank chosen against an unknown count is a guess, not a "
                        "placement"
                    )
                elif not 0 <= stage.bank < memory.banks:
                    problems.append(
                        f"{call.instr}: names bank {stage.bank} of {memory.name}, which has {memory.banks} bank(s)"
                    )

        # A call may not write rows it is also reading, nor write rows another operand of the same call
        # occupies. This is the 0/16384 failure, and it is entirely local to one call.
        for i, a in enumerate(call.stages):
            for b in call.stages[i + 1 :]:
                if a.memory == b.memory and _overlaps(a, b) and (a.writes or b.writes):
                    problems.append(
                        f"{call.instr}: {a.memory}[{a.row}:{a.row + _occupied(a)}] and "
                        f"{b.memory}[{b.row}:{b.row + _occupied(b)}] overlap and one of them writes. A result "
                        "written over an operand still being read returns wrong data with nothing "
                        "reported -- measured once at zero correct elements out of 16,384."
                    )

    problems.extend(_starved_banks(calls, memories, machine))
    return PlacementReport(problems=tuple(problems), checked=checked)


def _starved_banks(calls, memories, machine) -> list[str]:
    """Banks two contending engines share, where the memory declares an arbiter.

    Reported only where the machine declares BOTH an arbiter and which units contend: without the
    arbiter there is no priority to lose, and without the contender list there is nothing to say two
    units are the two. Inferring either would turn a fact about the device into a guess about it.
    """
    problems: list[str] = []
    for memory in memories.values():
        if memory.arbiter is None or len(memory.shared_by) < 2:
            continue
        by_bank: dict[int, set[str]] = {}
        for call in calls:
            if call.unit is None or call.unit not in memory.shared_by:
                continue
            for stage in call.stages:
                if stage.memory == memory.name and stage.bank is not None:
                    by_bank.setdefault(stage.bank, set()).add(call.unit)
        for bank, units in sorted(by_bank.items()):
            if len(units) > 1:
                problems.append(
                    f"{memory.name} bank {bank} is used by {sorted(units)}, which contend for it under "
                    f"the {memory.arbiter!r} arbiter. A bank shared between two engines starves the one "
                    "the arbiter deprioritises, so this decides whether the stage overlaps at all."
                )
    return problems

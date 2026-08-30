"""What each instruction defines and uses, what stays live, and how much register file that spends.

This is the layer between a decoded instruction stream and any dependence question asked of it. It
holds no register-file table, no operand-name convention and no notion of what a "destination" is
spelled like: every direction comes from a measured operand-direction model, and an operand whose
direction that model did not establish is carried through as UNRESOLVED rather than guessed.

Why unresolved has to survive to the end. A dependence graph built from a partially resolved
instruction is not a slightly worse graph, it is a graph with a MISSING EDGE, and a missing edge is
exactly the separation a reordering would delete. So an instruction with an unresolved operand does
not quietly contribute the operands it does know: it contributes those, and it also contributes the
fact that it is incomplete, and every consumer downstream reports what it built on.

THREE PASSES, and each mirrors a rule established elsewhere in this package:

* :func:`constant_state` is forward propagation with KILL semantics, the same shape as
  :func:`merlin.perf.dma_volume.propagate_constants`: a register written by anything the pass cannot
  evaluate becomes UNKNOWN rather than keeping a stale value, and a backward branch invalidates every
  register, because a value that differs per iteration is not a constant. It exists here so a
  transfer's staging address is a resolved number where the program actually established one, and
  UNKNOWN where it did not.
* :func:`liveness` is its backward mirror, over the real control-flow graph rather than over a
  straight line: ``live_in = (live_out - defs) | uses``, iterated to a fixed point so a backward
  branch makes the values a loop body reads at the top live at its bottom. That is not a detail --
  loop-carried liveness is the whole reason a tensor register cannot simply be reused.
* :func:`pressure` counts how many values of each register file are live at once. This is the
  resource a reordering pass spends FIRST and the one nothing here modelled before: moving a producer
  earlier to cover a latency extends its value's live range by exactly the distance it moved, and a
  file with as few slots as a tensor register file runs out long before instruction memory does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "Access", "Effects", "Instruction", "LivenessReport", "Pressure", "ValueRange",
    "constant_state", "effects_of", "liveness", "live_values", "pressure", "report", "value_ranges",
]


@dataclass(frozen=True, order=True)
class Access:
    """One architectural value: a state file and a slot in it.

    The file name is the oracle's own -- whatever it published when the direction probe watched that
    slot move. Nothing here interprets it, which is what lets a target with a register file this code
    has never heard of be analysed without a line changing."""

    file: str
    slot: int

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.file}[{self.slot}]"


@dataclass(frozen=True)
class Instruction:
    """One decoded instruction of an emitted program, in issue order."""

    index: int
    mnemonic: str
    operands: Mapping[str, int] = field(default_factory=dict)
    #: Where control goes when this instruction transfers it, as an instruction index. None for a
    #: straight-line instruction. Supplied by the caller from the machine's own measured branch
    #: contract, because a branch immediate is not portable knowledge.
    branch_target: int | None = None
    #: A label for reporting -- which part of the emitted program this instruction belongs to.
    section: str = ""

    @property
    def branches_backward(self) -> bool:
        return self.branch_target is not None and self.branch_target <= self.index


@dataclass(frozen=True)
class Effects:
    """What one instruction writes and reads, and what about it could not be established."""

    defs: tuple[Access, ...]
    uses: tuple[Access, ...]
    #: Operand names whose direction, or whose state file, the probe did not resolve. An instruction
    #: with any of these has an incomplete dependence footprint, and every consumer says so.
    unresolved: tuple[str, ...]
    #: True when the probe saw the instruction change observable state at all. False is a real,
    #: different answer from "it defines nothing": it means the oracle could not see whatever it did.
    observed: bool = True

    @property
    def complete(self) -> bool:
        return not self.unresolved


def effects_of(instruction: Instruction, directions: Any) -> Effects:
    """One instruction's defined and used values, from the MEASURED operand-direction model.

    A definition may span several consecutive slots -- a result in a format wider than one register
    lands in a run of them, and the probe measured how many. The run is expanded here rather than
    left implicit, because a dependence on the second half of a wide result is a dependence, and a
    graph that tracks only the leading slot loses it.
    """
    from merlin.targetgen import isa_direction as ID

    per_operand = (getattr(directions, "by_mnemonic", {}) or {}).get(instruction.mnemonic)
    if per_operand is None:
        return Effects(defs=(), uses=(), unresolved=("<the instruction was never probed>",),
                       observed=False)
    defs: list[Access] = []
    uses: list[Access] = []
    unresolved: list[str] = []
    observed = False
    for name, verdict in sorted(per_operand.items()):
        if name not in instruction.operands:
            continue
        value = int(instruction.operands[name])
        if verdict.direction == ID.UNKNOWN:
            unresolved.append(f"{name}: {verdict.reason}")
            continue
        observed = True
        if verdict.state_file is None:
            unresolved.append(f"{name}: direction {verdict.direction}, but no state file was attributed")
            continue
        width = max(1, len(verdict.written_slots))
        if verdict.direction in (ID.DEF, ID.DEF_USE):
            defs.extend(Access(verdict.state_file, value + i) for i in range(width))
        if verdict.direction in (ID.USE, ID.DEF_USE):
            uses.append(Access(verdict.state_file, value))
    return Effects(defs=tuple(defs), uses=tuple(uses), unresolved=tuple(unresolved), observed=observed)


# ---------------------------------------------------------------------------------------------------
# forward: which registers hold a constant this pass can name
# ---------------------------------------------------------------------------------------------------
def constant_state(instructions: "Sequence[Instruction]", effects: "Sequence[Effects]", *,
                   immediate_forms: Mapping[str, str],
                   zero_slot: Mapping[str, int] | None = None) -> list[dict[Access, int | None]]:
    """Per-instruction snapshots of which values hold a constant this pass can name.

    Forward propagation with KILL semantics, exactly as the movement-volume pass does it: a value
    written by anything other than a declared immediate form becomes UNKNOWN rather than keeping a
    stale value, so a program that loads a length once and rewrites the register later cannot have
    the old length attributed to the later transfer. A backward branch invalidates every value: one
    that differs per iteration is not a constant, and treating one as constant is how a loop-carried
    address becomes a confident lie.

    ``immediate_forms`` maps a mnemonic to the operand its constant travels in, so no spelling is
    assumed. ``zero_slot`` names, per file, the slot the register file hardwires to zero, where the
    target has one -- it is the only value that survives a kill, because nothing can write it.
    """
    state: dict[Access, int | None] = {}
    out: list[dict[Access, int | None]] = []
    invalidated = False
    zeros = dict(zero_slot or {})
    for instruction, effect in zip(instructions, effects):
        if invalidated:
            state = {}
        if instruction.branches_backward:
            invalidated = True
        form = instruction.mnemonic
        imm_operand = immediate_forms.get(form)
        for value in effect.defs:
            if zeros.get(value.file) == value.slot:
                continue
            if imm_operand is not None and imm_operand in instruction.operands:
                state[value] = int(instruction.operands[imm_operand])
            else:
                state[value] = None          # written by something unevaluatable -> UNKNOWN, not stale
        if effect.unresolved:
            # An instruction whose footprint is incomplete may have written anything, so nothing
            # survives it. Keeping the old values here is the flattering error: it would let an
            # address the program never established look established.
            state = {k: None for k in state}
        for file_name, slot in zeros.items():
            state[Access(file_name, slot)] = 0
        out.append(dict(state))
    return out


# ---------------------------------------------------------------------------------------------------
# backward: liveness over the real control-flow graph
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ValueRange:
    """One value's span: where it was defined and where it was last read."""

    value: Access
    defined_at: int
    last_use: int | None
    #: True when the value is still live at the end of the region, so its range does not close here.
    escapes: bool = False

    @property
    def length(self) -> int:
        return 0 if self.last_use is None else self.last_use - self.defined_at


@dataclass(frozen=True)
class Pressure:
    """How many values of one state file are live at once, and where the peak is."""

    file: str
    peak: int
    at_index: int
    capacity: int | None = None

    @property
    def fits(self) -> bool | None:
        """None when the file's capacity is not known -- which is NOT a pass."""
        return None if self.capacity is None else self.peak <= self.capacity

    def claim(self) -> str:
        if self.capacity is None:
            return (f"{self.file}: {self.peak} value(s) live at once (peak at instruction "
                    f"{self.at_index}); the file's capacity is UNKNOWN, so whether that fits was NOT "
                    "checked")
        verdict = "fits" if self.fits else "OVERFLOWS"
        return (f"{self.file}: {self.peak} of {self.capacity} slot(s) live at once (peak at "
                f"instruction {self.at_index}) -- {verdict}")


@dataclass(frozen=True)
class LivenessReport:
    """Live sets, value ranges and per-file pressure over one region of a program."""

    live_in: tuple[frozenset[Access], ...]
    live_out: tuple[frozenset[Access], ...]
    ranges: tuple[ValueRange, ...]
    pressure: tuple[Pressure, ...]
    #: Instructions whose dependence footprint was incomplete, with the reason. Everything above is
    #: conditional on these, and a consumer that does not surface them is over-claiming.
    incomplete: tuple[str, ...] = ()

    def claim(self) -> str:
        peaks = "; ".join(p.claim() for p in self.pressure)
        if self.incomplete:
            return (f"{peaks} -- CONDITIONAL: {len(self.incomplete)} instruction(s) have an "
                    "unresolved operand, so values they touch are missing from these sets")
        return peaks


def liveness(instructions: "Sequence[Instruction]", effects: "Sequence[Effects]", *,
             max_rounds: int = 64) -> tuple[tuple[frozenset[Access], ...], tuple[frozenset[Access], ...]]:
    """Backward live-value analysis over the instruction stream's own control-flow graph.

    ``live_in[i] = (live_out[i] - defs[i]) | uses[i]`` and ``live_out[i]`` is the union over
    successors, iterated to a fixed point. The fixed point is the part that matters: with a backward
    branch the successor set is cyclic, and one backward sweep would report a loop body's inputs as
    dead at the bottom of the body -- which reads as a register free to reuse, at exactly the place
    where reusing it is wrong.
    """
    n = len(instructions)
    successors: list[tuple[int, ...]] = []
    for i, instruction in enumerate(instructions):
        nxt: list[int] = []
        if i + 1 < n:
            nxt.append(i + 1)
        if instruction.branch_target is not None and 0 <= instruction.branch_target < n:
            nxt.append(int(instruction.branch_target))
        successors.append(tuple(dict.fromkeys(nxt)))

    live_in: list[frozenset[Access]] = [frozenset() for _ in range(n)]
    live_out: list[frozenset[Access]] = [frozenset() for _ in range(n)]
    for _ in range(max_rounds):
        changed = False
        for i in range(n - 1, -1, -1):
            out: set[Access] = set()
            for s in successors[i]:
                out |= live_in[s]
            inn = (out - set(effects[i].defs)) | set(effects[i].uses)
            if out != live_out[i] or inn != live_in[i]:
                live_out[i], live_in[i] = frozenset(out), frozenset(inn)
                changed = True
        if not changed:
            break
    return tuple(live_in), tuple(live_out)


def live_values(instructions: "Sequence[Instruction]", effects: "Sequence[Effects]") -> tuple[frozenset[Access], ...]:
    """Just the live-in sets, for a caller that does not need the pair."""
    return liveness(instructions, effects)[0]


def value_ranges(instructions: "Sequence[Instruction]", effects: "Sequence[Effects]",
                 live_out: "Sequence[frozenset[Access]]") -> tuple[ValueRange, ...]:
    """``[definition, last use]`` for every value the region defines.

    A value still live when the region ends does not get a closed range: it ESCAPES, and giving it a
    last use at the final instruction would report a register as free when the next iteration is
    about to read it."""
    last_use: dict[Access, int] = {}
    for i, effect in enumerate(effects):
        for value in effect.uses:
            last_use[value] = i
    end_live = set(live_out[-1]) if live_out else set()
    out: list[ValueRange] = []
    seen: set[tuple[Access, int]] = set()
    for i, effect in enumerate(effects):
        for value in effect.defs:
            if (value, i) in seen:
                continue
            seen.add((value, i))
            use = last_use.get(value)
            out.append(ValueRange(value=value, defined_at=i,
                                  last_use=(use if use is not None and use >= i else None),
                                  escapes=value in end_live))
    return tuple(out)


def pressure(live_in: "Sequence[frozenset[Access]]",
             capacities: Mapping[str, int] | None = None) -> tuple[Pressure, ...]:
    """Peak simultaneous live values per state file -- the resource a reordering spends first.

    A file whose capacity is not derivable reports its peak with the capacity UNKNOWN. That is not a
    pass: it says the peak was measured and the limit was not, which is the only honest thing to say
    about a register file nobody published the size of."""
    peaks: dict[str, tuple[int, int]] = {}
    for index, live in enumerate(live_in):
        counts: dict[str, int] = {}
        for value in live:
            counts[value.file] = counts.get(value.file, 0) + 1
        for file_name, count in counts.items():
            if count > peaks.get(file_name, (0, 0))[0]:
                peaks[file_name] = (count, index)
    caps = dict(capacities or {})
    return tuple(Pressure(file=name, peak=count, at_index=at, capacity=caps.get(name))
                 for name, (count, at) in sorted(peaks.items()))


def report(instructions: "Sequence[Instruction]", effects: "Sequence[Effects]", *,
           capacities: Mapping[str, int] | None = None) -> LivenessReport:
    """Liveness, ranges and pressure in one pass, carrying every incompleteness forward."""
    live_in, live_out = liveness(instructions, effects)
    incomplete = tuple(f"[{i}] {ins.mnemonic}: {'; '.join(eff.unresolved)}"
                       for i, (ins, eff) in enumerate(zip(instructions, effects)) if eff.unresolved)
    return LivenessReport(live_in=live_in, live_out=live_out,
                          ranges=value_ranges(instructions, effects, live_out),
                          pressure=pressure(live_in, capacities), incomplete=incomplete)

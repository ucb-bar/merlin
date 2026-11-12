"""Simulator-free checks of a kernel: static legality, configuration liveness, on-chip placement,
synchronisation on a given machine, exhaustive epilogue enumeration, the numerics contract a golden is
computed under, and the KIND of a schedule's cost (exact, estimated or lower bound).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from merlin.compiler.kernel_ir.ir import (
    NULL,
    Expr,
    InstructionSet,
    Kernel,
    KernelError,
    Loop,
    Ptr,
    check_structure,
    concretize,
    free_vars,
    instances,
    render,
)

# ---- epilogue_enum -----------------------------------------------------------------------------

Readout = Callable[[np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class FlipReport:
    """The outcome of one exhaustive comparison."""

    lo: int
    hi: int
    scale: float
    evaluated: int
    flips: int
    #: Up to ``max_examples`` disagreements as ``(accumulator, value_a, value_b)``.
    examples: tuple[tuple[int, int, int], ...]

    @property
    def exact(self) -> bool:
        return self.flips == 0 and self.evaluated == self.hi - self.lo + 1


def reachable_accumulator_range(
    k: int, a_range: tuple[int, int], b_range: tuple[int, int], bias_range: tuple[int, int] = (0, 0)
) -> tuple[int, int]:
    """Bounds of ``sum_{i<k} a_i * b_i + bias`` for operands in the given inclusive ranges."""
    if k <= 0:
        raise ValueError("reduction length must be positive")
    corners = [a * b for a in a_range for b in b_range]
    return k * min(corners) + bias_range[0], k * max(corners) + bias_range[1]


def enumerate_readout_flips(
    readout_a: Readout,
    readout_b: Readout,
    *,
    scale: float,
    lo: int,
    hi: int,
    chunk: int = 1 << 22,
    max_examples: int = 16,
) -> FlipReport:
    """Evaluate both readouts on every integer in ``[lo, hi]`` and count disagreements."""
    if hi < lo:
        raise ValueError("empty accumulator range")
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    flips = 0
    evaluated = 0
    examples: list[tuple[int, int, int]] = []
    start = lo
    while start <= hi:
        stop = min(hi, start + chunk - 1)
        acc = np.arange(start, stop + 1, dtype=np.int64)
        va = np.asarray(readout_a(acc, scale)).astype(np.int64)
        vb = np.asarray(readout_b(acc, scale)).astype(np.int64)
        if va.shape != acc.shape or vb.shape != acc.shape:
            raise ValueError("a readout changed the shape of its input")
        diff = np.nonzero(va != vb)[0]
        flips += int(diff.size)
        for idx in diff[: max(0, max_examples - len(examples))]:
            examples.append((int(acc[idx]), int(va[idx]), int(vb[idx])))
        evaluated += acc.size
        start = stop + 1
    return FlipReport(lo=lo, hi=hi, scale=float(scale), evaluated=evaluated, flips=flips, examples=tuple(examples))


# ---- static ------------------------------------------------------------------------------------
# G0: static legality of a kernel against a target's instruction set, before anything runs.
#
# Three layers, cheapest first:
#
# 1. target-free structure (``merlin.compiler.kernel_ir.ir.check_structure``);
# 2. every call names an instruction of the set, with exactly its operands, of the right kinds;
# 3. every DYNAMIC instance, in program order: flags are 0/1, the target's ``check`` holds on the concrete
#    operand values (with one ``state`` dict threaded through the kernel for cross-instruction rules), and
#    every byte extent the target's ``footprint`` reports lies inside its tensor argument, with no write
#    into a read-only argument.
#
# Layer 3 enumerates instances, so its cost is the kernel's instruction count; schedules issue loop
# macro-instructions, so that count is small. ``max_instances`` bounds it and says so when hit.

_KIND_OK = {
    "int": lambda v: isinstance(v, Expr),
    "flag": lambda v: isinstance(v, Expr),
    "ptr": lambda v: isinstance(v, Ptr) or v is NULL,
    "float": lambda v: isinstance(v, float),
}


def _static_calls(body):
    for s in body:
        if isinstance(s, Loop):
            yield from _static_calls(s.body)
        else:
            yield s


def check_kernel(
    kernel: Kernel, iset: InstructionSet, *, max_instances: int = 2_000_000, max_errors: int = 50
) -> list[str]:
    errors = check_structure(kernel)
    for c in _static_calls(kernel.body):
        d = iset.instrs.get(c.instr)
        if d is None:
            errors.append(f"unknown instruction {c.instr!r}")
            continue
        names = tuple(k for k, _ in c.args)
        if names != d.operand_names():
            errors.append(f"{c.instr}: operands {names} != {d.operand_names()}")
            continue
        for o, (_, v) in zip(d.operands, c.args):
            if not _KIND_OK[o.kind](v):
                errors.append(f"{c.instr}.{o.name}: expected a {o.kind} operand, got {v!r}")
    if errors:
        return errors
    state: dict = {}
    for n, (c, env) in enumerate(instances(kernel)):
        if n >= max_instances:
            errors.append(f"stopped after {max_instances} instances; the rest are unchecked")
            break
        d = iset.instr(c.instr)
        where = f"{c.instr}@{env}"
        try:
            values = concretize(c, env)
        except KernelError as exc:
            errors.append(f"{where}: {exc}")
            continue
        for o in d.operands:
            if o.kind == "flag" and values[o.name] not in (0, 1):
                errors.append(f"{where}: flag {o.name} = {values[o.name]}")
        if d.check is not None:
            errors.extend(f"{where}: {m}" for m in d.check(values, state))
        if d.footprint is not None:
            for operand, nbytes, access in d.footprint(values):
                p = values[operand]
                if p is None or nbytes == 0:
                    continue
                t = kernel.tensor(p.tensor)
                if p.offset < 0 or p.offset + nbytes > t.nbytes:
                    errors.append(
                        f"{where}: {operand} touches bytes [{p.offset}, {p.offset + nbytes}) "
                        f"of {t.name} ({t.nbytes} bytes)"
                    )
                if access == "write" and t.access == "read":
                    errors.append(f"{where}: {operand} writes read-only {t.name}")
        if len(errors) >= max_errors:
            errors.append("too many errors; stopped")
            return errors
    else:
        if iset.finish is not None:
            errors.extend(f"end of kernel: {m}" for m in iset.finish(state))
    return errors


# ---- config ------------------------------------------------------------------------------------
# Whether a kernel's configuration is live where it is used -- and where a configuration is dead.
#
# CONFIGURATION IS STATE, NOT STATEMENT ORDER. These accelerators are configured by instructions that set
# a register and return: a stride, a dataflow, a transpose, an activation. Nothing in a later instruction
# names the configuration it depends on, so the dependence exists only in the order the two were written.
# A reordering, a hoist, or a tile whose stride differs from the last one configured produces a device
# that is configured -- just not for this call. It returns wrong data, promptly, with nothing reported.
#
# WHY THIS IS GENERIC AND NOT ONE TARGET'S PROBLEM. One target's binding carries this by hand, field by
# field: "config_ld id 0 stride != the loop's", "config_st stride does not match", "config_ex transposes
# differ from the loop's (or no config_ex)". Three spellings of one question -- is the value this call
# assumes the value that is live? -- written out per field because the IR had no way to say it. Declared
# as ``sets``/``assumes`` on the call, the question is asked once here for every target, and the per-target
# work shrinks to naming the keys.
#
# THE SECOND FINDING IS A SAVING, NOT A DEFECT. A configuration that is overwritten before any call
# assumes it is DEAD: the instruction issued, cost its dispatch slot, and changed nothing that was read.
# That is the elidable-config case, and it is reported separately from the misconfigurations because it
# does not make the kernel wrong -- it makes it slower. Reporting the two in one list would force a caller
# to choose between ignoring wrong answers and ignoring free speedups.
#
# WHAT THIS DOES NOT DO. It does not know what any key MEANS. ``sets``/``assumes`` are opaque key/value
# pairs compared for equality, so nothing here has an opinion about strides or dataflows -- which is what
# keeps it target-free. A target that declares no configuration gets no findings rather than a default.


class _Varies:
    """A configuration whose value depended on a loop variable, seen from outside that loop.

    Not a value and not an absence. `sets stride=i` inside `for i in 0..5` leaves a different stride
    live on every iteration, and which one survives the loop is arithmetic this check does not do. The
    rendered text `"i"` is NOT that value: comparing it to a later `needs stride=i` in a different loop
    reports agreement between two unrelated numbers, and comparing it to `needs stride=4` reports a
    mismatch on a kernel that is correct. Both were measured. So the value is replaced by this at the
    loop's edge, and a read of it is reported as UNDECIDABLE rather than guessed in either direction.
    """

    __slots__ = ("var",)

    def __init__(self, var: str) -> None:
        self.var = var


def _text(value) -> str:
    return render(value) if isinstance(value, Expr) else repr(value)


@dataclass(frozen=True)
class ConfigReport:
    """What is wrong, what is wasted, and how much was looked at.

    ``problems`` are calls reading a configuration that is not live. ``dead`` are configurations no call
    read before they were replaced -- a cost, not an error.
    """

    problems: tuple[str, ...]
    dead: tuple[str, ...]
    #: Configuration keys the kernel uses at all. Zero means this kernel declares no configuration, in
    #: which case a clean report says nothing about it -- a caller printing only `ok` cannot tell an
    #: unconfigured kernel from a correctly configured one.
    keys: tuple[str, ...]
    #: Reads this check could not decide -- the live value varied with a loop that has since ended.
    #: Kept out of `problems` because it is not a defect, and out of silence because it is not clean.
    undecidable: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.problems


def _walk(body, live: dict, seen: dict, problems: list, dead: list, undecidable: list, keys: set, depth: int) -> None:
    """Interpret ``body`` in order, carrying the live configuration forward.

    ``live`` maps key -> (value, what set it). ``seen`` marks whether the live value has been read since
    it was set; an unread value that is replaced is the dead-configuration finding.
    """
    for stmt in body:
        if isinstance(stmt, Loop):
            # A loop body is walked TWICE when it can run twice, and the second pass is not redundant.
            # Iteration 0 sees the configuration left by the statements before the loop; every later
            # iteration sees what the loop's own TAIL left. Those are different states, so a body whose
            # head assumes a value its tail overwrites is correct on the first iteration and wrong on
            # all the rest:
            #
            #     config_a() sets stride=64
            #     for i in 0..4:
            #       mm()        needs stride=64      <-- right once, wrong three times
            #       config_b()  sets  stride=32
            #
            # An earlier version of this check walked the body once and justified it by claiming a
            # second pass would report nothing new. That was false, and this is the case it missed.
            # Two passes reach the fixpoint for this shape because the second pass starts from the
            # state the first one ended in, and a third would start from the same place again.
            _walk(stmt.body, live, seen, problems, dead, undecidable, keys, depth + 1)
            if stmt.extent > 1:
                _walk(stmt.body, live, seen, problems, dead, undecidable, keys, depth + 1)
            # Leaving the loop, any live value that mentioned its variable is no longer a value here.
            for key, (value, source) in list(live.items()):
                if isinstance(value, Expr) and stmt.var in free_vars(value):
                    live[key] = (_Varies(stmt.var), source)
            continue

        for key, want in stmt.assumes:
            keys.add(key)
            if key not in live:
                problems.append(
                    f"{stmt.instr}: needs {key}={_text(want)}, but no earlier call set {key}. The device "
                    "is configured for whatever ran before this kernel, which is not a property of this "
                    "kernel."
                )
                continue
            have, source = live[key]
            seen[key] = True
            if isinstance(have, _Varies):
                undecidable.append(
                    f"{stmt.instr}: needs {key}={_text(want)}, and the live value was set by {source} "
                    f"from an expression in {have.var!r}, so it differs per iteration of a loop that has "
                    "ended. Which value survives is arithmetic this check does not do -- reported as "
                    "undecidable rather than guessed."
                )
                continue
            if _text(have) != _text(want):
                problems.append(
                    f"{stmt.instr}: needs {key}={_text(want)}, but {source} left {key}={_text(have)} "
                    "live. The device is configured, just not for this call -- which returns wrong data "
                    "rather than an error."
                )

        for key, value in stmt.sets:
            keys.add(key)
            if key in live and not seen.get(key):
                prior, source = live[key]
                if not isinstance(prior, _Varies) and _text(prior) != _text(value):
                    dead.append(
                        f"{source}: set {key}={_text(prior)}, which {stmt.instr} replaced with "
                        f"{_text(value)} before any call read it. That configuration cost a dispatch "
                        "slot and changed nothing."
                    )
            live[key] = (value, stmt.instr)
            seen[key] = False


def check_config(kernel: Kernel) -> ConfigReport:
    """Configuration findings for ``kernel``, target-free.

    Takes no machine: a configuration key is whatever the target's binding chose to call it, and this
    compares values for equality without interpreting them. A machine parameter would suggest it
    validates keys against the device, which it does not and must not -- that belongs to the binding
    that knows what the keys mean.
    """
    problems: list[str] = []
    dead: list[str] = []
    keys: set[str] = set()
    undecidable: list[str] = []
    _walk(kernel.body, {}, {}, problems, dead, undecidable, keys, 0)
    # A loop body is walked twice, so a finding inside one is raised twice. Deduplicated by message
    # rather than by suppressing the second pass: which pass found it is an artefact of how this check
    # is written, and the reader should not have to know that a repeated line means a loop.
    return ConfigReport(
        problems=tuple(dict.fromkeys(problems)),
        dead=tuple(dict.fromkeys(dead)),
        undecidable=tuple(dict.fromkeys(undecidable)),
        keys=tuple(sorted(keys)),
    )


# ---- placement ---------------------------------------------------------------------------------
# Whether a kernel's on-chip placement is sound on a given machine.
#
# Two failures this is written against, both measured on shipped code, both silent.
#
# A DESTINATION WRITTEN ON TOP OF A LIVE OPERAND. A kernel carried a constant destination offset that was
# correct for one tile size and wrong for every larger one, so the result was written over the operand it
# was still reading. Measured: zero correct elements out of 16,384, with no error reported anywhere. The
# program ran, the harness completed, the numbers were wrong.
#
# A BANK SHARED BETWEEN TWO ENGINES. Where a scratchpad arbitrates in favour of one engine, any bank the
# two share starves the other -- so bank assignment decides whether a stage overlaps at all. That is a
# fact about the BANK, and it is invisible from an address.
#
# Neither is a question about the archetype: writing over a live operand returns wrong data whether or not
# hardware tracks dependencies, and an arbiter starves regardless. So unlike :mod:`~merlin.compiler.kernel_ir.checks`
# this asks one question of every machine -- what it needs from the machine is the memory's geometry, not
# its hazard model.


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


# ---- sync --------------------------------------------------------------------------------------
# Whether a kernel's synchronisation is sound ON A GIVEN MACHINE -- which is a different question per
# archetype, and that is the whole point of this file.
#
# The same kernel is correct on one machine and wrong on another, with no change to a single instruction.
# Where the hardware tracks dependencies, a consumer reading a producer's result needs nothing from the
# compiler: reordering cannot change the answer, so an absent wait is not a defect. Where the compiler is
# responsible for separation, the identical omission returns wrong data, silently, at full speed.
#
# So this check takes a :class:`~merlin.compiler.kernel_ir.machine.Machine` and asks the question that machine actually
# poses. A checker that asked one question would be wrong on half the corpus -- and wrong in the dangerous
# direction on the half where it passed everything.
#
# WHAT IT REFUSES TO DECIDE. A machine whose hazard model was never derived gets neither answer. Treating
# an unknown model as interlocked accepts every illegal schedule; treating it as explicit reports defects
# that may not exist. It is reported as undecidable, which is the third state the rest of this repo keeps
# and the reason ``Machine.hazard_resolution`` may be ``None`` but never defaulted.


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

    from merlin.compiler.kernel_ir.ir import has_carried_events

    if has_carried_events(kernel.body):
        return SyncReport(
            decidable=False,
            archetype=hazards,
            problems=tuple(problems),
            checked=len(calls),
            reason="lexical loop-carried completion requires the SCF event-edge/stage checker and target binding",
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


# ---- registry ----------------------------------------------------------------------------------
# Numerics contracts: the declared arithmetic a program's golden is computed under.
#
# A contract is a target-neutral definition of how an integer accumulator becomes the stored output:
# the scale granularity, the rounding, the saturation bounds and the activation. Goldens are generated
# UNDER a contract, and a program is graded against the golden of the contract it declares. Grading a
# per-tensor program against a per-channel golden fails for reasons that have nothing to do with the
# program, and grading a device epilogue against a host integer chain is impossible in principle (the
# device rounds an f32 product; the host chain does not).
#
# Contracts are built from a target's DERIVED readout facts (:func:`from_readout_facts`), i.e. the dict
# a target's own header-verified readout contract produces. The clamp bounds, dtypes and rounding come
# from there, never from a literal here.
#
# Readout arithmetic, exactly as a scalar C readout macro computes it on an FLT_EVAL_METHOD=0 host:
#
#     prod = f32(acc) * f32(scale)        # one IEEE-754 single multiply
#     y    = roundeven(prod)              # round half to even
#     out  = clamp(y, clamp_min, clamp_max)
#     out  = max(out, 0) if activation == "relu"
#
# ``numpy`` float32 arithmetic and ``numpy.rint`` reproduce each step bit-for-bit for |prod| < 2**63,
# which covers every accumulator of the declared width times a finite scale of magnitude < 2**31.

#: How the scale varies over the accumulator tile. ``row``/``column`` are one scale per accumulator
#: row / column of a 2-D tile; ``rank1`` is an outer product of a row vector and a column vector.
GRANULARITIES = ("tensor", "row", "column", "rank1")
#: Roundings a contract may declare. Only the one a readout contract has been verified to implement.
ROUNDINGS = ("half_even",)
#: The readout-facts schema this module knows how to consume.
READOUT_FACTS_SCHEMA = "scalar_narrow_readout_contract_v1"


class ContractError(ValueError):
    """A contract cannot be built, or a readout cannot be evaluated under it."""


def _int_dtype(spelling: str) -> np.dtype:
    """``i8``/``i16``/``i32``/``i64`` -> the numpy signed integer dtype. Fail closed on anything else."""
    if not isinstance(spelling, str) or not spelling.startswith("i"):
        raise ContractError(f"not a signed integer dtype spelling: {spelling!r}")
    bits = spelling[1:]
    if not bits.isdigit() or int(bits) not in (8, 16, 32, 64):
        raise ContractError(f"unsupported integer width in {spelling!r}")
    return np.dtype(f"int{int(bits)}")


@dataclass(frozen=True)
class NumericsContract:
    """One declared readout arithmetic. Two contracts with equal :meth:`digest` compute identical bits."""

    contract_id: str
    accumulator_dtype: str
    output_dtype: str
    scale_dtype: str
    scale_granularity: str
    rounding: str
    clamp_min: int
    clamp_max: int
    activations: tuple[str, ...] = ("none", "relu")
    #: Where the facts came from (header digests, macro names). Recorded, but NOT part of the
    #: semantics: two sources that derive the same arithmetic produce the same digest.
    provenance: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.scale_granularity not in GRANULARITIES:
            raise ContractError(f"unknown scale granularity {self.scale_granularity!r}")
        if self.rounding not in ROUNDINGS:
            raise ContractError(f"unknown rounding {self.rounding!r}")
        if self.scale_dtype != "f32":
            raise ContractError(f"only an f32 scale is declared, got {self.scale_dtype!r}")
        acc = _int_dtype(self.accumulator_dtype)
        out = _int_dtype(self.output_dtype)
        if not self.clamp_min < self.clamp_max:
            raise ContractError("clamp_min must be below clamp_max")
        info = np.iinfo(out)
        if self.clamp_min < info.min or self.clamp_max > info.max:
            raise ContractError(f"clamp [{self.clamp_min}, {self.clamp_max}] exceeds {self.output_dtype}")
        if np.iinfo(acc).bits < info.bits:
            raise ContractError("accumulator narrower than output")
        for act in self.activations:
            if act not in ("none", "relu"):
                raise ContractError(f"unknown activation {act!r}")

    def semantic_fields(self) -> dict[str, Any]:
        """Every field that changes the computed bits (i.e. all but provenance)."""
        return {
            "contract_id": self.contract_id,
            "accumulator_dtype": self.accumulator_dtype,
            "output_dtype": self.output_dtype,
            "scale_dtype": self.scale_dtype,
            "scale_granularity": self.scale_granularity,
            "rounding": self.rounding,
            "clamp_min": self.clamp_min,
            "clamp_max": self.clamp_max,
            "activations": list(self.activations),
        }

    def digest(self) -> str:
        """sha256 of the canonical semantic fields. A golden records this; the grader compares it."""
        blob = json.dumps(self.semantic_fields(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _scale_operand(self, acc: np.ndarray, scale) -> np.ndarray:
        s = np.asarray(scale, dtype=np.float32)
        if self.scale_granularity == "tensor":
            if s.size != 1:
                raise ContractError(f"a per-tensor contract takes one scale, got shape {s.shape}")
            return s.reshape(())
        if self.scale_granularity == "rank1":
            raise ContractError(
                "rank1 readout arithmetic is declared but its product order is not yet fixed; refuse rather than guess"
            )
        if acc.ndim != 2:
            raise ContractError(f"a per-{self.scale_granularity} readout needs a 2-D tile, got {acc.shape}")
        if self.scale_granularity == "row":
            if s.shape != (acc.shape[0],):
                raise ContractError(f"row scales must have shape ({acc.shape[0]},), got {s.shape}")
            return s[:, None]
        if s.shape != (acc.shape[1],):
            raise ContractError(f"column scales must have shape ({acc.shape[1]},), got {s.shape}")
        return s[None, :]

    def readout(self, acc, scale, *, activation: str = "none") -> np.ndarray:
        """Evaluate the readout on an integer accumulator tile. Returns ``output_dtype`` values."""
        if activation not in self.activations:
            raise ContractError(f"activation {activation!r} not admitted by {self.contract_id}")
        a = np.asarray(acc)
        if a.dtype.kind not in "iu":
            raise ContractError(f"accumulator must be an integer array, got {a.dtype}")
        info = np.iinfo(_int_dtype(self.accumulator_dtype))
        if a.size and (a.min() < info.min or a.max() > info.max):
            raise ContractError(f"accumulator value outside {self.accumulator_dtype}")
        s = self._scale_operand(a, scale)
        if not np.all(np.isfinite(s)):
            raise ContractError("non-finite scale")
        prod = a.astype(np.float32) * s  # one float32 multiply, IEEE round-to-nearest
        y = np.rint(prod)  # round half to even, still float32
        y = np.clip(y, self.clamp_min, self.clamp_max)
        if activation == "relu":
            y = np.maximum(y, 0)
        return y.astype(_int_dtype(self.output_dtype))


def from_readout_facts(readout: Mapping[str, Any], *, contract_id: str, granularity: str) -> NumericsContract:
    """Build a contract from a target's derived readout facts.

    ``readout`` is the dict a target's header-verified readout contract returns (schema
    ``scalar_narrow_readout_contract_v1``): accumulator/output/scale dtypes and clamp bounds. That
    schema is only produced after the target's scale and rounding macros were matched against the
    round-half-to-even construction, which is what licenses ``rounding="half_even"`` here.
    """
    if readout.get("schema") != READOUT_FACTS_SCHEMA:
        raise ContractError(f"unsupported readout facts schema {readout.get('schema')!r}")
    try:
        return NumericsContract(
            contract_id=contract_id,
            accumulator_dtype=str(readout["accumulator_dtype"]),
            output_dtype=str(readout["output_dtype"]),
            scale_dtype=str(readout["scale_dtype"]),
            scale_granularity=granularity,
            rounding="half_even",
            clamp_min=int(readout["clamp_min"]),
            clamp_max=int(readout["clamp_max"]),
            provenance=dict(readout.get("provenance") or {}),
        )
    except KeyError as missing:
        raise ContractError(f"readout facts lack {missing}") from None


#: Readout-level contracts, keyed by id. Model-level contracts (bias seeding, residual add, pooling)
#: compose one of these with further stages and are declared separately.
_BUILDERS: dict[str, Callable[[Mapping[str, Any]], NumericsContract]] = {
    "per_tensor_readout_v1": lambda r: from_readout_facts(r, contract_id="per_tensor_readout_v1", granularity="tensor"),
    "per_row_readout_v1": lambda r: from_readout_facts(r, contract_id="per_row_readout_v1", granularity="row"),
    "per_column_readout_v1": lambda r: from_readout_facts(r, contract_id="per_column_readout_v1", granularity="column"),
}


def known_contracts() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))


def contract(contract_id: str, readout: Mapping[str, Any]) -> NumericsContract:
    """The named contract, instantiated from one target's readout facts."""
    try:
        build = _BUILDERS[contract_id]
    except KeyError:
        raise ContractError(f"unknown contract {contract_id!r}; known: {known_contracts()}") from None
    return build(readout)


# ---- cost --------------------------------------------------------------------------------------
# What a schedule costs on a machine -- and, more often, why that cannot be answered exactly.
#
# A cost model's job is not to produce a number. It is to produce a number WITH ITS KIND, because the
# three cases are genuinely different and collapsing them is how a schedule gets ranked by a figure no
# arrangement of it can achieve:
#
# ``exact``        every instruction's cost on its unit was derived, and none of those costs is a floor.
# ``lower_bound``  every cost was derived, but at least one is a FLOOR measured on an idle machine. Real
#                  contention can only make the schedule slower, so the number bounds it from below and
#                  must never be reported as a prediction.
# ``undecidable``  some instruction has no derived cost on the unit it runs on. NOT a zero, and not the
#                  sum of the ones that are known: the missing terms are unbounded, so a partial sum is
#                  smaller than the truth by an unknown amount and reads as the schedule being fast.
#
# The last is the one that matters. A cost model that skipped underived terms would rank a schedule made
# of instructions nobody has measured as the cheapest one available, which is precisely backwards.
#
# WHAT IS AND IS NOT MODELLED. Issue is serial per queue -- units sharing an ordered command stream take
# each other's issue slots -- while EXECUTION overlaps unless two units share a resource. That split is
# the whole reason ``Unit`` separates ``queue`` from ``executes_on``: two units behind one in-order port
# still overlap, and on one target that overlap is the entire performance lever. What is deliberately NOT
# modelled here is anything that would need a data-dependent duration, a bank arbiter or a memory system:
# this prices ISSUE and DECLARED LATENCY, and says so.

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


def _unrolled_calls(body) -> list:
    out = []
    for stmt in body:
        if isinstance(stmt, Loop):
            out.extend(_unrolled_calls(stmt.body) * stmt.extent)
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
    calls = _unrolled_calls(kernel.body)
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


__all__ = [
    "Bound",
    "ConfigReport",
    "ContractError",
    "FlipReport",
    "GRANULARITIES",
    "KINDS",
    "NumericsContract",
    "PlacementReport",
    "SyncReport",
    "check_config",
    "check_placement",
    "check_sync",
    "contract",
    "enumerate_readout_flips",
    "from_readout_facts",
    "known_contracts",
    "price",
    "reachable_accumulator_range",
]

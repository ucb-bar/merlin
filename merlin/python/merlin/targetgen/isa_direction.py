"""Which of an instruction's operands is a DEFINITION and which is a USE -- derived by executing it.

The gap this closes. :meth:`~merlin.targetgen.isa_model.IsaModel.fields_of` gives an instruction's
operand NAMES and the word bits each one occupies. Nothing anywhere says which of those operands the
instruction WRITES and which it READS, and every dependence question -- read-after-write, live range,
register pressure, the separation two instructions must keep -- is a question about exactly that.

The tempting shortcut is to read the direction off the spelling: a name that looks like a destination
is a definition, one that looks like a source is a use. That is an assumption about a naming
convention, not a fact about the machine, and it is the class of assumption these tools exist to
remove. It also fails quietly: a format whose operand list is shared across a whole family declares
every field on every instruction in it, so a unary operation carries a second source operand that it
never reads. Believing that operand creates a dependence edge that does not exist, and an edge that
does not exist is a separation the schedule pays for nothing.

SO THE DIRECTION IS MEASURED, by the same differential-probing method that derived the field layouts:
write known architectural state, execute ONE instruction on the target's functional oracle, and read
the state back. Two comparisons decide the answer, and each is a difference rather than a value:

* **definition** -- run the instruction twice with the operand naming two different registers. If the
  slot that changed FOLLOWS the operand (the slot named by the first value changed on the first run
  and did not change on the second), the operand selects where the instruction writes. That is a
  definition, and the probe also learns WHICH state file it writes, because it saw which one moved.
* **use** -- run the instruction twice with a non-defining operand naming two different registers
  holding different content. If the observable effect differs, the instruction read that operand.

An operand that survives neither comparison is UNKNOWN with the reason recorded -- never "not a use".
The distinction matters in the direction that costs: an operand wrongly called unread drops a real
dependence edge, so a refusal has to stay visible rather than defaulting to the flattering answer.

FOUR THINGS THIS REFUSES, each because the probe genuinely cannot separate them:

* an operand whose word bits OVERLAP another operand's (a shipped encoder that packs a 7-bit source
  field across a 5-bit one shares a bit between them). Varying one silently varies the other, so the
  probe constrains its candidate values to leave every shared bit clear, and refuses the operand when
  two distinct such values do not exist;
* an operand that names a state file the oracle does not publish. A weight buffer that appears in no
  state read-back cannot be observed changing, and "not observed" is not "not written";
* an instruction whose effect is invisible under the preamble it was given -- a readout of an
  accumulator nothing has written reads zeros whichever bank it names;
* an instruction the probe could not run in isolation at all (it did not halt, or it faulted).

Target-agnostic by construction. Nothing here names an instruction, a register file, a field or a
count. The state files are whatever the oracle publishes; the operands are whatever the ISA model
declares; the handful of instructions needed to WRITE the initial state is a :class:`ProbeOps`
selection made by the caller, at the edge that is legitimately about one target -- the same shape as
the role selection the layer-scale workload generator takes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "DEF", "USE", "DEF_USE", "UNKNOWN",
    "DirectionError", "ProbeOps", "ProbeState", "OperandDirection", "DirectionModel",
    "state_from_debug_result", "shared_bits", "candidate_values", "derive_directions",
]

#: The operand selects the register the instruction WRITES.
DEF = "def"
#: The operand selects a register the instruction READS.
USE = "use"
#: Both -- the instruction reads the register it writes (an accumulate).
DEF_USE = "def_use"
#: The probe could not establish a direction. A distinct, inhabited answer: never read as "not a use".
UNKNOWN = "unknown"


#: Strength order for merging verdicts, whether across baselines or across preambles: an observation
#: beats a non-observation, and a definition beats a use, because a definition is the claim that needed
#: the stronger evidence and a use is what is concluded when that evidence was not found.
_STRENGTH = {UNKNOWN: 0, USE: 1, DEF: 2, DEF_USE: 3}


class DirectionError(RuntimeError):
    """The direction probe cannot run at all (no encodable preamble, no observable state)."""


@dataclass(frozen=True)
class ProbeOps:
    """The few instructions the probe needs in order to WRITE state before it reads it back.

    This is the one target-specific input, and it is a selection rather than a fact: a machine may
    offer several ways to materialise a constant and several ways to fill a tensor register, and
    which one the probe uses is a choice. Every name is checked against the derived ISA model before
    any program is emitted, so a mnemonic this target does not define is an error here rather than a
    silently mis-encoded word later.

    ``scalar_imm`` writes a small constant into a scalar register from an immediate; ``scalar_upper``
    supplies the high half so a full-width constant is reachable. ``stall`` holds issue for an
    immediate number of cycles -- REQUIRED, not optional, because this class of machine has no
    interlock: two state-writing instructions issued back to back leave only the first one's effect,
    which the probe would read as an instruction that writes nothing. ``halt`` terminates.

    ``seeders`` are ``(mnemonic, operands)`` pairs executed in order before the instruction under
    test, each followed by a settle. They exist to put NON-ZERO content into every state file the
    probe wants to observe, because an instruction that copies zeros onto zeros changes nothing and
    is indistinguishable from one that does nothing at all.
    """

    scalar_imm: str
    scalar_upper: str
    stall: str
    halt: str
    seeders: tuple[tuple[str, Mapping[str, int]], ...] = ()
    #: Cycles of settle after every state-writing instruction. Measured on the tier the probe runs on
    #: (the workload generator's settle probe supplies it); never picked.
    settle: int = 128
    #: Scalar registers the preamble loads with distinct sentinels, and the sentinel spacing.
    scalar_seeds: int = 8
    #: Operand-value ladder tried for each operand. Values are filtered against the encoding before
    #: use, so a value that does not fit a field, or that would disturb an overlapping one, is
    #: dropped rather than encoded.
    value_ladder: tuple[int, ...] = (0, 1, 16)
    #: Which operand of a seeder names the SLOT it fills. Needed to withdraw one seed at a time, which
    #: is how a use is attributed to a state file: the probe removes the content at exactly the slot
    #: the operand names and asks whether the instruction noticed.
    seed_slot_field: str = "vd"
    #: The operand value the file-attribution phase parks the operand under test at. It has to differ
    #: from the value every OTHER operand holds, or withdrawing a seed perturbs several operands at
    #: once and the answer cannot be attributed to any of them.
    attribution_value: int = 2

    def validate(self, isa: Any) -> None:
        names = [self.scalar_imm, self.scalar_upper, self.stall, self.halt]
        names += [mn for mn, _ in self.seeders]
        missing = sorted({n for n in names if isa.resolve(n) is None})
        if missing:
            raise DirectionError(
                f"{getattr(isa, 'target', '?')!r} defines no instruction for probe op(s): "
                f"{', '.join(missing)}")
        if int(self.settle) <= 0:
            raise DirectionError(
                "the probe settle is zero; on a machine with no interlock that makes every seeded "
                "state write invisible, and every instruction look as though it writes nothing")


@dataclass(frozen=True)
class ProbeState:
    """One read-back of architectural state, as ``{file: (slot_0, slot_1, ...)}``.

    A slot value is whatever the oracle publishes for it -- a register's contents where the oracle
    publishes values, a per-bank "holds something" flag where it publishes only presence. The probe
    never interprets a slot; it only asks whether two read-backs differ at it, which is all a
    difference-based derivation needs and is what lets a presence-only file carry the same weight as
    a value file.
    """

    files: Mapping[str, tuple]

    def changed_against(self, other: "ProbeState") -> dict[str, tuple[int, ...]]:
        """``{file: slots that differ}`` -- WHERE the instruction wrote. This is the view a definition
        is derived from, because a definition is a claim about a location."""
        return {name: tuple(slot for slot, _ in pairs)
                for name, pairs in self.signature_against(other).items()}

    def signature_against(self, other: "ProbeState") -> dict[str, tuple[tuple[int, Any], ...]]:
        """``{file: ((slot, new value), ...)}`` -- WHERE the instruction wrote AND WHAT it left there.

        Kept apart from :meth:`changed_against` on purpose. A use is a claim that the instruction READ
        something, and the evidence for that is a changed RESULT, which a location-only view cannot
        see: an arithmetic instruction writes the same register whatever its sources hold, so its
        sources are invisible to a slot comparison and plain in a value comparison. A file the oracle
        publishes as presence only carries a boolean here, so it contributes what it can and no more.
        """
        out: dict[str, tuple[tuple[int, Any], ...]] = {}
        for name, mine in self.files.items():
            theirs = other.files.get(name)
            if theirs is None:
                out[name] = tuple((i, v) for i, v in enumerate(mine))
                continue
            pairs = tuple((i, a) for i, (a, b) in enumerate(zip(mine, theirs)) if a != b)
            extra = tuple((i, mine[i]) for i in range(len(theirs), len(mine)))
            if pairs or extra:
                out[name] = pairs + extra
        return out

    def value_published(self, name: str) -> bool:
        """True when this file's slots carry values rather than a bare "holds something" flag.

        A presence-only file can still evidence a definition (the slot the instruction named is the
        one that lit up), but it cannot evidence what was written, so nothing that depends on the
        WRITTEN VALUE -- a read-modify-write, above all -- is separable there."""
        slots = self.files.get(name) or ()
        return any(not isinstance(v, bool) for v in slots)

    def differs_from(self, other: "ProbeState") -> bool:
        return bool(self.signature_against(other))


def state_from_debug_result(res: Mapping[str, Any]) -> ProbeState:
    """Flatten a functional-oracle debug read-back into a :class:`ProbeState`.

    Reads the runner's own published debug contract -- a scalar register vector under ``regs`` and a
    value-free on-chip population map under ``on_chip`` -- and nothing else. The file NAMES are the
    runner's; this function invents none and drops none, so a target whose oracle publishes a further
    file gets it observed for free, and one that publishes fewer simply has fewer observable operands
    (and correspondingly more honest refusals).
    """
    files: dict[str, tuple] = {}
    regs = res.get("regs")
    if isinstance(regs, (list, tuple)):
        files["scalar"] = tuple(int(v) for v in regs)
    on_chip = res.get("on_chip") or res.get("state_summary") or {}
    for name, value in (on_chip or {}).items():
        if isinstance(value, (list, tuple)):
            files[name] = tuple(value)
        elif isinstance(value, Mapping):
            for sub, seq in value.items():
                if isinstance(seq, (list, tuple)):
                    files[f"{name}.{sub}"] = tuple(seq)
        else:
            files[name] = (value,)
    if not files:
        raise DirectionError(
            "the oracle read-back publishes no architectural state, so no operand direction is "
            "observable; this is UNKNOWN for every operand, not an instruction that writes nothing")
    return ProbeState(files=files)


# ---------------------------------------------------------------------------------------------------
# encoding hygiene: which operand values can be varied without disturbing another operand
# ---------------------------------------------------------------------------------------------------
def shared_bits(fields: Mapping[str, Sequence[int | None]]) -> dict[str, frozenset[int]]:
    """``{operand: word bits it shares with some OTHER operand of the same instruction}``.

    A shipped encoder is free to overlap two operand fields, and one measured here does: a source
    field one bit wider than the register file it indexes lends its top bit to the next field along.
    Varying either operand then varies both, and a probe that ignores this reports the direction of
    whichever operand happened to move -- confidently, and possibly wrongly. So the overlap is
    computed first, and every candidate value is required to leave the shared bits clear.
    """
    owners: dict[int, set[str]] = {}
    for attr, bits in fields.items():
        for b in bits:
            if isinstance(b, int) and b >= 0:
                owners.setdefault(b, set()).add(attr)
    out: dict[str, frozenset[int]] = {}
    for attr, bits in fields.items():
        shared = {b for b in bits
                  if isinstance(b, int) and b >= 0 and len(owners.get(b, ())) > 1}
        out[attr] = frozenset(shared)
    return out


def candidate_values(fields: Mapping[str, Sequence[int | None]], attr: str,
                     ladder: Sequence[int]) -> list[int]:
    """The values of ``attr`` this encoding can carry WITHOUT disturbing any other operand.

    A value is kept when every operand bit it sets maps to a real, linear word bit that ``attr`` does
    not share with a sibling. Fewer than two survivors means the operand cannot be varied
    independently at all, and the caller refuses it rather than probing through the overlap.
    """
    bits = list(fields.get(attr) or ())
    if not bits or any(b == -1 for b in bits):
        return []
    shared = shared_bits(fields).get(attr, frozenset())
    keep: list[int] = []
    for value in ladder:
        if value < 0:
            continue
        ok = True
        for i, word_bit in enumerate(bits):
            if not (value >> i) & 1:
                continue
            if not isinstance(word_bit, int) or word_bit < 0 or word_bit in shared:
                ok = False
                break
        if ok and (value >> len(bits)) == 0:
            keep.append(int(value))
    return keep


# ---------------------------------------------------------------------------------------------------
# the derived answer
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OperandDirection:
    """One operand's measured direction, the state file it reaches, and how that was established."""

    mnemonic: str
    operand: str
    direction: str                    # DEF | USE | DEF_USE | UNKNOWN
    #: The state file the operand indexes, where a definition revealed it. None when not established.
    state_file: str | None
    #: Slots the instruction wrote, per probed operand value -- the evidence for a definition.
    written_slots: tuple[int, ...] = ()
    reason: str = ""

    @property
    def known(self) -> bool:
        return self.direction != UNKNOWN

    def claim(self) -> str:
        if self.direction == UNKNOWN:
            return f"{self.mnemonic}.{self.operand}: direction UNKNOWN -- {self.reason}"
        where = f" in {self.state_file}" if self.state_file else ""
        return f"{self.mnemonic}.{self.operand}: {self.direction}{where} ({self.reason})"


@dataclass(frozen=True)
class DirectionModel:
    """Every probed instruction's operand directions, plus what the probe refused and why."""

    target: str
    by_mnemonic: Mapping[str, Mapping[str, OperandDirection]] = field(default_factory=dict)
    #: Instructions the probe could not execute in isolation at all, with the reason each.
    refused: Mapping[str, str] = field(default_factory=dict)
    provenance: str = ""

    def defs_of(self, mnemonic: str) -> tuple[str, ...]:
        """Operands this instruction WRITES. Empty for an instruction that was never probed -- callers
        that need the difference between "writes nothing" and "not established" ask :meth:`resolved`."""
        return tuple(sorted(o for o, d in (self.by_mnemonic.get(mnemonic) or {}).items()
                            if d.direction in (DEF, DEF_USE)))

    def uses_of(self, mnemonic: str) -> tuple[str, ...]:
        return tuple(sorted(o for o, d in (self.by_mnemonic.get(mnemonic) or {}).items()
                            if d.direction in (USE, DEF_USE)))

    def file_of(self, mnemonic: str, operand: str) -> str | None:
        d = (self.by_mnemonic.get(mnemonic) or {}).get(operand)
        return d.state_file if d else None

    def resolved(self, mnemonic: str) -> bool:
        """True when EVERY declared operand of this instruction has a measured direction.

        The all-or-nothing bar is deliberate. A dependence built from an instruction with one
        unresolved operand is a dependence with an unknown edge in it, and consuming it as though it
        were complete is how a missing edge becomes a schedule that reads a stale register."""
        ent = self.by_mnemonic.get(mnemonic)
        return bool(ent) and all(d.known for d in ent.values())

    def unknown_operands(self) -> tuple[str, ...]:
        return tuple(sorted(f"{m}.{o}: {d.reason}" for m, ops in self.by_mnemonic.items()
                            for o, d in ops.items() if not d.known))

    def summary(self) -> dict:
        probed = len(self.by_mnemonic)
        full = sum(1 for m in self.by_mnemonic if self.resolved(m))
        operands = [d for ops in self.by_mnemonic.values() for d in ops.values()]
        return {
            "target": self.target,
            "instructions_probed": probed,
            "instructions_fully_resolved": full,
            "instructions_refused": len(self.refused),
            "operands": len(operands),
            "operands_def": sum(1 for d in operands if d.direction in (DEF, DEF_USE)),
            "operands_use": sum(1 for d in operands if d.direction in (USE, DEF_USE)),
            "operands_unknown": sum(1 for d in operands if not d.known),
            "provenance": self.provenance,
        }

    def to_json(self) -> dict:
        return {
            "target": self.target,
            "provenance": self.provenance,
            "summary": self.summary(),
            "refused": dict(self.refused),
            "by_mnemonic": {
                m: {o: {"direction": d.direction, "state_file": d.state_file,
                        "written_slots": list(d.written_slots), "reason": d.reason}
                    for o, d in sorted(ops.items())}
                for m, ops in sorted(self.by_mnemonic.items())},
        }

    def merge(self, other: "DirectionModel") -> "DirectionModel":
        """Combine two derivations of the same target, keeping the STRONGER evidence for each operand.

        More than one run is not redundancy, it is necessity. A state file the oracle publishes as
        presence only, with few enough slots that the probe cannot leave one both filled and empty,
        cannot evidence a write and a read under the same initial state: fill the accumulator and
        every definition into it is hidden (the bank already held something), leave it empty and every
        read of it is hidden (there was nothing to read). Two preambles resolve between them what
        neither resolves alone.

        WHY THE STRONGER VERDICT WINS RATHER THAN A DISAGREEMENT BEING REFUSED. The two verdicts are
        not symmetric claims. A definition is positive evidence -- the slot that changed tracked the
        operand, twice, in opposite directions. A use is the RESIDUAL conclusion drawn when that
        evidence was not found and the effect nonetheless varied. So a preamble reporting a use where
        another reports a definition is not contradicting it; it is reporting that it could not see
        what the other saw, which is exactly what a preamble that pre-filled the destination cannot
        see. Refusing both would throw away the observation and keep the failure to observe."""
        by: dict[str, dict[str, OperandDirection]] = {m: dict(ops) for m, ops in self.by_mnemonic.items()}
        for mnemonic, ops in other.by_mnemonic.items():
            slot = by.setdefault(mnemonic, {})
            for attr, theirs in ops.items():
                mine = slot.get(attr)
                if mine is None or _STRENGTH[theirs.direction] > _STRENGTH[mine.direction]:
                    slot[attr] = theirs
        refused = {m: why for m, why in {**dict(other.refused), **dict(self.refused)}.items()
                   if m not in by}
        return DirectionModel(target=self.target or other.target, by_mnemonic=by, refused=refused,
                              provenance=" + ".join(x for x in (self.provenance, other.provenance) if x))

    @classmethod
    def from_json(cls, blob: Mapping[str, Any]) -> "DirectionModel":
        by: dict[str, dict[str, OperandDirection]] = {}
        for m, ops in (blob.get("by_mnemonic") or {}).items():
            by[m] = {o: OperandDirection(mnemonic=m, operand=o, direction=str(e.get("direction")),
                                         state_file=e.get("state_file"),
                                         written_slots=tuple(e.get("written_slots") or ()),
                                         reason=str(e.get("reason") or ""))
                     for o, e in ops.items()}
        return cls(target=str(blob.get("target") or ""), by_mnemonic=by,
                   refused=dict(blob.get("refused") or {}),
                   provenance=str(blob.get("provenance") or ""))


# ---------------------------------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------------------------------
class _Emitter:
    """A flat instruction list encoded through the derived field maps. No labels: every probe program
    is straight-line by construction, because a probe that branches is no longer executing ONE
    instruction under known conditions."""

    def __init__(self, isa: Any, ops: ProbeOps):
        self._isa = isa
        self._ops = ops
        self.items: list[tuple[str, dict]] = []

    def emit(self, mnemonic: str, **operands: int) -> None:
        self.items.append((mnemonic, dict(operands)))

    def settle(self, cycles: int | None = None) -> None:
        fields = self._isa.fields_of(self._ops.stall)
        args = {name: 0 for name in fields}
        if "imm" not in fields:
            raise DirectionError(
                f"the selected stall {self._ops.stall!r} carries no immediate field, so the probe "
                "cannot hold issue for a measured number of cycles")
        args["imm"] = int(self._ops.settle if cycles is None else cycles)
        self.emit(self._ops.stall, **args)

    def _imm_width(self, mnemonic: str) -> int:
        bits = self._isa.fields_of(mnemonic).get("imm")
        if not bits:
            raise DirectionError(f"{mnemonic!r} carries no immediate field")
        return len(bits)

    def load_imm(self, rd: int, value: int) -> None:
        """Materialise a constant the way the ISA allows: an upper immediate plus a sign-corrected
        low add, exactly as the workload generator does. The low add sign-extends, so a low half with
        its top bit set is compensated in the upper half."""
        width = self._imm_width(self._ops.scalar_imm)
        lo_mask = (1 << width) - 1
        sign = 1 << (width - 1)
        lo = value & lo_mask
        hi = (value - (lo - (1 << width) if lo & sign else lo)) >> width
        upper_width = self._imm_width(self._ops.scalar_upper)
        if hi:
            if hi >> upper_width:
                raise DirectionError(f"constant {value} does not fit this ISA's upper immediate")
            self.emit(self._ops.scalar_upper, rd=rd, imm=hi & ((1 << upper_width) - 1))
            if lo:
                self.emit(self._ops.scalar_imm, rd=rd, rs1=rd, imm=lo)
        else:
            self.emit(self._ops.scalar_imm, rd=rd, rs1=0, imm=lo)

    def kernel_s(self, entry: str = "_start") -> str:
        from merlin.targetgen import isa_asm
        lines = [".section .text", f".globl {entry}", f".type {entry},@function", f"{entry}:"]
        for i, (mn, operands) in enumerate(self.items):
            word = isa_asm.assemble_line(self._isa, mn, operands)
            args = ", ".join(f"{k}={v}" for k, v in operands.items())
            lines.append(f"  .word 0x{word:08x}    # [{i}] {mn}{' ' + args if args else ''}")
        return "\n".join(lines) + "\n"


#: The two sentinel patterns the scalar preamble writes. Distinct at every index and distinct between
#: the two variants, so "this register changed" and "this register changed DIFFERENTLY" are both
#: readable. Nothing about the machine is encoded here: they are arbitrary marks.
_SENTINEL = (0x1000, 0x11)
_SENTINEL_ALT = (0x2000, 0x13)


def _preamble(isa: Any, ops: ProbeOps, *, scalar_perturb: int | None = None,
              omit_seed_slot: int | None = None) -> _Emitter:
    """Known architectural state: distinct sentinels in the scalar file, then the caller's seeders,
    each followed by a settle so its effect is actually committed before the next one issues.

    The two perturbations are what let a USE be attributed to a state file. ``scalar_perturb`` gives
    one scalar slot a different sentinel -- observable because the oracle publishes scalar VALUES.
    ``omit_seed_slot`` withdraws one seeder entirely, leaving its slot empty -- which is the only
    perturbation a presence-only file can show, since two different non-zero contents are the same
    "holds something" bit and no downstream instruction can distinguish them either."""
    e = _Emitter(isa, ops)
    for index in range(1, int(ops.scalar_seeds) + 1):
        base, step = _SENTINEL_ALT if index == scalar_perturb else _SENTINEL
        e.load_imm(index, base + index * step)
    for mnemonic, operands in ops.seeders:
        operands = dict(operands)
        if omit_seed_slot is not None and operands.get(ops.seed_slot_field) == omit_seed_slot:
            continue
        e.emit(mnemonic, **operands)
        e.settle()
    return e


def _program(isa: Any, ops: ProbeOps, under_test: tuple[str, dict] | None, *,
             scalar_perturb: int | None = None, omit_seed_slot: int | None = None) -> str:
    e = _preamble(isa, ops, scalar_perturb=scalar_perturb, omit_seed_slot=omit_seed_slot)
    if under_test is not None:
        e.emit(under_test[0], **under_test[1])
        e.settle()
    e.settle()
    # The terminator is emitted with NO operands, so it encodes as the model's own zero-operand word.
    # Writing its declared fields explicitly is how a terminator becomes a DIFFERENT system
    # instruction: a family that discriminates its members inside the immediate carries that
    # discriminator as a class default, and spelling the field out overwrites it.
    e.emit(ops.halt)
    return e.kernel_s()


def _base_operands(isa: Any, mnemonic: str, ladder: Sequence[int]) -> dict[str, int]:
    """The operand assignment every variant is measured against.

    Each operand sits at the SECOND value the encoding can carry cleanly where there is one, not the
    first. The reason is the register-file convention this class of machine shares: index zero is the
    fixed-zero register, so an instruction whose destination operand is parked there writes nothing
    and the whole probe compares two runs that both did nothing. Preferring the next value keeps the
    baseline run observable, which is what every later comparison is measured against."""
    fields = isa.fields_of(mnemonic)
    out: dict[str, int] = {}
    for attr in fields:
        values = candidate_values(fields, attr, ladder)
        out[attr] = (values[1] if len(values) > 1 else values[0]) if values else 0
    return out


def _baseline_assignments(isa: Any, mnemonic: str, ladder: Sequence[int]) -> list[dict[str, int]]:
    """The operand assignments the trials are run against -- more than one, and every one of them.

    Which assignment makes an instruction observable cannot be decided in advance, and picking wrong
    does not fail loudly, it fails silently. Park every operand at the register file's fixed-zero
    index and an arithmetic instruction writes a zero, which is invisible in a slot that already held
    nothing. Park them one index up and an operand that indexes a two-slot file addresses a slot that
    does not exist, so the instruction does nothing at all. Both produce a baseline in which nothing
    happened, and a probe measured against one of those learns nothing about anything.

    So both are tried, the trials run under each, and the STRONGEST verdict wins -- a definition seen
    under either assignment is a definition, because it was observed. Nothing is inferred from the
    assignment that failed to show it; the failure just means that assignment could not see it."""
    fields = isa.fields_of(mnemonic)
    values = {attr: candidate_values(fields, attr, ladder) for attr in fields}
    out: list[dict[str, int]] = []
    # First candidate: the smallest index that is neither the register file's conventional fixed zero
    # nor larger than a small file can hold -- and where the encoding cannot express that index at
    # all (an operand sharing its low bit with a sibling), the fixed zero rather than the next value
    # up, which on this kind of layout jumps to a slot most files do not have.
    preferred = {attr: (1 if 1 in v else (v[0] if v else 0)) for attr, v in values.items() if v}
    out.append(preferred)
    for rank in (1, 0):
        assignment = {attr: v[min(rank, len(v) - 1)] for attr, v in values.items() if v}
        if assignment not in out:
            out.append(assignment)
    return out


def derive_directions(isa: Any, ops: ProbeOps,
                      run_probe: "Callable[[str], Mapping[str, Any]]", *,
                      mnemonics: Sequence[str] | None = None,
                      progress: "Callable[[str, str], None] | None" = None) -> DirectionModel:
    """Measure every operand's direction by running the instruction on the target's own oracle.

    ``run_probe(kernel_s)`` executes one straight-line probe program and returns the oracle's debug
    read-back (``regs`` / ``on_chip``, plus ``halted``). It is supplied by the caller because the
    oracle, the tier and the budget are the caller's; nothing about them is decided here.

    The method, per instruction:

    1. Run the preamble ALONE. That read-back is the reference -- what the state looks like when the
       instruction under test did not execute at all. Comparing against a reference rather than
       against an assumed initial state is what makes every answer below a difference.
    2. For each BASELINE operand assignment in turn (see :func:`_baseline_assignments`), run the
       instruction under it, then re-run it once per candidate value of each operand, moving that one
       operand and holding the rest. Several baselines because no single one makes every instruction
       observable, and several values per operand because one pair does not always separate a wide
       result that lands in a RUN of slots.
    3. Classify from the change-sets. If the changed slot FOLLOWS the operand for some pair of
       values, it is a definition, and the file that moved is its file. If it does not follow but the
       instruction's effect differs across the operand's values, the instruction read it, so it is a
       use. If the effect is identical at every value, the probe saw no dependence and says so --
       which is weaker than "unused", and is recorded as UNKNOWN.
    4. A definition whose WRITTEN VALUE also depends on what the destination held is a definition and
       a use both; that is separable only on a file the oracle publishes as values, and is recorded
       as unseparated on a presence-only file rather than resolved by assumption.
    5. A use is then attributed to a state file by taking the content of the slot it names away, one
       file at a time, and asking whether the instruction noticed.

    Across baselines the STRONGEST verdict wins, because a definition seen under any assignment was
    observed and one that was not merely was not observed. Every failure mode is recorded, never
    smoothed: an instruction that does not halt, one whose program cannot be encoded, and one whose
    operands cannot be varied without disturbing a sibling all end up in the model with the reason
    attached.
    """
    ops.validate(isa)
    ladder = tuple(ops.value_ladder)

    reference_res = run_probe(_program(isa, ops, None))
    if not reference_res.get("halted"):
        raise DirectionError(
            "the probe PREAMBLE alone did not halt on this oracle, so no instruction can be measured "
            "against it; fix the preamble (or its budget) before reading any direction")
    reference = state_from_debug_result(reference_res)

    names = list(mnemonics) if mnemonics is not None else sorted(isa.by_mnemonic)
    verdict_none = OperandDirection("", "", UNKNOWN, None, reason="not yet probed")
    by_mnemonic: dict[str, dict[str, OperandDirection]] = {}
    refused: dict[str, str] = {}

    for mnemonic in names:
        fields = isa.fields_of(mnemonic)
        if not fields:
            refused[mnemonic] = "the ISA model declares no operands for it, so it has no directions"
            continue
        best: dict[str, OperandDirection] = {}
        failure: str | None = None
        for base_ops in _baseline_assignments(isa, mnemonic, ladder):
            try:
                base_res = run_probe(_program(isa, ops, (mnemonic, base_ops)))
            except Exception as exc:  # noqa: BLE001 - a program we cannot emit or run is a refusal
                failure = f"the probe program could not be run: {type(exc).__name__}: {exc}"
                continue
            if not base_res.get("halted"):
                failure = (f"the probe program did not halt "
                           f"({base_res.get('halt_reason') or 'no reason given'}), so its effect on "
                           "state cannot be attributed to this instruction")
                continue
            base_state = state_from_debug_result(base_res)
            base_changed = base_state.changed_against(reference)
            base_signature = base_state.signature_against(reference)
            if progress is not None:
                progress(mnemonic, "base")
            for attr in sorted(fields):
                values = candidate_values(fields, attr, ladder)
                if len(values) < 2:
                    shared = sorted(shared_bits(fields).get(attr, frozenset()))
                    verdict = OperandDirection(
                        mnemonic, attr, UNKNOWN, None, reason=(
                            f"this encoding cannot carry two distinct values of {attr!r} without "
                            f"disturbing an overlapping operand (word bits shared: "
                            f"{shared or 'none'}), so varying it does not isolate it"))
                    best.setdefault(attr, verdict)
                    continue
                if _STRENGTH.get((best.get(attr) or verdict_none).direction, 0) >= _STRENGTH[DEF]:
                    continue                       # already established as strongly as it can be
                trials: list[tuple[int, dict, dict]] = []
                stopped: str | None = None
                for value in values:
                    if value == base_ops.get(attr):
                        trials.append((value, base_changed, base_signature))
                        continue
                    variant = dict(base_ops)
                    variant[attr] = value
                    try:
                        res = run_probe(_program(isa, ops, (mnemonic, variant)))
                    except Exception as exc:  # noqa: BLE001 - a variant we cannot run is evidence lost
                        stopped = (f"the {attr}={value} variant could not be run: "
                                   f"{type(exc).__name__}: {exc}")
                        break
                    if not res.get("halted"):
                        stopped = (f"the {attr}={value} variant did not halt "
                                   f"({res.get('halt_reason') or 'no reason given'})")
                        break
                    st = state_from_debug_result(res)
                    trials.append((value, st.changed_against(reference),
                                   st.signature_against(reference)))
                verdict = (OperandDirection(mnemonic, attr, UNKNOWN, None, reason=stopped)
                           if stopped is not None
                           else _classify(mnemonic, attr, trials, reference))
                if verdict.direction == USE:
                    verdict = _attribute_use(isa, ops, run_probe, mnemonic, attr, base_ops,
                                             verdict, reference, ladder)
                current = best.get(attr)
                if current is None or _STRENGTH[verdict.direction] > _STRENGTH[current.direction]:
                    best[attr] = verdict
                if progress is not None:
                    progress(mnemonic, attr)
        if not best:
            refused[mnemonic] = failure or "no baseline assignment could be run"
            continue
        by_mnemonic[mnemonic] = best

    files = ", ".join(f"{k}[{len(v)}]" for k, v in sorted(reference.files.items()))
    provenance = (
        f"differential probe on the functional oracle: preamble seeds {ops.scalar_seeds} scalar "
        f"sentinel(s) and {len(ops.seeders)} state seeder(s), settle {ops.settle} cycles; "
        f"observed state files {files}")
    return DirectionModel(target=str(getattr(isa, "target", "") or ""), by_mnemonic=by_mnemonic,
                          refused=refused, provenance=provenance)


def _classify(mnemonic: str, attr: str,
              trials: "Sequence[tuple[int, Mapping[str, tuple[int, ...]], Mapping[str, tuple]]]",
              reference: ProbeState) -> OperandDirection:
    """Decide one operand's direction from the effects its several values produced.

    A DEFINITION is the strong reading and needs the strong evidence: for some pair of probed values
    the slot each one NAMES has to be among the slots that changed when it named it, and among the
    slots that did NOT change when the other value named something else. Both halves matter -- an
    instruction that writes a fixed slot whatever its operand says satisfies the first half alone,
    and calling that a definition would attribute a dependence to a register it never touches.

    Several values rather than two, because one pair is not always separable even when the operand is
    plainly a destination: a wide result occupies a RUN of consecutive slots, so two adjacent values
    write overlapping ranges and neither excludes the other. A pair further apart separates them, and
    asking for SOME pair to separate rather than ALL of them is what makes the answer robust to a
    register file whose slots the caller cannot know the grouping of in advance.

    A DEFINITION that also depends on what the destination already held is a read-modify-write, and
    it is separated only where the file publishes VALUES: with the sources held fixed, a pure
    definition leaves the same value wherever it writes, and an accumulate does not. On a file the
    oracle publishes as presence only, that separation is impossible and is recorded as such rather
    than resolved in the flattering direction.
    """
    if not any(sig for _v, _c, sig in trials):
        return OperandDirection(mnemonic, attr, UNKNOWN, None, reason=(
            "the instruction changed no observable state at any probed value of this operand, so "
            "nothing about it is established (the effect may be real and simply unobserved)"))

    files = sorted({name for _v, changed, _s in trials for name in changed})
    for file_name in files:
        for i, (v_a, changed_a, _sa) in enumerate(trials):
            for v_b, changed_b, _sb in trials[i + 1:]:
                in_a = set(changed_a.get(file_name, ()))
                in_b = set(changed_b.get(file_name, ()))
                if not (v_a in in_a and v_a not in in_b and v_b in in_b and v_b not in in_a):
                    continue
                direction, note = _read_modify_write(attr, file_name, trials, reference)
                return OperandDirection(
                    mnemonic, attr, direction, file_name,
                    written_slots=_widest_run(file_name, trials),
                    reason=(f"the changed slot followed the operand: {attr}={v_a} changed slot {v_a} "
                            f"of {file_name} and not slot {v_b}, and {attr}={v_b} the reverse{note}"))

    signatures = {tuple(sorted((n, tuple(p)) for n, p in sig.items())) for _v, _c, sig in trials}
    if len(signatures) > 1:
        values = ", ".join(str(v) for v, _c, _s in trials)
        return OperandDirection(
            mnemonic, attr, USE, None, reason=(
                f"the instruction's effect differed across {attr} in ({values}) without the written "
                "slot following it, so the instruction read this operand rather than writing through it"))

    return OperandDirection(mnemonic, attr, UNKNOWN, None, reason=(
        f"the effect was identical at every probed value of {attr}; the probe saw no dependence on "
        "this operand, which is weaker than establishing that it is unread"))


def _widest_run(file_name: str,
                trials: "Sequence[tuple[int, Mapping[str, tuple[int, ...]], Mapping[str, tuple]]]"
                ) -> tuple[int, ...]:
    """The widest run of consecutive slots any probed value wrote, starting at the slot it named.

    A definition is not always one slot wide: a result in a format wider than a register lands in a
    RUN of them, and how many is a fact about the datapath rather than something to assume. Taking the
    widest run seen, rather than the first, matters because a destination that lands on top of already
    occupied slots shows only its leading edge -- the run is under-counted exactly where the preamble
    happened to have filled the neighbours."""
    best: tuple[int, ...] = ()
    for value, changed, _sig in trials:
        slots = set(changed.get(file_name, ()))
        run = []
        cur = value
        while cur in slots:
            run.append(cur)
            cur += 1
        if len(run) > len(best):
            best = tuple(run)
    return best


def _read_modify_write(attr: str, file_name: str,
                       trials: "Sequence[tuple[int, Mapping[str, tuple[int, ...]], Mapping[str, tuple]]]",
                       reference: ProbeState) -> tuple[str, str]:
    """Whether a definition also READS its destination, where the file publishes values to tell."""
    if not reference.value_published(file_name):
        return DEF, (f"; whether it also reads {file_name} is NOT separable -- the oracle publishes "
                     "that file as presence only, so the written value cannot be compared")
    written: set = set()
    for value, _changed, sig in trials:
        for slot, new in sig.get(file_name, ()):
            if slot == value:
                written.add(new)
    if len(written) > 1:
        return DEF_USE, ("; the value written differed between destinations while the sources were "
                         "held fixed, so the instruction also read what the destination held")
    return DEF, "; the value written was the same at every destination, so it does not read it"


def _perturbation_files(isa: Any, ops: ProbeOps, run_probe: "Callable[[str], Mapping[str, Any]]",
                        reference: ProbeState, slot: int) -> dict[str, tuple[str, ProbeState]]:
    """Which state file each perturbation of ``slot`` moves, and the state the preamble then leaves.

    The perturbation NAMES its own file: run the preamble with one slot's content altered, diff it
    against the unperturbed preamble, and whatever moved is the file that slot lives in. Nothing is
    declared here -- a target whose oracle publishes a further file gets that file identified the same
    way, and a perturbation that moves nothing, or moves several files at once, is not usable as an
    attribution and is dropped.

    The perturbed preamble state comes back with it, and that is the load-bearing part: the
    instruction under test has to be measured against the preamble IT ran under, not against the
    unperturbed one. Measured against the wrong baseline, every perturbation "changes the effect" --
    because the perturbation itself is in the diff -- and every operand is attributed to every file.
    """
    found: dict[str, tuple[str, ProbeState]] = {}
    for kind, kwargs in (("scalar", {"scalar_perturb": slot}), ("seed", {"omit_seed_slot": slot})):
        try:
            res = run_probe(_program(isa, ops, None, **kwargs))
        except Exception:  # noqa: BLE001 - a preamble variant we cannot run is simply not available
            continue
        if not res.get("halted"):
            continue
        state = state_from_debug_result(res)
        moved = sorted(state.changed_against(reference))
        if len(moved) == 1:
            found[kind] = (moved[0], state)
    return found


def _attribute_use(isa: Any, ops: ProbeOps, run_probe: "Callable[[str], Mapping[str, Any]]",
                   mnemonic: str, attr: str, base_ops: Mapping[str, int],
                   verdict: OperandDirection, reference: ProbeState,
                   ladder: Sequence[int]) -> OperandDirection:
    """Which state FILE a use operand reads -- established by taking that content away.

    Knowing an instruction reads an operand is not enough to build a dependence: an edge joins a use
    to the definition of the SAME register in the SAME file, and a load reads its address from one
    file and writes its result into another. So the file is measured, not inferred from the operand's
    spelling or its field width.

    The method parks the operand under test at a slot no other operand of the instruction names, then
    withdraws the content of that slot one file at a time. The file whose withdrawal changes the
    instruction's effect is the file the operand reads. Withdrawal rather than alteration, because a
    file the oracle publishes only as presence cannot show two different contents apart -- absence is
    the one difference it can show.

    Attribution that does not land leaves the operand a USE with no file, which is honest and is
    strictly weaker: a dependence consumer must then treat the operand as an edge it cannot place.
    """
    fields = isa.fields_of(mnemonic)
    slot = int(ops.attribution_value)
    if not candidate_values(fields, attr, (slot,)):
        return OperandDirection(mnemonic, attr, USE, None, written_slots=verdict.written_slots,
                                reason=verdict.reason + (
                                    f"; its file was not attributed -- the encoding cannot carry "
                                    f"{attr}={slot}, the only value distinct from every sibling's"))
    if any(v == slot for k, v in base_ops.items() if k != attr):
        return OperandDirection(mnemonic, attr, USE, None, written_slots=verdict.written_slots,
                                reason=verdict.reason + (
                                    f"; its file was not attributed -- another operand also names "
                                    f"slot {slot}, so withdrawing that slot perturbs both"))
    assignment = dict(base_ops)
    assignment[attr] = slot
    files = _perturbation_files(isa, ops, run_probe, reference, slot)
    runs: dict[str, Any] = {}
    for tag, kwargs, baseline in (("plain", {}, reference),
                                  ("scalar", {"scalar_perturb": slot},
                                   (files.get("scalar") or (None, None))[1]),
                                  ("seed", {"omit_seed_slot": slot},
                                   (files.get("seed") or (None, None))[1])):
        if baseline is None:
            continue
        try:
            res = run_probe(_program(isa, ops, (mnemonic, assignment), **kwargs))
        except Exception as exc:  # noqa: BLE001
            return OperandDirection(mnemonic, attr, USE, None, written_slots=verdict.written_slots,
                                    reason=verdict.reason + (
                                        f"; its file was not attributed -- the {tag} run failed: "
                                        f"{type(exc).__name__}: {exc}"))
        if not res.get("halted"):
            return OperandDirection(mnemonic, attr, USE, None, written_slots=verdict.written_slots,
                                    reason=verdict.reason + (
                                        f"; its file was not attributed -- the {tag} run did not halt"))
        runs[tag] = state_from_debug_result(res).signature_against(baseline)

    hits = [tag for tag in ("scalar", "seed") if tag in runs and runs[tag] != runs["plain"]]
    named = [files[tag][0] for tag in hits if tag in files]
    if len(named) == 1:
        return OperandDirection(mnemonic, attr, USE, named[0], written_slots=verdict.written_slots,
                                reason=verdict.reason + (
                                    f"; withdrawing the content of slot {slot} of {named[0]} changed "
                                    f"the instruction's effect, so that is the file it reads"))
    if len(named) > 1:
        return OperandDirection(mnemonic, attr, USE, None, written_slots=verdict.written_slots,
                                reason=verdict.reason + (
                                    f"; its file was not attributed -- withdrawing slot {slot} of "
                                    f"more than one file ({', '.join(sorted(named))}) changed the "
                                    "effect, so the evidence does not separate them"))
    return OperandDirection(mnemonic, attr, USE, None, written_slots=verdict.written_slots,
                            reason=verdict.reason + (
                                f"; its file was not attributed -- withdrawing slot {slot} of any "
                                "seeded file left the effect unchanged"))

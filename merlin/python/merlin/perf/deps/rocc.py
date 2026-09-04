"""Turn a decoded accelerator command trace into the def-use footprint a dependence graph needs.

WHY THIS EXISTS. :func:`merlin.perf.deps.liveness.effects_of` reads a MEASURED operand-direction
model keyed on the architectural register operands ``rs1``/``rs2``. For a command-driven accelerator
those two are HOST general-purpose registers: they carry the payload, and every command in a program
writes them the same way from the same handful of scratch registers. A graph built on them says every
command depends on every other one, which is not a conservative approximation -- it is the wrong
graph, and it hides the reorderings that matter behind edges that do not exist.

The real dependences live one level down, in the ON-CHIP addresses the payload encodes: the staging
address a transfer writes and a compute reads, the accumulator address a compute writes and a readout
reads. The decoder already extracts those. This module is the adapter between the two, and it is the
missing link that makes :mod:`merlin.perf.depgraph` able to rank two orderings of the same commands.

WHAT IT TRACKS, AND WHAT IT CANNOT SEE

* On-chip state files, named by the ABI vocabulary declared below. Two commands touching the same
  address in the same file are dependent; two touching different addresses are not.
* NOT main memory. A transfer's off-chip operand resolves to a base pointer plus an offset, not to a
  value this graph can compare, so a dependence carried through memory is invisible here. That is
  reported in :attr:`untracked_files` on every program built, because an invisible dependence is a
  MISSING EDGE, and a missing edge is exactly the separation a reordering would delete.

WHY A WIDE DEFINITION IS EXPANDED AND A USE IS NOT. A transfer writes a run of consecutive rows and
the command says how many, so the run is expanded -- a later command that overwrites the second row
of that run really does have to stay after it. A consumer names only the base of the tile it reads;
how many rows it goes on to consume is a property of the machine's array, not of the command, so
expanding the use would be inventing edges rather than deriving them. Matching a use against the base
finds every producer the program actually named, and the def-side expansion is what catches overlap.

FLAG BITS IN AN ADDRESS ARE STRIPPED, OR THE INSTRUCTION IS REFUSED. Some address fields carry mode
bits beside the address. Left in, the same tile addressed once with a mode bit set and once without
looks like two different tiles and the dependence between them vanishes -- silently, in the direction
that flatters a reordering. The mask is a target fact and is DERIVED (:func:`flag_masks_for`); an
address whose command shows mode-bit evidence while no mask was supplied is carried as UNRESOLVED,
never stripped by guess and never used raw.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from merlin.perf.deps.liveness import Access, Effects, Instruction

__all__ = [
    "CONSUMES", "DEFINES", "FLAG_EVIDENCE", "UNRESOLVED_OPERAND_KIND", "WIDTH_FIELD",
    "effects_of_row", "flag_masks_for", "instructions_and_effects", "program_from_trace",
    "roles_for", "untracked_files",
]

#: Decoded field -> the on-chip state file it addresses, for fields that DEFINE a value and for
#: fields that CONSUME one. These are the emitted ABI's own field names, carried by the program
#: itself, not facts about any one device: a backend speaking this ABI emits them whatever its ISA.
#: A command declaring none of them contributes no edge, which is the truth for a mode-setting
#: command and is checked against the operand-resolution evidence below rather than assumed.
DEFINES: Mapping[str, str] = {"spad_addr": "spad", "c_addr": "acc"}
CONSUMES: Mapping[str, str] = {"a_spad": "spad", "weight_spad": "spad", "acc_addr": "acc"}

#: The decoded field carrying how many consecutive slots a definition spans. Absent -> one slot.
WIDTH_FIELD = "rows"

#: Decoded field -> the state file whose address that field proves carries mode bits. A command
#: reporting an accumulate mode or a readout width has decoded those from bits sitting inside the
#: accumulator address, so the address is not the address until they are stripped. Bound to a FILE
#: rather than to the command, because a command may name one address that carries mode bits and
#: another that does not, and refusing both would delete a dependence that is perfectly readable.
FLAG_EVIDENCE: Mapping[str, str] = {"accumulate": "acc", "readout": "acc"}

#: Command class -> the class that STAGES its destination. Some commands do not name what they
#: write: the ABI has an earlier command stage the destination and the later one write it. Without
#: this the writer has no definition at all, so nothing orders a readout after the command that
#: actually produced the value -- and an ordering that hoists the readout above it is scored as
#: legal and fast. Modelled as DEF_USE, because a command that accumulates also READS what is
#: already there. A writer with no stager before it is UNRESOLVED, never silently effect-free.
INHERITS_DESTINATION: Mapping[str, str] = {"COMPUTE_PRELOADED": "PRELOAD",
                                           "COMPUTE_ACCUMULATE": "PRELOAD"}

#: The operand-resolution verdict meaning the decoder could not establish a value at all. An operand
#: resolved to an off-chip base is a different, weaker state: it is resolved, and it addresses a file
#: this graph does not track (see :func:`untracked_files`).
UNRESOLVED_OPERAND_KIND = "unknown"

_OPERAND_KEYS = ("rs1", "rs2")


def untracked_files() -> tuple[str, ...]:
    """State this graph carries no edges for. Reported on every program, never silently omitted."""
    return ("dram",)


def flag_masks_for(target: str) -> dict[str, int]:
    """``{state file: bits to strip}``, DERIVED from ``target``'s own RTL facts.

    Raises rather than defaulting. A mask guessed here is not a small error: it silently merges or
    splits tiles, and either way the graph loses edges that a reordering is then free to violate.
    """
    from merlin.targetgen.rocc.decode import isa_constants

    isa = isa_constants(target)
    bits = 0
    for name in ("ACC_ACCUM", "FULL_C_BIT"):
        value = isa.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"target {target!r} does not derive {name}, so the accumulator address "
                             f"cannot be separated from its mode bits")
        bits |= int(value)
    return {"acc": bits}


def roles_for(target: str) -> dict[str, str]:
    """``{command class: structural role}`` for ``target``, from its derived ISA taxonomy.

    The role is what :func:`merlin.perf.depgraph.build_dag` groups an unpriced separation by, so two
    commands whose completion waits on the same pipeline share one unknown and cancel out of a
    difference. A class the taxonomy does not name keeps its own identity rather than being folded
    into someone else's unknown.
    """
    from merlin.targetgen.isa_model import isa_model_for_target
    from merlin.targetgen.rocc.decode import isa_constants

    isa = isa_constants(target)
    model = isa_model_for_target(target)
    # Both maps are keyed by the ENCODED function selector, and only by it. The ISA taxonomy spells a
    # command with the RTL's own name while the decoder reports its semantic class, and the two
    # spellings do not match -- joining on the name silently yields no role for anything, which
    # leaves every command its own separation class and refuses every comparison. Join on the
    # selector, which is the identity both were derived from.
    by_selector = {int(entry.get("funct7")): str((entry or {}).get("role") or "")
                   for entry in (model.by_mnemonic or {}).values()
                   if isinstance((entry or {}).get("funct7"), int)}
    roles = {str(klass): by_selector[int(selector)]
             for selector, klass in (isa.get("FUNCT_CLASS") or {}).items()
             if int(selector) in by_selector}
    # A configuring command is reported by its SUBTYPE, which shares its parent's selector: a subtype
    # is a narrowing of one command, not a command of its own, so it inherits the parent's role.
    parent = {klass for selector, klass in (isa.get("FUNCT_CLASS") or {}).items()
              if int(selector) in by_selector}
    for subtype in (isa.get("CONFIG_SUBTYPE") or {}).values():
        base = next((k for k in parent if str(subtype).startswith(k)), None)
        if base is not None:
            roles[str(subtype)] = roles[base]
    return roles


def _payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("decoded")
    return value if isinstance(value, Mapping) else {}


def _address(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def effects_of_row(row: Mapping[str, Any], *, flag_masks: Mapping[str, int] | None = None) -> Effects:
    """What one decoded command writes and reads on chip, and what about it stayed unresolved."""
    masks = dict(flag_masks or {})
    payload = _payload(row)
    klass = str(row.get("class") or "")
    unresolved: list[str] = []

    if klass in ("", "UNKNOWN"):
        return Effects(defs=(), uses=(), observed=False,
                       unresolved=("the decoder could not read this command, so its whole "
                                   "dependence footprint is unknown",))
    for key in _OPERAND_KEYS:
        operand = row.get(key)
        if isinstance(operand, Mapping) and operand.get("kind") == UNRESOLVED_OPERAND_KIND:
            unresolved.append(f"{key}: the decoder resolved no value, so any address it carries "
                              f"is invisible")

    flagged = {file for name, file in FLAG_EVIDENCE.items() if name in payload}
    width = _address(payload.get(WIDTH_FIELD))

    defs: list[Access] = []
    uses: list[Access] = []
    observed = False
    for fields, sink in ((DEFINES, defs), (CONSUMES, uses)):
        for name, file in sorted(fields.items()):
            address = _address(payload.get(name))
            if address is None:
                continue
            mask = masks.get(file)
            if file in flagged and mask is None:
                unresolved.append(
                    f"{name}: the command proves this address carries mode bits, and no mask was "
                    f"derived for file {file!r}, so its slot cannot be identified")
                continue
            observed = True
            base = address & ~int(mask) if mask is not None else address
            span = max(1, width) if (sink is defs and width is not None) else 1
            sink.extend(Access(file, base + i) for i in range(span))
    return Effects(defs=tuple(defs), uses=tuple(uses), unresolved=tuple(unresolved),
                   observed=observed)


def instructions_and_effects(trace: Any, *, flag_masks: Mapping[str, int] | None = None
                             ) -> tuple[tuple[Instruction, ...], tuple[Effects, ...]]:
    """``(instructions, effects)`` for one decoded trace, in issue order.

    The command's decoded CLASS is its mnemonic: that is the granularity the direction of its state
    is established at, and the granularity a structural role is assigned at. The payload travels as
    the instruction's operands so a consumer can read it back without re-decoding.
    """
    rows = trace.get("instructions") if isinstance(trace, Mapping) else trace
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("the trace declares no instruction list")
    ordered = [r for r in rows if isinstance(r, Mapping)]
    instructions = tuple(
        Instruction(index=i, mnemonic=str(r.get("class") or ""),
                    operands={k: v for k, v in _payload(r).items()
                              if isinstance(v, int) and not isinstance(v, bool)})
        for i, r in enumerate(ordered))
    effects = list(effects_of_row(r, flag_masks=flag_masks) for r in ordered)
    return instructions, tuple(_inherit_destinations(instructions, effects))


def _inherit_destinations(instructions: "Sequence[Instruction]",
                          effects: "Sequence[Effects]") -> list[Effects]:
    """Give each command that writes a destination it did not name the one staged for it.

    Walked forward so the stager is the nearest preceding one, which is what "preloaded" means. The
    inherited value is added as both a definition and a use: the writer orders after the stager, and
    every later reader orders after the WRITER rather than after the stager -- which is the edge that
    stops a readout being hoisted above the command that computed what it reads.
    """
    staged: dict[str, tuple[Access, ...]] = {}
    out: list[Effects] = []
    for instruction, effect in zip(instructions, effects):
        stager = INHERITS_DESTINATION.get(instruction.mnemonic)
        if stager is None:
            if effect.defs:
                staged[instruction.mnemonic] = effect.defs
            out.append(effect)
            continue
        inherited = staged.get(stager)
        if not inherited:
            out.append(Effects(defs=effect.defs, uses=effect.uses, observed=effect.observed,
                               unresolved=effect.unresolved + (
                                   f"writes a destination staged by a {stager} command, and no "
                                   f"{stager} command established one before it",)))
            continue
        out.append(Effects(defs=tuple(effect.defs) + inherited,
                           uses=tuple(effect.uses) + inherited,
                           unresolved=effect.unresolved, observed=True))
        staged[stager] = inherited
    return out


def program_from_trace(trace: Any, *, target: str | None = None,
                       flag_masks: Mapping[str, int] | None = None,
                       roles: Mapping[str, str] | None = None):
    """A :class:`merlin.perf.depgraph.Program` over a decoded command trace.

    Supply ``target`` to derive the address masks and structural roles from its RTL facts, or supply
    either one directly when they are already in hand. A command stream is straight-line by
    construction -- a loop construct in it means the static list is not what ran, and the decoder
    reports that as its own class, which becomes an unresolved instruction here rather than a graph
    quietly describing one iteration of something that ran many times.
    """
    from merlin.perf import depgraph

    if flag_masks is None and target is not None:
        flag_masks = flag_masks_for(target)
    if roles is None and target is not None:
        roles = roles_for(target)
    instructions, effects = instructions_and_effects(trace, flag_masks=flag_masks)
    regions = (depgraph.Region(name="[0,%d)" % len(instructions), start=0,
                               end=len(instructions), trips=1),) if instructions else ()
    return depgraph.Program(instructions=instructions, effects=effects, regions=regions,
                            roles=dict(roles or {}))

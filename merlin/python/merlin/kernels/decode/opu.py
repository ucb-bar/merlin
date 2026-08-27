"""Audit what a compiled object ACTUALLY emits for a matrix extension the disassembler cannot name.

The extension's instructions occupy reserved encoding slots, so no assembler knows their mnemonics and
`llvm-objdump` prints them as ``<unknown>``. Two consequences drive this module's whole design:

* **An audit must decode the raw encoding word, not the mnemonic.** Every count here comes from
  extracting opcode / funct3 / funct6 out of the 32-bit word and comparing those integers against a
  DERIVED encoding table (:mod:`targetgen.rtl.opu_isa`). Nothing string-matches a mnemonic, and nothing
  hardcodes a field value. This is also the difference between this tool and the one whose "100%
  coverage" claim was wrong: that one counted ``.insn`` occurrences in *source text*, which says what a
  programmer wrote, not what the compiler emitted or what the hardware will run.
* **The existing inert-lever digest is blind here.** ``mining.beam._emitted_digest`` hashes the mnemonic
  stream, and since all four of these instructions disassemble to the same ``<unknown>`` with no
  operands, a change that swaps an accumulate for a readout would hash identically and be marked inert.
  :func:`digest` therefore hashes the DECODED identity, so a matrix-extension change is visible to the
  same guard that protects every other lever.

The audit also reports what it could not account for. A word the disassembler declined to name and the
derived table does not claim is neither "ours" nor "fine" — it is recorded in ``unaccounted`` so a
mis-encoded instruction surfaces instead of being silently counted as absent.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Audit", "Decoded", "audit", "decode_stream", "digest", "fields_of"]

#: RISC-V 32-bit field positions. These are the *instruction format*, not a property of any target: an
#: R-type word is funct7[31:25] rs2[24:20] rs1[19:15] funct3[14:12] rd[11:7] opcode[6:0], and a
#: vector-format instruction splits funct7 into funct6[31:26] and vm[25].
_OPCODE_MASK = 0x7F  # derived-ok: RISC-V base encoding — opcode is bits[6:0] of every 32-bit
#                      word; a field WIDTH, not an opcode VALUE (the value is compared against
#                      the derived table in targetgen.rtl.opu_isa, never against a literal here)
_FUNCT3_SHIFT, _FUNCT3_MASK = 12, 0x7
_FUNCT6_SHIFT, _FUNCT6_MASK = 26, 0x3F
_VM_SHIFT = 25
_RD_SHIFT, _RS1_SHIFT, _RS2_SHIFT, _REG_MASK = 7, 15, 20, 0x1F

#: What a disassembler prints for a word it cannot name. Matching this is a statement about the TOOL's
#: output format, not about any instruction's semantics.
UNKNOWN_MNEMONIC = "<unknown>"


def fields_of(word: int) -> dict[str, int]:
    """Decode the R/vector-format fields of one 32-bit instruction word."""
    return {
        "opcode": word & _OPCODE_MASK,
        "funct3": (word >> _FUNCT3_SHIFT) & _FUNCT3_MASK,
        "funct6": (word >> _FUNCT6_SHIFT) & _FUNCT6_MASK,
        "vm": (word >> _VM_SHIFT) & 1,
        "rd": (word >> _RD_SHIFT) & _REG_MASK,
        "rs1": (word >> _RS1_SHIFT) & _REG_MASK,
        "rs2": (word >> _RS2_SHIFT) & _REG_MASK,
    }


@dataclass(frozen=True)
class Decoded:
    """One instruction, named by the derived table when the table claims its encoding."""

    index: int
    addr: int
    identity: str                  # the derived mnemonic, or the disassembler's own text
    from_extension: bool
    mnemonic: str = ""             # what the disassembler called it
    operands: tuple[str, ...] = ()
    fields: dict[str, int] = field(default_factory=dict)


def _word_of(hexcode: str) -> int | None:
    """The instruction word from an objdump hex column, or None when it is not a 32-bit word.

    A 16-bit compressed instruction cannot be one of these (they are all 32-bit), so a short column is
    declined rather than zero-extended into a bogus field decode.
    """
    token = (hexcode or "").strip().replace(" ", "")
    if len(token) != 8:
        return None
    try:
        return int(token, 16)
    except ValueError:
        return None


def decode_stream(insns: Sequence[Any], encodings: Mapping[str, Any]) -> list[Decoded]:
    """Name every instruction in a disassembly stream, using ``encodings`` for the unnameable ones.

    ``insns`` are :class:`kernels.decode.objdump.RawInsn`; ``encodings`` maps a name to anything
    carrying ``opcode`` / ``funct3`` / ``funct6`` (i.e. an :class:`targetgen.rtl.opu_isa.Encoding`).
    Matching is on the three integers, so a spelling difference between the RTL's vocabulary and a
    header's cannot cause a miss, and a value is never compared as a string.
    """
    table = {(int(e.opcode), int(e.funct3), int(e.funct6)): name
             for name, e in encodings.items()}
    out: list[Decoded] = []
    for i, insn in enumerate(insns):
        word = _word_of(getattr(insn, "hexcode", ""))
        f = fields_of(word) if word is not None else {}
        name = table.get((f.get("opcode"), f.get("funct3"), f.get("funct6"))) if f else None
        out.append(Decoded(index=i, addr=int(getattr(insn, "addr", 0)),
                           identity=name or str(getattr(insn, "mnemonic", "")),
                           from_extension=name is not None,
                           mnemonic=str(getattr(insn, "mnemonic", "")),
                           operands=tuple(getattr(insn, "operands", ()) or ()),
                           fields=f))
    return out


@dataclass(frozen=True)
class Audit:
    """What an object emits for the extension, and what could not be accounted for."""

    counts: dict[str, int] = field(default_factory=dict)
    total_insns: int = 0
    extension_insns: int = 0
    unaccounted: tuple[dict[str, Any], ...] = ()
    vector_config_insns: int = 0
    configured_before_each: dict[str, int] = field(default_factory=dict)
    unconfigured: tuple[str, ...] = ()
    digest: str = ""
    notes: tuple[str, ...] = ()

    @property
    def emitted_extension_ops(self) -> int:
        """Deliberately NOT called "coverage". A count of emitted instructions says nothing about how
        much of a model's work reached the unit, and conflating the two is a documented past failure."""
        return self.extension_insns

    def to_dict(self) -> dict[str, Any]:
        return {"counts": dict(self.counts), "total_insns": self.total_insns,
                "emitted_extension_ops": self.emitted_extension_ops,
                "unaccounted": [dict(u) for u in self.unaccounted],
                "vector_config_insns": self.vector_config_insns,
                "configured_before_each": dict(self.configured_before_each),
                "unconfigured": list(self.unconfigured), "digest": self.digest,
                "notes": list(self.notes)}


def digest(decoded: Sequence[Decoded]) -> str:
    """Hash of the DECODED identity stream — the inert-lever guard, made able to see this extension.

    Uses each instruction's derived name where there is one and the disassembler's mnemonic + operands
    otherwise. Addresses and the hex words are excluded so register-allocation noise and symbol offsets
    do not mask a genuine no-op as a change; the extension's operand *registers* are included, since for
    an unnameable instruction the register fields are the only thing distinguishing two uses of it.
    """
    lines = []
    for d in decoded:
        if d.from_extension:
            f = d.fields
            lines.append(f"{d.identity} rd={f.get('rd')} rs1={f.get('rs1')} rs2={f.get('rs2')}")
        else:
            lines.append(f"{d.mnemonic} {','.join(d.operands)}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def audit(insns: Sequence[Any], encodings: Mapping[str, Any], *,
          config_mnemonic_prefix: str = "vset") -> Audit:
    """Audit one disassembly stream against a derived encoding table.

    Reports a count for EVERY name in ``encodings``, including the ones that are zero: a readout with no
    counter is how an accumulate-without-extraction goes unnoticed, which is exactly the hole in the
    tool that preceded this one.

    ``configured_before_each`` counts, per extension instruction, how many vector-configuration
    instructions were emitted between it and the previous extension instruction. Zero for an
    instruction whose operand length is load-bearing means the length in effect was inherited from
    whatever ran before — the shape of the narrow-operand failure this audit exists to catch. The
    configuration mnemonic is matched by PREFIX because that is a property of the disassembler's naming
    of the base vector ISA, not of the extension. The prefix must stay short enough to cover the
    immediate-length form: ``vsetivli`` does not begin with ``vsetvl``, and a kernel that sets a constant
    length would otherwise have every instruction reported as inheriting one.
    """
    decoded = decode_stream(insns, encodings)
    counts = {name: 0 for name in encodings}
    unaccounted: list[dict[str, Any]] = []
    configured: dict[str, int] = {name: 0 for name in encodings}
    unconfigured: list[str] = []
    n_config = 0
    since_config = 0

    for d in decoded:
        if d.mnemonic.startswith(config_mnemonic_prefix):
            n_config += 1
            since_config += 1
            continue
        if d.from_extension:
            counts[d.identity] += 1
            configured[d.identity] += since_config
            if since_config == 0:
                unconfigured.append(d.identity)
            since_config = 0
            continue
        if d.mnemonic == UNKNOWN_MNEMONIC:
            # Neither nameable nor ours. Never counted as absent: a mis-encoded instruction looks
            # exactly like this, and dropping it would report a clean audit for a broken kernel.
            unaccounted.append({"index": d.index, "addr": d.addr, "fields": dict(d.fields)})

    notes: list[str] = []
    if unaccounted:
        notes.append(f"{len(unaccounted)} word(s) the disassembler could not name and the derived "
                     "table does not claim — inspect before trusting any count here")
    if unconfigured:
        notes.append(f"{len(unconfigured)} extension instruction(s) with no vector-configuration "
                     "instruction since the previous one: the operand length in effect was inherited")
    if not any(counts.values()):
        notes.append("no extension instruction was emitted at all")

    return Audit(counts=counts, total_insns=len(decoded),
                 extension_insns=sum(counts.values()), unaccounted=tuple(unaccounted),
                 vector_config_insns=n_config, configured_before_each=configured,
                 unconfigured=tuple(unconfigured), digest=digest(decoded), notes=tuple(notes))


def audit_object(obj_path, encodings: Mapping[str, Any], *, triple: str = "riscv64") -> Audit:
    """Audit a compiled object file."""
    from .objdump import tokenize
    return audit(tokenize(obj_path, triple=triple), encodings)


def audit_text(text: str, encodings: Mapping[str, Any]) -> Audit:
    """Audit already-disassembled text, so a saved objdump can be re-audited with no toolchain."""
    from .objdump import tokenize_text
    return audit(tokenize_text(text), encodings)

"""Derived assembler for a self-hosted-ISA target — packs operands into the EXACT bits the target's own
encoder uses, from the :class:`~merlin.targetgen.isa_model.IsaModel` field maps. It removes the single most
error-prone step of hand-writing raw ISA (bit-packing a 32-bit word from opcode/funct/field layout): the
agent writes ``MNEMONIC rd=1, rs1=2`` and gets the correct ``.word``.

Guardrails (this is a tool, not a solver): it encodes the syntax the AGENT chose — it never decides WHICH
instructions to emit, never reads a golden, and it **refuses rather than emit a silently-wrong word** when a
mnemonic is undefined, an operand is out of its field's range, or an operand field is non-linear. Pure
Python bit-ops over the derived model; no target name, no ``re``, no oracle.
"""
from __future__ import annotations

from .isa_model import IsaModel


class AssembleError(ValueError):
    """A line/operand the derived model cannot encode faithfully (undefined op, bad operand, non-linear
    field). Raised rather than emitting a wrong word — the whole point of the tool."""


def _representable_mask(bits: list[int | None]) -> int:
    """The operand-value bits this field can hold (index i set iff operand bit i maps to a real word bit)."""
    rep = 0
    for i, wb in enumerate(bits):
        if isinstance(wb, int) and wb >= 0:
            rep |= (1 << i)
    return rep


def _encode_operand(attr: str, bits: list[int | None], value: int) -> int:
    """Scatter an unsigned operand value into its word bits via the derived per-bit map. Refuses a
    non-linear field (a ``-1`` entry) and a value that does not fit the field's representable bits."""
    if value < 0:
        raise AssembleError(f"operand '{attr}'={value} is negative; provide the raw unsigned field value")
    if any(b == -1 for b in bits):
        raise AssembleError(f"operand '{attr}' maps to a non-linear field; cannot pack it safely — "
                            "emit this instruction as an explicit .word instead")
    if value & ~_representable_mask(bits):
        width = sum(1 for b in bits if isinstance(b, int) and b >= 0)
        raise AssembleError(f"operand '{attr}'={value} does not fit its {width}-bit field")
    word = 0
    for i, wb in enumerate(bits):
        if isinstance(wb, int) and wb >= 0 and (value >> i) & 1:
            word |= (1 << wb)
    return word


def assemble_line(model: IsaModel, mnemonic: str, operands: dict[str, int]) -> int:
    """Assemble one instruction → its 32-bit word. ``mnemonic`` is an op class name or an assembler alias
    the target's ISA exposes; ``operands`` are ``{field_name: unsigned_value}`` (omitted fields default to
    0, exactly as the model's own zero-operand encoding). Raises :class:`AssembleError` on an undefined
    mnemonic, an unknown/oversized operand, or a non-linear field."""
    ent = model.resolve(mnemonic)
    if ent is None:
        raise AssembleError(f"unknown instruction '{mnemonic}' — not defined by this target's ISA")
    fields = ent.get("fields") or {}
    unknown = set(operands) - set(fields)
    if unknown:
        valid = ", ".join(sorted(fields)) or "(none)"
        raise AssembleError(f"'{mnemonic}' has no operand(s) {sorted(unknown)}; valid operands: {valid}")
    word = int(ent.get("fixed_value", 0))
    for attr, value in operands.items():
        word |= _encode_operand(attr, fields[attr], int(value))
    return word & 0xFFFFFFFF


def _parse_operands(rest: str) -> dict[str, int]:
    """Parse ``rd=1, rs1=2, imm=0x10`` → {rd:1, rs1:2, imm:16}. Structured splitting only (no regex);
    values accept 0x/0b/decimal via ``int(x, 0)``."""
    ops: dict[str, int] = {}
    for tok in rest.replace(",", " ").split():
        if "=" not in tok:
            raise AssembleError(f"operand '{tok}' must be name=value")
        k, _, v = tok.partition("=")
        try:
            ops[k.strip()] = int(v.strip(), 0)
        except ValueError:
            raise AssembleError(f"operand '{k.strip()}' has non-integer value '{v.strip()}'")
    return ops


def assemble_text(model: IsaModel, text: str) -> list[int]:
    """Assemble a small mnemonic listing → the list of 32-bit words. One instruction per line:
    ``MNEMONIC field=value, field=value``. Also accepts ``.word 0x..`` / ``.word 123`` literal passthrough
    (for encodings the agent wants to hand-place) and skips blank lines and ``#`` / ``//`` / ``;`` comments.
    Raises :class:`AssembleError` (with the 1-based line number) on any line it cannot encode faithfully."""
    if model.is_empty():
        raise AssembleError("this target ships no ISA definition; the derived assembler is unavailable")
    words: list[int] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].split("//", 1)[0].split(";", 1)[0].strip()
        if not line:
            continue
        head, _, rest = line.partition(" ")
        try:
            if head == ".word":
                words.append(int(rest.strip(), 0) & 0xFFFFFFFF)
            else:
                words.append(assemble_line(model, head, _parse_operands(rest)))
        except AssembleError as e:
            raise AssembleError(f"line {lineno}: {e}") from None
    return words


def to_word_lines(words: list[int]) -> str:
    """Render assembled words as the ``.word 0x........`` lines the agent drops into ``kernel.S`` (which
    stock ``llvm-mc`` then assembles into IMEM words)."""
    return "".join(f".word 0x{w & 0xFFFFFFFF:08x}\n" for w in words)


# --- fixed-format encoder (the inverse of isa_disasm's field-layout decode) ------------------------
def _pack_field(word: int, hi: int, lo: int, value: int, name: str) -> int:
    """Place an unsigned value into the inclusive bit-range ``[hi:lo]``. Refuses a value that does not fit
    the field (rather than truncating into a silently-wrong word)."""
    if value < 0:
        raise AssembleError(f"field '{name}'={value} is negative; provide the raw unsigned field value")
    width = hi - lo + 1
    if value >> width:
        raise AssembleError(f"field '{name}'={value} does not fit its {width}-bit range [{hi}:{lo}]")
    return word | (value << lo)


def assemble_fixed(model: IsaModel, mnemonic: str, operands: dict[str, int] | None = None) -> int:
    """Assemble one instruction of a FIXED-FORMAT ISA (one field layout selected by an opcode field — the
    mlc ``isa_encoding`` derivation) → its ``inst_width``-bit word. ``mnemonic`` names an entry in the
    derived opcode table; ``operands`` are ``{field_name: unsigned_value}`` for the layout's fields (omitted
    fields default to 0). The opcode value is placed at the opcode field's low bit, which fills the opcode
    AND any extension bits carved contiguously above it (e.g. an address-space selector). Raises
    :class:`AssembleError` on an unknown mnemonic, an unknown/oversized field, or a non-fixed-format model."""
    if not model.is_fixed_format():
        raise AssembleError("assemble_fixed requires a fixed-format model (field layout + opcode table)")
    if mnemonic not in model.opcode_table:
        valid = ", ".join(sorted(model.opcode_table)) or "(none)"
        raise AssembleError(f"unknown instruction '{mnemonic}' — opcodes: {valid}")
    operands = operands or {}
    unknown = set(operands) - set(model.field_layout) | ({"opcode"} & set(operands))
    if unknown:
        valid = ", ".join(sorted(set(model.field_layout) - {"opcode"})) or "(none)"
        raise AssembleError(f"'{mnemonic}' has no settable field(s) {sorted(unknown)}; fields: {valid}")
    _, op_lo = model.field_layout["opcode"]
    word = int(model.opcode_table[mnemonic]) << op_lo
    for name, value in operands.items():
        hi, lo = model.field_layout[name]
        word = _pack_field(word, hi, lo, int(value), name)
    return word & ((1 << model.inst_width) - 1)


def to_data_lines(words: list[int], inst_width: int) -> str:
    """Render assembled words as the assembler data directive stock ``llvm-mc`` emits little-endian: a wide
    (>32-bit) word becomes ``.quad`` (8 bytes), else ``.word`` (4 bytes)."""
    if inst_width > 32:
        return "".join(f".quad 0x{w & ((1 << 64) - 1):016x}\n" for w in words)
    return to_word_lines(words)

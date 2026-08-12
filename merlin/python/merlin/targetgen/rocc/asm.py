"""Derived RoCC assembler for ``inline_asm_insn`` targets.

The companion of :mod:`rocc.decode`: given a short instruction listing, emit the CANONICAL LLVM-dialect
MLIR form of each RoCC ``.insn`` — an ``llvm.inline_asm`` op whose operands are SSA values produced by
``llvm.mlir.constant`` — using the target's RTL-derived ISA facts (``rocc_decode.isa_constants``). It is an
AUTHORING AID, not the answer: it does NOT compute operand values (choosing the config word / spad
addresses is the compiler's job); it guarantees the emission *form* is both **assemblable** and
**decodable**, so an agent cannot hand-roll the invalid inline-integer-literal operand form
(``llvm.inline_asm … "r,r" (65540, 16)``) that neither assembles NOR round-trips through the decoder.

Fully target-parameterized (facts come from ``isa_constants(target)``); refuses a class/funct the target's
facts do not define, and refuses a CONFIG whose ``rs1`` subtype bits contradict the requested subtype —
never emits a wrong instruction. Self-validating: every emitted program is fed back through
``rocc_decode.decode_text`` and the classes must match (else it raises).
"""
from __future__ import annotations

from . import decode as RD


class AsmError(ValueError):
    """The listing names a class/funct or operand the target's ISA facts do not permit."""


def _class_to_funct(isa: dict) -> dict[str, int]:
    """``class-name -> func7`` from the derived FUNCT_CLASS map (plus the CONFIG subtypes, all func7=0)."""
    inv = {v: k for k, v in isa["FUNCT_CLASS"].items()}
    config = inv.get("CONFIG")
    if config is not None:
        for sub in isa["CONFIG_SUBTYPE"].values():          # CONFIG_EX / CONFIG_LD / CONFIG_ST -> same func7
            inv[sub] = config
    return inv


def _config_subtype_bits(isa: dict, name: str) -> int | None:
    """The required ``rs1 & 0x3`` for a CONFIG subtype name, else None (not a CONFIG subtype)."""
    for bits, sub in isa["CONFIG_SUBTYPE"].items():
        if sub == name:
            return bits
    return None


def assemble_program(target: str, listing: list[tuple[str, int, int]], *,
                     kernel_symbol: str = "gemmini_kernel") -> str:
    """Render a full LLVM-dialect MLIR module for ``listing`` (each item ``(class_name, rs1, rs2)``).

    ``class_name`` is a derived instruction class (e.g. ``MVIN``, ``PRELOAD``, ``COMPUTE_PRELOADED``,
    ``MVOUT``, ``FLUSH``, ``CONFIG_EX``/``CONFIG_LD``/``CONFIG_ST``) or ``FENCE``. ``rs1``/``rs2`` are the
    two i64 source operand VALUES the caller's compiler computed. The op takes no result register (rd=x0);
    the RoCC funct3 and custom opcode are derived facts."""
    isa = RD.isa_constants(target)
    opcode, func3 = isa["CUSTOM_OPCODE"], isa["FUNCT3"]
    if opcode is None:
        raise AsmError(f"target {target!r} ships no RoCC custom opcode fact; the derived assembler "
                       f"is unavailable")
    cls2funct = _class_to_funct(isa)
    body: list[str] = []
    n = 0
    for name, rs1, rs2 in listing:
        name = name.strip().upper()
        if name == "FENCE":
            body.append('    llvm.inline_asm has_side_effects "fence", "" : () -> ()')
            continue
        funct = cls2funct.get(name)
        if funct is None:
            raise AsmError(f"unknown instruction class {name!r} for {target!r}; "
                           f"legal classes: {sorted(set(cls2funct))}")
        want = _config_subtype_bits(isa, name)              # None unless a CONFIG subtype
        if want is not None and (rs1 & 0x3) != want:
            raise AsmError(f"{name} requires (rs1 & 0x3) == {want}; got rs1={rs1} "
                           f"(rs1 & 0x3 == {rs1 & 0x3}). Set the low 2 bits of rs1 to select the subtype.")
        a, b = f"%c{n}", f"%c{n + 1}"
        n += 2
        body.append(f"    {a} = llvm.mlir.constant({rs1} : i64) : i64")
        body.append(f"    {b} = llvm.mlir.constant({rs2} : i64) : i64")
        body.append(f'    llvm.inline_asm has_side_effects ".insn r {hex(opcode)}, {hex(func3)}, '
                    f'{hex(funct)}, x0, $0, $1", "r,r" {a}, {b} : (i64, i64) -> ()')
    mlir = (f"module {{\n  llvm.func @{kernel_symbol}() {{\n"
            + "\n".join(body) + "\n    llvm.return\n  }\n}\n")
    _roundtrip_check(target, mlir, listing)
    return mlir


def _roundtrip_check(target: str, mlir: str, listing: list[tuple[str, int, int]]) -> None:
    """Decode the emitted text back and assert the decoded classes match the requested ones — proves the
    emitted form is decodable (the whole point) and guards against an encoder regression."""
    trace = RD.decode_text(mlir, source="rocc_asm", target=target)
    got = [i["class"] for i in trace["instructions"]]
    if "UNKNOWN" in got:
        raise AsmError(f"internal: emitted instruction did not decode (got UNKNOWN): {got}")
    want = [c.strip().upper() for c, _, _ in listing]
    if got != want:
        raise AsmError(f"internal: round-trip mismatch — emitted {want}, decoded {got}")


def assemble_text(target: str, text: str, **kw) -> str:
    """Parse a whitespace listing (one ``CLASS rs1 rs2`` per line; ``FENCE`` alone) and assemble it. Lines
    starting with ``#`` and blank lines are ignored. rs1/rs2 accept 0x-hex or decimal."""
    listing: list[tuple[str, int, int]] = []
    for ln in text.splitlines():
        ln = ln.split("#", 1)[0].strip()
        if not ln:
            continue
        parts = ln.split()
        if parts[0].upper() == "FENCE":
            listing.append(("FENCE", 0, 0))
            continue
        if len(parts) != 3:
            raise AsmError(f"expected `CLASS rs1 rs2` (or `FENCE`), got: {ln!r}")
        listing.append((parts[0], int(parts[1], 0), int(parts[2], 0)))
    return assemble_program(target, listing, **kw)

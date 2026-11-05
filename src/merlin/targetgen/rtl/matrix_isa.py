"""Derive a matrix extension's instruction encodings as MATCH/MASK pairs, from the RTL that decodes them.

:mod:`targetgen.rtl.opu_isa` identifies an instruction by ``(opcode, funct3, funct6)``. That is enough for
an extension whose every instruction owns a funct6 slot, and not enough in general: a matrix extension
routinely puts several instructions in ONE slot and tells them apart by a register field that is not a
register at all (``sf.vtzero.t`` / ``sf.vtmv.v.t`` / ``sf.vtdiscard`` share ``funct6`` and ``funct3`` and
differ only in ``rs2``), and a host core's configuration instructions are declared whole, as a 32-bit
``BitPat`` with don't-care bits. The general identity of an instruction is therefore the pair every
RISC-V decoder uses: the bits it decodes (``mask``) and their values (``match``). This module derives
that pair from the hardware's own sources, in two declaration shapes:

* **a decode table** -- an instruction object whose ``props`` list carries one entry per decode key
  (``F6(...)``, ``F3(...)``, ``RS1(...)``, ``RS2(...)``) plus a flag that selects the major opcode. Which
  prop names map to which instruction field, and which flag selects which opcode, are DECLARATIONS the
  caller passes (the unit's contract), not knowledge in this file;
* **a BitPat declaration** -- ``def NAME = BitPat("b1000100_00000_?????_111_?????_1010111")``.

Both are cross-checked against an independent ``.insn r`` header written by the unit's author, and
optionally against a simulator's ``MATCH_*``/``MASK_*`` table. Bits the RTL does not decode but an
assembler directive must still spell (the ``vm`` bit of an instruction whose decoder ignores it) are
taken from the header and recorded as such; bits nobody states are a gap, never a default.

Field POSITIONS are the RISC-V instruction format (opcode[6:0] rd[11:7] funct3[14:12] rs1[19:15]
rs2[24:20] funct7[31:25], funct7 = funct6[31:26] ++ vm[25]) and are the only constants here. No field
VALUE is written in this file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .opu_isa import (
    _VECTOR_UNIT_MIXIN_SUFFIX,
    _balanced_after,
    _bit_literal,
    _split_top_level,
    _strip_comments,
    bit_literal_defs,
    chisel_enum_ordinals,
    scala_param_names,
)

__all__ = [
    "INSN_BITS",
    "HeaderSite",
    "MaskedEncoding",
    "MatrixIsa",
    "bitpat_defs",
    "crosscheck_header",
    "crosscheck_opcode_table",
    "derive",
    "header_insn_sites",
    "match_case_table",
    "matrix_unit_params",
    "opcode_table_defs",
    "parse_bitpat",
    "seq_int_tuples",
    "seq_members",
]

#: Width of the instruction words these encodings describe.
INSN_BITS = 32
_WORD = (1 << INSN_BITS) - 1

#: RISC-V R-type / vector-format field positions as (lsb, width). These are the instruction FORMAT, not a
#: property of any target; every value placed in them is read from a source.
FORMAT: dict[str, tuple[int, int]] = {
    "opcode": (0, 7),  # derived-ok: RISC-V base instruction format (field position, not a value)
    "rd": (7, 5),
    "funct3": (12, 3),  # derived-ok: RISC-V base instruction format (field position, not a value)
    "rs1": (15, 5),
    "rs2": (20, 5),
    "vm": (25, 1),
    "funct6": (26, 6),
    "funct7": (25, 7),  # derived-ok: RISC-V base instruction format (field position, not a value)
}


def _field_mask(name: str) -> int:
    lsb, width = FORMAT[name]
    return ((1 << width) - 1) << lsb


def _get(word: int, name: str) -> int:
    lsb, width = FORMAT[name]
    return (word >> lsb) & ((1 << width) - 1)


def _put(value: int, name: str) -> int:
    lsb, width = FORMAT[name]
    if value < 0 or value >= (1 << width):
        raise ValueError(f"{value} does not fit the {width}-bit {name} field")
    return value << lsb


# ---------------------------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------------------------


def parse_bitpat(token: str) -> tuple[int, int, int] | None:
    """``(match, mask, width)`` for a ``BitPat("b10?1_0")`` token, or None when it is not one.

    ``?`` is a don't-care bit (mask 0), ``_`` is a visual separator. Only the binary form is accepted:
    that is the only one a ``BitPat`` string takes.
    """
    t = token.strip()
    head = "BitPat("
    if not t.startswith(head):
        return None
    inner = t[len(head) :].strip()
    if not inner.startswith('"'):
        return None
    end = inner.find('"', 1)
    if end < 0:
        return None
    lit = inner[1:end]
    if not lit.startswith("b"):
        return None
    bits = lit[1:].replace("_", "")
    if not bits or any(c not in "01?" for c in bits):
        return None
    match = mask = 0
    for c in bits:
        match <<= 1
        mask <<= 1
        if c != "?":
            mask |= 1
            match |= int(c)
    return match, mask, len(bits)


def bitpat_defs(text: str, container: str) -> dict[str, tuple[int, int, int]]:
    """``{NAME: (match, mask, width)}`` for every ``def NAME = BitPat("...")`` inside ``object container``."""
    body = _balanced_after(text, f"object {container}")
    if body is None:
        return {}
    out: dict[str, tuple[int, int, int]] = {}
    for raw in _strip_comments(body).splitlines():
        line = raw.strip()
        if not line.startswith("def ") or "=" not in line:
            continue
        name, _, rhs = line[len("def ") :].partition("=")
        name = name.strip()
        got = parse_bitpat(rhs.strip())
        if name.isidentifier() and got is not None:
            out[name] = got
    return out


def seq_members(text: str, seq_name: str) -> list[str]:
    """The dotted tokens of ``def <seq_name> = Seq(a.B, c.D.VV, ...)``, in order. Empty when absent."""
    i = text.find(f"def {seq_name}")
    if i < 0:
        return []
    body = _balanced_after(_strip_comments(text[i:]), "Seq", "(", ")")
    if body is None:
        return []
    return [t for t in _split_top_level(body) if t]


def _all_ints(tokens: list[str]) -> bool:
    return bool(tokens) and all(t.isdecimal() for t in tokens)


def seq_int_tuples(text: str, def_name: str) -> list[tuple[int, ...]]:
    """The integer tuples of ``def <def_name> ... = ... Seq((8, 4), (32, 1)) ...``, in order.

    Reads the first ``Seq(`` after the declaration, so a guarded form (``if (x) Seq(...) else Nil``)
    yields the declared list. Non-integer members are skipped, never coerced.
    """
    stripped = _strip_comments(text)
    i = stripped.find(f"def {def_name}")
    if i < 0:
        return []
    rhs = stripped[stripped.find("=", i) + 1 :]
    body = _balanced_after(rhs, "Seq", "(", ")")
    out: list[tuple[int, ...]] = []
    for tup in _split_top_level(body or ""):
        parts = [t.strip() for t in tup.strip().removeprefix("(").removesuffix(")").split(",")]
        if _all_ints(parts):
            out.append(tuple(int(t) for t in parts))
    return out


def match_case_table(text: str, container: str, def_name: str) -> dict[tuple[int, ...], int]:
    """``{(8, 4): 4, ...}`` from ``def <def_name>(...) = (...) match { case (8, 4) => 4 ... }`` in ``container``.

    Only integer cases are read; a wildcard ``case _`` is a default the caller must not rely on, so it is
    left out and a missing key is the caller's to report.
    """
    body = _balanced_after(_strip_comments(text), f"object {container}")
    if body is None:
        return {}
    i = body.find(f"def {def_name}")
    if i < 0:
        return {}
    table = _balanced_after(body[i:], "match", "{", "}") or ""
    out: dict[tuple[int, ...], int] = {}
    for line in table.splitlines():
        head, arrow, rhs = line.strip().partition("=>")
        head = head.strip()
        if not arrow or not head.startswith("case ("):
            continue
        key = [t.strip() for t in head[len("case ") :].removeprefix("(").removesuffix(")").split(",")]
        if _all_ints(key) and rhs.strip().isdecimal():
            out[tuple(int(t) for t in key)] = int(rhs.strip())
    return out


def _props_tokens(text: str, obj: str) -> list[str] | None:
    """The top-level entries of ``object <obj> ... { val props = Seq( ... ) }``, across lines."""
    stripped = _strip_comments(text)
    header = f"object {obj} "
    i = stripped.find(header)
    if i < 0:
        return None
    j = stripped.find("val props", i)
    nxt = stripped.find("\nobject ", i + len(header))
    if j < 0 or (0 <= nxt < j):
        return None
    body = _balanced_after(stripped[j:], "Seq", "(", ")")
    if body is None:
        return None
    return _split_top_level(body)


def _call(token: str) -> tuple[str, str] | None:
    """``("F6", "OPMFunct6.x")`` for ``F6(OPMFunct6.x)``; None for a token that is not a call."""
    t = token.strip()
    open_ = t.find("(")
    if open_ <= 0 or not t.endswith(")"):
        return None
    return t[:open_].strip(), t[open_ + 1 : -1].strip()


def _uint_literal(token: str) -> int | None:
    """The integer in ``31.U(5.W)`` / ``0.U`` / ``31``."""
    head = token.strip().split(".U", 1)[0].strip()
    try:
        return int(head, 0)
    except ValueError:
        return None


# ---------------------------------------------------------------------------------------------
# The derived table
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskedEncoding:
    """One instruction as a decoder sees it: the bits it decodes and their values.

    ``fill``/``fill_mask`` are bits the RTL decoder does NOT look at but that an assembler directive must
    still spell (e.g. a ``vm`` bit the decoder ignores). They come from the cross-check header, are
    recorded separately so nobody mistakes them for RTL facts, and are never part of the identity.
    """

    mnemonic: str
    match: int
    mask: int
    origin: str
    flags: dict[str, Any] = field(default_factory=dict)
    #: Decode-key conditions that are not instruction bits (e.g. a required element width).
    vtype: dict[str, int] = field(default_factory=dict)
    fill: int = 0
    fill_mask: int = 0
    fill_source: str = ""

    def __post_init__(self) -> None:
        # A match bit outside the mask can never be compared, so the pattern would match nothing.
        if self.match & ~self.mask or self.fill & ~self.fill_mask or self.fill_mask & self.mask:
            raise ValueError(f"{self.mnemonic}: inconsistent match/mask/fill bits")

    def matches(self, word: int) -> bool:
        return (int(word) & self.mask) == self.match

    def fixed(self, name: str) -> int | None:
        """A field's value when every one of its bits is decoded (or filled), else None."""
        m = _field_mask(name)
        known = self.mask | self.fill_mask
        if known & m != m:
            return None
        return _get(self.match | (self.fill & ~self.mask), name)

    # The (opcode, funct3, funct6) view, so code written against the triple keeps working.
    @property
    def opcode(self) -> int | None:
        return self.fixed("opcode")

    @property
    def funct3(self) -> int | None:
        return self.fixed("funct3")

    @property
    def funct6(self) -> int | None:
        return self.fixed("funct6")

    @property
    def funct7(self) -> int | None:
        return self.fixed("funct7")

    def insn_r(self, rd: str | None = None, rs1: str | None = None, rs2: str | None = None) -> str:
        """The ``.insn r`` directive for this instruction.

        A register field the encoding FIXES is spelled from the encoding (as ``x<n>``), and the caller
        must leave it None: an instruction distinguished by its ``rs2`` field would otherwise become a
        different instruction when a caller passed a different register. A field the encoding leaves
        free must be supplied.
        """
        for name in ("opcode", "funct3", "funct7"):
            if self.fixed(name) is None:
                raise ValueError(
                    f"{self.mnemonic}: {name} is not fully determined by the derivation or its cross-check; "
                    "refusing to spell an instruction with guessed bits"
                )
        ops = []
        for name, given in (("rd", rd), ("rs1", rs1), ("rs2", rs2)):
            fixed = self.fixed(name)
            if fixed is not None:
                if given is not None:
                    raise ValueError(f"{self.mnemonic}: {name} is fixed to x{fixed} by the encoding; do not pass one")
                ops.append(f"x{fixed}")
            elif given is None:
                raise ValueError(f"{self.mnemonic}: {name} is an operand of this instruction and must be given")
            else:
                ops.append(given)
        return f".insn r {self.opcode:#x}, {self.funct3:#x}, {self.funct7:#x}, {ops[0]}, {ops[1]}, {ops[2]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "match": f"{self.match:#010x}",
            "mask": f"{self.mask:#010x}",
            "origin": self.origin,
            "fields": {n: self.fixed(n) for n in ("opcode", "funct3", "funct6", "funct7", "rs1", "rs2", "rd")},
            "flags": dict(self.flags),
            "vtype": dict(self.vtype),
            "fill": f"{self.fill:#010x}",
            "fill_mask": f"{self.fill_mask:#010x}",
            "fill_source": self.fill_source,
        }


@dataclass(frozen=True)
class MatrixIsa:
    """The derived encodings, what could not be grounded, and every cross-check that ran.

    ``ok`` is True only when every requested instruction was derived, every one was checked against at
    least one independent source, and every check agreed. Code must not be emitted from one that is not.
    """

    encodings: dict[str, MaskedEncoding] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    gaps: tuple[str, ...] = ()
    crosschecks: tuple[dict[str, Any], ...] = ()

    @property
    def ok(self) -> bool:
        if self.gaps or not self.encodings or not self.crosschecks:
            return False
        if not all(c.get("agrees") for c in self.crosschecks):
            return False
        checked = {c.get("instruction") for c in self.crosschecks}
        return set(self.encodings) <= checked

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "sources": dict(self.sources),
            "gaps": list(self.gaps),
            "crosschecks": [dict(c) for c in self.crosschecks],
            "encodings": {k: e.to_dict() for k, e in sorted(self.encodings.items())},
        }


def _value_of(
    token: str, *, consts: Mapping[str, int], ordinals: Mapping[str, int], enum: str, width: int
) -> tuple[int, int] | str:
    """``(match, mask)`` of a decode-key VALUE within a field of ``width`` bits, or the reason it cannot
    be read. Accepts a BitPat, a ``N.U(w.W)`` literal, a consts name, or an enum member."""
    bp = parse_bitpat(token)
    if bp is not None:
        m, k, w = bp
        if w != width:
            return f"{token}: {w} bits for a {width}-bit field"
        return m, k
    lit = _uint_literal(token)
    if lit is not None:
        return lit, (1 << width) - 1
    qualifier, _, simple = token.strip().rpartition(".")
    if qualifier.rsplit(".", 1)[-1] == enum:
        if simple not in ordinals:
            return f"{token}: not a member of {enum}"
        return ordinals[simple], (1 << width) - 1
    if simple in consts:
        return consts[simple], (1 << width) - 1
    return f"{token}: not a literal, a consts name or an enum member"


def derive(
    *,
    consts: str | Path,
    instructions: str | Path,
    params: str | Path,
    insn_seq: str,
    consts_container: str,
    funct6_enum: str,
    opcode_name: str,
    decode_fields: Mapping[str, str],
    opcode_flags: Mapping[str, str] | None = None,
    form_funct3: Mapping[str, str] | None = None,
    vtype_fields: Sequence[str] = (),
    core_source: str | Path | None = None,
    core_container: str | None = None,
    core_insns: Sequence[str] = (),
) -> MatrixIsa:
    """Derive every instruction the unit implements, from the unit's decode table and the host core's
    ``BitPat`` declarations.

    ``decode_fields`` maps a decode-key prop to the instruction field it keys (``{"F6": "funct6", ...}``);
    ``opcode_flags`` maps a boolean prop to the consts name of the major opcode it selects (an instruction
    without any of those flags set uses ``opcode_name``); ``form_funct3`` maps an instantiation suffix
    (``.VV``) to the consts name of its funct3, for objects whose funct3 comes from the form rather than
    from a prop. ``vtype_fields`` are decode-key props that test the vector configuration rather than the
    instruction word; they are recorded, not encoded. ``core_*`` names the host core's ``BitPat`` object
    and which of its declarations belong to the unit.
    """
    gaps: list[str] = []
    paths = {"consts": Path(consts), "instructions": Path(instructions), "params": Path(params)}
    if core_source is not None:
        paths["core"] = Path(core_source)
    texts: dict[str, str] = {}
    for label, p in paths.items():
        try:
            texts[label] = p.read_text(encoding="utf-8")
        except OSError as exc:
            gaps.append(f"{label}: unreadable ({exc})")
    sources = {k: str(v) for k, v in paths.items()}
    if gaps:
        return MatrixIsa(sources=sources, gaps=tuple(gaps))

    consts_defs = bit_literal_defs(texts["consts"], consts_container)
    ordinals = chisel_enum_ordinals(texts["consts"], funct6_enum)
    members = seq_members(texts["params"], insn_seq)
    if not members:
        gaps.append(f"instruction sequence {insn_seq!r} is empty or absent")
    if opcode_name not in consts_defs:
        gaps.append(f"opcode {opcode_name!r} not found in {consts_container}")
    flag_opcodes = dict(opcode_flags or {})
    for flag, name in flag_opcodes.items():
        if name not in consts_defs:
            gaps.append(f"opcode {name!r} (selected by {flag}) not found in {consts_container}")
    if gaps:
        return MatrixIsa(sources=sources, gaps=tuple(gaps))

    encodings: dict[str, MaskedEncoding] = {}
    forms = dict(form_funct3 or {})
    for member in members:
        parts = member.split(".")
        obj, form = parts[-1], None
        if parts[-1] in forms and len(parts) >= 2:
            obj, form = parts[-2], parts[-1]
        tokens = _props_tokens(texts["instructions"], obj)
        if tokens is None:
            gaps.append(f"{obj}: no props declaration found")
            continue
        match = mask = 0
        flags: dict[str, Any] = {}
        vtype: dict[str, int] = {}
        opcode_const = opcode_name
        bad: list[str] = []
        if form is not None:
            f3 = forms[form]
            if f3 not in consts_defs:
                bad.append(f"form {form}: funct3 {f3!r} not in {consts_container}")
            else:
                match |= _put(consts_defs[f3], "funct3")
                mask |= _field_mask("funct3")
        for tok in tokens:
            call = _call(tok)
            if call is not None and call[0] in decode_fields:
                fname = decode_fields[call[0]]
                width = FORMAT[fname][1]
                got = _value_of(call[1], consts=consts_defs, ordinals=ordinals, enum=funct6_enum, width=width)
                if isinstance(got, str):
                    bad.append(got)
                    continue
                m, k = got
                match |= _put(m & k, fname)
                mask |= _put(k, fname)
                continue
            if call is not None and call[0] in vtype_fields:
                lit = _uint_literal(call[1])
                if lit is None:
                    bad.append(f"{tok}: vtype condition is not a literal")
                else:
                    vtype[call[0]] = lit
                continue
            if call is not None:
                flags[call[0]] = call[1]
                continue
            key, dot, val = tok.partition(".")
            if dot and val in ("Y", "N"):
                flags[key.strip()] = val == "Y"
                if key.strip() in flag_opcodes and val == "Y":
                    opcode_const = flag_opcodes[key.strip()]
        if bad:
            gaps.extend(f"{obj}: {b}" for b in bad)
            continue
        match |= _put(consts_defs[opcode_const], "opcode")
        mask |= _field_mask("opcode")
        encodings[obj] = MaskedEncoding(
            mnemonic=obj,
            match=match,
            mask=mask,
            origin=f"decode table {paths['instructions'].name}:{obj} (opcode {opcode_const})",
            flags=flags,
            vtype=vtype,
        )

    if core_source is not None:
        defs = bitpat_defs(texts["core"], str(core_container))
        for name in core_insns:
            got = defs.get(name)
            if got is None:
                gaps.append(f"{name}: no BitPat declaration in object {core_container}")
                continue
            m, k, w = got
            if w != INSN_BITS:
                gaps.append(f"{name}: BitPat is {w} bits, not a {INSN_BITS}-bit instruction")
                continue
            encodings[name] = MaskedEncoding(
                mnemonic=name, match=m, mask=k, origin=f"BitPat {paths['core'].name}:{core_container}.{name}"
            )

    _check_disjoint(encodings, gaps)
    return MatrixIsa(encodings=encodings, sources=sources, gaps=tuple(gaps))


def _check_disjoint(encodings: Mapping[str, MaskedEncoding], gaps: list[str]) -> None:
    """Two derived instructions that one word can decode as both are a derivation error, not a choice."""
    items = sorted(encodings.items())
    for i, (a, ea) in enumerate(items):
        for b, eb in items[i + 1 :]:
            common = ea.mask & eb.mask
            if (ea.match ^ eb.match) & common == 0 and ea.vtype == eb.vtype:
                gaps.append(f"{a} and {b} decode the same words; the derivation cannot tell them apart")


# ---------------------------------------------------------------------------------------------
# Independent sources
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HeaderSite:
    """One ``.insn r`` in a C header, with the bits it spells as literals."""

    site: str  # enclosing macro or function name
    match: int
    mask: int
    template: str


def _asm_template(text: str, start: int) -> str:
    """The asm template beginning at the string literal that contains ``start``.

    Adjacent literals concatenate; a macro call between them (``" VME_V(vd) "``) is an operand spelled at
    preprocessing time and becomes ``@``. Stops at the operand/clobber section or the closing paren.
    """
    q = text.rfind('"', 0, start)
    out: list[str] = []
    i = q + 1
    n = len(text)
    while i < n:
        end = text.find('"', i)
        if end < 0:
            break
        out.append(text[i:end])
        j = end + 1
        while j < n and text[j] in " \t\\\n":
            j += 1
        if j >= n or text[j] in ":);":
            break
        if text[j] == '"':
            i = j + 1
            continue
        # A preprocessor-spelled operand between two literals.
        depth, k = 0, j
        while k < n:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                if depth == 0:
                    break
                depth -= 1
                if depth == 0:
                    k += 1
                    break
            elif text[k] == '"' and depth == 0:
                break
            k += 1
        out.append("@")
        while k < n and text[k] in " \t\\\n":
            k += 1
        if k < n and text[k] == '"':
            i = k + 1
            continue
        break
    return "".join(out)


def _blank_c_comments(text: str) -> str:
    """``text`` with C comments replaced by spaces (newlines kept), so positions stay valid and a
    ``.insn r`` mentioned in prose is not read as an instruction."""
    out = list(text)
    i, n = 0, len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            i += 1
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
            continue
        i += 1
    return "".join(out)


_C_CONTROL = frozenset({"if", "for", "while", "switch", "do", "else"})


def _enclosing_name(text: str, pos: int) -> str | None:
    """The macro whose definition line, or the function whose body, contains ``pos``."""
    line_start = text.rfind("\n", 0, pos) + 1
    # A macro body may continue across backslash-newlines; walk back to its #define.
    k = line_start
    while True:
        line = text[k : text.find("\n", k) if text.find("\n", k) >= 0 else len(text)].lstrip()
        if line.startswith("#define "):
            head = line[len("#define ") :]
            return head.split("(", 1)[0].split()[0]
        prev_end = k - 1
        if prev_end <= 0 or not text[:prev_end].rstrip(" \t").endswith("\\"):
            break
        k = text.rfind("\n", 0, prev_end) + 1
    # Otherwise the innermost function definition whose brace-balanced body contains pos.
    depth = 0
    i = pos
    while i > 0:
        i -= 1
        c = text[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                head = text[text.rfind("\n", 0, text.rfind(")", 0, i)) + 1 : i]
                paren = head.find("(")
                if paren > 0 and head[:paren].split():
                    name = head[:paren].split()[-1].lstrip("*")
                    if name.isidentifier() and name not in _C_CONTROL:
                        return name
                    if name in _C_CONTROL:
                        continue  # a nested block: keep walking out to the function
                return None
            depth -= 1
    return None


def header_insn_sites(text: str) -> dict[str, list[HeaderSite]]:
    """``{site: [HeaderSite]}`` for every ``.insn r`` in a C header, keyed by enclosing macro/function.

    The literal fields (opcode, funct3, funct7 and any register written as ``x<n>``) form the site's
    ``(match, mask)``; operands spelled through ``%n`` or a macro stay unmasked.
    """
    out: dict[str, list[HeaderSite]] = {}
    marker = ".insn r"
    text = _blank_c_comments(text)
    pos = text.find(marker)
    while pos >= 0:
        template = _asm_template(text, pos)
        name = _enclosing_name(text, pos)
        body = template[template.find(marker) + len(marker) :] if marker in template else ""
        fields = [f.strip() for f in body.split(",")]
        if name and len(fields) == 6:
            ints = []
            for f in fields[:3]:
                try:
                    ints.append(int(f, 0))
                except ValueError:
                    ints = []
                    break
            if len(ints) == 3:
                match = _put(ints[0], "opcode") | _put(ints[1], "funct3") | _put(ints[2], "funct7")
                mask = _field_mask("opcode") | _field_mask("funct3") | _field_mask("funct7")
                for fname, tok in zip(("rd", "rs1", "rs2"), fields[3:]):
                    if tok.startswith("x") and tok[1:].isdigit():
                        match |= _put(int(tok[1:]), fname)
                        mask |= _field_mask(fname)
                out.setdefault(name, []).append(HeaderSite(name, match, mask, template.strip()))
        pos = text.find(marker, pos + len(marker))
    return out


#: The fields an emitted ``.insn r`` must spell in full, besides the register operands.
_EMIT_FIELDS = ("opcode", "funct3", "funct7")


def crosscheck_header(isa: MatrixIsa, header: str | Path, *, pairs: Mapping[str, str]) -> MatrixIsa:
    """Check every derived encoding against the header site ``pairs`` names, and fill emission bits.

    Agreement means: every bit BOTH sources state has the same value, and the header states every bit of
    the derived opcode/funct3/funct6. Bits the header spells that the RTL decoder ignores, within the
    fields an ``.insn r`` must carry, become the encoding's ``fill`` -- recorded as coming from the header.
    A site the header does not have, or a name with no declared pair, is a failed check, never a pass.
    """
    label = str(header)
    try:
        sites = header_insn_sites(Path(header).read_text(encoding="utf-8"))
    except OSError as exc:
        return MatrixIsa(
            encodings=isa.encodings,
            sources={**isa.sources, "header": label},
            gaps=isa.gaps,
            crosschecks=isa.crosschecks + ({"agrees": False, "source": label, "reason": f"unreadable: {exc}"},),
        )
    emit_mask = 0
    for f in _EMIT_FIELDS:
        emit_mask |= _field_mask(f)
    new: dict[str, MaskedEncoding] = {}
    records: list[dict[str, Any]] = []
    for name, enc in sorted(isa.encodings.items()):
        site_name = pairs.get(name)
        found = sites.get(site_name or "")
        if not site_name or not found:
            records.append(
                {
                    "instruction": name,
                    "source": label,
                    "site": site_name,
                    "agrees": False,
                    "reason": "no cross-check site declared" if not site_name else f"{site_name!r} not in header",
                }
            )
            new[name] = enc
            continue
        site_records = []
        for s in found:
            both = enc.mask & s.mask
            clash = (enc.match ^ s.match) & both
            identity = _field_mask("opcode") | _field_mask("funct3") | _field_mask("funct6")
            unstated = enc.mask & identity & ~s.mask
            site_records.append((s, clash, unstated))
        bad = [(s, c, u) for s, c, u in site_records if c or u]
        site = found[0]
        if bad:
            s, clash, unstated = bad[0]
            records.append(
                {
                    "instruction": name,
                    "source": label,
                    "site": site_name,
                    "agrees": False,
                    "reason": (
                        f"bits {clash:#010x} differ (rtl {enc.match & clash:#010x}, header {s.match & clash:#010x})"
                        if clash
                        else f"header does not state identity bits {unstated:#010x}"
                    ),
                    "template": s.template,
                }
            )
            new[name] = enc
            continue
        # Emission fill: bits of the always-spelled fields (and of any register the header fixes) that the
        # RTL does not decode. Every site must agree on them, or the header itself is ambiguous.
        fill_mask = (emit_mask | site.mask) & site.mask & ~enc.mask
        fills = {s.match & fill_mask for s in found}
        if len(fills) > 1:
            records.append(
                {
                    "instruction": name,
                    "source": label,
                    "site": site_name,
                    "agrees": False,
                    "reason": "the header's sites disagree on bits the RTL does not decode",
                }
            )
            new[name] = enc
            continue
        new[name] = MaskedEncoding(
            mnemonic=enc.mnemonic,
            match=enc.match,
            mask=enc.mask,
            origin=enc.origin,
            flags=enc.flags,
            vtype=enc.vtype,
            fill=site.match & fill_mask,
            fill_mask=fill_mask,
            fill_source=f"{label}:{site_name}" if fill_mask else "",
        )
        records.append(
            {
                "instruction": name,
                "source": label,
                "site": site_name,
                "agrees": True,
                "checked_bits": f"{enc.mask & site.mask:#010x}",
                "filled_bits": f"{fill_mask:#010x}",
                "sites": len(found),
            }
        )
    return MatrixIsa(
        encodings=new,
        sources={**isa.sources, "header": label},
        gaps=isa.gaps,
        crosschecks=isa.crosschecks + tuple(records),
    )


def opcode_table_defs(text: str) -> dict[str, tuple[int, int]]:
    """``{NAME: (match, mask)}`` from a riscv-opcodes style header's ``#define MATCH_NAME`` / ``MASK_NAME``."""
    matches: dict[str, int] = {}
    masks: dict[str, int] = {}
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) != 3 or parts[0] != "#define":
            continue
        name, value = parts[1], parts[2]
        try:
            v = int(value, 0)
        except ValueError:
            continue
        if name.startswith("MATCH_"):
            matches[name[len("MATCH_") :]] = v
        elif name.startswith("MASK_"):
            masks[name[len("MASK_") :]] = v
    return {k: (matches[k], masks[k]) for k in matches if k in masks}


def crosscheck_opcode_table(isa: MatrixIsa, table: str | Path, *, pairs: Mapping[str, str]) -> MatrixIsa:
    """Check the derivation against a simulator's MATCH/MASK table (e.g. a riscv-opcodes ``encoding.h``).

    Agreement: on every bit both decoders look at, the values are equal, and on every bit the simulator
    decodes that the derivation FILLED from its header, the fill equals the simulator's value -- so an
    emitted word is one this simulator decodes as the paired instruction. Bits only the simulator decodes
    (e.g. a register-field bit the kernel supplies as an operand) are reported, not judged.
    """
    label = str(table)
    try:
        defs = opcode_table_defs(Path(table).read_text(encoding="utf-8"))
    except OSError as exc:
        return MatrixIsa(
            encodings=isa.encodings,
            sources={**isa.sources, "opcode_table": label},
            gaps=isa.gaps,
            crosschecks=isa.crosschecks + ({"agrees": False, "source": label, "reason": f"unreadable: {exc}"},),
        )
    records: list[dict[str, Any]] = []
    for name, enc in sorted(isa.encodings.items()):
        other = pairs.get(name)
        if other is None or other not in defs:
            records.append(
                {"instruction": name, "source": label, "entry": other, "agrees": False, "reason": "no table entry"}
            )
            continue
        m, k = defs[other]
        clash = (enc.match ^ m) & (enc.mask & k)
        fill_clash = (enc.fill ^ m) & (enc.fill_mask & k)
        only_sim = k & ~(enc.mask | enc.fill_mask)
        agrees = not clash and not fill_clash
        records.append(
            {
                "instruction": name,
                "source": label,
                "entry": other,
                "agrees": agrees,
                "checked_bits": f"{enc.mask & k:#010x}",
                "operand_bits_only_simulator_decodes": f"{only_sim:#010x}",
                **(
                    {}
                    if agrees
                    else {"reason": f"decoded bits differ {clash:#010x}, filled bits differ {fill_clash:#010x}"}
                ),
            }
        )
    return MatrixIsa(
        encodings=isa.encodings,
        sources={**isa.sources, "opcode_table": label},
        gaps=isa.gaps,
        crosschecks=isa.crosschecks + tuple(records),
    )


# ---------------------------------------------------------------------------------------------
# Which vector unit carries the matrix unit
# ---------------------------------------------------------------------------------------------


def _find_def(text: str, name: str) -> int:
    """Index of ``def <name>`` not followed by more identifier characters (``def a`` is not ``def ab``)."""
    i = text.find(f"def {name}")
    while i >= 0:
        end = i + len("def ") + len(name)
        if end >= len(text) or not (text[end].isalnum() or text[end] == "_"):
            return i
        i = text.find(f"def {name}", end)
    return -1


def _params_flag(params_text: str, params_object: str, member: str, flag: str, *, depth: int = 0) -> bool | None:
    """The value of ``flag`` in ``params_object.member``, following ``.copy(...)`` bases to the class default.

    None when it cannot be grounded (a member that is not declared, a base that cannot be followed).
    """
    if depth > 16:
        return None
    body = _balanced_after(_strip_comments(params_text), f"object {params_object}")
    if body is None:
        return None
    i = _find_def(body, member)
    if i < 0:
        return None
    rhs = body[body.find("=", i) + 1 :].lstrip()
    call_head = rhs.split("(", 1)[0].strip()
    args = _balanced_after(rhs, call_head, "(", ")")
    if args is None:
        return None
    for arg in _split_top_level(args):
        key, sep, val = arg.partition("=")
        if sep and key.strip() == flag:
            v = val.strip()
            return True if v == "true" else False if v == "false" else None
    if call_head.endswith(".copy"):
        return _params_flag(params_text, params_object, call_head[: -len(".copy")].strip(), flag, depth=depth + 1)
    # A constructor call: the class's declared default.
    decl = _balanced_after(_strip_comments(params_text), f"case class {call_head}", "(", ")")
    if decl is None:
        return None
    for arg in _split_top_level(decl):
        name, _, rest = arg.partition(":")
        if name.strip() == flag:
            default = rest.partition("=")[2].strip()
            return True if default == "true" else False if default == "false" else None
    return None


def matrix_unit_params(
    config_text: str,
    config_class: str,
    *,
    mixin_text: str,
    params_text: str,
    params_object: str,
    enable_flag: str,
) -> tuple[dict[str, int], list[str]]:
    """``({vLen, dLen, ...}, notes)`` for the vector-unit mixin of ``config_class`` that ENABLES the unit.

    A heterogeneous SoC instantiates several vector units, and only some carry the matrix unit; reading the
    first one (what a plain VectorUnit lookup does) can return the wrong unit's geometry. Here each mixin's
    params argument (``VectorParams.opuMxParams``) is resolved against ``params_object`` and followed through
    its ``.copy`` chain to ``enable_flag``. Exactly the mixins whose flag is true are candidates; if they
    disagree on geometry, or none is found, the result is empty and the notes say why.
    """
    notes: list[str] = []
    body = _balanced_after(_strip_comments(config_text), f"class {config_class} extends Config(", "(", ")")
    if body is None:
        return {}, [f"config {config_class!r} not found"]
    picks: list[dict[str, int]] = []
    for mixin in _split_top_level(body, "+"):
        head = mixin.strip().removeprefix("new ").strip().split("(", 1)[0]
        simple = head.strip().rsplit(".", 1)[-1]
        if not simple.endswith(_VECTOR_UNIT_MIXIN_SUFFIX):
            continue
        call = _balanced_after(mixin, head.strip(), "(", ")")
        args = _split_top_level(call) if call else []
        names = scala_param_names(mixin_text, f"class {simple}")
        bound: dict[str, str] = {}
        position = 0
        for arg in args:
            key, sep, rhs = arg.partition("=")
            if sep and key.strip().isidentifier() and not rhs.startswith("="):
                bound[key.strip()] = rhs.strip()
            else:
                if position < len(names):
                    bound[names[position]] = arg.strip()
                position += 1
        enabled = None
        for token in bound.values():
            if token.startswith(f"{params_object}."):
                enabled = _params_flag(params_text, params_object, token[len(params_object) + 1 :], enable_flag)
        if enabled is None:
            notes.append(f"{simple}: could not resolve {enable_flag} from its params")
            continue
        if not enabled:
            notes.append(f"{simple}: {enable_flag} is false")
            continue
        ints: dict[str, int] = {}
        for k, v in bound.items():
            got = _bit_literal(v) if v.startswith('"') else _uint_literal(v)
            if got is not None:
                ints[k] = got
        picks.append(ints)
    if not picks:
        return {}, notes + [f"no vector unit in {config_class} enables {enable_flag}"]
    if any(p != picks[0] for p in picks[1:]):
        return {}, notes + [f"vector units enabling {enable_flag} disagree on geometry: {picks}"]
    return picks[0], notes

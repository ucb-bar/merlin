"""Derive and expose matrix ISA evidence from selected hardware declarations.

Ported from OPU-VME 80e00803d; hardware values remain caller-supplied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.development.targetgen.rtl.matrixchecks import (
    HeaderSite,
    crosscheck_header,
    crosscheck_opcode_table,
    header_insn_sites,
    opcode_table_defs,
)
from merlin.development.targetgen.rtl.matrixconfig import matrix_unit_params
from merlin.development.targetgen.rtl.matrixencoding import (
    FORMAT,
    INSN_BITS,
    MaskedEncoding,
    MatrixIsa,
    _field_mask,
    _put,
)
from merlin.development.targetgen.rtl.matrixparse import (
    _call,
    _props_tokens,
    _uint_literal,
    bitpat_defs,
    match_case_table,
    parse_bitpat,
    seq_int_tuples,
    seq_members,
)
from merlin.development.targetgen.rtl.scala_parse import bit_literal_defs, chisel_enum_ordinals

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

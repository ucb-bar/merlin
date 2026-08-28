"""Decode a self-hosted-ISA stream against the target's OWN derived field layout.

Named for what it does, not for the target it was first exercised on. It contains no bit position, no
opcode value and no instruction width: everything comes from the encoding mlc recovers from a target's
decoder RTL, so pointing it at another derived-ISA target decodes that one. A module named after one
target is a claim about scope, and this module's scope is every target with a derived encoding.

Unlike :mod:`kernels.decode.rocc`, which lays a command over the fixed RISC-V R-type format, nothing
about the field positions here is known in advance. They are read from the encoding mlc recovers from
the target's decoder RTL (``mlc_bridge.isa_encoding_for``: ``inst_width``, ``fields`` as ``[hi, lo]``
bit ranges, and an opcode table), so this module contains no bit position, no opcode value and no
instruction width. Point it at a different SIMT target and it decodes that one.

Two measured facts shape it:

* **The custom surface spans TWO encoding spaces.** A real kernel's unnameable words split between the
  custom opcode space and the REPURPOSED standard OP space — roughly a third and two thirds. A decoder
  that reads only the custom space sees a minority of what the target actually executes.
* **A disassembly probe must pin its ISA extensions.** With the extensions left to the tool's default,
  76% of a kernel's words came back unnamed and looked like a vast custom surface; given the extensions
  explicitly it is 15%, and that 15% is the real one. :func:`accountable` exists so a caller can tell
  "the tool could not name this" from "this is the endpoint's own instruction", which is the distinction
  that probe got wrong.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from merlin.kernels.decode.opu import UNKNOWN_MNEMONIC

__all__ = ["DerivedIsaAudit", "DerivedIsaInsn", "audit", "decode_stream", "digest",
           "encoding_for", "fields_of"]


def encoding_for(target: str) -> dict[str, Any]:
    """The target's RTL-derived instruction encoding, or ``{}`` when mlc/the cache is unavailable.

    Empty rather than a default layout: a decoder with no derived encoding must report that it decoded
    nothing, never decode against a guess.
    """
    from merlin.targetgen.rtl import mlc_bridge as _mb
    return _mb.isa_encoding_for(target) or {}


def _word_of(hexcode: str, width_bits: int) -> int | None:
    """The instruction word from an objdump hex column, at the DERIVED width.

    A column narrower than the declared width is declined rather than zero-extended: zero-extending a
    16-bit compressed word into a wide field layout produces a confident decode of an instruction that
    was never there.
    """
    token = (hexcode or "").strip().replace(" ", "")
    want = max(1, width_bits // 4)
    if len(token) != want:
        return None
    try:
        return int(token, 16)
    except ValueError:
        return None


def fields_of(word: int, layout: Mapping[str, Any]) -> dict[str, int]:
    """Extract every declared field from ``word`` using ``[hi, lo]`` bit ranges from the derivation."""
    out: dict[str, int] = {}
    for name, rng in (layout or {}).items():
        try:
            hi, lo = int(rng[0]), int(rng[1])
        except (TypeError, ValueError, IndexError):
            continue
        if hi < lo:
            hi, lo = lo, hi
        out[str(name)] = (word >> lo) & ((1 << (hi - lo + 1)) - 1)
    return out


@dataclass(frozen=True)
class DerivedIsaInsn:
    index: int
    addr: int
    identity: str                  # the derived opcode-space name, or the disassembler's text
    space: str = ""                # which derived opcode space the word falls in
    #: Roles the endpoint declares for this opcode space. Coarser than a per-instruction table -- a
    #: space groups many operations -- and that coarseness is the honest resolution of this evidence:
    #: the decoder says which space a word is in, not which operation it performs within it.
    roles: tuple[str, ...] = ()
    from_endpoint: bool = False
    mnemonic: str = ""
    operands: tuple[str, ...] = ()
    fields: dict[str, int] = field(default_factory=dict)
    #: Width IN BITS of the objdump hex column this came from, 0 when unknown. Kept because an entry
    #: narrower than the ISA's minimum instruction width is not an instruction at all -- objdump emits
    #: byte fallbacks where it cannot form one -- and counting those as unplaced INSTRUCTIONS inflates
    #: both the denominator and the gap. Recorded as the observation (a width), not the conclusion.
    hex_bits: int = 0


#: RISC-V R-type funct3 position, for resolving WHICH instruction a custom opcode space holds. The
#: instruction FORMAT, not a value: the values come from the target's own intrinsics header.
_F3_SHIFT, _F3_MASK = 12, 0x7


#: RISC-V R-type funct7 position. Format, not a value.
_F7_SHIFT, _F7_MASK = 25, 0x7F


def decode_stream(insns: Sequence[Any], encoding: Mapping[str, Any],
                  spaces: Sequence[str] = (), roles_of=None,
                  custom_table: "Mapping[tuple, str] | None" = None,
                  cede_funct7_in: Sequence[str] = ()) -> list[DerivedIsaInsn]:
    """Decode a stream against a derived encoding.

    ``spaces`` names the opcode-table entries that belong to this endpoint (e.g. the custom space plus
    a repurposed standard space). Empty means "report the space but claim nothing", which is the honest
    state for a target whose endpoint boundary has not been established.

    ``roles_of`` maps an opcode-space name to the roles the endpoint declares for it. Without it the
    stream is decoded but carries no MEANING, and a CCA lifted from it compares equal to everything —
    so an audit reports such a stream as claimed-but-unroled rather than as understood.
    """
    width = int(encoding.get("inst_width") or 32)
    layout = encoding.get("fields") or {}
    by_value: dict[int, str] = {}
    for name, value in (encoding.get("opcodes") or {}).items():
        by_value.setdefault(int(value), str(name))       # first name wins; aliases are reported as one
    mine = {str(s).upper().replace("-", "_") for s in spaces}
    out: list[DerivedIsaInsn] = []
    for i, insn in enumerate(insns):
        _hex = str(getattr(insn, "hexcode", "") or "").strip().replace(" ", "")
        word = _word_of(_hex, width)
        f = fields_of(word, layout) if word is not None else {}
        space = by_value.get(f.get("opcode"), "") if f else ""
        # Inside a custom opcode space the SPACE name says nothing about the operation. When the
        # target's own intrinsics header names the individual instructions, resolve to that finer
        # identity — the difference between "this word is in CUSTOM0" and "this word is a barrier".
        identity = space
        if custom_table and word is not None and space:
            # Scoped to the SHARED space only. Applied to every space it also cedes ordinary
            # arithmetic — a standard R-type instruction has funct7 bits like anything else — and the
            # coverage number falls while looking like a correctness fix. Caught by watching it move
            # the wrong way.
            if space in cede_funct7_in and ((word >> _F7_SHIFT) & _F7_MASK):
                # Another endpoint SHARES this opcode space and carries its operation in funct7; a
                # word with funct7 set is that endpoint's, not an unexplained one. Ceding it keeps the
                # unroled bucket meaningful — otherwise one engine's instructions are reported as the
                # other engine's gap, and the number that should drive work becomes noise.
                identity = ""
            else:
                finer = custom_table.get((f.get("opcode"),
                                          (word >> _F3_SHIFT) & _F3_MASK,
                                          (word >> _F7_SHIFT) & _F7_MASK))
                if finer:
                    identity = finer
        out.append(DerivedIsaInsn(
            index=i, addr=int(getattr(insn, "addr", 0)),
            identity=identity or str(getattr(insn, "mnemonic", "")),
            space=space if identity else "",
            roles=(tuple(roles_of(identity)) if (identity and roles_of) else ()),
            from_endpoint=bool(identity) and space.upper() in mine,
            mnemonic=str(getattr(insn, "mnemonic", "")),
            operands=tuple(getattr(insn, "operands", ()) or ()),
            fields=f, hex_bits=len(_hex) * 4))
    return out


def accountable(d: DerivedIsaInsn) -> bool:
    """Did SOMETHING name this word — the disassembler or the derived table?

    The distinction the 76%-unknown probe got wrong. A word the tool declined to name but the derived
    encoding places in a known opcode space is accounted for; a word neither can place is not, and must
    be reported rather than counted as absent.
    """
    return bool(d.space) or (d.mnemonic and d.mnemonic != UNKNOWN_MNEMONIC)


@dataclass(frozen=True)
class DerivedIsaAudit:
    space_counts: dict[str, int] = field(default_factory=dict)
    total_insns: int = 0
    endpoint_insns: int = 0
    named_by_disassembler: int = 0
    unaccounted: tuple[dict[str, Any], ...] = ()
    digest: str = ""

    @property
    def unaccounted_fraction(self) -> float:
        return (len(self.unaccounted) / self.total_insns) if self.total_insns else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"space_counts": dict(self.space_counts), "total_insns": self.total_insns,
                "endpoint_insns": self.endpoint_insns,
                "named_by_disassembler": self.named_by_disassembler,
                "unaccounted": [dict(u) for u in self.unaccounted],
                "unaccounted_fraction": round(self.unaccounted_fraction, 4),
                "digest": self.digest}


def digest(decoded: Sequence[DerivedIsaInsn]) -> str:
    """Hash of the decoded identity stream, so the inert-lever guard can see this endpoint too."""
    lines = []
    for d in decoded:
        if d.from_endpoint:
            f = d.fields
            lines.append(f"{d.identity} rd={f.get('rd')} rs1={f.get('rs1')} rs2={f.get('rs2')}")
        else:
            lines.append(f"{d.mnemonic} {','.join(d.operands)}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def audit(insns: Sequence[Any], target: str, endpoint=None) -> DerivedIsaAudit:
    """Audit one stream against ``target``'s derived encoding and its endpoint's declared spaces."""
    encoding = encoding_for(target)
    spaces = ()
    if endpoint is not None:
        from merlin.kernels import endpoints as _ep
        block = ((_ep._spec().get("endpoints") or {}).get(endpoint.name) or {})
        spaces = tuple((block.get("encoding") or {}).get("spaces") or ())
    decoded = decode_stream(insns, encoding, spaces)
    counts: dict[str, int] = {}
    unaccounted: list[dict[str, Any]] = []
    for d in decoded:
        if d.space:
            counts[d.space] = counts.get(d.space, 0) + 1
        if not accountable(d):
            unaccounted.append({"index": d.index, "addr": d.addr, "fields": dict(d.fields),
                                "mnemonic": d.mnemonic})
    return DerivedIsaAudit(
        space_counts=counts, total_insns=len(decoded),
        endpoint_insns=sum(1 for d in decoded if d.from_endpoint),
        named_by_disassembler=sum(1 for d in decoded
                                  if d.mnemonic and d.mnemonic != UNKNOWN_MNEMONIC),
        unaccounted=tuple(unaccounted), digest=digest(decoded))

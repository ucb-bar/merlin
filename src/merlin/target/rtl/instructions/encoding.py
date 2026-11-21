"""RISC-V instruction observations with explicit decoded and header-only bits.

Ported from OPU-VME 80e00803d; hardware values remain caller-supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

INSN_BITS = 32


_WORD = (1 << INSN_BITS) - 1


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

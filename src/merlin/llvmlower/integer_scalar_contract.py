"""Explicit original scalar declarations, not source or execution authority."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from merlin.common.digest import is_sha256


@dataclass(frozen=True)
class IntegerScalarLimits:
    source_bytes: int
    receipt_bytes: int
    nesting: int
    integer_bits: int
    operations: int

    def record(self):
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("scalar correspondence needs explicit positive reader limits")
        return dict(vars(self))


@dataclass(frozen=True)
class IntegerScalarSlot:
    name: str
    bits: int

    def record(self, limits):
        if (
            type(self.name) is not str
            or not self.name.isascii()
            or not self.name.isidentifier()
            or type(self.bits) is not int
            or not 0 < self.bits <= limits.integer_bits
        ):
            raise ValueError("original scalar slot needs its exact name and bounded integer width")
        return dict(vars(self))


@dataclass(frozen=True)
class OriginalIntegerScalarAbi:
    entry_symbol: str
    c_interface_symbol: str
    inputs: tuple[IntegerScalarSlot, ...]
    outputs: tuple[IntegerScalarSlot, ...]

    def record(self, limits):
        if (
            type(limits) is not IntegerScalarLimits
            or any(
                type(name) is not str or not name.isascii() or not name.isidentifier()
                for name in (self.entry_symbol, self.c_interface_symbol)
            )
            or self.entry_symbol == self.c_interface_symbol
            or not self.outputs
        ):
            raise ValueError("original scalar ABI requires distinct exact entries and complete results")
        limits.record()
        result = {"entry_symbol": self.entry_symbol, "c_interface_symbol": self.c_interface_symbol}
        for role, slots in (("inputs", self.inputs), ("outputs", self.outputs)):
            if (
                type(slots) is not tuple
                or any(type(slot) is not IntegerScalarSlot for slot in slots)
                or len({slot.name for slot in slots}) != len(slots)
            ):
                raise ValueError("original scalar ABI needs complete ordered distinct slot identities")
            result[role] = [slot.record(limits) for slot in slots]
        return result


@dataclass(frozen=True)
class IntegerScalarNumerics:
    arithmetic: str
    comparisons: str
    overflow_promises: str

    def record(self):
        if (
            self.arithmetic != "modular_bitvector"
            or self.comparisons != "typed_integer_predicates"
            or self.overflow_promises != "none"
            or any(type(value) is not str for value in vars(self).values())
        ):
            raise ValueError(
                "only explicitly selected modular integer semantics without overflow promises are supported"
            )
        return dict(vars(self))


@dataclass(frozen=True)
class OriginalIntegerScalarSource:
    path: Path
    sha256: str
    abi: OriginalIntegerScalarAbi
    numerics: IntegerScalarNumerics

    def record(self, limits):
        if (
            not isinstance(self.path, Path)
            or not self.path.is_absolute()
            or not is_sha256(self.sha256)
            or type(self.abi) is not OriginalIntegerScalarAbi
            or type(self.numerics) is not IntegerScalarNumerics
        ):
            raise ValueError(
                "scalar check needs exact original source identity, typed ordered ABI and numerical choices"
            )
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "abi": self.abi.record(limits),
            "numerics": self.numerics.record(),
        }

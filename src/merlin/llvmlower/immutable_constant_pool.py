"""Plan shared immutable byte storage under an explicit address-use contract.

Providers establish readonly uses, unobserved address identity, representation,
alignment and complete object ownership. This module checks all selected bytes
and bindings; it supplies no target packing, relocation or executable admission.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

ACCESS = "readonly_bytes_address_unobserved"


def _identifier(value):
    return (
        isinstance(value, str)
        and value.isascii()
        and value
        and (value[0].isalpha() or value[0] == "_")
        and all(c.isalnum() or c == "_" for c in value)
    )


def _digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not path.stat().st_size:
        raise ValueError("constant pool needs nonempty regular immutable operand files")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _equal(left, right):
    with left.open("rb") as lhs, right.open("rb") as rhs:
        while True:
            a, b = lhs.read(1 << 20), rhs.read(1 << 20)
            if a != b:
                return False
            if not a:
                return True


@dataclass(frozen=True)
class PoolOperand:
    group: str
    symbol: str
    path: Path
    alignment: int
    representation: str
    access: str


@dataclass(frozen=True)
class PoolBinding:
    operand: PoolOperand
    canonical_symbol: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class PoolBlob:
    symbol: str
    path: Path
    alignment: int
    representation: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ConstantPool:
    bindings: tuple[PoolBinding, ...]
    blobs: tuple[PoolBlob, ...]
    original_bytes: int
    pooled_bytes: int

    def recheck(self):
        """Reopen every source, including every duplicate, before/after emission."""
        selected = {blob.symbol: blob for blob in self.blobs}
        for blob in self.blobs:
            if _digest(blob.path) != blob.sha256 or blob.path.stat().st_size != blob.bytes:
                raise ValueError("constant pool representative changed")
        for binding in self.bindings:
            operand = binding.operand
            blob = selected[binding.canonical_symbol]
            if (
                operand.access != ACCESS
                or operand.representation != blob.representation
                or blob.alignment % operand.alignment
                or _digest(operand.path) != binding.sha256
                or operand.path.stat().st_size != binding.bytes
                or binding.bytes != blob.bytes
                or binding.sha256 != blob.sha256
                or not _equal(operand.path, blob.path)
            ):
                raise ValueError("constant pool operand or binding changed")

    def symbol_maps(self):
        result = {}
        for binding in self.bindings:
            result.setdefault(binding.operand.group, {})[binding.operand.symbol] = binding.canonical_symbol
        return result

    def receipt(self):
        self.recheck()
        return {
            "schema": "merlin.immutable_constant_pool.v1",
            "scope": "readonly_address_independent_byte_pool_not_relocation_or_runtime_admission",
            "original_bytes": self.original_bytes,
            "pooled_bytes": self.pooled_bytes,
            "bindings": [
                {
                    "group": b.operand.group,
                    "symbol": b.operand.symbol,
                    "path": str(b.operand.path),
                    "alignment": b.operand.alignment,
                    "representation": b.operand.representation,
                    "access": b.operand.access,
                    "canonical_symbol": b.canonical_symbol,
                    "bytes": b.bytes,
                    "sha256": b.sha256,
                }
                for b in self.bindings
            ],
            "blobs": [
                {
                    "symbol": b.symbol,
                    "path": str(b.path),
                    "alignment": b.alignment,
                    "representation": b.representation,
                    "bytes": b.bytes,
                    "sha256": b.sha256,
                }
                for b in self.blobs
            ],
        }


def plan_immutable_constant_pool(operands, *, symbol_prefix, reserved_symbols):
    """Merge only complete equal representations whose address identity is unobserved.

    Unknown/writable/identity-observing use contracts refuse, rather than being
    inferred from a readonly ELF section. Equal bytes with different provider
    representations retain distinct blobs. The caller owns all symbol rewrites
    and proves that its final linker places each shared span within target reach.
    """
    operands = tuple(operands)
    if not operands or not _identifier(symbol_prefix):
        raise ValueError("constant pool needs operands and a safe caller-selected symbol prefix")
    identities, snapshots = set(), []
    for operand in operands:
        if (
            not isinstance(operand, PoolOperand)
            or not isinstance(operand.group, str)
            or not operand.group
            or not _identifier(operand.symbol)
            or not isinstance(operand.path, Path)
            or (operand.group, operand.symbol) in identities
            or type(operand.alignment) is not int
            or operand.alignment <= 0
            or operand.alignment & (operand.alignment - 1)
            or not isinstance(operand.representation, str)
            or not operand.representation
            or operand.access != ACCESS
        ):
            raise ValueError("constant pool has duplicate, incomplete or unsafe operand contracts")
        identities.add((operand.group, operand.symbol))
        path = Path(operand.path)
        digest = _digest(path)
        snapshots.append((operand, path, path.stat().st_size, digest))
    occupied = set(reserved_symbols) | {operand.symbol for operand in operands}
    groups, bindings, blobs = {}, [], []
    cursors = {}
    for operand, path, size, digest in snapshots:
        cursor = cursors.get(operand.group, 0)
        cursor = (cursor + operand.alignment - 1) // operand.alignment * operand.alignment
        cursors[operand.group] = cursor + size
        key = operand.representation, size, digest
        if key in groups:
            index = groups[key]
            previous = blobs[index]
            if not _equal(path, previous.path):
                raise ValueError("constant pool digest equality does not establish byte equality")
            if operand.alignment > previous.alignment:
                blobs[index] = PoolBlob(
                    previous.symbol, previous.path, operand.alignment, previous.representation, size, digest
                )
        else:
            ordinal = len(blobs)
            while (symbol := symbol_prefix + str(ordinal)) in occupied:
                ordinal += 1
            occupied.add(symbol)
            index = len(blobs)
            groups[key] = index
            blobs.append(PoolBlob(symbol, path, operand.alignment, operand.representation, size, digest))
        bindings.append(PoolBinding(operand, blobs[index].symbol, size, digest))
    pooled = 0
    for blob in blobs:
        pooled = (pooled + blob.alignment - 1) // blob.alignment * blob.alignment + blob.bytes
    if blobs:
        alignment = blobs[-1].alignment
        pooled = (pooled + alignment - 1) // alignment * alignment
    # Report actual padded layouts, not just logical payload sizes.
    original = 0
    for group, cursor in cursors.items():
        alignment = next(o.alignment for o in reversed(operands) if o.group == group)
        original += (cursor + alignment - 1) // alignment * alignment
    plan = ConstantPool(tuple(bindings), tuple(blobs), original, pooled)
    plan.recheck()
    return plan

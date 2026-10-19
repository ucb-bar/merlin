"""Reuse a relocation-free constant object by reopening its complete contents.

ELF fields are public serialization, not ISA or ABI defaults. The provider must
supply a fresh reference object from its selected assembler and target options.
Only the readonly data and its exact symbol roster may differ from that object.
This proves storage identity, not packing semantics or executable admission.
"""

from __future__ import annotations

import hashlib
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConstantBlob:
    symbol: str
    path: Path
    alignment: int


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("constant object inputs must be regular files")
    return path


def _inspect(raw, section):
    def refuse():
        raise ValueError("constant object has unsupported or incomplete ELF structure")

    if len(raw) < 64 or raw[:4] != b"\x7fELF" or raw[4] not in (1, 2) or raw[5] not in (1, 2):
        refuse()
    endian = "<" if raw[5] == 1 else ">"
    wide = raw[4] == 2
    header_fmt = endian + ("HHIQQQIHHHHHH" if wide else "HHIIIIIHHHHHH")
    header = struct.unpack_from(header_fmt, raw, 16)
    kind, machine, version, entry, phoff, shoff, flags, ehsize, phsize, phnum, shsize, number, names = header
    fmt = endian + ("IIQQQQIIQQ" if wide else "IIIIIIIIII")
    if (
        kind != 1
        or version != 1
        or raw[6] != 1
        or entry
        or phoff
        or phnum
        or phsize
        or ehsize != 16 + struct.calcsize(header_fmt)
        or shsize != struct.calcsize(fmt)
        or not number
        or number > 4096
        or names >= number
        or shoff < ehsize
        or shoff + number * shsize > len(raw)
    ):
        refuse()
    rows = [struct.unpack_from(fmt, raw, shoff + i * shsize) for i in range(number)]
    if any(rows[0]):
        refuse()

    def payload(row):
        offset, size = row[4:6]
        if offset + size > len(raw) or size > 16 << 20:
            refuse()
        return raw[offset : offset + size]

    def string(table, index):
        stop = table.find(b"\0", index)
        if index >= len(table) or stop < 0:
            refuse()
        return table[index:stop].decode("ascii")

    if rows[names][1] != 3:
        refuse()
    names_table = payload(rows[names])
    named = [(string(names_table, row[0]), row) for row in rows]
    selected = [(i, row) for i, (name, row) in enumerate(named) if name == section]
    if len(selected) != 1:
        refuse()
    ordinal, constant = selected[0]
    if (
        constant[1] != 1
        or constant[2] != 2
        or constant[3]
        or constant[6]
        or constant[7]
        or constant[9]
        or constant[4] + constant[5] > len(raw)
        or not constant[8]
        or constant[8] & (constant[8] - 1)
    ):
        refuse()
    metadata = []
    tables = []
    for i, (name, row) in enumerate(named):
        if row[1] in (4, 9) and row[5]:
            raise ValueError("constant object contains relocations")
        if i != ordinal and row[2] & 2 and row[5]:
            raise ValueError("constant object contains other allocated storage or code")
        if row[1] == 2:
            tables.append(row)
        elif i and i != ordinal and row[1] != 3:
            metadata.append((name, row[1], row[2], row[3], row[8], row[9], hashlib.sha256(payload(row)).hexdigest()))
    if len(tables) != 1:
        refuse()
    table = tables[0]
    symbol_fmt = endian + ("IBBHQQ" if wide else "IIIBBH")
    symbol_size = struct.calcsize(symbol_fmt)
    if table[9] != symbol_size or table[5] % symbol_size or table[6] >= number or rows[table[6]][1] != 3:
        refuse()
    strings = payload(rows[table[6]])
    symbols = payload(table)
    exports = {}
    for offset in range(0, len(symbols), symbol_size):
        parsed = struct.unpack_from(symbol_fmt, symbols, offset)
        name, info, other, index, value, size = (
            parsed if wide else (parsed[0], parsed[3], parsed[4], parsed[5], parsed[1], parsed[2])
        )
        if not offset:
            if any(parsed):
                refuse()
            continue
        if info == 3 and not name and not other and 0 < index < number and not value and not size:
            continue  # Local section symbol, not an extra exported operand.
        name = string(strings, name)
        if info not in (16, 17) or other or index != ordinal or not name or name in exports:
            raise ValueError("constant object has extra, undefined or non-data symbols")
        exports[name] = (value, size)
    return {
        "identity": (raw[:16], machine, flags),
        "metadata": sorted(metadata),
        "offset": constant[4],
        "bytes": constant[5],
        "alignment": constant[8],
        "exports": exports,
    }


def reuse_constant_object(source, destination, blobs, *, reference_object, section):
    """Hardlink only an exact complete readonly data object, refusing every mismatch.

    ``blobs`` are freshly derived operand files in assembly order, each preceded
    by its declared alignment; the final operand receives its alignment as tail
    padding too. Fresh target reference metadata must match, all data bytes and
    zero padding are checked, and every exported symbol must bind exactly once.
    No source is rewritten, no compilation is claimed, and no cache is created.
    """
    blobs = tuple(blobs)
    source, reference = _regular(source), _regular(reference_object)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink() or not blobs or not section:
        raise ValueError("constant object reuse needs a fresh destination and complete operands")
    operands, names = [], set()
    for blob in blobs:
        if (
            not blob.symbol
            or not blob.symbol.isascii()
            or not (blob.symbol[0].isalpha() or blob.symbol[0] == "_")
            or any(not (c.isalnum() or c == "_") for c in blob.symbol)
            or blob.symbol in names
            or type(blob.alignment) is not int
            or blob.alignment <= 0
            or blob.alignment & (blob.alignment - 1)
        ):
            raise ValueError("constant object operand names or alignments are invalid")
        names.add(blob.symbol)
        path = _regular(blob.path)
        if not path.stat().st_size:
            raise ValueError("constant object operands must be nonempty")
        operands.append((blob, path.stat().st_size, _sha(path)))
    with reference.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        expected = _inspect(raw, section)
    with source.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        got = _inspect(raw, section)
        if got["identity"] != expected["identity"] or got["metadata"] != expected["metadata"]:
            raise ValueError("constant object differs from the fresh target reference")
        cursor, exports, roster = 0, {}, []

        def padding(stop):
            if stop > got["bytes"] or any(raw[got["offset"] + cursor : got["offset"] + stop]):
                raise ValueError("constant object has changed padding or extent")

        for blob, size, digest in operands:
            aligned = (cursor + blob.alignment - 1) // blob.alignment * blob.alignment
            padding(aligned)
            cursor = aligned
            if cursor + size > got["bytes"] or got["alignment"] % blob.alignment:
                raise ValueError("constant object operand extent or alignment differs")
            observed = hashlib.sha256()
            for start in range(cursor, cursor + size, 1 << 20):
                observed.update(raw[got["offset"] + start : got["offset"] + min(start + (1 << 20), cursor + size)])
            if observed.hexdigest() != digest:
                raise ValueError("constant object operand differs from freshly derived bytes")
            exports[blob.symbol] = (cursor, size)
            roster.append(
                {"symbol": blob.symbol, "offset": cursor, "bytes": size, "sha256": digest, "alignment": blob.alignment}
            )
            cursor += size
        stop = (cursor + blobs[-1].alignment - 1) // blobs[-1].alignment * blobs[-1].alignment
        padding(stop)
        if (
            stop != got["bytes"]
            or set(exports) != set(got["exports"])
            or any(
                got["exports"][name][0] != offset or got["exports"][name][1] not in (0, size)
                for name, (offset, size) in exports.items()
            )
        ):
            raise ValueError("constant object symbol membership, offsets or complete extent differ")
        object_digest = hashlib.sha256(raw).hexdigest()
    os.link(source, destination)
    if (
        not os.path.samefile(source, destination)
        or _sha(destination) != object_digest
        or any(_sha(blob.path) != digest for blob, _, digest in operands)
    ):
        raise ValueError("constant object or operands changed during storage binding")
    return {
        "schema": "merlin.constant_object_storage_reuse.v1",
        "scope": "complete_readonly_data_identity_not_packing_or_executable_admission",
        "source": str(source.resolve()),
        "sha256": object_digest,
        "bytes": destination.stat().st_size,
        "shared_inode": True,
        "reference_object_sha256": _sha(reference),
        "section": section,
        "operands": roster,
    }

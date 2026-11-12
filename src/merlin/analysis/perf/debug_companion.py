"""Admit debug-only ELF companions before attributing an existing PC census.

The caller supplies target tools, PC counts and symbolizer output. No execution,
ISA decoding, compiler invocation or timing inference occurs here. The current
binary reader deliberately supports ordinary little-endian ELF64 only.
"""

from __future__ import annotations

import hashlib
import struct
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class _Section:
    name: str
    kind: int
    flags: int
    address: int
    size: int
    link: int
    info: int
    alignment: int
    entry_size: int
    data: bytes


def _slice(data: bytes, offset: int, size: int) -> bytes:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ValueError("ELF range outside file")
    return data[offset : offset + size]


def _string(data: bytes, offset: int) -> str:
    if offset >= len(data) or b"\0" not in data[offset:]:
        raise ValueError("invalid ELF string")
    return data[offset : data.index(b"\0", offset)].decode("utf-8", errors="strict")


def _read(data: bytes):
    if len(data) < 64 or data[:7] != b"\x7fELF\x02\x01\x01":
        raise ValueError("ordinary little-endian ELF64 required")
    header = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
    kind, machine, version, entry, _, offset, flags, size, _, _, stride, count, names = header
    if version != 1 or size != 64 or stride != 64 or not count or not 0 < names < count:
        raise ValueError("unsupported ELF header or extended section numbering")
    rows = [struct.unpack("<IIQQQQIIQQ", _slice(data, offset + i * stride, stride)) for i in range(count)]
    table = _slice(data, rows[names][4], rows[names][5])
    sections = []
    for row in rows:
        name, typ, section_flags, address, start, length, link, info, align, entsize = row
        body = b"" if typ == 8 else _slice(data, start, length)
        sections.append(
            _Section(_string(table, name), typ, section_flags, address, length, link, info, align, entsize, body)
        )
    return (kind, machine, entry, flags), sections


def _allocated(sections):
    result = {}
    for section in sections:
        if section.flags & 2:
            if section.name in result:
                raise ValueError("duplicate allocated section name")
            result[section.name] = (
                section.kind,
                section.flags,
                section.address,
                section.size,
                section.alignment,
                section.data,
            )
    if not result:
        raise ValueError("ELF has no allocated sections")
    return result


def _relocations(sections):
    result = []
    for reloc in sections:
        if reloc.kind not in (4, 9):
            continue
        if reloc.info >= len(sections) or reloc.link >= len(sections):
            raise ValueError("invalid relocation section indices")
        destination = sections[reloc.info]
        if not destination.flags & 2:
            continue
        symbols = sections[reloc.link]
        if symbols.kind not in (2, 11) or symbols.entry_size != 24 or symbols.link >= len(sections):
            raise ValueError("invalid relocation symbol table")
        if sections[symbols.link].kind != 3 or len(symbols.data) % 24:
            raise ValueError("invalid relocation symbol strings or entries")
        strings = sections[symbols.link].data
        stride = 24 if reloc.kind == 4 else 16
        if reloc.entry_size != stride or len(reloc.data) % stride:
            raise ValueError("invalid relocation entries")
        for offset in range(0, len(reloc.data), stride):
            place, info = struct.unpack_from("<QQ", reloc.data, offset)
            addend = struct.unpack_from("<q", reloc.data, offset + 16)[0] if stride == 24 else None
            symbol_index, relocation_type = info >> 32, info & 0xFFFFFFFF
            raw = _slice(symbols.data, symbol_index * 24, 24)
            name, symbol_info, other, section_index, value, size = struct.unpack("<IBBHQQ", raw)
            if len(sections) <= section_index < 0xFF00:
                raise ValueError("symbol section index outside section table")
            owner = sections[section_index].name if section_index < len(sections) else section_index
            if section_index == 0xFFFF:
                raise ValueError("extended symbol section indices unsupported")
            symbol = (_string(strings, name), symbol_info, other, owner, value, size)
            result.append((destination.name, place, relocation_type, addend, symbol))
    return Counter(result)


def verify_debug_companion(control: bytes, companion: bytes, *, relocatable: bool = False) -> dict:
    """Require identical allocated data/address/flags and normalized relocations.

    Relocatable objects additionally require ELF ET_REL. Final images require
    ET_EXEC or ET_DYN. This does not establish source correctness, original
    numerical gates or equivalence of an unrelated compiler command.
    """
    if type(relocatable) is not bool:
        raise ValueError("explicit boolean object policy required")
    a_header, a = _read(control)
    b_header, b = _read(companion)
    if a_header != b_header or a_header[0] not in ((1,) if relocatable else (2, 3)):
        raise ValueError("ELF identity differs or wrong file kind")
    if _allocated(a) != _allocated(b):
        raise ValueError("allocated section bytes, address or attributes differ")
    if _relocations(a) != _relocations(b):
        raise ValueError("allocated-section relocations differ")
    return {
        "control_sha256": hashlib.sha256(control).hexdigest(),
        "companion_sha256": hashlib.sha256(companion).hexdigest(),
        "allocated_sections": len(_allocated(a)),
        "relocations": sum(_relocations(a).values()),
        "relocatable": relocatable,
    }


def attribute_symbolized_pcs(counts: dict[int, int], records: list[dict]) -> dict:
    """Aggregate LLVM symbolizer JSON records with complete address coverage.

    Every PC is accounted once, including records without source lines. Inline
    frames are retained as a stack; innermost aggregation is not a call timer.
    The caller must first verify the exact binary companion and census identity.
    """
    if any(type(pc) is not int or pc < 0 or type(n) is not int or n < 0 for pc, n in counts.items()):
        raise ValueError("nonnegative integer PCs and counts required")
    seen = set()
    functions, lines, stacks = Counter(), Counter(), Counter()
    for record in records:
        pc = int(record["Address"], 0)
        if pc not in counts or pc in seen:
            raise ValueError("unexpected or duplicate symbolized PC")
        seen.add(pc)
        frames = record["Symbol"]
        stack = tuple(
            (f.get("FunctionName", ""), f.get("FileName", ""), f.get("Line", 0), f.get("Column", 0)) for f in frames
        )
        if any(
            type(line) is not int or line < 0 or type(column) is not int or column < 0 for _, _, line, column in stack
        ):
            raise ValueError("invalid source coordinate")
        function, filename, line, _ = stack[0] if stack else ("", "", 0, 0)
        functions[function] += counts[pc]
        lines[(filename, line)] += counts[pc]
        stacks[stack] += counts[pc]
    if seen != counts.keys():
        raise ValueError("incomplete symbolizer coverage")
    return {
        "total": sum(counts.values()),
        "functions": dict(functions),
        "lines": dict(lines),
        "inline_stacks": dict(stacks),
    }

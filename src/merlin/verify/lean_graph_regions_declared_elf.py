"""Offline structural ELF metadata; never loads or executes the inspected file.

ELF32/ELF64, either byte order, program headers and dynamic string metadata only.
Extended program-header numbering and ambiguous virtual-address maps refuse.
Tag numbers here are ELF ABI format identifiers, not target ISA encodings.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


class UnsupportedElf(ValueError):
    """Insufficient/unambiguous offline format support; never a passing plan."""


@dataclass(frozen=True)
class Segment:
    kind: int
    offset: int
    address: int
    file_size: int
    memory_size: int


@dataclass(frozen=True)
class Elf:
    bits: int
    byte_order: str
    machine: int
    kind: int
    interpreter: str | None
    needed: tuple[str, ...]
    soname: str | None
    rpath: str | None
    runpath: str | None
    dynamic_tags: tuple[int, ...]
    unsupported_load_tags: tuple[int, ...]
    os_abi: int
    abi_version: int
    header_flags: int
    dynamic_entries: tuple[tuple[int, int], ...]


def region(data: bytes, offset: int, size: int) -> bytes:
    if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
        raise UnsupportedElf("ELF table/string lies outside file bytes")
    return data[offset : offset + size]


def cstring(data: bytes, offset: int, limit: int) -> str:
    raw = region(data, offset, limit)
    end = raw.find(b"\0")
    if end < 0:
        raise UnsupportedElf("unterminated ELF string")
    try:
        value = raw[:end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise UnsupportedElf("non-UTF8 ELF path/name") from error
    if not value:
        raise UnsupportedElf("empty ELF dependency/path string")
    return value


def parse(data: bytes) -> Elf:
    if type(data) is not bytes or len(data) < 16 or data[:4] != b"\x7fELF":
        raise UnsupportedElf("ELF identification bytes missing")
    if data[4] not in (1, 2) or data[5] not in (1, 2) or data[6] != 1:
        raise UnsupportedElf("unsupported ELF class, byte order or version")
    wide = data[4] == 2
    endian = "<" if data[5] == 1 else ">"
    header = struct.Struct(endian + ("HHIQQQIHHHHHH" if wide else "HHIIIIIHHHHHH"))
    values = header.unpack(region(data, 16, header.size))
    kind, machine, version = values[:3]
    offset, entry_size, count = values[4], values[8], values[9]
    if version != 1 or kind not in (2, 3) or count == 65535:
        raise UnsupportedElf("unsupported ELF type/version or extended program headers")
    if values[7] != 16 + header.size or offset < 16 + header.size:
        raise UnsupportedElf("unexpected ELF header extent/program-table position")
    program = struct.Struct(endian + ("IIQQQQQQ" if wide else "IIIIIIII"))
    if entry_size != program.size or count == 0:
        raise UnsupportedElf("unexpected or missing program-header table")
    region(data, offset, count * entry_size)
    segments = []
    for index in range(count):
        row = program.unpack(region(data, offset + index * entry_size, entry_size))
        if wide:
            segment = Segment(row[0], row[2], row[3], row[5], row[6])
        else:
            segment = Segment(row[0], row[1], row[2], row[4], row[5])
        region(data, segment.offset, segment.file_size)
        if segment.kind == 1 and segment.memory_size < segment.file_size:
            raise UnsupportedElf("PT_LOAD memory extent is smaller than file extent")
        segments.append(segment)
    interpreters = [segment for segment in segments if segment.kind == 3]
    dynamics = [segment for segment in segments if segment.kind == 2]
    if len(interpreters) > 1 or len(dynamics) > 1:
        raise UnsupportedElf("multiple PT_INTERP/PT_DYNAMIC segments")
    interpreter = None
    if interpreters:
        item = interpreters[0]
        interpreter = cstring(data, item.offset, item.file_size)
        if not interpreter.startswith("/"):
            raise UnsupportedElf("relative ELF interpreter")
    if not dynamics:
        return Elf(
            64 if wide else 32,
            "little" if endian == "<" else "big",
            machine,
            kind,
            interpreter,
            (),
            None,
            None,
            None,
            (),
            (),
            data[7],
            data[8],
            values[6],
            (),
        )
    dynamic = struct.Struct(endian + ("qQ" if wide else "iI"))
    section = dynamics[0]
    if section.file_size % dynamic.size:
        raise UnsupportedElf("misaligned dynamic table")
    entries = []
    terminated = False
    for position in range(0, section.file_size, dynamic.size):
        tag, value = dynamic.unpack(region(data, section.offset + position, dynamic.size))
        if tag == 0:
            terminated = True
            break
        entries.append((tag, value))
    if not terminated:
        raise UnsupportedElf("dynamic table has no DT_NULL")

    def single(tag):
        found = [value for key, value in entries if key == tag]
        if len(found) > 1:
            raise UnsupportedElf("ambiguous duplicate singleton dynamic tag: " + str(tag))
        return found[0] if found else None

    # DT_STRTAB and DT_STRSZ are required for every supported dynamic string.
    address, size = single(5), single(10)
    # CONFIG, DEPAUDIT, AUDIT, AUXILIARY and FILTER can select extra inputs.
    unsupported_tags = {0x6FFFFEFA, 0x6FFFFEFB, 0x6FFFFEFC, 0x7FFFFFFD, 0x7FFFFFFF}
    string_tags = {1, 14, 15, 29} | unsupported_tags
    if any(tag in string_tags for tag, _ in entries) and (address is None or size is None):
        raise UnsupportedElf("missing dynamic string table")
    if (address is None) != (size is None):
        raise UnsupportedElf("incomplete dynamic string-table extent")
    table = b""
    if address is not None and size is not None:
        matches = [
            segment.offset + address - segment.address
            for segment in segments
            if segment.kind == 1
            and segment.address <= address
            and address - segment.address <= segment.file_size
            and size <= segment.file_size - (address - segment.address)
        ]
        if len(matches) != 1:
            raise UnsupportedElf("ambiguous or absent PT_LOAD mapping for dynamic strings")
        table = region(data, matches[0], size)

    def text(tag):
        value = single(tag)
        return None if value is None else cstring(table, value, len(table) - value)

    needed = tuple(cstring(table, value, len(table) - value) for tag, value in entries if tag == 1)
    if len(needed) != len(set(needed)):
        raise UnsupportedElf("duplicate DT_NEEDED entries are outside this planner")
    extra = tuple(tag for tag, _ in entries if tag in unsupported_tags)
    return Elf(
        64 if wide else 32,
        "little" if endian == "<" else "big",
        machine,
        kind,
        interpreter,
        needed,
        text(14),
        text(15),
        text(29),
        tuple(tag for tag, _ in entries),
        extra,
        data[7],
        data[8],
        values[6],
        tuple(entries),
    )

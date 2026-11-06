"""Shared raw-program admission, startup, linker layout and ELF auditing."""

import re
import struct
from pathlib import Path
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from atlas_host_layout import ATLAS_DRAM_BASE, ATLAS_IMEM_BYTES

GRAPH_PROGRAM_WORDS = ATLAS_IMEM_BYTES // 4 - 5
_UNDECLARED_WORDS = object()

TEMPLATES = checked(Path(__file__).absolute().parents[1] / "runtime/host_templates")
TEMPLATE_BYTES = {name: checked(TEMPLATES / name).read_bytes() for name in ("smol_start.S", "smol_link.ld")}
START = TEMPLATE_BYTES["smol_start.S"].decode()
LINK = TEMPLATE_BYTES["smol_link.ld"].decode()
LINK = re.sub(r"(\.atlas_dram )0x[0-9a-fA-F]+", lambda match: match[1] + hex(ATLAS_DRAM_BASE), LINK)
LINK = re.sub(r"(\.atlas_dram[^{}]*\{.*?\. \+= )\d+;", r"\g<1>@DRAM_BYTES@;", LINK, flags=re.S)


def wrap(raw):
    if len(raw) % 4 or not 1 <= len(raw) // 4 <= ATLAS_IMEM_BYTES // 4 - 5:
        raise ValueError("program extent does not fit raw marker ABI")
    words = struct.unpack("<" + "I" * (len(raw) // 4), raw)
    if words[-1] != 0x73 or 0x73 in words[:-1] or 0x00100073 in words:
        raise ValueError("program must end in its only ECALL")
    for word in words[:-1]:
        if word & 127 == 0x73 and word >> 20 in (0xC10, 0xC11):
            kind, source = (word >> 12) & 7, (word >> 15) & 31
            if kind in (1, 5) or kind and source:
                raise ValueError("raw program writes a reserved mailbox")
    return raw[:-4] + struct.pack("<6I", 0x13, 0xC1102FF3, 0x13, 0xC10F9073, 0x13, 0x73)


def validate_graph_program(raw, declared_words=_UNDECLARED_WORDS):
    """Admit actual raw bytes to the graph runtime's bounded copy buffer."""
    wrap(raw)  # Preserve structural, mailbox and hardware IMEM admission.
    actual_words = len(raw) // 4
    if actual_words > GRAPH_PROGRAM_WORDS:
        raise ValueError("raw graph program exceeds runtime program buffer")
    if declared_words is not _UNDECLARED_WORDS and (type(declared_words) is not int or declared_words != actual_words):
        raise ValueError("declared graph program words differ from actual bytes")
    return actual_words


def elf_sections(raw):
    if raw[:7] != b"\x7fELF\x02\x01\x01" or struct.unpack_from("<HH", raw, 16) != (2, 243):
        raise ValueError("require executable ELF64 little-endian RISC-V")
    if struct.unpack_from("<I", raw, 48)[0] & 6 != 4:
        raise ValueError("require double-precision hardware floating ABI")
    entry = struct.unpack_from("<Q", raw, 24)[0]
    shoff = struct.unpack_from("<Q", raw, 40)[0]
    shsize, count, strindex = struct.unpack_from("<HHH", raw, 58)
    rows = [struct.unpack_from("<IIQQQQIIQQ", raw, shoff + i * shsize) for i in range(count)]
    namesrow = rows[strindex]
    names = raw[namesrow[4] : namesrow[4] + namesrow[5]]
    sections = {names[row[0] :].split(b"\0", 1)[0].decode(): row for row in rows}
    symbols = {}
    sym = sections[".symtab"]
    stringsrow = rows[sym[6]]
    strings = raw[stringsrow[4] : stringsrow[4] + stringsrow[5]]
    for offset in range(sym[4], sym[4] + sym[5], sym[9]):
        name, info, other, index, value, size = struct.unpack_from("<IBBHQQ", raw, offset)
        symbols[strings[name:].split(b"\0", 1)[0].decode()] = (value, size, index)
    return entry, sections, symbols


def audit_elf(raw, assets, dram_bytes, *, arena_base=ATLAS_DRAM_BASE):
    entry, sections, symbols = elf_sections(raw)
    if entry != 0x80000000 or symbols["__stack_top"][0] - symbols["__stack_bottom"][0] != 65536:
        raise ValueError("entry/stack differs")
    for name in ("atlas_workload_mailbox", "tohost", "fromhost"):
        if symbols[name][0] % 64:
            raise ValueError("mailbox/HTIF alignment differs")
    dram = sections[".atlas_dram"]
    if dram[1] != 8 or dram[3] != arena_base or dram[5] != dram_bytes:
        raise ValueError("NOLOAD DMA arena reservation differs")
    for label, data in assets.items():
        address, size, index = symbols[label]
        # Resolve by section index without depending on dict ordering.
        shoff = struct.unpack_from("<Q", raw, 40)[0]
        shsize = struct.unpack_from("<H", raw, 58)[0]
        section = struct.unpack_from("<IIQQQQIIQQ", raw, shoff + index * shsize)
        if size != len(data) or section[2] & 1:
            raise ValueError("embedded asset extent/writability differs")
        offset = section[4] + address - section[3]
        if raw[offset : offset + size] != data:
            raise ValueError("ELF embedded asset differs: " + label)
    phoff = struct.unpack_from("<Q", raw, 32)[0]
    phsize, phcount = struct.unpack_from("<HH", raw, 54)
    segments = []
    for i in range(phcount):
        kind, flags, off, va, pa, filesz, memsz, align = struct.unpack_from("<IIQQQQQQ", raw, phoff + i * phsize)
        if kind == 1:
            if (filesz > memsz or off + filesz > len(raw) or va != pa or va + memsz > 1 << 64 or flags & 3 == 3 or align
                and (align & (align - 1) or va % align != off % align)):
                raise ValueError("invalid physical load segment")
            if not (0x80000000 <= va and va + memsz <= 0x90000000 or va == arena_base and memsz == dram_bytes
                and filesz == 0):
                raise ValueError("load segment outside declared host/device DRAM")
            segments.append(dict(address=va, file_bytes=filesz, memory_bytes=memsz, flags=flags))
    for i, a in enumerate(segments):
        for b in segments[:i]:
            if a["address"] < b["address"] + b["memory_bytes"] and b["address"] < a["address"] + a["memory_bytes"]:
                raise ValueError("load segments overlap")
    for label, data in assets.items():
        address = symbols[label][0]
        if not any(segment["address"] <= address and address + len(data) <= segment["address"] + segment["file_bytes"]
            and not segment["flags"] & 2 for segment in segments):
            raise ValueError("immutable asset lacks a read-only file-backed load segment")
    return dict(entry=entry, stack_bytes=65536, dram_bytes=dram_bytes, segments=segments,
        symbols={name: symbols[name][0] for name in ("atlas_workload_mailbox", "atlas_resnet_state", "tohost", "fromhost",
        "__stack_bottom", "__stack_top")}, embedded_assets_exact=True)

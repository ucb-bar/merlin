"""Compare actual allocated object sections across a symbol-only rewrite."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path


def allocated(path: Path):
    """Read ELF64 section tables structurally, including NOBITS extents."""
    with path.open("rb") as stream:
        header = stream.read(64)
        if header[:6] != b"\x7fELF\x02\x01":
            raise ValueError("chain runtime requires little-endian ELF64 objects")
        table = struct.unpack_from("<Q", header, 40)[0]
        size, count, strings = struct.unpack_from("<HHH", header, 58)
        if size != 64 or not 0 < count < 65535 or strings >= count:
            raise ValueError("unsupported extended ELF section roster")
        stream.seek(table)
        sections = [struct.unpack("<IIQQQQIIQQ", stream.read(64)) for _ in range(count)]
        names = sections[strings]
        stream.seek(names[4])
        names = stream.read(names[5])
        result = {}
        for row in sections:
            name, kind, flags, _, offset, length, *_ = row
            if not flags & 2:
                continue
            label = names[name:].split(b"\0", 1)[0].decode("ascii")
            if label in result:
                raise ValueError("duplicate allocated object section")
            digest = None
            if kind != 8:  # ELF SHT_NOBITS has no stored bytes.
                stream.seek(offset)
                remaining, hasher = length, hashlib.sha256()
                while remaining:
                    block = stream.read(min(remaining, 1024 * 1024))
                    if not block:
                        raise ValueError("truncated allocated object section")
                    hasher.update(block)
                    remaining -= len(block)
                digest = hasher.hexdigest()
            result[label] = {"kind": kind, "flags": flags, "bytes": length, "sha256": digest}
        return result


def unchanged(source, product, *, weight_section=None):
    before, after = allocated(source), allocated(product)
    if weight_section is not None:
        if ".data" not in before:
            raise ValueError("weights object has no selected data section")
        # objcopy's explicit section flags preserve ALLOC/WRITE and initialized data.
        before[weight_section] = before.pop(".data")
    if before != after:
        raise ValueError("symbol namespace changed actual allocated instructions/data/storage")
    return {"status": "allocated_sections_unchanged", "sections": after}


def symbol_bytes(path, symbol):
    """Read one defined ELF64 symbol through its actual section index/extent."""
    with path.open("rb") as stream:
        header = stream.read(64)
        if header[:6] != b"\x7fELF\x02\x01":
            raise ValueError("unsupported object ABI for table layout proof")
        table = struct.unpack_from("<Q", header, 40)[0]
        stride, count, _ = struct.unpack_from("<HHH", header, 58)
        if stride != 64 or not 0 < count < 65535:
            raise ValueError("unsupported object section roster")
        stream.seek(table)
        sections = [struct.unpack("<IIQQQQIIQQ", stream.read(64)) for _ in range(count)]
        found = []
        for section in sections:
            if section[1] != 2:  # SHT_SYMTAB
                continue
            if section[9] != 24 or section[6] >= len(sections) or section[5] % 24:
                raise ValueError("invalid object symbol table")
            strings = sections[section[6]]
            stream.seek(strings[4])
            names = stream.read(strings[5])
            stream.seek(section[4])
            symbols = stream.read(section[5])
            for at in range(0, len(symbols), 24):
                name, _, _, index, offset, length = struct.unpack_from("<IBBHQQ", symbols, at)
                if names[name:].split(b"\0", 1)[0].decode("utf-8") != symbol:
                    continue
                if not 0 < index < len(sections) or not length:
                    raise ValueError("selected table symbol has no stored definition")
                stored = sections[index]
                if stored[1] == 8 or offset + length > stored[5]:
                    raise ValueError("selected table symbol exceeds its section")
                stream.seek(stored[4] + offset)
                found.append(stream.read(length))
        if len(found) != 1:
            raise ValueError("selected table symbol is absent or ambiguous")
        return found[0]

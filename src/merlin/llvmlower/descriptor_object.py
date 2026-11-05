"""Bounded compiler-emitted constant layout data, parsed without ISA logic.

ELF identifiers/header formats below are public gABI serialization vocabulary,
not target address, instruction, pointer-width or physical storage defaults.
"""

import struct

SECTION = ".merlin_descriptor_layout"


def layout_words(raw, *, count):
    """Require one complete readonly, relocation-free i64 constant array."""
    if type(count) is not int or count <= 0 or len(raw) < 64 or raw[:4] != b"\x7fELF":
        raise ValueError("descriptor layout needs a complete bounded ELF constant product")
    if raw[5] not in {1, 2} or raw[6] != 1:
        raise ValueError("descriptor layout ELF has unsupported byte encoding/version")
    order, endian = ("little", "<") if raw[5] == 1 else ("big", ">")
    if struct.unpack_from(endian + "H", raw, 16)[0] != 1:
        raise ValueError("descriptor layout requires its actual relocatable compiler object")
    if raw[4] == 2:
        shoff = struct.unpack_from(endian + "Q", raw, 40)[0]
        entsize, number, names = struct.unpack_from(endian + "HHH", raw, 58)
        fmt = endian + "IIQQQQIIQQ"
    elif raw[4] == 1:
        shoff = struct.unpack_from(endian + "I", raw, 32)[0]
        entsize, number, names = struct.unpack_from(endian + "HHH", raw, 46)
        fmt = endian + "IIIIIIIIII"
    else:
        raise ValueError("descriptor layout ELF class is unsupported")
    if (
        entsize != struct.calcsize(fmt)
        or not shoff
        or not number
        or names >= number
        or shoff + entsize * number > len(raw)
    ):
        raise ValueError("descriptor layout ELF section roster is missing or incomplete")
    rows = [struct.unpack_from(fmt, raw, shoff + ordinal * entsize) for ordinal in range(number)]

    def payload(row):
        start, size = row[4:6]
        if start + size > len(raw):
            raise ValueError("descriptor layout ELF section escapes the complete object")
        return raw[start : start + size]

    if rows[names][1] != 3:
        raise ValueError("descriptor layout ELF has no section-name string table")
    table, selected = payload(rows[names]), []
    for ordinal, row in enumerate(rows):
        index = row[0]
        stop = table.find(b"\0", index)
        if index >= len(table) or stop < 0:
            raise ValueError("descriptor layout ELF has an incomplete section name")
        if table[index:stop] == SECTION.encode():
            selected.append((ordinal, row))
    if len(selected) != 1:
        raise ValueError("descriptor layout ELF lacks its unique complete constant roster")
    ordinal, row = selected[0]
    if row[1] != 1 or row[2] & 5 or row[5] != count * 8:
        raise ValueError("descriptor layout constants are writable/executable or have changed membership")
    if any(entry[1] in {4, 9} and entry[7] == ordinal for entry in rows):
        raise ValueError("descriptor layout constants carry unresolved relocation semantics")
    return tuple(int.from_bytes(payload(row)[offset : offset + 8], order) for offset in range(0, count * 8, 8)), order

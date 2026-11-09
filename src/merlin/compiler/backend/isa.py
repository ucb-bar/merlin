"""Conservative RISC-V ISA decomposition for host implementation admission.

This preserves the existing package/ELF admission grammar; it is not a complete ISA
specification parser. Hardware facts and package qualification remain caller-owned.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def split_isa(value: str) -> tuple[str, set[str]]:
    """Expand the standard base letters for a conservative package/host ISA check."""
    head, *tail = value.lower().split("_")
    xlen = head[:4]
    if xlen not in {"rv32", "rv64"}:
        raise ValueError(f"unrecognized RISC-V ISA {value!r}")
    position = 4
    while position < len(head) and "a" <= head[position] <= "y":
        position += 1
    if position == 4:
        raise ValueError(f"unrecognized RISC-V ISA {value!r}")
    base = set(head[4:position])
    extensions = set(tail)
    if position < len(head):
        suffix = head[position:]
        if (
            not suffix.startswith("z")
            or len(suffix) == 1
            or not all("a" <= letter <= "z" or "0" <= letter <= "9" for letter in suffix[1:])
        ):
            raise ValueError(f"unrecognized RISC-V ISA {value!r}")
        extensions.add(suffix)
    if "g" in base:
        base.remove("g")
        base.update("imafd")
        extensions.update({"zicsr", "zifencei"})
    return xlen, base | extensions


def _dts_name_char(letter: str) -> bool:
    return letter.isalnum() or letter in "#,._+?@-/\\"


def _skip_dts_trivia(source: str, position: int) -> int:
    """Skip whitespace and both DTS comment forms; an open comment consumes EOF."""
    while position < len(source):
        if source[position].isspace():
            position += 1
        elif source.startswith("//", position):
            end = source.find("\n", position + 2)
            position = len(source) if end < 0 else end + 1
        elif source.startswith("/*", position):
            end = source.find("*/", position + 2)
            position = len(source) if end < 0 else end + 2
        else:
            break
    return position


def _dts_quoted_value(source: str, position: int) -> tuple[str | None, int]:
    """Return raw quoted bytes and the next position, skipping escaped quotes."""
    start = position + 1
    position = start
    while position < len(source):
        if source[position] == "\\":
            position += 2
        elif source[position] == '"':
            return source[start:position], position + 1
        else:
            position += 1
    return None, len(source)


def dts_string_values(source: str, property_name: str) -> list[str]:
    """Read complete DTS property tokens with one quoted value, outside comments/strings.

    The selected DTS is separately bound by its board-owned digest. This is a
    lexical reader for the host ISA/CPU declarations, not a full DTS parser.
    """
    if not property_name:
        raise ValueError("DTS property name must not be empty")
    values: list[str] = []
    position = 0
    while position < len(source):
        position = _skip_dts_trivia(source, position)
        if position >= len(source):
            break
        if source[position] == '"':
            _, position = _dts_quoted_value(source, position)
            continue
        if not _dts_name_char(source[position]):
            position += 1
            continue
        start = position
        while position < len(source) and _dts_name_char(source[position]):
            position += 1
        if source[start:position] != property_name:
            continue
        assignment = _skip_dts_trivia(source, position)
        if assignment >= len(source) or source[assignment] != "=":
            continue
        value_start = _skip_dts_trivia(source, assignment + 1)
        if value_start >= len(source) or source[value_start] != '"':
            continue
        value, position = _dts_quoted_value(source, value_start)
        if value:
            values.append(value)
    return values


def require_host_isa(cflags: list[str], host_isa: str) -> None:
    """Refuse a codegen package whose required instructions exceed the selected host ISA."""
    marches = [flag.removeprefix("-march=") for flag in cflags if flag.startswith("-march=")]
    if len(marches) != 1:
        raise ValueError("host package must declare exactly one -march for ISA compatibility")
    package_xlen, required = split_isa(marches[0])
    host_xlen, available = split_isa(host_isa)
    missing = sorted(required - available)
    if package_xlen != host_xlen or missing:
        raise ValueError(
            f"host package -march={marches[0]} is incompatible with selected host ISA {host_isa}: "
            f"missing {missing or ['matching XLEN']}"
        )


def require_host_isa_dts(cflags: list[str], dts: str | Path, *, expected_sha256: str | None = None) -> list[str]:
    """Check every CPU ISA in one DTS, optionally bound to a board-owned digest."""
    source = Path(dts)
    if not source.is_file():
        raise ValueError(f"selected host device tree is unavailable: {source}")
    data = source.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("selected host DTS bytes differ from the board's pinned SHA256")
    isas = dts_string_values(data.decode("utf-8"), "riscv,isa")
    if not isas:
        raise ValueError(f"selected host device tree has no riscv,isa declaration: {source}")
    for isa in isas:
        require_host_isa(cflags, isa)
    return isas


def implementation_functions():
    """Actual host ISA/selected-DTS parser and admission bindings for source custody."""
    return (
        implementation_functions,
        split_isa,
        _dts_name_char,
        _skip_dts_trivia,
        _dts_quoted_value,
        dts_string_values,
        require_host_isa,
        require_host_isa_dts,
    )

"""Structural readers for C headers, device trees, ELF segments and CSR accesses."""

from __future__ import annotations

import struct
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SourceReadError(ValueError):
    """A source file holds a form this reader does not parse, or a value it cannot evaluate."""


# --------------------------------------------------------------------------------------------- C headers


@dataclass(frozen=True)
class Header:
    """The object-like and function-like ``#define`` and the ``typedef`` declarations of one C header."""

    defines: dict[str, str]
    typedefs: dict[str, str]

    def integer(self, name: str) -> int | None:
        """The value of ``#define name <integer>``; ``None`` when absent or not a plain integer."""
        text = self.defines.get(name)
        if text is None:
            return None
        text = text.strip()
        while text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        while text and text[-1] in "uUlL":
            text = text[:-1]
        try:
            return int(text, 0)
        except ValueError:
            return None

    def c_type(self, name: str) -> dict[str, Any] | None:
        """``{kind, signed, bits}`` of a typedef chain ending in an exact-width integer or a float type."""
        seen = set()
        text = self.typedefs.get(name)
        while text is not None and text in self.typedefs and text not in seen:
            seen.add(text)
            text = self.typedefs[text]
        return None if text is None else exact_width_type(text)


def exact_width_type(text: str) -> dict[str, Any]:
    """``int8_t`` -> ``{kind: int, signed: True, bits: 8}``; ``float``/``double`` -> ``{kind: float}``."""
    words = text.split()
    if len(words) == 1 and words[0] in ("float", "double"):
        return {"kind": "float", "c_type": words[0]}
    if len(words) != 1 or not words[0].endswith("_t"):
        return {"kind": "unknown", "c_type": text}
    stem = words[0][: -len("_t")]
    signed = not stem.startswith("u")
    stem = stem if signed else stem[1:]
    if stem.startswith("int") and stem[3:].isdigit():
        return {"kind": "int", "signed": signed, "bits": int(stem[3:]), "c_type": words[0]}
    return {"kind": "unknown", "c_type": text}


def _strip_c_comments(text: str) -> str:
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            out.append(" ")
            i = n if end < 0 else end + 2
        elif text[i] in "\"'":
            quote, j = text[i], i + 1
            while j < n and text[j] != quote:
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def read_header(text: str) -> Header:
    """Parse ``#define`` (continuation lines joined) and single-line ``typedef`` declarations."""
    lines: list[str] = []
    pending = ""
    for raw in _strip_c_comments(text).splitlines():
        if raw.rstrip().endswith("\\"):
            pending += raw.rstrip()[:-1] + " "
            continue
        lines.append(pending + raw)
        pending = ""
    if pending:
        lines.append(pending)
    defines: dict[str, str] = {}
    typedefs: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            directive = stripped[1:].lstrip()
            keyword, _, rest = directive.partition(" ")
            if keyword != "define" or not rest.strip():
                continue
            rest = rest.strip()
            end = 0
            while end < len(rest) and (rest[end].isalnum() or rest[end] == "_"):
                end += 1
            name, value = rest[:end], rest[end:]
            if value.startswith("("):  # function-like: the parameter list is part of its definition
                close = value.find(")")
                value = value[: close + 1] + " " + value[close + 1 :]
            defines[name] = " ".join(value.split())
        elif stripped.startswith("typedef ") and stripped.endswith(";") and "(" not in stripped:
            words = stripped[: -len(";")].split()
            if len(words) >= 3:
                typedefs[words[-1]] = " ".join(words[1:-1])
    return Header(defines=defines, typedefs=typedefs)


# ----------------------------------------------------------------------------------------------- device trees


@dataclass
class DtsNode:
    name: str
    labels: list[str] = field(default_factory=list)
    props: dict[str, list[Any]] = field(default_factory=dict)
    children: list[DtsNode] = field(default_factory=list)

    def walk(self) -> Iterator[tuple[DtsNode | None, DtsNode]]:
        """``(parent, node)`` for this node and every descendant (the root's parent is ``None``)."""
        stack: list[tuple[DtsNode | None, DtsNode]] = [(None, self)]
        while stack:
            parent, node = stack.pop()
            yield parent, node
            stack.extend((node, child) for child in reversed(node.children))

    def strings(self, prop: str) -> list[str]:
        return [v for v in self.props.get(prop, []) if isinstance(v, str)]

    def cells(self, prop: str) -> list[int | str]:
        out: list[int | str] = []
        for value in self.props.get(prop, []):
            if isinstance(value, tuple) and value and value[0] == "cells":
                out.extend(value[1])
        return out


_DTS_NAME = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,._+-#@?/")


def _dts_tokens(text: str) -> Iterator[tuple[str, Any]]:
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
        elif c == '"':
            j, chars = i + 1, []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    chars.append(text[j + 1])
                    j += 2
                else:
                    chars.append(text[j])
                    j += 1
            yield "str", "".join(chars)
            i = j + 1
        elif c == "<":
            end = text.find(">", i)
            if end < 0:
                raise SourceReadError("unterminated cell list in device tree")
            values: list[int | str] = []
            for word in text[i + 1 : end].split():
                if word.startswith("&"):
                    values.append(word)
                else:
                    try:
                        values.append(int(word, 0))
                    except ValueError as error:
                        raise SourceReadError(f"device tree cell {word!r} is not an integer") from error
            yield "cells", values
            i = end + 1
        elif c == "[":
            end = text.find("]", i)
            yield "bytes", "".join(text[i + 1 : end].split())
            i = end + 1
        elif c in "{};=,":
            yield c, c
            i += 1
        elif c == "&":
            j = i + 1
            while j < n and text[j] in _DTS_NAME:
                j += 1
            yield "ref", text[i:j]
            i = j
        elif c == "/" and i + 1 < n and text[i + 1].isalpha() and text.find("/", i + 1) > i + 1:
            end = text.find("/", i + 1)
            yield "directive", text[i : end + 1]
            i = end + 1
        elif c in _DTS_NAME:
            j = i
            while j < n and text[j] in _DTS_NAME:
                j += 1
            if j < n and text[j] == ":":
                yield "label", text[i:j]
                i = j + 1
            else:
                yield "name", text[i:j]
                i = j
        else:
            raise SourceReadError(f"unexpected character {c!r} in device tree")


def read_dts(text: str) -> DtsNode:
    """The root node of a device-tree source (``/dts-v1/;`` form, one root, no includes)."""
    try:
        return _read_dts(list(_dts_tokens(text)))
    except IndexError as error:
        raise SourceReadError("device tree ends inside a node or property") from error


def _read_dts(tokens: list[tuple[str, Any]]) -> DtsNode:
    root = DtsNode(name="/")
    position = 0

    def body(node: DtsNode, at: int) -> int:
        while at < len(tokens):
            kind, value = tokens[at]
            if kind == "}":
                return at + 1
            labels = []
            while kind == "label":
                labels.append(value)
                at += 1
                kind, value = tokens[at]
            if kind != "name":
                raise SourceReadError(f"device tree: expected a name, found {value!r}")
            name = value
            at += 1
            kind, value = tokens[at]
            if kind == "{":
                child = DtsNode(name=name, labels=labels)
                node.children.append(child)
                at = body(child, at + 1)
                if at < len(tokens) and tokens[at][0] == ";":
                    at += 1
            elif kind == "=":
                values: list[Any] = []
                at += 1
                while tokens[at][0] != ";":
                    kind, value = tokens[at]
                    if kind == "str":
                        values.append(value)
                    elif kind in ("cells", "bytes"):
                        values.append((kind, value))
                    elif kind == "ref":
                        values.append(("ref", value))
                    at += 1
                node.props[name] = values
                at += 1
            elif kind == ";":
                node.props[name] = []
                at += 1
            else:
                raise SourceReadError(f"device tree: unexpected {value!r} after {name!r}")
        return at

    while position < len(tokens):
        kind, value = tokens[position]
        if kind == "directive":
            while position < len(tokens) and tokens[position][0] != ";":
                position += 1
            position += 1
        elif kind == "name" and value == "/" and tokens[position + 1][0] == "{":
            position = body(root, position + 2)
            if position < len(tokens) and tokens[position][0] == ";":
                position += 1
        else:
            raise SourceReadError(f"device tree: unexpected top-level {value!r}")
    return root


def _cell_number(cells: Sequence[int | str], start: int, count: int) -> int:
    value = 0
    for cell in cells[start : start + count]:
        if not isinstance(cell, int):
            raise SourceReadError(f"device tree reg holds a reference {cell!r}")
        value = (value << 32) | cell
    return value


def dts_memory(root: DtsNode) -> list[dict[str, Any]]:
    """Every ``device_type = "memory"`` node: ``{node, base, bytes, enabled}`` per ``reg`` entry."""
    out = []
    for parent, node in root.walk():
        if node.strings("device_type") != ["memory"]:
            continue
        scope = parent or root
        address_cells = scope.cells("#address-cells")
        size_cells = scope.cells("#size-cells")
        if len(address_cells) != 1 or len(size_cells) != 1:
            raise SourceReadError(f"device tree: {node.name}'s parent states no #address-cells/#size-cells")
        a, s = int(address_cells[0]), int(size_cells[0])
        reg = node.cells("reg")
        if not reg or len(reg) % (a + s):
            raise SourceReadError(f"device tree: {node.name} has no well-formed reg")
        enabled = node.strings("status") in ([], ["okay"], ["ok"])
        for at in range(0, len(reg), a + s):
            out.append(
                {
                    "node": node.name,
                    "base": _cell_number(reg, at, a),
                    "bytes": _cell_number(reg, at + a, s),
                    "enabled": enabled,
                }
            )
    return out


def dts_isas(root: DtsNode) -> list[str]:
    """``riscv,isa`` of every ``device_type = "cpu"`` node."""
    return [
        isa for _, node in root.walk() if node.strings("device_type") == ["cpu"] for isa in node.strings("riscv,isa")
    ]


# ------------------------------------------------------------------------------------------------------ ELF


def elf_load_segments(path: Path) -> list[tuple[int, int]]:
    """``(physical address, memory size)`` of every loadable segment with a non-empty memory image."""
    data = Path(path).read_bytes()
    if data[:4] != b"\x7fELF":
        raise SourceReadError(f"{path}: not an ELF file")
    wide, order = data[4], {1: "<", 2: ">"}.get(data[5])
    if wide not in (1, 2) or order is None:
        raise SourceReadError(f"{path}: unknown ELF class or byte order")
    if wide == 2:
        (phoff,) = struct.unpack_from(order + "Q", data, 32)
        phentsize, phnum = struct.unpack_from(order + "HH", data, 54)
        layout, fields = order + "IIQQQQQQ", ("type", "flags", "offset", "vaddr", "paddr", "filesz", "memsz", "align")
    else:
        (phoff,) = struct.unpack_from(order + "I", data, 28)
        phentsize, phnum = struct.unpack_from(order + "HH", data, 42)
        layout, fields = order + "IIIIIIII", ("type", "offset", "vaddr", "paddr", "filesz", "memsz", "flags", "align")
    loadable = 1  # PT_LOAD in the ELF specification's program-header type table
    out = []
    for index in range(phnum):
        row = dict(zip(fields, struct.unpack_from(layout, data, phoff + index * phentsize), strict=True))
        if row["type"] == loadable and row["memsz"]:
            out.append((row["paddr"], row["memsz"]))
    return out


_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def csr_accesses(listing: str, names: Iterable[str]) -> list[dict[str, Any]]:
    """Instructions of an ``objdump -d -M no-aliases`` listing that name one of the CSRs ``names``.

    With aliases off every CSR instruction spells its CSR operand by name (``csrrs a0,time,zero``;
    LLVM: ``csrrs a0, time, zero``). A line is ``<address>:<TAB><encoding><TAB><mnemonic><TAB><operands>``
    under a ``<address> <function>:`` header; operands are split at commas, and a trailing ``# comment``
    or `` <symbol>`` is not an operand. No mnemonic is assumed: any instruction naming the CSR touches it.
    """
    wanted = {str(n) for n in names}
    function: str | None = None
    hits: list[dict[str, Any]] = []
    for line in listing.splitlines():
        stripped = line.strip()
        if stripped.endswith(">:") and " <" in stripped:
            function = stripped[stripped.index(" <") + len(" <") : -len(">:")]
            continue
        address, colon, rest = line.partition(":")
        address = address.strip()
        if not colon or not address or not set(address) <= _HEX_DIGITS:
            continue
        fields = [f.strip() for f in rest.split("\t") if f.strip()]
        if len(fields) < 3:
            continue
        operands = [part.strip().split(" ")[0] for part in fields[2].partition("#")[0].split(",")]
        for operand in operands:
            if operand in wanted:
                hits.append(
                    {
                        "function": function,
                        "address": address,
                        "instruction": f"{fields[1]} {fields[2]}",
                        "csr": operand,
                    }
                )
    return hits

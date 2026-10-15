"""Structural readers behind the binding receipt: C parameter headers, device trees, ELF load segments and
CDE configuration sources (Chisel generators). Every value is parsed, never pattern-matched.

A reader that meets a form it does not understand raises :class:`SourceReadError`; the receipt records the
fact as unknown rather than guessing.
"""

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


# -------------------------------------------------------------------------------- CDE configuration sources


@dataclass(frozen=True)
class Token:
    kind: str  # ident | number | string | op | punct
    text: str
    line: int
    prefix: str = ""  # a string literal's interpolator (``x`` in ``x"8000_0000"``)


_OP_CHARS = frozenset("!#%&*+-/:<=>?@\\^|~")
_PUNCT = frozenset("()[]{},;.")


def scala_tokens(text: str) -> list[Token]:
    """Tokens of a Scala source: identifiers, numbers, (interpolated) strings, operators and punctuation."""
    out: list[Token] = []
    i, n, line = 0, len(text), 1
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            i += 1
        elif c.isspace():
            i += 1
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
        elif text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            line += text.count("\n", i, j)
            i = j
        elif c == '"':
            token, i, line = _scala_string(text, i, line, "")
            out.append(token)
        elif c == "'":
            if i + 2 < n and text[i + 2] == "'":
                out.append(Token("string", text[i + 1], line, "'"))
                i += 3
            elif i + 1 < n and text[i + 1] == "\\":
                end = text.find("'", i + 2)
                out.append(Token("string", text[i + 1 : end], line, "'"))
                i = end + 1
            else:
                out.append(Token("op", "'", line))
                i += 1
        elif c.isdigit():
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            if j + 1 < n and text[j] == "." and text[j + 1].isdigit():
                j += 1
                while j < n and (text[j].isalnum() or text[j] == "_"):
                    j += 1
            out.append(Token("number", text[i:j], line))
            i = j
        elif c.isalpha() or c in "_$":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] in "_$"):
                j += 1
            if j < n and text[j] == '"':
                token, i, line = _scala_string(text, j, line, text[i:j])
                out.append(token)
            else:
                out.append(Token("ident", text[i:j], line))
                i = j
        elif c == "`":
            end = text.find("`", i + 1)
            out.append(Token("ident", text[i + 1 : end], line))
            i = end + 1
        elif c in _PUNCT:
            out.append(Token("punct", c, line))
            i += 1
        elif c in _OP_CHARS:
            j = i + 1
            while j < n and text[j] in _OP_CHARS and not text.startswith("//", j) and not text.startswith("/*", j):
                j += 1
            out.append(Token("op", text[i:j], line))
            i = j
        else:
            i += 1
    return out


def _scala_string(text: str, start: int, line: int, prefix: str) -> tuple[Token, int, int]:
    if text.startswith('"""', start):
        end = text.find('"""', start + 3)
        end = len(text) if end < 0 else end
        body = text[start + 3 : end]
        return Token("string", body, line, prefix), end + 3, line + body.count("\n")
    j = start + 1
    while j < len(text) and text[j] != '"':
        j += 2 if text[j] == "\\" else 1
    return Token("string", text[start + 1 : j], line, prefix), j + 1, line


_OPEN = {"(": ")", "[": "]", "{": "}"}


def _close(tokens: Sequence[Token], at: int) -> int:
    """Index just past the bracket group opening at ``at``."""
    depth = 0
    for index in range(at, len(tokens)):
        token = tokens[index]
        if token.kind == "punct" and token.text in _OPEN:
            depth += 1
        elif token.kind == "punct" and token.text in _OPEN.values():
            depth -= 1
            if not depth:
                return index + 1
    raise SourceReadError("unbalanced brackets")


def _split(tokens: Sequence[Token], separator: tuple[str, str]) -> list[list[Token]]:
    """Split at ``separator`` (kind, text) tokens outside any bracket group."""
    parts: list[list[Token]] = [[]]
    depth = 0
    for token in tokens:
        if token.kind == "punct" and token.text in _OPEN:
            depth += 1
        elif token.kind == "punct" and token.text in _OPEN.values():
            depth -= 1
        if not depth and (token.kind, token.text) == separator:
            parts.append([])
            continue
        parts[-1].append(token)
    return [p for p in parts if p] if separator != ("punct", ",") else parts


def _qualified(tokens: Sequence[Token], at: int) -> tuple[str, int]:
    """``a.b.C`` starting at ``at``; returns the dotted name and the index after it."""
    if at >= len(tokens) or tokens[at].kind != "ident":
        raise SourceReadError("expected a name")
    parts, at = [tokens[at].text], at + 1
    while at + 1 < len(tokens) and tokens[at].text == "." and tokens[at + 1].kind == "ident":
        parts.append(tokens[at + 1].text)
        at += 2
    return ".".join(parts), at


@dataclass(frozen=True)
class ConfigClass:
    """One ``class Name(params) extends Parent(args)`` declaration."""

    fqn: str
    name: str
    package: str
    imports: tuple[tuple[str, str], ...]  # (prefix, member or "_")
    params: tuple[tuple[str, tuple[Token, ...] | None], ...]  # (name, default)
    parent: str
    parent_args: tuple[Token, ...] | None
    path: Path
    line: int


@dataclass(frozen=True)
class _Thunk:
    tokens: tuple[Token, ...]
    env: Mapping[str, _Thunk]


@dataclass(frozen=True)
class Leaf:
    """One partial function of a flattened configuration: its cases and the bindings it sees."""

    cases: Mapping[str, tuple[Token, ...]]
    up: str | None  # the name the function gives its lower-priority view, if it has one
    env: Mapping[str, _Thunk]
    origin: str  # "<path>:<line> <class>"
    via: tuple[str, ...] = ()  # the composition sites from the resolved configuration down to this function


@dataclass(frozen=True)
class Resolved:
    value: int
    origin: str
    via: tuple[str, ...]
    files: tuple[str, ...]


def _site(path: Path, line: int, text: str) -> str:
    return f"{path}:{line} {text}"


def _spelled(tokens: Sequence[Token], limit: int = 120) -> str:
    """A composition term as source-like text, shortened for provenance lines."""
    text = ""
    for token in tokens:
        word = token.prefix + (f'"{token.text}"' if token.kind == "string" else token.text)
        tight = not text or text[-1] in ".(" or word in (".", "(", ")", ",")
        text += ("" if tight else " ") + word
    return text if len(text) <= limit else text[: limit - 3] + "..."


#: Scala operator precedence by an operator's first character, lowest first (the language specification).
_TIERS = ("|", "^", "&", "=!", "<>", ":", "+-", "*/%")
_ARITHMETIC = frozenset({"+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^"})


def _tier(op: str) -> int | None:
    if op not in _ARITHMETIC:
        return None
    return next(index for index, starts in enumerate(_TIERS) if op[0] in starts)


class ConfigSources:
    """CDE configuration classes under source roots, resolved the way the config system composes them.

    ``A ++ B`` gives ``A`` priority; a partial function's case that reads its lower-priority view (``up``)
    modifies the value below it, any other case replaces it. ``config_class`` names the composition base
    (``Config``); ``interpolators`` maps a string-literal prefix to the radix of the integer it spells.
    """

    def __init__(self, roots: Iterable[Path], *, config_class: str, interpolators: Mapping[str, int]) -> None:
        self.roots = [Path(r) for r in roots]
        self.config_class = config_class
        self.interpolators = dict(interpolators)
        self._texts: dict[Path, str] | None = None
        self._declared: dict[str, list[Path]] = {}
        self._classes: dict[Path, list[ConfigClass]] = {}
        self.read_files: set[str] = set()

    # -- indexing
    def _all_texts(self) -> dict[Path, str]:
        if self._texts is None:
            self._texts = {}
            for root in self.roots:
                if not root.is_dir():
                    raise SourceReadError(f"configuration source root {root} is not a directory")
                for path in sorted(root.rglob("*.scala")):
                    text = path.read_text(encoding="utf-8", errors="replace")
                    self._texts[path] = text
                    at = text.find("class ")
                    while at >= 0:
                        start = stop = at + len("class ")
                        while stop < len(text) and (text[stop].isalnum() or text[stop] in "_$"):
                            stop += 1
                        self._declared.setdefault(text[start:stop], []).append(path)
                        at = text.find("class ", stop)
        return self._texts

    def _file_classes(self, path: Path) -> list[ConfigClass]:
        if path not in self._classes:
            self._classes[path] = _classes_in(path, scala_tokens(self._all_texts()[path]))
        return self._classes[path]

    def candidates(self, simple: str) -> list[ConfigClass]:
        self._all_texts()
        found = []
        for path in dict.fromkeys(self._declared.get(simple, ())):
            found += [c for c in self._file_classes(path) if c.name == simple]
        return found

    def find(self, reference: str, context: ConfigClass | None = None) -> ConfigClass:
        simple = reference.rsplit(".", 1)[-1]
        found = self.candidates(simple)
        if "." in reference:
            found = [c for c in found if c.fqn == reference or c.fqn.endswith("." + reference)]
        elif len(found) > 1 and context is not None:
            scopes = [context.package]
            scopes += [prefix for prefix, member in context.imports if member in ("_", simple)]
            package = context.package
            while "." in package:
                package = package.rsplit(".", 1)[0]
                scopes.append(package)
            for scope in scopes:
                scoped = [c for c in found if c.package == scope]
                if scoped:
                    found = scoped
                    break
        if len(found) != 1:
            where = f" (from {context.fqn})" if context else ""
            raise SourceReadError(
                f"configuration class {reference!r}{where} resolves to {len(found)} declarations: "
                + ", ".join(sorted(c.fqn for c in found))
            )
        self.read_files.add(str(found[0].path))
        return found[0]

    # -- flattening
    def leaves(self, reference: str) -> Iterator[Leaf]:
        """Partial functions of ``reference`` in priority order (highest first), lazily."""
        cls = self.find(reference)
        yield from self._class_leaves(cls, args=(), caller={}, via=(_site(cls.path, cls.line, cls.fqn),))

    def _class_leaves(
        self, cls: ConfigClass, *, args: Sequence[list[Token]], caller: Mapping[str, _Thunk], via: tuple[str, ...]
    ):
        env: dict[str, _Thunk] = {}
        positional = [a for a in args if not _is_named(a)]
        named = {a[0].text: a[2:] for a in args if _is_named(a)}
        for index, (name, default) in enumerate(cls.params):
            if name in named:
                env[name] = _Thunk(tuple(named[name]), caller)
            elif index < len(positional):
                env[name] = _Thunk(tuple(positional[index]), caller)
            elif default is not None:
                env[name] = _Thunk(default, env)
        if cls.parent.rsplit(".", 1)[-1] == self.config_class:
            yield from self._body_leaves(list(cls.parent_args or ()), env, cls, via)
            return
        parent = self.find(cls.parent, cls)
        parent_args = _split(list(cls.parent_args or ()), ("punct", ","))
        site = _site(cls.path, cls.line, f"{cls.fqn} extends {parent.fqn}")
        yield from self._class_leaves(parent, args=[a for a in parent_args if a], caller=env, via=(*via, site))

    def _body_leaves(self, tokens: list[Token], env: Mapping[str, _Thunk], cls: ConfigClass, via: tuple[str, ...]):
        if not tokens:
            return
        function = _partial_function(tokens)
        if function is not None:
            up, cases = function
            yield Leaf(cases=cases, up=up, env=env, origin=_site(cls.path, tokens[0].line, cls.fqn), via=via)
            return
        for term in _split(tokens, ("op", "++")):
            if not term or term[0].text != "new":
                raise SourceReadError(f"{cls.path}:{term[0].line if term else '?'}: unsupported configuration term")
            name, at = _qualified(term, 1)
            group = None
            if at < len(term) and term[at].text == "(":
                end = _close(term, at)
                group = term[at + 1 : end - 1]
                at = end
            if at != len(term):
                raise SourceReadError(f"{cls.path}:{term[0].line}: unsupported configuration term after {name}")
            site = _site(cls.path, term[0].line, _spelled(term))
            if name.rsplit(".", 1)[-1] == self.config_class:
                yield from self._body_leaves(list(group or ()), env, cls, (*via, site))
            else:
                target = self.find(name, cls)
                args = [a for a in _split(group or [], ("punct", ",")) if a]
                yield from self._class_leaves(target, args=args, caller=env, via=(*via, site))

    # -- lookup
    def resolve(self, reference: str, key: str, attribute: str | None) -> Resolved:
        """The integer value of ``key`` (or of its named field ``attribute``) in configuration ``reference``."""
        for leaf in self.leaves(reference):
            expression = leaf.cases.get(key)
            if expression is None:
                continue
            if attribute is None:
                value = self.evaluate(list(expression), leaf.env)
                return Resolved(value, leaf.origin, leaf.via, tuple(sorted(self.read_files)))
            named = _named_argument(expression, attribute)
            if named is not None:
                value = self.evaluate(named, leaf.env)
                return Resolved(value, leaf.origin, leaf.via, tuple(sorted(self.read_files)))
            if leaf.up is not None and any(t.kind == "ident" and t.text == leaf.up for t in expression):
                continue
            raise SourceReadError(f"{leaf.origin} redefines {key} without {attribute}")
        raise SourceReadError(f"no configuration fragment of {reference} defines {key}")

    # -- evaluation
    def evaluate(self, tokens: Sequence[Token], env: Mapping[str, _Thunk]) -> int:
        value, at = self._expression(list(tokens), 0, env, 0)
        if at != len(tokens):
            raise SourceReadError(f"line {tokens[at].line}: cannot evaluate {' '.join(t.text for t in tokens)!r}")
        return value

    def _expression(self, tokens: list[Token], at: int, env, floor: int) -> tuple[int, int]:
        left, at = self._primary(tokens, at, env)
        while at < len(tokens) and tokens[at].kind == "op":
            op = tokens[at].text
            tier = _tier(op)
            if tier is None or tier < floor:
                break
            right, at = self._expression(tokens, at + 1, env, tier + 1)
            left = _apply(op, left, right)
        return left, at

    def _primary(self, tokens: list[Token], at: int, env) -> tuple[int, int]:
        if at >= len(tokens):
            raise SourceReadError("expression ends early")
        token = tokens[at]
        if token.kind == "op" and token.text == "-":
            value, at = self._primary(tokens, at + 1, env)
            return -value, at
        if token.kind == "number":
            return _number(token), at + 1
        if token.kind == "string" and token.prefix in self.interpolators:
            try:
                return int(token.text.replace("_", ""), self.interpolators[token.prefix]), at + 1
            except ValueError as error:
                raise SourceReadError(f"line {token.line}: {token.prefix}{token.text!r} is not an integer") from error
        if token.kind == "punct" and token.text == "(":
            end = _close(tokens, at)
            return self.evaluate(tokens[at + 1 : end - 1], env), end
        if token.kind == "ident" and token.text in env:
            thunk = env[token.text]
            return self.evaluate(list(thunk.tokens), thunk.env), at + 1
        if token.kind == "ident" and at + 1 < len(tokens) and tokens[at + 1].text == "(":
            end = _close(tokens, at + 1)
            inner = tokens[at + 2 : end - 1]
            if token.text == "BigInt" and len(inner) == 1 and inner[0].kind == "string" and not inner[0].prefix:
                return int(inner[0].text), end
            if token.text == "BigInt":
                return self.evaluate(inner, env), end
        raise SourceReadError(f"line {token.line}: cannot evaluate {token.text!r}")


def _number(token: Token) -> int:
    text = token.text.replace("_", "")
    while text and text[-1] in "lL":
        text = text[:-1]
    try:
        return int(text, 0)
    except ValueError as error:
        raise SourceReadError(f"line {token.line}: {token.text!r} is not an integer literal") from error


def _apply(op: str, left: int, right: int) -> int:
    table = {
        "+": lambda: left + right,
        "-": lambda: left - right,
        "*": lambda: left * right,
        "/": lambda: left // right,
        "%": lambda: left % right,
        "<<": lambda: left << right,
        ">>": lambda: left >> right,
        "&": lambda: left & right,
        "|": lambda: left | right,
        "^": lambda: left ^ right,
    }
    if op not in table:
        raise SourceReadError(f"unsupported operator {op!r}")
    return table[op]()


def _is_named(arg: Sequence[Token]) -> bool:
    return len(arg) >= 3 and arg[0].kind == "ident" and arg[1].kind == "op" and arg[1].text == "="


def _named_argument(tokens: Sequence[Token], name: str) -> list[Token] | None:
    """The value tokens of ``name = <value>`` used as a named argument anywhere in ``tokens``."""
    for at in range(1, len(tokens) - 2):
        if (
            tokens[at].kind == "ident"
            and tokens[at].text == name
            and tokens[at + 1].kind == "op"
            and tokens[at + 1].text == "="
            and tokens[at - 1].text in ("(", ",")
        ):
            value: list[Token] = []
            depth = 0
            for token in tokens[at + 2 :]:
                if token.kind == "punct" and token.text in _OPEN:
                    depth += 1
                elif token.kind == "punct" and token.text in _OPEN.values():
                    if not depth:
                        break
                    depth -= 1
                elif not depth and token.text == ",":
                    break
                value.append(token)
            return value
    return None


def _partial_function(tokens: list[Token]) -> tuple[str | None, dict[str, tuple[Token, ...]]] | None:
    """``(site, here, up) => { case K => v ... }`` -> (the ``up`` name, ``{K: v}``); ``None`` otherwise."""
    if tokens[0].text != "(":
        return None
    end = _close(tokens, 0)
    if end >= len(tokens) or tokens[end].text != "=>":
        return None
    params = [p for p in _split(tokens[1 : end - 1], ("punct", ",")) if p]
    up = params[2][0].text if len(params) == 3 and len(params[2]) == 1 and params[2][0].text != "_" else None
    body = tokens[end + 1 :]
    if body and body[0].text == "{":
        body = body[1 : _close(body, 0) - 1]
    cases: dict[str, tuple[Token, ...]] = {}
    starts = []
    depth = 0
    for index, token in enumerate(body):
        if token.kind == "punct" and token.text in _OPEN:
            depth += 1
        elif token.kind == "punct" and token.text in _OPEN.values():
            depth -= 1
        elif not depth and token.kind == "ident" and token.text == "case":
            starts.append(index)
    for number, start in enumerate(starts):
        stop = starts[number + 1] if number + 1 < len(starts) else len(body)
        clause = body[start + 1 : stop]
        arrow = next((i for i, t in enumerate(clause) if t.text == "=>"), None)
        if arrow is None:
            continue
        pattern = clause[:arrow]
        guard = next((i for i, t in enumerate(pattern) if t.kind == "ident" and t.text == "if"), None)
        pattern = pattern[:guard] if guard is not None else pattern
        if pattern and all(t.kind == "ident" or t.text == "." for t in pattern):
            name, at = _qualified(pattern, 0)
            if at == len(pattern):
                cases.setdefault(name.rsplit(".", 1)[-1], tuple(clause[arrow + 1 :]))
    return up, cases


def _classes_in(path: Path, tokens: list[Token]) -> list[ConfigClass]:
    packages: list[str] = []
    imports: list[tuple[str, str]] = []
    out: list[ConfigClass] = []
    at = 0
    while at < len(tokens):
        token = tokens[at]
        if token.kind == "ident" and token.text == "package" and at + 1 < len(tokens):
            if tokens[at + 1].text == "object":
                at += 2
                continue
            name, at = _qualified(tokens, at + 1)
            packages.append(name)
            continue
        if token.kind == "ident" and token.text == "import" and at + 1 < len(tokens):
            name, at = _qualified(tokens, at + 1)
            if name.endswith("._"):
                imports.append((name[: -len("._")], "_"))
            elif at + 1 < len(tokens) and tokens[at].text == "." and tokens[at + 1].text == "{":
                end = _close(tokens, at + 1)
                for part in _split(tokens[at + 2 : end - 1], ("punct", ",")):
                    if part:
                        imports.append((name, part[0].text))
                at = end
            else:
                prefix, _, member = name.rpartition(".")
                imports.append((prefix, member))
            continue
        if token.kind == "ident" and token.text == "class" and (at == 0 or tokens[at - 1].text != "case"):
            parsed = _class_header(path, tokens, at, ".".join(packages), tuple(imports))
            if parsed is not None:
                out.append(parsed)
        at += 1
    return out


def _class_header(path, tokens, at, package, imports) -> ConfigClass | None:
    if at + 1 >= len(tokens) or tokens[at + 1].kind != "ident":
        return None
    name, cursor = tokens[at + 1].text, at + 2
    if cursor < len(tokens) and tokens[cursor].text == "[":
        cursor = _close(tokens, cursor)
    params: list[tuple[str, tuple[Token, ...] | None]] = []
    first = True
    while cursor < len(tokens) and tokens[cursor].text == "(":
        end = _close(tokens, cursor)
        if first:
            for part in _split(tokens[cursor + 1 : end - 1], ("punct", ",")):
                words = [t for t in part if t.text not in ("val", "var", "implicit", "private", "override")]
                if not words or words[0].kind != "ident":
                    continue
                equals = next((i for i, t in enumerate(words) if t.kind == "op" and t.text == "="), None)
                params.append((words[0].text, tuple(words[equals + 1 :]) if equals is not None else None))
        first = False
        cursor = end
    if cursor >= len(tokens) or tokens[cursor].text != "extends":
        return None
    parent, cursor = _qualified(tokens, cursor + 1)
    args = None
    if cursor < len(tokens) and tokens[cursor].text == "(":
        end = _close(tokens, cursor)
        args = tuple(tokens[cursor + 1 : end - 1])
    fqn = f"{package}.{name}" if package else name
    return ConfigClass(
        fqn=fqn,
        name=name,
        package=package,
        imports=imports,
        params=tuple(params),
        parent=parent,
        parent_args=args,
        path=path,
        line=tokens[at].line,
    )

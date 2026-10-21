"""A structural reader for the bodies of named modules in an elaborated (CHIRRTL) FIRRTL design.

WHY THIS EXISTS. Several facts about a design are not SHAPES (a memory's depth, a port's width) but
BEHAVIOURS: which event releases a dependency, which signal a stall waits on. Those are written in the
design as expressions -- ``node _T = and(entries[0].valid, eq(entries[0].bits.issued, UInt<1>(0h0)))``
-- and a reader that pattern-matches lines cannot tell ``valid && !issued`` from ``valid`` without
guessing at spellings. This module parses what it reads: statements into a guard-scoped list, every
expression into a tree, and it answers the one question the semantic extractors ask -- "what does this
signal's value depend on" -- by walking the definitions, not by matching text.

NOTHING HERE KNOWS A TARGET. Module, signal and field names arrive from the caller (in practice from a
target's own probe declaration, see :mod:`merlin.targetgen.rtl.semantic_facts`). The vocabulary below
is the FIRRTL language's own: statement keywords, the two ground literal constructors, and the bundle
``flip`` marker.

No regular expressions: statements are split on the language's own separators, expressions are read
by a character tokenizer with bracket depth, exactly as the grammar nests them.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import threading
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Keywords that open a module definition with a body we can read.
_MODULE_OPENERS = ("module ", "public module ")
#: Keywords that open any top-level definition (a body we skip ends at the next of these).
_TOP_OPENERS = ("module ", "public module ", "extmodule ", "intmodule ", "layer ", "type ", "option ")
#: The two ground-literal constructors of the language.
_LITERAL_CTORS = ("UInt", "SInt")
#: Declarations whose value is STATE or an interface, so a dependency walk stops at them.
_LEAF_DECLS = ("reg", "regreset", "input", "inst", "mem", "smem", "cmem", "mport")
#: Declarations whose value is whatever is connected to them, so a dependency walk continues through them.
_DRIVEN_DECLS = ("wire", "output")

# ------------------------------------------------------------------------------------------ reading

_BODY_CACHE_BYTES = 16 * 1024 * 1024
_BODY_CACHE_ENTRIES = 16
_body_cache: OrderedDict = OrderedDict()
_body_cache_bytes = 0
_body_cache_lock = threading.RLock()


def _reset_body_cache() -> None:
    global _body_cache, _body_cache_bytes, _body_cache_lock
    _body_cache = OrderedDict()
    _body_cache_bytes = 0
    _body_cache_lock = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_body_cache)


class _DigestReader(io.RawIOBase):
    """Hash exactly the raw bytes consumed by the ordinary text reader."""

    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.source.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count


def _body_cache_size(value) -> int:
    return sys.getsizeof(value) + (
        sum(_body_cache_size(item) for item in value) if isinstance(value, (tuple, frozenset)) else 0
    )


def _cache_bodies(key, bodies) -> None:
    global _body_cache_bytes
    frozen = tuple((name, tuple(lines)) for name, lines in bodies.items())
    # Include fixed per-entry bookkeeping, as well as the complete retained
    # key/line objects. Never retain the large source text or a verdict.
    size = _body_cache_size((key, frozen)) + 256
    if size > _BODY_CACHE_BYTES:
        return
    with _body_cache_lock:
        previous = _body_cache.pop(key, None)
        if previous is not None:
            _body_cache_bytes -= previous[1]
        _body_cache[key] = (frozen, size)
        _body_cache_bytes += size
        while len(_body_cache) > _BODY_CACHE_ENTRIES or _body_cache_bytes > _BODY_CACHE_BYTES:
            _, (_, removed_size) = _body_cache.popitem(last=False)
            _body_cache_bytes -= removed_size


def read_module_bodies(fir: str | Path, names: Iterable[str]) -> dict[str, list[tuple[int, str]]]:
    """``{module: [(line_no, raw_line), ...]}`` for each requested module that the design defines.

    Reopens and fully hashes the actual file on every call. Identical bytes and
    requested names may reuse only immutable selected lines, never source or
    qualification verdicts. Cache misses hash the bytes actually parsed, too.
    A module that is not defined is simply absent from the result -- the caller decides whether that is
    an UNKNOWN, since only it knows whether it asked for something the design must have.
    """
    wanted = frozenset(names)
    with open(fir, "rb") as source:
        digest = hashlib.file_digest(source, "sha256").digest()
    key = (digest, wanted)
    with _body_cache_lock:
        cached = _body_cache.get(key)
        if cached is not None:
            _body_cache.move_to_end(key)
            return {name: list(lines) for name, lines in cached[0]}
    with open(fir, "rb") as source:
        reader = _DigestReader(source)
        with io.TextIOWrapper(io.BufferedReader(reader), encoding="utf-8", errors="replace") as text:
            bodies = _read_module_lines(text, wanted)
            # The file may have changed since the lookup hash. Store only the
            # digest of the complete stream that produced these exact lines.
            parsed_digest = reader.digest.digest()
    _cache_bodies((parsed_digest, wanted), bodies)
    return bodies


def _read_module_lines(lines: Iterable[str], wanted: frozenset[str]) -> dict[str, list[tuple[int, str]]]:
    out: dict[str, list[tuple[int, str]]] = {}
    current: str | None = None
    for no, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if indent == 2 and stripped.startswith(_TOP_OPENERS):
            current = None
            if stripped.startswith(_MODULE_OPENERS):
                head = stripped.split(":", 1)[0].split()
                name = head[-1] if head else ""
                if name in wanted:
                    current = name
                    out[name] = [(no, line)]
            continue
        if current is not None:
            if indent < 4 and stripped:
                current = None
                continue
            out[current].append((no, line))
    return out


def split_locator(text: str) -> tuple[str, str | None]:
    """``(statement, locator)`` -- the source locator is the trailing ``@[file line:col]`` annotation."""
    at = text.rfind(" @[")
    if at < 0 or not text.rstrip().endswith("]"):
        return text.rstrip(), None
    return text[:at].rstrip(), text[at + 3 : text.rstrip().rfind("]")]


def locator_citation(locator: str | None) -> dict[str, Any] | None:
    """``{"file", "line"}`` out of a locator's first site (a locator may carry several)."""
    if not locator:
        return None
    first = locator.split(",", 1)[0].strip()
    parts = first.split()
    if len(parts) < 2:
        return None
    path, pos = parts[0], parts[1]
    line_text = pos.split(":", 1)[0]
    try:
        line = int(line_text)
    except ValueError:
        return None
    return {"file": path, "line": line}


# ------------------------------------------------------------------------------------- expressions


@dataclass(frozen=True)
class Ref:
    """A reference path. ``segments`` are field names (str), static indices (int) or ``None`` for a
    DYNAMIC index (whose index expression is kept separately in ``dynamic``)."""

    root: str
    segments: tuple[str | int | None, ...] = ()
    dynamic: tuple[Any, ...] = ()

    def text(self) -> str:
        out = self.root
        for seg in self.segments:
            if seg is None:
                out += "[*]"
            elif isinstance(seg, int):
                out += f"[{seg}]"
            else:
                out += f".{seg}"
        return out


@dataclass(frozen=True)
class Lit:
    kind: str  # "UInt" | "SInt"
    width: int | None
    value: int


@dataclass(frozen=True)
class Prim:
    op: str
    args: tuple[Any, ...]


@dataclass(frozen=True)
class IntArg:
    value: int


class FirrtlParseError(ValueError):
    """An expression the reader could not parse. Raised, never skipped: a dropped statement would make
    a dependency invisible, and invisible is read as "none"."""


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c in "()[],.<>":
            out.append(c)
            i += 1
            continue
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
            continue
        j = i
        while j < n and not text[j].isspace() and text[j] not in '()[],.<>"':
            j += 1
        out.append(text[i:j])
        i = j
    return out


def _int(tok: str) -> int | None:
    t = tok.strip()
    neg = t.startswith("-")
    if neg or t.startswith("+"):
        t = t[1:]
    try:
        if t[:1] == "0" and len(t) > 1 and t[1] in "hHbBoOdD":
            base = {"h": 16, "b": 2, "o": 8, "d": 10}[t[1].lower()]
            v = int(t[2:], base)
        else:
            v = int(t, 10)
    except (ValueError, IndexError):
        return None
    return -v if neg else v


class _Parser:
    def __init__(self, text: str):
        self.toks = _tokens(text)
        self.i = 0

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, want: str | None = None) -> str:
        tok = self.peek()
        if tok is None or (want is not None and tok != want):
            raise FirrtlParseError(f"expected {want!r}, found {tok!r} in {' '.join(self.toks)!r}")
        self.i += 1
        return tok

    def expr(self) -> Any:
        tok = self.take()
        if tok in _LITERAL_CTORS:
            width = None
            if self.peek() == "<":
                self.take("<")
                width = _int(self.take())
                self.take(">")
            self.take("(")
            value = _int(self.take())
            self.take(")")
            if value is None:
                raise FirrtlParseError(f"literal without an integer value near {tok!r}")
            return Lit(tok, width, value)
        as_int = _int(tok)
        if as_int is not None:
            return IntArg(as_int)
        if self.peek() == "(":
            self.take("(")
            args: list[Any] = []
            while self.peek() != ")":
                args.append(self.expr())
                if self.peek() == ",":
                    self.take(",")
            self.take(")")
            return Prim(tok, tuple(args))
        segments: list[str | int | None] = []
        dynamic: list[Any] = []
        while self.peek() in (".", "["):
            if self.take() == ".":
                segments.append(self.take())
            else:
                inner = self.expr()
                self.take("]")
                if isinstance(inner, IntArg):
                    segments.append(inner.value)
                else:
                    segments.append(None)
                    dynamic.append(inner)
        return Ref(tok, tuple(segments), tuple(dynamic))


def parse_expr(text: str) -> Any:
    p = _Parser(text)
    out = p.expr()
    if p.peek() is not None:
        raise FirrtlParseError(f"trailing tokens after an expression: {' '.join(p.toks[p.i :])!r}")
    return out


def _split_top_comma(text: str) -> tuple[str, str]:
    depth = 0
    for i, c in enumerate(text):
        if c in "([<{":
            depth += 1
        elif c in ")]>}":
            depth -= 1
        elif c == "," and depth == 0:
            return text[:i].strip(), text[i + 1 :].strip()
    raise FirrtlParseError(f"no top-level comma in {text!r}")


# -------------------------------------------------------------------------------------------- types


def parse_type(text: str) -> Any:
    """A FIRRTL type as nested data: ``("uint"|"sint", width)``, ``("vec", elem, n)``,
    ``("bundle", [(name, flipped, type), ...])`` or ``("other", text)``."""
    toks = _tokens(text.replace("{", " { ").replace("}", " } ").replace(":", " : "))
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        tok = toks[pos[0]]
        pos[0] += 1
        return tok

    def one():
        tok = take()
        if tok == "{":
            fields = []
            while peek() != "}":
                flipped = False
                name = take()
                if name == "flip":
                    flipped, name = True, take()
                if take() != ":":
                    raise FirrtlParseError(f"malformed bundle field near {name!r}")
                fields.append((name, flipped, one()))
                if peek() == ",":
                    take()
            take()
            base: Any = ("bundle", fields)
        elif tok in _LITERAL_CTORS:
            width = None
            if peek() == "<":
                take()
                width = _int(take())
                take()
            base = ("uint" if tok == "UInt" else "sint", width)
        else:
            base = ("other", tok)
            if peek() == "<":  # parametric non-ground type (Analog<N>, Probe<...>): skip its argument
                depth = 0
                while True:
                    t = take()
                    depth += 1 if t == "<" else (-1 if t == ">" else 0)
                    if depth == 0:
                        break
        while peek() == "[":
            take()
            n = _int(take())
            take()
            base = ("vec", base, n)
        return base

    return one()


def field_type(typ: Any, path: Iterable[str | int]) -> Any:
    """The sub-type at ``path`` (field names / vector indices), or None when the path does not exist."""
    cur = typ
    for seg in path:
        if cur is None:
            return None
        if isinstance(seg, int) or (isinstance(seg, str) and seg.isdigit()):
            cur = cur[1] if cur[0] == "vec" else None
            continue
        if cur[0] != "bundle":
            return None
        cur = next((t for (n, _f, t) in cur[1] if n == seg), None)
    return cur


def type_width(typ: Any) -> int | None:
    """Total bit width of a type, or None when any leaf is unsized/non-ground."""
    if typ is None:
        return None
    kind = typ[0]
    if kind in ("uint", "sint"):
        return typ[1]
    if kind == "vec":
        inner = type_width(typ[1])
        return None if inner is None or typ[2] is None else inner * typ[2]
    if kind == "bundle":
        total = 0
        for _n, _f, t in typ[1]:
            w = type_width(t)
            if w is None:
                return None
            total += w
        return total
    return None


# ------------------------------------------------------------------------------------------- bodies


@dataclass(frozen=True)
class Guard:
    cond: Any  # a parsed expression
    positive: bool

    def text(self) -> str:
        c = self.cond.text() if isinstance(self.cond, Ref) else repr(self.cond)
        return c if self.positive else f"not {c}"


@dataclass(frozen=True)
class Connect:
    lhs: Ref
    rhs: Any
    guards: tuple[Guard, ...]
    line: int
    locator: str | None


@dataclass
class ModuleBody:
    name: str
    line: int
    locator: str | None
    nodes: dict[str, tuple[Any, int, str | None]] = field(default_factory=dict)
    decls: dict[str, tuple[str, str, int]] = field(default_factory=dict)  # name -> (keyword, type text, line)
    connects: list[Connect] = field(default_factory=list)

    def connects_to(self, root: str) -> list[Connect]:
        return [c for c in self.connects if c.lhs.root == root]

    def port_type(self, name: str) -> Any:
        decl = self.decls.get(name)
        if decl is None or decl[0] not in ("input", "output"):
            return None
        return parse_type(decl[1])


def parse_module(name: str, lines: list[tuple[int, str]]) -> ModuleBody:
    """Parse one module body into nodes, declarations and guard-scoped connects."""
    head_no, head = lines[0]
    _, head_loc = split_locator(head)
    body = ModuleBody(name=name, line=head_no, locator=head_loc)
    stack: list[tuple[int, Guard]] = []
    last_popped: dict[int, Guard] = {}
    for no, raw in lines[1:]:
        text, loc = split_locator(raw)
        stripped = text.lstrip(" ")
        if not stripped:
            continue
        indent = len(text) - len(stripped)
        while stack and stack[-1][0] >= indent:
            ind, g = stack.pop()
            last_popped[ind] = g
        words = stripped.split(None, 1)
        kw = words[0]
        rest = words[1] if len(words) > 1 else ""
        if kw == "when":
            cond = parse_expr(rest.rstrip().rstrip(":").strip())
            stack.append((indent, Guard(cond, True)))
            continue
        if kw == "else":
            prior = last_popped.get(indent)
            if prior is None:
                raise FirrtlParseError(f"line {no}: `else` with no `when` at this indentation")
            stack.append((indent, Guard(prior.cond, False)))
            tail = rest.rstrip().rstrip(":").strip()
            if tail.startswith("when "):
                # `else when c :` -- a nested positive guard in the same scope
                cond = parse_expr(tail[len("when ") :].strip())
                stack.append((indent, Guard(cond, True)))
            continue
        guards = tuple(g for _i, g in stack)
        if kw == "node":
            lhs, _, rhs = rest.partition("=")
            body.nodes[lhs.strip()] = (parse_expr(rhs.strip()), no, loc)
        elif kw == "connect":
            lhs_text, rhs_text = _split_top_comma(rest)
            lhs = parse_expr(lhs_text)
            if not isinstance(lhs, Ref):
                raise FirrtlParseError(f"line {no}: connect to a non-reference {lhs_text!r}")
            body.connects.append(Connect(lhs, parse_expr(rhs_text), guards, no, loc))
        elif kw in ("wire", "reg", "regreset", "input", "output"):
            nm, _, typ = rest.partition(":")
            typ = typ.strip()
            if kw in ("reg", "regreset"):
                typ = _split_top_comma(typ)[0] if "," in typ else typ
            body.decls[nm.strip()] = (kw, typ, no)
        elif kw == "inst":
            nm, _, of = rest.partition(" of ")
            body.decls[nm.strip()] = ("inst", of.strip(), no)
        elif kw in ("mem", "smem", "cmem"):
            nm = rest.split(":", 1)[0].strip()
            body.decls[nm] = (kw, "", no)
        elif kw in ("read", "write", "infer", "rdwr") and rest.startswith("mport"):
            nm = rest[len("mport") :].split("=", 1)[0].strip()
            body.decls[nm] = ("mport", "", no)
        # printf / assert / invalidate / skip / stop / cover / attach carry no value dependency
    return body


def load_modules(fir: str | Path, names: Iterable[str]) -> dict[str, ModuleBody]:
    return {n: parse_module(n, ls) for n, ls in read_module_bodies(fir, names).items()}


# ----------------------------------------------------------------------------------- dependency walk


def _refs_in(expr: Any) -> Iterator[Ref]:
    if isinstance(expr, Ref):
        yield expr
        for d in expr.dynamic:
            yield from _refs_in(d)
    elif isinstance(expr, Prim):
        for a in expr.args:
            yield from _refs_in(a)


def _prims_in(expr: Any) -> Iterator[Prim]:
    if isinstance(expr, Prim):
        yield expr
        for a in expr.args:
            yield from _prims_in(a)
    elif isinstance(expr, Ref):
        for d in expr.dynamic:
            yield from _prims_in(d)


def _compatible(a: Ref, b: Ref) -> bool:
    """Whether two references can alias: one path is a prefix of the other, a dynamic index matching
    any static one. Over-approximates on purpose -- a walk that misses an alias misses a dependency."""
    for x, y in zip(a.segments, b.segments, strict=False):
        if x is None or y is None:
            continue
        if x != y:
            return False
    return True


class Cone:
    """The transitive fan-in of expressions inside ONE module, through nodes and wires.

    Stops at state and interfaces (registers, ports, instance pins, memories): those are the leaves a
    behavioural question is asked about. ``include_guards`` also walks the conditions under which a
    wire is driven, for questions whose answer lives in a ``when`` rather than in the data.
    """

    def __init__(self, module: ModuleBody, *, include_guards: bool = False):
        self.m = module
        self.include_guards = include_guards

    def walk(self, expr: Any) -> tuple[set[str], list[Prim], list[int]]:
        """``(leaf reference texts, every primitive operation, source lines visited)``."""
        leaves: set[str] = set()
        prims: list[Prim] = []
        lines: list[int] = []
        seen: set[tuple] = set()
        work: list[Any] = [expr]
        while work:
            e = work.pop()
            for p in _prims_in(e):
                prims.append(p)
            for r in _refs_in(e):
                key = (r.root, r.segments)
                if key in seen:
                    continue
                seen.add(key)
                if r.root in self.m.nodes:
                    node_expr, line, _loc = self.m.nodes[r.root]
                    lines.append(line)
                    work.append(node_expr)
                    continue
                decl = self.m.decls.get(r.root)
                # An OUTPUT port read inside its own module has the value connected to it, like a wire.
                if decl is not None and decl[0] in _DRIVEN_DECLS:
                    drivers = [c for c in self.m.connects_to(r.root) if _compatible(c.lhs, r)]
                    for c in drivers:
                        lines.append(c.line)
                        work.append(c.rhs)
                        if self.include_guards:
                            work.extend(g.cond for g in c.guards)
                    if drivers:
                        continue
                    # Undriven here: a flipped (input) field of an output bundle, or an unassigned wire.
                    # Either way it is an interface the module READS, so it is a leaf, never dropped.
                leaves.add(r.text())
        return leaves, prims, lines

    def leaves(self, expr: Any) -> set[str]:
        return self.walk(expr)[0]


def guards_match(guards: tuple[Guard, ...], spec: Iterable[str]) -> bool:
    """Whether ``spec`` -- guard texts like ``"is_load"`` / ``"not is_load"`` -- is a subsequence of the
    guard stack (outer ``when``s the spec does not mention are allowed)."""
    want = [s.strip() for s in spec]
    have = [g.text() for g in guards]
    i = 0
    for h in have:
        if i < len(want) and h == want[i]:
            i += 1
    return i == len(want)


def ref_matches(ref_text: str, pattern: str) -> bool:
    """``pattern`` with ``[*]`` standing for any index; a pattern also matches every sub-path of itself."""
    norm = []
    i = 0
    while i < len(ref_text):
        if ref_text[i] == "[":
            j = ref_text.index("]", i)
            norm.append("[*]")
            i = j + 1
        else:
            norm.append(ref_text[i])
            i += 1
    t = "".join(norm)
    p = pattern
    return t == p or t.startswith(p + ".") or t.startswith(p + "[")


def eq_constants(prims: Iterable[Prim], field_pattern: str) -> set[int]:
    """Every integer a reference matching ``field_pattern`` is compared EQUAL to (``eq(field, lit)``)."""
    out: set[int] = set()
    for p in prims:
        if p.op != "eq" or len(p.args) != 2:
            continue
        a, b = p.args
        for ref, lit in ((a, b), (b, a)):
            if isinstance(ref, Ref) and isinstance(lit, Lit) and ref_matches(ref.text(), field_pattern):
                out.add(lit.value)
    return out

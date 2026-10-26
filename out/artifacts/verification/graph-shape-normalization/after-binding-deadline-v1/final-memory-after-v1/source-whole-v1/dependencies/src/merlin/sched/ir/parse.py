"""Read a kernel back from its canonical text -- the inverse of ``Kernel.text``.

WHY THIS EXISTS. A schedule that can be printed but not read is an in-process object, not an artifact.
Three things this repo already wants are blocked by the missing direction:

* a schedule composed here cannot be handed to a minted package as DATA, so the only way to ship one is
  to bake it into a rendered template as literals, which is what `mint.py` does today;
* a vendored copy of the primitives (the rewrite vocabulary has no merlin imports, so it CAN travel into
  a package) cannot checkpoint what it composed, replay it, or diff two candidates across processes;
* a certificate cannot be attributed to the schedule that earned it the way it is attributed to bytes --
  a digest identifies a schedule, but nothing could recover the schedule a digest names.

THE PARSER VERIFIES ITSELF, AND THIS IS THE WHOLE DESIGN. A kernel's identity IS its text: the digest is
the sha256 of what ``Kernel.text`` prints. So a parser that is merely mostly right is worse than none --
it would hand back a kernel that looks like the one written down and carries a different identity, and
every digest-keyed thing downstream (the layer-bench key, a numerics contract, a cert) would be
attributed to the wrong schedule. Silently.

So ``parse_kernel`` re-renders what it built and refuses if the text differs. A bug in here is a
REFUSAL, never a kernel with a different digest, and ``digest(parse(text(k))) == digest(k)`` holds by
construction rather than by testing -- the test then only has to show the refusal can fire.

This module imports nothing outside ``merlin.sched.ir``, so it travels with the vendorable core.
Parsing is structural (a bracket-depth scanner and ``str.split``), never by pattern match: the repo's
no-regex rule exists because a too-narrow pattern silently drops valid-but-differently-spelled input,
and silently dropping a statement is exactly the failure the self-check above is written against.
"""

from __future__ import annotations

from .expr import Const, Expr, Scaled, Select, Sum, Var
from .kernel import NULL, Call, Kernel, KernelError, Loop, Ptr, Stage, TensorArg

__all__ = ["ParseError", "parse_kernel", "parse_expr"]

#: The clause keywords a call line may carry after its operand list, in the order ``Kernel.text``
#: emits them. Peeled from the right, so each one's value is whatever sits between it and the next.
_CLAUSES = (" on ", " after ", " in ", " sets ", " needs ")

_OPENERS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = {")": "(", "]": "[", "}": "{"}
_DIGITS = frozenset("0123456789")


class ParseError(KernelError):
    """The text does not denote a kernel, or denotes one that would not print back the same."""


# -- scanning ------------------------------------------------------------------------------------


def _split_top(s: str, sep: str) -> list[str]:
    """Split ``s`` on ``sep``, but only where no bracket is open.

    Every separator in this grammar also occurs INSIDE a value -- ``, `` inside a ``select``, ``=``
    inside its comparison, ``:`` inside a stage span -- so a plain split would cut a value in half.
    """
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS:
            depth -= 1
            if depth < 0:
                raise ParseError(f"unbalanced bracket in {s!r}")
        if depth == 0 and sep and s.startswith(sep, i):
            parts.append("".join(buf))
            buf = []
            i += len(sep)
            continue
        buf.append(ch)
        i += 1
    if depth:
        raise ParseError(f"unbalanced bracket in {s!r}")
    parts.append("".join(buf))
    return parts


def _match(s: str, start: int) -> int:
    """Index of the bracket closing the one at ``start``."""
    want = _OPENERS[s[start]]
    depth = 0
    for i in range(start, len(s)):
        if s[i] in _OPENERS:
            depth += 1
        elif s[i] in _CLOSERS:
            depth -= 1
            if depth == 0:
                if s[i] != want:
                    raise ParseError(f"mismatched bracket in {s!r}")
                return i
    raise ParseError(f"unclosed {s[start]!r} in {s!r}")


def _int(s: str) -> int:
    t = s.strip()
    body = t[1:] if t.startswith("-") else t
    if not body or set(body) - _DIGITS:
        raise ParseError(f"not an integer: {s!r}")
    return int(t)


def _is_ident(s: str) -> bool:
    return bool(s) and (s[0].isalpha() or s[0] == "_") and all(c.isalnum() or c == "_" for c in s)


# -- expressions ---------------------------------------------------------------------------------


def parse_expr(s: str) -> Expr:
    """One index expression, in the form ``merlin.sched.ir.expr.render`` prints.

    Built directly rather than through ``add``/``mul``, which simplify: a parser whose job is to
    recover what was written must not fold ``(i * 1)`` away, because the text it would then print is
    not the text it was given -- and that is a different digest.
    """
    s = s.strip()
    if not s:
        raise ParseError("empty index expression")
    if s.startswith("select(") and _match(s, 6) == len(s) - 1:
        parts = _split_top(s[7:-1], ", ")
        if len(parts) != 3:
            raise ParseError(f"select takes a condition and two branches: {s!r}")
        cond = _split_top(parts[0], " == ")
        if len(cond) != 2:
            raise ParseError(f"select condition must be `<var> == <int>`: {parts[0]!r}")
        return Select(cond[0].strip(), _int(cond[1]), parse_expr(parts[1]), parse_expr(parts[2]))
    if s.startswith("(") and _match(s, 0) == len(s) - 1:
        inner = s[1:-1]
        terms = _split_top(inner, " + ")
        if len(terms) > 1:
            return Sum(tuple(parse_expr(t) for t in terms))
        factors = _split_top(inner, " * ")
        if len(factors) == 2:
            return Scaled(parse_expr(factors[0]), _int(factors[1]))
        raise ParseError(f"a parenthesised expression is a sum or a scaled expression: {s!r}")
    if _is_ident(s):
        return Var(s)
    return Const(_int(s))


def _parse_value(s: str):
    """One operand value: an index expression, a pointer, ``null``, or a float."""
    s = s.strip()
    if s == "null":
        return NULL
    if s.startswith("&"):
        open_i = s.find("[")
        if open_i < 0 or _match(s, open_i) != len(s) - 1:
            raise ParseError(f"not a pointer into a tensor argument: {s!r}")
        return Ptr(s[1:open_i], parse_expr(s[open_i + 1 : -1]))
    # A float prints as its exact hex form, which no index expression can spell.
    body = s[1:] if s.startswith("-") else s
    if body.startswith("0x") or body in ("inf", "nan"):
        try:
            return float.fromhex(s)
        except ValueError as exc:
            raise ParseError(f"not a float: {s!r}") from exc
    return parse_expr(s)


def _parse_pairs(s: str, sep: str) -> tuple[tuple[str, object], ...]:
    """``k=v`` pairs separated by ``sep``, values parsed."""
    s = s.strip()
    if not s:
        return ()
    out: list[tuple[str, object]] = []
    for part in _split_top(s, sep):
        halves = _split_top(part, "=")
        if len(halves) < 2:
            raise ParseError(f"not a `key=value` pair: {part!r}")
        out.append((halves[0].strip(), _parse_value("=".join(halves[1:]))))
    return tuple(out)


# -- statements ----------------------------------------------------------------------------------


def _parse_stage(s: str) -> Stage:
    """``memory[row:row+rows]`` with an optional ``@bank`` and a trailing ``w`` for a writer."""
    open_i = s.find("[")
    if open_i < 0:
        raise ParseError(f"not a staging record: {s!r}")
    close_i = _match(s, open_i)
    span = _split_top(s[open_i + 1 : close_i], ":")
    if len(span) != 2:
        raise ParseError(f"a staging span is `first:last`: {s!r}")
    row, end = _int(span[0]), _int(span[1])
    if end < row:
        raise ParseError(f"staging span ends before it starts: {s!r}")
    tail = s[close_i + 1 :]
    writes = tail.endswith("w")
    if writes:
        tail = tail[:-1]
    bank = None
    at = tail.find("@")
    if at >= 0:
        bank = _int(tail[at + 1 :])
        tail = tail[:at]
    depth = 1
    if tail.startswith("x"):
        depth = _int(tail[1:])
        tail = ""
    if tail:
        raise ParseError(f"unexpected text after a staging span: {tail!r}")
    return Stage(s[:open_i], row, end - row, bank, writes, depth)


def _parse_call(content: str, lineno: int) -> Call:
    open_i = content.find("(")
    if open_i < 0:
        raise ParseError(f"line {lineno}: not a call and not a loop header: {content!r}")
    head = _split_top(content[:open_i], " = ")
    if len(head) > 2:
        raise ParseError(f"line {lineno}: a call produces at most one token: {content!r}")
    produces = head[0].strip() if len(head) == 2 else None
    instr = head[-1].strip()
    close_i = _match(content, open_i)
    args = _parse_pairs(content[open_i + 1 : close_i], ", ")

    rest = content[close_i + 1 :]
    found: dict[str, str] = {}
    for word in reversed(_CLAUSES):
        parts = _split_top(rest, word)
        if len(parts) > 2:
            raise ParseError(f"line {lineno}: {word.strip()!r} appears more than once")
        if len(parts) == 2:
            rest, found[word.strip()] = parts[0], parts[1]
    if rest.strip():
        raise ParseError(f"line {lineno}: unexpected text after the call: {rest.strip()!r}")

    stages = tuple(_parse_stage(p) for p in _split_top(found["in"], ",")) if "in" in found else ()
    return Call(
        instr,
        args,
        found.get("on", "").strip() or None,
        produces,
        tuple(found["after"].split(",")) if "after" in found else (),
        stages,
        _parse_pairs(found.get("sets", ""), ","),
        _parse_pairs(found.get("needs", ""), ","),
    )


def _parse_for(content: str, lineno: int) -> tuple[str, int]:
    if not content.endswith(":"):
        raise ParseError(f"line {lineno}: a loop header ends with ':'")
    parts = _split_top(content[4:-1], " in 0..")
    if len(parts) != 2:
        raise ParseError(f"line {lineno}: expected `for <var> in 0..<extent>:`")
    return parts[0].strip(), _int(parts[1])


def _parse_body(lines: list[tuple[int, str, int]], i: int, depth: int) -> tuple[tuple, int]:
    body: list = []
    while i < len(lines):
        level, content, lineno = lines[i]
        if level < depth:
            break
        if level > depth:
            raise ParseError(f"line {lineno}: unexpected indentation")
        if content.startswith("for "):
            var, extent = _parse_for(content, lineno)
            inner, i = _parse_body(lines, i + 1, depth + 1)
            body.append(Loop(var, extent, inner))
        else:
            body.append(_parse_call(content, lineno))
            i += 1
    return tuple(body), i


# -- the kernel ----------------------------------------------------------------------------------


def _parse_params(s: str) -> tuple[TensorArg, ...]:
    s = s.strip()
    if not s:
        return ()
    out: list[TensorArg] = []
    for part in _split_top(s, ", "):
        halves = _split_top(part, ": ")
        if len(halves) != 2:
            raise ParseError(f"a tensor argument is `name: dtype[shape] access`: {part!r}")
        rest = halves[1]
        open_i = rest.find("[")
        if open_i < 0:
            raise ParseError(f"a tensor argument declares a shape: {part!r}")
        close_i = _match(rest, open_i)
        shape = tuple(_int(d) for d in rest[open_i + 1 : close_i].split(","))
        out.append(TensorArg(halves[0].strip(), shape, rest[:open_i].strip(), rest[close_i + 1 :].strip()))
    return tuple(out)


def _lines(text: str) -> list[tuple[int, str, int]]:
    out: list[tuple[int, str, int]] = []
    for lineno, raw in enumerate(text.split("\n"), 1):
        if not raw.strip():
            continue
        stripped = raw.lstrip(" ")
        indent = len(raw) - len(stripped)
        if indent % 2:
            raise ParseError(f"line {lineno}: indent of {indent} is not a multiple of two")
        out.append((indent // 2, stripped, lineno))
    return out


def _first_difference(want: str, got: str) -> str:
    a, b = want.split("\n"), got.split("\n")
    for n, (x, y) in enumerate(zip(a, b), 1):
        if x != y:
            return f"line {n}: read {y!r}, given {x!r}"
    return f"the text has {len(a)} lines and what it parsed to prints {len(b)}"


def parse_kernel(text: str) -> Kernel:
    """The kernel ``text`` denotes, or ``ParseError``.

    The result is checked against its own input: if it does not print back byte for byte, it is not
    the kernel that was written down, and returning it would give that schedule a different digest.
    """
    if not text.endswith("\n"):
        text = text + "\n"
    lines = _lines(text)
    if not lines:
        raise ParseError("no kernel here: the text is empty")
    level, header, lineno = lines[0]
    if level or not header.startswith("kernel "):
        raise ParseError(f"line {lineno}: a kernel starts with `kernel <name>(<args>)`")
    open_i = header.find("(")
    if open_i < 0 or _match(header, open_i) != len(header) - 1:
        raise ParseError(f"line {lineno}: the kernel's argument list is not closed")
    name = header[len("kernel ") : open_i].strip()
    args = _parse_params(header[open_i + 1 : -1])

    i = 1
    attrs: tuple[tuple[str, str], ...] = ()
    if i < len(lines) and lines[i][0] == 1 and lines[i][1].startswith("attrs {"):
        raw = lines[i][1][len("attrs {") : -1]
        pairs = []
        for part in _split_top(raw, ", ") if raw.strip() else []:
            halves = _split_top(part, "=")
            if len(halves) < 2:
                raise ParseError(f"line {lines[i][2]}: not a `key=value` attribute: {part!r}")
            pairs.append((halves[0].strip(), "=".join(halves[1:])))
        attrs = tuple(pairs)
        i += 1

    body, i = _parse_body(lines, i, 1)
    if i != len(lines):
        raise ParseError(f"line {lines[i][2]}: unexpected indentation")

    kernel = Kernel(name, args, body, attrs)
    printed = kernel.text()
    if printed != text:
        raise ParseError(
            "this text does not print back the same, so the kernel read from it would carry a "
            f"different digest than the one it was written with -- {_first_difference(text, printed)}"
        )
    return kernel

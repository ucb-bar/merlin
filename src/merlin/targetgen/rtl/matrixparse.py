"""Read declared matrix decode fields and bounded Scala table forms.

Ported from OPU-VME 80e00803d; hardware values remain caller-supplied.
"""

from __future__ import annotations

from .scala_parse import _balanced_after, _split_top_level, _strip_comments


def parse_bitpat(token: str) -> tuple[int, int, int] | None:
    """``(match, mask, width)`` for a ``BitPat("b10?1_0")`` token, or None when it is not one.

    ``?`` is a don't-care bit (mask 0), ``_`` is a visual separator. Only the binary form is accepted:
    that is the only one a ``BitPat`` string takes.
    """
    t = token.strip()
    head = "BitPat("
    if not t.startswith(head):
        return None
    inner = t[len(head) :].strip()
    if not inner.startswith('"'):
        return None
    end = inner.find('"', 1)
    if end < 0:
        return None
    lit = inner[1:end]
    if not lit.startswith("b"):
        return None
    bits = lit[1:].replace("_", "")
    if not bits or any(c not in "01?" for c in bits):
        return None
    match = mask = 0
    for c in bits:
        match <<= 1
        mask <<= 1
        if c != "?":
            mask |= 1
            match |= int(c)
    return match, mask, len(bits)


def bitpat_defs(text: str, container: str) -> dict[str, tuple[int, int, int]]:
    """``{NAME: (match, mask, width)}`` for every ``def NAME = BitPat("...")`` inside ``object container``."""
    body = _balanced_after(text, f"object {container}")
    if body is None:
        return {}
    out: dict[str, tuple[int, int, int]] = {}
    for raw in _strip_comments(body).splitlines():
        line = raw.strip()
        if not line.startswith("def ") or "=" not in line:
            continue
        name, _, rhs = line[len("def ") :].partition("=")
        name = name.strip()
        got = parse_bitpat(rhs.strip())
        if name.isidentifier() and got is not None:
            out[name] = got
    return out


def seq_members(text: str, seq_name: str) -> list[str]:
    """The dotted tokens of ``def <seq_name> = Seq(a.B, c.D.VV, ...)``, in order. Empty when absent."""
    i = text.find(f"def {seq_name}")
    if i < 0:
        return []
    body = _balanced_after(_strip_comments(text[i:]), "Seq", "(", ")")
    if body is None:
        return []
    return [t for t in _split_top_level(body) if t]


def _all_ints(tokens: list[str]) -> bool:
    return bool(tokens) and all(t.isdecimal() for t in tokens)


def seq_int_tuples(text: str, def_name: str) -> list[tuple[int, ...]]:
    """The integer tuples of ``def <def_name> ... = ... Seq((8, 4), (32, 1)) ...``, in order.

    Reads the first ``Seq(`` after the declaration, so a guarded form (``if (x) Seq(...) else Nil``)
    yields the declared list. Non-integer members are skipped, never coerced.
    """
    stripped = _strip_comments(text)
    i = stripped.find(f"def {def_name}")
    if i < 0:
        return []
    rhs = stripped[stripped.find("=", i) + 1 :]
    body = _balanced_after(rhs, "Seq", "(", ")")
    out: list[tuple[int, ...]] = []
    for tup in _split_top_level(body or ""):
        parts = [t.strip() for t in tup.strip().removeprefix("(").removesuffix(")").split(",")]
        if _all_ints(parts):
            out.append(tuple(int(t) for t in parts))
    return out


def match_case_table(text: str, container: str, def_name: str) -> dict[tuple[int, ...], int]:
    """``{(8, 4): 4, ...}`` from ``def <def_name>(...) = (...) match { case (8, 4) => 4 ... }`` in ``container``.

    Only integer cases are read; a wildcard ``case _`` is a default the caller must not rely on, so it is
    left out and a missing key is the caller's to report.
    """
    body = _balanced_after(_strip_comments(text), f"object {container}")
    if body is None:
        return {}
    i = body.find(f"def {def_name}")
    if i < 0:
        return {}
    table = _balanced_after(body[i:], "match", "{", "}") or ""
    out: dict[tuple[int, ...], int] = {}
    for line in table.splitlines():
        head, arrow, rhs = line.strip().partition("=>")
        head = head.strip()
        if not arrow or not head.startswith("case ("):
            continue
        key = [t.strip() for t in head[len("case ") :].removeprefix("(").removesuffix(")").split(",")]
        if _all_ints(key) and rhs.strip().isdecimal():
            out[tuple(int(t) for t in key)] = int(rhs.strip())
    return out


def _props_tokens(text: str, obj: str) -> list[str] | None:
    """The top-level entries of ``object <obj> ... { val props = Seq( ... ) }``, across lines."""
    stripped = _strip_comments(text)
    header = f"object {obj} "
    i = stripped.find(header)
    if i < 0:
        return None
    j = stripped.find("val props", i)
    nxt = stripped.find("\nobject ", i + len(header))
    if j < 0 or (0 <= nxt < j):
        return None
    body = _balanced_after(stripped[j:], "Seq", "(", ")")
    if body is None:
        return None
    return _split_top_level(body)


def _call(token: str) -> tuple[str, str] | None:
    """``("F6", "OPMFunct6.x")`` for ``F6(OPMFunct6.x)``; None for a token that is not a call."""
    t = token.strip()
    open_ = t.find("(")
    if open_ <= 0 or not t.endswith(")"):
        return None
    return t[:open_].strip(), t[open_ + 1 : -1].strip()


def _uint_literal(token: str) -> int | None:
    """The integer in ``31.U(5.W)`` / ``0.U`` / ``31``."""
    head = token.strip().split(".U", 1)[0].strip()
    try:
        return int(head, 0)
    except ValueError:
        return None

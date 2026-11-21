"""Shared Scala declaration readers for matrix ISA derivation.

These syntax helpers are target-neutral; OPU encodings and facts live in Kodiak.
"""

from __future__ import annotations

_PLACEHOLDER = "_"

_VECTOR_UNIT_MIXIN_SUFFIX = "VectorUnit"


def _block_after(text: str, header: str) -> str | None:
    """The brace-balanced block following ``header``, or None when the header is absent.

    Balanced-brace scanning rather than a line pattern, so a nested block inside the body cannot end the
    read early and a body that opens on a later line is still found.
    """
    i = text.find(header)
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
    return None


def _balanced_after(text: str, header: str, opener: str = "{", closer: str = "}") -> str | None:
    """The delimiter-balanced span following ``header``. ``_block_after`` is the brace case of this.

    Scala configs are *parenthesised* (``extends Config( … )``) while its enums and traits are braced, so
    both delimiters are needed to read the same file.
    """
    i = text.find(header)
    if i < 0:
        return None
    j = text.find(opener, i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == opener:
            depth += 1
        elif text[k] == closer:
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
    return None


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` at nesting depth zero, so ``VectorParams(a, b)`` stays one argument."""
    out: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [s.strip() for s in out if s.strip()]


def scala_param_names(text: str, decl: str) -> list[str]:
    """The declared parameter names of ``decl`` (e.g. ``class WithShuttleVectorUnit``), in order.

    This is what makes a positional call site readable without assuming an argument order. Guessing that
    order is a real hazard here: ``(vLen, dLen)`` and ``(dLen, vLen)`` are both plausible, both parse, and
    swapping them yields a tile edge that is wrong by exactly a factor of two — which would silently
    certify a corpus against the wrong geometry rather than fail.
    """
    body = _balanced_after(_strip_comments(text), decl, "(", ")")
    if body is None:
        return []
    names = []
    for arg in _split_top_level(body):
        name, sep, _ = arg.partition(":")
        name = name.strip()
        if sep and name.isidentifier():
            names.append(name)
    return names


def _strip_comments(text: str) -> str:
    """Drop ``//`` line comments. A commented-out ``val x = Value`` must NOT consume an ordinal, and
    the OPU's own sources carry exactly that kind of annotation next to the ops of interest."""
    out = []
    for line in text.splitlines():
        i = line.find("//")
        out.append(line if i < 0 else line[:i])
    return "\n".join(out)


def chisel_enum_ordinals(text: str, enum_name: str) -> dict[str, int]:
    """``{member: ordinal}`` for a ``ChiselEnum``, which is what its funct6 encoding is.

    Handles the three declaration shapes these enums use: ``val a, b, c = Value`` (consecutive
    ordinals), ``val _ = Value`` (a reserved slot that advances the counter without naming anything), and
    ``val x = Value(0x40.U)`` — an EXPLICIT ordinal, which both names that value and resets where
    counting continues from, so treating it as just another slot would corrupt everything after it.
    ``def alias = other`` lines are aliases, not slots, and are skipped.
    """
    body = _block_after(text, f"object {enum_name}")
    if body is None:
        return {}
    ordinals: dict[str, int] = {}
    nxt = 0
    for raw in _strip_comments(body).splitlines():
        line = raw.strip()
        if not line.startswith("val ") or "=" not in line:
            continue
        names_part, value_part = line[len("val ") :].split("=", 1)
        value_part = value_part.strip()
        if not value_part.startswith("Value"):
            continue  # `val x = something_else` is not an enum slot
        names = [n.strip() for n in names_part.split(",") if n.strip()]
        if not names:
            continue
        explicit = _explicit_ordinal(value_part)
        if explicit is not None:
            # `Value(0x40.U)` pins this member and continues counting from there.
            if len(names) == 1 and names[0] != _PLACEHOLDER:
                ordinals[names[0]] = explicit
            nxt = explicit + 1
            continue
        for name in names:
            if name != _PLACEHOLDER:
                ordinals[name] = nxt
            nxt += 1
    return ordinals


def _explicit_ordinal(value_part: str) -> int | None:
    """The integer in ``Value(0x40.U)``, or None for a bare ``Value``."""
    if not value_part.startswith("Value("):
        return None
    inner = value_part[len("Value(") :]
    close = inner.find(")")
    if close < 0:
        return None
    token = inner[:close].strip().removesuffix(".U").strip()
    try:
        return int(token, 0)
    except ValueError:
        return None


def bit_literal_defs(text: str, container: str) -> dict[str, int]:
    """``{name: value}`` for ``def NAME = "b1010111".U`` style declarations inside a trait/object.

    Accepts an optional width suffix (``.U(3.W)``) and both binary and hex literals, because the same
    trait spells the opcode in binary and other constants in hex, and a reader that only understood one
    would drop the other silently.
    """
    body = _block_after(text, f"trait {container}")
    if body is None:
        body = _block_after(text, f"object {container}")
    if body is None:
        return {}
    out: dict[str, int] = {}
    for raw in _strip_comments(body).splitlines():
        line = raw.strip()
        if not line.startswith("def ") or "=" not in line:
            continue
        name, value = line[len("def ") :].split("=", 1)
        name = name.strip()
        if not name or not name.isidentifier():
            continue
        got = _bit_literal(value.strip())
        if got is not None:
            out[name] = got
    return out


def _bit_literal(value: str) -> int | None:
    """The integer in ``"b1010111".U`` / ``"h2a".U`` / ``0x40.U``, or None when it is not a literal."""
    if value.startswith('"'):
        end = value.find('"', 1)
        if end < 0:
            return None
        token = value[1:end]
        base = {"b": 2, "h": 16, "o": 8}.get(token[:1].lower())
        if base is None:
            return None
        digits = token[1:].replace("_", "")
        try:
            return int(digits, base)
        except ValueError:
            return None
    token = value.split()[0].removesuffix(".U").removesuffix(".W").strip()
    try:
        return int(token, 0)
    except ValueError:
        return None

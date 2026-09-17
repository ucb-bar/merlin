#!/usr/bin/env python3
"""The one front-matter reader for docs/ — shared by the docs gate and the hub generator.

Front-matter is a small, fixed YAML subset, so the repo parses it with the stdlib rather than
taking a PyYAML dependency into the git hooks. The subset is what the docs actually write, and
that is wider than one line: measured 2026-09-16, of the 83 docs carrying `code_refs`, 58 spell
it inline (`[a, b]`), 23 spell it as a block sequence, and 2 continue an inline list onto the
next line.

A reader that understood only the inline form did not fail on the other two -- it mis-read them,
which is worse. A block sequence produced an EMPTY value, so those 23 docs had no code_refs at
all and the drift detector never fired for them; a continued inline list produced the raw string
`"[merlin/...py,"`, which the caller then iterated CHARACTER BY CHARACTER, so every ref resolved
to @MISSING. Both spellings are valid YAML and both were silently wrong, so the fix belongs in
the parser and not in the 25 documents.

Scalars are unquoted, because `title: "Design: ..."` was reaching the generated hub with its
quotes still attached (and sorting under `"` rather than under D).
"""

from __future__ import annotations


def _unquote(value: str) -> str:
    """Strip one layer of matching quotes. YAML escapes are not used in this subset."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _flow_items(inner: str) -> list[str]:
    """Split the body of a `[a, b]` sequence. Items here never contain a comma."""
    return [_unquote(item.strip()) for item in inner.split(",") if item.strip()]


def parse(text: str) -> dict | None:
    """Return the front-matter mapping, or None when the file carries none.

    Understands scalars, inline sequences (including ones continued across lines), and block
    sequences. Anything else is kept as its raw scalar text rather than guessed at.
    """
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 4)
    if end == -1:
        return None

    lines = text[4:end].splitlines()
    fm: dict = {}
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip()
        i += 1
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()

        if value.startswith("["):
            # An inline sequence, which may not close on this line.
            buf = value
            while "]" not in buf and i < n:
                buf += " " + lines[i].strip()
                i += 1
            fm[key] = _flow_items(buf[1 : buf.rindex("]")] if "]" in buf else buf[1:])
        elif value:
            fm[key] = _unquote(value)
        else:
            # An empty value introduces a block sequence -- or is simply empty.
            items: list[str] = []
            while i < n:
                nxt = lines[i].strip()
                if not nxt:
                    i += 1
                    continue
                if not nxt.startswith("- "):
                    break
                items.append(_unquote(nxt[2:].strip()))
                i += 1
            fm[key] = items if items else ""
    return fm

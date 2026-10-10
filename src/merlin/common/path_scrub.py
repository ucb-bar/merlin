"""Scrub host filesystem paths out of text that crosses into an agent-visible document.

An error message from a build or a simulator routinely names files outside the agent's work tree, and
quoting it verbatim would disclose host layout the agent has no use for. This replaces
every absolute path in ``text`` with :data:`TOKEN`, except a path under one of the ``keep`` roots (the
agent's own visible work tree), and first rewrites any ``rewrite`` prefix (a host-private snapshot of
that tree) to the root the agent knows it by.

Structural, not pattern-based: a path starts with ``/`` at the beginning of the text or after a
delimiter, and runs to the next delimiter. Nothing here names a host, a user or a directory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

TOKEN = "<host-path>"

#: Characters that end a path, or may sit immediately before one.
_DELIMITERS = frozenset(" \t\r\n'\"`()[]{}<>,;=|")


def _norm(root: str | Path) -> str:
    text = str(root)
    return text.rstrip("/") if text != "/" else text


def scrub_host_paths(
    text: str,
    *,
    keep: Sequence[str | Path] = (),
    rewrite: Mapping[str | Path, str | Path] | None = None,
) -> str:
    """``text`` with every absolute path outside ``keep`` replaced by :data:`TOKEN`."""
    if not text:
        return text
    roots = sorted((_norm(k) for k in keep if str(k)), key=len, reverse=True)
    mapping = sorted(((_norm(a), _norm(b)) for a, b in (rewrite or {}).items()), key=lambda p: -len(p[0]))
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        char = text[i]
        if char == "/" and (i == 0 or text[i - 1] in _DELIMITERS or text[i - 1] == ":"):
            j = i
            while j < n and text[j] not in _DELIMITERS:
                j += 1
            if j == i + 1:  # a bare slash is not a path
                out.append(char)
                i += 1
                continue
            path = text[i:j]
            trailing = ""
            while path and path[-1] in ".:":
                trailing = path[-1] + trailing
                path = path[:-1]
            for source, target in mapping:
                if path == source or path.startswith(source + "/"):
                    path = target + path[len(source) :]
                    break
            visible = any(path == root or path.startswith(root + "/") for root in roots)
            out.append((path if visible else TOKEN) + trailing)
            i = j
            continue
        out.append(char)
        i += 1
    return "".join(out)

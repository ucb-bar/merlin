"""Read explicit lowered FIRRTL memory blocks without changing hardware sources.

The census consumes one declaration line per memory. Lowered FIRRTL instead
stores type and depth on indented property lines; this streaming observer joins
those exact fields into the same structural memory vocabulary. It never guesses
element decomposition, numeric formats, latency behavior or memory roles.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator


def _declaration(header: str, properties: list[str]) -> str:
    stripped = header.strip()
    lhs, separator, annotation = stripped.partition(":")
    if not separator or len(lhs.split()) != 2 or lhs.split()[0] != "mem":
        raise ValueError("malformed lowered FIRRTL memory declaration")
    selected = {}
    for line in properties:
        key, arrow, value = line.strip().partition("=>")
        if not arrow:
            raise ValueError("malformed lowered FIRRTL memory property")
        key, value = key.strip(), value.strip().split("@[", 1)[0].strip()
        if key in {"data-type", "depth"}:
            if key in selected:
                raise ValueError("duplicate lowered FIRRTL memory type or depth")
            selected[key] = value
    depth, typ = selected.get("depth"), selected.get("data-type")
    if depth is None or not depth.isdecimal() or int(depth) <= 0 or not typ:
        raise ValueError("lowered FIRRTL memory needs explicit positive depth and data type")
    # This is a structural observation string, never replacement FIRRTL.
    # Existing type parsing preserves an opaque/noninteger row as unknown.
    return header[: len(header) - len(header.lstrip())] + lhs + " : " + typ + " [" + depth + "] " + annotation + "\n"


def observed_memory_lines(lines: Iterable[str]) -> Iterator[str]:
    """Yield original lines, joining only exact lowered ``mem`` property blocks."""
    header = None
    properties = []
    indentation = 0
    for raw in lines:
        stripped = raw.strip()
        level = len(raw) - len(raw.lstrip())
        if header is not None:
            if stripped and level > indentation:
                properties.append(raw)
                continue
            if not stripped:
                continue
            yield _declaration(header, properties)
            header, properties = None, []
        if stripped.startswith("mem "):
            header, indentation = raw, level
        else:
            yield raw
    if header is not None:
        yield _declaration(header, properties)

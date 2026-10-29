"""Canonical JSON bytes and JSON file writers: one definition for every content-addressed document.

``canonical_json`` is the byte string merlin hashes: keys sorted, no insignificant whitespace, non-ASCII
escaped, and NaN/Infinity REFUSED -- JSON cannot represent them, so ``json.dumps`` would emit a
non-standard token that another reader rejects or re-hashes differently. Measured 2026-09-14: ten modules
defined their own ``_canonical_sha`` and eleven a ``_canonical``/``_canonical_json``. The ASCII and UTF-8
encoders among them produce identical bytes on every input (``json.dumps`` escapes non-ASCII by default),
so they collapse here with no change to any persisted digest. The genuinely different contracts keep an
explicit flag: ``ensure_ascii=False`` (raw UTF-8), ``trailing_newline=True`` (a newline-terminated file
body) and ``allow_nan=True`` (the lax spelling a caller used before, preserved rather than silently
tightened).

Named ``jsonio`` rather than ``json`` so it can never shadow the standard library.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from merlin.common.digest import sha256_bytes


def canonical_json(
    value: Any, *, ensure_ascii: bool = True, allow_nan: bool = False, trailing_newline: bool = False
) -> bytes:
    """Sorted-key, separator-minimal JSON bytes of ``value`` (see the module docstring for the flags)."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=ensure_ascii, allow_nan=allow_nan)
    if trailing_newline:
        text += "\n"
    return text.encode("utf-8")


def canonical_sha256(
    value: Any, *, ensure_ascii: bool = True, allow_nan: bool = False, trailing_newline: bool = False
) -> str:
    """SHA-256 of :func:`canonical_json` with the same flags."""
    return sha256_bytes(
        canonical_json(value, ensure_ascii=ensure_ascii, allow_nan=allow_nan, trailing_newline=trailing_newline)
    )


def _object_key(key: Any) -> str:
    """The member name ``json.dumps`` writes for ``key``."""
    if isinstance(key, str):
        return key
    if key is True or key is False or key is None:
        return json.dumps(key)
    if isinstance(key, (int, float)):
        return json.dumps(key)
    raise TypeError(f"keys must be str, int, float, bool or None, not {type(key).__name__}")


def _holds_no_object(items: list | tuple) -> bool:
    if any(isinstance(item, dict) for item in items):
        return False
    first = items[0] if items else None
    return not isinstance(first, (list, tuple)) or _holds_no_object(first)


def readable_json(value: Any, *, indent: int = 2) -> str:
    """``json.dumps(value, indent=indent)`` with every array of plain values on one line.

    ``indent`` puts each array element on its own line, so an embedded tensor is mostly indentation:
    a 65-million-element weight matrix in a capsule result took 1 GB indented and 229 MB without it,
    and that result was rewritten for every grade. Objects keep their indented layout for a human
    reader; arrays that hold no object are written compactly. ``json.loads`` of the result equals
    ``json.loads(json.dumps(value))``, so a reader or a digest of the parsed value sees no change.
    """
    pieces: list[str] = []

    def write(item: Any, level: int) -> None:
        if isinstance(item, dict):
            if not item:
                pieces.append("{}")
                return
            inner = "\n" + " " * (indent * (level + 1))
            pieces.append("{")
            for index, (key, member) in enumerate(item.items()):
                pieces.append(("," if index else "") + inner + json.dumps(_object_key(key)) + ": ")
                write(member, level + 1)
            pieces.append("\n" + " " * (indent * level) + "}")
        elif isinstance(item, (list, tuple)):
            if _holds_no_object(item):
                pieces.append(json.dumps(item, separators=(",", ":")))
                return
            inner = "\n" + " " * (indent * (level + 1))
            pieces.append("[")
            for index, member in enumerate(item):
                pieces.append(("," if index else "") + inner)
                write(member, level + 1)
            pieces.append("\n" + " " * (indent * level) + "]")
        else:
            pieces.append(json.dumps(item))

    write(value, 0)
    return "".join(pieces)


def strict_json_equal(observed: object, expected: object) -> bool:
    """Compare closed JSON values without Python's bool/int or tuple/list aliases.

    Floats, unknown containers and non-string object keys are deliberately not
    authority-bearing values under this comparator.
    """
    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        return (
            all(type(key) is str for key in observed)
            and observed.keys() == expected.keys()
            and all(strict_json_equal(observed[key], value) for key, value in expected.items())
        )
    if type(expected) is list:
        return len(observed) == len(expected) and all(
            strict_json_equal(item, value) for item, value in zip(observed, expected, strict=True)
        )
    return type(expected) in (str, int, bool, type(None)) and observed == expected


def write_canonical_json(path: str | Path, value: Any) -> None:
    """Write ``value`` as strict canonical JSON plus a trailing newline, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def write_pretty_json(path: str | Path, value: Any, *, mkdir: bool = False) -> None:
    """Write ``value`` indented (2), key-sorted, newline-terminated, for files a human reads."""
    path = Path(path)
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

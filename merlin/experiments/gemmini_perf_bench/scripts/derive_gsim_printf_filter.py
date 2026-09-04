#!/usr/bin/env python3
"""Derive a GSIM high-volume commit-trace filter from elaborated FIRRTL.

The derivation is deliberately independent of a target name, opcode, shape, or
performance result.  It selects the unique FIRRTL printf that self-identifies
as a disassembly stream, then derives a prefix long enough to include its first
format conversion and the following static separator.  The emitted record is
intended to be pinned alongside the filter shim and compile argv in the GSIM
build receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "merlin.gsim-printf-filter-derivation.v1"
_DISASSEMBLY_MARKER = "DASM("


class DerivationError(ValueError):
    """The FIRRTL does not identify one safely suppressible commit trace."""


@dataclass(frozen=True)
class FirrtlPrintf:
    line: int
    format: str
    source_locator: str | None


def _decode_firrtl_string(text: str, quote: int) -> tuple[str, int]:
    if quote >= len(text) or text[quote] != '"':
        raise DerivationError("FIRRTL printf format does not begin with a quote")
    chars: list[str] = []
    index = quote + 1
    escapes = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}
    while index < len(text):
        char = text[index]
        if char == '"':
            return "".join(chars), index + 1
        if char != "\\":
            chars.append(char)
            index += 1
            continue
        index += 1
        if index >= len(text):
            raise DerivationError("unterminated escape in FIRRTL printf format")
        escaped = text[index]
        if escaped not in escapes:
            raise DerivationError(
                f"unsupported FIRRTL printf escape \\{escaped}; refusing to guess")
        chars.append(escapes[escaped])
        index += 1
    raise DerivationError("unterminated FIRRTL printf format")


def _parse_printf(line: str, line_number: int) -> FirrtlPrintf | None:
    statement = line.lstrip()
    if not statement.startswith("printf("):
        return None

    quote = statement.find('"')
    if quote < 0:
        raise DerivationError(f"line {line_number}: FIRRTL printf has no format string")
    format_string, after_quote = _decode_firrtl_string(statement, quote)

    depth = 0
    in_string = False
    escaped = False
    closed = False
    for char in statement[:after_quote]:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
    for char in statement[after_quote:]:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                closed = True
                break
        if depth < 0:
            break
    if not closed or in_string or depth != 0:
        raise DerivationError(
            f"line {line_number}: printf is not one complete FIRRTL statement")

    locator: str | None = None
    _, marker, tail = statement.rpartition("@[")
    if marker:
        locator_text, closing, _ = tail.partition("]")
        if not closing or not locator_text:
            raise DerivationError(f"line {line_number}: malformed FIRRTL source locator")
        locator = locator_text
    return FirrtlPrintf(line=line_number, format=format_string, source_locator=locator)


def _derived_prefix(format_string: str) -> str:
    first = format_string.find("%")
    if first <= 0 or first + 1 >= len(format_string):
        raise DerivationError("commit trace lacks a stable literal before its first conversion")
    if format_string[first + 1] not in ("d", "x", "c"):
        raise DerivationError("commit trace begins with an unsupported GSIM conversion")
    second = format_string.find("%", first + 2)
    if second < 0:
        raise DerivationError("commit trace needs at least two conversions for a safe prefix")
    prefix = format_string[:second]
    if len(prefix) == first + 2:
        raise DerivationError("commit trace has no static separator after its first conversion")
    return prefix


def derive(firrtl: Path) -> dict[str, object]:
    printfs: list[FirrtlPrintf] = []
    with firrtl.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            parsed = _parse_printf(line, line_number)
            if parsed is not None:
                printfs.append(parsed)
    candidates = [row for row in printfs if _DISASSEMBLY_MARKER in row.format]
    if len(candidates) != 1:
        raise DerivationError(
            "expected exactly one FIRRTL disassembly printf, found "
            f"{len(candidates)}")
    selected = candidates[0]
    prefix = _derived_prefix(selected.format)
    collisions = [row for row in printfs if row.format.startswith(prefix)]
    if collisions != [selected]:
        raise DerivationError(
            "derived printf prefix is not unique within the elaborated FIRRTL")

    raw = firrtl.read_bytes()
    return {
        "schema_version": SCHEMA,
        "firrtl": {
            "path": str(firrtl.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "n_bytes": len(raw),
        },
        "selection": {
            "rule": "unique_printf_with_disassembly_marker",
            "marker": _DISASSEMBLY_MARKER,
            "line": selected.line,
            "source_locator": selected.source_locator,
            "format": selected.format,
            "suppress_prefix": prefix,
        },
        "inventory": {"printf_count": len(printfs), "candidate_count": 1},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firrtl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = derive(args.firrtl.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

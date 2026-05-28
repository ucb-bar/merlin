#!/usr/bin/env python3
"""Extract an IREE #util.composite<...xi8> constant arena to raw bytes.

The per-dispatch HAL wrappers emitted by IREE reference model constants as
subspans of a packed constant arena. The arena is visible in the stream/HAL
phase MLIR as:

  #composite_of_<N>b = #util.composite<Nxi8, [
      dense<[...]> : tensor<...xi32>,
      dense<"0x..."> : tensor<...xi8>,
      dense<0> : vector<...xi8>,
  ]>

This tool reconstructs the packed raw buffer in declaration order. It is meant
for real dispatch replay: the output file is the constant binding passed to
standalone per-dispatch VMFBs, replacing the previous zero-filled placeholder.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import struct
from collections.abc import Iterable

_HEADER_RE = re.compile(
    r"#(?P<name>composite[^=\s]*)\s*=\s*#util\.composite<(?P<size>\d+)xi8,\s*\[",
)
_DENSE_HEX_RE = re.compile(
    r'dense<"0x(?P<hex>[0-9A-Fa-f\s]+)">\s*:\s*(?P<type>\S+)',
    re.DOTALL,
)
_DENSE_SPLAT_RE = re.compile(
    r"dense<(?P<value>-?\d+)>\s*:\s*(?P<type>\S+)",
)
_DENSE_LIST_RE = re.compile(
    r"dense<\[(?P<values>[^\]]*)\]>\s*:\s*(?P<type>\S+)",
    re.DOTALL,
)
_TYPE_RE = re.compile(r"(?:tensor|vector)<(?P<shape>(?:\d+x)*)(?P<elem>[a-z]?\d+)>")


def _find_matching_bracket(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unterminated composite body")


def _iter_composites(text: str) -> Iterable[tuple[str, int, str]]:
    for match in _HEADER_RE.finditer(text):
        body_start = text.find("[", match.end() - 1)
        body_end = _find_matching_bracket(text, body_start)
        yield match.group("name"), int(match.group("size")), text[body_start + 1 : body_end]


def _num_elements(type_text: str) -> tuple[int, str]:
    match = _TYPE_RE.search(type_text)
    if not match:
        raise ValueError(f"unsupported dense type: {type_text}")
    shape_text = match.group("shape").rstrip("x")
    count = 1
    if shape_text:
        for dim in shape_text.split("x"):
            count *= int(dim)
    return count, match.group("elem")


def _pack_int(value: int, elem: str) -> bytes:
    if elem in ("i8", "si8", "ui8"):
        return struct.pack("b" if elem != "ui8" else "B", value)
    if elem in ("i16", "si16", "ui16"):
        return struct.pack("<h" if elem != "ui16" else "<H", value)
    if elem in ("i32", "si32", "ui32"):
        return struct.pack("<i" if elem != "ui32" else "<I", value)
    if elem in ("i64", "si64", "ui64"):
        return struct.pack("<q" if elem != "ui64" else "<Q", value)
    raise ValueError(f"unsupported integer element type: {elem}")


def _split_top_level_dense_entries(body: str) -> list[str]:
    entries: list[str] = []
    cur: list[str] = []
    angle = 0
    square = 0
    in_string = False
    prev = ""
    for ch in body:
        if ch == '"' and prev != "\\":
            in_string = not in_string
        if not in_string:
            if ch == "<":
                angle += 1
            elif ch == ">":
                angle -= 1
            elif ch == "[":
                square += 1
            elif ch == "]":
                square -= 1
            elif ch == "," and angle == 0 and square == 0:
                text = "".join(cur).strip()
                if text:
                    entries.append(text)
                cur = []
                prev = ch
                continue
        cur.append(ch)
        prev = ch
    text = "".join(cur).strip()
    if text:
        entries.append(text)
    return entries


def _decode_entry(entry: str) -> bytes:
    hex_match = _DENSE_HEX_RE.fullmatch(entry.strip())
    if hex_match:
        return bytes.fromhex("".join(hex_match.group("hex").split()))

    splat_match = _DENSE_SPLAT_RE.fullmatch(entry.strip())
    if splat_match:
        count, elem = _num_elements(splat_match.group("type"))
        return _pack_int(int(splat_match.group("value")), elem) * count

    list_match = _DENSE_LIST_RE.fullmatch(entry.strip())
    if list_match:
        count, elem = _num_elements(list_match.group("type"))
        raw_values = [v.strip() for v in list_match.group("values").split(",")]
        values = [int(v) for v in raw_values if v]
        if len(values) != count:
            raise ValueError(f"dense list has {len(values)} elements but type expects {count}: {entry[:120]}")
        return b"".join(_pack_int(v, elem) for v in values)

    raise ValueError(f"unsupported dense entry: {entry[:200]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-mlir", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--composite-name",
        default=None,
        help="Specific composite symbol to extract. Defaults to the first composite.",
    )
    args = parser.parse_args(argv)

    text = args.phase_mlir.read_text()
    selected: tuple[str, int, str] | None = None
    for composite in _iter_composites(text):
        if args.composite_name is None or composite[0] == args.composite_name:
            selected = composite
            break
    if selected is None:
        raise SystemExit(f"no matching #util.composite found in {args.phase_mlir}")

    name, expected_size, body = selected
    data = bytearray()
    for entry in _split_top_level_dense_entries(body):
        data.extend(_decode_entry(entry))
    if len(data) != expected_size:
        raise SystemExit(f"{name}: decoded {len(data)} bytes, expected {expected_size} bytes")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"wrote {args.out} ({len(data)} bytes from {name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

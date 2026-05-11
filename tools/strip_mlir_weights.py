#!/usr/bin/env python3
"""Strip embedded weight data from MLIR files.

Removes the ``{-#  dialect_resources: {    builtin: {`` section and
everything after it, which contains dense binary weight blobs that make
files hundreds of megabytes.  The resulting file keeps all ops, types,
and structure intact and is small enough to load in an editor or script.

Usage:
    python tools/strip_mlir_weights.py <file_or_directory> [--suffix .stripped]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

RESOURCE_MARKER = re.compile(r"^\{-#\s+dialect_resources:")


def strip_file(path: Path, suffix: str) -> Path | None:
    """Strip weights from *path*, writing to ``path.with_suffix(suffix)``.

    Returns the output path on success, ``None`` if no marker was found.
    """
    out_path = path.with_suffix(suffix + path.suffix)
    with open(path, encoding="utf-8") as f:
        lines: list[str] = []
        for line in f:
            if RESOURCE_MARKER.match(line):
                break
            lines.append(line)
        else:
            # Marker never found — file has no embedded weights.
            return None

    # Remove trailing blank lines before the marker.
    while lines and lines[-1].strip() == "":
        lines.pop()
    lines.append("\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        type=Path,
        help="MLIR file or directory containing .mlir files",
    )
    parser.add_argument(
        "--suffix",
        default=".stripped",
        help="Suffix inserted before .mlir for output files (default: .stripped)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original files instead of creating copies",
    )
    args = parser.parse_args()

    targets: list[Path] = []
    if args.target.is_dir():
        targets = sorted(args.target.rglob("*.mlir"))
    elif args.target.is_file():
        targets = [args.target]
    else:
        print(f"Error: {args.target} does not exist")
        return 1

    for path in targets:
        if args.in_place:
            # Read, strip, overwrite.
            text = path.read_text(encoding="utf-8")
            match = RESOURCE_MARKER.search(text)
            if match:
                stripped = text[: match.start()].rstrip() + "\n"
                path.write_text(stripped, encoding="utf-8")
                print(f"  stripped (in-place): {path}")
            else:
                print(f"  no weights: {path}")
        else:
            out = strip_file(path, args.suffix)
            if out:
                print(f"  {path} -> {out}")
            else:
                print(f"  no weights: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

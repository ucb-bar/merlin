"""Preserve raw MLIR dense-element bit patterns across xDSL text round trips."""

from __future__ import annotations


class DenseLiteralError(ValueError):
    """A round trip changed the order or type of dense constants."""


def preserve_raw_dense_bits(source: str, reprinted: str) -> str:
    """Restore hex splats that xDSL parses as integers instead of floating bit patterns.

    In MLIR, ``dense<0xFF800000> : tensor<...xf32>`` is negative infinity. The xDSL parser
    currently reads the bits as integer 4286578688 and prints a finite decimal. Match dense
    constants by operation order and shaped type; refuse if they cannot be paired exactly.
    """

    def dense_constants(lines: list[str]) -> list[tuple[int, int, int, str, str]]:
        rows = []
        for line_number, line in enumerate(lines):
            if "arith.constant" not in line or "dense<" not in line:
                continue
            start = line.index("dense<")
            end = line.find(">", start)
            type_starts = [i for kind in ("tensor<", "vector<") if (i := line.find(kind, end)) >= 0]
            type_start = min(type_starts, default=-1)
            type_end = line.find(">", type_start)
            if end < 0 or type_start < 0 or type_end < 0:
                raise DenseLiteralError("cannot locate a dense constant's shaped type")
            rows.append((line_number, start, end + 1, line[start + 6 : end], line[type_start : type_end + 1]))
        return rows

    original_lines = source.splitlines()
    reprinted_lines = reprinted.splitlines()
    originals = dense_constants(original_lines)
    selected = dense_constants(reprinted_lines)
    if len(originals) != len(selected):
        raise DenseLiteralError("round trip changed the number of dense constants")
    for (_, _, _, raw, original_type), (line_number, start, end, _, selected_type) in zip(
        originals, selected, strict=True
    ):
        if original_type != selected_type:
            raise DenseLiteralError("round trip reordered or retyped dense constants")
        if not raw.startswith("0x") or len(raw) <= 2 or not all(digit in "0123456789abcdefABCDEF" for digit in raw[2:]):
            continue
        reprinted_lines[line_number] = (
            reprinted_lines[line_number][:start] + f"dense<{raw}>" + reprinted_lines[line_number][end:]
        )
    return "\n".join(reprinted_lines) + "\n"

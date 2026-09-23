"""How many cycles a systolic array is occupied issuing a program's contractions.

WHY THIS EXISTS. A scoreboard built from movement bytes and launch counts cannot see mesh
efficiency. Measured on one whole model: changing which operand is held stationary moved the
array's issue count by 25% -- 21,383,000 cycles to 16,050,000, with idle column-slots falling from
25.30% to 0.48% -- while DRAM bytes per operand role stayed BIT-IDENTICAL and the launch count did
not change at all. Every field that scoreboard tracked was unmoved. A loop blind to this term
treats array efficiency as free and will spend it without noticing.

The count is the loop the sequencer itself runs. For a tile of `rows` streamed rows against a
`depth x cols` stationary block, an array of `array_rows x array_cols` issues
`rows * ceil(depth/array_rows) * ceil(cols/array_cols)` -- so a partial `cols` wastes array columns
for the whole tile while a partial `rows` costs nothing beyond the rows themselves. That asymmetry
is the entire reason operand orientation matters, and it is invisible in any byte or launch count.

WHAT IT REFUSES TO DO. A descriptor whose geometry cannot be read is COUNTED AND REPORTED, never
skipped. Silently ignoring an unreadable descriptor under-reports occupancy and makes the metric
look better the less the caller understands its own program -- the failure mode where a check that
finds nothing reports success.

Target-neutral: array dimensions are parameters, derived by the caller from the target's own facts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["tile_issue_cycles", "mesh_occupancy"]


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-int(numerator) // int(denominator))


def tile_issue_cycles(rows: int, depth: int, cols: int, *, array_rows: int, array_cols: int) -> int:
    """Issue cycles for one tile, and the ideal it would take with no partial-block waste."""
    if min(rows, depth, cols) < 0:
        raise ValueError(f"negative extent: rows={rows} depth={depth} cols={cols}")
    if min(array_rows, array_cols) < 1:
        raise ValueError(f"array dimensions must be positive: {array_rows}x{array_cols}")
    return int(rows) * _ceil_div(depth, array_rows) * _ceil_div(cols, array_cols)


def mesh_occupancy(tiles: Iterable[Mapping[str, Any]], *, array_rows: int, array_cols: int) -> dict[str, Any]:
    """Total issue cycles over `tiles`, with the idle-slot share and unreadable-tile census.

    Each tile supplies `rows`, `depth` and `cols`. `idle_slot_share` is the fraction of occupied
    array slots doing no useful work -- the direct measure of partial-block waste, and the term an
    orientation or tiling change actually moves.
    """
    total = ideal = unreadable = readable = 0
    for tile in tiles:
        try:
            rows, depth, cols = (int(tile["rows"]), int(tile["depth"]), int(tile["cols"]))
        except (KeyError, TypeError, ValueError):
            unreadable += 1
            continue
        readable += 1
        total += tile_issue_cycles(rows, depth, cols, array_rows=array_rows, array_cols=array_cols)
        # Slots that would be occupied if every block were full: the useful work, in slot-cycles.
        ideal += rows * depth * cols
    slots = total * array_rows * array_cols
    return {
        "schema": "mesh_occupancy_v1",
        "array": {"rows": array_rows, "cols": array_cols},
        "tiles_read": readable,
        # Reported, never silently dropped: an unreadable descriptor understates occupancy, which
        # would make the metric improve as the caller's understanding of its program got worse.
        "tiles_unreadable": unreadable,
        "issue_cycles": total,
        "useful_slot_cycles": ideal,
        "idle_slot_share": round(1.0 - ideal / slots, 6) if slots else None,
        "complete": unreadable == 0,
    }

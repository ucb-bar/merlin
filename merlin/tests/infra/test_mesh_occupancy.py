"""Array occupancy: the term a byte count and a launch count cannot see.

Measured on one whole model: changing which operand is stationary moved issue cycles 21,383,000 ->
16,050,000 (-24.94%) and idle column-slots 25.30% -> 0.48%, while per-role DRAM bytes were
bit-identical and the launch count was unchanged at 3787. Every field the loop scored was unmoved.
"""

from __future__ import annotations

import pytest

from merlin.perf.mesh_occupancy import mesh_occupancy, tile_issue_cycles

DIM = 16


def test_a_full_tile_wastes_nothing() -> None:
    assert tile_issue_cycles(64, 16, 16, array_rows=DIM, array_cols=DIM) == 64


def test_a_partial_column_block_costs_a_whole_block() -> None:
    """cols=7 on a 16-wide array occupies all 16 columns to do 7 columns of work."""
    assert tile_issue_cycles(64, 16, 7, array_rows=DIM, array_cols=DIM) == 64
    assert tile_issue_cycles(64, 16, 16, array_rows=DIM, array_cols=DIM) == 64


def test_the_asymmetry_between_rows_and_columns() -> None:
    """A partial `rows` costs only its rows; a partial `cols` costs a whole block. This asymmetry
    is why operand orientation moves the term at all."""
    narrow_rows = tile_issue_cycles(7, 256, 256, array_rows=DIM, array_cols=DIM)
    narrow_cols = tile_issue_cycles(256, 256, 7, array_rows=DIM, array_cols=DIM)
    assert narrow_rows == 7 * 16 * 16
    assert narrow_cols == 256 * 16 * 1
    assert narrow_cols > narrow_rows


def test_idle_share_is_zero_when_every_block_is_full() -> None:
    out = mesh_occupancy([{"rows": 64, "depth": 32, "cols": 16}], array_rows=DIM, array_cols=DIM)
    assert out["idle_slot_share"] == 0.0
    assert out["complete"] is True and out["tiles_unreadable"] == 0


def test_idle_share_reflects_partial_column_waste() -> None:
    """cols=7 of 16 leaves 9/16 of the columns idle."""
    out = mesh_occupancy([{"rows": 64, "depth": 16, "cols": 7}], array_rows=DIM, array_cols=DIM)
    assert out["idle_slot_share"] == pytest.approx(1 - 7 / 16, abs=1e-6)


def test_an_unreadable_tile_is_counted_not_skipped() -> None:
    """Dropping it silently would make occupancy improve as understanding got worse."""
    out = mesh_occupancy([{"rows": 64, "depth": 16, "cols": 16}, {"rows": 1}, {}], array_rows=DIM, array_cols=DIM)
    assert out["tiles_read"] == 1 and out["tiles_unreadable"] == 2
    assert out["complete"] is False


def test_totals_sum_over_tiles() -> None:
    tiles = [{"rows": 8, "depth": 16, "cols": 16}] * 10
    out = mesh_occupancy(tiles, array_rows=DIM, array_cols=DIM)
    assert out["issue_cycles"] == 80 and out["tiles_read"] == 10


def test_invalid_array_dimensions_are_refused() -> None:
    with pytest.raises(ValueError, match="array dimensions"):
        tile_issue_cycles(1, 1, 1, array_rows=0, array_cols=DIM)

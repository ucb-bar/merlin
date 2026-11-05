"""Wide transfers exactly join complete and partial preexisting tile loads."""

from dataclasses import replace

import pytest

from merlin.compile.scheduling.tile_load_coalescing import TileLoadCoalescing


@pytest.mark.parametrize("columns", [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129, 256, 2048])
@pytest.mark.parametrize("rows,stride", [(1, 16), (7, 16), (16, 16), (16, 256)])
def test_every_original_row_lane_and_source_index_is_preserved(columns, rows, stride):
    base, tile = 4096, 16
    capacity = base + (columns + 15) // 16 * stride
    plan = TileLoadCoalescing(rows, columns, tile, 64, base, stride, capacity)
    observed = []
    local_samples = set()
    for first, count, local in plan.commands():
        assert 0 < count <= 64 and first % tile == 0
        for row in range(rows):
            for offset in range(count):
                column = first + offset
                destination = local + offset // tile * stride + row, offset % tile
                assert destination == plan.location(row, column)
                assert destination not in local_samples
                local_samples.add(destination)
                observed.append((row, column, destination))
    expected = [(row, column, (base + column // tile * stride + row, column % tile))
                for first in range(0, columns, tile) for row in range(rows)
                for column in range(first, min(columns, first + tile))]
    assert sorted(observed) == sorted(expected)
    assert len(observed) == rows * columns
    assert max(row for row, _ in local_samples) < capacity


def test_unknown_overlap_capacity_and_element_indices_refuse():
    plan = TileLoadCoalescing(16, 65, 16, 64, 4096, 256, 5376)
    for change in ({"rows": 17}, {"block_stride": 15}, {"max_columns": 63}, {"columns": True},
                   {"local_base": -1}, {"capacity_rows": 5135}, {"columns": 1 << 63}):
        with pytest.raises(ValueError, match="complete bounded"):
            replace(plan, **change)
    for row, column in ((-1, 0), (16, 0), (0, 65), (True, 0)):
        with pytest.raises(ValueError, match="logical rectangle"):
            plan.location(row, column)

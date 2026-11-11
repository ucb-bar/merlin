"""Transport accounting and independent capacity/expansion refusals."""

import pytest

from merlin.semantic_compiler.operand_reuse import plan_operand_reuse


@pytest.mark.parametrize("tiles,tile_bytes,consumers", [(1, 3, 2), (7, 80, 19), (11, 128, 4)])
def test_complete_operand_reads_once_per_reuse_scope(tiles, tile_bytes, consumers):
    size = tiles * tile_bytes
    plan = plan_operand_reuse(
        tile_count=tiles, tile_bytes=tile_bytes, consumers=consumers, capacity_bytes=size, max_tiles=tiles
    )
    assert plan is not None
    assert plan.resident_bytes == size
    original_reads = [(consumer, tile) for consumer in range(consumers) for tile in range(tiles)]
    cached_reads = list(range(tiles))
    assert plan.avoided_read_bytes == (len(original_reads) - len(cached_reads)) * tile_bytes
    for capacity, maximum, uses in ((size - 1, tiles, consumers), (size, tiles - 1, consumers), (size, tiles, 1)):
        assert (
            plan_operand_reuse(
                tile_count=tiles, tile_bytes=tile_bytes, consumers=uses, capacity_bytes=capacity, max_tiles=maximum
            )
            is None
        )


@pytest.mark.parametrize("field", ["tile_count", "tile_bytes", "consumers", "capacity_bytes", "max_tiles"])
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_geometry_and_limits_are_errors(field, value):
    arguments = dict(tile_count=3, tile_bytes=7, consumers=2, capacity_bytes=21, max_tiles=3)
    arguments[field] = value
    with pytest.raises(ValueError):
        plan_operand_reuse(**arguments)


def test_zero_budget_declines_but_zero_geometry_is_invalid():
    arguments = dict(tile_count=3, tile_bytes=7, consumers=2, capacity_bytes=21, max_tiles=3)
    for field in ("capacity_bytes", "max_tiles"):
        assert plan_operand_reuse(**{**arguments, field: 0}) is None
    for field in ("tile_count", "tile_bytes", "consumers"):
        with pytest.raises(ValueError):
            plan_operand_reuse(**{**arguments, field: 0})

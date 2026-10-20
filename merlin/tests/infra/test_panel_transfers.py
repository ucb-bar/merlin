"""Execute actual packed addresses against an independent matrix/panel oracle."""

from dataclasses import replace
from itertools import product
from random import Random

import pytest

from merlin.compile.scheduling.panel_transfers import (
    AffineMatrixSource,
    PackedPanelLayout,
    PanelTransferLimits,
    check_panel_transfers,
    execute_panel_transfers,
    plan_packed_panels,
)
from merlin.compile.scheduling.storage_contracts import StorageWindow


def _source(shape, element_bytes, style="rows"):
    rows, columns = shape
    row_stride, column_stride = {
        "rows": (columns * element_bytes + 8, element_bytes),
        "transpose": (element_bytes, rows * element_bytes + 8),
        "broadcast_rows": (0, element_bytes),
        "broadcast_columns": (element_bytes, 0),
    }[style]
    allocation_start, first = 32, 48
    extent = first + (rows - 1) * row_stride + (columns - 1) * column_stride + element_bytes
    source = AffineMatrixSource(
        "external",
        shape,
        element_bytes,
        first,
        row_stride,
        column_stride,
        allocation_start,
        extent - allocation_start + 16,
    )
    # Distinct neighbouring values expose source-address corruption.
    payload = bytes((index * 37 + index // 251) % 256 for index in range(source.allocation_bytes))
    return source, payload


def _layout(source, tile=(4, 4), panel=(2, 3), alignment=8):
    return PackedPanelLayout(
        source.shape,
        source.element_bytes,
        tile,
        panel,
        alignment,
        StorageWindow("operand", "private", 96, 1_000_000, 16, 1_000_096, "invocation_private"),
    )


def _limits(**changes):
    return replace(PanelTransferLimits(16, 64, 1, 1, 1_000_000, 1_000_000, True), **changes)


def _expected_bytes(source, payload, tile, panel, alignment):
    # Independent panel coordinates and original affine source reads. This
    # oracle does not call a layout address method or project a plan request.
    height, width = tile[0] * panel[0], tile[1] * panel[1]
    stride = ((width * source.element_bytes + alignment - 1) // alignment) * alignment
    result = bytearray()
    for pr in range(0, source.shape[0], height):
        for pc in range(0, source.shape[1], width):
            for local_row in range(height):
                row = bytearray(stride)
                for local_column in range(width):
                    r, c = pr + local_row, pc + local_column
                    if r < source.shape[0] and c < source.shape[1]:
                        address = source.first_byte + r * source.row_stride_bytes + c * source.column_stride_bytes
                        start = address - source.allocation_start
                        first = local_column * source.element_bytes
                        row[first : first + source.element_bytes] = payload[start : start + source.element_bytes]
                result.extend(row)
    return bytes(result)


def _run(source, payload, layout, transfers):
    before, after = 17, 29
    destination = bytearray([0xD3]) * (before + layout.total_bytes + after)
    execute_panel_transfers(
        transfers,
        source=payload,
        source_start=source.allocation_start,
        destination=destination,
        destination_start=layout.storage.start - before,
    )
    assert destination[:before] == bytes([0xD3]) * before
    assert destination[-after:] == bytes([0xD3]) * after
    return bytes(destination[before:-after])


@pytest.mark.parametrize("shape,element_bytes", [((1, 1), 1), ((7, 13), 2), ((16, 23), 4), ((9, 6), 1)])
@pytest.mark.parametrize(
    "tile,panel,alignment,max_rows,max_width",
    [
        ((2, 3), (1, 1), 1, 1, 5),
        ((4, 4), (2, 3), 8, 3, 17),
        ((8, 5), (1, 4), 16, 16, 64),
    ],
)
@pytest.mark.parametrize("style", ["rows", "transpose", "broadcast_rows", "broadcast_columns"])
def test_exact_matrix_bytes_tails_and_padding_across_geometries(
    shape, element_bytes, tile, panel, alignment, max_rows, max_width, style
):
    source, payload = _source(shape, element_bytes, style)
    layout = _layout(source, tile, panel, alignment)
    plan = plan_packed_panels(source, layout, _limits(max_rows=max_rows, max_row_bytes=max_width))
    result = _run(source, payload, layout, plan.transfers)
    assert result == _expected_bytes(source, payload, tile, panel, alignment)
    receipt = check_panel_transfers(source, layout, plan.limits, plan.transfers)
    assert receipt["copied_bytes"] == shape[0] * shape[1] * element_bytes
    assert receipt["zero_bytes"] == layout.total_bytes - receipt["copied_bytes"]
    # Read actual primitive tile subviews using their declared physical stride.
    for tr, tc in product(range((shape[0] + tile[0] - 1) // tile[0]), range((shape[1] + tile[1] - 1) // tile[1])):
        first = layout.tile_first_byte(tr, tc) - layout.storage.start
        for r in range(min(tile[0], shape[0] - tr * tile[0])):
            for c in range(min(tile[1], shape[1] - tc * tile[1])):
                source_address = source.first_byte + (tr * tile[0] + r) * source.row_stride_bytes
                source_address += (tc * tile[1] + c) * source.column_stride_bytes
                expected = source_address - source.allocation_start
                actual = first + r * layout.row_stride_bytes + c * element_bytes
                assert result[actual : actual + element_bytes] == payload[expected : expected + element_bytes]


def test_packing_adjacent_tiles_enables_wider_transfers_without_changing_compute_tiles():
    source, payload = _source((8, 32), 1)
    narrow = _layout(source, (4, 4), (1, 1), 1)
    wide = _layout(source, (4, 4), (2, 8), 1)
    narrow_plan = plan_packed_panels(source, narrow, _limits())
    wide_plan = plan_packed_panels(source, wide, _limits())
    assert len(narrow_plan.transfers) == 16
    assert len(wide_plan.transfers) == 1
    assert wide_plan.transfers[0].row_bytes == 32 > wide.tile_shape[1]
    assert _run(source, payload, wide, wide_plan.transfers) == _expected_bytes(source, payload, (4, 4), (2, 8), 1)


def test_random_affine_views_and_panel_partitions_execute_exactly():
    rng = Random(91276)
    for _ in range(200):
        source, payload = _source(
            (rng.randrange(1, 18), rng.randrange(1, 25)),
            rng.choice((1, 2, 4)),
            rng.choice(("rows", "transpose", "broadcast_rows", "broadcast_columns")),
        )
        tile = (rng.randrange(1, 7), rng.randrange(1, 7))
        panel = (rng.randrange(1, 5), rng.randrange(1, 5))
        alignment = rng.choice((1, 3, 8))
        layout = _layout(source, tile, panel, alignment)
        limits = _limits(max_rows=rng.choice((1, 2, 7)), max_row_bytes=rng.choice((1, 3, 5, 17, 64)))
        plan = plan_packed_panels(source, layout, limits)
        assert _run(source, payload, layout, plan.transfers) == _expected_bytes(source, payload, tile, panel, alignment)


def test_width_splitting_preserves_address_alignment_and_exact_row_limits():
    source, payload = _source((4, 16), 1)
    layout = _layout(source, (4, 4), (1, 4), 4)
    limits = _limits(max_rows=2, max_row_bytes=6, address_alignment=4)
    plan = plan_packed_panels(source, layout, limits)
    assert len(plan.transfers) == 8
    assert all(operation.row_bytes == 4 and operation.rows == 2 for operation in plan.transfers)
    assert _run(source, payload, layout, plan.transfers) == _expected_bytes(source, payload, (4, 4), (1, 4), 4)


@pytest.mark.parametrize("missing", ["max_source_stride_bytes", "max_destination_stride_bytes"])
def test_missing_stride_capability_retains_single_row_transfers(missing):
    source, payload = _source((8, 16), 1)
    layout = _layout(source, (4, 4), (2, 4), 1)
    plan = plan_packed_panels(source, layout, _limits(**{missing: 0}))
    assert len(plan.transfers) == 8
    assert all(operation.rows == 1 for operation in plan.transfers)
    assert _run(source, payload, layout, plan.transfers) == _expected_bytes(source, payload, (4, 4), (2, 4), 1)


def test_unknown_zero_path_is_allowed_only_when_layout_needs_no_padding():
    source, _ = _source((8, 16), 1)
    plan_packed_panels(source, _layout(source, (4, 4), (2, 4), 1), _limits(zero_fill=False))
    with pytest.raises(ValueError, match="zero-fill"):
        plan_packed_panels(source, _layout(source, (4, 4), (2, 5), 1), _limits(zero_fill=False))


@pytest.mark.parametrize(
    "change",
    [
        {"source_byte": 49},
        {"destination_byte": 97},
        {"source_stride_bytes": 25},
        {"destination_stride_bytes": 17},
        {"source_byte": None},
        {"rows": 7},
    ],
)
def test_independent_actual_address_checker_rejects_corrupted_rectangles(change):
    source, payload = _source((8, 16), 1)
    layout = _layout(source, (4, 4), (2, 4), 1)
    plan = plan_packed_panels(source, layout, _limits())
    assert len(plan.transfers) == 1
    altered = (replace(plan.transfers[0], **change),)
    with pytest.raises(ValueError):
        check_panel_transfers(source, layout, plan.limits, altered)
    # An independent executor really observes wrong bytes, instead of using a
    # saved source request that could conceal a malformed instruction address.
    if set(change) <= {"source_byte", "source_stride_bytes", "rows"}:
        assert _run(source, payload, layout, altered) != _expected_bytes(source, payload, (4, 4), (2, 4), 1)


def test_padding_copy_duplicate_and_missing_writes_are_rejected():
    source, _ = _source((7, 13), 1)
    layout = _layout(source)
    plan = plan_packed_panels(source, layout, _limits())
    zero = next(i for i, operation in enumerate(plan.transfers) if operation.source_byte is None)
    altered = list(plan.transfers)
    altered[zero] = replace(altered[zero], source_byte=source.first_byte)
    for operations in (tuple(altered), plan.transfers[:-1], plan.transfers + plan.transfers[:1]):
        with pytest.raises(ValueError):
            check_panel_transfers(source, layout, plan.limits, operations)


def test_noncontiguous_affine_columns_cannot_be_claimed_as_a_wide_copy():
    source, _ = _source((4, 8), 2, "transpose")
    layout = _layout(source, (4, 4), (1, 2), 1)
    plan = plan_packed_panels(source, layout, _limits())
    operation = next(operation for operation in plan.transfers if operation.source_byte is not None)
    assert operation.row_bytes == 2
    with pytest.raises(ValueError, match="noncontiguous"):
        check_panel_transfers(source, layout, plan.limits, (replace(operation, row_bytes=4),))


def test_storage_assignment_capacity_ownership_and_source_alias_are_explicit():
    source, _ = _source((8, 16), 1)
    layout = _layout(source, (4, 4), (2, 4), 1)
    exact = replace(layout.storage, size=layout.total_bytes, capacity=layout.storage.start + layout.total_bytes)
    replace(layout, storage=exact)
    for window in (
        replace(exact, size=layout.total_bytes - 1),
        replace(exact, capacity=None),
        replace(exact, ownership="unknown"),
        replace(exact, start=97),
    ):
        with pytest.raises(ValueError):
            replace(layout, storage=window)
    alias = replace(layout, storage=replace(layout.storage, backing=source.backing))
    with pytest.raises(ValueError, match="alias"):
        plan_packed_panels(source, alias, _limits())


def test_refuse_unaligned_tails_without_reading_neighbouring_source_bytes():
    source, _ = _source((4, 13), 1)
    layout = _layout(source, (4, 4), (1, 4), 4)
    with pytest.raises(ValueError, match="alignment"):
        plan_packed_panels(source, layout, _limits(row_bytes_multiple=4))


def test_large_layout_refuses_before_enumeration_and_checker_has_its_own_budget():
    source = AffineMatrixSource("external", (10**9, 1), 1, 0, 1, 1, 0, 10**9)
    storage = StorageWindow("operand", "private", 0, 10**9, 1, 10**9, "invocation_private")
    layout = PackedPanelLayout(source.shape, 1, (1, 1), (1, 1), 1, storage)
    with pytest.raises(ValueError, match="planning budget"):
        plan_packed_panels(source, layout, _limits(), operation_budget=20)
    small_source, _ = _source((8, 16), 1)
    small_layout = _layout(small_source, (4, 4), (2, 4), 1)
    plan = plan_packed_panels(small_source, small_layout, _limits())
    with pytest.raises(ValueError, match="checking budget"):
        check_panel_transfers(small_source, small_layout, plan.limits, plan.transfers, checking_budget=7)


def test_negative_overflowed_and_unknown_affine_geometry_refuses():
    source, _ = _source((4, 8), 2)
    for change in (
        {"row_stride_bytes": -1},
        {"column_stride_bytes": None},
        {"shape": (True, 8)},
        {"allocation_bytes": 1},
        {"first_byte": 2**63 - 1},
    ):
        with pytest.raises(ValueError):
            replace(source, **change)
    layout = _layout(source)
    for coordinates in ((-1, 0), (0, -1), (100, 0), (0, 100), (True, 0)):
        with pytest.raises(ValueError):
            layout.tile_first_byte(*coordinates)

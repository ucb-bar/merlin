"""Checked affine-to-panel packing and rectangular transfer aggregation.

The source is an explicit immutable affine matrix view. Storage is an assigned
invocation-private byte window, after reservations. Panel sizes are choices;
transfer widths, strides and alignment are selected capabilities, never facts
inferred from a compute tile. This module supplies no physical instruction,
completion, ownership qualification or asynchronous overlap authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm, prod

from .storage_contracts import StorageWindow


def _integers(values, *, positive=False):
    return all(type(value) is int and (value > 0 if positive else value >= 0) and value < 2**63 for value in values)


def _pair(value):
    return type(value) is tuple and len(value) == 2 and _integers(value, positive=True)


@dataclass(frozen=True)
class AffineMatrixSource:
    """A read-only matrix view, with all addresses and strides measured in bytes.

    Zero strides represent broadcast reads. A transposed view is supported by
    exchanging its row/column strides. Negative strides retain another route.
    The bounding allocation is distinct from the logical view's first byte.
    """

    backing: str
    shape: tuple[int, int]
    element_bytes: int
    first_byte: int
    row_stride_bytes: int
    column_stride_bytes: int
    allocation_start: int
    allocation_bytes: int

    def __post_init__(self):
        if type(self.backing) is not str or not self.backing or not _pair(self.shape):
            raise ValueError("panel source requires an explicit backing and positive two-dimensional shape")
        if not _integers((self.element_bytes, self.allocation_bytes), positive=True) or not _integers(
            (self.first_byte, self.row_stride_bytes, self.column_stride_bytes, self.allocation_start)
        ):
            raise ValueError("panel source requires bounded nonnegative affine byte addresses")
        last = self.first_byte + (self.shape[0] - 1) * self.row_stride_bytes
        last += (self.shape[1] - 1) * self.column_stride_bytes + self.element_bytes
        if self.first_byte < self.allocation_start or last > self.allocation_start + self.allocation_bytes:
            raise ValueError("affine panel source exceeds its declared allocation")
        if max(last, self.allocation_start + self.allocation_bytes) >= 2**63:
            raise ValueError("affine panel source exceeds signed address arithmetic")


@dataclass(frozen=True)
class PanelTransferLimits:
    """Selected synchronous rectangular copy/zero capabilities in byte units.

    Every transferred row must satisfy address and length alignment. A stride
    limit of zero admits no multi-row operation of that kind. A single-row
    operation has no stride requirement. Zero fill means exact byte zeros.
    These declarations must be bound to actual instruction semantics by the
    target consumer; they are not extracted from a preferred schedule.
    """

    max_rows: int
    max_row_bytes: int
    address_alignment: int
    row_bytes_multiple: int
    max_source_stride_bytes: int
    max_destination_stride_bytes: int
    zero_fill: bool

    def __post_init__(self):
        if not _integers(
            (self.max_rows, self.max_row_bytes, self.address_alignment, self.row_bytes_multiple), positive=True
        ) or not _integers((self.max_source_stride_bytes, self.max_destination_stride_bytes)):
            raise ValueError("panel transfers require explicit positive limits and nonnegative stride bounds")
        if type(self.zero_fill) is not bool:
            raise ValueError("panel zero-fill support must be explicit")


@dataclass(frozen=True)
class PackedPanelLayout:
    """Panel-major storage, with rows shared by adjacent primitive column tiles.

    The layout covers a whole logical matrix. Panel tails and row-alignment
    gaps contain zeros. A compute consumer must separately prove that its
    addressing can read the tile subviews returned by ``tile_first_byte``.
    Storage capacity alone does not establish that addressing capability.
    """

    shape: tuple[int, int]
    element_bytes: int
    tile_shape: tuple[int, int]
    panel_tiles: tuple[int, int]
    row_alignment: int
    storage: StorageWindow

    def __post_init__(self):
        if not all(_pair(value) for value in (self.shape, self.tile_shape, self.panel_tiles)) or not _integers(
            (self.element_bytes, self.row_alignment), positive=True
        ):
            raise ValueError("panel layout requires positive bounded shapes, widths and row alignment")
        window = self.storage
        if (
            type(window) is not StorageWindow
            or any(type(value) is not str or not value for value in (window.backing, window.resource))
            or not _integers((window.start,))
            or not _integers((window.size, window.alignment), positive=True)
            or window.start % window.alignment
            or window.ownership != "invocation_private"
        ):
            raise ValueError("panel layout requires an aligned assigned invocation-private storage window")
        if window.capacity is None or not _integers((window.capacity,), positive=True):
            raise ValueError("panel layout requires a known physical byte capacity")
        if window.start + window.size > window.capacity or self.total_bytes > window.size:
            raise ValueError("packed panel layout exceeds assigned physical storage")
        if window.start + self.total_bytes >= 2**63:
            raise ValueError("packed panel layout exceeds signed address arithmetic")

    @property
    def panel_shape(self):
        return tuple(tile * count for tile, count in zip(self.tile_shape, self.panel_tiles, strict=True))

    @property
    def panel_counts(self):
        return tuple((extent + panel - 1) // panel for extent, panel in zip(self.shape, self.panel_shape, strict=True))

    @property
    def row_stride_bytes(self):
        width = self.panel_shape[1] * self.element_bytes
        return (width + self.row_alignment - 1) // self.row_alignment * self.row_alignment

    @property
    def panel_stride_bytes(self):
        return self.panel_shape[0] * self.row_stride_bytes

    @property
    def total_bytes(self):
        return prod(self.panel_counts) * self.panel_stride_bytes

    def tile_first_byte(self, tile_row, tile_column):
        """Address a primitive tile in the packed matrix, including edge tiles."""
        tile_counts = tuple((size + tile - 1) // tile for size, tile in zip(self.shape, self.tile_shape, strict=True))
        if not _integers((tile_row, tile_column)) or tile_row >= tile_counts[0] or tile_column >= tile_counts[1]:
            raise ValueError("requested compute tile lies outside the logical packed matrix")
        pr, lr = divmod(tile_row, self.panel_tiles[0])
        pc, lc = divmod(tile_column, self.panel_tiles[1])
        return (
            self.storage.start
            + (pr * self.panel_counts[1] + pc) * self.panel_stride_bytes
            + lr * self.tile_shape[0] * self.row_stride_bytes
            + lc * self.tile_shape[1] * self.element_bytes
        )


@dataclass(frozen=True)
class PanelTransfer:
    source_byte: int | None
    destination_byte: int
    rows: int
    row_bytes: int
    source_stride_bytes: int
    destination_stride_bytes: int


@dataclass(frozen=True)
class PackedPanelPlan:
    source: AffineMatrixSource
    layout: PackedPanelLayout
    limits: PanelTransferLimits
    transfers: tuple[PanelTransfer, ...]


def _validate_transfer(operation, limits):
    if (
        type(operation) is not PanelTransfer
        or not _integers((operation.rows, operation.row_bytes), positive=True)
        or not _integers(
            (operation.destination_byte, operation.source_stride_bytes, operation.destination_stride_bytes)
        )
    ):
        raise ValueError("invalid actual panel transfer operands")
    if operation.source_byte is not None and not _integers((operation.source_byte,)):
        raise ValueError("invalid actual panel source address")
    if operation.rows > limits.max_rows or operation.row_bytes > limits.max_row_bytes:
        raise ValueError("actual panel transfer exceeds selected rectangle limits")
    if operation.row_bytes % limits.row_bytes_multiple:
        raise ValueError("actual panel transfer length violates selected alignment")
    starts = (operation.destination_byte,) + (() if operation.source_byte is None else (operation.source_byte,))
    if any(address % limits.address_alignment for address in starts):
        raise ValueError("actual panel transfer address violates selected alignment")
    if operation.source_byte is None and (not limits.zero_fill or operation.source_stride_bytes):
        raise ValueError("actual panel zero path lacks selected exact zero-fill support")
    if operation.rows > 1:
        if (
            operation.destination_stride_bytes < operation.row_bytes
            or operation.destination_stride_bytes > limits.max_destination_stride_bytes
            or operation.destination_stride_bytes % limits.address_alignment
        ):
            raise ValueError("actual panel destination stride exceeds selected capabilities")
        if operation.source_byte is not None and (
            limits.max_source_stride_bytes == 0
            or operation.source_stride_bytes > limits.max_source_stride_bytes
            or operation.source_stride_bytes % limits.address_alignment
        ):
            raise ValueError("actual panel source stride exceeds selected capabilities")


def check_panel_transfers(source, layout, limits, transfers, *, checking_budget=1_000_000):
    """Check actual rectangles against the independent affine byte mapping.

    Each actual destination row is checked against the matrix coordinates of
    its physical panel row; no retained planner request is trusted. Sorted
    intervals prove exact coverage once, including zero tails/alignment gaps.
    The budget counts rectangle rows, not tensor elements or payload values.
    """
    if type(source) is not AffineMatrixSource or type(layout) is not PackedPanelLayout:
        raise ValueError("panel checking requires explicit source and layout contracts")
    if type(limits) is not PanelTransferLimits or not _integers((checking_budget,), positive=True):
        raise ValueError("panel checking requires selected limits and a bounded row budget")
    if source.shape != layout.shape or source.element_bytes != layout.element_bytes:
        raise ValueError("packed panel layout changes source shape or element width")
    if source.backing == layout.storage.backing and (
        source.allocation_start < layout.storage.start + layout.total_bytes
        and layout.storage.start < source.allocation_start + source.allocation_bytes
    ):
        raise ValueError("packed panel destination may alias its immutable source allocation")
    if type(transfers) is not tuple or len(transfers) > checking_budget:
        raise ValueError("actual panel transfers exceed bounded checking budget")
    intervals, copied, cleared, rows_seen = [], 0, 0, 0
    panel_h, panel_w = layout.panel_shape
    for operation in transfers:
        _validate_transfer(operation, limits)
        rows_seen += operation.rows
        if rows_seen > checking_budget:
            raise ValueError("actual panel rectangle rows exceed bounded checking budget")
        for index in range(operation.rows):
            destination = operation.destination_byte + index * operation.destination_stride_bytes
            offset = destination - layout.storage.start
            end = offset + operation.row_bytes
            if offset < 0 or end > layout.total_bytes:
                raise ValueError("actual panel transfer exceeds assigned layout extent")
            physical_row, column_byte = divmod(offset, layout.row_stride_bytes)
            if column_byte + operation.row_bytes > layout.row_stride_bytes:
                raise ValueError("actual panel transfer crosses a packed row boundary")
            panel_index, local_row = divmod(physical_row, panel_h)
            panel_row, panel_column = divmod(panel_index, layout.panel_counts[1])
            logical_row = panel_row * panel_h + local_row
            logical_column, element_byte = divmod(column_byte, source.element_bytes)
            logical_column += panel_column * panel_w
            valid = (
                min(panel_w, source.shape[1] - panel_column * panel_w) * source.element_bytes
                if logical_row < source.shape[0]
                else 0
            )
            if column_byte >= valid:
                if operation.source_byte is not None:
                    raise ValueError("actual panel padding must come from the exact zero path")
                cleared += operation.row_bytes
            else:
                if column_byte + operation.row_bytes > valid:
                    raise ValueError("actual panel copy crosses logical source into zero padding")
                expected = source.first_byte + logical_row * source.row_stride_bytes
                expected += logical_column * source.column_stride_bytes + element_byte
                actual = (
                    None
                    if operation.source_byte is None
                    else operation.source_byte + index * operation.source_stride_bytes
                )
                if actual != expected:
                    raise ValueError("actual panel transfer reads a different source byte")
                if (
                    source.column_stride_bytes != source.element_bytes
                    and element_byte + operation.row_bytes > source.element_bytes
                ):
                    raise ValueError("actual panel copy crosses noncontiguous affine source elements")
                copied += operation.row_bytes
            intervals.append((offset, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("actual panel destination has overlapping or missing byte coverage")
        cursor = end
    if cursor != layout.total_bytes:
        raise ValueError("actual panel destination has incomplete byte coverage")
    return {"copied_bytes": copied, "zero_bytes": cleared, "transfer_rows": rows_seen, "transfers": len(transfers)}


def _rectangles(rows, limits):
    """Join disjoint row segments only when their actual address steps agree."""
    current = rows[0]
    for following in rows[1:]:
        destination_step = following.destination_byte - current.destination_byte
        source_step = 0 if current.source_byte is None else following.source_byte - current.source_byte
        if current.rows > 1:
            destination_step //= current.rows
            source_step //= current.rows
        candidate = PanelTransfer(
            current.source_byte,
            current.destination_byte,
            current.rows + 1,
            current.row_bytes,
            source_step,
            destination_step,
        )
        exact_steps = (
            following.destination_byte == current.destination_byte + current.rows * destination_step
            and (
                current.source_byte is None or following.source_byte == current.source_byte + current.rows * source_step
            )
            and (
                current.rows == 1
                or (current.source_stride_bytes, current.destination_stride_bytes) == (source_step, destination_step)
            )
        )
        try:
            _validate_transfer(candidate, limits)
        except ValueError:
            exact_steps = False
        if exact_steps:
            current = candidate
        else:
            yield current
            current = following
    yield current


def plan_packed_panels(source, layout, limits, *, operation_budget=1_000_000):
    """Pack adjacent compute tiles into checked wide transfer panels.

    ``panel_tiles`` is an explicit optimization choice. Source scalar reads
    become wide copies only when its affine column stride proves contiguity.
    Both exact tails and alignment gaps are initialized without over-reading.
    Unknown/unaligned copies refuse instead of reading neighbouring bytes.
    """
    if (type(source), type(layout), type(limits)) != (AffineMatrixSource, PackedPanelLayout, PanelTransferLimits):
        raise ValueError("panel planning requires typed source, layout and transfer limits")
    if not _integers((operation_budget,), positive=True):
        raise ValueError("panel planning requires a bounded operation budget")
    if source.shape != layout.shape or source.element_bytes != layout.element_bytes:
        raise ValueError("packed panel layout changes source shape or element width")
    # Every physical row needs at least one operation, so decline before any
    # large matrix walk or allocation. Intermediate row segments are bounded.
    if layout.total_bytes // layout.row_stride_bytes > operation_budget:
        raise ValueError("packed panel rows exceed planning budget")
    panel_h, panel_w = layout.panel_shape
    width_limit = limits.max_row_bytes // limits.row_bytes_multiple * limits.row_bytes_multiple
    if width_limit == 0:
        raise ValueError("selected transfer width cannot satisfy its length alignment")
    groups, segment_count = {}, 0

    def append_segment(source_byte, destination, length):
        nonlocal segment_count
        while length:
            width = min(length, width_limit)
            if length > width_limit:
                step = lcm(limits.address_alignment, limits.row_bytes_multiple)
                width = width_limit // step * step
                if not width:
                    raise ValueError("selected transfer width cannot preserve address alignment when split")
            operation = PanelTransfer(source_byte, destination, 1, width, 0, 0)
            _validate_transfer(operation, limits)
            group = (destination % layout.row_stride_bytes, width, source_byte is None)
            groups.setdefault(group, []).append(operation)
            segment_count += 1
            if segment_count > operation_budget:
                raise ValueError("packed panel row segments exceed planning budget")
            length -= width
            destination += width
            if source_byte is not None:
                source_byte += width

    for panel_row in range(layout.panel_counts[0]):
        for panel_column in range(layout.panel_counts[1]):
            panel_index = panel_row * layout.panel_counts[1] + panel_column
            valid_columns = min(panel_w, source.shape[1] - panel_column * panel_w)
            for row in range(panel_h):
                destination = (
                    layout.storage.start + panel_index * layout.panel_stride_bytes + row * layout.row_stride_bytes
                )
                logical_row = panel_row * panel_h + row
                valid_bytes = valid_columns * source.element_bytes if logical_row < source.shape[0] else 0
                if valid_bytes:
                    address = source.first_byte + logical_row * source.row_stride_bytes
                    address += panel_column * panel_w * source.column_stride_bytes
                    if source.column_stride_bytes == source.element_bytes:
                        append_segment(address, destination, valid_bytes)
                    else:
                        for column in range(valid_columns):
                            append_segment(
                                address + column * source.column_stride_bytes,
                                destination + column * source.element_bytes,
                                source.element_bytes,
                            )
                if valid_bytes < layout.row_stride_bytes:
                    append_segment(None, destination + valid_bytes, layout.row_stride_bytes - valid_bytes)
    transfers = tuple(operation for rows in groups.values() for operation in _rectangles(rows, limits))
    check_panel_transfers(source, layout, limits, transfers, checking_budget=operation_budget)
    return PackedPanelPlan(source, layout, limits, transfers)


def execute_panel_transfers(transfers, *, source, source_start, destination, destination_start):
    """Execute actual byte addresses, independently of the planner's mapping.

    This synchronous host executor is useful for testing instruction adapters;
    it supplies no device execution or timing qualification.
    """
    if type(source) is not bytes or type(destination) is not bytearray:
        raise ValueError("panel execution requires immutable source bytes and a mutable destination")
    if not _integers((source_start, destination_start)):
        raise ValueError("panel execution requires explicit nonnegative allocation origins")
    for operation in transfers:
        if type(operation) is not PanelTransfer or not _integers((operation.rows, operation.row_bytes), positive=True):
            raise ValueError("invalid executable panel rectangle")
        if not _integers(
            (operation.destination_byte, operation.source_stride_bytes, operation.destination_stride_bytes)
        ):
            raise ValueError("invalid executable panel address or stride")
        for row in range(operation.rows):
            first = operation.destination_byte + row * operation.destination_stride_bytes - destination_start
            last = first + operation.row_bytes
            if first < 0 or last > len(destination):
                raise ValueError("executed panel write exceeds destination allocation")
            if operation.source_byte is None:
                destination[first:last] = bytes(operation.row_bytes)
            else:
                if not _integers((operation.source_byte,)):
                    raise ValueError("invalid executable panel source address")
                start = operation.source_byte + row * operation.source_stride_bytes - source_start
                if start < 0 or start + operation.row_bytes > len(source):
                    raise ValueError("executed panel read exceeds source allocation")
                destination[first:last] = source[start : start + operation.row_bytes]

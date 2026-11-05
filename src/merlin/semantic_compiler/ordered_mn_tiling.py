"""Exact outer tiles for one oriented matrix contraction.

Each output tile is an independent kernel request. The target binding supplies
qualified M/N geometry; this module only partitions row-major boundaries and
reassembles output bytes. Reduction order within each tile is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .model import KernelRequest, SemanticNode, TensorType
from .ordered_k_tiling import OrderedKTilingError, _validated_oriented_source


class OrderedMNTilingError(ValueError):
    """The source, geometry, or runtime bytes do not match an exact M/N partition."""


@dataclass(frozen=True)
class RowMajorMatrixSlice:
    tile_input: str
    source_input: str
    source_rows: int
    source_columns: int
    row_start: int
    column_start: int
    rows: int
    columns: int

    def __post_init__(self) -> None:
        if (
            not self.tile_input or not self.source_input
            or any(type(value) is not int for value in (
                self.source_rows, self.source_columns, self.row_start,
                self.column_start, self.rows, self.columns,
            ))
            or min(self.source_rows, self.source_columns, self.rows, self.columns) <= 0
            or self.row_start < 0 or self.column_start < 0
            or self.row_start + self.rows > self.source_rows
            or self.column_start + self.columns > self.source_columns
        ):
            raise OrderedMNTilingError("row-major matrix slice exceeds its source")

    def record(self) -> dict[str, int | str]:
        return {
            "tile_input": self.tile_input,
            "source_input": self.source_input,
            "source_rows": self.source_rows,
            "source_columns": self.source_columns,
            "row_start": self.row_start,
            "column_start": self.column_start,
            "rows": self.rows,
            "columns": self.columns,
        }


@dataclass(frozen=True)
class MatrixTile:
    m_start: int
    n_start: int
    request: KernelRequest
    input_slices: tuple[RowMajorMatrixSlice, RowMajorMatrixSlice]

    def record(self, *, include_valid_extent: bool = False) -> dict[str, object]:
        record: dict[str, object] = {
            "m_start": self.m_start,
            "n_start": self.n_start,
            "request_digest": self.request.digest(),
            "input_slices": [item.record() for item in self.input_slices],
        }
        if include_valid_extent:
            record["valid_m"] = self.input_slices[0].rows
            record["valid_n"] = self.input_slices[1].rows
        return record


@dataclass(frozen=True)
class OrderedMNTiling:
    source: KernelRequest
    tile_m: int
    tile_n: int
    tiles: tuple[MatrixTile, ...]
    tail_pad_element: bytes | None = None

    def __post_init__(self) -> None:
        try:
            lhs, rhs, _ = _validated_oriented_source(
                self.source, self.source.nodes[-1].type.numerical_policy,
            )
        except (OrderedKTilingError, IndexError) as exc:
            raise OrderedMNTilingError(str(exc)) from exc
        m_extent, n_extent = lhs.type.shape[0], rhs.type.shape[0]
        has_tail = bool(m_extent % self.tile_m or n_extent % self.tile_n) if (
            type(self.tile_m) is int and type(self.tile_n) is int
            and self.tile_m > 0 and self.tile_n > 0
        ) else False
        if (
            type(self.tile_m) is not int or type(self.tile_n) is not int
            or self.tile_m <= 0 or self.tile_n <= 0
            or has_tail != (self.tail_pad_element is not None)
            or (self.tail_pad_element is not None and (
                type(self.tail_pad_element) is not bytes or not self.tail_pad_element
            ))
        ):
            raise OrderedMNTilingError("M/N extents need exact tiles or declared tail padding")
        expected_coordinates = tuple(
            (m, n) for m in range(0, m_extent, self.tile_m)
            for n in range(0, n_extent, self.tile_n)
        )
        if len(self.tiles) != len(expected_coordinates):
            raise OrderedMNTilingError("M/N tiles omit or repeat an output region")
        for tile, (m, n) in zip(self.tiles, expected_coordinates):
            expected = _make_tile(self.source, self.tile_m, self.tile_n, m, n)
            if tile != expected:
                raise OrderedMNTilingError("M/N tile differs from its source region")

    def record(self) -> dict[str, object]:
        has_tail = self.tail_pad_element is not None
        record: dict[str, object] = {
            "schema": "merlin.ordered_mn_tiling.v2" if has_tail else "merlin.ordered_mn_tiling.v1",
            "source_request_digest": self.source.digest(),
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tiles": [tile.record(include_valid_extent=has_tail) for tile in self.tiles],
        }
        if has_tail:
            record["tail_pad_element_hex"] = self.tail_pad_element.hex()
        return record


def _make_tile(source: KernelRequest, tile_m: int, tile_n: int,
               m_start: int, n_start: int) -> MatrixTile:
    lhs, rhs, contraction = source.nodes
    valid_m = min(tile_m, lhs.type.shape[0] - m_start)
    valid_n = min(tile_n, rhs.type.shape[0] - n_start)
    lhs_tile = SemanticNode(
        lhs.id, "input", (), TensorType((tile_m, lhs.type.shape[1]), lhs.type.dtype,
                                        lhs.type.numerical_policy), lhs.attrs, "input",
    )
    rhs_tile = SemanticNode(
        rhs.id, "input", (), TensorType((tile_n, rhs.type.shape[1]), rhs.type.dtype,
                                        rhs.type.numerical_policy), rhs.attrs, "input",
    )
    result_tile = SemanticNode(
        contraction.id, contraction.op, contraction.inputs,
        TensorType((tile_m, tile_n), contraction.type.dtype,
                   contraction.type.numerical_policy),
        contraction.attrs, contraction.effect, contraction.index_maps,
    )
    request = KernelRequest(
        (lhs_tile, rhs_tile, result_tile), source.outputs, source.output_storages,
        source.input_storages, source.target_identity,
        lowering_policy=source.lowering_policy, source_identity=source.source_identity,
    )
    return MatrixTile(m_start, n_start, request, (
        RowMajorMatrixSlice(lhs.id, lhs.id, *lhs.type.shape,
                            m_start, 0, valid_m, lhs.type.shape[1]),
        RowMajorMatrixSlice(rhs.id, rhs.id, *rhs.type.shape,
                            n_start, 0, valid_n, rhs.type.shape[1]),
    ))


def partition_ordered_matmul_mn(
    source: KernelRequest, *, tile_m: int, tile_n: int, qualified_policy: str,
    tail_pad_element: bytes | None = None,
) -> OrderedMNTiling:
    """Partition independent output cells; pad tail rows only when qualified."""
    if not qualified_policy:
        raise OrderedMNTilingError("ordered M/N tiling needs a qualified policy")
    try:
        lhs, rhs, _ = _validated_oriented_source(source, qualified_policy)
    except OrderedKTilingError as exc:
        raise OrderedMNTilingError(str(exc)) from exc
    if type(tile_m) is not int or type(tile_n) is not int or tile_m <= 0 or tile_n <= 0:
        raise OrderedMNTilingError("M/N tile geometry must be positive integers")
    m_extent, n_extent = lhs.type.shape[0], rhs.type.shape[0]
    has_tail = bool(m_extent % tile_m or n_extent % tile_n)
    if has_tail and tail_pad_element is None:
        raise OrderedMNTilingError("M/N tails need a qualified pad element")
    if not has_tail and tail_pad_element is not None:
        raise OrderedMNTilingError("exact M/N tiles must not declare tail padding")
    return OrderedMNTiling(
        source, tile_m, tile_n,
        tuple(_make_tile(source, tile_m, tile_n, m, n)
              for m in range(0, m_extent, tile_m)
              for n in range(0, n_extent, tile_n)),
        tail_pad_element,
    )


def materialize_row_major_mn_inputs(
    tiling: OrderedMNTiling, source_bytes: Mapping[str, bytes], *, element_bytes: int,
) -> tuple[dict[str, bytes], ...]:
    """Return each tile's input boundary bytes in output row-major tile order."""
    if type(element_bytes) is not int or element_bytes <= 0:
        raise OrderedMNTilingError("element width must be positive")
    if tiling.tail_pad_element is not None and len(tiling.tail_pad_element) != element_bytes:
        raise OrderedMNTilingError("tail pad element width differs from tensor element")
    expected = {item.source_input: (item.source_rows, item.source_columns)
                for item in tiling.tiles[0].input_slices}
    if set(source_bytes) != set(expected):
        raise OrderedMNTilingError("runtime source inputs differ from M/N tiling")
    for name, (rows, columns) in expected.items():
        payload = source_bytes[name]
        if type(payload) is not bytes or len(payload) != rows * columns * element_bytes:
            raise OrderedMNTilingError("runtime source bytes differ from declared tensor")
    packed: list[dict[str, bytes]] = []
    for tile in tiling.tiles:
        inputs: dict[str, bytes] = {}
        for item in tile.input_slices:
            payload = source_bytes[item.source_input]
            source_row_bytes = item.source_columns * element_bytes
            tile_row_bytes = item.columns * element_bytes
            copied = b"".join(
                payload[row * source_row_bytes + item.column_start * element_bytes:
                        row * source_row_bytes + item.column_start * element_bytes + tile_row_bytes]
                for row in range(item.row_start, item.row_start + item.rows)
            )
            padded_rows = tile.request.node(item.tile_input).type.shape[0] - item.rows
            inputs[item.tile_input] = copied + (
                tiling.tail_pad_element * (padded_rows * item.columns)
                if padded_rows else b""
            )
        packed.append(inputs)
    return tuple(packed)


def assemble_row_major_mn_outputs(
    tiling: OrderedMNTiling, tile_outputs: Sequence[bytes], *, element_bytes: int,
) -> bytes:
    """Place all output rectangles into one row-major MxN result."""
    if type(element_bytes) is not int or element_bytes <= 0:
        raise OrderedMNTilingError("element width must be positive")
    if len(tile_outputs) != len(tiling.tiles) or any(
        type(payload) is not bytes or len(payload) != tiling.tile_m * tiling.tile_n * element_bytes
        for payload in tile_outputs
    ):
        raise OrderedMNTilingError("output tile bytes differ from declared M/N tiling")
    m_extent, n_extent = tiling.source.nodes[-1].type.shape
    output = bytearray(m_extent * n_extent * element_bytes)
    for tile, payload in zip(tiling.tiles, tile_outputs):
        valid_m, valid_n = (item.rows for item in tile.input_slices)
        for row in range(valid_m):
            destination = ((tile.m_start + row) * n_extent + tile.n_start) * element_bytes
            source_offset = row * tiling.tile_n * element_bytes
            count = valid_n * element_bytes
            output[destination:destination + count] = payload[source_offset:source_offset + count]
    return bytes(output)

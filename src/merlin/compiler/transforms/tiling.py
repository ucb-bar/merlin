"""Partition one declared ordered contraction into target-sized K tiles.

The caller supplies the target's K geometry and qualified numerical policy.
An explicitly qualified head pad can fill the first short tile before any
source reduction element. The target must establish that this pad leaves the
initial accumulator unchanged; no padding follows a source element.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from merlin.compiler.semantic.model import IndexMap, KernelRequest, SemanticNode, TensorType

# ---- contraction_k -----------------------------------------------------------------------------


class OrderedKTilingError(ValueError):
    """The source or selected geometry cannot be partitioned exactly."""


@dataclass(frozen=True)
class RowMajorKSlice:
    tile_input: str
    source_input: str
    rows: int
    source_columns: int
    k_start: int
    tile_columns: int
    source_count: int | None = None
    destination_column_start: int = 0

    def __post_init__(self) -> None:
        if (
            not self.tile_input
            or not self.source_input
            or any(
                type(value) is not int
                for value in (
                    self.rows,
                    self.source_columns,
                    self.k_start,
                    self.tile_columns,
                )
            )
            or self.rows <= 0
            or self.source_columns <= 0
            or self.tile_columns <= 0
            or (self.source_count is not None and (type(self.source_count) is not int or self.source_count <= 0))
            or type(self.destination_column_start) is not int
            or self.destination_column_start < 0
            or self.k_start < 0
            or self.k_start + (self.source_count or self.tile_columns) > self.source_columns
            or self.destination_column_start + (self.source_count or self.tile_columns) > self.tile_columns
        ):
            raise OrderedKTilingError("row-major K slice has invalid source bounds")

    def record(self) -> dict[str, int | str]:
        record: dict[str, int | str] = {
            "tile_input": self.tile_input,
            "source_input": self.source_input,
            "rows": self.rows,
            "source_columns": self.source_columns,
            "k_start": self.k_start,
            "tile_columns": self.tile_columns,
        }
        if self.source_count is not None:
            record["source_count"] = self.source_count
            record["destination_column_start"] = self.destination_column_start
        return record


@dataclass(frozen=True)
class OrderedKTiling:
    request: KernelRequest
    source_slices: tuple[RowMajorKSlice, ...]
    tile_count: int
    head_pad_element: bytes | None = None

    def __post_init__(self) -> None:
        if (
            type(self.tile_count) is not int
            or self.tile_count < 1
            or len(self.source_slices) != 2 * self.tile_count
            or tuple(item.tile_input for item in self.source_slices)
            != tuple(node_id for node_id, _ in self.request.input_storages)
        ):
            raise OrderedKTilingError("K slices do not match the tiled input boundary")
        sources = tuple(item.source_input for item in self.source_slices[:2])
        if len(set(sources)) != 2:
            raise OrderedKTilingError("K slices need two distinct logical sources")
        tile_columns = self.source_slices[0].tile_columns
        source_columns = self.source_slices[0].source_columns
        if self.source_slices[1].tile_columns != tile_columns or self.source_slices[1].source_columns != source_columns:
            raise OrderedKTilingError("K operands disagree on source or tile width")
        has_tail = self.head_pad_element is not None
        tail_count = source_columns % tile_columns
        if (
            (has_tail and (type(self.head_pad_element) is not bytes or not self.head_pad_element or not tail_count))
            or (not has_tail and tail_count)
            or self.tile_count != (source_columns + tile_columns - 1) // tile_columns
        ):
            raise OrderedKTilingError("K slices need exact tiles or a qualified head pad")
        for index in range(self.tile_count):
            for operand in range(2):
                item = self.source_slices[2 * index + operand]
                first = self.source_slices[operand]
                node = self.request.node(item.tile_input)
                expected_count = tail_count if has_tail and index == 0 else tile_columns
                expected_start = (
                    0 if index == 0 else (tail_count + (index - 1) * tile_columns if has_tail else index * tile_columns)
                )
                if (
                    item.source_input != sources[operand]
                    or (item.rows, item.source_columns, item.tile_columns)
                    != (first.rows, first.source_columns, first.tile_columns)
                    or item.k_start != expected_start
                    or (item.source_count if has_tail else item.tile_columns) != expected_count
                    or item.destination_column_start != (tile_columns - tail_count if has_tail and index == 0 else 0)
                    or (not has_tail and item.source_count is not None)
                    or node.type.shape != (item.rows, tile_columns)
                ):
                    raise OrderedKTilingError("K slices omit, overlap, or reorder source elements")

    def record(self) -> dict[str, object]:
        has_tail = self.head_pad_element is not None
        record: dict[str, object] = {
            "schema": "merlin.ordered_k_tiling.v2" if has_tail else "merlin.ordered_k_tiling.v1",
            "source_identity": self.request.source_identity,
            "tile_count": self.tile_count,
            "source_slices": [source_slice.record() for source_slice in self.source_slices],
            "tiled_request_digest": self.request.digest(),
        }
        if has_tail:
            record["head_pad_element_hex"] = self.head_pad_element.hex()
        return record


def _oriented_maps() -> tuple[IndexMap, ...]:
    return (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 1, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )


def _validated_oriented_source(
    source: KernelRequest,
    qualified_policy: str,
) -> tuple[SemanticNode, SemanticNode, SemanticNode]:
    """Check the exact, plain A[M,K] x W[N,K] source shared by tile transforms."""
    if (
        len(source.nodes) != 3
        or source.constants
        or source.physical_extents
        or len(source.outputs) != 1
        or len(source.input_storages) != 2
    ):
        raise OrderedKTilingError("ordered tiling needs one contraction and two plain inputs")
    lhs, rhs, contraction = source.nodes
    if (
        lhs.op != "input"
        or rhs.op != "input"
        or lhs.effect != rhs.effect
        or lhs.effect != "input"
        or lhs.index_maps
        or rhs.index_maps
        or contraction.op != "matmul"
        or contraction.effect != "pure"
        or contraction.inputs != (lhs.id, rhs.id)
        or contraction.attrs
        or source.outputs != (contraction.id,)
        or contraction.index_maps != _oriented_maps()
        or contraction.type.numerical_policy != qualified_policy
        or len(lhs.type.shape) != 2
        or len(rhs.type.shape) != 2
        or lhs.type.shape[1] != rhs.type.shape[1]
        or contraction.type.shape != (lhs.type.shape[0], rhs.type.shape[0])
        or lhs.type.dtype != rhs.type.dtype
        or lhs.type.numerical_policy != rhs.type.numerical_policy
        or set(source.input_storages) != {(lhs.id, "external"), (rhs.id, "external")}
        or source.output_storages != ("external",)
    ):
        raise OrderedKTilingError("ordered source graph, maps, or policy differs")
    return lhs, rhs, contraction


def partition_ordered_matmul_k(
    source: KernelRequest,
    *,
    tile_k: int,
    qualified_policy: str,
    head_pad_element: bytes | None = None,
) -> OrderedKTiling:
    """Split A[M,K] × W[N,K] into K-ascending tile inputs and accumulator nodes.

    ``qualified_policy`` must identify a target policy whose reset and
    continuation operations implement the same ordered reduction. For a short
    first tile, the target must also qualify ``head_pad_element`` as an exact
    identity on the reset accumulator before the first real source element.
    The caller materializes the recorded source slices at the boundary.
    """
    if type(tile_k) is not int or tile_k <= 0 or not qualified_policy:
        raise OrderedKTilingError("ordered K tiling needs positive geometry and a policy")
    lhs, rhs, contraction = _validated_oriented_source(source, qualified_policy)
    k_extent = lhs.type.shape[1]
    tail_count = k_extent % tile_k
    if tail_count and head_pad_element is None:
        raise OrderedKTilingError("ordered K tail needs a qualified head pad")
    if not tail_count and head_pad_element is not None:
        raise OrderedKTilingError("exact K tiles must not declare head padding")
    tile_count = (k_extent + tile_k - 1) // tile_k
    tiled_inputs: list[SemanticNode] = []
    slices: list[RowMajorKSlice] = []
    for index in range(tile_count):
        source_count = tail_count if tail_count and index == 0 else tile_k
        source_start = 0 if index == 0 else tail_count + (index - 1) * tile_k if tail_count else index * tile_k
        for original in (lhs, rhs):
            tile_id = f"{original.id}__k{index}"
            tiled_inputs.append(
                SemanticNode(
                    tile_id,
                    "input",
                    (),
                    TensorType((original.type.shape[0], tile_k), original.type.dtype, original.type.numerical_policy),
                    original.attrs,
                    "input",
                )
            )
            slices.append(
                RowMajorKSlice(
                    tile_id,
                    original.id,
                    original.type.shape[0],
                    k_extent,
                    source_start,
                    tile_k,
                    source_count if tail_count else None,
                    tile_k - tail_count if tail_count and index == 0 else 0,
                )
            )
    maps = _oriented_maps()
    reductions: list[SemanticNode] = []
    for index in range(tile_count):
        final = index == tile_count - 1
        node_id = contraction.id if final else f"{contraction.id}__k{index}"
        children = (f"{lhs.id}__k{index}", f"{rhs.id}__k{index}")
        reductions.append(
            SemanticNode(
                node_id,
                "matmul" if index == 0 else "matmul_accumulate",
                children if index == 0 else (*children, reductions[-1].id),
                contraction.type,
                index_maps=maps if index == 0 else (*maps, maps[-1]),
            )
        )
    tiled = KernelRequest(
        (*tiled_inputs, *reductions),
        source.outputs,
        source.output_storages,
        tuple((node.id, "external") for node in tiled_inputs),
        source.target_identity,
        lowering_policy=source.lowering_policy,
        source_identity=source.source_identity,
    )
    return OrderedKTiling(tiled, tuple(slices), tile_count, head_pad_element)


def materialize_row_major_k_slices(
    tiling: OrderedKTiling,
    source_bytes: Mapping[str, bytes],
    *,
    element_bytes: int,
) -> dict[str, bytes]:
    """Pack runtime tensor bytes into the exact physical input tile order."""
    if type(element_bytes) is not int or element_bytes <= 0:
        raise OrderedKTilingError("element width must be positive")
    if tiling.head_pad_element is not None and len(tiling.head_pad_element) != element_bytes:
        raise OrderedKTilingError("head pad element width differs from tensor element")
    expected_sources = {item.source_input for item in tiling.source_slices}
    if set(source_bytes) != expected_sources:
        raise OrderedKTilingError("runtime source inputs differ from the tiling record")
    result: dict[str, bytes] = {}
    for item in tiling.source_slices:
        payload = source_bytes[item.source_input]
        if type(payload) is not bytes or len(payload) != item.rows * item.source_columns * element_bytes:
            raise OrderedKTilingError("runtime source bytes differ from the declared tensor")
        row_bytes = item.source_columns * element_bytes
        source_count = item.source_count or item.tile_columns
        source_bytes_per_row = source_count * element_bytes
        pad = tiling.head_pad_element or b""
        result[item.tile_input] = b"".join(
            pad * item.destination_column_start
            + payload[
                row * row_bytes + item.k_start * element_bytes : row * row_bytes
                + item.k_start * element_bytes
                + source_bytes_per_row
            ]
            + pad * (item.tile_columns - item.destination_column_start - source_count)
            for row in range(item.rows)
        )
    return result


# Preserve published exception and record pickle spellings.
OrderedKTilingError.__module__ = "merlin.compiler.transforms.tiling"
RowMajorKSlice.__module__ = "merlin.compiler.transforms.tiling"
OrderedKTiling.__module__ = "merlin.compiler.transforms.tiling"


# ---- contraction_mn ----------------------------------------------------------------------------
# Exact outer tiles for one oriented matrix contraction.
#
# Each output tile is an independent kernel request. The target binding supplies
# qualified M/N geometry; this module only partitions row-major boundaries and
# reassembles output bytes. Reduction order within each tile is unchanged.


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
            not self.tile_input
            or not self.source_input
            or any(
                type(value) is not int
                for value in (
                    self.source_rows,
                    self.source_columns,
                    self.row_start,
                    self.column_start,
                    self.rows,
                    self.columns,
                )
            )
            or min(self.source_rows, self.source_columns, self.rows, self.columns) <= 0
            or self.row_start < 0
            or self.column_start < 0
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
                self.source,
                self.source.nodes[-1].type.numerical_policy,
            )
        except (OrderedKTilingError, IndexError) as exc:
            raise OrderedMNTilingError(str(exc)) from exc
        m_extent, n_extent = lhs.type.shape[0], rhs.type.shape[0]
        has_tail = (
            bool(m_extent % self.tile_m or n_extent % self.tile_n)
            if (type(self.tile_m) is int and type(self.tile_n) is int and self.tile_m > 0 and self.tile_n > 0)
            else False
        )
        if (
            type(self.tile_m) is not int
            or type(self.tile_n) is not int
            or self.tile_m <= 0
            or self.tile_n <= 0
            or has_tail != (self.tail_pad_element is not None)
            or (
                self.tail_pad_element is not None
                and (type(self.tail_pad_element) is not bytes or not self.tail_pad_element)
            )
        ):
            raise OrderedMNTilingError("M/N extents need exact tiles or declared tail padding")
        expected_coordinates = tuple(
            (m, n) for m in range(0, m_extent, self.tile_m) for n in range(0, n_extent, self.tile_n)
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


def _make_tile(source: KernelRequest, tile_m: int, tile_n: int, m_start: int, n_start: int) -> MatrixTile:
    lhs, rhs, contraction = source.nodes
    valid_m = min(tile_m, lhs.type.shape[0] - m_start)
    valid_n = min(tile_n, rhs.type.shape[0] - n_start)
    lhs_tile = SemanticNode(
        lhs.id,
        "input",
        (),
        TensorType((tile_m, lhs.type.shape[1]), lhs.type.dtype, lhs.type.numerical_policy),
        lhs.attrs,
        "input",
    )
    rhs_tile = SemanticNode(
        rhs.id,
        "input",
        (),
        TensorType((tile_n, rhs.type.shape[1]), rhs.type.dtype, rhs.type.numerical_policy),
        rhs.attrs,
        "input",
    )
    result_tile = SemanticNode(
        contraction.id,
        contraction.op,
        contraction.inputs,
        TensorType((tile_m, tile_n), contraction.type.dtype, contraction.type.numerical_policy),
        contraction.attrs,
        contraction.effect,
        contraction.index_maps,
    )
    request = KernelRequest(
        (lhs_tile, rhs_tile, result_tile),
        source.outputs,
        source.output_storages,
        source.input_storages,
        source.target_identity,
        lowering_policy=source.lowering_policy,
        source_identity=source.source_identity,
    )
    return MatrixTile(
        m_start,
        n_start,
        request,
        (
            RowMajorMatrixSlice(lhs.id, lhs.id, *lhs.type.shape, m_start, 0, valid_m, lhs.type.shape[1]),
            RowMajorMatrixSlice(rhs.id, rhs.id, *rhs.type.shape, n_start, 0, valid_n, rhs.type.shape[1]),
        ),
    )


def partition_ordered_matmul_mn(
    source: KernelRequest,
    *,
    tile_m: int,
    tile_n: int,
    qualified_policy: str,
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
        source,
        tile_m,
        tile_n,
        tuple(
            _make_tile(source, tile_m, tile_n, m, n)
            for m in range(0, m_extent, tile_m)
            for n in range(0, n_extent, tile_n)
        ),
        tail_pad_element,
    )


def materialize_row_major_mn_inputs(
    tiling: OrderedMNTiling,
    source_bytes: Mapping[str, bytes],
    *,
    element_bytes: int,
) -> tuple[dict[str, bytes], ...]:
    """Return each tile's input boundary bytes in output row-major tile order."""
    if type(element_bytes) is not int or element_bytes <= 0:
        raise OrderedMNTilingError("element width must be positive")
    if tiling.tail_pad_element is not None and len(tiling.tail_pad_element) != element_bytes:
        raise OrderedMNTilingError("tail pad element width differs from tensor element")
    expected = {item.source_input: (item.source_rows, item.source_columns) for item in tiling.tiles[0].input_slices}
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
                payload[
                    row * source_row_bytes + item.column_start * element_bytes : row * source_row_bytes
                    + item.column_start * element_bytes
                    + tile_row_bytes
                ]
                for row in range(item.row_start, item.row_start + item.rows)
            )
            padded_rows = tile.request.node(item.tile_input).type.shape[0] - item.rows
            inputs[item.tile_input] = copied + (
                tiling.tail_pad_element * (padded_rows * item.columns) if padded_rows else b""
            )
        packed.append(inputs)
    return tuple(packed)


def assemble_row_major_mn_outputs(
    tiling: OrderedMNTiling,
    tile_outputs: Sequence[bytes],
    *,
    element_bytes: int,
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
            output[destination : destination + count] = payload[source_offset : source_offset + count]
    return bytes(output)


# Preserve published exception and record pickle spellings.
OrderedMNTilingError.__module__ = "merlin.compiler.transforms.tiling"
RowMajorMatrixSlice.__module__ = "merlin.compiler.transforms.tiling"
MatrixTile.__module__ = "merlin.compiler.transforms.tiling"
OrderedMNTiling.__module__ = "merlin.compiler.transforms.tiling"


# ---- contraction_mnk ---------------------------------------------------------------------------
# Compose independent output tiles with ordered reduction tiles.
#
# The target supplies the tile geometry, numerical policy, and separately
# qualified pad bytes. Each resulting request retains the source's ordered K
# sequence; this module only moves declared boundary bytes and output tiles.


class OrderedMNKTilingError(ValueError):
    """An outer tile plan no longer describes the selected source exactly."""


@dataclass(frozen=True)
class OrderedMNKTiling:
    source: KernelRequest
    mn: OrderedMNTiling
    k_tiles: tuple[OrderedKTiling, ...]
    tile_k: int
    qualified_policy: str
    k_head_pad_element: bytes | None = None

    def __post_init__(self) -> None:
        if self.mn.source != self.source or len(self.k_tiles) != len(self.mn.tiles):
            raise OrderedMNKTilingError("M/N/K tiles do not cover the selected source")
        for output_tile, reduction in zip(self.mn.tiles, self.k_tiles):
            expected = partition_ordered_matmul_k(
                output_tile.request,
                tile_k=self.tile_k,
                qualified_policy=self.qualified_policy,
                head_pad_element=self.k_head_pad_element,
            )
            if reduction != expected:
                raise OrderedMNKTilingError("K tile differs from its output region")

    @property
    def requests(self) -> tuple[KernelRequest, ...]:
        """One separately compilable, ordered-reduction request per output tile."""
        return tuple(reduction.request for reduction in self.k_tiles)

    def record(self) -> dict[str, object]:
        return {
            "schema": "merlin.ordered_mnk_tiling.v1",
            "source_request_digest": self.source.digest(),
            "mn": self.mn.record(),
            "tile_k": self.tile_k,
            "qualified_policy": self.qualified_policy,
            "k_head_pad_element_hex": (None if self.k_head_pad_element is None else self.k_head_pad_element.hex()),
            "k": [reduction.record() for reduction in self.k_tiles],
        }


def partition_ordered_matmul_mnk(
    source: KernelRequest,
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    qualified_policy: str,
    mn_tail_pad_element: bytes | None = None,
    k_head_pad_element: bytes | None = None,
) -> OrderedMNKTiling:
    """Partition one oriented contraction without changing valid-cell K order.

    The first short K tile is head padded, so real source terms still appear
    in increasing K order. The caller must qualify that pad as an exact reset
    identity. M/N padding is a separate caller premise for invalid output
    lanes, and output assembly drops those lanes.
    """
    mn = partition_ordered_matmul_mn(
        source,
        tile_m=tile_m,
        tile_n=tile_n,
        qualified_policy=qualified_policy,
        tail_pad_element=mn_tail_pad_element,
    )
    reductions = tuple(
        partition_ordered_matmul_k(
            output_tile.request,
            tile_k=tile_k,
            qualified_policy=qualified_policy,
            head_pad_element=k_head_pad_element,
        )
        for output_tile in mn.tiles
    )
    return OrderedMNKTiling(source, mn, reductions, tile_k, qualified_policy, k_head_pad_element)


def materialize_row_major_mnk_inputs(
    tiling: OrderedMNKTiling,
    source_bytes: Mapping[str, bytes],
    *,
    element_bytes: int,
) -> tuple[dict[str, bytes], ...]:
    """Pack each output tile's complete, ordered K boundary from source bytes."""
    mn_inputs = materialize_row_major_mn_inputs(
        tiling.mn,
        source_bytes,
        element_bytes=element_bytes,
    )
    return tuple(
        materialize_row_major_k_slices(reduction, inputs, element_bytes=element_bytes)
        for reduction, inputs in zip(tiling.k_tiles, mn_inputs)
    )


def assemble_row_major_mnk_outputs(
    tiling: OrderedMNKTiling,
    tile_outputs: Sequence[bytes],
    *,
    element_bytes: int,
) -> bytes:
    """Trim invalid M/N lanes and join final accumulated output tiles."""
    return assemble_row_major_mn_outputs(
        tiling.mn,
        tile_outputs,
        element_bytes=element_bytes,
    )


@dataclass(frozen=True)
class LoopedMNKTiling:
    """Compact loop geometry and packed buffers for one ordered contraction.

    Unlike the explicit partition above, this plan does not build a request
    for every output tile. The target supplies its geometry, numerical policy
    and qualified pad values. Runtime tensors remain outside compilation.
    """

    source: KernelRequest
    tile_m: int
    tile_n: int
    tile_k: int
    qualified_policy: str
    input_element_bytes: int
    output_element_bytes: int
    mn_tail_pad_element: bytes | None = None
    k_head_pad_element: bytes | None = None

    def __post_init__(self) -> None:

        _validated_oriented_source(self.source, self.qualified_policy)
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.tile_m,
                self.tile_n,
                self.tile_k,
                self.input_element_bytes,
                self.output_element_bytes,
            )
        ):
            raise OrderedMNKTilingError("loop geometry and element widths must be positive integers")
        m, n, k = self.extents
        for needed, pad, label in (
            (bool(m % self.tile_m or n % self.tile_n), self.mn_tail_pad_element, "M/N"),
            (bool(k % self.tile_k), self.k_head_pad_element, "K"),
        ):
            if needed != (pad is not None) or (
                pad is not None and (type(pad) is not bytes or len(pad) != self.input_element_bytes)
            ):
                raise OrderedMNKTilingError(f"{label} tails need an explicit qualified pad of the input element width")

    @property
    def extents(self) -> tuple[int, int, int]:
        lhs, rhs, _ = self.source.nodes
        return lhs.type.shape[0], rhs.type.shape[0], lhs.type.shape[1]

    @property
    def counts(self) -> tuple[int, int, int]:
        return tuple(
            (extent + tile - 1) // tile
            for extent, tile in zip(
                self.extents,
                (self.tile_m, self.tile_n, self.tile_k),
                strict=True,
            )
        )

    @property
    def tile_bytes(self) -> tuple[int, int, int]:
        return (
            self.tile_m * self.tile_k * self.input_element_bytes,
            self.tile_n * self.tile_k * self.input_element_bytes,
            self.tile_m * self.tile_n * self.output_element_bytes,
        )

    @property
    def buffer_bytes(self) -> tuple[int, int, int]:
        m, n, k = self.counts
        a, w, out = self.tile_bytes
        return m * k * a, n * k * w, m * n * out

    def record(self) -> dict[str, object]:
        return {
            "schema": "merlin.looped_mnk_tiling.v1",
            "source_request_digest": self.source.digest(),
            "tile_shape": [self.tile_m, self.tile_n, self.tile_k],
            "loop_counts": list(self.counts),
            "loop_order": ["m", "n", "k"],
            "input_element_bytes": self.input_element_bytes,
            "output_element_bytes": self.output_element_bytes,
            "buffer_bytes": list(self.buffer_bytes),
            "input_tile_order": ["m,k", "n,k"],
            "output_tile_order": "m,n",
            "qualified_policy": self.qualified_policy,
            "mn_tail_pad_element_hex": None if self.mn_tail_pad_element is None else self.mn_tail_pad_element.hex(),
            "k_head_pad_element_hex": None if self.k_head_pad_element is None else self.k_head_pad_element.hex(),
        }


def materialize_looped_mnk_inputs(
    tiling: LoopedMNKTiling,
    source_bytes: Mapping[str, bytes],
) -> tuple[bytes, bytes]:
    """Pack each operand once, reusing A across N and W across M iterations.

    A is ordered by (M tile, K tile), W by (N tile, K tile). The short K
    segment is first and padded before its real elements. M/N tail rows
    occupy unobservable lanes and use their separately qualified pad value.
    """
    lhs, rhs, _ = tiling.source.nodes
    if set(source_bytes) != {lhs.id, rhs.id}:
        raise OrderedMNKTilingError("runtime source inputs differ from the looped plan")
    return tuple(materialize_looped_mnk_operand(tiling, node.id, source_bytes[node.id]) for node in (lhs, rhs))


def materialize_looped_mnk_operand(tiling: LoopedMNKTiling, source_id: str, payload: bytes) -> bytes:
    """Pack one logical operand, including an independently declared parameter."""
    lhs, rhs, _ = tiling.source.nodes
    if source_id not in (lhs.id, rhs.id):
        raise OrderedMNKTilingError("source operand is not an input of the looped plan")
    node, rows_per_tile = (lhs, tiling.tile_m) if source_id == lhs.id else (rhs, tiling.tile_n)
    width = tiling.input_element_bytes
    k_extent = tiling.extents[2]
    k_tail = k_extent % tiling.tile_k
    rows = node.type.shape[0]
    if type(payload) is not bytes or len(payload) != rows * k_extent * width:
        raise OrderedMNKTilingError("runtime tensor bytes differ from the looped source")
    packed = bytearray()
    for row_start in range(0, rows, rows_per_tile):
        for k_tile in range(tiling.counts[2]):
            count = k_tail if k_tail and k_tile == 0 else tiling.tile_k
            start = 0 if k_tile == 0 else (k_tail + (k_tile - 1) * tiling.tile_k if k_tail else k_tile * tiling.tile_k)
            pad = (tiling.k_head_pad_element or b"") * (tiling.tile_k - count)
            for row in range(row_start, row_start + rows_per_tile):
                if row >= rows:
                    assert tiling.mn_tail_pad_element is not None
                    packed.extend(tiling.mn_tail_pad_element * tiling.tile_k)
                else:
                    offset = (row * k_extent + start) * width
                    packed.extend(pad)
                    packed.extend(payload[offset : offset + count * width])
    return bytes(packed)


def assemble_looped_mnk_outputs(tiling: LoopedMNKTiling, output: bytes) -> bytes:
    """Trim row-major physical tiles into the source's logical M-by-N output.

    Target-specific lane/pair packing must be removed before calling this.
    """
    if type(output) is not bytes or len(output) != tiling.buffer_bytes[2]:
        raise OrderedMNKTilingError("looped output bytes differ from the declared tile buffer")
    m, n, _ = tiling.extents
    width = tiling.output_element_bytes
    result = bytearray(m * n * width)
    _, n_tiles, _ = tiling.counts
    for row in range(m):
        for n_tile in range(n_tiles):
            count = min(tiling.tile_n, n - n_tile * tiling.tile_n) * width
            tile = (row // tiling.tile_m) * n_tiles + n_tile
            start = tile * tiling.tile_bytes[2] + (row % tiling.tile_m) * tiling.tile_n * width
            dest = (row * n + n_tile * tiling.tile_n) * width
            result[dest : dest + count] = output[start : start + count]
    return bytes(result)


# Preserve published exception and record pickle spellings.
OrderedMNKTilingError.__module__ = "merlin.compiler.transforms.tiling"
OrderedMNKTiling.__module__ = "merlin.compiler.transforms.tiling"
LoopedMNKTiling.__module__ = "merlin.compiler.transforms.tiling"

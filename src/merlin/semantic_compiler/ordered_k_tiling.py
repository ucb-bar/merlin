"""Partition one declared ordered contraction into target-sized K tiles.

The caller supplies the target's K geometry and qualified numerical policy.
An explicitly qualified head pad can fill the first short tile before any
source reduction element. The target must establish that this pad leaves the
initial accumulator unchanged; no padding follows a source element.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .model import IndexMap, KernelRequest, SemanticNode, TensorType


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
            not self.tile_input or not self.source_input
            or any(type(value) is not int for value in (
                self.rows, self.source_columns, self.k_start, self.tile_columns,
            ))
            or self.rows <= 0 or self.source_columns <= 0 or self.tile_columns <= 0
            or (self.source_count is not None and (
                type(self.source_count) is not int or self.source_count <= 0))
            or type(self.destination_column_start) is not int
            or self.destination_column_start < 0
            or self.k_start < 0
            or self.k_start + (self.source_count or self.tile_columns) > self.source_columns
            or self.destination_column_start + (self.source_count or self.tile_columns)
            > self.tile_columns
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
            type(self.tile_count) is not int or self.tile_count < 1
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
        if (self.source_slices[1].tile_columns != tile_columns
                or self.source_slices[1].source_columns != source_columns):
            raise OrderedKTilingError("K operands disagree on source or tile width")
        has_tail = self.head_pad_element is not None
        tail_count = source_columns % tile_columns
        if (
            (has_tail and (type(self.head_pad_element) is not bytes
                           or not self.head_pad_element or not tail_count))
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
                expected_start = 0 if index == 0 else (
                    tail_count + (index - 1) * tile_columns if has_tail
                    else index * tile_columns
                )
                if (
                    item.source_input != sources[operand]
                    or (item.rows, item.source_columns, item.tile_columns)
                    != (first.rows, first.source_columns, first.tile_columns)
                    or item.k_start != expected_start
                    or (item.source_count if has_tail else item.tile_columns)
                    != expected_count
                    or item.destination_column_start != (
                        tile_columns - tail_count if has_tail and index == 0 else 0
                    )
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
    source: KernelRequest, qualified_policy: str,
) -> tuple[SemanticNode, SemanticNode, SemanticNode]:
    """Check the exact, plain A[M,K] x W[N,K] source shared by tile transforms."""
    if (
        len(source.nodes) != 3 or source.constants or source.physical_extents
        or len(source.outputs) != 1 or len(source.input_storages) != 2
    ):
        raise OrderedKTilingError("ordered tiling needs one contraction and two plain inputs")
    lhs, rhs, contraction = source.nodes
    if (
        lhs.op != "input" or rhs.op != "input"
        or lhs.effect != rhs.effect or lhs.effect != "input"
        or lhs.index_maps or rhs.index_maps
        or contraction.op != "matmul" or contraction.effect != "pure"
        or contraction.inputs != (lhs.id, rhs.id) or contraction.attrs
        or source.outputs != (contraction.id,)
        or contraction.index_maps != _oriented_maps()
        or contraction.type.numerical_policy != qualified_policy
        or len(lhs.type.shape) != 2 or len(rhs.type.shape) != 2
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
    source: KernelRequest, *, tile_k: int, qualified_policy: str,
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
        source_start = (0 if index == 0 else
                        tail_count + (index - 1) * tile_k if tail_count else index * tile_k)
        for original in (lhs, rhs):
            tile_id = f"{original.id}__k{index}"
            tiled_inputs.append(SemanticNode(
                tile_id, "input", (),
                TensorType((original.type.shape[0], tile_k),
                           original.type.dtype, original.type.numerical_policy),
                original.attrs, "input",
            ))
            slices.append(RowMajorKSlice(
                tile_id, original.id, original.type.shape[0], k_extent,
                source_start, tile_k,
                source_count if tail_count else None,
                tile_k - tail_count if tail_count and index == 0 else 0,
            ))
    maps = _oriented_maps()
    reductions: list[SemanticNode] = []
    for index in range(tile_count):
        final = index == tile_count - 1
        node_id = contraction.id if final else f"{contraction.id}__k{index}"
        children = (f"{lhs.id}__k{index}", f"{rhs.id}__k{index}")
        reductions.append(SemanticNode(
            node_id, "matmul" if index == 0 else "matmul_accumulate",
            children if index == 0 else (*children, reductions[-1].id),
            contraction.type, index_maps=maps if index == 0 else (*maps, maps[-1]),
        ))
    tiled = KernelRequest(
        (*tiled_inputs, *reductions), source.outputs, source.output_storages,
        tuple((node.id, "external") for node in tiled_inputs),
        source.target_identity, lowering_policy=source.lowering_policy,
        source_identity=source.source_identity,
    )
    return OrderedKTiling(tiled, tuple(slices), tile_count, head_pad_element)


def materialize_row_major_k_slices(
    tiling: OrderedKTiling, source_bytes: Mapping[str, bytes], *, element_bytes: int,
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
            + payload[row * row_bytes + item.k_start * element_bytes:
                      row * row_bytes + item.k_start * element_bytes + source_bytes_per_row]
            + pad * (item.tile_columns - item.destination_column_start - source_count)
            for row in range(item.rows)
        )
    return result

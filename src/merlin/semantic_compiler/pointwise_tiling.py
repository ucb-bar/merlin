"""Tile a pure affine pointwise operation without changing its scalar arithmetic."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

from .model import KernelRequest, pointwise_identity_maps


@dataclass(frozen=True)
class PointwiseTiling:
    source: KernelRequest
    tile_shape: tuple[int, int]
    element_bytes: int
    pad_element: bytes
    operand_pad_elements: tuple[bytes, ...] = ()
    compact_broadcasts: bool = False

    def __post_init__(self):
        source = self.source
        if (
            len(source.nodes) < 2
            or source.constants
            or source.physical_extents
            or source.boundary_layouts
            or source.output_layouts
            or len(source.outputs) != 1
            or len(self.tile_shape) != 2
            or any(type(x) is not int or x <= 0 for x in self.tile_shape)
            or type(self.element_bytes) is not int
            or self.element_bytes <= 0
            or type(self.pad_element) is not bytes
            or len(self.pad_element) != self.element_bytes
            or type(self.compact_broadcasts) is not bool
            or type(self.operand_pad_elements) is not tuple
            or (
                self.operand_pad_elements
                and (
                    len(self.operand_pad_elements) != len(source.nodes) - 1
                    or any(
                        type(pad) is not bytes or len(pad) != self.element_bytes for pad in self.operand_pad_elements
                    )
                )
            )
        ):
            raise ValueError("pointwise tiling needs a plain graph and explicit tile widths/padding")
        *inputs, output = source.nodes
        if (
            output.effect != "pure"
            or tuple(dict.fromkeys(output.inputs)) != tuple(node.id for node in inputs)
            or any(node.effect != "input" or node.op != "input" or node.index_maps for node in inputs)
            or source.outputs != (output.id,)
            or len(output.index_maps) != len(output.inputs) + 1
            or output.index_maps[-1] != pointwise_identity_maps(len(self.logical_shape), 1)[-1]
            or tuple(key for key, _ in source.input_storages) != tuple(node.id for node in inputs)
            or any(node.type.dtype != output.type.dtype for node in inputs)
        ):
            raise ValueError("pointwise tiling needs one computation and complete ordered operands/results")
        # Independent affine extrema prove every read is inside its source.
        for operand, node in enumerate(inputs):
            mapping = self.input_map(operand)
            if mapping.loop_rank != len(self.logical_shape) or len(mapping.coefficients) != len(node.type.shape):
                raise ValueError("pointwise operand map rank differs")
            for coefficients, offset, extent in zip(
                mapping.coefficients, mapping.offsets, node.type.shape, strict=True
            ):
                low = offset + sum(min(0, c * (n - 1)) for c, n in zip(coefficients, self.logical_shape, strict=True))
                high = offset + sum(max(0, c * (n - 1)) for c, n in zip(coefficients, self.logical_shape, strict=True))
                if low < 0 or high >= extent:
                    raise ValueError("pointwise operand map leaves its declared source")

    @property
    def logical_shape(self):
        return self.source.nodes[-1].type.shape

    @property
    def shape(self):
        """Matrix iteration space; the semantic graph retains the original rank."""
        return prod(self.logical_shape[:-1]), self.logical_shape[-1]

    def logical_indices(self, row: int, column: int):
        at = row * self.shape[1] + column
        indices = []
        for extent in reversed(self.logical_shape):
            at, coordinate = divmod(at, extent)
            indices.append(coordinate)
        return tuple(reversed(indices))

    def input_pad(self, operand: int) -> bytes:
        if type(operand) is not int or not 0 <= operand < len(self.source.nodes) - 1:
            raise ValueError("pointwise operand index is outside the source inputs")
        return self.operand_pad_elements[operand] if self.operand_pad_elements else self.pad_element

    def input_map(self, operand: int):
        self.input_pad(operand)
        key = self.source.nodes[operand].id
        output = self.source.nodes[-1]
        maps = [mapping for node, mapping in zip(output.inputs, output.index_maps[:-1], strict=True) if node == key]
        if not maps or any(mapping != maps[0] for mapping in maps):
            raise ValueError("shared pointwise uses need one identical physical view")
        return maps[0]

    @property
    def tile_counts(self):
        return tuple((n + t - 1) // t for n, t in zip(self.shape, self.tile_shape, strict=True))

    @property
    def tile_bytes(self):
        return prod(self.tile_shape) * self.element_bytes

    @property
    def packed_bytes(self):
        return prod(self.tile_counts) * self.tile_bytes

    def input_shape(self, operand: int):
        """Stored matrix shape after optional broadcast tile reuse.

        Reuse is valid only for lane-wise computations: extra values in the
        final output tile are unobserved, and cannot feed a reduction or effect.
        The provider must establish that property for its selected tile body.
        """
        self.input_pad(operand)  # Validate the operand before reading its map.
        mapping = self.input_map(operand)
        row_invariant = all(not any(row[:-1]) for row in mapping.coefficients)
        column_invariant = all(row[-1] == 0 for row in mapping.coefficients)
        return tuple(min(extent, tile) if self.compact_broadcasts and invariant else extent
                     for extent, tile, invariant in zip(
                         self.shape, self.tile_shape, (row_invariant, column_invariant), strict=True))

    def input_tile_counts(self, operand: int):
        return tuple((extent + tile - 1) // tile
                     for extent, tile in zip(self.input_shape(operand), self.tile_shape, strict=True))

    @property
    def input_packed_bytes(self):
        return tuple(prod(self.input_tile_counts(i)) * self.tile_bytes for i in range(len(self.source.nodes) - 1))

    def input_tile_strides(self, operand: int):
        """Tile offsets for one step in the output's row/column tile grid."""
        rows, columns = self.input_tile_counts(operand)
        return (columns if rows == self.tile_counts[0] else 0,
                1 if columns == self.tile_counts[1] else 0)

    def body_request(self):
        *operands, result = self.source.nodes
        return replace(
            self.source,
            nodes=(
                *(replace(node, type=replace(node.type, shape=self.tile_shape)) for node in operands),
                replace(
                    result,
                    type=replace(result.type, shape=self.tile_shape),
                    index_maps=pointwise_identity_maps(2, len(result.inputs)),
                ),
            ),
        )

    def record(self):
        record = {
            "schema": "merlin.pointwise_tiling.v1",
            "source_request_digest": self.source.digest(),
            "tile_shape": list(self.tile_shape),
            "tile_counts": list(self.tile_counts),
            "element_bytes": self.element_bytes,
            "packed_bytes": self.packed_bytes,
            "pad_element_hex": self.pad_element.hex(),
            "order": "row_tiles_then_column_tiles",
            "body_request_digest": self.body_request().digest(),
        }
        if len(self.logical_shape) != 2:
            record.update(
                schema="merlin.pointwise_tiling.v2",
                logical_shape=list(self.logical_shape),
                matrix_shape=list(self.shape),
                leading_dimensions=list(range(len(self.logical_shape) - 1)),
                column_dimension=len(self.logical_shape) - 1,
            )
        if self.operand_pad_elements:
            record["operand_pad_elements_hex"] = [pad.hex() for pad in self.operand_pad_elements]
        if self.compact_broadcasts:
            record.update(
                schema="merlin.pointwise_tiling.v3",
                input_tile_counts=[list(self.input_tile_counts(i)) for i in range(len(self.source.nodes) - 1)],
                input_packed_bytes=list(self.input_packed_bytes),
                broadcast_tail_rule="only-valid-output-lanes-observed; selected-body-must-be-lane-wise",
            )
        return record


def materialize_pointwise_input(plan: PointwiseTiling, operand: int, payload: bytes) -> bytes:
    """Expand broadcast/strided/transpose maps, initializing every tail lane."""
    if type(operand) is not int or not 0 <= operand < len(plan.source.nodes) - 1:
        raise ValueError("pointwise operand index is outside the source inputs")
    node = plan.source.nodes[operand]
    mapping = plan.input_map(operand)
    width = plan.element_bytes
    if type(payload) is not bytes or len(payload) != prod(node.type.shape) * width:
        raise ValueError("pointwise operand bytes differ from the source")
    result = bytearray()
    rows, columns = plan.input_shape(operand)
    for m in range(0, rows, plan.tile_shape[0]):
        for n in range(0, columns, plan.tile_shape[1]):
            for row in range(plan.tile_shape[0]):
                for col in range(plan.tile_shape[1]):
                    if m + row >= rows or n + col >= columns:
                        result.extend(plan.input_pad(operand))
                        continue
                    at = 0
                    logical = plan.logical_indices(m + row, n + col)
                    for index, extent in zip(mapping.apply(logical), node.type.shape, strict=True):
                        at = at * extent + index
                    result.extend(payload[at * width : (at + 1) * width])
    return bytes(result)


def assemble_pointwise_output(plan: PointwiseTiling, payload: bytes) -> bytes:
    if type(payload) is not bytes or len(payload) != plan.packed_bytes:
        raise ValueError("pointwise output bytes differ from the packed shape")
    result = bytearray(prod(plan.shape) * plan.element_bytes)
    tm, tn = plan.tile_shape
    for row in range(plan.shape[0]):
        for col in range(plan.shape[1]):
            at = (
                ((row // tm) * plan.tile_counts[1] + col // tn) * tm * tn + (row % tm) * tn + col % tn
            ) * plan.element_bytes
            out = (row * plan.shape[1] + col) * plan.element_bytes
            result[out : out + plan.element_bytes] = payload[at : at + plan.element_bytes]
    return bytes(result)

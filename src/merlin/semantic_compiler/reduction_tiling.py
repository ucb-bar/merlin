"""Pack seeded tensor reductions into matrix tiles without choosing arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod

from .model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from .pointwise_tiling import PointwiseTiling, assemble_pointwise_output, materialize_pointwise_input


@dataclass(frozen=True)
class ReductionTiling:
    source: KernelRequest
    axes: tuple[int, ...]
    tile_shape: tuple[int, int]
    element_bytes: int
    pad_element: bytes

    def __post_init__(self):
        if (
            len(self.source.nodes) != 3
            or self.source.constants
            or self.source.physical_extents
            or self.source.boundary_layouts
            or self.source.output_layouts
        ):
            raise ValueError("reduction tiling needs a plain data/seed/result graph")
        data, seed, out = self.source.nodes
        rank = len(data.type.shape)
        if (
            type(self.axes) is not tuple
            or not self.axes
            or any(type(axis) is not int for axis in self.axes)
            or tuple(sorted(set(self.axes))) != self.axes
            or any(type(axis) is not int or not 0 <= axis < rank for axis in self.axes)
            or len(self.axes) == rank
        ):
            raise ValueError("reduction axes must be distinct ascending dimensions with a ranked result")
        expected = tuple(data.type.shape[axis] for axis in self.kept_axes)
        projection = IndexMap(
            rank, tuple(tuple(int(i == axis) for i in range(rank)) for axis in self.kept_axes), (0,) * len(expected)
        )
        if (
            data.op != "input"
            or seed.op != "input"
            or data.effect != seed.effect
            or data.effect != "input"
            or data.index_maps
            or seed.index_maps
            or out.effect != "pure"
            or out.op != "reduce_sum"
            or out.inputs != (data.id, seed.id)
            or out.type.shape != expected
            or seed.type.shape != expected
            or any(node.type.dtype != data.type.dtype for node in (seed, out))
            or out.index_maps != (pointwise_identity_maps(rank, 1)[0], projection, projection)
            or self.source.outputs != (out.id,)
            or tuple(key for key, _ in self.source.input_storages) != out.inputs
        ):
            raise ValueError("reduction shapes, maps, seeds or boundary accounting differ")
        # The existing pointwise plans validate geometry, byte widths and pads.
        self.data_plan()
        self.seed_plan()

    @property
    def kept_axes(self):
        return tuple(i for i in range(len(self.source.nodes[0].type.shape)) if i not in self.axes)

    @property
    def output_shape(self):
        return self.source.nodes[-1].type.shape

    @property
    def matrix_shape(self):
        shape = self.source.nodes[0].type.shape
        return prod(self.output_shape), prod(shape[axis] for axis in self.axes)

    def _packing_plan(self, matrix_shape):
        storage = self.source.output_storages[0]
        type_ = TensorType(matrix_shape, self.source.nodes[0].type.dtype, "raw-bit-pattern")
        operand = SemanticNode("value", "input", (), type_, effect="input")
        out = SemanticNode("packed", "copy", (operand.id,), type_, index_maps=pointwise_identity_maps(2, 1))
        return PointwiseTiling(
            KernelRequest(
                (operand, out),
                (out.id,),
                (storage,),
                ((operand.id, storage),),
                self.source.target_identity,
                self.source.lowering_policy,
            ),
            self.tile_shape,
            self.element_bytes,
            self.pad_element,
        )

    def data_plan(self):
        return self._packing_plan(self.matrix_shape)

    def seed_plan(self):
        # Only column zero is observed. Other accumulator columns are allocated
        # and initialized, but never feed a row reduction.
        return self._packing_plan((self.matrix_shape[0], 1))

    @property
    def packed_sizes(self):
        return self.data_plan().packed_bytes, self.seed_plan().packed_bytes, self.seed_plan().packed_bytes

    def record(self):
        return {
            "schema": "merlin.reduction_tiling.v1",
            "source_request_digest": self.source.digest(),
            "axes": list(self.axes),
            "kept_axes": list(self.kept_axes),
            "output_shape": list(self.output_shape),
            "matrix_shape": list(self.matrix_shape),
            "tile_shape": list(self.tile_shape),
            "element_bytes": self.element_bytes,
            "pad_element_hex": self.pad_element.hex(),
            "packed_sizes": list(self.packed_sizes),
            "output_column": 0,
            "tile_order": "output_row_groups_then_reduction_tiles",
        }


def materialize_reduction_inputs(plan: ReductionTiling, data: bytes, seed: bytes):
    """Permute source axes, pack all reduction tails, and preserve the seed bits."""
    shape = plan.source.nodes[0].type.shape
    width = plan.element_bytes
    if (
        type(data) is not bytes
        or type(seed) is not bytes
        or len(data) != prod(shape) * width
        or len(seed) != prod(plan.output_shape) * width
    ):
        raise ValueError("reduction input bytes differ from data or seed shapes")
    order = plan.kept_axes + plan.axes
    canonical = bytearray()
    for coordinate in product(*(range(shape[axis]) for axis in order)):
        indices = dict(zip(order, coordinate, strict=True))
        at = 0
        for axis, extent in enumerate(shape):
            at = at * extent + indices[axis]
        canonical.extend(data[at * width : (at + 1) * width])
    return (
        materialize_pointwise_input(plan.data_plan(), 0, bytes(canonical)),
        materialize_pointwise_input(plan.seed_plan(), 0, seed),
    )


def assemble_reduction_output(plan: ReductionTiling, packed: bytes):
    """Extract exactly one value per output row; unused columns stay unobserved."""
    return assemble_pointwise_output(plan.seed_plan(), packed)

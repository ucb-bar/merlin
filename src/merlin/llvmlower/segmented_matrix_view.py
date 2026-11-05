"""Static row-segmented matrix views, preserving element width and ownership.

This helper proves an element-order map through explicit static slices and
linear reshapes. It edits no IR. A consumer must separately bind the owner to
dense physical storage, retain its lifetime and accept a read-only view ABI.
"""

from dataclasses import dataclass
from math import prod

from xdsl.dialects.builtin import NoneAttr, TensorType
from xdsl.dialects.linalg.ops import TransposeOp, YieldOp
from xdsl.dialects.tensor import CollapseShapeOp, ExpandShapeOp, ExtractSliceOp
from xdsl.ir import SSAValue


@dataclass(frozen=True)
class SegmentedRows:
    """Logical (rows, cols) indexing into one typed contiguous allocation.

    Offsets and strides are in elements. The allocation and byte width are
    explicit; no caller may substitute a differently typed semantic tensor.
    """

    rows: int
    cols: int
    segment_rows: int
    row_stride: int
    segment_stride: int
    source_elements: int
    element_bytes: int
    dtype: str
    origin: int = 0

    def __post_init__(self):
        values = (
            self.rows,
            self.cols,
            self.segment_rows,
            self.row_stride,
            self.segment_stride,
            self.source_elements,
            self.element_bytes,
        )
        if any(type(v) is not int or v <= 0 for v in values):
            raise ValueError("view extents, strides and width require positive integers")
        if type(self.origin) is not int or self.origin < 0:
            raise ValueError("view origin requires a nonnegative integer")
        if not self.dtype:
            raise ValueError("view element type is required")
        if self.row_stride < self.cols:
            raise ValueError("view rows overlap")
        segments = (self.rows + self.segment_rows - 1) // self.segment_rows
        separated = self.segment_stride >= (self.segment_rows - 1) * self.row_stride + self.cols
        # A transpose can interleave disjoint segments. Within each physical
        # row, their intervals are separated by segment_stride; successive
        # physical rows are separated by row_stride. Bounding spans overlap,
        # but the actual element intervals do not. This is a sufficient closed
        # proof, not an arbitrary-stride alias solver.
        interleaved = (self.segment_stride >= self.cols
                       and self.row_stride >= (segments - 1) * self.segment_stride + self.cols)
        if not separated and not interleaved:
            raise ValueError("view row segments overlap")
        endpoints = [self.offset(self.rows - 1, self.cols - 1)]
        if self.rows > self.segment_rows and self.rows % self.segment_rows:
            endpoints.append(self.offset(self.rows // self.segment_rows * self.segment_rows - 1, self.cols - 1))
        if max(endpoints) >= self.source_elements:
            raise ValueError("view endpoint exceeds the source allocation")

    def offset(self, row: int, col: int = 0) -> int:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError("view index is outside the logical matrix")
        return (
            self.origin
            + (row // self.segment_rows) * self.segment_stride
            + (row % self.segment_rows) * self.row_stride
            + col
        )

    def split_rows(self, start: int, count: int):
        """Yield logical-order runs with the explicit physical row stride.

        Source offsets need not increase between runs of an interleaved view.
        """
        if type(start) is not int or type(count) is not int or count <= 0:
            raise ValueError("row panel requires integer start and positive count")
        if start < 0 or start + count > self.rows:
            raise ValueError("row panel exceeds the logical matrix")
        if self.segment_stride == self.segment_rows * self.row_stride:
            yield start, count, self.offset(start)
            return
        stop = start + count
        while start < stop:
            width = min(stop - start, self.segment_rows - start % self.segment_rows)
            yield start, width, self.offset(start)
            start += width


@dataclass(frozen=True)
class MatrixViewProof:
    owner: SSAValue
    address: SegmentedRows
    chain: tuple

    def to_dict(self):
        return dict(
            kind="static_segmented_matrix_view",
            address=self.address.__dict__.copy(),
            owner_type=str(self.owner.type),
            chain_operations=[op.name for op in self.chain],
            physical_storage="Requires consumer closure of dense owner allocation and element width",
            ownership="Original tensor SSA owner retained; consumer must preserve lifetime and read-only access",
        )


def _shape(value):
    ty = value.type
    if not isinstance(ty, TensorType) or not isinstance(ty.encoding, NoneAttr):
        raise ValueError("view requires unencoded ranked tensors")
    shape = tuple(ty.get_shape())
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError("view requires positive static tensor extents")
    return shape


def _groups(op):
    return tuple(tuple(a.value.data for a in group) for group in op.reassociation)


def _transpose_source(value):
    """Read one exact tensor permutation, never an arbitrary scalar region."""
    op = value.owner
    if not isinstance(op, TransposeOp):
        return value, tuple(range(len(_shape(value)))), None
    if (op.attributes or set(op.properties) != {"permutation"}
            or len(op.operands) != 2 or len(op.results) != 1 or len(op.body.blocks) != 1):
        raise ValueError("view transpose requires an unmodified pure permutation")
    before, after = _shape(op.inputs[0]), _shape(value)
    permutation = tuple(op.permutation.get_values())
    body = tuple(op.body.block.ops)
    if (sorted(permutation) != list(range(len(before)))
            or after != tuple(before[i] for i in permutation)
            or op.operands[1].type != value.type or len(op.body.block.args) != 2
            or len(body) != 1 or not isinstance(body[0], YieldOp)
            or body[0].properties or body[0].attributes
            or tuple(body[0].operands) != (op.body.block.args[0],)
            or op.body.block.args[1].uses
            or op.inputs[0].type.get_element_type() != value.type.get_element_type()):
        raise ValueError("view transpose changes source values or coordinates")
    op.verify()
    return op.inputs[0], permutation, op


def _linear_source(value, chain):
    """Peel only whole-linearization reshapes, preserving all element order."""
    while isinstance(value.owner, (CollapseShapeOp, ExpandShapeOp)):
        op = value.owner
        before, after = _shape(op.operands[0]), _shape(value)
        if len(before) == 1:
            expected = (tuple(range(len(after))),)
        elif len(after) == 1:
            expected = (tuple(range(len(before))),)
        else:
            break
        if _groups(op) != expected or prod(before) != prod(after):
            raise ValueError("reshape does not prove whole linear element order")
        if value.type.get_element_type() != op.operands[0].type.get_element_type():
            raise ValueError("reshape changes element type")
        chain.append(op)
        value = op.operands[0]
    return value


def prove_segmented_matrix(value: SSAValue) -> MatrixViewProof:
    """Prove matrix rows from a batch-one spatial slice with contiguous columns.

    Nested static slices and an optional exact transpose before linearization
    compose exactly. Dynamic operands, rank reduction and element conversions
    refuse. Selected prefix dimensions must be singleton. A transpose supplies
    only an address map, not permission to remove a materialization or retain
    the source allocation.
    """
    logical = _shape(value)
    if len(logical) != 2 or not isinstance(value.owner, ExpandShapeOp):
        raise ValueError("matrix view requires an explicit flat-to-matrix reshape")
    chain = []
    expanded = value.owner
    flat = expanded.operands[0]
    if len(_shape(flat)) != 1 or _groups(expanded) != ((0, 1),):
        raise ValueError("matrix reshape requires whole linear reassociation")
    collapse = flat.owner
    if not isinstance(collapse, CollapseShapeOp):
        raise ValueError("matrix view requires an explicit whole collapse")
    sliced = collapse.operands[0]
    selected = _shape(sliced)
    rank = len(selected)
    if rank < 3 or any(dim != 1 for dim in selected[:-3]):
        raise ValueError("view requires singleton prefix and two spatial axes")
    if (
        _groups(collapse) != (tuple(range(rank)),)
        or logical != (selected[-3] * selected[-2], selected[-1])
        or _shape(flat) != (prod(selected),)
    ):
        raise ValueError("matrix shape disagrees with spatial linearization")
    chain.extend((expanded, collapse))
    offsets, strides = [0] * rank, [1] * rank
    source, permutation, transposed = _transpose_source(sliced)
    if transposed is not None:
        chain.append(transposed)
    count = 0
    while isinstance(source.owner, ExtractSliceOp):
        op = source.owner
        before, after = _shape(op.operands[0]), _shape(source)
        if len(op.operands) != 1 or len(before) != rank or len(after) != rank:
            raise ValueError("view requires static rank-preserving slices")
        arrays = [
            tuple(op.properties[key].get_values()) for key in ("static_offsets", "static_sizes", "static_strides")
        ]
        off, sizes, step = arrays
        if any(len(row) != rank for row in arrays) or sizes != after:
            raise ValueError("slice type and static sizes disagree")
        for axis in range(rank):
            if off[axis] < 0 or step[axis] <= 0 or off[axis] + (sizes[axis] - 1) * step[axis] >= before[axis]:
                raise ValueError("slice endpoint is outside its source")
            offsets[axis] = off[axis] + offsets[axis] * step[axis]
            strides[axis] *= step[axis]
        chain.append(op)
        source = op.operands[0]
        count += 1
    if not count:
        raise ValueError("matrix view has no explicit static slice")
    base_shape = _shape(source)
    elem = value.type.get_element_type()
    if any(op.operands[0].type.get_element_type() != elem for op in chain):
        raise ValueError("view chain changes element type")
    dense = [prod(base_shape[axis + 1 :]) for axis in range(rank)]
    physical = [strides[axis] * dense[axis] for axis in permutation]
    if physical[-1] != 1:
        raise ValueError("view requires singleton prefix and contiguous column axis")
    width = getattr(elem, "bitwidth", None)
    if width is None or width <= 0 or width % 8:
        raise ValueError("view element byte width is unknown")
    address = SegmentedRows(
        rows=logical[0],
        cols=logical[1],
        segment_rows=selected[-2],
        row_stride=physical[-2],
        segment_stride=physical[-3],
        source_elements=prod(base_shape),
        element_bytes=width // 8,
        dtype=str(elem),
        origin=sum(a * b for a, b in zip(offsets, dense, strict=True)),
    )
    owner = _linear_source(source, chain)
    if prod(_shape(owner)) != address.source_elements:
        raise ValueError("owner allocation extent differs from the source view")
    return MatrixViewProof(owner, address, tuple(chain))

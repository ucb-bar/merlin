"""Prove full initialization of immutable tensors assembled from static slices.

Partial tensors remain unavailable to computation. A completed, disjoint
full-axis partition becomes a concat of its initialized pieces, without
inventing a value for any element of tensor.empty.
"""

from dataclasses import dataclass

from .linalg_recognition import LinalgBridgeError
from .model import SemanticNode, TensorType


@dataclass(frozen=True)
class PartialTensor:
    type: TensorType
    axis: int
    pieces: tuple[tuple[int, int, str], ...]  # offset, extent, semantic source


def insert_partial(destination, type_, source, source_type, offsets, strides, previous, *, node_id):
    """Return the updated construction and, only when complete, a value node.

    Source and slice geometry have already passed the ordinary insert_slice
    checks. This proof admits nonoverlapping complete-axis slices; it does not
    approximate arbitrary rectangular coverage by a bounding interval.
    """
    if previous is None:
        owner = destination.owner
        if (
            getattr(owner, "name", None) != "tensor.empty"
            or owner.operands
            or owner.properties
            or owner.regions
            or owner.successors
            or tuple(owner.results) != (destination,)
            or any(not key.startswith("prov.") for key in owner.attributes)
        ):
            raise LinalgBridgeError("partial tensor assembly must begin with a static tensor.empty")
    elif previous.type != type_:
        raise LinalgBridgeError("partial tensor assembly changes its element type or shape")
    if any(stride != 1 for stride in strides):
        raise LinalgBridgeError("partial tensor assembly requires contiguous slices")
    differing = tuple(dim for dim, n in enumerate(type_.shape) if offsets[dim] or source_type.shape[dim] != n)
    if not differing:
        return None, SemanticNode(node_id, "identity", (source,), type_)
    if len(differing) != 1:
        raise LinalgBridgeError("partial tensor assembly requires complete slices of one axis")
    axis = differing[0]
    if previous is not None and previous.axis != axis:
        raise LinalgBridgeError("partial tensor assembly changes its partition axis")
    offset, extent = offsets[axis], source_type.shape[axis]
    pieces = () if previous is None else previous.pieces
    if any(offset < start + count and start < offset + extent for start, count, _ in pieces):
        raise LinalgBridgeError("partial tensor assembly has overlapping writes")
    pieces = tuple(sorted((*pieces, (offset, extent, source))))
    cursor = 0
    for start, count, _ in pieces:
        if start != cursor:
            return PartialTensor(type_, axis, pieces), None
        cursor += count
    if cursor != type_.shape[axis]:
        return PartialTensor(type_, axis, pieces), None
    return None, SemanticNode(node_id, "concat", tuple(key for _, _, key in pieces), type_, attrs=(("axis", axis),))

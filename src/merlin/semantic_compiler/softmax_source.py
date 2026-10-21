"""Recognize a closed stable row-softmax region in the parsed source graph.

Recognition preserves source arithmetic and unresolved reduction order. It
does not grant a target implementation or make a precision change equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from .model import IndexMap, KernelRequest, TensorType, pointwise_identity_maps


@dataclass(frozen=True)
class RowSoftmaxSource:
    source_digest: str
    scores: str
    result: str
    shape: tuple[int, ...]
    computations: tuple[str, ...]
    initializers: tuple[str, ...]

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """Independent dense prefix rows; the reduction axis stays intact."""
        return prod(self.shape[:-1]), self.shape[-1]

    def record(self) -> dict:
        record = {
            "source_request_digest": self.source_digest,
            "scores": self.scores,
            "result": self.result,
            "shape": list(self.shape),
            "computations": list(self.computations),
            "initializers": list(self.initializers),
            "source_reduction_order": "unresolved",
            "source_math_exp": "unqualified",
        }
        if len(self.shape) != 2:
            record["dense_row_view"] = {
                "source_shape": list(self.shape),
                "matrix_shape": list(self.matrix_shape),
                "row_axes": list(range(len(self.shape) - 1)),
                "reduction_axis": len(self.shape) - 1,
                "element_order": "row-major-unchanged",
            }
        return record


def recognize_row_softmax(request: KernelRequest, result_id: str) -> RowSoftmaxSource:
    """Check max/subtract/exp/sum/divide, including exact seeds and broadcasts.

    The result may have outside consumers. Other computed intermediates cannot
    escape the region. Immutable initializer constants may be shared with other
    regions and remain separately accounted for by the caller.
    """

    return _recognize_row_softmax(request, result_id)


def _recognize_row_softmax(request, result_id, enclosing=()):
    """The enclosing recognizer must separately check every extra consumer."""

    def require(condition, message):
        if not condition:
            raise ValueError("source row softmax: " + message)

    def node(name, operation, arity, attrs, maps):
        value = request.node(name)
        require(
            value.op == operation
            and value.effect == "pure"
            and len(value.inputs) == arity
            and dict(value.attrs) == attrs
            and value.index_maps == maps,
            "unsupported " + operation + " body, attributes or index maps",
        )
        return value

    shape = request.node(result_id).type.shape
    rank = len(shape)
    require(rank >= 2, "expected a rank-two or higher f32 result")
    identity = pointwise_identity_maps(rank, 1)[0]
    prefix = IndexMap(rank, identity.coefficients[:-1], (0,) * (rank - 1))
    row_broadcast = IndexMap(rank, (*identity.coefficients[:-1], (0,) * rank), (0,) * rank)
    broadcast = (identity, row_broadcast, identity)
    reduce_maps = (identity, prefix, prefix)
    result = node(result_id, "f32_div", 2, {"numeric_semantics": "ieee754-f32-div-rne-v1"}, broadcast)
    require(result.type == TensorType(shape, "f32", "raw-bit-pattern"), "expected an f32 result")
    row_shape = shape[:-1]
    exponential = node(
        result.inputs[0],
        "source_f32_exp_unqualified",
        1,
        {"numeric_semantics": "source-f32-exp-unqualified-v1"},
        pointwise_identity_maps(rank, 1),
    )
    shifted = node(exponential.inputs[0], "f32_sub", 2, {"numeric_semantics": "ieee754-f32-sub-rne-v1"}, broadcast)
    scores = request.node(shifted.inputs[0])
    require(scores.effect in {"input", "constant", "pure"}, "score producer has unresolved effects")
    for value in (scores, shifted, exponential):
        require(value.type == result.type, "score/exp/result shapes or types differ")

    def row_reduction(reshape_id, source_id, maximum):
        reshape = node(reshape_id, "reshape", 1, {}, ())
        require(reshape.type == TensorType((*row_shape, 1), "f32", "raw-bit-pattern"), "invalid broadcast reshape")
        views = [reshape]
        while request.node(views[-1].inputs[0]).op == "reshape":
            view = node(views[-1].inputs[0], "reshape", 1, {}, ())
            require(
                view.type.dtype == "f32"
                and view.type.numerical_policy == "raw-bit-pattern"
                and prod(view.type.shape) == prod(row_shape),
                "intermediate reshape changes row elements or type",
            )
            views.append(view)
        kind = "max" if maximum else "add"
        attrs = {
            "axis": rank - 1,
            "reduction_order": "unresolved",
            "numeric_semantics": "source-f32-maximum-unqualified-v1" if maximum else "source-f32-add-unqualified-v1",
            "scalar_operand_order" if maximum else "scalar_add_order": "input-accumulator",
        }
        reduction = node(views[-1].inputs[0], f"source_f32_reduce_{kind}_unqualified", 2, attrs, reduce_maps)
        require(
            reduction.inputs[0] == source_id and reduction.type == TensorType(row_shape, "f32", "raw-bit-pattern"),
            "reduction input or shape differs",
        )
        seed = request.node(reduction.inputs[1])
        bindings = {item.node_id: item for item in request.constants}
        binding = bindings.get(seed.id)
        require(
            seed.op == "constant"
            and seed.effect == "constant"
            and not seed.attrs
            and not seed.index_maps
            and seed.type == reduction.type
            and binding is not None
            and binding.encoding == "f32-le-raw",
            "reduction needs an explicit f32 row initializer",
        )
        expected = bytes.fromhex("000080ff") if maximum else bytes(4)
        require(bytes.fromhex(binding.data_hex) == expected * prod(row_shape), "wrong maximum or sum initializer")
        return tuple(views), reduction, seed

    max_views, maximum, max_seed = row_reduction(shifted.inputs[1], scores.id, True)
    sum_views, total, sum_seed = row_reduction(result.inputs[1], exponential.id, False)
    computations = {value.id for value in (maximum, *max_views, shifted, exponential, total, *sum_views, result)}
    hidden = computations - {result.id}
    allowed_consumers = computations | set(enclosing)
    require(not hidden.intersection(request.outputs), "an intermediate is a graph output")
    require(
        not any(value.id not in allowed_consumers and hidden.intersection(value.inputs) for value in request.nodes),
        "an intermediate has an outside consumer",
    )
    return RowSoftmaxSource(
        request.digest(),
        scores.id,
        result.id,
        shape,
        tuple(value.id for value in request.nodes if value.id in computations),
        (max_seed.id, sum_seed.id),
    )


@dataclass(frozen=True)
class MaskedRowSoftmaxSource(RowSoftmaxSource):
    """An unobserved additive mask followed by a complete stable softmax."""

    mask: str
    mask_map: IndexMap
    masked_scores: str
    mask_shape: tuple[int, ...]

    def record(self) -> dict:
        return {
            **super().record(),
            "mask": self.mask,
            "mask_map": self.mask_map.record(),
            "masked_scores": self.masked_scores,
            "mask_shape": list(self.mask_shape),
            "mask_values": "unqualified_source_f32_addends",
        }


def recognize_masked_row_softmax(request: KernelRequest, result_id: str) -> MaskedRowSoftmaxSource:
    """Recognize score + broadcast mask without assuming values or mask patterns.

    Numerical admission (for example, zero/-infinity masks with a nonempty
    row) belongs to the selected lowering policy. This recognizer establishes
    source structure, affine bounds and the absence of escaping intermediates.
    """
    region = recognize_row_softmax(request, result_id)
    masked = request.node(region.scores)
    identity = pointwise_identity_maps(len(region.shape), 1)[0]
    if (
        masked.op != "f32_add"
        or masked.effect != "pure"
        or len(masked.inputs) != 2
        or dict(masked.attrs) != {"numeric_semantics": "ieee754-f32-add-rne-v1"}
        or len(masked.index_maps) != 3
        or masked.index_maps[0] != identity
        or masked.index_maps[-1] != identity
    ):
        raise ValueError("source masked softmax: expected score + affine mask")
    scores, mask = (request.node(key) for key in masked.inputs)
    mapping = masked.index_maps[1]
    if (
        scores.id == mask.id
        or scores.type != masked.type
        or mask.type.dtype != "f32"
        or mask.type.numerical_policy != "raw-bit-pattern"
        or scores.effect not in {"input", "constant", "pure"}
        or mask.effect not in {"input", "constant", "pure"}
        or mapping.loop_rank != len(region.shape)
        or len(mapping.coefficients) != len(mask.type.shape)
    ):
        raise ValueError("source masked softmax: operand types or mask rank differ")
    for coefficients, offset, extent in zip(mapping.coefficients, mapping.offsets, mask.type.shape, strict=True):
        low = offset + sum(min(0, c * (n - 1)) for c, n in zip(coefficients, region.shape, strict=True))
        high = offset + sum(max(0, c * (n - 1)) for c, n in zip(coefficients, region.shape, strict=True))
        if low < 0 or high >= extent:
            raise ValueError("source masked softmax: mask map leaves its source")
    if masked.id in request.outputs or any(
        node.id not in region.computations and masked.id in node.inputs for node in request.nodes
    ):
        raise ValueError("source masked softmax: mask addition has an outside consumer")
    computations = {*region.computations, masked.id}
    return MaskedRowSoftmaxSource(
        region.source_digest,
        scores.id,
        region.result,
        region.shape,
        tuple(node.id for node in request.nodes if node.id in computations),
        region.initializers,
        mask.id,
        mapping,
        masked.id,
        mask.type.shape,
    )

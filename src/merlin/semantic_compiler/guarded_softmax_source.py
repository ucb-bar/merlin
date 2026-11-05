"""Recognize Boolean masking and an explicit zero result for fully masked rows."""

from dataclasses import dataclass
from math import prod

from .model import IndexMap, TensorType, pointwise_identity_maps
from .softmax_source import MaskedRowSoftmaxSource, _recognize_row_softmax


@dataclass(frozen=True)
class GuardedMaskedRowSoftmaxSource(MaskedRowSoftmaxSource):
    normalization_result: str

    def record(self):
        return {
            **super().record(),
            "mask_values": "boolean-true-excludes-score",
            "all_masked_rows": "positive-zero",
            "source_guard": "row-maximum-equals-negative-infinity",
            "normalization_result": self.normalization_result,
        }


def recognize_guarded_masked_row_softmax(request, result_id):
    """Check the complete select(mask,-inf,scores) -> guarded softmax region.

    The guard may share the normalization maximum or repeat its exact source
    operation. No finite-value or numerical-equivalence assumption is made here;
    the selected target policy must separately qualify its arithmetic domain.
    """
    computations, literals = set(), set()

    def require(condition, message):
        if not condition:
            raise ValueError("source guarded softmax: " + message)

    def pure(key, operation, arity):
        node = request.node(key)
        require(
            node.effect == "pure" and node.op == operation and len(node.inputs) == arity,
            "unexpected " + operation + " operation",
        )
        computations.add(key)
        return node

    result = pure(result_id, "select", 3)
    shape, rank = result.type.shape, len(result.type.shape)
    require(rank >= 2 and result.type == TensorType(shape, "f32", "raw-bit-pattern"), "result type differs")
    identity = pointwise_identity_maps(rank, 1)[0]
    require(not result.attrs and result.index_maps == (identity,) * 4, "result select maps differ")
    column = (*shape[:-1], 1)
    column_type = TensorType(column, "f32", "raw-bit-pattern")

    def literal(key, bits, target_shape):
        node = request.node(key)
        require(node.type == TensorType(target_shape, "f32", "raw-bit-pattern"), "literal type differs")
        if node.op == "gather":
            node = pure(key, "gather", 1)
            require(
                not node.attrs and node.index_maps == (IndexMap(rank, (), ()), identity),
                "literal broadcast map differs",
            )
            node = request.node(node.inputs[0])
            require(node.type == TensorType((), "f32", "raw-bit-pattern"), "broadcast literal is not scalar")
        binding = next((b for b in request.constants if b.node_id == node.id), None)
        require(
            node.op == "constant"
            and node.effect == "constant"
            and not node.attrs
            and not node.index_maps
            and binding is not None
            and binding.encoding == "f32-le-raw"
            and bytes.fromhex(binding.data_hex) == bits * prod(node.type.shape),
            "literal bytes differ",
        )
        literals.add(node.id)

    literal(result.inputs[1], bytes(4), shape)
    condition = pure(result.inputs[0], "gather", 1)
    projection = IndexMap(rank, (*identity.coefficients[:-1], (0,) * rank), (0,) * rank)
    require(
        condition.type == TensorType(shape, "i1", "boolean")
        and not condition.attrs
        and condition.index_maps == (projection, identity),
        "guard broadcast differs",
    )
    comparison = pure(condition.inputs[0], "mask", 2)
    require(
        comparison.type == TensorType(column, "i1", "boolean")
        and dict(comparison.attrs) == {"operation": "arith.cmpf", "predicate": "oeq"}
        and comparison.index_maps == pointwise_identity_maps(rank, 2),
        "guard is not ordered equality",
    )
    literal(comparison.inputs[1], bytes.fromhex("000080ff"), column)
    view = pure(comparison.inputs[0], "reshape", 1)
    require(view.type == column_type, "guard maximum shape differs")
    while True:
        require(
            not view.attrs
            and not view.index_maps
            and view.type.dtype == "f32"
            and view.type.numerical_policy == "raw-bit-pattern"
            and prod(view.type.shape) == prod(shape[:-1]),
            "guard reshape changes row values",
        )
        parent = request.node(view.inputs[0])
        if parent.op != "reshape":
            break
        view = pure(parent.id, "reshape", 1)
    guard_max = pure(parent.id, "source_f32_reduce_max_unqualified", 2)
    normalized = request.node(result.inputs[2])
    require(normalized.type == result.type, "normalization result type differs")
    core = _recognize_row_softmax(request, normalized.id, computations)
    maximum = next(
        request.node(key) for key in core.computations if request.node(key).op == "source_f32_reduce_max_unqualified"
    )
    literal(guard_max.inputs[1], bytes.fromhex("000080ff"), shape[:-1])
    require(
        guard_max.inputs[0] == maximum.inputs[0] and guard_max.semantic_key() == maximum.semantic_key(),
        "guard maximum differs from normalization maximum",
    )
    masked = pure(core.scores, "select", 3)
    require(
        masked.type == result.type and not masked.attrs and masked.index_maps == (identity,) * 4,
        "score mask select differs",
    )
    literal(masked.inputs[1], bytes.fromhex("000080ff"), shape)
    mask_view = request.node(masked.inputs[0])
    if mask_view.op == "gather":
        mask_view = pure(mask_view.id, "gather", 1)
        require(
            not mask_view.attrs
            and len(mask_view.index_maps) == 2
            and mask_view.index_maps[-1] == identity
            and mask_view.type == TensorType(shape, "i1", "boolean"),
            "mask broadcast differs",
        )
        mask = request.node(mask_view.inputs[0])
        mapping = mask_view.index_maps[0]
    else:
        mask, mapping = mask_view, identity
    scores = request.node(masked.inputs[2])
    require(
        scores.type == result.type
        and scores.effect in {"input", "constant", "pure"}
        and mask.effect in {"input", "constant", "pure"}
        and mask.type == TensorType(mask.type.shape, "i1", "boolean")
        and mapping.loop_rank == rank
        and len(mapping.coefficients) == len(mask.type.shape),
        "mask or scores have incompatible types",
    )
    for row, offset, extent in zip(mapping.coefficients, mapping.offsets, mask.type.shape, strict=True):
        low = offset + sum(min(0, c * (n - 1)) for c, n in zip(row, shape, strict=True))
        high = offset + sum(max(0, c * (n - 1)) for c, n in zip(row, shape, strict=True))
        require(0 <= low <= high < extent, "mask map leaves its source")
    computations.update(core.computations)
    hidden = computations - {result_id}
    require(not hidden.intersection(request.outputs), "intermediate is a graph output")
    require(
        not any(n.id not in computations and hidden.intersection(n.inputs) for n in request.nodes),
        "intermediate has an outside consumer",
    )
    return GuardedMaskedRowSoftmaxSource(
        core.source_digest,
        scores.id,
        result_id,
        shape,
        tuple(n.id for n in request.nodes if n.id in computations),
        tuple(n.id for n in request.nodes if n.id in literals | set(core.initializers)),
        mask.id,
        mapping,
        masked.id,
        mask.type.shape,
        normalized.id,
    )

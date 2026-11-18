"""Recognize private stable softmax followed by a complete value contraction.

This records source structure only. Quantizing unnormalized exponentials and
normalizing after the value contraction requires a separately selected target
precision policy; it is not an equality rewrite of the source graph.
"""

from dataclasses import dataclass
from math import prod

from .model import IndexMap, KernelRequest, pointwise_identity_maps
from .softmax_source import MaskedRowSoftmaxSource, recognize_masked_row_softmax, recognize_row_softmax


@dataclass(frozen=True)
class SoftmaxValueSource:
    source_digest: str
    scores: str
    values: str
    probabilities: str
    result: str
    batch_shape: tuple[int, ...]
    query_rows: int
    keys: int
    channels: int
    computations: tuple[str, ...]
    initializers: tuple[str, ...]
    mask: str | None = None

    def record(self):
        return {
            "schema": "merlin.source_softmax_value.v1",
            "source_request_digest": self.source_digest,
            "scores": self.scores,
            "values": self.values,
            "probabilities": self.probabilities,
            "result": self.result,
            "batch_shape": list(self.batch_shape),
            "query_rows": self.query_rows,
            "keys": self.keys,
            "channels": self.channels,
            "computations": list(self.computations),
            "initializers": list(self.initializers),
            "mask": self.mask,
            "source_reduction_order": "unresolved",
            "source_math_exp": "unqualified",
            "probability_boundary": "private",
        }


def recognize_softmax_value(request: KernelRequest, result_id: str, *, masked=False) -> SoftmaxValueSource:
    """Check dense independent batches, positive-zero seed, and private P.

    The result may escape. Probability, maximum, exponentials, and sum must not
    escape. Scores and values remain ordinary source boundaries, so upstream
    Q/K projections, GQA repetition, and score scaling retain their own source
    identities and cannot be silently absorbed or relabeled by this matcher.
    """
    def require(condition, message):
        if not condition:
            raise ValueError("source softmax value: " + message)

    require(not request.physical_extents and not request.boundary_layouts and not request.output_layouts,
            "requires an unallocated logical graph")
    result = request.node(result_id)
    attrs = dict(result.attrs)
    order = attrs.pop("scalar_add_order", None)
    require(result.op == "source_f32_matmul_unqualified" and result.effect == "pure"
            and len(result.inputs) == 3
            and order in {None, "accumulator-product", "product-accumulator"}
            and attrs == {"reduction_order": "unresolved", "numeric_semantics": "source-f32-mul-add-unqualified-v1"},
            "expected a complete source f32 value contraction")
    probability, values, seed = (request.node(key) for key in result.inputs)
    rank = len(result.type.shape)
    require(rank >= 2 and all(node.type.dtype == "f32"
            and node.type.numerical_policy == "raw-bit-pattern" and len(node.type.shape) == rank
            for node in (probability, values, seed, result)), "expected dense rank-matched f32 tensors")
    batch = result.type.shape[:-2]
    rows, channels = result.type.shape[-2:]
    keys = probability.type.shape[-1]
    require(probability.type.shape == (*batch, rows, keys)
            and values.type.shape == (*batch, keys, channels) and seed.type == result.type,
            "value contraction changes batches or matrix geometry")
    loops = rank + 1
    axes = tuple(tuple(int(i == j) for j in range(loops)) for i in range(loops))
    prefix = axes[:rank - 2]
    maps = tuple(IndexMap(loops, (*prefix, *pair), (0,) * rank) for pair in (
        (axes[-3], axes[-1]), (axes[-1], axes[-2]), (axes[-3], axes[-2])))
    require(result.index_maps == (*maps, maps[-1]), "value contraction mixes batch, query, key, or channel axes")
    binding = next((item for item in request.constants if item.node_id == seed.id), None)
    require(seed.op == "constant" and seed.effect == "constant" and not seed.attrs and not seed.index_maps
            and binding is not None and binding.encoding == "f32-le-raw"
            and bytes.fromhex(binding.data_hex) == bytes(4 * prod(result.type.shape)),
            "value contraction needs an exact positive-zero initializer")
    # model2MLIR flattens independent leading dimensions around batch_matmul.
    # Only element-order-preserving views and identity copies are transparent.
    views = []
    original = probability
    while original.op in {"reshape", "copy", "gather"} and original.effect == "pure":
        require(len(original.inputs) == 1 and not original.attrs, "probability view has extra inputs or attributes")
        parent = request.node(original.inputs[0])
        require(parent.type.dtype == "f32" and parent.type.numerical_policy == "raw-bit-pattern"
                and prod(parent.type.shape) == prod(original.type.shape), "probability view changes type or element count")
        require((original.op == "reshape" and not original.index_maps)
                or (parent.type == original.type and original.index_maps == pointwise_identity_maps(len(parent.type.shape), 1)),
                "probability view changes element order")
        views.append(original.id)
        original = parent
    softmax = (recognize_masked_row_softmax if masked else recognize_row_softmax)(request, original.id)
    require(softmax.shape[-2:] == (rows, keys) and prod(softmax.shape[:-2]) == prod(batch),
            "softmax and contraction batch geometry differ")
    private = {original.id, *views}
    allowed = {*views, result.id}
    require(not private.intersection(request.outputs)
            and all(node.id in allowed or not private.intersection(node.inputs) for node in request.nodes),
            "probabilities have an outside consumer or are a graph output")
    computations = {*softmax.computations, *views, result.id}
    return SoftmaxValueSource(
        request.digest(), softmax.scores, values.id, original.id, result.id, batch,
        rows, keys, channels, tuple(node.id for node in request.nodes if node.id in computations),
        (*softmax.initializers, seed.id), softmax.mask if isinstance(softmax, MaskedRowSoftmaxSource) else None,
    )

"""Recognize private source RMS normalization without choosing target precision."""

from dataclasses import dataclass
from math import prod
import struct

from .model import IndexMap, TensorType, pointwise_identity_maps


@dataclass(frozen=True)
class RMSNormSource:
    source_digest: str
    hidden: str
    gamma: str
    result: str
    shape: tuple[int, ...]
    epsilon_bits: str
    computations: tuple[str, ...]
    initializers: tuple[str, ...]
    hidden_consumers: tuple[str, ...]

    @property
    def matrix_shape(self):
        return prod(self.shape[:-1]), self.shape[-1]

    def record(self):
        return dict(schema="merlin.source_rmsnorm.v1", source_request_digest=self.source_digest,
                    hidden=self.hidden, gamma=self.gamma, result=self.result, shape=list(self.shape),
                    epsilon_bits=self.epsilon_bits, computations=list(self.computations),
                    initializers=list(self.initializers), preserved_hidden_consumers=list(self.hidden_consumers),
                    source_reduction_order="unresolved", source_rsqrt="unqualified",
                    source_intermediates="private", dense_row_view=list(self.matrix_shape))


def recognize_rmsnorm(request, result_id, *, epsilon_bits=None):
    """Check H*rsqrt(mean(H*H)+epsilon)*gamma and every private use.

    H and gamma remain ordinary boundaries, including upstream producers and
    independent residual consumers. Exact literals and affine broadcasts are
    checked; numerical admission and the BF16 hierarchy are separate policies.
    Infer any finite positive binary32 epsilon from its original bytes unless
    the caller supplies an explicit epsilon_bits constraint.
    """
    def require(condition, reason):
        if not condition:
            raise ValueError("source RMSNorm: " + reason)

    require(not request.physical_extents and not request.boundary_layouts and not request.output_layouts,
            "requires an unallocated logical graph")
    computations, literals = set(), set()
    full = request.node(result_id).type
    shape, rank = full.shape, len(full.shape)
    require(rank >= 2 and full == TensorType(shape, "f32", "raw-bit-pattern"), "expected dense f32 rows")
    identity = pointwise_identity_maps(rank, 1)[0]

    def projection(source_shape, target_shape):
        n = len(target_shape)
        require(len(source_shape) <= n, "broadcast rank exceeds result")
        axes = pointwise_identity_maps(n, 1)[0].coefficients
        offset = n - len(source_shape)
        require(all(a == 1 or a == b for a, b in zip(source_shape, target_shape[offset:])),
                "broadcast extent differs")
        return IndexMap(n, tuple((0,) * n if size == 1 else axes[offset+i]
                                 for i, size in enumerate(source_shape)), (0,) * len(source_shape))

    def binary(key, operation):
        value = request.node(key)
        require(value.effect == "pure" and value.op == "f32_" + operation and len(value.inputs) == 2
                and dict(value.attrs) == {"numeric_semantics": f"ieee754-f32-{operation}-rne-v1"}
                and value.type.dtype == "f32" and value.type.numerical_policy == "raw-bit-pattern",
                "unexpected " + operation + " expression")
        expected = tuple(projection(request.node(k).type.shape, value.type.shape) for k in value.inputs)
        # Singleton dimensions may be represented by their identity axis in an
        # equal-shape pointwise op; both maps address precisely the same lanes.
        canonical = pointwise_identity_maps(len(value.type.shape), 1)[0]
        def equivalent(mapping, want):
            normalize = lambda rows: tuple(tuple(c if size != 1 else 0 for c, size in zip(row, value.type.shape)) for row in rows)
            return (mapping.loop_rank == want.loop_rank and mapping.offsets == want.offsets
                    and normalize(mapping.coefficients) == normalize(want.coefficients))
        require(len(value.index_maps) == 3 and value.index_maps[-1] == canonical
                and all(equivalent(mapping, want) for mapping, want in zip(value.index_maps, expected)),
                "arithmetic broadcast maps differ")
        computations.add(value.id)
        return value

    def views(key, *, count):
        value = request.node(key)
        while value.op in {"reshape", "copy", "gather"} and value.effect == "pure":
            require(len(value.inputs) == 1 and not value.attrs, "view carries extra semantics")
            parent = request.node(value.inputs[0])
            require(parent.type.dtype == value.type.dtype == "f32"
                    and parent.type.numerical_policy == value.type.numerical_policy == "raw-bit-pattern"
                    and prod(parent.type.shape) == prod(value.type.shape) == count,
                    "view changes element count or type")
            require((value.op == "reshape" and not value.index_maps)
                    or (parent.type == value.type
                        and value.index_maps == pointwise_identity_maps(len(value.type.shape), 1)),
                    "view changes element order")
            computations.add(value.id)
            value = parent
        return value

    def literal(key, expected, target_shape):
        value = request.node(key)
        require(value.type.dtype == "f32" and value.type.numerical_policy == "raw-bit-pattern",
                "literal type differs")
        if value.op == "gather" and value.effect == "pure":
            require(len(value.inputs) == 1 and not value.attrs and value.type.shape == target_shape,
                    "literal gather shape differs")
            parent = request.node(value.inputs[0])
            require(parent.type.shape == () and value.index_maps ==
                    (IndexMap(len(target_shape), (), ()), pointwise_identity_maps(len(target_shape), 1)[0]),
                    "literal gather map differs")
            computations.add(value.id)
            value = parent
        value = views(value.id, count=prod(value.type.shape))
        require(value.type.shape in {(), target_shape}, "literal shape differs")
        binding = next((b for b in request.constants if b.node_id == value.id), None)
        require(value.op == "constant" and value.effect == "constant" and not value.attrs and not value.index_maps
                and binding is not None and binding.encoding == "f32-le-raw", "literal bytes differ")
        raw = bytes.fromhex(binding.data_hex)
        if expected is None:
            expected = raw[:4]
        require(len(expected) == 4 and raw == expected * prod(value.type.shape), "literal bytes differ")
        literals.add(value.id)
        return expected

    result = binary(result_id, "mul")
    require(result.type == full, "result type differs")
    # Gamma is the last multiplication, independent of scalar operand order.
    candidates = [(request.node(result.inputs[i]), request.node(result.inputs[1-i])) for i in (0, 1)]
    candidates = [(n, g) for n, g in candidates if n.type == full and n.op == "f32_mul"
                  and g.type.shape in {(shape[-1],), (1,) * (rank-1) + (shape[-1],)}]
    require(len(candidates) == 1, "expected a final gamma broadcast product")
    normalized, gamma_view = candidates[0]
    gamma = views(gamma_view.id, count=shape[-1])
    require(gamma.type.shape in {(shape[-1],), (1,) * (rank-1) + (shape[-1],)}
            and gamma.effect in {"input", "constant", "pure"}, "gamma boundary differs")
    normalized = binary(normalized.id, "mul")
    candidates = [(request.node(normalized.inputs[i]), request.node(normalized.inputs[1-i])) for i in (0, 1)]
    candidates = [(h, inv) for h, inv in candidates
                  if h.type == full and inv.type == TensorType((*shape[:-1], 1), "f32", "raw-bit-pattern")]
    require(len(candidates) == 1, "expected H times broadcast inverse")
    hidden, inverse_view = candidates[0]
    require(hidden.effect in {"input", "constant", "pure"}, "H has unresolved effects")
    inverse = views(inverse_view.id, count=prod(shape[:-1]))
    require(inverse.op == "source_f32_rsqrt_unqualified" and inverse.effect == "pure"
            and len(inverse.inputs) == 1
            and dict(inverse.attrs) == {"numeric_semantics": "source-f32-rsqrt-unqualified-v1"}
            and inverse.index_maps == pointwise_identity_maps(len(inverse.type.shape), 1), "rsqrt expression differs")
    computations.add(inverse.id)
    shifted = binary(inverse.inputs[0], "add")
    require(shifted.type == inverse.type, "epsilon addition type differs")
    mean_candidates = []
    for i in (0, 1):
        candidate = request.node(shifted.inputs[i])
        while candidate.op == "reshape" and len(candidate.inputs) == 1:
            candidate = request.node(candidate.inputs[0])
        if candidate.op in {"f32_div", "f32_mul"}:
            mean_candidates.append((shifted.inputs[i], shifted.inputs[1-i]))
    require(len(mean_candidates) == 1, "mean and epsilon are not explicit")
    mean_id, epsilon = mean_candidates[0]
    source_epsilon = int.from_bytes(literal(epsilon, None, shifted.type.shape), "little")
    require(0 < source_epsilon < 0x7f800000, "epsilon must be finite and positive")
    source_epsilon_bits = f"0x{source_epsilon:08x}"
    require(epsilon_bits is None or epsilon_bits == source_epsilon_bits,
            "source epsilon differs from selected epsilon")
    epsilon_bits = source_epsilon_bits
    mean = views(mean_id, count=prod(shape[:-1]))
    mean = binary(mean.id, mean.op[4:])
    require(mean.type.shape in {shape[:-1], (*shape[:-1], 1)}, "mean type differs")
    # Division by exact width, or multiplication by an exact reciprocal width.
    choices = [(mean.inputs[0], mean.inputs[1])]
    if mean.op == "f32_mul":
        choices.append((mean.inputs[1], mean.inputs[0]))
    reduction_choice = []
    for data, factor in choices:
        candidate = request.node(data)
        while candidate.op == "reshape" and len(candidate.inputs) == 1:
            candidate = request.node(candidate.inputs[0])
        if candidate.op == "source_f32_reduce_add_unqualified":
            reduction_choice.append((data, factor))
    require(len(reduction_choice) == 1, "expected a private source mean reduction")
    data, factor = reduction_choice[0]
    literal(factor, struct.pack("<f", shape[-1] if mean.op == "f32_div" else 1 / shape[-1]), mean.type.shape)
    reduction = views(data, count=prod(shape[:-1]))
    prefix = IndexMap(rank, identity.coefficients[:-1], (0,) * (rank-1))
    require(reduction.type == TensorType(shape[:-1], "f32", "raw-bit-pattern")
            and reduction.effect == "pure" and len(reduction.inputs) == 2
            and dict(reduction.attrs) == {"axis": rank-1, "reduction_order": "unresolved",
                "numeric_semantics": "source-f32-add-unqualified-v1", "scalar_add_order": "input-accumulator"}
            and reduction.index_maps == (identity, prefix, prefix), "source reduction order, type, or maps differ")
    computations.add(reduction.id)
    literal(reduction.inputs[1], bytes(4), shape[:-1])
    square = binary(reduction.inputs[0], "mul")
    require(square.type == full and square.inputs == (hidden.id, hidden.id)
            and square.index_maps == pointwise_identity_maps(rank, 2), "square is not H times itself")
    private = computations - {result.id}
    require(not private.intersection(request.outputs)
            and all(n.id in computations or not private.intersection(n.inputs) for n in request.nodes),
            "normalization intermediate escapes")
    require(hidden.id not in computations and gamma.id not in computations, "boundary is absorbed by the norm")
    return RMSNormSource(request.digest(), hidden.id, gamma.id, result.id, shape, epsilon_bits,
                         tuple(n.id for n in request.nodes if n.id in computations),
                         tuple(n.id for n in request.nodes if n.id in literals),
                         tuple(n.id for n in request.nodes if n.id not in computations and hidden.id in n.inputs))

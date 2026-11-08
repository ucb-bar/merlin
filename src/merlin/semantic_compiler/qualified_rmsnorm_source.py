"""Recognize decomposed source square/mean/rsqrt/scale RMSNorm exactly."""
from dataclasses import dataclass
import struct

from .model import IndexMap, TensorType, pointwise_identity_maps


@dataclass(frozen=True)
class RMSNormSource:
    source_digest: str
    data: str
    gamma: str
    result: str
    shape: tuple[int, int]
    epsilon_f32_bits: int
    computations: tuple[str, ...]
    initializers: tuple[str, ...]

    def record(self):
        return dict(source_request_digest=self.source_digest, data=self.data, gamma=self.gamma,
                    result=self.result, shape=list(self.shape), epsilon_f32_bits=self.epsilon_f32_bits,
                    computations=list(self.computations), initializers=list(self.initializers),
                    source_reduction_order="unresolved", source_rsqrt="unqualified")


def recognize_rmsnorm(source, result_id):
    def require(test, reason):
        if not test:
            raise ValueError("source RMSNorm: " + reason)

    constants = {c.node_id: c for c in source.constants}
    computations, initializers = set(), set()

    def node(key, op, arity, attrs, maps):
        n = source.node(key)
        require(n.op == op and n.effect == "pure" and len(n.inputs) == arity
                and dict(n.attrs) == attrs and n.index_maps == maps, "unsupported " + op + " body, indexing, or attributes")
        require(n.type.dtype == "f32" and n.type.numerical_policy == "raw-bit-pattern", "expected f32 source arithmetic")
        computations.add(key)
        return n

    def literal(key, shape, expected=None):
        n = source.node(key)
        require(n.effect == n.op == "constant" and n.type == TensorType(shape, "f32", "raw-bit-pattern")
                and not n.attrs and not n.index_maps and not n.inputs, "invalid literal tensor")
        c = constants.get(key)
        require(c is not None and c.encoding == "f32-le-raw", "literal bytes are not authenticated f32")
        raw = bytes.fromhex(c.data_hex)
        codes = tuple(code for (code,) in struct.iter_unpack("<I", raw))
        require(codes and len(set(codes)) == 1, "literal tensor is not a splat")
        if expected is not None:
            require(codes[0] == expected, "wrong literal value")
        initializers.add(key)
        return codes[0]

    out = source.node(result_id)
    shape = out.type.shape
    require(len(shape) == 2, "expected a matrix")
    rows, width = shape
    identity = pointwise_identity_maps(2, 1)[0]
    gamma_map = IndexMap(2, ((0, 1),), (0,))
    final_maps = ((gamma_map, identity, identity), (identity, gamma_map, identity))
    require(out.index_maps in final_maps, "gamma broadcast must cover columns")
    out = node(result_id, "f32_mul", 2, {"numeric_semantics": "ieee754-f32-mul-rne-v1"}, out.index_maps)
    gamma_index = 0 if out.index_maps == final_maps[0] else 1
    gamma = source.node(out.inputs[gamma_index])
    normalized = source.node(out.inputs[1 - gamma_index])
    require(gamma.type == TensorType((width,), "f32", "raw-bit-pattern"), "gamma shape differs")
    broadcast = IndexMap(2, ((1, 0), (0, 0)), (0, 0))
    normalized = node(normalized.id, "f32_mul", 2, {"numeric_semantics": "ieee754-f32-mul-rne-v1"}, (identity, broadcast, identity))
    data = source.node(normalized.inputs[0])
    require(data.type == normalized.type == out.type == TensorType(shape, "f32", "raw-bit-pattern"), "data and result shapes differ")
    inverse = node(normalized.inputs[1], "source_f32_rsqrt_unqualified", 1,
                   {"numeric_semantics": "source-f32-rsqrt-unqualified-v1"}, pointwise_identity_maps(2, 1))
    shifted = node(inverse.inputs[0], "f32_add", 2, {"numeric_semantics": "ieee754-f32-add-rne-v1"}, pointwise_identity_maps(2, 2))
    require(shifted.type == inverse.type == TensorType((rows, 1), "f32", "raw-bit-pattern"), "row-statistic shape differs")
    epsilon = literal(shifted.inputs[1], (rows, 1))
    require(0 < epsilon < 0x7f800000, "epsilon must be positive finite")
    view = node(shifted.inputs[0], "reshape", 1, {}, ())
    require(view.type == shifted.type, "mean broadcast reshape differs")
    mean = node(view.inputs[0], "f32_div", 2, {"numeric_semantics": "ieee754-f32-div-rne-v1"}, pointwise_identity_maps(1, 2))
    require(mean.type == TensorType((rows,), "f32", "raw-bit-pattern"), "mean shape differs")
    literal(mean.inputs[1], (rows,), struct.unpack("<I", struct.pack("<f", float(width)))[0])
    row_map = IndexMap(2, ((1, 0),), (0,))
    reduction = source.node(mean.inputs[0])
    require(reduction.op == "source_f32_reduce_add_unqualified", "expected unresolved source sum")
    reduction = node(reduction.id, reduction.op, 2,
                     dict(axis=1, reduction_order="unresolved", numeric_semantics="source-f32-add-unqualified-v1", scalar_add_order="input-accumulator"),
                     (identity, row_map, row_map))
    require(reduction.type == mean.type, "sum shape differs")
    literal(reduction.inputs[1], (rows,), 0)
    square = node(reduction.inputs[0], "f32_mul", 2, {"numeric_semantics": "ieee754-f32-mul-rne-v1"}, pointwise_identity_maps(2, 2))
    require(square.inputs == (data.id, data.id) and square.type == data.type, "expected square of the same input")
    for key in computations - {result_id}:
        require(key not in source.outputs and all(n.id in computations for n in source.nodes if key in n.inputs), "intermediate escapes normalization")
    require(data.id not in computations and gamma.id not in computations, "normalization boundary cycle")
    return RMSNormSource(source.digest(), data.id, gamma.id, result_id, shape, epsilon,
                         tuple(n.id for n in source.nodes if n.id in computations),
                         tuple(n.id for n in source.nodes if n.id in initializers))

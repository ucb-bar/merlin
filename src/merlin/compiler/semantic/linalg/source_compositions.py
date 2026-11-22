"""Recover single BF16 source ops that the frontend decomposed, as one declared node.

model2MLIR lowers a BF16 GELU, layer norm, or unfolded (im2col) convolution to
an exact widening, binary32 opmath and one final BF16 rounding spread over
several generics. When every intermediate of such a chain is private, the chain
is one source op: this pass names it and keeps the final node's identity.
Shared or escaping intermediates, other dtypes and other bodies keep their
ordinary nodes. No target implementation is selected here.

GELU keeps the chain's arithmetic and literal bits, and the convolution keeps
its unresolved reduction order. A layer norm node names the source aten op, whose
declared semantics is the pinned torch wheel's dispatched CPU kernel (chunked Welford moments), not the two-pass
moments of the decomposition.

Temporary: the generic replacement is model2MLIR emitting each of these ops as
one generic tagged with its numerical semantics, as it already does for BF16
sigmoid and batch norm; this pass is then deleted.
"""

from __future__ import annotations

from dataclasses import replace
from math import prod
from struct import pack, unpack

from merlin.compiler.semantic.linalg.linalg_recognition import _f32_reduce_maps
from merlin.compiler.semantic.model import IndexMap, SemanticNode, TensorType, pointwise_identity_maps

GELU_SEMANTICS = "bf16-gelu-f32-opmath-v1"
LAYER_NORM_SEMANTICS = "torch-2.11-cpu-fma-bf16-layer-norm-v1"
CONTRACTION_SEMANTICS = "source-bf16-f32-affine-contraction-v1"
FINITE_BF16 = "finite-bf16"
_GELU_FORMS = {
    "tanh_gelu_f32": ("tanh", "0x3f000000,0x3f800000,0x3d372713,0x3f4c422a"),
    "erf_gelu_f32": ("none", "0x3f000000,0x3f800000,0x3f3504f3"),
}


class _Graph:
    def __init__(self, nodes, outputs, constants):
        self.nodes = {node.id: node for node in nodes}
        self.outputs = set(outputs)
        self.constants = {binding.node_id: binding for binding in constants}
        self.users = {node.id: [] for node in nodes}
        for node in nodes:
            for key in node.inputs:
                self.users[key].append(node.id)

    def node(self, key):
        return self.nodes[key]


def _attrs(node):
    return {key: value for key, value in node.attrs if not key.startswith("prov.")}


def _is(node, op, semantics=None, inputs=None):
    return (node.op == op and node.effect == "pure"
            and (semantics is None or _attrs(node) == {"numeric_semantics": semantics})
            and (inputs is None or len(node.inputs) == inputs))


def _f32(node, shape=None):
    return node.type == TensorType(node.type.shape if shape is None else shape, "f32", "raw-bit-pattern")


def _cast(graph, key, op, shape):
    """An exact widening or RNE narrowing with identity maps, or None."""
    node = graph.node(key)
    semantics = "bf16-to-f32-exact-v1" if op == "bf16_to_f32" else "f32-to-bf16-rne-v1"
    if not _is(node, op, semantics, 1) or node.type.shape != shape:
        return None
    if node.index_maps != pointwise_identity_maps(len(shape), 1):
        return None
    source = graph.node(node.inputs[0])
    dtypes = ("bf16", "f32") if op == "bf16_to_f32" else ("f32", "bf16")
    if source.type != TensorType(shape, dtypes[0], "raw-bit-pattern") or node.type.dtype != dtypes[1]:
        return None
    return node


def _broadcast(mapping, operand_shape, result_shape):
    """True when ``mapping`` reads a trailing-aligned broadcast of the operand."""
    rank, offset = len(result_shape), len(result_shape) - len(operand_shape)
    if offset < 0 or mapping.loop_rank != rank or len(mapping.coefficients) != len(operand_shape) or any(mapping.offsets):
        return False
    for axis, (row, extent) in enumerate(zip(mapping.coefficients, operand_shape)):
        unit = tuple(int(i == offset + axis) for i in range(rank))
        if extent == 1:
            if any(c and result_shape[i] != 1 for i, c in enumerate(row)):
                return False
        elif row != unit or result_shape[offset + axis] != extent:
            return False
    return True


def _binary(graph, key, op, shape):
    node = graph.node(key)
    if not _is(node, op, f"ieee754-f32-{op[4:]}-rne-v1", 2) or not _f32(node, shape):
        return None
    if len(node.index_maps) != 3 or node.index_maps[-1] != pointwise_identity_maps(len(shape), 1)[0]:
        return None
    if not all(_broadcast(m, graph.node(k).type.shape, shape) for m, k in zip(node.index_maps, node.inputs)):
        return None
    return node


def _splat(graph, key, shape):
    """Raw little-endian f32 bits of a constant splat of ``shape``, or None."""
    node = graph.node(key)
    binding = graph.constants.get(key)
    if (node.op != "constant" or node.effect != "constant" or node.inputs or node.index_maps or _attrs(node)
            or not _f32(node, shape) or binding is None or binding.encoding != "f32-le-raw"):
        return None
    raw = bytes.fromhex(binding.data_hex)
    if len(raw) != 4 * prod(shape) or raw != raw[:4] * prod(shape):
        return None
    return int.from_bytes(raw[:4], "little")


def _views(graph, key, shape, private):
    """Follow private dense row-major reshapes down to their f32 producer."""
    node = graph.node(key)
    while node.op == "reshape" and node.effect == "pure":
        if (len(node.inputs) != 1 or _attrs(node) or node.index_maps or node.id in graph.outputs
                or len(graph.users[node.id]) != 1):
            return None
        private.add(node.id)
        node = graph.node(node.inputs[0])
    if node.type.dtype != "f32" or prod(node.type.shape) != prod(shape):
        return None
    return node


def _private(graph, keys, root):
    """Every private value is read only inside the group and is not an output."""
    group = set(keys) | {root}
    return all(key not in graph.outputs and set(graph.users[key]) <= group for key in keys)


def _gelu(graph, narrow):
    shape = narrow.type.shape
    if not _cast(graph, narrow.id, "f32_to_bf16", shape):
        return None
    body = graph.node(narrow.inputs[0])
    attrs = _attrs(body)
    form = _GELU_FORMS.get(attrs.get("operation"))
    if (form is None or not _is(body, "source_f32_composite_unqualified", None, 1) or not _f32(body, shape)
            or attrs != {"numeric_semantics": "source-f32-composite-unqualified-v1",
                         "operation": attrs["operation"], "literal_bits": form[1]}
            or body.index_maps != pointwise_identity_maps(len(shape), 1)):
        return None
    wide = _cast(graph, body.inputs[0], "bf16_to_f32", shape)
    if wide is None or not _private(graph, (wide.id, body.id), narrow.id):
        return None
    node = SemanticNode(narrow.id, "bf16_gelu", wide.inputs, narrow.type,
                        attrs=(("approximate", form[0]), ("input_domain", FINITE_BF16),
                               ("numeric_semantics", GELU_SEMANTICS)),
                        index_maps=pointwise_identity_maps(len(shape), 1))
    return node, {wide.id, body.id}


def _mean(graph, key, x, axes, outer, private):
    """sum(x over axes) / count from a +0 seed, with the source order left unresolved."""
    count = prod(x.type.shape[axis] for axis in axes)
    divide = _binary(graph, key, "f32_div", outer)
    if divide is None or _splat(graph, divide.inputs[1], outer) != unpack("<I", pack("<f", float(count)))[0]:
        return None
    total = graph.node(divide.inputs[0])
    axis_attr = ("axis", axes[0]) if len(axes) == 1 else ("axes", ",".join(map(str, axes)))
    if (not _is(total, "source_f32_reduce_add_unqualified", None, 2) or not _f32(total, outer)
            or total.inputs[0] != x.id or _splat(graph, total.inputs[1], outer) != 0
            or total.attrs != (axis_attr, ("reduction_order", "unresolved"),
                               ("numeric_semantics", "source-f32-add-unqualified-v1"),
                               ("scalar_add_order", "input-accumulator"))
            or total.index_maps != _f32_reduce_maps(len(x.type.shape), axes)):
        return None
    private.update((divide.id, total.id))
    return divide


def _layer_norm(graph, narrow):
    shape = narrow.type.shape
    rank = len(shape)
    identity = pointwise_identity_maps(rank, 1)[0]
    if not _cast(graph, narrow.id, "f32_to_bf16", shape):
        return None
    private, affine = set(), {}
    value = graph.node(narrow.inputs[0])
    # Optional affine terms in source order: (normalized * weight) + bias, each widened exactly.
    for role, op in (("bias", "f32_add"), ("weight", "f32_mul")):
        node = _binary(graph, value.id, op, shape)
        term = graph.node(node.inputs[1]) if node is not None else None
        cast = _cast(graph, term.id, "bf16_to_f32", term.type.shape) if term is not None else None
        if cast is None or node.index_maps[0] != identity:
            continue
        affine[role] = cast
        private.update((node.id, cast.id))
        value = graph.node(node.inputs[0])
    normalized = _binary(graph, value.id, "f32_mul", shape)
    if normalized is None or normalized.index_maps[0] != identity:
        return None
    centered = _binary(graph, normalized.inputs[0], "f32_sub", shape)
    view = graph.node(normalized.inputs[1])
    inverse = _views(graph, view.id, view.type.shape, private)
    if centered is None or inverse is None or centered.index_maps[0] != identity:
        return None
    keep = inverse.type.shape
    if not (_is(inverse, "source_f32_rsqrt_unqualified", "source-f32-rsqrt-unqualified-v1", 1)
            and len(keep) == rank and view.type.shape == keep
            and inverse.index_maps == pointwise_identity_maps(rank, 1)):
        return None
    shifted = _binary(graph, inverse.inputs[0], "f32_add", keep)
    epsilon = _splat(graph, shifted.inputs[1], keep) if shifted is not None else None
    if epsilon is None or epsilon >= 0x7F800000:
        return None
    variance = _views(graph, shifted.inputs[0], keep, private)
    total = graph.node(variance.inputs[0]) if variance is not None and variance.op == "f32_div" else None
    attrs = dict(total.attrs) if total is not None else {}
    axes = (attrs["axis"],) if "axis" in attrs else tuple(int(a) for a in str(attrs.get("axes", "")).split(",") if a)
    k = len(axes)
    if not k or k >= rank or axes != tuple(range(rank - k, rank)) or keep != (*shape[:rank - k], *(1,) * k):
        return None
    outer = shape[:rank - k]
    square = _binary(graph, total.inputs[0], "f32_mul", shape)
    wide = _cast(graph, centered.inputs[0], "bf16_to_f32", shape)
    if (square is None or wide is None or square.inputs != (centered.id, centered.id)
            or square.index_maps != pointwise_identity_maps(rank, 2)
            or _mean(graph, variance.id, square, axes, outer, private) is None):
        return None
    mean = _views(graph, centered.inputs[1], keep, private)
    if mean is None or _mean(graph, mean.id, wide, axes, outer, private) is None:
        return None
    if any(cast.type.shape != shape[rank - k:] for cast in affine.values()):
        return None
    private.update((normalized.id, inverse.id, shifted.id, centered.id, square.id, wide.id))
    if not _private(graph, private, narrow.id):
        return None
    roles = tuple(role for role in ("weight", "bias") if role in affine)
    trailing = IndexMap(rank, tuple(tuple(int(i == j) for i in range(rank)) for j in axes), (0,) * k)
    node = SemanticNode(
        narrow.id, "bf16_layer_norm", (wide.inputs[0], *(affine[role].inputs[0] for role in roles)), narrow.type,
        attrs=(("affine_operands", ",".join(roles)), ("epsilon_f32_bits", epsilon), ("input_domain", FINITE_BF16),
               ("normalized_rank", k), ("numeric_semantics", LAYER_NORM_SEMANTICS)),
        index_maps=(identity, *(trailing,) * len(roles), identity))
    return node, private


def _unit(rank, axis):
    return tuple(int(i == axis) for i in range(rank))


def _gathered_contraction(graph, narrow):
    """bf16(sum_k f32(w[co,k]) * f32(gather(x)[k,p]) + f32(bias[co])), an unfolded convolution.

    The weight's trailing dimensions equal the gather's reduction dimensions and
    the gather's remaining dimensions are (1, spatial...), so the matmul's
    row-major [Cout, P] result is the NCHW output with batch one.
    """
    shape = narrow.type.shape
    if len(shape) < 2 or shape[0] != 1 or not _cast(graph, narrow.id, "f32_to_bf16", shape):
        return None
    private, rank = set(), len(shape)
    value, bias = graph.node(narrow.inputs[0]), None
    channel = IndexMap(rank, (_unit(rank, 1),), (0,))
    identity = pointwise_identity_maps(rank, 1)[0]
    if _is(value, "f32_add", "ieee754-f32-add-rne-v1", 2) and _f32(value, shape):
        bias = _cast(graph, value.inputs[1], "bf16_to_f32", (shape[1],))
        if bias is None or value.index_maps != (identity, channel, identity):
            return None
        private.update((value.id, bias.id))
        value = graph.node(value.inputs[0])
    matmul = _views(graph, value.id, shape, private)
    attrs = _attrs(matmul) if matmul is not None else {}
    attrs.pop("scalar_add_order", None)
    if (matmul is None or not _is(matmul, "source_f32_matmul_unqualified", None, 3)
            or attrs != {"reduction_order": "unresolved", "numeric_semantics": "source-f32-mul-add-unqualified-v1"}
            or matmul.index_maps != tuple(IndexMap(3, rows, (0, 0)) for rows in (
                ((1, 0, 0), (0, 0, 1)), ((0, 0, 1), (0, 1, 0)), ((1, 0, 0), (0, 1, 0)), ((1, 0, 0), (0, 1, 0))))):
        return None
    cout, positions = matmul.type.shape
    if cout != shape[1] or _splat(graph, matmul.inputs[2], (cout, positions)) != 0:
        return None
    lhs, rhs = (graph.node(key) for key in matmul.inputs[:2])
    weight = _views(graph, lhs.id, lhs.type.shape, private)
    gather = _views(graph, rhs.id, rhs.type.shape, private)
    if weight is None or gather is None or len(lhs.type.shape) != 2 or lhs.type.shape[0] != cout:
        return None
    k = lhs.type.shape[1]
    weight = _cast(graph, weight.id, "bf16_to_f32", weight.type.shape)
    if (weight is None or weight.type.shape[0] != cout or prod(weight.type.shape[1:]) != k
            or rhs.type.shape != (k, positions)):
        return None
    dims = gather.type.shape
    split = len(weight.type.shape) - 1
    if (gather.op != "gather" or gather.effect != "pure" or len(gather.inputs) != 1 or _attrs(gather)
            or dims[:split] != weight.type.shape[1:] or dims[split:] != (1, *shape[2:])
            or len(gather.index_maps) != 2 or gather.index_maps[1] != pointwise_identity_maps(len(dims), 1)[0]):
        return None
    source = _cast(graph, gather.inputs[0], "bf16_to_f32", graph.node(gather.inputs[0]).type.shape)
    if source is None:
        return None
    private.update((matmul.id, weight.id, gather.id, source.id))
    if not _private(graph, private, narrow.id):
        return None
    # One loop domain (co, gather dims...); the gather's leading dims are reduced.
    loops = (cout, *dims)
    width = len(loops)
    read = gather.index_maps[0]
    maps = [IndexMap(width, tuple((0, *row) for row in read.coefficients), read.offsets),
            IndexMap(width, tuple(_unit(width, axis) for axis in range(split + 1)), (0,) * (split + 1))]
    if bias is not None:
        maps.append(IndexMap(width, (_unit(width, 0),), (0,)))
    maps.append(IndexMap(width, (_unit(width, split + 1), _unit(width, 0),
                                 *(_unit(width, split + 2 + i) for i in range(rank - 2))), (0,) * rank))
    node = SemanticNode(
        narrow.id, "bf16_affine_contraction",
        (source.inputs[0], weight.inputs[0], *((bias.inputs[0],) if bias is not None else ())), narrow.type,
        attrs=(("affine_operands", "bias" if bias is not None else ""), ("input_domain", FINITE_BF16),
               ("loop_extents", ",".join(map(str, loops))), ("numeric_semantics", CONTRACTION_SEMANTICS),
               ("reduction_loops", ",".join(str(1 + i) for i in range(split))),
               ("reduction_order", "unresolved")),
        index_maps=tuple(maps))
    return node, private


def collapse_source_compositions(nodes, outputs, constants):
    """Return nodes and constant bindings with each closed chain as one node."""
    graph = _Graph(nodes, outputs, constants)
    replaced, removed = {}, set()
    for node in nodes:
        if node.op != "f32_to_bf16" or node.id in removed:
            continue
        for recognize in (_gelu, _layer_norm, _gathered_contraction):
            found = recognize(graph, node)
            if found is not None and not found[1] & (removed | set(replaced)):
                replaced[node.id] = found[0]
                removed |= found[1]
                break
    if not replaced:
        return tuple(nodes), tuple(constants)
    kept = [replaced.get(node.id, node) for node in nodes if node.id not in removed]
    used = {key for node in kept for key in node.inputs} | set(outputs)
    dropped = {node.id for node in kept if node.effect == "constant" and node.id not in used
               and any(node.id in graph.node(key).inputs for key in removed)}
    return (tuple(node for node in kept if node.id not in dropped),
            tuple(binding for binding in constants if binding.node_id not in dropped))

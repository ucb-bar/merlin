"""Whole-prefix MAC bounds using declared widths and literal source operands."""


def require_mesh_mac_bound(numeric, *, mesh_signed_bits=None):
    """Validate a whole-prefix bound, never just a cancellation-prone final sum.

    Both signed-byte inputs include -128, so a product can be +16384. No D
    payload is licensed here. The provider separately checks garbage mesh D,
    unit scales, and full-width accumulator reads on the actual schedule.
    """
    if type(mesh_signed_bits) is not int or mesh_signed_bits <= 1:
        raise ValueError("native arithmetic requires declared signed mesh MAC width")
    limit = 1 << (mesh_signed_bits - 1)
    proof = numeric.get("bound_proof")
    if not isinstance(proof, dict):
        raise ValueError("native arithmetic requires a per-primitive MAC bound")
    k = proof.get("primitive_k")
    if type(k) is not int or k <= 0:
        raise ValueError("native arithmetic requires a positive primitive K")
    bound = k * 128 * 128
    if bound >= limit:
        raise ValueError("a mesh MAC partial can wrap at the selected mesh width in WS or OS")
    declared = proof.get("max_mesh_partial_sum")
    bits = proof.get("required_signed_mac_bits")
    if (
        type(declared) is not int
        or not 0 <= declared < limit
        or type(bits) is not int
        or not 1 <= bits <= mesh_signed_bits
    ):
        raise ValueError("provider MAC bound is absent or exceeds the declared mesh width")
    return dict(
        method="all_prefixes_full_signed_byte_products",
        primitive_k=k,
        conservative_absolute_bound=bound,
        mesh_signed_bits=mesh_signed_bits,
        dataflows=["WS", "OS"],
    )


def constant_tensor(value):
    """Read only literal IR constants and exact splat/fill constructors.

    Runtime arguments and view operations deliberately refuse. In particular a
    transpose must never be mistaken for its flattened constant input.
    """
    import numpy as np
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerAttr

    shape = tuple(value.type.get_shape()) if hasattr(value.type, "get_shape") else ()
    if any(d <= 0 for d in shape):
        raise ValueError("constant tensor needs positive static extents")
    owner = value.owner
    name = getattr(owner, "name", None)
    if name == "arith.constant":
        attr = owner.properties.get("value", owner.attributes.get("value"))
        if isinstance(attr, IntegerAttr):
            return np.full(shape, int(attr.value.data), dtype=object)
        if isinstance(attr, DenseIntOrFPElementsAttr):
            listed = list(attr.get_values())
            if any(type(x) is not int for x in listed):
                raise ValueError("constant tensor must contain integers")
            if len(listed) == 1:
                return np.full(shape, listed[0], dtype=object)
            return np.asarray(listed, dtype=object).reshape(shape)
    if name in {"tensor.splat", "linalg.fill"}:
        scalar = constant_tensor(owner.operands[0])
        if scalar.size == 1:
            return np.full(shape, scalar.item(), dtype=object)
    raise ValueError("value bound requires constant weights and initializer in the IR")


def signed_matmul_shape(root):
    """Check rank-two signed i8 products widened into i32, returning M,K,N."""
    from xdsl.dialects.builtin import ArrayAttr
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp, MatmulOp

    from merlin.transforms.dense_ir import recognize_signed_integer_matmul

    if isinstance(root, (GenericOp, MatmulOp)):
        properties = dict(root.properties)
        if isinstance(root, MatmulOp):
            properties["iterator_types"] = ArrayAttr(
                [
                    IteratorTypeAttr(IteratorType.PARALLEL),
                    IteratorTypeAttr(IteratorType.PARALLEL),
                    IteratorTypeAttr(IteratorType.REDUCTION),
                ]
            )
        normalized = GenericOp.create(
            operands=root.operands,
            result_types=[v.type for v in root.results],
            attributes=dict(root.attributes),
            properties=properties,
            regions=[region.clone() for region in root.regions],
        )
        try:
            recognize_signed_integer_matmul(normalized)
        finally:
            normalized.drop_all_references()
    else:
        raise ValueError("value bound requires a signed rank-two matmul")
    values = (*root.operands, root.results[0])
    if [str(getattr(v.type, "element_type", None)) for v in values] != ["i8", "i8", "i32", "i32"]:
        raise ValueError("value bound requires signed i8 products and i32 accumulator")
    shapes = [tuple(v.type.get_shape()) for v in values]
    if any(len(s) != 2 or any(d <= 0 for d in s) for s in shapes):
        raise ValueError("value bound requires positive static rank-two shapes")
    (m, k), (rk, n), init, result = shapes
    if rk != k or init != (m, n) or result != init:
        raise ValueError("matmul shapes do not agree")
    return m, k, n


def require_value_mac_bound(root, *, mesh_signed_bits):
    """Prove every OS partial from literal weights, signed input range and seed.

    Summing the negative/positive product endpoints separately covers every
    prefix and every reduction order, including cancellation-prone weights.
    The bound includes the initializer; no assumption of a zero D is hidden.
    A refusal leaves the caller's independently admitted WS route available.
    """
    from merlin.transforms.certificates import ir_digest

    if type(mesh_signed_bits) is not int or mesh_signed_bits <= 1:
        raise ValueError("value bound requires declared signed mesh MAC width")
    m, k, n = signed_matmul_shape(root)
    weights = constant_tensor(root.operands[1])
    seed = constant_tensor(root.operands[2])
    lower, upper = [], []
    for column in range(n):
        low, high = int(seed[:, column].min()), int(seed[:, column].max())
        for weight in weights[:, column]:
            products = (-128 * int(weight), 127 * int(weight))
            low += min(0, *products)
            high += max(0, *products)
        lower.append(low)
        upper.append(high)
    limit = 1 << (mesh_signed_bits - 1)
    if min(lower) < -limit or max(upper) >= limit:
        raise ValueError("constant-weight OS partial can wrap at the selected mesh width")
    return {
        "method": "constant_weight_all_prefix_intervals",
        "dataflow": "OS",
        "source_ir_sha256": ir_digest(root),
        "shape_m_k_n": [m, k, n],
        "mesh_signed_bits": mesh_signed_bits,
        "input_range": [-128, 127],
        "column_lower": lower,
        "column_upper": upper,
        "constant_weights_required": True,
    }

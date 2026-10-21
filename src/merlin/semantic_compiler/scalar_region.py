"""Expand closed pointwise scalar SSA into typed tensor nodes.

This preserves the source operations and their rounding boundaries. It grants
neither target support nor permission to replace binary64 by a narrower type.
Math intrinsics retain an unresolved source numerical policy.
"""

from dataclasses import dataclass
from math import isnan, prod
from struct import pack

from merlin.frontends.linalg_patterns import InvalidLinalgPattern, _checked_static_linalg_shell

from .linalg_recognition import LinalgBridgeError, _linear_affine_expression, _type
from .model import ConstantBinding, IndexMap, SemanticNode, TensorType, pointwise_identity_maps
from .reference import TensorValue


@dataclass(frozen=True)
class ScalarRegion:
    nodes: tuple[SemanticNode, ...]
    constants: tuple[ConstantBinding, ...]
    values: dict[str, TensorValue]
    output: str


def translate_bucketize(op, sources, *, node_id):
    """Reuse the checked literal-boundary count, including its zero seed."""
    from merlin.frontends.bucketize_source import recognize_static_bucketize_body

    try:
        pattern = recognize_static_bucketize_body(op)
    except InvalidLinalgPattern as error:
        raise LinalgBridgeError(str(error)) from error
    if len(sources) != 2 or any(source is None for source in sources):
        raise LinalgBridgeError("boundary count reads an uninitialized source")
    rank = len(pattern.shape) + 1
    maps = []
    for mapping in op.get_indexing_maps():
        rows = tuple(_linear_affine_expression(expression, rank) for expression in mapping.data.results)
        maps.append(IndexMap(rank, tuple(row for row, _ in rows), tuple(offset for _, offset in rows)))
    return SemanticNode(
        node_id,
        "bucketize",
        sources,
        _type(op.results[0].type),
        attrs=(
            ("numeric_semantics", "f32-unordered-boundary-count-i64-v1"),
            ("right", pattern.right),
            ("boundary_count", pattern.boundary_count),
        ),
        index_maps=tuple(maps),
    )


def _tensor(shape, scalar):
    dtype = str(scalar)
    policy = {
        "i1": "boolean",
        "i64": "signed-i64",
        "f32": "raw-bit-pattern",
        "f64": "raw-bit-pattern",
        "bf16": "raw-bit-pattern",
    }.get(dtype)
    if policy is None:
        raise LinalgBridgeError(f"scalar region has no source semantics for {dtype}")
    return TensorType(shape, dtype, policy)


def _operation(op, types, result):
    """Return a typed semantic operation and its exact arithmetic attributes."""
    name = op.name
    dtype = result.dtype
    if name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf"}:
        if len(types) == 2 and dtype in {"bf16", "f32", "f64"} and all(t == result for t in types):
            arithmetic = name[6:-1]
            return f"{dtype}_{arithmetic}", (("numeric_semantics", f"ieee754-{dtype}-{arithmetic}-rne-v1"),)
    elif name == "arith.negf":
        if types == (result,) and dtype in {"f32", "f64"}:
            return f"{dtype}_neg", (("numeric_semantics", f"{dtype}-sign-bit-flip-v1"),)
    elif name in {"arith.addi", "arith.subi", "arith.muli"}:
        if name == "arith.muli" and len(types) == 2 and dtype == "i1" and all(t == result for t in types):
            return "mask", (("operation", "arith.andi"), ("predicate", None))
        if len(types) == 2 and dtype == "i64" and all(t == result for t in types):
            arithmetic = name[6:-1]
            return f"i64_{arithmetic}", (("numeric_semantics", f"signed-i64-{arithmetic}-wrap-v1"),)
    elif name in {"arith.andi", "arith.xori"}:
        if len(types) == 2 and dtype == "i1" and all(t == result for t in types):
            return "mask", (("operation", name), ("predicate", None))
    elif name in {"arith.cmpf", "arith.cmpi"}:
        from xdsl.dialects.builtin import IntegerAttr

        predicate = op.properties.get("predicate")
        predicates = (
            {1: "oeq", 2: "ogt", 3: "oge", 4: "olt", 5: "ole", 6: "one"}
            if name == "arith.cmpf"
            else {0: "eq", 1: "ne", 2: "slt", 3: "sle", 4: "sgt", 5: "sge"}
        )
        admitted = {"f32", "f64"} if name == "arith.cmpf" else {"i64"}
        if (
            len(types) == 2
            and types[0] == types[1]
            and types[0].dtype in admitted
            and dtype == "i1"
            and isinstance(predicate, IntegerAttr)
            and predicate.value.data in predicates
        ):
            return "mask", (("operation", name), ("predicate", predicates[predicate.value.data]))
    elif name == "arith.select":
        if types == (TensorType(result.shape, "i1", "boolean"), result, result):
            return "select", ()
    elif name in {"arith.sitofp", "arith.extf", "arith.truncf", "arith.fptosi"} and len(types) == 1:
        source = types[0].dtype
        if name == "arith.sitofp" and source == "i64" and dtype in {"f32", "f64"}:
            return f"signed_i64_to_{dtype}", (("numeric_semantics", f"signed-i64-to-{dtype}-rne-v1"),)
        if name == "arith.extf" and (source, dtype) in {("bf16", "f32"), ("f32", "f64")}:
            return f"{source}_to_{dtype}", (("numeric_semantics", f"{source}-to-{dtype}-exact-v1"),)
        if name == "arith.truncf" and (source, dtype) in {("f64", "f32"), ("f32", "bf16")}:
            return f"{source}_to_{dtype}", (("numeric_semantics", f"{source}-to-{dtype}-rne-v1"),)
        if name == "arith.fptosi" and source in {"f32", "f64"} and dtype == "i64":
            return f"{source}_to_signed_i64", (
                ("numeric_semantics", f"{source}-to-signed-i64-toward-zero-in-range-v1"),
            )
    elif name in {"math.powf", "math.sin", "math.cos", "math.exp", "math.rsqrt", "math.tanh"}:
        arity = 2 if name == "math.powf" else 1
        if len(types) == arity and dtype in {"f32", "f64"} and all(t == result for t in types):
            operation = name[5:]
            return f"source_{dtype}_{operation}_unqualified", (
                ("numeric_semantics", f"source-{dtype}-{operation}-unqualified-v1"),
            )
    raise LinalgBridgeError(f"no admitted scalar region semantics for {name} and its operand types")


def translate_scalar_region(op, sources, *, prefix, max_elements=1 << 20):
    """Translate one closed static pointwise region with singleton projections.

    Only input block arguments may terminate the scalar DAG. An output block
    argument, an external captured scalar or an unknown effect is rejected.
    """
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr
    from xdsl.traits import Pure

    try:
        shell = _checked_static_linalg_shell(op, singleton_projection_inputs=True)
    except InvalidLinalgPattern as error:
        raise LinalgBridgeError(str(error)) from error
    if len(op.outputs) != 1 or len(op.results) != 1 or len(sources) != len(op.inputs):
        raise LinalgBridgeError("scalar region needs one result and all declared inputs")
    if not shell.shape or any(n <= 0 for n in shell.shape):
        raise LinalgBridgeError("scalar region needs a positive static tensor shape")
    try:
        op.verify()
    except Exception as error:  # noqa: BLE001 - xDSL owns the source type/effect checks
        raise LinalgBridgeError(f"scalar region failed source verification: {error}") from error
    nodes, bindings, values = [], [], {}
    materialized_elements = 0
    env = {}
    identity = pointwise_identity_maps(len(shell.shape), 1)[-1]
    for index, (argument, source, value, mapping) in enumerate(zip(shell.args, sources, op.inputs, shell.input_maps)):
        type_ = _type(value.type)
        rows = tuple(_linear_affine_expression(expression, len(shell.shape)) for expression in mapping.results)
        logical_map = IndexMap(len(shell.shape), tuple(row for row, _ in rows), tuple(offset for _, offset in rows))
        if type_.shape != shell.shape or logical_map != identity:
            expanded = TensorType(shell.shape, type_.dtype, type_.numerical_policy)
            view = f"{prefix}_input{index}"
            nodes.append(SemanticNode(view, "gather", (source,), expanded, index_maps=(logical_map, identity)))
            source, type_ = view, expanded
        env[argument] = source, type_
    for index, scalar in enumerate(shell.body):
        if scalar.name == "linalg.yield":
            if (
                index != len(shell.body) - 1
                or len(scalar.operands) != 1
                or scalar.operands[0] not in env
                or scalar.properties
                or any(not k.startswith("prov.") for k in scalar.attributes)
            ):
                raise LinalgBridgeError("scalar region has an unknown or uninitialized yielded value")
            output, output_type = env[scalar.operands[0]]
            if output_type != _type(op.results[0].type):
                raise LinalgBridgeError("scalar region yields a different tensor type")
            return ScalarRegion(tuple(nodes), tuple(bindings), values, output)
        if (
            not type(scalar).has_trait(Pure)
            or scalar.regions
            or scalar.successors
            or len(scalar.results) != 1
            or any(not key.startswith("prov.") for key in scalar.attributes)
        ):
            raise LinalgBridgeError("scalar region contains an unknown effect or result signature")
        if scalar.name == "arith.constant":
            allowed = {"value"}
        elif scalar.name == "arith.cmpf":
            allowed = {"predicate", "fastmath"}
        elif scalar.name == "arith.cmpi":
            allowed = {"predicate"}
        elif scalar.name in {"arith.addi", "arith.subi", "arith.muli"}:
            allowed = {"overflowFlags"}
        elif scalar.name.startswith("math.") or scalar.name in {
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
            "arith.negf",
        }:
            allowed = {"fastmath"}
        else:
            allowed = set()
        if set(scalar.properties) - allowed or any(
            flag in scalar.properties and scalar.properties[flag].data for flag in ("fastmath", "overflowFlags")
        ):
            raise LinalgBridgeError("scalar region has unknown properties or relaxed arithmetic flags")
        result = _tensor(shell.shape, scalar.results[0].type)
        node_id = f"{prefix}_s{index}"
        if scalar.name == "arith.constant":
            literal = scalar.properties.get("value")
            count = prod(shell.shape)
            materialized_elements += count
            if materialized_elements > max_elements:
                from .target_binding import NativeCompilationError

                raise NativeCompilationError("resource_limit", "scalar literals exceed the materialized element budget")
            if result.dtype in {"f32", "f64"} and isinstance(literal, FloatAttr) and str(literal.type) == result.dtype:
                if isnan(literal.value.data):
                    raise LinalgBridgeError("scalar literal NaN payload cannot be recovered from its parsed float")
                payload = pack("<f" if result.dtype == "f32" else "<d", literal.value.data)
                encoding = f"{result.dtype}-le-raw"
                bits = int.from_bytes(payload, "little")
            elif result.dtype == "i64" and isinstance(literal, IntegerAttr) and str(literal.type) == "i64":
                bits, encoding = literal.value.data, "i64-le"
                payload = bits.to_bytes(8, "little", signed=True)
            elif result.dtype == "i1" and isinstance(literal, IntegerAttr) and str(literal.type) == "i1":
                bits, encoding = literal.value.data & 1, "i1-raw"
                payload = bytes((bits,))
            else:
                raise LinalgBridgeError("scalar region constant has an unsupported exact encoding")
            if scalar.operands:
                raise LinalgBridgeError("scalar constant unexpectedly reads an operand")
            node = SemanticNode(node_id, "constant", (), result, effect="constant")
            bindings.append(ConstantBinding(node_id, encoding, (payload * count).hex()))
            values[node_id] = TensorValue(result, (bits,) * count)
        else:
            if any(value not in env for value in scalar.operands):
                raise LinalgBridgeError("scalar region reads a captured or uninitialized scalar")
            inputs = tuple(env[value] for value in scalar.operands)
            operation, attrs = _operation(scalar, tuple(t for _, t in inputs), result)
            node = SemanticNode(
                node_id,
                operation,
                tuple(key for key, _ in inputs),
                result,
                attrs,
                index_maps=pointwise_identity_maps(len(shell.shape), len(inputs)),
            )
        nodes.append(node)
        env[scalar.results[0]] = node_id, result
    raise LinalgBridgeError("scalar region has no final yield")

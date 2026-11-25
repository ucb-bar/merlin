"""Translate the scalar source family, preserving numerical boundaries."""

from __future__ import annotations

from itertools import product
from math import prod

from xdsl.dialects.linalg.attrs import IteratorType

from merlin.compiler.frontend.patterns.comparisons import (
    StaticIotaResourceLimit,
    recognize_f32_nonzero,
    recognize_static_iota,
)
from merlin.compiler.frontend.patterns.linalg_composite_math import recognize_static_composite_math_body
from merlin.compiler.frontend.patterns.linalg_patterns import InvalidLinalgPattern, _checked_static_linalg_shell
from merlin.compiler.semantic.linalg.linalg_recognition import (
    LinalgBridgeError,
    _embedding_lookup_operands,
    _linear_affine_expression,
    _static_index_copy_map,
    _type,
)
from merlin.compiler.semantic.linalg.linalg_state import TranslationState
from merlin.compiler.semantic.model import ConstantBinding, IndexMap, SemanticNode, pointwise_identity_maps
from merlin.compiler.semantic.reference import TensorValue

# ---- linalg_scalar -----------------------------------------------------------------------------


def translate(state: TranslationState, op, ordinal):
    nodes = state.nodes
    env = state.env
    constants = state.constants
    constant_bindings = state.constant_bindings
    append_scalar_region = state.append_scalar_region
    if op.name == "linalg.generic" and len(op.results) == len(op.outputs) == 2:
        from merlin.compiler.frontend.patterns.comparisons import recognize_static_i64_argmin

        if len(op.inputs) != 1:
            raise LinalgBridgeError("two-result source reduction needs one admitted input")
        source_type = _type(op.inputs[0].type)
        index_bits = max(2, max(source_type.shape).bit_length() + 1)
        try:
            pattern = recognize_static_i64_argmin(op, index_bits=index_bits)
        except InvalidLinalgPattern as error:
            raise LinalgBridgeError(str(error)) from error
        source_id = env.get(op.inputs[0])
        if source_id is None:
            raise LinalgBridgeError("minimum/first-index reduction reads an uninitialized input")
        # These are pure projections of the same mathematical reduction.
        # A physical multi-result instruction uses the existing descriptor
        # bundle mechanism; separate realizations must count their work.
        for result_index, (value, operation) in enumerate(zip(op.results, ("i64_min_value", "i64_first_min_index"))):
            node = SemanticNode(
                f"op{ordinal}r{result_index}",
                operation,
                (source_id,),
                _type(value.type),
                attrs=(
                    ("axis", pattern.axis),
                    ("minimum_index_bits", index_bits),
                    ("numeric_semantics", "signed-i64-min-first-index-v1"),
                ),
            )
            nodes.append(node)
            env[value] = node.id
        return True

    if op.name == "linalg.generic" and len(op.results) == 1 and len(op.outputs) == 1:
        indexed = translate_constant_index_read(op, env, nodes, constants, node_id=f"op{ordinal}r0")
        if indexed is not None:
            nodes.append(indexed)
            env[op.results[0]] = indexed.id
            return True
        element_types = tuple(str(getattr(value.type, "element_type", None)) for value in (*op.inputs, *op.results))
        if (
            len(op.inputs) == 1
            and _type(op.inputs[0].type).dtype in {"bf16", "fp8_e4m3", "i8", "i32"}
            and len(op.regions) == 1
            and len(op.regions[0].blocks) == 1
            and tuple(item.name for item in op.regions[0].block.ops) == ("linalg.yield",)
        ):
            value, destination = op.inputs[0], op.outputs[0]
            if env.get(value) is None or env.get(destination, "undefined") is not None:
                raise LinalgBridgeError("indexed copy needs a defined source and an empty destination")
            maps = _static_index_copy_map(op, _type(value.type), _type(op.results[0].type))
            node = SemanticNode(f"op{ordinal}r0", "gather", (env[value],), _type(op.results[0].type), index_maps=maps)
            nodes.append(node)
            env[op.results[0]] = node.id
            return True
        if (
            len(op.inputs) == 1
            and element_types in {("i1", "i64"), ("i64", "i64")}
            and any(item.data == IteratorType.REDUCTION for item in op.iterator_types)
        ):
            from merlin.compiler.frontend.patterns.linalg_integer_reductions import (
                recognize_static_integer_reduction,
            )

            source_type = _type(op.inputs[0].type)
            index_bits = max(2, max(source_type.shape).bit_length() + 1)
            try:
                pattern = recognize_static_integer_reduction(op, index_bits=index_bits)
            except InvalidLinalgPattern as error:
                raise LinalgBridgeError(str(error)) from error
            source_id = env.get(op.inputs[0])
            if source_id is None:
                raise LinalgBridgeError("integer prefix sum reads an uninitialized input")
            node = SemanticNode(
                f"op{ordinal}r0",
                "i64_cumsum",
                (source_id,),
                _type(op.results[0].type),
                attrs=(
                    ("axis", pattern.axis[0]),
                    ("minimum_index_bits", index_bits),
                    ("numeric_semantics", "signed-i64-prefix-sum-wrap-v1"),
                ),
            )
            nodes.append(node)
            env[op.results[0]] = node.id
            return True
        if element_types == ("f32", "f32", "i64") and any(
            item.data == IteratorType.REDUCTION for item in op.iterator_types
        ):
            from merlin.compiler.semantic.linalg.scalar_regions import translate_bucketize

            node = translate_bucketize(op, tuple(env.get(value) for value in op.inputs), node_id=f"op{ordinal}r0")
            nodes.append(node)
            env[op.results[0]] = node.id
            return True
        if (
            ("f64" in element_types and not any(item.name == "tensor.extract" for item in op.walk()))
            or (
                element_types[-1] == "f32"
                and len(op.regions) == 1
                and len(op.regions[0].blocks) == 1
                and tuple(item.name for item in op.regions[0].block.ops)
                in {
                    ("math.sin", "linalg.yield"),
                    ("math.cos", "linalg.yield"),
                    ("math.powf", "linalg.yield"),
                }
            )
            or (len(op.inputs) == 3 and all(item.data == IteratorType.PARALLEL for item in op.iterator_types))
            or (
                op.inputs
                and all(dtype == "i64" for dtype in element_types)
                and all(item.data == IteratorType.PARALLEL for item in op.iterator_types)
            )
            or (
                len(op.inputs) == 1
                and (element_types[0], element_types[-1]) in {("bf16", "f32"), ("f32", "bf16"), ("f32", "i64")}
            )
            or (
                element_types == ("i64", "f32")
                and len(op.regions) == 1
                and len(op.regions[0].blocks) == 1
                and tuple(item.name for item in op.regions[0].block.ops) == ("arith.sitofp", "linalg.yield")
            )
            or (
                "bf16" in element_types
                and element_types[-1] == "f32"
                and all(item.data == IteratorType.PARALLEL for item in op.iterator_types)
            )
            or (
                element_types[-1] == "bf16"
                and len(op.inputs) == 2
                and all(dtype == "bf16" for dtype in element_types)
                and all(key.startswith("prov.") for key in op.attributes)
                and all(item.data == IteratorType.PARALLEL for item in op.iterator_types)
            )
            or (
                element_types
                and all(dtype == "i1" for dtype in element_types)
                and all(item.data == IteratorType.PARALLEL for item in op.iterator_types)
            )
        ):
            append_scalar_region(op, ordinal)
            return True

    if (
        op.name == "linalg.generic"
        and len(op.results) == 1
        and (
            (str(getattr(op.results[0].type, "element_type", None)) == "i1" and len(op.inputs) == 2)
            or (
                str(getattr(op.results[0].type, "element_type", None)) == "i64"
                and len(op.inputs) == 1
                and str(getattr(op.inputs[0].type, "element_type", None)) == "i1"
            )
        )
    ):
        from merlin.compiler.semantic.linalg.scalar_regions import recognize_mask

        try:
            _, maps, attrs = recognize_mask(op)
        except InvalidLinalgPattern as error:
            raise LinalgBridgeError(str(error)) from error
        source_ids = tuple(env.get(value) for value in op.inputs)
        if any(value is None for value in source_ids):
            raise LinalgBridgeError("mask reads an uninitialized input")
        node_id = f"op{ordinal}r0"
        nodes.append(SemanticNode(node_id, "mask", source_ids, _type(op.results[0].type), attrs=attrs, index_maps=maps))
        env[op.results[0]] = node_id
        return True

    if op.name == "linalg.generic" and len(op.results) == 1:
        # A signed reciprocal contains a cast as well as a division. Keep
        # both rounding boundaries instead of treating its i64 input as
        # an embedding index or assuming a fixed mask population.
        if len(op.inputs) == 1 and _type(op.inputs[0].type).dtype == "i64":
            try:
                composite = recognize_static_composite_math_body(op)
            except InvalidLinalgPattern:
                composite = None
            if composite is not None and composite.operation == "reciprocal_signed_i64_to_f32":
                source_id = env.get(op.inputs[0])
                if source_id is None or env.get(op.outputs[0], "undefined") is not None:
                    raise LinalgBridgeError("signed reciprocal needs a defined input and an empty output")
                type_ = _type(op.results[0].type)
                node_id = f"op{ordinal}r0"
                cast_id, one_id = f"{node_id}_cast", f"{node_id}_one"
                maps = pointwise_identity_maps(len(type_.shape), 1)
                nodes.extend(
                    (
                        SemanticNode(
                            cast_id,
                            "signed_i64_to_f32",
                            (source_id,),
                            type_,
                            attrs=(("numeric_semantics", "signed-i64-to-f32-rne-v1"),),
                            index_maps=maps,
                        ),
                        SemanticNode(one_id, "constant", (), type_, effect="constant"),
                        SemanticNode(
                            node_id,
                            "f32_div",
                            (one_id, cast_id),
                            type_,
                            attrs=(("numeric_semantics", "ieee754-f32-div-rne-v1"),),
                            index_maps=pointwise_identity_maps(len(type_.shape), 2),
                        ),
                    )
                )
                count = prod(type_.shape)
                constants[one_id] = TensorValue(type_, (0x3F800000,) * count)
                constant_bindings.append(ConstantBinding(one_id, "f32-le-raw", "0000803f" * count))
                env[op.results[0]] = node_id
                return True
        try:
            generated = recognize_static_iota(op) if not op.inputs else None
            truth_shape = (
                recognize_f32_nonzero(op)
                if len(op.inputs) == 1 and str(getattr(op.results[0].type, "element_type", None)) == "i1"
                else None
            )
        except InvalidLinalgPattern:
            generated, truth_shape = None, None
        except StaticIotaResourceLimit as error:
            from merlin.compiler.semantic.target_binding import NativeCompilationError

            raise NativeCompilationError("resource_limit", str(error)) from error
        if generated is not None:
            node_id = f"op{ordinal}r0"
            type_ = _type(op.results[0].type)
            nodes.append(SemanticNode(node_id, "constant", (), type_, effect="constant"))
            constants[node_id] = TensorValue(type_, generated)
            data = b"".join(value.to_bytes(8, "little", signed=True) for value in generated)
            constant_bindings.append(ConstantBinding(node_id, "i64-le", data.hex()))
            env[op.results[0]] = node_id
            return True
        if truth_shape is not None:
            source_id = env.get(op.inputs[0])
            if source_id is None:
                raise LinalgBridgeError("truth conversion reads an uninitialized tensor")
            node_id = f"op{ordinal}r0"
            nodes.append(
                SemanticNode(
                    node_id,
                    "f32_nonzero",
                    (source_id,),
                    _type(op.results[0].type),
                    attrs=(("numeric_semantics", "f32-une-zero-v1"),),
                    index_maps=pointwise_identity_maps(len(truth_shape), 1),
                )
            )
            env[op.results[0]] = node_id
            return True

    return False


# ---- constant_indexing -------------------------------------------------------------------------
# Recover affine reads from declared constant indices, without sampling data.
#
# The complete finite output domain is checked. Runtime indices and non-affine
# reads remain unsupported; this does not grant a target gather instruction.


def _flat(indices, shape):
    if len(indices) != len(shape) or any(not 0 <= i < n for i, n in zip(indices, shape)):
        raise LinalgBridgeError("constant index reads outside its declared tensor bounds")
    offset = 0
    for index, extent in zip(indices, shape):
        offset = offset * extent + index
    return offset


def _constant_indices(roots, nodes, constants, budget):
    """Evaluate only constant signed-i64 movement and modular arithmetic."""
    by_id = {node.id: node for node in nodes}
    pending, needed = list(roots), set()
    while pending:
        key = pending.pop()
        if key in needed:
            continue
        node = by_id[key]
        needed.add(key)
        budget -= prod(node.type.shape)
        if budget < 0:
            from merlin.compiler.semantic.target_binding import NativeCompilationError

            raise NativeCompilationError("resource_limit", "constant indexing exceeds its element work budget")
        if node.type.dtype != "i64" or node.type.numerical_policy != "signed-i64":
            raise LinalgBridgeError("constant indexing requires signed-i64 index tensors")
        if node.op not in {"constant", "identity", "reshape", "gather", "concat", "i64_add", "i64_sub", "i64_mul"}:
            raise LinalgBridgeError("index tensor is not a supported declared compile-time constant")
        pending.extend(node.inputs)
    values = {}
    for node in nodes:
        if node.id not in needed:
            continue
        operands = [values[key] for key in node.inputs]
        if node.op == "constant":
            value = constants.get(node.id)
            if value is None or value.type != node.type:
                raise LinalgBridgeError("index constant has no matching declared bytes")
            result = value.elements
        elif node.op in {"identity", "reshape"}:
            result = operands[0]
        elif node.op == "gather":
            source = by_id[node.inputs[0]]
            result = tuple(
                operands[0][_flat(node.index_maps[0].apply(point), source.type.shape)]
                for point in product(*(range(n) for n in node.type.shape))
            )
        elif node.op == "concat":
            axis = dict(node.attrs)["axis"]
            result = []
            for point in product(*(range(n) for n in node.type.shape)):
                at = list(point)
                for key, operand in zip(node.inputs, operands):
                    shape = by_id[key].type.shape
                    if at[axis] < shape[axis]:
                        result.append(operand[_flat(at, shape)])
                        break
                    at[axis] -= shape[axis]
            result = tuple(result)
        else:
            operation = node.op.removeprefix("i64_")
            if dict(node.attrs) != {"numeric_semantics": f"signed-i64-{operation}-wrap-v1"}:
                raise LinalgBridgeError("constant index arithmetic has a different rounding policy")
            result = []
            for a, b in zip(*operands, strict=True):
                value = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b}[operation]()
                result.append((value + (1 << 63)) % (1 << 64) - (1 << 63))
            result = tuple(result)
        if len(result) != prod(node.type.shape):
            raise LinalgBridgeError("constant index element count differs from its declared shape")
        values[node.id] = result
    return {key: values[key] for key in roots}, by_id


def translate_constant_index_read(op, env, nodes, constants, *, node_id, max_work=1 << 20):
    """Recognize loop indices and constant index casts feeding a captured read.

    Return None for other scalar bodies. A recognized read must satisfy every
    check; it cannot fall through to a matcher with weaker index requirements.
    """
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return None
    body = tuple(op.regions[0].block.ops)
    if not any(item.name == "tensor.extract" for item in body):
        return None
    if not op.inputs:
        return None
    # Preserve the existing runtime embedding path only after its complete
    # semantic proof succeeds; a similar-looking body gets no exemption.
    if any(item.name == "linalg.index" for item in body):
        try:
            _embedding_lookup_operands(op)
        except LinalgBridgeError:
            pass
        else:
            return None
    if (
        len(body) < 3
        or tuple(item.name for item in body[-2:]) != ("tensor.extract", "linalg.yield")
        or any(item.name not in {"linalg.index", "arith.index_cast"} for item in body[:-2])
    ):
        return None
    try:
        shell = _checked_static_linalg_shell(op, singleton_projection_inputs=True)
        op.verify()
    except Exception as error:  # noqa: BLE001 - fail closed at the parsed source boundary
        raise LinalgBridgeError(f"constant indexed read has an invalid Linalg shell: {error}") from error
    rank = len(shell.shape)
    if prod(shell.shape) * max(1, len(body) - 2) > max_work:
        from merlin.compiler.semantic.target_binding import NativeCompilationError

        raise NativeCompilationError("resource_limit", "constant indexing exceeds its finite-domain work budget")
    if (
        any(
            set(item.properties) != ({"dim"} if item.name == "linalg.index" else set())
            or any(not k.startswith("prov.") for k in item.attributes)
            for item in body
        )
        or any(item.regions or item.successors for item in body)
        or any(str(arg.type) != "i64" for arg in shell.args[:-1])
        or env.get(op.outputs[0], "undefined") is not None
    ):
        raise LinalgBridgeError(
            "constant indexed read needs only i64 casts, loop indices, a tensor read, and an empty output"
        )
    indices, read, yielded = body[:-2], body[-2], body[-1]
    casts = tuple(item for item in indices if item.name == "arith.index_cast")
    loops = tuple(item for item in indices if item.name == "linalg.index")
    from xdsl.dialects.builtin import IntegerAttr

    if (
        len(casts) != len(op.inputs)
        or any(
            tuple(cast.operands) != (arg,) or len(cast.results) != 1 or str(cast.results[0].type) != "index"
            for cast, arg in zip(casts, shell.args[:-1])
        )
        or any(
            loop.operands
            or len(loop.results) != 1
            or str(loop.results[0].type) != "index"
            or not isinstance(dim := loop.properties.get("dim"), IntegerAttr)
            or not 0 <= dim.value.data < rank
            for loop in loops
        )
        or len(read.results) != 1
        or tuple(read.operands[1:]) != tuple(index.results[0] for index in indices)
        or tuple(yielded.operands) != (read.results[0],)
        or read.operands[0] not in env
        or env[read.operands[0]] is None
    ):
        raise LinalgBridgeError("constant indexed read has an unknown table, index, or yielded value")
    table_type, result_type = _type(read.operands[0].type), _type(op.results[0].type)
    if len(table_type.shape) != len(indices) or (table_type.dtype, table_type.numerical_policy) != (
        result_type.dtype,
        result_type.numerical_policy,
    ):
        raise LinalgBridgeError("constant indexed read has incompatible table dimensions or element type")
    roots = tuple(env.get(value) for value in op.inputs)
    if any(key is None for key in roots):
        raise LinalgBridgeError("constant indexed read uses an uninitialized index tensor")
    values, by_id = _constant_indices(roots, nodes, constants, max_work)
    maps = []
    for mapping in shell.input_maps:
        rows = tuple(_linear_affine_expression(expr, rank) for expr in mapping.results)
        maps.append(IndexMap(rank, tuple(row for row, _ in rows), tuple(offset for _, offset in rows)))

    def coordinate(point):
        cast_values = {
            cast.results[0]: values[key][_flat(mapping.apply(point), by_id[key].type.shape)]
            for cast, key, mapping in zip(casts, roots, maps, strict=True)
        }
        cast_values.update((loop.results[0], point[loop.properties["dim"].value.data]) for loop in loops)
        return tuple(cast_values[index.results[0]] for index in indices)

    origin = coordinate((0,) * rank)
    differences = []
    for axis, extent in enumerate(shell.shape):
        unit = tuple(int(dim == axis) for dim in range(rank))
        differences.append(tuple(b - a for a, b in zip(origin, coordinate(unit))) if extent > 1 else (0,) * len(origin))
    source_map = IndexMap(rank, tuple(tuple(delta[dim] for delta in differences) for dim in range(len(origin))), origin)
    for point in product(*(range(n) for n in shell.shape)):
        actual = coordinate(point)
        _flat(actual, table_type.shape)
        if actual != source_map.apply(point):
            raise LinalgBridgeError("declared constant indices do not form an affine tensor read")
    return SemanticNode(
        node_id,
        "gather",
        (env[read.operands[0]],),
        result_type,
        index_maps=(source_map, pointwise_identity_maps(rank, 1)[-1]),
    )

"""Recognize typed Linalg bodies and affine index maps without selecting hardware."""

from __future__ import annotations

from struct import pack, unpack

from .model import IndexMap, TensorType, pointwise_identity_maps


class LinalgBridgeError(ValueError):
    pass


def _maps() -> tuple[IndexMap, ...]:
    return (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 0, 1), (0, 1, 0)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )


def _transpose_maps(permutation: tuple[int, ...]) -> tuple[IndexMap, ...]:
    rank = len(permutation)
    inverse = tuple(permutation.index(axis) for axis in range(rank))
    source = IndexMap(
        rank,
        tuple(tuple(int(loop == inverse[axis]) for loop in range(rank)) for axis in range(rank)),
        (0,) * rank,
    )
    output = IndexMap(
        rank,
        tuple(tuple(int(loop == axis) for loop in range(rank)) for axis in range(rank)),
        (0,) * rank,
    )
    return source, output


def _oriented_matmul_maps() -> tuple[IndexMap, ...]:
    return (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 1, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )


def _oriented_matmul_accumulate_maps() -> tuple[IndexMap, ...]:
    maps = _oriented_matmul_maps()
    return (maps[0], maps[1], maps[2], maps[2])


def _source_f32_batched_matmul(op) -> tuple[object, object, object, tuple[IndexMap, ...], str]:
    """Recognize a batched contraction while retaining its unresolved f32 order."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import DenseArrayBase
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp, YieldOp
    from xdsl.ir.affine import AffineDimExpr

    if (
        type(op) is not GenericOp
        or len(op.inputs) != 2
        or len(op.outputs) != 1
        or len(op.results) != 1
        or op.successors
        or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}
        or any(not key.startswith("prov.") for key in op.attributes)
        or len(op.regions) != 1
        or len(op.regions[0].blocks) != 1
    ):
        raise LinalgBridgeError("source f32 batched matmul needs one closed Linalg region")
    segments = op.properties["operandSegmentSizes"]
    if (
        not isinstance(segments, DenseArrayBase)
        or str(segments.elt_type) != "i32"
        or tuple(segments.iter_values()) != (2, 1)
    ):
        raise LinalgBridgeError("source f32 batched matmul has invalid operand segments")
    lhs, rhs, init = (*op.inputs, op.outputs[0])
    lhs_type, rhs_type, init_type, result_type = (_type(value.type) for value in (lhs, rhs, init, op.results[0]))
    result_rank = len(result_type.shape)
    if (
        result_rank < 3
        or any(
            value.dtype != "f32" or value.numerical_policy != "raw-bit-pattern"
            for value in (lhs_type, rhs_type, init_type, result_type)
        )
        or any(len(value.shape) != result_rank for value in (lhs_type, rhs_type, init_type))
        or init_type != result_type
        or lhs_type.shape[:-2] != result_type.shape[:-2]
        or rhs_type.shape[:-2] != result_type.shape[:-2]
        or lhs_type.shape[-2] != result_type.shape[-2]
        or rhs_type.shape[-1] != result_type.shape[-1]
        or lhs_type.shape[-1] != rhs_type.shape[-2]
    ):
        raise LinalgBridgeError("source f32 batched matmul has incompatible shapes or types")
    iterators = op.iterator_types.data
    if (
        len(iterators) != result_rank + 1
        or any(not isinstance(iterator, IteratorTypeAttr) for iterator in iterators)
        or tuple(iterator.data for iterator in iterators)
        != (IteratorType.PARALLEL,) * result_rank + (IteratorType.REDUCTION,)
    ):
        raise LinalgBridgeError("source f32 batched matmul has invalid reduction iterators")
    batch_axes = tuple(range(result_rank - 2))
    m_axis, n_axis, k_axis = result_rank - 2, result_rank - 1, result_rank
    expected_axes = (
        (*batch_axes, m_axis, k_axis),
        (*batch_axes, k_axis, n_axis),
        (*batch_axes, m_axis, n_axis),
    )
    maps = tuple(attribute.data for attribute in op.get_indexing_maps())
    if len(maps) != 3 or any(
        mapping.num_dims != result_rank + 1
        or mapping.num_symbols != 0
        or len(mapping.results) != len(axes)
        or any(
            type(expression) is not AffineDimExpr or expression.position != axis
            for expression, axis in zip(mapping.results, axes, strict=True)
        )
        for mapping, axes in zip(maps, expected_axes, strict=True)
    ):
        raise LinalgBridgeError("source f32 batched matmul has nonstandard index maps")
    block = op.regions[0].block
    body = tuple(block.ops)
    if (
        tuple(str(argument.type) for argument in block.args) != ("f32",) * 3
        or len(body) != 3
        or tuple(nested.name for nested in body) != ("arith.mulf", "arith.addf", "linalg.yield")
        or type(body[2]) is not YieldOp
        or any(
            nested.regions or nested.successors or any(not key.startswith("prov.") for key in nested.attributes)
            for nested in body
        )
        or any(set(arithmetic.properties) - {"fastmath"} for arithmetic in body[:2])
        or any(
            (flags := arithmetic.properties.get("fastmath")) is not None
            and (not isinstance(flags, arith.FastMathFlagsAttr) or flags.data)
            for arithmetic in body[:2]
        )
        or body[2].properties
        or any(len(arithmetic.results) != 1 or str(arithmetic.results[0].type) != "f32" for arithmetic in body[:2])
        or tuple(body[0].operands) != tuple(block.args[:2])
        or tuple(body[2].operands) != (body[1].results[0],)
    ):
        raise LinalgBridgeError("source f32 batched matmul has an unqualified scalar body")
    add_operands = tuple(body[1].operands)
    if add_operands == (body[0].results[0], block.args[2]):
        add_order = "product-accumulator"
    elif add_operands == (block.args[2], body[0].results[0]):
        add_order = "accumulator-product"
    else:
        raise LinalgBridgeError("source f32 batched matmul adds the wrong scalar operands")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - xDSL owns source verification
        raise LinalgBridgeError(f"source f32 batched matmul failed verification: {exc}") from exc
    index_maps = tuple(
        IndexMap(
            result_rank + 1,
            tuple(tuple(int(loop == axis) for loop in range(result_rank + 1)) for axis in axes),
            (0,) * len(axes),
        )
        for axes in (*expected_axes, expected_axes[-1])
    )
    return lhs, rhs, init, index_maps, add_order


def _f32_reduce_maps(rank: int, axes: tuple[int, ...]) -> tuple[IndexMap, ...]:
    source = IndexMap(
        rank,
        tuple(tuple(int(loop == dim) for loop in range(rank)) for dim in range(rank)),
        (0,) * rank,
    )
    projection = IndexMap(
        rank,
        tuple(tuple(int(loop == dim) for loop in range(rank)) for dim in range(rank) if dim not in axes),
        (0,) * (rank - len(axes)),
    )
    return source, projection, projection


def _recognize_f32_reduce(op) -> tuple[object, object, tuple[int, ...], str, str, str]:
    """Check one f32 add or maximum reduction without inventing its order.

    The attribute is a source promise. Parsing cannot prove that a prior
    producer or an external model intended this execution order.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import DenseArrayBase, StringAttr
    from xdsl.dialects.linalg.ops import ReduceOp, YieldOp

    if (
        type(op) is not ReduceOp
        or set(op.properties) != {"dimensions"}
        or any(key != "merlin.reduction_order" and not key.startswith("prov.") for key in op.attributes)
        or op.successors
        or len(op.operands) != 2
        or tuple(op.operands) != (op.input, op.init)
        or len(op.results) != 1
        or len(op.regions) != 1
        or len(op.regions[0].blocks) != 1
    ):
        raise LinalgBridgeError("ordered f32 reduction needs a closed registered linalg.reduce")
    order = op.attributes.get("merlin.reduction_order")
    if order is not None and (not isinstance(order, StringAttr) or order.data != "axis-ascending"):
        raise LinalgBridgeError("f32 linalg.reduce has an unqualified numerical-order declaration")
    source_type, init_type, result_type = (_type(value.type) for value in (op.input, op.init, op.results[0]))
    dimensions = op.properties["dimensions"]
    if not isinstance(dimensions, DenseArrayBase) or str(dimensions.elt_type) != "i64":
        raise LinalgBridgeError("ordered f32 reduction needs a typed i64 axis")
    axes = tuple(dimensions.iter_values())
    rank = len(source_type.shape)
    if (
        not axes
        or len(axes) >= rank
        or len(set(axes)) != len(axes)
        or any(type(axis) is not int or not 0 <= axis < rank for axis in axes)
        or order is not None
        and len(axes) != 1
        or rank < 2
        or source_type.dtype != "f32"
        or source_type.numerical_policy != "raw-bit-pattern"
        or init_type != result_type
        or result_type
        != TensorType(
            tuple(extent for dim, extent in enumerate(source_type.shape) if dim not in axes),
            "f32",
            "raw-bit-pattern",
        )
    ):
        raise LinalgBridgeError("ordered f32 reduction has an unsupported axis, shape, or type")
    block = op.regions[0].block
    body = tuple(block.ops)
    if len(block.args) != 2 or len(body) != 2:
        raise LinalgBridgeError("ordered f32 reduction needs one add and one yield")
    add, yld = body
    flags = add.properties.get("fastmath")
    if (
        add.name not in {"arith.addf", "arith.maximumf"}
        or type(yld) is not YieldOp
        or any(
            nested.regions or nested.successors or any(not key.startswith("prov.") for key in nested.attributes)
            for nested in body
        )
        or set(add.properties) - {"fastmath"}
        or yld.properties
        or flags is not None
        and (not isinstance(flags, arith.FastMathFlagsAttr) or flags.data)
        or tuple(str(arg.type) for arg in block.args) != ("f32", "f32")
        or tuple(add.operands) not in (tuple(block.args), tuple(reversed(block.args)))
        or len(add.results) != 1
        or str(add.results[0].type) != "f32"
        or tuple(yld.operands) != (add.results[0],)
    ):
        raise LinalgBridgeError("f32 reduction scalar body or fast-math flags are unqualified")
    add_order = "input-accumulator" if tuple(add.operands) == tuple(block.args) else "accumulator-input"
    if order is not None and (add.name != "arith.addf" or add_order != "input-accumulator"):
        raise LinalgBridgeError("ordered f32 reduction changes its declared scalar add order or operation")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - xDSL owns source verification
        raise LinalgBridgeError(f"ordered f32 reduction failed verification: {exc}") from exc
    return op.input, op.init, axes, "unresolved" if order is None else order.data, add_order, add.name


def _linear_affine_expression(expression, rank: int) -> tuple[tuple[int, ...], int]:
    """Accept only integer-affine dimension/add/constant-multiply expressions."""
    from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr

    if isinstance(expression, AffineDimExpr) and 0 <= expression.position < rank:
        return tuple(int(axis == expression.position) for axis in range(rank)), 0
    if isinstance(expression, AffineConstantExpr):
        return (0,) * rank, expression.value
    if isinstance(expression, AffineBinaryOpExpr):
        left, left_offset = _linear_affine_expression(expression.lhs, rank)
        right, right_offset = _linear_affine_expression(expression.rhs, rank)
        if expression.kind is AffineBinaryOpKind.Add:
            return tuple(a + b for a, b in zip(left, right)), left_offset + right_offset
        if expression.kind is AffineBinaryOpKind.Mul:
            if not any(left):
                return tuple(left_offset * value for value in right), left_offset * right_offset
            if not any(right):
                return tuple(right_offset * value for value in left), right_offset * left_offset
    raise LinalgBridgeError("indexed copy has a nonlinear or unsupported affine expression")


def _static_index_copy_map(op, source_type: TensorType, result_type: TensorType) -> tuple[IndexMap, ...]:
    """Check a one-input all-parallel Linalg region yielding its input scalar."""
    from xdsl.dialects.linalg.attrs import IteratorType

    if (
        len(op.inputs) != 1
        or len(op.outputs) != 1
        or len(op.results) != 1
        or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}
        or tuple(op.properties["operandSegmentSizes"].get_values()) != (1, 1)
        or any(not key.startswith("prov.") for key in op.attributes)
        or len(op.body.blocks) != 1
    ):
        raise LinalgBridgeError("indexed copy has an unsupported Linalg signature")
    body = op.body.block
    operations = tuple(body.ops)
    if (
        len(body.args) != 2
        or len(operations) != 1
        or operations[0].name != "linalg.yield"
        or len(operations[0].operands) != 1
        or operations[0].operands[0] is not body.args[0]
        or operations[0].properties
        or any(not key.startswith("prov.") for key in operations[0].attributes)
        or body.args[0].type != op.inputs[0].type.element_type
        or body.args[1].type != op.results[0].type.element_type
        or (source_type.dtype, source_type.numerical_policy) != (result_type.dtype, result_type.numerical_policy)
        or op.results[0].type != op.outputs[0].type
    ):
        raise LinalgBridgeError("indexed copy body must yield the unchanged input scalar")
    rank = len(result_type.shape)
    maps = tuple(attribute.data for attribute in op.get_indexing_maps())
    if (
        len(maps) != 2
        or len(op.get_iterator_types()) != rank
        or any(attribute.data is not IteratorType.PARALLEL for attribute in op.get_iterator_types())
        or any(mapping.num_dims != rank or mapping.num_symbols != 0 for mapping in maps)
        or len(maps[0].results) != len(source_type.shape)
        or len(maps[1].results) != rank
    ):
        raise LinalgBridgeError("indexed copy needs all-parallel loops and static affine maps")
    converted = []
    for mapping in maps:
        rows = tuple(_linear_affine_expression(expression, rank) for expression in mapping.results)
        converted.append(IndexMap(rank, tuple(row for row, _ in rows), tuple(offset for _, offset in rows)))
    identity = IndexMap(
        rank,
        tuple(tuple(int(axis == dim) for axis in range(rank)) for dim in range(rank)),
        (0,) * rank,
    )
    if converted[1] != identity:
        raise LinalgBridgeError("indexed copy destination map must cover every output element once")
    for row, offset, source_extent in zip(
        converted[0].coefficients,
        converted[0].offsets,
        source_type.shape,
    ):
        minimum = offset + sum(
            min(0, coefficient * (extent - 1)) for coefficient, extent in zip(row, result_type.shape)
        )
        maximum = offset + sum(
            max(0, coefficient * (extent - 1)) for coefficient, extent in zip(row, result_type.shape)
        )
        if minimum < 0 or maximum >= source_extent:
            raise LinalgBridgeError("indexed copy input map exceeds source bounds")
    try:
        op.verify()
    except Exception as error:  # noqa: BLE001 - retain the parsed operation's type and region checks
        raise LinalgBridgeError(f"indexed copy failed source verification: {error}") from error
    return tuple(converted)


def _static_f32_square_maps(op, source_type: TensorType, result_type: TensorType) -> tuple[IndexMap, ...]:
    """Recognize one strict f32 multiplication of each source element by itself."""
    from xdsl.dialects import arith
    from xdsl.dialects.linalg.attrs import IteratorType

    if (
        len(op.inputs) != 1
        or len(op.outputs) != 1
        or len(op.results) != 1
        or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}
        or tuple(op.properties["operandSegmentSizes"].get_values()) != (1, 1)
        or any(not key.startswith("prov.") for key in op.attributes)
        or len(op.body.blocks) != 1
        or source_type != result_type
        or source_type.dtype != "f32"
        or source_type.numerical_policy != "raw-bit-pattern"
    ):
        raise LinalgBridgeError("f32 square has an unsupported type or Linalg signature")
    body = op.body.block
    operations = tuple(body.ops)
    if len(body.args) != 2 or len(operations) != 2:
        raise LinalgBridgeError("f32 square needs one multiplication and one yield")
    multiply, yield_op = operations
    flags = multiply.properties.get("fastmath")
    if (
        (multiply.name, yield_op.name) != ("arith.mulf", "linalg.yield")
        or any(
            nested.regions or nested.successors or any(not key.startswith("prov.") for key in nested.attributes)
            for nested in operations
        )
        or set(multiply.properties) - {"fastmath"}
        or yield_op.properties
        or flags is not None
        and (not isinstance(flags, arith.FastMathFlagsAttr) or flags.data)
        or tuple(multiply.operands) != (body.args[0], body.args[0])
        or len(multiply.results) != 1
        or str(multiply.results[0].type) != "f32"
        or tuple(yield_op.operands) != (multiply.results[0],)
        or any(str(arg.type) != "f32" for arg in body.args)
    ):
        raise LinalgBridgeError("f32 square must multiply its input by itself without fast math")
    rank = len(result_type.shape)
    maps = tuple(attribute.data for attribute in op.get_indexing_maps())
    if (
        len(maps) != 2
        or len(op.get_iterator_types()) != rank
        or any(attribute.data is not IteratorType.PARALLEL for attribute in op.get_iterator_types())
        or any(mapping.num_dims != rank or mapping.num_symbols != 0 for mapping in maps)
    ):
        raise LinalgBridgeError("f32 square needs all-parallel static indexing")
    converted = tuple(
        IndexMap(rank, tuple(row for row, _ in rows), tuple(offset for _, offset in rows))
        for rows in (
            tuple(_linear_affine_expression(expression, rank) for expression in mapping.results) for mapping in maps
        )
    )
    if converted != pointwise_identity_maps(rank, 1):
        raise LinalgBridgeError("f32 square requires identical source and output index maps")
    return pointwise_identity_maps(rank, 2)


def _embedding_lookup_operands(op) -> tuple[object, object, object]:
    """Recognize a checked tensor-table read indexed by one signed i64 tensor."""
    from xdsl.dialects.builtin import NoneAttr
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if (
        len(op.inputs) != 1
        or len(op.outputs) != 1
        or len(op.results) != 1
        or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}
        or tuple(op.properties["operandSegmentSizes"].get_values()) != (1, 1)
        or any(not key.startswith("prov.") for key in op.attributes)
        or len(op.body.blocks) != 1
    ):
        raise LinalgBridgeError("embedding lookup has an unsupported Linalg signature")
    indices, destination = op.inputs[0], op.outputs[0]
    body = op.body.block
    body_operations = tuple(body.ops)
    if len(body_operations) != 4:
        raise LinalgBridgeError("embedding lookup body must read one table row and one output channel")
    cast, channel, extract, yield_op = body_operations
    if (
        len(body.args) != 2
        or tuple(arg.name for arg in (cast, channel, extract, yield_op))
        != ("arith.index_cast", "linalg.index", "tensor.extract", "linalg.yield")
        or any(
            nested.regions or nested.successors or any(not key.startswith("prov.") for key in nested.attributes)
            for nested in (cast, channel, extract, yield_op)
        )
        or cast.properties
        or extract.properties
        or yield_op.properties
        or set(channel.properties) != {"dim"}
        or str(body.args[0].type) != "i64"
        or body.args[1].type != op.outputs[0].type.element_type
        or tuple(cast.operands) != (body.args[0],)
        or len(cast.results) != 1
        or str(cast.results[0].type) != "index"
        or channel.operands
        or len(channel.results) != 1
        or str(channel.results[0].type) != "index"
        or len(extract.operands) != 3
        or len(extract.results) != 1
        or tuple(extract.operands[1:]) != (cast.results[0], channel.results[0])
        or extract.results[0].type != body.args[1].type
        or tuple(yield_op.operands) != (extract.results[0],)
    ):
        raise LinalgBridgeError("embedding lookup body must read one table row and one output channel")
    table = extract.operands[0]
    indices_type, table_type, destination_type, result_type = (
        _type(value.type) for value in (indices, table, destination, op.results[0])
    )
    output_rank = len(result_type.shape)
    if (
        indices_type.dtype != "i64"
        or indices_type.numerical_policy != "signed-i64"
        or table_type.dtype not in {"f32", "f64", "bf16", "fp8_e4m3"}
        or table_type.numerical_policy != "raw-bit-pattern"
        or len(table_type.shape) != 2
        or result_type != destination_type
        or result_type != TensorType((*indices_type.shape, table_type.shape[1]), table_type.dtype, "raw-bit-pattern")
        or any(not isinstance(value.type.encoding, NoneAttr) for value in (indices, table, destination, op.results[0]))
        or getattr(getattr(channel.properties["dim"], "value", None), "data", None) != output_rank - 1
        or len(op.get_iterator_types()) != output_rank
        or any(item.data is not IteratorType.PARALLEL for item in op.get_iterator_types())
    ):
        raise LinalgBridgeError("embedding lookup has an unsupported shape, type, or channel index")
    dimensions = tuple(AffineDimExpr(dim) for dim in range(output_rank))
    expected = (
        AffineMap(output_rank, 0, dimensions[:-1]),
        AffineMap(output_rank, 0, dimensions),
    )
    if tuple(item.data for item in op.get_indexing_maps()) != expected:
        raise LinalgBridgeError("embedding lookup maps differ from the output prefix and channel")
    try:
        op.verify()
    except Exception as error:  # noqa: BLE001 - source tensor types and scalar SSA must verify
        raise LinalgBridgeError(f"embedding lookup failed source verification: {error}") from error
    return indices, table, destination


def _type(attribute) -> TensorType:
    from xdsl.dialects.builtin import NoneAttr
    from xdsl.dialects.builtin import TensorType as XdslTensorType

    if not isinstance(attribute, XdslTensorType):
        raise LinalgBridgeError("only ranked tensor values are admitted")
    if not isinstance(attribute.encoding, NoneAttr):
        raise LinalgBridgeError("source tensor encoding needs an explicit logical-layout translation")
    shape = tuple(int(dimension) for dimension in attribute.get_shape())
    dtype = str(attribute.element_type)
    if dtype == "i1":
        policy = "boolean"
    elif dtype == "i8":
        policy = "signed-i8"
    elif dtype == "i32":
        policy = "i32-wrap-k-ascending"
    elif dtype == "i64":
        policy = "signed-i64"
    elif dtype == "bf16":
        policy = "raw-bit-pattern"
    elif dtype in {"f32", "f64"}:
        # Source arithmetic needs a separate declared precision policy.
        # Its boundary values and bit-preserving moves still have exact types.
        policy = "raw-bit-pattern"
    elif dtype == "f8E4M3FN":
        dtype = "fp8_e4m3"
        policy = "raw-bit-pattern"
    else:
        raise LinalgBridgeError(f"no admitted numerical policy for {dtype}")
    try:
        return TensorType(shape, dtype, policy)
    except ValueError as exc:
        raise LinalgBridgeError(f"unadmitted tensor type {attribute}: {exc}") from exc


def _scalar_integer(op) -> int:
    attr = op.properties.get("value", op.attributes.get("value"))
    value = getattr(getattr(attr, "value", None), "data", None)
    if type(value) is not int or str(op.results[0].type) != "i32":
        raise LinalgBridgeError("fill scalar must be an explicit i32 integer constant")
    if not -(1 << 31) <= value < (1 << 31):
        raise LinalgBridgeError("fill constant is outside signed i32")
    return value


def _scalar_i64(op) -> int:
    attr = op.properties.get("value", op.attributes.get("value"))
    value = getattr(getattr(attr, "value", None), "data", None)
    if type(value) is not int or str(op.results[0].type) != "i64":
        raise LinalgBridgeError("fill scalar must be an explicit i64 integer constant")
    if not -(1 << 63) <= value < (1 << 63):
        raise LinalgBridgeError("fill constant is outside signed i64")
    return value


def _i64_to_f32_bits(value: int) -> int:
    """Round a signed i64 exactly to binary32, ties to even."""
    if not -(1 << 63) <= value < (1 << 63):
        raise LinalgBridgeError("signed integer cast is outside i64")
    magnitude = abs(value)
    if magnitude == 0:
        return 0
    exponent = magnitude.bit_length() - 1
    shift = max(0, exponent - 23)
    significand = magnitude >> shift
    if shift:
        remainder = magnitude - (significand << shift)
        halfway = 1 << (shift - 1)
        if remainder > halfway or remainder == halfway and significand & 1:
            significand += 1
    else:
        significand <<= 23 - exponent
    if significand == 1 << 24:
        significand >>= 1
        exponent += 1
    return (int(value < 0) << 31) | ((exponent + 127) << 23) | (significand & ((1 << 23) - 1))


def _scalar_f32_bits(op) -> int:
    from math import isnan

    from xdsl.dialects.builtin import FloatAttr

    attr = op.properties.get("value", op.attributes.get("value"))
    if str(op.results[0].type) != "f32" or not isinstance(attr, FloatAttr):
        raise LinalgBridgeError("fill scalar must be an explicit f32 constant")
    value = float(attr.value.data)
    if isnan(value):
        raise LinalgBridgeError("f32 NaN scalar payload cannot be preserved by this bridge")
    try:
        return unpack("<I", pack("<f", value))[0]
    except OverflowError as exc:
        raise LinalgBridgeError("f32 fill scalar is outside its storage range") from exc


def _named_matmul_maps_are_default(op) -> bool:
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    d0, d1, d2 = (AffineDimExpr(i) for i in range(3))
    expected = (
        AffineMap(3, 0, (d0, d2)),
        AffineMap(3, 0, (d2, d1)),
        AffineMap(3, 0, (d0, d1)),
    )
    return tuple(attr.data for attr in op.get_indexing_maps()) == expected and tuple(
        attr.data for attr in op.get_iterator_types()
    ) == (IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.REDUCTION)

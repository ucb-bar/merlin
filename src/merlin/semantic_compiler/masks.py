"""Typed comparison masks with checked singleton projections."""

from itertools import product
from struct import unpack

from .model import IndexMap, SemanticNode, TensorType


def recognize_mask(op):
    """Use the shared closed Linalg recognizer; return no target placement decision."""
    from merlin.frontends.linalg_patterns import InvalidLinalgPattern, recognize_static_projected_pointwise

    from .linalg_recognition import _linear_affine_expression

    pattern = recognize_static_projected_pointwise(op)
    if pattern.operation not in {"arith.cmpi", "arith.cmpf", "arith.andi", "arith.xori", "arith.extui"}:
        raise InvalidLinalgPattern("mask operation is not a checked comparison, Boolean operation or widening cast")
    rank = len(pattern.shape)
    maps = tuple(
        IndexMap(rank, tuple(row for row, _ in rows), tuple(offset for _, offset in rows))
        for rows in (
            tuple(_linear_affine_expression(expression, rank) for expression in mapping.data.results)
            for mapping in op.get_indexing_maps()
        )
    )
    attrs = (("operation", pattern.operation), ("predicate", pattern.predicate))
    return pattern.shape, maps, attrs


def evaluate_mask(node: SemanticNode, operands):
    """Recheck maps and types without consulting the Linalg source recognizer."""
    from .reference import TensorValue

    attrs = dict(node.attrs)
    if set(attrs) != {"operation", "predicate"} or len(operands) not in (1, 2):
        raise ValueError("mask has an unknown operation, type or attribute")
    operation, predicate = attrs["operation"], attrs["predicate"]
    if operation == "arith.extui":
        expected, predicates = ("i1", "boolean"), {None}
        output_type, arity = TensorType(node.type.shape, "i64", "signed-i64"), 1
    elif operation == "arith.cmpi":
        expected, predicates = ("i64", "signed-i64"), {"eq", "ne", "slt", "sle", "sgt", "sge"}
    elif operation == "arith.cmpf":
        dtype = operands[0].type.dtype
        if dtype not in {"f32", "f64"}:
            raise ValueError("floating mask needs a declared IEEE source type")
        expected, predicates = (dtype, "raw-bit-pattern"), {"oeq", "one", "olt", "ole", "ogt", "oge"}
    elif operation in {"arith.andi", "arith.xori"}:
        expected, predicates = ("i1", "boolean"), {None}
    else:
        raise ValueError("mask operation has no exact reference")
    if operation != "arith.extui":
        output_type, arity = TensorType(node.type.shape, "i1", "boolean"), 2
    if node.type != output_type or len(operands) != arity:
        raise ValueError("mask output type or operation arity differs")
    if predicate not in predicates or any(
        (value.type.dtype, value.type.numerical_policy) != expected for value in operands
    ):
        raise ValueError("mask predicate or operand numerical policy differs")
    rank = len(node.type.shape)
    if len(node.index_maps) != arity + 1:
        raise ValueError("mask needs one map per input and one output map")
    identity = IndexMap(rank, tuple(tuple(int(i == j) for j in range(rank)) for i in range(rank)), (0,) * rank)
    if node.index_maps[-1] != identity:
        raise ValueError("mask output map is not identity")
    for mapping, operand in zip(node.index_maps[:-1], operands, strict=True):
        shape = operand.type.shape
        if mapping.loop_rank != rank or len(mapping.coefficients) != len(shape):
            raise ValueError("mask input map has incorrect rank")
        axes = []
        for row, offset, extent in zip(mapping.coefficients, mapping.offsets, shape, strict=True):
            selected = [axis for axis, value in enumerate(row) if value]
            if offset or any(value not in (0, 1) for value in row) or len(selected) > 1:
                raise ValueError("mask input map is not a singleton projection")
            if not selected:
                if extent != 1:
                    raise ValueError("mask constant projection needs a singleton extent")
            elif selected[0] in axes or extent != node.type.shape[selected[0]]:
                raise ValueError("mask input projection has incompatible extents")
            else:
                axes.append(selected[0])

    def at(operand, mapping, index):
        flat = 0
        for coordinate, extent in zip(mapping.apply(index), operand.type.shape, strict=True):
            flat = flat * extent + coordinate
        value = operand.elements[flat]
        if operation == "arith.cmpf":
            return unpack(
                "<d" if operand.type.dtype == "f64" else "<f",
                value.to_bytes(8 if operand.type.dtype == "f64" else 4, "little"),
            )[0]
        return value

    result = []
    for index in product(*(range(extent) for extent in node.type.shape)):
        args = tuple(at(value, mapping, index) for value, mapping in zip(operands, node.index_maps[:-1], strict=True))
        lhs, rhs = args if len(args) == 2 else (args[0], None)
        if operation == "arith.extui":
            value = lhs
        elif operation == "arith.andi":
            value = lhs & rhs
        elif operation == "arith.xori":
            value = lhs ^ rhs
        else:
            ordered = lhs == lhs and rhs == rhs
            compare = predicate[1:] if operation == "arith.cmpf" or predicate.startswith("s") else predicate
            value = (
                ordered
                and {
                    "eq": lhs == rhs,
                    "ne": lhs != rhs,
                    "lt": lhs < rhs,
                    "le": lhs <= rhs,
                    "gt": lhs > rhs,
                    "ge": lhs >= rhs,
                }[compare]
            )
        result.append(int(value))
    return TensorValue(node.type, tuple(result))

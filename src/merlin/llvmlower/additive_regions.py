"""Retype closed additive tensor regions without changing their static indexing.

Only zero, addition, subtraction and exact extensions act on data. Index work,
static loops and their branch predicates remain explicit source operations.
Unknown data operations refuse. This supports reductions expressed as gathers
as well as ordinary linalg reduction dimensions.
"""


def positive_zero_seed(value):
    """Prove a literal zero initializer, retaining the IEEE sign of zero."""
    import math

    op = value.owner
    if getattr(op, "name", None) in {"tensor.splat", "linalg.fill"}:
        return positive_zero_seed(op.operands[0])
    if getattr(op, "name", None) != "arith.constant":
        return False
    attr = op.properties.get("value", op.attributes.get("value"))
    values = list(attr.get_values()) if hasattr(attr, "get_values") else [attr.value.data]
    return bool(values) and all(v == 0 and math.copysign(1, v) > 0 for v in values)


def data_sources(op, element_predicate):
    """Return explicit and captured tensor data operands, excluding unused shapes."""
    from xdsl.dialects.builtin import TensorType

    result = []
    block = op.body.block
    for i, value in enumerate(op.inputs):
        if block.args[i].uses and element_predicate(value.type.element_type):
            result.append(value)
    for child in op.body.walk():
        if child.name == "tensor.extract":
            source = child.operands[0]
            if (
                isinstance(source.type, TensorType)
                and element_predicate(source.type.element_type)
                and source not in result
            ):
                result.append(source)
    return result


def retype_additive(op, replacements, element, *, floating):
    """Clone a checked additive generic; attach captured tensors as real inputs."""
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IntegerAttr, IntegerType, TensorType
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir.affine import AffineConstantExpr, AffineMap
    from xdsl.rewriter import Rewriter

    def data_type(typ):
        return (
            str(typ) in {"f16", "bf16", "f32", "f64"}
            if floating
            else isinstance(typ, IntegerType) and typ.width.data >= 128
        )

    allowed = {"arith.addf", "arith.subf", "arith.extf"} if floating else {"arith.addi", "arith.subi", "arith.extsi"}
    data_ops = 0
    for child in op.body.walk():
        if child.name in {"arith.constant", "scf.for", "scf.if", "scf.yield", "linalg.yield", "tensor.extract"}:
            if child.name == "arith.constant" and data_type(child.results[0].type):
                if child.value.value.data != 0:
                    return None
            continue
        touches_data = any(data_type(v.type) for v in (*child.operands, *child.results))
        if touches_data and child.name not in allowed:
            return None
        data_ops += touches_data and child.name in {"arith.addf", "arith.subf", "arith.addi", "arith.subi"}
    if not data_ops:
        return None
    if op.body.block.args[-1].uses and not positive_zero_seed(op.outputs[0]):
        return None
    if not all(data_type(v.type.element_type) for v in op.outputs):
        return None
    operations = []
    mapper = dict(replacements)
    # Every seed and unused shape operand is a literal zero. The caller proves
    # that original data seeds are zero; unused tensor.empty is never read.
    for value in op.operands:
        if value not in mapper:
            if value in op.inputs:
                index = list(op.inputs).index(value)
                if op.body.block.args[index].uses:
                    if data_type(value.type.element_type):
                        return None
                    # A used index/control tensor is neither a data seed nor
                    # a shape placeholder. Retain it; the provider must still
                    # prove that every gather address is closed and static.
                    mapper[value] = value
                    continue
            constant = arith.ConstantOp(IntegerAttr(0, element))
            splat = tensor.SplatOp(constant.result, [], TensorType(element, value.type.get_shape()))
            operations.extend([constant, splat])
            mapper[value] = splat.result
    cloned = op.clone(value_mapper=mapper)
    for region in [cloned.body, *(r for child in cloned.body.walk() for r in child.regions)]:
        for block in region.blocks:
            for arg in list(block.args):
                if data_type(arg.type):
                    Rewriter.replace_value_with_new_type(arg, element)
    for child in list(cloned.body.walk()):
        if child.name == "arith.constant" and data_type(child.results[0].type):
            Rewriter.replace_op(child, arith.ConstantOp(IntegerAttr(0, element)))
        elif child.name in ({"arith.extf"} if floating else {"arith.extsi"}):
            if data_type(child.results[0].type) or child.operands[0].type == child.results[0].type:
                Rewriter.replace_op(child, [], [child.operands[0]])
        elif floating and child.name in {"arith.addf", "arith.subf"}:
            cls = arith.AddiOp if child.name == "arith.addf" else arith.SubiOp
            Rewriter.replace_op(child, cls(*child.operands))
        else:
            for value in list(child.results):
                if data_type(value.type):
                    Rewriter.replace_value_with_new_type(value, element)
    inputs = [mapper[v] for v in op.inputs]
    maps = list(op.get_indexing_maps())
    rank = maps[-1].data.num_dims
    captures = [v for v in replacements if v not in op.inputs]
    for value in captures:
        cloned.body.block.insert_arg(element, len(inputs))
        inputs.append(replacements[value])
        maps.insert(-1, AffineMapAttr(AffineMap(rank, 0, (AffineConstantExpr(0),) * len(value.type.get_shape()))))
    region = cloned.detach_region(0)
    result_type = TensorType(element, op.results[0].type.get_shape())
    result = L.GenericOp(
        inputs=inputs,
        outputs=[mapper[op.outputs[0]]],
        body=region,
        indexing_maps=ArrayAttr(maps),
        iterator_types=op.get_iterator_types(),
        result_types=[result_type],
    )
    operations.append(result)
    return operations, result


def expose_reduce_domains(module):
    """Spell a named reduction with its complete static iteration maps."""
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir.affine import AffineDimExpr, AffineMap
    from xdsl.rewriter import Rewriter

    for op in list(module.walk()):
        if op.name != "linalg.reduce" or len(op.operands) != 2 or len(op.results) != 1:
            continue
        rank = len(op.operands[0].type.get_shape())
        dimensions = set(op.dimensions.get_values())
        if not dimensions <= set(range(rank)):
            continue
        clone = op.clone()
        generic = L.GenericOp(
            inputs=[op.operands[0]],
            outputs=[op.operands[1]],
            body=clone.detach_region(0),
            indexing_maps=ArrayAttr(
                [
                    AffineMapAttr(AffineMap.identity(rank)),
                    AffineMapAttr(
                        AffineMap(rank, 0, tuple(AffineDimExpr(d) for d in range(rank) if d not in dimensions))
                    ),
                ]
            ),
            iterator_types=ArrayAttr(
                [
                    L.IteratorTypeAttr(L.IteratorType.REDUCTION if d in dimensions else L.IteratorType.PARALLEL)
                    for d in range(rank)
                ]
            ),
            result_types=[op.results[0].type],
        )
        generic.attributes.update(op.attributes)
        Rewriter.replace_op(op, generic)

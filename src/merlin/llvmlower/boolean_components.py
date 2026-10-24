"""Boolean AND as integer numerators followed by an exact truth predicate."""


def rewrite_boolean_components(module):
    from math import prod

    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import IntegerAttr, TensorType, i8, i32
    from xdsl.dialects.linalg import ops as L
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower._ir import BATCHED_ITERATORS, BATCHED_MAPS, identity, mac, point, reshape

    bounds = {}
    changed = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.results) != 1 or str(op.results[0].type.element_type) != "i1":
            continue
        body = list(op.body.block.ops)
        if len(body) != 2 or body[0].name != "arith.andi":
            continue
        if len(op.inputs) != 2 or list(body[0].operands) != list(op.body.block.args[:2]):
            continue
        if any(it.data != L.IteratorType.PARALLEL for it in op.get_iterator_types()):
            continue
        shape = op.results[0].type.get_shape()
        operations = []
        inputs = []
        for i, source in enumerate(op.inputs):
            value = point(
                [source],
                TensorType(i8, shape),
                lambda args, emit: emit(arith.ExtUIOp(args[0], i8)),
                operations,
                maps=[list(op.get_indexing_maps())[i], identity(len(shape))],
                iterators=op.get_iterator_types(),
            )
            bounds[value] = 1
            inputs.append(value)
        zero = arith.ConstantOp(IntegerAttr(0, i32))
        seed = tensor.SplatOp(zero.result, [], TensorType(i32, shape))
        operations.extend([zero, seed])
        flat = []
        for value in inputs:
            reshaped = reshape(value, [prod(shape), 1, 1], operations)
            bounds[reshaped] = 1
            flat.append(reshaped)
        batch_seed = tensor.SplatOp(zero.result, [], TensorType(i32, [prod(shape), 1, 1]))
        operations.append(batch_seed)
        contraction = mac(flat, batch_seed.result, BATCHED_MAPS, BATCHED_ITERATORS)
        operations.append(contraction)
        result = reshape(contraction.results[0], shape, operations)
        bounds[result] = 2
        # This predicate decodes the complete device-computed AND numerator.
        final = point(
            [result],
            op.results[0].type,
            lambda args, emit: emit(arith.CmpiOp(args[0], emit(arith.ConstantOp(IntegerAttr(0, i32))), "ne")),
            operations,
            iterators=op.get_iterator_types(),
            host=True,
        )
        Rewriter.replace_op(op, operations, [final])
        changed += 1
    module.verify()
    return bounds, changed

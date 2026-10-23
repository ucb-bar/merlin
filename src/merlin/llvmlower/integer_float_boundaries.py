"""Keep native signed-integer conversion before true-division host epilogues.

The byte producers compute the complete integer numerator. One explicit integer
RNE conversion reproduces the source's sitofp, including any rounding before
subsequent floating operations. No division is moved across that boundary.
"""


def rewrite_integer_float_boundaries(module, input_bounds, *, output_grids=None):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import i128
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower._ir import emitter, generic, point, splat
    from merlin.llvmlower.fixed_point_ir import FORMAT, encode_fixed_integer
    from merlin.llvmlower.pointwise_primitives import expose_pointwise_primitives

    expose_pointwise_primitives(module)
    function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    bounds = dict(zip(function.body.block.args, input_bounds))
    added = {}
    count = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.inputs) != 1 or len(op.results) != 1:
            continue
        if any(it.data != L.IteratorType.PARALLEL for it in op.get_iterator_types()):
            continue
        body = list(op.body.block.ops)
        if (
            len(body) != 2
            or body[0].name not in {"arith.sitofp", "arith.uitofp"}
            or body[0].operands[0] is not op.body.block.args[0]
        ):
            continue
        source = op.inputs[0]
        if str(source.type.element_type) not in {"i1", "i8", "i16", "i32", "i64"}:
            continue
        if str(op.results[0].type.element_type) not in FORMAT:
            continue
        magnitude = bounds.get(source)
        if magnitude is None:
            from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

            literal = _constant_of(source)
            if type(literal) is int:
                magnitude = abs(literal)
        if magnitude is None:
            magnitude = 1 << (source.type.element_type.width.data - 1)
        unsigned = body[0].name == "arith.uitofp"
        if unsigned and str(source.type.element_type) != "i1":
            continue
        operations = []
        seed = splat(0, i128, op.results[0].type.get_shape(), operations)
        block = Block(arg_types=[source.type.element_type, i128])
        extend = (arith.ExtUIOp if unsigned else arith.ExtSIOp)(block.args[0], i128)
        block.add_op(L.YieldOp(emitter(block)(extend)))
        integer = generic([source], seed, block, op.get_indexing_maps(), op.get_iterator_types())
        operations.append(integer)
        added[integer.results[0]] = magnitude
        identity = list(op.get_indexing_maps())[-1]
        converted = point(
            [integer.results[0]],
            op.results[0].type,
            lambda args, emit: encode_fixed_integer(args[0], 0, op.results[0].type.element_type, emit),
            operations,
            maps=[identity, identity],
            iterators=op.get_iterator_types(),
            host=True,
        )
        if output_grids is not None:
            # Integer-to-float rounding preserves integrality. The rounded
            # magnitude is bounded by twice the original integer magnitude.
            output_grids[converted] = (0, 2 * magnitude)
        Rewriter.replace_op(op, operations, [converted])
        count += 1
    module.verify()
    return added, count

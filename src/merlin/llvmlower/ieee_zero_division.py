"""Separable native IEEE zero-divisor readout over a complete device numerator.

This policy selects the recorded x86 CPU invalid-result convention: negative
quiet default NaN for binary16/32/64, positive canonical quiet NaN for BF16.
It uses the computed numerator's bits; no eager outputs or original floating
arithmetic are replayed. Finite-grid admission remains a separate obligation.
"""


def rewrite_zero_division(module):
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import FloatAttr, IntegerType
    from xdsl.dialects.linalg import ops as L
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower._ir import constants, identity, point
    from merlin.llvmlower.fixed_point_ir import FORMAT
    from merlin.llvmlower.ieee_pointwise import canonical_bf16_storage
    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    count = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.inputs) != 2 or len(op.results) != 1:
            continue
        body = list(op.body.block.ops)
        if (
            len(body) != 2
            or body[0].name != "arith.divf"
            or any(it.data != L.IteratorType.PARALLEL for it in op.get_iterator_types())
        ):
            continue
        if list(body[0].operands) != list(op.body.block.args[:2]) or _constant_of(op.inputs[1]) != 0:
            continue
        typ = op.results[0].type
        token = str(typ.element_type)
        if token not in FORMAT:
            continue
        one = arith.ConstantOp(FloatAttr(1.0, typ.element_type))
        ones = tensor.SplatOp(one.result, [], op.inputs[1].type)
        operations = [one, ones]
        numerator = point(
            [op.inputs[0], ones.result],
            typ,
            lambda args, emit: emit(arith.MulfOp(args[0], args[1])),
            operations,
            maps=op.get_indexing_maps(),
            iterators=op.get_iterator_types(),
        )
        fraction, exponent, _ = FORMAT[token]
        storage = IntegerType(1 + fraction + exponent)

        def readout(args, emit):
            constant = constants(emit, storage)
            raw = emit(arith.BitcastOp(args[0], storage))
            divisor_raw = emit(arith.BitcastOp(args[1], storage))
            sign_mask = constant(1 << (fraction + exponent))
            sign = emit(arith.AndIOp(emit(arith.XOrIOp(raw, divisor_raw)), sign_mask))
            magnitude = emit(arith.AndIOp(raw, constant((1 << (fraction + exponent)) - 1)))
            is_zero = emit(arith.CmpiOp(magnitude, constant(0), "eq"))
            infinity = constant(((1 << exponent) - 1) << fraction)
            signed_infinity = emit(arith.OrIOp(infinity, sign))
            invalid_nan = ((1 << exponent) - 1) << fraction | 1 << (fraction - 1)
            if token != "bf16":
                invalid_nan |= 1 << (fraction + exponent)
            result = emit(arith.SelectOp(is_zero, constant(invalid_nan), signed_infinity))
            nan = emit(arith.CmpiOp(magnitude, infinity, "ugt"))
            quiet_payload = emit(arith.OrIOp(raw, constant(1 << (fraction - 1))))
            if token == "bf16":
                quiet_payload = constant(0x7FC0)
            result = emit(arith.SelectOp(nan, quiet_payload, result))
            return canonical_bf16_storage(op, emit(arith.BitcastOp(result, typ.element_type)), emit)

        rank = len(typ.get_shape())
        result = point(
            [numerator, op.inputs[1]],
            typ,
            readout,
            operations,
            maps=[identity(rank), list(op.get_indexing_maps())[1], identity(rank)],
            iterators=op.get_iterator_types(),
            host=True,
        )
        Rewriter.replace_op(op, operations, [result])
        count += 1
    module.verify()
    return count

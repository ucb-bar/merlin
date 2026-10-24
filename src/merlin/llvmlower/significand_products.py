"""Exact pointwise IEEE products using dtype-bounded significand limb GEMMs.

Exponents remain runtime representation metadata. Full significands (including
every subnormal) are device operands; their exact product receives one integer
RNE readout. No exponent span or limb count is inferred from tensor data.
"""


def rewrite_significand_products(module):
    from math import prod

    from xdsl.dialects import arith
    from xdsl.dialects.builtin import TensorType, i64
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower import _ir
    from merlin.llvmlower._ir import BATCHED_ITERATORS, BATCHED_MAPS, identity, mac, reshape, splat
    from merlin.llvmlower.fixed_point_ir import FORMAT
    from merlin.llvmlower.floating_component_ir import point
    from merlin.llvmlower.ieee_dyadic_ir import decode, encode
    from merlin.llvmlower.ieee_pointwise import canonical_bf16_storage, overlay
    from merlin.llvmlower.uniform_numeric_literals import compile_time_operand

    bounds, proofs = {}, []
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.inputs) != 2 or len(op.results) != 1:
            continue
        body = list(op.body.block.ops)
        token = str(op.results[0].type.element_type)
        if token not in {"f16", "bf16", "f32"} or any(it.data.value != "parallel" for it in op.get_iterator_types()):
            continue
        if (
            len(body) != 2
            or body[0].name != "arith.mulf"
            or list(body[0].operands) != list(op.body.block.args[:2])
            or list(body[1].operands) != list(body[0].results)
        ):
            continue
        if not any(compile_time_operand(value) for value in op.inputs):
            continue
        shape = op.results[0].type.get_shape()
        maps = list(op.get_indexing_maps())
        if maps[-1] != identity(len(shape)):
            continue
        operations, expanded = [], []
        magnitude = (1 << (FORMAT[token][0] + 1)) - 1
        operand_bounds = []
        for source, amap in zip(op.inputs, maps[:-1]):
            from math import isfinite

            from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

            literal = _constant_of(source)
            bound = magnitude
            if literal is not None and isfinite(literal):
                fraction, exponent, _ = FORMAT[token]
                if token == "bf16":
                    import struct

                    from merlin.targetgen.exact_arithmetic import round_binary

                    rounded = round_binary(literal, 16, bfloat16=True)
                    word = struct.unpack("<I", struct.pack("<f", rounded))[0] >> 16
                else:
                    word = int.from_bytes(source.type.element_type.pack([float(literal)]), "little")
                field = word >> fraction & ((1 << exponent) - 1)
                coefficient = word & ((1 << fraction) - 1)
                coefficient |= 1 << fraction if field else 0
                if word >> (fraction + exponent):
                    coefficient = -coefficient
                bound = abs(coefficient)
                packed = splat(coefficient, i64, source.type.get_shape(), operations)
            else:
                packed = point(
                    [source],
                    TensorType(i64, source.type.get_shape()),
                    lambda args, emit: decode(args[0], emit)[0],
                    operations,
                )
            bounds[packed] = bound
            operand_bounds.append(bound)
            broadcast = _ir.point(
                [packed],
                TensorType(i64, shape),
                lambda args, emit: args[0],
                operations,
                maps=[amap, identity(len(shape))],
                iterators=op.get_iterator_types(),
            )
            reshaped = reshape(broadcast, [prod(shape), 1, 1], operations)
            bounds[broadcast] = bounds[reshaped] = bound
            expanded.append(reshaped)
        seed = splat(0, i64, [prod(shape), 1, 1], operations)
        integer = mac(expanded, seed, BATCHED_MAPS, BATCHED_ITERATORS)
        operations.append(integer)
        bounds[seed] = 0
        product_bound = prod(operand_bounds)
        bounds[integer.results[0]] = product_bound
        reshaped = reshape(integer.results[0], shape, operations)
        bounds[reshaped] = product_bound

        def readout(args, emit):
            a, b = [decode(v, emit) for v in args[1:3]]
            exponent = emit(arith.AddiOp(a[1], b[1]))
            negative_zero = emit(arith.XOrIOp(a[2], b[2]))
            rounded = encode(args[0], exponent, op.results[0].type.element_type, emit, zero_negative=negative_zero)
            return canonical_bf16_storage(op, overlay(op, list(args[1:3]), rounded, emit), emit)

        result = point(
            [reshaped, *op.inputs],
            op.results[0].type,
            readout,
            operations,
            maps=[identity(len(shape)), *maps],
            iterators=op.get_iterator_types(),
        )
        Rewriter.replace_op(op, operations, [result])
        proofs.append(
            dict(
                form="typed_ieee_significand_product",
                eligibility="at_least_one_compile_time_operand",
                dtype=token,
                significand_absolute_bound=magnitude,
                product_absolute_bound=product_bound,
                exponent="runtime_ieee_exponent_sum",
                source_data_arithmetic=True,
                rounding="single_integer_RNE_with_gradual_underflow_and_overflow",
            )
        )
    module.verify()
    return bounds, proofs

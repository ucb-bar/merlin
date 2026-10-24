"""Full-domain literal integer quotients via exact reciprocal products.

For 0<=n<2**w and d>0, M=ceil(2**s/d), s=w+ceil(log2(d)),
gives floor(n*M/2**s) in {floor(n/d), floor(n/d)+1}. One exact
multiply/compare correction recovers the quotient. Zero and CPU trapping
signed overflow domains are not licensed by this value-producing theorem.
"""


def literal_divisor(op):
    from merlin.llvmlower._ir import input_literal

    literal = input_literal(op.operands[1])
    return literal if type(literal) is int else None


def reciprocal(width, divisor):
    if width not in {8, 16, 32, 64} or not divisor:
        raise ValueError("unproved literal division domain")
    d = abs(divisor)
    shift = width + (d - 1).bit_length()
    multiplier = ((1 << shift) + d - 1) // d
    # Exact universal inequalities; these are not sampled numerator checks.
    assert 0 <= multiplier * d - (1 << shift) < d
    assert ((1 << width) - 1) * (multiplier * d - (1 << shift)) < d * (1 << shift)
    return multiplier, shift


def cpu_floor_wrap(proof, width):
    """Check the complete signed-overflow exception class, independent of inputs."""
    if width not in {8, 16, 32, 64}:
        return False
    expected = b"".join((value % (1 << width)).to_bytes(width // 8, "little") for value in (-(1 << (width - 1)), 0, -1))
    return any(
        row.get("width") == width
        and row.get("mode") == "floor"
        and row.get("divisor") == -1
        and row.get("observation", {}).get("status") == "value"
        and row["observation"].get("dtype") == "torch.int" + str(width)
        and row["observation"].get("bits") == list(expected)
        for row in proof.get("integer_checks", {}).get("exceptions", [])
    )


def rewrite_integer_divisions(module, *, wrapping_floor_widths=()):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, IntegerType, TensorType
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower import _ir
    from merlin.llvmlower._ir import constants
    from merlin.llvmlower.floating_component_ir import point

    bounds, proofs = {}, []
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.results) != 1:
            continue
        body = list(op.body.block.ops)
        if len(body) != 2 or body[0].name not in {"arith.divsi", "arith.remsi", "arith.floordivsi"}:
            continue
        scalar = body[0]
        dtype = op.results[0].type.element_type
        if not isinstance(dtype, IntegerType) or dtype.width.data not in {8, 16, 32, 64}:
            continue
        divisor = literal_divisor(scalar)
        wrapping_floor = scalar.name == "arith.floordivsi" and dtype.width.data in wrapping_floor_widths
        if divisor is None or not divisor or (divisor == -1 and dtype.width.data >= 32 and not wrapping_floor):
            continue
        if any(it.data.value != "parallel" for it in op.get_iterator_types()):
            continue
        arg = scalar.operands[0]
        if arg not in op.body.block.args[: len(op.inputs)]:
            continue
        index = op.body.block.args.index(arg)
        maps = list(op.get_indexing_maps())
        source, amap = op.inputs[index], maps[index]
        wide = IntegerType(256)
        multiplier, shift = reciprocal(dtype.width.data, divisor)
        operations = []

        def magnitude(args, emit):
            value = emit(arith.ExtSIOp(args[0], wide))
            zero = emit(arith.ConstantOp(IntegerAttr(0, wide)))
            return emit(arith.SelectOp(emit(arith.CmpiOp(value, zero, "slt")), emit(arith.SubiOp(zero, value)), value))

        packed = point([source], TensorType(wide, source.type.get_shape()), magnitude, operations)
        bounds[packed] = 1 << (dtype.width.data - 1)
        integer = _ir.point(
            [packed],
            TensorType(wide, op.results[0].type.get_shape()),
            lambda args, emit: emit(arith.MuliOp(args[0], emit(arith.ConstantOp(IntegerAttr(multiplier, wide))))),
            operations,
            maps=[amap, maps[-1]],
            iterators=op.get_iterator_types(),
        )
        bounds[integer] = (1 << (dtype.width.data - 1)) * multiplier

        def quotient(args, emit):
            c = constants(emit, wide)
            original = emit(arith.ExtSIOp(args[1], wide))
            negative = emit(arith.CmpiOp(original, c(0), "slt"))
            n = emit(arith.SelectOp(negative, emit(arith.SubiOp(c(0), original)), original))
            q = emit(arith.ShRUIOp(args[0], c(shift)))
            excess = emit(arith.CmpiOp(emit(arith.MuliOp(q, c(abs(divisor)))), n, "ugt"))
            q = emit(arith.SubiOp(q, emit(arith.SelectOp(excess, c(1), c(0)))))
            signed = emit(arith.SelectOp(negative, emit(arith.SubiOp(c(0), q)), q))
            if divisor < 0:
                signed = emit(arith.SubiOp(c(0), signed))
            remainder = emit(arith.SubiOp(original, emit(arith.MuliOp(signed, c(divisor)))))
            result = signed
            if scalar.name == "arith.remsi":
                result = remainder
            elif scalar.name == "arith.floordivsi":
                residual = emit(arith.CmpiOp(remainder, c(0), "ne"))
                sign_differs = emit(arith.CmpiOp(original, c(0), "sge" if divisor < 0 else "slt"))
                adjust = emit(arith.AndIOp(residual, sign_differs))
                result = emit(arith.SubiOp(signed, emit(arith.SelectOp(adjust, c(1), c(0)))))
            return emit(arith.TruncIOp(result, dtype))

        result = point(
            [integer, source],
            op.results[0].type,
            quotient,
            operations,
            maps=[maps[-1], amap, maps[-1]],
            iterators=op.get_iterator_types(),
        )
        Rewriter.replace_op(op, operations, [result])
        proofs.append(
            dict(
                form="full_dtype_literal_quotient",
                width=dtype.width.data,
                divisor=divisor,
                multiplier=str(multiplier),
                shift=shift,
                rounding=scalar.name,
                proof="universal_reciprocal_interval_and_one_correction",
                source_data_arithmetic=True,
            )
        )
    module.verify()
    return bounds, proofs

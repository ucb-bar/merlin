"""Optional IEEE special readout, separate from exact finite component arithmetic."""


def finite_values(values):
    """Input-defined finite subproblem replaces each special by exact positive zero."""
    from math import isfinite

    return [value if isfinite(value) else 0 for value in values]


def special(value, emit):
    """Classify inf/NaN by bits without consuming or quieting the operand."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, IntegerType

    from merlin.llvmlower.fixed_point_ir import FORMAT

    fraction, exponent, _ = FORMAT[str(value.type)]
    storage = IntegerType(1 + fraction + exponent)
    raw = emit(arith.BitcastOp(value, storage))
    mask = emit(arith.ConstantOp(IntegerAttr(((1 << exponent) - 1) << fraction, storage)))
    field = emit(arith.AndIOp(raw, mask))
    return emit(arith.CmpiOp(field, mask, "eq"))


def finite_or_zero(value, emit):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr

    flag = special(value, emit)
    zero = emit(arith.ConstantOp(FloatAttr(0.0, value.type)))
    return emit(arith.SelectOp(flag, zero, value))


def canonical_bf16_storage(op, value, emit):
    """Certified scalar C10 versus AVX2 vector NaN storage of op's BF16 result; finite words untouched.

    Only a BF16 value of an op that records a CPU storage profile changes.
    """
    import json

    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, i16

    from merlin.llvmlower.native_cpu_layout import vector_predicate

    if str(value.type) != "bf16" or "merlin.cpu_storage_profile" not in op.attributes:
        return value
    profile = json.loads(op.attributes["merlin.cpu_storage_profile"].data)
    raw = emit(arith.BitcastOp(value, i16))
    magnitude = emit(arith.AndIOp(raw, emit(arith.ConstantOp(IntegerAttr(0x7FFF, i16)))))
    nan = emit(arith.CmpiOp(magnitude, emit(arith.ConstantOp(IntegerAttr(0x7F80, i16))), "ugt"))
    vector = vector_predicate(profile, emit)
    selected = emit(
        arith.SelectOp(
            vector, emit(arith.ConstantOp(IntegerAttr(-1, i16))), emit(arith.ConstantOp(IntegerAttr(0x7FC0, i16)))
        )
    )
    result = emit(arith.BitcastOp(selected, value.type))
    return emit(arith.SelectOp(nan, result, value))


def dependence_proof(op):
    """Prove the exact element dependence set from a closed scalar/index DAG."""
    from xdsl.ir.affine import AffineMap

    maps = [a.data for a in op.get_indexing_maps()]
    if (
        op.name != "linalg.generic"
        or any(it.data.value != "parallel" for it in op.get_iterator_types())
        or maps[-1] != AffineMap.identity(len(op.results[0].type.get_shape()))
        or any(amap.num_symbols for amap in maps)
    ):
        raise ValueError("special readout requires a closed parallel output index map")
    arguments = op.body.block.args
    dependencies = set()
    allowed = {
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.negf",
        "arith.extf",
        "arith.truncf",
        "arith.constant",
        "math.fma",
    }

    def visit(value):
        if value in arguments:
            index = arguments.index(value)
            if index >= len(op.inputs):
                raise ValueError("special readout cannot depend on an unproved output seed")
            dependencies.add(index)
            return
        owner = value.owner
        if getattr(owner, "name", None) not in allowed or owner.parent_block() is not op.body.block:
            raise ValueError("special readout has an unproved scalar dependence")
        for operand in owner.operands:
            visit(operand)

    visit(op.body.block.last_op.operands[0])
    return dict(
        method="closed_parallel_scalar_DAG_and_affine_input_maps",
        output_shape=list(op.results[0].type.get_shape()),
        output_index_map=str(maps[-1]),
        input_dependencies=[dict(operand=i, index_map=str(maps[i])) for i in sorted(dependencies)],
        affected_predicate="OR(is_nonfinite(input[operand][index_map(output_index)]))",
        finite_dependence_outputs="retain_exact_device_integer_readout_without_overlay",
    )


def overlay(op, inputs, fallback, emit):
    """Evaluate the original scalar source only on an affected output element.

    No finite floating subproblem is replayed: the false branch returns the
    exact integer readout. A closed parallel primitive is required by the caller.
    A closed per-dtype source proof and the independent final bit grader remain gates.
    """
    from xdsl.dialects import arith, scf
    from xdsl.ir import Block, Region

    from merlin.llvmlower.fixed_point_ir import FORMAT

    proof = dependence_proof(op)
    dependencies = {row["operand"] for row in proof["input_dependencies"]}
    flags = [
        special(value, emit)
        for index, value in enumerate(inputs)
        if index in dependencies and str(value.type) in FORMAT
    ]
    if not flags:
        return fallback
    flag = flags[0]
    for other in flags[1:]:
        flag = emit(arith.OrIOp(flag, other))
    mapping = dict(zip(op.body.block.args[: len(op.inputs)], inputs))
    selected = Block()

    def selected_emit(child):
        selected.add_op(child)
        return child.results[0]

    for child in list(op.body.block.ops)[:-1]:
        if child.name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf", "math.fma"}:
            mapping[child.results[0]] = special_arithmetic(
                child.name, [mapping[v] for v in child.operands], selected_emit
            )
        elif child.name in {"arith.extf", "arith.truncf"}:
            mapping[child.results[0]] = special_conversion(
                child.name, mapping[child.operands[0]], child.results[0].type, selected_emit
            )
        else:
            copied = child.clone(value_mapper=mapping)
            selected.add_op(copied)
            mapping.update(zip(child.results, copied.results, strict=True))
    yielded = op.body.block.last_op.operands[0]
    selected.add_op(scf.YieldOp(mapping[yielded]))
    retained = Block([scf.YieldOp(fallback)])
    return emit(scf.IfOp(flag, [fallback.type], Region(selected), Region(retained)))


def special_arithmetic(name, operands, emit):
    """Closed x86 IEEE add/sub/mul/div propagation, expressed entirely in bits."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerType

    from merlin.llvmlower._ir import constants
    from merlin.llvmlower.fixed_point_ir import FORMAT

    typ = operands[0].type
    fraction, exponent, _ = FORMAT[str(typ)]
    integer = IntegerType(1 + fraction + exponent)

    c = constants(emit, integer)

    def binary(cls, a, b):
        return emit(cls(a, b))

    def choose(flag, a, b):
        return emit(arith.SelectOp(flag, a, b))

    def eq(a, b):
        return emit(arith.CmpiOp(a, b, "eq"))

    sign_mask = c(1 << (fraction + exponent))
    infinity = c(((1 << exponent) - 1) << fraction)
    magnitude_mask = c((1 << (fraction + exponent)) - 1)
    zero = c(0)
    raws = [emit(arith.BitcastOp(value, integer)) for value in operands]
    magnitudes = [binary(arith.AndIOp, raw, magnitude_mask) for raw in raws]
    signs = [binary(arith.AndIOp, raw, sign_mask) for raw in raws]
    nans = [emit(arith.CmpiOp(magnitude, infinity, "ugt")) for magnitude in magnitudes]
    infs = [eq(magnitude, infinity) for magnitude in magnitudes]
    quiet = c(1 << (fraction - 1))
    default = c(
        (((1 << exponent) - 1) << fraction)
        | (1 << (fraction - 1))
        | (0 if str(typ) == "bf16" else 1 << (fraction + exponent))
    )
    if name in {"arith.addf", "arith.subf"}:
        right_sign = signs[1] if name == "arith.addf" else binary(arith.XOrIOp, signs[1], sign_mask)
        opposite = emit(arith.CmpiOp(signs[0], right_sign, "ne"))
        invalid = binary(arith.AndIOp, binary(arith.AndIOp, infs[0], infs[1]), opposite)
        sign = choose(infs[0], signs[0], right_sign)
    elif name == "arith.mulf":
        invalid = binary(
            arith.OrIOp,
            binary(arith.AndIOp, infs[0], eq(magnitudes[1], zero)),
            binary(arith.AndIOp, infs[1], eq(magnitudes[0], zero)),
        )
        sign = binary(arith.XOrIOp, signs[0], signs[1])
    elif name == "arith.divf":
        invalid = binary(arith.AndIOp, infs[0], infs[1])
        sign = binary(arith.XOrIOp, signs[0], signs[1])
    elif name == "math.fma":
        product_sign = binary(arith.XOrIOp, signs[0], signs[1])
        product_inf = binary(arith.OrIOp, infs[0], infs[1])
        invalid_product = binary(
            arith.OrIOp,
            binary(arith.AndIOp, infs[0], eq(magnitudes[1], zero)),
            binary(arith.AndIOp, infs[1], eq(magnitudes[0], zero)),
        )
        opposite = emit(arith.CmpiOp(product_sign, signs[2], "ne"))
        invalid_sum = binary(arith.AndIOp, binary(arith.AndIOp, product_inf, infs[2]), opposite)
        invalid = binary(arith.OrIOp, invalid_product, invalid_sum)
        sign = choose(product_inf, product_sign, signs[2])
    else:
        raise ValueError("unproved IEEE scalar special operation: " + name)
    raw = choose(invalid, default, binary(arith.OrIOp, infinity, sign))
    if name == "arith.divf":
        # A finite numerator divided by an infinite divisor yields signed zero.
        raw = choose(binary(arith.AndIOp, infs[1], emit(arith.CmpiOp(magnitudes[0], infinity, "ult"))), sign, raw)
    # Unprotected competing NaNs refuse. Complete source guards override this
    # provisional word with their proved priority. BF16 opmath canonicalizes.
    propagated = binary(arith.OrIOp, raws[-1], quiet)
    has_nan = nans[-1]
    for nan, word in reversed(list(zip(nans[:-1], raws[:-1], strict=True))):
        propagated = choose(nan, binary(arith.OrIOp, word, quiet), propagated)
        has_nan = binary(arith.OrIOp, nan, has_nan)
    if str(typ) == "bf16":
        propagated = c(((1 << exponent) - 1) << fraction | (1 << (fraction - 1)))
    raw = choose(has_nan, propagated, raw)
    return emit(arith.BitcastOp(raw, typ))


def special_conversion(name, operand, destination, emit, *, finite_fallback=None):
    """Retain NaN sign/payload when RV floating conversions would canonicalize it."""
    from xdsl.dialects import arith, scf
    from xdsl.dialects.builtin import IntegerType, i64
    from xdsl.ir import Block, Region

    from merlin.llvmlower._ir import constants, emitter
    from merlin.llvmlower.fixed_point_ir import FORMAT

    source_fraction, source_exponent, _ = FORMAT[str(operand.type)]
    fraction, exponent, _ = FORMAT[str(destination)]
    source_storage = IntegerType(1 + source_fraction + source_exponent)
    storage = IntegerType(1 + fraction + exponent)
    flag = special(operand, emit)
    selected = Block()

    e = emitter(selected)
    c = constants(e, i64)
    raw = e(arith.BitcastOp(operand, source_storage))
    if source_storage != i64:
        raw = e(arith.ExtUIOp(raw, i64))
    payload = e(arith.AndIOp(raw, c((1 << source_fraction) - 1)))
    nan = e(arith.CmpiOp(payload, c(0), "ne"))
    if fraction > source_fraction:
        payload = e(arith.ShLIOp(payload, c(fraction - source_fraction)))
    elif fraction < source_fraction:
        payload = e(arith.ShRUIOp(payload, c(source_fraction - fraction)))
    if not (str(operand.type) == "bf16" and str(destination) == "f32"):
        payload = e(arith.OrIOp(payload, c(1 << (fraction - 1))))
    if str(destination) == "bf16":
        payload = c(1 << (fraction - 1))
    payload = e(arith.SelectOp(nan, payload, c(0)))
    sign = e(arith.ShRUIOp(raw, c(source_fraction + source_exponent)))
    sign = e(arith.ShLIOp(sign, c(fraction + exponent)))
    if str(destination) == "bf16":
        sign = e(arith.SelectOp(nan, c(0), sign))
    raw = e(arith.OrIOp(e(arith.OrIOp(sign, c(((1 << exponent) - 1) << fraction))), payload))
    if storage != i64:
        raw = e(arith.TruncIOp(raw, storage))
    value = e(arith.BitcastOp(raw, destination))
    selected.add_op(scf.YieldOp(value))
    finite = Block()
    if finite_fallback is None:
        converted = (arith.ExtFOp if name == "arith.extf" else arith.TruncFOp)(operand, destination)
        finite.add_ops([converted, scf.YieldOp(converted.result)])
    else:
        finite.add_op(scf.YieldOp(finite_fallback))
    return emit(scf.IfOp(flag, [destination], Region(selected), Region(finite)))

"""Native sign and closed compare/select epilogues over exact integer words.

Every payload bit is preserved, including signed zeros and NaN payloads. The
linear data term is an exact byte component producer; the sign-bit offset is
explicit modular host recombination. No floating conversion is involved.
"""


def rewrite_native_sign_components(
    module, *, canonical_bf16_nan=False, defer_floating_negation=False, cpu_profiles=None
):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, IntegerType, TensorType, UnitAttr
    from xdsl.dialects.linalg import ops as L
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower._ir import constants, point

    widths = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}
    bounds = {}
    proofs = []
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.inputs) != 1 or len(op.results) != 1:
            continue
        if any(it.data != L.IteratorType.PARALLEL for it in op.get_iterator_types()):
            continue
        body = list(op.body.block.ops)
        argument = op.body.block.args[0]
        negate = len(body) == 2 and body[0].name == "arith.negf" and body[0].operands[0] is argument
        literals = set()
        literal_casts = {"arith.sitofp", "arith.uitofp", "arith.extf", "arith.truncf"}
        for child in body[:-3]:
            if child.name == "arith.constant" or (
                child.name in literal_casts and all(value in literals for value in child.operands)
            ):
                literals.update(child.results)
            else:
                break
        clamp = False
        if len(body) >= 4:
            compare, select, terminator = body[-3:]
            choices = set(compare.operands)
            clamp = (
                sum(len(child.results) for child in body[:-3]) == len(literals)
                and compare.name in {"arith.cmpi", "arith.cmpf"}
                and argument in choices
                and len(choices) == 2
                and (choices - {argument}) <= literals
                and select.name == "arith.select"
                and select.operands[0] is compare.results[0]
                and set(select.operands[1:]) == choices
                and terminator.name == "linalg.yield"
                and list(terminator.operands) == list(select.results)
            )
        if not (negate or clamp):
            continue
        source = op.inputs[0]
        bits = widths.get(str(source.type.element_type))
        floating = bits is not None
        if negate and floating and defer_floating_negation:
            # Keep the finite dyadic grid visible to following arithmetic.
            # The floating component pass lowers negation with its zero sign.
            continue
        if isinstance(source.type.element_type, IntegerType) and source.type.element_type.width.data in {8, 16, 32, 64}:
            bits = source.type.element_type.width.data
        if bits is None or source.type.element_type != op.results[0].type.element_type:
            continue
        cpu_profile = (
            (cpu_profiles or {}).get(source) if floating and str(source.type.element_type) in {"f16", "bf16"} else None
        )
        if floating and str(source.type.element_type) in {"f16", "bf16"} and cpu_profiles is not None:
            if cpu_profile is None:
                continue
        element = IntegerType(bits)
        shape = source.type.get_shape() if clamp else op.results[0].type.get_shape()
        operations = []
        packed = point(
            [source],
            TensorType(element, source.type.get_shape()),
            lambda args, emit: emit(arith.BitcastOp(args[0], element)) if floating else args[0],
            operations,
        )
        bounds[packed] = 1 << (bits - 1)
        offset = -(1 << (bits - 1)) if negate else 0
        changed = point(
            [packed],
            TensorType(element, shape),
            lambda args, emit: emit(arith.AddiOp(args[0], emit(arith.ConstantOp(IntegerAttr(offset, element))))),
            operations,
            maps=None if clamp else op.get_indexing_maps(),
            iterators=None if clamp else op.get_iterator_types(),
        )
        bounds[changed] = 1 << (bits - 1)

        def readout(args, emit):
            if not floating:
                return args[0]
            raw = args[0]
            if cpu_profile is not None and str(source.type.element_type) in {"f16", "bf16"}:
                from merlin.llvmlower.native_cpu_layout import vector_predicate

                c = constants(emit, element)
                fraction = 7 if str(source.type.element_type) == "bf16" else 10
                infinity = 0x7F80 if fraction == 7 else 0x7C00
                magnitude = emit(arith.AndIOp(raw, c(0x7FFF)))
                nan = emit(arith.CmpiOp(magnitude, c(infinity), "ugt"))
                vector = vector_predicate(cpu_profile, emit)
                corrected = (
                    c(0x7FC0 if negate else 0xFFFF) if fraction == 7 else emit(arith.OrIOp(raw, c(1 << (fraction - 1))))
                )
                # Sign change's scalar C10 path converts; threshold's vector
                # path converts. The other path retains the original word.
                if negate:
                    from xdsl.dialects.builtin import i1

                    vector = emit(arith.XOrIOp(vector, emit(arith.ConstantOp(IntegerAttr(1, i1)))))
                flag = emit(arith.AndIOp(nan, vector))
                raw = emit(arith.SelectOp(flag, corrected, raw))
            if negate and canonical_bf16_nan and str(source.type.element_type) == "bf16":
                c = constants(emit, element)
                mask, infinity, canonical = c(0x7FFF), c(0x7F80), c(0x7FC0)
                nan = emit(arith.CmpiOp(emit(arith.AndIOp(raw, mask)), infinity, "ugt"))
                raw = emit(arith.SelectOp(nan, canonical, raw))
            return emit(arith.BitcastOp(raw, source.type.element_type))

        # The closed sign/clamp proof licenses this host readout, including
        # BF16 NaN canonicalization and the original literal compare/select.
        result = point(
            [changed],
            TensorType(source.type.element_type, shape),
            readout,
            operations,
            iterators=None if clamp else op.get_iterator_types(),
            host=not clamp,
        )
        if clamp:
            retained = op.clone(value_mapper={source: result})
            retained.attributes["merlin.exact_host_arithmetic"] = UnitAttr()
            operations.append(retained)
            result = retained.results[0]
        Rewriter.replace_op(op, operations, [result])
        proofs.append(
            dict(
                source="arith.negf" if negate else "closed_constant_compare_select",
                representation="ieee_storage_bits" if floating else "native_signed_integer",
                width=bits,
                arithmetic="sign_bit_offset_modulo_storage_width" if negate else "source_compare_select_retained_host",
                nan_payload=(
                    "exhaustive_CPU_scalar_vector_storage_forms"
                    if cpu_profile is not None
                    else "canonical_positive_quiet"
                    if negate and canonical_bf16_nan and str(source.type.element_type) == "bf16"
                    else "preserved"
                    if negate
                    else "source_predicate_semantics"
                ),
                **({"cpu_storage_profile": cpu_profile} if cpu_profile is not None else {}),
            )
        )
    module.verify()
    return bounds, proofs

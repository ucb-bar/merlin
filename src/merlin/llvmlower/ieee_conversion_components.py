"""Typed storage conversions with complete word producers and integer RNE readout."""


def rewrite_ieee_conversions(module, *, copy_profiles=None):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, IntegerType, TensorType
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower import _ir
    from merlin.llvmlower.floating_component_ir import point
    from merlin.llvmlower.ieee_dyadic_ir import decode, encode
    from merlin.llvmlower.ieee_pointwise import canonical_bf16_storage, special_conversion

    bounds, proofs = {}, []
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.inputs) != 1 or len(op.results) != 1:
            continue
        body = list(op.body.block.ops)
        source, target = op.inputs[0], op.results[0].type.element_type
        if (
            len(body) != 2
            or body[0].name not in {"arith.extf", "arith.truncf"}
            or body[0].operands[0] is not op.body.block.args[0]
            or any(it.data.value != "parallel" for it in op.get_iterator_types())
            or str(source.type.element_type) not in {"f16", "bf16", "f32"}
            or str(target) not in {"f16", "bf16", "f32", "f64"}
        ):
            continue
        storage = IntegerType(source.type.element_type.bitwidth)
        operations = []
        packed = point(
            [source],
            TensorType(storage, source.type.get_shape()),
            lambda args, emit: emit(arith.BitcastOp(args[0], storage)),
            operations,
        )
        bounds[packed] = 1 << (storage.width.data - 1)
        integer = _ir.point(
            [packed],
            TensorType(storage, op.results[0].type.get_shape()),
            lambda args, emit: emit(arith.AddiOp(args[0], emit(arith.ConstantOp(IntegerAttr(0, storage))))),
            operations,
            maps=op.get_indexing_maps(),
            iterators=op.get_iterator_types(),
        )
        bounds[integer] = 1 << (storage.width.data - 1)

        def readout(args, emit):
            original = emit(arith.BitcastOp(args[0], source.type.element_type))
            significand, exponent, negative = decode(original, emit)
            finite = encode(significand, exponent, target, emit, zero_negative=negative)
            result = special_conversion(body[0].name, original, target, emit, finite_fallback=finite)
            context = (copy_profiles or {}).get(source)
            if context is not None and (str(source.type.element_type), str(target)) in {
                ("f16", "f32"),
                ("f32", "bf16"),
            }:
                from merlin.llvmlower.native_cpu_layout import vector_predicate

                integer = IntegerType(target.bitwidth)
                source_storage = IntegerType(source.type.element_type.bitwidth)
                sf = 10 if str(source.type.element_type) == "f16" else 23
                se = 5 if sf == 10 else 8
                raw = emit(arith.BitcastOp(original, source_storage))
                magnitude = emit(
                    arith.AndIOp(raw, emit(arith.ConstantOp(IntegerAttr((1 << (sf + se)) - 1, source_storage))))
                )
                nan = emit(
                    arith.CmpiOp(
                        magnitude, emit(arith.ConstantOp(IntegerAttr(((1 << se) - 1) << sf, source_storage))), "ugt"
                    )
                )
                if str(target) == "bf16":
                    replacement = emit(arith.BitcastOp(emit(arith.ConstantOp(IntegerAttr(-1, integer))), target))
                else:
                    # Dense copy uses eight float lanes; the software tail
                    # selects +all-payload NaN. Strided copies use F16C too.
                    if not context["dense"]:
                        return result
                    from xdsl.dialects.builtin import i1

                    vector = vector_predicate(context, emit)
                    scalar = emit(arith.XOrIOp(vector, emit(arith.ConstantOp(IntegerAttr(1, i1)))))
                    nan = emit(arith.AndIOp(nan, scalar))
                    replacement = emit(
                        arith.BitcastOp(emit(arith.ConstantOp(IntegerAttr(0x7FFFFFFF, integer))), target)
                    )
                result = emit(arith.SelectOp(nan, replacement, result))
            return canonical_bf16_storage(op, result, emit)

        result = point([integer], op.results[0].type, readout, operations)
        Rewriter.replace_op(op, operations, [result])
        proofs.append(
            dict(
                form="typed_storage_conversion",
                source=str(source.type.element_type),
                target=str(target),
                domain="all_source_storage_words",
                rounding="integer_RNE; explicit_NaN_bit_readout; source_path_guards_retained",
                context="tensor_copy" if copy_profiles is not None else "typed_opmath",
            )
        )
    module.verify()
    return bounds, proofs

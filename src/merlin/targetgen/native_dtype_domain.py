"""Input-independent domains and semantic obligations for native components.

These are storage domains, not observations of calibration or example tensors.
An IEEE finite grid alone does not prove an IEEE arithmetic implementation.
"""

from merlin.llvmlower.fixed_point_ir import FORMAT


def _word_generic(op):
    """Closed native-width sign or positive-zero threshold, with exact wiring."""
    from xdsl.dialects.builtin import FloatAttr

    parent = op.parent_op()
    if parent is None or parent.name != "linalg.generic" or len(parent.inputs) != 1 or len(parent.results) != 1:
        return False
    if any(it.data.value != "parallel" for it in parent.get_iterator_types()):
        return False
    body = list(parent.body.block.ops)
    arg = parent.body.block.args[0]
    if parent.results[0].type.element_type != arg.type or parent.body.block.args[-1].uses:
        return False
    if op.name == "arith.negf":
        return (
            len(body) == 2
            and body[0] is op
            and list(op.operands) == [arg]
            and body[1].name == "linalg.yield"
            and list(body[1].operands) == list(op.results)
        )
    if len(body) != 4:
        return False
    literal, compare, select, yielded = body
    value = literal.properties.get("value")
    import math

    return (
        literal.name == "arith.constant"
        and isinstance(value, FloatAttr)
        and value.value.data == 0.0
        and math.copysign(1.0, value.value.data) == 1.0
        and compare is op
        and compare.predicate.value.data == 4
        and list(compare.operands) == [arg, literal.results[0]]
        and select.name == "arith.select"
        and list(select.operands) == [compare.results[0], literal.results[0], arg]
        and yielded.name == "linalg.yield"
        and list(yielded.operands) == list(select.results)
    )


def dtype_domain(dtype):
    """Return the complete storage bound and (where applicable) finite grid."""
    if dtype.startswith("i") and dtype[1:].isdigit():
        width = int(dtype[1:])
        return dict(dtype=dtype, absolute_bound=1 if width == 1 else 1 << (width - 1), finite_grid=None)
    real = "f32" if dtype == "complex<f32>" else "f64" if dtype == "complex<f64>" else dtype
    if real not in FORMAT:
        raise ValueError(f"no complete native storage domain for {dtype}")
    fraction, exponent, bias = FORMAT[real]
    minimum = 1 - bias - fraction
    maximum = ((1 << (fraction + 1)) - 1) << (((1 << exponent) - 2 - bias - fraction) - minimum)
    return dict(dtype=dtype, absolute_bound=None, finite_grid=(minimum, maximum), ieee_special_values=True)


def _storage_component_semantics(module):
    """Prove closed storage/ring primitives, or state the missing theorem.

    Integer add/sub/multiply and signed extensions are equations in Z/(2**w).
    Their promotions and truncations remain explicit typed source boundaries.
    IEEE sign changes and ordered zero-threshold selections at native f32/f64
    precision preserve every input word, including signaling NaN payloads. Reduced
    precision CPU kernels can use different scalar and vector NaN conversions;
    their source arithmetic is deliberately not licensed by this theorem.

    No eager observations, tensor payloads or provenance names enter this decision.
    """
    from xdsl.dialects.builtin import IntegerType, TensorType

    floating = []
    integer_widths = set()
    structural = {
        "builtin.module",
        "func.func",
        "func.return",
        "linalg.generic",
        "linalg.yield",
        "linalg.matmul",
        "linalg.batch_matmul",
        "linalg.reduce",
        "linalg.index",
        "linalg.fill",
        "linalg.copy",
        "linalg.broadcast",
        "linalg.transpose",
        "tensor.empty",
        "tensor.splat",
        "tensor.cast",
        "tensor.reshape",
        "tensor.dim",
        "tensor.extract",
        "tensor.from_elements",
        "tensor.extract_slice",
        "tensor.insert_slice",
        "tensor.concat",
        "tensor.collapse_shape",
        "tensor.expand_shape",
        "tensor.pad",
        "tensor.yield",
        "scf.for",
        "scf.if",
        "scf.yield",
        "affine.apply",
    }
    scalar = {
        "arith.constant",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.extsi",
        "arith.extui",
        "arith.trunci",
        "arith.index_cast",
        "arith.index_castui",
        "arith.andi",
        "arith.ori",
        "arith.xori",
        "arith.cmpi",
        "arith.select",
        "arith.bitcast",
        "arith.negf",
        "arith.cmpf",
    }
    for op in module.walk():
        for value in (*op.operands, *op.results):
            typ = value.type.element_type if isinstance(value.type, TensorType) else value.type
            if isinstance(typ, IntegerType):
                integer_widths.add(typ.width.data)
        for key in ("fastmath", "overflowFlags"):
            flag = op.properties.get(key, op.attributes.get(key))
            if flag is not None and str(flag).split("<")[-1] != "none>":
                return dict(
                    status="refused",
                    code="native_relaxed_arithmetic_unproved",
                    reason="native semantic proof requires unflagged arithmetic",
                )
        if op.name in {"func.call", "func.call_indirect"} or op.name.startswith("math."):
            return dict(
                status="refused",
                code="native_external_semantics_unproved",
                reason="native component family has no general theorem for calls or math intrinsics",
            )
        # A source-level typed conversion is observable (promotion, NaN payload,
        # overflow and intermediate rounding); a finite sample cannot close it.
        if op.name in {"arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui", "arith.extf", "arith.truncf"}:
            return dict(
                status="refused",
                code="native_conversion_semantics_unproved",
                reason="typed floating conversion/promotion lacks a full-domain CPU semantic proof",
            )
        if op.name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf", "linalg.matmul", "linalg.batch_matmul"}:
            # Integer named contractions are handled by the modular theorem.
            if all(isinstance(getattr(v.type, "element_type", None), IntegerType) for v in op.operands):
                continue
            return dict(
                status="refused",
                code="native_ieee_arithmetic_semantics_unproved",
                reason=(
                    "floating arithmetic lacks a per-dtype proof of CPU rounding, signed zero, "
                    "overflow and NaN payloads"
                ),
            )
        if op.name in {"arith.negf", "arith.cmpf"}:
            types = {str(value.type) for value in op.operands}
            if not types <= {"f32", "f64"}:
                return dict(
                    status="refused",
                    code="native_reduced_precision_nan_semantics_unproved",
                    reason=(
                        "reduced-precision scalar/vector CPU NaN conversions lack a general source equivalence proof"
                    ),
                )
            if not _word_generic(op):
                return dict(
                    status="refused",
                    code="native_ieee_word_semantics_unproved",
                    reason="IEEE source is outside the closed native-width sign/positive-zero-threshold family",
                )
            floating.append(op.name)
        if op.name in {"arith.maximumf", "arith.minimumf", "arith.maxnumf", "arith.minnumf"}:
            return dict(
                status="refused",
                code="native_ieee_selection_semantics_unproved",
                reason="IEEE extrema require a signed-zero and NaN selection theorem",
            )
        if op.name in {
            "arith.divsi",
            "arith.divui",
            "arith.remsi",
            "arith.remui",
            "arith.floordivsi",
            "arith.ceildivsi",
        }:
            from merlin.llvmlower.integer_division_components import literal_divisor

            divisor = literal_divisor(op)
            if op.name in {"arith.divsi", "arith.remsi"} and divisor is not None and divisor > 0:
                # A positive source literal excludes both zero and MIN/-1.
                # The modular accumulator boundary preceding this quotient stays explicit.
                continue
            return dict(
                status="refused",
                code="native_integer_division_semantics_unproved",
                reason="integer division requires full-domain zero-divisor and signed-overflow semantics",
            )
        if op.name not in structural | scalar:
            return dict(
                status="refused",
                code="native_primitive_semantics_unproved",
                reason=f"no general native semantic theorem for primitive {op.name}",
            )
    return dict(
        status="proven",
        method="typed_storage_and_modular_ring_v1",
        integer_widths=sorted(integer_widths),
        integer_arithmetic="modulo_each_source_storage_width",
        promotion="explicit_source_extension_and_truncation_boundaries",
        floating_primitives=sorted(set(floating)),
        ieee_scope="native_width_sign_and_retained_ordered_compare_select",
        input_domain="all_storage_bit_patterns",
        runtime_input_observations=False,
    )


def component_semantics(module):
    proof = _storage_component_semantics(module)
    if proof["status"] == "proven":
        return proof
    numerical = {
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "math.fma",
        "arith.extf",
        "arith.truncf",
        "arith.sitofp",
        "arith.uitofp",
    }
    if not any(op.name in numerical for op in module.walk()) and proof["code"] in {
        "native_ieee_word_semantics_unproved",
        "native_reduced_precision_nan_semantics_unproved",
    }:
        if not all(_word_generic(op) for op in module.walk() if op.name in {"arith.negf", "arith.cmpf"}):
            return dict(
                status="refused",
                code="native_ieee_word_semantics_unproved",
                reason="word-only source is outside the closed sign/positive-zero-threshold wiring theorem",
            )
    from merlin.targetgen.native_typed_semantics import prove

    return prove(module)

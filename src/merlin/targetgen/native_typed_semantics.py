"""Composition theorem for closed typed pointwise storage/IEEE programs.

The CPU profile is certified per dtype/form, never against a captured example.
All typed RNE boundaries and explicit path/NaN selectors stay in the program.
Reductions need a different theorem about order and competing NaN payloads.
"""


def prove(module):
    import math

    from merlin.llvmlower._ir import input_literal as literal
    from merlin.targetgen.native_ieee_certificate import certificate
    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    try:
        proof = certificate()
    except (OSError, KeyError, ValueError) as exc:
        return dict(status="refused", code="native_cpu_certificate_unavailable", reason=str(exc))
    # Literal-domain simplification is a proof step. Preserve the emitted
    # source and its original IEEE boundaries; no input observations are used.
    from merlin.llvmlower.uniform_numeric_literals import fold_uniform_numeric_literals

    storage_inputs = {
        str(arg.type.element_type)
        for function in module.walk()
        if function.name == "func.func" and function.sym_name.data == "forward"
        for arg in function.body.block.args
    }
    if storage_inputs & {"f16", "bf16"}:
        module = module.clone()
        fold_uniform_numeric_literals(module, close_guards=True)
    arithmetic = {"arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.negf", "math.fma"}
    conversions = {"arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp"}
    scalar = (
        arithmetic
        | conversions
        | {
            "arith.constant",
            "arith.addi",
            "arith.subi",
            "arith.muli",
            "arith.extsi",
            "arith.extui",
            "arith.trunci",
            "arith.index_cast",
            "arith.index_castui",
            "arith.bitcast",
            "arith.cmpi",
            "arith.cmpf",
            "arith.select",
            "arith.andi",
            "arith.ori",
            "arith.xori",
            "arith.shli",
            "arith.shrsi",
            "arith.shrui",
            "arith.minui",
            "arith.maxui",
            "arith.minsi",
            "arith.maxsi",
        }
    )
    structural = {
        "builtin.module",
        "func.func",
        "func.return",
        "linalg.generic",
        "linalg.yield",
        "linalg.index",
        "linalg.copy",
        "linalg.broadcast",
        "linalg.transpose",
        "linalg.fill",
        "tensor.empty",
        "tensor.splat",
        "tensor.cast",
        "tensor.reshape",
        "tensor.dim",
        "tensor.extract",
        "tensor.extract_slice",
        "tensor.insert_slice",
        "tensor.concat",
        "tensor.collapse_shape",
        "tensor.expand_shape",
        "tensor.pad",
        "tensor.yield",
        "affine.apply",
    }
    divisions = {"arith.divsi", "arith.floordivsi", "arith.remsi", "arith.divui", "arith.remui"}
    primitives = set()
    reduced_profile = False
    integer_floor_wrap_widths = set()
    functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward"]
    storage = {str(arg.type.element_type) for function in functions for arg in function.body.block.args}
    if storage & {"f64", "complex<f64>"}:
        return dict(
            status="refused",
            code="native_binary64_arithmetic_form_unproved",
            reason=(
                "binary64 arithmetic/conversion forms are outside this CPU certificate; "
                "storage-only proofs remain separate"
            ),
        )
    for function in functions:
        for value in function.body.block.last_op.operands:
            if (
                str(getattr(value.type, "element_type", "")) == "i1"
                and value not in function.body.block.args
                and _constant_of(value) is None
            ):
                return dict(
                    status="refused",
                    code="native_numeric_predicate_family_unproved",
                    reason="a conversion prelude cannot place an unproved numerical Boolean readout",
                )
    for check in proof.get("arithmetic_checks", []):
        if check["form"]["dtype"] in storage and check["status"] != "proven":
            return dict(
                status="refused",
                code="native_cpu_arithmetic_certificate_refused",
                reason="CPU per-dtype arithmetic form has a full-domain counterexample",
                form=check["form"],
                counterexample=check["counterexample"],
            )

    def has_nan_guard(value):
        """Require the complete explicit invalid/default + quieted-input DAG.

        Every unprotected arithmetic-word escape refuses. Index selects after
        the complete guard may choose different proved operand priorities.
        """
        from xdsl.dialects.builtin import IntegerAttr

        from merlin.llvmlower.fixed_point_ir import FORMAT

        primitive = value.owner
        block = primitive.parent_block()
        fraction, exponent, _ = FORMAT[str(value.type)]
        quiet = 1 << (fraction - 1)
        default = ((1 << (fraction + exponent + 1)) - 1) ^ ((1 << (fraction - 1)) - 1)
        width_mask = (1 << (fraction + exponent + 1)) - 1
        operands = set(primitive.operands)
        required = {v for v in operands if literal(v) is None or math.isnan(literal(v))}

        def constant_bits(word):
            cast = word.owner
            if getattr(cast, "name", None) != "arith.bitcast":
                return None
            owner = cast.operands[0].owner
            if getattr(owner, "name", None) != "arith.constant":
                return None
            attr = owner.properties.get("value")
            return (attr.value.data & width_mask) if isinstance(attr, IntegerAttr) else None

        def nan_select(op):
            if op.name != "arith.select":
                return None
            condition = op.operands[0].owner
            if (
                getattr(condition, "name", None) != "arith.cmpf"
                or condition.operands[0] is not condition.operands[1]
                or condition.predicate.value.data not in {7, 14}
            ):
                return None
            yes, no = op.operands[1:]
            if condition.predicate.value.data == 7:
                yes, no = no, yes
            return condition.operands[0], yes, no

        def quiet_word(word, original):
            cast = word.owner
            if getattr(cast, "name", None) != "arith.bitcast":
                return False
            union = cast.operands[0].owner
            if getattr(union, "name", None) != "arith.ori":
                return False
            for raw, mask in (tuple(union.operands), tuple(reversed(union.operands))):
                bitcast = raw.owner
                constant = mask.owner
                if (
                    getattr(bitcast, "name", None) == "arith.bitcast"
                    and bitcast.operands[0] is original
                    and getattr(constant, "name", None) == "arith.constant"
                    and isinstance(constant.properties.get("value"), IntegerAttr)
                    and constant.properties["value"].value.data == quiet
                ):
                    return True
            return False

        states = {}
        for op in block.ops:
            choice = nan_select(op)
            if choice is None:
                continue
            tested, yes, no = choice
            if tested is value and no is value and constant_bits(yes) == default:
                states[op.results[0]] = set()
            elif no in states and tested in operands and quiet_word(yes, tested):
                states[op.results[0]] = states[no] | {tested}
        memo = {}

        def protected(word):
            if word in memo:
                return memo[word]
            if word in states and states[word] >= required:
                return True
            memo[word] = False
            if not word.uses:
                return True
            for use in word.uses:
                consumer = use.operation
                if (
                    consumer.name == "arith.cmpf"
                    and consumer.operands[0] is word
                    and consumer.operands[1] is word
                    and consumer.predicate.value.data in {7, 14}
                ):
                    continue
                if (
                    consumer.name != "arith.select"
                    or consumer.results[0] not in states
                    or not protected(consumer.results[0])
                ):
                    return False
            memo[word] = True
            return True

        return bool(states) and protected(value)

    for op in module.walk():
        if op.name in arithmetic and any(str(v.type) == "f64" for v in (*op.operands, *op.results)):
            return dict(
                status="refused",
                code="native_binary64_arithmetic_form_unproved",
                reason="exact binary32 widening does not certify binary64 arithmetic",
            )
        if op.name == "arith.cmpf":
            from merlin.targetgen.native_dtype_domain import _word_generic

            nan_classification = op.operands[0] is op.operands[1] and op.predicate.value.data in {7, 14}
            if not nan_classification and not _word_generic(op):
                return dict(
                    status="refused",
                    code="native_float_predicate_family_unproved",
                    reason="floating predicate is outside NaN readout and the closed device word-selection family",
                )
        if op.name == "linalg.generic" and any(it.data.value != "parallel" for it in op.get_iterator_types()):
            return dict(
                status="refused",
                code="native_aggregate_nan_order_unproved",
                reason="aggregate CPU order/NaN payload selection lacks a full-domain theorem",
            )
        if op.name in {"arith.mulf", "math.fma"}:
            from merlin.llvmlower.uniform_numeric_literals import compile_time_operand

            if not any(compile_time_operand(v) for v in op.operands[:2]):
                return dict(
                    status="refused",
                    code="native_parallel_runtime_product_construction",
                    reason="parallel products require a compile-time operand, including fused products",
                )
        if op.name in divisions:
            if str(op.results[0].type) == "index":
                continue
            from merlin.llvmlower.integer_division_components import cpu_floor_wrap, literal_divisor

            divisor = literal_divisor(op)
            width = op.results[0].type.width.data
            wrapping_floor = op.name == "arith.floordivsi" and cpu_floor_wrap(proof, width)
            if divisor == -1 and wrapping_floor:
                integer_floor_wrap_widths.add(width)
            if divisor is None or divisor == 0 or (divisor == -1 and width >= 32 and not wrapping_floor):
                return dict(
                    status="refused",
                    code="native_integer_division_exception_domain",
                    reason="full dtype domain includes divisor-zero or INT_MIN/-1 without a defined CPU value theorem",
                )
            if divisor == -1 and op.results[0].type.width.data == 8:
                rows = proof.get("integer_checks", {}).get("quotients", [])
                if not rows or any(row.get("overflow_domain") != "value" for row in rows):
                    return dict(
                        status="refused",
                        code="native_integer_division_exception_domain",
                        reason="signed-byte overflow has no checked CPU value theorem",
                    )
        elif op.name not in scalar | structural:
            return dict(
                status="refused",
                code="native_primitive_semantics_unproved",
                reason="no certified typed pointwise theorem for " + op.name,
            )
        for key in ("fastmath", "overflowFlags"):
            flag = op.properties.get(key, op.attributes.get(key))
            if flag is not None and str(flag).split("<")[-1] != "none>":
                return dict(
                    status="refused",
                    code="native_relaxed_arithmetic_unproved",
                    reason="typed RNE theorem requires unflagged arithmetic",
                )
        if op.name in arithmetic | conversions:
            primitives.add(op.name)
        if op.name == "arith.divf":
            divisor = literal(op.operands[1])
            if divisor is None:
                return dict(
                    status="refused",
                    code="native_runtime_float_division_unproved",
                    reason="runtime floating division requires a full-domain quotient and zero-divisor producer",
                )
            if not math.isfinite(divisor):
                return dict(
                    status="refused",
                    code="native_nonfinite_literal_division_unproved",
                    reason="nonfinite literal division has no certified complete integer producer",
                )
            if divisor and math.isfinite(divisor):
                from fractions import Fraction

                ratio = Fraction(float(divisor))
                from merlin.llvmlower.ieee_dyadic_ir import finite_dtype_grid

                quantum, bound = finite_dtype_grid(str(op.results[0].type))
                reciprocal = abs(1 / ratio) / Fraction(2) ** quantum
                if (
                    abs(ratio.numerator) & (abs(ratio.numerator) - 1)
                    or reciprocal.denominator != 1
                    or reciprocal > bound
                ):
                    return dict(
                        status="refused",
                        code="native_reciprocal_opmath_unproved",
                        reason="CPU scalar reciprocal is not an exact finite value of the compute dtype",
                    )
        if op.name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf", "math.fma"}:
            if not has_nan_guard(op.results[0]):
                runtime = {v for v in op.operands if literal(v) is None}
                constants = [literal(v) for v in op.operands if literal(v) is not None]
                if any(math.isnan(v) for v in constants) or (len(runtime) > 1 and storage != {"bf16"}):
                    return dict(
                        status="refused",
                        code="native_competing_nan_priority_unproved",
                        reason="unguarded competing NaN words depend on CPU instruction/loop order",
                    )
                if "bf16" in storage:
                    reduced_profile = True
        if op.name in {"arith.extf", "arith.truncf"}:
            if not (
                op.name == "arith.extf" and str(op.operands[0].type) == "f32" and str(op.results[0].type) == "f64"
            ) and any(str(v.type) not in {"f16", "bf16", "f32"} for v in (*op.operands, *op.results)):
                return dict(
                    status="refused",
                    code="native_conversion_form_unproved",
                    reason="floating conversion form is outside the checked reduced/binary32 storage domains",
                )
    return dict(
        status="proven",
        method="closed_typed_pointwise_IEEE_composition_v1",
        certificate_sha256=proof["content_sha256"],
        conversion_forms=[row["key"] for row in proof["conversions"]],
        input_domain="all_storage_bit_patterns",
        runtime_input_observations=False,
        finite_arithmetic="one exact integer producer and RNE at every original typed boundary",
        primitives=sorted(primitives),
        ieee_scope="explicit_source_NaN_and_index_selectors_preserved",
        integer_arithmetic="modular; certified literal quotient correction; typed promotion retained",
        integer_floor_wrap_widths=sorted(integer_floor_wrap_widths),
        cpu_reduced_storage_profile_required=reduced_profile,
        cpu_loop_profile=proof["loop_profile"],
    )

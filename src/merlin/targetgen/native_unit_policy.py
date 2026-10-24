"""Structural native-unit eligibility, independent of source overload names.

Representation splitting and exact host readout do not change a producer's
mathematical role. Parallel runtime products, nonlinear rounding, and predicate
identities must not become native units merely because component lowering can
express them with contractions. Index arithmetic remains address selection.
"""


def construction_evidence(module, *, excluded=None, result_abi=None):
    from merlin.llvmlower import image_folds as image
    from merlin.llvmlower import structural_folds as structural
    from merlin.targetgen.native_construction_patterns import Payloads, dataflow

    # Classify the canonical arithmetic role, without mutating source bytes.
    # A one-contribution monoid identity is storage movement, not a predicate
    # implementation. The emitted movement path repeats this exact proof.
    candidates = any(
        (op.name == "arith.cmpf" and op.predicate.value.data in {1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13})
        or getattr(getattr(op, "op_name", None), "data", op.name)
        in {"math.floor", "math.ceil", "math.trunc", "math.round", "math.roundeven"}
        or op.name == "linalg.index"
        or op.name == "arith.cmpi"
        or op.name == "linalg.reduce"
        or (op.name == "linalg.generic" and any(str(v.type.element_type) == "i1" for v in op.inputs))
        or (
            op.name == "linalg.generic"
            and (
                any(hasattr(v.type, "get_shape") and 0 in v.type.get_shape() for v in op.operands)
                or any(it.data.value == "reduction" for it in op.get_iterator_types())
            )
        )
        for op in module.walk()
    )
    if candidates:
        canonical = module.clone()
        order = (
            structural.fold_static_unit_indices,
            image.fold_integer_image_predicates,
            image.fold_integral_image_identities,
            image.fold_ieee_identity_selectors,
            structural.normalize_singleton_boundaries,
            image.fold_nan_guarded_neutral,
            image.fold_boolean_unit_image,
            image.fold_boolean_image_predicates,
            image.fold_boolean_domain,
            structural.fold_empty_reductions,
            lambda m: (
                structural.fold_singleton_reductions(m) if any(op.name == "linalg.reduce" for op in m.walk()) else 0
            ),
            structural.fold_singleton_selectors,
            structural.fold_singleton_carriers,
        )
        if sum(fold(canonical) for fold in order):
            from merlin.llvmlower.entry_weight_projection import _pure_dce

            for function in canonical.walk():
                if function.name == "func.func":
                    _pure_dce(function)
            module = canonical
    evidence = []
    payloads = Payloads(module)

    literal_memo = {}

    def literal(value):
        return dataflow(value, literal_memo, literal_dependencies, all)

    def literal_dependencies(value):
        owner = value.owner
        name = getattr(owner, "name", None)
        if name in {"arith.constant", "linalg.index"}:
            return True
        if name == "linalg.generic":
            block = owner.body.block
            return (block.last_op.operands[value.index],)
        if name in {
            "tensor.splat",
            "arith.extf",
            "arith.truncf",
            "arith.sitofp",
            "arith.uitofp",
            "arith.extsi",
            "arith.extui",
            "arith.trunci",
            "tensor.reshape",
            "tensor.cast",
        }:
            return (owner.operands[0],)
        if name is not None and (name.startswith("arith.") or name.startswith("math.")):
            return tuple(owner.operands) if owner.operands else False
        if name is None and hasattr(owner, "parent_op"):
            parent = owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
                return (parent.inputs[value.index],)
        return False

    index_memo = {}

    def index_payload(value):
        return dataflow(value, index_memo, index_dependencies)

    def index_dependencies(value):
        owner = value.owner
        name = getattr(owner, "name", None)
        if name == "linalg.index":
            return True
        if name is None and hasattr(owner, "parent_op"):
            parent = owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
                return (parent.inputs[value.index],)
            return False
        if name in {"tensor.extract", "arith.constant"}:
            return False
        if name == "linalg.generic":
            return (owner.body.block.last_op.operands[value.index],)
        if name == "scf.if":
            return tuple(region.block.last_op.operands[value.index] for region in owner.regions)
        # Selection predicates are addresses; selected values are payloads.
        operands = owner.operands[1:] if name == "arith.select" else owner.operands
        return tuple(operands)

    from merlin.targetgen.native_payload_carriers import TABLE, UNKNOWN, LiteralPayloads

    literal_payloads = LiteralPayloads()

    logical_literals = []
    for ordinal, op in enumerate(module.walk()):
        if op.name == "func.return":
            masks = [literal_payloads.classify(v) for v in op.operands]
            from merlin.targetgen.logical_literal_results import uniform_logical_literal

            for index, value in enumerate(op.operands):
                abi = (result_abi or {}).get(index, {})
                if masks[index] & TABLE and uniform_logical_literal(value, abi):
                    masks[index] &= ~TABLE
                    logical_literals.append(
                        dict(
                            result_index=index,
                            dtype=abi["dtype"],
                            shape=abi["shape"],
                            proof="every declared complex storage word is identical; closed literal movement only",
                        )
                    )
            for mask, reason in (
                (TABLE, "nonuniform_literal_result_construction"),
                (UNKNOWN, "unproved_payload_carrier"),
            ):
                if any(value & mask for value in masks):
                    evidence.append(
                        dict(
                            ordinal=ordinal,
                            operation=op.name,
                            reason=reason,
                            **({"carriers": sorted(literal_payloads.unknown_operators)} if mask == UNKNOWN else {}),
                        )
                    )
        for reason in payloads.patterns(op):
            evidence.append(dict(ordinal=ordinal, operation=op.name, reason=reason))
        if op.name == "linalg.generic":
            yielded = op.body.block.last_op.operands
            if any(str(v.type) not in {"i1", "index"} and index_payload(v) for v in yielded):
                evidence.append(dict(ordinal=ordinal, operation=op.name, reason="index_generated_payload_arithmetic"))
        if op.name in {"math.floor", "math.ceil", "math.trunc", "math.round", "math.roundeven"}:
            if not literal(op.operands[0]):
                evidence.append(dict(ordinal=ordinal, operation=op.name, reason="nonlinear_rounding_construction"))
        parent = op.parent_op()
        boolean_product = (
            op.name == "arith.andi"
            and str(op.results[0].type) == "i1"
            and parent is not None
            and parent.name == "linalg.generic"
            and any(str(v.type.element_type) == "i1" for v in parent.results)
            and payloads.readout(op.results[0])
        )
        if payloads.forced_special_division(op):
            # x / 0 and x mod 0 with a closed zero divisor make every result an
            # IEEE special value (inf or NaN) whatever the data: outside the
            # quantized contract and not native arithmetic.
            evidence.append(dict(ordinal=ordinal, operation=op.name, reason="forced_ieee_special_division"))
        if op.name in {"arith.mulf", "arith.muli"} and payloads.product_reduction(op):
            # acc * x folded over a reduced extent is a chain of runtime
            # products, not a contraction (which adds products into acc).
            evidence.append(dict(ordinal=ordinal, operation=op.name, reason="runtime_product_reduction"))
        if op.name in {"arith.mulf", "arith.muli"} and payloads.literal_coefficient_table(op):
            # A contraction against a closed table of several distinct nonzero
            # weights emulates a tabulated linear transform (DFT twiddles,
            # interpolation weights). Selections, sums and one-scale tables
            # (values in {0, c}) remain ordinary constant operands.
            evidence.append(dict(ordinal=ordinal, operation=op.name, reason="nonuniform_literal_coefficient_contraction"))
        if op.name not in {"arith.mulf", "arith.muli"} and not boolean_product:
            continue
        if any(str(value.type) == "index" for value in op.operands) or any(literal(v) for v in op.operands):
            continue
        if payloads.reduction_product(op):
            continue
        parent = op.parent_op()
        while parent is not None and parent.name != "linalg.generic":
            parent = parent.parent_op()
        if parent is None or any(it.data.value != "parallel" for it in parent.get_iterator_types()):
            continue
        # Only a producer-dependent epilogue retains a contraction role.
        # An unrelated reduction elsewhere in the module grants no exemption.
        if not any(payloads.contraction_dependent(v) for v in op.operands):
            evidence.append(dict(ordinal=ordinal, operation=op.name, reason="parallel_runtime_product_construction"))
    selected = (
        evidence
        if excluded is None
        else [
            entry for entry in evidence if entry["reason"] in excluded or entry["reason"] == "unproved_payload_carrier"
        ]
    )
    return dict(
        schema="merlin.native_unit_policy.v1",
        status="host" if selected else "eligible",
        evidence=selected,
        classified_constructions=evidence,
        **({"logical_literal_results": logical_literals} if logical_literals else {}),
    )

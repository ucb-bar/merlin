"""Index-preserving special propagation beside exact finite integer producers.

The closed family is sums of scalar products and gathers, with finite literal
divisors. Finite terms are discarded in the propagation program. Consequently
it performs no finite floating arithmetic and cannot replace device work.
"""


def propagation_source(op, source_grid):
    """Return a checked special-only clone and a dependence proof, or refuse.

    Addition preserves a special, including opposite infinities (invalid NaN).
    A product of source leaves is special whenever either leaf is special,
    including zero times infinity. No sum may feed a product: its discarded
    finite value would then be needed to determine the product's sign/zero.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower.additive_regions import positive_zero_seed
    from merlin.llvmlower.fixed_point_ir import FORMAT
    from merlin.llvmlower.ieee_pointwise import special, special_arithmetic, special_conversion
    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    if op.name != "linalg.generic" and all(
        hasattr(op, name) for name in ("body", "inputs", "outputs", "get_indexing_maps", "get_iterator_types")
    ):
        from xdsl.dialects.builtin import ArrayAttr
        from xdsl.dialects.linalg import ops as L

        cloned = op.clone()
        generic = L.GenericOp(
            inputs=list(op.inputs),
            outputs=list(op.outputs),
            body=cloned.detach_region(0),
            indexing_maps=ArrayAttr(op.get_indexing_maps()),
            iterator_types=ArrayAttr(op.get_iterator_types()),
            result_types=[v.type for v in op.results],
        )
        generic.attributes.update(op.attributes)
        return propagation_source(generic, source_grid)
    if op.name != "linalg.generic" or len(op.results) != 1:
        raise ValueError("aggregate propagation requires one explicit generic result")
    if op.body.block.args[-1].uses and not positive_zero_seed(op.outputs[0]):
        raise ValueError("aggregate propagation requires a positive zero seed")
    inputs = list(op.body.block.args[: len(op.inputs)])

    def leaf(value):
        owner = value.owner
        if value in inputs or getattr(owner, "name", None) in {"tensor.extract", "arith.constant"}:
            return True
        return getattr(owner, "name", None) == "arith.extf" and leaf(owner.operands[0])

    data_allowed = {
        "arith.constant",
        "tensor.extract",
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.extf",
        "scf.for",
        "scf.if",
        "scf.yield",
        "linalg.yield",
    }
    arithmetic = []
    for child in op.body.walk():
        if getattr(child.properties.get("fastmath"), "data", frozenset()):
            raise ValueError("aggregate propagation refuses floating fastmath flags")
        if any(str(v.type) in FORMAT for v in (*child.operands, *child.results)) and child.name not in data_allowed:
            raise ValueError("unproved aggregate special data operation: " + child.name)
        if child.name == "arith.mulf" and not all(leaf(v) for v in child.operands):
            raise ValueError("aggregate product requires original scalar leaves")
        if child.name == "arith.divf":
            value = child.operands[1]
            if value in inputs:
                value = op.inputs[inputs.index(value)]
            literal = _constant_of(value)
            from math import isfinite

            if literal is None or not isfinite(float(literal)) or abs(literal) < 1:
                raise ValueError("aggregate special divisor must be a finite literal of magnitude at least one")
        if child.name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf"}:
            arithmetic.append(child)
    if not arithmetic:
        raise ValueError("aggregate propagation has no source data arithmetic")
    # Static maps and gather controls are checked independently by the same
    # closed index interpreter used for component producer qualification.
    from merlin.llvmlower.static_polynomials import analyze_static_polynomial

    statistics = {}
    analyze_static_polynomial(op, require_bilinear=False, statistics=statistics)
    from fractions import Fraction

    from merlin.llvmlower.additive_regions import data_sources

    grids = [source_grid(v) for v in data_sources(op, lambda t: str(t) in FORMAT)]
    if not grids or any(g is None for g in grids):
        raise ValueError("aggregate propagation lacks finite operand range proofs")
    magnitude = max(Fraction(b) * Fraction(2) ** e for e, b in grids)
    formats = []
    for child in op.body.walk():
        if child.name == "arith.constant" and str(child.results[0].type) in FORMAT:
            magnitude = max(magnitude, abs(Fraction.from_float(float(child.value.value.data))))
        if child.name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf"}:
            formats.append(FORMAT[str(child.results[0].type)])
    count = statistics["floating_work"] + 1
    precision = min(f for f, _, _ in formats)
    # N*u <= 1/4 bounds accumulated relative rounding by a geometric
    # series <2. Add the absolute subnormal rounding envelope as well.
    # Products contain only leaves; sums cannot be amplified by products
    # or divisors of magnitude below one.
    if count > 1 << (precision - 2):
        raise ValueError("aggregate IEEE intermediate rounding envelope exceeds its proof budget")
    tiny = max(Fraction(2) ** (1 - bias - f) for f, _, bias in formats)
    envelope = 2 * count * (max(magnitude, magnitude * magnitude) + tiny)
    maximum = min(Fraction((1 << (f + 1)) - 1) * Fraction(2) ** (bias - f) for f, _, bias in formats)
    if envelope > maximum:
        raise ValueError("aggregate finite terms can overflow; special propagation depends on host instruction order")
    clone = op.clone()
    for child in list(clone.body.walk()):
        if child.name not in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.extf"}:
            continue
        operations = []

        def emit(operation):
            operations.append(operation)
            return operation.results[0]

        if child.name == "arith.extf":
            value = special_conversion(child.name, child.operands[0], child.results[0].type, emit)
        else:
            flags = [special(v, emit) for v in child.operands]
            affected = emit(arith.OrIOp(*flags))
            propagated = special_arithmetic(child.name, list(child.operands), emit)
            zero = emit(arith.ConstantOp(FloatAttr(0.0, child.results[0].type)))
            value = emit(arith.SelectOp(affected, propagated, zero))
        Rewriter.replace_op(child, operations, [value])
    proof = dict(
        method="closed_source_index_maps_and_static_gather_control",
        source_index_maps=[str(m.data) for m in op.get_indexing_maps()],
        source_scalar_region=str(op),
        affected_predicate="OR(is_nonfinite(each_live_source_read_at_its_original_index))",
        propagation="special_only_sum_of_leaf_products_with_finite_literal_divisors",
        finite_intermediate_proof=dict(
            source_arithmetic_count=statistics["floating_work"],
            absolute_envelope=str(envelope),
            finite_limit=str(maximum),
        ),
        finite_dependence_outputs="retain_exact_device_integer_readout_without_overlay",
    )
    return clone, proof


def aggregate_overlay(propagation, fallback, operations):
    """Select special words only; every finite output retains its device result."""
    from xdsl.dialects import arith

    from merlin.llvmlower._ir import point
    from merlin.llvmlower.ieee_pointwise import special

    operations.append(propagation)

    def select(args, emit):
        return emit(arith.SelectOp(special(args[0], emit), args[0], args[1]))

    return point([propagation.results[0], fallback], fallback.type, select, operations)

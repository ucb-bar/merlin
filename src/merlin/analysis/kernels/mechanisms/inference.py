"""Bounded mechanism questions from source domains and normalized storage effects.

Packing arithmetic describes conditional alternatives, not a legal rewrite or
the schedule a binary executed. Repeated reads retain the selected projector's
backing/alias authority; no opcode spelling supplies tensor or effect semantics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import prod

from xdsl.dialects.builtin import IntegerAttr, IntegerType, TensorType
from xdsl.ir import Operation
from xdsl.ir.affine import AffineDimExpr

# ---- mechanism_inference -----------------------------------------------------------------------


@dataclass(frozen=True)
class ReductionLanes:
    """A selected adapter's physical reduction width and its exact evidence pointer."""

    width: int
    evidence: str

    def __post_init__(self):
        if type(self.width) is not int or self.width < 1 or not self.evidence:
            raise ValueError("reduction lanes require a positive width and selected evidence")


def packing_profiles(reduction, *, lanes=None):
    """Price contiguous suffix groupings without asserting their index/numeric legality."""
    if (
        not reduction
        or len(reduction) > 32
        or any(type(n) is not int or n < 1 for n in reduction)
        or sum(n.bit_length() for n in reduction) > 256
    ):
        raise ValueError("packing analysis requires bounded positive reduction extents")
    useful = prod(reduction)
    profiles = []
    for suffix in range(1, len(reduction) + 1):
        groups, elements = prod(reduction[:-suffix]), prod(reduction[-suffix:])
        pieces = groups * (-(-elements // lanes.width)) if lanes else None
        physical = pieces * lanes.width if lanes else None
        profiles.append(
            dict(
                collapsed_suffix_axes=suffix,
                independent_groups=groups,
                elements_per_group=elements,
                pieces=pieces,
                physical_extent=physical,
                unused_lanes=None if physical is None else physical - useful,
                status="conditional_arithmetic" if lanes else "unknown_physical_width",
                legality="unknown",
                emitted_application="unknown",
            )
        )
    return dict(
        useful_reduction_extent=useful,
        reduction_extents=list(reduction),
        lanes=None if lanes is None else dict(width=lanes.width, evidence=lanes.evidence),
        alternatives=profiles,
        required_proof=[
            "indexing and boundary padding under the new grouping",
            "accumulation, rounding, saturation and overflow order",
            "selected consumer and actual emitted reduction partition",
        ],
        performance="unknown",
    )


def source_mechanisms(module, *, evidence_id, lanes=None, max_operations=4096, max_witnesses=16):
    """Read actual yielded MACs and exact tensor SSA sharing, preserving unknown domains."""
    from xdsl.dialects.builtin import TensorType

    from merlin.analysis.programs.model_macs import observe_model_macs

    if max_operations < 1 or max_witnesses < 1:
        raise ValueError("mechanism analysis requires finite positive budgets")
    positions = {}
    for index, operation in enumerate(module.walk()):
        if index >= max_operations:
            return dict(
                status="unknown",
                reason="source_mechanism_operation_budget_exhausted",
                findings=[],
                optimization_application="unknown",
            )
        positions[operation] = index
    domains, uses, value_names = [], defaultdict(list), {}
    missing = []
    for operation, shape in observe_model_macs(module):
        if operation is None or shape.status != "derived":
            missing.append(dict(operation=None if operation is None else positions[operation], reason=shape.reason))
            continue
        witness = f"{evidence_id}#operation:{positions[operation]}"
        domains.append(
            dict(
                operation=positions[operation],
                source_op_index=shape.source_op_index,
                witness=witness,
                dtypes=list(shape.dtypes),
                parallel_extents=list(shape.parallel),
                packing=packing_profiles(shape.reduction, lanes=lanes),
            )
        )
        # Tensor SSA is immutable; equal shapes or matching addresses are not SSA equality.
        # Output/init operands are deliberately excluded from shared input evidence.
        for operand_index, value in enumerate(tuple(operation.inputs)):
            if not isinstance(value.type, TensorType):
                continue
            value_names.setdefault(value, f"tensor:{len(value_names)}")
            uses[value].append(dict(operation=positions[operation], operand=operand_index, witness=witness))
    shared = [
        dict(
            value=value_names[value],
            shape=list(value.type.get_shape()),
            dtype=str(value.type.get_element_type()),
            consumers=members[:max_witnesses],
            consumers_total=len(members),
            consumers_omitted=max(0, len(members) - max_witnesses),
            status="observed_immutable_ssa_sharing",
            storage_reuse="unknown",
            required_proof=[
                "source-to-selected-consumer binding",
                "load and preparation lifetime",
                "capacity and non-overwrite across all consumers",
            ],
        )
        for value, members in uses.items()
        if len({row["operation"] for row in members}) > 1
    ]

    return dict(
        status="partial" if missing else "observed",
        origin="parsed_source_ir",
        reductions=domains[:max_witnesses],
        reductions_total=len(domains),
        reductions_omitted=max(0, len(domains) - max_witnesses),
        shared_inputs=shared[:max_witnesses],
        shared_inputs_total=len(shared),
        shared_inputs_omitted=max(0, len(shared) - max_witnesses),
        unknowns=missing[:max_witnesses],
        unknowns_total=len(missing),
        product_views=source_product_views(module, evidence_id=evidence_id, max_operations=max_operations),
        optimization_application="unknown",
        numerical_equivalence="unknown",
        performance="unknown",
    )


def repeated_read_mechanisms(model, *, max_comparisons=100_000, max_witnesses=16):
    """Find invariant reads in observed affine recurrences without expanding them.

    All accesses to the same canonical backing participate in invalidation. A
    repeated external read is not evidence that the destination remains live.
    Completion, capacity, hidden effects and cross-buffer aliasing stay separate.
    """
    if max_comparisons < 1 or max_witnesses < 1:
        raise ValueError("reuse analysis requires finite positive budgets")
    stages = {stage.key: stage for stage in model.stages}
    findings, unknowns, comparisons = [], [], 0
    if any(not stage.accesses for stage in model.stages):
        unknowns.append("normalized_effect_roster_incomplete")
    for loop in model.loops:
        if loop.status != "inferred" or loop.extent < 2 or any(key not in stages for key in loop.body):
            unknowns.append("unsupported_or_unbound_recurrence")
            continue
        body = [stages[key] for key in loop.body]
        accesses = [(i, j, access) for i, stage in enumerate(body) for j, access in enumerate(stage.accesses)]
        for i, j, access in accesses:
            if access.mode != "read":
                continue
            if type(access.offset) is not int or access.extent is None or f"{i}:{j}" not in loop.strides:
                unknowns.append("read_extent_or_affine_recurrence_unresolved")
                continue
            if loop.strides[f"{i}:{j}"] != 0:
                continue
            conflict = False
            unresolved = not model.complete or model.truncated
            for wi, wj, write in accesses:
                comparisons += 1
                if comparisons > max_comparisons:
                    return dict(
                        status="unknown",
                        reason="reuse_comparison_budget_exhausted",
                        findings=[],
                        optimization_application="unknown",
                    )
                if (write.space, write.buffer) != (access.space, access.buffer):
                    continue
                if write.mode not in ("write", "read_write"):
                    continue
                if write.unit != access.unit or type(write.offset) is not int or write.extent is None:
                    unresolved = True
                    continue
                stride = loop.strides.get(f"{wi}:{wj}")
                if type(stride) is not int:
                    unresolved = True
                    continue
                # An envelope may overapproximate gaps between writes. Treat
                # overlap as a refusal, never as proof the source changed.
                end = write.offset + (loop.extent - 1) * stride
                lo, hi = min(write.offset, end), max(write.offset, end) + write.extent
                conflict |= max(lo, access.offset) < min(hi, access.offset + access.extent)
            status = (
                "unknown" if unresolved else "intervening_write_possible" if conflict else "observed_invariant_read"
            )
            findings.append(
                dict(
                    recurrence=loop.key,
                    stage=body[i].key,
                    access=j,
                    backing=dict(space=access.space, buffer=access.buffer, unit=access.unit),
                    offset=access.offset,
                    extent=access.extent,
                    repetitions=loop.extent,
                    total_read_units=access.extent * loop.extent,
                    repeated_read_units=access.extent * (loop.extent - 1),
                    status=status,
                    witness=f"schedule:{model.identity}#stages:{body[i].key}:access:{j}",
                    scope="selected normalized backing/effects; observed finite recurrence",
                    reuse_legality="unknown",
                    optimization_application="unknown",
                    required_proof=[
                        "complete effects and canonical alias mapping",
                        "destination storage lifetime",
                        "capacity with simultaneous live values",
                        "completion and dependency ordering",
                    ],
                )
            )
    return dict(
        status="partial" if unknowns or not model.complete else "observed",
        findings=findings[:max_witnesses],
        findings_total=len(findings),
        findings_omitted=max(0, len(findings) - max_witnesses),
        unknowns=unknowns[:max_witnesses],
        optimization_application="unknown",
        performance="unknown",
    )


# ---- source_product_views ----------------------------------------------------------------------
# Logical operand/readout coordinates from actual typed source SSA.
#
# This is a source-mechanism diagnostic, not a source-to-command/binary proof.
# No physical packing, numerical comparison or capture owner is selected here.


def _matrix(value):
    typ = value.type
    if not isinstance(typ, TensorType):
        raise ValueError("static_rank_two_tensor_required")
    shape = list(typ.get_shape())
    if len(shape) != 2 or any(type(n) is not int or not 0 < n < 1 << 63 for n in shape):
        raise ValueError("static_rank_two_tensor_required")
    return shape


def _numeric(op):
    """Refine the existing MAC matcher to a zero-initialized modular integer body."""
    from merlin.analysis.programs.plan_evidence import _source_has_multiply_accumulate

    if not _source_has_multiply_accumulate(op):
        raise ValueError("yielded_mac_recurrence_unproved")
    body = op.regions[0].block
    accumulator = body.args[2].type
    if not isinstance(accumulator, IntegerType) or not 0 < accumulator.width.data <= 128:
        raise ValueError("integer_accumulator_required")
    addition = body.last_op.operands[0].owner
    if addition.name != "arith.addi":
        raise ValueError("integer_additive_recurrence_required")
    other = next(value for value in addition.operands if value is not body.args[2])
    multiply = other.owner
    if multiply.name != "arith.muli":
        raise ValueError("integer_product_required")
    seen, extensions = {addition, multiply, body.last_op}, {}
    for arithmetic in (addition, multiply):
        if getattr(arithmetic.properties.get("overflowFlags"), "data", None):
            raise ValueError("overflow_flagged_arithmetic_requires_numeric_adapter")
    for value in multiply.operands:
        extension = "identity"
        if isinstance(value.owner, Operation) and value.owner.name in ("arith.extsi", "arith.extui"):
            seen.add(value.owner)
            extension = "signed" if value.owner.name == "arith.extsi" else "unsigned"
            if value.type != accumulator or not isinstance(value.owner.operands[0].type, IntegerType):
                raise ValueError("operand_extension_domain_unproved")
            value = value.owner.operands[0]
        if value not in body.args[:2] or value in extensions:
            raise ValueError("operand_extension_domain_unproved")
        if extension == "identity" and value.type != accumulator:
            raise ValueError("operand_extension_domain_unproved")
        extensions[value] = extension
    if set(body.ops) != seen or len(extensions) != 2:
        raise ValueError("additional_numeric_or_effectful_body_requires_adapter")
    init = tuple(op.outputs)[0].owner
    if not isinstance(init, Operation) or init.name != "linalg.fill" or len(tuple(init.inputs)) != 1:
        raise ValueError("zero_initialization_unproved")
    scalar = tuple(init.inputs)[0]
    constant = scalar.owner
    value = constant.properties.get("value") if isinstance(constant, Operation) else None
    if (
        not isinstance(constant, Operation)
        or constant.name != "arith.constant"
        or not isinstance(value, IntegerAttr)
        or value.value.data != 0
        or scalar.type != accumulator
    ):
        raise ValueError("zero_initialization_unproved")
    return dict(
        arithmetic="modular_integer_mac",
        accumulator_dtype=str(accumulator),
        initialization=0,
        operand_extensions=[extensions[arg] for arg in body.args[:2]],
        reduction_order="source_linalg_reduction",
        readout="identity",
    )


def source_product_views(module, *, evidence_id, max_operations=4096, max_products=128, max_placements=256):
    """Extend source diagnostics with exact static slice/concat coordinates.

    The caller supplies its original verified typed module, as the existing
    source parser/diagnostic owner does. Coordinates index immutable logical
    function arguments, never physical buffers. Contractions reuse the existing yielded-MAC/domain observer and
    additionally require ordinary matrix maps and an explicit integer recipe.
    Unsupported roots/readouts and exhausted budgets remain visible UNKNOWN.
    """
    from merlin.analysis.programs.model_macs import observe_model_macs

    if min(max_operations, max_products, max_placements) < 1:
        raise ValueError("source views require finite positive budgets")
    positions = {}
    for index, operation in enumerate(module.walk()):
        if index >= max_operations:
            return dict(status="unknown", reason="source_product_view_operation_budget_exhausted")
        positions[operation] = index
    functions = [op for op in module.body.block.ops if op.name == "func.func" and op.body.blocks]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        return dict(status="unknown", reason="single_source_entry_required")
    function = functions[0]
    block = function.body.block
    domains = {op: shape for op, shape in observe_model_macs(module) if op is not None}
    views, readouts, products, outputs, unknowns = {}, {}, [], [], []
    returned_placements = 0
    for index, argument in enumerate(block.args):
        try:
            shape = _matrix(argument)
            views[argument] = dict(
                argument=index,
                argument_shape=shape,
                shape=shape,
                offsets=[0, 0],
                strides=[1, 1],
                dtype=str(argument.type.get_element_type()),
            )
        except ValueError:
            pass

    def fail(operation, reason):
        unknowns.append(dict(operation=positions[operation], reason=reason))

    for op in block.ops:
        try:
            if op.name == "tensor.extract_slice":
                if len(op.operands) != 1 or len(op.results) != 1 or op.operands[0] not in views:
                    raise ValueError("slice_requires_bound_static_input_view")
                parent = views[op.operands[0]]
                offsets, sizes, strides = [
                    list(op.properties[key].get_values())
                    for key in ("static_offsets", "static_sizes", "static_strides")
                ]
                if (
                    any(len(value) != 2 for value in (offsets, sizes, strides))
                    or _matrix(op.results[0]) != sizes
                    or op.operands[0].type.get_element_type() != op.results[0].type.get_element_type()
                    or any(
                        o < 0 or s <= 0 or step <= 0 or o + (s - 1) * step >= extent
                        for o, s, step, extent in zip(offsets, sizes, strides, parent["shape"], strict=True)
                    )
                ):
                    raise ValueError("slice_offsets_sizes_or_strides_unproved")
                views[op.results[0]] = dict(
                    parent,
                    shape=sizes,
                    offsets=[
                        a + b * step for a, b, step in zip(parent["offsets"], offsets, parent["strides"], strict=True)
                    ],
                    strides=[a * b for a, b in zip(parent["strides"], strides, strict=True)],
                )
            elif op in domains:
                domain = domains[op]
                if domain.status != "derived" or len(tuple(op.inputs)) != 2 or len(tuple(op.outputs)) != 1:
                    raise ValueError("complete_matrix_product_domain_unproved")
                maps = tuple(attr.data for attr in op.get_indexing_maps())
                if (
                    len(maps) != 3
                    or any(mapping.num_dims != 3 or mapping.num_symbols for mapping in maps)
                    or any(not isinstance(expr, AffineDimExpr) for mapping in maps for expr in mapping.results)
                    or tuple(tuple(expr.position for expr in mapping.results) for mapping in maps)
                    != ((0, 2), (2, 1), (0, 1))
                    or tuple(item.data.value for item in op.get_iterator_types())
                    != ("parallel", "parallel", "reduction")
                ):
                    raise ValueError("ordinary_matrix_indexing_required")
                lhs, rhs = tuple(op.inputs)
                if lhs not in views or rhs not in views or len(op.results) != 1:
                    raise ValueError("product_operand_view_unbound")
                a, b, shape = _matrix(lhs), _matrix(rhs), _matrix(op.results[0])
                if a[1] != b[0] or shape != [a[0], b[1]]:
                    raise ValueError("matrix_product_geometry_mismatch")
                numeric = _numeric(op)
                if len(products) >= max_products:
                    return dict(status="unknown", reason="source_product_view_product_budget_exhausted")
                index = len(products)
                products.append(
                    dict(
                        product=index,
                        operation=positions[op],
                        witness=f"{evidence_id}#operation:{positions[op]}",
                        lhs=views[lhs],
                        rhs=views[rhs],
                        output_shape=shape,
                        reduction=dict(extent=a[1], lhs_axis=1, rhs_axis=0),
                        numeric=numeric,
                    )
                )
                readouts[op.results[0]] = [dict(product=index, offsets=[0, 0], shape=shape)]
            elif op.name == "tensor.concat":
                axis = op.properties["dim"].value.data
                if axis not in (0, 1) or len(op.results) != 1:
                    raise ValueError("static_matrix_concat_axis_required")
                shape = _matrix(op.results[0])
                spans, cursor = [], 0
                for value in op.operands:
                    if value in readouts and len(readouts[value]) > max_placements - len(spans):
                        return dict(status="unknown", reason="source_product_view_placement_budget_exhausted")
                    part = _matrix(value)
                    if value not in readouts or part[1 - axis] != shape[1 - axis]:
                        raise ValueError("concat_requires_complete_product_readout_views")
                    if value.type.get_element_type() != op.results[0].type.get_element_type():
                        raise ValueError("concat_dtype_mismatch")
                    for span in readouts[value]:
                        offsets = list(span["offsets"])
                        offsets[axis] += cursor
                        spans.append(dict(span, offsets=offsets))
                    cursor += part[axis]
                if not spans or cursor != shape[axis]:
                    raise ValueError("concat_output_domain_mismatch")
                readouts[op.results[0]] = spans
            elif op.name == "func.return":
                for index, value in enumerate(op.operands):
                    if value not in readouts:
                        raise ValueError("returned_value_requires_product_view_adapter")
                    if len(readouts[value]) > max_placements - returned_placements:
                        return dict(status="unknown", reason="source_product_view_placement_budget_exhausted")
                    returned_placements += len(readouts[value])
                    outputs.append(dict(output=index, shape=_matrix(value), placements=readouts[value]))
            elif op.name not in ("tensor.empty", "arith.constant", "linalg.fill"):
                raise ValueError("source_operation_requires_view_adapter")
        except (ValueError, AttributeError, KeyError, IndexError) as error:
            fail(op, str(error) if isinstance(error, ValueError) else "source_view_structure_unavailable")
    if (
        not outputs
        or block.last_op is None
        or block.last_op.name != "func.return"
        or len(outputs) != len(tuple(function.function_type.outputs))
    ):
        unknowns.append(dict(operation=positions[function], reason="complete_ordered_output_domain_unproved"))
    used = {span["product"] for output in outputs for span in output["placements"]}
    if used != set(range(len(products))):
        unknowns.append(dict(operation=positions[function], reason="not_all_source_products_reach_readouts"))
    for product in products:
        product["readout_uses"] = sum(
            span["product"] == product["product"] for output in outputs for span in output["placements"]
        )
    return dict(
        status="partial" if unknowns else "observed",
        origin="typed_source_ssa",
        coordinate_scope="logical_function_arguments_and_ordered_returns",
        products=products,
        products_total=len(products),
        outputs=outputs,
        outputs_total=len(outputs),
        coverage=dict(
            source_products_total=len(domains),
            source_products_bound=len(products),
            ordered_returns_total=len(tuple(function.function_type.outputs)),
            ordered_returns_bound=len(outputs),
            products_omitted=0,
            outputs_omitted=0,
        ),
        unknowns=unknowns,
        source_to_command_buffer="unknown",
        source_capture_equivalence="unknown",
        emitted_application="unknown",
        numerical_equivalence="unknown",
        performance="unknown",
    )

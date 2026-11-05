"""Logical operand/readout coordinates from actual typed source SSA.

This is a source-mechanism diagnostic, not a source-to-command/binary proof.
No physical packing, numerical comparison or capture owner is selected here.
"""

from __future__ import annotations

from xdsl.dialects.builtin import IntegerAttr, IntegerType, TensorType
from xdsl.ir import Operation
from xdsl.ir.affine import AffineDimExpr


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
    from merlin.perf.compiler_plan_evidence import _source_has_multiply_accumulate

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
    if (not isinstance(constant, Operation) or constant.name != "arith.constant"
            or not isinstance(value, IntegerAttr) or value.value.data != 0 or scalar.type != accumulator):
        raise ValueError("zero_initialization_unproved")
    return dict(arithmetic="modular_integer_mac", accumulator_dtype=str(accumulator), initialization=0,
                operand_extensions=[extensions[arg] for arg in body.args[:2]],
                reduction_order="source_linalg_reduction", readout="identity")


def source_product_views(module, *, evidence_id, max_operations=4096, max_products=128, max_placements=256):
    """Extend source diagnostics with exact static slice/concat coordinates.

    The caller supplies its original verified typed module, as the existing
    source parser/diagnostic owner does. Coordinates index immutable logical
    function arguments, never physical buffers. Contractions reuse the existing yielded-MAC/domain observer and
    additionally require ordinary matrix maps and an explicit integer recipe.
    Unsupported roots/readouts and exhausted budgets remain visible UNKNOWN.
    """
    from merlin.perf.model_macs import observe_model_macs

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
            views[argument] = dict(argument=index, argument_shape=shape, shape=shape,
                                   offsets=[0, 0], strides=[1, 1], dtype=str(argument.type.get_element_type()))
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
                offsets, sizes, strides = [list(op.properties[key].get_values()) for key in (
                    "static_offsets", "static_sizes", "static_strides")]
                if (any(len(value) != 2 for value in (offsets, sizes, strides)) or _matrix(op.results[0]) != sizes
                        or op.operands[0].type.get_element_type() != op.results[0].type.get_element_type()
                        or any(o < 0 or s <= 0 or step <= 0 or o + (s - 1) * step >= extent
                               for o, s, step, extent in zip(offsets, sizes, strides, parent["shape"], strict=True))):
                    raise ValueError("slice_offsets_sizes_or_strides_unproved")
                views[op.results[0]] = dict(parent, shape=sizes,
                    offsets=[a + b * step for a, b, step in zip(
                        parent["offsets"], offsets, parent["strides"], strict=True)],
                    strides=[a * b for a, b in zip(parent["strides"], strides, strict=True)])
            elif op in domains:
                domain = domains[op]
                if domain.status != "derived" or len(tuple(op.inputs)) != 2 or len(tuple(op.outputs)) != 1:
                    raise ValueError("complete_matrix_product_domain_unproved")
                maps = tuple(attr.data for attr in op.get_indexing_maps())
                if (len(maps) != 3 or any(mapping.num_dims != 3 or mapping.num_symbols for mapping in maps)
                        or any(not isinstance(expr, AffineDimExpr) for mapping in maps for expr in mapping.results)
                        or tuple(tuple(expr.position for expr in mapping.results) for mapping in maps)
                        != ((0, 2), (2, 1), (0, 1))
                        or tuple(item.data.value for item in op.get_iterator_types())
                        != ("parallel", "parallel", "reduction")):
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
                products.append(dict(product=index, operation=positions[op],
                    witness=f"{evidence_id}#operation:{positions[op]}",
                    lhs=views[lhs], rhs=views[rhs], output_shape=shape,
                    reduction=dict(extent=a[1], lhs_axis=1, rhs_axis=0), numeric=numeric))
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
    if (not outputs or block.last_op is None or block.last_op.name != "func.return"
            or len(outputs) != len(tuple(function.function_type.outputs))):
        unknowns.append(dict(operation=positions[function], reason="complete_ordered_output_domain_unproved"))
    used = {span["product"] for output in outputs for span in output["placements"]}
    if used != set(range(len(products))):
        unknowns.append(dict(operation=positions[function], reason="not_all_source_products_reach_readouts"))
    for product in products:
        product["readout_uses"] = sum(span["product"] == product["product"]
                                      for output in outputs for span in output["placements"])
    return dict(status="partial" if unknowns else "observed", origin="typed_source_ssa",
                coordinate_scope="logical_function_arguments_and_ordered_returns",
                products=products, products_total=len(products), outputs=outputs, outputs_total=len(outputs),
                coverage=dict(source_products_total=len(domains), source_products_bound=len(products),
                              ordered_returns_total=len(tuple(function.function_type.outputs)),
                              ordered_returns_bound=len(outputs), products_omitted=0, outputs_omitted=0),
                unknowns=unknowns, source_to_command_buffer="unknown", source_capture_equivalence="unknown",
                emitted_application="unknown", numerical_equivalence="unknown", performance="unknown")

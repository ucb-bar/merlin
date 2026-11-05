"""Preserve source-proved per-member readouts in a shared contraction outline.

The existing output-fusion theorem owns all arithmetic. This bridge transports
its exact cut, constants and original result views; it supplies no target law.
"""

from dataclasses import replace


def extend_sibling_readouts(plan, *, path, values, extents=None):
    """Return an uncommitted typed-source plan; raw observers keep their chain.

    Constant readout operands must be source constants or independently bound
    capture values. Folded bias words are explicit immutable caller inputs. The
    raw contraction core remains separately available for ordinary CB emission.
    """
    import numpy as np
    from xdsl.dialects import arith, func, tensor
    from xdsl.dialects.builtin import ArrayAttr, DenseIntOrFPElementsAttr, IntegerAttr, TensorType, i32, i64
    from xdsl.ir import Block, Operation, Region

    from merlin.compile.analysis.contraction_shapes import contraction_extents
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_output_program as GOP
    from merlin.xdsl_dialects.lowering import output_fusion as OF

    from .group_host_call import _device_result
    from .source_region import _sibling_tensor_operation_is_pure

    if path is None:
        return plan
    if extents is None:
        extents = contraction_extents(plan.host)
    decisions, records, frontiers, relayouts, erased, prepared = [], [], [], [], list(plan.selected), []
    inputs = list(plan.inputs)
    for member, (root, geometry) in enumerate(zip(plan.selected, plan.receipt["source_geometries"], strict=True)):
        m, k, n = geometry
        chain = OF.source_chain(root, values)
        decision = OF.decide(chain, path, values, extents)
        record = dict(
            member=member,
            output_dtype="i32",
            attributes=dict(epilogue=[], output_dtype="i32"),
            bias_values=[],
            bias_argument=None,
            output_fusion=decision.record(),
            absorbed=False,
        )
        frontier, relayout = root.results[0], None
        if decision.cut == "requant" and "residual" not in decision.stages:
            members = [root, *decision.members]
            group = CG.Group(member, "source_output_path", root, members=members, fusion=decision.record())
            frontier = decision.members[-1].results[0]
            relayout, device_type, reason = _device_result(
                group, members, dict(op="matmul", M=m, K=k, N=n), frontier.type
            )
            if reason:
                raise ValueError("shared source readout result view: " + reason)
            if tuple(device_type.get_shape()) != (m, n) or str(device_type.element_type) != "i8":
                raise ValueError("shared source readout needs the original complete matrix i8 exit")
            if any(op.parent is not plan.block or not _sibling_tensor_operation_is_pure(op) for op in decision.members):
                raise ValueError("shared source readout crosses an unproved operation effect")
            offsets = GOP._integer_offsets(decision.proof, n)
            attributes = dict(
                GOP._readout(decision.proof, decision.stages, n),
                output_dtype="i8",
                epilogue=(["bias_add"] if any(offsets) else [])
                + ["acc_scale"]
                + (["relu"] if "relu" in decision.stages else []),
            )
            if any(offsets):
                constant = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(i32, (n,)), offsets))
                record["bias_argument"] = len(inputs)
                inputs.append(constant.result)
                prepared.append(constant)
            record.update(
                output_dtype="i8",
                attributes=attributes,
                bias_values=list(offsets),
                absorbed=True,
                source_result_type=str(frontier.type),
                device_result_type=str(device_type),
            )
            erased.extend(decision.members)
        else:
            decision = None
        decisions.append(decision)
        frontiers.append(frontier)
        relayouts.append(relayout)
        records.append(record)
    if not any(record["absorbed"] for record in records):
        return replace(plan, receipt={**plan.receipt, "readouts": records})

    raw = plan.core.body.block.first_op
    body = Block(arg_types=[value.type for value in inputs])
    raw_map = dict(zip(raw.body.block.args, body.args[: len(plan.inputs)], strict=True))
    for op in tuple(raw.body.block.ops)[:-1]:
        clone = op.clone(value_mapper=raw_map)
        body.add_op(clone)
        raw_map.update(zip(op.results, clone.results, strict=True))
    mapped = dict(zip(plan.inputs, body.args[: len(plan.inputs)], strict=True))
    mapped.update(
        (root.results[0], raw_map[value])
        for root, value in zip(plan.selected, raw.body.block.last_op.arguments, strict=True)
    )
    constant_elements = 0

    def bind(value):
        nonlocal constant_elements
        if value in mapped:
            return mapped[value]
        owner = value.owner
        if isinstance(owner, Operation) and owner.parent is plan.block and _sibling_tensor_operation_is_pure(owner):
            for operand in owner.operands:
                bind(operand)
            clone = owner.clone(value_mapper=mapped)
            body.add_op(clone)
            mapped.update(zip(owner.results, clone.results, strict=True))
            return mapped[value]
        array = values.array(value)
        if array is None or not isinstance(value.type, TensorType):
            raise ValueError("shared readout has no immutable source binding for a numerical operand")
        array = np.asarray(array)
        shape = tuple(value.type.get_shape())
        if array.size != int(np.prod(shape)):
            raise ValueError("shared readout numerical operand shape changed")
        constant_elements += int(array.size)
        if constant_elements > 1 << 16:
            raise ValueError("shared readout constant-materialization bound exceeded")
        constant = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(value.type, array.reshape(-1).tolist()))
        body.add_op(constant)
        mapped[value] = constant.result
        return constant.result

    results = []
    for root, decision, frontier, geometry in zip(
        plan.selected, decisions, frontiers, plan.receipt["source_geometries"], strict=True
    ):
        result = mapped[root.results[0]] if decision is None else bind(frontier)
        if decision is not None and tuple(result.type.get_shape()) != (geometry[0], geometry[2]):
            # The existing result-view reader already proved that every source
            # coordinate is contiguous and ordered; flatten/expand is its inverse.
            rank = len(result.type.get_shape())
            collapsed = tensor.CollapseShapeOp(
                operands=[result],
                result_types=[TensorType(result.type.element_type, (geometry[0] * geometry[2],))],
                properties={"reassociation": ArrayAttr([ArrayAttr([IntegerAttr(i, i64) for i in range(rank)])])},
            )
            body.add_op(collapsed)
            expanded = tensor.ExpandShapeOp(
                collapsed.result,
                (),
                ArrayAttr([ArrayAttr([IntegerAttr(i, i64) for i in range(2)])]),
                (geometry[0], geometry[2]),
                TensorType(result.type.element_type, (geometry[0], geometry[2])),
            )
            body.add_op(expanded)
            result = expanded.result
        results.append(result)
    body.add_op(func.ReturnOp(*results))
    callee = func.FuncOp(
        plan.callee.sym_name.data, ([value.type for value in inputs], [value.type for value in results]), Region(body)
    )
    callee.verify()
    positions = {op: i for i, op in enumerate(plan.block.ops)}
    return replace(
        plan,
        inputs=tuple(inputs),
        outputs=tuple(frontiers),
        callee=callee,
        erased=tuple(sorted(set(erased), key=positions.get)),
        relayouts=tuple(relayouts),
        prepared_inputs=tuple(prepared),
        receipt={**plan.receipt, "readouts": records},
    )

"""Independent source/destination shape checks for derived dynamic models.

Only original compiler members and actual standard bodies participate. Runtime
arrays, references and producer implementation imports are excluded.
"""

from xdsl.dialects import builtin, func

from .model_shape_facts import PROFILE, ShapeFacts


def bind_kernel_shapes(facts, function, spec, shapes, dimensions, call, results):
    if not isinstance(function, func.FuncOp) or len(function.body.blocks) != 1:
        raise ValueError("symbolic kernel requires its executable standard function")
    inputs = [buffer for buffer in spec["buffers"] if buffer["effect"] != "write"]
    outputs = [buffer for buffer in spec["buffers"] if buffer["effect"] != "read"]
    scalars = spec.get("runtime_scalars", [])
    if len(function.body.block.args) != len(inputs) + len(scalars) or len(function.function_type.outputs) != len(
        outputs
    ):
        raise ValueError("symbolic kernel public argument/result roster differs from its ABI")
    local = ShapeFacts()
    for argument, buffer in zip(function.body.block.args[: len(inputs)], inputs, strict=True):
        if not _buffer_type(argument.type, buffer):
            raise ValueError("symbolic kernel argument type differs from its source ABI")
        local.tensors[argument] = tuple(shapes[buffer["name"]])
    for argument, scalar in zip(function.body.block.args[len(inputs) :], scalars, strict=True):
        if argument.type != getattr(builtin, scalar["dtype"]):
            raise ValueError("symbolic kernel scalar type differs from its source ABI")
        expression = dimensions.get(scalar["name"])
        if isinstance(expression, str) or type(expression) is int and expression >= 0:
            local.integers[argument] = expression
    if any(
        not _buffer_type(type_, buffer) for type_, buffer in zip(function.function_type.outputs, outputs, strict=True)
    ):
        raise ValueError("symbolic kernel result types differ from its source ABI")
    terminal = function.body.block.last_op
    if not isinstance(terminal, func.ReturnOp) or [local.shape(value) for value in terminal.operands] != [
        tuple(shapes[b["name"]]) for b in outputs
    ]:
        raise ValueError("symbolic kernel result dimensions lack an executable standard-MLIR proof")
    for value, name in zip(call.results, results, strict=True):
        facts.tensors[value] = tuple(shapes[name])


def _buffer_type(type_, buffer):
    return (
        isinstance(type_, builtin.TensorType)
        and isinstance(type_.encoding, builtin.NoneAttr)
        and str(type_.element_type) == buffer["dtype"]
        and len(type_.get_shape()) == len(buffer["shape"])
        and all(
            actual < 0 if isinstance(wanted, str) else actual == wanted
            for actual, wanted in zip(type_.get_shape(), buffer["shape"], strict=True)
        )
    )


def checked_model_shapes(parent, bound, bindings, interfaces):
    calls = bindings.get("calls", [])
    entries = [
        op for op in parent.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward" and op.body.blocks
    ]
    dynamic = len(entries) == 1 and any(
        isinstance(type_, builtin.TensorType) and any(n < 0 for n in type_.get_shape())
        for type_ in (*entries[0].function_type.inputs, *entries[0].function_type.outputs)
    )
    selected = bindings.get("model_shape_profile")
    present = dynamic or any("symbolic_tensor_shapes" in call for call in calls) or "runtime_shape_contract" in bindings
    if selected is None and not present:
        return None
    if selected != PROFILE or len(entries) != 1:
        raise ValueError("derived symbolic model shapes require their explicit profile")
    original_ordinals = [call.get("parent_call_ordinal") for call in calls]
    if any(type(n) is not int for n in original_ordinals) or original_ordinals != sorted(set(original_ordinals)):
        raise ValueError("derived symbolic model calls require original source order")
    from .derived_runtime_arguments import checked_runtime_call_shapes

    facts = ShapeFacts.entry(entries[0])
    call_shapes, maximum = {}, {}
    for record in calls:
        digest = record["implementation_sha256"]
        spec, function = interfaces[digest]
        limits = {}
        call_shapes[record["parent_call_ordinal"]] = checked_runtime_call_shapes(
            parent,
            spec,
            record,
            tensor_views=bindings.get("tensor_view_binding_profile") is not None,
            source_views=bindings.get("source_tensor_view_binding_profile") is not None,
            provenance=bindings["kernels"][digest],
            live_operands=True,
            checked_indices=bindings.get("argument_binding_profile") == "t2m.checked-index-runtime-arguments.v1",
            shape_facts=facts,
            kernel_function=function,
            maximum_shapes=limits,
        )
        maximum[record["parent_call_ordinal"]] = limits
    contract = facts.contract(entries[0])
    bound_entries = [
        op for op in bound.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward" and op.body.blocks
    ]
    if (
        len(bound_entries) != 1
        or bound_entries[0].function_type != entries[0].function_type
        or bindings.get("runtime_shape_contract") != contract
        or ShapeFacts.entry(bound_entries[0]).contract(bound_entries[0]) != contract
    ):
        raise ValueError("derived symbolic model contract differs from the original and executable public interfaces")
    if (
        ("guard_storage_bytes" in bindings) != ("guard_storage_scope" in bindings)
        or "guard_storage_scope" in bindings
        and bindings["guard_storage_scope"] != "sum-of-source-shape-upper-bounds"
    ):
        raise ValueError("derived symbolic scratch requires its explicit source-bound scope")
    return {"contract": contract, "call_shapes": call_shapes, "maximum_shapes": maximum}

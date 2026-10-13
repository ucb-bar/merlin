"""Closed SSA proofs of source payload/control roles, independent of placement."""

from xdsl.dialects import builtin

from merlin.common import mlir_query as mq


def _block(member):
    if len(member.regions) != 1 or len(member.regions[0].blocks) != 1:
        return None
    return member.regions[0].block


def _select_roles(member, inputs):
    block = _block(member)
    if block is None or len(block.args) < len(inputs):
        return None
    operations = tuple(block.ops)
    if [mq.op_name(op) for op in operations] != ["arith.select", "linalg.yield"]:
        return None
    select, terminator = operations
    if (len(select.operands) != 3 or len(select.results) != 1
        or tuple(terminator.operands) != tuple(select.results)):
        return None
    arguments = tuple(block.args[:len(inputs)])
    if any(value not in arguments for value in select.operands):
        return None
    condition, lhs, rhs = select.operands
    if (condition.type != builtin.i1 or lhs.type != rhs.type or lhs.type != select.results[0].type
        or condition is lhs or condition is rhs):
        return None
    predicate = arguments.index(condition)
    payload = sorted({arguments.index(lhs), arguments.index(rhs)})
    return {"inputs": tuple(inputs[index] for index in payload), "control_indices": (predicate,),
            "basis": "exact_select_SSA_condition_and_value_roles"}


def _integer_generator(member, inputs):
    """Observe real arithmetic width of a closed input-free index generator."""
    block = _block(member)
    if inputs or block is None or any(argument.uses for argument in block.args):
        return None
    operations = tuple(block.ops)
    if not operations or mq.op_name(operations[-1]) != "linalg.yield":
        return None
    yielded = tuple(operations[-1].operands)
    if len(yielded) != 1 or not isinstance(yielded[0].type, builtin.IntegerType):
        return None
    integer = yielded[0].type
    arithmetic = []
    for operation in operations[:-1]:
        name = mq.op_name(operation)
        if operation.regions or len(operation.results) != 1:
            return None
        if name == "linalg.index":
            if operation.operands or not isinstance(operation.results[0].type, builtin.IndexType):
                return None
        elif name == "arith.constant":
            if operation.operands or operation.results[0].type not in {integer, builtin.IndexType()}:
                return None
        elif name == "arith.index_cast":
            if (len(operation.operands) != 1 or not isinstance(operation.operands[0].type, builtin.IndexType)
                or operation.results[0].type != integer):
                return None
        elif name in {"arith.addi", "arith.muli"}:
            if len(operation.operands) != 2 or any(value.type != integer
                    for value in (*operation.operands, *operation.results)):
                return None
            arithmetic.append(operation)
        else:
            return None
        if any(getattr(value.owner, "parent", None) is not block for value in operation.operands):
            return None
    reached, pending = set(), [yielded[0]]
    while pending:
        value = pending.pop()
        owner = value.owner
        if owner in reached:
            continue
        if owner not in operations[:-1]:
            return None
        reached.add(owner)
        pending.extend(owner.operands)
    if not any(operation in reached for operation in arithmetic):
        return None
    return {"inputs": tuple(arithmetic[0].operands), "control_indices": (),
            "basis": "closed_index_generator_actual_integer_arithmetic"}


def payload_roles(member, inputs):
    """Return typed payload values; unknown roles retain the full original roster."""
    name = mq.op_name(member)
    if name == "tensor.extract":
        if member.operands and len(member.results) == 1:
            source, *indices = member.operands
            shape = source.type.get_shape() if isinstance(source.type, builtin.TensorType) else None
            if (shape is not None and len(indices) == len(shape)
                and all(isinstance(index.type, builtin.IndexType) for index in indices)
                and member.results[0].type == source.type.element_type):
                return {"inputs": (source,), "control_indices": tuple(range(1, len(indices) + 1)),
                        "basis": "typed_tensor_extract_source_and_index_roles"}
        return {"inputs": (), "control_indices": (), "basis": "unproved_tensor_extract_roles"}
    if name == "linalg.generic":
        selected = _select_roles(member, inputs) or _integer_generator(member, inputs)
        if selected is not None:
            return selected
    return {"inputs": tuple(inputs), "control_indices": (), "basis": "original_operand_roles"}

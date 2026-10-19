"""Find private tensor producer/consumer pairs that can execute together."""

from __future__ import annotations


def pure_tensor_compute(op) -> bool:
    """Recognize tensor-only structured arithmetic, refusing unknown effects.

    Structured linalg operations do not advertise scalar-body effects through
    xDSL's generic effect query. Check their actual bodies and tensor operands;
    memref operations, nested regions, captures and unknown scalar ops refuse.
    """
    from xdsl.dialects.builtin import AnyFloat, IndexType, IntegerType, TensorType
    from xdsl.dialects.linalg.abstract_ops import LinalgStructuredOperation
    from xdsl.traits import is_side_effect_free

    if not op.regions:
        return is_side_effect_free(op)
    if not isinstance(op, LinalgStructuredOperation):
        return False
    if (not op.results or any(not isinstance(v.type, TensorType) for v in op.results)
            or any(not isinstance(v.type, (TensorType, AnyFloat, IntegerType, IndexType))
                   for v in op.inputs)
            or any(not isinstance(v.type, TensorType) for v in op.outputs)
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1):
        return False
    block = op.regions[0].block
    local = set(block.args)
    for step in block.ops:
        if step.regions or any(value not in local for value in step.operands):
            return False
        if step.name != "linalg.yield" and not is_side_effect_free(step):
            return False
        local.update(step.results)
    return True


def destination_only_tensor_call(op) -> bool:
    """A destination-passing call cannot mutate the delayed tensor inputs.

    The external declaration must explicitly promise read-only inputs and a
    single private write-only destination allocated by tensor.empty or splat. Device state may
    change, but the delayed pure tensor arithmetic neither reads nor writes it.
    """
    from xdsl.dialects.builtin import TensorType
    from xdsl.traits import SymbolTable

    if op.name != "func.call" or len(op.results) != 1 or not op.arguments:
        return False
    if any(not isinstance(v.type, TensorType) for v in (*op.arguments, *op.results)):
        return False
    destination = op.arguments[-1]
    if (getattr(destination.owner, "name", None) not in {"tensor.empty", "tensor.splat"}
            or tuple(use.operation for use in destination.uses) != (op,)
            or op.results[0].type != destination.type):
        return False
    declaration = SymbolTable.lookup_symbol(op, op.callee)
    attrs = getattr(declaration, "arg_attrs", None)
    if declaration is None or declaration.body.blocks or attrs is None:
        return False
    modes = [getattr(a.data.get("bufferization.access"), "data", None) for a in attrs]
    return modes == ["read"] * (len(op.arguments) - 1) + ["write"]


def private_tensor_pairs(candidates, *, can_cross=pure_tensor_compute):
    """Return disjoint single-use pairs across independently pure operations.

    The caller supplies supported operations, not names or provenance hints.
    Only a sole direct input use can hide the producer's result. The pair may
    execute at the consumer's position when every crossed operation passes the
    supplied legality predicate (pure computation by default). A caller can
    also admit destination-only calls whose declarations protect tensor inputs.
    This is an analysis: the caller's IR is never mutated.
    """
    candidates = tuple(candidates)
    allowed, used, pairs = set(candidates), set(), []
    for producer in candidates:
        if producer in used or len(producer.results) != 1 or not pure_tensor_compute(producer):
            continue
        uses = tuple(producer.results[0].uses)
        if len(uses) != 1:
            continue
        consumer = uses[0].operation
        if (consumer not in allowed or consumer in used or consumer is producer
                or consumer.parent is not producer.parent or not pure_tensor_compute(consumer)
                or producer.results[0] not in consumer.inputs):
            continue
        crossed = producer.next_op
        while crossed is not None and crossed is not consumer and can_cross(crossed):
            crossed = crossed.next_op
        if crossed is consumer:
            pairs.append((producer, consumer))
            used.update((producer, consumer))
    return tuple(pairs)


def private_tensor_chain(operations, *, can_cross=pure_tensor_compute) -> bool:
    """Check an ordered pure chain whose private results each have one input use.

    Only pure tensor arithmetic moves. A caller can admit independent external
    destination calls as barriers crossed by that arithmetic; this never grants
    permission to reorder already lowered device calls themselves.
    """
    operations = tuple(operations)
    if (len(operations) < 2 or len(set(operations)) != len(operations)
            or any(len(op.results) != 1 or not hasattr(op, "inputs")
                   or not pure_tensor_compute(op) for op in operations)):
        return False
    for producer, consumer in zip(operations, operations[1:]):
        uses = tuple(producer.results[0].uses)
        if (producer.parent is None or producer.parent is not consumer.parent
                or len(uses) != 1 or uses[0].operation is not consumer
                or producer.results[0] not in consumer.inputs):
            return False
        crossed = producer.next_op
        while crossed is not None and crossed is not consumer and can_cross(crossed):
            crossed = crossed.next_op
        if crossed is not consumer:
            return False
    return True


def replace_private_tensor_chain(module, operations, *, symbol, inputs, can_cross=pure_tensor_compute):
    """Replace a provider-qualified static pure chain by one synchronous DPS call.

    Providers own endpoint arithmetic, rounding, completion, exceptional and cost
    contracts. Core proves privacy/effects, exact external input membership and
    declaration compatibility. No numerical equivalence follows from a symbol.
    Source output initializers must be unused; arbitrary reductions refuse.
    """
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, NoneAttr, StringAttr, SymbolRefAttr, TensorType
    from xdsl.ir import Region
    from xdsl.traits import SymbolTable

    operations, inputs = tuple(operations), tuple(inputs)
    if not private_tensor_chain(operations, can_cross=can_cross):
        raise ValueError("tensor chain is observable, unordered or crosses unsupported effects")
    root = operations[0]
    while root.parent_op() is not None:
        root = root.parent_op()
    if root is not module or not symbol or not symbol.isascii():
        raise ValueError("tensor chain belongs to a different module or invalid symbol")
    internal = {op.results[0] for op in operations}
    external = {value for op in operations for value in op.inputs if value not in internal}
    if len(set(inputs)) != len(inputs) or set(inputs) != external:
        raise ValueError("tensor chain endpoint does not receive its exact external inputs")
    result_type = operations[-1].results[0].type
    if (any(not isinstance(value.type, TensorType) for value in inputs)
            or not isinstance(result_type, TensorType) or not isinstance(result_type.encoding, NoneAttr)
            or any(dim <= 0 for dim in result_type.get_shape())
            or any(arg.uses for op in operations for arg in op.body.block.args[len(op.inputs):])):
        raise ValueError("tensor chain requires static tensor results and unused output initializers")
    signature = tuple(value.type for value in inputs) + (result_type,)
    declaration = SymbolTable.lookup_symbol(operations[0], SymbolRefAttr(symbol))
    read = DictionaryAttr({"bufferization.access": StringAttr("read")})
    write = DictionaryAttr({"bufferization.access": StringAttr("write")})
    attrs = ArrayAttr([read] * len(inputs) + [write])
    if declaration is not None:
        if (not isinstance(declaration, func.FuncOp) or declaration.body.blocks
                or tuple(declaration.function_type.inputs) != signature
                or tuple(declaration.function_type.outputs) != (result_type,)
                or declaration.arg_attrs != attrs):
            raise ValueError("tensor chain endpoint collides with an incompatible declaration")
    last = operations[-1]
    destination = tensor.EmptyOp((), result_type)
    call = func.CallOp(symbol, [*inputs, destination.results[0]], [result_type])
    last.parent.insert_ops_before([destination, call], last)
    last.results[0].replace_all_uses_with(call.results[0])
    for op in reversed(operations):
        op.detach()
        op.erase()
    if declaration is None:
        module.body.block.add_op(func.FuncOp(symbol, (signature, [result_type]), Region(),
                                             visibility="private", arg_attrs=attrs))
    module.verify()
    return call

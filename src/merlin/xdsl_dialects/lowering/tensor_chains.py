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

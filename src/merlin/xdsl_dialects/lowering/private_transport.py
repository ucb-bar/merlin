"""Change an unobservable call edge's representation without moving work.

Providers qualify equivalent producer and consumer entries, including rounding
and exceptional behavior. Merlin proves the edge is private, its views preserve
flat contents, effects are bounded, and all unrelated values retain their types.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects import func, tensor
from xdsl.dialects.builtin import (
    AnyFloat,
    ArrayAttr,
    DictionaryAttr,
    IntegerType,
    NoneAttr,
    StringAttr,
    SymbolRefAttr,
    TensorType,
)
from xdsl.ir import Region
from xdsl.traits import SymbolTable

from .shared_preparation import contiguous_tensor_root
from .tensor_chains import destination_only_tensor_call, pure_tensor_compute


@dataclass(frozen=True)
class PrivateTensorTransport:
    """Provider contract for equivalent complete synchronous call endpoints.

    The producer writes its complete destination in ``element_type``; the
    consumer reads that representation at the selected input position. Their
    composition must preserve all original results and refusals. Names do not
    prove equivalence; qualification remains the provider's obligation.
    """

    producer_symbol: str
    consumer_symbol: str
    element_type: AnyFloat | IntegerType


def prune_unobserved_transport_views(module):
    """Clone and drop unused static shape-only views before privacy analysis.

    Removed views have one unencoded static tensor input and result, preserve
    contiguous flat contents, and have no users. Calls, allocations, arithmetic,
    dynamic/encoded views and all observed branches survive. This cleanup grants
    no permission to ignore actual fanout or change an endpoint representation.
    """
    module.verify()
    result, removed = module.clone(), 0
    for op in reversed(tuple(result.walk())):
        if (op.name not in {"tensor.collapse_shape", "tensor.expand_shape"}
                or len(op.operands) != 1 or len(op.results) != 1 or tuple(op.results[0].uses)):
            continue
        types = (op.operands[0].type, op.results[0].type)
        if (any(not isinstance(t, TensorType) or not isinstance(t.encoding, NoneAttr)
                or any(d <= 0 for d in t.get_shape()) for t in types)
                or contiguous_tensor_root(op.results[0]) is not contiguous_tensor_root(op.operands[0])):
            continue
        op.detach()
        op.erase()
        removed += 1
    result.verify()
    return result, removed


def private_call_edges(module):
    """Find single-use destination-call edges through static shape-only views."""
    result = []
    for producer in module.walk():
        if not destination_only_tensor_call(producer):
            continue
        value, views = producer.results[0], []
        typ = value.type
        if (not isinstance(typ.encoding, NoneAttr) or any(d <= 0 for d in typ.get_shape())):
            continue
        while len(tuple(value.uses)) == 1:
            use = next(iter(value.uses))
            op = use.operation
            if (op.name not in {"tensor.collapse_shape", "tensor.expand_shape"}
                    or len(op.operands) != 1 or len(op.results) != 1
                    or op.parent is not producer.parent
                    or contiguous_tensor_root(op.results[0]) is not contiguous_tensor_root(value)):
                break
            views.append(op)
            value = op.results[0]
        uses = tuple(value.uses)
        if len(uses) != 1:
            continue
        use = uses[0]
        consumer = use.operation
        if (not destination_only_tensor_call(consumer) or consumer.parent is not producer.parent
                or use.index == len(consumer.arguments) - 1):
            continue
        crossed = producer.next_op
        while (crossed is not None and crossed is not consumer
               and (pure_tensor_compute(crossed) or destination_only_tensor_call(crossed))):
            crossed = crossed.next_op
        if crossed is consumer:
            result.append((producer, tuple(views), consumer, use.index))
    return tuple(result)


def rewrite_private_transports(module, select):
    """Return a verified clone and receipts for disjoint provider-qualified edges.

    Consumer order, shapes, every public value and every arithmetic body remain
    unchanged. Unknown effects, extra consumers, observed intermediate outputs,
    encoded/dynamic views, and non-private destinations retain original execution.
    This is an explicit legality mechanism, not a numerical or cost policy.
    """
    module.verify()
    result = module.clone()
    declarations, changed, used = {}, [], set()
    for producer, views, consumer, index in private_call_edges(result):
        if producer in used or consumer in used:
            continue
        spec = select(producer, consumer, index)
        if spec is None:
            continue
        if any(SymbolTable.lookup_symbol(owner, SymbolRefAttr(name)) is not None
               for owner, name in ((producer, spec.producer_symbol), (consumer, spec.consumer_symbol))):
            raise ValueError("private transport entry collides with an existing symbol")
        old_type = producer.results[0].type
        if (not isinstance(spec.element_type, (AnyFloat, IntegerType))
                or spec.element_type == old_type.element_type):
            raise ValueError("private transport requires a different scalar representation")
        new_type = TensorType(spec.element_type, old_type.get_shape())
        block = producer.parent
        destination = tensor.EmptyOp((), new_type)
        first = func.CallOp(spec.producer_symbol,
                            [*producer.arguments[:-1], destination.results[0]], [new_type])
        first.attributes.update(producer.attributes)
        block.insert_ops_before([destination, first], producer)
        value = first.results[0]
        for view in views:
            changed_view = type(view).create(
                operands=[value], result_types=[TensorType(spec.element_type, view.results[0].type.get_shape())],
                properties=dict(view.properties), attributes=dict(view.attributes))
            block.insert_op_before(changed_view, view)
            value = changed_view.results[0]
        arguments = list(consumer.arguments)
        arguments[index] = value
        second = func.CallOp(spec.consumer_symbol, arguments, [v.type for v in consumer.results])
        second.attributes.update(consumer.attributes)
        block.insert_op_before(second, consumer)
        for old, new in zip(consumer.results, second.results, strict=True):
            old.replace_all_uses_with(new)
        for call in (first, second):
            symbol = call.callee.root_reference.data
            signature = tuple(v.type for v in call.arguments)
            if declarations.setdefault(symbol, signature) != signature:
                raise ValueError("private transport entry has inconsistent signatures")
        changed.append({"producer": producer.callee.root_reference.data,
                        "consumer": consumer.callee.root_reference.data,
                        "producer_entry": spec.producer_symbol, "consumer_entry": spec.consumer_symbol,
                        "consumer_input": index, "private_views": len(views),
                        "original_type": str(old_type), "transport_type": str(new_type)})
        used.update((producer, consumer))
        for old in (consumer, *reversed(views), producer):
            old.detach()
            old.erase()
    existing = {op.sym_name.data for op in result.body.block.ops if isinstance(op, func.FuncOp)}
    read = DictionaryAttr({"bufferization.access": StringAttr("read")})
    write = DictionaryAttr({"bufferization.access": StringAttr("write")})
    for symbol, signature in declarations.items():
        if symbol in existing:
            raise ValueError("private transport entry collides with an existing symbol")
        result.body.block.add_op(func.FuncOp(
            symbol, (signature, [signature[-1]]), Region(), visibility="private",
            arg_attrs=ArrayAttr([read] * (len(signature) - 1) + [write])))
    result.verify()
    return result, tuple(changed)

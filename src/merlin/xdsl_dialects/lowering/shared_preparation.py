"""Share an explicit immutable tensor preparation between read-only consumers.

The provider specifies a deterministic representation and equivalent consumer
entries. This pass owns SSA equivalence, effect barriers, allocation lifetimes,
and rewriting. It never moves consumers or retains implicit pointer caches.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from xdsl.dialects import arith, func, tensor
from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    IntegerType,
    NoneAttr,
    StringAttr,
    TensorType,
)
from xdsl.dialects.linalg.attrs import IteratorType
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Region
from xdsl.ir.affine import AffineMap

from .tensor_chains import destination_only_tensor_call, pure_tensor_compute


@dataclass(frozen=True)
class TensorPreparation:
    """Provider contract: deterministic preparation and equivalent consumer.

    ``arguments`` selects original call arguments in the replacement's order;
    the prepared value replaces ``input_index``. The last private destination
    must remain last. Dropped inputs must be read-only in the original contract.
    The provider remains responsible for numerical equivalence and completion.
    """

    input_index: int
    prepare_symbol: str
    prepared_type: TensorType
    consumer_symbol: str
    arguments: tuple[int, ...]


def contiguous_tensor_root(value):
    """Identify equal flat contents through static, unencoded shape-only views.

    Equal roots alone do not establish equal interpretations: the grouping also
    requires identical consumer input types. Slices, casts and transposes stop.
    """
    while getattr(value.owner, "name", None) in {"tensor.expand_shape", "tensor.collapse_shape"}:
        owner = value.owner
        source = owner.operands[0]
        types = (source.type, value.type)
        if (
            any(
                not isinstance(t, TensorType)
                or not isinstance(t.encoding, NoneAttr)
                or any(d <= 0 for d in t.get_shape())
                for t in types
            )
            or types[0].element_type != types[1].element_type
            or prod(types[0].get_shape()) != prod(types[1].get_shape())
        ):
            break
        value = source
    return value


def _preparation_input_key(value):
    """Equal flat views or identical closed, elementwise floating extensions.

    Extension stays in the key: a rounded narrow value and its widened value
    never compare equal. Only the same source SSA, input/output types and exact
    scalar operation establish equivalence; arbitrary casts still stop.
    """
    root = contiguous_tensor_root(value)
    op = root.owner
    if (
        not isinstance(op, GenericOp)
        or not pure_tensor_compute(op)
        or len(op.inputs) != 1
        or len(op.outputs) != 1
        or len(op.results) != 1
    ):
        return root
    source, destination = op.inputs[0], op.outputs[0]
    types = (source.type, destination.type, root.type)
    if (
        any(
            not isinstance(t, TensorType) or not isinstance(t.encoding, NoneAttr) or any(d <= 0 for d in t.get_shape())
            for t in types
        )
        or types[1] != types[2]
        or types[0].get_shape() != types[1].get_shape()
    ):
        return root
    rank = len(types[0].get_shape())
    if (
        len(op.iterator_types) != rank
        or any(it.data != IteratorType.PARALLEL for it in op.iterator_types)
        or tuple(m.data for m in op.indexing_maps) != (AffineMap.identity(rank),) * 2
    ):
        return root
    body = op.body.block
    steps = tuple(body.ops)
    if (
        len(body.args) != 2
        or len(steps) != 2
        or not isinstance(steps[0], arith.ExtFOp)
        or not isinstance(steps[1], YieldOp)
        or steps[0].input is not body.args[0]
        or steps[0].result.type != types[1].element_type
        or body.args[1].uses
        or steps[0].attributes
        or steps[0].properties
        or tuple(steps[1].operands) != (steps[0].result,)
    ):
        return root
    return ("elementwise_extf", types[0], types[1], contiguous_tensor_root(source))


def _bytes(spec):
    typ = spec.prepared_type
    if (
        not isinstance(typ, TensorType)
        or not isinstance(typ.encoding, NoneAttr)
        or not isinstance(typ.element_type, IntegerType)
        or typ.element_type.width.data != 8
        or any(d <= 0 for d in typ.get_shape())
    ):
        raise ValueError("shared preparation requires a static unencoded byte tensor")
    return prod(typ.get_shape())


def share_tensor_preparations(module, select, *, max_live_bytes: int):
    """Return a verified clone and receipts for preparations shared at least twice.

    Unknown effects split groups. Explicit access contracts protect tensors
    across destination-only calls. Preparations live in ordinary SSA buffers,
    independent of target workspaces. A caller-supplied budget bounds the sum
    of overlapping new buffers; rejected candidates retain the original calls.
    This proves legality, not profitability; selection remains an explicit policy.
    """
    if type(max_live_bytes) is not int or max_live_bytes <= 0:
        raise ValueError("shared preparation needs a positive live byte budget")
    module.verify()
    result = module.clone()
    groups = []
    for function in tuple(result.body.block.ops):
        if not isinstance(function, func.FuncOp) or not function.body.blocks:
            continue
        for block in function.body.blocks:
            pending = {}
            for position, op in enumerate(tuple(block.ops)):
                if not (pure_tensor_compute(op) or destination_only_tensor_call(op)):
                    groups.extend(pending.values())
                    pending = {}
                    continue
                if not isinstance(op, func.CallOp):
                    continue
                spec = select(op)
                if spec is None:
                    continue
                if not destination_only_tensor_call(op):
                    continue
                indices = spec.arguments
                if (
                    not 0 <= spec.input_index < len(op.arguments) - 1
                    or not indices
                    or indices[-1] != len(op.arguments) - 1
                    or spec.input_index not in indices[:-1]
                    or len(set(indices)) != len(indices)
                    or any(type(i) is not int or not 0 <= i < len(op.arguments) for i in indices)
                ):
                    raise ValueError("shared preparation has invalid consumer argument mapping")
                size = _bytes(spec)
                original = op.arguments[spec.input_index]
                key = (_preparation_input_key(original), original.type, spec.prepare_symbol, spec.prepared_type)
                pending.setdefault(key, []).append((position, op, spec, size))
            groups.extend(pending.values())
    groups = sorted((g for g in groups if len(g) > 1), key=lambda g: g[0][0])
    declarations = {}
    records, accepted = [], []
    for group in groups:
        start, first, spec, size = group[0]
        end = group[-1][0]
        overlapping = [(a, b, n) for block, a, b, n in accepted if block is first.parent and a <= end and start <= b]
        # Count at every interval boundary, rather than summing disjoint lives.
        points = {start, *(max(start, a) for a, _, _ in overlapping)}
        if any(size + sum(n for a, b, n in overlapping if a <= p <= b) > max_live_bytes for p in points):
            continue
        block = first.parent
        destination = tensor.EmptyOp((), spec.prepared_type)
        prepare = func.CallOp(
            spec.prepare_symbol, [first.arguments[spec.input_index], destination.results[0]], [spec.prepared_type]
        )
        block.insert_ops_before([destination, prepare], first)
        signatures = [(spec.prepare_symbol, tuple(v.type for v in prepare.arguments))]
        consumers = []
        for _, call, current, _ in group:
            arguments = [
                prepare.results[0] if i == current.input_index else call.arguments[i] for i in current.arguments
            ]
            replacement = func.CallOp(current.consumer_symbol, arguments, [v.type for v in call.results])
            block.insert_op_before(replacement, call)
            for old, new in zip(call.results, replacement.results, strict=True):
                old.replace_all_uses_with(new)
            consumers.append(call.callee.root_reference.data)
            call.detach()
            call.erase()
            signatures.append((current.consumer_symbol, tuple(v.type for v in arguments)))
        for symbol, signature in signatures:
            if declarations.setdefault(symbol, signature) != signature:
                raise ValueError("shared preparation symbol has inconsistent types")
        accepted.append((block, start, end, size))
        records.append(
            {
                "prepare_symbol": spec.prepare_symbol,
                "consumers": consumers,
                "prepared_bytes": size,
                "removed_preparations": len(group) - 1,
                "first_operation": start,
                "last_operation": end,
            }
        )
    existing = {op.sym_name.data for op in result.body.block.ops if isinstance(op, func.FuncOp)}
    read = DictionaryAttr({"bufferization.access": StringAttr("read")})
    write = DictionaryAttr({"bufferization.access": StringAttr("write")})
    for symbol, signature in declarations.items():
        if symbol in existing:
            raise ValueError("shared preparation entry collides with an existing symbol")
        result.body.block.add_op(
            func.FuncOp(
                symbol,
                (signature, [signature[-1]]),
                Region(),
                visibility="private",
                arg_attrs=ArrayAttr([read] * (len(signature) - 1) + [write]),
            )
        )
    result.verify()
    return result, tuple(records)

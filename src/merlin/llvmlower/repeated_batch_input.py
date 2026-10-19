"""Prove exact contiguous batch repetition through static copy/view chains.

The original tensor and call arguments remain materialized and unchanged.
A provider may use this coordinate fact to retain a private preparation across
adjacent equal batch slices. Numerical/effect equivalence, input immutability,
private preparation ownership and synchronous completion are separate provider
obligations. No preparation, allocation, movement or target policy is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

from .ordered_fma_groups import _context_snapshot, _snapshot


@dataclass(frozen=True)
class RepeatedBatchInput:
    value: object
    source: object
    shape: tuple[int, int, int]
    repeat: int
    _witness: tuple = field(repr=False)
    _contexts: tuple = field(repr=False)

    def to_dict(self):
        return {
            "shape": list(self.shape),
            "contiguous_batch_repeat": self.repeat,
            "source_type": str(self.source.type),
            "scope": "exact source coordinates; provider preparation/effect/lifetime proof required",
        }


def _shape(value):
    from xdsl.dialects.builtin import NoneAttr, TensorType

    typ = value.type
    if (not isinstance(typ, TensorType) or not isinstance(typ.encoding, NoneAttr)
            or not typ.get_shape() or any(d <= 0 for d in typ.get_shape())
            or prod(typ.get_shape()) >= 1 << 63):
        raise ValueError("batch repetition requires positive static unencoded tensors")
    return tuple(typ.get_shape())


def prove_repeated_batch_input(value) -> RepeatedBatchInput:
    """Prove ``tensor[b] == tensor[b // repeat * repeat]`` bit for bit.

    Accepted copies yield their sole input without scalar computation. Static
    linear reshapes preserve flat element order; transposes may swap only the
    two matrix axes. A singleton broadcast must repeat the final batch factor,
    leaving matrix axes untouched. Other broadcast orders and unknown producers
    refuse rather than guessing equality from shapes or captured values.
    """
    from xdsl.dialects import tensor
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.dialects.linalg.ops import GenericOp, TransposeOp, YieldOp
    from xdsl.ir.affine import AffineMap

    from .tensor_broadcast import singleton_projection_axes

    logical = _shape(value)
    if len(logical) != 3:
        raise ValueError("batch repetition requires a rank-three logical input")
    batch, rows, cols = logical
    current, matrix = value, (rows, cols)
    chain = []
    while True:
        operation = current.owner
        shape = _shape(current)
        if isinstance(operation, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
            if (len(operation.operands) != 1
                    or prod(_shape(operation.operands[0])) != prod(shape)
                    or operation.operands[0].type.element_type != current.type.element_type):
                raise ValueError("batch repetition requires exact static linear reshapes")
            operation.verify()
            chain.append(operation)
            current = operation.operands[0]
            continue
        if len(shape) < 3 or shape[-2:] != matrix or prod(shape[:-2]) != batch:
            raise ValueError("batch view mixes batch and matrix coordinates")
        if isinstance(operation, TransposeOp):
            rank = len(shape)
            permutation = tuple(operation.permutation.get_values())
            allowed = (*range(rank - 2), rank - 1, rank - 2)
            if permutation != allowed or len(operation.inputs) != 1 or len(operation.outputs) != 1:
                raise ValueError("batch repetition permits only matrix-axis transposes")
            source = operation.inputs[0]
            source_shape = _shape(source)
            if (source.type.element_type != current.type.element_type
                    or source_shape != (*shape[:-2], shape[-1], shape[-2])):
                raise ValueError("batch transpose source shape changed")
            if (len(operation.body.blocks) != 1 or len(operation.body.block.args) != 2
                    or len(tuple(operation.body.block.ops)) != 1
                    or not isinstance(operation.body.block.last_op, YieldOp)
                    or tuple(operation.body.block.last_op.operands) != (operation.body.block.args[0],)):
                raise ValueError("batch transpose must preserve source scalar values")
            chain.extend(operation.walk())
            current, matrix = source, source_shape[-2:]
            continue
        if not isinstance(operation, GenericOp):
            raise ValueError("input has no proved contiguous batch broadcast")
        if (len(operation.inputs) != 1 or len(operation.outputs) != 1
                or len(operation.results) != 1 or len(operation.body.blocks) != 1
                or operation.library_call is not None):
            raise ValueError("batch repetition requires a sole-input pure tensor copy")
        block = operation.body.block
        scalar = tuple(block.ops)
        source = operation.inputs[0]
        source_shape = _shape(source)
        if (len(block.args) != 2 or len(scalar) != 1 or not isinstance(scalar[0], YieldOp)
                or tuple(scalar[0].operands) != (block.args[0],)
                or source.type.element_type != current.type.element_type
                or operation.outputs[0].type != current.type
                or len(source_shape) != len(shape)
                or len(operation.iterator_types) != len(shape)
                or any(iterator.data != IteratorType.PARALLEL for iterator in operation.iterator_types)
                or len(operation.indexing_maps) != 2
                or operation.indexing_maps.data[1].data != AffineMap.identity(len(shape))):
            raise ValueError("batch repetition requires exact parallel copies without arithmetic")
        axes = singleton_projection_axes(operation.indexing_maps.data[0].data, shape, source_shape)
        if axes != tuple(range(len(shape))):
            raise ValueError("batch copy has no same-axis singleton projection")
        changed = [axis for axis, (before, after) in enumerate(zip(source_shape, shape, strict=True))
                   if before != after]
        chain.extend(operation.walk())
        if not changed:
            current = source
            continue
        first = min(changed)
        if (max(changed) >= len(shape) - 2
                or any(d != 1 for d in source_shape[first:-2])):
            raise ValueError("broadcast does not repeat the final contiguous batch factor")
        repeat = prod(shape[first:-2])
        if repeat <= 1 or batch % repeat:
            raise ValueError("batch broadcast has no complete repeated groups")
        contexts = {}
        for item in chain:
            parent = item.parent_op()
            while parent is not None:
                contexts[parent] = _context_snapshot(parent)
                parent = parent.parent_op()
        return RepeatedBatchInput(value, source, logical, repeat,
                                  tuple(_snapshot(item) for item in dict.fromkeys(chain)),
                                  tuple(contexts.values()))


def validate_repeated_batch_input(proof: RepeatedBatchInput) -> None:
    if (type(proof) is not RepeatedBatchInput
            or any(_snapshot(item.operation) != item for item in proof._witness)
            or any(_context_snapshot(item.operation) != item for item in proof._contexts)):
        raise ValueError("batch repetition source or context changed after analysis")
    actual = prove_repeated_batch_input(proof.value)
    if (actual.source is not proof.source or actual.shape != proof.shape or actual.repeat != proof.repeat):
        raise ValueError("batch repetition coordinates changed after analysis")


@dataclass(frozen=True)
class RepeatedBatchConsumer:
    """An explicitly qualified provider alternative with the original tensor ABI."""

    symbol: str
    alternative: str
    argument: int
    repeat: int


def rewrite_repeated_batch_consumers(module, contracts):
    """Retarget proved read-only inputs without changing operands or output ABI.

    The provider separately qualifies exact arithmetic, rounding, exceptional
    partial outputs, private retained preparation and synchronous completion.
    Every selected call is re-proved before any mutation; there is no operation
    movement or source view removal. Original declaration attributes are copied.
    """
    import json

    from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
    from xdsl.dialects.func import CallOp, FuncOp

    contracts = tuple(contracts)
    if not contracts:
        return []
    module.verify()
    declarations = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, FuncOp)}
    reserved = set(declarations)
    selected, plans = set(), []
    for contract in contracts:
        if type(contract) is not RepeatedBatchConsumer or contract.symbol in selected:
            raise ValueError("batch reuse requires distinct typed consumer selections")
        selected.add(contract.symbol)
        name = contract.alternative
        if type(name) is not str or not name.isascii() or not name.isidentifier() or name in reserved:
            raise ValueError("batch reuse alternative symbol collides or is invalid")
        reserved.add(name)
        declaration = declarations.get(contract.symbol)
        if declaration is None or declaration.body.blocks:
            raise ValueError("batch reuse requires an original bodyless declaration")
        types = tuple(declaration.function_type.inputs)
        index = contract.argument
        if type(index) is not int or not 0 <= index < len(types):
            raise ValueError("batch reuse argument is outside its original declaration")
        if type(contract.repeat) is not int or contract.repeat <= 1:
            raise ValueError("batch reuse needs a complete repeated group")
        attrs = declaration.arg_attrs
        if attrs is None or getattr(attrs.data[index].data.get("bufferization.access"), "data", None) != "read":
            raise ValueError("batch reuse input lacks declared read-only access")
        calls = [op for op in module.walk() if isinstance(op, CallOp)
                 and op.callee.root_reference.data == contract.symbol]
        if not calls:
            raise ValueError("batch reuse selected consumer has no source calls")
        proofs = []
        for call in calls:
            if (len(call.callee.nested_references)
                    or tuple(value.type for value in call.arguments) != types
                    or tuple(value.type for value in call.results) != tuple(declaration.function_type.outputs)):
                raise ValueError("batch reuse call differs from its original tensor ABI")
            proof = prove_repeated_batch_input(call.arguments[index])
            if proof.repeat != contract.repeat:
                raise ValueError("batch reuse alternative differs from proved source repetition")
            validate_repeated_batch_input(proof)
            proofs.append(proof)
        plans.append((contract, declaration, calls, proofs))
    reports = []
    for contract, declaration, calls, proofs in plans:
        alternative = declaration.clone()
        alternative.properties["sym_name"] = StringAttr(contract.alternative)
        alternative.attributes["merlin.repeated_batch_input"] = StringAttr(json.dumps({
            "argument": contract.argument, "repeat": contract.repeat,
            "provider_obligation": "original_results_rounding_effects_refusal_and_synchronous_private_preparation",
        }, sort_keys=True))
        declaration.parent.insert_op_before(alternative, declaration)
        for call in calls:
            call.properties["callee"] = SymbolRefAttr(contract.alternative)
        reports.append({"symbol": contract.symbol, "alternative": contract.alternative,
                        "argument": contract.argument, "calls": len(calls),
                        "input_proofs": [proof.to_dict() for proof in proofs]})
    module.verify()
    return reports

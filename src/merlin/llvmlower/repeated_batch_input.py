"""Prove exact contiguous batch repetition through static copy/view chains.

Default binding keeps original tensors and call arguments unchanged. Explicit
compact binding supplies one matrix per source-proved group through ordinary
tensor views and, if required, a compact transpose. A provider may retain private
preparation across equal slices. Numerical/effect equivalence, input immutability,
private preparation ownership and synchronous completion are separate provider
obligations. No native preparation, scheduling or default target policy is supplied.
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
    matrix_transposed: bool
    _witness: tuple = field(repr=False)
    _contexts: tuple = field(repr=False)

    def to_dict(self):
        return {
            "shape": list(self.shape),
            "contiguous_batch_repeat": self.repeat,
            "source_type": str(self.source.type),
            "matrix_transposed": self.matrix_transposed,
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
    current, matrix, transposed = value, (rows, cols), False
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
            transposed = not transposed
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
        return RepeatedBatchInput(value, source, logical, repeat, transposed,
                                  tuple(_snapshot(item) for item in dict.fromkeys(chain)),
                                  tuple(contexts.values()))


def validate_repeated_batch_input(proof: RepeatedBatchInput) -> None:
    if (type(proof) is not RepeatedBatchInput
            or any(_snapshot(item.operation) != item for item in proof._witness)
            or any(_context_snapshot(item.operation) != item for item in proof._contexts)):
        raise ValueError("batch repetition source or context changed after analysis")
    actual = prove_repeated_batch_input(proof.value)
    if (actual.source is not proof.source or actual.shape != proof.shape or actual.repeat != proof.repeat
            or actual.matrix_transposed != proof.matrix_transposed):
        raise ValueError("batch repetition coordinates changed after analysis")


def private_repeated_batch_inputs(uses):
    """Select a closed set whose expanded copy/view results can become dead.

    ``uses`` maps explicit ``(consumer, argument)`` pairs to current source
    proofs. Keep removing selections with an escaping intermediate until the
    complete remaining use graph closes. Source values before the broadcast
    may have other users. This is a structural cost guard, not a numerical,
    alias, DCE, scheduling or measured-profitability grant. It mutates no IR.
    """
    from xdsl.ir import Operation

    uses = dict(uses)
    chains = {}
    for key, proof in uses.items():
        if type(proof) is not RepeatedBatchInput:
            raise ValueError("private batch selection requires an exact source proof")
        if type(key) is not tuple or len(key) != 2:
            raise ValueError("private batch selection requires explicit consumer/argument pairs")
        consumer, argument = key
        if (not isinstance(consumer, Operation) or consumer.parent is None
                or type(argument) is not int or not 0 <= argument < len(consumer.operands)
                or consumer.operands[argument] is not proof.value):
            raise ValueError("private batch selection differs from its original input use")
        validate_repeated_batch_input(proof)
        chains[key] = frozenset(item.operation for item in proof._witness)
    selected = set(uses)
    while selected:
        operations = set().union(*(chains[key] for key in selected))
        escaped = {key for key in selected if any(
            use.operation not in operations and (use.operation, use.index) not in selected
            for operation in chains[key] for result in operation.results for use in result.uses)}
        if not escaped:
            break
        selected.difference_update(escaped)
    return frozenset(selected)


def _compact_input(proof, anchor):
    """Build ordinary tensor views and, when required, a compact matrix transpose.

    Call only after every selected source proof and provider ABI was validated.
    Other users retain their original materialized tensors. Upstream lowering
    owns dead-code elimination, buffer aliasing and view/copy realization.
    """
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import ArrayAttr, DenseArrayBase, IntegerAttr, TensorType, i64
    from xdsl.dialects.linalg.ops import TransposeOp

    def emit(operation):
        anchor.parent.insert_op_before(operation, anchor)
        return operation.results[0]

    source = proof.source
    source_shape = _shape(source)
    element = source.type.element_type
    groups = proof.shape[0] // proof.repeat
    shape = (groups, *source_shape[-2:])
    if source_shape != shape:
        reassociation = ArrayAttr([ArrayAttr([IntegerAttr(axis, i64) for axis in range(len(source_shape))])])
        flat = emit(tensor.CollapseShapeOp.create(
            operands=[source], result_types=[TensorType(element, [prod(source_shape)])],
            properties={"reassociation": reassociation}))
        reassociation = ArrayAttr([ArrayAttr([IntegerAttr(axis, i64) for axis in range(3)])])
        source = emit(tensor.ExpandShapeOp(flat, (), reassociation, shape, TensorType(element, shape)))
    if proof.matrix_transposed:
        out_type = TensorType(element, (groups, *proof.shape[1:]))
        empty = emit(tensor.EmptyOp([], out_type))
        source = emit(TransposeOp(source, empty, DenseArrayBase.from_list(i64, [0, 2, 1]), out_type))
    return source


@dataclass(frozen=True)
class RepeatedBatchConsumer:
    """Qualified provider alternative; empty ``calls`` selects every use."""

    symbol: str
    alternative: str
    argument: int
    repeat: int
    calls: tuple = ()
    compact: bool = False


def rewrite_repeated_batch_consumers(module, contracts):
    """Retarget proved read-only inputs without changing operands or output ABI.

    Explicit compact mode replaces only the selected input ABI with one matrix
    per group. The provider must expand reads by ``batch // repeat``. Pure source
    views/copies remain for other users; ordinary lowering may remove dead ones.
    The provider separately qualifies exact arithmetic, rounding, exceptional
    partial outputs, private retained preparation and synchronous completion.
    Every selected call is re-proved before any mutation; there is no operation
    movement or source view removal. Original declaration attributes are copied.
    """
    import json

    from xdsl.dialects.builtin import FunctionType, StringAttr, SymbolRefAttr, TensorType
    from xdsl.dialects.func import CallOp, FuncOp

    contracts = tuple(contracts)
    if not contracts:
        return []
    module.verify()
    declarations = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, FuncOp)}
    reserved = set(declarations)
    selected, plans = set(), []
    source_calls = tuple(op for op in module.walk() if isinstance(op, CallOp))
    for contract in contracts:
        if type(contract) is not RepeatedBatchConsumer:
            raise ValueError("batch reuse requires typed consumer selections")
        if type(contract.compact) is not bool:
            raise ValueError("compact batch input requires an explicit boolean provider contract")
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
        if type(contract.calls) is not tuple:
            raise ValueError("batch reuse call selection must be an explicit tuple")
        calls = list(contract.calls) if contract.calls else [op for op in source_calls
                                                            if op.callee.root_reference.data == contract.symbol]
        if not calls:
            raise ValueError("batch reuse selected consumer has no source calls")
        if (any(not isinstance(call, CallOp) for call in calls)
                or len(set(calls)) != len(calls)
                or any(call in selected or call not in source_calls
                       or call.callee.root_reference.data != contract.symbol for call in calls)):
            raise ValueError("batch reuse selected calls overlap or differ from the original declaration")
        selected.update(calls)
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
        if contract.compact:
            types = list(declaration.function_type.inputs)
            proof = proofs[0]
            types[contract.argument] = TensorType(proof.value.type.element_type,
                                                 (proof.shape[0] // proof.repeat, *proof.shape[1:]))
            alternative.properties["function_type"] = FunctionType.from_lists(types, declaration.function_type.outputs)
        alternative.attributes["merlin.repeated_batch_input"] = StringAttr(json.dumps({
            "argument": contract.argument, "repeat": contract.repeat,
            **({"compact_input": True} if contract.compact else {}),
            "provider_obligation": "original_results_rounding_effects_refusal_and_synchronous_private_preparation",
        }, sort_keys=True))
        declaration.parent.insert_op_before(alternative, declaration)
        for call, proof in zip(calls, proofs, strict=True):
            if contract.compact:
                arguments = list(call.arguments)
                arguments[contract.argument] = _compact_input(proof, call)
                call.operands = arguments
            call.properties["callee"] = SymbolRefAttr(contract.alternative)
        reports.append({"symbol": contract.symbol, "alternative": contract.alternative,
                        "argument": contract.argument, "calls": len(calls),
                        "input_proofs": [proof.to_dict() for proof in proofs],
                        **({"compact_input": True} if contract.compact else {})})
    module.verify()
    return reports

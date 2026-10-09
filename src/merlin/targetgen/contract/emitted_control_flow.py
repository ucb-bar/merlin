"""Opt-in structural observation of complete bounded emitted LLVM CFGs.

Block arguments and successor operands retain actual cyclic SSA joins. Per-block
operations are static inventory, never a dynamic execution order or trip count.
Raw typed properties, predicates and pointer indices are retained without
interpreting flags, addresses, source coverage or instruction effects. The
straight-line emitted_dataflow API deliberately keeps its prior refusal.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .emitted_dataflow import DataflowUnavailable


@dataclass(frozen=True)
class ControlFlowValue:
    ordinal: int
    block: int
    definition: int | None
    type: str
    bits: int


@dataclass(frozen=True)
class ControlFlowEdge:
    successor: int
    arguments: tuple[int, ...]
    parameters: tuple[int, ...]


@dataclass(frozen=True)
class ControlFlowOperation:
    ordinal: int
    block: int
    name: str
    operands: tuple[int, ...]
    results: tuple[int, ...]
    properties: tuple[tuple[str, str], ...]
    edges: tuple[ControlFlowEdge, ...]


@dataclass(frozen=True)
class ControlFlowBlock:
    ordinal: int
    arguments: tuple[int, ...]
    operations: tuple[int, ...]


@dataclass(frozen=True)
class EmittedControlFlow:
    source_sha256: str
    entry_symbol: str
    pointer_bits: int
    arguments: tuple[int, ...]
    values: tuple[ControlFlowValue, ...]
    blocks: tuple[ControlFlowBlock, ...]
    operations: tuple[ControlFlowOperation, ...]
    unknown: tuple[str, ...] = (
        "dynamic_execution_paths",
        "loop_trip_counts",
        "termination",
        "complete_output_stores",
        "pointer_ranges",
        "layout",
        "alias",
        "source_equivalence",
        "instruction_effects",
        "resources",
        "physical_runtime",
        "timing",
    )
    scope: str = "static CFG/typed SSA inventory only; no dynamic execution or semantic/runtime authority"


def observe_emitted_control_flow(text, *, entry_symbol, pointer_bits, max_blocks=256, max_operations=4096):
    """Retain every supported block, phi edge, typed definition and operation.

    Observation bounds limit only this reader. They do not establish compiled
    program resource bounds. This observer never folds phi values, assumes a
    unique pointer origin, evaluates a predicate or treats lexical block order
    as actual execution. Unsupported dispatch, regions and types refuse.
    """
    from xdsl.context import Context
    from xdsl.dialects import builtin, llvm
    from xdsl.irdl.dominance import DominanceInfo
    from xdsl.parser import Parser
    from xdsl.utils.exceptions import ParseError, VerifyException

    from .compile_only import require_pointer_entry

    if (
        type(text) is not str
        or type(entry_symbol) is not str
        or not entry_symbol.isascii()
        or not entry_symbol.isidentifier()
        or type(pointer_bits) is not int
        or not 1 <= pointer_bits <= 256
        or type(max_blocks) is not int
        or not 1 <= max_blocks <= 1024
        or type(max_operations) is not int
        or not 1 <= max_operations <= 100000
    ):
        raise DataflowUnavailable("CFG observation needs explicit bounded source/entry/pointer selections")
    context = Context()
    context.load_dialect(builtin.Builtin)
    context.load_dialect(llvm.LLVM)
    try:
        module = Parser(context, text).parse_module()
        module.verify()
    except (ParseError, VerifyException) as error:
        raise DataflowUnavailable("emitted LLVM CFG cannot be parsed and verified completely") from error
    members = tuple(module.body.block.ops)
    if module.attributes or module.properties or len(members) != 1 or type(members[0]) is not llvm.FuncOp:
        raise DataflowUnavailable("CFG observation requires one complete plain function without external dispatch")
    function = members[0]
    try:
        require_pointer_entry(text, entry_symbol=entry_symbol, pointer_arity=len(function.function_type.inputs))
    except ValueError as error:
        raise DataflowUnavailable("CFG entry lacks the selected plain pointer ABI") from error
    if (
        function.attributes
        or set(function.properties) - {"sym_name", "function_type", "CConv", "linkage", "visibility_", "unnamed_addr"}
        or any(
            type(function.properties[name]) is not builtin.IntegerAttr or function.properties[name].value.data != 0
            for name in ("visibility_", "unnamed_addr")
            if name in function.properties
        )
    ):
        raise DataflowUnavailable("CFG observation cannot interpret function metadata")
    blocks = tuple(function.body.blocks)
    operations = tuple(op for block in blocks for op in block.ops)
    if not blocks or len(blocks) > max_blocks or not operations or len(operations) > max_operations:
        raise DataflowUnavailable("CFG block/operation roster exceeds the selected observation bound")
    block_ids = {block: index for index, block in enumerate(blocks)}
    op_ids = {op: index for index, op in enumerate(operations)}
    definitions, values, locations = {}, [], {}

    def define(value, block, operation):
        if value in definitions:
            raise DataflowUnavailable("CFG SSA definition repeats")
        type_ = value.type
        if type(type_) is builtin.IntegerType and 1 <= type_.width.data <= 256:
            width = type_.width.data
        elif type(type_) is llvm.LLVMPointerType and type(type_.addr_space) is builtin.NoneAttr:
            width = pointer_bits
        else:
            raise DataflowUnavailable("CFG observation has an unsupported scalar/pointer type")
        ordinal = len(values)
        definitions[value] = ordinal
        locations[value] = (block, operation)
        values.append(
            ControlFlowValue(
                ordinal, block_ids[block], op_ids[operation] if operation is not None else None, str(type_), width
            )
        )

    for block in blocks:
        for argument in block.args:
            define(argument, block, None)
        for operation in block.ops:
            for result in operation.results:
                define(result, block, operation)
    # Structural edges are complete before the shared dominance computation;
    # opaque/successor-bearing operations cannot hide another graph.
    allowed = {
        "llvm.mlir.constant",
        "llvm.add",
        "llvm.sub",
        "llvm.mul",
        "llvm.or",
        "llvm.and",
        "llvm.shl",
        "llvm.ptrtoint",
        "llvm.inttoptr",
        "llvm.bitcast",
        "llvm.icmp",
        "llvm.getelementptr",
        "llvm.load",
        "llvm.store",
        "llvm.inline_asm",
        "llvm.br",
        "llvm.cond_br",
        "llvm.return",
    }
    for block in blocks:
        for op in block.ops:
            if op.name not in allowed or op.regions or op.attributes:
                raise DataflowUnavailable("unsupported CFG operation/metadata: " + op.name)
            if op.successors and type(op) not in {llvm.BrOp, llvm.CondBrOp}:
                raise DataflowUnavailable("CFG operation has unmodeled successor semantics")
            if any(successor not in block_ids or successor is blocks[0] for successor in op.successors):
                raise DataflowUnavailable("CFG successor escapes the function or reenters its pointer entry")
            if type(op) in {llvm.BrOp, llvm.CondBrOp, llvm.ReturnOp} and op is not block.last_op:
                raise DataflowUnavailable("CFG terminator precedes another operation")
        if type(block.last_op) not in {llvm.BrOp, llvm.CondBrOp, llvm.ReturnOp}:
            raise DataflowUnavailable("CFG block has no complete supported terminator")
    dominance = DominanceInfo(function.body)
    observed = []
    for block in blocks:
        for op in block.ops:
            for operand in op.operands:
                if operand not in locations:
                    raise DataflowUnavailable("CFG operand escapes its complete definition roster")
                defining_block, defining_op = locations[operand]
                if (defining_block is block and defining_op is not None and op_ids[defining_op] >= op_ids[op]) or (
                    defining_block is not block and not dominance.dominates(defining_block, block)
                ):
                    raise DataflowUnavailable("CFG operand definition does not dominate its use")
            edges = []
            branches = (
                ((op.successor, op.arguments),)
                if type(op) is llvm.BrOp
                else ((op.then_block, op.then_arguments), (op.else_block, op.else_arguments))
                if type(op) is llvm.CondBrOp
                else ()
            )
            for successor, arguments in branches:
                if tuple(arg.type for arg in arguments) != tuple(arg.type for arg in successor.args):
                    raise DataflowUnavailable("CFG edge omits or changes successor argument types")
                edges.append(
                    ControlFlowEdge(
                        block_ids[successor],
                        tuple(definitions[v] for v in arguments),
                        tuple(definitions[v] for v in successor.args),
                    )
                )
            if type(op) is llvm.ReturnOp and (op.operands or op.results or op.properties):
                raise DataflowUnavailable("CFG return differs from the original complete void pointer boundary")
            observed.append(
                ControlFlowOperation(
                    op_ids[op],
                    block_ids[block],
                    op.name,
                    tuple(definitions[v] for v in op.operands),
                    tuple(definitions[v] for v in op.results),
                    tuple((name, str(value)) for name, value in sorted(op.properties.items())),
                    tuple(edges),
                )
            )
    return EmittedControlFlow(
        hashlib.sha256(text.encode()).hexdigest(),
        entry_symbol,
        pointer_bits,
        tuple(definitions[value] for value in blocks[0].args),
        tuple(values),
        tuple(
            ControlFlowBlock(
                block_ids[block], tuple(definitions[v] for v in block.args), tuple(op_ids[op] for op in block.ops)
            )
            for block in blocks
        ),
        tuple(observed),
    )

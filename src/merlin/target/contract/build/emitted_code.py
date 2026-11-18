"""Typed observations of closed straight-line emitted LLVM dataflow.

The observer records actual SSA definitions, memory actions and input-only
inline assembly. It interprets no instruction template, target operand, source
operation or hardware effect. Those interpretations require independently
selected source owners. Unsupported IR refuses; an observation is never a
source, layout, alias, execution, resource or correctness qualification.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# ---- emitted_dataflow --------------------------------------------------------------------------


class DataflowUnavailable(ValueError):
    """The complete emitted body cannot be followed in the supported subset."""


@dataclass(frozen=True)
class EmittedValue:
    ordinal: int
    kind: str
    type: str
    bits: int | None
    operands: tuple[int, ...] = ()
    constant: int | None = None
    argument: int | None = None


@dataclass(frozen=True)
class EmittedAction:
    ordinal: int
    kind: str
    operands: tuple[int, ...]
    results: tuple[int, ...] = ()
    assembly: str | None = None
    constraints: str | None = None
    side_effects: bool | None = None
    alignment: int | None = None
    volatile: bool | None = None


@dataclass(frozen=True)
class EmittedDataflow:
    source_sha256: str
    entry_symbol: str
    pointer_bits: int
    arguments: tuple[int, ...]
    values: tuple[EmittedValue, ...]
    actions: tuple[EmittedAction, ...]
    scope: str = "complete supported emitted SSA/actions only; no source/ISA/effect/runtime authority"

    def value(self, ordinal):
        if type(ordinal) is not int or not 0 <= ordinal < len(self.values):
            raise DataflowUnavailable("emitted value is outside the complete definition roster")
        return self.values[ordinal]

    def constant(self, ordinal):
        """Evaluate only defined constant bit-vector expressions, with no ISA meaning."""
        return self.value(ordinal).constant

    def argument_origin(self, ordinal):
        """Observe an unchanged argument through width-preserving casts/identities.

        Nonzero offsets, truncation, load values and unknown expressions have no
        observed argument origin. No address range or pointer legality follows.
        """
        for _ in range(len(self.values)):
            value = self.value(ordinal)
            if value.kind == "argument":
                return value.argument
            if value.kind in {"llvm.ptrtoint", "llvm.inttoptr", "llvm.bitcast"}:
                ordinal = value.operands[0]
            elif value.kind in {"llvm.add", "llvm.or"}:
                bases = [
                    base
                    for base, other in (value.operands, tuple(reversed(value.operands)))
                    if self.constant(other) == 0
                ]
                if not bases:
                    return None
                ordinal = bases[0]
            elif value.kind == "llvm.sub" and self.constant(value.operands[1]) == 0:
                ordinal = value.operands[0]
            else:
                return None
        raise DataflowUnavailable("emitted argument-origin chain is cyclic or exceeds its roster")


def observe_emitted_dataflow(text, *, entry_symbol, pointer_bits, max_operations=4096):
    """Parse actual LLVM bytes, retaining every supported action in program order.

    ``pointer_bits`` is an explicit caller selection, not inferred from emitted
    metadata. Bounds limit the observer, not the compiled program's resources.
    No selected backend, evaluator, default target or instruction table is used.
    """
    from xdsl.context import Context
    from xdsl.dialects import builtin, llvm
    from xdsl.parser import Parser
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.target.contract.build.services import require_pointer_entry

    if (
        type(text) is not str
        or type(entry_symbol) is not str
        or not entry_symbol.isascii()
        or not entry_symbol.isidentifier()
        or type(pointer_bits) is not int
        or not 1 <= pointer_bits <= 256
        or type(max_operations) is not int
        or not 1 <= max_operations <= 100000
    ):
        raise DataflowUnavailable("emitted observation needs explicit bounded source/entry/pointer selections")
    context = Context()
    context.load_dialect(builtin.Builtin)
    context.load_dialect(llvm.LLVM)
    try:
        module = Parser(context, text).parse_module()
        module.verify()
    except (ParseError, VerifyException) as error:
        raise DataflowUnavailable("emitted LLVM cannot be parsed and verified completely") from error
    if module.attributes or module.properties:
        raise DataflowUnavailable("uninterpreted emitted module metadata")
    members = tuple(module.body.block.ops)
    if len(members) != 1 or type(members[0]) is not llvm.FuncOp:
        raise DataflowUnavailable("emitted observation requires one complete function without external dispatch")
    function = members[0]
    try:
        require_pointer_entry(text, entry_symbol=entry_symbol, pointer_arity=len(function.function_type.inputs))
    except ValueError as error:
        raise DataflowUnavailable("emitted entry lacks the selected plain pointer ABI") from error
    if (
        function.attributes
        or set(function.properties) - {"sym_name", "function_type", "CConv", "linkage", "visibility_", "unnamed_addr"}
        or any(
            type(function.properties.get(name)) is not builtin.IntegerAttr or function.properties[name].value.data != 0
            for name in ("visibility_", "unnamed_addr")
            if name in function.properties
        )
    ):
        raise DataflowUnavailable("uninterpreted emitted function metadata")
    if len(function.body.blocks) != 1:
        raise DataflowUnavailable("emitted control-flow join is unsupported")
    block = function.body.block
    operations = tuple(block.ops)
    if not operations or len(operations) > max_operations:
        raise DataflowUnavailable("emitted operation roster is empty or exceeds the selected observation bound")
    values, actions, definitions = [], [], {}

    def bits(value):
        type_ = value.type
        if type(type_) is builtin.IntegerType and 1 <= type_.width.data <= 256:
            return type_.width.data
        if type(type_) is llvm.LLVMPointerType and type(type_.addr_space) is builtin.NoneAttr:
            return pointer_bits
        raise DataflowUnavailable("emitted scalar or pointer type is unsupported")

    def define(value, kind, operands=(), *, constant=None, argument=None):
        if value in definitions:
            raise DataflowUnavailable("emitted SSA definition repeats")
        ordinal = len(values)
        definitions[value] = ordinal
        width = bits(value)
        if kind in {"llvm.add", "llvm.sub", "llvm.mul", "llvm.or", "llvm.and", "llvm.shl"}:
            left, right = (values[index].constant for index in operands)
            if left is not None and right is not None:
                if kind == "llvm.shl" and right >= width:
                    raise DataflowUnavailable("constant shift is outside defined LLVM bit-vector semantics")
                constant = {
                    "llvm.add": lambda: left + right,
                    "llvm.sub": lambda: left - right,
                    "llvm.mul": lambda: left * right,
                    "llvm.or": lambda: left | right,
                    "llvm.and": lambda: left & right,
                    "llvm.shl": lambda: left << right,
                }[kind]() % (1 << width)
        values.append(EmittedValue(ordinal, kind, str(value.type), width, operands, constant, argument))
        return ordinal

    def operands(op):
        if any(value not in definitions for value in op.operands):
            raise DataflowUnavailable("emitted operand has no preceding definition")
        return tuple(definitions[value] for value in op.operands)

    def no_flags(op):
        if op.attributes or op.regions or op.successors:
            raise DataflowUnavailable("emitted attributes, regions or successors are unsupported")

    arguments = tuple(define(value, "argument", argument=index) for index, value in enumerate(block.args))
    for index, op in enumerate(operations):
        no_flags(op)
        args = operands(op)
        if type(op) is llvm.ConstantOp:
            value = op.properties.get("value")
            if (
                set(op.properties) != {"value"}
                or args
                or len(op.results) != 1
                or type(value) is not builtin.IntegerAttr
                or type(op.results[0].type) is not builtin.IntegerType
                or value.type != op.results[0].type
            ):
                raise DataflowUnavailable("emitted constant has unproved scalar semantics")
            define(op.results[0], "constant", constant=value.value.data % (1 << bits(op.results[0])))
        elif op.name in {"llvm.add", "llvm.sub", "llvm.mul", "llvm.or", "llvm.and", "llvm.shl"}:
            flags = op.properties
            if (
                len(args) != 2
                or len(op.results) != 1
                or type(op.results[0].type) is not builtin.IntegerType
                or any(value.type != op.results[0].type for value in op.operands)
                or set(flags) - {"overflowFlags"}
                or "overflowFlags" in flags
                and (type(flags["overflowFlags"]) is not builtin.IntegerAttr or flags["overflowFlags"].value.data != 0)
            ):
                raise DataflowUnavailable("emitted integer operation has unsupported flags or types")
            define(op.results[0], op.name, args)
        elif op.name in {"llvm.ptrtoint", "llvm.inttoptr", "llvm.bitcast"}:
            if len(args) != 1 or len(op.results) != 1 or op.properties:
                raise DataflowUnavailable("emitted cast has unsupported properties or arity")
            source, target = op.operands[0].type, op.results[0].type

            def pointer(value):
                return type(value) is llvm.LLVMPointerType and type(value.addr_space) is builtin.NoneAttr

            valid = (
                op.name == "llvm.ptrtoint"
                and pointer(source)
                and type(target) is builtin.IntegerType
                or op.name == "llvm.inttoptr"
                and type(source) is builtin.IntegerType
                and pointer(target)
                or op.name == "llvm.bitcast"
                and pointer(source)
                and pointer(target)
            )
            if not valid or bits(op.operands[0]) != pointer_bits or bits(op.results[0]) != pointer_bits:
                raise DataflowUnavailable("emitted cast loses the explicitly selected pointer width")
            define(op.results[0], op.name, args)
        elif type(op) in {llvm.LoadOp, llvm.StoreOp}:
            allowed = {"alignment", "ordering", "volatile_"}
            ordering = op.properties.get("ordering")
            if (
                set(op.properties) - allowed
                or ordering is not None
                and (type(ordering) is not builtin.IntegerAttr or ordering.value.data != 0)
                or "volatile_" in op.properties
                and type(op.properties["volatile_"]) is not builtin.UnitAttr
            ):
                raise DataflowUnavailable("atomic or unmodeled emitted memory semantics")
            alignment = op.properties.get("alignment")
            if alignment is not None and type(alignment) is not builtin.IntegerAttr:
                raise DataflowUnavailable("emitted memory alignment lacks an integer declaration")
            alignment = alignment.value.data if alignment is not None else None
            if alignment is not None and (alignment < 1 or alignment & (alignment - 1)):
                raise DataflowUnavailable("emitted memory alignment is malformed")
            if type(op) is llvm.LoadOp:
                if len(args) != 1 or len(op.results) != 1 or type(op.results[0].type) is not builtin.IntegerType:
                    raise DataflowUnavailable("emitted load has unsupported type or arity")
                result = define(op.results[0], "memory_read", args)
                actions.append(
                    EmittedAction(
                        index, "load", args, (result,), alignment=alignment, volatile="volatile_" in op.properties
                    )
                )
            else:
                if len(args) != 2 or op.results or type(op.operands[0].type) is not builtin.IntegerType:
                    raise DataflowUnavailable("emitted store has unsupported type or arity")
                actions.append(
                    EmittedAction(index, "store", args, alignment=alignment, volatile="volatile_" in op.properties)
                )
        elif type(op) is llvm.InlineAsmOp:
            tail = op.properties.get("tail_call_kind")
            if (
                op.results
                or set(op.properties) - {"asm_string", "constraints", "has_side_effects", "tail_call_kind"}
                or type(op.properties.get("asm_string")) is not builtin.StringAttr
                or type(op.properties.get("constraints")) is not builtin.StringAttr
                or "has_side_effects" in op.properties
                and type(op.properties["has_side_effects"]) is not builtin.UnitAttr
                or tail is not None
                and (type(tail) is not llvm.TailCallKindAttr or tail.data is not llvm.TailCallKind.NONE)
            ):
                raise DataflowUnavailable("emitted assembly results/options are unsupported")
            constraints = op.constraints.data
            fields = tuple(field.strip() for field in constraints.split(",")) if constraints else ()
            inputs = tuple(field for field in fields if not field.startswith("~{"))
            if (
                len(inputs) != len(args)
                or any(field not in {"r", "i"} for field in inputs)
                or any(field.startswith("~{") and (not field.endswith("}") or len(field) <= 3) for field in fields)
                or any(not field.startswith("~{") for field in fields[len(inputs) :])
            ):
                raise DataflowUnavailable("emitted assembly input/clobber constraints are unsupported")
            actions.append(
                EmittedAction(
                    index,
                    "assembly",
                    args,
                    assembly=op.asm_string.data,
                    constraints=constraints,
                    side_effects="has_side_effects" in op.properties,
                )
            )
        elif type(op) is llvm.ReturnOp:
            if args or op.results or op.properties or index != len(operations) - 1:
                raise DataflowUnavailable("emitted return is not the complete final void boundary")
            actions.append(EmittedAction(index, "return", ()))
        else:
            raise DataflowUnavailable("unsupported emitted operation: " + op.name)
    if type(operations[-1]) is not llvm.ReturnOp:
        raise DataflowUnavailable("emitted body has no complete return")
    return EmittedDataflow(
        hashlib.sha256(text.encode()).hexdigest(), entry_symbol, pointer_bits, arguments, tuple(values), tuple(actions)
    )


# ---- emitted_control_flow ----------------------------------------------------------------------
# Opt-in structural observation of complete bounded emitted LLVM CFGs.
#
# Block arguments and successor operands retain actual cyclic SSA joins. Per-block
# operations are static inventory, never a dynamic execution order or trip count.
# Raw typed properties, predicates and pointer indices are retained without
# interpreting flags, addresses, source coverage or instruction effects. The
# straight-line emitted_dataflow API deliberately keeps its prior refusal.


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


def observe_emitted_control_flow(
    text,
    *,
    entry_symbol,
    pointer_bits,
    max_blocks=256,
    max_operations=4096,
    max_source_bytes=2000000,
    max_nesting=64,
    max_values=None,
    max_edges=None,
    max_integer_bits=256,
):
    """Retain every supported block, phi edge, typed definition and operation.

    Source/lexical limits run before parser allocation. Observation bounds limit
    only this reader, without establishing compiled program resource bounds.
    This observer never folds phi values, assumes a
    unique pointer origin, evaluates a predicate or treats lexical block order
    as actual execution. Unsupported dispatch, regions and types refuse.
    """
    from xdsl.context import Context
    from xdsl.dialects import builtin, llvm
    from xdsl.irdl.dominance import DominanceInfo
    from xdsl.parser import Parser
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.target.contract.build.services import require_pointer_entry
    from merlin.target.contract.interfaces.grammar import MlirSourceUnavailable, admit_mlir_source

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
        or any(value is not None and (type(value) is not int or value <= 0) for value in (max_values, max_edges))
        or type(max_integer_bits) is not int
        or not 1 <= max_integer_bits <= 256
    ):
        raise DataflowUnavailable("CFG observation needs explicit bounded source/entry/pointer selections")
    try:
        admit_mlir_source(
            text,
            max_source_bytes=max_source_bytes,
            max_nesting=max_nesting,
            max_integer_bits=max_integer_bits,
            allow_dense=False,
            allow_dense_resource=False,
        )
    except MlirSourceUnavailable as error:
        raise DataflowUnavailable(str(error)) from error
    context = Context()
    context.load_dialect(builtin.Builtin)
    context.load_dialect(llvm.LLVM)
    try:
        module = Parser(context, text).parse_module()
        module.verify()
    except (ParseError, VerifyException, RecursionError) as error:
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
    if (
        max_values is not None
        and sum(len(block.args) for block in blocks) + sum(len(op.results) for op in operations) > max_values
        or max_edges is not None
        and sum(len(op.successors) for op in operations) > max_edges
    ):
        raise DataflowUnavailable("CFG value/edge roster exceeds the selected observation bound")
    block_ids = {block: index for index, block in enumerate(blocks)}
    op_ids = {op: index for index, op in enumerate(operations)}
    definitions, values, locations = {}, [], {}

    def define(value, block, operation):
        if value in definitions:
            raise DataflowUnavailable("CFG SSA definition repeats")
        type_ = value.type
        if type(type_) is builtin.IntegerType and 1 <= type_.width.data <= max_integer_bits:
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

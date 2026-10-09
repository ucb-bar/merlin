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

    from .compile_only import require_pointer_entry

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

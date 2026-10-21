"""Exact bounded traversal of a small, statically resolvable LLVM CFG.

Clients supply values for arguments and select opaque operations with no used results
to observe. Integer arithmetic follows declared-width wrap semantics. Memory
loads, calls producing values, unsupported pointer layouts and unresolved
control flow are refused; a partial trace never certifies a complete count.
No target instruction or device ABI is interpreted here.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerType
from xdsl.ir import Operation
from xdsl.traits import SymbolTable


class StaticTraceError(ValueError):
    """The trace cannot be followed exactly under the supplied inputs."""


@dataclass(frozen=True)
class StaticInt:
    value: int
    bits: int

    def __post_init__(self):
        if type(self.value) is not int:
            raise StaticTraceError("integer value required")
        if type(self.bits) is not int or self.bits <= 0:
            raise StaticTraceError("integer width must be positive")
        object.__setattr__(self, "value", self.value % (1 << self.bits))

    @property
    def signed(self) -> int:
        return self.value - (1 << self.bits) if self.value >= (1 << (self.bits - 1)) else self.value


@dataclass(frozen=True)
class StaticPointer:
    base: object
    offset: StaticInt


@dataclass(frozen=True)
class StaticAddress:
    """Integer representation of a symbolic pointer, not a known integer base."""

    base: object
    offset: StaticInt


StaticValue = StaticInt | StaticPointer | StaticAddress


@dataclass(frozen=True)
class StaticExecutionStep:
    operation: Operation
    inputs: tuple[StaticValue, ...]


def _integer(value: StaticValue) -> StaticInt:
    if not isinstance(value, StaticInt):
        raise StaticTraceError("integer operand required")
    return value


def _bits(typ) -> int:
    if not isinstance(typ, IntegerType):
        raise StaticTraceError("only integer scalar values can be evaluated")
    return typ.width.data


def _flags_absent(op: Operation) -> None:
    if op.properties.get("isDisjoint") is not None:
        raise StaticTraceError(f"{op.name}: poison-producing arithmetic flags unsupported")
    for key in ("overflowFlags", "nonNeg", "noWrapFlags"):
        value = op.properties.get(key)
        if value is not None and getattr(getattr(value, "value", None), "data", None) != 0:
            raise StaticTraceError(f"{op.name}: poison-producing arithmetic flags unsupported")


def trace_static_function(
    function: llvm.FuncOp,
    arguments: Sequence[StaticValue],
    *,
    observe: Callable[[Operation], bool],
    pointer_index_bits: int,
    pointer_bits: int | None = None,
    global_addresses: Mapping[str, StaticPointer] | None = None,
    max_steps: int = 10000000,
) -> Iterator[StaticExecutionStep]:
    """Yield selected operations in execution order or raise on any uncertainty.

    Pointer index width is an explicit data-layout fact. Symbolic base identities
    are bookkeeping only and confer no alignment, storage extent or noalias
    permission. Global addresses must name declared globals; no storage is inferred.
    Symbolic ptrtoint requires an explicit pointer representation width, currently
    equal to the index and result widths. Only constant add/sub and a same-width
    inttoptr roundtrip preserve these symbolic integers; their absolute bits,
    comparisons and bitwise arithmetic are unknown. Addresses are never dereferenced.
    The caller supplies verified IR. Consuming the iterator to completion is
    required: selected operations yielded before an error form a partial trace.
    """
    if type(pointer_index_bits) is not int or pointer_index_bits <= 0:
        raise StaticTraceError("positive pointer index width required")
    if pointer_bits is not None and (type(pointer_bits) is not int or pointer_bits <= 0):
        raise StaticTraceError("positive pointer representation width required")
    if type(max_steps) is not int or max_steps <= 0:
        raise StaticTraceError("positive trace limit required")
    if not function.body.blocks:
        raise StaticTraceError("function body required")
    block = function.body.blocks.first
    if len(arguments) != len(block.args):
        raise StaticTraceError("argument count differs")
    for argument, value in zip(block.args, arguments):
        if isinstance(argument.type, IntegerType):
            if not isinstance(value, StaticInt) or value.bits != argument.type.width.data:
                raise StaticTraceError("integer argument type differs")
        elif isinstance(argument.type, llvm.LLVMPointerType):
            if not isinstance(value, StaticPointer) or value.offset.bits != pointer_index_bits:
                raise StaticTraceError("pointer argument type or index width differs")
        else:
            raise StaticTraceError("unsupported static argument type")
    values = dict(zip(block.args, arguments))
    steps = 0
    globals_snapshot = dict(global_addresses) if global_addresses is not None else {}
    global_state = {}
    for name, pointer in globals_snapshot.items():
        if type(name) is not str or not name or not isinstance(pointer, StaticPointer):
            raise StaticTraceError("global mapping requires symbol names and static pointers")
        if not isinstance(pointer.offset, StaticInt) or pointer.offset.bits != pointer_index_bits:
            raise StaticTraceError("global pointer index width differs")
        declared = SymbolTable.lookup_symbol(function, name)
        if not isinstance(declared, llvm.GlobalOp) or declared.thread_local_ is not None:
            raise StaticTraceError("global mapping requires a declared non-thread-local global")
        if declared.addr_space.value.data != 0:
            raise StaticTraceError("global address space is unsupported")
        global_state[name] = (pointer.base, pointer.offset, pointer.offset.value, pointer.offset.bits)

    def check_globals():
        if global_addresses is None:
            return
        if set(global_addresses) != set(globals_snapshot):
            raise StaticTraceError("global mapping changed during trace")
        for name, pointer in globals_snapshot.items():
            base, offset, value, bits = global_state[name]
            if (
                global_addresses[name] is not pointer
                or pointer.base is not base
                or pointer.offset is not offset
                or offset.value != value
                or offset.bits != bits
            ):
                raise StaticTraceError("global mapping changed during trace")

    def address_width(bits):
        if pointer_bits != pointer_index_bits or bits != pointer_bits:
            raise StaticTraceError("symbolic pointer conversion requires explicit equal pointer widths")

    def read(value):
        if value not in values:
            raise StaticTraceError("SSA operand is not statically resolved")
        return values[value]

    while True:
        jump = False
        for op in block.ops:
            check_globals()
            steps += 1
            if steps > max_steps:
                raise StaticTraceError("static trace step limit exceeded")
            inputs = tuple(read(v) for v in op.operands)
            selected = observe(op)
            check_globals()
            if selected:
                unused_void = (
                    len(op.results) == 1
                    and isinstance(op.results[0].type, llvm.LLVMVoidType)
                    and not op.results[0].uses
                )
                if (op.results and not unused_void) or op.regions or op.successors or op.has_trait(llvm.IsTerminator):
                    raise StaticTraceError("observed operation must be result-free or unused void and non-control")
                yield StaticExecutionStep(op, inputs)
                check_globals()
                continue
            _flags_absent(op)
            result = None
            if op.name == "llvm.mlir.constant":
                attr = op.properties.get("value")
                if not hasattr(attr, "value") or not hasattr(attr.value, "data"):
                    raise StaticTraceError("only integer constants can be evaluated")
                result = StaticInt(attr.value.data, _bits(op.results[0].type))
            elif isinstance(op, (llvm.AddOp, llvm.SubOp, llvm.MulOp)):
                if any(isinstance(v, StaticAddress) for v in inputs):
                    a, b = inputs
                    if isinstance(op, llvm.AddOp) and isinstance(b, StaticAddress):
                        a, b = b, a
                    if (
                        not isinstance(a, StaticAddress)
                        or not isinstance(b, StaticInt)
                        or isinstance(op, llvm.MulOp)
                        or a.offset.bits != b.bits
                        or _bits(op.results[0].type) != a.offset.bits
                    ):
                        raise StaticTraceError("unsupported symbolic address arithmetic")
                    delta = -b.value if isinstance(op, llvm.SubOp) else b.value
                    result = StaticAddress(a.base, StaticInt(a.offset.value + delta, a.offset.bits))
                    values[op.results[0]] = result
                    continue
                a, b = map(_integer, inputs)
                if a.bits != b.bits:
                    raise StaticTraceError("integer operand widths differ")
                if isinstance(op, llvm.AddOp):
                    value = a.value + b.value
                elif isinstance(op, llvm.SubOp):
                    value = a.value - b.value
                else:
                    value = a.value * b.value
                result = StaticInt(value, _bits(op.results[0].type))
            elif isinstance(op, (llvm.AndOp, llvm.OrOp, llvm.XOrOp)):
                a, b = map(_integer, inputs)
                bits = _bits(op.results[0].type)
                if a.bits != b.bits or a.bits != bits:
                    raise StaticTraceError("integer operand or result widths differ")
                if isinstance(op, llvm.AndOp):
                    value = a.value & b.value
                elif isinstance(op, llvm.OrOp):
                    value = a.value | b.value
                else:
                    value = a.value ^ b.value
                result = StaticInt(value, bits)
            elif isinstance(op, (llvm.SExtOp, llvm.ZExtOp, llvm.TruncOp)):
                value = _integer(inputs[0])
                result = StaticInt(
                    value.signed if isinstance(op, llvm.SExtOp) else value.value, _bits(op.results[0].type)
                )
            elif isinstance(op, llvm.IntToPtrOp):
                if isinstance(inputs[0], StaticAddress):
                    address_width(inputs[0].offset.bits)
                    if getattr(op.results[0].type.addr_space, "data", 0) != 0:
                        raise StaticTraceError("symbolic pointer address space is unsupported")
                    result = StaticPointer(inputs[0].base, inputs[0].offset)
                else:
                    result = StaticPointer(None, StaticInt(_integer(inputs[0]).value, pointer_index_bits))
            elif isinstance(op, llvm.PtrToIntOp):
                address_width(_bits(op.results[0].type))
                if getattr(op.operands[0].type.addr_space, "data", 0) != 0:
                    raise StaticTraceError("symbolic pointer address space is unsupported")
                pointer = inputs[0]
                if not isinstance(pointer, StaticPointer) or pointer.offset.bits != pointer_bits:
                    raise StaticTraceError("pointer conversion index width differs")
                result = StaticAddress(pointer.base, pointer.offset)
            elif isinstance(op, llvm.AddressOfOp):
                name = op.global_name
                if name.nested_references or name.root_reference.data not in globals_snapshot:
                    raise StaticTraceError("global address has no explicit binding")
                if getattr(op.result.type.addr_space, "data", 0) != 0:
                    raise StaticTraceError("global address space is unsupported")
                result = globals_snapshot[name.root_reference.data]
            elif isinstance(op, llvm.GEPOp):
                if op.properties.get("inbounds") is not None:
                    raise StaticTraceError("inbounds pointer extent proof is not supplied")
                pointer = inputs[0]
                if not isinstance(pointer, StaticPointer) or pointer.offset.bits != pointer_index_bits:
                    raise StaticTraceError("pointer index width differs")
                element = op.properties["elem_type"]
                if not isinstance(element, IntegerType) or element.width.data % 8:
                    raise StaticTraceError("GEP requires a byte-sized integer element layout")
                indices = tuple(op.properties["rawConstantIndices"].iter_values())
                if len(indices) != 1:
                    raise StaticTraceError("only one-dimensional GEP is supported")
                index = _integer(inputs[1]).signed if indices[0] == llvm.GEP_USE_SSA_VAL else indices[0]
                result = StaticPointer(
                    pointer.base,
                    StaticInt(pointer.offset.value + index * (element.width.data // 8), pointer_index_bits),
                )
            elif isinstance(op, llvm.ICmpOp):
                a, b = map(_integer, inputs)
                if a.bits != b.bits:
                    raise StaticTraceError("comparison operand widths differ")
                predicate = llvm.ICmpPredicateFlag.from_int(op.predicate.value.data)
                signed = predicate in (
                    llvm.ICmpPredicateFlag.SLT,
                    llvm.ICmpPredicateFlag.SLE,
                    llvm.ICmpPredicateFlag.SGT,
                    llvm.ICmpPredicateFlag.SGE,
                )
                x, y = (a.signed, b.signed) if signed else (a.value, b.value)
                comparison = {
                    llvm.ICmpPredicateFlag.EQ: x == y,
                    llvm.ICmpPredicateFlag.NE: x != y,
                    llvm.ICmpPredicateFlag.SLT: x < y,
                    llvm.ICmpPredicateFlag.ULT: x < y,
                    llvm.ICmpPredicateFlag.SLE: x <= y,
                    llvm.ICmpPredicateFlag.ULE: x <= y,
                    llvm.ICmpPredicateFlag.SGT: x > y,
                    llvm.ICmpPredicateFlag.UGT: x > y,
                    llvm.ICmpPredicateFlag.SGE: x >= y,
                    llvm.ICmpPredicateFlag.UGE: x >= y,
                }[predicate]
                result = StaticInt(int(comparison), 1)
            elif isinstance(op, llvm.BrOp):
                block = op.successors[0]
                if len(block.args) != len(inputs):
                    raise StaticTraceError("branch arguments differ")
                values.update(zip(block.args, inputs))
                jump = True
                break
            elif isinstance(op, llvm.CondBrOp):
                cond = _integer(read(op.cond))
                if cond.bits != 1:
                    raise StaticTraceError("branch condition must be i1")
                target, operands = (
                    (op.then_block, op.then_arguments) if cond.value else (op.else_block, op.else_arguments)
                )
                transferred = tuple(read(v) for v in operands)
                if len(target.args) != len(transferred):
                    raise StaticTraceError("conditional branch arguments differ")
                block = target
                values.update(zip(block.args, transferred))
                jump = True
                break
            elif isinstance(op, llvm.ReturnOp):
                return
            else:
                raise StaticTraceError(f"unsupported static operation: {op.name}")
            if result is not None:
                values[op.results[0]] = result
        if not jump:
            raise StaticTraceError("CFG block lacks a followed terminator")

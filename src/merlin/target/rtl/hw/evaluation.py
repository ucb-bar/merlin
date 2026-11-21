"""Bounded two-state evaluation of an explicit, closed combinational HW body.

This is a local source-expression observation, not an independent target runtime
model. Instances, state, memories and unrecognized operations refuse. Complete
output values do not assign resource roles, address spaces, tensor axes, capacity,
protocol, four-state behavior or physical/timing correspondence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    Signedness,
    StringAttr,
    UnitAttr,
    UnregisteredAttr,
)
from xdsl.dialects.comb import ICMP_COMPARISON_OPERATIONS

from merlin.target.contract.interfaces.grammar import admit_mlir_source
from merlin.target.rtl.firrtl.ports import _hw_port_entries
from merlin.target.rtl.hw.graph import _attribute, _integer, _name, parse_generic_hw

# ---- hw_combinational --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationLimits:
    """Explicit limits; nodes count input/result/output slots, not cycle cost."""

    source_bytes: int
    nodes: int
    scalar_bits: int
    cases: int
    bit_work: int

    def __post_init__(self):
        for value in (self.source_bytes, self.nodes, self.scalar_bits, self.cases, self.bit_work):
            if type(value) is not int or value <= 0:
                raise ValueError("combinational observation limits must be positive integers")


@dataclass(frozen=True)
class ScalarPort:
    name: str
    width: int


@dataclass(frozen=True)
class _Expression:
    kind: str
    width: int
    operands: tuple[int, ...]
    parameter: int | None = None


@dataclass(frozen=True)
class PreparedCombinationalObservation:
    """Immutable local expression; constructing it grants no admission authority."""

    source_sha256: str
    module: str
    inputs: tuple[ScalarPort, ...]
    outputs: tuple[ScalarPort, ...]
    expressions: tuple[_Expression, ...]
    output_values: tuple[int, ...]
    per_case_bit_work: int
    limits: EvaluationLimits

    def evaluate(self, cases: Sequence[Mapping[str, int]]) -> tuple[dict[str, int], ...]:
        """Validate the whole stimulus roster/budget before allocating results."""
        if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
            raise ValueError("combinational stimuli require a bounded explicit sequence")
        count = len(cases)
        if count > self.limits.cases or count * self.per_case_bit_work > self.limits.bit_work:
            raise ValueError("combinational observation exceeds its case or bit-work budget")
        names = {port.name for port in self.inputs}
        frozen = []
        for ordinal in range(count):
            case = cases[ordinal]
            if not isinstance(case, Mapping) or set(case) != names:
                raise ValueError("combinational stimulus must name every original input exactly")
            row = []
            for port in self.inputs:
                value = case[port.name]
                if type(value) is not int or not 0 <= value < 1 << port.width:
                    raise ValueError("combinational stimulus is outside its original unsigned bitvector")
                row.append(value)
            frozen.append(tuple(row))
        if len(cases) != count:
            raise ValueError("combinational original stimulus roster changed during validation")
        result = []
        for case in frozen:
            values = list(case)
            widths = [port.width for port in self.inputs]
            for expression in self.expressions:
                args = [values[index] for index in expression.operands]
                if expression.kind == "hw.constant":
                    value = expression.parameter
                elif expression.kind == "comb.extract":
                    value = args[0] >> expression.parameter
                elif expression.kind == "comb.concat":
                    value = 0
                    for index, arg in zip(expression.operands, args, strict=True):
                        value = (value << widths[index]) | arg
                elif expression.kind == "comb.replicate":
                    value = 0
                    for _ in range(expression.parameter):
                        value = (value << widths[expression.operands[0]]) | args[0]
                elif expression.kind == "comb.shru":
                    # The public Comb fold defines overshifts as zero. Avoid
                    # passing an unbounded shift count to the host operator.
                    value = 0 if args[1] >= expression.width else args[0] >> args[1]
                elif expression.kind == "seq.from_clock":
                    value = args[0]
                elif expression.kind == "comb.add":
                    value = sum(args)
                elif expression.kind == "comb.and":
                    value = args[0]
                    for arg in args[1:]:
                        value &= arg
                elif expression.kind in {"comb.or", "comb.xor"}:
                    value = args[0]
                    for arg in args[1:]:
                        value = value | arg if expression.kind == "comb.or" else value ^ arg
                elif expression.kind == "comb.sub":
                    value = args[0] - args[1]
                elif expression.kind == "comb.mux":
                    value = args[1] if args[0] else args[2]
                elif expression.kind == "comb.icmp":
                    value = _compare(args, widths[expression.operands[0]], expression.parameter)
                else:
                    raise ValueError("unsupported prepared combinational expression")
                values.append(value & ((1 << expression.width) - 1))
                widths.append(expression.width)
            result.append(
                {port.name: values[index] for port, index in zip(self.outputs, self.output_values, strict=True)}
            )
        return tuple(result)


def _compare(args, width, predicate):
    # Older prepared equality expressions used None. Fresh preparation keeps
    # the original predicate; pinned source records are never upgraded here.
    if predicate is None:
        predicate = ICMP_COMPARISON_OPERATIONS.index("eq")
    if type(predicate) is not int or not 0 <= predicate < len(ICMP_COMPARISON_OPERATIONS):
        raise ValueError("unsupported prepared integer comparison predicate")
    name = ICMP_COMPARISON_OPERATIONS[predicate]
    left, right = args
    if name in {"slt", "sle", "sgt", "sge"}:
        sign, domain = 1 << (width - 1), 1 << width
        left = left - domain if left & sign else left
        right = right - domain if right & sign else right
    if name == "eq":
        return int(left == right)
    if name == "ne":
        return int(left != right)
    if name in {"slt", "ult"}:
        return int(left < right)
    if name in {"sle", "ule"}:
        return int(left <= right)
    if name in {"sgt", "ugt"}:
        return int(left > right)
    if name in {"sge", "uge"}:
        return int(left >= right)
    raise ValueError("unsupported prepared integer comparison predicate")


def _width(value, limits):
    typ = value.type
    if (
        not isinstance(typ, IntegerType)
        or typ.signedness.data != Signedness.SIGNLESS
        or not 0 < typ.width.data <= limits.scalar_bits
    ):
        raise ValueError("combinational observation requires bounded signless integer SSA")
    return typ.width.data


def _ports(module, block, output, limits):
    typ = _attribute(module, "module_type")
    if not isinstance(typ, UnregisteredAttr) or typ.attr_name.data != "hw.modty":
        raise ValueError("combinational observation requires the complete original module type")
    entries = _hw_port_entries("(" + typ.value.data + ")")
    if entries is None:
        raise ValueError("combinational module ports cannot be read completely")
    inputs, outputs, names = [], [], set()
    for entry in entries:
        head, separator, declared_type = entry.partition(":")
        words = head.split()
        if not separator or len(words) != 2 or words[0] not in {"input", "output"} or words[1] in names:
            raise ValueError("combinational observation refuses opaque, duplicate or inout ports")
        names.add(words[1])
        roster, values = (inputs, block.args) if words[0] == "input" else (outputs, output.operands)
        if len(roster) >= len(values):
            raise ValueError("combinational module port roster differs from original SSA")
        width = _width(values[len(roster)], limits)
        if declared_type.strip() != "i" + str(width):
            raise ValueError("combinational declared port type differs from original SSA")
        roster.append(ScalarPort(words[1], width))
        if len(inputs) + len(outputs) > limits.nodes:
            raise ValueError("combinational original port roster exceeds its node budget")
    if len(inputs) != len(block.args) or len(outputs) != len(output.operands) or not outputs:
        raise ValueError("combinational observation requires every original input/output slot")
    return tuple(inputs), tuple(outputs)


def _expression(op, widths, width, *, conditional_logic=False):
    kind, parameter = _name(op), None
    expected = {
        "hw.constant": {"value"},
        "comb.extract": {"lowBit"},
        "comb.icmp": {"predicate", "twoState"},
        "comb.replicate": set(),
    }
    allowed = expected.get(kind, {"twoState"}) | {"op_name__", "sv.namehint"}
    if not (set(op.attributes) | set(op.properties)) <= allowed:
        raise ValueError("combinational observation refuses unknown operation attributes")
    if set(op.attributes) & set(op.properties):
        raise ValueError("combinational operation has ambiguous attribute/property ownership")
    two_state = op.attributes.get("twoState", op.properties.get("twoState"))
    if two_state is not None and not isinstance(two_state, UnitAttr):
        raise ValueError("combinational observation refuses malformed two-state annotation")
    hint = _attribute(op, "sv.namehint")
    if hint is not None and not isinstance(hint, StringAttr):
        raise ValueError("combinational observation refuses malformed source-name metadata")
    if kind == "hw.constant" and not widths:
        parameter = _integer(op, "value")
        constant = op.attributes.get("value", op.properties.get("value"))
        valid = parameter is not None and isinstance(constant, IntegerAttr) and constant.type == op.results[0].type
    elif kind == "comb.extract" and len(widths) == 1:
        parameter = _integer(op, "lowBit")
        valid = parameter is not None and 0 <= parameter and parameter + width <= widths[0]
    elif kind == "comb.concat":
        valid = bool(widths) and sum(widths) == width
    elif kind == "comb.replicate" and len(widths) == 1:
        valid = width >= widths[0] and width % widths[0] == 0
        parameter = width // widths[0]
    elif kind == "comb.shru":
        valid = widths == [width, width]
    elif kind in {"comb.add", "comb.and"}:
        valid = len(widths) >= 2 and all(bits == width for bits in widths)
    elif conditional_logic and kind in {"comb.or", "comb.xor"}:
        valid = len(widths) >= 2 and all(bits == width for bits in widths)
    elif conditional_logic and kind == "comb.sub":
        valid = widths == [width, width]
    elif kind == "comb.mux":
        valid = widths == [1, width, width]
    elif kind == "comb.icmp":
        predicate = _attribute(op, "predicate")
        parameter = _integer(op, "predicate")
        valid = len(widths) == 2 and widths[0] == widths[1] and width == 1
        valid = (
            valid
            and isinstance(predicate, IntegerAttr)
            and predicate.type == IntegerType(64)
            and parameter is not None
            and 0 <= parameter < len(ICMP_COMPARISON_OPERATIONS)
        )
    else:
        valid = False
    if not valid:
        raise ValueError("unsupported or ill-typed combinational operation: " + kind)
    return kind, parameter


def prepare_combinational_observation(
    text: str, *, module: str, limits: EvaluationLimits
) -> PreparedCombinationalObservation:
    """Prepare every node/output of a selected local module, with no role lift."""
    # Reject obviously oversized text before allocating an encoded copy. The
    # shared guard then counts UTF-8 bytes exactly before parsing attributes.
    if type(text) is not str or len(text) > limits.source_bytes:
        raise ValueError("combinational source exceeds its explicit parse budget")
    admit_mlir_source(
        text,
        max_source_bytes=limits.source_bytes,
        max_nesting=64,
        max_integer_bits=max(64, limits.scalar_bits),
        allow_dense=False,
        allow_dense_resource=False,
    )
    parsed = parse_generic_hw(text, reject_dense_literals=True)
    matches = [
        op
        for op in parsed.walk()
        if _name(op) == "hw.module"
        and isinstance(_attribute(op, "sym_name"), StringAttr)
        and _attribute(op, "sym_name").data == module
    ]
    if len(matches) != 1:
        raise ValueError("combinational observation requires one explicitly selected module")
    selected = matches[0]
    parameters = _attribute(selected, "parameters")
    if parameters is not None and (not isinstance(parameters, ArrayAttr) or parameters.data):
        raise ValueError("combinational observation refuses unresolved module parameters")
    if len(selected.regions) != 1 or len(selected.regions[0].blocks) != 1:
        raise ValueError("combinational module requires one complete block")
    block = selected.regions[0].block
    output = block.last_op
    if output is None or _name(output) != "hw.output" or output.results or output.regions:
        raise ValueError("combinational module requires an explicit original output terminator")
    if (set(output.attributes) | set(output.properties)) - {"op_name__"}:
        raise ValueError("combinational output has unsupported attributes")
    inputs, outputs = _ports(selected, block, output, limits)
    indices = {value: index for index, value in enumerate(block.args)}
    widths = {value: _width(value, limits) for value in block.args}
    expressions = []
    work = sum(port.width for port in (*inputs, *outputs))
    if work > limits.bit_work:
        raise ValueError("combinational port roster exceeds its bit-work budget")
    for op in block.ops:
        if op is output:
            break
        if len(inputs) + len(outputs) + len(expressions) >= limits.nodes or op.regions or len(op.results) != 1:
            raise ValueError("combinational observation exceeds its node budget or encounters opaque structure")
        if any(value not in indices for value in op.operands):
            raise ValueError("combinational observation refuses unresolved or non-dominating SSA")
        width = _width(op.results[0], limits)
        operand_widths = [widths[value] for value in op.operands]
        kind, parameter = _expression(op, operand_widths, width)
        expressions.append(_Expression(kind, width, tuple(indices[value] for value in op.operands), parameter))
        indices[op.results[0]] = len(indices)
        widths[op.results[0]] = width
        work += width + sum(operand_widths)
        if work > limits.bit_work:
            raise ValueError("combinational preparation exceeds its source-derived bit-work budget")
    if any(value not in indices for value in output.operands):
        raise ValueError("combinational output has no complete original source expression")
    return PreparedCombinationalObservation(
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        module,
        inputs,
        outputs,
        tuple(expressions),
        tuple(indices[value] for value in output.operands),
        work,
        limits,
    )


# ---- hw_cycle_eval -----------------------------------------------------------------------------
# Two-state, cycle-by-cycle evaluation of ONE elaborated HW module, for directed probes.
#
# WHY. Some ordering properties of a hardware unit are behaviours, not shapes: whether a command the
# dispatch table holds waits for an older one to ISSUE or to COMPLETE is decided by which event
# clears a dependency bit, and that is read most directly by driving the module's own ports and
# watching what it issues. :mod:`.command_ordering` does exactly that, through this evaluator.
#
# WHAT IT EVALUATES. The module's own top-level body: combinational ``comb`` operations, ``hw``
# wires/constants/arrays and ``seq`` registers (with their synchronous reset when present). Every
# value is a two-state unsigned bitvector. Nested regions (``sv.always`` / ``sv.ifdef`` bodies:
# assertions and prints) carry no values into the datapath and are not evaluated.
#
# WHAT IT REFUSES. An operation outside that vocabulary, a memory, or an instance result is evaluated
# only if it can reach a register or an output; if it can, preparation raises ``ValueError`` and the
# caller records UNKNOWN. A register with no reset starts at zero -- recorded by the caller as the
# evaluation's initialisation premise, because the hardware leaves it undefined.
#
# This is a local observation of one module's logic, not a model of the system it sits in: what is
# connected to its ports is whatever the caller drives.

#: Register operations and their operand layout: (next, clock[, reset, reset value]).
_REGISTERS = frozenset({"seq.firreg", "seq.compreg"})
#: Operations that only re-label a value.
_PASS = frozenset({"hw.wire", "seq.from_clock", "seq.to_clock", "hw.bitcast"})
_IGNORED = frozenset({"sv.always", "sv.ifdef", "sv.initial", "hw.output", "sv.verbatim", "sv.macro.decl"})


def _bits(value) -> int | None:
    typ = value.type
    if isinstance(typ, IntegerType):
        return typ.width.data
    return None


def _array_shape(value) -> tuple[int, int] | None:
    """``(length, element bits)`` of a ``!hw.array<NxiW>`` value, read from its type text."""
    text = str(value.type)
    head, sep, rest = text.partition("<")
    if not sep or not head.endswith("hw.array"):
        return None
    body = rest.rstrip(">")
    count, sep, element = body.partition("x")
    if not sep or not count.isdigit() or not element.startswith("i") or not element[1:].isdigit():
        return None
    return int(count), int(element[1:])


@dataclass(frozen=True)
class Step:
    kind: str
    out: int
    args: tuple[int, ...]
    width: int
    param: Any = None


class CycleModel:
    """A compiled single-module cycle evaluator. ``step(inputs)`` returns the outputs seen BEFORE the
    clock edge (combinational in the current state) and then advances the registers."""

    def __init__(self, design, module: str, *, max_nodes: int = 200_000):
        view = design.module(module)
        if view.extern:
            raise ValueError(f"module {module!r} is external; nothing to evaluate")
        self.module = module
        self.inputs = list(view.inputs)
        self.outputs = list(view.outputs)
        block = view.op.regions[0].block
        index: dict[Any, int] = {}
        widths: list[int] = []

        def slot(value, width) -> int:
            index[value] = len(widths)
            widths.append(width)
            return index[value]

        self._input_slots = {}
        for name in self.inputs:
            value = view.input_value[name]
            # clocks are one-bit carriers here; their value is never read for data
            self._input_slots[name] = slot(value, _bits(value) or 1)
        ops = [op for op in block.ops if _name(op) not in _IGNORED]
        if len(ops) > max_nodes:
            raise ValueError(f"module {module!r} exceeds the evaluation node budget")
        registers = [op for op in ops if _name(op) in _REGISTERS]
        for op in registers:
            width = _bits(op.results[0])
            if width is None:
                raise ValueError(f"register {op} is not an integer")
            slot(op.results[0], width)
        self._register_ops = registers
        # what any register or output can reach: everything else (assertion logic) is skipped
        needed: set[Any] = set()
        stack = [operand for op in registers for operand in op.operands]
        stack += [view.output_value[name] for name in self.outputs]
        producer = {op.results[0]: op for op in ops if len(op.results) == 1}
        while stack:
            value = stack.pop()
            if value in needed:
                continue
            needed.add(value)
            op = producer.get(value)
            if op is not None and _name(op) not in _REGISTERS:
                stack.extend(op.operands)
        for op in ops:
            if len(op.results) > 1 and any(result in needed for result in op.results):
                raise ValueError(f"multi-result operation {_name(op)} reaches state in {module!r}")
        # A module body is a graph region: operands may be defined after their use. Order the needed
        # combinational operations so every operand is computed first (registers and inputs are leaves).
        steps: list[Step] = []
        done: set[Any] = set(index)
        on_path: set[Any] = set()
        for root in [op.results[0] for op in ops if len(op.results) == 1 and _name(op) not in _REGISTERS]:
            if root not in needed or root in done:
                continue
            stack: list[tuple[Any, bool]] = [(root, False)]
            while stack:
                value, expanded = stack.pop()
                if value in done:
                    continue
                op = producer.get(value)
                if op is None:
                    raise ValueError(f"a value reaching state in {module!r} has no producer in its body")
                if expanded:
                    on_path.discard(value)
                    steps.append(self._compile(_name(op), op, value, index, slot))
                    done.add(value)
                    continue
                if value in on_path:
                    raise ValueError(f"combinational cycle in {module!r}")
                on_path.add(value)
                stack.append((value, True))
                stack.extend((operand, False) for operand in op.operands if operand not in done)
        self._steps = steps
        self._widths = widths
        self._size = len(widths)
        self._next = [
            (index[op.operands[0]], index[op.operands[2]], index[op.operands[3]])
            if len(op.operands) == 4
            else (index[op.operands[0]], None, None)
            for op in registers
        ]
        self._register_slots = [index[op.results[0]] for op in registers]
        self._output_slots = {name: index[view.output_value[name]] for name in self.outputs}
        self.register_names = [
            (_attribute(op, "name").data if _attribute(op, "name") is not None else f"reg{i}")
            for i, op in enumerate(registers)
        ]
        self.state = [0] * len(registers)
        self.unreset_registers = sum(1 for op in registers if len(op.operands) != 4)

    # ------------------------------------------------------------------ compilation
    def _compile(self, kind, op, result, index, slot) -> Step:
        try:
            args = tuple(index[operand] for operand in op.operands)
        except KeyError as exc:
            raise ValueError(f"{kind} in {self.module!r} reads a value defined later or outside") from exc
        shape = _array_shape(result)
        width = _bits(result)
        if width is None and shape is None:
            raise ValueError(f"{kind} in {self.module!r} produces an unsupported type {result.type}")
        out = slot(result, width if width is not None else 0)
        if kind == "hw.constant":
            attr = _attribute(op, "value")
            if not isinstance(attr, IntegerAttr):
                raise ValueError("constant without an integer value")
            return Step(kind, out, (), width, attr.value.data & ((1 << width) - 1))
        if kind in _PASS:
            return Step("pass", out, args[:1], width or 0)
        if kind == "comb.extract":
            return Step(kind, out, args, width, _integer(op, "lowBit"))
        if kind == "comb.concat":
            return Step(kind, out, args, width, tuple(_bits(o) for o in op.operands))
        if kind == "comb.replicate":
            return Step(kind, out, args, width, (_bits(op.operands[0]), width // _bits(op.operands[0])))
        if kind == "comb.icmp":
            name = ICMP_COMPARISON_OPERATIONS[_integer(op, "predicate")]
            return Step(kind, out, args, 1, (name, _bits(op.operands[0])))
        if kind in {
            "comb.and",
            "comb.or",
            "comb.xor",
            "comb.add",
            "comb.mul",
            "comb.sub",
            "comb.divu",
            "comb.modu",
            "comb.shl",
            "comb.shru",
            "comb.mux",
            "comb.parity",
        }:
            return Step(kind, out, args, width)
        if kind == "hw.array_create":
            return Step(kind, out, args, 0, shape)
        if kind == "hw.array_get":
            arr = _array_shape(op.operands[0])
            return Step(kind, out, args, width, arr)
        raise ValueError(f"operation {kind} reaches state in {self.module!r} and is not evaluated")

    # ------------------------------------------------------------------ evaluation
    def _evaluate(self, inputs: Mapping[str, int]) -> list:
        values: list = [0] * self._size
        for name, value in inputs.items():
            values[self._input_slots[name]] = value
        for slot_index, value in zip(self._register_slots, self.state, strict=True):
            values[slot_index] = value
        for step in self._steps:
            kind, args, width = step.kind, step.args, step.width
            if kind == "comb.mux":
                values[step.out] = values[args[1]] if values[args[0]] else values[args[2]]
                continue
            if kind == "comb.and":
                acc = values[args[0]]
                for a in args[1:]:
                    acc &= values[a]
                values[step.out] = acc
                continue
            if kind == "comb.or":
                acc = values[args[0]]
                for a in args[1:]:
                    acc |= values[a]
                values[step.out] = acc
                continue
            if kind == "comb.xor":
                acc = values[args[0]]
                for a in args[1:]:
                    acc ^= values[a]
                values[step.out] = acc
                continue
            if kind == "comb.icmp":
                values[step.out] = _icmp(step.param, values[args[0]], values[args[1]])
                continue
            if kind == "comb.extract":
                values[step.out] = (values[args[0]] >> step.param) & ((1 << width) - 1)
                continue
            if kind == "comb.concat":
                acc = 0
                for a, bits in zip(args, step.param, strict=True):
                    acc = (acc << bits) | values[a]
                values[step.out] = acc
                continue
            if kind == "hw.constant":
                values[step.out] = step.param
                continue
            if kind == "pass":
                values[step.out] = values[args[0]]
                continue
            mask = (1 << width) - 1 if width else 0
            if kind == "comb.add":
                values[step.out] = sum(values[a] for a in args) & mask
            elif kind == "comb.sub":
                values[step.out] = (values[args[0]] - values[args[1]]) & mask
            elif kind == "comb.mul":
                acc = 1
                for a in args:
                    acc *= values[a]
                values[step.out] = acc & mask
            elif kind == "comb.divu":
                d = values[args[1]]
                values[step.out] = (values[args[0]] // d) & mask if d else 0
            elif kind == "comb.modu":
                d = values[args[1]]
                values[step.out] = (values[args[0]] % d) & mask if d else 0
            elif kind == "comb.shl":
                s = values[args[1]]
                values[step.out] = (values[args[0]] << s) & mask if s < width else 0
            elif kind == "comb.shru":
                s = values[args[1]]
                values[step.out] = values[args[0]] >> s if s < width else 0
            elif kind == "comb.parity":
                values[step.out] = bin(values[args[0]]).count("1") & 1
            elif kind == "comb.replicate":
                bits, times = step.param
                acc = 0
                for _ in range(times):
                    acc = (acc << bits) | values[args[0]]
                values[step.out] = acc
            elif kind == "hw.array_create":
                # CIRCT lists array elements from the highest index down
                values[step.out] = tuple(values[a] for a in reversed(args))
            elif kind == "hw.array_get":
                arr, at = values[args[0]], values[args[1]]
                values[step.out] = arr[at] if at < len(arr) else 0
            else:  # pragma: no cover - _compile admits only the kinds above
                raise ValueError(kind)
        return values

    def peek(self, inputs: Mapping[str, int]) -> dict[str, int]:
        """The outputs under ``inputs`` in the current state, without a clock edge."""
        values = self._evaluate(self._full(inputs))
        return {name: values[slot] for name, slot in self._output_slots.items()}

    def step(self, inputs: Mapping[str, int]) -> dict[str, int]:
        """Outputs before the edge, then one rising edge (registers take their next value)."""
        values = self._evaluate(self._full(inputs))
        outputs = {name: values[slot] for name, slot in self._output_slots.items()}
        nxt = []
        for (src, reset, reset_value), current in zip(self._next, self.state, strict=True):
            nxt.append(values[reset_value] if reset is not None and values[reset] else values[src])
        self.state = nxt
        return outputs

    def _full(self, inputs: Mapping[str, int]) -> dict[str, int]:
        unknown = set(inputs) - set(self._input_slots)
        if unknown:
            raise ValueError(f"{self.module!r} has no input(s) {sorted(unknown)}")
        full = dict.fromkeys(self._input_slots, 0)
        for name, value in inputs.items():
            full[name] = value & ((1 << self._widths[self._input_slots[name]]) - 1)
        return full

    def reset_state(self) -> None:
        self.state = [0] * len(self._register_ops)


def _icmp(param, left: int, right: int) -> int:
    name, bits = param
    if name == "eq":
        return int(left == right)
    if name == "ne":
        return int(left != right)
    if name[0] == "s":
        sign = 1 << (bits - 1)
        left = left - (sign << 1) if left & sign else left
        right = right - (sign << 1) if right & sign else right
        name = "u" + name[1:]
    if name == "ult":
        return int(left < right)
    if name == "ule":
        return int(left <= right)
    if name == "ugt":
        return int(left > right)
    return int(left >= right)

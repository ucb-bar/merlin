"""Two-state, cycle-by-cycle evaluation of ONE elaborated HW module, for directed probes.

WHY. Some ordering properties of a hardware unit are behaviours, not shapes: whether a command the
dispatch table holds waits for an older one to ISSUE or to COMPLETE is decided by which event
clears a dependency bit, and that is read most directly by driving the module's own ports and
watching what it issues. :mod:`.command_ordering` does exactly that, through this evaluator.

WHAT IT EVALUATES. The module's own top-level body: combinational ``comb`` operations, ``hw``
wires/constants/arrays and ``seq`` registers (with their synchronous reset when present). Every
value is a two-state unsigned bitvector. Nested regions (``sv.always`` / ``sv.ifdef`` bodies:
assertions and prints) carry no values into the datapath and are not evaluated.

WHAT IT REFUSES. An operation outside that vocabulary, a memory, or an instance result is evaluated
only if it can reach a register or an output; if it can, preparation raises ``ValueError`` and the
caller records UNKNOWN. A register with no reset starts at zero -- recorded by the caller as the
evaluation's initialisation premise, because the hardware leaves it undefined.

This is a local observation of one module's logic, not a model of the system it sits in: what is
connected to its ports is whatever the caller drives.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.dialects.comb import ICMP_COMPARISON_OPERATIONS

from .hw_observations import _attribute, _integer, _name

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

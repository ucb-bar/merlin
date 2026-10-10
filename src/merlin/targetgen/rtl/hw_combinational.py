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

from merlin.targetgen.contract.mlir_source_admission import admit_mlir_source

from .hw_graph import parse_generic_hw
from .hw_observations import _attribute, _integer, _name
from .ports import _hw_port_entries


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

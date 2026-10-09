"""Bounded conditional expressions feeding explicit original instance inputs.

Module arguments and direct instance results are independent symbolic roots.
They are not observations of reachable state, storage effects or execution.
Undefined source branches require a separate original-source validity premise;
two-state values in this observation cannot supply that premise.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, StringAttr, SymbolRefAttr, UnregisteredAttr
from xdsl.ir import BlockArgument, OpResult

from merlin.targetgen.contract.mlir_source_admission import admit_mlir_source

from .hw_combinational import (
    EvaluationLimits,
    PreparedCombinationalObservation,
    ScalarPort,
    _Expression,
    _expression,
    _width,
)
from .hw_graph import parse_generic_hw
from .hw_observations import _attribute, _name
from .ports import _hw_port_entries


@dataclass(frozen=True)
class OriginalPort:
    direction: str
    name: str
    type: str


@dataclass(frozen=True)
class SymbolicRoot:
    port: ScalarPort
    kind: str
    name: str
    instance: str | None = None
    module: str | None = None


@dataclass(frozen=True)
class PreparedInstanceInputObservation:
    """Conditional local SSA only; complete consumer signature is preserved."""

    module: str
    instance: str
    consumer_module: str
    consumer_ports: tuple[OriginalPort, ...]
    roots: tuple[SymbolicRoot, ...]
    expression: PreparedCombinationalObservation

    def evaluate(self, cases):
        return self.expression.evaluate(cases)


def _signature(op):
    typ, parameters = _attribute(op, "module_type"), _attribute(op, "parameters")
    if not isinstance(typ, UnregisteredAttr) or typ.attr_name.data != "hw.modty":
        raise ValueError("instance source requires a lossless original module type")
    if parameters is not None and (not isinstance(parameters, ArrayAttr) or parameters.data):
        raise ValueError("instance source refuses unresolved module parameters")
    entries = _hw_port_entries("(" + typ.value.data + ")")
    if entries is None:
        raise ValueError("instance source ports cannot be read completely")
    ports = []
    for entry in entries:
        head, separator, type_text = entry.partition(":")
        words = head.split()
        if not separator or len(words) != 2 or words[0] not in {"input", "output"}:
            raise ValueError("instance source refuses unreadable or inout ports")
        ports.append(OriginalPort(words[0], words[1], type_text.strip()))
    if len({port.name for port in ports}) != len(ports):
        raise ValueError("instance source port names must be unique")
    if _name(op) == "hw.module":
        if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
            raise ValueError("instance source requires one original module block")
        block = op.regions[0].block
        output = block.last_op
        if output is None or _name(output) != "hw.output" or output.results or output.regions:
            raise ValueError("instance source requires the original output terminator")
        for direction, values in (("input", block.args), ("output", output.operands)):
            selected = tuple(port for port in ports if port.direction == direction)
            if len(selected) != len(values) or any(
                port.type != str(value.type) for port, value in zip(selected, values, strict=True)
            ):
                raise ValueError("instance source declared ports differ from original SSA")
    return tuple(ports)


def _instance(op, modules):
    names, outputs = _attribute(op, "argNames"), _attribute(op, "resultNames")
    module, instance, parameters = (_attribute(op, key) for key in ("moduleName", "instanceName", "parameters"))
    if (
        not isinstance(module, SymbolRefAttr)
        or module.nested_references.data
        or not isinstance(instance, StringAttr)
        or not isinstance(names, ArrayAttr)
        or not isinstance(outputs, ArrayAttr)
        or not all(isinstance(name, StringAttr) for name in (*names.data, *outputs.data))
    ):
        raise ValueError("instance source requires complete explicit names and module selection")
    if parameters is not None and (not isinstance(parameters, ArrayAttr) or parameters.data):
        raise ValueError("instance source refuses unresolved instance parameters")
    selected = modules.get(module.root_reference.data)
    if selected is None:
        raise ValueError("instance source callee is unavailable in the original module roster")
    ports = _signature(selected)
    for direction, labels, values in (
        ("input", names.data, op.operands),
        ("output", outputs.data, op.results),
    ):
        expected = tuple(port for port in ports if port.direction == direction)
        if (
            len(labels) != len(expected)
            or len(values) != len(expected)
            or any(
                name.data != port.name or str(value.type) != port.type
                for port, name, value in zip(expected, labels, values, strict=True)
            )
        ):
            raise ValueError("instance source port binding differs from the complete original callee")
    return instance.data, module.root_reference.data, ports


def prepare_instance_input_observation(
    text: str, *, module: str, instance: str, ports: tuple[str, ...], limits: EvaluationLimits
) -> PreparedInstanceInputObservation:
    """Follow selected sinks; state/unknown nodes refuse, instance results stop."""
    if type(text) is not str or len(text) > limits.source_bytes:
        raise ValueError("instance source exceeds its explicit parse budget")
    if not isinstance(ports, tuple) or not ports or any(not isinstance(name, str) for name in ports):
        raise ValueError("instance observation requires an explicit nonempty original input port tuple")
    if len(set(ports)) != len(ports):
        raise ValueError("instance observation selected input ports must be unique")
    admit_mlir_source(
        text,
        max_source_bytes=limits.source_bytes,
        max_nesting=64,
        max_integer_bits=max(64, limits.scalar_bits),
        allow_dense=False,
        allow_dense_resource=False,
    )
    parsed = parse_generic_hw(text, reject_dense_literals=True)
    modules = {}
    for op in parsed.walk():
        if _name(op) not in {"hw.module", "hw.module.extern"}:
            continue
        name = _attribute(op, "sym_name")
        if not isinstance(name, StringAttr) or name.data in modules:
            raise ValueError("instance source module symbols must be complete and unique")
        modules[name.data] = op
    parent = modules.get(module)
    if parent is None or _name(parent) != "hw.module":
        raise ValueError("instance observation requires an explicitly selected original body")
    signature = _signature(parent)
    block = parent.regions[0].block
    arguments = dict(zip(block.args, (p.name for p in signature if p.direction == "input"), strict=True))
    instances = {}
    for op in block.ops:
        if _name(op) == "hw.instance":
            info = _instance(op, modules)
            if info[0] in instances:
                raise ValueError("instance source names must be unique in the selected body")
            instances[info[0]] = op, info
    if instance not in instances:
        raise ValueError("instance observation selected consumer is unavailable")
    consumer, (_, callee, consumer_ports) = instances[instance]
    input_names = tuple(p.name for p in consumer_ports if p.direction == "input")
    bindings = dict(zip(input_names, consumer.operands, strict=True))
    if any(name not in bindings for name in ports):
        raise ValueError("instance observation selected port is not an original consumer input")
    sinks = tuple(bindings[name] for name in ports)
    outputs = tuple(ScalarPort(name, _width(value, limits)) for name, value in zip(ports, sinks, strict=True))
    indices, widths, roots, expressions, visiting = {}, {}, [], [], set()
    work = sum(port.width for port in outputs)

    def charge(value, operands=()):
        nonlocal work
        if len(indices) + len(outputs) >= limits.nodes:
            raise ValueError("instance expression exceeds its source-derived node budget")
        width = _width(value, limits)
        work += width + sum(widths[operand] for operand in operands)
        if work > limits.bit_work:
            raise ValueError("instance expression exceeds its source-derived bit-work budget")
        indices[value], widths[value] = len(indices), width
        return width

    # Inputs must precede all expressions in the shared evaluator's slot layout.
    order, root_values = [], []
    for sink in sinks:
        stack = [(sink, False)]
        while stack:
            value, ready = stack.pop()
            if value in widths:
                continue
            if not ready and len(widths) + len(visiting) + len(outputs) >= limits.nodes:
                raise ValueError("instance dependency walk exceeds its source-derived node budget")
            if isinstance(value, BlockArgument):
                if value not in arguments:
                    raise ValueError("instance expression has a nonlocal block argument")
                width = charge(value)
                root_values.append(value)
                roots.append(
                    SymbolicRoot(ScalarPort("root_" + str(len(roots)), width), "module_input", arguments[value])
                )
                continue
            if not isinstance(value, OpResult) or value.owner.parent is not block:
                raise ValueError("instance expression has a nonlocal or unknown SSA producer")
            op = value.owner
            if _name(op) == "hw.instance":
                name, source_module, source_ports = _instance(op, modules)
                result_ports = tuple(port for port in source_ports if port.direction == "output")
                width = charge(value)
                root_values.append(value)
                roots.append(
                    SymbolicRoot(
                        ScalarPort("root_" + str(len(roots)), width),
                        "opaque_instance_result",
                        result_ports[value.index].name,
                        name,
                        source_module,
                    )
                )
                continue
            if op.regions or len(op.results) != 1:
                raise ValueError("instance expression refuses state, opaque regions or multiple results")
            if _name(op) not in {
                "hw.constant",
                "comb.extract",
                "comb.concat",
                "comb.add",
                "comb.sub",
                "comb.and",
                "comb.or",
                "comb.xor",
                "comb.mux",
                "comb.icmp",
            }:
                raise ValueError("instance expression refuses unsupported or stateful producers")
            if not ready:
                if value in visiting:
                    raise ValueError("instance expression source contains a cyclic dependency")
                visiting.add(value)
                stack.append((value, True))
                stack.extend((operand, False) for operand in reversed(op.operands))
                continue
            visiting.remove(value)
            width = charge(value, op.operands)
            _expression(op, [widths[operand] for operand in op.operands], width, conditional_logic=True)
            order.append(value)
    indices = {value: index for index, value in enumerate(root_values)}
    for value in order:
        op = value.owner
        kind, parameter = _expression(op, [widths[arg] for arg in op.operands], widths[value], conditional_logic=True)
        expressions.append(_Expression(kind, widths[value], tuple(indices[arg] for arg in op.operands), parameter))
        indices[value] = len(indices)
    expression = PreparedCombinationalObservation(
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        module,
        tuple(root.port for root in roots),
        outputs,
        tuple(expressions),
        tuple(indices[value] for value in sinks),
        work,
        limits,
    )
    return PreparedInstanceInputObservation(module, instance, callee, consumer_ports, tuple(roots), expression)

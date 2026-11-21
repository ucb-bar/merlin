"""Bounded conditional expressions feeding explicit original instance inputs.

Module arguments and direct instance results are independent symbolic roots.
They are not observations of reachable state, storage effects or execution.
Undefined source branches require a separate original-source validity premise;
two-state values in this observation cannot supply that premise.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from xdsl.dialects.builtin import ArrayAttr, DenseArrayBase, IntegerType, StringAttr, SymbolRefAttr, UnregisteredAttr
from xdsl.ir import BlockArgument, OpResult

from merlin.target.contract.interfaces.grammar import admit_mlir_source
from merlin.target.rtl.firrtl.ports import _hw_port_entries
from merlin.target.rtl.hw.evaluation import (
    EvaluationLimits,
    PreparedCombinationalObservation,
    ScalarPort,
    _Expression,
    _expression,
    _width,
)
from merlin.target.rtl.hw.graph import _attribute, _integer, _module_name, _name, parse_generic_hw

# ---- hw_instance_inputs ------------------------------------------------------------------------


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


# ---- hw_memory_ports ---------------------------------------------------------------------------
# Exact local FIRRTL-flavored memory declarations and conditional port bindings.
#
# The Seq dialect supplies memory and port meanings. Original input names,
# integer widths and instance geometry supply no command, tensor or allocation
# roles. State and instance results remain explicitly independent symbolic roots;
# this reader never evaluates memory contents or follows temporal transfers.

SCHEMA = "merlin.hw_memory_port_observations.v1"
_PORTS = {"seq.firmem.read_port", "seq.firmem.write_port", "seq.firmem.read_write_port"}
_COMBINATIONAL = {
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
}
_UNKNOWN = (
    "command_and_decoder_correspondence",
    "state_and_instance_transfer_semantics",
    "software_scalar_and_tensor_axis_correspondence",
    "allocation_capacity_and_physical_tails",
    "initial_memory_contents_and_reachable_state",
    "physical_alias_lifetime_order_and_completion",
    "complete_memory_and_instruction_domain",
    "undefined_source_branch_validity",
)


@dataclass(frozen=True)
class MemoryPortLimits:
    """Explicit metadata bounds; no memory or tensor values are allocated."""

    modules: int
    memories: int
    ports: int
    nodes: int
    scalar_bits: int
    bit_work: int
    expression_depth: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("memory observations require positive explicit metadata limits")


def _memory_type(typ):
    if not isinstance(typ, UnregisteredAttr) or typ.attr_name.data != "seq.firmem":
        raise ValueError("memory declaration requires its exact original Seq memory type")
    tokens = typ.value.data.replace(",", " , ").split()
    if (
        len(tokens) not in {3, 6}
        or tokens[1] != "x"
        or not tokens[0].isdecimal()
        or not tokens[2].isdecimal()
        or (len(tokens) == 6 and (tokens[3:5] != [",", "mask"] or not tokens[5].isdecimal()))
    ):
        raise ValueError("memory declaration type is outside the complete supported grammar")
    depth, width = int(tokens[0]), int(tokens[2])
    mask = int(tokens[5]) if len(tokens) == 6 else None
    if depth <= 0 or width <= 0 or (mask is not None and mask <= 0):
        raise ValueError("memory declaration requires positive depth, width and optional mask width")
    return depth, width, mask


def _bindings(op, *, width, address_bits, mask_width):
    name = _name(op)
    if op.regions or not op.operands:
        raise ValueError("memory port requires one complete original local operation")
    if name == "seq.firmem.read_port":
        if len(op.operands) not in {3, 4} or _attribute(op, "operandSegmentSizes") is not None:
            raise ValueError("memory read port has an unsupported operand roster")
        labels = ["memory", "address", "clock", *(["enable"] if len(op.operands) == 4 else [])]
    else:
        segments = _attribute(op, "operandSegmentSizes")
        if not isinstance(segments, DenseArrayBase) or segments.elt_type != IntegerType(32):
            raise ValueError("memory write port needs original i32 operand segment sizes")
        sizes = list(segments.get_values())
        rw = name == "seq.firmem.read_write_port"
        fixed = [1, 1, 1, None, 1, *([1] if rw else []), None]
        if (
            len(sizes) != len(fixed)
            or sum(sizes) != len(op.operands)
            or any(
                size not in {0, 1} if expected is None else size != expected
                for size, expected in zip(sizes, fixed, strict=True)
            )
        ):
            raise ValueError("memory port has an incomplete or inconsistent original segment roster")
        fields = ["memory", "address", "clock", "enable", "data", *(["mode"] if rw else []), "mask"]
        labels = [field for field, size in zip(fields, sizes, strict=True) if size]
    bound = dict(zip(labels, op.operands, strict=True))
    expected = {"address": address_bits, "enable": 1, "data": width, "mode": 1, "mask": mask_width}
    for label, value in bound.items():
        if label == "memory":
            continue
        if label == "clock":
            if str(value.type) != "!seq.clock":
                raise ValueError("memory port clock differs from its original Seq clock type")
        elif expected[label] is None or value.type != IntegerType(expected[label]):
            raise ValueError("memory port type differs from its original declared memory")
    read = name != "seq.firmem.write_port"
    if len(op.results) != int(read) or (read and op.results[0].type != IntegerType(width)):
        raise ValueError("memory port result roster differs from its original declared memory")
    return {label: value for label, value in bound.items() if label != "memory"}


def memory_port_observations(parsed, *, limits: MemoryPortLimits):
    """Retain every original firmem and port; unknown roots are never elided."""
    if type(limits) is not MemoryPortLimits:
        raise ValueError("memory observations require the original explicit limits")
    modules = {}
    for op in parsed.walk():
        if _name(op) not in {"hw.module", "hw.module.extern"}:
            continue
        name = _module_name(op)
        if name in modules or len(modules) >= limits.modules:
            raise ValueError("memory source has duplicate modules or exceeds its module budget")
        modules[name] = op
    totals = {"memories": 0, "ports": 0, "nodes": 0, "bit_work": 0}
    records = []
    for module, op in modules.items():
        if _name(op) != "hw.module":
            continue
        block = op.regions[0].block
        children = list(block.ops)
        memories = [child for child in children if _name(child) == "seq.firmem"]
        if not memories:
            continue
        signature = _signature(op)
        inputs = dict(zip(block.args, (port.name for port in signature if port.direction == "input"), strict=True))
        ordinals = {child: ordinal for ordinal, child in enumerate(children)}
        nodes, visiting = {}, set()

        def width(value):
            typ = value.type
            if str(typ) == "!seq.clock":
                return None
            if not isinstance(typ, IntegerType) or typ != IntegerType(typ.width.data):
                raise ValueError("memory binding requires a scalar signless original type")
            if not 0 < typ.width.data <= limits.scalar_bits:
                raise ValueError("memory binding exceeds its original scalar-width budget")
            return typ.width.data

        def trace(value, depth=0):
            if value in nodes:
                return nodes[value]["id"]
            if depth >= limits.expression_depth or value in visiting:
                raise ValueError("memory expression is cyclic or exceeds its depth budget")
            bits = width(value)
            if isinstance(value, BlockArgument):
                if value not in inputs:
                    raise ValueError("memory expression has a nonlocal block argument")
                identity = "input:" + str(value.index)
                row = {
                    "id": identity,
                    "kind": "module_input",
                    "ordinal": value.index,
                    "name": inputs[value],
                    "type": str(value.type),
                }
            elif isinstance(value, OpResult) and value.owner.parent is block:
                producer = value.owner
                identity = "op:" + str(ordinals[producer]) + ":" + str(value.index)
                row = {
                    "id": identity,
                    "type": str(value.type),
                    "operation": _name(producer),
                    "ordinal": ordinals[producer],
                    "result_ordinal": value.index,
                }
                if _name(producer) == "hw.instance":
                    name, callee, ports = _instance(producer, modules)
                    outputs = tuple(port for port in ports if port.direction == "output")
                    row.update(
                        kind="opaque_instance_result",
                        instance=name,
                        module=callee,
                        port=outputs[value.index].name,
                        original_callee_ports=[asdict(port) for port in ports],
                    )
                elif _name(producer) in {"seq.firreg", "seq.compreg", "seq.compreg.ce", "seq.shiftreg"}:
                    row.update(kind="state_result")
                elif _name(producer) in _COMBINATIONAL and bits is not None:
                    if producer.regions or len(producer.results) != 1:
                        raise ValueError("memory combinational producer has unsupported regions/results")
                    visiting.add(value)
                    operands = [trace(operand, depth + 1) for operand in producer.operands]
                    visiting.remove(value)
                    try:
                        kind, parameter = _expression(
                            producer, [width(operand) for operand in producer.operands], bits, conditional_logic=True
                        )
                    except ValueError as error:
                        row.update(kind="unsupported_result", reason=str(error), operands=operands)
                    else:
                        row.update(kind="combinational", expression=kind, parameter=parameter, operands=operands)
                    totals["bit_work"] += sum(width(operand) or 0 for operand in producer.operands)
                else:
                    row.update(kind="unsupported_result")
            else:
                raise ValueError("memory expression has a nonlocal or unavailable original producer")
            totals["nodes"] += 1
            totals["bit_work"] += bits or 0
            if totals["nodes"] > limits.nodes or totals["bit_work"] > limits.bit_work:
                raise ValueError("complete memory observations exceed their aggregate expression budget")
            row["width"] = bits
            nodes[value] = row
            return identity

        def interval(value):
            row = nodes[value]
            if row["kind"] in {"module_input", "opaque_instance_result", "state_result"} and row["width"]:
                return {"root": row["id"], "low_bit": 0, "width": row["width"]}
            if row["kind"] != "combinational":
                return None
            producer = value.owner
            if _name(producer) == "comb.extract":
                base = interval(producer.operands[0])
                if base:
                    return {
                        "root": base["root"],
                        "low_bit": base["low_bit"] + _integer(producer, "lowBit"),
                        "width": row["width"],
                    }
            if _name(producer) == "comb.concat":
                parts = [interval(operand) for operand in reversed(producer.operands)]
                if parts and all(parts):
                    first, size = parts[0], 0
                    for part in parts:
                        if part["root"] != first["root"] or part["low_bit"] != first["low_bit"] + size:
                            return None
                        size += part["width"]
                    return {"root": first["root"], "low_bit": first["low_bit"], "width": size}
            return None

        declarations = []
        for memory in memories:
            totals["memories"] += 1
            if totals["memories"] > limits.memories or memory.operands or memory.regions or len(memory.results) != 1:
                raise ValueError("memory source exceeds its budget or has an unsupported declaration roster")
            depth, bits, mask = _memory_type(memory.results[0].type)
            if bits > limits.scalar_bits:
                raise ValueError("declared memory width exceeds its explicit scalar metadata budget")
            latency = {key: _integer(memory, key) for key in ("readLatency", "writeLatency")}
            ruw, wuw = _integer(memory, "ruw"), _integer(memory, "wuw")
            if (
                any(value is None or value < 0 for value in latency.values())
                or ruw not in {0, 1, 2}
                or wuw not in {0, 1}
            ):
                raise ValueError("memory source lacks supported explicit latency/collision declarations")
            ports = []
            users = {use.operation for use in memory.results[0].uses}
            for port in sorted(users, key=lambda child: ordinals.get(child, -1)):
                if port.parent is not block or _name(port) not in _PORTS or port.operands[0] is not memory.results[0]:
                    raise ValueError("memory handle has an unsupported or nonlocal original use")
                totals["ports"] += 1
                if totals["ports"] > limits.ports:
                    raise ValueError("complete memory observations exceed their aggregate port budget")
                bindings = _bindings(port, width=bits, address_bits=max(1, (depth - 1).bit_length()), mask_width=mask)
                refs = {key: trace(value) for key, value in bindings.items()}
                ports.append(
                    {
                        "ordinal": ordinals[port],
                        "operation": _name(port),
                        "bindings": refs,
                        "implicit_enable": "true" if "enable" not in bindings else None,
                        "implicit_write_mask": "all_bits_enabled"
                        if "data" in bindings and "mask" not in bindings
                        else None,
                        "data_source_interval": interval(bindings["data"]) if "data" in bindings else None,
                        "result_types": [str(value.type) for value in port.results],
                    }
                )
            name = _attribute(memory, "name")
            declarations.append(
                {
                    "ordinal": ordinals[memory],
                    "name": name.data if isinstance(name, StringAttr) else None,
                    "type": str(memory.results[0].type),
                    "depth": depth,
                    "width": bits,
                    "mask_width": mask,
                    "declared_storage_bits": depth * bits,
                    "address_domain": {"minimum": 0, "maximum": depth - 1, "out_of_range": "unestablished"},
                    "read_latency": latency["readLatency"],
                    "write_latency": latency["writeLatency"],
                    "read_under_write": ("undefined", "old", "new")[ruw],
                    "write_under_write": ("undefined", "port_order")[wuw],
                    "initialization_declared": _attribute(memory, "init") is not None,
                    "ports": ports,
                }
            )
        records.append(
            {
                "module": module,
                "original_ports": [asdict(port) for port in signature],
                "memories": declarations,
                "expressions": list(nodes.values()),
            }
        )
    return {
        "schema": SCHEMA,
        "scope": "local declared Seq memory and conditional scalar port bindings only",
        "modules": records,
        "cost": totals,
        "limits": asdict(limits),
        "unknowns": list(_UNKNOWN),
        "memory_contents_evaluated": False,
        "command_tensor_allocation_or_temporal_admission": False,
    }

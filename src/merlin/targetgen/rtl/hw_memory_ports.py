"""Exact local FIRRTL-flavored memory declarations and conditional port bindings.

The Seq dialect supplies memory and port meanings. Original input names,
integer widths and instance geometry supply no command, tensor or allocation
roles. State and instance results remain explicitly independent symbolic roots;
this reader never evaluates memory contents or follows temporal transfers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from xdsl.dialects.builtin import DenseArrayBase, IntegerType, StringAttr, UnregisteredAttr
from xdsl.ir import BlockArgument, OpResult

from .hw_combinational import _expression
from .hw_instance_inputs import _instance, _signature
from .hw_observations import _attribute, _integer, _module_name, _name

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

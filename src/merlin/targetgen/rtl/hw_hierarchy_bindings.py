"""Bounded hierarchical SSA connectivity to exact declared memory ports.

Defined module input/output bindings may be followed combinationally. State,
memory reads and external or parameterized bodies remain explicit stops. An
instance occurrence is source structure, never command, tensor or effect credit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IntegerType, StringAttr, SymbolRefAttr, UnregisteredAttr
from xdsl.ir import BlockArgument, OpResult

from .hw_combinational import _expression
from .hw_instance_inputs import OriginalPort, _instance, _signature
from .hw_memory_ports import _COMBINATIONAL, MemoryPortLimits, _bindings, memory_port_observations
from .hw_observations import _attribute, _module_name, _name
from .ports import _hw_port_entries

SCHEMA = "merlin.hw_hierarchical_memory_bindings.v1"
_STATE = {"seq.firreg", "seq.compreg", "seq.compreg.ce", "seq.shiftreg"}
_UNKNOWN = (
    "command_operand_and_decoder_semantics",
    "state_transfer_reachability_and_initialization",
    "memory_read_values_and_collision_validity",
    "external_parameterized_and_unsupported_body_semantics",
    "software_dtype_and_tensor_axis_correspondence",
    "allocation_capacity_use_and_physical_tails",
    "physical_alias_lifetime_order_and_completion",
    "undefined_source_branch_validity",
    "elaboration_to_physical_device_correspondence",
)


@dataclass(frozen=True)
class HierarchyBindingLimits:
    modules: int
    occurrences: int
    memory_occurrences: int
    memory_ports: int
    port_bindings: int
    nodes: int
    scalar_bits: int
    bit_work: int
    hierarchy_depth: int
    expression_depth: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("hierarchical bindings require explicit positive metadata limits")


def _parameters(op):
    value = _attribute(op, "parameters")
    if value is not None and not isinstance(value, ArrayAttr):
        raise ValueError("hierarchical source has unreadable original parameters")
    legacy = _attribute(op, "oldParameters")
    if legacy is not None and not isinstance(legacy, DictionaryAttr):
        raise ValueError("hierarchical source has unreadable original legacy parameters")
    return (value is not None and bool(value.data)) or (legacy is not None and bool(legacy.data))


def _static_signature(op):
    """Retain parameterized declarations only when exact static types still join."""
    if not _parameters(op):
        return _signature(op)
    typ = _attribute(op, "module_type")
    if not isinstance(typ, UnregisteredAttr) or typ.attr_name.data != "hw.modty":
        raise ValueError("hierarchical source lacks its original declared module type")
    entries = _hw_port_entries("(" + typ.value.data + ")")
    if entries is None:
        raise ValueError("hierarchical source cannot read complete declared ports")
    ports = []
    for entry in entries:
        head, separator, type_text = entry.partition(":")
        words = head.split()
        if not separator or len(words) != 2 or words[0] not in {"input", "output"}:
            raise ValueError("hierarchical source refuses unsupported declared port bindings")
        ports.append(OriginalPort(words[0], words[1], type_text.strip()))
    if len({port.name for port in ports}) != len(ports):
        raise ValueError("hierarchical source has duplicate declared port names")
    if _name(op) == "hw.module":
        if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
            raise ValueError("hierarchical source requires one complete original body")
        block = op.regions[0].block
        output = block.last_op
        if output is None or _name(output) != "hw.output" or output.results or output.regions:
            raise ValueError("hierarchical source lacks its original output roster")
        for direction, values in (("input", block.args), ("output", output.operands)):
            selected = [port for port in ports if port.direction == direction]
            if len(selected) != len(values) or any(
                port.type != str(value.type) for port, value in zip(selected, values, strict=True)
            ):
                raise ValueError("hierarchical declaration differs from its complete original SSA")
    return tuple(ports)


def _binding(op, modules):
    module = _attribute(op, "moduleName")
    if (
        not isinstance(module, SymbolRefAttr)
        or module.nested_references.data
        or module.root_reference.data not in modules
    ):
        raise ValueError("hierarchical binding has no exact original callee")
    callee = modules[module.root_reference.data]
    if not (_parameters(op) or _parameters(callee)):
        return (*_instance(op, modules), None)
    instance = _attribute(op, "instanceName")
    if not isinstance(instance, StringAttr):
        raise ValueError("hierarchical binding lacks its original instance name")
    ports = _static_signature(callee)
    for field, direction, values in (("argNames", "input", op.operands), ("resultNames", "output", op.results)):
        names = _attribute(op, field)
        selected = [port for port in ports if port.direction == direction]
        if (
            not isinstance(names, ArrayAttr)
            or len(names.data) != len(selected)
            or len(values) != len(selected)
            or any(
                not isinstance(name, StringAttr) or name.data != port.name or str(value.type) != port.type
                for name, port, value in zip(names.data, selected, values, strict=True)
            )
        ):
            raise ValueError("parameterized binding lacks exact static named port/index/type correspondence")
    return instance.data, module.root_reference.data, ports, "parameterized_body"


def hierarchical_memory_bindings(parsed, *, root: str, local_limits: MemoryPortLimits, limits: HierarchyBindingLimits):
    """Preflight the complete rooted roster, then follow conditional scalar SSA."""
    if type(root) is not str or not root or type(limits) is not HierarchyBindingLimits:
        raise ValueError("hierarchical bindings require explicit original root and limits")
    local = memory_port_observations(parsed, limits=local_limits)
    local_units = {row["module"]: row for row in local["modules"]}
    modules = {}
    for op in parsed.walk():
        if _name(op) in {"hw.module", "hw.module.extern"}:
            name = _module_name(op)
            if name in modules or len(modules) >= limits.modules:
                raise ValueError("hierarchical source has duplicate modules or exceeds its module budget")
            modules[name] = op
    if root not in modules or _name(modules[root]) != "hw.module" or _parameters(modules[root]):
        raise ValueError("hierarchical source requires the exact selected unparameterized root body")
    definitions = {}

    def definition(module):
        if module in definitions:
            return definitions[module]
        op = modules[module]
        ports = _static_signature(op)
        block = op.regions[0].block if _name(op) == "hw.module" else None
        children, names = {}, set()
        if block is not None and not _parameters(op):
            for ordinal, child in enumerate(block.ops):
                if _name(child) == "hw.instance":
                    info = _binding(child, modules)
                    if info[0] in names or child.regions:
                        raise ValueError("hierarchical source has duplicate or unsupported instance bindings")
                    names.add(info[0])
                    children[child] = ordinal, info
        definitions[module] = ports, block, children
        return definitions[module]

    counts = ("occurrences", "memory_occurrences", "memory_ports", "port_bindings")
    sizes = {}

    def size(module, stop=None, ancestors=()):
        if len(ancestors) >= limits.hierarchy_depth or module in ancestors:
            raise ValueError("hierarchical roster is recursive or exceeds its declared depth budget")
        if stop is not None or _name(modules[module]) == "hw.module.extern":
            return dict(zip(counts, (1, 0, 0, 0), strict=True)), 1
        if module in sizes:
            result, depth = sizes[module]
            if len(ancestors) + depth > limits.hierarchy_depth:
                raise ValueError("hierarchical roster exceeds its declared depth budget")
            return result, depth
        _, _, children = definition(module)
        memories = local_units.get(module, {}).get("memories", [])
        result = dict(zip(counts, (1, len(memories), sum(len(row["ports"]) for row in memories), 0), strict=True))
        depth = 1
        for child, (_, (_, callee, ports, boundary)) in children.items():
            descendant, height = size(callee, boundary, ancestors + (module,))
            for field in counts:
                result[field] += descendant[field]
            result["port_bindings"] += len(ports)
            depth = max(depth, height + 1)
            if any(result[field] > getattr(limits, field) for field in counts):
                raise ValueError("complete hierarchical roster exceeds its pre-expansion metadata budget")
        if any(result[field] > getattr(limits, field) for field in counts):
            raise ValueError("complete hierarchical roster exceeds its pre-expansion metadata budget")
        sizes[module] = result, depth
        return result, depth

    expected, _ = size(root)
    frames, children_by_frame, blocks, frame_records = [], {}, {}, []

    def expand(module, path, parent=None, incoming=None, stop=None):
        ports, block, children = definition(module)
        index = len(frames)
        boundary = stop or ("external_body" if block is None else None)
        frames.append((module, block, parent, incoming, boundary))
        frame_records.append(
            {
                "id": index,
                "module": module,
                "path": path,
                "parent": parent,
                "stop": boundary,
                "original_ports": [asdict(port) for port in ports],
                "original_parameters": str(_attribute(modules[module], "parameters")),
                "original_legacy_parameters": str(_attribute(modules[module], "oldParameters")),
                "instance_parameters": str(_attribute(incoming, "parameters")) if incoming is not None else None,
                "instance_legacy_parameters": str(_attribute(incoming, "oldParameters"))
                if incoming is not None
                else None,
            }
        )
        if boundary is not None:
            return index
        blocks[index] = {op: ordinal for ordinal, op in enumerate(block.ops)}
        for child, (ordinal, (name, callee, _, child_stop)) in children.items():
            children_by_frame[index, child] = expand(
                callee, path + [{"ordinal": ordinal, "instance": name, "module": callee}], index, child, child_stop
            )
        return index

    expand(root, [])
    assert len(frames) == expected["occurrences"]
    nodes, visiting, totals = {}, set(), {**expected, "nodes": 0, "bit_work": 0}

    def width(value):
        typ = value.type
        if str(typ) == "!seq.clock":
            return None
        if (
            not isinstance(typ, IntegerType)
            or typ != IntegerType(typ.width.data)
            or not 0 < typ.width.data <= limits.scalar_bits
        ):
            raise ValueError("hierarchical scalar binding has unsupported type or exceeds its explicit width budget")
        return typ.width.data

    def trace(frame, value, depth=0):
        key = frame, value
        if key in nodes:
            return nodes[key]["id"]
        if key in visiting or depth >= limits.expression_depth:
            raise ValueError("hierarchical expression is cyclic or exceeds its declared depth budget")
        if totals["nodes"] >= limits.nodes:
            raise ValueError("hierarchical expression exceeds its complete node budget")
        bits = width(value)
        totals["nodes"] += 1
        totals["bit_work"] += bits or 0
        if totals["bit_work"] > limits.bit_work:
            raise ValueError("hierarchical expression exceeds its complete bit-work budget")
        module, block, parent, incoming, _ = frames[frame]
        row = {"id": totals["nodes"] - 1, "frame": frame, "type": str(value.type), "width": bits}
        visiting.add(key)
        if isinstance(value, BlockArgument) and value.block is block:
            ports = [port for port in definition(module)[0] if port.direction == "input"]
            row.update(ordinal=value.index, port=ports[value.index].name)
            if parent is None:
                row.update(kind="root_input")
            else:
                row.update(
                    kind="instance_input_binding", operands=[trace(parent, incoming.operands[value.index], depth + 1)]
                )
        elif isinstance(value, OpResult) and value.owner.parent is block:
            op = value.owner
            row.update(ordinal=blocks[frame][op], result_ordinal=value.index, operation=_name(op))
            if _name(op) == "hw.instance":
                child = children_by_frame[frame, op]
                callee, child_block, _, _, stop = frames[child]
                ports = [port for port in definition(callee)[0] if port.direction == "output"]
                row.update(child_frame=child, port=ports[value.index].name)
                if stop is not None:
                    row.update(kind="opaque_instance_result", stop=stop)
                else:
                    row.update(
                        kind="instance_output_binding",
                        operands=[trace(child, child_block.last_op.operands[value.index], depth + 1)],
                    )
            elif _name(op) in _STATE:
                row.update(kind="state_result")
            elif _name(op) in {"seq.firmem.read_port", "seq.firmem.read_write_port"}:
                row.update(kind="memory_read_result", memory_ordinal=blocks[frame][op.operands[0].owner])
            elif _name(op) in _COMBINATIONAL and bits is not None:
                if op.regions or len(op.results) != 1:
                    raise ValueError("hierarchical combinational producer has unsupported regions/results")
                operands = [trace(frame, operand, depth + 1) for operand in op.operands]
                operand_widths = [width(operand) for operand in op.operands]
                totals["bit_work"] += sum(operand_widths)
                if totals["bit_work"] > limits.bit_work:
                    raise ValueError("hierarchical expression exceeds its complete bit-work budget")
                try:
                    expression, parameter = _expression(op, operand_widths, bits, conditional_logic=True)
                except ValueError as error:
                    row.update(kind="unsupported_result", reason=str(error), operands=operands)
                else:
                    row.update(kind="combinational", expression=expression, parameter=parameter, operands=operands)
            else:
                row.update(kind="unsupported_result")
        else:
            raise ValueError("hierarchical source has a nonlocal or unavailable original SSA producer")
        visiting.remove(key)
        nodes[key] = row
        return row["id"]

    by_id = {}

    def interval(identity):
        row = by_id[identity]
        if (
            row["kind"]
            in {"root_input", "state_result", "memory_read_result", "opaque_instance_result", "unsupported_result"}
            and row["width"]
        ):
            return {"root": identity, "low_bit": 0, "width": row["width"]}
        if row["kind"] in {"instance_input_binding", "instance_output_binding"}:
            return interval(row["operands"][0])
        if row["kind"] == "combinational" and row["expression"] == "comb.extract":
            base = interval(row["operands"][0])
            if base:
                return {"root": base["root"], "low_bit": base["low_bit"] + row["parameter"], "width": row["width"]}
        if row["kind"] == "combinational" and row["expression"] == "comb.concat":
            parts = [interval(operand) for operand in reversed(row["operands"])]
            if parts and all(parts):
                first, size = parts[0], 0
                for part in parts:
                    if part["root"] != first["root"] or part["low_bit"] != first["low_bit"] + size:
                        return None
                    size += part["width"]
                return {"root": first["root"], "low_bit": first["low_bit"], "width": size}
        return None

    memories = []
    for frame, (module, block, _, _, stop) in enumerate(frames):
        if stop is not None or module not in local_units:
            continue
        operations = list(block.ops)
        for declaration in local_units[module]["memories"]:
            ports = []
            for original in declaration["ports"]:
                op = operations[original["ordinal"]]
                bound = _bindings(
                    op,
                    width=declaration["width"],
                    address_bits=max(1, (declaration["depth"] - 1).bit_length()),
                    mask_width=declaration["mask_width"],
                )
                refs = {field: trace(frame, value) for field, value in bound.items()}
                ports.append(
                    {
                        "ordinal": original["ordinal"],
                        "operation": original["operation"],
                        "bindings": refs,
                        "implicit_enable": original["implicit_enable"],
                        "implicit_write_mask": original["implicit_write_mask"],
                        "result_types": original["result_types"],
                    }
                )
            memories.append(
                {
                    "frame": frame,
                    "declaration": {key: value for key, value in declaration.items() if key != "ports"},
                    "ports": ports,
                }
            )
    by_id.update((row["id"], row) for row in nodes.values())
    for memory in memories:
        for port in memory["ports"]:
            port["data_source_interval"] = interval(port["bindings"]["data"]) if "data" in port["bindings"] else None
    assert len(memories) == expected["memory_occurrences"]
    assert sum(len(row["ports"]) for row in memories) == expected["memory_ports"]
    return {
        "schema": SCHEMA,
        "root": root,
        "frames": frame_records,
        "memories": memories,
        "expressions": sorted(nodes.values(), key=lambda row: row["id"]),
        "cost": totals,
        "limits": asdict(limits),
        "local_limits": asdict(local_limits),
        "unknowns": list(_UNKNOWN),
        "scope": "conditional hierarchical SSA connectivity only",
        "memory_contents_evaluated": False,
        "command_capacity_axis_or_temporal_admission": False,
    }

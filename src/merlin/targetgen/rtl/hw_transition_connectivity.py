"""Exact hierarchical source connectivity for every original state operand slot.

Operand roles come from the typed state reader. Original module ports supply
identity and types only: neither their names nor widths assign command meaning.
State, memory and unsupported expressions remain unevaluated source stops.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from xdsl.dialects.builtin import IntegerType
from xdsl.ir import BlockArgument, OpResult

from .hw_address_transitions import address_state_transitions
from .hw_array_selection import ArraySelectionLimits, preflight_array_selections, typed_array_selection
from .hw_combinational import _expression
from .hw_hierarchy_bindings import _binding, _static_signature
from .hw_memory_ports import _COMBINATIONAL
from .hw_observations import _module_name, _name

SCHEMA = "merlin.hw_transition_operand_connectivity.v1"
_STATE = {"seq.firreg", "seq.compreg", "seq.compreg.ce", "seq.shiftreg"}
_UNKNOWN = (
    "original_command_interface_semantic_correspondence",
    "decoded_command_and_resource_axis_roles",
    "clock_reset_initialization_and_state_reachability",
    "memory_values_collision_and_undefined_branch_validity",
    "unsupported_and_opaque_expression_semantics",
    "enabled_address_range_allocation_capacity_and_physical_tails",
    "physical_alias_order_completion_and_device_correspondence",
)


@dataclass(frozen=True)
class TransitionConnectivityLimits:
    frames: int
    port_bindings: int
    operand_bindings: int
    traversal_steps: int
    nodes: int
    scalar_bits: int
    bit_work: int
    expression_depth: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("transition connectivity requires explicit positive metadata limits")


def transition_operand_connectivity(
    parsed, *, root, local_limits, hierarchy_limits, transition_limits, limits, array_limits=None
):
    """Follow all original state operands; never select a port by a role hint."""
    if type(limits) is not TransitionConnectivityLimits:
        raise ValueError("transition connectivity requires explicit original traversal limits")
    if array_limits is not None and type(array_limits) is not ArraySelectionLimits:
        raise ValueError("array connectivity requires explicit whole-roster aggregate limits")
    transfers = address_state_transitions(
        parsed, root=root, local_limits=local_limits, hierarchy_limits=hierarchy_limits, limits=transition_limits
    )
    frames = transfers["source_frames"]
    expected = {
        "frames": len(frames),
        "port_bindings": sum(len(frame["original_ports"]) for frame in frames if frame["parent"] is not None),
        "operand_bindings": sum(len(row["operand_types"]) for row in transfers["state_transfers"]),
    }
    if any(value > getattr(limits, field) for field, value in expected.items()):
        raise ValueError("complete transfer roster exceeds its pre-connectivity metadata budget")
    totals = {**expected, "traversal_steps": 0, "nodes": 0, "bit_work": 0}

    def charge(field, amount=1):
        totals[field] += amount
        if totals[field] > getattr(limits, field):
            raise ValueError("transition connectivity exceeds its explicit " + field + " budget")

    modules = {_module_name(op): op for op in parsed.walk() if _name(op) in {"hw.module", "hw.module.extern"}}
    definitions = {}

    def definition(frame):
        module = frames[frame]["module"]
        if module not in definitions:
            op = modules[module]
            ports = _static_signature(op)
            block = op.regions[0].block if _name(op) == "hw.module" else None
            operations = list(block.ops) if block is not None else []
            definitions[module] = ports, block, operations, {op: index for index, op in enumerate(operations)}
        return definitions[module]

    incoming, children = {}, {}
    for index, frame in enumerate(frames):
        ports, _, _, _ = definition(index)
        if frame["id"] != index or frame["original_ports"] != [asdict(port) for port in ports]:
            raise ValueError("transition connectivity lost its exact original frame/port roster")
        parent = frame["parent"]
        if parent is None:
            if index != 0 or frame["path"] or frame["module"] != root:
                raise ValueError("transition connectivity lost its exact original root occurrence")
            continue
        if not 0 <= parent < index or not frame["path"]:
            raise ValueError("transition connectivity has an unsupported original parent occurrence")
        instance = definition(parent)[2][frame["path"][-1]["ordinal"]]
        name, callee, signature, boundary = _binding(instance, modules)
        if (
            frame["path"][:-1] != frames[parent]["path"]
            or frame["path"][-1]["instance"] != name
            or callee != frame["module"]
            or signature != ports
            or (boundary is not None and boundary != frame["stop"])
            or (parent, instance) in children
        ):
            raise ValueError("transition connectivity lacks exact original named port/index/type correspondence")
        incoming[index] = instance
        children[parent, instance] = index

    array_cost = None
    if array_limits is not None:
        original_modules = [frame["module"] for frame in frames if frame["stop"] is None]
        array_cost = preflight_array_selections(
            {module: definitions[module][2] for module in set(original_modules)},
            original_modules,
            limits=array_limits,
            scalar_bits=limits.scalar_bits,
        )

    def width(value):
        if str(value.type) == "!seq.clock":
            return None
        if (
            not isinstance(value.type, IntegerType)
            or value.type != IntegerType(value.type.width.data)
            or not 0 < value.type.width.data <= limits.scalar_bits
        ):
            raise ValueError("transition connectivity requires its exact bounded scalar type")
        return value.type.width.data

    # Preserve every original slot, including operands of unsupported state kinds.
    selected = []
    for transfer in transfers["state_transfers"]:
        op = definition(transfer["frame"])[2][transfer["ordinal"]]
        if (
            _name(op) != transfer["operation"]
            or [str(value.type) for value in op.operands] != transfer["operand_types"]
        ):
            raise ValueError("transition connectivity lost its complete original state operand roster")
        roles = ["next", "clock", *(["reset", "reset_value"] if len(op.operands) == 4 else [])]
        if _name(op) != "seq.firreg":
            roles = [None] * len(op.operands)
        for ordinal, (role, value) in enumerate(zip(roles, op.operands, strict=True)):
            width(value)
            selected.append((transfer, ordinal, role, value))
    nodes, visiting = {}, set()

    def trace(frame, value, depth=0):
        charge("traversal_steps")
        key = frame, value
        if key in nodes:
            return nodes[key]["id"]
        if key in visiting or depth >= limits.expression_depth:
            raise ValueError("transition connectivity is cyclic or exceeds its expression depth budget")
        charge("nodes")
        bits = width(value)
        charge("bit_work", bits or 0)
        ports, block, _, ordinals = definition(frame)
        row = {"id": totals["nodes"] - 1, "frame": frame, "type": str(value.type), "width": bits}
        visiting.add(key)
        if isinstance(value, BlockArgument) and value.block is block:
            inputs = [port for port in ports if port.direction == "input"]
            row.update(ordinal=value.index, port=inputs[value.index].name)
            parent = frames[frame]["parent"]
            if parent is None:
                row.update(kind="root_input")
            else:
                row.update(
                    kind="instance_input_binding",
                    operands=[trace(parent, incoming[frame].operands[value.index], depth + 1)],
                )
        elif isinstance(value, OpResult) and value.owner.parent is block:
            op = value.owner
            row.update(ordinal=ordinals[op], result_ordinal=value.index, operation=_name(op))
            if _name(op) == "hw.instance":
                child = children[frame, op]
                signature, child_block, _, _ = definition(child)
                outputs = [port for port in signature if port.direction == "output"]
                row.update(child_frame=child, port=outputs[value.index].name)
                if frames[child]["stop"] is not None:
                    row.update(kind="opaque_instance_result", stop=frames[child]["stop"])
                else:
                    row.update(
                        kind="instance_output_binding",
                        operands=[trace(child, child_block.last_op.operands[value.index], depth + 1)],
                    )
            elif _name(op) in _STATE:
                row.update(kind="state_result")
            elif _name(op) in {"seq.firmem.read_port", "seq.firmem.read_write_port"}:
                row.update(kind="memory_read_result", memory_ordinal=ordinals[op.operands[0].owner])
            elif _name(op) == "hw.array_get" and array_limits is not None:
                try:
                    selection = typed_array_selection(op, scalar_bits=limits.scalar_bits, limits=array_limits)
                except ValueError as error:
                    row.update(kind="unsupported_result", reason=str(error))
                else:
                    charge("bit_work", selection.index_width + len(selection.elements) * selection.element_width)
                    operands = [trace(frame, value, depth + 1) for value in (selection.index, *selection.elements)]
                    row.update(
                        operands=operands,
                        array_selection={
                            "creation_ordinal": ordinals[op.operands[0].owner],
                            "original_array_type": str(op.operands[0].type),
                            "element_count": len(selection.elements),
                            "runtime_index_to_creation_operand": list(reversed(range(len(selection.elements)))),
                            "index_width": selection.index_width,
                            "defined_index_maximum": len(selection.elements) - 1,
                            "full_original_index_domain_defined": selection.full_index_domain_defined,
                            "relation": "structural scalar dependencies only; no state or memory history",
                        },
                    )
                    if selection.full_index_domain_defined:
                        row.update(kind="combinational", expression="hw.array_get", parameter=len(selection.elements))
                    else:
                        row.update(kind="unsupported_result", reason="original index domain has out-of-range branches")
            elif _name(op) in _COMBINATIONAL and bits is not None:
                if op.regions or len(op.results) != 1:
                    raise ValueError("transition combinational producer has unsupported regions/results")
                widths = [width(operand) for operand in op.operands]
                if any(bits is None for bits in widths):
                    raise ValueError("transition combinational producer has a noninteger operand")
                charge("bit_work", sum(widths))
                operands = [trace(frame, operand, depth + 1) for operand in op.operands]
                try:
                    expression, parameter = _expression(op, widths, bits, conditional_logic=True)
                except ValueError as error:
                    row.update(kind="unsupported_result", reason=str(error), operands=operands)
                else:
                    row.update(kind="combinational", expression=expression, parameter=parameter, operands=operands)
            else:
                row.update(kind="unsupported_result")
        else:
            raise ValueError("transition connectivity has a nonlocal or unavailable original SSA producer")
        visiting.remove(key)
        nodes[key] = row
        return row["id"]

    operands = [
        {
            "state_expression": transfer["state_expression"],
            "frame": transfer["frame"],
            "state_ordinal": transfer["ordinal"],
            "operation": transfer["operation"],
            "operand_ordinal": ordinal,
            "primitive_role": role,
            "type": str(value.type),
            "expression": trace(transfer["frame"], value),
        }
        for transfer, ordinal, role, value in selected
    ]
    assert len(operands) == expected["operand_bindings"]
    record = {
        "schema": SCHEMA,
        "root": root,
        "source_frames": frames,
        "original_root_ports": frames[0]["original_ports"],
        "operand_bindings": operands,
        "expressions": sorted(nodes.values(), key=lambda row: row["id"]),
        "cost": totals,
        "limits": asdict(limits),
        "transition_limits": asdict(transition_limits),
        "hierarchy_limits": asdict(hierarchy_limits),
        "local_limits": asdict(local_limits),
        "unknowns": list(_UNKNOWN),
        "scope": "exact hierarchical original state operand connectivity only",
        "clock_events_evaluated": False,
        "command_axis_capacity_or_temporal_admission": False,
    }
    if array_limits is not None:
        record.update(array_selection_limits=asdict(array_limits), array_selection_cost=array_cost)
    return record

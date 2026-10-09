"""Exact local FirReg transfer expressions behind every rooted memory address.

The public Seq primitive defines the operand roles and reset priority. This
reader exposes those expressions without evaluating a clock event or proving
reachable state, decoded roles, address validity, capacity or temporal closure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr, UnitAttr
from xdsl.ir import BlockArgument, OpResult

from .hw_combinational import _expression
from .hw_hierarchy_bindings import HierarchyBindingLimits, _binding, _static_signature, hierarchical_memory_bindings
from .hw_memory_ports import _COMBINATIONAL
from .hw_observations import _attribute, _module_name, _name

SCHEMA = "merlin.hw_address_state_transitions.v1"
_STATE = {"seq.firreg", "seq.compreg", "seq.compreg.ce", "seq.shiftreg"}
_UNKNOWN = (
    "clock_event_history_and_reset_validity",
    "initial_values_and_state_reachability",
    "memory_read_values_and_collision_validity",
    "instance_and_unsupported_expression_semantics",
    "command_operands_and_decode_correspondence",
    "enabled_address_range_and_undefined_branch_validity",
    "software_axes_allocation_capacity_and_physical_tails",
    "physical_alias_order_completion_and_device_correspondence",
)


@dataclass(frozen=True)
class AddressTransitionLimits:
    address_bindings: int
    state_occurrences: int
    address_state_bindings: int
    traversal_steps: int
    operand_bindings: int
    nodes: int
    scalar_bits: int
    bit_work: int
    expression_depth: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("address transitions require explicit positive metadata limits")


def address_state_transitions(parsed, *, root, local_limits, hierarchy_limits, limits):
    """Recompute the complete address roster, then expose bounded local transfers."""
    if type(limits) is not AddressTransitionLimits or type(hierarchy_limits) is not HierarchyBindingLimits:
        raise ValueError("address transitions require explicit original hierarchy and transfer limits")
    hierarchy = hierarchical_memory_bindings(parsed, root=root, local_limits=local_limits, limits=hierarchy_limits)
    original = {row["id"]: row for row in hierarchy["expressions"]}
    if hierarchy["cost"]["memory_ports"] > limits.address_bindings:
        raise ValueError("complete address roster exceeds its pre-transfer budget")
    modules = {_module_name(op): op for op in parsed.walk() if _name(op) in {"hw.module", "hw.module.extern"}}
    totals = {
        field: 0
        for field in (
            "address_bindings",
            "state_occurrences",
            "address_state_bindings",
            "traversal_steps",
            "operand_bindings",
            "nodes",
            "bit_work",
        )
    }

    def charge(field, amount=1):
        totals[field] += amount
        if totals[field] > getattr(limits, field):
            raise ValueError("address transitions exceed their explicit " + field + " budget")

    addresses, selected, address_stops = [], set(), set()
    for memory in hierarchy["memories"]:
        for port in memory["ports"]:
            charge("address_bindings")
            pending, seen, stops = [port["bindings"]["address"]], set(), set()
            while pending:
                charge("traversal_steps")
                identity = pending.pop()
                if identity in seen:
                    continue
                seen.add(identity)
                row = original[identity]
                if row["kind"] in {"combinational", "instance_input_binding", "instance_output_binding"}:
                    pending.extend(row.get("operands", []))
                else:
                    stops.add(identity)
            states = sorted(identity for identity in stops if original[identity]["kind"] == "state_result")
            address_stops.update(stops)
            charge("address_state_bindings", len(states))
            for identity in states:
                if identity not in selected:
                    charge("state_occurrences")
                    selected.add(identity)
            address = original[port["bindings"]["address"]]
            depth = memory["declaration"]["depth"]
            addresses.append(
                {
                    "frame": memory["frame"],
                    "memory_ordinal": memory["declaration"]["ordinal"],
                    "port_ordinal": port["ordinal"],
                    "operation": port["operation"],
                    "address_expression": address["id"],
                    "address_type": address["type"],
                    "declared_depth": depth,
                    "enable_expression": port["bindings"].get("enable"),
                    "implicit_enable": port["implicit_enable"],
                    "state_expressions": states,
                    "other_stops": sorted(stops - set(states)),
                    "range_obligation": {
                        "statement": "when original enable is active, unsigned address index < declared depth",
                        "address_bits": address["width"],
                        "typed_index_domain_size_power_of_two": address["width"],
                        "declared_depth": depth,
                        "proved": False,
                    },
                }
            )
    frames = {row["id"]: row for row in hierarchy["frames"]}
    definitions = {}

    def definition(frame):
        module = frames[frame]["module"]
        if module not in definitions:
            op = modules[module]
            block = op.regions[0].block
            children = list(block.ops)
            definitions[module] = (
                block,
                children,
                {op: ordinal for ordinal, op in enumerate(children)},
                _static_signature(op),
            )
        return definitions[module]

    def width(value):
        if str(value.type) == "!seq.clock":
            return None
        if (
            not isinstance(value.type, IntegerType)
            or value.type != IntegerType(value.type.width.data)
            or not 0 < value.type.width.data <= limits.scalar_bits
        ):
            raise ValueError("address transfer requires its exact bounded scalar type")
        return value.type.width.data

    # Preflight all selected state operand rosters before any local expression work.
    bindings = {}
    for identity in sorted(selected):
        row = original[identity]
        _, children, _, _ = definition(row["frame"])
        op = children[row["ordinal"]]
        if (
            _name(op) != row["operation"]
            or row["result_ordinal"] >= len(op.results)
            or str(op.results[row["result_ordinal"]].type) != row["type"]
        ):
            raise ValueError("address state lost its exact original operation/result correspondence")
        charge("operand_bindings", len(op.operands))
        width(op.results[row["result_ordinal"]])
        if _name(op) != "seq.firreg":
            bindings[identity] = op, None
            continue
        if op.regions or len(op.results) != 1 or len(op.operands) not in {2, 4}:
            raise ValueError("FirReg requires its complete next/clock and optional reset/value operand roster")
        current = op.results[0]
        width(current)
        if op.operands[0].type != current.type or str(op.operands[1].type) != "!seq.clock":
            raise ValueError("FirReg next/clock types differ from the original primitive")
        if len(op.operands) == 4 and (op.operands[2].type != IntegerType(1) or op.operands[3].type != current.type):
            raise ValueError("FirReg reset/value types differ from the original primitive")
        if _attribute(op, "operandSegmentSizes") is not None or not isinstance(_attribute(op, "name"), StringAttr):
            raise ValueError("FirReg has unsupported segments or missing original name declaration")
        asynchronous, preset = _attribute(op, "isAsync"), _attribute(op, "preset")
        if asynchronous is not None and (not isinstance(asynchronous, UnitAttr) or len(op.operands) != 4):
            raise ValueError("FirReg asynchronous reset requires its exact reset operands and unit annotation")
        if preset is not None and (not isinstance(preset, IntegerAttr) or preset.type != current.type):
            raise ValueError("FirReg preset has a malformed original scalar type")
        bindings[identity] = (
            op,
            "asynchronous" if asynchronous is not None else "synchronous" if len(op.operands) == 4 else "absent",
        )

    nodes, visiting = {}, set()

    def trace(frame, value, depth=0):
        key = frame, value
        if key in nodes:
            return nodes[key]["id"]
        if key in visiting or depth >= limits.expression_depth:
            raise ValueError("address transfer expression is cyclic or exceeds its depth budget")
        charge("nodes")
        bits = width(value)
        charge("bit_work", bits or 0)
        block, _, ordinals, ports = definition(frame)
        row = {"id": totals["nodes"] - 1, "frame": frame, "type": str(value.type), "width": bits}
        visiting.add(key)
        if isinstance(value, BlockArgument) and value.block is block:
            inputs = [port for port in ports if port.direction == "input"]
            row.update(kind="local_module_input", ordinal=value.index, port=inputs[value.index].name)
        elif isinstance(value, OpResult) and value.owner.parent is block:
            op = value.owner
            row.update(ordinal=ordinals[op], result_ordinal=value.index, operation=_name(op))
            if _name(op) in _STATE:
                row.update(kind="state_result")
            elif _name(op) in {"seq.firmem.read_port", "seq.firmem.read_write_port"}:
                row.update(kind="memory_read_result", memory_ordinal=ordinals[op.operands[0].owner])
            elif _name(op) == "hw.instance":
                name, callee, signature, boundary = _binding(op, modules)
                outputs = [port for port in signature if port.direction == "output"]
                row.update(
                    kind="instance_result",
                    instance=name,
                    callee=callee,
                    port=outputs[value.index].name,
                    stop=boundary or "local_instance_boundary",
                )
            elif _name(op) in _COMBINATIONAL and bits is not None:
                if op.regions or len(op.results) != 1:
                    raise ValueError("address transfer combinational producer has unsupported regions/results")
                widths = [width(operand) for operand in op.operands]
                if any(bits is None for bits in widths):
                    raise ValueError("address transfer combinational producer has a noninteger operand")
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
            raise ValueError("address transfer has a nonlocal or unavailable original SSA producer")
        visiting.remove(key)
        nodes[key] = row
        return row["id"]

    transfers = []
    for identity, (op, reset_kind) in bindings.items():
        row = original[identity]
        transfer = {
            "state_expression": identity,
            "frame": row["frame"],
            "module": frames[row["frame"]]["module"],
            "ordinal": row["ordinal"],
            "result_ordinal": row["result_ordinal"],
            "operation": _name(op),
            "result_type": row["type"],
            "operand_types": [str(value.type) for value in op.operands],
            "original_attributes": {key: str(value) for key, value in op.attributes.items()},
            "original_properties": {key: str(value) for key, value in op.properties.items()},
            "relation": None,
            "hold_mux": None,
        }
        if reset_kind is None:
            transfer.update(stop="unsupported_state_primitive")
        else:
            refs = dict(
                zip(
                    ["next", "clock", *(["reset", "reset_value"] if len(op.operands) == 4 else [])],
                    (trace(row["frame"], value) for value in op.operands),
                    strict=True,
                )
            )
            transfer.update(
                reset_kind=reset_kind, bindings=refs, initialization="unproved", preset=str(_attribute(op, "preset"))
            )
            if reset_kind == "asynchronous":
                transfer.update(stop="asynchronous_event_relation_unsupported")
            else:
                transfer["relation"] = {
                    "primitive": "seq.firreg",
                    "clock_event": "primitive-declared rising clock edge; event validity unproved",
                    "reset_priority": "reset_value if reset is active, otherwise next"
                    if reset_kind == "synchronous"
                    else "no reset operands; next",
                    "clock": refs["clock"],
                    "next": refs["next"],
                    "reset": refs.get("reset"),
                    "reset_value": refs.get("reset_value"),
                    "evaluated": False,
                }
                nxt = op.operands[0]
                expression = nodes[row["frame"], nxt]
                if isinstance(nxt, OpResult) and expression.get("expression") == "comb.mux":
                    arms = nxt.owner.operands[1:]
                    holds = [index == 0 for index, value in enumerate(arms) if value is op.results[0]]
                    if holds:
                        transfer["hold_mux"] = {
                            "condition": expression["operands"][0],
                            "hold_when_condition": holds,
                            "true_value": expression["operands"][1],
                            "false_value": expression["operands"][2],
                            "identity": "exact same original register result SSA",
                            "applies": "ordinary next branch only; reset retains priority",
                        }
        transfers.append(transfer)
    return {
        "schema": SCHEMA,
        "root": root,
        "source_frames": hierarchy["frames"],
        "memory_declarations": [
            {"frame": memory["frame"], "declaration": memory["declaration"]} for memory in hierarchy["memories"]
        ],
        "addresses": addresses,
        "address_stop_expressions": [original[identity] for identity in sorted(address_stops)],
        "state_transfers": transfers,
        "expressions": sorted(nodes.values(), key=lambda row: row["id"]),
        "cost": totals,
        "limits": asdict(limits),
        "hierarchy_limits": asdict(hierarchy_limits),
        "local_limits": asdict(local_limits),
        "unknowns": list(_UNKNOWN),
        "scope": "local typed source transfer relation only",
        "clock_events_evaluated": False,
        "temporal_or_command_capacity_axis_admission": False,
    }

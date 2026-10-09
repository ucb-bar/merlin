"""Conditional unsigned type-domain containment for exact original memory ports.

A defined known N-bit index lies in [0, 2**N). Containment in original declared
depth proves no definedness, state/event validity, allocation or physical effect.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from xdsl.dialects.builtin import IntegerType
from xdsl.ir import BlockArgument, OpResult

from .hw_address_transitions import address_state_transitions
from .hw_memory_ports import _bindings, _memory_type
from .hw_observations import _module_name, _name

SCHEMA = "merlin.hw_conditional_index_ranges.v1"
_UNKNOWN = (
    "original_address_defined_known_bit_premise",
    "enabled_state_and_event_validity",
    "initialization_and_memory_history",
    "command_resource_axis_allocation_capacity_and_effect_correspondence",
)


@dataclass(frozen=True)
class IndexRangeLimits:
    addresses: int
    scalar_bits: int
    depth_bits: int
    proof_bits: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("index ranges require explicit positive whole-roster budgets")


def _domain_cost(typ, depth, limits):
    if type(limits) is not IndexRangeLimits:
        raise ValueError("index ranges require explicit type/depth budgets")
    if (
        not isinstance(typ, IntegerType)
        or type(typ.width.data) is not int
        or typ != IntegerType(typ.width.data)
        or not 0 < typ.width.data <= limits.scalar_bits
    ):
        raise ValueError("index range requires its bounded original signless integer type")
    if type(depth) is not int or depth <= 0 or depth.bit_length() > limits.depth_bits:
        raise ValueError("index range requires its bounded positive original declared depth")
    return typ.width.data + depth.bit_length()


def known_unsigned_index_domain(typ, depth, *, limits):
    """A symbolic known-bit domain only; allocate no enormous endpoint integers."""
    if _domain_cost(typ, depth, limits) > limits.proof_bits:
        raise ValueError("index range exceeds its pre-expansion proof budget")
    return {
        "original_address_type": str(typ),
        "address_bits": typ.width.data,
        "declared_depth": depth,
        "unsigned_minimum": 0,
        "unsigned_upper_bound_exclusive_power_of_two": typ.width.data,
        "conditional_domain_contained": typ.width.data < depth.bit_length(),
        "premise": "original address is defined known bits interpreted as an unsigned index",
        "address_definedness_proved": False,
    }


def index_range_observations(parsed, *, root, local_limits, hierarchy_limits, transition_limits, limits):
    """Recompute all original addresses and exact native joins before range proofs."""
    if type(limits) is not IndexRangeLimits:
        raise ValueError("index ranges require explicit original complete-roster budgets")
    transfers = address_state_transitions(
        parsed, root=root, local_limits=local_limits, hierarchy_limits=hierarchy_limits, limits=transition_limits
    )
    addresses = transfers["addresses"]
    if len(addresses) > limits.addresses:
        raise ValueError("complete original address roster exceeds its pre-expansion range budget")
    modules = {
        _module_name(op): (op.regions[0].block, list(op.regions[0].block.ops))
        for op in parsed.walk()
        if _name(op) == "hw.module"
    }
    frames = {row["id"]: row for row in transfers["source_frames"]}
    declarations = {
        (row["frame"], row["declaration"]["ordinal"]): row["declaration"] for row in transfers["memory_declarations"]
    }

    def original_binding(row):
        frame = frames[row["frame"]]
        block, operations = modules[frame["module"]]
        port, memory = operations[row["port_ordinal"]], operations[row["memory_ordinal"]]
        if (
            _name(port) != row["operation"]
            or _name(memory) != "seq.firmem"
            or not isinstance(port.operands[0], OpResult)
            or port.operands[0].owner is not memory
            or port.operands[0].index != 0
        ):
            raise ValueError("index range lost its exact original port/declaration correspondence")
        depth, width, mask = _memory_type(port.operands[0].type)
        address = _bindings(port, width=width, address_bits=max(1, (depth - 1).bit_length()), mask_width=mask)[
            "address"
        ]
        declaration = declarations[row["frame"], row["memory_ordinal"]]
        if (
            depth != declaration["depth"]
            or depth != row["declared_depth"]
            or str(port.operands[0].type) != declaration["type"]
            or str(address.type) != row["address_type"]
            or address.type.width.data != row["range_obligation"]["address_bits"]
        ):
            raise ValueError("index range differs from its exact original type/depth fields")
        if isinstance(address, OpResult) and address.owner.parent is block:
            source = {"operation_ordinal": operations.index(address.owner), "result_ordinal": address.index}
        elif isinstance(address, BlockArgument) and address.block is block:
            source = {"argument_ordinal": address.index}
        else:
            raise ValueError("index range has a nonlocal original address SSA")
        return frame, port, address, depth, source

    # Count the entire original roster before constructing any domain/proof rows.
    total_bits = 0
    seen = set()
    for row in addresses:
        identity = row["frame"], row["memory_ordinal"], row["port_ordinal"]
        if identity in seen:
            raise ValueError("index range has duplicate original address membership")
        seen.add(identity)
        _, _, address, depth, _ = original_binding(row)
        total_bits += _domain_cost(address.type, depth, limits)
        if total_bits > limits.proof_bits:
            raise ValueError("complete original range proof roster exceeds its pre-expansion budget")
    rows = []
    for row in addresses:
        frame, port, address, depth, source = original_binding(row)
        rows.append(
            {
                "frame": row["frame"],
                "module": frame["module"],
                "path": frame["path"],
                "memory_ordinal": row["memory_ordinal"],
                "port_ordinal": row["port_ordinal"],
                "operation": row["operation"],
                "original_port_operand_types": [str(value.type) for value in port.operands],
                "original_port_result_types": [str(value.type) for value in port.results],
                "original_address_source": {**source, "type": str(address.type)},
                "original_range_obligation": row["range_obligation"],
                "known_bit_domain": known_unsigned_index_domain(address.type, depth, limits=limits),
                "unknowns": list(_UNKNOWN),
            }
        )
    return {
        "schema": SCHEMA,
        "root": root,
        "source_frames": transfers["source_frames"],
        "addresses": rows,
        "cost": {"addresses": len(rows), "proof_bits": total_bits},
        "limits": asdict(limits),
        "transition_limits": asdict(transition_limits),
        "hierarchy_limits": asdict(hierarchy_limits),
        "local_limits": asdict(local_limits),
        "unknowns": list(_UNKNOWN),
        "state_events_evaluated": False,
        "address_definedness_granted": False,
        "command_capacity_axis_or_effect_admission": False,
    }

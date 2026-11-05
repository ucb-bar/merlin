"""Conditional bit wiring from original local partitions to memory data ports.

Intermediate module input and instance result identities survive hierarchical
bindings. Only extracts and concatenations preserve bit correspondence here;
all other producers stop the relation. No source value or temporal event is
evaluated, and no software, command, tensor or physical role is assigned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.ir import OpResult

from .hw_combinational import _expression
from .hw_hierarchy_bindings import HierarchyBindingLimits, hierarchical_memory_bindings
from .hw_memory_ports import MemoryPortLimits
from .hw_observations import _attribute, _name
from .hw_packing import equal_partitions

SCHEMA = "merlin.hw_partition_memory_bindings.v1"
_WIRE = {"comb.extract", "comb.concat"}


@dataclass(frozen=True)
class PartitionMemoryLimits:
    """Whole source and relation metadata bounds, never memory value counts."""

    operations: int
    extractions: int
    partitions: int
    slices: int
    local_trace_work: int
    root_match_work: int
    interval_pieces: int
    cut_memberships: int
    traversal_work: int
    expression_depth: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("partition memory bindings require explicit positive metadata limits")


@dataclass(frozen=True)
class _Interval:
    partition: int
    frame: int
    source_low_bit: int
    destination_low_bit: int
    width: int


def _preflight(parsed, hierarchy_limits, limits):
    operations, extractions, transparent = 0, 0, {}
    for op in parsed.walk():
        operations += 1
        if operations > limits.operations:
            raise ValueError("complete partition source exceeds its operation budget")
        if _name(op) not in _WIRE:
            continue
        if op.regions or len(op.results) != 1:
            raise ValueError("partition source has unsupported original wire regions/results")
        values = (*op.operands, *op.results)
        widths = []
        for value in values:
            typ = value.type
            if (
                not isinstance(typ, IntegerType)
                or typ != IntegerType(typ.width.data)
                or not 0 < typ.width.data <= hierarchy_limits.scalar_bits
            ):
                raise ValueError("partition wire requires bounded original signless scalar types")
            widths.append(typ.width.data)
        _expression(op, widths[:-1], widths[-1])
        if _name(op) == "comb.extract":
            low = _attribute(op, "lowBit")
            if not isinstance(low, IntegerAttr) or low.type != IntegerType(32):
                raise ValueError("partition extract requires its original i32 low-bit field")
        elif _attribute(op, "twoState") is not None:
            raise ValueError("partition concatenation has an unsupported original annotation")
        transparent[op] = tuple(op.operands)
        extractions += int(_name(op) == "comb.extract")
        if extractions > limits.extractions:
            raise ValueError("complete partition source exceeds its extraction budget")

    # Admit every transparent path before the historical local partition reader
    # performs recursive tracing, including paths unused by a memory endpoint.
    heights, weights, visiting = {}, {}, set()

    def height(op):
        if op in heights:
            return heights[op]
        if op in visiting or len(visiting) >= limits.expression_depth:
            raise ValueError("partition wire source is cyclic or exceeds its depth budget")
        visiting.add(op)
        depth, weight = 1, 1
        for value in transparent[op]:
            if isinstance(value, OpResult) and value.owner in transparent:
                depth = max(depth, 1 + height(value.owner))
                weight += weights[value.owner]
            else:
                weight += 1
            if weight > limits.local_trace_work:
                raise ValueError("partition wire source exceeds its pre-expansion local trace budget")
        if depth > limits.expression_depth:
            raise ValueError("partition wire source exceeds its depth budget")
        visiting.remove(op)
        heights[op] = depth
        weights[op] = weight
        return depth

    for op in transparent:
        height(op)
    trace_work = sum(weights[op] for op in transparent if _name(op) == "comb.extract")
    if trace_work > limits.local_trace_work:
        raise ValueError("complete partition source exceeds its pre-expansion local trace budget")
    return operations, extractions, trace_work


def partition_memory_bindings(
    parsed,
    *,
    root: str,
    local_limits: MemoryPortLimits,
    hierarchy_limits: HierarchyBindingLimits,
    limits: PartitionMemoryLimits,
):
    """Derive fresh same-source occurrence intervals; retain all original ports.

    A returned interval states bit identity conditional on defined known bits.
    It does not prove that a write occurs, that the value is a software scalar,
    or that an opaque result carries an earlier command or memory read.
    """
    if (
        type(limits) is not PartitionMemoryLimits
        or type(hierarchy_limits) is not HierarchyBindingLimits
        or type(local_limits) is not MemoryPortLimits
    ):
        raise ValueError("partition bindings require all original explicit limits")
    operations, extractions, local_trace_work = _preflight(parsed, hierarchy_limits, limits)
    partitions = equal_partitions(parsed)
    rows = partitions["partitions"]
    slices = sum(len(row["slices"]) for row in rows)
    if len(rows) > limits.partitions or slices > limits.slices:
        raise ValueError("complete local partition roster exceeds its relation budget")
    hierarchy = hierarchical_memory_bindings(parsed, root=root, local_limits=local_limits, limits=hierarchy_limits)
    nodes = {row["id"]: row for row in hierarchy["expressions"]}
    frames = {row["id"]: row for row in hierarchy["frames"]}
    match_work = len(nodes) * len(rows)
    if match_work > limits.root_match_work:
        raise ValueError("complete occurrence join exceeds its pre-expansion match budget")
    roots = {}
    for identity, node in nodes.items():
        matches = []
        for ordinal, partition in enumerate(rows):
            if partition["module"] != frames[node["frame"]]["module"] or node["width"] != partition["packed_width"]:
                continue
            original = partition["root"]
            if original["kind"] == "module_input":
                matched = (
                    node["kind"] in {"root_input", "instance_input_binding"}
                    and node.get("ordinal") == original["ordinal"]
                    and node.get("port") == original["name"]
                )
            else:
                matched = (
                    node["kind"] in {"opaque_instance_result", "instance_output_binding"}
                    and node.get("ordinal") == original["producer_ordinal"]
                    and node.get("result_ordinal") == original["output_ordinal"]
                    and node.get("port") == original["output"]
                    and frames[node["child_frame"]]["module"] == original["module"]
                    and frames[node["child_frame"]]["path"][-1]["instance"] == original["instance"]
                )
            if matched:
                matches.append(ordinal)
        roots[identity] = tuple(matches)

    memo, visiting = {}, set()
    cost = {"operations": operations, "extractions": extractions, "partitions": len(rows), "slices": slices}
    cost.update(
        local_trace_work=local_trace_work,
        root_match_work=match_work,
        interval_pieces=0,
        cut_memberships=0,
        traversal_work=0,
    )

    def charge(field, amount):
        if cost[field] + amount > getattr(limits, field):
            raise ValueError("complete partition relation exceeds its " + field + " budget")
        cost[field] += amount

    def trace(identity):
        if identity in memo:
            return memo[identity]
        if identity in visiting or len(visiting) >= limits.expression_depth:
            raise ValueError("partition relation is cyclic or exceeds its depth budget")
        visiting.add(identity)
        node = nodes[identity]
        charge("traversal_work", 1)
        result, cuts = [], set()

        def append(piece):
            charge("interval_pieces", 1)
            result.append(piece)

        def merge(stops):
            charge("cut_memberships", len(stops))
            cuts.update(stops)

        for partition in roots[identity]:
            append(_Interval(partition, node["frame"], 0, 0, node["width"]))
        operands = node.get("operands", [])
        if node["kind"] in {"instance_input_binding", "instance_output_binding"}:
            if len(operands) != 1 or nodes[operands[0]]["width"] != node["width"]:
                raise ValueError("partition relation differs from its original typed binding")
            pieces, stops = trace(operands[0])
            merge(stops)
            for piece in pieces:
                charge("traversal_work", 1)
                append(piece)
        elif node["kind"] == "combinational" and node["expression"] == "comb.extract":
            low = node["parameter"]
            pieces, stops = trace(operands[0])
            merge(stops)
            for piece in pieces:
                charge("traversal_work", 1)
                start = max(low, piece.destination_low_bit)
                end = min(low + node["width"], piece.destination_low_bit + piece.width)
                if start < end:
                    append(
                        _Interval(
                            piece.partition,
                            piece.frame,
                            piece.source_low_bit + start - piece.destination_low_bit,
                            start - low,
                            end - start,
                        )
                    )
        elif node["kind"] == "combinational" and node["expression"] == "comb.concat":
            low = 0
            for operand in reversed(operands):
                charge("traversal_work", 1)
                pieces, stops = trace(operand)
                merge(stops)
                for piece in pieces:
                    charge("traversal_work", 1)
                    append(
                        _Interval(
                            piece.partition,
                            piece.frame,
                            piece.source_low_bit,
                            low + piece.destination_low_bit,
                            piece.width,
                        )
                    )
                low += nodes[operand]["width"]
            if low != node["width"]:
                raise ValueError("partition relation differs from its original concatenation width")
        else:
            # Roots are recorded even when their upstream producer is stopped.
            # An opaque output's own bit partition grants no output value.
            merge((identity,))
        visiting.remove(identity)
        memo[identity] = tuple(result), frozenset(cuts)
        return memo[identity]

    ports = []
    for memory in hierarchy["memories"]:
        for port in memory["ports"]:
            data = port["bindings"].get("data")
            pieces, cuts = trace(data) if data is not None else ((), ())
            charge("interval_pieces", len(pieces))
            charge("cut_memberships", len(cuts))
            ports.append(
                {
                    "frame": memory["frame"],
                    "memory_ordinal": memory["declaration"]["ordinal"],
                    "port_ordinal": port["ordinal"],
                    "operation": port["operation"],
                    "data_expression": data,
                    "intervals": [asdict(piece) for piece in pieces],
                    "cuts": sorted(cuts),
                    "data_status": "conditional_bit_relation"
                    if pieces
                    else "unproved"
                    if data is not None
                    else "no_write_data_operand",
                }
            )
    return {
        "schema": SCHEMA,
        "root": root,
        "local_partitions": partitions,
        "hierarchy": hierarchy,
        "ports": ports,
        "cost": cost,
        "limits": asdict(limits),
        "scope": "exact occurrence bit wiring conditional on defined known bits",
        "unknowns": hierarchy["unknowns"],
        "source_values_evaluated": False,
        "command_capacity_axis_or_temporal_admission": False,
        "packing_mapping_admission": False,
    }

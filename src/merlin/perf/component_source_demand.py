"""Bounded logical demand from original functionalized tensor programs.

An explicit topological schedule determines when values are last needed. These
are eager logical tensor payloads, not allocated buffers, bus traffic, physical
capacity, asynchronous lifetimes or predicted costs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from merlin.common.jsonio import canonical_sha256
from merlin.targetgen import component_program

SCHEMA = "merlin.component_source_demand.v1"


@dataclass(frozen=True)
class SourceDemandLimits:
    max_inputs: int
    max_nodes: int
    max_outputs: int
    max_scalar_bits: int
    max_extent_bits: int
    max_payload_bits: int

    def verify(self):
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("source demand requires explicit positive metadata/arithmetic limits")


def _width(dtype, limits):
    if (
        type(dtype) is not str
        or len(dtype) > len(str(limits.max_scalar_bits)) + 1
        or not dtype.startswith("i")
        or not dtype[1:].isascii()
        or not dtype[1:].isdigit()
    ):
        raise ValueError("source demand scalar format is unsupported")
    width = int(dtype[1:])
    if not 1 <= width <= limits.max_scalar_bits or dtype != "i" + str(width):
        raise ValueError("source demand scalar width exceeds its explicit limit")
    return width


def _shape(shape, limits):
    if (
        type(shape) is not list
        or len(shape) != 2
        or any(type(value) is not int or value < 1 or value.bit_length() > limits.max_extent_bits for value in shape)
    ):
        raise ValueError("source demand requires bounded positive rank-two extents")


def _metadata(program, operand_dtype, accumulator_dtype, limits):
    """Bound the closed source before analyzer copying or large arithmetic."""
    if type(program) is not dict or set(program) != {"inputs", "nodes", "outputs"}:
        raise ValueError("source demand needs the original closed tensor program")
    for key, maximum in (("inputs", limits.max_inputs), ("nodes", limits.max_nodes), ("outputs", limits.max_outputs)):
        if type(program[key]) is not list or len(program[key]) > maximum:
            raise ValueError("source demand metadata limit exceeded: " + key)
    operand, accumulator = _width(operand_dtype, limits), _width(accumulator_dtype, limits)
    extents, widths = [], [operand, accumulator]
    for row in program["inputs"]:
        if type(row) is not dict or set(row) != {"name", "role", "shape", "dtype"}:
            raise ValueError("source demand input layout or fields are unsupported")
        _shape(row["shape"], limits)
        dtype = {"operand": operand_dtype, "accumulator": accumulator_dtype}.get(row["dtype"], row["dtype"])
        widths.append(_width(dtype, limits))
        extents.extend(value.bit_length() for value in row["shape"])
    for row in program["nodes"]:
        if (
            type(row) is not dict
            or set(row) != {"name", "op", "inputs"}
            or type(row["op"]) is not str
            or not 1 <= len(row["op"]) <= 128
            or type(row["inputs"]) is not list
            or len(row["inputs"]) > 2
        ):
            raise ValueError("source demand node layout or fields are unsupported")
    for row in program["outputs"]:
        if type(row) is not dict or set(row) != {"name", "value"}:
            raise ValueError("source demand needs the complete original output roster")
    names = [row["name"] for row in program["inputs"] + program["nodes"] + program["outputs"]]
    references = [name for row in program["nodes"] for name in row["inputs"]]
    references += [row["value"] for row in program["outputs"]]
    if any(type(name) is not str or not 1 <= len(name) <= 128 for name in names + references):
        raise ValueError("source demand identifiers exceed the closed metadata grammar")
    # Three extents bound contraction work; two bound tensor payload. Include
    # scalar container width and the whole roster before products or summation.
    bits = 3 * max(extents, default=1) + max(widths).bit_length()
    bits += (len(names) + 1).bit_length()
    if bits > limits.max_payload_bits:
        raise ValueError("source demand payload arithmetic exceeds its explicit bit budget")


def derive_source_demand(*, program, operand_dtype, accumulator_dtype, schedule, limits):
    """Derive complete source features or explicit UNKNOWN, never a timing rank.

    ``schedule`` contains every original node once. It orders already
    functionalized SSA dependencies; it does not change source alias/epoch
    resolution, numerical order within operations or publication membership.
    Unsupported source grammar is UNKNOWN; malformed schedule selection refuses.
    No shaped values, inputs or references are allocated by this computation.
    """
    if type(limits) is not SourceDemandLimits:
        raise ValueError("source demand requires the exact typed metadata limits")
    limits.verify()
    result = {
        "schema": SCHEMA,
        "scope": "logical source demand only",
        "limits": asdict(limits),
        "authority": "none",
        "physical_costs": "UNKNOWN",
        "payload_basis": "ceil(scalar bits/8) per logical tensor element; not a physical encoding",
        "liveness_basis": "eager SSA values at each operation, released after last source use/publication",
        "operand_basis": "one complete logical operand payload per non-alias operand occurrence; not traffic",
    }
    try:
        _metadata(program, operand_dtype, accumulator_dtype, limits)
        typed = component_program.analyze(program, operand_dtype=operand_dtype, accumulator_dtype=accumulator_dtype)
    except (ValueError, TypeError, KeyError) as error:
        return {**result, "status": "UNKNOWN", "missing": [str(error)]}
    nodes = {row["name"]: row for row in typed["nodes"]}
    if (
        type(schedule) is not tuple
        or len(schedule) != len(nodes)
        or any(type(name) is not str or not 1 <= len(name) <= 128 for name in schedule)
        or len(set(schedule)) != len(schedule)
        or set(schedule) != set(nodes)
    ):
        raise ValueError("source demand schedule must contain every original node exactly once")
    position = {name: index for index, name in enumerate(schedule)}
    for row in typed["nodes"]:
        if any(name in nodes and position[name] >= position[row["name"]] for name in row["actual_inputs"]):
            raise ValueError("source demand schedule violates original SSA dependencies")
    types = {row["name"]: row for row in typed["inputs"] + typed["nodes"]}
    canonical, dependencies, depth = {}, {}, {}
    for row in typed["inputs"]:
        canonical[row["name"]], dependencies[row["name"]], depth[row["name"]] = row["name"], [], 0
    for row in typed["nodes"]:
        name, args = row["name"], row["actual_inputs"]
        canonical[name] = canonical[args[0]] if row["op"] == "alias" else name
        dependencies[name] = [canonical[value] for value in args]
        depth[name] = 1 + max(depth[value] for value in args)
    publication = [{**row, "logical_value": canonical[row["actual_value"]]} for row in typed["outputs"]]
    publish_at = len(schedule)
    values = {name: row for name, row in types.items() if canonical[name] == name}
    extents = {name: row["shape"][0] * row["shape"][1] for name, row in values.items()}
    payload = {name: extents[name] * ((_width(row["dtype"], limits) + 7) // 8) for name, row in values.items()}
    starts = {name: position.get(name, -1) for name in values}
    consumers = {name: [] for name in values}
    operand_payload, macs, element_results = 0, 0, 0
    actions = []
    for name in schedule:
        row, args = nodes[name], dependencies[name]
        # An alias is a source value relation, not a tensor element visit.
        reads = 0 if row["op"] == "alias" else sum(payload[value] for value in args)
        writes = 0 if row["op"] == "alias" else payload[name]
        for value in args:
            consumers[value].append(position[name])
        node_macs = (
            (
                types[row["actual_inputs"][0]]["shape"][0]
                * types[row["actual_inputs"][0]]["shape"][1]
                * types[row["actual_inputs"][1]]["shape"][1]
            )
            if row["op"] == "matmul"
            else 0
        )
        operand_payload += reads
        macs += node_macs
        element_results += 0 if row["op"] == "alias" else extents[name]
        actions.append(
            {
                "node": name,
                "operation": row["op"],
                "original_ordinal": row["ordinal"],
                "schedule_index": position[name],
                "logical_inputs": args,
                "operand_payload_bytes": reads,
                "result_payload_bytes": writes,
                "macs": node_macs,
                "dependency_depth": depth[name],
            }
        )
    published = {row["logical_value"] for row in publication}
    ends = {
        name: max(consumers[name] + ([publish_at] if name in published else []), default=starts[name])
        for name in values
    }
    intervals = [
        {
            "value": name,
            "defined_at": starts[name],
            "last_needed_at": ends[name],
            "payload_bytes": payload[name],
            "consumer_positions": consumers[name],
            "published": name in published,
        }
        for name in values
    ]
    boundaries = []
    for index in range(-1, publish_at + 1):
        live = [name for name in values if starts[name] <= index <= ends[name]]
        boundaries.append({"position": index, "values": live, "payload_bytes": sum(payload[name] for name in live)})
    demand = {
        "input_payload_bytes": sum(payload[row["name"]] for row in typed["inputs"]),
        "materialized_result_payload_bytes": sum(payload[name] for name in nodes if canonical[name] == name),
        "operand_payload_bytes": operand_payload,
        "publication_payload_bytes": sum(payload[row["logical_value"]] for row in publication),
        "macs": macs,
        "element_results": element_results,
        "dependency_depth": max(depth.values(), default=0),
        "peak_eager_logical_payload_bytes": max(row["payload_bytes"] for row in boundaries),
        "multiple_consumer_values": sum(len(set(uses)) > 1 for uses in consumers.values()),
        "reused_weight_values": sum(
            len(set(consumers[row["name"]])) > 1 for row in typed["inputs"] if row["role"] == "weight"
        ),
    }
    return {
        **result,
        "status": "observed",
        "missing": [],
        "source_program_sha256": canonical_sha256(program),
        "typed_program_sha256": canonical_sha256(typed),
        "schedule": list(schedule),
        "selected_storage": typed["selected_storage"],
        "actions": actions,
        "publication": publication,
        "logical_aliases": typed["logical_aliases"],
        "logical_epochs": typed["logical_epochs"],
        "demand": demand,
        "value_intervals": intervals,
        "boundaries": boundaries,
        "unproved": [
            "physical allocation/address/capacity",
            "traffic and overlap",
            "async lifetime/completion",
            "cycles/counter units",
            "cold/warm costs",
            "ranking/calibration",
            "hardware applicability",
        ],
    }

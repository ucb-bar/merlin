"""Independent source-only graph pairs from the existing generic topology owner.

Only original tensor semantics are emitted. Logical aliases and epochs become
SSA; the source pair proves no physical reuse, device ordering or numerics.
Metadata counts are derived from fixed bounded source prototypes before unroll.
"""

from __future__ import annotations

import copy

from . import component_graph_variants as G
from .rtl_intake import RtlIntakeRefusal

SCHEMA = "merlin.component_compile_only_plan.v2"
GRAPH_FIELDS = {"operation_owners", "graph_family", "depth", "fanout"}
OPERATIONS = ("matmul", "copy", "add")


def is_graph(row):
    return isinstance(row, dict) and row.get("operation") == "component_program"


def validate(row):
    G.validate(row["graph_family"])
    if set(row["dimensions"]) != {"M", "K", "N"}:
        raise RtlIntakeRefusal("source-only graph dimensions need exactly M/K/N")
    if not isinstance(row["operation_owners"], dict) or set(row["operation_owners"]) != set(OPERATIONS):
        raise RtlIntakeRefusal("source-only graph must independently select every generated primitive owner")
    # Parameter validation is shape-independent; no topology is constructed.
    G.parameters({"M": 1, "K": 1, "N": 1, "depth": row["depth"], "fanout": row["fanout"]})


def members(plan):
    """Keep the complete original pair and detect staging name collisions."""
    result = []
    for row in plan["members"]:
        if plan["schema"] == SCHEMA and is_graph(row):
            for variant in G.VARIANTS:
                selected = copy.deepcopy(row)
                selected["name"] += "__" + variant
                selected["graph_variant"] = variant
                result.append(selected)
        else:
            result.append(copy.deepcopy(row))
    if len({row["name"] for row in result}) != len(result):
        raise RtlIntakeRefusal("source-only expanded members repeat an original staging name")
    return tuple(result)


def counts(row, budget):
    """Exact source node/output counts from bounded factory prototypes only."""
    if not is_graph(row):
        if row["operation"] not in {"copy", "matmul"}:
            raise RtlIntakeRefusal("source-only graph budget has no cost for the selected primitive")
        return {"nodes": 1, "outputs": 1}
    values = G.parameters({"M": 1, "K": 1, "N": 1, "depth": row["depth"], "fanout": row["fanout"]})
    if any(value.bit_length() > budget["max_extent_bits"] for value in (values["depth"], values["fanout"])):
        raise RtlIntakeRefusal("source-only graph parameters exceed their metadata bit budget")
    variant = row.get("graph_variant")
    if variant not in G.VARIANTS:
        raise RtlIntakeRefusal("source-only graph needs an original issued representation")
    base = G.program({**values, "depth": 0, "fanout": 2}, variant)
    stage = G.program({**values, "depth": 1, "fanout": 2}, variant)
    extra = G.program({**values, "depth": 1, "fanout": 3}, variant)
    result = {}
    for field in ("nodes", "outputs"):
        result[field] = len(base[field]) + values["depth"] * (
            len(stage[field]) - len(base[field]) + (values["fanout"] - 2) * (len(extra[field]) - len(stage[field]))
        )
        if result[field] > budget["max_" + field]:
            raise RtlIntakeRefusal("source-only graph " + field + " budget exceeded before topology construction")
    return result


def program(row, values, budget):
    expected = counts(row, budget)
    actual = G.program({**values, "depth": row["depth"], "fanout": row["fanout"]}, row["graph_variant"])
    if {key: len(actual[key]) for key in expected} != expected:
        raise RtlIntakeRefusal("source-only graph actual source counts differ from its symbolic metadata")
    return actual

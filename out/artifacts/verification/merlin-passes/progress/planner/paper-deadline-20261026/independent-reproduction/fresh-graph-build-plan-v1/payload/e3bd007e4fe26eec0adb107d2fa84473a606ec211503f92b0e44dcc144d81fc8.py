"""Actual complete entry roster and finite reachability hints, never proof results.

All definitions and declarations are retained. Header/closure facts are emitted
as Lean obligations elsewhere. Python comparisons are preparation diagnostics.
"""

import json

from .lean_graph_loops_segment_projection import _one_block
from .lean_graph_loops_shape_types import kind


def headers(exported):
    result = []
    operations = _one_block(exported["tree"])["operations"]
    for operation, prototype in zip(operations, exported["functions"], strict=True):
        if operation["name"] != "func.func":
            raise ValueError("entry roster requires every top-level operation to be a function")
        properties = dict(operation["properties"])
        result.append(dict(
            name=prototype["name"],
            visibility=properties.get("sym_visibility", "<default-public>"),
            declaration=prototype["declaration"],
            parameters=[kind(row) for row in prototype["parameters"]],
            returned=[kind(row) for row in prototype["returned"]],
        ))
    return result


def actions(code):
    """All scoped action nodes, including nested bodies; retain occurrence order."""
    while code["tag"] != "returned":
        if code["tag"] == "action":
            yield code
        elif code["tag"] == "region":
            yield from actions(code["body"])
        code = code["rest"]


def callgraph(module):
    # Lean callTarget uses the FIRST matching key. Module.Valid separately checks consistency.
    targets = {}
    for key, name in module["calls"]:
        targets.setdefault(key, name)
    return {
        function["name"]: [
            dict(key=action["key"], target=targets[action["key"]])
            for action in actions(function["code"]) if action["key"] in targets
        ]
        for function in module["functions"]
    }


def reachable(name, *graphs):
    result, pending = [], [name]
    while pending:
        current = pending.pop(0)
        if current in result:
            continue
        result.append(current)
        for graph in graphs:
            pending.extend(row["target"] for row in graph.get(current, ()))
    return result


def literal_opcodes(*modules):
    """Keep exact metadata keys; scalar literal values remain opaque parameters."""
    result = []
    for module in modules:
        for function in module["functions"]:
            for action in actions(function["code"]):
                metadata = json.loads(action["key"])
                if metadata["name"] != "arith.constant" or metadata["index_constant"] is not None:
                    continue
                outputs = action["results"]
                if action["operands"] or len(outputs) != 1 or outputs[0]["kind"]["tag"] != "scalar":
                    raise ValueError("scalar literal interpretation requires zero inputs and one scalar result")
                row = [action["key"], outputs[0]["kind"]["type"]]
                if row not in result:
                    result.append(row)
    return result


def roster(source_export, final_export, source, final, kernel):
    before, after = headers(source_export), headers(final_export)
    first_graph, last_graph = callgraph(source), callgraph(final)
    rows, refusals = [], []
    if before != after:
        refusals.append(dict(code="boundary_roster_or_interface_changed", detail="complete ordered headers differ"))
    for header in before + after:
        if header["declaration"]:
            refusals.append(dict(code="declaration_not_supported", entry=header["name"]))
    names = [header["name"] for header in before]
    if len(names) != len(set(names)):
        refusals.append(dict(code="duplicate_entry_name"))
    for header in before:
        name = header["name"]
        closure = reachable(name, first_graph, last_graph)
        rows.append(dict(name=name, route="kernel" if name == kernel else "unchanged", reachable=closure))
        if name != kernel and kernel in closure:
            refusals.append(dict(code="unproved_changed_kernel_caller_context", entry=name, reachable=closure))
    return dict(
        source_headers=before, final_headers=after, rows=rows,
        source_callgraph=first_graph, final_callgraph=last_graph,
        kernel_reachable=reachable(kernel, first_graph, last_graph),
        literals=literal_opcodes(source, final), refusals=refusals,
        coverage="ALL_LISTED_DEFINITIONS_CANDIDATE" if not refusals else "REFUSED_DOMAIN",
        semantic_validation="UNOBSERVED",
    )

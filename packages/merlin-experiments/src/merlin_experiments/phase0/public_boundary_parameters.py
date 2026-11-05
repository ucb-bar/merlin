"""Replay source parameter roles without importing original parameter values.

Generated constants are fresh deterministic buffers. A lifted export parameter
is not a dynamic user input; the subsequent capture must retain that distinction.
"""

from __future__ import annotations


def source_role(node):
    kind = node.get("input_kind")
    if node["op"] == "get_attr":
        kind, target = "ATTRIBUTE", node["target"]
    elif node["op"] == "placeholder" and kind in {"PARAMETER", "BUFFER", "CONSTANT_TENSOR"}:
        target = node.get("input_target")
    else:
        return None
    if not isinstance(target, str) or not target or node["args"] or node["kwargs"] or len(node["results"]) != 1:
        raise ValueError("source parameter lacks an exact original role and owner")
    return {"node": node["id"], "kind": kind, "target": target}


def split(program):
    parameters = [row for row in program["inputs"] if "source_parameter" in row]
    dynamic = [row for row in program["inputs"] if "source_parameter" not in row]
    return dynamic, parameters


def captured_inputs(nodes, program):
    """Match the fresh original export's exact buffer and user-input roster."""
    dynamic, parameters = split(program)
    users, constants = [], {}
    for node in nodes:
        if node["op"] == "get_attr":
            target = node["target"]
        elif node["op"] == "placeholder":
            kind = node.get("input_kind")
            if kind in (None, "USER_INPUT"):
                users.append(node)
                continue
            if kind != "BUFFER":
                raise ValueError("fresh source parameter is not its constructor-owned buffer")
            target = node.get("input_target")
        else:
            continue
        if not isinstance(target, str) or target in constants or node["args"] or node["kwargs"]:
            raise ValueError("fresh source buffer has missing, duplicate or opaque ownership")
        constants[target] = node
    if len(users) != len(dynamic) or set(constants) != {row["name"] for row in parameters}:
        raise ValueError("fresh source changed its exact dynamic-input/frozen-buffer roster")
    expected = [*dynamic, *parameters]
    actual = [*users, *(constants[row["name"]] for row in parameters)]
    return actual, expected

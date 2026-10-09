"""Narrow original typed add premises; independent review still owns arithmetic."""

from __future__ import annotations

from .frontend_operator_effects import _argument_values
from .frontend_use_def import original_use_def_semantics


def defaults_request(trace, observation):
    relation = original_use_def_semantics(trace)
    return {
        "schema": "merlin.original_schema_defaults_request.v1",
        "graph_sha256": relation.graph_sha256,
        "rows": [
            {"target": row["target"], "schema": row["schema"]}
            for row in observation["rows"]
            if row["status"] == "observed"
        ],
    }


def add_forms(trace, observation, defaults, *, numerical_semantics):
    relation = original_use_def_semantics(trace)
    graph = trace["graphs"]["original"]
    request = defaults_request(trace, observation)
    if (
        defaults.get("schema") != "merlin.native_schema_defaults_observation.v1"
        or defaults.get("graph_sha256") != relation.graph_sha256
        or len(defaults.get("rows", [])) != len(request["rows"])
        or [row.get("request") for row in defaults["rows"]] != request["rows"]
    ):
        raise ValueError("typed add requires complete original public schema defaults")
    observed = {row["target"]: row for row in observation["rows"]}
    default_rows = {row["request"]["target"]: row for row in defaults["rows"]}
    values = {value["id"]: value for node in graph["nodes"] for value in node["results"]}
    forms = []
    for node in graph["nodes"]:
        if node["op"] != "call_function" or node["target"] != "aten.add.Tensor":
            continue
        try:
            row, declared = observed[node["target"]], default_rows.get(node["target"])
            if row["status"] != "observed" or graph.get("operator_schemas", {}).get(node["target"]) != row["schema"]:
                raise ValueError("captured/public/registered add schemas are not identical")
            arguments = row["arguments"]
            if (
                [arg["name"] for arg in arguments] != ["self", "other", "alpha"]
                or [arg["type"] for arg in arguments[:2]] != ["Tensor", "Tensor"]
                or arguments[2]["type"] not in {"number", "Scalar"}
                or any(arg["alias"] is not None for arg in arguments)
                or len(row["returns"]) != 1
                or row["returns"][0]["type"] != "Tensor"
                or row["returns"][0]["alias"] is not None
                or declared is None
                or declared["status"] != "observed"
            ):
                raise ValueError("public add schema has no exact supported direct unaliased Tensor signature")
            expected_defaults = declared["defaults"]
            if len(expected_defaults) != 3 or any(
                set(default) != {"ordinal", "name", "has_default", "default"}
                or type(default["ordinal"]) is not int
                or type(default["has_default"]) is not bool
                or default["ordinal"] != index
                or default["name"] != argument["name"]
                or default["has_default"] is not argument["has_default"]
                for index, (default, argument) in enumerate(zip(expected_defaults, arguments, strict=True))
            ):
                raise ValueError("original default roster differs from exact typed schema arguments")
            selected, _ = _argument_values(node, arguments)
            alpha = selected.get(2)
            if 2 not in selected:
                alpha_default = expected_defaults[2]
                if (
                    alpha_default["has_default"] is not True
                    or alpha_default["default"] != {"kind": "int", "value": 1}
                    or type(alpha_default["default"].get("value")) is not int
                ):
                    raise ValueError("original public default alpha is not exact integer one")
                alpha = 1
            if type(alpha) is not int or alpha != 1:
                raise ValueError("original alpha is outside the exact unit-alpha add domain")
            operands = [values[selected[index]["value_id"]] for index in (0, 1)]
            if len(node["results"]) != 1:
                raise ValueError("original add does not have one complete typed result")
            result = node["results"][0]
            dtype = numerical_semantics["operand_dtype"]
            dtype = "int" + dtype[1:] if dtype.startswith("i") and dtype[1:].isdigit() else dtype
            if (
                not dtype.startswith("int")
                or not dtype[3:].isdigit()
                or any(value["kind"] != "tensor" or value.get("dtype") != dtype for value in [*operands, result])
            ):
                raise ValueError(
                    "original input/result dtype or promotion differs from the selected signed integer domain"
                )
            shapes = [value.get("shape") for value in [*operands, result]]
            if any(
                not isinstance(shape, list)
                or len(shape) != 2
                or any(type(extent) is not int or extent < 1 for extent in shape)
                for shape in shapes
            ):
                raise ValueError("original add rank/static extent is outside the rank-two source domain")
            if any(shape != shapes[0] for shape in shapes[1:]):
                raise ValueError("original add broadcasting is outside the exact equal-shape source domain")
            overflow = numerical_semantics.get("overflow")
            if numerical_semantics["model"]["engine"] != "integer_reference" or overflow not in {
                "bounded_exact",
                "modular_wrap",
            }:
                raise ValueError("selected add overflow/reference semantics are unimplemented")
            forms.append(
                {
                    "node": node["id"],
                    "target": node["target"],
                    "status": "supported",
                    "operand_dtypes": [dtype, dtype],
                    "result_dtypes": [dtype],
                    "rank": 2,
                    "alpha": 1,
                    "broadcasting": "none",
                    "overflow": overflow,
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            forms.append({"node": node["id"], "target": node["target"], "status": "unknown", "reason": str(error)})
    return forms

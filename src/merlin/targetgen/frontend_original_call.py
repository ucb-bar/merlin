"""Original typed call bindings from exact public/native schema defaults.

Bindings preserve every original argument and result slot. They grant no
operation correspondence, numerical policy, purity or physical effect meaning.
"""

from __future__ import annotations

import copy
import math

from .frontend_operator_effects import _argument_values, _zero_return_rows
from .frontend_typed_add import defaults_request
from .frontend_use_def import original_use_def_semantics


def default_value(value):
    """Decode only the fixed observer's exact finite scalar/list vocabulary."""
    if not isinstance(value, dict):
        raise ValueError("original public default is not a typed literal")
    kind = value.get("kind")
    if kind == "none" and set(value) == {"kind"}:
        return None
    if kind in {"bool", "int", "str"} and set(value) == {"kind", "value"}:
        if type(value["value"]).__name__ == kind:
            return value["value"]
    if kind == "float" and set(value) == {"kind", "value_hex"}:
        if isinstance(value["value_hex"], str):
            try:
                decoded = float.fromhex(value["value_hex"])
            except OverflowError as error:
                raise ValueError("original public default overflows the finite literal domain") from error
            if math.isfinite(decoded) and decoded.hex() == value["value_hex"]:
                return decoded
    if kind == "list" and set(value) == {"kind", "items"} and isinstance(value["items"], list):
        decoded = [default_value(item) for item in value["items"]]
        if all(item is None or type(item) in {bool, int, str, float} for item in decoded):
            return decoded
    raise ValueError("original public default is outside the exact finite scalar/list domain")


def _literal(value):
    if value is None:
        return {"kind": "none"}
    if type(value) in {bool, int, str}:
        return {"kind": type(value).__name__, "value": value}
    if type(value) is float and math.isfinite(value):
        return {"kind": "float", "value_hex": value.hex()}
    if type(value) is list:
        return {"kind": "list", "items": [_literal(item) for item in value]}
    if (
        isinstance(value, dict)
        and set(value) == {"kind", "value"}
        and value["kind"]
        in {
            "dtype",
            "device",
            "layout",
            "memory_format",
        }
        and isinstance(value["value"], str)
    ):
        return copy.deepcopy(value)
    raise ValueError("original argument is outside the exact serialized literal vocabulary")


def _value(value):
    # Exact roster identity and types are retained; original shape extents do
    # not become factory geometry or authoring policy.
    return {
        key: copy.deepcopy(value.get(key))
        for key in (
            "id",
            "kind",
            "dtype",
            "storage_dtype",
            "compute_dtype",
            "device",
            "layout",
        )
    } | {"rank": len(value["shape"]) if isinstance(value.get("shape"), list) else None}


def call_contracts(trace, observation, defaults, *, zero_returns=None):
    """Bind every original call to its complete original typed/default roster."""
    relation = original_use_def_semantics(trace)
    request = defaults_request(trace, observation, version=2)
    if (
        defaults.get("schema") != "merlin.native_schema_defaults_observation.v2"
        or defaults.get("graph_sha256") != relation.graph_sha256
        or not isinstance(defaults.get("rows"), list)
        or [row.get("request") for row in defaults["rows"]] != request["rows"]
    ):
        raise ValueError("original calls require the complete exact public scalar/list default roster")
    graph = trace["graphs"]["original"]
    observed = {row["target"]: row for row in observation["rows"]}
    if len(observed) != len(observation["rows"]):
        raise ValueError("original call schema roster repeats an operator")
    declared = {row["request"]["target"]: row for row in defaults["rows"]}
    values = {value["id"]: value for node in graph["nodes"] for value in node["results"]}
    zero_rows = _zero_return_rows(trace, observation, zero_returns)
    contracts = []
    for node in graph["nodes"]:
        if node["op"] != "call_function":
            continue
        result = {
            "node": node["id"],
            "target": node["target"],
            "result_roster": [_value(value) for value in node["results"]],
            "result_metadata": copy.deepcopy(node.get("result_metadata")),
            "result_arity": len(node["results"]),
        }
        try:
            schema = observed[node["target"]]
            native = declared[node["target"]]
            if (
                schema["status"] != "observed"
                or native["status"] != "observed"
                or graph.get("operator_schemas", {}).get(node["target"]) != schema["schema"]
            ):
                raise ValueError("original call lost exact captured/public/registered schema equality")
            arguments = schema["arguments"]
            if "result_arity" in node and (
                type(node["result_arity"]) is not int or node["result_arity"] != len(node["results"])
            ):
                raise ValueError("original declared result arity differs from its complete typed roster")
            if schema["returns"]:
                if len(schema["returns"]) != len(node["results"]):
                    raise ValueError("original result count differs from its exact public schema")
                if any(
                    returned["type"] != "Tensor" or value["kind"] != "tensor"
                    for returned, value in zip(schema["returns"], node["results"], strict=True)
                ):
                    raise ValueError("original schema result kind has no exact supported Tensor correspondence")
            elif zero_rows.get(node["id"], {}).get("status") != "observed":
                raise ValueError("zero-return call lacks its native bridge and original observed None metadata")
            rows = native["defaults"]
            if len(rows) != len(arguments) or any(
                set(row) != {"ordinal", "name", "has_default", "default"}
                or type(row["ordinal"]) is not int
                or row["ordinal"] != index
                or type(row["has_default"]) is not bool
                or row["name"] != argument["name"]
                or row["has_default"] is not argument["has_default"]
                or (not row["has_default"] and row["default"] is not None)
                for index, (row, argument) in enumerate(zip(rows, arguments, strict=True))
            ):
                raise ValueError("public defaults differ from the complete original schema argument roster")
            selected, paths = _argument_values(node, arguments)
            bindings = []
            for index, argument in enumerate(arguments):
                explicit = index in selected
                if not explicit and rows[index]["has_default"] is not True:
                    raise ValueError("original required argument has no explicit binding or public default")
                value = selected[index] if explicit else default_value(rows[index]["default"])
                if isinstance(value, dict) and set(value) == {"node_id", "value_id"}:
                    typed = values[value["value_id"]]
                    encoded = {"kind": "ssa", "node_id": value["node_id"], "value": _value(typed)}
                else:
                    encoded = _literal(value)
                bindings.append(
                    {
                        "ordinal": index,
                        **copy.deepcopy(argument),
                        "binding": "explicit" if explicit else "default",
                        "path": paths[index] if explicit else None,
                        "value": encoded,
                    }
                )
            result.update(
                status="bound",
                schema=schema["schema"],
                arguments=bindings,
                schema_returns=copy.deepcopy(schema["returns"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            result.update(status="unknown", reason=str(error))
        contracts.append(result)
    return contracts

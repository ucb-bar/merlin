"""Conservative original argument/result effects from observed typed schemas.

Source SSA and canonical schema observations must agree. This reader produces
shape-free mandatory effect classes and private exact witnesses, not physical
aliasing, allocation ownership or a claim that schemas enumerate all effects.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .frontend_use_def import original_use_def_semantics


@dataclass(frozen=True)
class OriginalOperatorEffects:
    graph_sha256: str
    effect_classes: tuple[str, ...]
    witnesses_json: str
    unknowns_json: str
    tensor_bindings_json: str = "[]"
    zero_returns_json: str = "[]"

    def witnesses(self):
        return json.loads(self.witnesses_json)

    def unknowns(self):
        return json.loads(self.unknowns_json)

    def tensor_bindings(self):
        return json.loads(self.tensor_bindings_json)

    def zero_returns(self):
        return json.loads(self.zero_returns_json)

    def public_semantics(self):
        return {"graph_sha256": self.graph_sha256, "effect_classes": list(self.effect_classes)}


def _alias(row):
    alias = row["alias"]
    if alias is None:
        return set(), False
    if (
        not isinstance(alias, dict)
        or set(alias) != {"before", "after", "write"}
        or type(alias["write"]) is not bool
        or any(
            not isinstance(alias[key], list) or any(not isinstance(item, str) or not item for item in alias[key])
            for key in ("before", "after")
        )
    ):
        raise ValueError("operator effects need actual typed alias observations")
    before, after = set(alias["before"]), set(alias["after"])
    if "*" in before | after or before != after:
        raise ValueError("wildcard or changing alias sets have no supported exact effect relation")
    return before, alias["write"]


def _argument_values(node, arguments):
    positional = [index for index, row in enumerate(arguments) if not row["kwarg_only"]]
    if len(node["args"]) > len(positional):
        raise ValueError("original source positional arguments exceed the observed schema")
    names = {row["name"]: index for index, row in enumerate(arguments)}
    if len(names) != len(arguments) or set(node["kwargs"]) - set(names):
        raise ValueError("original source keyword arguments differ from the observed schema")
    selected = {index: value for index, value in zip(positional, node["args"])}
    paths = {index: "args/" + str(offset) for offset, index in enumerate(positional[: len(node["args"])])}
    for name, value in node["kwargs"].items():
        index = names[name]
        if index in selected:
            raise ValueError("original source argument is bound twice")
        selected[index], paths[index] = value, "kwargs/" + name
    for index, row in enumerate(arguments):
        if index not in selected and not row["has_default"]:
            raise ValueError("original source required argument has no concrete binding")
    return selected, paths


def scalar_literal(value):
    """Keep source kind, exact integer value and floating signed zero separate."""
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": str(value)}
    if type(value) is float and math.isfinite(value):
        return {"type": "float", "value_hex": value.hex()}
    raise ValueError("only exact finite original Python numeric literals are supported")


def original_tensor_argument_requests(trace, observation):
    """Select original scalar slots without interpreting operation names as policy."""
    relation = original_use_def_semantics(trace)
    observed = {row["target"]: row for row in observation["rows"]}
    graph = trace["graphs"]["original"]
    rows = []
    for node in graph["nodes"]:
        found = observed.get(node["target"])
        if (
            node["op"] != "call_function"
            or not found
            or found["status"] != "observed"
            or graph.get("operator_schemas", {}).get(node["target"]) != found["schema"]
        ):
            continue
        try:
            values, paths = _argument_values(node, found["arguments"])
        except (KeyError, TypeError, ValueError):
            continue
        for index, argument in enumerate(found["arguments"]):
            if argument["type"] != "Tensor" or argument["alias"] is not None or index not in values:
                continue
            try:
                literal = scalar_literal(values[index])
            except ValueError:
                continue
            rows.append(
                {
                    "node": node["id"],
                    "target": node["target"],
                    "schema": found["schema"],
                    "argument_index": index,
                    "argument_path": paths[index],
                    "literal": literal,
                }
            )
    return {"schema": "merlin.original_tensor_argument_request.v1", "graph_sha256": relation.graph_sha256, "rows": rows}


def original_zero_return_requests(trace, observation):
    """Select only exact original calls whose observed schema has no results.

    Missing metadata stays in the request as missing; no result or effect meaning
    comes from an operator name. Historical None slots remain unchanged.
    """
    relation = original_use_def_semantics(trace)
    graph = trace["graphs"]["original"]
    observed = {row["target"]: row for row in observation["rows"]}
    rows = []
    for node in graph["nodes"]:
        found = observed.get(node["target"])
        if (
            node["op"] == "call_function"
            and found
            and found["status"] == "observed"
            and graph.get("operator_schemas", {}).get(node["target"]) == found["schema"]
            and found["returns"] == []
        ):
            rows.append(
                {
                    "node": node["id"],
                    "target": node["target"],
                    "schema": found["schema"],
                    "results": node["results"],
                    "result_metadata": node.get("result_metadata"),
                }
            )
    return {"schema": "merlin.original_zero_return_request.v1", "graph_sha256": relation.graph_sha256, "rows": rows}


def _zero_return_rows(trace, observation, zero_returns):
    if zero_returns is None:
        return {}
    from .torch_zero_return_observer import _metadata_is_none

    request = original_zero_return_requests(trace, observation)
    if (
        set(zero_returns) != {"schema", "graph_sha256", "rows", "runtime", "scope"}
        or zero_returns["schema"] != "merlin.native_zero_return_observation.v1"
        or zero_returns["graph_sha256"] != request["graph_sha256"]
        or not isinstance(zero_returns["rows"], list)
        or len(zero_returns["rows"]) != len(request["rows"])
    ):
        raise ValueError("zero-return observations need the complete exact original request roster")
    found = {}
    for expected, row in zip(request["rows"], zero_returns["rows"], strict=True):
        if row.get("request") != expected or row.get("status") not in {"observed", "unknown"}:
            raise ValueError("zero-return observation changed an original call or complete metadata roster")
        if row["status"] == "observed":
            native = row.get("native")
            if (
                set(row) != {"request", "status", "native"}
                or not isinstance(native, dict)
                or set(native) != {"schema", "return_count", "empty_stack_is_none"}
                or native["schema"] != expected["schema"]
                or type(native["return_count"]) is not int
                or native["return_count"] != 0
                or native["empty_stack_is_none"] is not True
                or not _metadata_is_none(expected)
            ):
                raise ValueError("zero-return binding is not the actual supported native None relation")
        elif set(row) != {"request", "status", "reason"} or not isinstance(row["reason"], str):
            raise ValueError("unobserved zero return must retain its actual refusal")
        if expected["node"] in found:
            raise ValueError("zero-return observation repeats an original call")
        found[expected["node"]] = row
    return found


def _tensor_argument_rows(trace, observation, tensor_arguments):
    if tensor_arguments is None:
        return {}
    request = original_tensor_argument_requests(trace, observation)
    if (
        set(tensor_arguments) != {"schema", "graph_sha256", "rows", "runtime", "scope"}
        or tensor_arguments["schema"] != "merlin.native_tensor_argument_observation.v1"
        or tensor_arguments["graph_sha256"] != request["graph_sha256"]
        or not isinstance(tensor_arguments["rows"], list)
        or len(tensor_arguments["rows"]) != len(request["rows"])
    ):
        raise ValueError("Tensor argument observations need the complete exact original request roster")
    found = {}
    for expected, row in zip(request["rows"], tensor_arguments["rows"], strict=True):
        if row.get("request") != expected or row.get("status") not in {"observed", "unknown"}:
            raise ValueError("Tensor argument observation changed an original operator, slot or literal")
        if row["status"] == "observed":
            native = row.get("native")
            if (
                set(row) != {"request", "status", "native"}
                or not isinstance(native, dict)
                or set(native)
                != {
                    "schema",
                    "argument_name",
                    "source_allows_number",
                    "wrapped_number",
                    "shape",
                    "dtype",
                    "element_bytes",
                    "literal",
                    "disjoint_from_prior_live_boxes",
                }
                or native["schema"] != expected["schema"]
                or native["source_allows_number"] is not True
                or native["wrapped_number"] is not True
                or native["shape"] != []
                or native["literal"] != expected["literal"]
                or not isinstance(native["dtype"], str)
                or not native["dtype"]
                or type(native["element_bytes"]) is not int
                or native["element_bytes"] <= 0
                or native["disjoint_from_prior_live_boxes"] is not True
            ):
                raise ValueError("Tensor literal binding is not the actual supported native wrapped scalar relation")
        elif set(row) != {"request", "status", "reason"} or not isinstance(row["reason"], str):
            raise ValueError("unobserved Tensor argument must retain its actual refusal")
        identity = expected["node"], expected["argument_path"]
        if identity in found:
            raise ValueError("Tensor argument observation repeats an original slot")
        found[identity] = row
    return found


def _bindings(node, arguments, values, scalar_rows):
    selected, paths = _argument_values(node, arguments)
    for index, row in enumerate(arguments):
        value = selected.get(index)
        if row["type"] == "Tensor":
            if not isinstance(value, dict) or set(value) != {"node_id", "value_id"} or value["value_id"] not in values:
                scalar = scalar_rows.get((node["id"], paths.get(index)))
                if scalar is None or row["alias"] is not None:
                    raise ValueError("schema tensor argument has no exact original tensor value")
                if scalar["status"] != "observed":
                    raise ValueError("original Tensor scalar conversion is unobserved: " + scalar["reason"])
                if scalar["native"]["argument_name"] != row["name"]:
                    raise ValueError("native Tensor argument name differs from the original observed schema")
            elif values[value["value_id"]]["kind"] != "tensor":
                raise ValueError("schema tensor argument disagrees with original value kind")
        elif row["alias"] is not None:
            raise ValueError("non-scalar tensor aliases require an unsupported container binding")
    return selected, paths


def original_operator_effects(trace, observation, *, tensor_arguments=None, zero_returns=None):
    """Replay all original uses before joining each exact observed schema row.

    Narrow support covers direct Tensor arguments and direct Tensor returns.
    Unknown lists, conditional alias sets, unresolved schemas and unmatched
    result rosters remain explicit. A schema with no alias annotations is not
    asserted pure: random, exception, global state and other effects are outside
    this source relation.
    """
    relation = original_use_def_semantics(trace)
    graph = trace["graphs"]["original"]
    schemas = graph.get("operator_schemas", {})
    if observation.get("schema") != "merlin.native_operator_schema_observation.v1":
        raise ValueError("original effects require the fixed native observation schema")
    rows = observation.get("rows")
    if (
        not isinstance(rows, list)
        or len({row["target"] for row in rows}) != len(rows)
        or {row["target"] for row in rows} != set(relation.operations)
    ):
        raise ValueError("schema observation must cover the complete exact original call roster")
    observed = {row["target"]: row for row in rows}
    scalar_rows = _tensor_argument_rows(trace, observation, tensor_arguments)
    zero_rows = _zero_return_rows(trace, observation, zero_returns)
    values = {result["id"]: result for node in graph["nodes"] for result in node["results"]}
    witnesses, unknowns = [], []
    for node in graph["nodes"]:
        if node["op"] not in {"call_function", "call_method", "call_module"}:
            continue
        target, found = node["target"], observed[node["target"]]
        if found["status"] != "observed":
            unknowns.append({"target": target, "reason": found.get("reason", "schema is unobserved")})
            continue
        if node["op"] != "call_function" or schemas.get(target) != found["schema"]:
            unknowns.append({"target": target, "reason": "original captured schema or call kind differs"})
            continue
        try:
            arguments, returns = found["arguments"], found["returns"]
            bindings, paths = _bindings(node, arguments, values, scalar_rows)
            inputs = [_alias(row) for row in arguments]
            outputs = [_alias(row) for row in returns]
            zero = zero_rows.get(node["id"])
            if returns == [] and zero is not None:
                if zero["status"] != "observed":
                    raise ValueError("original zero-result correspondence is unobserved: " + zero["reason"])
            elif len(node["results"]) != len(returns) or any(
                row["type"] != "Tensor" or result["kind"] != "tensor" for row, result in zip(returns, node["results"])
            ):
                raise ValueError("original result roster is not the exact supported direct Tensor returns")
            local = []
            for index, (aliases, write) in enumerate(inputs):
                if aliases and index not in bindings:
                    raise ValueError("aliased input has no explicit original value")
                if write:
                    local.append(
                        {
                            "kind": "may_write_argument",
                            "node": node["id"],
                            "target": target,
                            "argument_path": paths[index],
                            "input_value": bindings[index]["value_id"],
                        }
                    )
            for result_index, (aliases, _) in enumerate(outputs):
                if aliases and not any(aliases & input_aliases for input_aliases, _ in inputs):
                    raise ValueError("aliased return has no supported original input alias relation")
                for input_index, (input_aliases, _) in enumerate(inputs):
                    if aliases & input_aliases:
                        local.append(
                            {
                                "kind": "may_alias_result",
                                "node": node["id"],
                                "target": target,
                                "argument_path": paths[input_index],
                                "input_value": bindings[input_index]["value_id"],
                                "result_value": node["results"][result_index]["id"],
                            }
                        )
            witnesses += local
        except (KeyError, TypeError, ValueError) as exc:
            unknowns.append({"target": target, "reason": str(exc)})
    return OriginalOperatorEffects(
        relation.graph_sha256,
        tuple(sorted({row["kind"] for row in witnesses})),
        json.dumps(witnesses, sort_keys=True, separators=(",", ":")),
        json.dumps(unknowns, sort_keys=True, separators=(",", ":")),
        json.dumps(list(scalar_rows.values()), sort_keys=True, separators=(",", ":")),
        json.dumps(list(zero_rows.values()), sort_keys=True, separators=(",", ":")),
    )

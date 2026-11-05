"""Bounded original casts and metadata assertions without semantic admission.

Fresh geometry preserves rank, storage and exact arguments. Assertions retain
their observed Python None slot separately from the empty dispatcher result.
Example dimensions/strides never select geometry or disappear into a no-op.
"""

from __future__ import annotations

import copy
import json

from merlin.common.jsonio import canonical_json

from .frontend_original_call import call_contracts, default_value
from .original_operator_sources import SOURCE_SCHEMA, OriginalOperatorSource, _tensor

CAST_SCHEMA = "merlin.original_dtype_cast_form.v1"
ASSERTION_SCHEMA = "merlin.original_metadata_assertion_form.v1"
CAST = "aten.to.dtype"
ASSERTION = "aten._assert_tensor_metadata.default"
_DTYPES = {
    "int8": 8,
    "int16": 16,
    "int32": 32,
    "int64": 64,
    "float16": 16,
    "bfloat16": 16,
    "float32": 32,
    "float64": 64,
}
_ALIAS = {"before": ["a"], "after": ["a"], "write": False}


def _literal(argument, kind):
    value = argument["value"]
    if value == {"kind": "none"}:
        return None
    if type(value) is not dict or set(value) != {"kind", "value"} or value["kind"] != kind:
        raise ValueError("original metadata argument needs its exact serialized literal kind")
    if type(value["value"]) is not str:
        raise ValueError("original metadata argument needs an exact literal spelling")
    return value["value"]


def _binding(call):
    target, arguments = call["target"], call["arguments"]
    cast = target == CAST
    names = (
        ["self", "dtype", "non_blocking", "copy", "memory_format"]
        if cast
        else ["a", "size", "stride", "dtype", "device", "layout"]
    )
    types = (
        ["Tensor", "int", "bool", "bool", "Optional[int]"]
        if cast
        else [
            "Tensor",
            "Optional[List[int]]",
            "Optional[List[int]]",
            "Optional[int]",
            "Optional[Device]",
            "Optional[int]",
        ]
    )
    if (
        target not in {CAST, ASSERTION}
        or [row["name"] for row in arguments] != names
        or [row["type"] for row in arguments] != types
        or any(type(row["ordinal"]) is not int or row["ordinal"] != i for i, row in enumerate(arguments))
        or [row["kwarg_only"] for row in arguments] != ([False] * 5 if cast else [False] * 4 + [True] * 2)
        or any(type(row["kwarg_only"]) is not bool or type(row["has_default"]) is not bool for row in arguments)
        or [row["has_default"] for row in arguments]
        != ([False, False, True, True, True] if cast else [False] + [True] * 5)
        or canonical_json(arguments[0]["alias"]) != canonical_json(_ALIAS if cast else None)
        or any(row["alias"] is not None for row in arguments[1:])
        or type(call["result_arity"]) is not int
        or call["result_arity"] != 1
        or len(call["result_roster"]) != 1
    ):
        raise ValueError("metadata source needs its complete original argument/alias/result schema")
    tensor = arguments[0]["value"]
    rank = tensor["value"]["rank"]
    if type(rank) is not int or rank < 0:
        raise ValueError("metadata source needs the original static CPU Tensor rank")
    dtype = _tensor(arguments[0], rank=rank, dtypes=_DTYPES)
    result = call["result_roster"][0]
    if cast:
        destination = _literal(arguments[1], "dtype")
        if destination not in {"torch." + name for name in _DTYPES}:
            raise ValueError("cast destination needs the exact original supported torch dtype")
        output_dtype = destination.removeprefix("torch.")
        flags = [default_value(row["value"]) for row in arguments[2:4]]
        if any(type(flag) is not bool for flag in flags) or default_value(arguments[4]["value"]) is not None:
            raise ValueError("cast needs exact boolean flags and original memory_format=None")
        if canonical_json(call["schema_returns"]) != canonical_json(
            [{"type": "Tensor", "alias": _ALIAS, "name": "", "kwarg_only": False, "has_default": False}]
        ) and canonical_json(call["schema_returns"]) != canonical_json([{"type": "Tensor", "alias": _ALIAS}]):
            raise ValueError("cast needs its original Tensor(a) return alias")
        if (
            result["kind"] != "tensor"
            or type(result["rank"]) is not int
            or result["rank"] != rank
            or result["dtype"] != output_dtype
            or result["storage_dtype"] != output_dtype
            or result["device"] != "cpu"
            or result["layout"] != "torch.strided"
        ):
            raise ValueError("cast result must preserve original rank and exact requested storage")
        parameters = {"dtype": destination, "non_blocking": flags[0], "copy": flags[1], "memory_format": None}
        return dtype, rank, [output_dtype], parameters
    if default_value(arguments[1]["value"]) is not None or default_value(arguments[2]["value"]) is not None:
        raise ValueError("assertion size/stride conditions cannot borrow original example geometry")
    parameters = {"size": None, "stride": None, **{row["name"]: _literal(row, row["name"]) for row in arguments[3:]}}
    if (
        parameters["dtype"] not in {None, "torch." + dtype}
        or parameters["device"] not in {None, "cpu"}
        or parameters["layout"] not in {None, "torch.strided"}
    ):
        raise ValueError("assertion metadata condition differs from original CPU strided input storage")
    expected = {
        "schema": "m2m.frontend_result_metadata.v1",
        "status": "observed",
        "container": "single",
        "values": [{"result_id": result["id"], "kind": "none"}],
    }
    if (
        call["schema_returns"] != []
        or result["kind"] != "unknown"
        or any(value is not None for key, value in result.items() if key not in {"id", "kind"})
        or canonical_json(call["result_metadata"]) != canonical_json(expected)
    ):
        raise ValueError("assertion needs empty dispatcher results and the exact original observed Python None slot")
    return dtype, rank, [], parameters


def metadata_forms(trace, observation, defaults, *, numerical_semantics=None, zero_returns=None):
    """Retain each original cast/assertion, including every unsupported condition."""
    values = {value["id"]: value for node in trace["graphs"]["original"]["nodes"] for value in node["results"]}
    forms = []
    for call in call_contracts(trace, observation, defaults, zero_returns=zero_returns):
        if call["target"] not in {CAST, ASSERTION}:
            continue
        form = {
            "form_schema": CAST_SCHEMA if call["target"] == CAST else ASSERTION_SCHEMA,
            **call,
            "source_numerical_semantics": copy.deepcopy(numerical_semantics),
        }
        try:
            if call["status"] != "bound":
                raise ValueError(call["reason"])
            dtype, rank, outputs, parameters = _binding(call)
            if call["target"] == CAST:
                operand = call["arguments"][0]["value"]["value"]["id"]
                result = call["result_roster"][0]["id"]
                shapes = [values[identity].get("shape") for identity in (operand, result)]
                if (
                    any(
                        type(shape) is not list
                        or len(shape) != rank
                        or any(type(extent) is not int or extent < 1 for extent in shape)
                        for shape in shapes
                    )
                    or shapes[0] != shapes[1]
                ):
                    raise ValueError("cast needs complete original equal input/result shape relation")
                # Original extents establish equality only, never generated geometry.
                form["shape_relation"] = {"kind": "equal", "operand": operand, "result": result, "rank": rank}
            form.update(
                status="supported", operand_dtypes=[dtype], result_dtypes=outputs, rank=rank, parameters=parameters
            )
        except (KeyError, TypeError, ValueError) as error:
            form.update(status="unknown", reason=str(error))
        forms.append(form)
    return forms


def metadata_source(form, *, extent, max_tensor_elements):
    """Construct the actual original call after full logical input/output bounds."""
    if (
        type(form) is not dict
        or form.get("form_schema") not in {CAST_SCHEMA, ASSERTION_SCHEMA}
        or form.get("status") != "supported"
        or type(extent) is not int
        or extent < 1
        or type(max_tensor_elements) is not int
        or max_tensor_elements < 1
    ):
        raise ValueError("metadata source needs a supported form and explicit positive geometry/budget")
    dtype, rank, outputs, parameters = _binding(form)
    if form["target"] == CAST:
        relation = {
            "kind": "equal",
            "operand": form["arguments"][0]["value"]["value"]["id"],
            "result": form["result_roster"][0]["id"],
            "rank": rank,
        }
        if canonical_json(form.get("shape_relation")) != canonical_json(relation):
            raise ValueError("cast source changed its original equal-shape relation")
    expected = {
        "operand_dtypes": [dtype],
        "result_dtypes": outputs,
        "rank": rank,
        "parameters": parameters,
        "form_schema": CAST_SCHEMA if form["target"] == CAST else ASSERTION_SCHEMA,
    }
    if canonical_json({key: form[key] for key in expected}) != canonical_json(expected):
        raise ValueError("metadata source changed its original ordered storage/condition/result binding")
    arrays = 1 + len(outputs)
    if rank > max_tensor_elements:
        raise ValueError("metadata rank exceeds its complete logical budget before allocation")
    count = 1
    for axis in range(rank):
        count *= extent + axis
        if arrays * count > max_tensor_elements:
            raise ValueError("metadata source exceeds complete tensor-element budget before allocation")
    if arrays * count > max_tensor_elements:
        raise ValueError("metadata source exceeds complete tensor-element budget before allocation")
    shape = [extent + axis for axis in range(rank)]
    output = (
        [{"name": "Y", "kind": "tensor", "dtype": outputs[0], "shape": shape}]
        if outputs
        else [{"name": "None", "kind": "none", "original_result_id": form["result_roster"][0]["id"]}]
    )
    metadata = {
        "schema": SOURCE_SCHEMA,
        "target": form["target"],
        "inputs": [{"name": "X", "dtype": dtype, "shape": shape}],
        "outputs": output,
        "parameters": copy.deepcopy(parameters),
        "schema_alias": copy.deepcopy(form["arguments"][0]["alias"]),
        "dispatcher_result_count": len(form["schema_returns"]),
        "original_result_metadata": copy.deepcopy(form["result_metadata"]),
        "source_numerical_semantics": copy.deepcopy(form["source_numerical_semantics"]),
        "tensor_elements": arrays * count,
        "logical_payload_bytes": count * sum(_DTYPES[t] // 8 for t in [dtype, *outputs]),
        "scalar_products": 0,
        "scope": "original source construction only; numeric/alias/effect/target admission unproved",
    }
    rendered = {
        key: (
            value
            if key in {"dtype", "layout"} and value is not None
            else "torch.device(" + repr(value) + ")"
            if key == "device" and value is not None
            else repr(value)
        )
        for key, value in parameters.items()
    }
    keywords = ", ".join(key + "=" + value for key, value in rendered.items())
    loader = (
        "import torch\n\nclass Model(torch.nn.Module):\n    def forward(self, X):\n"
        f"        return torch.ops.{form['target']}(X, {keywords})\n\ndef get_model_and_inputs():\n"
        f"    return Model(), (torch.zeros({shape!r}, dtype=torch.{dtype}),)\n"
    )
    return OriginalOperatorSource(loader, json.dumps(metadata, sort_keys=True, allow_nan=False))

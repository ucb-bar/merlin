"""Exact original triu calls and independently bounded batched sources.

The public operator preserves shape/storage and masks only its last two axes.
Original sizes prove the input/result shape relation; fresh geometry uses only
an explicit extent. Construction grants no numerical, effect or target admission.
"""

from __future__ import annotations

import copy
import json

from merlin.common.jsonio import canonical_json

from .frontend_original_call import call_contracts
from .original_operator_sources import SOURCE_SCHEMA, OriginalOperatorSource, _argument, _tensor

FORM_SCHEMA = "merlin.original_triu_form.v1"
TARGET = "aten.triu.default"
_DTYPES = {"float32": 32, "int8": 8, "int16": 16, "int32": 32, "int64": 64}


def _binding(call):
    arguments, returns = call["arguments"], call["schema_returns"]
    if (
        call["target"] != TARGET
        or [row["name"] for row in arguments] != ["self", "diagonal"]
        or [row["type"] for row in arguments] != ["Tensor", "int"]
        or any(row["alias"] is not None for row in arguments)
        or len(returns) != 1
        or returns[0]["type"] != "Tensor"
        or returns[0]["alias"] is not None
        or type(call["result_arity"]) is not int
        or call["result_arity"] != 1
        or len(call["result_roster"]) != 1
    ):
        raise ValueError("triu needs its exact original direct Tensor/int argument and Tensor result schema")
    original = _argument(arguments[0])["value"]
    rank = original["rank"]
    if type(rank) is not int or rank < 2:
        raise ValueError("triu needs an exact original Tensor rank of at least two")
    dtype = _tensor(arguments[0], rank=rank, dtypes=_DTYPES)
    result = call["result_roster"][0]
    if (
        result["kind"] != "tensor"
        or type(result["rank"]) is not int
        or result["rank"] != rank
        or result["dtype"] != dtype
        or result["storage_dtype"] != dtype
        or result["layout"] != "torch.strided"
        or result["device"] != "cpu"
    ):
        raise ValueError("triu must preserve the original CPU strided rank and input/result storage")
    diagonal = _argument(arguments[1])
    if type(diagonal) is not int or not -(1 << 63) <= diagonal < 1 << 63:
        raise ValueError("triu diagonal must retain its exact original signed64 integer")
    return dtype, rank, {"diagonal": diagonal}


def triu_forms(trace, observation, defaults, *, numerical_semantics=None, zero_returns=None):
    """Bind each original occurrence and its positive same-shape relation."""
    values = {value["id"]: value for node in trace["graphs"]["original"]["nodes"] for value in node["results"]}
    forms = []
    for call in call_contracts(trace, observation, defaults, zero_returns=zero_returns):
        if call["target"] != TARGET:
            continue
        form = {"form_schema": FORM_SCHEMA, **call, "source_numerical_semantics": copy.deepcopy(numerical_semantics)}
        try:
            if call["status"] != "bound":
                raise ValueError(call["reason"])
            dtype, rank, parameters = _binding(call)
            identities = [_argument(call["arguments"][0])["value"]["id"], call["result_roster"][0]["id"]]
            shapes = [values[identity].get("shape") for identity in identities]
            if any(
                type(shape) is not list
                or len(shape) != rank
                or any(type(value) is not int or value < 1 for value in shape)
                for shape in shapes
            ):
                raise ValueError("triu needs complete positive original input/result shape relations")
            if shapes[0] != shapes[1]:
                raise ValueError("triu original result must have exactly the input shape")
            form.update(
                status="supported",
                rank=rank,
                operand_dtypes=[dtype],
                result_dtypes=[dtype],
                parameters=parameters,
                shape_relation="same_shape",
                matrix_axes=[rank - 2, rank - 1],
            )
        except (KeyError, TypeError, ValueError) as error:
            form.update(status="unknown", reason=str(error))
        forms.append(form)
    return forms


def triu_source(form, *, extent, max_tensor_elements):
    """Bound rank metadata and complete logical counts before shape allocation."""
    if (
        type(form) is not dict
        or form.get("form_schema") != FORM_SCHEMA
        or form.get("status") != "supported"
        or type(extent) is not int
        or extent < 1
        or type(max_tensor_elements) is not int
        or max_tensor_elements < 1
    ):
        raise ValueError("triu needs its supported original form and explicit positive geometry/budget")
    dtype, rank, parameters = _binding(form)
    if 2 * rank > max_tensor_elements:
        raise ValueError("triu rank metadata exceeds its budget before geometry allocation")
    if (
        type(form["rank"]) is not int
        or form["rank"] != rank
        or canonical_json(form["operand_dtypes"]) != canonical_json([dtype])
        or canonical_json(form["result_dtypes"]) != canonical_json([dtype])
        or canonical_json(form["parameters"]) != canonical_json(parameters)
        or form["shape_relation"] != "same_shape"
        or canonical_json(form["matrix_axes"]) != canonical_json([rank - 2, rank - 1])
    ):
        raise ValueError("triu changed its original type, diagonal, rank, shape relation or matrix axes")
    count = 1
    for axis in range(rank):
        count *= extent + axis
        if 2 * count > max_tensor_elements:
            raise ValueError("triu exceeds complete tensor-element budget before geometry allocation")
    shape = [extent + axis for axis in range(rank)]
    elements = 2 * count
    metadata = {
        "schema": SOURCE_SCHEMA,
        "target": TARGET,
        "inputs": [{"name": "X", "dtype": dtype, "shape": shape}],
        "outputs": [{"name": "Y", "kind": "tensor", "dtype": dtype, "shape": shape}],
        "parameters": parameters,
        "shape_relation": "same_shape",
        "matrix_axes": [rank - 2, rank - 1],
        "source_numerical_semantics": copy.deepcopy(form["source_numerical_semantics"]),
        "tensor_elements": elements,
        "logical_payload_bytes": elements * (_DTYPES[dtype] // 8),
        "scalar_products": 0,
        "scope": "typed original triu source only; numerical/effect/packing/target admission unproved",
    }
    loader = (
        "import torch\n\n"
        "class Model(torch.nn.Module):\n"
        "    def forward(self, X):\n"
        f"        return torch.ops.aten.triu.default(X, diagonal={parameters['diagonal']})\n\n"
        "def get_model_and_inputs():\n"
        f"    return Model(), (torch.zeros({shape!r}, dtype=torch.{dtype}),)\n"
    )
    return OriginalOperatorSource(loader, json.dumps(metadata, sort_keys=True, allow_nan=False))

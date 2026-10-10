"""Exact original rank-two transpose calls and independently bounded sources.

The public Tensor(a) schema and original axis arguments are retained. Fresh
geometry comes only from an explicit extent, never an example's dimensions.
Construction establishes no numerical, alias, physical or target admission.
"""

from __future__ import annotations

import copy
import json

from merlin.common.jsonio import canonical_json

from .frontend_original_call import call_contracts
from .original_operator_sources import SOURCE_SCHEMA, OriginalOperatorSource, _argument, _tensor

FORM_SCHEMA = "merlin.original_transpose_form.v1"
TARGET = "aten.transpose.int"
_DTYPES = {"float32": 32, "int8": 8, "int16": 16, "int32": 32, "int64": 64}
_ALIAS = {"before": ["a"], "after": ["a"], "write": False}


def _binding(call):
    arguments, returns = call["arguments"], call["schema_returns"]
    if (
        call["target"] != TARGET
        or [row["name"] for row in arguments] != ["self", "dim0", "dim1"]
        or [row["type"] for row in arguments] != ["Tensor", "int", "int"]
        or canonical_json(arguments[0]["alias"]) != canonical_json(_ALIAS)
        or any(row["alias"] is not None for row in arguments[1:])
        or len(returns) != 1
        or returns[0]["type"] != "Tensor"
        or canonical_json(returns[0]["alias"]) != canonical_json(_ALIAS)
        or type(call["result_arity"]) is not int
        or call["result_arity"] != 1
        or len(call["result_roster"]) != 1
    ):
        raise ValueError("transpose needs its exact original Tensor(a) argument/result and axis schema")
    dtype = _tensor(arguments[0], rank=2, dtypes=_DTYPES)
    original = _argument(arguments[0])["value"]
    result = call["result_roster"][0]
    if (
        type(original["rank"]) is not int
        or type(result["rank"]) is not int
        or result["rank"] != 2
        or result["kind"] != "tensor"
        or result["dtype"] != dtype
        or result["storage_dtype"] != dtype
        or result["layout"] != "torch.strided"
        or result["device"] != "cpu"
    ):
        raise ValueError("transpose must preserve original rank-two CPU strided input/result storage")
    axes = [_argument(row) for row in arguments[1:]]
    if any(type(axis) is not int or not -2 <= axis < 2 for axis in axes):
        raise ValueError("transpose axes must be exact original integers in the rank-two dimension domain")
    normalized = [axis + 2 if axis < 0 else axis for axis in axes]
    permutation = [0, 1]
    permutation[normalized[0]], permutation[normalized[1]] = permutation[normalized[1]], permutation[normalized[0]]
    return dtype, dict(zip(("dim0", "dim1"), axes, strict=True)), permutation


def transpose_forms(trace, observation, defaults, *, numerical_semantics=None, zero_returns=None):
    """Bind all original transpose occurrences; unsupported forms stay explicit."""
    forms = []
    for call in call_contracts(trace, observation, defaults, zero_returns=zero_returns):
        if call["target"] != TARGET:
            continue
        form = {"form_schema": FORM_SCHEMA, **call, "source_numerical_semantics": copy.deepcopy(numerical_semantics)}
        try:
            if call["status"] != "bound":
                raise ValueError(call["reason"])
            dtype, parameters, permutation = _binding(call)
            form.update(
                status="supported",
                rank=2,
                operand_dtypes=[dtype],
                result_dtypes=[dtype],
                parameters=parameters,
                permutation=permutation,
            )
        except (KeyError, TypeError, ValueError) as error:
            form.update(status="unknown", reason=str(error))
        forms.append(form)
    return forms


def transpose_source(form, *, extent, max_tensor_elements):
    """Construct rectangular source geometry after complete logical budgeting."""
    if (
        type(form) is not dict
        or form.get("form_schema") != FORM_SCHEMA
        or form.get("status") != "supported"
        or type(extent) is not int
        or extent < 1
        or type(max_tensor_elements) is not int
        or max_tensor_elements < 1
    ):
        raise ValueError("transpose needs a supported original form and explicit positive geometry/budget")
    dtype, parameters, permutation = _binding(form)
    if (
        type(form["rank"]) is not int
        or form["rank"] != 2
        or form["operand_dtypes"] != [dtype]
        or form["result_dtypes"] != [dtype]
        or canonical_json(form["parameters"]) != canonical_json(parameters)
        or canonical_json(form["permutation"]) != canonical_json(permutation)
    ):
        raise ValueError("transpose changed its original type, axes, alias or complete result binding")
    elements = 2 * extent * (extent + 1)
    if elements > max_tensor_elements:
        raise ValueError("transpose exceeds complete tensor-element budget before geometry allocation")
    shape = [extent, extent + 1]
    output_shape = [shape[axis] for axis in permutation]
    metadata = {
        "schema": SOURCE_SCHEMA,
        "target": TARGET,
        "inputs": [{"name": "X", "dtype": dtype, "shape": shape}],
        "outputs": [{"name": "Y", "kind": "tensor", "dtype": dtype, "shape": output_shape}],
        "parameters": parameters,
        "permutation": permutation,
        "schema_alias": copy.deepcopy(_ALIAS),
        "source_numerical_semantics": copy.deepcopy(form["source_numerical_semantics"]),
        "tensor_elements": elements,
        "logical_payload_bytes": elements * (_DTYPES[dtype] // 8),
        "scalar_products": 0,
        "scope": "typed original transpose source only; numerical/alias/effect/target admission unproved",
    }
    loader = (
        "import torch\n\n"
        "class Model(torch.nn.Module):\n"
        "    def forward(self, X):\n"
        f"        return torch.ops.aten.transpose.int(X, {parameters['dim0']}, {parameters['dim1']})\n\n"
        "def get_model_and_inputs():\n"
        f"    return Model(), (torch.zeros({shape!r}, dtype=torch.{dtype}),)\n"
    )
    return OriginalOperatorSource(loader, json.dumps(metadata, sort_keys=True, allow_nan=False))

"""Original reshape bindings and bounded contiguous source construction.

Positive fixed literals constrain the total product; a single -1 retains its
original syntax and requires unique inference. Fresh leading extents choose
only legal input factorizations. Schema aliases remain declarations, not an
observed alias, effect, numerical or physical-resource permission.
"""

from __future__ import annotations

import copy
import json
import math

from merlin.common.jsonio import canonical_json

from .frontend_original_call import call_contracts
from .original_operator_sources import SOURCE_SCHEMA, OriginalOperatorSource, _argument, _tensor

FORM_SCHEMA = "merlin.original_reshape_form.v1"
TARGET = "aten.reshape.default"
_SCHEMA = "aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)"
_ALIAS = {"before": ["a"], "after": ["a"], "write": False}
_DTYPES = {"float32": 32, "int8": 8, "int16": 16, "int32": 32, "int64": 64}
_MAX_INDEX = (1 << 63) - 1


def _product(shape):
    if type(shape) is not list or not shape:
        raise ValueError("reshape needs complete nonempty positive shape metadata")
    product = 1
    for dimension in shape:
        if type(dimension) is not int or not 1 <= dimension <= _MAX_INDEX:
            raise ValueError("reshape positive dimensions must fit the signed64 index domain")
        if product > _MAX_INDEX // dimension:
            raise ValueError("reshape shape product exceeds the signed64 index domain")
        product *= dimension
    return product


def _literal(shape):
    if type(shape) is not list or not shape:
        raise ValueError("reshape needs its exact nonempty original shape literal")
    if any(type(dimension) is not int or not (dimension == -1 or 1 <= dimension <= _MAX_INDEX) for dimension in shape):
        raise ValueError("reshape supports positive signed64 dimensions and one exact -1 only")
    if shape.count(-1) > 1:
        raise ValueError("reshape cannot uniquely infer multiple -1 dimensions")
    product = _product([dimension for dimension in shape if dimension != -1] or [1])
    return product, shape.index(-1) if -1 in shape else None


def _resolve(shape, elements):
    product, inferred = _literal(shape)
    if type(elements) is not int or not 1 <= elements <= _MAX_INDEX:
        raise ValueError("reshape needs a positive signed64 original element count")
    if inferred is None:
        if elements != product:
            raise ValueError("reshape fixed original literal product differs from the input element count")
    elif elements % product:
        raise ValueError("reshape original -1 inference needs exact product divisibility")
    result = list(shape)
    if inferred is not None:
        result[inferred] = elements // product
    return result


def _contiguous(shape, strides):
    _product(shape)
    if type(strides) is not list or len(strides) != len(shape):
        raise ValueError("reshape needs complete original contiguous stride metadata")
    expected = 1
    for dimension, stride in zip(reversed(shape), reversed(strides), strict=True):
        if type(stride) is not int or not 1 <= stride <= _MAX_INDEX:
            raise ValueError("reshape original strides must be positive signed64 integers")
        # Positive singleton axes do not constrain the contiguous address map.
        if dimension != 1 and stride != expected:
            raise ValueError("reshape original layout is not proved contiguous")
        expected *= dimension


def _binding(call, *, metadata_budget=None):
    arguments, returns = call["arguments"], call["schema_returns"]
    if (
        call["target"] != TARGET
        or call["schema"] != _SCHEMA
        or [row["name"] for row in arguments] != ["self", "shape"]
        or [row["type"] for row in arguments] != ["Tensor", "List[int]"]
        or canonical_json([row["ordinal"] for row in arguments]) != canonical_json([0, 1])
        or any(
            row["binding"] != "explicit" or row["has_default"] is not False or row["kwarg_only"] is not False
            for row in arguments
        )
        or arguments[0]["path"] not in {"args/0", "kwargs/self"}
        or arguments[1]["path"] not in {"args/1", "kwargs/shape"}
        or canonical_json([row["alias"] for row in arguments]) != canonical_json([_ALIAS, None])
        or len(returns) != 1
        or returns[0]["type"] != "Tensor"
        or canonical_json(returns[0]["alias"]) != canonical_json(_ALIAS)
        or type(call["result_arity"]) is not int
        or call["result_arity"] != 1
        or len(call["result_roster"]) != 1
    ):
        raise ValueError("reshape needs its exact original required Tensor(a)/shape and Tensor(a) result schema")
    original = _argument(arguments[0])["value"]
    rank = original["rank"]
    if type(rank) is not int or rank < 1:
        raise ValueError("reshape needs a complete positive original input rank")
    encoded = arguments[1]["value"]
    if type(encoded) is not dict or encoded.get("kind") != "list" or type(encoded.get("items")) is not list:
        raise ValueError("reshape shape must retain its original typed list literal")
    if metadata_budget is not None and 2 * (rank + len(encoded["items"])) > metadata_budget:
        raise ValueError("reshape rank/stride metadata exceeds its budget before geometry allocation")
    dtype = _tensor(arguments[0], rank=rank, dtypes=_DTYPES)
    shape = _argument(arguments[1])
    _literal(shape)
    result = call["result_roster"][0]
    if (
        result["kind"] != "tensor"
        or type(result["rank"]) is not int
        or result["rank"] != len(shape)
        or result["dtype"] != dtype
        or result["storage_dtype"] != dtype
        or result["layout"] != "torch.strided"
        or result["device"] != "cpu"
    ):
        raise ValueError("reshape must preserve original CPU strided input/result storage and literal result rank")
    return dtype, rank, shape


def _premises(geometry, rank, shape):
    if type(geometry) is not dict or set(geometry) != {
        "input_shape",
        "input_strides",
        "output_shape",
        "output_strides",
    }:
        raise ValueError("reshape lost its complete original shape/stride witness")
    original, output = geometry["input_shape"], geometry["output_shape"]
    if type(original) is not list or len(original) != rank:
        raise ValueError("reshape original input shape differs from its typed rank")
    elements = _product(original)
    if canonical_json(output) != canonical_json(_resolve(shape, elements)):
        raise ValueError("reshape original output shape differs from exact literal inference")
    _contiguous(original, geometry["input_strides"])
    _contiguous(output, geometry["output_strides"])


def reshape_forms(trace, observation, defaults, *, numerical_semantics=None, zero_returns=None):
    """Preserve every original reshape, including unimplemented layout/domain cases."""
    values = {value["id"]: value for node in trace["graphs"]["original"]["nodes"] for value in node["results"]}
    forms = []
    for call in call_contracts(trace, observation, defaults, zero_returns=zero_returns):
        if call["target"] != TARGET:
            continue
        form = {"form_schema": FORM_SCHEMA, **call, "source_numerical_semantics": copy.deepcopy(numerical_semantics)}
        try:
            if call["status"] != "bound":
                raise ValueError(call["reason"])
            dtype, rank, shape = _binding(call)
            original = values[_argument(call["arguments"][0])["value"]["id"]]
            output = values[call["result_roster"][0]["id"]]
            geometry = {
                "input_shape": original.get("shape"),
                "input_strides": original.get("stride"),
                "output_shape": output.get("shape"),
                "output_strides": output.get("stride"),
            }
            _premises(geometry, rank, shape)
            form.update(
                status="supported",
                rank=rank,
                operand_dtypes=[dtype],
                result_dtypes=[dtype],
                parameters={"shape": copy.deepcopy(shape)},
                original_geometry=copy.deepcopy(geometry),
                shape_relation="equal_positive_product",
                layout_relation="contiguous",
            )
        except (KeyError, TypeError, ValueError) as error:
            form.update(status="unknown", reason=str(error))
        forms.append(form)
    return forms


def reshape_source(form, *, extent, max_tensor_elements):
    """Choose a legal factorization with an exact fresh leading input extent."""
    if (
        type(form) is not dict
        or form.get("form_schema") != FORM_SCHEMA
        or form.get("status") != "supported"
        or type(extent) is not int
        or not 1 <= extent <= _MAX_INDEX
        or type(max_tensor_elements) is not int
        or max_tensor_elements < 1
    ):
        raise ValueError("reshape needs its supported original form and explicit positive signed64 geometry/budget")
    dtype, rank, literal = _binding(form, metadata_budget=max_tensor_elements)
    if 2 * (rank + len(literal)) > max_tensor_elements:
        raise ValueError("reshape rank/stride metadata exceeds its budget before geometry allocation")
    if (
        type(form["rank"]) is not int
        or form["rank"] != rank
        or canonical_json(form["operand_dtypes"]) != canonical_json([dtype])
        or canonical_json(form["result_dtypes"]) != canonical_json([dtype])
        or canonical_json(form["parameters"]) != canonical_json({"shape": literal})
        or form["shape_relation"] != "equal_positive_product"
        or form["layout_relation"] != "contiguous"
    ):
        raise ValueError("reshape changed its original literal, rank, storage or shape/layout relation")
    _premises(form["original_geometry"], rank, literal)
    product, inferred = _literal(literal)
    if inferred is not None and product > _MAX_INDEX // extent:
        raise ValueError("reshape inferred fresh product exceeds the signed64 index domain")
    count = product if inferred is None else product * extent
    if 2 * count > max_tensor_elements:
        raise ValueError("reshape exceeds complete tensor-element budget before geometry allocation")
    if count % extent or (rank == 1 and count != extent):
        raise ValueError("reshape requested fresh leading extent cannot preserve its original literal product and rank")
    remaining = count // extent
    input_shape = [extent]
    for axis in range(1, rank - 1):
        dimension = math.gcd(remaining, extent + axis)
        input_shape.append(dimension)
        remaining //= dimension
    if rank > 1:
        input_shape.append(remaining)
    output_shape = _resolve(literal, count)
    elements = 2 * count
    metadata = {
        "schema": SOURCE_SCHEMA,
        "target": TARGET,
        "inputs": [{"name": "X", "dtype": dtype, "shape": input_shape}],
        "outputs": [{"name": "Y", "kind": "tensor", "dtype": dtype, "shape": output_shape}],
        "parameters": {"shape": copy.deepcopy(literal)},
        "shape_relation": "equal_positive_product",
        "layout_relation": "contiguous",
        "schema_aliases": {"self": copy.deepcopy(_ALIAS), "result": copy.deepcopy(_ALIAS)},
        "geometry_scope": "fixed_literal_product_factorization"
        if inferred is None
        else "unique_positive_minus_one_inference",
        "source_numerical_semantics": copy.deepcopy(form["source_numerical_semantics"]),
        "tensor_elements": elements,
        "logical_payload_bytes": elements * (_DTYPES[dtype] // 8),
        "scalar_products": 0,
        "scope": "original contiguous reshape source only; alias/effect/numerical/packing/target admission unproved",
    }
    loader = (
        "import torch\n\n"
        "class Model(torch.nn.Module):\n"
        "    def forward(self, X):\n"
        f"        return torch.ops.aten.reshape.default(X, {literal!r})\n\n"
        "def get_model_and_inputs():\n"
        f"    return Model(), (torch.zeros({input_shape!r}, dtype=torch.{dtype}),)\n"
    )
    return OriginalOperatorSource(loader, json.dumps(metadata, sort_keys=True, allow_nan=False))

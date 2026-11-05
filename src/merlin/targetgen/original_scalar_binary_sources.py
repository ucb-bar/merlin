"""Bounded original Tensor calls whose second argument is a FloatLiteral.

The loader preserves Python's literal kind and exact value rather than adding
a second SSA or tensorizing a scalar. Construction proves no registered scalar
conversion, numerical policy, promotion, alias, effects or target support.
"""

from __future__ import annotations

import copy
import json

from merlin.common.jsonio import canonical_json

from .frontend_original_call import call_contracts, default_value
from .original_operator_sources import SOURCE_SCHEMA, OriginalOperatorSource

FORM_SCHEMA = "merlin.original_scalar_binary_form.v1"
INTEGER_FORM_SCHEMA = "merlin.original_scalar_binary_form.v2"
TARGETS = frozenset({"aten.mul.Tensor", "aten.div.Tensor"})


def _tensor(value):
    if (
        type(value) is not dict
        or value.get("kind") != "tensor"
        or type(value.get("id")) is not str
        or not value["id"]
        or type(value.get("rank")) is not int
        or value["rank"] < 1
        or value.get("dtype") != "float32"
        or value.get("storage_dtype") != "float32"
        or value.get("compute_dtype") not in {None, "float32"}
        or value.get("layout") != "torch.strided"
        or value.get("device") != "cpu"
    ):
        raise ValueError("scalar binary needs an original direct positive-rank strided CPU f32 Tensor")
    return value["rank"]


def integer_tensor_binding(call, rows):
    """Join one original signed64 literal to its observed Tensor argument.

    This is a data reader. The live schema intake must separately replay its
    complete native request, original source and getter before construction.
    Wrapped storage establishes no promotion or operator arithmetic contract.
    """
    literal = call["arguments"][1]["value"]
    scalar = default_value(literal)
    if type(scalar) is not int or not -(1 << 63) <= scalar < (1 << 63):
        raise ValueError("scalar binary v2 requires an exact original signed64 Python integer")
    expected = {
        "node": call["node"],
        "target": call["target"],
        "schema": call["schema"],
        "argument_index": 1,
        "argument_path": "args/1",
        "literal": {"type": "int", "value": str(scalar)},
    }
    if type(rows) is not list:
        raise ValueError("integer scalar source requires its original native Tensor-binding row")
    matches = [
        row
        for row in rows
        if type(row) is dict
        and type(row.get("request")) is dict
        and row["request"].get("node") == call["node"]
        and row["request"].get("argument_path") == "args/1"
    ]
    if len(matches) != 1:
        raise ValueError("integer scalar source needs exactly one original Tensor-binding row")
    row = matches[0]
    native = {
        "schema": call["schema"],
        "argument_name": "other",
        "source_allows_number": True,
        "wrapped_number": True,
        "shape": [],
        "dtype": "torch.int64",
        "element_bytes": 8,
        "literal": expected["literal"],
        "disjoint_from_prior_live_boxes": True,
    }
    if canonical_json(row) != canonical_json({"request": expected, "status": "observed", "native": native}):
        raise ValueError("integer scalar source changed original literal, native wrapped storage or argument identity")
    return copy.deepcopy(row)


def _binding(call, *, version=1):
    target = call["target"]
    arguments = call["arguments"]
    if (
        target not in TARGETS
        or call["schema"] != target.replace("aten.", "aten::", 1) + "(Tensor self, Tensor other) -> Tensor"
        or type(arguments) is not list
        or len(arguments) != 2
        or type(call["result_arity"]) is not int
        or call["result_arity"] != 1
        or type(call["schema_returns"]) is not list
        or len(call["schema_returns"]) != 1
        or call["schema_returns"][0]["type"] != "Tensor"
        or call["schema_returns"][0]["alias"] is not None
        or type(call["result_roster"]) is not list
        or len(call["result_roster"]) != 1
    ):
        raise ValueError("scalar binary needs its exact original Tensor argument and complete result schema")
    for ordinal, (name, argument) in enumerate(zip(("self", "other"), arguments, strict=True)):
        if (
            type(argument["ordinal"]) is not int
            or argument["ordinal"] != ordinal
            or argument["name"] != name
            or argument["type"] != "Tensor"
            or argument["alias"] is not None
            or argument["kwarg_only"] is not False
            or argument["has_default"] is not False
            or argument["binding"] != "explicit"
            or argument["path"] != f"args/{ordinal}"
        ):
            raise ValueError("scalar binary changed the original argument order, binding or public schema")
    first = arguments[0]["value"]
    if (
        type(first) is not dict
        or set(first) != {"kind", "node_id", "value"}
        or first["kind"] != "ssa"
        or type(first["node_id"]) is not str
        or not first["node_id"]
    ):
        raise ValueError("scalar binary requires exactly one original Tensor SSA first argument")
    rank = _tensor(first["value"])
    literal = arguments[1]["value"]
    if type(literal) is not dict or literal.get("kind") not in ({"float", "int"} if version == 2 else {"float"}):
        raise ValueError("scalar binary requires the explicit original finite FloatLiteral, not a second SSA")
    scalar = default_value(literal)
    if literal["kind"] == "int":
        integer_tensor_binding(call, [call.get("tensor_binding")])
    elif version == 2 and call.get("tensor_binding") is not None:
        raise ValueError("floating scalar source cannot inherit an integer Tensor-binding row")
    result = call["result_roster"][0]
    if _tensor(result) != rank or result["id"] == first["value"]["id"]:
        raise ValueError("scalar binary must preserve original input/result rank and distinct result identity")
    return rank, literal, scalar


def scalar_binary_forms(
    trace, observation, defaults, *, numerical_semantics=None, zero_returns=None, version=1, tensor_bindings=None
):
    """Retain every selected mul/div call, including unsupported literal domains."""
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("scalar binary forms require their explicit supported source version")
    forms = []
    for call in call_contracts(trace, observation, defaults, zero_returns=zero_returns):
        if call["target"] not in TARGETS:
            continue
        form = {
            "form_schema": FORM_SCHEMA if version == 1 else INTEGER_FORM_SCHEMA,
            **call,
            "source_numerical_semantics": copy.deepcopy(numerical_semantics),
        }
        try:
            if call["status"] != "bound":
                raise ValueError(call["reason"])
            if version == 2 and call["arguments"][1]["value"].get("kind") == "int":
                form["tensor_binding"] = integer_tensor_binding(call, tensor_bindings)
            rank, literal, _ = _binding(form, version=version)
            form.update(
                status="supported",
                rank=rank,
                operand_dtypes=["float32"],
                result_dtypes=["float32"],
                parameters={"other": copy.deepcopy(literal), "shape_relation": "same_tensor_shape"},
            )
        except (KeyError, TypeError, ValueError) as error:
            form.update(status="unknown", reason=str(error))
        forms.append(form)
    return forms


def scalar_binary_source(form, *, extent, max_tensor_elements):
    """Construct one-input source after complete rank and f32 payload bounds.

    The element budget bounds both logical input and output, hence also their
    exact four-byte storage cost. Original shape extents do not select geometry.
    """
    if (
        type(form) is not dict
        or form.get("form_schema") not in {FORM_SCHEMA, INTEGER_FORM_SCHEMA}
        or form.get("status") != "supported"
        or type(extent) is not int
        or extent < 1
        or type(max_tensor_elements) is not int
        or max_tensor_elements < 1
    ):
        raise ValueError("scalar binary source needs its supported original form and positive geometry/budget")
    try:
        rank, literal, scalar = _binding(form, version=2 if form["form_schema"] == INTEGER_FORM_SCHEMA else 1)
        expected = {"other": literal, "shape_relation": "same_tensor_shape"}
        if (
            type(form["rank"]) is not int
            or form["rank"] != rank
            or canonical_json(form["parameters"]) != canonical_json(expected)
            or canonical_json(form["operand_dtypes"]) != canonical_json(["float32"])
            or canonical_json(form["result_dtypes"]) != canonical_json(["float32"])
        ):
            raise ValueError("scalar binary changed its original literal, rank or ordered storage types")
        if 2 * rank > max_tensor_elements:
            raise ValueError("scalar binary rank metadata exceeds its complete budget before allocation")
        count = 1
        for axis in range(rank):
            dimension = extent + axis
            if count > (max_tensor_elements // 2) // dimension:
                raise ValueError("scalar binary exceeds its complete tensor payload budget before allocation")
            count *= dimension
    except (KeyError, TypeError) as error:
        raise ValueError("scalar binary source lost its complete original binding") from error
    shape = [extent + axis for axis in range(rank)]
    metadata = {
        "schema": SOURCE_SCHEMA,
        "target": form["target"],
        "inputs": [{"name": "X", "dtype": "float32", "shape": shape}],
        "outputs": [{"name": "Y", "kind": "tensor", "dtype": "float32", "shape": shape}],
        "parameters": copy.deepcopy(expected),
        "source_numerical_semantics": copy.deepcopy(form["source_numerical_semantics"]),
        "tensor_elements": 2 * count,
        "logical_payload_bytes": 8 * count,
        # Existing source budget vocabulary: one scalar work slot per output.
        # This is no target multiplication or performance correspondence.
        "scalar_products": count,
        "scope": "typed scalar source only; conversion, numerical policy, owner, effects and hardware unqualified",
    }
    if form["form_schema"] == INTEGER_FORM_SCHEMA:
        metadata["original_form_schema"] = INTEGER_FORM_SCHEMA
        metadata["original_tensor_binding"] = copy.deepcopy(form.get("tensor_binding"))
    loader = (
        "import torch\n\nclass Model(torch.nn.Module):\n"
        "    def forward(self, X):\n"
        f"        return torch.ops.{form['target']}(X, {scalar!r})\n\n"
        "def get_model_and_inputs():\n"
        f"    return Model(), (torch.zeros({shape!r}, dtype=torch.float32),)\n"
    )
    return OriginalOperatorSource(loader, json.dumps(metadata, sort_keys=True, allow_nan=False))

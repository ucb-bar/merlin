"""Original typed add calls with exact right-aligned broadcast relations.

Original extents establish equality and singleton relations only. Fresh source
dimensions come from an explicit bounded extent. Construction grants no dtype
promotion, alias, numerical-domain, physical or target permission.
"""

from __future__ import annotations

import copy
import json
import math

from merlin.common.jsonio import canonical_json

from .frontend_original_call import call_contracts
from .original_operator_sources import _MATMUL_DTYPES, SOURCE_SCHEMA, OriginalOperatorSource, _argument, _tensor

FORM_SCHEMA = "merlin.original_broadcast_add_form.v1"
TARGET = "aten.add.Tensor"


def _binding(call):
    arguments = call["arguments"]
    if (
        call["target"] != TARGET
        or [row["name"] for row in arguments] != ["self", "other", "alpha"]
        or [row["type"] for row in arguments[:2]] != ["Tensor", "Tensor"]
        or arguments[2]["type"] not in {"number", "Scalar"}
        or any(row["alias"] is not None for row in arguments)
        or type(call["result_arity"]) is not int
        or call["result_arity"] != 1
        or len(call["result_roster"]) != 1
        or len(call["schema_returns"]) != 1
        or call["schema_returns"][0]["type"] != "Tensor"
        or call["schema_returns"][0]["alias"] is not None
    ):
        raise ValueError("broadcast add needs its complete original direct Tensor argument/result schema")
    alpha = _argument(arguments[2])
    if type(alpha) is not int or alpha != 1:
        raise ValueError("broadcast add requires the original exact integer unit alpha")
    originals = [_argument(row) for row in arguments[:2]]
    ranks = [row["value"]["rank"] for row in originals]
    if any(type(rank) is not int or rank < 0 for rank in ranks):
        raise ValueError("broadcast add needs exact original nonnegative Tensor ranks")
    dtypes = [_tensor(row, rank=rank, dtypes=_MATMUL_DTYPES) for row, rank in zip(arguments[:2], ranks, strict=True)]
    if originals[0]["value"]["id"] == originals[1]["value"]["id"]:
        raise ValueError("broadcast add does not implement shared operand identity constraints")
    result = call["result_roster"][0]
    if (
        result["kind"] != "tensor"
        or type(result["rank"]) is not int
        or result["rank"] != max(ranks)
        or dtypes[0] != dtypes[1]
        or result["dtype"] != dtypes[0]
        or result["storage_dtype"] != result["dtype"]
        or result["layout"] != "torch.strided"
        or result["device"] != "cpu"
    ):
        raise ValueError("broadcast add must preserve original rank and ordered input/result storage without promotion")
    return dtypes, [result["dtype"]], ranks


def broadcast_add_forms(trace, observation, defaults, *, numerical_semantics=None, zero_returns=None):
    """Derive exact original singleton/equality relations without retaining sizes."""
    values = {value["id"]: value for node in trace["graphs"]["original"]["nodes"] for value in node["results"]}
    forms = []
    for call in call_contracts(trace, observation, defaults, zero_returns=zero_returns):
        if call["target"] != TARGET:
            continue
        form = {"form_schema": FORM_SCHEMA, **call, "source_numerical_semantics": copy.deepcopy(numerical_semantics)}
        try:
            if call["status"] != "bound":
                raise ValueError(call["reason"])
            operands, results, ranks = _binding(call)
            identities = [_argument(row)["value"]["id"] for row in call["arguments"][:2]]
            identities.append(call["result_roster"][0]["id"])
            shapes = [values[identity].get("shape") for identity in identities]
            if any(
                type(shape) is not list
                or len(shape) != rank
                or any(type(value) is not int or value < 1 for value in shape)
                for shape, rank in zip(shapes, [*ranks, max(ranks)], strict=True)
            ):
                raise ValueError("broadcast add needs complete positive original shape relations")
            padded = [[1] * (max(ranks) - rank) + shape for shape, rank in zip(shapes[:2], ranks, strict=True)]
            expected = []
            for left, right in zip(*padded, strict=True):
                if left != right and left != 1 and right != 1:
                    raise ValueError("original add operands violate right-aligned singleton broadcasting")
                expected.append(max(left, right))
            if shapes[2] != expected:
                raise ValueError("original add result differs from the complete broadcast shape relation")
            parameters = {
                "alpha": 1,
                "broadcasting": "right_aligned",
                "operand_axes": [["singleton" if value == 1 else "varying" for value in shape] for shape in shapes[:2]],
                "output_axes": ["singleton" if value == 1 else "varying" for value in expected],
            }
            form.update(status="supported", operand_dtypes=operands, result_dtypes=results, parameters=parameters)
        except (KeyError, TypeError, ValueError) as error:
            form.update(status="unknown", reason=str(error))
        forms.append(form)
    return forms


def _parameters(form, ranks):
    parameters = form["parameters"]
    if (
        type(parameters) is not dict
        or set(parameters) != {"alpha", "broadcasting", "operand_axes", "output_axes"}
        or type(parameters["alpha"]) is not int
        or parameters["alpha"] != 1
        or parameters["broadcasting"] != "right_aligned"
        or type(parameters["operand_axes"]) is not list
        or len(parameters["operand_axes"]) != 2
        or type(parameters["output_axes"]) is not list
        or len(parameters["output_axes"]) != max(ranks)
    ):
        raise ValueError("broadcast add lost its exact original alpha/rank/broadcast parameters")
    for pattern, rank in zip(parameters["operand_axes"], ranks, strict=True):
        if type(pattern) is not list or len(pattern) != rank:
            raise ValueError("broadcast add changed an original operand rank")
    patterns = [*parameters["operand_axes"], parameters["output_axes"]]
    if any(type(kind) is not str or kind not in {"singleton", "varying"} for row in patterns for kind in row):
        raise ValueError("broadcast add has an unsupported original axis relation")
    padded = [
        ["singleton"] * (max(ranks) - rank) + row for row, rank in zip(parameters["operand_axes"], ranks, strict=True)
    ]
    expected = ["varying" if "varying" in pair else "singleton" for pair in zip(*padded, strict=True)]
    if parameters["output_axes"] != expected:
        raise ValueError("broadcast add result rank/relations differ from its original operands")
    return parameters


def broadcast_add_source(form, *, extent, max_tensor_elements):
    """Bound all original input/output and rank metadata before geometry allocation."""
    if (
        type(form) is not dict
        or form.get("form_schema") != FORM_SCHEMA
        or form.get("status") != "supported"
        or type(extent) is not int
        or extent < 1
        or type(max_tensor_elements) is not int
        or max_tensor_elements < 1
    ):
        raise ValueError("broadcast add source needs its supported original form and explicit positive geometry/budget")
    operands, results, ranks = _binding(form)
    # Even all-singleton shapes must bound the dimension metadata separately
    # through the same explicit limit before pattern/geometry expansion.
    if sum(ranks) + max(ranks) > max_tensor_elements:
        raise ValueError("broadcast add rank metadata exceeds its preallocation budget")
    parameters = _parameters(form, ranks)
    if canonical_json(form["operand_dtypes"]) != canonical_json(operands) or canonical_json(
        form["result_dtypes"]
    ) != canonical_json(results):
        raise ValueError("broadcast add changed its original ordered storage types")
    counts = []
    for pattern, rank in zip(
        [*parameters["operand_axes"], parameters["output_axes"]], [*ranks, max(ranks)], strict=True
    ):
        count = 1
        offset = max(ranks) - rank
        for axis, kind in enumerate(pattern):
            count *= 1 if kind == "singleton" else extent + offset + axis + 1
            if count > max_tensor_elements:
                raise ValueError("broadcast add exceeds its complete tensor budget before geometry allocation")
        counts.append(count)
    elements = sum(counts)
    if elements > max_tensor_elements:
        raise ValueError("broadcast add exceeds its complete tensor budget before geometry allocation")
    shape = [1 if kind == "singleton" else extent + axis + 1 for axis, kind in enumerate(parameters["output_axes"])]
    inputs = []
    for name, pattern, rank in zip(("X", "W"), parameters["operand_axes"], ranks, strict=True):
        offset = len(shape) - rank
        dimensions = [1 if kind == "singleton" else shape[offset + axis] for axis, kind in enumerate(pattern)]
        inputs.append({"name": name, "dtype": results[0], "shape": dimensions})
    outputs = [{"name": "Y", "kind": "tensor", "dtype": results[0], "shape": shape}]
    metadata = {
        "schema": SOURCE_SCHEMA,
        "target": TARGET,
        "inputs": inputs,
        "outputs": outputs,
        "parameters": copy.deepcopy(parameters),
        "source_numerical_semantics": copy.deepcopy(form["source_numerical_semantics"]),
        "tensor_elements": elements,
        "logical_payload_bytes": elements * (_MATMUL_DTYPES[results[0]] // 8),
        "scalar_products": math.prod(shape),
        "scope": "typed original broadcast source only; no numerical-domain, owner, alias or hardware admission",
    }
    examples = ", ".join(f"torch.zeros({row['shape']!r}, dtype=torch.{row['dtype']})" for row in inputs)
    loader = (
        "import torch\n\nclass Model(torch.nn.Module):\n"
        "    def forward(self, X, W):\n"
        "        return torch.ops.aten.add.Tensor(X, W, alpha=1)\n\n"
        "def get_model_and_inputs():\n"
        f"    return Model(), ({examples},)\n"
    )
    return OriginalOperatorSource(loader, json.dumps(metadata, sort_keys=True, allow_nan=False))

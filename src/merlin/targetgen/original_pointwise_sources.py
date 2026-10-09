"""Original unary pointwise sources with independently bounded fresh geometry.

Exact call/default/type bindings select a source, never a numerical comparison,
operation owner, physical effect or target capability. Original tensor extents
do not choose the source geometry.
"""

from __future__ import annotations

import copy
import json
import math

from .frontend_original_call import call_contracts
from .original_operator_sources import SOURCE_SCHEMA, OriginalOperatorSource, _argument, _tensor

FORM_SCHEMA = "merlin.original_pointwise_form.v1"
_DTYPES = {"int8": 8, "float16": 16, "bfloat16": 16, "float32": 32, "float64": 64}
_OPERATIONS = {"aten.relu.default", "aten.round.default", "aten.clamp.default"}


def _binding(call):
    target = call["target"]
    arguments = call["arguments"]
    if (
        target not in _OPERATIONS
        or [row["name"] for row in arguments]
        != (["self", "min", "max"] if target == "aten.clamp.default" else ["self"])
        or arguments[0]["type"] != "Tensor"
        or any(row["alias"] is not None for row in arguments)
        or call["result_arity"] != 1
        or len(call["schema_returns"]) != 1
        or call["schema_returns"][0]["type"] != "Tensor"
        or call["schema_returns"][0]["alias"] is not None
        or len(call["result_roster"]) != 1
    ):
        raise ValueError("pointwise source needs its complete exact unaliased Tensor schema/result roster")
    original = _argument(arguments[0])
    if not isinstance(original, dict) or original.get("kind") != "ssa":
        raise ValueError("pointwise source needs its actual original Tensor operand")
    rank = original["value"]["rank"]
    if type(rank) is not int or rank < 0:
        raise ValueError("pointwise source needs the original static Tensor rank")
    dtype = _tensor(arguments[0], rank=rank, dtypes=_DTYPES)
    result = call["result_roster"][0]
    if (
        result["kind"] != "tensor"
        or result["rank"] != rank
        or result["dtype"] != dtype
        or result["storage_dtype"] != dtype
        or result["layout"] != "torch.strided"
        or result["device"] != "cpu"
    ):
        raise ValueError("pointwise source must preserve every original input/result storage type and rank")
    parameters = {}
    if target == "aten.clamp.default":
        for argument in arguments[1:]:
            value = _argument(argument)
            if argument["type"] not in {"Optional[number]", "Optional[Scalar]"} or (
                value is not None
                and (type(value) not in {int, float} or (type(value) is float and not math.isfinite(value)))
            ):
                raise ValueError("clamp source needs exact original finite optional scalar bounds")
            parameters[argument["name"]] = value
        if parameters["min"] is None and parameters["max"] is None:
            raise ValueError("clamp source needs an original lower or upper bound")
    return dtype, rank, parameters


def pointwise_forms(trace, observation, defaults, *, numerical_semantics=None, zero_returns=None):
    """Bind original unary calls without granting their numerical or effect domain."""
    forms = []
    for call in call_contracts(trace, observation, defaults, zero_returns=zero_returns):
        if call["target"] not in _OPERATIONS:
            continue
        form = {
            "form_schema": FORM_SCHEMA,
            **call,
            "source_numerical_semantics": copy.deepcopy(numerical_semantics),
        }
        try:
            if call["status"] != "bound":
                raise ValueError(call["reason"])
            dtype, rank, parameters = _binding(call)
            form.update(
                status="supported", operand_dtypes=[dtype], result_dtypes=[dtype], rank=rank, parameters=parameters
            )
        except (KeyError, TypeError, ValueError) as error:
            form.update(status="unknown", reason=str(error))
        forms.append(form)
    return forms


def pointwise_source(form, *, extent, max_tensor_elements):
    """Construct a source only after checking its complete logical geometry cost."""
    if (
        not isinstance(form, dict)
        or form.get("form_schema") != FORM_SCHEMA
        or form.get("status") != "supported"
        or type(extent) is not int
        or extent < 1
        or type(max_tensor_elements) is not int
        or max_tensor_elements < 1
    ):
        raise ValueError("pointwise source needs a supported original form and explicit positive geometry/budget")
    dtype, rank, parameters = _binding(form)
    if (
        form["operand_dtypes"] != [dtype]
        or form["result_dtypes"] != [dtype]
        or form["rank"] != rank
        or json.dumps(form["parameters"], sort_keys=True, allow_nan=False)
        != json.dumps(parameters, sort_keys=True, allow_nan=False)
    ):
        raise ValueError("pointwise source changed its complete original type/argument/result binding")
    # Bound the rank-driven prototype before constructing a dimension list.
    # Its dimensions depend only on explicit extent and original rank.
    count = 1
    for axis in range(rank):
        count *= extent + axis
        if 2 * count > max_tensor_elements:
            raise ValueError("pointwise source exceeds its complete tensor-element budget before allocation")
    if 2 * count > max_tensor_elements:
        raise ValueError("pointwise source exceeds its complete tensor-element budget before allocation")
    shape = [extent + axis for axis in range(rank)]
    inputs = [{"name": "X", "dtype": dtype, "shape": shape}]
    outputs = [{"name": "Y", "kind": "tensor", "dtype": dtype, "shape": shape}]
    metadata = {
        "schema": SOURCE_SCHEMA,
        "target": form["target"],
        "inputs": inputs,
        "outputs": outputs,
        "parameters": copy.deepcopy(parameters),
        "source_numerical_semantics": copy.deepcopy(form["source_numerical_semantics"]),
        "tensor_elements": 2 * count,
        "logical_payload_bytes": 2 * count * (_DTYPES[dtype] // 8),
        "scalar_products": 0,
        "scope": "typed original-form source construction only; numerical/owner/effect/target admission unproved",
    }
    arguments = ["X", *[repr(parameters[name]) for name in ("min", "max") if name in parameters]]
    loader = (
        "import torch\n\n"
        "class Model(torch.nn.Module):\n"
        "    def forward(self, X):\n"
        f"        return torch.ops.{form['target']}({', '.join(arguments)})\n\n"
        "def get_model_and_inputs():\n"
        f"    return Model(), (torch.zeros({shape!r}, dtype=torch.{dtype}),)\n"
    )
    return OriginalOperatorSource(loader, json.dumps(metadata, sort_keys=True, allow_nan=False))

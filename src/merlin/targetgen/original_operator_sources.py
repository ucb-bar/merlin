"""Typed original operator forms and bounded ordinary frontend sources.

Construction preserves the original call's types and parameters. It issues no
software correspondence or numerical admission; a source exists even when the
independently selected policy cannot qualify it.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass

from .frontend_original_call import call_contracts, default_value
from .software_spec import validate_numerical_semantics

FORM_SCHEMA = "merlin.original_conv2d_form.v1"
SOURCE_SCHEMA = "merlin.original_operator_source.v1"
_FLOAT_DTYPES = {"float16": 16, "bfloat16": 16, "float32": 32, "float64": 64}


def _argument(binding):
    value = binding["value"]
    return value if value["kind"] == "ssa" else default_value(value)


def _pair(value, *, positive):
    if (
        not isinstance(value, list)
        or len(value) not in {1, 2}
        or any(type(item) is not int or item < (1 if positive else 0) for item in value)
    ):
        raise ValueError("conv2d requires exact one/two-axis integer stride/padding/dilation")
    return value * 2 if len(value) == 1 else value


def _tensor(binding, *, rank):
    value = _argument(binding)
    if (
        not isinstance(value, dict)
        or value.get("kind") != "ssa"
        or value["value"]["kind"] != "tensor"
        or value["value"]["rank"] != rank
        or value["value"]["dtype"] not in _FLOAT_DTYPES
        or value["value"]["storage_dtype"] != value["value"]["dtype"]
        or value["value"]["layout"] != "torch.strided"
        or value["value"]["device"] != "cpu"
    ):
        raise ValueError("conv2d source requires a complete direct strided CPU floating tensor binding")
    return value["value"]["dtype"]


def conv2d_forms(trace, observation, defaults, *, numerical_semantics=None, zero_returns=None):
    """Select actual conv2d forms without granting their selected numeric policy."""
    forms = []
    for call in call_contracts(trace, observation, defaults, zero_returns=zero_returns):
        if call["target"] != "aten.conv2d.default":
            continue
        form = {"form_schema": FORM_SCHEMA, **call, "source_numerical_semantics": copy.deepcopy(numerical_semantics)}
        try:
            if call["status"] != "bound":
                raise ValueError(call["reason"])
            arguments = call["arguments"]
            if (
                [arg["name"] for arg in arguments]
                != ["input", "weight", "bias", "stride", "padding", "dilation", "groups"]
                or [arg["type"] for arg in arguments]
                != ["Tensor", "Tensor", "Optional[Tensor]", "List[int]", "List[int]", "List[int]", "int"]
                or any(arg["alias"] is not None for arg in arguments)
                or len(call["schema_returns"]) != 1
                or call["schema_returns"][0]["type"] != "Tensor"
                or call["schema_returns"][0]["alias"] is not None
                or len(call["result_roster"]) != 1
            ):
                raise ValueError("conv2d source has no exact complete direct Tensor schema/result roster")
            dtypes = [_tensor(arg, rank=4) for arg in arguments[:2]]
            bias = _argument(arguments[2])
            if bias is not None:
                dtypes.append(_tensor(arguments[2], rank=1))
            identities = [_argument(arg)["value"]["id"] for arg in arguments[: 2 + (bias is not None)]]
            if len(identities) != len(set(identities)):
                raise ValueError("conv2d source factory does not implement shared operand identity constraints")
            result = call["result_roster"][0]
            if (
                result["kind"] != "tensor"
                or result["rank"] != 4
                or result["dtype"] != dtypes[0]
                or any(dtype != dtypes[0] for dtype in dtypes)
                or result["storage_dtype"] != result["dtype"]
                or result["layout"] != "torch.strided"
                or result["device"] != "cpu"
            ):
                raise ValueError("conv2d original input/bias/output dtype or rank is incompatible")
            stride, padding, dilation, groups = (_argument(arg) for arg in arguments[3:])
            for value, positive in ((stride, True), (padding, False), (dilation, True)):
                _pair(value, positive=positive)
            if type(groups) is not int or groups < 1:
                raise ValueError("conv2d groups must be an exact positive integer")
            form.update(
                status="supported",
                operand_dtypes=dtypes,
                result_dtypes=[result["dtype"]],
                parameters={
                    "stride": stride,
                    "padding": padding,
                    "dilation": dilation,
                    "groups": groups,
                    "bias": bias is not None,
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            form.update(status="unknown", reason=str(error))
        forms.append(form)
    return forms


def _dtype(value):
    from merlin.common.quant_formats import get

    return get(value).name


def policy_compatibility(form, numerical_semantics):
    """Check selected policy types only, never issue numerical/correspondence proof."""
    try:
        if form["status"] != "supported":
            raise ValueError("original conv2d form is unsupported")
        policy = validate_numerical_semantics(copy.deepcopy(numerical_semantics))
        if policy["model"]["engine"] != "specir_fp_reduce":
            raise ValueError("original floating conv2d cannot use an integer surrogate numerical engine")
        if any(_dtype(dtype) != _dtype(policy["operand_dtype"]) for dtype in form["operand_dtypes"]) or any(
            _dtype(dtype) != _dtype(policy["readout_dtype"]) for dtype in form["result_dtypes"]
        ):
            raise ValueError("selected numerical policy changes original operand/result dtypes")
        return {
            "status": "dtype_compatible",
            "numerical_semantics": policy,
            "scope": (
                "declared source dtype/engine compatibility only; "
                "reference, comparison, correspondence and whole domain unqualified"
            ),
        }
    except (KeyError, TypeError, ValueError) as error:
        return {"status": "unknown", "reason": str(error)}


@dataclass(frozen=True)
class OriginalOperatorSource:
    loader: str
    metadata_json: str

    def metadata(self):
        return json.loads(self.metadata_json)


def conv2d_source(form, *, extent, max_tensor_elements):
    """Render ordinary PyTorch conv2d with independently bounded fresh geometry.

    The complete metadata cost is checked before the loader can allocate its
    example tensors. Original shape extents never select geometry. This source
    has the normal get_model_and_inputs API for upstream FX/MLIR capture.
    """
    if (
        not isinstance(form, dict)
        or form.get("form_schema") != FORM_SCHEMA
        or form.get("status") != "supported"
        or form.get("target") != "aten.conv2d.default"
        or type(extent) is not int
        or extent < 1
        or type(max_tensor_elements) is not int
        or max_tensor_elements < 1
    ):
        raise ValueError("conv2d source requires an actual supported form and explicit positive geometry/budget")
    parameters = form["parameters"]
    if set(parameters) != {"stride", "padding", "dilation", "groups", "bias"}:
        raise ValueError("conv2d source parameters differ from the complete original call")
    selected = form["arguments"]
    actual_parameters = {
        key: _argument(selected[index])
        for key, index in (
            ("stride", 3),
            ("padding", 4),
            ("dilation", 5),
            ("groups", 6),
        )
    }
    actual_parameters["bias"] = _argument(selected[2]) is not None
    if parameters != actual_parameters:
        raise ValueError("conv2d source parameters changed the exact original argument bindings")
    stride = _pair(parameters["stride"], positive=True)
    padding = _pair(parameters["padding"], positive=False)
    dilation = _pair(parameters["dilation"], positive=True)
    groups, bias = parameters["groups"], parameters["bias"]
    if type(groups) is not int or groups < 1 or type(bias) is not bool:
        raise ValueError("conv2d source requires exact original groups and optional bias form")
    dtype = form["result_dtypes"][0]
    actual_dtypes = [_tensor(arg, rank=4) for arg in selected[:2]]
    if bias:
        actual_dtypes.append(_tensor(selected[2], rank=1))
    if (
        dtype not in _FLOAT_DTYPES
        or len(form["result_dtypes"]) != 1
        or form["operand_dtypes"] != [dtype] * (3 if bias else 2)
        or form["operand_dtypes"] != actual_dtypes
        or len(form["result_roster"]) != 1
        or form["result_roster"][0]["dtype"] != dtype
    ):
        raise ValueError("conv2d source must preserve every original input/result dtype")
    # Fresh rectangular geometry, parameterized only by the explicit small
    # extent and actual operator semantics. No original tensor extents are used.
    kernel = [extent, extent + 1]
    image = [d * (k - 1) + 1 + (extent - 1) * s for d, k, s in zip(dilation, kernel, stride, strict=True)]
    spatial = [
        (i + 2 * p - d * (k - 1) - 1) // s + 1
        for i, p, d, k, s in zip(image, padding, dilation, kernel, stride, strict=True)
    ]
    input_shape = [extent, groups * extent, *image]
    weight_shape = [groups * extent, extent, *kernel]
    output_shape = [extent, groups * extent, *spatial]
    inputs = [{"name": "X", "dtype": dtype, "shape": input_shape}, {"name": "W", "dtype": dtype, "shape": weight_shape}]
    if bias:
        inputs.append({"name": "Bias", "dtype": dtype, "shape": [groups * extent]})
    outputs = [{"name": "Y", "kind": "tensor", "dtype": dtype, "shape": output_shape}]
    elements = sum(math.prod(row["shape"]) for row in [*inputs, *outputs])
    if elements > max_tensor_elements:
        raise ValueError("conv2d source exceeds the explicit complete tensor-element budget before allocation")
    metadata = {
        "schema": SOURCE_SCHEMA,
        "target": form["target"],
        "inputs": inputs,
        "outputs": outputs,
        "parameters": copy.deepcopy(parameters),
        "source_numerical_semantics": copy.deepcopy(form["source_numerical_semantics"]),
        "tensor_elements": elements,
        "logical_payload_bytes": elements * (_FLOAT_DTYPES[dtype] // 8),
        "scalar_products": math.prod(output_shape) * extent * math.prod(kernel),
        "scope": "typed original-form source construction only; no numerical, operation-owner or hardware admission",
    }
    names = [row["name"] for row in inputs]
    signature = ", ".join(names)
    arguments = ["X", "W", "Bias" if bias else "None"]
    arguments += [repr(parameters[key]) for key in ("stride", "padding", "dilation", "groups")]
    examples = ", ".join(f"torch.zeros({row['shape']!r}, dtype=torch.{dtype})" for row in inputs)
    loader = (
        "import torch\n\n"
        "class Model(torch.nn.Module):\n"
        f"    def forward(self, {signature}):\n"
        f"        return torch.ops.aten.conv2d.default({', '.join(arguments)})\n\n"
        "def get_model_and_inputs():\n"
        f"    return Model(), ({examples},)\n"
    )
    return OriginalOperatorSource(loader, json.dumps(metadata, sort_keys=True, allow_nan=False))

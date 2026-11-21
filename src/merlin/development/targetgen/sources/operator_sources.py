"""Original typed call bindings from exact public/native schema defaults.

Bindings preserve every original argument and result slot. They grant no
operation correspondence, numerical policy, purity or physical effect meaning.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass

from merlin.development.targetgen.sources.frontend_typed_add import defaults_request
from merlin.development.targetgen.sources.frontend_use_def import original_use_def_semantics
from merlin.development.targetgen.sources.operator_effects import _argument_values, _zero_return_rows
from merlin.target.spec.software import validate_numerical_semantics

# ---- frontend_original_call --------------------------------------------------------------------


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


# ---- original_operator_sources -----------------------------------------------------------------
# Typed original operator forms and bounded ordinary frontend sources.
#
# Construction preserves the original call's types and parameters. It issues no
# software correspondence or numerical admission; a source exists even when the
# independently selected policy cannot qualify it.

FORM_SCHEMA = "merlin.original_conv2d_form.v1"
MATMUL_FORM_SCHEMA = "merlin.original_matmul_form.v1"
ADD_FORM_SCHEMA = "merlin.original_add_form.v1"
SOURCE_SCHEMA = "merlin.original_operator_source.v1"
_FLOAT_DTYPES = {"float16": 16, "bfloat16": 16, "float32": 32, "float64": 64}
_MATMUL_DTYPES = _FLOAT_DTYPES | {"int8": 8}


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


def _tensor(binding, *, rank, dtypes=None):
    dtypes = _FLOAT_DTYPES if dtypes is None else dtypes
    value = _argument(binding)
    if (
        not isinstance(value, dict)
        or value.get("kind") != "ssa"
        or value["value"]["kind"] != "tensor"
        or value["value"]["rank"] != rank
        or value["value"]["dtype"] not in dtypes
        or value["value"]["storage_dtype"] != value["value"]["dtype"]
        or value["value"]["layout"] != "torch.strided"
        or value["value"]["device"] != "cpu"
    ):
        raise ValueError("original source requires a complete direct strided CPU supported tensor binding")
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
    from merlin.common.target_data import get

    return get(value).name


def policy_compatibility(form, numerical_semantics):
    """Check selected policy types only, never issue numerical/correspondence proof."""
    try:
        if form["status"] != "supported":
            raise ValueError("original operator form is unsupported")
        if any(dtype not in _FLOAT_DTYPES for dtype in [*form["operand_dtypes"], *form["result_dtypes"]]):
            raise ValueError("original integer source requires an independent arithmetic/overflow policy relation")
        policy = validate_numerical_semantics(copy.deepcopy(numerical_semantics))
        if policy["model"]["engine"] != "specir_fp_reduce":
            raise ValueError("original floating source cannot use an integer surrogate numerical engine")
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


def _matmul_types(call):
    arguments = call["arguments"]
    if (
        call["target"] != "aten.matmul.default"
        or [arg["name"] for arg in arguments] != ["self", "other"]
        or [arg["type"] for arg in arguments] != ["Tensor", "Tensor"]
        or any(arg["alias"] is not None for arg in arguments)
        or len(call["schema_returns"]) != 1
        or call["schema_returns"][0]["type"] != "Tensor"
        or call["schema_returns"][0]["alias"] is not None
        or len(call["result_roster"]) != 1
        or call["result_arity"] != 1
    ):
        raise ValueError("matmul source has no exact complete direct Tensor schema/result roster")
    dtypes = [_tensor(arg, rank=2, dtypes=_MATMUL_DTYPES) for arg in arguments]
    if _argument(arguments[0])["value"]["id"] == _argument(arguments[1])["value"]["id"]:
        raise ValueError("matmul source factory does not implement shared operand identity constraints")
    result = call["result_roster"][0]
    if (
        result["kind"] != "tensor"
        or result["rank"] != 2
        or result["dtype"] != dtypes[0]
        or dtypes[1] != dtypes[0]
        or result["storage_dtype"] != result["dtype"]
        or result["layout"] != "torch.strided"
        or result["device"] != "cpu"
    ):
        raise ValueError("matmul original input/output dtype or rank is incompatible")
    return dtypes, [result["dtype"]]


def matmul_forms(trace, observation, defaults, *, numerical_semantics=None, zero_returns=None):
    """Select rank-two original matmul, preserving signed i8 readout as i8.

    These are ordinary source forms, not a widening contraction policy. No
    reduction order, overflow domain or numerical correspondence is granted.
    """
    forms = []
    for call in call_contracts(trace, observation, defaults, zero_returns=zero_returns):
        if call["target"] != "aten.matmul.default":
            continue
        form = {
            "form_schema": MATMUL_FORM_SCHEMA,
            **call,
            "source_numerical_semantics": copy.deepcopy(numerical_semantics),
        }
        try:
            if call["status"] != "bound":
                raise ValueError(call["reason"])
            operands, results = _matmul_types(call)
            form.update(status="supported", operand_dtypes=operands, result_dtypes=results, parameters={})
        except (KeyError, TypeError, ValueError) as error:
            form.update(status="unknown", reason=str(error))
        forms.append(form)
    return forms


def matmul_source(form, *, extent, max_tensor_elements):
    """Construct typed original matmul using independent bounded geometry."""
    if (
        not isinstance(form, dict)
        or form.get("form_schema") != MATMUL_FORM_SCHEMA
        or form.get("status") != "supported"
        or type(extent) is not int
        or extent < 1
        or type(max_tensor_elements) is not int
        or max_tensor_elements < 1
    ):
        raise ValueError("matmul source requires an actual supported form and explicit positive geometry/budget")
    operands, results = _matmul_types(form)
    if form["operand_dtypes"] != operands or form["result_dtypes"] != results or form["parameters"] != {}:
        raise ValueError("matmul source changed the complete original type/argument/result binding")
    dtype = results[0]
    inputs = [
        {"name": "X", "dtype": dtype, "shape": [extent, extent + 1]},
        {"name": "W", "dtype": dtype, "shape": [extent + 1, extent + 2]},
    ]
    outputs = [{"name": "Y", "kind": "tensor", "dtype": dtype, "shape": [extent, extent + 2]}]
    elements = sum(math.prod(row["shape"]) for row in [*inputs, *outputs])
    if elements > max_tensor_elements:
        raise ValueError("matmul source exceeds the explicit complete tensor-element budget before allocation")
    metadata = {
        "schema": SOURCE_SCHEMA,
        "target": form["target"],
        "inputs": inputs,
        "outputs": outputs,
        "parameters": {},
        "source_numerical_semantics": copy.deepcopy(form["source_numerical_semantics"]),
        "tensor_elements": elements,
        "logical_payload_bytes": elements * (_MATMUL_DTYPES[dtype] // 8),
        "scalar_products": math.prod(outputs[0]["shape"]) * (extent + 1),
        "scope": "typed original-form source construction only; no numerical, operation-owner or hardware admission",
    }
    examples = ", ".join(f"torch.zeros({row['shape']!r}, dtype=torch.{dtype})" for row in inputs)
    loader = (
        "import torch\n\n"
        "class Model(torch.nn.Module):\n"
        "    def forward(self, X, W):\n"
        "        return torch.ops.aten.matmul.default(X, W)\n\n"
        "def get_model_and_inputs():\n"
        f"    return Model(), ({examples},)\n"
    )
    return OriginalOperatorSource(loader, json.dumps(metadata, sort_keys=True, allow_nan=False))


def _add_types(call):
    arguments = call["arguments"]
    if (
        call["target"] != "aten.add.Tensor"
        or [arg["name"] for arg in arguments] != ["self", "other", "alpha"]
        or [arg["type"] for arg in arguments[:2]] != ["Tensor", "Tensor"]
        or arguments[2]["type"] not in {"number", "Scalar"}
        or any(arg["alias"] is not None for arg in arguments)
        or len(call["schema_returns"]) != 1
        or call["schema_returns"][0]["type"] != "Tensor"
        or call["schema_returns"][0]["alias"] is not None
        or len(call["result_roster"]) != 1
        or call["result_arity"] != 1
    ):
        raise ValueError("add source has no exact complete direct Tensor schema/result roster")
    alpha = _argument(arguments[2])
    if type(alpha) is not int or alpha != 1:
        raise ValueError("original add source requires exact integer unit alpha")
    dtypes = [_tensor(arg, rank=2, dtypes=_MATMUL_DTYPES) for arg in arguments[:2]]
    if _argument(arguments[0])["value"]["id"] == _argument(arguments[1])["value"]["id"]:
        raise ValueError("add source factory does not implement shared operand identity constraints")
    result = call["result_roster"][0]
    if (
        result["kind"] != "tensor"
        or result["rank"] != 2
        or result["dtype"] != dtypes[0]
        or dtypes[1] != dtypes[0]
        or result["storage_dtype"] != result["dtype"]
        or result["layout"] != "torch.strided"
        or result["device"] != "cpu"
    ):
        raise ValueError("add original input/output dtype or rank is incompatible")
    return dtypes, [result["dtype"]]


def original_add_forms(trace, observation, defaults, *, numerical_semantics=None, zero_returns=None):
    """Bind typed original equal-shape add independently from numeric admission."""
    values = {value["id"]: value for node in trace["graphs"]["original"]["nodes"] for value in node["results"]}
    forms = []
    for call in call_contracts(trace, observation, defaults, zero_returns=zero_returns):
        if call["target"] != "aten.add.Tensor":
            continue
        form = {
            "form_schema": ADD_FORM_SCHEMA,
            **call,
            "source_numerical_semantics": copy.deepcopy(numerical_semantics),
        }
        try:
            if call["status"] != "bound":
                raise ValueError(call["reason"])
            operands, results = _add_types(call)
            ids = [_argument(arg)["value"]["id"] for arg in call["arguments"][:2]]
            ids += [call["result_roster"][0]["id"]]
            shapes = [values[identity].get("shape") for identity in ids]
            if any(
                not isinstance(shape, list)
                or len(shape) != 2
                or any(type(extent) is not int or extent < 1 for extent in shape)
                for shape in shapes
            ) or any(shape != shapes[0] for shape in shapes[1:]):
                raise ValueError("original add broadcasting or incomplete shape relation is unsupported")
            # Original extents prove only the equal-shape form. They are never
            # retained as geometry or used to choose any generated dimensions.
            form.update(
                status="supported",
                operand_dtypes=operands,
                result_dtypes=results,
                parameters={"alpha": 1, "broadcasting": "none"},
            )
        except (KeyError, TypeError, ValueError) as error:
            form.update(status="unknown", reason=str(error))
        forms.append(form)
    return forms


def add_source(form, *, extent, max_tensor_elements):
    """Construct exact typed unit-alpha add with fresh rectangular geometry."""
    if (
        not isinstance(form, dict)
        or form.get("form_schema") != ADD_FORM_SCHEMA
        or form.get("status") != "supported"
        or type(extent) is not int
        or extent < 1
        or type(max_tensor_elements) is not int
        or max_tensor_elements < 1
    ):
        raise ValueError("add source requires an actual supported form and explicit positive geometry/budget")
    operands, results = _add_types(form)
    parameters = {"alpha": 1, "broadcasting": "none"}
    if form["operand_dtypes"] != operands or form["result_dtypes"] != results or form["parameters"] != parameters:
        raise ValueError("add source changed the complete original type/argument/result binding")
    dtype, shape = results[0], [extent, extent + 1]
    inputs = [{"name": name, "dtype": dtype, "shape": shape} for name in ("X", "W")]
    outputs = [{"name": "Y", "kind": "tensor", "dtype": dtype, "shape": shape}]
    elements = 3 * math.prod(shape)
    if elements > max_tensor_elements:
        raise ValueError("add source exceeds the explicit complete tensor-element budget before allocation")
    metadata = {
        "schema": SOURCE_SCHEMA,
        "target": form["target"],
        "inputs": inputs,
        "outputs": outputs,
        "parameters": parameters,
        "source_numerical_semantics": copy.deepcopy(form["source_numerical_semantics"]),
        "tensor_elements": elements,
        "logical_payload_bytes": elements * (_MATMUL_DTYPES[dtype] // 8),
        "scalar_products": math.prod(shape),
        "scope": "typed original-form source construction only; no numerical, operation-owner or hardware admission",
    }
    examples = ", ".join(f"torch.zeros({row['shape']!r}, dtype=torch.{dtype})" for row in inputs)
    loader = (
        "import torch\n\nclass Model(torch.nn.Module):\n"
        "    def forward(self, X, W):\n"
        "        return torch.ops.aten.add.Tensor(X, W, alpha=1)\n\n"
        "def get_model_and_inputs():\n"
        f"    return Model(), ({examples},)\n"
    )
    return OriginalOperatorSource(loader, json.dumps(metadata, sort_keys=True, allow_nan=False))

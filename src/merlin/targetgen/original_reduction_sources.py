"""Exact original reduction calls and independently bounded source geometry.

Axes and normalized dimensions retain their original literal/default bindings.
Original positive contiguous shapes witness relations, never fresh dimensions.
Construction supplies no reference, numerical, effect or hardware admission.
"""

from __future__ import annotations

import copy
import json
import math

from merlin.common.jsonio import canonical_json

from .frontend_original_call import call_contracts
from .original_operator_sources import SOURCE_SCHEMA, OriginalOperatorSource, _argument, _tensor

FORM_SCHEMA = "merlin.original_reduction_form.v1"
_MAX_INDEX = (1 << 63) - 1
_DTYPES = {"float32": 32}
_DECLARATIONS = {
    "aten.mean.dim": (
        "aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
        (
            ("self", "Tensor", False, False),
            ("dim", "Optional[List[int]]", False, False),
            ("keepdim", "bool", True, False),
            ("dtype", "Optional[int]", True, True),
        ),
    ),
    "aten.softmax.int": (
        "aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
        (("self", "Tensor", False, False), ("dim", "int", False, False), ("dtype", "Optional[int]", True, False)),
    ),
    "aten.layer_norm.default": (
        "aten::layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, Tensor? bias=None, "
        "float eps=1.0000000000000001e-05, bool cudnn_enable=True) -> Tensor",
        (
            ("input", "Tensor", False, False),
            ("normalized_shape", "List[int]", False, False),
            ("weight", "Optional[Tensor]", True, False),
            ("bias", "Optional[Tensor]", True, False),
            ("eps", "float", True, False),
            ("cudnn_enable", "bool", True, False),
        ),
    ),
}
_DEFAULTS = {"keepdim": False, "dtype": None, "weight": None, "bias": None, "eps": 1e-5, "cudnn_enable": True}


def _product(shape):
    if type(shape) is not list:
        raise ValueError("reduction needs exact positive shape metadata")
    count = 1
    for dimension in shape:
        if type(dimension) is not int or not 1 <= dimension <= _MAX_INDEX:
            raise ValueError("reduction dimensions must be positive signed64 integers")
        if count > _MAX_INDEX // dimension:
            raise ValueError("reduction shape product exceeds the signed64 index domain")
        count *= dimension
    return count


def _axes(dimensions, rank):
    if dimensions is None or dimensions == []:
        return list(range(rank))
    if type(dimensions) is not list or any(type(d) is not int or not -rank <= d < rank for d in dimensions):
        raise ValueError("reduction axes must retain exact in-range original integers")
    axes = [d % rank for d in dimensions]
    if len(set(axes)) != len(axes):
        raise ValueError("reduction cannot repeat an original axis, including negative aliases")
    return axes


def _binding(call, *, input_shape, input_strides):
    schema, declaration = _DECLARATIONS[call["target"]]
    arguments, returns = call["arguments"], call["schema_returns"]
    if (
        call["schema"] != schema
        or len(arguments) != len(declaration)
        or canonical_json([row["ordinal"] for row in arguments]) != canonical_json(list(range(len(declaration))))
        or any(row["alias"] is not None for row in arguments)
        or len(returns) != 1
        or returns[0]["type"] != "Tensor"
        or returns[0]["alias"] is not None
        or type(call["result_arity"]) is not int
        or call["result_arity"] != 1
        or len(call["result_roster"]) != 1
    ):
        raise ValueError("reduction needs its exact ordered unaliased original argument/result schema")
    for ordinal, (row, (name, kind, default, kwarg)) in enumerate(zip(arguments, declaration, strict=True)):
        if (
            row["name"] != name
            or row["type"] != kind
            or row["has_default"] is not default
            or row["kwarg_only"] is not kwarg
            or row["binding"] not in {"explicit", "default"}
        ):
            raise ValueError("reduction changed an original argument kind, default or keyword binding")
        if row["binding"] == "default":
            if (
                not default
                or row["path"] is not None
                or canonical_json(_argument(row)) != canonical_json(_DEFAULTS[name])
            ):
                raise ValueError("reduction lost the exact independently observed public default")
        elif row["path"] not in ({f"kwargs/{name}"} if kwarg else {f"args/{ordinal}", f"kwargs/{name}"}):
            raise ValueError("reduction changed an original explicit argument path")
    value = _argument(arguments[0])["value"]
    rank = value["rank"]
    if type(rank) is not int or not 1 <= rank <= _MAX_INDEX:
        raise ValueError("reduction needs a complete positive original input rank")
    if type(input_shape) is not list or len(input_shape) != rank:
        raise ValueError("reduction original shape differs from its typed input rank")
    # Both form derivation and source replay bind rank to complete geometry
    # before any axis/range expansion. No separate rank constant is inferred.
    _contiguous(input_shape, input_strides)
    dtype = _tensor(arguments[0], rank=rank, dtypes=_DTYPES)
    target = call["target"]
    tensors = [arguments[0]]
    if target == "aten.mean.dim":
        dim, keepdim, output_dtype = (_argument(row) for row in arguments[1:])
        axes = _axes(dim, rank)
        if type(keepdim) is not bool or output_dtype is not None:
            raise ValueError("mean source needs exact Boolean keepdim and unchanged dtype=None")
        parameters = {"dim": dim, "keepdim": keepdim, "dtype": None}
        output_rank = rank if keepdim else rank - len(axes)
    elif target == "aten.softmax.int":
        dim, output_dtype = (_argument(row) for row in arguments[1:])
        axes = _axes([dim], rank)
        if output_dtype is not None:
            raise ValueError("softmax source does not implement original dtype promotion")
        parameters, output_rank = {"dim": dim, "dtype": None}, rank
    else:
        normalized = _argument(arguments[1])
        if type(normalized) is not list or not 1 <= len(normalized) <= rank:
            raise ValueError("layer norm needs its exact nonempty trailing normalized-shape literal")
        _product(normalized)
        weight, bias = (_argument(row) for row in arguments[2:4])
        for argument, selected in zip(arguments[2:4], (weight, bias), strict=True):
            if selected is not None:
                if _tensor(argument, rank=len(normalized), dtypes=_DTYPES) != dtype:
                    raise ValueError("layer norm must preserve exact same-storage affine inputs")
                tensors.append(argument)
        eps, cudnn = (_argument(row) for row in arguments[4:])
        if type(eps) is not float or not math.isfinite(eps) or type(cudnn) is not bool:
            raise ValueError("layer norm needs its exact finite FloatLiteral epsilon and Boolean cudnn flag")
        parameters = {
            "normalized_shape": normalized,
            "weight": weight is not None,
            "bias": bias is not None,
            "eps": eps,
            "cudnn_enable": cudnn,
        }
        axes, output_rank = list(range(rank - len(normalized), rank)), rank
    identities = [_argument(row)["value"]["id"] for row in tensors]
    if len(identities) != len(set(identities)):
        raise ValueError("reduction source does not implement shared operand identity constraints")
    result = call["result_roster"][0]
    if (
        result["kind"] != "tensor"
        or type(result["rank"]) is not int
        or result["rank"] != output_rank
        or result["dtype"] != dtype
        or result["storage_dtype"] != dtype
        or result["layout"] != "torch.strided"
        or result["device"] != "cpu"
    ):
        raise ValueError("reduction result must retain its exact CPU strided rank/storage relation")
    return dtype, rank, parameters, axes, tensors


def _output_shape(target, shape, parameters, axes):
    if target != "aten.mean.dim":
        return list(shape)
    return (
        [1 if axis in axes else dimension for axis, dimension in enumerate(shape)]
        if parameters["keepdim"]
        else [dimension for axis, dimension in enumerate(shape) if axis not in axes]
    )


def _contiguous(shape, strides):
    _product(shape)
    if type(strides) is not list or len(strides) != len(shape):
        raise ValueError("reduction needs complete original contiguous stride metadata")
    expected = 1
    for dimension, stride in zip(reversed(shape), reversed(strides), strict=True):
        if type(stride) is not int or not 1 <= stride <= _MAX_INDEX or (dimension != 1 and stride != expected):
            raise ValueError("reduction original layout is not proved positive contiguous")
        expected *= dimension


def _premises(geometry, target, rank, parameters, axes, tensor_count):
    if type(geometry) is not dict or set(geometry) != {
        "input_shapes",
        "input_strides",
        "output_shape",
        "output_strides",
    }:
        raise ValueError("reduction lost its complete original shape/stride witness")
    shapes, strides = geometry["input_shapes"], geometry["input_strides"]
    if (
        type(shapes) is not list
        or type(strides) is not list
        or len(shapes) != tensor_count
        or len(strides) != tensor_count
    ):
        raise ValueError("reduction changed its complete ordered original input geometry")
    if type(shapes[0]) is not list or len(shapes[0]) != rank:
        raise ValueError("reduction original shape differs from its typed input rank")
    for shape, stride in zip(shapes, strides, strict=True):
        _contiguous(shape, stride)
    if target == "aten.layer_norm.default":
        literal = parameters["normalized_shape"]
        if canonical_json(shapes[0][-len(literal) :]) != canonical_json(literal) or any(
            canonical_json(shape) != canonical_json(literal) for shape in shapes[1:]
        ):
            raise ValueError("layer norm original input suffix and affine shapes must equal its literal")
    output = _output_shape(target, shapes[0], parameters, axes)
    if canonical_json(geometry["output_shape"]) != canonical_json(output):
        raise ValueError("reduction original output shape differs from its exact axis relation")
    _contiguous(output, geometry["output_strides"])


def reduction_forms(trace, observation, defaults, *, numerical_semantics=None, zero_returns=None):
    """Retain every original reduction call and every unsupported premise."""
    values = {value["id"]: value for node in trace["graphs"]["original"]["nodes"] for value in node["results"]}
    forms = []
    for call in call_contracts(trace, observation, defaults, zero_returns=zero_returns):
        if call["target"] not in _DECLARATIONS:
            continue
        form = {"form_schema": FORM_SCHEMA, **call, "source_numerical_semantics": copy.deepcopy(numerical_semantics)}
        try:
            if call["status"] != "bound":
                raise ValueError(call["reason"])
            input_value = values[_argument(call["arguments"][0])["value"]["id"]]
            dtype, rank, parameters, axes, tensors = _binding(
                call, input_shape=input_value.get("shape"), input_strides=input_value.get("stride")
            )
            inputs = [values[_argument(row)["value"]["id"]] for row in tensors]
            result = values[call["result_roster"][0]["id"]]
            geometry = {
                "input_shapes": [value.get("shape") for value in inputs],
                "input_strides": [value.get("stride") for value in inputs],
                "output_shape": result.get("shape"),
                "output_strides": result.get("stride"),
            }
            _premises(geometry, call["target"], rank, parameters, axes, len(tensors))
            form.update(
                status="supported",
                rank=rank,
                operand_dtypes=[dtype] * len(tensors),
                result_dtypes=[dtype],
                parameters=copy.deepcopy(parameters),
                reduction_axes=axes,
                original_geometry=copy.deepcopy(geometry),
                layout_relation="contiguous",
            )
        except (KeyError, TypeError, ValueError) as error:
            form.update(status="unknown", reason=str(error))
        forms.append(form)
    return forms


def reduction_source(form, *, extent, max_tensor_elements):
    """Preflight complete rank/logical costs before allocating fresh geometry."""
    if (
        type(form) is not dict
        or form.get("form_schema") != FORM_SCHEMA
        or form.get("status") != "supported"
        or form.get("target") not in _DECLARATIONS
        or type(extent) is not int
        or not 1 <= extent <= _MAX_INDEX
        or type(max_tensor_elements) is not int
        or max_tensor_elements < 1
    ):
        raise ValueError("reduction needs its supported original form and explicit finite extent/budget")
    geometry = form["original_geometry"]
    if (
        type(geometry) is not dict
        or set(geometry) != {"input_shapes", "input_strides", "output_shape", "output_strides"}
        or type(geometry["input_shapes"]) is not list
        or not geometry["input_shapes"]
        or type(geometry["input_strides"]) is not list
        or not geometry["input_strides"]
    ):
        raise ValueError("reduction lost its complete original input shape/stride witness")
    dtype, rank, parameters, axes, tensors = _binding(
        form,
        input_shape=geometry["input_shapes"][0],
        input_strides=geometry["input_strides"][0],
    )
    if (
        type(form["rank"]) is not int
        or form["rank"] != rank
        or form["layout_relation"] != "contiguous"
        or canonical_json(form["operand_dtypes"]) != canonical_json([dtype] * len(tensors))
        or canonical_json(form["result_dtypes"]) != canonical_json([dtype])
        or canonical_json(form["parameters"]) != canonical_json(parameters)
        or canonical_json(form["reduction_axes"]) != canonical_json(axes)
    ):
        raise ValueError("reduction changed its exact original rank, storage, axes, scalar defaults or layout")
    _premises(form["original_geometry"], form["target"], rank, parameters, axes, len(tensors))
    normalized = parameters.get("normalized_shape", [])
    leading = rank - len(normalized)
    count, output_count = 1, 1
    for axis in range(rank):
        dimension = extent + axis if axis < leading else normalized[axis - leading]
        if dimension > _MAX_INDEX or count > _MAX_INDEX // dimension:
            raise ValueError("reduction fresh shape product exceeds the signed64 index domain")
        count *= dimension
        if form["target"] != "aten.mean.dim" or axis not in axes:
            output_count *= dimension
        if count + output_count > max_tensor_elements:
            raise ValueError("reduction exceeds complete tensor-element budget before geometry allocation")
    affine_count = _product(normalized) * (len(tensors) - 1) if normalized else 0
    elements = count + output_count + affine_count
    if elements > max_tensor_elements:
        raise ValueError("reduction exceeds complete affine/input/output tensor budget before geometry allocation")
    shape = [extent + axis for axis in range(leading)] + list(normalized)
    output_shape = _output_shape(form["target"], shape, parameters, axes)
    names = ["X"]
    if parameters.get("weight"):
        names.append("Weight")
    if parameters.get("bias"):
        names.append("Bias")
    inputs = [{"name": name, "dtype": dtype, "shape": shape if name == "X" else list(normalized)} for name in names]
    metadata = {
        "schema": SOURCE_SCHEMA,
        "target": form["target"],
        "inputs": inputs,
        "outputs": [{"name": "Y", "kind": "tensor", "dtype": dtype, "shape": output_shape}],
        "parameters": copy.deepcopy(parameters),
        "reduction_axes": list(axes),
        "layout_relation": "contiguous",
        "source_numerical_semantics": copy.deepcopy(form["source_numerical_semantics"]),
        "tensor_elements": elements,
        "logical_payload_bytes": elements * (_DTYPES[dtype] // 8),
        "scalar_products": 0,
        "geometry_scope": (
            "original_literal_shape_only"
            if leading == 0
            else "original_literal_suffix_and_fresh_leading_dimensions"
            if normalized
            else "fresh_rank_axis_geometry"
        ),
        "scope": "typed original reduction source only; numerical/effect/packing/target admission unproved",
    }
    if form["target"] == "aten.mean.dim":
        expression = f"torch.ops.aten.mean.dim(X, {parameters['dim']!r}, keepdim={parameters['keepdim']!r}, dtype=None)"
    elif form["target"] == "aten.softmax.int":
        expression = f"torch.ops.aten.softmax.int(X, {parameters['dim']!r}, dtype=None)"
    else:
        expression = (
            f"torch.ops.aten.layer_norm.default(X, {normalized!r}, "
            f"{'Weight' if parameters['weight'] else 'None'}, {'Bias' if parameters['bias'] else 'None'}, "
            f"eps={parameters['eps']!r}, cudnn_enable={parameters['cudnn_enable']!r})"
        )
    arguments = ", ".join(names)
    prototypes = ", ".join(f"torch.zeros({row['shape']!r}, dtype=torch.{dtype})" for row in inputs)
    loader = (
        "import torch\n\nclass Model(torch.nn.Module):\n"
        f"    def forward(self, {arguments}):\n        return {expression}\n\n"
        f"def get_model_and_inputs():\n    return Model(), ({prototypes},)\n"
    )
    return OriginalOperatorSource(loader, json.dumps(metadata, sort_keys=True, allow_nan=False))

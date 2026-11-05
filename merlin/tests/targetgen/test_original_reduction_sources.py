"""Independently declared reduction shape controls; no framework observations."""

import ast
import copy
import math

import pytest

from merlin.targetgen import original_reduction_sources as S
from merlin.targetgen.frontend_trace import _digest

_SCHEMAS = {
    "mean": "aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
    "softmax": "aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
    "layer_norm": "aten::layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, Tensor? bias=None, "
    "float eps=1.0000000000000001e-05, bool cudnn_enable=True) -> Tensor",
}
_TARGETS = {"mean": "aten.mean.dim", "softmax": "aten.softmax.int", "layer_norm": "aten.layer_norm.default"}
_OMITTED = object()


def strides(shape):
    result, stride = [], 1
    for dimension in reversed(shape):
        result.append(stride)
        stride *= dimension
    return list(reversed(result))


def literal(value):
    if value is None:
        return {"kind": "none"}
    if type(value) is float:
        return {"kind": "float", "value_hex": value.hex()}
    return {"kind": type(value).__name__, "value": value}


def declarations(
    family="mean",
    input_shape=(2, 3, 4),
    *,
    dim=(0, -1),
    keepdim=_OMITTED,
    normalized_shape=(4,),
    weight=True,
    bias=True,
    eps=_OMITTED,
    cudnn_enable=_OMITTED,
    dtype="float32",
    output_shape=None,
    selected_dtype=_OMITTED,
):
    """Minimal artificial schema rows, explicitly substituted for native seams."""
    target, schema = _TARGETS[family], _SCHEMAS[family]
    specs = [("input" if family == "layer_norm" else "self", "Tensor", False, False, None)]
    shapes = [("input", list(input_shape))]

    def reference(name):
        return {"node_id": name, "value_id": name + ":0"}

    args, kwargs = [reference("input")], {}
    if family == "mean":
        specs += [
            ("dim", "Optional[List[int]]", False, False, None),
            ("keepdim", "bool", True, False, False),
            ("dtype", "Optional[int]", True, True, None),
        ]
        dimensions = list(dim) if dim is not None else None
        args.append(dimensions)
        if keepdim is not _OMITTED:
            kwargs["keepdim"] = keepdim
        if selected_dtype is not _OMITTED:
            kwargs["dtype"] = selected_dtype
        axes = (
            list(range(len(input_shape)))
            if dimensions is None or not dimensions
            else [d % len(input_shape) for d in dimensions]
        )
        retained = False if keepdim is _OMITTED else keepdim
        inferred = (
            [1 if a in axes else n for a, n in enumerate(input_shape)]
            if retained
            else [n for a, n in enumerate(input_shape) if a not in axes]
        )
    elif family == "softmax":
        specs += [("dim", "int", False, False, None), ("dtype", "Optional[int]", True, False, None)]
        args.append(dim)
        if selected_dtype is not _OMITTED:
            kwargs["dtype"] = selected_dtype
        inferred = list(input_shape)
    else:
        specs += [
            ("normalized_shape", "List[int]", False, False, None),
            ("weight", "Optional[Tensor]", True, False, None),
            ("bias", "Optional[Tensor]", True, False, None),
            ("eps", "float", True, False, 1e-5),
            ("cudnn_enable", "bool", True, False, True),
        ]
        args.append(list(normalized_shape))
        if weight:
            shapes.append(("weight", list(normalized_shape)))
            kwargs["weight"] = reference("weight")
        if bias:
            shapes.append(("bias", list(normalized_shape)))
            kwargs["bias"] = reference("bias")
        if eps is not _OMITTED:
            kwargs["eps"] = eps
        if cudnn_enable is not _OMITTED:
            kwargs["cudnn_enable"] = cudnn_enable
        inferred = list(input_shape)
    shapes.append(("reduction", inferred if output_shape is None else list(output_shape)))
    values = {
        name: {
            "id": name + ":0",
            "kind": "tensor",
            "dtype": dtype,
            "storage_dtype": dtype,
            "shape": shape,
            "stride": strides(shape),
            "layout": "torch.strided",
            "device": "cpu",
        }
        for name, shape in shapes
    }
    nodes = [
        {
            "id": name,
            "ordinal": i,
            "op": "placeholder",
            "target": name,
            "args": [],
            "kwargs": {},
            "results": [values[name]],
        }
        for i, (name, _) in enumerate(shapes[:-1])
    ]
    nodes += [
        {
            "id": "reduction",
            "ordinal": len(nodes),
            "op": "call_function",
            "target": target,
            "args": args,
            "kwargs": kwargs,
            "results": [values["reduction"]],
            "result_arity": 1,
        }
    ]
    nodes += [
        {
            "id": "output",
            "ordinal": len(nodes),
            "op": "output",
            "target": "output",
            "args": [reference("reduction")],
            "kwargs": {},
            "results": [],
        }
    ]
    edges = []
    for prefix, arguments in (("args", enumerate(args)), ("kwargs", kwargs.items())):
        for slot, arg in arguments:
            if type(arg) is dict and "value_id" in arg:
                value = values[arg["node_id"]]
                edges.append(
                    {
                        "producer_node_id": arg["node_id"],
                        "producer_value_id": arg["value_id"],
                        "consumer_node_id": "reduction",
                        "argument_path": f"{prefix}/{slot}",
                        "value_kind": "tensor",
                        "dtype": dtype,
                        "shape": value["shape"],
                    }
                )
    edges.append(
        {
            "producer_node_id": "reduction",
            "producer_value_id": "reduction:0",
            "consumer_node_id": "output",
            "argument_path": "args/0",
            "value_kind": "tensor",
            "dtype": dtype,
            "shape": values["reduction"]["shape"],
        }
    )
    graph = {
        "schema": "m2m.frontend_graph.v1",
        "stage": "original",
        "status": "complete",
        "call_count": 1,
        "operator_schemas": {target: schema},
        "nodes": nodes,
        "edges": edges,
    }
    graph["sha256"] = _digest(graph)
    observation = {
        "rows": [
            {
                "target": target,
                "schema": schema,
                "status": "observed",
                "arguments": [
                    {"name": name, "type": kind, "alias": None, "has_default": default, "kwarg_only": kwarg}
                    for name, kind, default, kwarg, _ in specs
                ],
                "returns": [{"type": "Tensor", "alias": None}],
            }
        ]
    }
    defaults = {
        "schema": "merlin.native_schema_defaults_observation.v2",
        "graph_sha256": graph["sha256"],
        "rows": [
            {
                "request": {"target": target, "schema": schema},
                "status": "observed",
                "defaults": [
                    {"ordinal": i, "name": name, "has_default": default, "default": literal(value) if default else None}
                    for i, (name, _, default, _, value) in enumerate(specs)
                ],
            }
        ],
    }
    return {"schema": "m2m.frontend_trace.v1", "graphs": {"original": graph}}, observation, defaults


def form(*args, **kwargs):
    return S.reduction_forms(*declarations(*args, **kwargs), numerical_semantics={"unqualified": True})[0]


def rehash(trace, defaults):
    graph = trace["graphs"]["original"]
    graph.pop("sha256")
    graph["sha256"] = defaults["graph_sha256"] = _digest(graph)


@pytest.mark.parametrize(
    "family,options,extent,expected,output",
    [
        ("mean", {"input_shape": (3, 2, 5), "dim": [0, -1]}, 3, [3, 4, 5], [4]),
        ("mean", {"input_shape": (2, 3, 4, 5, 2, 3), "dim": [3, 5]}, 2, [2, 3, 4, 5, 6, 7], [2, 3, 4, 6]),
        ("mean", {"dim": [2, 0], "keepdim": True}, 2, [2, 3, 4], [1, 3, 1]),
        ("mean", {"dim": None}, 2, [2, 3, 4], []),
        ("mean", {"dim": []}, 2, [2, 3, 4], []),
        ("softmax", {"input_shape": (3, 5), "dim": -1}, 3, [3, 4], [3, 4]),
        ("softmax", {"dim": 0}, 4, [4, 5, 6], [4, 5, 6]),
        ("layer_norm", {"input_shape": (5, 4)}, 3, [3, 4], [3, 4]),
        ("layer_norm", {"input_shape": (2, 7, 3, 5), "normalized_shape": [3, 5]}, 3, [3, 4, 3, 5], [3, 4, 3, 5]),
        ("layer_norm", {"input_shape": (4,), "weight": False, "bias": False}, 9, [4], [4]),
    ],
)
def test_original_axes_literals_defaults_and_full_fresh_outputs(family, options, extent, expected, output):
    original = form(family, **options)
    assert original["status"] == "supported"
    source = S.reduction_source(original, extent=extent, max_tensor_elements=100000)
    metadata = source.metadata()
    assert metadata["inputs"][0]["shape"] == expected and metadata["outputs"][0]["shape"] == output
    total = sum(math.prod(row["shape"]) for row in metadata["inputs"] + metadata["outputs"])
    assert metadata["tensor_elements"] == total and metadata["logical_payload_bytes"] == total * 4
    assert metadata["scalar_products"] == 0 and metadata["source_numerical_semantics"] == {"unqualified": True}
    forward = next(
        n for n in ast.walk(ast.parse(source.loader)) if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    operation = forward.body[0].value
    assert ast.unparse(operation.func) == "torch.ops." + _TARGETS[family]
    assert [argument.arg for argument in forward.args.args] == ["self"] + [row["name"] for row in metadata["inputs"]]
    assert ast.unparse(operation.args[0]) == "X"
    assert "admission unproved" in metadata["scope"]
    if family == "mean":
        assert ast.literal_eval(operation.args[1]) == options.get("dim", (0, -1))
        assert metadata["parameters"]["keepdim"] == options.get("keepdim", False)
    elif family == "softmax":
        assert ast.literal_eval(operation.args[1]) == options["dim"]
    else:
        assert ast.literal_eval(operation.args[1]) == list(options.get("normalized_shape", (4,)))


@pytest.mark.parametrize("weight,bias", [(True, True), (True, False), (False, True), (False, False)])
def test_optional_affine_roster_and_original_epsilon_signed_zero(weight, bias):
    original = form("layer_norm", (2, 4), weight=weight, bias=bias, eps=-0.0, cudnn_enable=False)
    source = S.reduction_source(original, extent=3, max_tensor_elements=1000)
    metadata = source.metadata()
    assert [r["name"] for r in metadata["inputs"]] == ["X"] + (["Weight"] if weight else []) + (
        ["Bias"] if bias else []
    )
    assert metadata["parameters"]["eps"].hex() == "-0x0.0p+0" and metadata["parameters"]["cudnn_enable"] is False
    tree = ast.parse(source.loader)
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "torch.ops.aten.layer_norm.default"
    )
    assert [ast.unparse(a) for a in call.args[2:]] == ["Weight" if weight else "None", "Bias" if bias else "None"]
    assert ast.literal_eval(next(k.value for k in call.keywords if k.arg == "eps")).hex() == "-0x0.0p+0"


def test_original_extents_are_not_fresh_geometry():
    first, second = form("mean", (2, 3, 4)), form("mean", (7, 5, 9))
    assert first["original_geometry"] != second["original_geometry"]
    assert S.reduction_source(first, extent=3, max_tensor_elements=1000) == S.reduction_source(
        second, extent=3, max_tensor_elements=1000
    )


@pytest.mark.parametrize(
    "family,options,extent,total",
    [
        ("mean", {"input_shape": (3,), "dim": None}, 1, 2),
        ("softmax", {"input_shape": (3, 5), "dim": -1}, 1, 4),
        ("mean", {"input_shape": (3,), "dim": None}, 30, 31),
        ("softmax", {"input_shape": (3, 5), "dim": -1}, 4, 40),
        ("layer_norm", {"input_shape": (2, 3), "normalized_shape": (3,)}, 4, 30),
    ],
)
def test_complete_logical_input_affine_and_output_budget_boundary(family, options, extent, total):
    original = form(family, **options)
    assert S.reduction_source(original, extent=extent, max_tensor_elements=total).metadata()["tensor_elements"] == total
    with pytest.raises(ValueError, match="budget"):
        S.reduction_source(original, extent=extent, max_tensor_elements=total - 1)


@pytest.mark.parametrize(
    "family,options",
    [
        ("mean", {"dim": [0, -3]}),
        ("mean", {"dim": [3]}),
        ("mean", {"dim": [True]}),
        ("mean", {"keepdim": 1}),
        ("mean", {"selected_dtype": 6}),
        ("softmax", {"dim": 3}),
        ("softmax", {"dim": -4}),
        ("softmax", {"dim": True}),
        ("softmax", {"dim": -1, "selected_dtype": 6}),
        ("layer_norm", {"input_shape": (2, 3), "normalized_shape": [4]}),
        ("layer_norm", {"normalized_shape": []}),
        ("layer_norm", {"normalized_shape": [True]}),
        ("layer_norm", {"normalized_shape": [1, 2, 3, 4]}),
        ("layer_norm", {"eps": 1}),
        ("layer_norm", {"cudnn_enable": 1}),
        ("mean", {"input_shape": (2, 0, 3)}),
    ],
)
def test_unproved_axes_storage_scalar_or_shape_cases_remain_unknown(family, options):
    original = form(family, **options)
    assert original["status"] == "unknown"
    with pytest.raises(ValueError):
        S.reduction_source(original, extent=3, max_tensor_elements=1000)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float64", "int8", "int64"])
@pytest.mark.parametrize("family", ["mean", "softmax", "layer_norm"])
def test_unsupported_original_formats_never_promote(dtype, family):
    options = {"dim": -1} if family == "softmax" else {}
    original = form(family, dtype=dtype, **options)
    assert original["status"] == "unknown" and original["result_roster"][0]["storage_dtype"] == dtype


@pytest.mark.parametrize("change", ["axes", "literal", "dtype", "rank", "layout", "geometry", "epsilon", "version"])
def test_reopening_refuses_form_scalar_type_aliases_and_shape_drift(change):
    original = form("layer_norm", (2, 4), eps=-0.0)
    if change == "axes":
        original["reduction_axes"] = [True]
    elif change == "literal":
        original["parameters"]["normalized_shape"] = [True]
    elif change == "dtype":
        original["result_dtypes"] = ["int8"]
    elif change == "rank":
        original["rank"] = 2.0
    elif change == "layout":
        original["layout_relation"] = "unknown"
    elif change == "geometry":
        original["original_geometry"]["input_strides"][0] = [1, 2]
    elif change == "epsilon":
        original["parameters"]["eps"] = 0.0
    else:
        original["form_schema"] = "merlin.original_reduction_form.v0"
    with pytest.raises(ValueError):
        S.reduction_source(original, extent=3, max_tensor_elements=1000)


@pytest.mark.parametrize("change", ["alias", "default", "order", "required", "result", "strides"])
def test_original_schema_defaults_complete_result_and_contiguity_are_required(change):
    trace, observation, defaults = declarations("softmax", dim=-1)
    if change == "alias":
        observation["rows"][0]["arguments"][0]["alias"] = {"before": ["a"], "after": ["a"], "write": False}
    elif change == "default":
        defaults["rows"][0]["defaults"][2]["default"] = literal(1)
    elif change == "order":
        observation["rows"][0]["arguments"].reverse()
    elif change == "required":
        trace["graphs"]["original"]["nodes"][1]["args"].pop()
        rehash(trace, defaults)
    elif change == "result":
        trace["graphs"]["original"]["nodes"][1]["result_arity"] = 2
        rehash(trace, defaults)
    else:
        trace["graphs"]["original"]["nodes"][0]["results"][0]["stride"] = [1, 1, 1]
        rehash(trace, defaults)
    original = S.reduction_forms(trace, observation, defaults)[0]
    assert original["status"] == "unknown"


def test_rank_and_index_overflow_refuse_before_large_geometry_or_loader_allocation(monkeypatch):
    original = form("softmax", (2, 3), dim=-1)
    huge = copy.deepcopy(original)
    huge["arguments"][0]["value"]["value"]["rank"] = 1 << 40
    huge["result_roster"][0]["rank"] = 1 << 40
    monkeypatch.setattr(S, "OriginalOperatorSource", lambda *a: pytest.fail("loader allocation before preflight"))
    with pytest.raises(ValueError, match="typed input rank"):
        S.reduction_source(huge, extent=1, max_tensor_elements=1000)
    with pytest.raises(ValueError, match="signed64 index"):
        S.reduction_source(original, extent=(1 << 63) - 1, max_tensor_elements=1 << 64)


def test_declared_rank_refuses_before_axes_expand_in_derivation_and_source(monkeypatch):
    trace, observation, defaults = declarations("mean", dim=None)
    original = S.reduction_forms(trace, observation, defaults)[0]
    contracts = S.call_contracts(trace, observation, defaults)
    contracts[0]["arguments"][0]["value"]["value"]["rank"] = 1 << 40
    monkeypatch.setattr(S, "call_contracts", lambda *a, **k: contracts)
    monkeypatch.setattr(S, "_axes", lambda *a: pytest.fail("axis expansion before complete rank binding"))
    refused = S.reduction_forms(trace, observation, defaults)[0]
    assert refused["status"] == "unknown" and "typed input rank" in refused["reason"]
    original["arguments"][0]["value"]["value"]["rank"] = 1 << 40
    with pytest.raises(ValueError, match="typed input rank"):
        S.reduction_source(original, extent=1, max_tensor_elements=1 << 64)

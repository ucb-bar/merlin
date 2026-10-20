"""Independent schema-to-source triu controls; no numerical or target grant."""

import ast
import copy

import pytest

from merlin.targetgen import original_triangular_sources as S
from merlin.targetgen.frontend_trace import _digest


def declarations(shape=(7, 9), *, dtype="float32", diagonal=None, output_shape=None):
    """A declared pure public-schema fixture, not a native observation."""
    schema = "aten::triu(Tensor self, SymInt diagonal=0) -> Tensor"
    arguments = [
        {"name": name, "type": kind, "alias": None, "kwarg_only": False, "has_default": index == 1}
        for index, (name, kind) in enumerate((("self", "Tensor"), ("diagonal", "int")))
    ]
    values = [
        {
            "id": name + ":0",
            "kind": "tensor",
            "dtype": dtype,
            "storage_dtype": dtype,
            "shape": list(dimensions),
            "layout": "torch.strided",
            "device": "cpu",
        }
        for name, dimensions in (("input", shape), ("triu", shape if output_shape is None else output_shape))
    ]

    def reference(name):
        return {"node_id": name, "value_id": name + ":0"}

    graph = {
        "schema": "m2m.frontend_graph.v1",
        "stage": "original",
        "status": "complete",
        "call_count": 1,
        "operator_schemas": {S.TARGET: schema},
        "nodes": [
            {
                "id": "input",
                "ordinal": 0,
                "op": "placeholder",
                "target": "input",
                "args": [],
                "kwargs": {},
                "results": [values[0]],
            },
            {
                "id": "triu",
                "ordinal": 1,
                "op": "call_function",
                "target": S.TARGET,
                "args": [reference("input")] if diagonal is None else [reference("input"), diagonal],
                "kwargs": {},
                "results": [values[1]],
                "result_arity": 1,
            },
            {
                "id": "output",
                "ordinal": 2,
                "op": "output",
                "target": "output",
                "args": [reference("triu")],
                "kwargs": {},
                "results": [],
            },
        ],
        "edges": [
            {
                "producer_node_id": name,
                "producer_value_id": name + ":0",
                "consumer_node_id": consumer,
                "argument_path": "args/0",
                "value_kind": "tensor",
                "dtype": dtype,
                "shape": values[index]["shape"],
            }
            for index, (name, consumer) in enumerate((("input", "triu"), ("triu", "output")))
        ],
    }
    graph["sha256"] = _digest(graph)
    observation = {
        "rows": [
            {
                "target": S.TARGET,
                "schema": schema,
                "status": "observed",
                "arguments": arguments,
                "returns": [{"type": "Tensor", "alias": None}],
            }
        ]
    }
    defaults = {
        "schema": "merlin.native_schema_defaults_observation.v2",
        "graph_sha256": graph["sha256"],
        "rows": [
            {
                "request": {"target": S.TARGET, "schema": schema},
                "status": "observed",
                "defaults": [
                    {
                        "ordinal": index,
                        "name": row["name"],
                        "has_default": row["has_default"],
                        "default": {"kind": "int", "value": 0} if index == 1 else None,
                    }
                    for index, row in enumerate(arguments)
                ],
            }
        ],
    }
    return {"schema": "m2m.frontend_trace.v1", "graphs": {"original": graph}}, observation, defaults


def form(*args, **kwargs):
    return S.triu_forms(*declarations(*args, **kwargs), numerical_semantics={"unqualified": True})[0]


@pytest.mark.parametrize("dtype,bits", [("float32", 32), ("int8", 8), ("int16", 16), ("int32", 32), ("int64", 64)])
@pytest.mark.parametrize("shape,fresh,diagonal", [((7, 9), [2, 3], 0), ((3, 7, 9), [2, 3, 4], -3)])
def test_original_rank_storage_and_last_two_axes_choose_independent_geometry(dtype, bits, shape, fresh, diagonal):
    original = form(shape, dtype=dtype, diagonal=diagonal)
    assert original["status"] == "supported"
    assert "shape" not in original["arguments"][0]["value"]["value"]
    source = S.triu_source(original, extent=2, max_tensor_elements=1000)
    metadata = source.metadata()
    count = 6 if len(shape) == 2 else 24
    assert metadata["inputs"] == [{"name": "X", "dtype": dtype, "shape": fresh}]
    assert metadata["outputs"] == [{"name": "Y", "kind": "tensor", "dtype": dtype, "shape": fresh}]
    assert metadata["matrix_axes"] == [len(shape) - 2, len(shape) - 1]
    assert metadata["shape_relation"] == "same_shape"
    assert metadata["tensor_elements"] == 2 * count
    assert metadata["logical_payload_bytes"] == 2 * count * bits // 8
    assert metadata["scalar_products"] == 0
    forward = next(
        node
        for node in ast.walk(ast.parse(source.loader))
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    operation = forward.body[0].value
    assert ast.unparse(operation.func) == "torch.ops.aten.triu.default"
    assert [ast.unparse(arg) for arg in operation.args] == ["X"]
    assert [(kw.arg, ast.literal_eval(kw.value)) for kw in operation.keywords] == [("diagonal", diagonal)]
    assert "admission unproved" in metadata["scope"]


@pytest.mark.parametrize("diagonal", [None, -(1 << 63), -9, -1, 0, 1, 9, (1 << 63) - 1])
def test_default_and_full_signed_diagonal_are_not_clamped_or_normalized(diagonal):
    original = form(diagonal=diagonal)
    expected = 0 if diagonal is None else diagonal
    assert original["arguments"][1]["binding"] == ("default" if diagonal is None else "explicit")
    assert original["parameters"] == {"diagonal": expected}
    source = S.triu_source(original, extent=1, max_tensor_elements=4)
    assert source.metadata()["parameters"] == {"diagonal": expected}
    assert f"diagonal={expected}" in source.loader


@pytest.mark.parametrize("diagonal", [True, 1.0, -(1 << 63) - 1, 1 << 63, "1"])
def test_unimplemented_scalar_kinds_or_signed64_overflow_stay_unknown(diagonal):
    assert form(diagonal=diagonal)["status"] == "unknown"


@pytest.mark.parametrize("shape", [(), (7,), (0, 3), (3, -1), (True, 3)])
def test_missing_nonpositive_or_low_rank_shapes_stay_unknown(shape):
    assert form(shape)["status"] == "unknown"


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float64", "uint8", "bool", "complex64"])
def test_unsupported_storage_is_retained_without_promotion(dtype):
    assert form(dtype=dtype)["status"] == "unknown"


def test_original_same_shape_relation_is_required_before_fresh_geometry():
    assert form(output_shape=(9, 7))["status"] == "unknown"
    assert form(output_shape=(7, 9, 1))["status"] == "unknown"
    first = form((7, 9))
    second = form((11, 13))
    assert S.triu_source(first, extent=2, max_tensor_elements=12) == S.triu_source(
        second, extent=2, max_tensor_elements=12
    )


@pytest.mark.parametrize(
    "defect",
    [
        "rank",
        "axes",
        "shape_relation",
        "parameters",
        "result_storage",
        "input_storage",
        "input_alias",
        "result_alias",
        "argument_type",
        "argument_order",
        "result_count",
        "result_kind",
        "dtype_list",
    ],
)
def test_factory_reopens_complete_original_arguments_types_and_result_roster(defect):
    original = form()
    if defect == "rank":
        original["rank"] = True
    elif defect == "axes":
        original["matrix_axes"] = [True, 0]
    elif defect == "shape_relation":
        original["shape_relation"] = "broadcast"
    elif defect == "parameters":
        original["parameters"]["diagonal"] = True
    elif defect == "result_storage":
        original["result_roster"][0]["storage_dtype"] = "int32"
    elif defect == "input_storage":
        original["arguments"][0]["value"]["value"]["storage_dtype"] = "float64"
    elif defect == "input_alias":
        original["arguments"][0]["alias"] = {"write": True}
    elif defect == "result_alias":
        original["schema_returns"][0]["alias"] = {"before": ["a"], "after": ["a"], "write": False}
    elif defect == "argument_type":
        original["arguments"][1]["type"] = "number"
    elif defect == "argument_order":
        original["arguments"].reverse()
    elif defect == "result_count":
        original["result_arity"] = True
    elif defect == "result_kind":
        original["result_roster"][0]["kind"] = "scalar"
    else:
        original["operand_dtypes"] = ["int8"]
    with pytest.raises(ValueError):
        S.triu_source(original, extent=2, max_tensor_elements=100)


def test_complete_logical_counts_and_rank_metadata_are_bounded_before_allocation():
    for original, extent, maximum in ((form(), 2, 11), (form((3, 7, 9)), 2, 47), (form(), 10**100, 100)):
        with pytest.raises(ValueError, match="before geometry allocation"):
            S.triu_source(original, extent=extent, max_tensor_elements=maximum)
    original = form()
    huge = copy.deepcopy(original)
    huge["rank"] = huge["arguments"][0]["value"]["value"]["rank"] = huge["result_roster"][0]["rank"] = 10**100
    with pytest.raises(ValueError, match="rank metadata"):
        S.triu_source(huge, extent=1, max_tensor_elements=100)
    assert S.triu_source(original, extent=2, max_tensor_elements=12).metadata()["tensor_elements"] == 12
    for key in ("extent", "max_tensor_elements"):
        selected = {"extent": 2, "max_tensor_elements": 100, key: True}
        with pytest.raises(ValueError):
            S.triu_source(original, **selected)


def test_policy_is_copied_without_becoming_a_numeric_or_effect_admission():
    original = form()
    source = S.triu_source(original, extent=2, max_tensor_elements=12)
    original["source_numerical_semantics"]["unqualified"] = False
    original["parameters"]["diagonal"] = 7
    assert source.metadata()["source_numerical_semantics"] == {"unqualified": True}
    assert source.metadata()["parameters"] == {"diagonal": 0}

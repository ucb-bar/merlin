"""Pure original reshape schema/source controls; no framework or alias grant."""

import ast
import copy
import math

import pytest

from merlin.targetgen import original_reshape_sources as S
from merlin.targetgen.frontend_trace import _digest


def strides(shape):
    result, stride = [], 1
    for dimension in reversed(shape):
        result.append(stride)
        stride *= dimension
    return list(reversed(result))


def declarations(input_shape=(6, 4), shape=(4, 6), *, dtype="float32", output_shape=None):
    """Independently declared schema fixture; its rows are not observations."""
    schema = "aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)"
    alias = {"before": ["a"], "after": ["a"], "write": False}
    arguments = [
        {"name": name, "type": kind, "alias": copy.deepcopy(binding), "kwarg_only": False, "has_default": False}
        for name, kind, binding in (("self", "Tensor", alias), ("shape", "List[int]", None))
    ]
    inferred = list(shape)
    if output_shape is None and inferred.count(-1) == 1 and all(d == -1 or (type(d) is int and d > 0) for d in shape):
        product = math.prod(d for d in shape if d != -1)
        inferred[inferred.index(-1)] = math.prod(input_shape) // product
    output_shape = inferred if output_shape is None else output_shape
    values = [
        {
            "id": name + ":0",
            "kind": "tensor",
            "dtype": dtype,
            "storage_dtype": dtype,
            "shape": list(dimensions),
            "stride": strides(dimensions),
            "layout": "torch.strided",
            "device": "cpu",
        }
        for name, dimensions in (("input", input_shape), ("reshape", output_shape))
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
                "id": "reshape",
                "ordinal": 1,
                "op": "call_function",
                "target": S.TARGET,
                "args": [reference("input"), list(shape)],
                "kwargs": {},
                "results": [values[1]],
                "result_arity": 1,
            },
            {
                "id": "output",
                "ordinal": 2,
                "op": "output",
                "target": "output",
                "args": [reference("reshape")],
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
            for index, (name, consumer) in enumerate((("input", "reshape"), ("reshape", "output")))
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
                "returns": [{"type": "Tensor", "alias": copy.deepcopy(alias)}],
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
                    {"ordinal": index, "name": row["name"], "has_default": False, "default": None}
                    for index, row in enumerate(arguments)
                ],
            }
        ],
    }
    return {"schema": "m2m.frontend_trace.v1", "graphs": {"original": graph}}, observation, defaults


def form(*args, **kwargs):
    return S.reshape_forms(*declarations(*args, **kwargs), numerical_semantics={"unqualified": True})[0]


def rehash(graph, defaults):
    graph.pop("sha256")
    graph["sha256"] = defaults["graph_sha256"] = _digest(graph)


@pytest.mark.parametrize("dtype,bits", [("float32", 32), ("int8", 8), ("int16", 16), ("int32", 32), ("int64", 64)])
@pytest.mark.parametrize(
    "input_shape,literal,extent,fresh,output",
    [
        ((6, 4), (4, 6), 2, [2, 12], [4, 6]),
        ((2, 3, 4), (3, -1), 2, [2, 3, 1], [3, 2]),
        ((17,), (-1,), 5, [5], [5]),
        ((5, 7, 3), (7, 15), 5, [5, 3, 7], [7, 15]),
    ],
)
def test_exact_literal_rank_storage_and_complete_fresh_product(
    dtype, bits, input_shape, literal, extent, fresh, output
):
    original = form(input_shape, literal, dtype=dtype)
    assert original["status"] == "supported"
    source = S.reshape_source(original, extent=extent, max_tensor_elements=1000)
    metadata = source.metadata()
    assert metadata["inputs"] == [{"name": "X", "dtype": dtype, "shape": fresh}]
    assert metadata["outputs"] == [{"name": "Y", "kind": "tensor", "dtype": dtype, "shape": output}]
    assert metadata["tensor_elements"] == 2 * math.prod(fresh) == 2 * math.prod(output)
    assert metadata["logical_payload_bytes"] == 2 * math.prod(fresh) * bits // 8
    assert metadata["scalar_products"] == 0
    assert metadata["parameters"] == {"shape": list(literal)}
    assert metadata["schema_aliases"] == {
        "self": {"before": ["a"], "after": ["a"], "write": False},
        "result": {"before": ["a"], "after": ["a"], "write": False},
    }
    assert metadata["geometry_scope"] == (
        "unique_positive_minus_one_inference" if -1 in literal else "fixed_literal_product_factorization"
    )
    assert "admission unproved" in metadata["scope"]
    forward = next(
        node
        for node in ast.walk(ast.parse(source.loader))
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    operation = forward.body[0].value
    assert ast.unparse(operation.func) == "torch.ops.aten.reshape.default"
    assert [ast.unparse(arg) for arg in operation.args[:1]] == ["X"]
    assert ast.literal_eval(operation.args[1]) == list(literal)
    assert operation.keywords == []


def test_original_sizes_are_only_layout_witnesses_not_fresh_input_geometry():
    first = form((6, 4), (4, 6))
    second = form((3, 8), (4, 6))
    assert first["original_geometry"] != second["original_geometry"]
    assert S.reshape_source(first, extent=2, max_tensor_elements=48) == S.reshape_source(
        second, extent=2, max_tensor_elements=48
    )
    assert S.reshape_source(first, extent=3, max_tensor_elements=48).metadata()["inputs"][0]["shape"] == [3, 8]
    assert (
        S.reshape_source(first, extent=2, max_tensor_elements=48).metadata()["geometry_scope"]
        == "fixed_literal_product_factorization"
    )


@pytest.mark.parametrize(
    "input_shape,shape",
    [
        ((6, 4), ()),
        ((0, 4), (0, 4)),
        ((6, 4), (0, -1)),
        ((6, 4), (-1, -1)),
        ((6, 4), (-2, 12)),
        ((6, 4), (True, 24)),
        ((6, 4), (2.0, 12)),
        ((6, 4), (5, -1)),
        ((6, 4), (5, 5)),
        ((), (1,)),
        (((1 << 62), 4), (-1,)),
        ((6, 4), ((1 << 62), 4)),
        ((True, 24), (-1,)),
    ],
)
def test_unimplemented_or_illegal_original_shape_domain_stays_unknown(input_shape, shape):
    assert form(input_shape, shape)["status"] == "unknown"


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float64", "uint8", "bool", "complex64"])
def test_unsupported_storage_is_retained_without_promotion(dtype):
    assert form(dtype=dtype)["status"] == "unknown"


@pytest.mark.parametrize(
    "field,value", [("stride", None), ("stride", [1, 6]), ("stride", [4]), ("stride", [4, True]), ("shape", [6, 5])]
)
@pytest.mark.parametrize("index", [0, 1])
def test_complete_original_shape_and_contiguous_stride_premises_are_required(field, value, index):
    trace, schema, defaults = declarations()
    graph = trace["graphs"]["original"]
    graph["nodes"][index]["results"][0][field] = value
    if field == "shape":
        graph["edges"][index]["shape"] = value
    rehash(graph, defaults)
    assert S.reshape_forms(trace, schema, defaults)[0]["status"] == "unknown"


def test_positive_singleton_stride_does_not_change_contiguous_address_map():
    trace, schema, defaults = declarations((1, 24), (1, 4, 6))
    graph = trace["graphs"]["original"]
    graph["nodes"][0]["results"][0]["stride"][0] = 999
    graph["nodes"][1]["results"][0]["stride"][0] = 42
    rehash(graph, defaults)
    original = S.reshape_forms(trace, schema, defaults)[0]
    assert original["status"] == "supported"
    assert S.reshape_source(original, extent=2, max_tensor_elements=48).metadata()["outputs"][0]["shape"] == [1, 4, 6]


@pytest.mark.parametrize(
    "defect",
    [
        "literal",
        "parameters",
        "schema",
        "input_alias",
        "result_alias",
        "argument_order",
        "argument_type",
        "argument_ordinal",
        "argument_path",
        "omitted",
        "input_storage",
        "result_storage",
        "result_rank",
        "result_count",
        "dtype_list",
        "layout",
        "relation",
        "witness",
        "witness_stride",
        "witness_output",
    ],
)
def test_factory_reopens_exact_call_and_original_shape_stride_relation(defect):
    original = form()
    if defect == "literal":
        original["arguments"][1]["value"]["items"][0]["value"] = 3
    elif defect == "parameters":
        original["parameters"]["shape"][0] = 3
    elif defect == "schema":
        original["schema"] = original["schema"].replace("(a)", "")
    elif defect == "input_alias":
        original["arguments"][0]["alias"]["write"] = True
    elif defect == "result_alias":
        original["schema_returns"][0]["alias"]["write"] = 0
    elif defect == "argument_order":
        original["arguments"].reverse()
    elif defect == "argument_type":
        original["arguments"][1]["type"] = "Tensor"
    elif defect == "argument_ordinal":
        original["arguments"][0]["ordinal"] = False
    elif defect == "argument_path":
        original["arguments"][1]["path"] = "args/0"
    elif defect == "omitted":
        original["arguments"][1]["binding"] = "default"
    elif defect == "input_storage":
        original["arguments"][0]["value"]["value"]["storage_dtype"] = "int32"
    elif defect == "result_storage":
        original["result_roster"][0]["storage_dtype"] = "int32"
    elif defect == "result_rank":
        original["result_roster"][0]["rank"] = True
    elif defect == "result_count":
        original["result_arity"] = True
    elif defect == "dtype_list":
        original["operand_dtypes"] = ["int8"]
    elif defect == "layout":
        original["layout_relation"] = "unknown"
    elif defect == "relation":
        original["shape_relation"] = "broadcast"
    elif defect == "witness":
        original["original_geometry"].pop("input_strides")
    elif defect == "witness_stride":
        original["original_geometry"]["input_strides"] = [1, 6]
    else:
        original["original_geometry"]["output_shape"] = [6, 4]
    with pytest.raises((ValueError, KeyError)):
        S.reshape_source(original, extent=2, max_tensor_elements=1000)


@pytest.mark.parametrize("input_shape,literal,extent", [((6, 4), (4, 6), 5), ((24,), (4, 6), 2), ((6,), (2, -1), 3)])
def test_impossible_requested_fresh_leading_extent_remains_unavailable(input_shape, literal, extent):
    with pytest.raises(ValueError, match="requested fresh leading extent"):
        S.reshape_source(form(input_shape, literal), extent=extent, max_tensor_elements=1000)


def test_complete_elements_rank_and_index_domains_are_bounded_before_geometry_allocation():
    original = form()
    assert S.reshape_source(original, extent=2, max_tensor_elements=48).metadata()["tensor_elements"] == 48
    with pytest.raises(ValueError, match="complete tensor-element budget before geometry allocation"):
        S.reshape_source(original, extent=2, max_tensor_elements=47)
    huge = copy.deepcopy(original)
    huge["rank"] = huge["arguments"][0]["value"]["value"]["rank"] = 10**100
    with pytest.raises(ValueError, match="rank/stride metadata"):
        S.reshape_source(huge, extent=1, max_tensor_elements=1000)
    inferred = form((1, 2), (2, -1))
    with pytest.raises(ValueError, match="signed64 index"):
        S.reshape_source(inferred, extent=(1 << 63) - 1, max_tensor_elements=1 << 70)
    for selected in (
        {"extent": True, "max_tensor_elements": 1000},
        {"extent": 2, "max_tensor_elements": True},
        {"extent": 1 << 63, "max_tensor_elements": 1000},
    ):
        with pytest.raises(ValueError):
            S.reshape_source(original, **selected)


def test_policy_and_alias_declarations_are_copied_without_granting_observed_effects():
    original = form()
    source = S.reshape_source(original, extent=2, max_tensor_elements=48)
    original["source_numerical_semantics"]["unqualified"] = False
    original["arguments"][0]["alias"]["write"] = True
    original["parameters"]["shape"][0] = 3
    assert source.metadata()["source_numerical_semantics"] == {"unqualified": True}
    assert source.metadata()["parameters"] == {"shape": [4, 6]}
    assert source.metadata()["schema_aliases"]["self"]["write"] is False

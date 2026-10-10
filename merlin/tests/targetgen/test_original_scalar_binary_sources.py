"""Minimal declared scalar calls exercise construction without source admission.

The pure schema observations below are independent fixtures, not native schema,
conversion, numerical ownership or original protected coverage evidence.
"""

import ast
import copy

import pytest

from merlin.targetgen import original_scalar_binary_sources as S
from merlin.targetgen.frontend_trace import _digest
from merlin.targetgen.original_operator_sources import policy_compatibility


def declarations(target="aten.mul.Tensor", scalar=1.0, *, shape=(7, 9), dtype="float32"):
    schema = target.replace("aten.", "aten::", 1) + "(Tensor self, Tensor other) -> Tensor"
    arguments = [
        {"name": name, "type": "Tensor", "alias": None, "kwarg_only": False, "has_default": False}
        for name in ("self", "other")
    ]
    tensor = {
        "id": "input:0",
        "kind": "tensor",
        "dtype": dtype,
        "storage_dtype": dtype,
        "shape": list(shape),
        "layout": "torch.strided",
        "device": "cpu",
    }
    result = {**tensor, "id": "binary:0"}
    reference = {"node_id": "input", "value_id": "input:0"}
    graph = {
        "schema": "m2m.frontend_graph.v1",
        "stage": "original",
        "status": "complete",
        "call_count": 1,
        "operator_schemas": {target: schema},
        "nodes": [
            {
                "id": "input",
                "ordinal": 0,
                "op": "placeholder",
                "target": "X",
                "args": [],
                "kwargs": {},
                "results": [tensor],
            },
            {
                "id": "binary",
                "ordinal": 1,
                "op": "call_function",
                "target": target,
                "args": [reference, scalar],
                "kwargs": {},
                "results": [result],
                "result_arity": 1,
            },
            {
                "id": "output",
                "ordinal": 2,
                "op": "output",
                "target": "output",
                "args": [{"node_id": "binary", "value_id": "binary:0"}],
                "kwargs": {},
                "results": [],
            },
        ],
        "edges": [
            {
                "producer_node_id": producer,
                "producer_value_id": producer + ":0",
                "consumer_node_id": consumer,
                "argument_path": "args/0",
                "value_kind": "tensor",
                "dtype": dtype,
                "shape": list(shape),
            }
            for producer, consumer in (("input", "binary"), ("binary", "output"))
        ],
    }
    graph["sha256"] = _digest(graph)
    observation = {
        "rows": [
            {
                "target": target,
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
                "request": {"target": target, "schema": schema},
                "status": "observed",
                "defaults": [
                    {"ordinal": index, "name": name, "has_default": False, "default": None}
                    for index, name in enumerate(("self", "other"))
                ],
            }
        ],
    }
    return {"schema": "m2m.frontend_trace.v1", "graphs": {"original": graph}}, observation, defaults


def form(target="aten.mul.Tensor", scalar=1.0, **kwargs):
    trace, schemas, defaults = declarations(target, scalar, **kwargs)
    rows = S.scalar_binary_forms(trace, schemas, defaults, numerical_semantics={"pending_original_policy": True})
    assert len(rows) == 1 and rows[0]["status"] == "supported"
    return rows[0]


@pytest.mark.parametrize("target,scalar", [("aten.mul.Tensor", 1.0), ("aten.div.Tensor", 2.23606797749979)])
def test_minimal_original_float_literal_call_has_one_input_and_complete_typed_result(target, scalar):
    original = form(target, scalar)
    source = S.scalar_binary_source(original, extent=2, max_tensor_elements=100)
    metadata = source.metadata()
    assert metadata["inputs"] == [{"name": "X", "dtype": "float32", "shape": [2, 3]}]
    assert metadata["outputs"] == [{"name": "Y", "kind": "tensor", "dtype": "float32", "shape": [2, 3]}]
    assert metadata["tensor_elements"] == 12 and metadata["logical_payload_bytes"] == 48
    assert metadata["scalar_products"] == 6
    assert metadata["parameters"]["other"] == {"kind": "float", "value_hex": scalar.hex()}
    assert metadata["source_numerical_semantics"] == original["source_numerical_semantics"]
    tree = ast.parse(source.loader)
    model = next(row for row in tree.body if isinstance(row, ast.ClassDef))
    forward = next(row for row in model.body if isinstance(row, ast.FunctionDef))
    assert [row.arg for row in forward.args.args] == ["self", "X"]
    call = forward.body[0].value
    assert isinstance(call, ast.Call) and len(call.args) == 2
    assert isinstance(call.args[0], ast.Name) and call.args[0].id == "X"
    assert isinstance(call.args[1], ast.Constant) and type(call.args[1].value) is float
    assert call.args[1].value.hex() == scalar.hex()
    assert "torch.ops." + target + "(X, " + repr(scalar) + ")" in source.loader
    assert policy_compatibility(original, {"model": {"engine": "integer_reference"}})["status"] == "unknown"


@pytest.mark.parametrize("scalar", [0.0, -0.0, 0.1, -3.5])
def test_float_literal_kind_and_signed_zero_survive_without_tensorization_or_pre_rounding(scalar):
    original = form(scalar=scalar)
    source = S.scalar_binary_source(original, extent=1, max_tensor_elements=4)
    assert source.metadata()["parameters"]["other"]["value_hex"] == scalar.hex()
    assert "X, " + repr(scalar) + ")" in source.loader
    assert "torch.tensor" not in source.loader and "torch.float64" not in source.loader


@pytest.mark.parametrize("scalar", [True, 1, None, "1.0", [1.0]])
def test_unimplemented_literal_kinds_remain_required_unknown_forms(scalar):
    rows = S.scalar_binary_forms(*declarations(scalar=scalar))
    assert len(rows) == 1 and rows[0]["status"] == "unknown"


@pytest.mark.parametrize(
    "change",
    [
        "literal",
        "rank",
        "storage",
        "promotion",
        "second_ssa",
        "reversed",
        "argument_schema",
        "alias",
        "result_roster",
        "result_rank",
        "dtype",
        "default",
        "arity_bool",
    ],
)
def test_original_bindings_types_order_and_complete_result_roster_cannot_be_substituted(change):
    original = form()
    if change == "literal":
        original["parameters"]["other"]["value_hex"] = (-1.0).hex()
    elif change == "rank":
        original["rank"] = 1
    elif change == "storage":
        original["arguments"][0]["value"]["value"]["storage_dtype"] = "bfloat16"
    elif change == "promotion":
        original["result_roster"][0]["dtype"] = "float64"
    elif change == "second_ssa":
        original["arguments"][1]["value"] = copy.deepcopy(original["arguments"][0]["value"])
    elif change == "reversed":
        original["arguments"].reverse()
    elif change == "argument_schema":
        original["arguments"][1]["type"] = "number"
    elif change == "alias":
        original["schema_returns"][0]["alias"] = {"may_alias": True}
    elif change == "result_roster":
        original["result_roster"].append(copy.deepcopy(original["result_roster"][0]))
    elif change == "result_rank":
        original["result_roster"][0]["rank"] = True
    elif change == "dtype":
        original["operand_dtypes"] = ["float64"]
    elif change == "default":
        original["arguments"][1]["binding"] = "default"
    else:
        original["result_arity"] = True
    with pytest.raises(ValueError):
        S.scalar_binary_source(original, extent=2, max_tensor_elements=100)


@pytest.mark.parametrize("rank,extent,limit", [(2, 2, 11), (1000000000, 1, 100), (2, 1000000000, 100)])
def test_complete_logical_payload_and_rank_budgets_precede_geometry_allocation(rank, extent, limit):
    original = form()
    original["rank"] = original["arguments"][0]["value"]["value"]["rank"] = rank
    original["result_roster"][0]["rank"] = rank
    with pytest.raises(ValueError, match="before allocation"):
        S.scalar_binary_source(original, extent=extent, max_tensor_elements=limit)


@pytest.mark.parametrize("field", ["extent", "max_tensor_elements"])
def test_boolean_cannot_replace_an_explicit_positive_integer_budget(field):
    supplied = {"extent": 2, "max_tensor_elements": 100, field: True}
    with pytest.raises(ValueError, match="positive geometry/budget"):
        S.scalar_binary_source(form(), **supplied)


def test_source_freezes_policy_literal_and_fresh_geometry_never_copies_original_extents():
    original = form(shape=(10007, 10009))
    source = S.scalar_binary_source(original, extent=1, max_tensor_elements=4)
    original["parameters"]["other"]["value_hex"] = (0.5).hex()
    original["source_numerical_semantics"]["pending_original_policy"] = False
    metadata = source.metadata()
    assert metadata["inputs"][0]["shape"] == [1, 2]
    assert metadata["parameters"]["other"]["value_hex"] == (1.0).hex()
    assert metadata["source_numerical_semantics"] == {"pending_original_policy": True}

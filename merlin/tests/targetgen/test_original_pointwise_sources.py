"""Exact original unary bindings construct sources without numerical authority."""

import pytest

from merlin.targetgen import original_pointwise_sources as P
from merlin.targetgen.frontend_original_call import _literal


def form(target="aten.relu.default", *, dtype="float32", rank=2, bounds=(-0.0, 3.5)):
    original = {
        "id": "x",
        "kind": "tensor",
        "dtype": dtype,
        "storage_dtype": dtype,
        "rank": rank,
        "layout": "torch.strided",
        "device": "cpu",
    }
    arguments = [
        {
            "name": "self",
            "type": "Tensor",
            "alias": None,
            "value": {"kind": "ssa", "node_id": "input", "value": original},
        }
    ]
    parameters = {}
    if target == "aten.clamp.default":
        for name, value in zip(("min", "max"), bounds, strict=True):
            arguments.append({"name": name, "type": "Optional[number]", "alias": None, "value": _literal(value)})
            parameters[name] = value
    return {
        "form_schema": P.FORM_SCHEMA,
        "status": "supported",
        "target": target,
        "arguments": arguments,
        "result_arity": 1,
        "schema_returns": [{"type": "Tensor", "alias": None}],
        "result_roster": [{**original, "id": "y"}],
        "rank": rank,
        "operand_dtypes": [dtype],
        "result_dtypes": [dtype],
        "parameters": parameters,
        "source_numerical_semantics": {"deliberately_unqualified": True},
    }


@pytest.mark.parametrize("target", ["aten.relu.default", "aten.round.default", "aten.clamp.default"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "float64", "int8"])
def test_fresh_rectangular_geometry_preserves_original_storage_and_bounds(target, dtype):
    original = form(target, dtype=dtype)
    source = P.pointwise_source(original, extent=2, max_tensor_elements=100)
    metadata = source.metadata()
    assert metadata["inputs"] == [{"name": "X", "dtype": dtype, "shape": [2, 3]}]
    assert metadata["outputs"] == [{"name": "Y", "kind": "tensor", "dtype": dtype, "shape": [2, 3]}]
    assert metadata["tensor_elements"] == 12 and metadata["scalar_products"] == 0
    assert metadata["source_numerical_semantics"] == original["source_numerical_semantics"]
    assert metadata["parameters"] == original["parameters"]
    assert "admission unproved" in metadata["scope"]
    if target == "aten.clamp.default":
        assert "X, -0.0, 3.5" in source.loader


@pytest.mark.parametrize("change", ["dtype", "rank", "alias", "scalar", "result_kind", "missing_result", "bound_kind"])
def test_source_cannot_change_original_types_arguments_alias_or_result_roster(change):
    original = form("aten.clamp.default")
    if change == "dtype":
        original["result_dtypes"] = ["int8"]
    elif change == "rank":
        original["rank"] = 1
    elif change == "alias":
        original["schema_returns"][0]["alias"] = {"may_alias": True}
    elif change == "scalar":
        original["parameters"]["min"] = 0.0
    elif change == "result_kind":
        original["result_roster"][0]["kind"] = "scalar"
    elif change == "missing_result":
        original["result_roster"] = []
    else:
        original["arguments"][1]["value"] = {"kind": "bool", "value": True}
    with pytest.raises(ValueError):
        P.pointwise_source(original, extent=2, max_tensor_elements=100)


@pytest.mark.parametrize("rank,limit", [(0, 1), (2, 11), (10**9, 100)])
def test_complete_input_and_output_budget_precedes_geometry_and_loader_allocation(rank, limit):
    with pytest.raises(ValueError, match="before allocation"):
        P.pointwise_source(form(rank=rank), extent=2, max_tensor_elements=limit)


def test_scalar_rank_has_complete_one_element_input_and_output():
    metadata = P.pointwise_source(form(rank=0), extent=17, max_tensor_elements=2).metadata()
    assert metadata["inputs"][0]["shape"] == [] and metadata["outputs"][0]["shape"] == []
    assert metadata["tensor_elements"] == 2


def test_source_freezes_caller_policy_and_parameters():
    original = form("aten.clamp.default")
    source = P.pointwise_source(original, extent=1, max_tensor_elements=100)
    original["parameters"]["max"] = 100
    original["source_numerical_semantics"]["deliberately_unqualified"] = False
    assert source.metadata()["parameters"]["max"] == 3.5
    assert source.metadata()["source_numerical_semantics"]["deliberately_unqualified"] is True


def test_rank_and_budget_require_exact_positive_integer_selections():
    original = form(rank=True)
    with pytest.raises(ValueError, match="static Tensor rank"):
        P.pointwise_source(original, extent=2, max_tensor_elements=100)
    for key in ("extent", "max_tensor_elements"):
        selected = {"extent": 2, "max_tensor_elements": 100, key: True}
        with pytest.raises(ValueError, match="positive geometry/budget"):
            P.pointwise_source(form(), **selected)


def test_bounds_keep_none_and_inverted_order_without_inventing_numeric_semantics():
    for bounds in ((None, 7), (7, None), (7, -3)):
        original = form("aten.clamp.default", bounds=bounds)
        assert P.pointwise_source(original, extent=1, max_tensor_elements=100).metadata()["parameters"] == dict(
            zip(("min", "max"), bounds, strict=True)
        )
    with pytest.raises(ValueError, match="lower or upper bound"):
        P.pointwise_source(form("aten.clamp.default", bounds=(None, None)), extent=1, max_tensor_elements=100)

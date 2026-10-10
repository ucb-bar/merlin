"""Independent schema fixtures check construction, never native or semantic admission."""

import ast
import copy

import pytest
from test_original_scalar_binary_sources import declarations as binary_declarations

from merlin.targetgen import original_metadata_sources as M
from merlin.targetgen.frontend_operator_effects import original_zero_return_requests
from merlin.targetgen.frontend_trace import _digest
from merlin.targetgen.frontend_typed_add import defaults_request


def declarations(target=M.CAST, *, dtype="int8", result_dtype="float32", shape=(101, 103), conditions=None):
    trace, observation, defaults = binary_declarations(target, shape=shape, dtype=dtype)
    graph = trace["graphs"]["original"]
    call = graph["nodes"][1]
    cast = target == M.CAST
    names = (
        ["self", "dtype", "non_blocking", "copy", "memory_format"]
        if cast
        else ["a", "size", "stride", "dtype", "device", "layout"]
    )
    types = (
        ["Tensor", "int", "bool", "bool", "Optional[int]"]
        if cast
        else [
            "Tensor",
            "Optional[List[int]]",
            "Optional[List[int]]",
            "Optional[int]",
            "Optional[Device]",
            "Optional[int]",
        ]
    )
    schema = (
        (
            "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, "
            "MemoryFormat? memory_format=None) -> Tensor(a)"
        )
        if cast
        else (
            "aten::_assert_tensor_metadata(Tensor a, SymInt[]? size=None, SymInt[]? stride=None, "
            "ScalarType? dtype=None, *, Device? device=None, Layout? layout=None) -> ()"
        )
    )
    arguments = [
        {
            "name": name,
            "type": kind,
            "alias": copy.deepcopy(M._ALIAS) if cast and i == 0 else None,
            "kwarg_only": not cast and i >= 4,
            "has_default": i >= (2 if cast else 1),
        }
        for i, (name, kind) in enumerate(zip(names, types, strict=True))
    ]
    returns = [{"type": "Tensor", "alias": copy.deepcopy(M._ALIAS)}] if cast else []
    observation["rows"][0].update(schema=schema, arguments=arguments, returns=returns)
    graph["operator_schemas"][target] = schema
    reference = copy.deepcopy(call["args"][0])
    call.update(args=[reference], kwargs={} if cast else {"dtype": {"kind": "dtype", "value": "torch." + dtype}})
    if cast:
        call["args"].append({"kind": "dtype", "value": "torch." + result_dtype})
        call["results"][0].update(dtype=result_dtype, storage_dtype=result_dtype)
        graph["edges"][1]["dtype"] = result_dtype
    else:
        call["results"] = [
            {
                "id": "binary:0",
                "kind": "unknown",
                **dict.fromkeys(("shape", "dtype", "storage_dtype", "compute_dtype", "device", "layout", "stride")),
            }
        ]
        call["result_metadata"] = {
            "schema": "m2m.frontend_result_metadata.v1",
            "status": "observed",
            "container": "single",
            "values": [{"result_id": "binary:0", "kind": "none"}],
        }
        graph["edges"][1].update(value_kind="unknown", dtype=None, shape=None)
    if conditions:
        call["kwargs"].update(copy.deepcopy(conditions))
    graph["sha256"] = _digest({key: value for key, value in graph.items() if key != "sha256"})
    request = defaults_request(trace, observation, version=2)
    defaults.update(
        graph_sha256=graph["sha256"],
        rows=[
            {
                "request": request["rows"][0],
                "status": "observed",
                "defaults": [
                    {
                        "ordinal": i,
                        "name": row["name"],
                        "has_default": row["has_default"],
                        "default": {"kind": "bool", "value": False}
                        if cast and i in {2, 3}
                        else {"kind": "none"}
                        if row["has_default"]
                        else None,
                    }
                    for i, row in enumerate(arguments)
                ],
            }
        ],
    )
    zero_request = original_zero_return_requests(trace, observation)
    zero = {
        "schema": "merlin.native_zero_return_observation.v1",
        "graph_sha256": graph["sha256"],
        "rows": [
            {
                "request": row,
                "status": "observed",
                "native": {"schema": schema, "return_count": 0, "empty_stack_is_none": True},
            }
            for row in zero_request["rows"]
        ],
        "runtime": {},
        "scope": "pure diagnostic substitute, not native evidence",
    }
    return trace, observation, defaults, zero


def form(target=M.CAST, **options):
    trace, schema, defaults, zero = declarations(target, **options)
    return M.metadata_forms(
        trace, schema, defaults, numerical_semantics={"pending_original_policy": True}, zero_returns=zero
    )[0]


@pytest.mark.parametrize("shape", [[101, 104], None, []])
def test_same_rank_different_original_cast_result_shape_is_not_rewritten_as_equal(shape):
    trace, schema, defaults, zero = declarations()
    graph = trace["graphs"]["original"]
    graph["nodes"][1]["results"][0]["shape"] = shape
    graph["edges"][1]["shape"] = shape
    graph["sha256"] = _digest({key: value for key, value in graph.items() if key != "sha256"})
    defaults["graph_sha256"] = graph["sha256"]
    zero["graph_sha256"] = graph["sha256"]
    original = M.metadata_forms(trace, schema, defaults, zero_returns=zero)[0]
    assert original["status"] == "unknown"
    with pytest.raises(ValueError):
        M.metadata_source(original, extent=2, max_tensor_elements=12)


@pytest.mark.parametrize("change", ["missing", "operand", "result", "rank", "scalar_alias"])
def test_cast_source_rechecks_the_bound_shape_relation_before_construction(change):
    original = form()
    assert original["shape_relation"] == {
        "kind": "equal",
        "operand": original["arguments"][0]["value"]["value"]["id"],
        "result": original["result_roster"][0]["id"],
        "rank": 2,
    }
    if change == "missing":
        original.pop("shape_relation")
    elif change == "scalar_alias":
        original["shape_relation"]["rank"] = 2.0
    else:
        original["shape_relation"][change] = 3 if change == "rank" else "other"
    with pytest.raises(ValueError, match="equal-shape relation"):
        M.metadata_source(original, extent=2, max_tensor_elements=12)


@pytest.mark.parametrize(
    "input_dtype,output_dtype",
    [("int8", "int32"), ("float32", "float16"), ("int64", "float32"), ("float32", "float32")],
)
@pytest.mark.parametrize("flags", [(False, False), (True, False), (False, True), (True, True)])
def test_cast_preserves_actual_original_storage_flags_and_fresh_geometry(input_dtype, output_dtype, flags):
    original = form(
        dtype=input_dtype, result_dtype=output_dtype, conditions=dict(zip(("non_blocking", "copy"), flags, strict=True))
    )
    assert original["status"] == "supported"
    source = M.metadata_source(original, extent=2, max_tensor_elements=12)
    metadata = source.metadata()
    assert metadata["inputs"] == [{"name": "X", "dtype": input_dtype, "shape": [2, 3]}]
    assert metadata["outputs"] == [{"name": "Y", "kind": "tensor", "dtype": output_dtype, "shape": [2, 3]}]
    assert metadata["parameters"] == {
        "dtype": "torch." + output_dtype,
        "non_blocking": flags[0],
        "copy": flags[1],
        "memory_format": None,
    }
    assert metadata["logical_payload_bytes"] == 6 * (M._DTYPES[input_dtype] + M._DTYPES[output_dtype]) // 8
    assert metadata["schema_alias"] == M._ALIAS and metadata["dispatcher_result_count"] == 1
    assert metadata["source_numerical_semantics"] == {"pending_original_policy": True}
    tree = ast.parse(source.loader)
    forward = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "forward")
    assert isinstance(forward.body[0], ast.Return) and isinstance(forward.body[0].value, ast.Call)
    assert {row.arg for row in forward.body[0].value.keywords} == set(metadata["parameters"])


@pytest.mark.parametrize(
    "conditions",
    [
        {},
        {"device": {"kind": "device", "value": "cpu"}},
        {"layout": {"kind": "layout", "value": "torch.strided"}},
        {"dtype": None},
    ],
)
def test_assertion_is_retained_as_actual_zero_result_call_with_one_none_metadata_slot(conditions):
    original = form(M.ASSERTION, conditions=conditions)
    assert original["status"] == "supported"
    source = M.metadata_source(original, extent=2, max_tensor_elements=6)
    metadata = source.metadata()
    assert metadata["outputs"] == [{"name": "None", "kind": "none", "original_result_id": "binary:0"}]
    assert metadata["dispatcher_result_count"] == 0 and metadata["tensor_elements"] == 6
    assert metadata["original_result_metadata"] == original["result_metadata"]
    tree = ast.parse(source.loader)
    forward = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "forward")
    call = forward.body[0].value
    assert isinstance(call, ast.Call) and ast.unparse(call.func) == "torch.ops.aten._assert_tensor_metadata.default"
    assert isinstance(call.args[0], ast.Name) and call.args[0].id == "X"
    assert {row.arg for row in call.keywords} == {"size", "stride", "dtype", "device", "layout"}


@pytest.mark.parametrize(
    "target,conditions",
    [
        (M.CAST, {"copy": 1}),
        (M.CAST, {"non_blocking": 0}),
        (M.CAST, {"memory_format": {"kind": "memory_format", "value": "torch.contiguous_format"}}),
        (M.ASSERTION, {"size": [101, 103]}),
        (M.ASSERTION, {"stride": [103, 1]}),
        (M.ASSERTION, {"dtype": {"kind": "dtype", "value": "torch.float32"}}),
        (M.ASSERTION, {"device": {"kind": "device", "value": "cuda"}}),
        (M.ASSERTION, {"layout": {"kind": "layout", "value": "torch.sparse_coo"}}),
    ],
)
def test_unsupported_original_conditions_remain_required_unknown(target, conditions):
    assert form(target, conditions=conditions)["status"] == "unknown"


@pytest.mark.parametrize("target", [M.CAST, M.ASSERTION])
@pytest.mark.parametrize("change", ["parameters", "ordinal", "rank", "storage", "alias", "result", "arity", "form"])
def test_source_rechecks_complete_original_bindings_and_exact_scalar_types(target, change):
    original = form(target)
    if change == "parameters":
        original["parameters"]["dtype"] = "torch.float64"
    elif change == "ordinal":
        original["arguments"][0]["ordinal"] = False
    elif change == "rank":
        original["rank"] = True
    elif change == "storage":
        original["arguments"][0]["value"]["value"]["storage_dtype"] = "float32"
    elif change == "alias":
        original["arguments"][0]["alias"] = {"before": ["x"], "after": ["x"], "write": False}
    elif change == "result":
        original["result_roster"].append(copy.deepcopy(original["result_roster"][0]))
    elif change == "arity":
        original["result_arity"] = True
    else:
        original["form_schema"] = M.ASSERTION_SCHEMA if target == M.CAST else M.CAST_SCHEMA
    with pytest.raises(ValueError):
        M.metadata_source(original, extent=2, max_tensor_elements=100)


@pytest.mark.parametrize("change", ["missing", "not_none", "other_id", "container", "unknown", "native_count"])
def test_zero_result_source_requires_the_original_native_bridge_and_metadata(change):
    trace, schema, defaults, zero = declarations(M.ASSERTION)
    if change == "missing":
        zero = None
    elif change == "native_count":
        zero["rows"][0]["native"]["return_count"] = False
    else:
        node = trace["graphs"]["original"]["nodes"][1]
        if change == "not_none":
            node["result_metadata"]["values"][0]["kind"] = "tensor"
        elif change == "other_id":
            node["result_metadata"]["values"][0]["result_id"] = "other"
        elif change == "container":
            node["result_metadata"]["container"] = "tuple"
        else:
            zero["rows"][0] = {"request": zero["rows"][0]["request"], "status": "unknown", "reason": "absent bridge"}
    if change in {"not_none", "other_id", "container", "native_count"}:
        with pytest.raises(ValueError):
            M.metadata_forms(trace, schema, defaults, zero_returns=zero)
    else:
        assert M.metadata_forms(trace, schema, defaults, zero_returns=zero)[0]["status"] == "unknown"


@pytest.mark.parametrize(
    "target,rank,extent,limit",
    [(M.CAST, 2, 2, 11), (M.ASSERTION, 2, 2, 5), (M.CAST, 10**9, 1, 100), (M.ASSERTION, 2, 10**9, 100)],
)
def test_complete_logical_budget_precedes_shape_and_loader_allocation(target, rank, extent, limit):
    original = form(target, shape=(1,) * min(rank, 3))
    original["rank"] = original["arguments"][0]["value"]["value"]["rank"] = rank
    if target == M.CAST:
        original["result_roster"][0]["rank"] = rank
        original["shape_relation"]["rank"] = rank
    with pytest.raises(ValueError, match="before allocation"):
        M.metadata_source(original, extent=extent, max_tensor_elements=limit)


@pytest.mark.parametrize("target", [M.CAST, M.ASSERTION])
def test_scalar_sources_keep_one_logical_input_and_no_invented_assertion_tensor(target):
    source = M.metadata_source(form(target, shape=()), extent=1009, max_tensor_elements=2)
    assert source.metadata()["inputs"][0]["shape"] == []
    assert source.metadata()["tensor_elements"] == (2 if target == M.CAST else 1)

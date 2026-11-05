"""Independent small typed schemas and complete logical broadcast outputs.

These declarations exercise source construction; they are not protected original
owners, framework/source correspondence or hardware qualification.
"""

import copy
import math
from dataclasses import replace

import pytest

from merlin.targetgen import original_broadcast_add_sources as S
from merlin.targetgen import original_operator_sources as L
from merlin.targetgen.frontend_trace import _digest
from merlin.targetgen.original_operator_reference import (
    OriginalReferenceBudget,
    OriginalReferencePolicy,
    prepare_original_reference,
)
from merlin.targetgen.original_reference_values import TypedReferenceTensor as T


def selected_policy(dtype="float32", arithmetic=None):
    floating = dtype == "float32"
    return OriginalReferencePolicy(
        S.TARGET,
        (dtype, dtype),
        (dtype,),
        "float32" if floating else "int32",
        arithmetic or ("finite_f32" if floating else "bounded_exact"),
        "accumulator_format",
        "elementwise",
        "per_step",
        "rne" if floating else "exact_integer",
        True,
        False,
        "after_reduction",
        0.0,
        0.0,
        "preserve" if floating else "ignore",
    )


def declarations(left=(7, 9), right=(9,), output=None, dtype="float32", *, alpha=None):
    """An explicit pure schema fixture, never an observed native schema grant."""
    rank = max(len(left), len(right))
    padded = [[1] * (rank - len(shape)) + list(shape) for shape in (left, right)]
    output = [max(a, b) for a, b in zip(*padded, strict=True)] if output is None else list(output)
    schema = "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
    arguments = [
        {"name": name, "type": kind, "alias": None, "kwarg_only": index == 2, "has_default": index == 2}
        for index, (name, kind) in enumerate((("self", "Tensor"), ("other", "Tensor"), ("alpha", "number")))
    ]
    values = [
        {
            "id": name + ":0",
            "kind": "tensor",
            "dtype": dtype,
            "storage_dtype": dtype,
            "shape": list(shape),
            "layout": "torch.strided",
            "device": "cpu",
        }
        for name, shape in zip(("left", "right", "add"), (left, right, output), strict=True)
    ]

    def reference(name):
        return {"node_id": name, "value_id": name + ":0"}

    nodes = [
        {
            "id": name,
            "ordinal": index,
            "op": "placeholder",
            "target": name,
            "args": [],
            "kwargs": {},
            "results": [values[index]],
        }
        for index, name in enumerate(("left", "right"))
    ]
    nodes += [
        {
            "id": "add",
            "ordinal": 2,
            "op": "call_function",
            "target": S.TARGET,
            "args": [reference("left"), reference("right")],
            "kwargs": {} if alpha is None else {"alpha": alpha},
            "results": [values[2]],
            "result_arity": 1,
        },
        {
            "id": "output",
            "ordinal": 3,
            "op": "output",
            "target": "output",
            "args": [reference("add")],
            "kwargs": {},
            "results": [],
        },
    ]
    graph = {
        "schema": "m2m.frontend_graph.v1",
        "stage": "original",
        "status": "complete",
        "call_count": 1,
        "operator_schemas": {S.TARGET: schema},
        "nodes": nodes,
        "edges": [
            {
                "producer_node_id": name,
                "producer_value_id": name + ":0",
                "consumer_node_id": consumer,
                "argument_path": path,
                "value_kind": "tensor",
                "dtype": dtype,
                "shape": values[index]["shape"],
            }
            for index, (name, consumer, path) in enumerate(
                (("left", "add", "args/0"), ("right", "add", "args/1"), ("add", "output", "args/0"))
            )
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
                        "default": {"kind": "int", "value": 1} if index == 2 else None,
                    }
                    for index, row in enumerate(arguments)
                ],
            }
        ],
    }
    return {"schema": "m2m.frontend_trace.v1", "graphs": {"original": graph}}, observation, defaults


def form(*args, policy=None, **kwargs):
    return S.broadcast_add_forms(
        *declarations(*args, **kwargs), numerical_semantics=(policy or selected_policy()).record()
    )[0]


def contract(left=(7, 9), right=(9,), *, dtype="float32", extent=1, arithmetic=None):
    policy = selected_policy(dtype, arithmetic)
    original = form(left, right, dtype=dtype, policy=policy)
    source = S.broadcast_add_source(original, extent=extent, max_tensor_elements=100000)
    return prepare_original_reference(
        original,
        source,
        extent=extent,
        policy=policy,
        budget=OriginalReferenceBudget(100000, 800000, 200000, 100000),
        output_byteorder="little",
    )


@pytest.mark.parametrize(
    "left,right,inputs,output",
    [
        ((7, 9), (9,), [[2, 3], [3]], [2, 3]),
        ((2, 3, 5, 7), (2, 3, 5, 7), [[2, 3, 4, 5]] * 2, [2, 3, 4, 5]),
        ((1, 7, 1), (5, 1, 9), [[1, 3, 1], [2, 1, 4]], [2, 3, 4]),
        ((), (9,), [[], [2]], [2]),
        ((), (), [[], []], []),
        ((1, 1, 1), (1,), [[1, 1, 1], [1]], [1, 1, 1]),
    ],
)
def test_exact_original_broadcast_relations_choose_fresh_bounded_geometry(left, right, inputs, output):
    original = form(left, right)
    assert original["status"] == "supported"
    source = S.broadcast_add_source(original, extent=1, max_tensor_elements=10000)
    metadata = source.metadata()
    assert [row["shape"] for row in metadata["inputs"]] == inputs
    assert metadata["outputs"][0]["shape"] == output
    assert metadata["tensor_elements"] == sum(math.prod(shape) for shape in [*inputs, output])
    assert metadata["scalar_products"] == math.prod(output)
    assert [row["name"] for row in metadata["inputs"] + metadata["outputs"]] == ["X", "W", "Y"]
    assert "alpha=1" in source.loader and "no numerical-domain" in metadata["scope"]


@pytest.mark.parametrize("left,right", [((7, 9), (9,)), ((1, 7, 1), (5, 1, 9)), ((), (9,)), ((), ())])
@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_complete_output_projection_visits_every_aligned_and_singleton_input(left, right, dtype):
    checked = contract(left, right, dtype=dtype)
    metadata = checked.verify()
    inputs = tuple(
        T.from_values(
            row["name"],
            dtype,
            row["shape"],
            [((index * (operand + 2)) % 11) - 5 for index in range(math.prod(row["shape"]))],
            byteorder="big",
        )
        for operand, row in enumerate(metadata["inputs"])
    )
    # Independent Cartesian-coordinate reference, not the evaluator's flat-index loop.
    from itertools import product

    output = metadata["outputs"][0]
    expected = []
    for coordinate in product(*(range(width) for width in output["shape"])):
        values = []
        for tensor in inputs:
            aligned = coordinate[len(coordinate) - len(tensor.shape) :]
            flat = 0
            for index, width in zip(aligned, tensor.shape, strict=True):
                flat = flat * width + (0 if width == 1 else index)
            values.append(tensor.values()[flat])
        expected.append(sum(values))
    actual = T.from_values("Y", dtype, output["shape"], expected, byteorder="little")
    assert checked.evaluate(inputs) == (actual,)
    assert checked.compare(inputs, (actual,))["checked_elements"] == len(expected)
    changed = T.from_values("Y", dtype, output["shape"], [*expected[:-1], expected[-1] + 1], byteorder="little")
    result = checked.compare(inputs, (changed,))
    assert not result["passed"] and result["mismatches"][0]["index"] == len(expected) - 1


def test_f32_signed_zero_subnormal_and_rounding_and_integer_wrap_stay_selected():
    checked = contract((7, 9), (9,))
    inputs = (
        T.from_values("X", "float32", (2, 3), [-0.0, 2**-149, 1, -0.0, -(2**-149), 1], byteorder="little"),
        T.from_values("W", "float32", (3,), [-0.0, 2**-149, 2**-24], byteorder="little"),
    )
    exact = T.from_values("Y", "float32", (2, 3), [-0.0, 2**-148, 1, -0.0, 0.0, 1], byteorder="little")
    assert checked.evaluate(inputs) == (exact,)
    changed = replace(exact, data=b"\0\0\0\0" + exact.data[4:])
    assert not checked.compare(inputs, (changed,))["passed"]
    wrapping = contract((), (9,), dtype="int8", arithmetic="modular_wrap")
    integers = (
        T.from_values("X", "int8", (), [127], byteorder="little"),
        T.from_values("W", "int8", (2,), [1, -1], byteorder="little"),
    )
    assert wrapping.evaluate(integers)[0].values() == (-128, 126)
    with pytest.raises(ValueError, match="readout overflow"):
        contract((), (9,), dtype="int8").evaluate(integers)


@pytest.mark.parametrize(
    "defect",
    [
        "incompatible",
        "result_shape",
        "zero",
        "symbolic",
        "alpha_float",
        "alpha_bool",
        "alpha_nonunit",
        "default_bool",
        "alias",
        "return_count",
        "promotion",
    ],
)
def test_incomplete_or_incompatible_original_bindings_remain_required_unknown(defect):
    args = {"incompatible": ((7, 9), (8,)), "zero": ((0, 9), (9,)), "symbolic": (("n", 9), (9,))}.get(
        defect, ((7, 9), (9,))
    )
    # Incompatible/symbolic shapes need an explicit result to avoid deriving it in this fixture.
    trace, schemas, defaults = declarations(
        *args, output=(7, 9), alpha={"alpha_float": 1.0, "alpha_bool": True, "alpha_nonunit": 2}.get(defect)
    )
    if defect == "result_shape":
        trace, schemas, defaults = declarations(output=(7, 8))
    elif defect == "default_bool":
        defaults["rows"][0]["defaults"][2]["default"] = {"kind": "bool", "value": True}
    elif defect == "alias":
        schemas["rows"][0]["arguments"][0]["alias"] = {"before": ["a"], "after": ["a"], "write": False}
    elif defect == "return_count":
        schemas["rows"][0]["returns"] = []
    elif defect == "promotion":
        graph = trace["graphs"]["original"]
        graph["nodes"][2]["results"][0]["dtype"] = "float64"
        graph["edges"][2]["dtype"] = "float64"
        graph["sha256"] = _digest({key: value for key, value in graph.items() if key != "sha256"})
        defaults["graph_sha256"] = graph["sha256"]
    original = S.broadcast_add_forms(trace, schemas, defaults)[0]
    assert original["status"] == "unknown"


def test_complete_budgets_and_original_parameters_are_checked_before_allocation():
    original = form()
    for extent, budget in ((10**100, 100), (True, 100), (1, 14), (1, True)):
        with pytest.raises(ValueError):
            S.broadcast_add_source(original, extent=extent, max_tensor_elements=budget)
    source = S.broadcast_add_source(original, extent=1, max_tensor_elements=15)
    assert source.metadata()["tensor_elements"] == 15
    changed = copy.deepcopy(original)
    changed["arguments"][0]["value"]["value"]["rank"] = 10**9
    with pytest.raises(ValueError):
        S.broadcast_add_source(changed, extent=1, max_tensor_elements=100)
    for key, value in (
        ("alpha", True),
        ("output_axes", ["varying"]),
        ("operand_axes", [["varying", "invalid"], ["varying"]]),
    ):
        changed = copy.deepcopy(original)
        changed["parameters"][key] = value
        with pytest.raises(ValueError):
            S.broadcast_add_source(changed, extent=1, max_tensor_elements=100)
    original["source_numerical_semantics"]["atol"] = 1.0
    assert source.metadata()["source_numerical_semantics"]["atol"] == 0.0


def test_legacy_factory_and_selected_policy_meanings_remain_unchanged():
    trace, schema, defaults = declarations()
    assert L.original_add_forms(trace, schema, defaults)[0]["status"] == "unknown"
    supported = form()
    assert (
        L.policy_compatibility(
            supported,
            {
                "model": {"engine": "integer_reference"},
                "operand_dtype": "int8",
                "accumulator_dtype": "i32",
                "readout_dtype": "i32",
            },
        )["status"]
        == "unknown"
    )
    changed = copy.deepcopy(supported)
    changed["source_numerical_semantics"] = {"not_the_selected_policy": True}
    source = S.broadcast_add_source(changed, extent=1, max_tensor_elements=100)
    with pytest.raises(ValueError, match="numerical"):
        prepare_original_reference(
            changed,
            source,
            extent=1,
            policy=selected_policy(),
            budget=OriginalReferenceBudget(100, 1000, 100, 10000),
            output_byteorder="little",
        )

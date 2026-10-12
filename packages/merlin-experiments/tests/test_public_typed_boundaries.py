"""Tiny typed public source controls; no capture, simulator or qualification."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from merlin_experiments.phase0 import public_boundaries as B
from merlin_experiments.phase0 import public_boundary_semantics as S
from merlin_experiments.phase0 import public_typed_semantics as T
from test_public_joint_boundaries import POLICY, produce, trace

from merlin.common.paths import repo_root
from merlin.targetgen import capsule_golden
from merlin.targetgen.capsule_source import _model_output_numeric_policy


def dtype(name):
    return {"kind": "dtype", "value": "torch." + name}


def representation_program():
    rows = [
        ("round", "aten.round.default", [{"value": "X"}], {}, [2, 3, 4], "float32"),
        ("clamp", "aten.clamp.default", [{"value": "round"}, -4, 4], {}, [2, 3, 4], "float32"),
        ("packed", "aten.to.dtype", [{"value": "clamp"}, dtype("int8")], {}, [2, 3, 4], "int8"),
        ("wide", "aten.to.dtype", [{"value": "packed"}, dtype("int32")], {}, [2, 3, 4], "int32"),
        ("product", "aten.bmm.default", [{"value": "wide"}, {"value": "W"}], {}, [2, 3, 5], "int32"),
        ("reduced", "aten.sum.dim_IntList", [{"value": "product"}, [-1], True], {}, [2, 3, 1], "int64"),
        ("floating", "aten.to.dtype", [{"value": "reduced"}, dtype("float32")], {}, [2, 3, 1], "float32"),
        ("average", "aten.mean.dim", [{"value": "floating"}, [1], True], {}, [2, 1, 1], "float32"),
    ]
    return {
        "inputs": [
            {"name": "X", "shape": [2, 3, 4], "dtype": "float32"},
            {"name": "W", "shape": [2, 4, 5], "dtype": "int32"},
        ],
        "nodes": [dict(name=n, target=t, args=a, kwargs=k, shape=s, dtype=d) for n, t, a, k, s, d in rows],
        "outputs": ["packed", "average"],
    }


def execute(program, dimensions=None):
    torch = pytest.importorskip("torch")
    cuts, missing = S.cuts(trace(program), max_nodes=96, typed=True)
    assert len(cuts) == 1 and missing == []
    region = cuts[0]
    bindings = dimensions or dict(zip(region["dimensions"], [3, 5, 7, 4], strict=True))
    actual = S.instantiate(region, bindings)
    namespace = {}
    exec(compile(B.render(actual), "typed-public-source", "exec"), namespace)
    module, inputs = namespace["get_model_and_inputs"]()
    return actual, inputs, module(*inputs), torch


def test_real_cast_quantized_representation_batched_reduction_and_all_liveouts():
    program, inputs, outputs, torch = execute(representation_program())
    x, w = (v.numpy() for v in inputs)
    packed = np.clip(np.rint(x), -4, 4).astype(np.int8)
    product = np.matmul(packed.astype(np.int32), w)
    average = product.sum(-1, keepdims=True, dtype=np.int64).astype(np.float32).mean(1, keepdims=True)
    assert len(outputs) == 2 and len(program["outputs"]) == 2
    np.testing.assert_array_equal(outputs[0].numpy(), packed)
    np.testing.assert_allclose(outputs[1].numpy(), average, rtol=1e-6)
    assert outputs[0].dtype is torch.int8 and outputs[1].dtype is torch.float32
    assert len(program["nodes"]) == 8 and all(v.dtype == torch.int32 for v in inputs[1:])
    assert np.count_nonzero(packed) > 0
    # Prior float-only source contract never relabels this integer graph.
    _, missing = S.cuts(trace(representation_program()), max_nodes=96)
    assert missing


def test_higher_rank_broadcast_contraction_softmax_and_explicit_reduction_dtype():
    program = {
        "inputs": [
            {"name": "X", "shape": [2, 1, 3, 4], "dtype": "float32"},
            {"name": "W", "shape": [1, 5, 4, 6], "dtype": "float32"},
        ],
        "nodes": [
            {
                "name": "C",
                "target": "aten.matmul.default",
                "args": [{"value": "X"}, {"value": "W"}],
                "kwargs": {},
                "shape": [2, 5, 3, 6],
                "dtype": "float32",
            },
            {
                "name": "P",
                "target": "aten.softmax.int",
                "args": [{"value": "C"}, -1],
                "kwargs": {},
                "shape": [2, 5, 3, 6],
                "dtype": "float32",
            },
            {
                "name": "R",
                "target": "aten.sum.dim_IntList",
                "args": [{"value": "P"}, [-2]],
                "kwargs": {"dtype": dtype("float64")},
                "shape": [2, 5, 6],
                "dtype": "float64",
            },
        ],
        "outputs": ["R"],
    }
    actual, inputs, outputs, _ = execute(program, {"D0": 3, "D1": 4, "D2": 5, "D3": 2, "D4": 7})
    x, w = (v.numpy() for v in inputs)
    c = x @ w
    exponent = np.exp(c - c.max(-1, keepdims=True))
    expected = (exponent / exponent.sum(-1, keepdims=True)).sum(-2, dtype=np.float64)
    np.testing.assert_allclose(outputs[0].numpy(), expected, rtol=2e-6, atol=1e-6)
    assert actual["nodes"][-1]["dtype"] == "float64"


def test_host_only_conversion_reduction_is_a_real_source_member():
    source = {
        "inputs": [{"name": "X", "shape": [3, 7], "dtype": "int8"}],
        "nodes": [
            {
                "name": "C",
                "target": "aten._to_copy.default",
                "args": [{"value": "X"}],
                "kwargs": {"dtype": dtype("int32")},
                "shape": [3, 7],
                "dtype": "int32",
            },
            {
                "name": "R",
                "target": "aten.sum.dim_IntList",
                "args": [{"value": "C"}, [1], True],
                "kwargs": {},
                "shape": [3, 1],
                "dtype": "int64",
            },
        ],
        "outputs": ["R"],
    }
    actual, inputs, outputs, _ = execute(source, {"D0": 4, "D1": 9})
    np.testing.assert_array_equal(outputs[0].numpy(), inputs[0].numpy().sum(1, keepdims=True, dtype=np.int64))
    assert not {r["target"] for r in actual["nodes"]} & T.CONTRACTIONS


@pytest.mark.parametrize("defect", ["dtype", "effect", "device", "batch"])
def test_original_storage_effect_or_shape_drift_is_explicit_unknown(defect):
    program = copy.deepcopy(representation_program())
    if defect == "dtype":
        program["nodes"][2]["dtype"] = "int16"
    elif defect == "effect":
        program["nodes"][1]["target"] = "aten.copy_.default"
    elif defect == "device":
        program["nodes"][2]["kwargs"]["memory_format"] = "unknown"
    else:
        program["inputs"][1]["shape"][0] = 7
    _, missing = S.cuts(trace(program), max_nodes=96, typed=True)
    assert missing and all(row["status"] == "UNKNOWN" for row in missing)


def test_explicit_normal_recipe_selects_versioned_ordinary_template():
    root = repo_root()
    recipe = yaml.safe_load((root / "examples/gemmini/phase0/recipe-public-connected.yaml").read_text())
    template = yaml.safe_load((root / "experiments/templates/phase0/performance.yaml").read_text())
    policy = B.validate(recipe["public_joint_boundaries"])
    assert template["public_boundary_factories"][B.SCHEMA_V2] == B.template_for(policy)
    assert template["public_boundary_factory"] == B.TEMPLATE
    assert policy["max_nodes"] >= 96 and policy["max_logical_elements"] >= 8 << 20


def test_typed_joint_pressure_constructs_real_mixed_storage_members():
    entries, products, report = produce(
        capacity=2048, source=trace(representation_program()), policy={**POLICY, "schema": B.SCHEMA_V2, "max_nodes": 96}
    )
    assert entries and report["schema"] == B.SCHEMA_V2
    for entry in entries:
        member = entry["public_boundary"]
        assert member["schema"] == B.SCHEMA_V2 and member["program"]["schema"] == S.TYPED_SCHEMA
        assert {row["dtype"] for row in member["program"]["nodes"]} == {"float32", "int8", "int32", "int64"}
        assert len(member["program"]["outputs"]) == 2
        assert products[entry["loader"]] == B.render(member["program"]).encode()
        assert member["pressure"]["hardware_mapping"] == "UNKNOWN"
    assert {point["point"]["fits"] for point in (e["public_boundary"] for e in entries)} == {True, False}


def test_mixed_public_outputs_preserve_exact_integer_and_original_float_gates():
    policy = _model_output_numeric_policy(
        [{"dtype": "i8"}, {"dtype": "f32"}],
        [[3], [1.0]],
        SimpleNamespace(atol=0.01, rtol=0.02),
        output_names=["quantized", "floating"],
    )
    assert policy["outputs"]["quantized"]["compare"] == "exact_int"
    assert policy["outputs"]["floating"]["atol"] == 0.01
    expected = {"quantized": [3], "floating": [1.0]}
    assert capsule_golden.compare(expected, {"quantized": [3], "floating": [1.001]}, policy)["status"] == "pass"
    assert capsule_golden.compare(expected, {"quantized": [4], "floating": [1.001]}, policy)["status"] == "fail"
    assert capsule_golden.compare(expected, {"quantized": [3]}, policy)["status"] == "fail"

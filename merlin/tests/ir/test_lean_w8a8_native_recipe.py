"""Whole dynamic W8A8 output against an ordered finite-input numerical recipe.

These native comparisons complement the parametric Lean component proofs. They do not run
Lean's host float evaluator for max/min, prove the recipe independently reviewed, or issue C5
receipts. No production quantization helpers are used to compute the reference answers.
"""

from __future__ import annotations

import json

import pytest
from test_lean_w8a8_body import exported, lowered
from test_lean_w8a8_scalar import parts

from merlin.verify.lean import differential as diff


def quantize_rows(values):
    import numpy as np

    assert values.dtype == np.float32 and np.isfinite(values).all()
    quantized = np.empty(values.shape, dtype=np.int8)
    scales = np.empty(values.shape[0], dtype=np.float32)
    for row_index, row in enumerate(values):
        maximum = np.float32(0)
        for value in row:
            maximum = np.maximum(np.abs(value), maximum)
        # The user-approved recipe includes the explicit all-zero-row branch.
        scale = (
            np.float32(1e-5) if maximum == 0 else np.maximum(np.float32(maximum / np.float32(127)), np.float32(1e-5))
        )
        inverse = np.float32(np.float32(1) / scale)
        scales[row_index] = scale
        for column, value in enumerate(row):
            rounded = np.rint(np.float32(value * inverse))
            bounded = np.maximum(np.minimum(rounded, np.float32(127)), np.float32(-127))
            quantized[row_index, column] = np.int8(bounded)
    return quantized, scales


def dynamic_matmul_reference(left, right):
    import numpy as np

    m, k = left.shape
    assert right.shape[0] == k and k * 16384 < 2**31
    qleft, left_scale = quantize_rows(left)
    qright_rows, right_scale = quantize_rows(right.T)
    result = np.empty((m, right.shape[1]), dtype=np.float32)
    for row in range(m):
        for column in range(right.shape[1]):
            accumulator = sum(int(qleft[row, inner]) * int(qright_rows[column, inner]) for inner in range(k))
            converted = np.float32(accumulator)
            result[row, column] = np.float32(np.float32(converted * left_scale[row]) * right_scale[column])
    return result


def samples_for_shape(m, k, n):
    import numpy as np

    random = np.random.default_rng(20261020)
    rounding = np.array([127, 0.5, 1.5, 2.5, -0.5, -1.5, -127, 0], dtype=np.float32)
    return [
        (np.zeros((m, k), dtype=np.float32), np.zeros((k, n), dtype=np.float32)),
        (random.uniform(-5, 5, (m, k)).astype(np.float32), random.uniform(-7, 7, (k, n)).astype(np.float32)),
        (
            np.resize(np.array([0, 5e-6, -5e-6, 1e-7], dtype=np.float32), (m, k)),
            np.resize(np.array([0, -2e-6, 3e-6, -1e-7], dtype=np.float32), (k, n)),
        ),
        (np.resize(rounding, (m, k)), np.resize(rounding[::-1], (n, k)).T.copy()),
        (
            random.uniform(-1, 1, (m, k)).astype(np.float32)
            * np.resize(np.array([1e-6, 0.1, 1000], dtype=np.float32), m)[:, None],
            random.uniform(-1, 1, (k, n)).astype(np.float32)
            * np.resize(np.array([300, 1e-6, 0.25, 7], dtype=np.float32), n)[None, :],
        ),
    ]


def execute_recipe_case(module, samples, tmp_path, lean_tools, fidelity_tools):
    import numpy as np

    def words(values):
        return [int(x) for x in values.view(np.uint32).flat]

    document = exported(module, tmp_path, lean_tools)
    inputs, outputs = diff.signature(json.loads(document.read_text()), "forward")
    arguments = [[words(left), words(right)] for left, right in samples]
    expected = [[str(x) for x in words(dynamic_matmul_reference(left, right))] for left, right in samples]
    actual = diff.execute_mlir(
        tmp_path / "actual.mlir",
        arguments,
        tmp_path,
        fidelity_tools,
        inputs=inputs,
        outputs=outputs,
        function="forward",
    )
    (tmp_path / "recipe-results.json").write_text(
        json.dumps(
            {"scope": "native_dynamic_w8a8_recipe_only", "whole_pass_verified": False, "results": expected}, indent=2
        )
    )
    return actual["results"], expected


@pytest.mark.parametrize("shape", [(2, 8, 3), (3, 9, 4), (1, 0, 2), (0, 3, 2)])
def test_complete_dynamic_w8a8_against_native_recipe(tmp_path, lean_tools, fidelity_tools, shape):
    module, _, _ = lowered(*shape)
    actual, expected = execute_recipe_case(module, samples_for_shape(*shape), tmp_path, lean_tools, fidelity_tools)
    assert actual == expected


def test_reference_rounding_and_group_scales():
    import numpy as np

    values = np.array([[127, 0.5, 1.5, 2.5, -0.5, -1.5, -127, 0]], dtype=np.float32)
    quantized, scales = quantize_rows(np.concatenate([values, values * np.float32(2)], axis=0))
    assert quantized.tolist() == [[127, 0, 2, 2, 0, -2, -127, 0]] * 2
    assert scales.tolist() == [1.0, 2.0]
    quantized, scales = quantize_rows(np.zeros((2, 0), dtype=np.float32))
    assert quantized.shape == (2, 0)
    assert scales.tolist() == [np.float32(1e-5)] * 2


@pytest.mark.parametrize("mutation", ["rounding", "scale_axes", "initial_accumulator", "upper_bound"])
def test_native_recipe_detects_valid_mlir_corruptions(tmp_path, lean_tools, fidelity_tools, mutation):
    from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, f32, i32

    shape = (3, 8, 3)
    module, _, contraction = lowered(*shape)
    if mutation == "initial_accumulator":
        contraction.outputs[0].owner.inputs[0].owner.properties["value"] = IntegerAttr(1, i32)
    elif mutation == "scale_axes":
        _, _, requant = next(part for part in parts(module) if part[0] == "requantize2")
        maps = list(requant.indexing_maps)
        requant.properties["indexing_maps"] = ArrayAttr([maps[0], maps[2], maps[1], maps[3]])
    else:
        _, _, quantize = next(part for part in parts(module) if part[0] == "quantize")
        body = list(quantize.body.block.ops)
        if mutation == "rounding":
            body[1].name = "math.round"
        else:
            body[2].operands[1].owner.properties["value"] = FloatAttr(126.0, f32)
    actual, expected = execute_recipe_case(module, samples_for_shape(*shape), tmp_path, lean_tools, fidelity_tools)
    assert actual != expected

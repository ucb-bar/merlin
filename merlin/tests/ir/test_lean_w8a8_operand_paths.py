"""Actual mixed-byte and per-channel source paths, including layout/channel correspondence."""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_module import check_module
from test_lean_w8a8_native_recipe import quantize_rows

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_quant_int import lower_contraction_int8
from merlin.llvmlower.passes_xdsl import lower_quant_ext
from merlin.verify.lean import differential as diff


def mixed_source(integer_left, accumulator_first=False, unsigned=False):
    left, right = ("i8", "f32") if integer_left else ("f32", "i8")
    byte = "%av" if integer_left else "%bv"
    cast = "uitofp" if unsigned else "sitofp"
    product = "%converted, %bv" if integer_left else "%av, %converted"
    addition = "%acc, %product" if accumulator_first else "%product, %acc"
    return parse_mlir_text(f"""func.func @forward(%a: tensor<2x3x{left}>, %b: tensor<3x4x{right}>) -> tensor<2x4xf32> {{
      %e = tensor.empty() : tensor<2x4xf32>
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%e : tensor<2x4xf32>) -> tensor<2x4xf32>
      %r = linalg.generic {{indexing_maps = [affine_map<(m,n,k) -> (m,k)>,
        affine_map<(m,n,k) -> (k,n)>, affine_map<(m,n,k) -> (m,n)>],
        iterator_types = ["parallel", "parallel", "reduction"]}}
        ins(%a, %b : tensor<2x3x{left}>, tensor<3x4x{right}>) outs(%init : tensor<2x4xf32>) {{
          ^bb0(%av: {left}, %bv: {right}, %acc: f32):
            %converted = arith.{cast} {byte} : i8 to f32
            %product = arith.mulf {product} : f32
            %sum = arith.addf {addition} : f32
            linalg.yield %sum : f32
        }} -> tensor<2x4xf32>
      return %r : tensor<2x4xf32>
    }}""")


@pytest.mark.parametrize("integer_left", [False, True])
@pytest.mark.parametrize("accumulator_first", [False, True])
def test_actual_mixed_inputs_and_add_order(tmp_path, lean_tools, integer_left, accumulator_first):
    before = mixed_source(integer_left, accumulator_first)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    checked = check_module(before, after, tmp_path, lean_tools)
    assert checked["conditional_match"] and checked["qualification"]["qualification_passed"], checked


@pytest.mark.parametrize("integer_left", [False, True])
def test_unsigned_source_cast_cannot_select_signed_reference(tmp_path, lean_tools, integer_left):
    before = mixed_source(integer_left, unsigned=True)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    checked = check_module(before, after, tmp_path, lean_tools)
    assert not checked["expanded"] and not checked["conditional_match"]


def calibrated_source(transpose_left=False, transpose_right=False, *, zero_point=0, wrong_axis=False):
    shapes = [(3, 2) if transpose_left else (2, 3), (4, 3) if transpose_right else (3, 4)]
    axes = [1 if transpose_left else 0, 0 if transpose_right else 1]
    if wrong_axis:
        # Equal dimensions keep the incorrect channel choice well-typed in the emitted program.
        shapes = [(3, 3), (3, 3)]
        axes = [0, 0]
    m, k, n = (3, 3, 3) if wrong_axis else (2, 3, 4)
    declarations, code, inputs = [], [], []
    for name, shape, axis, transpose, output_shape in zip(
        ("a", "b"), shapes, axes, (transpose_left, transpose_right), ((m, k), (k, n)), strict=True
    ):
        ty = "x".join(map(str, shape))
        parameters = shape[axis]
        declarations += [f"%{name}q: tensor<{ty}xi8>", f"%{name}s: tensor<{parameters}xf32>"]
        code += [
            f"%{name}z = tensor.splat %zp : tensor<{parameters}xi64>",
            f'%{name}d = "quant_ext.dequantize_per_channel"(%{name}q, %{name}s, %{name}z) '
            f"<{{axis = {axis} : i64, quant_min = -127 : i64, quant_max = 127 : i64}}> : "
            f"(tensor<{ty}xi8>, tensor<{parameters}xf32>, tensor<{parameters}xi64>) -> tensor<{ty}xf32>",
        ]
        if transpose:
            result_ty = "x".join(map(str, output_shape))
            code += [
                f"%{name}e = tensor.empty() : tensor<{result_ty}xf32>",
                f"%{name} = linalg.transpose ins(%{name}d : tensor<{ty}xf32>) "
                f"outs(%{name}e : tensor<{result_ty}xf32>) permutation = [1, 0]",
            ]
        inputs.append(f"%{name}" if transpose else f"%{name}d")
    return parse_mlir_text(f"""func.func @forward({", ".join(declarations)}) -> tensor<{m}x{n}xf32> {{
      %zp = arith.constant {zero_point} : i64
      {" ".join(code)}
      %e = tensor.empty() : tensor<{m}x{n}xf32>
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%e : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
      %r = linalg.matmul ins({", ".join(inputs)} : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>)
        outs(%init : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
      return %r : tensor<{m}x{n}xf32>
    }}""")


@pytest.mark.parametrize("transpose_left,transpose_right", [(False, False), (True, False), (False, True), (True, True)])
def test_channel_scale_correspondence_through_transposes(tmp_path, lean_tools, transpose_left, transpose_right):
    before = calibrated_source(transpose_left, transpose_right)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    checked = check_module(before, after, tmp_path, lean_tools, scope="captured")
    assert checked["conditional_match"], checked
    assert not checked["qualification"]["qualification_passed"]


@pytest.mark.parametrize("change", [{"zero_point": 1}, {"wrong_axis": True}])
def test_per_channel_source_contract_is_recomputed(tmp_path, lean_tools, change):
    before = calibrated_source(**change)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    checked = check_module(before, after, tmp_path, lean_tools, scope="captured")
    assert not checked["expanded"] and not checked["conditional_match"]


def native(module, arguments, tmp_path, lean_tools, fidelity_tools):
    document = exported(module, tmp_path, lean_tools)
    inputs, outputs = diff.signature(json.loads(document.read_text()), "forward")
    return diff.execute_mlir(
        tmp_path / "actual.mlir",
        arguments,
        tmp_path,
        fidelity_tools,
        inputs=inputs,
        outputs=outputs,
        function="forward",
    )["results"]


def words(array):
    width = np.uint32 if array.dtype == np.float32 else np.uint8
    return [str(int(x)) for x in array.reshape(-1).view(width)]


@pytest.mark.parametrize("integer_left", [False, True])
def test_mixed_input_native_requantization_order(tmp_path, lean_tools, fidelity_tools, integer_left):
    module = mixed_source(integer_left)
    assert lower_contraction_int8(module) == 1
    rng = np.random.default_rng(301)
    left = (
        rng.integers(-128, 128, size=(2, 3), dtype=np.int8)
        if integer_left
        else rng.normal(size=(2, 3)).astype(np.float32)
    )
    right = (
        rng.normal(size=(3, 4)).astype(np.float32)
        if integer_left
        else rng.integers(-128, 128, size=(3, 4), dtype=np.int8)
    )
    if integer_left:
        quantized, scales = quantize_rows(right.T)
        expected = (left.astype(np.int32) @ quantized.T.astype(np.int32)).astype(np.float32) * scales[None, :]
    else:
        quantized, scales = quantize_rows(left)
        expected = (quantized.astype(np.int32) @ right.astype(np.int32)).astype(np.float32) * scales[:, None]
    assert native(module, [[words(left), words(right)]], tmp_path, lean_tools, fidelity_tools) == [words(expected)]


@pytest.mark.parametrize("transpose_left,transpose_right", [(False, False), (True, True)])
def test_calibrated_native_requantization_uses_correct_channels(
    tmp_path, lean_tools, fidelity_tools, transpose_left, transpose_right
):
    module = calibrated_source(transpose_left, transpose_right)
    assert lower_contraction_int8(module) == 1
    assert lower_quant_ext(module) == 2
    left = np.array([[-128, 1, 127], [4, -5, 6]], dtype=np.int8)
    right = np.arange(-6, 6, dtype=np.int8).reshape(3, 4)
    left_scale = np.array([0.25, 2], dtype=np.float32)
    right_scale = np.array([3, 0.125, 2, 0.5], dtype=np.float32)
    accumulator = (left.astype(np.int32) @ right.astype(np.int32)).astype(np.float32)
    expected = np.float32(np.float32(accumulator * left_scale[:, None]) * right_scale[None, :])
    args = [
        words(left.T.copy() if transpose_left else left),
        words(left_scale),
        words(right.T.copy() if transpose_right else right),
        words(right_scale),
    ]
    assert native(module, [args], tmp_path, lean_tools, fidelity_tools) == [words(expected)]

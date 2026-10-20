"""Whole-function comparison against a source-derived, explicitly unreviewed policy draft."""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_quant_int import lower_contraction_int8


def source(m=2, k=3, n=4):
    return parse_mlir_text(f"""func.func @forward(%a: tensor<{m}x{k}xf32>,
        %b: tensor<{k}x{n}xf32>) -> tensor<{m}x{n}xf32> {{
      %e = tensor.empty() : tensor<{m}x{n}xf32>
      %z = arith.constant 0.0 : f32
      %f = linalg.fill ins(%z : f32) outs(%e : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
      %r = linalg.matmul ins(%a, %b : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>)
        outs(%f : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
      return %r : tensor<{m}x{n}xf32>
    }}""")


def check_module(before, after, tmp_path, tools, scope="all"):
    for name in ("before", "after"):
        (tmp_path / name).mkdir()
    before_doc = exported(before, tmp_path / "before", tools, generic=True, allow_unregistered=True)
    after_doc = exported(after, tmp_path / "after", tools, generic=True, allow_unregistered=True)
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_module.lean"),
            str(before_doc),
            str(after_doc),
            scope,
            str(tmp_path / "reference.mlir"),
        ],
        cwd=repo_root() / "merlin/lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["whole_pass_verified"] is False
    assert report["policy_reviewed"] is False
    assert (proc.returncode == 0) == report["conditional_match"]
    driver = subprocess.run(
        [str(tools.lean_exe), "check-w8a8-draft", str(before_doc), str(after_doc), "64", scope],
        capture_output=True,
        text=True,
        check=False,
    )
    assert driver.returncode == 1, driver.stdout + driver.stderr
    qualified = json.loads(driver.stdout)
    assert qualified["conditional_match"] == report["conditional_match"]
    assert qualified["verdict"] == "abstracted"
    assert qualified["whole_pass_verified"] is False
    assert qualified["policy_reviewed"] is False
    report["qualification"] = qualified
    return report


@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 0, 2), (0, 3, 2)])
def test_actual_dynamic_function_matches_source_derived_reference(tmp_path, lean_tools, shape):
    before = source(*shape)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report = check_module(before, after, tmp_path, lean_tools)
    assert report["conditional_match"], report
    assert report["qualification"]["qualification_passed"], report


def test_captured_scope_does_not_widen_unquantized_source(tmp_path, lean_tools):
    before = source()
    report = check_module(before, before.clone(), tmp_path, lean_tools, scope="captured")
    assert report["conditional_match"], report


def test_omitting_selected_quantization_is_refused(tmp_path, lean_tools):
    before = source()
    report = check_module(before, before.clone(), tmp_path, lean_tools)
    assert not report["conditional_match"]


@pytest.mark.parametrize("mutation", ["quantized_source", "scale_source", "contraction_inputs", "scale_floor"])
def test_valid_mlir_cross_wiring_and_body_mutations_refused(tmp_path, lean_tools, mutation):
    from test_lean_w8a8_scalar import parts
    from xdsl.dialects.builtin import FloatAttr, f32

    before = source(2, 2, 2)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    function = next(op for op in after.walk() if op.name == "func.func")
    if mutation == "quantized_source":
        quantizer = next(op for kind, _, op in parts(after) if kind == "quantize")
        quantizer.operands = [function.body.block.args[1], *quantizer.operands[1:]]
    elif mutation == "scale_source":
        requantizer = next(op for kind, _, op in parts(after) if kind == "requantize2")
        operands = list(requantizer.operands)
        requantizer.operands = [operands[0], operands[2], operands[1], operands[3]]
    elif mutation == "contraction_inputs":
        contraction = next(
            op
            for op in after.walk()
            if op.name == "linalg.generic" and [str(arg.type) for arg in op.body.block.args] == ["i8", "i8", "i32"]
        )
        contraction.operands = [contraction.inputs[1], contraction.inputs[0], *contraction.outputs]
    else:
        scale = next(op for kind, _, op in parts(after) if kind == "scale")
        list(scale.body.block.ops)[2].properties["value"] = FloatAttr(1e-4, f32)
    report = check_module(before, after, tmp_path, lean_tools)
    assert report["expanded"] and not report["conditional_match"], report


@pytest.mark.parametrize("mutation", ["wrong_source_body", "nonzero_source_init"])
def test_unsupported_source_cannot_choose_its_target_as_reference(tmp_path, lean_tools, mutation):
    from xdsl.dialects.builtin import FloatAttr, f32

    before = source()
    matmul = next(op for op in before.walk() if op.name == "linalg.matmul")
    if mutation == "wrong_source_body":
        from xdsl.dialects.builtin import ArrayAttr
        from xdsl.dialects.linalg import ops as linalg

        replacement = linalg.GenericOp(
            inputs=matmul.inputs,
            outputs=matmul.outputs,
            body=matmul.body.clone(),
            indexing_maps=matmul.indexing_maps,
            iterator_types=ArrayAttr(
                [
                    linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL),
                    linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL),
                    linalg.IteratorTypeAttr(linalg.IteratorType.REDUCTION),
                ]
            ),
            result_types=[matmul.results[0].type],
        )
        block = matmul.parent_block()
        block.insert_op_before(replacement, matmul)
        matmul.results[0].replace_all_uses_with(replacement.results[0])
        block.erase_op(matmul)
        replacement.body.block.first_op.operands = [replacement.body.block.args[0]] * 2
    else:
        matmul.outputs[0].owner.inputs[0].owner.properties["value"] = FloatAttr(1.0, f32)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report = check_module(before, after, tmp_path, lean_tools)
    assert not report["expanded"] and not report["conditional_match"]


def test_widening_captured_scope_is_refused(tmp_path, lean_tools):
    before = source()
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report = check_module(before, after, tmp_path, lean_tools, scope="captured")
    assert report["expanded"] and not report["conditional_match"]


def calibrated_source(*, reshape_left=False, transpose_right=False, zero_point=0):
    m, k, n = 2, 3, 4
    left_storage = f"1x{m}x{k}" if reshape_left else f"{m}x{k}"
    right_storage = f"{n}x{k}" if transpose_right else f"{k}x{n}"
    left_layout = (
        (f"%a = tensor.collapse_shape %ad [[0, 1], [2]] : tensor<1x{m}x{k}xf32> into tensor<{m}x{k}xf32>")
        if reshape_left
        else ""
    )
    right_layout = (
        (
            f"%be = tensor.empty() : tensor<{k}x{n}xf32>\n"
            f"%b = linalg.transpose ins(%bd : tensor<{n}x{k}xf32>) "
            f"outs(%be : tensor<{k}x{n}xf32>) permutation = [1, 0]"
        )
        if transpose_right
        else ""
    )
    left, right = ("%a" if reshape_left else "%ad"), ("%b" if transpose_right else "%bd")
    return parse_mlir_text(f"""func.func @forward(%aq: tensor<{left_storage}xi8>,
        %bq: tensor<{right_storage}xi8>, %as: tensor<f32>, %bs: tensor<f32>) -> tensor<{m}x{n}xf32> {{
      %zp = arith.constant {zero_point} : i64
      %z = tensor.splat %zp : tensor<i64>
      %ad = "quant_ext.dequantize_per_tensor"(%aq, %as, %z)
        <{{quant_min = -128 : i64, quant_max = 127 : i64, input_dtype = i8}}>
        : (tensor<{left_storage}xi8>, tensor<f32>, tensor<i64>) -> tensor<{left_storage}xf32>
      %bd = "quant_ext.dequantize_per_tensor"(%bq, %bs, %z)
        <{{quant_min = -128 : i64, quant_max = 127 : i64, input_dtype = i8}}>
        : (tensor<{right_storage}xi8>, tensor<f32>, tensor<i64>) -> tensor<{right_storage}xf32>
      {left_layout}
      {right_layout}
      %e = tensor.empty() : tensor<{m}x{n}xf32>
      %zero = arith.constant 0.0 : f32
      %f = linalg.fill ins(%zero : f32) outs(%e : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
      %r = linalg.matmul ins({left}, {right} : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>)
        outs(%f : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
      return %r : tensor<{m}x{n}xf32>
    }}""")


@pytest.mark.parametrize("reshape_left,transpose_right", [(False, False), (True, False), (False, True), (True, True)])
def test_actual_calibrated_reuse_matches_source_producers(tmp_path, lean_tools, reshape_left, transpose_right):
    before = calibrated_source(reshape_left=reshape_left, transpose_right=transpose_right)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report = check_module(before, after, tmp_path, lean_tools, scope="captured")
    assert report["conditional_match"], report
    # Captured i64 zero-point semantics are not in the frozen source contract. A complete
    # source-derived match must never hide that separate support/definedness refusal.
    assert not report["qualification"]["qualification_passed"]


def test_unknown_calibration_contract_refuses_instead_of_blessing_output(tmp_path, lean_tools):
    before = calibrated_source(zero_point=1)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report = check_module(before, after, tmp_path, lean_tools, scope="captured")
    assert not report["expanded"] and not report["conditional_match"]


def test_unknown_dequantization_cannot_narrow_captured_reach(tmp_path, lean_tools):
    from xdsl.dialects.builtin import StringAttr

    before = calibrated_source()
    for operation in before.walk():
        if getattr(getattr(operation, "op_name", None), "data", "") == "quant_ext.dequantize_per_tensor":
            operation.attributes["op_name__"] = StringAttr("quant_ext.dequantize_per_group")
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report = check_module(before, after, tmp_path, lean_tools, scope="captured")
    assert not report["expanded"] and not report["conditional_match"]

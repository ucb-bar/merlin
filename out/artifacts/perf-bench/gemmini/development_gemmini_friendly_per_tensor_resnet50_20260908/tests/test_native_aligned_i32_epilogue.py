"""Regression gates for the declared native-aligned accumulator epilogue."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mlir_oot.frontend.integer_prepare import prepare_int8_text
from mlir_oot.gemmini_opt import Pipeline


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (ROOT.parent / "development_phase2_native_aligned_per_tensor_i8_20260908"
           / "validation/native_aligned_capture/linalg.mlir")


def _source(*, bias_axis: int = 1, relu: bool = False) -> str:
    bias_map = f"(d{bias_axis})"
    relu_ops = "" if not relu else r"""
    %re = tensor.empty() : tensor<2x4xf32>
    %r = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%biased : tensor<2x4xf32>) outs(%re : tensor<2x4xf32>) {
      ^bb1(%x: f32, %o: f32):
        %relu_zero = arith.constant 0.0 : f32
        %m = arith.maximumf %x, %relu_zero : f32
        linalg.yield %m : f32
    } -> tensor<2x4xf32>
"""
    quant_input = "%r" if relu else "%biased"
    return rf"""
builtin.module {{ func.func @forward(%aq: tensor<2x3xi8>, %wq: tensor<3x4xi8>,
    %bq: tensor<4xi32>) -> tensor<2x4xi8> {{
  %zi = arith.constant 0 : i64
  %zp = tensor.splat %zi : tensor<i64>
  %asf = arith.constant 0.25 : f32
  %as = tensor.splat %asf : tensor<f32>
  %wsf = arith.constant 0.5 : f32
  %ws = tensor.splat %wsf : tensor<f32>
  %bsf = arith.constant 0.125 : f32
  %bs = tensor.splat %bsf : tensor<f32>
  %osf = arith.constant 0.25 : f32
  %os = tensor.splat %osf : tensor<f32>
  %a = "quant_ext.dequantize_per_tensor"(%aq, %as, %zp)
    <{{quant_min = -128 : i64, quant_max = 127 : i64}}> :
    (tensor<2x3xi8>, tensor<f32>, tensor<i64>) -> tensor<2x3xf32>
  %w = "quant_ext.dequantize_per_tensor"(%wq, %ws, %zp)
    <{{quant_min = -127 : i64, quant_max = 127 : i64}}> :
    (tensor<3x4xi8>, tensor<f32>, tensor<i64>) -> tensor<3x4xf32>
  %b = "quant_ext.dequantize_per_tensor"(%bq, %bs, %zp)
    <{{quant_min = -2147483648 : i64, quant_max = 2147483647 : i64}}> :
    (tensor<4xi32>, tensor<f32>, tensor<i64>) -> tensor<4xf32>
  %e = tensor.empty() : tensor<2x4xf32>
  %z = arith.constant 0.0 : f32
  %f = linalg.fill ins(%z : f32) outs(%e : tensor<2x4xf32>) -> tensor<2x4xf32>
  %mm = linalg.matmul {{prov.region_id = "contract", prov.family = "contraction", prov.op = "matmul"}}
    ins(%a, %w : tensor<2x3xf32>, tensor<3x4xf32>)
    outs(%f : tensor<2x4xf32>) -> tensor<2x4xf32>
  %be = tensor.empty() : tensor<2x4xf32>
  %biased = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->{bias_map}>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}}
    ins(%mm, %b : tensor<2x4xf32>, tensor<4xf32>) outs(%be : tensor<2x4xf32>) {{
    ^bb0(%x: f32, %bv: f32, %o: f32):
      %sum = arith.addf %x, %bv : f32
      linalg.yield %sum : f32
  }} -> tensor<2x4xf32>
  {relu_ops}
  %q = "quant_ext.quantize_per_tensor"({quant_input}, %os, %zp)
    <{{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}}> :
    (tensor<2x4xf32>, tensor<f32>, tensor<i64>) -> tensor<2x4xi8>
  func.return %q : tensor<2x4xi8>
}} }}
"""


@pytest.mark.parametrize("relu", [False, True])
def test_i32_bias_scalar_scale_and_optional_relu_stay_in_one_mesh_task(relu: bool) -> None:
    normalized, preparation = prepare_int8_text(
        _source(relu=relu), native_aligned_epilogue=True)
    fold = preparation["native_aligned_i32_epilogue"]
    assert fold == {
        "admitted": 1,
        "refused": {},
        "source_contract": "i32_bias_then_one_f32_multiplier_roundeven_saturate_i8",
    }

    pipe = Pipeline(normalized, enable_source_conv=True).run()
    assert pipe.declined is None
    cb = pipe.plan.command_buffer
    tasks = cb["params"]["global_program_plan"]["tasks"]
    assert [task["kind"] for task in tasks] == ["contraction"]
    receipt = cb["params"]["target_neutral_quantized_epilogues"]
    assert receipt["selected_count"] == 1
    assert receipt["formed"][0]["numeric_contract"]["bias_domain"] == "accumulator_i32"
    assert receipt["formed"][0]["numeric_contract"]["multiplier"] == pytest.approx(0.5)
    commit = cb["commands"][-1]
    assert commit["attributes"]["epilogue"] == [
        "bias", "acc_scale", *(["relu"] if relu else [])]
    assert commit["attributes"]["acc_scale"] == pytest.approx(0.5)
    assert commit["attributes"]["bias"] in cb["tensors"]
    contraction = next(item for item in pipe.plan.tasks if hasattr(item, "epilogue"))
    assert tasks[0]["reads"] == [contraction.lhs, contraction.rhs,
                                  commit["attributes"]["bias"]]
    assert contraction.epilogue.bias is not None
    assert contraction.epilogue.output_dtype == "i8"
    bias_loads = [ins for ins in pipe.instrs
                  if ins.kind == "mvin" and ins.bufs == [contraction.epilogue.bias]]
    assert len(bias_loads) == 1
    stores = [ins for ins in pipe.instrs if ins.kind == "config_st"]
    assert stores[-1].attrs["acc_scale"] == pytest.approx(0.5)
    assert stores[-1].attrs["acc_act"] == int(relu)


def test_wrong_bias_axis_fails_closed() -> None:
    # Shape is deliberately transposed so a valid length-2 row bias exists but Gemmini's
    # repeating D preload cannot represent it for C[M,N].
    source = _source().replace("%bq: tensor<4xi32>", "%bq: tensor<2xi32>")
    source = source.replace("tensor<4xi32>, tensor<f32>", "tensor<2xi32>, tensor<f32>")
    source = source.replace("tensor<4xf32>", "tensor<2xf32>").replace("(d1)>", "(d0)>")
    normalized, preparation = prepare_int8_text(source, native_aligned_epilogue=True)
    assert preparation["native_aligned_i32_epilogue"] == {
        "admitted": 0,
        "refused": {"bias_axis_not_gemmini_column": 1},
        "source_contract": "i32_bias_then_one_f32_multiplier_roundeven_saturate_i8",
    }
    pipe = Pipeline(normalized, enable_source_conv=True).run()
    assert pipe.declined is None
    assert pipe.plan.command_buffer["params"]["target_neutral_quantized_epilogues"][
        "selected_count"] == 0


def test_native_contract_rewrite_is_explicitly_opt_in() -> None:
    _normalized, preparation = prepare_int8_text(_source())
    assert preparation["native_aligned_i32_epilogue"] == {"enabled": False}


@pytest.mark.parametrize(
    "edit,reason",
    [
        (("%bsf = arith.constant 0.125 : f32",
          "%bsf = arith.constant 0.126 : f32"),
         "bias_scale_is_not_exact_accumulator_scale"),
        (("  %b = \"quant_ext.dequantize_per_tensor\"(%bq, %bs, %zp)",
          "  %bone = arith.constant 1 : i64\n"
          "  %bzp = tensor.splat %bone : tensor<i64>\n"
          "  %b = \"quant_ext.dequantize_per_tensor\"(%bq, %bs, %bzp)"),
         "bias_is_not_symmetric_i32_accumulator_units"),
    ],
)
def test_numeric_contract_mismatches_are_counted_refusals(
        edit: tuple[str, str], reason: str) -> None:
    source = _source().replace(*edit)
    _normalized, preparation = prepare_int8_text(source, native_aligned_epilogue=True)
    assert preparation["native_aligned_i32_epilogue"]["admitted"] == 0
    assert preparation["native_aligned_i32_epilogue"]["refused"] == {reason: 1}


def test_declared_roundeven_saturate_and_relu_boundary_values() -> None:
    accumulators = np.asarray(
        [-300, -257, -256, -255, -3, -2, -1, 0, 1, 2, 3, 253, 254, 255, 256, 300],
        dtype=np.int32)
    bias = np.asarray([0, 1, -1, 0] * 4, dtype=np.int32)
    scaled = (accumulators + bias).astype(np.float32) * np.float32(0.5)
    saturated = np.rint(scaled).clip(-128, 127).astype(np.int8)
    assert saturated.tolist() == [
        -128, -128, -128, -128, -2, 0, -1, 0, 0, 2, 1, 126, 127, 127, 127, 127]
    assert np.maximum(saturated, np.int8(0)).tolist() == [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 126, 127, 127, 127, 127]


@pytest.mark.skipif(not CAPTURE.exists(), reason="full public native-aligned capture unavailable")
def test_full_capture_structural_contract_is_53_refusals_plus_terminal_fc() -> None:
    # Run in a fresh process: this is the exact public full-model source and exercises the
    # production parser/pass registry rather than a hand-assembled internal operation graph.
    code = r"""
import json, sys
from pathlib import Path
from mlir_oot.frontend.integer_prepare import prepare_int8_text
_text, report = prepare_int8_text(
    Path(sys.argv[1]).read_text(), native_aligned_epilogue=True)
print(json.dumps(report["native_aligned_i32_epilogue"], sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(CAPTURE)], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["admitted"] == 1
    assert report["refused"] == {"bias_axis_not_gemmini_column": 53}

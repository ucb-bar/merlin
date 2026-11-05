"""An explicit input ABI split retains the original quantization arithmetic."""

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.entry_quantization_split import split_entry_quantization
from merlin.xdsl_dialects._common import text


SOURCE = """module {func.func @forward(%w:tensor<2x7xi32>,%x:tensor<2x7xf32>)->tensor<2x7xi32>{
  %s=arith.constant 0.125:f32
  %ss=tensor.splat %s:tensor<f32>
  %z=arith.constant 3:i64
  %zz=tensor.splat %z:tensor<i64>
  %q="quant_ext.quantize_per_tensor"(%x,%ss,%zz)<{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}> {prov.region_id="input_q"}:(tensor<2x7xf32>,tensor<f32>,tensor<i64>)->tensor<2x7xi8>
  %e=tensor.empty():tensor<2x7xi32>
  %r=linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}
    ins(%q,%w:tensor<2x7xi8>,tensor<2x7xi32>) outs(%e:tensor<2x7xi32>) {
    ^bb0(%v:i8,%w0:i32,%unused:i32):
      %a=arith.extsi %v:i8 to i32
      %b=arith.muli %a,%w0:i32
      linalg.yield %b:i32
    } ->tensor<2x7xi32>
  return %r:tensor<2x7xi32>}}
"""


def split(source=SOURCE):
    return split_entry_quantization(parse_mlir_text(source), entry="forward", argument_index=1,
                                    preparer_entry="prepare_input")


def test_original_abi_constant_operations_and_unmoved_forward_remain_exact():
    original = parse_mlir_text(SOURCE)
    before = text(original, generic=True)
    result = split_entry_quantization(original, entry="forward", argument_index=1,
                                      preparer_entry="prepare_input")
    assert text(original, generic=True) == before
    assert str(result.forward.body.block.first_op.body.block.args[0].type) == "tensor<2x7xi32>"
    assert str(result.forward.body.block.first_op.body.block.args[1].type) == "tensor<2x7xi8>"
    assert "quant_ext.quantize_per_tensor" not in text(result.forward, generic=True)
    assert "quant_ext.quantize_per_tensor" in text(result.preparer, generic=True)
    assert 'prov.region_id = "input_q"' in text(result.preparer, generic=True)
    from merlin.llvmlower.passes_xdsl import lower_quant_ext
    from merlin.runtime.linalg_numpy import evaluate

    values = np.array([-1e9, -16.375, -16, -0.4375, -0.3125, -0.0625, -0.,
                       0., 0.0625, 0.1875, 0.3125, 15, 15.5625, 1e9], np.float32).reshape(2, 7)
    weight = np.arange(-7, 7, dtype=np.int32).reshape(2, 7)
    lower_quant_ext(original)
    prep, forward = result.preparer.clone(), result.forward.clone()
    lower_quant_ext(prep)
    prepared = evaluate(prep, [values], function="prepare_input")[0]
    actual = evaluate(forward, [weight, prepared])[0]
    expected = evaluate(original, [weight, values])[0]
    np.testing.assert_array_equal(actual, expected)


def test_shared_original_constant_remains_in_forward():
    source = SOURCE.replace("return %r:tensor<2x7xi32>", "return %r,%ss:tensor<2x7xi32>,tensor<f32>")
    source = source.replace("->tensor<2x7xi32>{", "->(tensor<2x7xi32>,tensor<f32>){", 1)
    result = split(source)
    assert "tensor.splat" in text(result.forward, generic=True)
    result.forward.verify()


def test_dense_original_scalar_parameters_are_retained():
    source = SOURCE.replace("%s=arith.constant 0.125:f32\n  %ss=tensor.splat %s:tensor<f32>",
                            "%ss=arith.constant dense<0.125>:tensor<f32>")
    source = source.replace("%z=arith.constant 3:i64\n  %zz=tensor.splat %z:tensor<i64>",
                            "%zz=arith.constant dense<3>:tensor<i64>")
    result = split(source)
    assert "dense<" in text(result.preparer)
    assert "arith.constant" not in text(result.forward)


@pytest.mark.parametrize("mutation", ["extra_input_use", "dynamic_parameter", "unknown_semantics", "shape", "scale", "internal_call", "index"])
def test_missing_exact_entry_boundary_refuses(mutation):
    source = SOURCE
    index = 1
    if mutation == "extra_input_use":
        source = source.replace("return %r:tensor<2x7xi32>", "return %r,%x:tensor<2x7xi32>,tensor<2x7xf32>")
        source = source.replace("->tensor<2x7xi32>{", "->(tensor<2x7xi32>,tensor<2x7xf32>){", 1)
    elif mutation == "dynamic_parameter":
        source = source.replace("%s=arith.constant 0.125:f32", "%s=arith.constant 0.125:f32\n  %s2=arith.addf %s,%s:f32").replace("tensor.splat %s:", "tensor.splat %s2:")
    elif mutation == "unknown_semantics":
        source = source.replace('prov.region_id="input_q"', 'unmodeled_effect="unknown"')
    elif mutation == "shape":
        source = source.replace("tensor<2x7xf32>", "tensor<?x7xf32>")
    elif mutation == "scale":
        source = source.replace("0.125:f32", "0.0:f32")
    elif mutation == "internal_call":
        source = source.replace("%s=arith.constant", "%recursive=func.call @forward(%w,%x):(tensor<2x7xi32>,tensor<2x7xf32>)->tensor<2x7xi32>\n  %s=arith.constant", 1)
    else:
        index = 7
    with pytest.raises(ValueError):
        split_entry_quantization(parse_mlir_text(source), entry="forward", argument_index=index,
                                 preparer_entry="prepare_input")


def test_native_original_and_prepared_forward_composition(tmp_path, monkeypatch):
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.lower import lower_model

    if not toolchain.available():
        pytest.skip("selected native compiler unavailable")
    monkeypatch.setenv("MERLIN_SPLIT_FRESH_DEALLOCS", "0")
    values = np.array([-1e9, -16.375, -16, -0.4375, -0.3125, -0.0625, -0.,
                       0., 0.0625, 0.1875, 0.3125, 15, 15.5625, 1e9], np.float32).reshape(2, 7)
    weight = np.arange(-7, 7, dtype=np.int32).reshape(2, 7)
    selection = split()

    def run(module, name, arguments, output, directory):
        lowered = lower_model(text(module, generic=True), directory, targets=("host",))
        image = HostModel.load(str(lowered.host_so), name=name,
                               image_policy=PrivateHostImagePolicy(directory.resolve()))
        image([(a.ctypes.data, a.shape) for a in (*arguments, output)])

    expected = np.empty_like(weight)
    run(parse_mlir_text(SOURCE), "forward", [weight, values], expected, tmp_path / "original")
    prepared = np.empty(values.shape, np.int8)
    run(selection.preparer, "prepare_input", [values], prepared, tmp_path / "prepare")
    actual = np.empty_like(expected)
    run(selection.forward, "forward", [weight, prepared], actual, tmp_path / "forward")
    np.testing.assert_array_equal(prepared, np.clip(np.rint(values * np.float32(8)) + 3, -128, 127).astype(np.int8))
    np.testing.assert_array_equal(actual, expected)

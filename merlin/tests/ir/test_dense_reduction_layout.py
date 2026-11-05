"""Channel-last contraction alternatives preserve source values and observations."""

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms.certificates import ir_text
from merlin.transforms.dense_reduction_layout import rewrite_module


def source(channels=5, height=2, width=3):
    k = height * width
    count = channels * k
    return f"""module {{
 func.func @forward(%x:tensor<1x{height}x{width}x{channels}xi8>,
                    %w:tensor<{k}x1xi8>) -> tensor<{channels}x1xi32> {{
  %empty=tensor.empty():tensor<1x{channels}x{height}x{width}xi8>
  %t=linalg.transpose ins(%x:tensor<1x{height}x{width}x{channels}xi8>)
      outs(%empty:tensor<1x{channels}x{height}x{width}xi8>) permutation=[0,3,1,2]
  %flat=tensor.collapse_shape %t [[0,1,2,3]] :
      tensor<1x{channels}x{height}x{width}xi8> into tensor<{count}xi8>
  %a=tensor.expand_shape %flat [[0,1]] output_shape [{channels},{k}] :
      tensor<{count}xi8> into tensor<{channels}x{k}xi8>
  %z=arith.constant 0:i32
  %init=tensor.splat %z:tensor<{channels}x1xi32>
  %r=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,
      affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
      iterator_types=["parallel","parallel","reduction"]}}
      ins(%a,%w:tensor<{channels}x{k}xi8>,tensor<{k}x1xi8>)
      outs(%init:tensor<{channels}x1xi32>) {{
  ^bb0(%av:i8,%bv:i8,%acc:i32):
   %aa=arith.extsi %av:i8 to i32
   %bb=arith.extsi %bv:i8 to i32
   %p=arith.muli %aa,%bb:i32
   %s=arith.addi %acc,%p:i32
   linalg.yield %s:i32
  }} -> tensor<{channels}x1xi32>
  return %r:tensor<{channels}x1xi32>
 }} }}"""


def biased_source(channels=5, height=2, width=3):
    text = source(channels, height, width)
    shape = f"tensor<1x{channels}x{height}x{width}xi8>"
    insertion = f"""
  %bias=arith.constant dense<3>:tensor<{channels}xi8>
  %broadcast=tensor.expand_shape %bias [[0,1,2,3]] output_shape [1,{channels},1,1]:
      tensor<{channels}xi8> into tensor<1x{channels}x1x1xi8>
  %e_bias=tensor.empty():{shape}
  %biased=linalg.generic {{indexing_maps=[affine_map<(n,c,h,w)->(n,c,h,w)>,
      affine_map<(n,c,h,w)->(0,c,0,0)>,affine_map<(n,c,h,w)->(n,c,h,w)>],
      iterator_types=["parallel","parallel","parallel","parallel"]}}
      ins(%t,%broadcast:{shape},tensor<1x{channels}x1x1xi8>) outs(%e_bias:{shape}) {{
  ^bb1(%v:i8,%bias_value:i8,%unused:i8):
   %add=arith.addi %v,%bias_value:i8
   linalg.yield %add:i8
  }} -> {shape}
"""
    return text.replace("  %flat=", insertion+"  %flat=").replace("collapse_shape %t", "collapse_shape %biased")


def quantized_source(channels=5, height=2, width=3):
    text = source(channels, height, width)
    before, rest = text.split("  %flat=", 1)
    before = before.replace("xi8>", "xf32>")
    before = before.replace(f"%w:tensor<{height*width}x1xf32>", f"%w:tensor<{height*width}x1xi8>")
    shape = f"tensor<1x{channels}x{height}x{width}xi8>"
    quantize = f'''
  %scale=arith.constant dense<0.5>:tensor<f32>
  %zero=arith.constant dense<0>:tensor<i64>
  %q="quant_ext.quantize_per_tensor"(%t,%scale,%zero)
     <{{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}}>:
     (tensor<1x{channels}x{height}x{width}xf32>,tensor<f32>,tensor<i64>)->{shape}
'''
    return before+quantize+"  %flat="+rest.replace("collapse_shape %t", "collapse_shape %q", 1)


def test_layout_selection_keeps_unlowered_quantization_boundaries():
    from merlin.common import mlir_query as mq

    module = parse_mlir_text(quantized_source())
    assert rewrite_module(module).rewritten == 1
    quantize, = (op for op in module.walk() if mq.op_name(op) == "quant_ext.quantize_per_tensor")
    assert quantize.results[0].type.get_shape() == (1, 2, 3, 5)
    assert not any(op.name == "linalg.transpose" for op in module.walk())
    assert quantize.properties["quant_min"].value.data == -128
    assert quantize.properties["quant_max"].value.data == 127


def test_axis_dependent_quantization_remains_a_layout_boundary():
    module = parse_mlir_text(quantized_source().replace("quantize_per_tensor", "quantize_per_channel"))
    before = ir_text(module)
    assert rewrite_module(module).rewritten == 0
    assert ir_text(module) == before


def test_existing_channel_layout_survives_projected_broadcasts():
    module = parse_mlir_text(biased_source())
    report = rewrite_module(module)
    assert report.rewritten == report.preferred_layouts == 1
    assert not any(op.name == "linalg.transpose" for op in module.walk())
    module.verify()


@pytest.mark.parametrize("shape", [(5, 2, 3), (17, 1, 7), (32, 7, 7)])
def test_exact_layout_rewrite_keeps_reduction_order_and_external_shape(shape):
    module = parse_mlir_text(source(*shape))
    report = rewrite_module(module)
    assert report.rewritten == 1
    assert report.eliminated_transpose_bytes == np.prod(shape)
    assert not any(op.name == "linalg.transpose" for op in module.walk())
    product = next(op for op in module.walk() if op.name == "linalg.generic")
    assert product.results[0].type.get_shape() == (1, shape[0])
    assert [op.name for op in product.body.block.ops] == [
        "arith.extsi", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"]
    module.verify()


def test_source_orientation_survives_immutable_ownership_and_host_operand_binding():
    from fake_quant_layer import Oracle
    from merlin.llvmlower.device_offload import _device_operands
    from merlin.llvmlower.group_host_call import _logical_group_call
    from merlin.xdsl_dialects.lowering import compute_groups as CG, group_command as GC
    from merlin.xdsl_dialects.lowering.group_prepack import device_weight

    module = parse_mlir_text(source())
    assert rewrite_module(module).rewritten == 1
    group, = (g for g in CG.form_groups(module, "synthetic", oracle=Oracle()) if g.placement != CG.HOST)
    stated = GC.program(group, weight_args={1})
    assert (stated.stored_operand, stated.stored_arg, stated.transposed) == (0, 1, False)
    assert tuple(stated.entry[key] for key in ("M", "K", "N")) == (1, 6, 5)
    assert GC.stationary_is_activation(stated.entry)
    pair, why = _device_operands(group, stated)
    assert not why and pair == tuple(group.root.operands[:2])
    original_weight = np.arange(6, dtype=np.int8).reshape(6, 1)
    prepared = device_weight(group, stated, original_weight)
    np.testing.assert_array_equal(prepared, original_weight.T)
    operands, constants, roles, why = _logical_group_call(
        group, stated, stated.entry, pair, {"weight": {"array": "immutable_left"}},
        {"immutable_left": prepared}, "")
    assert not why and roles == ["input_0", "weight_0"]
    assert operands[1] is pair[1]
    assert len(constants) == 1 and operands[0] is constants[0].results[0]
    assert tuple(operands[0].type.get_shape()) == (1, 6)
    assert tuple(operands[1].type.get_shape()) == (6, 5)
    assert tuple(constants[0].value.get_values()) == tuple(original_weight.reshape(-1))


def test_ordinary_group_offload_preserves_source_selected_orientation(tmp_path, monkeypatch):
    from logical_device_target import install
    from test_device_shim_logical_abi import _write_capture
    from merlin.llvmlower.device_offload import rewrite_groups_to_device

    install(monkeypatch)
    original = source()
    _write_capture(tmp_path, original, {"ones": np.ones((6, 1), dtype=np.int8)})
    module = parse_mlir_text(original)
    assert rewrite_module(module).rewritten == 1
    routed = rewrite_groups_to_device(module, "gemmini", select=lambda _: True,
                                      capture=tmp_path / "model.mlir")
    assert len(routed.routed) == 1, routed.skipped
    entry, = routed.entries.values()
    assert tuple(entry[key] for key in ("M", "K", "N")) == (1, 6, 5)
    assert entry["stationary_operand"] == "activation"
    program, = routed.programs.values()
    assert (program["stored_operand"], program["stored_arg"], program["transposed"]) == (0, 1, False)
    module.verify()


def test_orientation_request_does_not_admit_transposing_index_maps():
    from fake_quant_layer import Oracle
    from merlin.xdsl_dialects.lowering import compute_groups as CG, group_command as GC

    module = parse_mlir_text(source())
    assert rewrite_module(module).rewritten == 1
    root = next(op for op in module.walk() if op.name == "linalg.generic")
    # Keep identical static types but permute the coordinates; a selection
    # annotation cannot confer a matrix ABI the real source does not have.
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr
    from xdsl.ir.affine import AffineMap, AffineDimExpr

    root.properties["indexing_maps"] = ArrayAttr([
        AffineMapAttr(AffineMap(3, 0, (AffineDimExpr(2), AffineDimExpr(0)))),
        *tuple(root.indexing_maps)[1:],
    ])
    group, = (g for g in CG.form_groups(module, "synthetic", oracle=Oracle()) if g.placement != CG.HOST)
    with pytest.raises(CG.NoCapsuleForm, match="orientation"):
        GC.program(group, weight_args={1})


@pytest.mark.parametrize("change", [
    lambda text: text.replace("arith.extsi %av", "arith.extui %av"),
    lambda text: text.replace("arith.addi %acc,%p", "arith.subi %acc,%p"),
    lambda text: text.replace("[0,3,1,2]", "[0,3,2,1]").replace("1x5x2x3", "1x5x3x2"),
    lambda text: text.replace("module {", "module {func.func private @opaque()", 1)
                          .replace("  %z=", "  func.call @opaque():()->()\n  %z="),
])
def test_unknown_recurrence_order_and_opaque_effects_keep_source(change):
    module = parse_mlir_text(change(source()))
    before = ir_text(module)
    assert rewrite_module(module).rewritten == 0
    assert ir_text(module) == before


@pytest.mark.parametrize("shape,biased", [((5, 2, 3), False), ((17, 1, 7), False),
                                        ((5, 2, 3), True), ((5, 2, 3), "quantized")])
def test_native_original_and_transformed_match_all_values_and_preserve_inputs(tmp_path, shape, biased):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import build_host_shared
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.target_data_layout import _query
    from merlin.llvmlower.toolchain import clang

    c, h, w = shape
    original = (quantized_source(*shape) if biased == "quantized"
                else biased_source(*shape) if biased else source(*shape))
    module = parse_mlir_text(original)
    assert rewrite_module(module).rewritten == 1
    x = ((np.arange(h*w*c, dtype=np.int32)*53+7) % 256-128).astype(np.int8).reshape(1,h,w,c)
    if biased == "quantized":
        x = x.astype(np.float32)*np.float32(0.75)
    weight = ((np.arange(h*w, dtype=np.int32)*31+3) % 256-128).astype(np.int8).reshape(h*w,1)
    values = (np.clip(np.rint(x*np.float32(2)), -128, 127).astype(np.int8) if biased == "quantized"
              else (x.astype(np.int16)+3).astype(np.int8) if biased else x)
    expected = values.reshape(h*w,c).astype(np.int32).T @ weight.astype(np.int32)
    before_x, before_weight = x.copy(), weight.copy()
    for name, text in (("control", original), ("candidate", ir_text(module))):
        symbol = f"weighted_{c}_{h}_{w}_{biased}_{name}"
        text = text.replace("@forward(", "@"+symbol+"(").replace(
            'sym_name = "forward"', 'sym_name = "'+symbol+'"')
        lowered = lower_model(text, tmp_path/name, targets=(), index_bits=64,
                              data_layout=_query(clang(), ("-O2", "-fPIC")))
        library = build_host_shared(lowered.ll_path, tmp_path/(symbol+".so"))
        invoke = HostModel.load(str(library), name=symbol)
        guard = np.full(c+16, 0x5A5A5A5A, np.int32)
        output = guard[8:-8].reshape(c,1)
        invoke([(value.ctypes.data, value.shape) for value in (x, weight, output)])
        np.testing.assert_array_equal(output, expected, err_msg=name)
        np.testing.assert_array_equal(x, before_x)
        np.testing.assert_array_equal(weight, before_weight)
        assert np.all(guard[:8] == 0x5A5A5A5A) and np.all(guard[-8:] == 0x5A5A5A5A)

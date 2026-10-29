"""Permutation propagation preserves broadcast coordinates and scalar order."""

from xdsl.ir.affine import AffineExpr, AffineMap

from merlin.common.mlir_query import parse
from merlin.llvmlower.layout_propagation import rewrite_module

BASE = """builtin.module {
 func.func @f(%x: tensor<1x2x3x4xf32>, %bias: tensor<4xf32>) -> tensor<1x2x3x4xf32> {
  %e = tensor.empty() : tensor<1x4x2x3xf32>
  %t = linalg.transpose ins(%x:tensor<1x2x3x4xf32>) outs(%e:tensor<1x4x2x3xf32>) permutation = [0, 3, 1, 2]
  %o = tensor.empty() : tensor<1x4x2x3xf32>
  %a = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d1)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>], iterator_types = ["parallel","parallel","parallel","parallel"]} ins(%t,%bias:tensor<1x4x2x3xf32>,tensor<4xf32>) outs(%o:tensor<1x4x2x3xf32>) {
  ^bb0(%v:f32,%b:f32,%init:f32):
    %sum = arith.addf %v,%b : f32
    %r = math.roundeven %sum : f32
    linalg.yield %r : f32
  } -> tensor<1x4x2x3xf32>
  %e2 = tensor.empty() : tensor<1x2x3x4xf32>
  %r = linalg.transpose ins(%a:tensor<1x4x2x3xf32>) outs(%e2:tensor<1x2x3x4xf32>) permutation = [0, 2, 3, 1]
  func.return %r : tensor<1x2x3x4xf32>
 }
}"""


def test_scalar_quantization_keeps_exact_coordinate_and_parameter_expression():
    from merlin.llvmlower.layout_conformance import layout_expression

    source = '''module {
 func.func @f(%x:tensor<2x3xf32>,%s:tensor<f32>,%z:tensor<i64>)->tensor<2x3xi8>{
  %e=tensor.empty():tensor<3x2xf32>
  %t=linalg.transpose ins(%x:tensor<2x3xf32>) outs(%e:tensor<3x2xf32>) permutation=[1,0]
  %q="quant_ext.quantize_per_tensor"(%t,%s,%z)
    <{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}>:
    (tensor<3x2xf32>,tensor<f32>,tensor<i64>)->tensor<3x2xi8>
  %o=tensor.empty():tensor<2x3xi8>
  %r=linalg.transpose ins(%q:tensor<3x2xi8>) outs(%o:tensor<2x3xi8>) permutation=[1,0]
  return %r:tensor<2x3xi8>
 }}'''
    control, candidate = parse(source), parse(source)
    report = rewrite_module(candidate, pointwise_only=True)
    assert report.scalar_regions == 1 and report.canceled == 1
    assert layout_expression(control) == layout_expression(candidate)
    quantize = next(op for op in candidate.walk() if getattr(getattr(op, "op_name", None), "data", "")
                    == "quant_ext.quantize_per_tensor")
    from xdsl.dialects.builtin import IntegerAttr, i64
    quantize.properties["quant_max"] = IntegerAttr(126, i64)
    assert layout_expression(control) != layout_expression(candidate)


def result_op(module):
    return next(op for op in module.walk() if op.name == "func.return").operands[0].owner


def test_broadcast_channel_axis_changes_and_scalar_order_stays_exact():
    m = parse(BASE)
    report = rewrite_module(m)
    m.verify()
    op = result_op(m)
    assert op.name == "linalg.generic"
    assert op.operands[0] is next(x for x in m.walk() if x.name == "func.func").body.block.args[0]
    assert op.properties["indexing_maps"].data[1].data == AffineMap(4, 0, (AffineExpr.dimension(3),))
    assert [x.name for x in op.regions[0].block.ops] == ["arith.addf", "math.roundeven", "linalg.yield"]
    assert report.canceled == 1 and report.scalar_regions == 1


def test_static_padding_offsets_sizes_and_destination_follow_axes():
    source = BASE.replace(
        "  %e2 = tensor.empty()",
        """  %zero = arith.constant -7.250000e+00 : f32
  %pad = tensor.splat %zero : tensor<1x4x4x5xf32>
  %padded = "tensor.insert_slice"(%a,%pad) <{static_offsets = array<i64: 0,0,1,1>, static_sizes = array<i64: 1,4,2,3>, static_strides = array<i64: 1,1,1,1>, operandSegmentSizes = array<i32: 1,1,0,0,0>}> : (tensor<1x4x2x3xf32>,tensor<1x4x4x5xf32>) -> tensor<1x4x4x5xf32>
  %e2 = tensor.empty()""",
    )
    source = source.replace("%a:tensor<1x4x2x3xf32>) outs(%e2", "%padded:tensor<1x4x4x5xf32>) outs(%e2")
    source = source.replace("tensor<1x2x3x4xf32>", "tensor<1x4x5x4xf32>")
    # Keep the input tensor shape independent of the padded result.
    source = source.replace("%x: tensor<1x4x5x4xf32>", "%x: tensor<1x2x3x4xf32>")
    source = source.replace("%x:tensor<1x4x5x4xf32>", "%x:tensor<1x2x3x4xf32>")
    m = parse(source)
    rewrite_module(m)
    op = result_op(m)
    assert op.name == "tensor.insert_slice"
    assert list(op.properties["static_offsets"].get_values()) == [0, 1, 1, 0]
    assert list(op.properties["static_sizes"].get_values()) == [1, 2, 3, 4]
    assert op.operands[1].owner.name == "tensor.splat"
    assert op.operands[1].type.get_shape() == (1, 4, 5, 4)


def test_index_observing_region_keeps_explicit_boundary_transpose():
    source = BASE.replace("    %sum = arith.addf", "    %idx = linalg.index 1 : index\n    %sum = arith.addf")
    m = parse(source)
    report = rewrite_module(m)
    assert result_op(m).name == "linalg.transpose"
    assert report.scalar_regions == 0
    assert report.boundary_transposes == 2


def test_coordinate_and_work_checks_share_producers_without_stale_metadata():
    from merlin.llvmlower.layout_conformance import layout_expression

    source = BASE.replace("  %e2 = tensor.empty()", """  %o2 = tensor.empty() : tensor<1x4x2x3xf32>
  %b = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>], iterator_types = ["parallel","parallel","parallel","parallel"]} ins(%a:tensor<1x4x2x3xf32>) outs(%o2:tensor<1x4x2x3xf32>) {
  ^bb0(%v:f32,%init:f32):
    %r = arith.negf %v : f32
    linalg.yield %r : f32
  } -> tensor<1x4x2x3xf32>
  %e2 = tensor.empty()""").replace("ins(%a:tensor<1x4x2x3xf32>) outs(%e2", "ins(%b:tensor<1x4x2x3xf32>) outs(%e2")
    original, candidate = parse(source), parse(source)
    before, work = layout_expression(original, include_work=True)
    assert sum(work.values()) == 3 * 24
    rewrite_module(candidate)
    assert (before, work) == layout_expression(candidate, include_work=True)
    negate = next(op for op in candidate.walk() if op.name == "arith.negf")
    from xdsl.dialects.builtin import StringAttr
    negate.attributes["semantic_marker"] = StringAttr("changed")
    assert before != layout_expression(candidate)

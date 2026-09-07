"""The pointwise-im2col view rewrite: exact real pattern, refusals, and frozen default."""
from __future__ import annotations

from merlin.common import mlir_query as mq
from merlin.llvmlower import im2col_identity_view as iv
from merlin.llvmlower import im2col_pack as ip
from merlin.llvmlower import impr_features as F
from merlin.llvmlower import perop_blocks as pb


def _module(*, n=1, c=4, kh=1, kw=1, oh=3, ow=5, sh=1, sw=1):
    ih, iw = (oh - 1) * sh + kh, (ow - 1) * sw + kw
    k, m, f = c * kh * kw, n * oh * ow, 8
    return f"""
module {{
  func.func @forward(%in: tensor<{n}x{c}x{ih}x{iw}xi8>, %w: tensor<{f}x{k}xi8>)
      -> tensor<{f}x{m}xi32> {{
    %c0 = arith.constant 0 : i32
    %e = tensor.empty() : tensor<{c}x{kh}x{kw}x{n}x{oh}x{ow}xi8>
    %g = linalg.generic {{
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5)
                           -> (d3, d0, d4 * {sh} + d1, d5 * {sw} + d2)>,
                         affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}}
      ins(%in : tensor<{n}x{c}x{ih}x{iw}xi8>)
      outs(%e : tensor<{c}x{kh}x{kw}x{n}x{oh}x{ow}xi8>) {{
    ^bb0(%a: i8, %o: i8):
      linalg.yield %a : i8
    }} -> tensor<{c}x{kh}x{kw}x{n}x{oh}x{ow}xi8>
    %flat = tensor.collapse_shape %g [[0, 1, 2, 3, 4, 5]]
      : tensor<{c}x{kh}x{kw}x{n}x{oh}x{ow}xi8> into tensor<{k * m}xi8>
    %col = tensor.expand_shape %flat [[0, 1]] output_shape [{k}, {m}]
      : tensor<{k * m}xi8> into tensor<{k}x{m}xi8>
    %ae = tensor.empty() : tensor<{f}x{m}xi32>
    %acc = linalg.fill ins(%c0 : i32) outs(%ae : tensor<{f}x{m}xi32>) -> tensor<{f}x{m}xi32>
    %mm = linalg.generic {{
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                         affine_map<(d0, d1, d2) -> (d2, d1)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]}}
      ins(%w, %col : tensor<{f}x{k}xi8>, tensor<{k}x{m}xi8>)
      outs(%acc : tensor<{f}x{m}xi32>) {{
    ^bb0(%x: i8, %y: i8, %o: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %p = arith.muli %xe, %ye : i32
      %s = arith.addi %p, %o : i32
      linalg.yield %s : i32
    }} -> tensor<{f}x{m}xi32>
    func.return %mm : tensor<{f}x{m}xi32>
  }}
}}
"""


def test_feature_is_registered_and_default_off():
    assert iv.ensure_registered() == iv.FEATURE
    feat = F.get(iv.FEATURE)
    assert feat.edit_pipeline is None and feat.edit_schedule is None


def test_real_n1_pointwise_pattern_becomes_one_row_major_view():
    module = mq.parse(_module())
    report = iv.rewrite_module(module)
    assert report.to_dict() == {"viewed": 1, "refusals": {}}
    text = str(module)
    assert "merlin.im2col_identity_view" in text
    assert "tensor<4x15xi8>" in text
    assert "tensor<4x1x1x1x3x5xi8>" not in text
    assert text.count("linalg.generic") == 1  # only the contraction remains


def test_nonidentity_geometry_is_counted_and_untouched():
    for kwargs in ({"kh": 3, "kw": 3}, {"sh": 2, "sw": 2}, {"n": 2}):
        module = mq.parse(_module(**kwargs))
        before = str(module)
        report = iv.rewrite_module(module)
        assert report.viewed == 0
        assert report.refusals == {"refused_nonidentity_geometry": 1}
        assert str(module) == before


def test_viewed_pointwise_and_panel_packed_spatial_paths_compose_without_overlap():
    pointwise = mq.parse(_module())
    assert iv.rewrite_module(pointwise).viewed == 1
    # The view still feeds the ordinary register-blocked matmul; panel packing must not recreate a
    # copy for it.
    point_table = {pb.shape_key("linalg.matmul", (8, 15), (4,)): (4, 15)}
    assert ip.rewrite_module(pointwise, point_table).packed == 0

    spatial = mq.parse(_module(kh=3, kw=3))
    assert iv.rewrite_module(spatial).viewed == 0
    spatial_table = {pb.shape_key("linalg.matmul", (8, 15), (36,)): (4, 15)}
    assert ip.rewrite_module(spatial, spatial_table).packed == 1

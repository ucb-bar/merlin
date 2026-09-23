"""A convolution in generic form is work, and a census weighted by work has to see it."""

from __future__ import annotations

from merlin.common import mlir_query as mq
from merlin.targetgen import model_coverage as MC

_CONV = """
#in = affine_map<(n, f, oh, ow, c, kh, kw) -> (n, c, oh + kh, ow + kw)>
#wt = affine_map<(n, f, oh, ow, c, kh, kw) -> (f, c, kh, kw)>
#out = affine_map<(n, f, oh, ow, c, kh, kw) -> (n, f, oh, ow)>
func.func @conv(%x: tensor<1x3x8x8xf32>, %w: tensor<4x3x3x3xf32>,
                %o: tensor<1x4x6x6xf32>) -> tensor<1x4x6x6xf32> {
  %r = linalg.generic {indexing_maps = [#in, #wt, #out],
        iterator_types = ["parallel", "parallel", "parallel", "parallel",
                          "reduction", "reduction", "reduction"]}
      ins(%x, %w : tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>)
      outs(%o : tensor<1x4x6x6xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %m = arith.mulf %a, %b : f32
      %s = arith.addf %c, %m : f32
      linalg.yield %s : f32
  } -> tensor<1x4x6x6xf32>
  return %r : tensor<1x4x6x6xf32>
}
"""


def test_a_generic_convolution_carries_its_multiply_accumulates() -> None:
    (region,) = MC.regions_from_module(mq.parse(_CONV))
    assert (region.m, region.k, region.n) == (1 * 4 * 6, 3 * 3 * 3, 6)
    assert region.m * region.k * region.n == 3888


def test_an_elementwise_generic_is_not_priced_as_a_contraction() -> None:
    elementwise = _CONV.replace('"reduction", "reduction", "reduction"', '"parallel", "parallel", "parallel"')
    (region,) = MC.regions_from_module(mq.parse(elementwise))
    assert region.m is None and region.k is None

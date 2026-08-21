"""Contraction coverage prices the contractions the matcher missed.

The load-bearing case is `test_a_contracted_extent_that_also_appears_in_the_output_is_counted`. The
obvious way to find a reduced extent -- take the input dimension that is missing from the output --
silently undercounted attention's `scores.V` by 196x on the real model, because that op contracts
over a second 196 while a 196 is also in its result. The extents must come from the indexing maps.
"""
from __future__ import annotations

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.xdsl_dialects.lowering.contraction_coverage import (
    classify_generic,
    contraction_coverage,
)

# scores.V shape: out 4x196x64, contracting over a SECOND 196. This is the shape that breaks the
# shape-subtraction shortcut, so it is the shape the test uses.
AV = """
module {
  func.func @forward(%s: tensor<4x196x196xf32>, %v: tensor<4x196x64xf32>) -> tensor<4x196x64xf32> {
    %e = tensor.empty() : tensor<4x196x64xf32>
    %o = linalg.generic {indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                                          affine_map<(b, m, n, k) -> (b, k, n)>,
                                          affine_map<(b, m, n, k) -> (b, m, n)>],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
         ins(%s, %v : tensor<4x196x196xf32>, tensor<4x196x64xf32>)
         outs(%e : tensor<4x196x64xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %m = arith.mulf %a, %b : f32
      %d = arith.addf %c, %m : f32
      linalg.yield %d : f32
    } -> tensor<4x196x64xf32>
    return %o : tensor<4x196x64xf32>
  }
}
"""

ELEMENTWISE = """
module {
  func.func @forward(%x: tensor<8xf32>) -> tensor<8xf32> {
    %e = tensor.empty() : tensor<8xf32>
    %o = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                         iterator_types = ["parallel"]}
         ins(%x : tensor<8xf32>) outs(%e : tensor<8xf32>) {
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %a : f32
      linalg.yield %s : f32
    } -> tensor<8xf32>
    return %o : tensor<8xf32>
  }
}
"""

MOVEMENT = """
module {
  func.func @forward(%x: tensor<8xf32>) -> tensor<8xf32> {
    %e = tensor.empty() : tensor<8xf32>
    %o = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                         iterator_types = ["parallel"]}
         ins(%x : tensor<8xf32>) outs(%e : tensor<8xf32>) {
    ^bb0(%a: f32, %b: f32):
      linalg.yield %a : f32
    } -> tensor<8xf32>
    return %o : tensor<8xf32>
  }
}
"""

MATMUL = """
module {
  func.func @forward(%a: tensor<8x4xf32>, %b: tensor<4x16xf32>) -> tensor<8x16xf32> {
    %e = tensor.empty() : tensor<8x16xf32>
    %m = linalg.matmul ins(%a, %b : tensor<8x4xf32>, tensor<4x16xf32>)
                       outs(%e : tensor<8x16xf32>) -> tensor<8x16xf32>
    return %m : tensor<8x16xf32>
  }
}
"""


def _cov(src: str):
    with IR_LOCK:
        return contraction_coverage(mq.parse(src))


def _only_generic(src: str):
    with IR_LOCK:
        return next(iter(mq.walk(mq.parse(src), "linalg.generic")))


def test_a_contracted_extent_that_also_appears_in_the_output_is_counted():
    """4*196*64 output elements, each contracting 196 -> 9,834,496 MACs, not 4*196*64 = 50,176."""
    rep = _cov(AV)
    assert len(rep.unlowered) == 1
    got = rep.unlowered[0]
    assert got.macs == 4 * 196 * 64 * 196 == 9_834_496
    assert dict(got.loop_extents) == {0: 4, 1: 196, 2: 64, 3: 196}
    assert rep.unpriceable == []


def test_a_matmul_is_lowered_work_not_a_miss():
    rep = _cov(MATMUL)
    assert rep.lowered_macs == 8 * 16 * 4
    assert rep.unlowered == []
    assert rep.unlowered_share == 0.0


def test_the_share_is_relative_to_all_contraction_work():
    with IR_LOCK:
        both = contraction_coverage(mq.parse(AV.replace("@forward", "@a")))
    assert both.unlowered_macs == 9_834_496
    assert both.lowered_macs == 0
    # every contraction missed => the whole denominator is the miss
    assert both.unlowered_share == 1.0


def test_elementwise_and_movement_are_not_contractions():
    assert classify_generic(_only_generic(ELEMENTWISE)) == "elementwise"
    assert classify_generic(_only_generic(MOVEMENT)) == "movement"
    for src in (ELEMENTWISE, MOVEMENT):
        rep = _cov(src)
        assert rep.unlowered == []
        assert rep.total_macs == 0
        assert rep.unlowered_share == 0.0, "no work must not read as 100% missed"


def test_a_reduction_with_a_multiply_add_body_is_a_contraction():
    assert classify_generic(_only_generic(AV)) == "contraction"


def test_labels_count_every_generic_seen():
    rep = _cov(AV)
    assert rep.labels == {"contraction": 1}

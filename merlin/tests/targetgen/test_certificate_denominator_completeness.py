"""A contraction the matcher never matched is in NEITHER side of the recall — so the recall reads high.

Both halves of Acceleratable Region Recall are built from the routing demands, and the demands come from
the matmul matcher. A contraction that stays a ``linalg.generic`` therefore never becomes a demand: it is
not a false fallback, not an ineligible acceleration, just absent. Measured on ``spectformer_int8_full``,
that absence was 8.5% of the model's contraction MACs — attention's Q·Kᵀ and scores·V, sixteen ops, the
canonical matrix-unit workload — and the published recall was computed without them.

These tests pin the correction: the certificate prices the unmatched mass from the module itself and
states a recall FLOOR beside the headline number, so a reader gets a bracket instead of one end of one.
"""
from __future__ import annotations

from merlin.targetgen import coverage_certificate as cc
from merlin.targetgen.compute_units import SemanticCapability
from merlin.targetgen.routing import OpDemand, RouteResult

# One matmul the matcher DOES see (4x8 by 8x16) beside one attention-shaped generic it does not
# (out 2x6x4, contracting a second 6). The generic is the shape that breaks the naive
# "reduced extent = the input dim missing from the output" shortcut, which is why it is the one used.
MIXED = """
module {
  func.func @forward(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>,
                     %s: tensor<2x6x6xf32>, %v: tensor<2x6x4xf32>)
      -> (tensor<4x16xf32>, tensor<2x6x4xf32>) {
    %e0 = tensor.empty() : tensor<4x16xf32>
    %mm = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
                        outs(%e0 : tensor<4x16xf32>) -> tensor<4x16xf32>
    %e1 = tensor.empty() : tensor<2x6x4xf32>
    %g = linalg.generic {indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                                          affine_map<(b, m, n, k) -> (b, k, n)>,
                                          affine_map<(b, m, n, k) -> (b, m, n)>],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
         ins(%s, %v : tensor<2x6x6xf32>, tensor<2x6x4xf32>)
         outs(%e1 : tensor<2x6x4xf32>) {
    ^bb0(%x: f32, %y: f32, %c: f32):
      %p = arith.mulf %x, %y : f32
      %d = arith.addf %c, %p : f32
      linalg.yield %d : f32
    } -> tensor<2x6x4xf32>
    return %mm, %g : tensor<4x16xf32>, tensor<2x6x4xf32>
  }
}
"""

MATCHED_ONLY = """
module {
  func.func @forward(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
    %e = tensor.empty() : tensor<4x16xf32>
    %mm = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
                        outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %mm : tensor<4x16xf32>
  }
}
"""

_MATMUL_MACS = 4 * 8 * 16       # 512, what the matcher saw
_GENERIC_MACS = 2 * 6 * 4 * 6   # 288, what it did not


def _plan_one_accelerated_matmul():
    """A plan whose single region is the matched matmul, routed to the accelerator."""
    d = OpDemand(op="matmul", in_fmt="f32", weight_fmt="f32", site="mm", m=4, k=8, n=16)
    r = RouteResult(demand=d, unit="mesh", acc="f32", gap=None)
    return {"results": [r], "mesh": [r], "fallback": [], "scalar_rvv": []}


def _cap_map():
    return {"contraction": SemanticCapability(family="contraction", dtypes=("f32",))}


def test_unmatched_contraction_is_priced_from_the_module():
    got = cc.denominator_completeness(MIXED)
    assert got["matched_contraction_macs"] == _MATMUL_MACS
    assert got["unmatched_contraction_macs"] == _GENERIC_MACS
    assert got["n_unmatched_contractions"] == 1
    assert abs(got["unmatched_contraction_share"]
               - _GENERIC_MACS / (_MATMUL_MACS + _GENERIC_MACS)) < 1e-9
    assert got["caveats"], "an unmatched contraction must be stated, not just counted"


def test_recall_floor_is_below_the_headline_when_the_matcher_missed_work():
    """The whole point: the same compilation reports two numbers that bracket the truth."""
    cert = cc.build(_plan_one_accelerated_matmul(), _cap_map(), target=None, linalg_mlir=MIXED)
    m = cert["metrics"]
    assert m["acceleratable_flop_recall"] == 1.0, "every region the matcher produced was accelerated"
    floor = m["acceleratable_flop_recall_lower_bound"]
    # 2*512 accelerated over 2*(512+288): the unmatched mass charged entirely to the denominator.
    assert abs(floor - (2 * _MATMUL_MACS) / (2 * (_MATMUL_MACS + _GENERIC_MACS))) < 1e-9
    assert floor < m["acceleratable_flop_recall"]
    assert cert["unmatched_contraction_flops"] == 2 * _GENERIC_MACS, "a MAC is two flops, not one"


def test_region_recall_has_a_floor_too_not_just_flop_recall():
    """REGION recall is the number that actually gets quoted, and it was the one with no floor.

    The flop floor existed from the start; the region one did not, so the most-cited figure in the
    certificate was also the only headline metric with nothing bracketing it. One eligible region was
    matched and accelerated (recall 1.0) while one contraction stayed a `linalg.generic` and never
    became a demand -- charge it to the denominator and the floor is 1/2.
    """
    cert = cc.build(_plan_one_accelerated_matmul(), _cap_map(), target=None, linalg_mlir=MIXED)
    m = cert["metrics"]
    assert m["acceleratable_region_recall"] == 1.0
    assert m["acceleratable_region_recall_lower_bound"] == 0.5
    assert m["acceleratable_region_recall_lower_bound"] < m["acceleratable_region_recall"]
    assert cert["unmatched_contraction_regions"] == 1


def test_every_headline_recall_carries_a_bound():
    """Structural, so a recall added later cannot ship without one: each `*_recall` key has a
    `*_recall_lower_bound` sibling. A new unbracketed headline is exactly how the last one got in."""
    m = cc.build(_plan_one_accelerated_matmul(), _cap_map(), target=None, linalg_mlir=MIXED)["metrics"]
    recalls = [k for k in m if k.endswith("_recall")]
    assert recalls, "the certificate reports no recall at all -- the metric block moved"
    missing = [k for k in recalls if f"{k}_lower_bound" not in m]
    assert not missing, f"recall(s) reported with no lower bound beside them: {missing}"


def test_the_two_numbers_coincide_when_nothing_was_missed():
    """A bound that is always pessimistic would be ignored; it must be tight when the matcher is complete."""
    cert = cc.build(_plan_one_accelerated_matmul(), _cap_map(), target=None, linalg_mlir=MATCHED_ONLY)
    m = cert["metrics"]
    assert m["acceleratable_flop_recall_lower_bound"] == m["acceleratable_flop_recall"] == 1.0
    assert cert["denominator_completeness"]["n_unmatched_contractions"] == 0
    assert cert["denominator_completeness"]["caveats"] == []


def test_no_module_reports_the_absence_rather_than_implying_completeness():
    """Omitting the module must not read as 'the matcher missed nothing'."""
    cert = cc.build(_plan_one_accelerated_matmul(), _cap_map(), target=None)
    assert cert["denominator_completeness"] is None
    assert cert["unmatched_contraction_flops"] == 0
    assert cert["metrics"]["acceleratable_flop_recall_lower_bound"] == \
        cert["metrics"]["acceleratable_flop_recall"]


def test_an_unparseable_module_surfaces_the_error_instead_of_vanishing():
    """Fail closed: a certificate that silently drops the block looks exactly like a complete one."""
    cert = cc.build(_plan_one_accelerated_matmul(), _cap_map(), target=None,
                    linalg_mlir="this is not MLIR {{{")
    got = cert["denominator_completeness"]
    assert got is not None and "error" in got
    assert "UNKNOWN" in got["note"]


def test_the_certificate_still_builds_when_the_module_cannot_be_priced():
    """The completeness probe is advisory; it must never take the recall down with it."""
    cert = cc.build(_plan_one_accelerated_matmul(), _cap_map(), target=None, linalg_mlir="{{{")
    assert cert["metrics"]["acceleratable_region_recall"] == 1.0
    assert cert["n_regions"] == 1

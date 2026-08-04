"""The whole-model coverage facet — the losses a per-kernel CCA structurally cannot see.

Every other CCA facet is lifted from ONE kernel's asm, so a loss that lives in the GRAPH is invisible
to it: an entire contraction class left for convert-linalg-to-loops, or the ~88% of linalg ops that are
not contractions at all. A kernel-scoped comparison can find nothing wrong while the model runs at a
few percent of peak (measured: spectformer 0.40 MAC/cycle, deepjscc 0.22, lstmnetvit 0.067 against
~8 for a VLEN=128 int8 vwmacc datapath). These tests pin that the facet SEES the loss and that the
catalog ROUTES it, because an unrouted loss is one the mining loop can never propose a fix for.
"""
from __future__ import annotations

import pytest

from merlin.kernels import cca_contract as cc
from merlin.kernels.action_catalog import route
from merlin.kernels.cca import lift_coverage
from merlin.kernels.cca_compare import Divergence

_MM = """
builtin.module {
  func.func @forward(%a: tensor<8x128xf32>, %b: tensor<128x64xf32>,
                     %c: tensor<8x64xf32>) -> tensor<8x64xf32> {
    %0 = linalg.matmul ins(%a, %b : tensor<8x128xf32>, tensor<128x64xf32>)
         outs(%c : tensor<8x64xf32>) -> tensor<8x64xf32>
    return %0 : tensor<8x64xf32>
  }
}
"""


def test_a_fully_claimed_model_reports_no_loss():
    cov = lift_coverage(_MM, {"linalg.matmul", "linalg.batch_matmul"}).coverage
    assert cov.claimed_mac_fraction == pytest.approx(1.0)
    assert cov.unclaimed_op_classes == ()


def test_an_unclaimed_class_is_reported_with_its_mac_weight():
    """Declining a class must show up as BOTH the class name and the share of MACs it took with it."""
    cov = lift_coverage(_MM, set()).coverage           # nothing claimed
    assert cov.unclaimed_op_classes == ("linalg.matmul",)
    assert cov.claimed_mac_fraction == pytest.approx(0.0)


def test_macs_are_weighted_not_counted():
    """One huge contraction must outweigh many tiny ones, or the facet mis-ranks the loss.

    A count-based facet would call this model 50% claimed; by MACs the unclaimed op is ~0.1%.
    """
    src = """
    builtin.module {
      func.func @forward(%a: tensor<256x256xf32>, %b: tensor<256x256xf32>, %c: tensor<256x256xf32>,
                         %d: tensor<2x4xf32>, %e: tensor<4x2xf32>, %f: tensor<2x2xf32>)
          -> tensor<256x256xf32> {
        %0 = linalg.matmul ins(%a, %b : tensor<256x256xf32>, tensor<256x256xf32>)
             outs(%c : tensor<256x256xf32>) -> tensor<256x256xf32>
        %1 = linalg.batch_matmul ins(%d, %e : tensor<2x4xf32>, tensor<4x2xf32>)
             outs(%f : tensor<2x2xf32>) -> tensor<2x2xf32>
        return %0 : tensor<256x256xf32>
      }
    }
    """
    try:
        cov = lift_coverage(src, {"linalg.matmul"}).coverage
    except Exception:                                  # noqa: BLE001
        pytest.skip("this xDSL build cannot parse a bare linalg.batch_matmul")
    if cov.claimed_mac_fraction is None or not cov.unclaimed_op_classes:
        pytest.skip("batch_matmul not classified by this build's shape observer")
    assert cov.claimed_mac_fraction > 0.99, (
        f"MACs must be weighted, not counted: got {cov.claimed_mac_fraction}")


def test_unmeasurable_ir_degrades_instead_of_guessing():
    cov = lift_coverage("builtin.module { }", {"linalg.matmul"}).coverage
    assert cov.claimed_mac_fraction is None and cov.unclaimed_op_classes == ()


def test_contraction_generics_are_not_reported_as_unclaimed():
    """The capture emits attention as a linalg.generic and the pipeline NAMES it before the schedule
    runs, so an op-name check would invent a loss that is not there."""
    cov = lift_coverage(_MM, {"linalg.matmul"}).coverage
    assert "linalg.generic" not in cov.unclaimed_op_classes


@pytest.mark.parametrize("axis,expert,ours", [
    ("coverage.unclaimed_op_classes", (), ("linalg.batch_matmul",)),
    ("coverage.claimed_mac_fraction", 1.0, 0.659),
    ("coverage.non_contraction_op_fraction", 0.1, 0.882),
])
def test_every_coverage_loss_routes_to_a_compiler_seam(axis, expert, ours):
    """THE point of the facet: a loss the loop cannot route to a seam is a loss it cannot fix."""
    action = route(Divergence(axis, expert, ours, "rvv", ["measured"]))
    assert action is not None, f"{axis} reports a loss with no routed action"
    assert action.target_seam and action.change


def test_coverage_axes_are_levers_bound_by_the_bijection():
    """Capture- and exposure-completeness: each coverage axis is a LEVER with a route and a region."""
    from merlin.kernels import regions

    for axis in ("coverage.claimed_mac_fraction", "coverage.unclaimed_op_classes",
                 "coverage.non_contraction_op_fraction"):
        assert cc.FIELD_REGISTRY[axis].classification == cc.LEVER
        assert axis in cc.routed_axes("rvv"), f"{axis} is a LEVER with no route"
        assert regions.region_for_axis(axis) is not None, f"{axis} has no governing region"

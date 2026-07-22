"""Workstream A: the default-off Gemmini codegen feature registry + its action_catalog routing.

Mirrors ``test_impr_features`` for RVV: the load-bearing invariant is that with no features enabled the
codegen options are byte-identical to the frozen baseline, so a feature only changes codegen when a fork
explicitly turns it on. Also checks that the two seeded features route from a SpatialFacet divergence to
their concrete gemmini seam file (the "which section do I modify" answer).
"""
from __future__ import annotations

import pytest

from merlin.llvmlower import gemmini_features as gf
from merlin.kernels import action_catalog as ac
from merlin.kernels.cca_compare import Divergence


def test_no_features_is_byte_identical_baseline():
    """features == frozenset() -> options returned UNCHANGED (the frozen WS / non-resident baseline)."""
    base = gf.GemminiCodegenOpts()
    out = gf.apply_opts(base, frozenset())
    assert out == base
    assert out.dataflow == "ws" and out.accumulator_resident is False


def test_each_feature_changes_exactly_its_axis():
    base = gf.GemminiCodegenOpts()
    df = gf.apply_opts(base, frozenset({"gemmini_dataflow_select"}))
    assert df.dataflow == "os" and df.accumulator_resident is False
    ar = gf.apply_opts(base, frozenset({"gemmini_accumulator_resident"}))
    assert ar.accumulator_resident is True and ar.dataflow == "ws"


def test_features_compose_and_are_order_stable():
    both = gf.apply_opts(gf.GemminiCodegenOpts(),
                         frozenset({"gemmini_dataflow_select", "gemmini_accumulator_resident"}))
    assert both.dataflow == "os" and both.accumulator_resident is True


def test_normalize_rejects_unknown_feature():
    with pytest.raises(KeyError):
        gf.normalize(["gemmini_dataflow_select", "not_a_feature"])
    assert gf.normalize(None) == frozenset()


def test_edit_opts_does_not_mutate_input():
    base = gf.GemminiCodegenOpts()
    gf.apply_opts(base, frozenset({"gemmini_dataflow_select"}))
    assert base.dataflow == "ws"  # frozen dataclass; edit returns a new value


def test_spatial_divergence_routes_to_gemmini_seam():
    """A mined SpatialFacet divergence resolves to the concrete gemmini_features seam FILE."""
    d = Divergence(axis="spatial.dataflow", expert="os", ours="ws", backend="gemmini", evidence=[])
    a = ac.route(d)
    assert a is not None and a.action_class == "HEURISTIC"
    loc = ac.seam_location(a.target_seam)
    assert loc["seam_file"].endswith("llvmlower/gemmini_features.py")
    # accumulator-residency routes too, as a PASS.
    d2 = Divergence(axis="spatial.accumulator_resident", expert=True, ours=False, backend="gemmini",
                    evidence=[])
    a2 = ac.route(d2)
    assert a2 is not None and a2.action_class == "PASS"

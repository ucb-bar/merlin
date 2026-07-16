"""WS-C C4: the improvement-category layer (the beam's 'what should we improve?' question)."""
from __future__ import annotations

from merlin.kernels import categories as C, cca_contract
from merlin.kernels.cca_compare import Divergence


def test_check_categories_clean():
    # every RVV lever axis has an improvement category; every mapped category is declared
    assert C.check_categories() == []


def test_every_lever_axis_has_a_category():
    for ax in cca_contract.leverable_axes("rvv"):
        assert C.category_for_axis(ax) in C.CATEGORIES, ax


def test_category_assignment_matches_the_named_buckets():
    assert C.category_for_axis("compute.accumulator_resident") == "register-residency"
    assert C.category_for_axis("compute.contraction_form") == "instruction-selection"
    assert C.category_for_axis("compute.register_block") == "tiling-dataflow"
    assert C.category_for_axis("compute.epilogue") == "fusion-layout"


def test_categorize_groups_divergences_by_what_to_improve():
    divs = [Divergence("compute.accumulator_resident", True, False, "rvv"),
            Divergence("compute.contraction_form", "fused_fma", "mul_add", "rvv"),
            Divergence("vector.lmul", 4.0, 2.0, "rvv")]
    grouped = C.categorize(divs)
    assert set(grouped) == {"register-residency", "instruction-selection"}
    # both lmul + contraction_form are instruction-selection
    assert {d.axis for d in grouped["instruction-selection"]} == {"compute.contraction_form", "vector.lmul"}


def test_uncategorized_axis_not_dropped():
    # an axis with no category lands under None (surfaced, never silently dropped)
    grouped = C.categorize([Divergence("compute.mr_adapts_to_m", True, False, "rvv")])
    assert None in grouped

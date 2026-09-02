"""The generalization claim: how much of Core ATen the captured models contain, and how much routes.

The claim has to be bounded and checkable, and the ways it can quietly become a slogan are specific:
a denominator that is not actually Core ATen, a numerator that swallows regions nobody could classify,
and a precision gap reported as a routing gap. Each is pinned here.
"""

from __future__ import annotations

import pathlib

import pytest

from merlin.common.paths import artifacts_dir
from merlin.targetgen import aten_coverage as AC


def _captures() -> dict:
    root = artifacts_dir() / "recaptures"
    if not root.is_dir():
        return {}
    return {d.name.replace("_fp32_consistent", "").replace("_consistent", ""): d / "model.mlir"
            for d in sorted(root.iterdir()) if (d / "model.mlir").is_file()}


@pytest.fixture(scope="module")
def opset():
    try:
        return AC.core_opset()
    except Exception as exc:                       # noqa: BLE001
        pytest.skip(f"Core ATen opset needs the m2m venv: {exc}")


def test_the_denominator_is_the_core_opset_not_the_decomposition_table(opset):
    """`torch._decomp.core_aten_decompositions()` LOOKS like the Core ATen opset and is its rough
    complement -- the ops decomposed AWAY on the way to Core ATen. Measured: ~1004 entries against 188
    core-tagged overloads. Using it would report coverage of about a fifth of the true figure against
    a set that is not the one meant."""
    assert opset["n_core"] == len(opset["ops"])
    assert 100 < opset["n_core"] < 400, (
        f"{opset['n_core']} core ops is outside the plausible band; if this is ~1000 the decomposition "
        f"table has been used as the denominator")
    assert all(o.startswith("aten.") for o in opset["ops"])
    assert opset["torch"], "the opset is a property of a torch VERSION and must record which"


def test_the_census_separates_core_from_non_core(opset):
    caps = _captures()
    if not caps:
        pytest.skip("no captured models")
    cen = AC.census(caps, opset=opset)
    assert cen["n_observed_core"] <= cen["n_core"]
    core = set(opset["ops"])
    assert not (set(cen["non_core_observed"]) & core), (
        "an op cannot be both core and non-core")
    assert set(cen["observed_core"]) <= core
    # A non-core op a model contains is REAL work the compiler must handle. Folding it into the core
    # count would inflate the numerator against a denominator that never contained it.
    for name, d in cen["per_model"].items():
        if d.get("status") == "ok":
            assert not (set(d["non_core"]) & core), name


def test_unclassified_regions_are_never_folded_into_either_bucket(opset):
    """A region whose family could not be determined is evidence neither of coverage nor of a gap.
    The routing denominator is routed + fallback; unclassified sits beside it."""
    caps = _captures()
    if not caps:
        pytest.skip("no captured models")
    rep = AC.coverage(caps, "gemmini", opset=opset)
    rt = rep["routing"]
    assert rt["denominator"] == rt["routed"] + rt["fallback"]
    assert rt["unclassified"] >= 0
    assert rt["unclassified"] not in (rt["routed"], rt["denominator"]) or rt["unclassified"] == 0


def test_a_precision_gap_is_not_reported_as_a_routing_gap(opset):
    """Precision is judged only over the family-admitted subset. A region whose family is admitted but
    whose dtype is not is a PRECISION gap; reporting it as a routing gap misattributes it -- and on an
    int8-only target against f32 captures that is the majority of regions."""
    caps = _captures()
    if not caps:
        pytest.skip("no captured models")
    rep = AC.coverage(caps, "gemmini", opset=opset)
    assert "precision" in rep and "routing" in rep
    prec = rep["precision"]
    assert prec["dtype_ok"] + prec["dtype_blocked"] <= rep["routing"]["routed"], (
        "precision is judged over the family-admitted subset, so it cannot exceed it")


def test_the_claim_sentence_carries_only_what_the_numbers_support(opset):
    caps = _captures()
    if not caps:
        pytest.skip("no captured models")
    rep = AC.coverage(caps, "gemmini", opset=opset)
    sentence = AC.claim_sentence(rep)
    assert str(rep["opset"]["n_core"]) in sentence
    assert str(rep["observed"]["n_core_observed"]) in sentence
    assert "unclassified" in sentence, "the honest caveat must survive into the claim"


def test_an_unreadable_capture_is_reported_not_counted_as_zero(opset, tmp_path):
    bad = tmp_path / "missing.mlir"
    rep = AC.coverage({"ghost": bad}, "gemmini", opset=opset)
    assert rep["per_model"]["ghost"]["status"] == "unreadable"
    assert rep["routing"]["denominator"] == 0, (
        "a model we could not read contributes no regions, rather than contributing zeros")


# --- the work column ---------------------------------------------------------------------------------
# A region COUNT answers "how many ops" and says nothing about how much of the model those ops are: a
# hundred elementwise regions and one attention matmul are 101 regions and nowhere near 101 units of
# work. The claim was documented with this third number and shipped without it.

def test_the_claim_states_the_work_weighting_when_it_could_price_it():
    sent = AC.claim_sentence({
        "n_models": 2, "opset": {"n_core": 188},
        "observed": {"n_core_observed": 40, "core_fraction": 40 / 188},
        "target": "t",
        "routing": {"routed": 30, "denominator": 40, "routed_fraction": 0.75, "unclassified": 1},
        "work": {"routed_fraction": 0.93, "exact": True},
    })
    assert "93.0% of the roster's total loop-nest work" in sent
    assert "lower bound" not in sent


def test_a_partially_recovered_nest_is_stated_as_a_lower_bound():
    """`iteration_space` reports `complete=False` when a nest was only partially recovered. A lower
    bound that reads as a measurement is how a heavy op gets ranked light."""
    sent = AC.claim_sentence({
        "n_models": 1, "opset": {"n_core": 188},
        "observed": {"n_core_observed": 10, "core_fraction": 10 / 188},
        "target": "t",
        "routing": {"routed": 5, "denominator": 10, "routed_fraction": 0.5, "unclassified": 0},
        "work": {"routed_fraction": 0.5, "exact": False},
    })
    assert "lower bound" in sent


def test_a_claim_that_could_not_price_the_work_says_nothing_about_it():
    """Silence is honest; a stated weighting nobody computed is not."""
    sent = AC.claim_sentence({
        "n_models": 1, "opset": {"n_core": 188},
        "observed": {"n_core_observed": 10, "core_fraction": 10 / 188},
        "target": "t",
        "routing": {"routed": 5, "denominator": 10, "routed_fraction": 0.5, "unclassified": 0},
        "work": {"routed_fraction": None, "exact": False},
    })
    assert "loop-nest work" not in sent and sent.endswith(".")

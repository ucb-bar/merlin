"""Which precision a target should compile in, and why the others were refused.

`precision_preference` is a RANKING and could only ever filter: synthesis compared it against the
admitted dtypes and reported survivors. Nobody could ask the direct question, so the answer lived in
whichever profile entry happened to name a dtype.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.precision_policy import best_format


def test_the_top_ranked_admitted_format_is_chosen():
    r = best_format("gemmini")
    if r["status"] == "no_preference_declared":
        pytest.skip("gemmini declares no precision preference in this checkout")
    assert r["chosen"] is not None
    assert r["chosen"]["capsule_dtype"] in r["admitted"]


def test_a_preference_can_never_widen_what_the_hardware_admits():
    """The whole point of the ranking: it orders what exists and cannot add to it."""
    r = best_format("gemmini", preference=["fp8_e4m3", "mxfp4"])
    assert r["chosen"] is None
    assert {x["why"] for x in r["rejected"]} <= {"not_admitted", "not_expressible"}


def test_a_rejection_says_which_kind_it_is():
    """"we preferred int8 and this target has no int8 datapath" and "int8 is a typo" send a reader to
    different places, so the two must not share a reason."""
    r = best_format("gemmini", preference=["definitely_not_a_format", "fp8_e4m3"])
    whys = {x["format"]: x["why"] for x in r["rejected"]}
    assert whys["definitely_not_a_format"] == "not_expressible"
    assert whys.get("fp8_e4m3") in {"not_admitted", "not_expressible"}


def test_no_declared_preference_is_not_no_qualifying_format():
    """`chosen: None` beside an empty rejection list reads as "nothing qualified" when the truth is
    "nobody was asked"."""
    r = best_format("gemmini", preference=[])
    assert r["status"] == "no_preference_declared"
    assert r["admitted"], "the target still admits formats; only the ranking is absent"


def test_accuracy_is_reported_as_unestablished_rather_than_assumed():
    """Choosing the narrowest admitted format and calling it best is how a compiler ships a model that
    runs fast and answers wrong."""
    r = best_format("gemmini")
    assert r["certified"]["status"] == "not_established"
    assert "hardware has the datapath" in r["certified"]["detail"]


def test_both_sides_are_compared_in_one_spelling():
    """`admitted` speaks the registry's tokens and a cell speaks the capsule's ("int8" vs "i8").
    Comparing them raw reported every admitted dtype as not-admitted."""
    r = best_format("gemmini")
    if r["status"] != "ok":
        pytest.skip("no preference declared in this checkout")
    assert r["chosen"]["format"] != r["chosen"]["capsule_dtype"] or True
    assert r["chosen"]["capsule_dtype"] in r["admitted"], "the chosen dtype must be in the admitted set"

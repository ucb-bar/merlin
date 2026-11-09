from __future__ import annotations

import pytest

from merlin.targetgen.precision_policy import best_format

pytestmark = pytest.mark.target("gemmini")


def test_a_preference_can_never_widen_what_the_hardware_admits():
    """The whole point of the ranking: it orders what exists and cannot add to it."""
    r = best_format("gemmini", preference=["fp8_e4m3", "mxfp4"])
    assert r["chosen"] is None
    assert {x["why"] for x in r["rejected"]} <= {"not_admitted", "not_expressible"}

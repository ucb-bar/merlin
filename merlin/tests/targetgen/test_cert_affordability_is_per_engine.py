"""The certification ceiling is priced per (target, ENGINE), and the mixture it replaced is visible.

WHAT WAS WRONG. ``conformance._cert_affordability`` asked ``cert_cost.fit_for(target)`` with no
``engine=``, which draws ONE line through every certification a target has, whatever answered it.
Measured on this repo's systolic target against the identical ELF, GSIM answers a capsule in 3.31 s
where Verilator takes 86.83 s, so such a line prices a capsule at neither engine's cost. It also did
its own inversion inline -- ``(budget - intercept) / per_element`` -- which bypasses
``cert_cost.max_elements_within`` and with it the extrapolation clamp, the rule that a prediction past
the measured range times the margin is not an answer.

MEASURED WHEN THIS LANDED (2026-09-21, gemmini, 300 s budget):

    mixed, all engines   19,245 elements   r2 0.149   over 3..150,528 measured
    gsim                  6,500 elements   r2 0.230   over 3..6,144
    verilator               923 elements   r2 0.604   over 240..1,024

and the ceiling written into the tracked corpus is 875. So the apparent 22x staleness of that 875 was
never a corpus that got cheaper -- it was two devices added together. Split by engine, the tracked
number is within 6% of the slowest engine's figure. Regenerating on the mixed fit would have raised
every capsule's allowance 22x on the WORST-fitting line of the three.

WHICH ENGINE BINDS. The slowest one the target has history on, because the cap has to be payable on
whichever engine the run picks; the cheapest would emit capsules nobody can afford the moment a run
chooses the other. That rule is asserted below rather than described, and so is the visibility of the
spread -- the per-engine ceilings ride in the returned document, so a reader of a capsule's cap reason
can see which machine priced it.
"""

from __future__ import annotations

import pytest

from merlin.targetgen import cert_affordability as CA
from merlin.targetgen import cert_cost as CC
from merlin.targetgen import corpus_synth as CS
from merlin.targetgen.conformance import _cert_affordability

TARGET = "gemmini"
BUDGET = 300.0


@pytest.fixture(scope="module")
def priced() -> dict:
    return _cert_affordability(TARGET, budget_s=BUDGET)


def test_the_ceiling_names_the_engine_it_belongs_to(priced) -> None:
    """A certification second is a property of the simulator. A ceiling that does not say which one it
    was measured on cannot be compared to the next one, and this number is written into tracked
    capsules where it outlives the run that produced it."""
    if not priced.get("max_elements"):
        pytest.skip("this checkout has no measured certification history for the target")
    assert priced["engine"], "the ceiling must name its engine"
    assert priced["engines"], "the per-engine spread must travel with the chosen figure"
    assert priced["engine"] in priced["engines"]


def test_the_chosen_engine_is_the_one_that_binds(priced) -> None:
    """The slowest engine with a fit, i.e. the smallest ceiling. Choosing the cheapest would size the
    corpus for a machine the run may not use."""
    if not priced.get("max_elements"):
        pytest.skip("this checkout has no measured certification history for the target")
    usable = {e: n for e, n in priced["engines"].items() if n}
    assert priced["max_elements"] == min(usable.values())
    assert usable[priced["engine"]] == priced["max_elements"]


def test_the_extrapolation_clamp_is_not_bypassed(priced) -> None:
    """The ceiling must be exactly what ``cert_cost.max_elements_within`` returns for that engine's
    fit -- the one place the clamp and the "floor alone exceeds the budget" refusal live. The inline
    division this replaced could answer past the evidence and did not say so."""
    if not priced.get("max_elements"):
        pytest.skip("this checkout has no measured certification history for the target")
    fit = CA.fit_for(TARGET, priced["engine"])
    assert fit is not None
    through_the_clamp = CC.max_elements_within(
        CC.CostFit(
            target=fit.target,
            intercept_s=fit.intercept_s,
            per_element_s=fit.per_element_s,
            r2=fit.r2,
            n_samples=fit.n_samples,
            elements_min=fit.elements_min,
            elements_max=fit.elements_max,
            metric=fit.metric,
        ),
        BUDGET,
    )
    assert priced["max_elements"] == through_the_clamp


def test_the_ceiling_is_not_the_mixed_fit(priced) -> None:
    """The regression this is about. A target with more than one engine's history must not be priced
    on the line through both -- and where that line differs, the per-engine answer must win."""
    if not priced.get("max_elements"):
        pytest.skip("this checkout has no measured certification history for the target")
    mixed = CC.fit_for(TARGET)
    if mixed is None or not mixed.mixed_engines:
        pytest.skip("this checkout's history is single-engine, so there is no mixture to avoid")
    assert priced["max_elements"] != CC.max_elements_within(mixed, BUDGET), (
        "the ceiling still equals the mixed-fit answer, so the engine axis is not reaching the caller"
    )


def test_the_spread_between_engines_is_reported_rather_than_averaged(priced) -> None:
    """What makes the staleness of an older tracked ceiling readable instead of a mystery: both
    engines' numbers are in the document, so a reader can see that a 20x jump is a change of machine
    and not a change of corpus."""
    if not priced.get("max_elements"):
        pytest.skip("this checkout has no measured certification history for the target")
    assert priced["basis"].startswith("fitted on")
    assert priced["engine"] in priced["basis"]
    assert "r2" in priced["basis"], "a ceiling that does not carry its fit quality invites being quoted"


def test_the_cap_reason_written_into_a_capsule_names_the_engine() -> None:
    """The reason string is the only thing that survives into the tracked corpus, so the engine has to
    be in it. Two capsules capped years apart are otherwise indistinguishable from a corpus that got
    more expensive."""
    entry: dict = {"name": "T", "M": 64, "K": 16, "N": 64}
    spec_doc = {
        "cert_affordability": {"max_elements": 16, "budget_s": 300.0, "engine": "an_engine"},
        "boundaries": {"tile_edge": 16},
        "oracle_tiers": ["L2", "L3"],
    }
    reason = CS.cap_to_affordable(entry, spec_doc)
    assert reason and "an_engine" in reason
    assert entry["max_oracle_tier"] == "L2"


def test_a_capsule_inside_the_ceiling_is_left_alone() -> None:
    """The other half of the mutation: a cap that fires on everything is not a cap."""
    entry: dict = {"name": "T", "M": 16, "K": 16, "N": 16}
    spec_doc = {
        "cert_affordability": {"max_elements": 4096, "budget_s": 300.0, "engine": "an_engine"},
        "boundaries": {"tile_edge": 16},
        "oracle_tiers": ["L2", "L3"],
    }
    assert CS.cap_to_affordable(entry, spec_doc) is None
    assert "max_oracle_tier" not in entry


def test_an_unmeasured_target_is_still_refused_rather_than_borrowed() -> None:
    """A ``(target, engine)`` pair nobody has certified yields no fit, and the producer must not fill
    that in from another engine or another target."""
    assert CA.fit_for("no_such_target_at_all", "no_such_engine") is None

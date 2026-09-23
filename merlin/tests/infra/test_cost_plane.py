"""Nothing gated on cost; this is the cheapest honest version of a plane that does.

Two decidable rules and one refusal, and the shape of each matters more than the arithmetic:

* a DECLARED ceiling the capsule states, evaluated with the same comparator that bounds every other
  declared quantity;
* a DERIVED floor -- the cycles the array must spend issuing this program's own work -- which needs
  no slack constant and therefore refutes an impossible count without inventing anything;
* and a refusal to compare counts from two designs, or from designs nobody named.

Everything else is ``incomplete``, which is never a pass. The plane lands at the ``report`` phase:
it computes the whole verdict and blocks nothing, because a plane that flips to ``fail`` on day one
is indistinguishable from a regression. The ``fail`` phase is implemented, and the tests below
exercise it -- a gate whose failing path has never run is a gate nobody has shown can fail.
"""

from __future__ import annotations

import pytest

from merlin.perf import cost_plane as CP

TIMING_TIER = "L3"


def _capsule(*, timing_tier: str | None = TIMING_TIER, projected=None, max_timing_tier=None):
    capsule: dict = {}
    if timing_tier is not None:
        capsule["performance"] = {"acceptance": {"evidence": {"timing_tier": timing_tier}}}
    if projected is not None:
        capsule.setdefault("performance", {}).setdefault("cost", {})["projected_cycles"] = projected
    if max_timing_tier is not None:
        capsule[CP.TIMING_CEILING_FIELD] = max_timing_tier
    return capsule


def _tiers(cycles, *, tier: str = TIMING_TIER, **design):
    record = {"status": "pass", "cycles": cycles}
    record.update(
        {
            "substrate": "sim",
            "hw_config": "cfg",
            "hwdb_config_artifact_sha256": "d" * 64,
            "engine": "engine_a",
            **design,
        }
    )
    return {tier: record}


ARRAY = {"array_rows": 16, "array_cols": 16}


# --------------------------------------------------------------------------------------------
# Phase: report lands blocking nothing; fail blocks a decided failure
# --------------------------------------------------------------------------------------------


def test_a_measurement_within_its_declared_ceiling_is_admitted() -> None:
    out = CP.assess(_capsule(projected=5_000), tiers=_tiers(4_000), macs=1_000, **ARRAY)
    assert out["status"] == CP.STATUS_WITHIN
    assert out["admitted"] is True and out["blocking"] is False
    assert out["axis"] == "timing"


def test_the_report_phase_records_the_failure_and_blocks_nothing() -> None:
    """Day one. The verdict is complete and the run is scored by the rule it was scored by before."""
    out = CP.assess(_capsule(projected=5_000), tiers=_tiers(9_000), macs=1_000, phase=CP.PHASE_REPORT, **ARRAY)
    assert out["status"] == CP.STATUS_OVER
    assert out["enforced"] is False and out["blocking"] is False
    assert out["admitted"] is False


def test_the_fail_phase_blocks_a_measurement_over_the_declared_ceiling() -> None:
    """THE MUTATION THAT MUST FAIL: the same capsule, one cycle count past its own ceiling."""
    out = CP.assess(_capsule(projected=5_000), tiers=_tiers(5_001), macs=1_000, phase=CP.PHASE_FAIL, **ARRAY)
    assert out["status"] == CP.STATUS_OVER
    assert out["blocking"] is True
    assert "5001 > max 5000" in out["comparison"]["reason"]


def test_the_ceiling_uses_the_declared_bound_grammar_not_a_second_one() -> None:
    """Exactly at the ceiling passes, as ``{max: N}`` means everywhere else in this repo."""
    assert CP.assess(_capsule(projected=5_000), tiers=_tiers(5_000), macs=1, **ARRAY)["status"] == CP.STATUS_WITHIN


def test_an_unknown_phase_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError, match="phase must be one of"):
        CP.assess(_capsule(), tiers=_tiers(1), phase="enforce")


# --------------------------------------------------------------------------------------------
# The derived floor: refutable with nothing invented
# --------------------------------------------------------------------------------------------


def test_a_count_below_the_arrays_own_issue_floor_is_refuted() -> None:
    """A program cannot finish faster than its sequencer's loop, so such a count is not about it.

    This is the half of the plane that needs no declaration at all: the floor is
    ``work / array slots``, both derived, and no slack factor decides it.
    """
    out = CP.assess(_capsule(), tiers=_tiers(3), macs=16 * 16 * 100, phase=CP.PHASE_FAIL, **ARRAY)
    assert out["status"] == CP.STATUS_BELOW_FLOOR
    assert out["blocking"] is True
    assert out["floor_cycles"] == pytest.approx(100.0)


def test_the_same_program_one_cycle_above_the_floor_is_not_refuted() -> None:
    """The mutation's counterpart: the rule fires on the impossible count and on nothing else."""
    out = CP.assess(_capsule(), tiers=_tiers(100), macs=16 * 16 * 100, phase=CP.PHASE_FAIL, **ARRAY)
    assert out["status"] != CP.STATUS_BELOW_FLOOR


def test_declared_tiles_give_a_tighter_floor_than_macs_alone() -> None:
    """The sequencer's loop charges the partial block; a division over MACs cannot see it."""
    from merlin.perf.mesh_occupancy import tile_issue_cycles

    tiles = [{"rows": 64, "depth": 16, "cols": 4}]
    floor = CP.derived_floor(tiles=tiles, **ARRAY)
    assert floor["cycles"] == float(tile_issue_cycles(64, 16, 4, array_rows=16, array_cols=16))
    assert floor["cycles"] > (64 * 16 * 4) / (16 * 16)


def test_no_array_geometry_leaves_the_floor_unknown_rather_than_zero() -> None:
    """A floor of zero admits every measurement -- a check that cannot fail."""
    from merlin.perf.decompose import is_unknown

    floor = CP.derived_floor(macs=1_000, array_rows=None, array_cols=16)
    assert is_unknown(floor["cycles"])
    assert "array_rows" in floor["reason"]


def test_no_work_and_no_tiles_leaves_the_floor_unknown() -> None:
    from merlin.perf.decompose import is_unknown

    floor = CP.derived_floor(macs=None, **ARRAY)
    assert is_unknown(floor["cycles"])
    assert "nothing to divide" in floor["reason"]


# --------------------------------------------------------------------------------------------
# Incomplete: every state the plane cannot decide says so, and none of them passes
# --------------------------------------------------------------------------------------------


def test_no_declared_ceiling_is_incomplete_and_says_why_no_slack_was_invented() -> None:
    out = CP.assess(_capsule(), tiers=_tiers(9_999), macs=16, **ARRAY)
    assert out["status"] == CP.STATUS_INCOMPLETE
    assert out["admitted"] is False
    assert "slack" in out["reason"]
    assert out["array_efficiency"] is not None, "the ratio is still reported when it can be"


def test_a_capsule_that_declares_no_timing_rung_is_incomplete() -> None:
    out = CP.assess(_capsule(timing_tier=None), tiers=_tiers(1_000), macs=16, **ARRAY)
    assert out["status"] == CP.STATUS_INCOMPLETE
    assert "timing_tier" in out["reason"]


def test_a_member_capped_out_of_the_measurement_matrix_is_an_exclusion_not_a_failure() -> None:
    """``max_timing_tier`` is a deliberate exclusion; reporting it as a cost failure would blame a
    member for a measurement nobody bought."""
    out = CP.assess(
        _capsule(projected=1, max_timing_tier="L2"), tiers=_tiers(9_999), macs=16, phase=CP.PHASE_FAIL, **ARRAY
    )
    assert out["status"] == CP.STATUS_INCOMPLETE
    assert out["blocking"] is False
    assert CP.TIMING_CEILING_FIELD in out["reason"]


def test_a_tier_that_reported_no_cycles_is_incomplete_not_fast() -> None:
    out = CP.assess(_capsule(projected=10), tiers={TIMING_TIER: {"status": "pass"}}, macs=16, **ARRAY)
    assert out["status"] == CP.STATUS_INCOMPLETE
    assert out["measured_cycles"] is None


def test_a_word_in_the_projection_slot_is_not_parsed_into_a_bound() -> None:
    """Every shipped member fills the slot with when a number would arrive, not with one.

    The reason now comes from :mod:`merlin.perf.cycle_bound`, which owns the vocabulary the capsule
    schema and the corpus generator also read, so a word all three admit reads the same way here.
    """
    from merlin.perf import cycle_bound

    ceiling, basis = CP.declared_ceiling(_capsule(projected="derived_at_preflight"))
    assert ceiling is None
    assert cycle_bound.NO_CYCLE_BOUND["derived_at_preflight"] in basis

    # A word the vocabulary does NOT know is a refusal, not silence. This reader used to treat any
    # non-integer as "no ceiling", which made a typo indistinguishable from a declaration -- a
    # capsule that meant to owe a number quietly owed nothing.
    ceiling, basis = CP.declared_ceiling(_capsule(projected="derived_at_prefilght"))
    assert ceiling is None
    assert "neither a cycle count nor a declared reason" in basis

    # ...and a count written as a string looks like a bound to a reviewer while resolving to none.
    ceiling, basis = CP.declared_ceiling(_capsule(projected="4096"))
    assert ceiling is None and "neither a cycle count" in basis

    # MUTATION: an integer is still read as the bound it is.
    assert CP.declared_ceiling(_capsule(projected=4096)) == (
        4096,
        "declared in performance.cost.projected_cycles",
    )


# --------------------------------------------------------------------------------------------
# The refusal: a cycle verdict abstains across designs
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", CP.DESIGN_KEYS)
def test_comparing_cycle_counts_from_two_designs_refuses(key: str) -> None:
    """Inherited from ``firesim_checkpoint.compare``: cycles from different designs are not
    comparable, and an approximate answer is worse than none because it gets cited."""
    now = CP.assess(_capsule(projected=10_000), tiers=_tiers(4_000), macs=16, **ARRAY)
    before = CP.assess(_capsule(projected=10_000), tiers=_tiers(8_000, **{key: "something_else"}), macs=16, **ARRAY)
    if key == "tier":
        before = CP.assess(
            _capsule(timing_tier="L2", projected=10_000), tiers=_tiers(8_000, tier="L2"), macs=16, **ARRAY
        )
    with pytest.raises(CP.CrossDesignComparison, match="differ in " + key):
        CP.compare(now, before)


def test_a_design_nobody_named_refuses_rather_than_matching_another_unnamed_one() -> None:
    """Two records that both say nothing compare EQUAL, which would read as the same machine. That
    is the flattering direction, and the one nobody audits."""
    bare = {TIMING_TIER: {"status": "pass", "cycles": 100}}
    now = CP.assess(_capsule(projected=10_000), tiers=bare, macs=16, **ARRAY)
    before = CP.assess(_capsule(projected=10_000), tiers=bare, macs=16, **ARRAY)
    assert now["design"]["engine"] is None
    with pytest.raises(CP.CrossDesignComparison, match="not stated on both sides"):
        CP.compare(now, before)


def test_two_counts_from_the_same_named_design_compare() -> None:
    now = CP.assess(_capsule(projected=10_000), tiers=_tiers(4_000), macs=16, **ARRAY)
    before = CP.assess(_capsule(projected=10_000), tiers=_tiers(8_000), macs=16, **ARRAY)
    change = CP.compare(now, before)
    assert change["status"] == "measured"
    assert change["ratio"] == pytest.approx(0.5)


def test_the_cost_plane_has_a_production_caller() -> None:
    """A gate nothing calls is indistinguishable from a gate that always passes."""
    from merlin.targetgen import capsule_grade  # noqa: PLC0415

    assert capsule_grade._COST_PLANE is CP

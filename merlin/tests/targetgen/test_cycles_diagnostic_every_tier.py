"""``cycles_diagnostic`` must harvest EVERY tier that reports cycles, not one fixed tier.

The grader used to read the cycle count out of a single hardcoded tier. Which tier actually holds a
capsule's cycle count is a property of the run: the ladder runs the cheapest oracle for THAT target
first, and a capsule can carry a count under one tier and nothing under another. The failure mode is
asymmetric and therefore invisible -- passing capsules clear every tier and are all present, while
failures carry a count only at the tier that refuted them, so the diagnostic reads "failures have no
cycles" when the truth is the opposite.
"""
from __future__ import annotations

import json

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.capsule_grade import cycles_by_tier

LADDER = ("L0", "L1", "L2", "L3", "L4", "L5")


def test_a_count_under_a_tier_other_than_the_cheapest_is_still_harvested():
    # The shape a refuted capsule actually has: the cheap tier never recorded, the elaborated tier did.
    got = cycles_by_tier({"L0": "pass", "L1": "pass", "L2": "pass",
                          "L4": {"status": "fail", "cycles": 1378}}, ladder=LADDER)
    assert got == {"L4": 1378}


def test_two_tiers_are_kept_side_by_side_as_a_comparand():
    got = cycles_by_tier({"L4": {"cycles": 3078}, "L3": {"cycles": 3078}}, ladder=LADDER)
    assert got == {"L3": 3078, "L4": 3078}
    assert list(got) == ["L3", "L4"], "ordered by the ladder, not by the order the tiers ran"


def test_a_tier_outside_the_declared_ladder_is_not_dropped():
    got = cycles_by_tier({"LX": {"cycles": 7}, "L3": {"cycles": 9}}, ladder=LADDER)
    assert got == {"L3": 9, "LX": 7}


def test_a_bare_string_tier_record_reports_nothing_and_is_not_read_as_zero():
    # Persisted score files flatten tier records to bare strings; a string has no fields, so it must
    # contribute no entry at all rather than a fabricated 0.
    assert cycles_by_tier({"L3": "pass", "L4": "fail"}, ladder=LADDER) == {}
    assert cycles_by_tier(None) == {}
    assert cycles_by_tier({}) == {}


def test_a_tier_that_ran_but_reported_no_cycles_contributes_no_entry():
    assert cycles_by_tier({"L3": {"status": "pass"}, "L4": {"status": "pass", "cycles": None}},
                          ladder=LADDER) == {}


# --- replay against a real graded run ---------------------------------------------------------------
# The regression this fixes is quantitative, so it is checked against real per-capsule records rather
# than only synthetic ones. Untracked run output: skipped (loudly) when absent, never silently passed.
_RUN = (repo_root() / "out/runs/atlas/capsule-bench/merlin_assisted/merlincirct_atlassg1"
        / "grading_public/runs/atlas-capsule-bench")


@pytest.mark.skipif(not _RUN.is_dir(),
                    reason=f"DID NOT RUN: graded-run replay corpus absent at {_RUN}")
def test_replay_a_graded_run_recovers_the_failures_a_single_tier_dropped():
    single_tier, every_tier, failures_recovered = 0, 0, 0
    for result_path in sorted(_RUN.glob("*/capsule_result.json")):
        record = json.loads(result_path.read_text())
        tiers = record.get("tiers") or {}
        harvested = cycles_by_tier(tiers, ladder=LADDER)
        if harvested:
            every_tier += 1
        old = tiers.get("L3")  # what a single fixed-tier read would have found
        if isinstance(old, dict) and old.get("cycles") is not None:
            single_tier += 1
        elif harvested and record.get("status") != "pass":
            failures_recovered += 1
    assert every_tier > single_tier, (
        f"replay corpus must contain a count outside the single tier "
        f"(every_tier={every_tier}, single_tier={single_tier})")
    assert failures_recovered >= 1, (
        "the point of the fix: capsules that FAILED carry their cycle count under the tier that "
        "refuted them, and a single-tier harvest loses exactly those")

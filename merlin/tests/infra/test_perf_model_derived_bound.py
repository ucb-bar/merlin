"""The optimisation ORDER is taken on the derived bound, and both ceilings stay in the report.

``rank_headroom`` decides which shape the loop works on next. Ranking it by distance to the OBSERVED
ceiling makes "no headroom left" mean "no better than this loop has already been" -- measured at
31.3% of structural on one device, against 96.2% for a hand-written schedule on the same device. So
the order is taken on the facts-derived bound wherever the target's own facts supply one, while the
observed ceiling stays beside it, because the gap between the two is the finding.

Each test carries a mutation that must make it fail.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from merlin_experiments.phase2 import calibration

from merlin.common.paths import repo_root
from merlin.perf.decompose import UNKNOWN

sys.path.insert(0, str(repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"))
import perf_model as PM  # noqa: E402

DATA = repo_root() / "merlin" / "tests" / "data" / "derived_bound"


def _machine(name: str = "device_a"):
    from merlin.perf import derived_bound as DB

    facts = json.loads((DATA / f"{name}_facts.json").read_text(encoding="utf-8"))
    return DB.machine_from_facts(name, facts=facts, measure_fill=False)


def _points() -> list[calibration.MeasuredPoint]:
    """Two shapes whose SPREAD is small and whose distance from the machine is large.

    Both sit near the best rate this loop has ever produced, which is the situation the observed
    ceiling cannot see out of: it reports them at ~100% attained while both are at a third of the
    array.
    """
    return [
        calibration.MeasuredPoint("nearly_best", 8_000_000, 100_000, "x"),  # 80 mac/cycle
        calibration.MeasuredPoint("best_so_far", 8_100_000, 100_000, "x"),  # 81 mac/cycle
    ]


def test_the_observed_ceiling_calls_both_shapes_finished_and_the_derived_bound_does_not():
    """The defect, stated as a comparison the report now makes for you."""
    points = _points()
    observed = calibration.achievable_ceiling(points, provenance="unit test")
    assert observed.known and observed.value == pytest.approx(81.0)

    machine = _machine()
    by_observed = PM.rank_headroom(points, achievable=observed, structural=256)
    assert all(h.share_of_achievable > 0.98 for h in by_observed), "both look finished against history"

    by_derived = PM.rank_headroom(points, achievable=observed, structural=256, machine=machine)
    assert all(h.share_of_derived < 0.35 for h in by_derived), "against the machine, neither is close"
    assert all(h.factor_to_derived > 3.0 for h in by_derived)
    # Both ceilings survive in the row, which is what makes the gap readable rather than replaced.
    assert all(h.share_of_achievable is not None for h in by_derived)
    assert by_derived[0].ranking_share == by_derived[0].share_of_derived

    # MUTATION: with no machine there is no derived bound and the order falls back to the observed
    # ceiling -- today's behaviour, unchanged, rather than a fabricated bound.
    assert all(h.share_of_derived is None for h in by_observed)
    assert by_observed[0].ranking_share == by_observed[0].share_of_achievable


def test_a_derived_bound_a_point_beats_is_dropped_rather_than_used_to_rank_it():
    """A refuted bound must not steer the loop: it is wrong, and the ranking says so by omission."""
    machine = _machine()
    impossible = calibration.MeasuredPoint("beats_the_array", 8_000_000, 8_000, "x")  # 1000 mac/cycle
    bounds = PM.derived_bounds([impossible], machine)
    assert bounds["beats_the_array"].refuted

    ranked = PM.rank_headroom(
        [impossible, *_points()],
        achievable=calibration.achievable_ceiling(_points(), provenance="unit test"),
        structural=256,
        machine=machine,
    )
    refuted = next(h for h in ranked if h.point.capsule == "beats_the_array")
    assert refuted.derived_rate is None and refuted.share_of_derived is None
    assert refuted.ranking_share == refuted.share_of_achievable

    # MUTATION: a point that does NOT beat the bound keeps its derived rate.
    ok = next(h for h in ranked if h.point.capsule == "best_so_far")
    assert ok.derived_rate == pytest.approx(256.0)


def test_a_facts_bundle_that_cannot_be_read_leaves_todays_behaviour_in_place(tmp_path: Path):
    """No default and no nameplate: an unreadable bundle means no derived bound at all."""
    missing = tmp_path / "nothing.json"
    assert PM.derived_machine(missing, "device_a") is None

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"facts": {}}), encoding="utf-8")
    machine = PM.derived_machine(empty, "device_a")
    assert machine is not None and machine.peak_macs_per_cycle is not None
    assert machine.refusals, "an empty body must say what it could not ground"
    # Nothing resolved, so there is no bound -- not a large one. The ranking must fall back to the
    # observed ceiling rather than be steered by an infinity.
    bound = PM.derived_bounds(_points(), machine)["best_so_far"]
    assert not bound.bounded and bound.rate is UNKNOWN
    ranked = PM.rank_headroom(
        _points(),
        achievable=calibration.achievable_ceiling(_points(), provenance="unit test"),
        structural=256,
        machine=machine,
    )
    assert all(h.derived_rate is None and h.ranking_share == h.share_of_achievable for h in ranked)

    # MUTATION: a bundle that IS readable produces a bound, so the refusal above is about the input.
    good = tmp_path / "good.json"
    good.write_text((DATA / "device_a_facts.json").read_text(encoding="utf-8"), encoding="utf-8")
    assert PM.derived_machine(good, "device_a").peak_macs_per_cycle == 256


def test_the_report_carries_the_gap_between_the_loops_history_and_the_machine():
    """``derived_over_achievable`` is the number this whole exercise is about."""
    points = _points()
    observed = calibration.achievable_ceiling(points, provenance="unit test")
    machine = _machine()
    bounds = PM.derived_bounds(points, machine)
    derived = max(b.partial_rate for b in bounds.values())
    assert derived / float(observed.value) > 3.0

    rendered = PM.render(
        {
            "target": "t",
            "points": len(points),
            "structural_mac_per_cycle": 256,
            "structural_basis": "facts",
            "achievable_mac_per_cycle": float(observed.value),
            "achievable_basis": observed.provenance,
            "achievable_share_of_structural": float(observed.value) / 256,
            "derived_mac_per_cycle": derived,
            "derived_machine": machine.to_dict(),
            "derived_unresolved": sorted(machine.refusals),
            "derived_refuted_by": {},
            "derived_over_achievable": derived / float(observed.value),
            "headroom": [],
            "prediction_error": {"status": "unavailable"},
        }
    )
    assert "derived bound" in rendered and "below it" in rendered
    # The UNKNOWN terms that loosened the bound are named in the report, not silently dropped.
    assert "loosened by UNKNOWN term(s)" in rendered

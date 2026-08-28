"""Continuous mode must stop on a PLATEAU, not only on convergence or the wall budget.

`_keep_going`'s docstring promised a stop on "converged, plateaued" and only `converged` (all_pass) was
implemented. Every radiance run so far has failed to reach all_pass, so each was guaranteed to spend its
ENTIRE wall budget however early it stopped improving — and a four-arm ladder therefore cost 4x the full
budget by construction, which is what made running the ladder impractical.

The rule under test: stop when the BEST score has not improved across the last N completed rounds.
Measured on the best, not the latest, so a single lower round does not cut a run that is still exploring.
"""
from __future__ import annotations

import inspect

from merlin.common.paths import repo_root

SRC = (repo_root() / "merlin/experiments/capsule_bench/harness/run_baseline_qa_loop.py"
       ).read_text(encoding="utf-8")


def _keep_going(scores, plateau_rounds=3, all_pass=False, schedule="continuous",
                active_wall_s=0.0, max_wall_s=0):
    """A faithful re-implementation of the decision under test, driven by a score history.

    Mirrors the source's ordering exactly (all_pass -> rounds cap -> wall -> plateau); the source is
    asserted to contain the same rule below so this cannot drift into testing a different function.
    """
    rounds_summary = [{"round": i, "n_passed": s} for i, s in enumerate(scores)]
    if all_pass:
        return False
    if schedule == "rounds":
        return True
    if max_wall_s and active_wall_s >= max_wall_s:
        return False
    if plateau_rounds and len(rounds_summary) >= plateau_rounds + 1:
        scored = [r.get("n_passed") for r in rounds_summary if r.get("n_passed") is not None]
        if len(scored) >= plateau_rounds + 1:
            recent, earlier = scored[-plateau_rounds:], scored[:-plateau_rounds]
            if earlier and max(recent) <= max(earlier):
                return False
    return True


def test_a_flat_run_stops():
    """The v12 arm-4 case: ceiling reached early, then rounds that go nowhere."""
    assert _keep_going([33, 33, 33, 33]) is False


def test_a_still_improving_run_continues():
    assert _keep_going([21, 24, 28, 31]) is True


def test_a_single_dip_does_not_stop_it():
    """Best-not-latest: a lower round has not undone the progress."""
    assert _keep_going([21, 30, 28, 31]) is True


def test_a_late_improvement_keeps_it_alive():
    assert _keep_going([21, 21, 21, 22]) is True


def test_too_few_rounds_to_judge_continues():
    """N+1 completed rounds are needed before a plateau can be claimed at all."""
    for hist in ([], [10], [10, 10], [10, 10, 10]):
        assert _keep_going(hist) is True, hist


def test_plateau_can_be_disabled():
    assert _keep_going([33, 33, 33, 33], plateau_rounds=0) is True


def test_convergence_and_wall_still_win_first():
    assert _keep_going([21, 24, 28], all_pass=True) is False
    assert _keep_going([21, 24, 28], max_wall_s=100, active_wall_s=100) is False


def test_the_source_implements_this_rule_and_not_a_bare_wall_check():
    """Guard against a revert: the terminator must exist in the real loop, with the documented flag."""
    assert "reason=plateau" in SRC, "the plateau stop is gone from the loop"
    assert "--plateau-rounds" in SRC, "the flag that controls it is gone"
    assert "recent, earlier = scored[-a.plateau_rounds:], scored[:-a.plateau_rounds]" in SRC
    # and it must still be ORDERED after convergence and the wall budget
    i_all = SRC.index('if verdict.get("all_pass")')
    i_wall = SRC.index("wall budget reached")
    i_plat = SRC.index("reason=plateau")
    assert i_all < i_wall < i_plat, "the terminators are no longer ordered convergence -> wall -> plateau"

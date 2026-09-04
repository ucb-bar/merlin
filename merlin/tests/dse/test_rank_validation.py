"""A scorer must earn exposure by measurement, and the default answer is no.

Two signals already failed this bar in practice: the correctness simulator at 46.1% and a per-command
cost model at 39.3%, both below chance on within-workload ordering. The second is the dangerous kind
-- accurate on absolute magnitude, anti-predictive on ordering -- so these tests fix that a perfect
scorer passes, an inverted one fails LOUDLY rather than quietly, and a scorer that decides nothing is
never credited for its ties.
"""
from __future__ import annotations

import pytest

from merlin.perf import rank_validation as RV


def _programs():
    """Two workloads, three programs each; measured time encoded in the name for clarity."""
    return [
        RV.Program(workload="A", program="a1", measured=100.0, group="F"),
        RV.Program(workload="A", program="a2", measured=200.0, group="F"),
        RV.Program(workload="A", program="a3", measured=300.0, group="F"),
        RV.Program(workload="B", program="b1", measured=150.0, group="G"),
        RV.Program(workload="B", program="b2", measured=250.0, group="G"),
        RV.Program(workload="B", program="b3", measured=350.0, group="G"),
    ]


def _perfect():
    return {p.program: p.measured for p in _programs()}


def _inverted():
    return {p.program: -p.measured for p in _programs()}


def test_pairs_are_within_workload_only():
    """Comparing across workloads asks a question the search never asks."""
    pairs = RV.ordered_pairs(_programs())
    assert len(pairs) == 6                       # C(3,2) per workload, two workloads
    assert all(a.workload == b.workload for a, b in pairs)


def test_a_tie_in_the_oracle_is_not_a_pair():
    rows = [RV.Program("A", "x", 100.0), RV.Program("A", "y", 100.0)]
    assert RV.ordered_pairs(rows) == []


def test_a_perfect_scorer_agrees_on_every_decided_pair():
    a = RV.agreement(RV.ordered_pairs(_programs()), _perfect())
    assert a.decided == 6 and a.agreed == 6 and a.rate == 1.0


def test_an_inverted_scorer_scores_zero_not_one():
    """The failure mode that matters: it must read as 0.0, never be silently absolved."""
    a = RV.agreement(RV.ordered_pairs(_programs()), _inverted())
    assert a.rate == 0.0


def test_a_constant_scorer_decides_nothing_and_is_not_credited():
    """Counting ties as agreement is how a function that knows nothing scores 100%."""
    flat = {p.program: 1.0 for p in _programs()}
    a = RV.agreement(RV.ordered_pairs(_programs()), flat)
    assert a.decided == 0 and a.undecided == 6 and a.rate is None


def test_a_missing_score_is_undecided_not_a_guess():
    partial = {"a1": 1.0}
    a = RV.agreement(RV.ordered_pairs(_programs()), partial)
    assert a.agreed == 0 and a.decided == 0


def test_a_margin_trades_decided_count_for_confidence_and_reports_both():
    scores = {"a1": 0.0, "a2": 0.5, "a3": 10.0, "b1": 0.0, "b2": 0.5, "b3": 10.0}
    loose = RV.agreement(RV.ordered_pairs(_programs()), scores, margin=0.0)
    tight = RV.agreement(RV.ordered_pairs(_programs()), scores, margin=1.0)
    assert tight.decided < loose.decided
    assert tight.undecided > loose.undecided


def test_slices_expose_a_scorer_that_learned_only_one_workload():
    """Good overall, useless on a held-out slice, is exactly what leave-one-out is for."""
    scores = dict(_perfect())
    for name in ("b1", "b2", "b3"):              # invert workload B only
        scores[name] = -scores[name]
    per = RV.held_out(_programs(), scores, by="workload")
    assert per["A"].rate == 1.0
    assert per["B"].rate == 0.0


@pytest.mark.parametrize("by", ["workload", "group"])
def test_slicing_works_on_both_keys(by):
    per = RV.held_out(_programs(), _perfect(), by=by)
    assert len(per) == 2 and all(a.rate == 1.0 for a in per.values())


def test_an_unknown_slice_key_is_refused():
    with pytest.raises(ValueError):
        RV.held_out(_programs(), _perfect(), by="whatever")


# ------------------------------------------------------------------ the verdict

def _verdict(scores, **over):
    kwargs = {"minimum_rate": 0.70, "minimum_decided": 4, "minimum_slice_decided": 2}
    kwargs.update(over)
    programs = _programs()
    overall = RV.agreement(RV.ordered_pairs(programs), scores)
    return RV.verdict(overall, RV.held_out(programs, scores), **kwargs)


def test_a_perfect_scorer_with_enough_evidence_is_exposable():
    assert _verdict(_perfect())["exposable"] is True


def test_a_below_chance_scorer_is_refused_and_named_as_such():
    """The 39.3% case. It must not read as merely 'below the bar'."""
    out = _verdict(_inverted())
    assert out["exposable"] is False
    assert any("chance" in r for r in out["reasons"])


def test_too_little_evidence_refuses_however_good_the_rate():
    out = _verdict(_perfect(), minimum_decided=1000)
    assert out["exposable"] is False
    assert any("below the required 1000" in r for r in out["reasons"])


def test_a_scorer_failing_one_slice_is_refused_even_if_overall_passes():
    scores = dict(_perfect())
    for name in ("b1", "b2", "b3"):
        scores[name] = -scores[name]
    out = _verdict(scores, minimum_decided=1)
    assert out["exposable"] is False
    assert any("slice" in r for r in out["reasons"])


def test_the_verdict_always_carries_the_counts_it_was_based_on():
    out = _verdict(_perfect())
    assert out["overall"]["decided"] == 6
    assert out["thresholds"]["chance"] == RV.CHANCE


# ------------------------------------------------------------------ one slice is not evidence

def test_a_scorer_evidenced_by_only_one_slice_is_refused():
    """Measured case: a heuristic scored 0.804 overall on 158 decided pairs, ALL from one family,
    while a workload inside that same family scored 0.486 -- below chance. Every other slice decided
    nothing, so a check that only fails slices with evidence would have called it exposable. Silence
    from the other slices is missing evidence, not a pass."""
    lone = RV.Agreement(pairs=200, decided=158, agreed=127, undecided=42)   # 0.804
    silent = RV.Agreement(pairs=40, decided=0, agreed=0, undecided=40)
    out = RV.verdict(lone, {"PK": lone, "PM": silent, "PR": silent},
                     minimum_rate=0.70, minimum_decided=100, minimum_slice_decided=20)
    assert out["exposable"] is False
    assert any("below the required 2" in r for r in out["reasons"])
    assert out["qualifying_slices"] == ["PK"]


def test_two_qualifying_slices_that_both_pass_are_exposable():
    good = RV.Agreement(pairs=200, decided=150, agreed=120, undecided=50)   # 0.80
    out = RV.verdict(good, {"A": good, "B": good},
                     minimum_rate=0.70, minimum_decided=100, minimum_slice_decided=20)
    assert out["exposable"] is True


def test_a_qualifying_slice_below_the_bar_still_refuses():
    good = RV.Agreement(pairs=200, decided=150, agreed=120, undecided=50)
    poor = RV.Agreement(pairs=200, decided=100, agreed=48, undecided=100)   # 0.48
    out = RV.verdict(good, {"A": good, "B": poor},
                     minimum_rate=0.70, minimum_decided=100, minimum_slice_decided=20)
    assert out["exposable"] is False
    assert any("fall below the bar" in r for r in out["reasons"])

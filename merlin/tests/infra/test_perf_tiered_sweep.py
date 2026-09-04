"""The sweep is ordered by measured cost and may stop only a candidate that has already lost.

One tuning measurement sweeps every corpus member, and the members do not cost the same: measured
median simulation time across this corpus spans 10.2 s to 178.8 s, a 17.5x spread. A candidate that
is behind on the cheap members is behind, and paying for the slowest ones to confirm it spends the
budget on a conclusion already reached.

The stop is one-directional and that is the whole safety argument: it may end a sweep for a candidate
losing everywhere it has been asked, and it may never shorten one for a candidate that is winning,
because a winning prefix predicts nothing about a member not yet measured.
"""
from __future__ import annotations

import json
import sys

from merlin.common.paths import merlin_dir

sys.path.insert(0, str(merlin_dir() / "experiments" / "gemmini_perf_bench" / "scripts"))

import perf_agent_stage as PAS  # noqa: E402


class _Member:
    def __init__(self, capsule, family="PK"):
        self.capsule, self.family = capsule, family


class _Feedback:
    """Only the attributes the predicate reads."""
    MINIMUM_REFUTING_PREFIX = PAS.DevelopmentGsimFeedback.MINIMUM_REFUTING_PREFIX
    _refuted_so_far = PAS.DevelopmentGsimFeedback._refuted_so_far


def _cell(delta, *, comparable=True):
    return {"comparable": comparable, "candidate_minus_baseline_cycles": delta}


def _losing(n):
    return [_cell(+10) for _ in range(n)]


# ---------------------------------------------------------------- ordering

def test_the_sweep_orders_cheapest_measured_member_first():
    members = [_Member("slow"), _Member("cheap"), _Member("mid")]
    cost = {"slow": 178.5, "cheap": 10.2, "mid": 44.8}
    ordered, basis = PAS.order_members_by_cost(members, cost)
    assert [m.capsule for m in ordered] == ["cheap", "mid", "slow"]
    assert "measured simulation seconds" in basis


def test_an_unpriced_member_sorts_last_never_first():
    """Absence of a cost must not read as cheap: it might be the slowest member in the corpus."""
    members = [_Member("unknown"), _Member("cheap")]
    ordered, _ = PAS.order_members_by_cost(members, {"cheap": 10.0})
    assert [m.capsule for m in ordered] == ["cheap", "unknown"]


def test_without_history_the_declared_order_stands_and_says_so():
    members = [_Member("b"), _Member("a")]
    ordered, basis = PAS.order_members_by_cost(members, {})
    assert [m.capsule for m in ordered] == ["b", "a"]
    assert "no measured simulation cost" in basis


# ---------------------------------------------------------------- the stop

def test_a_candidate_losing_on_every_measured_member_stops_the_sweep():
    fb = _Feedback()
    assert fb._refuted_so_far(_losing(3), index=2, total=10) is True


def test_a_single_loss_is_not_enough_to_stop():
    """The cheapest member is the one most dominated by fixed cost; one loss is an anecdote."""
    fb = _Feedback()
    assert fb._refuted_so_far(_losing(1), index=0, total=10) is False
    assert fb._refuted_so_far(_losing(2), index=1, total=10) is False


def test_one_win_anywhere_keeps_the_sweep_going():
    """The rule is one-directional: a win must never be cut short."""
    fb = _Feedback()
    cells = _losing(3) + [_cell(-5)]
    assert fb._refuted_so_far(cells, index=3, total=10) is False


def test_a_tie_keeps_the_sweep_going():
    fb = _Feedback()
    assert fb._refuted_so_far(_losing(3) + [_cell(0)], index=3, total=10) is False


def test_an_incomparable_member_keeps_the_sweep_going():
    """A member that could not be compared means the picture is partial, not that it lost."""
    fb = _Feedback()
    cells = _losing(3) + [_cell(None, comparable=False)]
    assert fb._refuted_so_far(cells, index=3, total=10) is False


def test_the_last_member_never_triggers_a_stop():
    """Stopping at the end saves nothing and would only complicate the record."""
    fb = _Feedback()
    assert fb._refuted_so_far(_losing(5), index=4, total=5) is False


# ---------------------------------------------------------------- the record

def test_an_unmeasured_member_is_recorded_not_omitted():
    """A missing cell and an unmeasured one are different claims; only one is honest."""
    row = PAS._unmeasured_cell(_Member("PM15_m64n64", family="PM"), reason="already behind")
    assert row["measured"] is False
    assert row["skip_reason"] == "already behind"
    assert row["capsule"] == "PM15_m64n64"
    # every measurement-valued field is null, never zero
    for field in ("baseline_gsim_cycles", "candidate_gsim_cycles",
                  "candidate_minus_baseline_cycles", "declared_macs"):
        assert row[field] is None, field
    assert row["comparable"] is False
    assert row["verdict"] == "refused"


def test_the_unmeasured_cell_matches_the_redacted_schema_exactly():
    """The schema is an exact key set, so a skipped cell must be the same shape as a measured one."""
    row = PAS._unmeasured_cell(_Member("x"), reason="r")
    source = (merlin_dir() / "experiments" / "gemmini_perf_bench" / "scripts"
              / "perf_agent_stage.py").read_text(encoding="utf-8")
    start = source.index("cell_fields = {")
    declared = source[start:source.index("}", start)]
    for key in row:
        assert f'"{key}"' in declared, f"the redacted schema does not admit {key!r}"


def test_harvest_reports_no_cost_rather_than_zero_for_an_absent_history(tmp_path):
    assert PAS.harvest_member_cost([tmp_path]) == {}
    assert PAS.harvest_member_cost([]) == {}


def test_harvest_takes_the_median_of_repeated_measurements(tmp_path):
    for i, seconds in enumerate((10.0, 30.0, 20.0)):
        d = tmp_path / f"r{i}"
        d.mkdir()
        (d / "capsule_result.json").write_text(json.dumps({
            "capsule": "K", "tiers": {"L3": {"timing": {"sim_active_s": seconds}}}}),
            encoding="utf-8")
    assert PAS.harvest_member_cost([tmp_path]) == {"K": 20.0}

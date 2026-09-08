"""The performance campaign's own analysis: cost, attempts, and the gaps it refuses to fill.

Every reader here gets a PAIRED test, as `agentreport/AGENT.md` requires: one that proves it finds
the thing, one that proves it REFUSES with a reason when the thing is absent. A zero plots as a
finding and a refusal plots as a gap, and a campaign that spent nothing looks exactly like a
campaign whose receipts are missing unless the two are kept apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from merlin.agentreport import phase2_campaign as PC
from merlin.perf.optimization_ledger import Attempt, Delta, Ledger


@dataclass
class _Call:
    """The shape `agentreport.phase2.BrokerCall` presents; constructed directly so these tests do
    not depend on a run tree."""

    action: str
    elapsed_s: float
    returncode: int | None
    state: str = "complete"
    index: int = 0


class TestTheCostTiersAreBoundsNotAnActionNameTable:
    """The action vocabulary is DERIVED per run from the candidate's manifest, so naming actions in
    the library would describe one campaign and mislabel the next."""

    def test_a_free_action_is_under_a_second(self):
        assert PC.tier_of(0.0) == "free"
        assert PC.tier_of(0.9) == "free"

    def test_a_compile_is_seconds(self):
        assert PC.tier_of(1.0) == "compile"
        assert PC.tier_of(3.2) == "compile"
        assert PC.tier_of(18.9) == "compile"

    def test_a_measurement_is_the_top_tier(self):
        assert PC.tier_of(97.8) == "measure"
        assert PC.tier_of(104.3) == "measure"
        assert PC.tier_of(1e9) == "measure"

    def test_every_tier_carries_the_argument_for_what_it_is(self):
        for name, _, why in PC.COST_TIERS:
            assert len(why) > 20, f"{name} needs a reason, not a label"


class TestARefusalIsNotAFailure:
    """A refusal means the action declined to run: same wall time, no evidence. 154 of 1,296 calls
    refused while 11 actually failed, and reporting them together hides both."""

    def test_the_refusal_codes_are_declared(self):
        assert 125 in PC.REFUSAL_RETURNCODES and 126 in PC.REFUSAL_RETURNCODES
        assert 0 not in PC.REFUSAL_RETURNCODES and 1 not in PC.REFUSAL_RETURNCODES

    def test_refusals_and_failures_are_counted_separately(self):
        budget = PC.summarize_broker_calls([
            _Call("measure", 100.0, 0), _Call("measure", 100.0, 125),
            _Call("measure", 100.0, 126), _Call("measure", 100.0, 1)])
        row = budget.actions[0]
        assert row.calls == 4 and row.refused == 2 and row.failed == 1
        assert row.refused_seconds == pytest.approx(200.0)
        assert row.refusal_rate == pytest.approx(0.5)

    def test_refused_wall_time_is_reported_as_its_own_fraction(self):
        budget = PC.summarize_broker_calls([
            _Call("cheap", 1.0, 0), _Call("dear", 99.0, 125)])
        block = budget.to_dict()
        assert block["refused"] == 1
        assert block["refused_call_fraction"] == pytest.approx(0.5)
        assert block["refused_wall_fraction"] == pytest.approx(0.99), (
            "half the CALLS and 99% of the WALL -- the two must not be conflated")

    def test_a_call_with_no_return_code_is_neither_credited_nor_charged(self):
        budget = PC.summarize_broker_calls([_Call("x", 10.0, None), _Call("x", 10.0, 0)])
        row = budget.actions[0]
        assert row.calls == 2 and row.refused == 0 and row.failed == 0
        status = budget.availability.get("outcomes")
        assert status.kind == "unavailable"
        assert "neither a success nor a refusal" in status.reason

    def test_all_coded_calls_leave_the_outcomes_field_MEASURED(self):
        budget = PC.summarize_broker_calls([_Call("x", 1.0, 0), _Call("x", 1.0, 125)])
        assert budget.availability.get("outcomes").kind == "measured"


class TestAnEmptyBudgetRefusesRatherThanReportingZero:
    def test_no_receipts_is_a_GAP_with_a_reason(self):
        budget = PC.summarize_broker_calls([])
        assert budget.calls == 0
        status = budget.availability.get("budget")
        assert status.kind == "unavailable"
        assert "would read as a campaign that spent nothing" in status.reason

    def test_receipts_present_makes_it_MEASURED(self):
        budget = PC.summarize_broker_calls([_Call("x", 1.0, 0)])
        assert budget.availability.get("budget").kind == "measured"


class TestTheTierTableIsTheShapeOfTheCampaign:
    def test_calls_and_wall_are_grouped_by_tier(self):
        budget = PC.summarize_broker_calls([
            _Call("analysis", 0.0, 0), _Call("analysis", 0.0, 0),
            *[_Call("emit", 3.0, 0) for _ in range(10)],
            _Call("sim", 100.0, 0), _Call("sim", 100.0, 125)])
        tiers = budget.by_tier()
        assert tiers["free"]["calls"] == 2 and tiers["free"]["wall_seconds"] == 0.0
        assert tiers["compile"]["calls"] == 10 and tiers["compile"]["wall_seconds"] == 30.0
        assert tiers["measure"]["calls"] == 2 and tiers["measure"]["refused"] == 1

    def test_the_measure_tier_can_dominate_from_a_minority_of_calls(self):
        """The ratio that decides how a campaign should be shaped."""
        budget = PC.summarize_broker_calls(
            [*[_Call("emit", 3.0, 0) for _ in range(100)],
             *[_Call("sim", 100.0, 0) for _ in range(20)]])
        tiers = budget.by_tier()
        assert tiers["compile"]["calls"] > tiers["measure"]["calls"]
        assert tiers["measure"]["wall_seconds"] > tiers["compile"]["wall_seconds"]


class TestTheAttemptSummaryGroupsTheThreeWaysAReaderAsks:
    def _ledger(self):
        return Ledger(target="t", attempts=[
            Attempt(mechanism="a", scope="global", found_by="i1", verdict="helped",
                    deltas=(Delta("w", "m", 2.0, 1.0, "inst"),)),
            Attempt(mechanism="b", scope="build", found_by="i2", verdict="refuted",
                    hypothesis="the branch was believed", deltas=(Delta("w", "m", 1.0, 2.0, "inst"),)),
            Attempt(mechanism="c", scope="frontend", found_by="i3", verdict="blocked",
                    blocked_by="an export that carries no parameters"),
            Attempt(mechanism="d", scope="global", found_by="i1", verdict="unmeasured"),
        ])

    def test_it_counts_by_verdict_and_by_scope(self):
        out = PC.summarize_attempts(self._ledger())
        assert out.total == 4
        assert out.by_verdict == {"helped": 1, "refuted": 1, "blocked": 1, "unmeasured": 1}
        assert out.by_scope == {"global": 2, "build": 1, "frontend": 1}
        assert out.scopes_reached == 3

    def test_only_measured_verdicts_are_counted_as_carrying_a_measurement(self):
        out = PC.summarize_attempts(self._ledger())
        assert out.measured_attempts == 2, "helped + refuted; blocked and unmeasured are not"

    def test_instruments_are_deduplicated(self):
        out = PC.summarize_attempts(self._ledger())
        assert out.instruments == ["i1", "i2", "i3"]

    def test_refuted_rows_carry_WHY_the_branch_is_dead(self):
        out = PC.summarize_attempts(self._ledger())
        assert len(out.refuted) == 1
        assert out.refuted[0]["why_the_branch_is_dead"] == "the branch was believed"
        assert out.refuted[0]["deltas"], "a refutation without its measurement is an opinion"

    def test_live_blockers_carry_what_blocks_them(self):
        out = PC.summarize_attempts(self._ledger())
        assert len(out.live_blockers) == 1
        assert "carries no parameters" in out.live_blockers[0]["blocked_by"]

    def test_an_empty_ledger_is_a_GAP_not_a_campaign_that_tried_nothing(self):
        out = PC.summarize_attempts(Ledger(target="t"))
        status = out.availability.get("attempts")
        assert status.kind == "unavailable"
        assert "rather than one whose" in status.reason

    def test_integrity_problems_are_REPORTED_not_dropped(self):
        """A row that does not stand up is a fact about the record-keeping."""
        bad = Ledger(target="t", attempts=[
            Attempt(mechanism="x", scope="global", found_by="", verdict="helped")])
        out = PC.summarize_attempts(bad)
        assert len(out.integrity_problems) == 1
        assert out.integrity_problems[0]["problems"], "the reason must travel with it"


class TestTheWholeAnalysisSaysWhatItCannotDecide:
    def _facts(self, **over):
        base = dict(target="t", stages=2, tool_spans=10, point_events=3,
                    broker_calls=[_Call("sim", 100.0, 0)],
                    ledger=Ledger(target="t", attempts=[
                        Attempt(mechanism="a", scope="global", found_by="i", verdict="helped",
                                deltas=(Delta("w", "m", 2.0, 1.0, "inst"),))]),
                    outcomes=[{"model": "w", "elf_bytes": 1}])
        base.update(over)
        return base

    def test_a_complete_analysis_scores_one(self):
        out = PC.build_analysis(**self._facts())
        block = out.to_dict()
        assert block["availability_score"] == 1.0
        assert all(s["kind"] != "unavailable" for s in block["availability"].values())

    def test_no_stage_telemetry_is_a_GAP(self):
        out = PC.build_analysis(**self._facts(stages=0))
        assert out.availability.get("stages").kind == "unavailable"

    def test_no_tool_spans_means_the_LANE_SHAPE_is_unknown(self):
        out = PC.build_analysis(**self._facts(tool_spans=0))
        status = out.availability.get("lane_shape")
        assert status.kind == "unavailable"
        assert "serially or in parallel" in status.reason

    def test_no_outcome_means_the_campaigns_PRODUCT_is_unstated(self):
        out = PC.build_analysis(**self._facts(outcomes=[]))
        assert out.availability.get("outcomes").kind == "unavailable"

    def test_the_score_falls_as_gaps_appear(self):
        full = PC.build_analysis(**self._facts()).to_dict()["availability_score"]
        gapped = PC.build_analysis(
            **self._facts(stages=0, tool_spans=0, outcomes=[])).to_dict()["availability_score"]
        assert gapped < full

    def test_the_nested_ledgers_are_merged_under_their_own_prefixes(self):
        block = PC.build_analysis(**self._facts()).to_dict()
        assert "budget.budget" in block["availability"]
        assert "attempts.attempts" in block["availability"]

    def test_it_carries_its_schema(self):
        block = PC.build_analysis(**self._facts()).to_dict()
        assert block["schema"] == "merlin_phase2_campaign_analysis_v1"
        assert block["budget"]["schema"] == "merlin_phase2_budget_v1"
        assert block["attempts"]["schema"] == "merlin_phase2_attempts_v1"


class TestItRunsOnThisTreesRealCampaign:
    """The acceptance test: the numbers this analysis is generated from must still be there."""

    def _analysis(self):
        from merlin.agentreport.phase2 import read_phase2
        from merlin.common.paths import artifacts_dir, runs_dir
        roots = [runs_dir() / "gemmini" / "perf-bench" / "agent_stages",
                 artifacts_dir() / "perf-bench" / "gemmini"]
        stages = [c for r in roots if r.is_dir() for c in sorted(r.iterdir())
                  if c.is_dir() and ((c / "agent" / "tools.jsonl").is_file()
                                     or (c / "control").is_dir()
                                     or (c / "global_control").is_dir())]
        if not stages:
            pytest.skip("no phase-2 telemetry in this tree")
        calls, spans, points = [], 0, 0
        for stage in stages:
            facts = read_phase2(stage)
            calls.extend(facts.broker_calls)
            spans += len(facts.spanset.spans)
            points += facts.n_point_events
        return PC.build_analysis(target="gemmini", stages=len(stages), tool_spans=spans,
                                 point_events=points, broker_calls=calls, ledger=Ledger("gemmini"),
                                 outcomes=[{"model": "present"}])

    def test_the_measure_tier_dominates_the_wall_time(self):
        tiers = self._analysis().budget.by_tier()
        assert tiers["measure"]["wall_seconds"] > tiers["compile"]["wall_seconds"] * 5, (
            "if the expensive tier stops dominating, the campaign's cost shape has changed and the "
            "analysis' central claim needs re-deriving")

    def test_some_brokered_calls_REFUSED_and_they_are_attributed(self):
        budget = self._analysis().budget
        assert budget.refused > 0
        assert budget.refused_seconds > 0
        worst = max(budget.actions, key=lambda a: a.refusal_rate)
        assert worst.refusal_rate > 0.5, "at least one action refuses most of the time"

"""Unit tests for :mod:`merlin.perf.select` -- bounded candidate selection.

Everything here runs against a FAKE evaluator: a deterministic dict lookup. No oracle, no simulator,
no compiler. That is a requirement of the acceptance criterion, not a convenience -- a stop condition
whose test needs a working oracle is a stop condition nobody can regression-test.

Three properties are locked in:

* **an axis whose evidence is missing reports UNKNOWN and offers no candidate.** Both axes here are
  partly blocked upstream, and a selection loop that fills the gap with a plausible default spends a
  real budget exploring a space nothing measured;
* **each of the four stop conditions fires and does NOT fire**, on hand-built states;
* **the convergence curve is denominated in the measured scarce unit**, so the same code produces a
  differently-denominated curve when measurement elects a different unit.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.perf import budget as bud
from merlin.perf import select as sel
from merlin.perf.amplification import MovementObservation, amplification
from merlin.perf.decompose import UNKNOWN, ResourceKind, Trait, Unavailable, activity_from_busy, is_unknown
from merlin.perf.headroom import headroom

KINDS = {"move": ResourceKind.MOVEMENT, "arith": ResourceKind.COMPUTE,
         "unattributed": ResourceKind.FIXED}
MOVEMENT_TRAIT = Trait("explicit_movement", True, evidence="fixture declares a movement engine")


def _source(*, total=1000, move=700, arith=200, fixed=100, completion=True):
    return activity_from_busy("w", total, {"move": move, "arith": arith, "unattributed": fixed},
                              KINDS, partitioned=False, completion_observable=completion,
                              provenance="fixture activity decomposition")


def _amp(*, moved=8192, useful=2048, transfers=32):
    return amplification(MovementObservation("w", moved_bytes=moved, useful_bytes=useful,
                                             transfers=transfers),
                         trait=MOVEMENT_TRAIT)


def _channels():
    """A measured regime where the synthesis call dominates a datapoint."""
    return [bud.channel_from_samples("synthesis_call", seconds=[2184.0], dollars=[25.33]),
            bud.channel_from_samples("oracle_query", seconds=[0.276], items_per_datapoint=26)]


def _budget(**limits) -> bud.Budget:
    b = bud.budget_from_channels(_channels(), **limits)
    assert isinstance(b, bud.Budget)
    return b


# --- the two axes, and only two --------------------------------------------------------------------

def test_there_are_exactly_two_axes():
    assert [a.value for a in sel.Axis] == ["dma_tiling", "overlap_policy"]


def test_derive_axes_always_reports_both_even_when_one_is_unknown():
    axes = sel.derive_axes(_source(), amplification=_amp(), headroom=None)
    assert set(axes) == set(sel.Axis)
    assert axes[sel.Axis.OVERLAP].established is None


# --- DMA / descriptor axis ---------------------------------------------------------------------------

def test_dma_axis_is_unknown_without_a_derivable_per_command_byte_volume():
    """The upstream block: with the command count unknown the granule is not derivable, so a sweep
    would be a sweep over an assumed descriptor. No candidate is offered."""
    ev = sel.dma_axis(_source(), _amp(transfers=None))
    assert ev.established is None
    assert ev.candidates == ()
    assert any("byte volume" in m for m in ev.missing)
    assert isinstance(ev.unavailable, Unavailable)


def test_dma_axis_is_unknown_when_no_movement_resource_carried_cycles():
    ev = sel.dma_axis(_source(move=0, arith=900), _amp())
    assert ev.established is None
    assert ev.candidates == ()
    assert any("movement resource" in m for m in ev.missing)


def test_dma_axis_propagates_an_unavailable_amplification():
    amp = amplification(MovementObservation("w", moved_bytes=0, useful_bytes=2048, transfers=8),
                        trait=MOVEMENT_TRAIT)
    assert isinstance(amp, Unavailable)
    ev = sel.dma_axis(_source(), amp)
    assert ev.established is None
    assert ev.missing == amp.missing


def test_dma_axis_sweeps_descriptor_shapes_derived_from_the_observation():
    ev = sel.dma_axis(_source(), _amp())          # 32 commands of 256 B; floor is 8
    assert ev.established is True
    shapes = sorted(dict(c.setting)["transfers"] for c in ev.candidates)
    assert shapes == [8, 16]                       # the halving ladder down to the derived floor
    assert all(dict(c.setting)["block_bytes"] == 256.0 for c in ev.candidates)
    # fewer commands move fewer bytes, so the saving grows as the ladder descends
    by_shape = {dict(c.setting)["transfers"]: float(c.saving_hi) for c in ev.candidates}
    assert by_shape[8] > by_shape[16] > 0


def test_dma_axis_reports_no_lever_rather_than_unknown_when_already_at_the_floor():
    """``False`` is a finding about the target; ``None`` would mean the evidence was missing."""
    ev = sel.dma_axis(_source(), _amp(moved=2048, useful=2048, transfers=8))
    assert ev.established is False
    assert ev.candidates == ()


# --- overlap axis -------------------------------------------------------------------------------------

def test_overlap_axis_is_unknown_when_the_concurrency_traits_are_not_established():
    src = _source(completion=None)
    hr = headroom(src)
    assert isinstance(hr, Unavailable)
    ev = sel.overlap_axis(src, hr)
    assert ev.established is None
    assert ev.candidates == ()
    assert ev.missing == hr.missing


def test_overlap_candidate_is_a_ceiling_with_an_open_interval_when_overlap_is_unobserved():
    src = _source()
    ev = sel.overlap_axis(src, headroom(src))
    assert ev.established is True
    (cand,) = ev.candidates
    assert cand.is_upper_bound
    assert float(cand.saving_hi) == 200.0          # min(move 700, arith 200)
    assert float(cand.saving_lo) == 0.0            # the program may already overlap it entirely


def test_observing_the_realised_overlap_collapses_the_interval_and_the_voi():
    """VOI ranks what is worth QUERYING. A saving already pinned down is worth implementing, not
    re-measuring, and scores zero here by construction."""
    src = _source()
    ev = sel.overlap_axis(src, headroom(src, observed_overlap_cycles=50))
    (cand,) = ev.candidates
    assert not cand.is_upper_bound
    assert float(cand.saving_lo) == float(cand.saving_hi) == 150.0
    v = sel.voi(cand, reference_cycles=1000, budget=_budget())
    assert v.uncertainty == 0.0
    assert v.score == 0.0


def test_candidates_from_does_not_multiply_the_axes_together():
    src = _source()
    axes = sel.derive_axes(src, amplification=_amp(), headroom=headroom(src))
    cands = sel.candidates_from(axes)
    assert len(cands) == len(axes[sel.Axis.DMA_TILING].candidates) + \
        len(axes[sel.Axis.OVERLAP].candidates)
    assert {c.axis for c in cands} == set(sel.Axis)
    assert all(len(c.setting) == 2 for c in cands)   # each point names ONE axis, never a compound


# --- VOI: three factors, not four ----------------------------------------------------------------------

def test_voi_is_impact_times_uncertainty_over_cost():
    src = _source()
    (cand,) = sel.overlap_axis(src, headroom(src)).candidates
    v = sel.voi(cand, reference_cycles=1000, budget=_budget(), cost_units=2.0)
    assert v.impact == pytest.approx(0.2)
    assert v.uncertainty == pytest.approx(1.0)
    assert v.cost_units == 2.0
    assert v.score == pytest.approx(0.1)


def test_generality_is_dropped_and_the_reason_travels_with_the_score():
    assert not hasattr(sel.VOI, "generality")
    src = _source()
    (cand,) = sel.overlap_axis(src, headroom(src)).candidates
    d = sel.voi(cand, reference_cycles=1000, budget=_budget()).to_dict()
    assert set(d) >= {"impact", "uncertainty", "cost_units", "score"}
    assert "constant across candidates" in d["generality"]


def test_an_unknown_factor_makes_the_score_unknown_and_never_zero():
    cand = sel.Candidate(axis=sel.Axis.DMA_TILING, workload="w", setting=(("transfers", 4),),
                         baseline_cycles=1000, saving_hi=100.0, saving_lo=UNKNOWN,
                         rationale="the interval was not split")
    v = sel.voi(cand, reference_cycles=1000, budget=_budget())
    assert is_unknown(v.uncertainty)
    assert is_unknown(v.score)
    assert not v.known
    assert v.missing


def test_rank_retains_unscorable_candidates_but_sorts_them_last():
    good = sel.Candidate(sel.Axis.OVERLAP, "w", (("group_a", "arith"), ("group_b", "move")),
                         1000, 200.0, 0.0, "ceiling")
    blind = sel.Candidate(sel.Axis.DMA_TILING, "w", (("transfers", 4),), 1000, 500.0, UNKNOWN,
                          "interval not split")
    out = sel.rank([blind, good], reference_cycles=1000, budget=_budget())
    assert [v.candidate_id for v in out] == [good.id, blind.id]
    assert len(out) == 2


def test_the_ranking_is_denominated_in_the_scarce_unit_so_cost_can_flip_it():
    """The same two candidates, the same savings: only the per-candidate cost in the scarce unit
    changes, and the order changes with it."""
    big = sel.Candidate(sel.Axis.OVERLAP, "w", (("group_a", "a"), ("group_b", "b")),
                        1000, 300.0, 0.0, "big but expensive")
    small = sel.Candidate(sel.Axis.OVERLAP, "w", (("group_a", "c"), ("group_b", "d")),
                          1000, 100.0, 0.0, "small but cheap")
    flat = sel.rank([big, small], reference_cycles=1000, budget=_budget())
    assert [v.candidate_id for v in flat] == [big.id, small.id]
    priced = sel.rank([big, small], reference_cycles=1000, budget=_budget(),
                      cost_units={big.id: 10.0, small.id: 1.0})
    assert [v.candidate_id for v in priced] == [small.id, big.id]


# --- the four stop conditions: each firing, and each NOT firing -----------------------------------------

POLICY = sel.StopPolicy()


def _state(**kw) -> sel.SearchState:
    base = dict(baseline_cycles=1000, best_cycles=900.0, budget=_budget(limit_items=10))
    base.update(kw)
    return sel.SearchState(**base)


def test_attainment_reached_fires_at_ninety_percent_of_the_conservative_target():
    v = sel.attainment_reached(_state(best_cycles=850.0, attainable_cycles=800.0), POLICY)
    assert v.fired and "94.1%" in v.reason


def test_attainment_reached_does_not_fire_below_the_threshold():
    v = sel.attainment_reached(_state(best_cycles=1000.0, attainable_cycles=800.0), POLICY)
    assert not v.fired and "below" in v.reason


def test_attainment_reached_does_not_fire_on_an_unknown_target():
    """An unresolved bound must never read as a bound that was reached."""
    v = sel.attainment_reached(_state(best_cycles=810.0, attainable_cycles=UNKNOWN), POLICY)
    assert not v.fired
    assert v.missing


def test_predicted_remaining_below_fires_when_nothing_left_promises_three_percent():
    v = sel.predicted_remaining_below(_state(best_cycles=1000.0, predicted_best_cycles=980.0),
                                      POLICY)
    assert v.fired and "2.00%" in v.reason


def test_predicted_remaining_below_does_not_fire_while_a_candidate_promises_more():
    v = sel.predicted_remaining_below(_state(best_cycles=1000.0, predicted_best_cycles=800.0),
                                      POLICY)
    assert not v.fired


def test_predicted_remaining_below_does_not_fire_when_no_candidate_carries_a_prediction():
    v = sel.predicted_remaining_below(_state(best_cycles=1000.0, predicted_best_cycles=UNKNOWN),
                                      POLICY)
    assert not v.fired
    assert v.missing


def test_plateaued_fires_after_three_consecutive_sub_one_percent_queries():
    v = sel.plateaued(_state(improvements=(0.20, 0.004, 0.002, 0.001)), POLICY)
    assert v.fired and "last 3 queries" in v.reason


def test_plateaued_does_not_fire_when_one_of_the_three_cleared_the_bar():
    v = sel.plateaued(_state(improvements=(0.004, 0.05, 0.001)), POLICY)
    assert not v.fired


def test_plateaued_does_not_fire_on_a_history_shorter_than_the_rule():
    v = sel.plateaued(_state(improvements=(0.001, 0.001)), POLICY)
    assert not v.fired and "needs 3" in v.reason


def test_budget_exhausted_fires_and_names_the_unit_it_is_denominated_in():
    b = _budget(limit_items=1)
    b.charge(label="a")
    v = sel.budget_exhausted(_state(budget=b), POLICY)
    assert v.fired and "synthesis_call" in v.reason


def test_budget_exhausted_does_not_fire_while_items_remain():
    b = _budget(limit_items=3)
    b.charge(label="a")
    v = sel.budget_exhausted(_state(budget=b), POLICY)
    assert not v.fired and "2 remaining" in v.reason


def test_check_stop_always_returns_all_four_verdicts():
    verdicts = sel.check_stop(_state())
    assert [v.name for v in verdicts] == [
        "attainment_reached", "predicted_remaining_below", "plateaued", "budget_exhausted"]
    assert all(v.reason for v in verdicts)


# --- the loop, over a fake evaluator ---------------------------------------------------------------------

class FakeEvaluator:
    """Deterministic cycle counts by candidate id. Counts its calls so a test can prove the loop
    stopped rather than merely reported that it did."""

    def __init__(self, cycles: dict[str, float], default: float = 1000.0):
        self.cycles = cycles
        self.default = default
        self.calls: list[str] = []

    def __call__(self, cand: sel.Candidate) -> float:
        self.calls.append(cand.id)
        return self.cycles.get(cand.id, self.default)


def _cands(n: int, *, hi: float = 300.0, lo: float = 0.0) -> list[sel.Candidate]:
    return [sel.Candidate(sel.Axis.OVERLAP, "w", (("group_a", f"g{i}"), ("group_b", "move")),
                          1000, hi - i, lo, f"fixture candidate {i}") for i in range(n)]


def test_the_loop_stops_on_the_budget_and_leaves_the_rest_unrun():
    cands = _cands(5)
    ev = FakeEvaluator({})
    b = _budget(limit_items=2)
    res = sel.search(cands, evaluate=ev, budget=b, baseline_cycles=1000)
    assert len(ev.calls) == 2
    assert res.stopped_by == ("budget_exhausted",)
    assert any("did not run" in r or "not reached" in r for _, r in res.skipped)


def test_the_loop_stops_once_the_conservative_attainable_target_is_reached():
    cands = _cands(5)
    ev = FakeEvaluator({cands[0].id: 850.0})
    res = sel.search(cands, evaluate=ev, budget=_budget(limit_items=10), baseline_cycles=1000,
                     attainable_cycles=800.0)
    assert len(ev.calls) == 1
    assert res.stopped_by == ("attainment_reached",)
    assert res.best_cycles == 850.0


def test_the_loop_stops_after_three_queries_that_each_improved_under_one_percent():
    cands = _cands(6)
    ranked = sel.rank(cands, reference_cycles=1000, budget=_budget())
    order = [v.candidate_id for v in ranked]
    ev = FakeEvaluator({order[0]: 995.0, order[1]: 994.0, order[2]: 993.0})
    res = sel.search(cands, evaluate=ev, budget=_budget(limit_items=10), baseline_cycles=1000)
    assert len(ev.calls) == 3
    assert res.stopped_by == ("plateaued",)


def test_the_loop_stops_when_nothing_remaining_predicts_three_percent():
    strong = sel.Candidate(sel.Axis.OVERLAP, "w", (("group_a", "big"), ("group_b", "move")),
                           1000, 300.0, 0.0, "worth a query")
    weak = [sel.Candidate(sel.Axis.OVERLAP, "w", (("group_a", f"t{i}"), ("group_b", "move")),
                          1000, 5.0, 0.0, "barely moves") for i in range(3)]
    ev = FakeEvaluator({strong.id: 700.0})
    res = sel.search([strong, *weak], evaluate=ev, budget=_budget(limit_items=10),
                     baseline_cycles=1000)
    assert ev.calls == [strong.id]
    assert res.stopped_by == ("predicted_remaining_below",)


def test_an_identical_candidate_is_served_from_cache_and_charged_nothing():
    """Content-addressed the way oracle_schedule content-addresses a verdict: identical settings
    cannot produce a different answer, so paying for them again buys nothing."""
    a = _cands(1)[0]
    twin = sel.Candidate(a.axis, a.workload, a.setting, a.baseline_cycles, a.saving_hi,
                         a.saving_lo, "a different rationale, the same point")
    assert twin.digest == a.digest
    ev = FakeEvaluator({})
    b = _budget(limit_items=10)
    sel.search([a, twin], evaluate=ev, budget=b, baseline_cycles=1000)
    assert len(ev.calls) == 1
    assert b.spent_items == 1.0


def test_the_search_runs_with_no_oracle_at_all():
    """The whole acceptance criterion in one line: a fake evaluator, and nothing else is touched."""
    ev = FakeEvaluator({})
    res = sel.search(_cands(3), evaluate=ev, budget=_budget(limit_items=3), baseline_cycles=1000)
    assert res.to_dict()["queries"]
    assert all(isinstance(c, str) for c in ev.calls)


# --- convergence over the MEASURED scarce unit ------------------------------------------------------------

def test_the_convergence_curve_is_plotted_over_the_measured_scarce_unit():
    cands = _cands(4)
    ranked = [v.candidate_id for v in sel.rank(cands, reference_cycles=1000, budget=_budget())]
    ev = FakeEvaluator({ranked[0]: 900.0, ranked[1]: 800.0, ranked[2]: 780.0})
    b = _budget(limit_items=3)
    res = sel.search(cands, evaluate=ev, budget=b, baseline_cycles=1000)
    rows = sel.convergence_rows(res)
    assert rows[0] == {"query": 0, "cumulative_items": 0.0, "cumulative_seconds": 0.0,
                       "cumulative_dollars": 0.0, "best_cycles": 1000.0, "improvement": 0.0,
                       "unit": "synthesis_call", "candidate_id": ""}
    assert [r["cumulative_items"] for r in rows] == [0.0, 1.0, 2.0, 3.0]
    assert [r["best_cycles"] for r in rows] == [1000.0, 900.0, 800.0, 780.0]
    # the x axis carries the MEASURED price, not an assumed one
    assert rows[-1]["cumulative_seconds"] == pytest.approx(3 * 2184.0)
    assert rows[-1]["cumulative_dollars"] == pytest.approx(3 * 25.33)


def test_the_curve_is_denominated_differently_when_measurement_elects_a_different_unit():
    """Same loop, same candidates: only the measured channel prices change."""
    slow = [bud.channel_from_samples("synthesis_call", seconds=[2184.0], dollars=[25.33]),
            bud.channel_from_samples("deep_sim_query", seconds=[2700.0], items_per_datapoint=26)]
    b = bud.budget_from_channels(slow, limit_items=2)
    assert isinstance(b, bud.Budget)
    res = sel.search(_cands(3), evaluate=FakeEvaluator({}), budget=b, baseline_cycles=1000)
    rows = sel.convergence_rows(res)
    assert {r["unit"] for r in rows} == {"deep_sim_query"}
    assert rows[-1]["cumulative_seconds"] == pytest.approx(2 * 2700.0)


def test_write_convergence_emits_the_curve_and_reports_a_missing_plot_as_not_run(tmp_path):
    assert repo_root().is_dir()
    res = sel.search(_cands(3), evaluate=FakeEvaluator({}), budget=_budget(limit_items=2),
                     baseline_cycles=1000)
    out = sel.write_convergence(res, tmp_path / "curve")
    assert out["csv"].is_file() and out["json"].is_file()
    header = out["csv"].read_text().splitlines()[0]
    assert header.split(",")[1] == "cumulative_items"
    assert out["plot"] == "written" or out["plot"].startswith("not_run: ")
    if out["plot"] == "written":
        assert out["png"].is_file()


def test_the_observed_command_count_comes_from_the_amplification_split():
    """With heterogeneous descriptors the granule is the LARGEST command, so dividing the moved
    bytes by it undercounts the commands issued -- and would then drop a real descriptor shape from
    the sweep by mistaking it for the shape already in use."""
    amp = amplification(
        MovementObservation("w", moved_bytes=1088, useful_bytes=512,
                            transfer_bytes=(256, 256, 256, 64, 64, 64, 64, 64)),
        trait=MOVEMENT_TRAIT)
    assert float(amp.block_bytes) == 256.0
    assert int(amp.transfers_min) == 2
    assert round(1088 / 256) == 4                    # what the naive count would have said
    ev = sel.dma_axis(_source(), amp)
    assert ev.established is True
    assert sorted(dict(c.setting)["transfers"] for c in ev.candidates) == [2, 4]
    assert "8 command(s)" in ev.candidates[0].rationale


def test_emit_product_writes_under_the_out_root_via_the_artifacts_helpers(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    res = sel.search(_cands(3), evaluate=FakeEvaluator({}), budget=_budget(limit_items=2),
                     baseline_cycles=1000)
    out = sel.emit_product(res, target="tgt_under_test")
    prod = out["product"]
    assert prod.path.is_relative_to(tmp_path / "out" / "artifacts")
    assert prod.manifest_path.is_file()
    assert (prod.path / "convergence.csv").is_file()
    assert (prod.path / "convergence.json").is_file()

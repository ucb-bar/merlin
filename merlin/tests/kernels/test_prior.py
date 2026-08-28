"""The outcome prior turns a corpus of attempts into something the search can rank by. Every test
here is about a way a prior can claim more than it measured."""
import json

import pytest

from merlin.mining.prior import (LEDGER_ENV, OutcomePrior, family_strategy_map,
                                 ledger_path, load_outcome_prior, prior_fn_from)


def _ledger(tmp_path, rows, name="ledger.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def _row(strategy, outcome):
    return {"strategy_num": strategy, "outcome": outcome}


# --------------------------------------------------------------------------- missing corpus

def test_no_ledger_named_is_unknown_not_an_empty_corpus(monkeypatch):
    monkeypatch.delenv(LEDGER_ENV, raising=False)
    assert ledger_path() is None
    assert load_outcome_prior() is None


def test_a_named_but_absent_ledger_is_also_unknown(tmp_path):
    assert load_outcome_prior(tmp_path / "does_not_exist.jsonl") is None


def test_the_env_seam_names_the_ledger(tmp_path, monkeypatch):
    p = _ledger(tmp_path, [_row("S1", "improved")])
    monkeypatch.setenv(LEDGER_ENV, str(p))
    prior = load_outcome_prior()
    assert prior is not None and prior.source == str(p)


# --------------------------------------------------------------------------- what counts

def test_correct_but_no_gain_is_not_folded_in_with_the_successes():
    """It is a real negative result. Counting it as improvement flatters every rate."""
    prior = OutcomePrior(by_strategy={}, total_attempts=10, total_improved=2, source="x")
    assert prior.base_rate == pytest.approx(0.2)


def test_compile_errors_stay_in_the_denominator(tmp_path):
    """An attempt that never built still consumed a candidate slot. Dropping compile errors reports
    the rate among attempts that happened to build, which is not the rate a planner faces."""
    rows = [_row("S1", "improved")] + [_row("S1", "compile_error")] * 3
    prior = load_outcome_prior(_ledger(tmp_path, rows))
    assert prior.by_strategy["S1"].attempts == 4
    assert prior.rate_for("S1", min_attempts=1) == pytest.approx(0.25)


def test_an_unparseable_line_is_counted_unusable_not_skipped_silently(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text('{"strategy_num": "S1", "outcome": "improved"}\nnot json at all\n', encoding="utf-8")
    prior = load_outcome_prior(p)
    assert prior.total_attempts == 1 and prior.unusable_rows == 1


# --------------------------------------------------------------------------- too little evidence

def test_an_empty_prior_reports_no_rate_for_anything():
    ev = OutcomePrior(by_strategy={}, total_attempts=0, total_improved=0, source="x")
    assert ev.rate_for("S1") is None and ev.base_rate is None


def test_min_attempts_gates_the_rate(tmp_path):
    rows = [_row("S1", "improved"), _row("S1", "incorrect")]
    prior = load_outcome_prior(_ledger(tmp_path, rows))
    assert prior.rate_for("S1", min_attempts=5) is None      # too thin to report
    assert prior.rate_for("S1", min_attempts=2) == pytest.approx(0.5)


def test_an_unknown_strategy_gets_none_not_the_base_rate(tmp_path):
    """The base rate is a fact about the corpus, not a belief about THIS strategy. Returning it would
    assert evidence nobody collected."""
    rows = [_row("S1", "improved")] * 10
    prior = load_outcome_prior(_ledger(tmp_path, rows))
    assert prior.base_rate == pytest.approx(1.0)
    assert prior.rate_for("S_never_seen") is None


# --------------------------------------------------------------------------- coverage honesty

def test_unlabelled_attempts_are_counted_not_discarded(tmp_path):
    rows = [_row("S1", "improved")] * 5 + [_row("", "incorrect")] * 5
    prior = load_outcome_prior(_ledger(tmp_path, rows))
    assert prior.total_attempts == 5 and prior.unlabelled_attempts == 5
    assert prior.unusable_rows == 0          # unlabelled is NOT unusable


def test_a_skewed_unlabelled_population_is_reported(tmp_path):
    """The labelled rows improving at a different rate from the unlabelled ones means the labelled
    set is not a random sample -- so its rates are conditional, not corpus properties."""
    rows = [_row("S1", "improved")] * 5 + [_row("S1", "incorrect")] * 5 + [_row("", "incorrect")] * 10
    prior = load_outcome_prior(_ledger(tmp_path, rows))
    problems = prior.coverage_problems()
    assert any("NO strategy label" in p for p in problems)
    assert any("NOT a random sample" in p for p in problems)


def test_a_representative_unlabelled_population_raises_no_alarm(tmp_path):
    rows = [_row("S1", "improved")] * 5 + [_row("S1", "incorrect")] * 5 + \
           [_row("", "improved")] * 5 + [_row("", "incorrect")] * 5
    prior = load_outcome_prior(_ledger(tmp_path, rows))
    assert not any("random sample" in p for p in prior.coverage_problems())


def test_corpus_rate_and_labelled_rate_are_reported_separately(tmp_path):
    rows = [_row("S1", "improved")] * 5 + [_row("", "incorrect")] * 5
    prior = load_outcome_prior(_ledger(tmp_path, rows))
    assert prior.base_rate == pytest.approx(1.0)          # among labelled
    assert prior.corpus_base_rate == pytest.approx(0.5)   # over everything countable


# --------------------------------------------------------------------------- the correspondence

class _Action:
    def __init__(self, family):
        self.action_family = family


class _Prop:
    def __init__(self, family):
        self.action = _Action(family)


def test_an_action_with_no_declared_strategy_is_unmeasured_not_average(tmp_path):
    """Nothing derives a ledger strategy from a compiler seam. Guessing would produce a confident
    mapping nobody measured, so an undeclared family gets None."""
    rows = [_row("S1", "improved")] * 10
    prior = load_outcome_prior(_ledger(tmp_path, rows))
    fn = prior_fn_from(prior, family_strategy_map([("tiling", "S1")]))
    assert fn(_Prop("tiling")) == pytest.approx(1.0)
    assert fn(_Prop("something_else")) is None


def test_prior_fn_over_a_missing_corpus_returns_none_for_everything():
    fn = prior_fn_from(None, family_strategy_map([("tiling", "S1")]))
    assert fn(_Prop("tiling")) is None


def test_a_proposal_with_no_action_has_no_strategy():
    class _Bare:
        action = None
    fn = prior_fn_from(None, family_strategy_map([("tiling", "S1")]))
    assert fn(_Bare()) is None

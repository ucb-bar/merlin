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


class TestLearningFromOurOwnRuns:
    """Two axes at very different prices: whether an action LANDS (a build answers it) and whether it
    HELPS (hardware answers it). The tests are mostly about not letting the cheap axis be read as the
    expensive one, and not letting absent evidence count as negative evidence."""

    def _node(self, seam="schedule:x", **kw):
        n = {"applied_seams": [seam], "gate_ok": True, "inert": False}
        n.update(kw)
        return n

    def test_an_unmeasured_candidate_is_evidence_about_nothing(self):
        """Under a build-only search most candidates never run. Counting those as 'did not improve'
        would drag every prior toward zero on evidence nobody collected."""
        from merlin.mining.prior import classify_node
        assert classify_node(self._node(speedup=None)) is None

    def test_an_inert_candidate_is_evidence_about_neither_axis(self):
        """Byte-identical emitted code means the action never applied."""
        from merlin.mining.prior import INERT, classify_node, seam_evidence_from_nodes
        assert classify_node(self._node(inert=True, speedup=2.0)) == INERT
        ev = seam_evidence_from_nodes([self._node(inert=True, speedup=2.0)])["schedule:x"]
        assert ev.inert == 1 and ev.measured == 0 and ev.promise_checked == 0

    def test_a_broken_candidate_is_incorrect_not_a_regression(self):
        from merlin.mining.prior import classify_node
        assert classify_node(self._node(gate_ok=False, speedup=9.0)) == "incorrect"

    def test_improvement_needs_to_beat_the_noise_margin(self):
        from merlin.mining.prior import classify_node
        assert classify_node(self._node(speedup=1.30, parent_speedup=1.0,
                                        margin_improved=True)) == "improved"
        assert classify_node(self._node(speedup=1.001, parent_speedup=1.0,
                                        margin_improved=False)) == "correct_no_gain"
        assert classify_node(self._node(speedup=0.80, parent_speedup=1.0,
                                        margin_improved=False)) == "regressed"

    def test_the_two_axes_are_accumulated_separately(self):
        """A candidate can land its promise and still not help, or help while nothing checked it."""
        from merlin.mining.prior import seam_evidence_from_nodes
        nodes = [
            self._node(speedup=1.5, parent_speedup=1.0, margin_improved=True,
                       search_step={"promise_checkable": True, "achieved": True}),
            self._node(speedup=1.0, parent_speedup=1.0, margin_improved=False,
                       search_step={"promise_checkable": True, "achieved": True}),
            self._node(speedup=None, search_step={"promise_checkable": True, "achieved": False}),
        ]
        ev = seam_evidence_from_nodes(nodes)["schedule:x"]
        assert ev.promise_checked == 3 and ev.promise_kept == 2
        assert ev.measured == 2 and ev.improved == 1
        assert ev.unmeasured == 1
        assert ev.landing_rate == pytest.approx(2 / 3)
        assert ev.improvement_rate == pytest.approx(0.5)

    def test_an_unverifiable_step_is_counted_apart_from_a_failed_one(self):
        from merlin.mining.prior import seam_evidence_from_nodes
        ev = seam_evidence_from_nodes([
            self._node(speedup=None, search_step={"promise_checkable": False, "achieved": False})
        ])["schedule:x"]
        assert ev.unverifiable == 1 and ev.promise_checked == 0
        assert ev.landing_rate is None      # nothing was checked, so there is no rate

    def test_the_node_is_credited_to_the_action_it_ADDED(self):
        """applied_seams is a lineage; the last entry is what this fork added on top of its parent."""
        from merlin.mining.prior import seam_evidence_from_nodes
        node = self._node(speedup=None)
        node["applied_seams"] = ["schedule:parent", "schedule:child"]
        ev = seam_evidence_from_nodes([node])
        assert set(ev) == {"schedule:child"}

    def test_a_seam_with_too_little_evidence_yields_no_prior(self):
        from merlin.mining.prior import landing_prior_fn, seam_evidence_from_nodes

        class _A:
            target_seam = "schedule:x"

        class _P:
            action = _A()

        ev = seam_evidence_from_nodes([
            self._node(speedup=None, search_step={"promise_checkable": True, "achieved": True})])
        assert landing_prior_fn(ev, min_attempts=3)(_P()) is None
        assert landing_prior_fn(ev, min_attempts=1)(_P()) == pytest.approx(1.0)


class TestShapeScopeIsAClaimOrSilence:
    """An empty shape_regimes makes an action fire on every regime. Whether that is a validated claim
    or nobody having said is the difference between a one-shot emission that rests on evidence and one
    that is guessing."""

    def _action(self, **kw):
        from merlin.kernels.action_catalog import CompilerAction
        base = dict(divergence_axis="a", action_class="KNOB", target_seam="s", change="c",
                    forkable_now=True, expected_effect="e", backend="rvv")
        base.update(kw)
        return CompilerAction(**base)

    def test_the_three_states_are_distinguishable(self):
        from merlin.kernels.action_catalog import shape_scope
        assert shape_scope(self._action(shape_regimes=("skinny",))) == "regimes"
        assert shape_scope(self._action(shape_agnostic=True)) == "agnostic"
        assert shape_scope(self._action()) == "unspecified"

    def test_unspecified_still_applies_everywhere(self):
        """Behaviour is unchanged -- only the claim it makes is now separable from silence."""
        from merlin.kernels.action_catalog import applies_to_shape
        assert applies_to_shape(self._action(), "square_large")

    def test_unvalidated_scope_names_the_guesses_being_made(self):
        from merlin.kernels.action_catalog import unvalidated_scope
        actions = [self._action(shape_regimes=("skinny",)), self._action(shape_agnostic=True),
                   self._action()]
        guessed = unvalidated_scope(actions, "skinny")
        assert len(guessed) == 1 and guessed[0].shape_regimes == ()

    def test_an_action_that_does_not_fire_here_is_not_a_guess_here(self):
        from merlin.kernels.action_catalog import unvalidated_scope
        assert unvalidated_scope([self._action(shape_regimes=("skinny",))], "square_large") == ()

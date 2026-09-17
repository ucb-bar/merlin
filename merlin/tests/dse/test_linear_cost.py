"""Linear per-command cost model: prediction, band, folds, round-trip, and fail-closed resolution."""

import json

import pytest

import merlin.perf.linear_cost as LC
from merlin.perf.linear_cost import CostModelUnavailable, LinearCostModel, fit_linear

EVENTS = ("config", "mvin_A", "mvin2_B", "compute", "mvout", "fence")
FOLD = {"bias": {"into": "mvin_A", "scale": {"datapath_bits_ratio": ["accumulator", "input"]}}}


def _model(**kw):
    return LinearCostModel(
        const=50.0,
        coeff={"config": 5, "mvin_A": 20, "mvin2_B": 20, "compute": 16, "mvout": 18, "fence": 40},
        error={"mape": 0.1, "max_abs_pct": 0.2, "n_points": 14},
        events=EVENTS,
        **kw,
    )


def test_predict_linear():
    m = _model()
    # one full tile: mvin2_B + mvin_A + compute + mvout + fence
    ev = {"mvin2_B": 1, "mvin_A": 1, "compute": 1, "mvout": 1, "fence": 1}
    assert m.predict(ev) == 50 + 20 + 20 + 16 + 18 + 40


def test_predict_with_band_uses_mape():
    m = _model()
    cyc, band = m.predict_with_band({"compute": 10})
    assert band == cyc * 0.1


def test_resident_rhs_ordering_is_recovered():
    """The model must rank hoisted < baseline (the Stage-F decision) for a reused RHS."""
    m = _model()
    R = 16
    baseline = m.predict({"mvin2_B": R, "mvin_A": R, "compute": R, "mvout": R, "fence": 1})
    hoisted = m.predict({"mvin2_B": 1, "mvin_A": R, "compute": R, "mvout": R, "fence": 1})
    assert hoisted < baseline  # the whole point of resident_packed_tensor


def test_save_load_round_trip(tmp_path):
    m = _model()
    p = tmp_path / "c.json"
    m.save(p)
    m2 = LinearCostModel.load(p)
    assert m2.const == m.const and m2.coeff == m.coeff and m2.error == m.error
    # a bare coefficient file declares no vocabulary: its keys are the priced events, sorted
    assert m2.priced_events() == tuple(sorted(EVENTS)) and m2.folds == {}


def test_fold_is_priced_by_the_datapath_width_ratio(monkeypatch):
    widths = {"accumulator": 32, "input": 8}
    monkeypatch.setattr(LC, "datapath_bits", lambda target, name: widths[name])
    m = _model(folds=FOLD, target="t")
    assert m.fold_scale("bias") == 4
    assert m.predict({"compute": 1, "bias": 1}) - m.predict({"compute": 1}) == 20 * 4


def test_a_fold_without_a_target_fails_closed():
    m = _model(folds=FOLD)
    with pytest.raises(CostModelUnavailable, match="without a target"):
        m.predict({"compute": 1, "bias": 1})
    # a program that issues no folded command needs no scale
    assert m.predict({"compute": 1}) == 66


def test_a_fold_whose_width_is_not_in_the_rtl_facts_fails_closed(monkeypatch):
    from merlin.targetgen.rtl import facts as rtl_facts

    monkeypatch.setattr(
        rtl_facts, "load_facts", lambda target: {"facts": {"datapaths": [{"name": "input", "dtype": "i8"}]}}
    )
    m = _model(folds=FOLD, target="t")
    with pytest.raises(CostModelUnavailable, match="no 'accumulator' datapath"):
        m.predict({"bias": 1})


def test_a_target_without_a_calibration_has_no_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_TARGETS_DIR", str(tmp_path))
    assert LC.cost_model_artifact("t") is None
    assert LC.cost_model_artifact("") is None
    with pytest.raises(CostModelUnavailable, match="no calibrated cost model"):
        LinearCostModel.for_target("t")


def test_for_target_reads_the_declared_vocabulary(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_TARGETS_DIR", str(tmp_path))
    home = tmp_path / "t" / "cost_model"
    home.mkdir(parents=True)
    _model().save(home / "coefficients.json")
    (home / "vocabulary.json").write_text(json.dumps({"events": list(EVENTS), "folds": FOLD}))
    m = LinearCostModel.for_target("t")
    assert m.priced_events() == EVENTS and m.folds == FOLD and m.target == "t"

    (home / "vocabulary.json").write_text(json.dumps({"events": [*EVENTS, "never_fitted"]}))
    with pytest.raises(CostModelUnavailable, match="never fitted"):
        LinearCostModel.for_target("t")


def test_fit_linear_recovers_a_noise_free_model():
    truth = _model()
    rows = []
    for i, e in enumerate(EVENTS):
        for n in (1, 5):
            ev = {k: 0.0 for k in EVENTS}
            ev[e], ev["fence"] = n, 1 + i % 2
            rows.append({"events": ev, "cycles": truth.predict(ev)})
    fit = fit_linear(rows, EVENTS, meta={"fit": "test"})
    assert fit.const == pytest.approx(truth.const)
    assert fit.coeff == pytest.approx(truth.coeff)
    assert fit.error["mape"] == pytest.approx(0.0, abs=1e-9) and fit.error["n_points"] == len(rows)
    assert fit.events == EVENTS and fit.meta == {"fit": "test"}

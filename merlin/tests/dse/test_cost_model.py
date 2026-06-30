"""Gemmini cost-model logic: linear prediction, mvin3 byte-folding, band, round-trip."""
from merlin.cost_model.gemmini import EVENTS, MVIN3_BYTE_RATIO, GemminiCostModel


def _model():
    return GemminiCostModel(
        const=50.0,
        coeff={"config": 5, "mvin_A": 20, "mvin2_B": 20, "compute": 16, "mvout": 18, "fence": 40},
        error={"mape": 0.1, "max_abs_pct": 0.2, "n_points": 14})


def test_predict_linear():
    m = _model()
    # one full tile: mvin2_B + mvin_A + compute + mvout + fence
    ev = {"mvin2_B": 1, "mvin_A": 1, "compute": 1, "mvout": 1, "fence": 1}
    assert m.predict(ev) == 50 + 20 + 20 + 16 + 18 + 40


def test_mvin3_folds_into_mvin_by_byte_ratio():
    m = _model()
    base = m.predict({"compute": 1})
    withbias = m.predict({"compute": 1, "mvin3_bias": 1})
    assert withbias - base == 20 * MVIN3_BYTE_RATIO


def test_predict_with_band_uses_mape():
    m = _model()
    cyc, band = m.predict_with_band({"compute": 10})
    assert band == cyc * 0.1


def test_resident_rhs_ordering_is_recovered():
    """Cost model must rank hoisted < baseline (the Stage-F decision) for reused RHS."""
    m = _model()
    R = 16
    baseline = m.predict({"mvin2_B": R, "mvin_A": R, "compute": R, "mvout": R, "fence": 1})
    hoisted = m.predict({"mvin2_B": 1, "mvin_A": R, "compute": R, "mvout": R, "fence": 1})
    assert hoisted < baseline  # the whole point of resident_packed_tensor


def test_save_load_round_trip(tmp_path):
    m = _model()
    p = tmp_path / "c.json"
    m.save(p)
    m2 = GemminiCostModel.load(p)
    assert m2.const == m.const and m2.coeff == m.coeff
    assert set(m2.coeff) == set(EVENTS)

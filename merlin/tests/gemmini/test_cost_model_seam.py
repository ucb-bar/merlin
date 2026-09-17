"""This target's linear cost model is reached through the generic seam, never by importing it.

The coefficients and vocabulary live in the target's own directory and are resolved by name; the bias
fold's scale is the accumulator/input datapath width ratio read from the RTL facts, not a literal.
"""

import pytest

from merlin.perf.linear_cost import LinearCostModel, cost_model_artifact, datapath_bits

TARGET = "gemmini"


def test_the_model_resolves_by_target_name():
    artifact = cost_model_artifact(TARGET)
    assert artifact is not None and artifact.is_file()
    model = LinearCostModel.for_target(TARGET)
    assert model.priced_events() and set(model.priced_events()) <= set(model.coeff)
    assert model.error.get("mape", 0.0) > 0.0, "a calibrated model must carry its error band"


def test_the_bias_fold_is_scaled_by_the_rtl_datapath_widths():
    model = LinearCostModel.for_target(TARGET)
    ratio = datapath_bits(TARGET, "accumulator") / datapath_bits(TARGET, "input")
    assert model.fold_scale("mvin3_bias") == ratio
    into = model.folds["mvin3_bias"]["into"]
    delta = model.predict({"compute": 1, "mvin3_bias": 1}) - model.predict({"compute": 1})
    assert delta == pytest.approx(model.coeff[into] * ratio)


def test_a_bare_load_prices_the_coefficient_keys_without_folds():
    model = LinearCostModel.load(cost_model_artifact(TARGET))
    assert model.priced_events() == tuple(sorted(model.coeff)) and model.folds == {}

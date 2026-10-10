"""Ordinary complete-cost feedback refuses numeric boolean substitutions."""

from copy import deepcopy

import pytest
from merlin_experiments.phase2 import component_workflow as W
from test_component_screening import document

from merlin.xdsl_dialects.lowering.global_plan import CycleInterval


def _interval_type_feedback(resolved):
    """Use the ordinary feedback reader, without any execution capability."""
    selected = document()
    selected.pop("screening")
    selected["evidence"] = selected["evidence"][:1]
    if not resolved:
        interval = CycleInterval.unknown("unqualified complete cost").to_dict()
        row = selected["evidence"][0]
        for arm in ("baseline", "candidate"):
            row[arm] = deepcopy(interval)
            for regime in row["complete_cost"][arm]["regimes"].values():
                regime["total"] = deepcopy(interval)
                for region in regime["regions"]:
                    region["cycles"] = deepcopy(interval)
    return selected


@pytest.mark.parametrize("resolved", [False, True])
def test_normal_feedback_preserves_exact_boolean_complete_cost_states(resolved):
    selected = _interval_type_feedback(resolved)
    assert W.validate_component_feedback(selected) == selected
    assert selected["promotion"] == "NO_FINAL_ACCEPTANCE"
    assert all(row[arm]["resolved"] is resolved for row in selected["evidence"] for arm in ("baseline", "candidate"))


@pytest.mark.parametrize("arm", ["baseline", "candidate"])
@pytest.mark.parametrize("regime", ["cold", "warm"])
@pytest.mark.parametrize("location", ["total", "region"])
@pytest.mark.parametrize("resolved", [False, True])
def test_normal_feedback_refuses_numeric_nested_complete_cost_states(arm, regime, location, resolved):
    selected = _interval_type_feedback(resolved)
    report = selected["evidence"][0]["complete_cost"][arm]["regimes"][regime]
    interval = report["total"] if location == "total" else report["regions"][0]["cycles"]
    interval["resolved"] = int(resolved)
    with pytest.raises(ValueError, match="interval schema"):
        W.validate_component_feedback(selected)

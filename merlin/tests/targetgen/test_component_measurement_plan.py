"""Pure scheduling arithmetic cannot issue physical or runtime owners."""

from dataclasses import replace

import pytest

from merlin.common.jsonio import canonical_sha256
from merlin.perf.component_measurement_plan import plan_development_measurements as plan
from merlin.perf.component_screen import ComponentOpportunity, ComponentScreenPolicy
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval


def row(label, baseline=(10, 12), candidate=(11, 13), seconds=2, legal=True, correct=True):
    return ComponentOpportunity(
        canonical_sha256(label),
        "independent_family",
        "warm",
        CycleInterval(*baseline, ("synthetic arithmetic only",)),
        CycleInterval(*candidate, ("synthetic arithmetic only",)),
        seconds,
        legal,
        correct,
    )


def decide(rows, **updates):
    kwargs = dict(
        applicability={r.id: "IN_DOMAIN" for r in rows},
        owner_available=True,
        purpose="development_performance",
        max_measurements=2,
    )
    kwargs.update(updates)
    return plan(tuple(rows), **kwargs)


def test_disjoint_predictions_defer_only_development_but_overlap_uses_cheapest_distinguishing_case():
    rows = [
        row("faster", candidate=(1, 2)),
        row("slower", candidate=(20, 30)),
        row("slow_overlap", seconds=9),
        row("cheap_overlap", seconds=1),
        row("remaining", seconds=3),
    ]
    decisions, selected = decide(rows)
    assert [r.action for r in decisions] == ["DEFER", "DEFER", "PENDING", "MEASURE", "MEASURE"]
    assert selected == (rows[3].id, rows[4].id)
    assert len(decisions) == len(rows)


@pytest.mark.parametrize("purpose", ["development_correctness", "held", "final_performance", True])
def test_original_correctness_held_and_final_denominators_cannot_be_pruned(purpose):
    with pytest.raises(ValueError, match="correctness, held or final"):
        decide([row("case")], purpose=purpose)


def test_missing_owners_and_new_joint_cells_never_defer_even_disjoint_intervals():
    original = row("case", candidate=(1, 2))
    decisions, selected = decide([original], owner_available=False)
    assert decisions[0].action == "UNAVAILABLE" and not selected
    decisions, selected = decide([original], applicability={original.id: "UNKNOWN"})
    assert decisions[0].action == "REQUALIFY" and not selected
    decisions, selected = decide([replace(original, candidate=CycleInterval.unknown("unsupported feature"))])
    assert decisions[0].action == "REQUALIFY" and not selected


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("legal", False, "REFUSAL"),
        ("correct", False, "REFUSAL"),
        ("legal", None, "UNAVAILABLE"),
        ("correct", None, "UNAVAILABLE"),
    ],
)
def test_qualification_gates_remain_distinct_from_prediction_intervals(field, value, expected):
    original = row("case", candidate=(1, 2))
    decisions, selected = decide([replace(original, **{field: value})])
    assert decisions[0].action == expected and not selected


@pytest.mark.parametrize("seconds", [None, True, float("nan"), 0, -1])
def test_unknown_or_invalid_experiment_cost_cannot_select_a_measurement(seconds):
    decisions, selected = decide([row("case", seconds=seconds)])
    assert decisions[0].action == "UNAVAILABLE" and not selected


def test_exact_touching_intervals_still_require_measurement():
    original = row("case", candidate=(12, 20))
    decisions, selected = decide([original])
    assert decisions[0].action == "MEASURE" and selected == (original.id,)


@pytest.mark.parametrize("defect", ["empty", "duplicate", "missing", "bool_budget", "oversized", "alien"])
def test_closed_membership_and_pre_expansion_bounds(defect):
    rows = [row("case")]
    updates = {}
    if defect == "empty":
        rows = []
    elif defect == "duplicate":
        rows *= 2
    elif defect == "missing":
        updates["applicability"] = {}
    elif defect == "bool_budget":
        updates["max_measurements"] = True
    elif defect == "oversized":
        updates["max_members"] = 10001
    else:
        rows = [replace(rows[0], regime="held")]
    with pytest.raises(ValueError):
        decide(rows, **updates)


def test_original_screening_acceptance_thresholds_are_unchanged():
    assert ComponentScreenPolicy().to_dict() == {
        "minimum_rank_rate": 0.95,
        "minimum_decided": 100,
        "minimum_slice_decided": 20,
        "minimum_slices": 3,
        "minimum_predictions": 20,
        "maximum_relative_error": 0.10,
        "minimum_interval_coverage": 0.95,
    }

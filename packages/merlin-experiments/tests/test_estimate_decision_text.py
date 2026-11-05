"""Agent-facing performance text makes estimates the basis for decisions, under honest evidence labels.

On one run the author was told UNKNOWN timing "stays unranked" and "cannot establish a speedup" for
every large case, whose complete observation could not fit a grade's budget; it stopped deciding on
anything. Estimates (complete, else the calibrated fast estimate) now decide, with a revert rule.
"""

from merlin_experiments.phase1.feedback import joint_outcome_brief as O
from merlin_experiments.phase1.feedback import next_steps as N
from merlin_experiments.phase1.feedback import performance_brief as PB
from merlin_experiments.phase1.feedback import shape_panel as SP
from merlin_experiments.phase1.providers import authoring_prompt as AP

BLOCKING = ("stays unranked", "is unranked", "remain unranked", "unranked hypothesis",
            "obtain cycles before deciding", "never claim physical speedup", "no instruction-count fallback",
            "UNKNOWN stays outside cycle rankings")


def _texts():
    signals = O.performance_signals({"status": "UNKNOWN"})
    return {
        "next_steps": N.DECIDE_ON_ESTIMATES,
        "authoring": AP.COEVOLUTION_FEEDBACK_GUIDANCE,
        "shape_panel": SP.RULE,
        "unknown_timing": PB.unknown_timing(["conv"], {})["meaning"],
        "missing_timing": signals["performance_objective"]["missing_timing"],
    }


def test_agent_facing_text_decides_on_estimates_and_keeps_a_revert_rule():
    for where, text in _texts().items():
        assert not any(phrase in text for phrase in BLOCKING), where
    assert "revert" in N.DECIDE_ON_ESTIMATES and "fast estimate" in N.DECIDE_ON_ESTIMATES
    assert "fast_estimate" in AP.COEVOLUTION_FEEDBACK_GUIDANCE and "revert" in AP.COEVOLUTION_FEEDBACK_GUIDANCE
    assert "fast_estimate" in _texts()["missing_timing"]


def test_outcome_points_a_deferred_case_at_its_fast_estimate():
    outcomes = {"next_priority": "timing_model_or_observation", "next_owner": "operator", "next_action": "inspect"}
    O.decide_on_fast_estimate(outcomes, {"status": "UNKNOWN", "cycles": None})
    assert outcomes["next_owner"] == "operator"  # no estimate: unchanged
    O.decide_on_fast_estimate(outcomes, {"status": "ESTIMATED", "cycles": 3_900_000, "low_cycles": 2_500_000,
                                         "high_cycles": 6_400_000})
    assert outcomes["next_priority"] == "performance_search" and outcomes["next_owner"] == "compiler"
    assert "3900000" in outcomes["next_action"] and "2500000-6400000" in outcomes["next_action"]
    assert "revert" in outcomes["next_action"]

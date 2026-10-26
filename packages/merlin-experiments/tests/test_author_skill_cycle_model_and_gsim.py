"""The cycle-model-and-gsim skill quotes the evidence labels, fields and rules the harness publishes."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

_CHECKS = importlib.util.spec_from_file_location("author_skill_perf_checks",
                                                 Path(__file__).with_name("author_skill_perf_checks.py"))
C = importlib.util.module_from_spec(_CHECKS)
_CHECKS.loader.exec_module(C)

NAME = "cycle-model-and-gsim"


def test_skill_layout_is_generic_and_its_fields_and_commands_exist():
    C.check_layout(NAME)
    C.check_generic(NAME)
    assert C.missing_fields(NAME) == []
    assert C.bad_commands(NAME) == []
    assert {"roofline-and-utilization", "ab-library-pass"} <= C.skills_named(NAME)


def test_evidence_labels_are_the_ones_the_writers_emit():
    from merlin_experiments.phase1.feedback import performance_brief, performance_headroom

    _, body = C.split(NAME)
    writers = inspect.getsource(performance_brief) + inspect.getsource(performance_headroom)
    for label in ("measured: gSIM cycles of this request", "measured: latest gSIM certificate"):
        assert label in writers and label in body
    assert "modeled: Spike timing model (self-check)" in writers
    assert "modeled: Spike timing model ({performance['status'].lower()})" in writers
    assert "`modeled: Spike timing model (provisional\\|qualified\\|self-check)`" in body


def test_the_comparison_rule_is_the_calibration_rule():
    from merlin_experiments.phase1.feedback import model_calibration

    assert "descriptive" in model_calibration.USE and "not a tuned timing" in model_calibration.USE
    assert "small/medium" in model_calibration.USE and "Spike-only" in model_calibration.USE
    fitted = model_calibration.fit([{"ratio": r, "family": "f", "bound_class": "compute"}
                                    for r in (0.9, 1.0, 1.1, 1.2, 1.3)])
    assert set(fitted["levels"]["family"]["f"]) == {"n", "median_measured_over_modeled", "interquartile", "spread"}
    assert model_calibration.fit([{"ratio": 1.0}], min_pairs=5)["levels"]["all"]["all"]["status"] == "uncalibrated"
    applied = model_calibration.apply(1000, fitted, family="f")
    assert {"calibrated_estimate", "calibrated_interval", "model_confidence"} <= set(applied)
    _, body = C.split(NAME)
    for word in ("median_measured_over_modeled", "interquartile", "spread", "uncalibrated",
                 "calibrated_estimate", "calibrated_interval", "model_confidence"):
        assert word in body
    assert "does not prove schedule ranking" in body and "Do not wait for calibration" in body


def test_only_complete_accounting_counts_as_a_modeled_estimate():
    from merlin_experiments.phase1.feedback import joint_performance_report

    report = {"counter_windows": [{"estimated_cycles": 500, "begin_pc": 1, "end_pc": 2}], "estimated_cycles": 500,
              "scope_sha256": "a" * 64, "unresolved": ["an unpriced event"], "timing_complete": True}
    observation = {"status": "observed", "complete_counts": True,
                   "scope_binding": {"begin_pc": 1, "end_pc": 2, "scope_sha256": "a" * 64}}
    value = joint_performance_report.project(report, observation, {"structural_compute_floor_cycles": 100},
                                             selection_sha256="b" * 64, reason="r")
    assert value["status"] == "UNKNOWN" and value["complete_event_accounting"] is False
    report["unresolved"] = []
    value = joint_performance_report.project(report, observation, {"structural_compute_floor_cycles": 100},
                                             selection_sha256="b" * 64, reason="r")
    assert value["status"] == "PROVISIONAL" and value["estimated_whole_soc_cycles"] == 500
    assert value["structural_floor_to_estimate_ratio"] == 0.2

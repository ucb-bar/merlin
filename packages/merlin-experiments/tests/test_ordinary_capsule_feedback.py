"""Descriptive feedback arithmetic; synthetic fixtures qualify no target timing."""

from copy import deepcopy

import pytest
from merlin_experiments.phase2 import ordinary_capsule_feedback as F


@pytest.fixture
def compiled():
    return {
        "status": "observed",
        "elf_sha256": "e" * 64,
        "candidate_diagnostics": {
            "candidate_features": {"features": {"source_macs": 1000}},
            "host": {"status": "observed", "logical_load_bytes": 1000000},
            "representations": {"representation_directives": [{"packing": "declared"}]},
        },
    }


@pytest.fixture
def machine():
    return {
        "peak_macs_per_cycle": 100,
        "read_bytes_per_cycle": 16,
        "write_bytes_per_cycle": 16,
        "basis": {"compute": "synthetic explicit arithmetic inventory"},
        "unresolved": {},
        "scope_sha256": "s" * 64,
        "device_sha256": "d" * 64,
    }


@pytest.fixture
def raw(compiled):
    return {
        "schema": "merlin.annotated_spike_report.v2",
        "elf_sha256": compiled["elf_sha256"],
        "estimated_cycles": 7,
        "timing_complete": False,
        "instructions": 2,
        "execution_completed": True,
        "functional_status": "PASS",
        "raw_functional_exit_code": 0,
        "unresolved": ["native completion and resources unknown"],
        "feature_counts": {
            "cpu.integer": 2,
            "target.nominal_padded_macs": 4096,
            "target.useful_shape_macs": 1200,
            "target.requested_read_bytes": 100000,
        },
        "feature_observation": {
            "schema": "merlin.spike_feature_observation.v1",
            "complete": True,
            "scope_instructions": 2,
            "counted_cpu_instructions": 2,
            "scope_kind": "whole_program",
            "window_index": None,
            "primary_cpu_features": ["cpu.integer"],
            "unresolved": [],
            "qualification": "not_established",
        },
    }


def join(compiled, machine, **kwargs):
    return F.join_capsule_feedback(
        {"name": "public-example", "kind": "model"}, compiled, target="fixture-target", machine=machine, **kwargs
    )


def measurement(compiled, machine, cycles):
    return {
        "kind": "measured_target_cycles",
        "measured_now": True,
        "cycles": cycles,
        "elf_sha256": compiled["elf_sha256"],
        "functional_status": "PASS",
        "complete_source_scope": True,
        **{key: machine[key] for key in ("scope_sha256", "device_sha256")},
    }


def test_useful_source_work_is_distinct_from_padded_requests_and_no_bytes_price_movement(compiled, machine, raw):
    report = join(compiled, machine, raw_report=raw)
    assert report["source_work"]["macs"] == 1000
    assert report["dynamic"]["feature_counts"]["target.nominal_padded_macs"] == 4096
    assert report["dynamic"]["feature_counts"]["target.useful_shape_macs"] == 1200
    assert report["roofline"]["floor"]["compute_floor_cycles"] == 10
    assert report["roofline"]["floor"]["movement_floor_cycles"] is None
    assert report["roofline"]["floor"]["compulsory_read_bytes"] is None
    assert report["dynamic"]["timing_cycles"] is None
    assert report["timing_cycles"] is None
    assert report["roofline"]["attainment"] is None


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x["feature_observation"].update(complete=False),
        lambda x: x["feature_observation"].update(scope_instructions=3),
        lambda x: x.update(elf_sha256="different"),
        lambda x: x["feature_counts"].update({"target.nominal_padded_macs": -1}),
        lambda x: x["feature_counts"].update({"cpu.integer": True}),
    ],
)
def test_incomplete_or_mismatched_count_report_is_unknown(compiled, machine, raw, change):
    change(raw)
    report = join(compiled, machine, raw_report=raw)
    assert report["dynamic"]["status"] == "UNKNOWN"
    assert report["dynamic"]["feature_counts"] is None
    assert report["timing_cycles"] is None


def test_qualified_empirical_prediction_never_supplies_measured_cycles_or_attainment(compiled, machine):
    report = join(
        compiled,
        machine,
        prediction={
            "status": "qualified_screen",
            "estimated_cycles": 5,
            "interval": [4, 6],
            "screen_terms": {"features": [{"name": "command", "estimated_cycles": 5}]},
        },
    )
    assert report["screen"]["estimated_cycles"] == 5
    assert "correlated" in report["screen"]["interpretation"]
    assert report["timing_cycles"] is None
    assert report["roofline"]["status"] == "UNKNOWN"
    assert report["roofline"]["attainment"] is None
    assert not report["convergence_eligible"]


@pytest.mark.parametrize("cycles", [True, -1, float("nan"), None])
def test_invalid_empirical_estimates_remain_unknown(compiled, machine, cycles):
    report = join(compiled, machine, prediction={"status": "qualified_screen", "estimated_cycles": cycles})
    assert report["screen"]["status"] == "UNKNOWN"
    assert report["screen"]["estimated_cycles"] is None
    assert report["timing_cycles"] is None


def test_unknown_normalized_source_does_not_become_zero_or_descriptor_work(compiled, machine):
    compiled["candidate_diagnostics"]["candidate_features"]["features"]["source_macs"] = None
    report = F.join_capsule_feedback({"source_macs": 999999}, compiled, target="fixture-target", machine=machine)
    assert report["source_work"]["status"] == "UNKNOWN"
    assert report["source_work"]["macs"] is None
    assert report["roofline"]["floor"]["roofline_cycles"] is None


def test_actual_paired_rows_can_refute_partial_floor_without_clamping_or_converging(compiled, machine):
    baseline = join(compiled, machine, measurement=measurement(compiled, machine, 20))
    candidate = join(compiled, machine, measurement=measurement(compiled, machine, 9))
    assert baseline["roofline"]["attainment"] == 0.5
    assert baseline["roofline"]["gap_cycles"] == 10
    assert candidate["roofline"]["status"] == "floor_refuted"
    assert candidate["roofline"]["gap_cycles"] == -1
    assert candidate["roofline"]["attainment"] is None
    assert not baseline["roofline"]["convergence_eligible"]
    assert set(baseline["roofline"]["floor"]["unresolved"]) >= {
        "complete_call_host",
        "packing_layout",
        "ports",
        "overlap",
        "placement",
    }


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "empirical_screen"},
        {"measured_now": False},
        {"cycles": True},
        {"cycles": float("nan")},
        {"elf_sha256": "different"},
        {"functional_status": "FAIL"},
        {"scope_sha256": "different"},
        {"device_sha256": "different"},
        {"complete_source_scope": False},
    ],
)
def test_predictions_or_foreign_measurements_cannot_price_floor(compiled, machine, change):
    observed = {**measurement(compiled, machine, 20), **change}
    report = join(compiled, machine, measurement=observed)
    assert report["timing_cycles"] is None
    assert report["roofline"]["attainment"] is None


def test_facts_only_derivation_and_missing_arithmetic_do_not_use_geometry_or_supplied_rate(compiled, machine):
    facts = {"facts": {"arrays": [{"rows": 2, "cols": 4}]}}
    report = F.join_capsule_feedback({}, compiled, target="fixture-target", facts=facts, machine=machine)
    assert report["machine"]["peak_macs_per_cycle"] is None
    assert report["roofline"]["floor"]["roofline_cycles"] is None
    facts["facts"]["arrays"][0]["mac_idiom"] = {"muls": 1}
    report = F.join_capsule_feedback({}, compiled, target="fixture-target", facts=facts)
    assert report["machine"]["peak_macs_per_cycle"] == 8
    assert report["roofline"]["floor"]["compute_floor_cycles"] == 125


def test_join_does_not_mutate_inputs_and_retains_host_packing_scope(compiled, machine, raw):
    originals = deepcopy((compiled, machine, raw))
    report = join(compiled, machine, raw_report=raw)
    assert (compiled, machine, raw) == originals
    assert report["host_and_packing"]["host"]["logical_load_bytes"] == 1000000
    assert report["host_and_packing"]["physical_traffic_bytes"] is None
    assert report["compiled"]["representations"]["representation_directives"]
    report["dynamic"]["feature_counts"]["cpu.integer"] = 100
    assert raw["feature_counts"]["cpu.integer"] == 2

"""Synthetic same-window transport/arithmetic, never device qualification."""

import copy

import pytest
from merlin_timing.optimization_feedback import project
from merlin_timing.scoped_resources import SCHEMA


def performance():
    return dict(
        scope_sha256="a" * 64,
        elf_sha256="c" * 64,
        model_sha256="b" * 64,
        estimated_whole_soc_cycles=10,
        structural_roofline=dict(
            useful_macs=1000,
            structural_peak_macs_per_cycle=256,
            structural_compute_floor_cycles=4,
            logical_request_ideal_service={"read": {"requested_bytes": 32}},
        ),
        scoped_resource_feedback=dict(
            schema=SCHEMA,
            status="modeled",
            source_kind="modeled",
            scope_sha256="a" * 64,
            model_sha256="b" * 64,
            elf_sha256="c" * 64,
            estimated_cycles_same_window=10,
            activity_status="modeled",
            compute=dict(useful_macs=1000, nominal_padded_macs=2000),
            commands=dict(total=5, fences=1),
            host={"instruction_counts": {"cpu.integer": 100}},
            traffic=[
                dict(boundary="device_requested_read", bytes=64, source_kind="observed_requests_over_modeled_cycles")
            ],
            resources=[
                dict(
                    resource_id=0,
                    name="generic.compute",
                    active_fraction=0.8,
                    capacity_normalized_activity=0.4,
                    average_modeled_waiting_entries=0,
                    queue_active_cycles=0,
                    peak_queue_entries=0,
                )
            ],
            overlaps=[dict(left=0, right=1, active_cycles=3, fraction_of_window=0.3, source_kind="modeled")],
        ),
        # Deliberately inconsistent and enormous. These must never enter ROI feedback.
        modeled_resource_activity={"scope": "whole_trace", "busy_unit_cycles": 10**12},
        cycles=999999,
    )


def test_compact_real_metrics_keep_modeled_reservations_separate_from_physical_unknowns():
    source = performance()
    before = copy.deepcopy(source)
    value = project(source)
    assert value["status"] == "modeled"
    assert value["cycles"] == dict(value=10, source_kind="modeled", unit="model_cycles", measured_cycles=None)
    assert value["work"]["compute_ceiling_attainment"] == 1000 / (256 * 10)
    assert value["work"]["structural_floor_to_modeled_cycles"] == 0.4
    assert value["work"]["useful_fraction_of_nominal_padded_work"] == 0.5
    assert value["reservations"][0]["reservation_active_fraction"] == 0.8
    assert value["reservations"][0]["modeled_average_waiting_entries"] == 0
    assert value["physical"]["accelerator_queue_pressure"] is None
    assert value["physical"]["dependency_stall_cycles"] is None
    assert value["physical"]["storage_byte_highwater"] is None
    assert value["commands"]["compute"] is None
    assert value["traffic"][0]["requested_to_source_logical_ratio"] == 2
    assert value["traffic"][0]["bytes_per_modeled_cycle"] == 6.4
    assert value["overlaps"][0]["fraction_of_window"] == 0.3
    assert {s["opportunity"] for s in value["suggestions"]} == {
        "reduce_nominal_padded_work",
        "inspect_synchronization_frequency",
        "inspect_requested_traffic_amplification",
    }
    assert not value["qualification_issued"] and not value["resource_cycles_additive"]
    assert "999999" not in str(value) and str(10**12) not in str(value)
    assert source == before


def test_compact_host_feedback_keeps_busy_primary_classes_over_zero_counters():
    source = performance()
    counts = {"cpu.atomic": 0, "cpu.bitmanip": 0, "cpu.branch": 100,
              "cpu.branch_taken": 90, "cpu.branch_not_taken": 10,
              "cpu.custom": 200, "cpu.csr": 1, "cpu.divide": 0,
              "cpu.integer": 1000, "cpu.fence": 50, "cpu.load": 20,
              "cpu.fp_multiply": 400, "cpu.fp_multiply.f32": 400}
    source["scoped_resource_feedback"]["host"]["instruction_counts"] = counts
    value = project(source)
    assert value["host_instruction_counts"]["cpu.integer"] == 1000
    assert value["host_instruction_counts"]["cpu.fp_multiply"] == 400
    assert value["host_instruction_counts"]["cpu.fence"] == 50
    assert "cpu.branch_taken" not in value["host_instruction_counts"]
    assert "cpu.fp_multiply.f32" not in value["host_instruction_counts"]
    assert len(value["host_instruction_counts"]) == 8
    assert value["host_instruction_classes_omitted"] == 2


@pytest.mark.parametrize("change", ["scope", "model", "elf", "foreign_elf", "cycles", "incomplete", "absent"])
def test_missing_or_mismatched_roi_does_not_borrow_other_cycles_or_whole_trace(change):
    source = performance()
    scoped = source["scoped_resource_feedback"]
    if change in {"scope", "model"}:
        scoped[change + "_sha256"] = "d" * 64
    elif change == "elf":
        del scoped["elf_sha256"]
    elif change == "foreign_elf":
        scoped["elf_sha256"] = "d" * 64
    elif change == "cycles":
        scoped["estimated_cycles_same_window"] = 11
    elif change == "incomplete":
        scoped["status"] = "UNKNOWN"
    else:
        source.pop("scoped_resource_feedback")
    value = project(source)
    assert value["status"] == "UNKNOWN"
    assert value["cycles"]["value"] is None
    assert value["work"]["source_useful_macs"] is None
    assert value["reservations"] == [] and value["suggestions"] == []


def test_missing_resource_provider_and_source_work_remain_unknown_without_losing_valid_counts():
    source = performance()
    source["structural_roofline"] = {}
    scoped = source["scoped_resource_feedback"]
    scoped["activity_status"] = "UNKNOWN"
    value = project(source)
    assert value["cycles"]["value"] == 10
    assert value["work"]["source_useful_macs"] is None
    assert value["work"]["observed_shape_macs"] == 1000
    assert value["work"]["nominal_padded_macs"] == 2000
    assert value["work"]["compute_ceiling_attainment"] is None
    assert value["reservations"] == [] and value["overlaps"] == []
    assert value["reservation_status"] == "UNKNOWN"
    assert value["traffic"][0]["requested_to_source_logical_ratio"] is None
    assert value["physical"]["array_occupancy"] is None


def test_explicit_zero_work_is_distinct_from_unknown_and_compact_bounds_hold():
    source = performance()
    source["structural_roofline"]["useful_macs"] = 0
    scoped = source["scoped_resource_feedback"]
    scoped["commands"] = {}
    scoped["resources"] *= 20
    scoped["overlaps"] *= 20
    value = project(source)
    assert value["work"]["source_useful_macs"] == 0
    assert value["work"]["compute_ceiling_attainment"] == 0
    assert value["commands"]["target_fences"] is None
    assert len(value["reservations"]) == 8 and value["reservation_rows_omitted"] == 12
    assert len(value["overlaps"]) == 4

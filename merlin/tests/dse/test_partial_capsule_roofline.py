"""Partial work remains actionable without promoting logical bytes to traffic."""

import pytest

from merlin.perf.capsule_roofline import partial_roofline


def test_logical_request_ideal_cannot_become_compulsory_movement_or_whole_cost():
    row = partial_roofline(
        [(2, 4, 8)],
        {"peak_macs_per_cycle": 64, "read_bytes_per_cycle": 4, "write_bytes_per_cycle": 2},
        logical_read_bytes=1000,
        logical_write_bytes=200,
    )
    assert row["structural_compute_floor_cycles"] == 1
    service = row["logical_request_ideal_service"]
    assert service["read"]["ideal_requested_service_cycles"] == 250
    assert service["write"]["ideal_requested_service_cycles"] == 100
    assert row["status"] == "partial"
    assert row["whole_call_cycles"] is row["complete_attainment"] is None
    assert row["convergence_eligible"] is False
    assert "actual compulsory interface traffic" in row["missing_terms"][0]


def test_original_count_requests_remain_distinct_from_logical_unique_payload():
    row = partial_roofline(
        [(2, 4, 8)], {"peak_macs_per_cycle": 64, "read_bytes_per_cycle": 4},
        logical_read_bytes=100, client_requests={
            "read_bytes": 600, "write_bytes": None, "provenance": "original checked count report",
        },
    )
    assert row["logical_request_ideal_service"]["read"]["ideal_requested_service_cycles"] == 25
    actual = row["provided_client_request_ideal_service"]["service"]
    assert actual["read"]["ideal_requested_service_cycles"] == 150
    assert actual["write"]["ideal_requested_service_cycles"] is None
    assert row["structural_compute_floor_cycles"] == 1
    assert row["whole_call_cycles"] is None


def test_two_different_machines_price_known_work_without_target_names():
    slow = partial_roofline([(3, 7, 5)], {"peak_macs_per_cycle": 32, "read_bytes_per_cycle": 8},
                            logical_read_bytes=240)
    wide = partial_roofline([(3, 7, 5)], {"peak_macs_per_cycle": 128, "read_bytes_per_cycle": 32},
                            logical_read_bytes=240)
    assert slow["useful_macs"] == wide["useful_macs"] == 105
    assert (slow["structural_compute_floor_cycles"], wide["structural_compute_floor_cycles"]) == (4, 1)
    assert slow["logical_request_ideal_service"]["read"]["ideal_requested_service_cycles"] == 30
    assert wide["logical_request_ideal_service"]["read"]["ideal_requested_service_cycles"] == 7.5
    assert wide["logical_request_ideal_service"]["write"]["ideal_requested_service_cycles"] is None


def test_padding_is_conditional_and_does_not_replace_useful_structural_floor():
    row = partial_roofline([(1, 8, 8)], {"peak_macs_per_cycle": 32},
                           padded_macs=512, padding_basis="explicit hypothetical full tiles")
    assert row["structural_compute_floor_cycles"] == 2
    assert row["padded_work"]["conditional_compute_service_cycles"] == 16
    assert row["padded_work"]["compulsory_workload_floor"] is False


def test_store_pressure_never_establishes_physical_placement_or_ports():
    row = partial_roofline(None, {}, store_requests={
        "one_copy": {"capacity_bytes": 1024, "requested_bytes": 768, "provenance": "full-buffer premise"},
        "two_copies": {"capacity_bytes": 1024, "requested_bytes": 1536, "provenance": "two-buffer premise"},
        "unknown": {"capacity_bytes": None, "requested_bytes": 64, "provenance": "capacity absent"},
    })
    assert row["storage_requests"]["one_copy"]["pressure_ratio"] == 0.75
    assert row["storage_requests"]["one_copy"]["conditional_request_fits"] is True
    assert row["storage_requests"]["two_copies"]["conditional_request_fits"] is False
    assert row["storage_requests"]["unknown"]["conditional_request_fits"] is None
    assert all(not store["physical_placement_proven"] for store in row["storage_requests"].values())
    assert row["whole_call_cycles"] is None


def test_missing_arithmetic_counts_do_not_suppress_known_logical_work():
    row = partial_roofline([(2, 4, 8)], {"array_rows": 8, "array_cols": 8}, logical_read_bytes=64)
    assert row["useful_macs"] == 64
    assert row["structural_compute_floor_cycles"] is None
    assert row["logical_request_ideal_service"]["read"]["requested_bytes"] == 64
    assert "compute" in row["machine_unresolved"]
    assert partial_roofline(None, None)["status"] == "unknown"


@pytest.mark.parametrize("kwargs", [
    {"logical_read_bytes": True}, {"logical_write_bytes": -1}, {"logical_read_bytes": 2**64},
    {"padded_macs": 512}, {"padding_basis": "no count"},
    {"padded_macs": 1, "padding_basis": "omits useful products"},
    {"store_requests": {"x": {"capacity_bytes": True, "requested_bytes": 1, "provenance": "x"}}},
    {"store_requests": {"x": {"capacity_bytes": 2**64, "requested_bytes": 1, "provenance": "x"}}},
    {"store_requests": {"x": {"capacity_bytes": 16, "requested_bytes": True, "provenance": "x"}}},
    {"client_requests": {"read_bytes": 1, "write_bytes": 2}},
    {"client_requests": {"read_bytes": True, "write_bytes": None, "provenance": "x"}},
])
def test_malformed_or_unproven_demands_refuse(kwargs):
    with pytest.raises(ValueError):
        partial_roofline([(2, 4, 8)], {"peak_macs_per_cycle": 32}, **kwargs)

"""The movement peak comes from the timing structure's own ports; the staged roofline keeps stages apart."""

from __future__ import annotations

import copy

import pytest

from merlin.perf import capsule_roofline as CR
from merlin.perf import derived_bound as DB
from merlin.perf import movement_peak as MP
from merlin.perf import staged_roofline as SR

#: A synthetic machine: an 8x8 array, an operand store of four banks and a result store of two, and a
#: movement engine with a read client (data on the response) and a write client (data on the request).
_BODY: dict = {
    "arrays": [{"name": "mesh", "rows": 8, "cols": 8, "mac_idiom": {"muls": 1}}],
    "datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}],
    "memories": [
        {"name": "scratchpad", "banks": 4, "bytes": 8192, "depth": 256, "row_elems": 8},
        {"name": "accumulator", "banks": 2, "bytes": 4096, "depth": 32, "lanes": 8},
    ],
    "timing_structure": {
        "memory_clients": {
            "status": "derived",
            "clients": {
                "reader": {
                    "request_data_bits": 64,
                    "response_data_bits": 64,
                    "request_channels": 1,
                    "beat_bytes": 8,
                    "max_request_bytes": 32,
                    "in_flight": {"status": "derived", "value": 4},
                },
                "writer": {
                    "request_data_bits": 64,
                    "response_data_bits": None,
                    "request_channels": 1,
                    "beat_bytes": 8,
                    "max_request_bytes": 32,
                    "in_flight": {"status": "derived", "value": 4},
                },
            },
        },
        "fit_from_measurement": ["memory_round_trip_latency_cycles"],
    },
}


def _machine(body=_BODY) -> DB.Machine:
    return DB.machine_from_facts("synthetic", facts={"facts": body}, measure_fill=False)


def test_direction_is_structural_and_the_in_flight_window_is_a_binding_latency_not_a_rate() -> None:
    peak = MP.movement_peak({"facts": _BODY})
    assert peak["status"] == "derived"
    assert peak["read_bytes_per_cycle"] == 8 and peak["write_bytes_per_cycle"] == 8
    reader = peak["clients"]["reader"]
    assert reader["direction"] == MP.READ and peak["clients"]["writer"]["direction"] == MP.WRITE
    assert reader["in_flight_bytes"] == 128 and reader["binding_latency_cycles"] == 16
    assert reader["sustained_bytes_per_cycle"] == 8.0
    assert "memory_round_trip_latency_cycles" in peak["fit_from_measurement"]


def test_a_fitted_round_trip_above_the_binding_latency_caps_the_sustained_rate() -> None:
    peak = MP.movement_peak({"facts": _BODY}, round_trip_latency_cycles=32)
    assert peak["read_bytes_per_cycle"] == pytest.approx(4.0)
    assert MP.movement_peak({"facts": _BODY}, round_trip_latency_cycles=8)["read_bytes_per_cycle"] == 8.0
    with pytest.raises(ValueError):
        MP.movement_peak({"facts": _BODY}, round_trip_latency_cycles=0)


def test_missing_ports_stay_unknown_with_a_reason() -> None:
    body = copy.deepcopy(_BODY)
    del body["timing_structure"]
    peak = MP.movement_peak(body)
    assert peak["status"] == "unknown" and peak["total_bytes_per_cycle"] is None
    assert "memory_clients" in peak["unresolved"]
    body = copy.deepcopy(_BODY)
    del body["timing_structure"]["memory_clients"]["clients"]["writer"]
    assert MP.movement_peak(body)["status"] == "unknown"


def test_the_derived_bound_and_capsule_roofline_read_the_ports() -> None:
    machine = _machine()
    assert machine.dram_bytes_per_cycle == 16.0
    assert "timing_structure.memory_clients" in machine.provenance["dram_bytes_per_cycle"]
    bounds = CR.machine_bounds("synthetic", facts={"facts": _BODY})
    assert bounds["read_bytes_per_cycle"] == 8 and bounds["write_bytes_per_cycle"] == 8
    assert "movement" not in bounds["unresolved"]


def test_padded_and_useful_arithmetic_are_both_reported() -> None:
    machine = _machine()
    doc = SR.staged_roofline([SR.Contraction(9, 8, 8)], machine, MP.movement_peak({"facts": _BODY}), result_bytes=1)
    compute = doc["compute"]
    assert compute["useful_macs"] == 9 * 8 * 8 and compute["padded_macs"] == 16 * 8 * 8
    assert compute["padding_efficiency"] == pytest.approx(9 / 16)
    assert doc["attainable_useful_macs_per_cycle"] == pytest.approx(compute["useful_macs"] / doc["floor_cycles"])


def test_requested_unique_and_physical_bytes_are_separate_and_tails_cost_whole_beats() -> None:
    machine = _machine()
    ports = MP.movement_peak({"facts": _BODY})
    aligned = SR.staged_roofline([SR.Contraction(8, 8, 8)], machine, ports, result_bytes=1)["movement"]
    assert aligned["unique_bytes"] == aligned["requested_bytes"] == aligned["physical_bytes"]
    tail = SR.staged_roofline([SR.Contraction(8, 9, 8)], machine, ports, result_bytes=1)["movement"]
    assert tail["requested_bytes"]["read"] == 8 * 9 + 9 * 8
    # Each A row is one 8-byte segment plus a one-byte tail, which costs a whole beat.
    assert tail["physical_bytes"]["read"] > tail["requested_bytes"]["read"]


def test_serial_stages_are_summed_and_merlin_host_compute_is_excluded() -> None:
    ports = MP.movement_peak({"facts": _BODY})
    overlapped = SR.staged_roofline([SR.Contraction(64, 64, 64)], _machine(), ports, result_bytes=1)
    assert overlapped["composition"]["overlap_admissible"] is True
    terms = overlapped["compute"]["padded_cycles"], overlapped["movement"]["cycles"]
    assert overlapped["floor_cycles"] == pytest.approx(max(terms))
    assert overlapped["status"] == "partial" and "accelerator_host_stage" in overlapped["unresolved"]

    single = copy.deepcopy(_BODY)
    single["memories"][1]["banks"] = 1
    single["memories"][1]["depth"] = 64
    serial = SR.staged_roofline(
        [SR.Contraction(64, 64, 64)],
        _machine(single),
        ports,
        result_bytes=1,
        accelerator_host_cycles=100,
        merlin_host_operations={"linalg.softmax": 1},
    )
    assert serial["composition"]["overlap_admissible"] is False
    terms = serial["compute"]["padded_cycles"], serial["movement"]["cycles"]
    assert serial["floor_cycles"] == pytest.approx(sum(terms) + 100)
    assert serial["host"]["merlin_supplied_operations"] == {"linalg.softmax": 1}
    assert serial["status"] == "derived"


def test_a_measurement_below_the_floor_refutes_it() -> None:
    doc = SR.staged_roofline(
        [SR.Contraction(64, 64, 64)], _machine(), MP.movement_peak({"facts": _BODY}), result_bytes=1
    )
    assert SR.position(doc, int(doc["floor_cycles"]) - 1)["status"] == "refuted"
    assert SR.position(doc, doc["floor_cycles"] * 2)["attainment"] == pytest.approx(0.5)

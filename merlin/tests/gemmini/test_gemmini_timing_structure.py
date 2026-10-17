"""Gemmini timing structure derived from the FireSimGemminiRocketConfig RTL vs. the Chisel defaults.

The derived values come from the elaborated HW (facts-v6, trimmed into a fixture). The Chisel
``GemminiRocketConfig`` defaults below are a cross-check ONLY: they are never an input to any
extractor. Where the RTL legitimately carries more structure than one Chisel parameter names, the
difference is asserted together with its explanation instead of being smoothed over.

Set ``MERLIN_GEMMINI_RTL_FACTS`` to a full facts artifact to run the same checks against it.
"""

import json
import os
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

FIXTURE = merlin_dir() / "tests" / "data" / "rtl_facts" / "gemmini_firesim_timing_structure.json"

# GemminiRocketConfig defaults (generators/gemmini Configs.scala defaultConfig): validation only.
CHISEL = {
    "reservation_station_entries_ld": 8,
    "reservation_station_entries_st": 4,
    "reservation_station_entries_ex": 16,
    "ld_queue_length": 8,
    "st_queue_length": 2,
    "ex_queue_length": 8,
    "max_in_flight_mem_reqs": 16,
    "dma_maxbytes": 64,
    "dma_buswidth": 128,
    "dataflow": "both",
    "mesh": (16, 16),
    "tile": (1, 1),
}


def _sources():
    sources = [("fixture", json.loads(FIXTURE.read_text()))]
    selected = os.environ.get("MERLIN_GEMMINI_RTL_FACTS")
    if selected:
        facts = json.loads(Path(selected).read_text())["facts"]
        sources.append(("selected", {**facts, "timing_structure": facts["timing_structure"]}))
    return sources


@pytest.fixture(params=_sources(), ids=lambda source: source[0])
def derived(request):
    return request.param[1]


def test_dispatch_entries_match_reservation_station_defaults(derived):
    classes = derived["timing_structure"]["command_classes"]["classes"]
    for name, key in (("load", "ld"), ("store", "st"), ("execute", "ex")):
        assert classes[name]["entries"]["status"] == "derived"
        assert classes[name]["entries"]["value"] == CHISEL[f"reservation_station_entries_{key}"]


def test_command_queues_match_queue_lengths_with_the_unroller_stage_explained(derived):
    classes = derived["timing_structure"]["command_classes"]["classes"]
    load, store, execute = (classes[name]["command_queue"]["chain"] for name in ("load", "store", "execute"))
    assert [hop["entries"] for hop in load] == [CHISEL["ld_queue_length"]]
    assert [hop["entries"] for hop in store] == [CHISEL["st_queue_length"]]
    # ex_queue_length sizes ExecuteController's cmd_q. The intact command first crosses the
    # TransposePreloadUnroller's own 2-entry queue (a fixed MultiHeadedQueue, not parameterized by
    # ex_queue_length), so the RTL buffers 2 + 8 issued execute commands.
    assert [(hop["instance"].rsplit("/", 1)[-1], hop["entries"]) for hop in execute] == [
        ("q", 2),
        ("cmd_q", CHISEL["ex_queue_length"]),
    ]
    assert classes["execute"]["command_queue"]["entries_total"] == 2 + CHISEL["ex_queue_length"]


def test_memory_clients_match_in_flight_maxbytes_and_bus_width(derived):
    clients = derived["timing_structure"]["memory_clients"]["clients"]
    for name in ("dma_read", "dma_write"):
        client = clients[name]
        assert client["in_flight"]["value"] == CHISEL["max_in_flight_mem_reqs"]
        assert client["id_space"] == CHISEL["max_in_flight_mem_reqs"]
        assert client["max_request_bytes"] == CHISEL["dma_maxbytes"]
        assert client["request_data_bits"] == CHISEL["dma_buswidth"]
        assert client["max_request_beats"] == CHISEL["dma_maxbytes"] * 8 // CHISEL["dma_buswidth"]


def test_dataflow_and_array_geometry_match(derived):
    structure = derived["timing_structure"]
    assert structure["dataflow"]["support"] == CHISEL["dataflow"]
    assert set(structure["dataflow"]["modes"].values()) == {"weight_stationary", "output_stationary"}
    rows, cols = CHISEL["mesh"]
    tile_rows, tile_cols = CHISEL["tile"]
    assert structure["dataflow"]["cell_occurrences"] == rows * cols * tile_rows * tile_cols
    mesh = structure["array_paths"]["mesh_activation_to_partial_sum_ws"]
    assert (mesh["sources"], mesh["sinks"]) == (rows * tile_rows, cols * tile_cols)
    # One register per tile hop in each direction plus the registered row input: an activation
    # entering row i reaches column j's partial-sum output after (rows + 1) - i + j stages.
    assert mesh["affine"] == {"per_source_index": -1, "per_sink_index": 1, "offset": rows + 1}


def test_fixed_latencies_are_classified_not_guessed(derived):
    latencies = derived["timing_structure"]["fixed_latencies"]
    assert (latencies["accumulator_adder"]["status"], latencies["accumulator_adder"]["min_stages"]) == ("fixed", 1)
    # SRAM read latency 1, then a flow-through response queue: 1 when it is empty, more when it holds.
    assert latencies["scratchpad_bank_read"]["min_stages"] == 1
    assert latencies["scratchpad_bank_read"]["status"] == "variable"
    assert all(row["status"] != "fixed" or row["max_stages"] == row["min_stages"] for row in latencies.values())


def test_queue_module_names_agree_with_derived_depths(derived):
    # Chisel names a Queue(gen, entries) module Queue<entries>_<type>: an independent label, read
    # here only to cross-check the structurally derived depth of every such buffer.
    for row in derived.get("buffers") or derived["timing_structure"]["buffers"]:
        name = row["module"]
        if name.startswith("Queue") and name[5:].split("_", 1)[0].isdigit():
            assert row["locations"] == [int(name[5:].split("_", 1)[0])], name


def test_simulation_configuration_is_structurally_the_same_accelerator(derived):
    correspondence = derived["simulation_correspondence"]
    assert correspondence["status"] in {"identical", "structurally_equivalent"}
    assert correspondence["configs"]["target"]["config"] == "FireSimGemminiRocketConfig"
    assert correspondence["configs"]["simulation"]["config"] == "GemminiGsimFireSimAlignedConfig"
    assert correspondence["firrtl_module_bodies"]["differ"] == []
    assert not correspondence["hw_file_level_operations"]["anchored_in_unit"]

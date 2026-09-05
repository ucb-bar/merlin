from pathlib import Path

from merlin.perf.bus_beat_probe import (
    assess_compiled_simulator,
    derive_counter_beat_monitors,
    measure_physical_beat_trace,
)


_HW = """
hw.module @Leaf(in %ready : i1, in %bus_valid : i1, in %bus_bits_size : i4,
                in %bus_bits_data : i128, in %reset_counter : i1, out count : i32) {
  %one = hw.constant 1 : i16
  %zero = hw.constant 0 : i32
  %event = comb.and bin %ready, %bus_valid : i1
  %shift = comb.shl bin %one, %bus_bits_size : i16
  %add = comb.add bin %count, %shift : i32
  %hold = comb.mux bin %event, %add, %count : i32
  %next = comb.mux bin %reset_counter, %zero, %hold : i32
  %count = seq.firreg %next clock %clock : i32
  hw.output %count : i32
}
"""


def _candidate(extent: str = "%bus_bits_size") -> dict:
    return {
        "status": "structurally_proved",
        "counter_field": "opaque-counter",
        "accumulator_proof": {
            "status": "proved",
            "leaf_module": "Leaf",
            "update_predicate_ssa": "%event",
            "encoded_extent_ssa": extent,
        },
    }


def test_circt_derives_width_and_handshake_but_never_physical_semantics() -> None:
    proof = derive_counter_beat_monitors(
        _HW, {"artifact_sha256": "a" * 64, "candidates": [_candidate()]}, source="core.hw.mlir")

    assert proof["status"] == "structural_only"
    assert proof["physical_byte_facts"] == []
    monitor = proof["monitors"][0]
    assert monitor["status"] == "monitor_derivable"
    assert monitor["valid_port"] == "%bus_valid"
    assert monitor["ready_port"] == "%ready"
    assert monitor["data_port"] == "%bus_bits_data"
    assert monitor["data_width_bits"] == 128
    assert monitor["beat_width_bytes"] == 16
    assert monitor["mechanical_formula"] == "accepted_event_count * 16"
    assert monitor["physical_semantics"] == "unknown"


def test_missing_data_sibling_refuses_instead_of_using_encoded_extent_as_bytes() -> None:
    hw = _HW.replace("in %bus_bits_data : i128, ", "")
    proof = derive_counter_beat_monitors(
        hw, {"artifact_sha256": "b" * 64, "candidates": [_candidate()]})

    assert proof["status"] == "unknown"
    assert proof["monitors"][0]["status"] == "unknown"
    assert proof["physical_byte_facts"] == []


def test_non_byte_aligned_data_width_refuses() -> None:
    hw = _HW.replace("i128, in %reset_counter", "i127, in %reset_counter")
    proof = derive_counter_beat_monitors(
        hw, {"artifact_sha256": "c" * 64, "candidates": [_candidate()]})
    assert proof["monitors"][0]["status"] == "unknown"


def test_prebuilt_model_without_trace_api_or_window_is_machine_readable_unknown(tmp_path: Path) -> None:
    simulator = tmp_path / "simulator"
    simulator.write_bytes(b"ELF model")
    public = tmp_path / "VModel.h"
    public.write_text("class VModel {};\n", encoding="utf-8")
    metadata = tmp_path / "VModel_classes.mk"
    metadata.write_text("VM_TRACE = 0\nVM_TRACE_VCD = 0\n", encoding="utf-8")

    report = assess_compiled_simulator(
        simulator=simulator, public_header=public, build_metadata=metadata,
        required_ports=("bus_valid", "ready", "bus_bits_data"), exact_window_marker=None)

    assert report["status"] == "unknown"
    assert report["trace_compiled"] is False
    assert report["public_signal_api"] is False
    assert report["measurement"] is None
    assert any("VM_TRACE disabled" in problem for problem in report["problems"])
    assert any("exact measurement-window" in problem for problem in report["problems"])


def test_trace_capability_still_refuses_without_exact_window_marker(tmp_path: Path) -> None:
    simulator = tmp_path / "simulator"
    simulator.write_bytes(b"ELF model")
    public = tmp_path / "VModel.h"
    public.write_text("bus_valid ready bus_bits_data\n", encoding="utf-8")
    metadata = tmp_path / "VModel_classes.mk"
    metadata.write_text("VM_TRACE = 1\n", encoding="utf-8")

    report = assess_compiled_simulator(
        simulator=simulator, public_header=public, build_metadata=metadata,
        required_ports=("bus_valid", "ready", "bus_bits_data"), exact_window_marker=None)
    assert report["trace_compiled"] is True
    assert report["public_signal_api"] is True
    assert report["status"] == "unknown"
    assert report["measurement"] is None


def test_complete_content_bound_trace_counts_accepted_beats_at_circt_width() -> None:
    write_hw = (_HW.replace("@Leaf", "@WriteLeaf")
                .replace("%ready", "%write_ready")
                .replace("%bus_", "%write_bus_"))
    proof = derive_counter_beat_monitors(
        _HW + write_hw,
        {"artifact_sha256": "d" * 64, "candidates": [
            _candidate(),
            {
                **_candidate("%write_bus_bits_size"),
                "counter_field": "another-opaque-counter",
                "accumulator_proof": {
                    **_candidate()["accumulator_proof"],
                    "leaf_module": "WriteLeaf",
                    "update_predicate_ssa": "%event",
                    "encoded_extent_ssa": "%write_bus_bits_size",
                },
            },
        ]})
    proof_sha = proof["artifact_sha256"]
    circt_sha = proof["circt_hw"]["sha256"]
    bindings = [
        {
            "fact_kind": "bus_payload_binding", "module": "Leaf",
            "valid_port": "%bus_valid", "ready_port": "%ready", "data_port": "%bus_bits_data",
            "direction": "read",
            "derived_from_rtl": True, "circt_hw_sha256": circt_sha,
            "monitor_proof_sha256": proof_sha, "evidence_sha256": "e" * 64,
            "exhaustive_for_direction": True,
        },
        {
            "fact_kind": "bus_payload_binding", "module": "WriteLeaf",
            "valid_port": "%write_bus_valid", "ready_port": "%write_ready",
            "data_port": "%write_bus_bits_data", "direction": "write",
            "derived_from_rtl": True, "circt_hw_sha256": circt_sha,
            "monitor_proof_sha256": proof_sha, "evidence_sha256": "f" * 64,
            "exhaustive_for_direction": True,
        },
    ]
    trace = {
        "inputs": {
            "monitor_proof_sha256": proof_sha, "circt_hw_sha256": circt_sha,
            "simulator_binary_sha256": "1" * 64,
        },
        "window": {"start_cycle": 8, "end_cycle": 10,
                   "marker_evidence_sha256": "2" * 64},
        "samples": [
            {"cycle": 8, "signals": {
                "%ready": 1, "%bus_valid": 1, "%bus_bits_data": 0,
                "%write_ready": 1, "%write_bus_valid": 0, "%write_bus_bits_data": 0,
            }},
            {"cycle": 9, "signals": {
                "%ready": 0, "%bus_valid": 1, "%bus_bits_data": 0,
                "%write_ready": 1, "%write_bus_valid": 1, "%write_bus_bits_data": 0,
            }},
        ],
    }

    result = measure_physical_beat_trace(trace, proof, semantic_bindings=bindings)
    assert result["status"] == "exact"
    assert result["accepted_beats"] == {"read": 1, "write": 1}
    assert result["physical_bytes"] == {"read": 16, "write": 16, "total": 32}


def test_trace_without_exhaustive_semantics_refuses_even_if_signals_are_present() -> None:
    proof = derive_counter_beat_monitors(
        _HW, {"artifact_sha256": "a" * 64, "candidates": [_candidate()]})
    result = measure_physical_beat_trace({}, proof, semantic_bindings=[])
    assert result["status"] == "unknown"
    assert result["physical_bytes"] is None
    assert any("both physical payload directions" in problem for problem in result["problems"])

"""The target adapter binds exact inputs but does not promote names into semantics."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

from merlin.runtime.backends.base import get_backend


def _module():
    backend = get_backend("gemmini")
    return importlib.import_module(f"{backend.__name__}.counter_byte_bindings")


def test_probe_content_addresses_inputs_and_emits_no_unproved_fact(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    circt = tmp_path / "core.hw.mlir"
    header = tmp_path / "counter.h"
    circt.write_text("synthetic circt", encoding="utf-8")
    header.write_text("synthetic header", encoding="utf-8")
    monkeypatch.setattr(module, "_input_artifacts", lambda: (circt, header))

    def extract(hw_text, header_text, **identities):
        assert hw_text == "synthetic circt" and header_text == "synthetic header"
        assert identities["top_module"] == "Gemmini"
        return {
            "schema": "merlin.counter-byte-binding-probe.v1", "status": "unknown",
            "counter_facts": [], "structurally_proved_candidates": 1,
            "candidates": [{"binding_status": "unknown"}],
            "inputs": {
                "circt_core_hw": {"source": identities["source"],
                                  "sha256": hashlib.sha256(hw_text.encode()).hexdigest()},
                "counter_header": {"source": identities["header_source"],
                                   "sha256": hashlib.sha256(header_text.encode()).hexdigest()},
            },
        }

    monkeypatch.setattr(module, "extract_external_additive_counters", extract)

    artifact = module.probe_counter_byte_bindings()

    assert artifact["status"] == "unknown"
    assert artifact["counter_facts"] == []
    assert artifact["structurally_proved_candidates"] == 1
    assert artifact["candidates"][0]["binding_status"] == "unknown"
    assert artifact["inputs"]["circt_core_hw"]["sha256"]
    assert artifact["artifact_sha256"] == module._canonical_sha256({
        key: value for key, value in artifact.items() if key != "artifact_sha256"})


def _structural() -> dict:
    return {
        "schema": "old", "artifact_sha256": "f" * 64, "status": "unknown",
        "counter_facts": [], "why": "semantic proof absent",
        "inputs": {
            "circt_core_hw": {"source": "core", "sha256": "a" * 64},
            "counter_header": {"source": "header", "sha256": "b" * 64},
        },
        "candidates": [
            {"counter_field": "OPAQUE_ONE", "status": "structurally_proved"},
            {"counter_field": "OPAQUE_TWO", "status": "structurally_proved"},
        ],
    }


def _point(direction: str, payload: int, one: int, two: int) -> dict:
    inputs = {
        "circt_core_hw_sha256": "a" * 64,
        "counter_header_sha256": "b" * 64,
        "rtl_facts_sha256": "c" * 64,
        "simulator_binary_sha256": "d" * 64,
    }
    emitter = {
        "status": "accepted", "direction": direction,
        "requested_payload_bytes": payload,
        "emitted_mlir_sha256": "1" * 64, "llvm_ir_sha256": "2" * 64,
        "object_sha256": "3" * 64, "object_kernel_disassembly_sha256": "4" * 64,
        "header_custom_instruction_count": 2, "object_custom_instruction_count": 2,
    }
    return {
        "direction": direction, "requested_payload_bytes": payload,
        "input_bindings": inputs, "console_sha256": "e" * 64,
        "result": {
            "status": "measured", "direction": direction,
            "requested_payload_bytes": payload, "correct": True, "cycles": 9,
            "oracle": {"derived_from_rtl": True}, "emitter": emitter,
            "elf_sha256": "5" * 64,
            "raw_counter_readings": {
                "counter_header_sha256": "b" * 64,
                "readings": {"OPAQUE_ONE": one, "OPAQUE_TWO": two},
            },
        },
    }


def _campaign(*, read_multiplier: int = 1, corrupt_copy: bool = False) -> dict:
    points = []
    for payload in (8, 16, 24, 32):
        points.append(_point("read", payload, payload * read_multiplier, 0))
        points.append(_point("write", payload, 0, payload // 2))
        points.append(_point("copy", payload, payload + int(corrupt_copy), payload // 2))
    receipt = {"path": "generated-header", "sha256": "6" * 64, "macro": "LIMIT"}
    module = _module()
    return {
        "inputs": {
            "circt_core_hw_sha256": "a" * 64,
            "counter_header_sha256": "b" * 64,
            "rtl_facts_sha256": "c" * 64,
            "rtl_facts_core_hw_sha256": "a" * 64,
            "simulator_binary_sha256": "d" * 64,
            "coordinate_derivation": {
                "method": "target_header_command_limit_multiples",
                "required_points": 4, "sizes_bytes": [8, 16, 24, 32],
                "capability_receipt": receipt,
                "capability_receipt_sha256": module._canonical_sha256(receipt),
            },
        },
        "points": points,
    }


def test_complete_differential_campaign_promotes_without_using_counter_names() -> None:
    module = _module()
    artifact = module.evaluate_differential_evidence(_structural(), _campaign())

    assert artifact["status"] == "proved"
    assert artifact["differential_evidence"]["direction_assignment"] == {
        "read": "OPAQUE_ONE", "write": "OPAQUE_TWO"}
    assert [(fact["direction"], fact["unit_bytes"]) for fact in artifact["counter_facts"]] == [
        ("read", 1), ("write", 2)]
    assert all(fact["artifact_sha256"] == "a" * 64 for fact in artifact["counter_facts"])


def test_real_failure_shape_falsifies_byte_unit_instead_of_becoming_scale_factor() -> None:
    module = _module()
    artifact = module.evaluate_differential_evidence(
        _structural(), _campaign(read_multiplier=4))

    assert artifact["status"] == "unknown" and artifact["counter_facts"] == []
    assert "payload/raw=1/4" in artifact["why"]
    assert artifact["differential_evidence"]["byte_unit_evidence"]["read"]["status"] == "falsified"
    first = artifact["differential_evidence"]["scale_observations"]["read"][0]
    assert first["payload_bytes"] == 8 and first["raw_count"] == 32


def test_one_read_witness_keeps_direction_separate_from_falsified_unit() -> None:
    module = _module()
    campaign = _campaign(read_multiplier=4)
    campaign["points"] = [campaign["points"][0]]
    artifact = module.evaluate_differential_evidence(_structural(), campaign)

    evidence = artifact["differential_evidence"]
    assert evidence["direction_evidence"]["read"] == {
        "status": "witnessed", "counter_field": "OPAQUE_ONE",
        "isolated_points": 1, "promotion_eligible": False}
    assert evidence["byte_unit_evidence"]["read"]["status"] == "falsified"
    assert artifact["counter_facts"] == []


def test_one_substituted_circt_binding_invalidates_whole_campaign() -> None:
    module = _module()
    campaign = _campaign()
    campaign["points"][0]["input_bindings"]["circt_core_hw_sha256"] = "9" * 64
    artifact = module.evaluate_differential_evidence(_structural(), campaign)
    assert artifact["status"] == "unknown" and artifact["counter_facts"] == []
    assert "exact campaign input" in artifact["why"]


def test_non_target_counter_must_be_zero_and_reading_set_exhaustive() -> None:
    module = _module()
    campaign = _campaign()
    campaign["points"][0]["result"]["raw_counter_readings"]["readings"]["OPAQUE_TWO"] = 1
    artifact = module.evaluate_differential_evidence(_structural(), campaign)
    assert artifact["status"] == "unknown" and artifact["counter_facts"] == []
    assert "activate exactly one" in artifact["why"]


def test_copy_must_cross_check_both_directional_scales() -> None:
    module = _module()
    artifact = module.evaluate_differential_evidence(
        _structural(), _campaign(corrupt_copy=True))
    assert artifact["status"] == "unknown" and artifact["counter_facts"] == []
    assert "copy point does not cross-check" in artifact["why"]


def test_four_points_and_exact_compiled_program_receipts_are_mandatory() -> None:
    module = _module()
    campaign = _campaign()
    campaign["points"] = [point for point in campaign["points"]
                          if point["requested_payload_bytes"] != 32]
    campaign["points"][0]["result"]["emitter"].pop("object_kernel_disassembly_sha256")
    artifact = module.evaluate_differential_evidence(_structural(), campaign)
    assert artifact["status"] == "unknown" and artifact["counter_facts"] == []
    assert "at least four" in artifact["why"] and "disassembly hash" in artifact["why"]

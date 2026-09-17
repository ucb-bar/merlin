"""Command-buffer diagnostics expose representations while refusing invented occupancy."""
import hashlib
import json

import pytest

from merlin.perf.command_buffer_diagnostics import representation_activity


def _digest(value) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _verified_transition_evidence(command_buffer):
    row = {
        "id": "layout-copy", "status": "verified", "materialized": True,
        "execution_multiplicity_verified": True,
        "load_payload_bytes": 24, "store_payload_bytes": 24,
    }
    evidence = {
        "schema": "physical_transition_evidence_v1", "status": "verified",
        "source_sha256": "a" * 64, "lowered_sha256": "b" * 64,
        "command_buffer_sha256": _digest(command_buffer), "transitions": [row],
        "encoding_activity": {
            "status": "verified", "executed_transition_count": 1,
            "materialized_transition_count": 1, "physical_read_bytes": 24,
            "physical_write_bytes": 24, "physical_bytes": 48,
            "basis": "fixture emitted-address proof",
        },
    }
    evidence["receipt_sha256"] = _digest(evidence)
    return evidence


def test_representation_directives_and_unknown_timeline_are_both_explicit() -> None:
    audit = representation_activity({
        "tensors": {
            "weight": {"shape": [8, 8], "dtype": "i8", "role": "weight"},
            "output": {"shape": [8, 8], "dtype": "i32", "role": "output",
                       "physical": {"unstack": 2}},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "weight", "dst": "resident"},
             "attributes": {"layout": "packed"}},
            {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "output"},
             "attributes": {"output_dtype": "i32", "epilogue": ["relu"]}},
        ],
    })

    assert audit["command_counts"] == {"COMMIT": 1, "RES_PACK": 1}
    assert audit["representation_directive_count"] == 2
    assert audit["tensors"]["output"]["physical_status"] == "declared"
    assert audit["tensors"]["weight"]["physical_status"] == "UNKNOWN"
    assert audit["occupancy"]["status"] == "UNKNOWN"
    assert audit["emitted_encoding_transitions"]["count"] is None


def test_declined_whole_model_carries_placement_without_becoming_zero_work() -> None:
    audit = representation_activity({
        "tensors": {}, "commands": [],
        "declined": {"op": "host_lane", "reason": "cannot roll the full model"},
        "params": {"lane_placement": [
            {"region": "a", "family": "contraction", "lane": "array"},
            {"region": "b", "family": "elementwise", "lane": "vector"},
            {"region": "c", "family": "contraction", "lane": "array"},
        ]},
    })

    assert audit["lowering"]["status"] == "declined"
    assert audit["placement"]["lane_counts"] == {"array": 2, "vector": 1}
    assert audit["placement"]["adjacent_lane_transitions"] == 2


def test_verified_emitted_transition_exposes_exact_materialization_and_bytes() -> None:
    command_buffer = {
        "tensors": {"value": {"shape": [6], "dtype": "i32"}},
        "commands": [],
        "params": {"global_program_plan": {"physical_transitions": [{"id": "layout-copy"}]}},
    }
    evidence = _verified_transition_evidence(command_buffer)

    activity = representation_activity(
        command_buffer, physical_transition_evidence=evidence)["emitted_encoding_transitions"]

    assert activity["status"] == "verified"
    assert activity["count"] == activity["materialized_count"] == 1
    assert activity["physical_read_bytes"] == activity["physical_write_bytes"] == 24
    assert activity["physical_bytes"] == 48
    assert activity["evidence_sha256"] == evidence["receipt_sha256"]
    assert activity["calibration_source_schema"] == "phase2_analytical_feature_calibration_v1"


@pytest.mark.parametrize("mutation", [
    "declaration_only", "changed_command", "changed_receipt", "malformed_bytes", "partial",
])
def test_unverified_or_changed_transition_evidence_cannot_become_activity(mutation) -> None:
    command_buffer = {"tensors": {}, "commands": [], "params": {}}
    evidence = _verified_transition_evidence(command_buffer)
    if mutation == "declaration_only":
        evidence = None
    elif mutation == "changed_command":
        command_buffer["params"]["changed_after_verification"] = True
    elif mutation == "changed_receipt":
        evidence["encoding_activity"]["physical_bytes"] += 1
    elif mutation == "malformed_bytes":
        evidence["transitions"][0]["load_payload_bytes"] = "24"
        evidence["receipt_sha256"] = _digest({key: value for key, value in evidence.items()
                                               if key != "receipt_sha256"})
    else:
        evidence["status"] = "UNKNOWN"
        evidence["receipt_sha256"] = _digest({key: value for key, value in evidence.items()
                                               if key != "receipt_sha256"})

    activity = representation_activity(
        command_buffer, physical_transition_evidence=evidence)["emitted_encoding_transitions"]

    assert activity["status"] == "UNKNOWN"
    assert activity["count"] is None
    assert activity["materialized_count"] is None
    assert activity["physical_bytes"] is None

"""Native suite admission separates explicit host work from device dtype support."""

from copy import deepcopy

import pytest

from merlin.targetgen import capsule_runner as runner
from merlin.targetgen import eligibility
from merlin.targetgen.compute_units import SemanticCapability
from merlin.targetgen.owned_host_leaf_execution import declared_host_only, selected_host_leaf


def host_capsule(dtype="i32", axis="host_lane", count=1):
    return {
        "name": "neutral_host_program",
        "kind": "model_slice",
        "operation": {"op": "model"},
        "inputs": [{"name": "input", "role": "input", "dtype": dtype, "shape": [4, 8]}],
        "semantic": {
            "semantic_family": "elementwise_map",
            "generalization_axis": axis,
            "eligible": False,
            "must_accelerate": False,
        },
        "expected": {"instruction_classes": [], "modes": {}},
        "required_oracle_tiers": ["L0", "L1", "L2"],
        "software_screen": {
            "status": "admitted",
            "constraints_status": "matched",
            "program_sha256": "1" * 64,
            "placements": {"host": 1},
            "decisions": [
                {
                    "operation": "source_scalar_body",
                    "mlir_operation": "linalg.generic",
                    "count": count,
                    "classification": "host_required",
                    "placement": "host",
                    "status": "admitted",
                }
            ],
        },
    }


@pytest.fixture(autouse=True)
def public_device_capability(monkeypatch):
    capability = {"elementwise_map": SemanticCapability("elementwise_map", dtypes=("int8",))}
    monkeypatch.setattr(eligibility, "capability_map_for_target", lambda target: capability)


@pytest.mark.parametrize(
    "dtype,axis,count", [("i32", "host_lane", 2), ("f32", "host_lane", 1), ("i32", "composition", 1)]
)
def test_explicit_host_work_is_graded_and_counted_without_claiming_device_support(dtype, axis, count):
    cap = host_capsule(dtype, axis, count)
    before = deepcopy(cap)
    assert declared_host_only(cap) and selected_host_leaf(cap)
    kept, withheld = runner._split_ineligible([cap], "neutral")
    assert kept == [cap] and withheld == []
    assert runner._gate_counts({"capsule": cap["name"], "status": "fail"}, [cap], "neutral")
    assert cap == before  # required numerical/native tiers and acceleration policy unchanged
    assert "status" not in cap and "pass" not in cap


@pytest.mark.parametrize(
    "damage",
    [
        "no_semantic",
        "no_screen",
        "no_eligible",
        "no_demand",
        "required_acceleration",
        "positive_eligibility",
        "no_digest",
        "bad_digest",
        "no_constraints",
        "unknown_constraints",
        "missing_decisions",
        "missing_placements",
        "count_mismatch",
        "no_classification",
        "unknown_classification",
        "mixed_lane",
        "refused_decision",
        "invalid_count",
        "missing_operation",
        "instruction_obligation",
        "required_lane",
        "malformed_decisions",
        "malformed_semantic",
    ],
)
def test_missing_or_conflicting_declarations_cannot_bypass_absent_device_dtype(damage):
    cap = host_capsule()
    screen = cap["software_screen"]
    row = screen["decisions"][0]
    if damage == "no_semantic":
        cap.pop("semantic")
    elif damage == "no_screen":
        cap.pop("software_screen")
    elif damage == "no_eligible":
        cap["semantic"].pop("eligible")
    elif damage == "no_demand":
        cap["semantic"].pop("must_accelerate")
    elif damage == "required_acceleration":
        cap["semantic"]["must_accelerate"] = True
    elif damage == "positive_eligibility":
        cap["semantic"]["eligible"] = True
    elif damage == "no_digest":
        screen.pop("program_sha256")
    elif damage == "bad_digest":
        screen["program_sha256"] = "unknown"
    elif damage == "no_constraints":
        screen.pop("constraints_status")
    elif damage == "unknown_constraints":
        screen["constraints_status"] = "unknown"
    elif damage == "missing_decisions":
        screen.pop("decisions")
    elif damage == "missing_placements":
        screen.pop("placements")
    elif damage == "count_mismatch":
        screen["placements"]["host"] = 2
    elif damage == "no_classification":
        row.pop("classification")
    elif damage == "unknown_classification":
        row["classification"] = "unclassified"
    elif damage == "mixed_lane":
        row["placement"] = "accelerator"
    elif damage == "refused_decision":
        row["status"] = "refused"
    elif damage == "invalid_count":
        row["count"] = True
    elif damage == "missing_operation":
        row.pop("mlir_operation")
    elif damage == "instruction_obligation":
        cap["expected"]["instruction_classes"] = ["compute"]
    elif damage == "required_lane":
        cap["lanes"] = {"require": ["on_mesh"]}
    elif damage == "malformed_decisions":
        screen["decisions"] = ["host"]
    elif damage == "malformed_semantic":
        cap["semantic"] = []
    assert not declared_host_only(cap)
    kept, withheld = runner._split_ineligible([cap], "neutral")
    assert not kept and len(withheld) == 1
    assert withheld[0]["status"] == "not_graded"
    assert not runner._gate_counts({"capsule": cap["name"]}, [cap], "neutral")


def test_existing_negative_lane_and_native_device_routes_remain_graded():
    negative = host_capsule()
    negative.pop("software_screen")
    negative["lanes"] = {"forbid": ["on_mesh"]}
    native = host_capsule("i8")
    native.pop("software_screen")
    native["semantic"].update(eligible=True, must_accelerate=True)
    native["name"] = "device"
    kept, withheld = runner._split_ineligible([negative, native], "neutral")
    assert kept == [negative, native] and not withheld
    assert all(runner._gate_counts({"capsule": cap["name"]}, kept, "neutral") for cap in kept)

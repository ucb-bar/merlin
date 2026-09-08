from __future__ import annotations

import hashlib
import json

from merlin.perf import phase2_feature_calibration as calibration


def _write(path, value):
    raw = (
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        if isinstance(value, dict)
        else str(value).encode()
    )
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _quantity(value, unit, scope):
    return {"value": value, "unit": unit, "scope": scope}


def _pair(tmp_path, *, ident, target_sha, feature_values, outcomes, controls="same-controls"):
    controls_ref = _write(
        tmp_path / f"{ident}.controls",
        {
            "schema": calibration.CONTROLS_SCHEMA,
            "status": "declared",
            "target_sha256": target_sha,
            "varied_feature_pointer": "/quantities/feature",
            "invariant_pointers": ["/environment", "/measurement_window"],
            "description": controls,
        },
    )
    arms = []
    for name, feature_value, measured in zip(("first", "second"), feature_values, outcomes, strict=True):
        source = _write(tmp_path / f"{ident}.{name}.source", f"{ident}-{name}-source")
        evidence = {
            "schema": calibration.OBSERVATION_SCHEMA,
            "status": "measured",
            "bindings": {
                "target_sha256": target_sha,
                "source_sha256": source["sha256"],
                "controls_sha256": controls_ref["sha256"],
            },
            "quantities": {
                "feature": _quantity(feature_value, "emitted_op", "emitted_feature"),
                **measured,
            },
        }
        arms.append({"source": source, "evidence": _write(tmp_path / f"{ident}.{name}.json", evidence)})
    return {"id": ident, "controls": controls_ref, "first": arms[0], "second": arms[1]}


def _base_request(target, *, kind, pairs):
    measurements = (
        [
            {
                "coefficient": "cycles_per_unit",
                "evidence_pointer": "/quantities/cycles",
                "unit": "cycle",
                "scope": "target_execution",
            }
        ]
        if kind == "compute"
        else [
            {
                "coefficient": "physical_bytes_per_unit",
                "evidence_pointer": "/quantities/physical_bytes",
                "unit": "byte",
                "scope": "physical_target_interface",
            },
            {
                "coefficient": "commands_per_unit",
                "evidence_pointer": "/quantities/commands",
                "unit": "command",
                "scope": "target_execution",
            },
        ]
    )
    return {
        "schema": calibration.REQUEST_SCHEMA,
        "target_descriptor": target,
        "feature": {
            "id": "issued-operation",
            "kind": kind,
            "pointer": "/target_activity/issued/operation_count",
            "evidence_pointer": "/quantities/feature",
            "unit": "emitted_op",
            "resource": "execution-resource",
            "effects": ["scheduling"],
        },
        "measurements": measurements,
        "controlled_pairs": pairs,
    }


def test_compute_preparer_and_validator_recompute_content_bound_pair(tmp_path):
    target = _write(tmp_path / "target.json", "target descriptor")
    pair = _pair(
        tmp_path,
        ident="compute",
        target_sha=target["sha256"],
        feature_values=(2, 6),
        outcomes=(
            {"cycles": _quantity(20, "cycle", "target_execution")},
            {"cycles": _quantity(32, "cycle", "target_execution")},
        ),
    )

    prepared = calibration.prepare_feature_calibration(_base_request(target, kind="compute", pairs=[pair]))

    assert prepared["status"] == "ready"
    assert prepared["calibration"]["feature"]["cycles_per_unit"] == {"lo": 3.0, "hi": 3.0}
    assert prepared["calibration"]["derivation"]["n_fitted_parameters"] == 1
    assert prepared["calibration"]["derivation"]["n_distinct_points"] == 2
    assert prepared["target_execution_performed"] is False
    assert prepared["simulator_execution_performed"] is False

    receipt = _write(tmp_path / "compute-receipt.json", prepared["calibration"])
    validated = calibration.validate_feature_calibration(receipt["path"], expected_target_sha256=target["sha256"])

    assert validated["status"] == "ready"
    assert validated["feature"]["pointer"] == "/target_activity/issued/operation_count"
    assert validated["feature"]["unit"] == "emitted_op"
    assert {row["purpose"] for row in validated["evidence_files"]} == {
        "exact target descriptor",
        "paired controlled-variable contract",
        "controlled source",
        "controlled measured observation",
    }


def test_physical_movement_needs_four_paired_points_for_two_parameters(tmp_path):
    target = _write(tmp_path / "target.json", "target descriptor")
    pair = _pair(
        tmp_path,
        ident="movement-a",
        target_sha=target["sha256"],
        feature_values=(1, 3),
        outcomes=(
            {
                "physical_bytes": _quantity(8, "byte", "physical_target_interface"),
                "commands": _quantity(1, "command", "target_execution"),
            },
            {
                "physical_bytes": _quantity(40, "byte", "physical_target_interface"),
                "commands": _quantity(3, "command", "target_execution"),
            },
        ),
    )

    incomplete = calibration.prepare_feature_calibration(_base_request(target, kind="movement", pairs=[pair]))

    assert incomplete["status"] == "incomplete"
    assert incomplete["calibration"] is None
    assert "at least two distinct points per fitted parameter" in incomplete["missing"][0]["reason"]

    second = _pair(
        tmp_path,
        ident="movement-b",
        target_sha=target["sha256"],
        feature_values=(2, 5),
        outcomes=(
            {
                "physical_bytes": _quantity(24, "byte", "physical_target_interface"),
                "commands": _quantity(2, "command", "target_execution"),
            },
            {
                "physical_bytes": _quantity(72, "byte", "physical_target_interface"),
                "commands": _quantity(5, "command", "target_execution"),
            },
        ),
    )
    ready = calibration.prepare_feature_calibration(_base_request(target, kind="movement", pairs=[pair, second]))

    assert ready["status"] == "ready"
    assert ready["calibration"]["feature"]["physical_bytes_per_unit"] == 16.0
    assert ready["calibration"]["feature"]["commands_per_unit"] == 1
    assert ready["calibration"]["derivation"]["n_distinct_points"] == 4


def test_observation_binding_mismatch_is_refused(tmp_path):
    target = _write(tmp_path / "target.json", "target descriptor")
    pair = _pair(
        tmp_path,
        ident="compute",
        target_sha="0" * 64,
        feature_values=(1, 2),
        outcomes=(
            {"cycles": _quantity(2, "cycle", "target_execution")},
            {"cycles": _quantity(4, "cycle", "target_execution")},
        ),
    )

    result = calibration.prepare_feature_calibration(_base_request(target, kind="compute", pairs=[pair]))

    assert result["status"] == "refused"
    assert result["calibration"] is None
    assert any("target_sha256" in row["reason"] for row in result["refusals"])


def test_missing_controlled_observations_stays_explicitly_incomplete(tmp_path):
    target = _write(tmp_path / "target.json", "target descriptor")

    result = calibration.prepare_feature_calibration(_base_request(target, kind="compute", pairs=[]))

    assert result["status"] == "incomplete"
    assert result["calibration"] is None
    assert result["refusals"] == []
    assert result["full_model_execution_performed"] is False


def test_validator_refuses_a_coefficient_not_derived_from_bound_points(tmp_path):
    target = _write(tmp_path / "target.json", "target descriptor")
    pair = _pair(
        tmp_path,
        ident="compute",
        target_sha=target["sha256"],
        feature_values=(1, 2),
        outcomes=(
            {"cycles": _quantity(2, "cycle", "target_execution")},
            {"cycles": _quantity(5, "cycle", "target_execution")},
        ),
    )
    prepared = calibration.prepare_feature_calibration(_base_request(target, kind="compute", pairs=[pair]))
    receipt = prepared["calibration"]
    receipt["feature"]["cycles_per_unit"] = {"lo": 1.0, "hi": 1.0}

    result = calibration.validate_feature_calibration(receipt)

    assert result["status"] == "refused"
    assert result["feature"] is None
    assert any("does not equal" in row["reason"] for row in result["refusals"])

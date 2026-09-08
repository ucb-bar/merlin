from __future__ import annotations

import hashlib
import json

from merlin.perf import phase2_calibration_bundle as bundle
from merlin.perf.phase2_analytical_provider import build_fast_evaluator_installation


def _sha(data: bytes | str) -> str:
    raw = data.encode() if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def _write(path, data: bytes | str):
    raw = data.encode() if isinstance(data, str) else data
    path.write_bytes(raw)
    return {"path": str(path), "sha256": _sha(raw)}


def _derived_counter():
    body = {
        "schema": "phase2_trusted_counter_preparation_v1",
        "status": "derived",
        "composition": {"operator": "partial", "eta": 0.5},
    }
    body["receipt_sha256"] = _sha(json.dumps(body, sort_keys=True, separators=(",", ":")))
    return body


def _derived_movement():
    fit = {
        "schema": "merlin_movement_balance_v1",
        "status": "derived",
        "peak_bytes_per_cycle": 8.0,
        "base_latency_cycles": 2.0,
        "domain_bytes": [16, 256],
        "n_distinct_sizes": 4,
        "residual_cycles": [0.0, 0.0, 0.0, 0.0],
    }
    body = {"schema": "phase2_controlled_movement_preparation_v1",
            "status": "derived", "fit": fit}
    body["receipt_sha256"] = _sha(json.dumps(body, sort_keys=True, separators=(",", ":")))
    return body


def _stub_evidence(monkeypatch):
    monkeypatch.setattr(bundle, "_counter_evidence", lambda _adapter: (
        _derived_counter(), {"operator": "partial", "eta": 0.5}, [], []))
    monkeypatch.setattr(bundle, "_movement_evidence", lambda _adapter: (
        _derived_movement(), [], []))


def _adapter(target_ref, features=()):
    return {
        "schema": bundle.ADAPTER_SCHEMA,
        "target_descriptor": target_ref,
        "counter_harvest": {},
        "movement_series": {},
        "accelerator_compute_roles": ["execute"],
        "risk_score": 0.2,
        "feature_calibration_receipts": list(features),
    }


def _feature_receipt(tmp_path, target_sha, kind, feature):
    source = _write(tmp_path / f"{kind}.source", f"measured-{kind}")
    document = {
        "schema": bundle.FEATURE_SCHEMA,
        "status": "derived",
        "target_sha256": target_sha,
        "source_files": [source],
        "derivation": {"evidence_sha256s": [source["sha256"]]},
        "feature": feature,
    }
    if feature.get("cycles_per_unit") is not None:
        document["derivation"].update({"n_fitted_parameters": 1, "n_distinct_points": 2})
    path = tmp_path / f"{kind}.json"
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return _write(path, raw)


def test_trusted_counters_and_movement_do_not_invent_feature_coefficients(tmp_path, monkeypatch):
    _stub_evidence(monkeypatch)
    target = _write(tmp_path / "target.yaml", "target bytes")

    receipt = bundle.prepare_phase2_calibration(_adapter(target))

    assert receipt["status"] == "incomplete"
    assert receipt["calibration"] is None
    assert receipt["refusals"] == []
    assert {row["field"] for row in receipt["missing"]} == {
        "features.compute", "features.movement", "features.encoding",
    }
    assert receipt["target_execution_performed"] is False
    assert receipt["full_model_simulation_performed"] is False


def test_a_digest_mismatch_is_a_refusal_not_missing_evidence(tmp_path):
    target = _write(tmp_path / "target.yaml", "target bytes")
    target["sha256"] = _sha("different bytes")

    receipt = bundle.prepare_phase2_calibration(_adapter(target))

    assert receipt["status"] == "refused"
    assert receipt["calibration"] is None
    assert any(row["field"] == "target_descriptor" for row in receipt["refusals"])


def test_complete_content_addressed_receipts_build_a_provider_accepted_calibration(
        tmp_path, monkeypatch):
    _stub_evidence(monkeypatch)
    target = _write(tmp_path / "target.yaml", "target bytes")
    target_sha = target["sha256"]
    features = [
        _feature_receipt(tmp_path, target_sha, "compute", {
            "id": "compute", "pointer": "/issued/compute", "resource": "arithmetic",
            "kind": "compute", "cycles_per_unit": {"lo": 1.0, "hi": 1.2},
            "floor_cycles_per_unit": 0.5, "effects": ["tiling"],
        }),
        _feature_receipt(tmp_path, target_sha, "movement", {
            "id": "movement", "pointer": "/issued/movement", "resource": "transfer",
            "kind": "movement", "physical_bytes_per_unit": 16.0,
            "commands_per_unit": 1, "floor_cycles_per_unit": 1.0,
            "effects": ["movement"],
        }),
        _feature_receipt(tmp_path, target_sha, "encoding", {
            "id": "encoding", "pointer": "/encoding/count", "resource": "transfer",
            "kind": "encoding", "physical_bytes_per_unit": 16.0,
            "commands_per_unit": 1, "transitions_per_unit": 1,
            "floor_cycles_per_unit": 1.0, "effects": ["encoding"],
        }),
    ]

    receipt = bundle.prepare_phase2_calibration(_adapter(target, features))

    assert receipt["status"] == "ready"
    calibration = receipt["calibration"]
    assert calibration["schema"] == bundle.CALIBRATION_SCHEMA
    assert calibration["target_sha256"] == target_sha
    assert len(calibration["features"]) == 3
    assert all(feature["provenance_sha256"] in calibration["evidence_sha256s"]
               for feature in calibration["features"])

    members = tuple(_sha(f"member-{index}") for index in range(4))
    installation = build_fast_evaluator_installation(
        members, classification_member_sha256=members[0],
        corpus_sha256_by_member={member: _sha("corpus-" + member) for member in members},
        calibration=calibration,
        quality_observer=lambda **_kwargs: {}, quality_observer_sha256=_sha("observer"),
    )
    assert installation.provider is not None
    assert installation.provider_binding["target_sha256"] == target_sha


def test_cycle_feature_receipt_needs_two_points_per_fitted_parameter(tmp_path, monkeypatch):
    _stub_evidence(monkeypatch)
    target = _write(tmp_path / "target.yaml", "target bytes")
    feature = _feature_receipt(tmp_path, target["sha256"], "compute", {
        "id": "compute", "pointer": "/issued/compute", "resource": "arithmetic",
        "kind": "compute", "cycles_per_unit": {"lo": 1.0, "hi": 1.0},
    })
    path = tmp_path / "compute.json"
    document = json.loads(path.read_text())
    document["derivation"]["n_distinct_points"] = 1
    feature = _write(path, json.dumps(document, sort_keys=True, separators=(",", ":")))

    receipt = bundle.prepare_phase2_calibration(_adapter(target, [feature]))

    assert receipt["status"] == "refused"
    assert any("two points per fitted parameter" in row["reason"]
               for row in receipt["refusals"])

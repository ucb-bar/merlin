"""Production causal evidence uses paired full-model Merlin controls, not comparators."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from merlin.compare.freeze import sha256_paths
from merlin.compare.paper_full_model_ablation import (
    FullModelAblationError,
    validate_manifest,
)
from merlin.compare.paper_attribution import (
    CausalEvidenceError,
    _result_matches,
    freeze_causal_evidence,
)


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def _ref(path: Path, root: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_paths([path])}


def _binding() -> dict:
    return {
        "study_identity_sha256": "1" * 64, "study_label": "frozen-study",
        "target": "unit-test", "model": "model", "checkpoint": "checkpoint",
        "fidelity": "full", "precision": "fp32", "core_count": 1,
        "comparator": "executorch_xnnpack", "compiler_policy_sha256": "2" * 64,
        "compiler_source_sha256": "3" * 64, "runtime_sha256": "4" * 64,
        "capture_sha256": "5" * 64, "capture_session_identity_sha256": "6" * 64,
        "session_protocol_sha256": _canonical_sha({"kind": "continuous"}),
        "ours": {"backend": "merlin_frozen", "kind": "compiler", "runtime": "merlin",
                 "quantization": "none", "package_sha256": "7" * 64,
                 "source_sha256": "3" * 64},
        "comparator_backend": {
            "backend": "executorch_xnnpack", "kind": "external_runtime",
            "runtime": "executorch", "quantization": "none",
            "package_sha256": "8" * 64, "source_sha256": "9" * 64,
        },
    }


def _fixture(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    from merlin.compare import paper
    monkeypatch.setattr(paper, "validate_paper_result", lambda _result: None)
    root = tmp_path / "causal"
    root.mkdir()
    binding = _binding()
    binding_sha = _canonical_sha(binding)
    delta = root / "transform-delta.yaml"
    delta_document = {
        "schema_version": 1, "kind": "paper_compiler_transform_delta_v1",
        "status": "frozen", "control_policy_sha256": "a" * 64,
        "treatment_policy_sha256": "2" * 64,
        "changed_components": ["fusion_layout", "runtime_synchronization"],
        "unchanged_build_configuration_sha256": "b" * 64,
    }
    _write_yaml(delta, delta_document)
    build_receipts = {role: {"role": role} for role in ("control", "treatment")}
    contract = root / "pair-contract.yaml"
    contract_document = {
        "schema_version": 1, "kind": "paper_full_model_ablation_pair_contract_v1",
        "status": "frozen", "pair_id": "model-fp32-1c", "binding_sha256": binding_sha,
        "measurement_study_sha256": "c" * 64,
        "intervention": {
            "id": "frozen-policy-vs-disabled-transform", "scope": "compiler_full_model_transform",
            "isolated_change": "compiler_policy",
            "changed_components": ["fusion_layout", "runtime_synchronization"],
            "control_policy_sha256": "a" * 64, "treatment_policy_sha256": "2" * 64,
            "delta_manifest": _ref(delta, root),
        },
        "pairing": {
            "metric": "end_to_end_latency_ns", "direction": "lower_is_better",
            "primary_scope": "end_to_end", "sample_unit": "continuous_session",
            "per_run_samples": 3, "pair_count": 3,
            "schedule": [
                {"pair_index": 0, "order": "control_first"},
                {"pair_index": 1, "order": "treatment_first"},
                {"pair_index": 2, "order": "control_first"},
            ],
        },
        "environment": {
            "target": "unit-test", "require_same_board_identity": True,
            "require_same_vlen": True, "require_performance_governor": True,
            "require_current_equals_max": True, "maximum_thermal_delta_millic": 5000,
        },
        "arms": {
            "control": {
                "backend": "merlin_ablation_control", "runtime": "merlin",
                "compiler_source_sha256": "3" * 64, "runtime_sha256": "4" * 64,
                "package_sha256": "d" * 64, "policy_sha256": "a" * 64,
                "binary_sha256": "e" * 64, "build_configuration_sha256": "b" * 64,
                "build_receipt_sha256": _canonical_sha(build_receipts["control"]),
            },
            "treatment": {
                "backend": "merlin_frozen", "runtime": "merlin",
                "compiler_source_sha256": "3" * 64, "runtime_sha256": "4" * 64,
                "package_sha256": "7" * 64, "policy_sha256": "2" * 64,
                "binary_sha256": "f" * 64, "build_configuration_sha256": "b" * 64,
                "build_receipt_sha256": _canonical_sha(build_receipts["treatment"]),
            },
        },
    }
    _write_yaml(contract, contract_document)

    samples = {"control": [120, 110, 130], "treatment": [80, 90, 100]}
    starts = {
        (0, "control"): (100, 110), (0, "treatment"): (120, 130),
        (1, "treatment"): (200, 210), (1, "control"): (220, 230),
        (2, "control"): (300, 310), (2, "treatment"): (320, 330),
    }
    runs = []
    probe = json.dumps({
        "schema_version": 1, "kind": "merlin_board_probe_v1", "identity": "board-1",
        "vlen_bits": 256, "vlen_source": "csr", "governor": "performance",
        "current_khz": 1600000, "max_khz": 1600000, "max_thermal_millic": 50000,
    }, sort_keys=True)
    for pair_index, schedule in enumerate(contract_document["pairing"]["schedule"]):
        row = {"pair_index": pair_index, "order": schedule["order"]}
        for role in ("control", "treatment"):
            arm = contract_document["arms"][role]
            started, ended = starts[(pair_index, role)]
            raw_path = root / f"raw-{pair_index}-{role}.json"
            raw = {
                "driver": {"started_monotonic_ns": started, "ended_monotonic_ns": ended},
                "functional_output_sha256": "0" * 64,
                "build_receipt": build_receipts[role],
                "board_receipts": {"before": {"probe": probe}, "after": {"probe": probe}},
            }
            raw_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
            receipt_path = root / f"receipt-{pair_index}-{role}.yaml"
            _write_yaml(receipt_path, {"raw_measurement": {
                "path": raw_path.name,
                "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()}})
            result_path = root / f"result-{pair_index}-{role}.yaml"
            result = {
                "run_id": f"run-{pair_index}-{role}", "study_label": "frozen-study",
                "target": "unit-test", "model": "model", "checkpoint": "checkpoint",
                "fidelity": "full", "precision": "fp32", "core_count": 1,
                "backend": arm["backend"], "runtime": "merlin",
                "artifact_sha256": "5" * 64, "session": {"kind": "continuous"},
                "lifecycle": {"status": "pass"}, "correctness": {"gate_ok": True},
                "quality": {"gate_ok": True},
                "timing": {"scope": "end_to_end", "samples": [
                    samples[role][pair_index] - 1, samples[role][pair_index],
                    samples[role][pair_index] + 1]},
                "measurement_receipt": {
                    "path": str(receipt_path.resolve()),
                    "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                    "aet_run_id": f"run-{pair_index}-{role}",
                    "command_sha256": "1" * 64,
                },
                "provenance": {
                    "compiler_policy_sha256": arm["policy_sha256"],
                    "compiler_source_sha256": "3" * 64, "runtime_sha256": "4" * 64,
                    "package_sha256": arm["package_sha256"], "binary": arm["binary_sha256"],
                    "capture_session_identity_sha256": "6" * 64,
                },
            }
            _write_yaml(result_path, result)
            row[role] = {
                "result": _ref(result_path, root), "receipt": _ref(receipt_path, root),
                "issuance_fingerprint": hashlib.sha256(
                    f"fingerprint-{pair_index}-{role}".encode()).hexdigest(),
            }
        runs.append(row)
    evidence = root / "pair-evidence.yaml"
    evidence_document = {
        "schema_version": 1, "kind": "paper_full_model_ablation_pair_evidence_v1",
        "status": "complete", "pair_id": "model-fp32-1c", "binding_sha256": binding_sha,
        "contract_sha256": sha256_paths([contract]), "runs": runs,
        "summary": {"pair_count": 3, "control_median_ns": 120,
                    "treatment_median_ns": 90, "treatment_better_pairs": 3,
                    "treatment_improved": True},
    }
    _write_yaml(evidence, evidence_document)
    manifest = root / "manifest.yaml"
    manifest_document = {
        "schema_version": 2, "kind": "paper_full_model_causal_evidence_manifest_v2",
        "status": "frozen", "study_identity_sha256": "1" * 64,
        "records": [{
            "model": "model", "precision": "fp32", "core_count": 1,
            "comparator": "executorch_xnnpack", "binding": binding,
            "binding_sha256": binding_sha, "pair_contract": _ref(contract, root),
            "pair_evidence": _ref(evidence, root),
        }],
    }
    _write_yaml(manifest, manifest_document)
    return manifest, binding


def _verifier(receipt_path: Path, **_kwargs) -> dict[str, str]:
    return {"receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "command_sha256": "1" * 64}


def _validate(manifest: Path, binding: dict):
    return validate_manifest(
        manifest, {}, expected_binding=lambda *_args, **_kwargs: binding,
        study_identity_sha256="1" * 64, hasher=sha256_paths,
        receipt_verifier=_verifier)


def test_full_model_pair_is_claim_ready_only_after_exact_controller_pairs(
        tmp_path, monkeypatch):
    manifest, binding = _fixture(tmp_path, monkeypatch)

    records = _validate(manifest, binding)

    record = records[("model", "fp32", 1, "executorch_xnnpack")]
    assert record["treatment_binary_sha256"] == "f" * 64
    assert "3/3 paired runs" in record["why"]
    assert "fusion layout, runtime synchronization" in record["how"]
    assert record["retained_ablation"]["pair_contract_sha256"] == sha256_paths([
        manifest.parent / "pair-contract.yaml"])


def test_full_model_pair_rejects_partial_evidence(tmp_path, monkeypatch):
    manifest, binding = _fixture(tmp_path, monkeypatch)
    evidence_path = manifest.parent / "pair-evidence.yaml"
    evidence = yaml.safe_load(evidence_path.read_text(encoding="utf-8"))
    evidence["runs"].pop()
    _write_yaml(evidence_path, evidence)
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    document["records"][0]["pair_evidence"]["sha256"] = sha256_paths([evidence_path])
    _write_yaml(manifest, document)

    with pytest.raises(FullModelAblationError, match="partial"):
        _validate(manifest, binding)


def test_full_model_pair_rejects_duplicate_cell_records(tmp_path, monkeypatch):
    manifest, binding = _fixture(tmp_path, monkeypatch)
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    document["records"].append(copy.deepcopy(document["records"][0]))
    _write_yaml(manifest, document)

    with pytest.raises(FullModelAblationError, match="duplicate"):
        _validate(manifest, binding)


def test_full_model_pair_rejects_a_non_improving_treatment_even_with_refreshed_digests(
        tmp_path, monkeypatch):
    manifest, binding = _fixture(tmp_path, monkeypatch)
    evidence_path = manifest.parent / "pair-evidence.yaml"
    evidence = yaml.safe_load(evidence_path.read_text(encoding="utf-8"))
    for row in evidence["runs"]:
        result_path = manifest.parent / row["treatment"]["result"]["path"]
        result = yaml.safe_load(result_path.read_text(encoding="utf-8"))
        result["timing"]["samples"] = [199, 200, 201]
        _write_yaml(result_path, result)
        row["treatment"]["result"]["sha256"] = sha256_paths([result_path])
    evidence["summary"] = {
        "pair_count": 3, "control_median_ns": 120, "treatment_median_ns": 200,
        "treatment_better_pairs": 0, "treatment_improved": False,
    }
    _write_yaml(evidence_path, evidence)
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    document["records"][0]["pair_evidence"]["sha256"] = sha256_paths([evidence_path])
    _write_yaml(manifest, document)

    with pytest.raises(FullModelAblationError, match="repeatable treatment improvement"):
        _validate(manifest, binding)


def test_k1_freeze_rejects_legacy_comparator_as_ablation_control(tmp_path):
    manifest = tmp_path / "legacy.yaml"
    _write_yaml(manifest, {"schema_version": 1, "records": []})
    raw = {"target": "k1", "reporting": {"causal_attribution": {"path": str(manifest)}}}

    with pytest.raises(CausalEvidenceError, match="schema-v2 paired full-model"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_matrix_result_must_use_the_exact_paired_treatment_binary():
    binding = _binding()
    common = {
        "model": "model", "precision": "fp32", "core_count": 1,
        "study_label": "frozen-study", "target": "unit-test", "checkpoint": "checkpoint",
        "fidelity": "full", "artifact_sha256": "5" * 64,
        "session": {"kind": "continuous"}, "quantization": "none",
    }
    provenance = {
        "study_sha256": "0" * 64, "compiler_policy_sha256": "2" * 64,
        "compiler_source_sha256": "3" * 64, "runtime_sha256": "4" * 64,
        "capture_session_identity_sha256": "6" * 64,
    }
    ours = {**common, "backend": "merlin_frozen", "runtime": "merlin",
            "provenance": {**provenance, "package_sha256": "7" * 64, "binary": "f" * 64}}
    comparator = {
        **common, "backend": "executorch_xnnpack", "runtime": "executorch",
        "provenance": {**provenance, "framework_package_sha256": "8" * 64,
                       "framework_source_sha256": "9" * 64},
    }
    record = {"binding": binding, "treatment_binary_sha256": "f" * 64}
    assert _result_matches(record, ours, comparator, study_sha256="0" * 64) is None

    ours["provenance"]["binary"] = "e" * 64
    assert "paired full-model treatment" in str(_result_matches(
        record, ours, comparator, study_sha256="0" * 64))


@pytest.mark.parametrize("mutation, message", [
    ("binary", "provenance"),
    ("output", "functional correctness"),
    ("environment", "board identity"),
    ("order", "order"),
])
def test_full_model_pair_rejects_tampered_identity_correctness_environment_or_order(
        tmp_path, monkeypatch, mutation, message):
    manifest, binding = _fixture(tmp_path, monkeypatch)
    evidence_path = manifest.parent / "pair-evidence.yaml"
    evidence = yaml.safe_load(evidence_path.read_text(encoding="utf-8"))
    run = evidence["runs"][0]
    if mutation == "binary":
        result_path = manifest.parent / run["treatment"]["result"]["path"]
        result = yaml.safe_load(result_path.read_text(encoding="utf-8"))
        result["provenance"]["binary"] = "1" * 64
        _write_yaml(result_path, result)
        run["treatment"]["result"]["sha256"] = sha256_paths([result_path])
    elif mutation in {"output", "environment"}:
        receipt_path = manifest.parent / run["treatment"]["receipt"]["path"]
        receipt = yaml.safe_load(receipt_path.read_text(encoding="utf-8"))
        raw_path = receipt_path.parent / receipt["raw_measurement"]["path"]
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if mutation == "output":
            raw["functional_output_sha256"] = "1" * 64
        else:
            probe = json.loads(raw["board_receipts"]["after"]["probe"])
            probe["identity"] = "other-board"
            raw["board_receipts"]["after"]["probe"] = json.dumps(probe, sort_keys=True)
        raw_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
        receipt["raw_measurement"]["sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        _write_yaml(receipt_path, receipt)
        run["treatment"]["receipt"]["sha256"] = sha256_paths([receipt_path])
        result_path = manifest.parent / run["treatment"]["result"]["path"]
        result = yaml.safe_load(result_path.read_text(encoding="utf-8"))
        result["measurement_receipt"]["sha256"] = hashlib.sha256(
            receipt_path.read_bytes()).hexdigest()
        _write_yaml(result_path, result)
        run["treatment"]["result"]["sha256"] = sha256_paths([result_path])
    else:
        run["order"] = "treatment_first"
    _write_yaml(evidence_path, evidence)
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    document["records"][0]["pair_evidence"]["sha256"] = sha256_paths([evidence_path])
    _write_yaml(manifest, document)

    with pytest.raises(FullModelAblationError, match=message):
        _validate(manifest, binding)

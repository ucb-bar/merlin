"""Validate frozen, full-model compiler ablations used by paper win claims.

The comparison backend is not an ablation control.  A causal record therefore has two independent
parts: the ordinary Merlin-versus-comparator binding, and a paired Merlin-control-versus-Merlin-
treatment experiment.  This module validates the latter.  Each sample is a separately issued
measurement-controller run and pairs are executed in an alternating AB/BA order on one board.

The module is intentionally a verifier, not a benchmark driver.  It cannot turn handwritten timing
rows into evidence: every run must replay through ``paper_measurement_controller.verify_receipt``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml


_HEX = frozenset("0123456789abcdef")
_COMPONENTS = frozenset({
    "tiling_dataflow", "fusion_layout", "register_residency",
    "instruction_selection", "runtime_synchronization",
})


class FullModelAblationError(ValueError):
    """A production full-model causal contract or its evidence is invalid."""


def _sha(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in _HEX for character in text)


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _closed(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FullModelAblationError(f"{label} must be a mapping")
    extra, missing = sorted(set(value) - fields), sorted(fields - set(value))
    if extra or missing:
        raise FullModelAblationError(
            f"{label} is closed; unrecognized={extra} missing={missing}")
    return value


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise FullModelAblationError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise FullModelAblationError(f"{label} must be a mapping")
    return value


def _retained(root: Path, value: object, label: str,
              hasher: Callable[[list[Path]], str]) -> tuple[Path, str]:
    value = _closed(value, {"path", "sha256"}, label)
    relative = Path(str(value["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise FullModelAblationError(f"{label}.path must be relative and contained")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise FullModelAblationError(f"{label}.path escapes its manifest") from error
    digest = str(value["sha256"])
    if not path.is_file() or not _sha(digest) or hasher([path]) != digest:
        raise FullModelAblationError(f"{label} digest differs from retained bytes")
    return path, digest


def _raw_measurement(receipt_path: Path) -> dict[str, Any]:
    receipt = _load(receipt_path, "full-model controller receipt")
    raw_ref = receipt.get("raw_measurement")
    if not isinstance(raw_ref, Mapping) or set(raw_ref) != {"path", "sha256"}:
        raise FullModelAblationError("controller receipt has no closed raw-measurement reference")
    relative = Path(str(raw_ref["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise FullModelAblationError("controller raw-measurement path is unsafe")
    raw_path = (receipt_path.parent / relative).resolve()
    try:
        raw_path.relative_to(receipt_path.parent.resolve())
    except ValueError as error:
        raise FullModelAblationError("controller raw measurement escapes its receipt") from error
    if (not raw_path.is_file() or not _sha(raw_ref["sha256"])
            or _content_sha(raw_path) != raw_ref["sha256"]):
        raise FullModelAblationError("controller raw-measurement digest differs")
    try:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FullModelAblationError("controller raw measurement is invalid JSON") from error
    if not isinstance(raw, dict):
        raise FullModelAblationError("controller raw measurement must be a mapping")
    return raw


def _probe(raw: Mapping[str, Any], endpoint: str) -> Mapping[str, Any]:
    try:
        text = raw["board_receipts"][endpoint]["probe"]
        value = json.loads(text)
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise FullModelAblationError(
            f"controller raw measurement lacks its {endpoint} board probe") from error
    expected = {"schema_version", "kind", "identity", "vlen_bits", "vlen_source",
                "governor", "current_khz", "max_khz", "max_thermal_millic"}
    value = _closed(value, expected, f"{endpoint} board probe")
    if (value["schema_version"] != 1 or value["kind"] != "merlin_board_probe_v1"
            or value["vlen_source"] != "csr" or value["governor"] != "performance"
            or value["current_khz"] != value["max_khz"]
            or not str(value["identity"]).strip()
            or any(type(value[field]) is not int or value[field] <= 0 for field in (
                "vlen_bits", "current_khz", "max_khz", "max_thermal_millic"))):
        raise FullModelAblationError(f"{endpoint} board probe is not claim-safe")
    return value


def _contract(path: Path, *, binding: Mapping[str, Any], binding_sha256: str,
              hasher: Callable[[list[Path]], str]) -> tuple[dict[str, Any], dict[str, str]]:
    root = path.parent
    value = _closed(_load(path, "full-model ablation pair contract"), {
        "schema_version", "kind", "status", "pair_id", "binding_sha256",
        "measurement_study_sha256", "intervention", "pairing", "environment", "arms",
    }, "full-model ablation pair contract")
    if (value["schema_version"] != 1
            or value["kind"] != "paper_full_model_ablation_pair_contract_v1"
            or value["status"] != "frozen" or not str(value["pair_id"]).strip()
            or value["binding_sha256"] != binding_sha256
            or not _sha(value["measurement_study_sha256"])):
        raise FullModelAblationError("full-model pair contract identity is invalid")

    intervention = _closed(value["intervention"], {
        "id", "scope", "isolated_change", "changed_components", "control_policy_sha256",
        "treatment_policy_sha256", "delta_manifest",
    }, "full-model intervention")
    components = intervention["changed_components"]
    if (not str(intervention["id"]).strip()
            or intervention["scope"] != "compiler_full_model_transform"
            or intervention["isolated_change"] != "compiler_policy"
            or not isinstance(components, list) or not components
            or len(set(components)) != len(components)
            or any(component not in _COMPONENTS for component in components)
            or not _sha(intervention["control_policy_sha256"])
            or intervention["control_policy_sha256"] == intervention["treatment_policy_sha256"]
            or intervention["treatment_policy_sha256"] != binding[
                "compiler_policy_sha256"]):
        raise FullModelAblationError(
            "full-model intervention is not an isolated typed compiler-policy change")
    delta_path, delta_sha = _retained(
        root, intervention["delta_manifest"], "compiler transformation delta", hasher)
    delta = _closed(_load(delta_path, "compiler transformation delta"), {
        "schema_version", "kind", "status", "control_policy_sha256",
        "treatment_policy_sha256", "changed_components",
        "unchanged_build_configuration_sha256",
    }, "compiler transformation delta")
    if (delta["schema_version"] != 1
            or delta["kind"] != "paper_compiler_transform_delta_v1"
            or delta["status"] != "frozen"
            or delta["control_policy_sha256"] != intervention["control_policy_sha256"]
            or delta["treatment_policy_sha256"] != intervention["treatment_policy_sha256"]
            or delta["changed_components"] != components
            or not _sha(delta["unchanged_build_configuration_sha256"])):
        raise FullModelAblationError("compiler transformation delta differs from the pair contract")

    pairing = _closed(value["pairing"], {
        "metric", "direction", "primary_scope", "sample_unit", "per_run_samples",
        "pair_count", "schedule",
    }, "full-model pairing")
    pair_count = pairing["pair_count"]
    schedule = pairing["schedule"]
    if (pairing["metric"] != "end_to_end_latency_ns"
            or pairing["direction"] != "lower_is_better"
            or pairing["primary_scope"] != "end_to_end"
            or pairing["sample_unit"] != "continuous_session"
            or type(pairing["per_run_samples"]) is not int
            or pairing["per_run_samples"] < 3
            or type(pair_count) is not int or pair_count < 3
            or not isinstance(schedule, list) or len(schedule) != pair_count):
        raise FullModelAblationError("full-model pair schedule is incomplete")
    for index, row in enumerate(schedule):
        row = _closed(row, {"pair_index", "order"}, f"pair schedule[{index}]")
        expected_order = "control_first" if index % 2 == 0 else "treatment_first"
        if row["pair_index"] != index or row["order"] != expected_order:
            raise FullModelAblationError("full-model pair schedule must be balanced alternating AB/BA")

    environment = _closed(value["environment"], {
        "target", "require_same_board_identity", "require_same_vlen",
        "require_performance_governor", "require_current_equals_max",
        "maximum_thermal_delta_millic",
    }, "full-model pair environment")
    if (environment["target"] != binding["target"]
            or environment["require_same_board_identity"] is not True
            or environment["require_same_vlen"] is not True
            or environment["require_performance_governor"] is not True
            or environment["require_current_equals_max"] is not True
            or type(environment["maximum_thermal_delta_millic"]) is not int
            or environment["maximum_thermal_delta_millic"] < 0):
        raise FullModelAblationError("full-model pair environment gates are incomplete")

    arms = _closed(value["arms"], {"control", "treatment"}, "full-model pair arms")
    normalized: dict[str, str] = {"transformation_delta_sha256": delta_sha}
    for role in ("control", "treatment"):
        arm = _closed(arms[role], {
            "backend", "runtime", "compiler_source_sha256", "runtime_sha256",
            "package_sha256", "policy_sha256", "binary_sha256",
            "build_configuration_sha256", "build_receipt_sha256",
        }, f"full-model {role} arm")
        for field in set(arm) - {"backend", "runtime"}:
            if not _sha(arm[field]):
                raise FullModelAblationError(f"full-model {role} has invalid {field}")
        if (arm["runtime"] != binding["ours"]["runtime"]
                or arm["compiler_source_sha256"] != binding["compiler_source_sha256"]
                or arm["runtime_sha256"] != binding["runtime_sha256"]
                or arm["policy_sha256"] != intervention[f"{role}_policy_sha256"]
                or arm["build_configuration_sha256"] != delta[
                    "unchanged_build_configuration_sha256"]):
            raise FullModelAblationError(
                f"full-model {role} differs outside the frozen compiler-policy intervention")
        normalized[f"{role}_binary_sha256"] = str(arm["binary_sha256"])
        normalized[f"{role}_build_receipt_sha256"] = str(arm["build_receipt_sha256"])
        normalized[f"{role}_policy_sha256"] = str(arm["policy_sha256"])
    control, treatment = arms["control"], arms["treatment"]
    if (control["backend"] != "merlin_ablation_control"
            or treatment["backend"] != binding["ours"]["backend"]
            or treatment["package_sha256"] != binding["ours"]["package_sha256"]
            or treatment["policy_sha256"] != binding["compiler_policy_sha256"]
            or control["package_sha256"] == treatment["package_sha256"]
            or control["binary_sha256"] == treatment["binary_sha256"]):
        raise FullModelAblationError(
            "full-model control/treatment package or binary identities are not isolated")
    return dict(value), normalized


def _arm_run(root: Path, value: object, *, role: str, pair_index: int,
             contract: Mapping[str, Any], binding: Mapping[str, Any],
             hasher: Callable[[list[Path]], str],
             verifier: Callable[..., Mapping[str, Any]]) -> dict[str, Any]:
    value = _closed(value, {"result", "receipt", "issuance_fingerprint"},
                    f"pair {pair_index} {role} run")
    result_path, result_sha = _retained(
        root, value["result"], f"pair {pair_index} {role} result", hasher)
    receipt_path, receipt_sha = _retained(
        root, value["receipt"], f"pair {pair_index} {role} receipt", hasher)
    fingerprint = str(value["issuance_fingerprint"])
    if not _sha(fingerprint):
        raise FullModelAblationError(f"pair {pair_index} {role} issuance fingerprint is invalid")
    result = _load(result_path, f"pair {pair_index} {role} result")
    try:
        from .paper import validate_paper_result
        validate_paper_result(result)
        root_evidence = verifier(
            receipt_path, expected_result=result,
            expected_study_sha256=contract["measurement_study_sha256"],
            trusted_issuance_fingerprint=fingerprint)
    except (ValueError, OSError) as error:
        raise FullModelAblationError(
            f"pair {pair_index} {role} controller receipt is invalid: {error}") from error
    if root_evidence.get("receipt_sha256") != _content_sha(receipt_path):
        raise FullModelAblationError(f"pair {pair_index} {role} receipt replay differs")
    measurement_ref = _closed(result.get("measurement_receipt"), {
        "path", "sha256", "aet_run_id", "command_sha256",
    }, f"pair {pair_index} {role} result measurement receipt")
    if (Path(str(measurement_ref["path"])).resolve() != receipt_path
            or measurement_ref["sha256"] != _content_sha(receipt_path)
            or measurement_ref["aet_run_id"] != result.get("run_id")
            or measurement_ref["command_sha256"] != root_evidence.get("command_sha256")):
        raise FullModelAblationError(
            f"pair {pair_index} {role} result does not name the replayed controller receipt")

    arm = contract["arms"][role]
    expected = {
        "study_label": binding["study_label"], "target": binding["target"],
        "model": binding["model"], "checkpoint": binding["checkpoint"],
        "fidelity": binding["fidelity"], "precision": binding["precision"],
        "core_count": binding["core_count"], "backend": arm["backend"],
        "runtime": arm["runtime"], "artifact_sha256": binding["capture_sha256"],
    }
    if any(result.get(field) != expected_value for field, expected_value in expected.items()):
        raise FullModelAblationError(f"pair {pair_index} {role} result identity differs")
    if _canonical_sha(result.get("session")) != binding["session_protocol_sha256"]:
        raise FullModelAblationError(f"pair {pair_index} {role} session differs")
    provenance = result.get("provenance", {}) or {}
    expected_provenance = {
        "compiler_policy_sha256": arm["policy_sha256"],
        "compiler_source_sha256": arm["compiler_source_sha256"],
        "runtime_sha256": arm["runtime_sha256"], "package_sha256": arm["package_sha256"],
        "binary": arm["binary_sha256"],
        "capture_session_identity_sha256": binding["capture_session_identity_sha256"],
    }
    if any(provenance.get(field) != expected_value
           for field, expected_value in expected_provenance.items()):
        raise FullModelAblationError(f"pair {pair_index} {role} provenance differs")
    if (result.get("lifecycle", {}).get("status") != "pass"
            or result.get("correctness", {}).get("gate_ok") is not True
            or result.get("quality", {}).get("gate_ok") is not True):
        raise FullModelAblationError(f"pair {pair_index} {role} did not pass correctness")
    timing = result.get("timing", {}) or {}
    samples = timing.get("samples")
    if (timing.get("scope") != "end_to_end" or not isinstance(samples, list)
            or len(samples) != contract["pairing"]["per_run_samples"]
            or any(type(sample) is not int or sample <= 0 for sample in samples)):
        raise FullModelAblationError(
            f"pair {pair_index} {role} has the wrong repeated end-to-end session samples")
    raw = _raw_measurement(receipt_path)
    driver = raw.get("driver", {}) or {}
    started, ended = driver.get("started_monotonic_ns"), driver.get("ended_monotonic_ns")
    if (type(started) is not int or type(ended) is not int or ended <= started
            or raw.get("functional_output_sha256") is None
            or not _sha(raw.get("functional_output_sha256"))
            or _canonical_sha(raw.get("build_receipt")) != arm["build_receipt_sha256"]):
        raise FullModelAblationError(f"pair {pair_index} {role} raw provenance is invalid")
    return {
        "sample_ns": _median(samples), "started_ns": started, "ended_ns": ended,
        "functional_output_sha256": raw["functional_output_sha256"],
        "correctness_sha256": _canonical_sha({
            "correctness": result["correctness"], "quality": result["quality"]}),
        "before": dict(_probe(raw, "before")), "after": dict(_probe(raw, "after")),
        "result_sha256": result_sha, "receipt_sha256": receipt_sha,
        "issuance_fingerprint": fingerprint,
    }


def _median(values: list[int]) -> int:
    value = statistics.median(values)
    return int(value) if int(value) == value else int(value * 2) / 2


def validate_manifest(path: Path, raw: Mapping[str, Any], *,
                      expected_binding: Callable[..., dict[str, Any]],
                      study_identity_sha256: str,
                      hasher: Callable[[list[Path]], str],
                      receipt_verifier: Callable[..., Mapping[str, Any]] | None = None,
                      ) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    """Validate a schema-v2 causal manifest and return report-ready indexed records."""
    from .paper_measurement_controller import verify_receipt
    verifier = verify_receipt if receipt_verifier is None else receipt_verifier
    manifest = _closed(_load(path, "full-model causal evidence manifest"), {
        "schema_version", "kind", "status", "study_identity_sha256", "records",
    }, "full-model causal evidence manifest")
    if (manifest["schema_version"] != 2
            or manifest["kind"] != "paper_full_model_causal_evidence_manifest_v2"
            or manifest["status"] != "frozen"
            or manifest["study_identity_sha256"] != study_identity_sha256):
        raise FullModelAblationError("full-model causal manifest identity differs from the study")
    records = manifest["records"]
    if not isinstance(records, list):
        raise FullModelAblationError("full-model causal manifest records must be a list")
    indexed: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for record_index, record in enumerate(records):
        record = _closed(record, {
            "model", "precision", "core_count", "comparator", "binding",
            "binding_sha256", "pair_contract", "pair_evidence",
        }, f"full-model causal record[{record_index}]")
        try:
            key = (str(record["model"]), str(record["precision"]),
                   int(record["core_count"]), str(record["comparator"]))
        except (TypeError, ValueError) as error:
            raise FullModelAblationError("full-model causal record key is invalid") from error
        if key in indexed:
            raise FullModelAblationError(f"duplicate full-model causal record {key}")
        binding = expected_binding(
            raw, model=key[0], precision=key[1], core_count=key[2], comparator=key[3])
        binding_sha = _canonical_sha(binding)
        if record["binding"] != binding or record["binding_sha256"] != binding_sha:
            raise FullModelAblationError(f"full-model causal record {key} binding differs")
        contract_path, contract_sha = _retained(
            path.parent, record["pair_contract"], "full-model pair contract", hasher)
        contract, normalized = _contract(
            contract_path, binding=binding, binding_sha256=binding_sha, hasher=hasher)
        evidence_path, evidence_sha = _retained(
            path.parent, record["pair_evidence"], "full-model pair evidence", hasher)
        evidence = _closed(_load(evidence_path, "full-model pair evidence"), {
            "schema_version", "kind", "status", "pair_id", "binding_sha256",
            "contract_sha256", "runs", "summary",
        }, "full-model pair evidence")
        if (evidence["schema_version"] != 1
                or evidence["kind"] != "paper_full_model_ablation_pair_evidence_v1"
                or evidence["status"] != "complete" or evidence["pair_id"] != contract["pair_id"]
                or evidence["binding_sha256"] != binding_sha
                or evidence["contract_sha256"] != contract_sha):
            raise FullModelAblationError("full-model pair evidence differs from its contract")
        runs = evidence["runs"]
        schedule = contract["pairing"]["schedule"]
        if not isinstance(runs, list) or len(runs) != len(schedule):
            raise FullModelAblationError("full-model pair evidence is partial")
        controls: list[int] = []
        treatments: list[int] = []
        output_sha: str | None = None
        correctness_sha: str | None = None
        board_identity: str | None = None
        vlen_bits: int | None = None
        temperatures: list[int] = []
        retained_runs: dict[str, list[str]] = {"control": [], "treatment": []}
        for pair_index, (declared, run) in enumerate(zip(schedule, runs, strict=True)):
            run = _closed(run, {"pair_index", "order", "control", "treatment"},
                          f"full-model pair evidence run[{pair_index}]")
            if run["pair_index"] != pair_index or run["order"] != declared["order"]:
                raise FullModelAblationError("full-model pair evidence order differs")
            arm_rows = {
                role: _arm_run(
                    evidence_path.parent, run[role], role=role, pair_index=pair_index,
                    contract=contract, binding=binding, hasher=hasher, verifier=verifier)
                for role in ("control", "treatment")
            }
            first, second = (("control", "treatment") if run["order"] == "control_first"
                             else ("treatment", "control"))
            if arm_rows[first]["ended_ns"] > arm_rows[second]["started_ns"]:
                raise FullModelAblationError(
                    "full-model pair controller runs overlap or violate declared AB/BA order")
            for role, arm_row in arm_rows.items():
                if output_sha is None:
                    output_sha = arm_row["functional_output_sha256"]
                    correctness_sha = arm_row["correctness_sha256"]
                    board_identity = arm_row["before"]["identity"]
                    vlen_bits = arm_row["before"]["vlen_bits"]
                if (arm_row["functional_output_sha256"] != output_sha
                        or arm_row["correctness_sha256"] != correctness_sha):
                    raise FullModelAblationError(
                        "full-model pair arms do not have identical functional correctness")
                for endpoint in ("before", "after"):
                    probe = arm_row[endpoint]
                    if probe["identity"] != board_identity or probe["vlen_bits"] != vlen_bits:
                        raise FullModelAblationError(
                            "full-model pair board identity or RVV VLEN changed")
                    temperatures.append(int(probe["max_thermal_millic"]))
                retained_runs[role].extend([
                    arm_row["result_sha256"], arm_row["receipt_sha256"],
                    arm_row["issuance_fingerprint"]])
            controls.append(arm_rows["control"]["sample_ns"])
            treatments.append(arm_rows["treatment"]["sample_ns"])
        if max(temperatures) - min(temperatures) > contract["environment"][
                "maximum_thermal_delta_millic"]:
            raise FullModelAblationError("full-model pair thermal regime exceeded its frozen bound")
        control_median, treatment_median = _median(controls), _median(treatments)
        better = sum(treatment < control for treatment, control in zip(
            treatments, controls, strict=True))
        summary = _closed(evidence["summary"], {
            "pair_count", "control_median_ns", "treatment_median_ns",
            "treatment_better_pairs", "treatment_improved",
        }, "full-model pair summary")
        expected_summary = {
            "pair_count": len(controls), "control_median_ns": control_median,
            "treatment_median_ns": treatment_median, "treatment_better_pairs": better,
            "treatment_improved": treatment_median < control_median and better > len(controls) // 2,
        }
        if dict(summary) != expected_summary or summary["treatment_improved"] is not True:
            raise FullModelAblationError(
                "full-model pair does not support a repeatable treatment improvement")
        components = contract["intervention"]["changed_components"]
        component_text = ", ".join(str(component).replace("_", " ") for component in components)
        why = (f"With the frozen full-model session and board regime held fixed, the Merlin "
               f"treatment improved {better}/{len(controls)} paired runs.")
        how = (f"The isolated compiler-policy intervention changed only {component_text}; "
               f"paired median latency changed from {control_median} ns to "
               f"{treatment_median} ns.")
        indexed[key] = {
            "model": key[0], "precision": key[1], "core_count": key[2],
            "comparator": key[3], "binding": binding, "binding_sha256": binding_sha,
            "ablation": dict(record["pair_evidence"]),
            "structural": dict(contract["intervention"]["delta_manifest"]),
            "retained_ablation": {
                "pair_contract_sha256": contract_sha,
                "pair_evidence_sha256": evidence_sha,
                "transformation_delta_sha256": normalized["transformation_delta_sha256"],
                "control_artifact_sha256": normalized["control_binary_sha256"],
                "treatment_artifact_sha256": normalized["treatment_binary_sha256"],
                "control_measurement_roots_sha256": _canonical_sha(retained_runs["control"]),
                "treatment_measurement_roots_sha256": _canonical_sha(
                    retained_runs["treatment"]),
            },
            "treatment_binary_sha256": normalized["treatment_binary_sha256"],
            "why": why, "how": how,
        }
    return indexed


def main(argv: list[str] | None = None) -> int:
    """Verify a complete pre-freeze full-model evidence manifest without editing it."""
    parser = argparse.ArgumentParser(
        prog="merlin-paper-full-model-ablation",
        description="Verify paired full-model compiler causal evidence")
    parser.add_argument("--study", type=Path, required=True,
                        help="study YAML whose non-lifecycle identity the pairs bind")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="schema-v2 frozen full-model evidence manifest")
    arguments = parser.parse_args(argv)
    from .freeze import sha256_paths
    from .paper import PaperStudySpec
    from .paper_attribution import _study_identity_sha, expected_binding
    spec = PaperStudySpec.from_yaml(arguments.study)
    records = validate_manifest(
        arguments.manifest.resolve(), spec.canonical_dict(),
        expected_binding=expected_binding,
        study_identity_sha256=_study_identity_sha(spec.canonical_dict()),
        hasher=sha256_paths)
    print(json.dumps({
        "status": "pass", "manifest": str(arguments.manifest.resolve()),
        "record_count": len(records),
        "cells": ["/".join((model, precision, f"{cores}c", comparator))
                  for model, precision, cores, comparator in sorted(records)],
    }, sort_keys=True))
    return 0


__all__ = ["FullModelAblationError", "main", "validate_manifest"]


if __name__ == "__main__":  # pragma: no cover - CLI is a thin verifier wrapper
    raise SystemExit(main())

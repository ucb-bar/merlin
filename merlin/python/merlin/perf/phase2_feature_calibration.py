"""Prepare and revalidate Phase-2 feature coefficients from paired evidence.

This module is a host-only evidence join.  It does not execute a target, invoke a simulator, or
infer a coefficient from whole-program measurements.  A coefficient is derived only from explicitly
paired controlled observations whose target, source, controls, and evidence bytes are all bound by
SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

REQUEST_SCHEMA = "phase2_analytical_feature_calibration_request_v1"
OBSERVATION_SCHEMA = "phase2_controlled_feature_observation_v1"
CONTROLS_SCHEMA = "phase2_controlled_feature_controls_v1"
FEATURE_SCHEMA = "phase2_analytical_feature_calibration_v1"
PREPARATION_SCHEMA = "phase2_analytical_feature_calibration_preparation_v1"
VALIDATION_SCHEMA = "phase2_analytical_feature_calibration_validation_v1"

_REQUIRED_MEASUREMENTS = {
    "compute": {
        "cycles_per_unit": ("cycle", "target_execution"),
    },
    "movement": {
        "physical_bytes_per_unit": ("byte", "physical_target_interface"),
        "commands_per_unit": ("command", "target_execution"),
    },
}


class _EvidenceError(ValueError):
    def __init__(self, message: str, *, integrity: bool = False):
        super().__init__(message)
        self.integrity = integrity


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _load_json(source: Mapping[str, Any] | Path) -> tuple[dict[str, Any], str, Path]:
    if isinstance(source, Mapping):
        document = dict(source)
        return document, _digest(document), Path.cwd()
    path = Path(source)
    if path.is_symlink():
        raise _EvidenceError(f"{path}: JSON document is a symlink", integrity=True)
    if not path.is_file():
        raise _EvidenceError(f"{path}: JSON document is unavailable")
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise _EvidenceError(f"{path}: invalid JSON: {exc}", integrity=True) from exc
    if not isinstance(document, Mapping):
        raise _EvidenceError(f"{path}: JSON document must contain an object", integrity=True)
    return dict(document), hashlib.sha256(raw).hexdigest(), path.parent


def _verified_file(reference: Any, label: str, base: Path) -> tuple[Path, str]:
    row = _mapping(reference)
    raw_path = str(row.get("path") or "")
    expected = row.get("sha256")
    if not raw_path:
        raise _EvidenceError(f"{label}: path is absent")
    if not _is_sha256(expected):
        raise _EvidenceError(f"{label}: sha256 is absent or invalid", integrity=True)
    path = Path(raw_path)
    path = path if path.is_absolute() else base / path
    if path.is_symlink():
        raise _EvidenceError(f"{label}: {path} is a symlink", integrity=True)
    if not path.is_file():
        raise _EvidenceError(f"{label}: {path} is unavailable")
    actual = _file_digest(path)
    if actual != expected:
        raise _EvidenceError(f"{label}: sha256 mismatch (declared {expected}, actual {actual})", integrity=True)
    return path, actual


def _pointer(document: Mapping[str, Any], pointer: Any, label: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise _EvidenceError(f"{label}: must be an absolute JSON pointer", integrity=True)
    value: Any = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, Mapping) and token in value:
            value = value[token]
        elif (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and token.isdigit()
            and int(token) < len(value)
        ):
            value = value[int(token)]
        else:
            raise _EvidenceError(f"{label}: {pointer} is absent")
    return value


def _fraction(value: Any, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _EvidenceError(f"{label}: value must be numeric")
    if not math.isfinite(float(value)) or value < 0:
        raise _EvidenceError(f"{label}: value must be finite and non-negative")
    return Fraction(str(value))


def _quantity(document: Mapping[str, Any], pointer: Any, unit: str, scope: str, label: str) -> Fraction:
    quantity = _mapping(_pointer(document, pointer, label))
    if not quantity:
        raise _EvidenceError(f"{label}: pointer must select a quantity object")
    if quantity.get("unit") != unit or quantity.get("scope") != scope:
        raise _EvidenceError(f"{label}: quantity must declare unit {unit!r} and scope {scope!r}", integrity=True)
    return _fraction(quantity.get("value"), label)


def _controls_contract(path: Path, *, target_sha256: str, feature_pointer: str, label: str) -> None:
    try:
        document = json.loads(path.read_bytes())
    except (TypeError, ValueError) as exc:
        raise _EvidenceError(f"{label}: invalid JSON: {exc}", integrity=True) from exc
    if not isinstance(document, Mapping):
        raise _EvidenceError(f"{label}: document must contain an object", integrity=True)
    if document.get("schema") != CONTROLS_SCHEMA or document.get("status") != "declared":
        raise _EvidenceError(f"{label}: no controlled-variable contract is present", integrity=True)
    if document.get("target_sha256") != target_sha256:
        raise _EvidenceError(f"{label}: target_sha256 does not bind the exact target", integrity=True)
    if document.get("varied_feature_pointer") != feature_pointer:
        raise _EvidenceError(f"{label}: varied feature pointer does not match the observation", integrity=True)
    invariants = tuple(_sequence(document.get("invariant_pointers")))
    if not invariants or len(set(invariants)) != len(invariants):
        raise _EvidenceError(f"{label}: invariant JSON pointers are absent or duplicated", integrity=True)
    for pointer in invariants:
        if not isinstance(pointer, str) or not pointer.startswith("/") or pointer == feature_pointer:
            raise _EvidenceError(
                f"{label}: invariant pointers must be absolute and exclude the varied feature", integrity=True
            )


def _problem(exc: _EvidenceError, field: str) -> dict[str, str]:
    return {
        "field": field,
        "severity": "refusal" if exc.integrity else "missing",
        "reason": str(exc),
    }


def _feature_spec(raw: Any) -> dict[str, Any]:
    row = _mapping(raw)
    kind = str(row.get("kind") or "")
    if kind not in _REQUIRED_MEASUREMENTS:
        raise _EvidenceError("feature.kind must be compute or movement", integrity=True)
    ident = str(row.get("id") or "")
    pointer = row.get("pointer")
    unit = str(row.get("unit") or "")
    evidence_pointer = row.get("evidence_pointer")
    resource = str(row.get("resource") or "")
    if not ident or not resource or not unit:
        raise _EvidenceError("feature needs non-empty id, resource, and unit")
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise _EvidenceError("feature.pointer must be an absolute provider JSON pointer", integrity=True)
    if not isinstance(evidence_pointer, str) or not evidence_pointer.startswith("/"):
        raise _EvidenceError("feature.evidence_pointer must be an absolute JSON pointer", integrity=True)
    effects = tuple(sorted({str(value) for value in _sequence(row.get("effects")) if str(value)}))
    return {
        "id": ident,
        "kind": kind,
        "pointer": pointer,
        "evidence_pointer": evidence_pointer,
        "unit": unit,
        "resource": resource,
        "effects": list(effects),
    }


def _measurements(kind: str, raw: Any) -> list[dict[str, str]]:
    required = _REQUIRED_MEASUREMENTS[kind]
    rows: dict[str, dict[str, str]] = {}
    for index, value in enumerate(_sequence(raw)):
        row = _mapping(value)
        coefficient = str(row.get("coefficient") or "")
        pointer = row.get("evidence_pointer")
        unit = str(row.get("unit") or "")
        scope = str(row.get("scope") or "")
        if coefficient in rows:
            raise _EvidenceError(f"measurements[{index}]: duplicate coefficient {coefficient!r}", integrity=True)
        if coefficient not in required:
            raise _EvidenceError(
                f"measurements[{index}]: coefficient {coefficient!r} is not valid for {kind}", integrity=True
            )
        if not isinstance(pointer, str) or not pointer.startswith("/"):
            raise _EvidenceError(
                f"measurements[{index}].evidence_pointer must be an absolute JSON pointer", integrity=True
            )
        expected_unit, expected_scope = required[coefficient]
        if (unit, scope) != (expected_unit, expected_scope):
            raise _EvidenceError(
                f"measurements[{index}] must use unit {expected_unit!r} and scope {expected_scope!r}", integrity=True
            )
        rows[coefficient] = {
            "coefficient": coefficient,
            "evidence_pointer": pointer,
            "unit": unit,
            "scope": scope,
        }
    missing = sorted(set(required) - set(rows))
    if missing:
        raise _EvidenceError(f"measurements do not define required coefficients {missing}")
    return [rows[name] for name in sorted(rows)]


def _observation(
    arm: Any,
    *,
    label: str,
    target_sha256: str,
    controls_sha256: str,
    feature: Mapping[str, Any],
    measurements: Sequence[Mapping[str, str]],
    base: Path,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    row = _mapping(arm)
    source_path, source_sha = _verified_file(row.get("source"), f"{label}.source", base)
    evidence_path, evidence_sha = _verified_file(row.get("evidence"), f"{label}.evidence", base)
    try:
        document = json.loads(evidence_path.read_bytes())
    except (TypeError, ValueError) as exc:
        raise _EvidenceError(f"{label}.evidence: invalid JSON: {exc}", integrity=True) from exc
    if not isinstance(document, Mapping):
        raise _EvidenceError(f"{label}.evidence: document must contain an object", integrity=True)
    if document.get("schema") != OBSERVATION_SCHEMA or document.get("status") != "measured":
        raise _EvidenceError(f"{label}.evidence: no measured controlled-observation contract is present")
    bindings = _mapping(document.get("bindings"))
    expected = {
        "target_sha256": target_sha256,
        "source_sha256": source_sha,
        "controls_sha256": controls_sha256,
    }
    for name, digest in expected.items():
        if bindings.get(name) != digest:
            raise _EvidenceError(f"{label}.evidence: {name} does not bind the exact declared bytes", integrity=True)
    feature_value = _quantity(
        document, feature["evidence_pointer"], str(feature["unit"]), "emitted_feature", f"{label}.feature"
    )
    measured: dict[str, Fraction] = {}
    for measurement in measurements:
        name = measurement["coefficient"]
        measured[name] = _quantity(
            document, measurement["evidence_pointer"], measurement["unit"], measurement["scope"], f"{label}.{name}"
        )
    return {
        "source_sha256": source_sha,
        "evidence_sha256": evidence_sha,
        "feature_value": feature_value,
        "measurements": measured,
    }, [
        {"path": str(source_path), "sha256": source_sha, "purpose": "controlled source"},
        {"path": str(evidence_path), "sha256": evidence_sha, "purpose": "controlled measured observation"},
    ]


def _derive(
    document: Mapping[str, Any],
    *,
    base: Path,
    target_sha256: str,
    feature: Mapping[str, Any],
    measurements: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    problems: list[dict[str, str]] = []
    pairs: list[dict[str, Any]] = []
    files: list[dict[str, str]] = []
    seen_arms: set[tuple[str, str]] = set()
    seen_pair_ids: set[str] = set()
    for index, raw in enumerate(_sequence(document.get("controlled_pairs"))):
        field = f"controlled_pairs[{index}]"
        pair = _mapping(raw)
        ident = str(pair.get("id") or "")
        try:
            if not ident:
                raise _EvidenceError(f"{field}: id is absent")
            if ident in seen_pair_ids:
                raise _EvidenceError(f"{field}: pair id {ident!r} is duplicated", integrity=True)
            seen_pair_ids.add(ident)
            controls_path, controls_sha = _verified_file(pair.get("controls"), f"{field}.controls", base)
            _controls_contract(
                controls_path,
                target_sha256=target_sha256,
                feature_pointer=str(feature["evidence_pointer"]),
                label=f"{field}.controls",
            )
            first, first_files = _observation(
                pair.get("first"),
                label=f"{field}.first",
                target_sha256=target_sha256,
                controls_sha256=controls_sha,
                feature=feature,
                measurements=measurements,
                base=base,
            )
            second, second_files = _observation(
                pair.get("second"),
                label=f"{field}.second",
                target_sha256=target_sha256,
                controls_sha256=controls_sha,
                feature=feature,
                measurements=measurements,
                base=base,
            )
            if first["source_sha256"] == second["source_sha256"]:
                raise _EvidenceError(f"{field}: paired points must bind distinct source bytes", integrity=True)
            if first["evidence_sha256"] == second["evidence_sha256"]:
                raise _EvidenceError(f"{field}: paired points must bind distinct evidence bytes", integrity=True)
            if first["feature_value"] == second["feature_value"]:
                raise _EvidenceError(f"{field}: paired feature values are not distinct")
            binding = (first["source_sha256"], first["evidence_sha256"])
            other_binding = (second["source_sha256"], second["evidence_sha256"])
            if binding in seen_arms or other_binding in seen_arms:
                raise _EvidenceError(f"{field}: a controlled point is reused", integrity=True)
            seen_arms.update((binding, other_binding))
            delta_feature = second["feature_value"] - first["feature_value"]
            slopes = {
                measurement["coefficient"]: (
                    second["measurements"][measurement["coefficient"]]
                    - first["measurements"][measurement["coefficient"]]
                )
                / delta_feature
                for measurement in measurements
            }
            if any(value < 0 for value in slopes.values()):
                raise _EvidenceError(f"{field}: paired observations do not establish non-negative coefficients")
            files.append(
                {"path": str(controls_path), "sha256": controls_sha, "purpose": "paired controlled-variable contract"}
            )
            files.extend((*first_files, *second_files))
            pairs.append(
                {
                    "id": ident,
                    "controls_sha256": controls_sha,
                    "first_source_sha256": first["source_sha256"],
                    "first_evidence_sha256": first["evidence_sha256"],
                    "second_source_sha256": second["source_sha256"],
                    "second_evidence_sha256": second["evidence_sha256"],
                    "feature_values": [float(first["feature_value"]), float(second["feature_value"])],
                    "slopes": {name: float(value) for name, value in sorted(slopes.items())},
                }
            )
        except _EvidenceError as exc:
            problems.append(_problem(exc, field))

    n_parameters = len(measurements)
    n_points = 2 * len(pairs)
    if n_points < 2 * n_parameters:
        problems.append(
            {
                "field": "controlled_pairs",
                "severity": "missing",
                "reason": (
                    f"{n_points} valid paired points cannot fit {n_parameters} parameters; "
                    "at least two distinct points per fitted parameter are required"
                ),
            }
        )
    if problems or n_points < 2 * n_parameters:
        return None, pairs, files, problems

    slope_sets = {
        measurement["coefficient"]: [Fraction(str(pair["slopes"][measurement["coefficient"]])) for pair in pairs]
        for measurement in measurements
    }
    derived = {
        "id": feature["id"],
        "pointer": feature["pointer"],
        "unit": feature["unit"],
        "resource": feature["resource"],
        "kind": feature["kind"],
        "effects": feature["effects"],
    }
    if feature["kind"] == "compute":
        values = slope_sets["cycles_per_unit"]
        derived["cycles_per_unit"] = {"lo": float(min(values)), "hi": float(max(values))}
    else:
        for name in ("physical_bytes_per_unit", "commands_per_unit"):
            values = slope_sets[name]
            if len(set(values)) != 1:
                problems.append(
                    {
                        "field": f"coefficients.{name}",
                        "severity": "missing",
                        "reason": (
                            f"paired evidence does not establish one exact {name}; observed controlled slopes disagree"
                        ),
                    }
                )
        command = slope_sets["commands_per_unit"][0]
        if command.denominator != 1:
            problems.append(
                {
                    "field": "coefficients.commands_per_unit",
                    "severity": "missing",
                    "reason": "paired evidence does not establish an integral commands-per-unit value",
                }
            )
        if problems:
            return None, pairs, files, problems
        derived["physical_bytes_per_unit"] = float(slope_sets["physical_bytes_per_unit"][0])
        derived["commands_per_unit"] = int(command)
    return derived, pairs, files, problems


def _prepare(document: Mapping[str, Any], *, source_sha256: str, base: Path, expected_schema: str) -> dict[str, Any]:
    problems: list[dict[str, str]] = []
    files: list[dict[str, str]] = []
    target_sha = ""
    try:
        if document.get("schema") != expected_schema:
            raise _EvidenceError(f"expected schema {expected_schema}", integrity=True)
        target_path, target_sha = _verified_file(document.get("target_descriptor"), "target_descriptor", base)
        files.append({"path": str(target_path), "sha256": target_sha, "purpose": "exact target descriptor"})
        if expected_schema == FEATURE_SCHEMA and document.get("target_sha256") != target_sha:
            raise _EvidenceError(
                "feature receipt target_sha256 does not bind the exact target descriptor", integrity=True
            )
        feature = _feature_spec(document.get("feature"))
        derivation = _mapping(document.get("derivation")) if expected_schema == FEATURE_SCHEMA else document
        measurements = _measurements(feature["kind"], derivation.get("measurements"))
    except _EvidenceError as exc:
        problems.append(_problem(exc, "contract"))
        feature = {}
        derivation = {}
        measurements = []

    if expected_schema == REQUEST_SCHEMA:
        for index, reference in enumerate(_sequence(document.get("candidate_evidence_files"))):
            try:
                path, digest = _verified_file(reference, f"candidate_evidence_files[{index}]", base)
                files.append(
                    {
                        "path": str(path),
                        "sha256": digest,
                        "purpose": "candidate evidence without a controlled-point contract",
                    }
                )
            except _EvidenceError as exc:
                problems.append(_problem(exc, f"candidate_evidence_files[{index}]"))

    derived = None
    pairs: list[dict[str, Any]] = []
    if target_sha and feature and measurements:
        derived, pairs, evidence_files, derivation_problems = _derive(
            derivation, base=base, target_sha256=target_sha, feature=feature, measurements=measurements
        )
        files.extend(evidence_files)
        problems.extend(derivation_problems)
    if problems:
        derived = None

    if derived is not None and expected_schema == FEATURE_SCHEMA:
        claimed = dict(_mapping(document.get("feature")))
        claimed.pop("evidence_pointer", None)
        if claimed != derived:
            problems.append(
                {
                    "field": "feature",
                    "severity": "refusal",
                    "reason": "claimed coefficient does not equal the paired-evidence derivation",
                }
            )
            derived = None
        counts = _mapping(document.get("derivation"))
        if counts.get("n_fitted_parameters") != len(measurements) or counts.get("n_distinct_points") != 2 * len(pairs):
            problems.append(
                {
                    "field": "derivation",
                    "severity": "refusal",
                    "reason": "claimed fit counts do not equal the revalidated paired evidence",
                }
            )
            derived = None
        if counts.get("pair_derivations") != pairs:
            problems.append(
                {
                    "field": "derivation.pair_derivations",
                    "severity": "refusal",
                    "reason": "claimed pair derivations do not equal the revalidated observations",
                }
            )
            derived = None

    unique_files: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in files:
        key = (row["path"], row["sha256"])
        if key not in seen:
            seen.add(key)
            unique_files.append(row)
    if expected_schema == FEATURE_SCHEMA:
        declared_sources = list(_sequence(document.get("source_files")))
        declared_evidence = list(_sequence(document.get("evidence_files")))
        actual_sources = [row for row in unique_files if row["purpose"] == "controlled source"]
        actual_evidence = [row for row in unique_files if row["purpose"] != "controlled source"]
        if declared_sources != actual_sources or declared_evidence != actual_evidence:
            problems.append(
                {
                    "field": "evidence_files",
                    "severity": "refusal",
                    "reason": "declared source/evidence inventory does not equal the revalidated files",
                }
            )
            derived = None
    refusals = [problem for problem in problems if problem["severity"] == "refusal"]
    missing = [problem for problem in problems if problem["severity"] == "missing"]
    status = "refused" if refusals else ("ready" if derived is not None else "incomplete")
    result = {
        "status": status,
        "execution": "host_evidence_join_only",
        "target_execution_performed": False,
        "simulator_execution_performed": False,
        "full_model_execution_performed": False,
        "source_sha256": source_sha256,
        "target_sha256": target_sha or None,
        "feature": derived,
        "controlled_pairs": pairs,
        "evidence_files": sorted(unique_files, key=lambda row: (row["purpose"], row["path"])),
        "missing": missing,
        "refusals": refusals,
    }
    if derived is not None and expected_schema == REQUEST_SCHEMA:
        derivation = {
            "method": "controlled_pair_difference_v1",
            "preparation_request_sha256": source_sha256,
            "feature_evidence_pointer": feature["evidence_pointer"],
            "measurements": measurements,
            "n_fitted_parameters": len(measurements),
            "n_distinct_points": 2 * len(pairs),
            "controlled_pairs": list(_sequence(document.get("controlled_pairs"))),
            "pair_derivations": pairs,
        }
        receipt_feature = dict(derived)
        receipt_feature["evidence_pointer"] = feature["evidence_pointer"]
        result["calibration"] = {
            "schema": FEATURE_SCHEMA,
            "status": "derived",
            "target_descriptor": dict(_mapping(document.get("target_descriptor"))),
            "target_sha256": target_sha,
            "source_files": [row for row in unique_files if row["purpose"] == "controlled source"],
            "evidence_files": [
                row
                for row in unique_files
                if row["purpose"]
                not in {
                    "controlled source",
                    "candidate evidence without a controlled-point contract",
                }
            ],
            "derivation": derivation,
            "feature": receipt_feature,
        }
        result["calibration_sha256"] = _digest(result["calibration"])
    else:
        result["calibration"] = None
    return result


def prepare_feature_calibration(request_source: Mapping[str, Any] | Path) -> dict[str, Any]:
    """Derive one strict feature receipt, or explain why the evidence is incomplete/refused."""
    try:
        request, request_sha, base = _load_json(request_source)
    except _EvidenceError as exc:
        result = {
            "schema": PREPARATION_SCHEMA,
            "status": "refused",
            "calibration": None,
            "missing": [],
            "refusals": [_problem(exc, "request")],
            "target_execution_performed": False,
            "simulator_execution_performed": False,
            "full_model_execution_performed": False,
        }
        result["receipt_sha256"] = _digest(result)
        return result
    result = _prepare(request, source_sha256=request_sha, base=base, expected_schema=REQUEST_SCHEMA)
    result["schema"] = PREPARATION_SCHEMA
    result["request_sha256"] = request_sha
    result["receipt_sha256"] = _digest(result)
    return result


def validate_feature_calibration(
    receipt_source: Mapping[str, Any] | Path, *, expected_target_sha256: str | None = None
) -> dict[str, Any]:
    """Re-read every bound byte and recompute a claimed feature coefficient."""
    try:
        receipt, receipt_sha, base = _load_json(receipt_source)
    except _EvidenceError as exc:
        result = {
            "schema": VALIDATION_SCHEMA,
            "status": "refused",
            "feature": None,
            "missing": [],
            "refusals": [_problem(exc, "receipt")],
        }
        result["receipt_sha256"] = _digest(result)
        return result
    result = _prepare(receipt, source_sha256=receipt_sha, base=base, expected_schema=FEATURE_SCHEMA)
    if (
        expected_target_sha256 is not None
        and result.get("target_sha256") is not None
        and result["target_sha256"] != expected_target_sha256
    ):
        result["refusals"].append(
            {
                "field": "target_sha256",
                "severity": "refusal",
                "reason": "feature receipt does not bind the calibration bundle target",
            }
        )
        result["status"] = "refused"
        result["feature"] = None
    result["schema"] = VALIDATION_SCHEMA
    result["receipt_file_sha256"] = receipt_sha
    result["receipt_sha256"] = _digest(result)
    return result


def write_json(document: Mapping[str, Any], output: Path) -> Path:
    path = Path(output)
    if path.is_symlink():
        raise ValueError(f"refusing to overwrite symlink {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(document, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path)
    args = parser.parse_args(argv)
    result = prepare_feature_calibration(args.request)
    write_json(result, args.output)
    if args.calibration_output is not None and result.get("calibration") is not None:
        write_json(result["calibration"], args.calibration_output)
    print(
        json.dumps(
            {"status": result["status"], "output": str(args.output), "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTROLS_SCHEMA",
    "FEATURE_SCHEMA",
    "OBSERVATION_SCHEMA",
    "PREPARATION_SCHEMA",
    "REQUEST_SCHEMA",
    "VALIDATION_SCHEMA",
    "prepare_feature_calibration",
    "validate_feature_calibration",
    "write_json",
]

"""Prepare a content-addressed calibration for the Phase-2 host evaluator.

This module only joins evidence that already exists.  It never runs a target and it never estimates
a coefficient from an uncontrolled workload corpus.  Target-owned names (oracle engines, counter
roles, proof modules, and emitted-feature selectors) arrive through one adapter descriptor.

The output is either a ``phase2_host_analytical_calibration_v1`` accepted by
``phase2_analytical_provider``, or a receipt whose ``missing``/``refusals`` fields say exactly why
one could not be built.  In particular, a trusted occupancy partition and a movement slope do not
establish cycles per emitted compute instruction, physical bytes per emitted movement instruction,
or the cost of an encoding transition.  Those terms require separate, content-addressed feature
calibration receipts and remain UNKNOWN when no such receipts are supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.perf import attribution as attribution_lib
from merlin.perf import counter_harvest
from merlin.perf import headroom
from merlin.perf import hw_counters
from merlin.perf import movement_balance
from merlin.perf.decompose import ResourceKind, Unavailable

ADAPTER_SCHEMA = "phase2_host_analytical_calibration_adapter_v1"
FEATURE_SCHEMA = "phase2_analytical_feature_calibration_v1"
PREPARATION_SCHEMA = "phase2_host_analytical_calibration_preparation_v1"
CALIBRATION_SCHEMA = "phase2_host_analytical_calibration_v1"


class _EvidenceError(ValueError):
    def __init__(self, message: str, *, integrity: bool = False):
        super().__init__(message)
        self.integrity = integrity


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _EvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise _EvidenceError(f"{label} must be finite and non-negative")
    return result


def _load_json(source: Mapping[str, Any] | Path) -> tuple[dict[str, Any], str, str]:
    if isinstance(source, Mapping):
        document = dict(source)
        return document, _digest(document), "canonical_mapping"
    path = Path(source)
    if path.is_symlink():
        raise _EvidenceError(f"{path}: descriptor is a symlink", integrity=True)
    if not path.is_file():
        raise _EvidenceError(f"{path}: descriptor is unavailable")
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise _EvidenceError(f"{path}: invalid JSON: {exc}", integrity=True) from exc
    if not isinstance(document, Mapping):
        raise _EvidenceError(f"{path}: descriptor must contain an object", integrity=True)
    return dict(document), hashlib.sha256(raw).hexdigest(), "exact_file_bytes"


def _verified_file(reference: Any, label: str) -> tuple[Path, str]:
    row = _mapping(reference)
    path = Path(str(row.get("path") or ""))
    expected = row.get("sha256")
    if not _is_sha256(expected):
        raise _EvidenceError(f"{label}: expected sha256 is absent or invalid", integrity=True)
    if path.is_symlink():
        raise _EvidenceError(f"{label}: {path} is a symlink", integrity=True)
    if not path.is_file():
        raise _EvidenceError(f"{label}: {path} is unavailable")
    actual = _sha256_bytes(path)
    if actual != expected:
        raise _EvidenceError(
            f"{label}: sha256 mismatch (declared {expected}, actual {actual})", integrity=True)
    return path, actual


def _evidence_row(path: Path, digest: str, purpose: str) -> dict[str, str]:
    return {"path": str(path), "sha256": digest, "purpose": purpose}


def _record_problem(problems: list[dict[str, str]], exc: Exception, field: str) -> None:
    problems.append({
        "field": field,
        "severity": "refusal" if isinstance(exc, _EvidenceError) and exc.integrity else "missing",
        "reason": str(exc),
    })


def _counter_evidence(adapter: Mapping[str, Any]) -> tuple[
        dict[str, Any], dict[str, Any] | None, list[dict[str, str]], list[dict[str, str]]]:
    spec = _mapping(adapter.get("counter_harvest"))
    evidence: list[dict[str, str]] = []
    problems: list[dict[str, str]] = []
    if not spec:
        problem = {"field": "counter_harvest", "severity": "missing",
                   "reason": "no trusted-counter evidence descriptor was supplied"}
        return {"status": "unavailable", "reason": problem["reason"]}, None, evidence, [problem]

    try:
        header_path, header_sha = _verified_file(spec.get("counter_header"), "counter header")
        proof_path, proof_sha = _verified_file(
            _mapping(spec.get("partition_proof")).get("artifact"),
            "counter partition proof artifact")
    except _EvidenceError as exc:
        _record_problem(problems, exc, "counter_harvest")
        return {"status": "unavailable", "reason": str(exc)}, None, evidence, problems
    evidence.extend((
        _evidence_row(header_path, header_sha, "target-declared counter header"),
        _evidence_row(proof_path, proof_sha, "counter partition proof artifact"),
    ))

    engines = tuple(str(value) for value in _sequence(spec.get("declared_engines")) if str(value))
    kinds = {str(name): str(kind) for name, kind in _mapping(spec.get("resource_kinds")).items()}
    proof = _mapping(spec.get("partition_proof"))
    root_value = str(spec.get("runs_root") or "")
    root = Path(root_value) if root_value else None
    if root is None or not root.is_dir():
        problem = {"field": "counter_harvest.runs_root", "severity": "missing",
                   "reason": f"counter run root {root} is unavailable"}
        return {"status": "unavailable", "reason": problem["reason"]}, None, evidence, [problem]
    if not engines:
        problem = {"field": "counter_harvest.declared_engines", "severity": "refusal",
                   "reason": "the target adapter declares no oracle-engine vocabulary"}
        return {"status": "refused", "reason": problem["reason"]}, None, evidence, [problem]

    header_text = header_path.read_text(encoding="utf-8", errors="replace")
    counters = hw_counters.derive_occupancy_counters(header_text)
    known_kinds = {kind.value for kind in ResourceKind}
    if (not counters.complete() or set(kinds) != set(counters.engines)
            or not set(kinds.values()) <= known_kinds):
        problem = {
            "field": "counter_harvest.resource_kinds", "severity": "refusal",
            "reason": ("counter header does not derive a complete partition, or the target adapter "
                       "does not assign exactly one resource kind to every derived engine"),
        }
        return {"status": "refused", "reason": problem["reason"]}, None, evidence, [problem]

    harvested = counter_harvest.harvest_counter_runs(root, engines=engines)
    seen_paths: set[Path] = set()
    for run in harvested.runs:
        seen_paths.add(run.console)
    for refusal in harvested.refusals:
        path = Path(str(refusal.get("console") or ""))
        if path.is_file():
            seen_paths.add(path)
    for path in sorted(seen_paths):
        evidence.append(_evidence_row(path, _sha256_bytes(path), "harvested counter console"))

    valid_runs = []
    sources = []
    overlap_by_workload: dict[str, int] = {}
    validation_refusals: list[dict[str, str]] = []
    codes = hw_counters.event_codes(header_text)
    trusted_counts: dict[str, int] = {}
    for run in harvested.trusted():
        trusted_counts[run.workload] = trusted_counts.get(run.workload, 0) + 1
    proof_text = proof_path.read_text(encoding="utf-8", errors="replace")
    for run in harvested.trusted():
        if trusted_counts[run.workload] != 1:
            validation_refusals.append({
                "console": str(run.console),
                "reason": (f"{trusted_counts[run.workload]} trusted consoles share workload "
                           f"identity {run.workload!r}; composition keys would be ambiguous"),
            })
            continue
        console_text = run.console.read_text(encoding="utf-8", errors="replace")
        if hw_counters.parse_counter_schema(console_text) != header_sha:
            validation_refusals.append({
                "console": str(run.console),
                "reason": "measured counter-schema digest does not match the bound target header",
            })
            continue
        observed = hw_counters.eta_from_counters(
            dict(run.readings), counters, hw_text=proof_text, codes=codes,
            module=str(proof.get("module") or ""),
            counter_module=str(proof.get("counter_module") or ""),
            measurement_cycles=run.total_cycles, source=str(proof_path))
        partition = _mapping(observed.get("partition_proof"))
        if observed.get("state") != "measured" or partition.get("status") != "proved":
            validation_refusals.append({
                "console": str(run.console),
                "reason": str(observed.get("why") or "the occupancy partition was not proved"),
            })
            continue
        try:
            source = attribution_lib.activity_from_counter_readings(
                run.readings, workload=run.workload, total_cycles=run.total_cycles,
                header_text=header_text, kind_of=kinds,
                provenance=f"console sha256:{_sha256_bytes(run.console)}")
        except ValueError as exc:
            validation_refusals.append({"console": str(run.console), "reason": str(exc)})
            continue
        across_kinds = sum(
            int(run.readings[name]) for combination, name in counters.by_combination.items()
            if len({kinds[engine] for engine in combination}) >= 2)
        sources.append(source)
        overlap_by_workload[run.workload] = across_kinds
        valid_runs.append({**run.to_dict(), "console_sha256": _sha256_bytes(run.console),
                           "partition_proof": partition,
                           "overlap_cycles_across_kinds": across_kinds})

    composition = headroom.composition_operator(
        sources, observed_overlap_cycles=overlap_by_workload) if sources else Unavailable(
            "composition operator", ("at least one proved trusted counter run",))
    composition_row = None
    if isinstance(composition, Unavailable):
        problems.append({"field": "composition", "severity": "missing", "reason": str(composition)})
    else:
        operator, eta = composition
        composition_row = {"operator": operator.value, "eta": eta}

    block: dict[str, Any] = {
        "schema": "phase2_trusted_counter_preparation_v1",
        "status": "derived" if valid_runs else "unavailable",
        "counter_header_sha256": header_sha,
        "partition_artifact_sha256": proof_sha,
        "harvest": harvested.to_dict(),
        "validated_runs": valid_runs,
        "validation_refusals": validation_refusals,
        "composition": composition_row,
    }
    receipt_sha = _digest(block)
    block["receipt_sha256"] = receipt_sha
    if not valid_runs:
        problems.append({"field": "counter_harvest.validated_runs", "severity": "missing",
                         "reason": "no trusted counter console passed schema and partition proof"})
    return block, composition_row, evidence, problems


def _movement_evidence(adapter: Mapping[str, Any]) -> tuple[
        dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    spec = _mapping(adapter.get("movement_series"))
    evidence: list[dict[str, str]] = []
    problems: list[dict[str, str]] = []
    if not spec:
        problem = {"field": "movement_series", "severity": "missing",
                   "reason": "no controlled movement series descriptor was supplied"}
        return {"status": "unavailable", "reason": problem["reason"]}, evidence, [problem]
    try:
        package_receipt, package_sha = _verified_file(
            spec.get("package_receipt"), "compiler-package identity receipt")
    except _EvidenceError as exc:
        _record_problem(problems, exc, "movement_series.package_receipt")
        return {"status": "unavailable", "reason": str(exc)}, evidence, problems
    evidence.append(_evidence_row(
        package_receipt, package_sha, "compiler-package identity for movement series"))

    root_value = str(spec.get("runs_root") or "")
    engine = str(spec.get("engine") or "")
    package_value = str(spec.get("package_path") or "")
    root = Path(root_value) if root_value else None
    package_path = Path(package_value) if package_value else None
    if (root is None or not root.is_dir() or not engine or package_path is None
            or package_path.is_symlink() or not package_path.is_dir()):
        problem = {"field": "movement_series", "severity": "missing",
                   "reason": "movement run root, engine, or compiler-package directory is unavailable"}
        return {"status": "unavailable", "reason": problem["reason"]}, evidence, [problem]

    if package_receipt.parent.resolve() != package_path.resolve():
        problem = {"field": "movement_series.package_receipt", "severity": "refusal",
                   "reason": "compiler-package identity receipt is outside the declared package"}
        return {"status": "refused", "reason": problem["reason"]}, evidence, [problem]

    samples, refusals = movement_balance.samples_from_capsule_runs(
        root, engine=engine, package=package_sha)
    accepted = []
    for sample in samples:
        matches = [path for path in root.rglob("capsule_result.json")
                   if path.parent.name == sample.program]
        if len(matches) != 1:
            refusals.append({"program": sample.program,
                             "reason": "the movement sample has no unique result receipt"})
            continue
        result_path = matches[0]
        command_path = result_path.parent / "generated" / "command_buffer.json"
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            refusals.append({"program": sample.program, "reason": str(exc)})
            continue
        tiers = [row for row in _mapping(result.get("tiers")).values()
                 if isinstance(row, Mapping) and row.get("engine") == engine
                 and row.get("derived_from_rtl") is True and row.get("cycles") == sample.cycles]
        declared_packages = {
            str(row.get("toolchain") or _mapping(row.get("submission")).get("package") or "")
            for row in tiers
        }
        if len(tiers) != 1 or declared_packages != {str(package_path)}:
            refusals.append({
                "program": sample.program,
                "reason": "sample is not bound to the declared cycle-accurate engine/package",
            })
            continue
        accepted.append(sample)
        evidence.extend((
            _evidence_row(result_path, _sha256_bytes(result_path), "movement sample result"),
            _evidence_row(command_path, _sha256_bytes(command_path),
                          "movement sample emitted command buffer"),
        ))

    fitted = movement_balance.fit(accepted, engine=engine, package=package_sha)
    block = {"schema": "phase2_controlled_movement_preparation_v1",
             "status": fitted.status, "package_receipt_sha256": package_sha,
             "fit": fitted.to_dict(), "refusals": refusals}
    block["receipt_sha256"] = _digest(block)
    if not fitted.derived:
        problems.append({"field": "movement_balance", "severity": "missing",
                         "reason": fitted.reason})
    return block, evidence, problems


def _feature_evidence(adapter: Mapping[str, Any], target_sha: str) -> tuple[
        list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    features: list[dict[str, Any]] = []
    evidence: list[dict[str, str]] = []
    problems: list[dict[str, str]] = []
    for index, reference in enumerate(_sequence(adapter.get("feature_calibration_receipts"))):
        field = f"feature_calibration_receipts[{index}]"
        try:
            path, receipt_sha = _verified_file(reference, field)
            document, _unused, _source = _load_json(path)
            if (document.get("schema") != FEATURE_SCHEMA or document.get("status") != "derived"
                    or document.get("target_sha256") != target_sha):
                raise _EvidenceError(
                    f"{field}: receipt is not a derived calibration for the bound target",
                    integrity=True)
            source_hashes = []
            for source_index, source_ref in enumerate(_sequence(document.get("source_files"))):
                source_path, source_sha = _verified_file(
                    source_ref, f"{field}.source_files[{source_index}]")
                source_hashes.append(source_sha)
                evidence.append(_evidence_row(
                    source_path, source_sha, "raw analytical-feature calibration evidence"))
            derivation = _mapping(document.get("derivation"))
            if set(_sequence(derivation.get("evidence_sha256s"))) != set(source_hashes):
                raise _EvidenceError(
                    f"{field}: derivation is not bound to exactly its raw evidence files",
                    integrity=True)
            feature = dict(_mapping(document.get("feature")))
            if feature.get("cycles_per_unit") is not None:
                parameters = int(_number(
                    derivation.get("n_fitted_parameters"), f"{field} fitted parameters"))
                points = int(_number(
                    derivation.get("n_distinct_points"), f"{field} distinct points"))
                if parameters <= 0 or points < 2 * parameters:
                    raise _EvidenceError(
                        f"{field}: cycle fit needs at least two points per fitted parameter",
                        integrity=True)
            feature["provenance_sha256"] = receipt_sha
            features.append(feature)
            evidence.append(_evidence_row(path, receipt_sha, "analytical-feature calibration receipt"))
        except _EvidenceError as exc:
            _record_problem(problems, exc, field)

    present = {str(row.get("kind") or "") for row in features}
    for kind in ("compute", "movement", "encoding"):
        if kind not in present:
            problems.append({
                "field": f"features.{kind}", "severity": "missing",
                "reason": (f"no content-addressed {kind} feature calibration was supplied; "
                           "counter activity and movement slope do not establish this coefficient"),
            })
    return features, evidence, problems


def prepare_phase2_calibration(
        adapter_source: Mapping[str, Any] | Path) -> dict[str, Any]:
    """Build a ready calibration or an explicit incomplete/refusal receipt without target execution."""
    try:
        adapter, adapter_sha, adapter_source_kind = _load_json(adapter_source)
    except _EvidenceError as exc:
        return {"schema": PREPARATION_SCHEMA, "status": "refused", "calibration": None,
                "missing": [], "refusals": [{"field": "adapter", "reason": str(exc)}]}
    if adapter.get("schema") != ADAPTER_SCHEMA:
        return {"schema": PREPARATION_SCHEMA, "status": "refused",
                "adapter_sha256": adapter_sha, "calibration": None, "missing": [],
                "refusals": [{"field": "adapter.schema",
                              "reason": f"expected {ADAPTER_SCHEMA}"}]}

    evidence: list[dict[str, str]] = []
    problems: list[dict[str, str]] = []
    target_sha = ""
    try:
        target_path, target_sha = _verified_file(
            adapter.get("target_descriptor"), "target descriptor")
        evidence.append(_evidence_row(target_path, target_sha, "exact target descriptor"))
    except _EvidenceError as exc:
        _record_problem(problems, exc, "target_descriptor")

    counter, composition, rows, counter_problems = _counter_evidence(adapter)
    evidence.extend(rows)
    problems.extend(counter_problems)
    movement, rows, movement_problems = _movement_evidence(adapter)
    evidence.extend(rows)
    problems.extend(movement_problems)
    features, rows, feature_problems = _feature_evidence(adapter, target_sha)
    evidence.extend(rows)
    problems.extend(feature_problems)

    roles = tuple(sorted({str(value) for value in _sequence(
        adapter.get("accelerator_compute_roles")) if str(value)}))
    if not roles:
        problems.append({"field": "accelerator_compute_roles", "severity": "missing",
                         "reason": "target descriptor declares no emitted compute roles"})
    try:
        risk = _number(adapter.get("risk_score"), "risk score")
        if risk > 1:
            raise _EvidenceError("risk score must be in [0, 1]")
    except _EvidenceError as exc:
        risk = None
        _record_problem(problems, exc, "risk_score")

    # Bind derived receipts as evidence too.  Their bodies are embedded below, so their content
    # address is independently recomputable without access to the original machine paths.
    derived_hashes = [str(counter.get("receipt_sha256") or ""),
                      str(movement.get("receipt_sha256") or "")]
    file_rows = []
    seen_files: set[tuple[str, str]] = set()
    for row in evidence:
        key = (row["path"], row["sha256"])
        if key not in seen_files:
            seen_files.add(key)
            file_rows.append(row)

    calibration = None
    missing = [problem for problem in problems if problem["severity"] == "missing"]
    refusals = [problem for problem in problems if problem["severity"] == "refusal"]
    if (not missing and not refusals and target_sha and composition is not None
            and movement.get("status") == "derived" and risk is not None):
        counter_receipt = str(counter["receipt_sha256"])
        movement_receipt = str(movement["receipt_sha256"])
        evidence_hashes = tuple(dict.fromkeys((
            adapter_sha, *(row["sha256"] for row in file_rows),
            counter_receipt, movement_receipt,
            *(str(feature["provenance_sha256"]) for feature in features),
        )))
        movement_row = dict(_mapping(movement.get("fit")))
        movement_row["provenance_sha256"] = movement_receipt
        calibration = {
            "schema": CALIBRATION_SCHEMA,
            "target_sha256": target_sha,
            "evidence_sha256s": list(evidence_hashes),
            "composition": {**composition, "provenance_sha256": counter_receipt},
            "movement_balance": movement_row,
            "accelerator_compute_roles": list(roles),
            "risk_score": risk,
            "features": features,
        }

    status = "refused" if refusals else ("ready" if calibration is not None else "incomplete")
    result = {
        "schema": PREPARATION_SCHEMA,
        "status": status,
        "execution": "host_evidence_join_only",
        "target_execution_performed": False,
        "full_model_simulation_performed": False,
        "adapter_sha256": adapter_sha,
        "adapter_source": adapter_source_kind,
        "target_sha256": target_sha or None,
        "evidence_files": sorted(file_rows, key=lambda row: (row["purpose"], row["path"])),
        "derived_receipt_sha256s": [value for value in derived_hashes if _is_sha256(value)],
        "counter_evidence": counter,
        "movement_evidence": movement,
        "feature_calibrations": features,
        "missing": missing,
        "refusals": refusals,
        "calibration": calibration,
    }
    result["receipt_sha256"] = _digest(result)
    return result


def write_preparation(receipt: Mapping[str, Any], output: Path) -> Path:
    """Write one canonical receipt.  Callers choose the generated-artifact location."""
    path = Path(output)
    if path.is_symlink():
        raise ValueError(f"refusing to overwrite symlink {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = prepare_phase2_calibration(args.adapter)
    write_preparation(receipt, args.output)
    print(json.dumps({"status": receipt["status"],
                      "receipt_sha256": receipt.get("receipt_sha256"),
                      "output": str(args.output),
                      "missing": len(receipt.get("missing", ())),
                      "refusals": len(receipt.get("refusals", ()))}, sort_keys=True))
    return 0 if receipt["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADAPTER_SCHEMA", "CALIBRATION_SCHEMA", "FEATURE_SCHEMA", "PREPARATION_SCHEMA",
    "prepare_phase2_calibration", "write_preparation",
]

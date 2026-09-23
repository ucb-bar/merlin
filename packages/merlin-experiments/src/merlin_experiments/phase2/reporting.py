"""Current GSIM report admission from parent-pinned evidence and verified handoffs.

This host-owned module does not launch engines or qualify candidate submissions.
Native callers supply handoffs from the full candidate verifier. Historical Verilator
report policy remains separate; neither its schemas nor scientific claims are inferred.
"""

from __future__ import annotations

import json
import stat
from collections.abc import Mapping
from pathlib import Path

from merlin.common import digest
from merlin_experiments.phase2 import measurement_evidence as ME
from merlin_experiments.phase2 import statistics


class ReportingGateError(RuntimeError):
    """A report would overstate, mix, or lose attribution for the measured campaign."""


def require_read_only(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.exists():
        raise ReportingGateError(f"{label} is absent or linked: {path}")
    if stat.S_IMODE(path.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ReportingGateError(f"{label} is writable: {path}")


def read_immutable_json(
    path: Path, expected_sha256: object, *, label: str, expected_n_bytes: object | None = None
) -> object:
    if not digest.is_sha256(expected_sha256):
        raise ReportingGateError(f"{label} has no valid lowercase SHA-256")
    require_read_only(path, label=label)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ReportingGateError(f"{label} is not an ordinary file: {path}")
    payload = path.read_bytes()
    if expected_n_bytes is not None:
        if (
            isinstance(expected_n_bytes, bool)
            or not isinstance(expected_n_bytes, int)
            or expected_n_bytes <= 0
            or len(payload) != expected_n_bytes
        ):
            raise ReportingGateError(f"{label} byte count does not match its digest record")
    observed = digest.sha256_bytes(payload)
    if observed != expected_sha256:
        raise ReportingGateError(f"{label} digest mismatch: observed {observed}, expected {expected_sha256}")
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ReportingGateError(f"{label} is not valid JSON: {path}") from exc


def load_current_experiment(
    parent_path: Path, *, expected_sha256: str, handoffs: Mapping[str, object]
) -> tuple[dict, list[dict], dict]:
    """Report current paired GSIM evidence through its existing parent hash commitments.

    The native caller obtains handoffs through the real candidate verifier. This reader
    never constructs a candidate qualification or infers trust from a mutable child.
    Historical Verilator campaigns retain their separate reader and scientific policy.
    """
    parent = read_immutable_json(parent_path, expected_sha256, label="parent experiment manifest")
    if (
        not isinstance(parent, Mapping)
        or parent.get("schema") != "merlin.agentic-performance-experiment.v1"
        or parent.get("status") != "GO"
        or parent.get("selection") != "all_three_trials_all_predeclared_cells_no_best_of_no_drop"
    ):
        raise ReportingGateError("unsupported or incomplete current experiment manifest")
    declaration = parent.get("declaration") or {}
    trials = declaration.get("trials") or []
    phases = declaration.get("measurement_phases") or []
    if (
        not isinstance(trials, list)
        or not trials
        or len(set(trials)) != len(trials)
        or set(handoffs) != set(trials)
        or phases != ["tuning", "held_out"]
    ):
        raise ReportingGateError("current experiment has incomplete trial/phase handoffs")
    evidence = parent.get("trials") or []
    if not isinstance(evidence, list) or len(evidence) != len(trials):
        raise ReportingGateError("current experiment lacks exact candidate evidence")
    evidence_by_trial = {row.get("trial"): row for row in evidence if isinstance(row, Mapping)}
    if set(evidence_by_trial) != set(trials):
        raise ReportingGateError("current experiment repeats or omits candidate evidence")
    for trial in trials:
        handoff = handoffs[trial]
        if evidence_by_trial[trial].get("agent_evidence_sha256") != handoff.record_sha256 or dict(
            handoff.agent_contract
        ) != (declaration.get("trial_contracts") or {}).get(trial):
            raise ReportingGateError("current experiment candidate differs from its declared trial")
    receipts = parent.get("measurement_manifests") or []
    expected_order = [(trial, phase) for trial in trials for phase in phases]
    if not isinstance(receipts, list) or len(receipts) != len(expected_order):
        raise ReportingGateError("current experiment omits declared measurement children")
    rows = []
    seen_paths = set()
    for receipt, (trial, phase) in zip(receipts, expected_order, strict=True):
        if not isinstance(receipt, Mapping) or receipt.get("trial") != trial:
            raise ReportingGateError("current measurement child order differs from its declaration")
        child_path = Path(str(receipt.get("path") or ""))
        if not child_path.is_absolute() or child_path.resolve() in seen_paths:
            raise ReportingGateError("current measurement child path is absent, relative or duplicated")
        seen_paths.add(child_path.resolve())
        handoff = handoffs[trial]
        if phase == "tuning":
            corpus_manifest = handoff.corpus_manifest_sha256
            corpus_bytes = handoff.corpus_sha256
            certificate = declaration.get("gsim_certificate_sha256")
        else:
            holdout = parent.get("holdout") or {}
            corpus_manifest = holdout.get("manifest_sha256")
            corpus_bytes = holdout.get("capsules_sha256")
            certificate = (parent.get("heldout_gsim_certificate") or {}).get("sha256")
        binding = ME.MeasurementBinding(
            phase,
            handoff.functional_run_id,
            handoff.functional_submission_sha256,
            handoff.record_sha256,
            handoff.candidate_sha256,
            corpus_manifest,
            corpus_bytes,
            certificate,
        )
        if not digest.is_sha256(receipt.get("sha256")):
            raise ReportingGateError("current measurement child lacks its parent hash pin")
        try:
            verified = ME.verify_paired_measurement(child_path, expected=binding, manifest_sha256=receipt["sha256"])
        except (ValueError, RuntimeError, OSError) as exc:
            raise ReportingGateError(f"current paired measurement is invalid: {exc}") from exc
        if verified.manifest.get("schema") != "paired_arm4_performance_campaign_v2":
            raise ReportingGateError("current experiment child has an unsupported paired schema")
        rows.extend(ME.statistics_rows(verified.rows, trial=trial))
    predeclared = read_immutable_json(
        parent_path.parent / "statistics_predeclaration.json",
        parent.get("statistics_predeclaration_sha256"),
        label="statistics predeclaration",
    )
    result = statistics.evaluate(predeclared, rows, trial_evidence=evidence)
    if result.get("status") != "admitted" or result != parent.get("statistics"):
        raise ReportingGateError("current experiment statistics differ from its exact sealed evidence")
    return dict(parent), rows, result

"""Paired GSIM measurement evidence shared by producers, adoption and reporting.

This module verifies existing receipts; it does not create a seal or discover runs.
Callers own candidate admission and pass explicit expected identities.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common.digest import is_sha256 as _is_sha256
from merlin_experiments.phase2 import campaign as PC

ARMS = ("baseline", "candidate")
REPLICATES = ("r000", "r001")
SIMULATORS = ("spike", "gsim")
PHASES = ("tuning", "held_out")


class MeasurementEvidenceError(ValueError):
    """Paired measurement evidence differs from the caller's declared inputs."""


@dataclass(frozen=True, order=True)
class ResultIdentity:
    phase: str
    arm: str
    family: str
    capsule: str
    simulator: str
    replicate: str

    @property
    def label(self) -> str:
        return "/".join((self.phase, self.arm, self.family, self.capsule, self.simulator, self.replicate))


def completion_report(results: Sequence[Mapping[str, Any]], expected: Sequence[ResultIdentity]) -> dict[str, Any]:
    wanted = tuple(expected)
    if not wanted or len(set(wanted)) != len(wanted):
        raise PC.CampaignGateError("expected identities are empty or duplicated")
    observed: dict[ResultIdentity, Mapping[str, Any]] = {}
    for row in results:
        identity = ResultIdentity(
            *(str(row.get(key) or "") for key in ("phase", "arm", "family", "capsule", "simulator", "replicate"))
        )
        if (
            identity.phase not in PHASES
            or identity.arm not in ARMS
            or identity.simulator not in SIMULATORS
            or identity.replicate not in REPLICATES
        ):
            raise PC.CampaignGateError(f"invalid result identity: {identity.label}")
        if identity in observed:
            raise PC.CampaignGateError(f"duplicate result: {identity.label}")
        observed[identity] = row
    extras = sorted(set(observed) - set(wanted))
    if extras:
        raise PC.CampaignGateError(f"unexpected results: {[item.label for item in extras]}")
    passed = failed = 0
    for identity in wanted:
        row = observed.get(identity)
        if row is None:
            continue
        if identity.simulator == "spike":
            valid = row.get("correct") is True and row.get("cycles") is None and not row.get("citable")
        else:
            provenance, cycles = row.get("provenance"), row.get("cycles")
            valid = (
                row.get("correct") is True
                and isinstance(cycles, int)
                and not isinstance(cycles, bool)
                and cycles > 0
                and row.get("citable") is True
                and isinstance(provenance, Mapping)
                and provenance.get("tier") == "L3"
                and provenance.get("simulator") == identity.simulator
                and provenance.get("oracle_kind") == f"rtl_{identity.simulator}"
                and provenance.get("derived_from_rtl") is True
                and provenance.get("cycle_accurate") is True
                and _is_sha256(provenance.get("elf_sha256"))
            )
            valid = (
                valid and isinstance(row.get("qualification"), Mapping) and row["qualification"].get("admitted") is True
            )
        passed += int(valid)
        failed += int(not valid)
    missing = len(wanted) - len(observed)
    return {
        "expected": len(wanted),
        "reported": len(observed),
        "passed": passed,
        "failed": failed,
        "missing": missing,
        "complete": missing == failed == 0 and passed == len(wanted),
    }


@dataclass(frozen=True)
class MeasurementBinding:
    phase: str
    functional_run_id: str
    functional_submission_sha256: str
    candidate_record_sha256: str
    candidate_sha256: str
    corpus_manifest_sha256: str
    corpus_capsules_sha256: str
    certificate_sha256: str


@dataclass(frozen=True)
class VerifiedPairedMeasurement:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    rows: tuple[dict[str, Any], ...]

    def receipt(self) -> dict[str, str]:
        return {"path": str(self.manifest_path), "sha256": self.manifest_sha256}


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _read_measurement(manifest_path: Path, expected_sha256: str | None = None, *, require_current_schema: bool = False):
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise MeasurementEvidenceError("paired measurement manifest is absent or linked")
    payload = manifest_path.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and (not _is_sha256(expected_sha256) or observed != expected_sha256):
        raise MeasurementEvidenceError("paired measurement manifest digest changed")
    manifest = json.loads(payload)
    if not isinstance(manifest, Mapping):
        raise MeasurementEvidenceError("paired measurement manifest must be a mapping")
    if require_current_schema and manifest.get("schema") != "paired_arm4_performance_campaign_v2":
        raise MeasurementEvidenceError(
            "unsupported paired campaign schema (expected paired_arm4_performance_campaign_v2)"
        )
    if manifest.get("status") != "GO":
        raise MeasurementEvidenceError(f"paired measurement did not reach GO: {manifest_path}")
    results = manifest.get("raw_results") or {}
    path = Path(str(results.get("paired_cells") or ""))
    if path.is_symlink() or not path.is_file():
        raise MeasurementEvidenceError("paired result cells are absent or changed")
    result_payload = path.read_bytes()
    if hashlib.sha256(result_payload).hexdigest() != results.get("paired_cells_sha256"):
        raise MeasurementEvidenceError("paired result cells are absent or changed")
    result = json.loads(result_payload)
    if not isinstance(result, Mapping):
        raise MeasurementEvidenceError("paired result document must be a mapping")
    if require_current_schema and result.get("schema") != "paired_arm4_result_cells_v2":
        raise MeasurementEvidenceError("unsupported paired result schema (expected paired_arm4_result_cells_v2)")
    cells = result.get("cells") or []
    if not isinstance(cells, list) or any(not isinstance(row, Mapping) for row in cells):
        raise MeasurementEvidenceError("paired result cells must be mappings")
    return manifest, cells, observed


def statistics_rows(results: Sequence[Mapping[str, Any]], *, trial: str) -> list[dict[str, Any]]:
    """Project every GSIM cell; this projection alone does not admit evidence."""
    rows = []
    for cell in results:
        if cell.get("simulator") != "gsim":
            continue
        provenance = cell.get("provenance") or {}
        rows.append(
            {
                "identity": {
                    "trial": trial,
                    "subject": cell.get("arm"),
                    "family": f"{cell.get('phase')}:{cell.get('family')}",
                    "capsule": cell.get("capsule"),
                    "simulator": "gsim",
                    "replicate": cell.get("replicate"),
                },
                "tier": provenance.get("tier"),
                "correct": cell.get("correct"),
                "cycle_accurate": provenance.get("cycle_accurate"),
                "cycles": cell.get("cycles"),
                "oracle": {
                    "kind": provenance.get("oracle_kind"),
                    "derived_from_rtl": provenance.get("derived_from_rtl"),
                },
            }
        )
    return rows


def read_statistics_rows(
    manifest_path: Path, *, trial: str, manifest_sha256: str | None = None
) -> list[dict[str, Any]]:
    """Rehash recorded result bytes before projection; callers separately admit the manifest."""
    _manifest, cells, _digest = _read_measurement(manifest_path, manifest_sha256)
    return statistics_rows(cells, trial=trial)


def verify_paired_measurement(
    manifest_path: Path, *, expected: MeasurementBinding, manifest_sha256: str | None = None
) -> VerifiedPairedMeasurement:
    """Recompute actual paired evidence against explicit trusted candidate/corpus bindings.

    Adoption can omit the manifest hash before checkpointing a fresh child; replay and
    reporting must supply the hash already bound by their parent receipt.
    """
    if (
        expected.phase not in PHASES
        or not isinstance(expected.functional_run_id, str)
        or not expected.functional_run_id
        or any(
            not _is_sha256(value)
            for value in (
                expected.functional_submission_sha256,
                expected.candidate_record_sha256,
                expected.candidate_sha256,
                expected.corpus_manifest_sha256,
                expected.corpus_capsules_sha256,
                expected.certificate_sha256,
            )
        )
    ):
        raise MeasurementEvidenceError("paired measurement expected binding is incomplete")
    manifest, raw_cells, observed_sha = _read_measurement(manifest_path, manifest_sha256, require_current_schema=True)
    certificate = manifest.get("gsim_certificate") or {}
    corpus = manifest.get("frozen_corpus") or {}
    completion = manifest.get("completion") or {}
    engine_policy = manifest.get("engine_policy") or {}
    plan = manifest.get("measurement_plan") or {}
    if plan.get("schema") != "paired_arm4_measurement_plan_v3":
        raise MeasurementEvidenceError(
            "unsupported paired measurement plan schema (expected paired_arm4_measurement_plan_v3)"
        )
    if manifest.get("measurement_plan_sha256") != hashlib.sha256(_canonical_bytes(plan)).hexdigest():
        raise MeasurementEvidenceError("paired measurement plan digest is invalid")
    try:
        expected_results = tuple(ResultIdentity(**row) for row in plan.get("expected_results") or [])
        recomputed_completion = completion_report(raw_cells, expected_results)
    except Exception as exc:
        raise MeasurementEvidenceError(f"paired raw evidence cannot be revalidated: {exc}") from exc
    if recomputed_completion != completion:
        raise MeasurementEvidenceError("paired completion does not match raw evidence")
    identities = {
        "phase": expected.phase,
        "functional_run_id": expected.functional_run_id,
        "functional_submission_sha256": expected.functional_submission_sha256,
        "candidate_record_sha256": expected.candidate_record_sha256,
        "candidate_sha256": expected.candidate_sha256,
    }
    if any(manifest.get(key) != value for key, value in identities.items()):
        raise MeasurementEvidenceError("paired measurement manifest identity differs from its trial declaration")
    if (
        certificate.get("sha256") != expected.certificate_sha256
        or corpus.get("manifest_sha256") != expected.corpus_manifest_sha256
        or corpus.get("capsules_sha256") != expected.corpus_capsules_sha256
        or corpus.get("visibility") != expected.phase
    ):
        raise MeasurementEvidenceError("paired measurement certificate/corpus identity differs from declaration")
    expected_cells = completion.get("expected")
    if (
        isinstance(expected_cells, bool)
        or not isinstance(expected_cells, int)
        or expected_cells <= 0
        or completion.get("reported") != expected_cells
        or completion.get("passed") != expected_cells
        or completion.get("failed") != 0
        or completion.get("missing") != 0
        or completion.get("complete") is not True
        or engine_policy.get("rtl_execution_backends") != ["gsim"]
        or engine_policy.get("timing_authority") != "gsim"
        or engine_policy.get("verilator") != "prelaunch_certificate_qualification_only"
        or manifest.get("identity_before") != manifest.get("identity_after")
        or (manifest.get("fork_before") or {}).get("ok") is not True
        or (manifest.get("fork_after") or {}).get("ok") is not True
    ):
        raise MeasurementEvidenceError("paired measurement lacks complete GSIM-only identity/fork evidence")
    return VerifiedPairedMeasurement(manifest_path.resolve(), observed_sha, manifest, tuple(raw_cells))

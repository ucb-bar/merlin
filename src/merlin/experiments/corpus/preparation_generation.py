"""Verify completed corpus generation and its immutable source lineage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..spec import SpecError, read_yaml
from .preparation_io import ordinary_tree


def _verify_completed_frozen_sources(plan: dict, run_dir: Path) -> None:
    """Admit only recorded run-owned source bytes, not a live producer/runtime."""
    from merlin.development.testgen import freeze

    from ..adapters import PHASE0_MODULE
    from ..runner import _command_value, _phase0_input_paths, fingerprint

    command = plan["phases"]["0"]
    if command.get("module") != PHASE0_MODULE or command["argv"][1:3] != ["-m", PHASE0_MODULE]:
        raise SpecError("completed Phase-0 module binding is not the recorded derivation implementation")
    paths, pins = plan.get("input_paths"), plan.get("inputs")
    if not isinstance(paths, dict) or not isinstance(pins, dict) or set(paths) != set(pins):
        raise SpecError("completed Phase-0 input closure differs from its frozen pin inventory")
    membership = plan.get("phase0_operator_inputs")
    if not isinstance(membership, dict) or {
        name: path for name, path in paths.items() if name.startswith("phase0:operator:")
    } != _phase0_input_paths(membership):
        raise SpecError("completed Phase-0 operator selection differs from its frozen pin inventory")
    for name, record in membership.items():
        if name == "phase0:operator:application_demands_sidecar":
            continue  # Derived from the selected requirement, not a command argument.
        option = name.removeprefix("phase0:operator:")
        value = command["inputs"].get(option)
        if (record is None and value is not None) or (record is not None and value != record["path"]):
            raise SpecError(f"completed Phase-0 operator input changed: {name}")
        flag = "--" + option.replace("_", "-")
        if value is not None and _command_value(command, flag) != value:
            raise SpecError(f"completed Phase-0 command differs from its selected input: {name}")
    for name, selected in paths.items():
        pin = pins[name]
        if not isinstance(selected, str) or not isinstance(pin, dict) or pin.get("path") != selected:
            raise SpecError(f"completed Phase-0 input pin changed: {name}")
        path = Path(selected)
        if not path.is_absolute() or path != path.resolve(strict=True) or not path.is_relative_to(run_dir):
            raise SpecError(f"completed Phase-0 input escapes its ordinary run: {name}")
        ordinary_tree(path)
        if fingerprint(path) != pin.get("sha256"):
            raise SpecError(f"completed Phase-0 frozen input changed: {name}")
    for key in ("phase0_source_snapshot", "phase0_evidence_bundle", "phase0_m2m_runtime_receipt"):
        selected = plan.get(key)
        if selected is None:
            continue
        path = Path(selected)
        if not path.is_absolute() or path != path.resolve(strict=True) or not path.is_relative_to(run_dir):
            raise SpecError(f"completed Phase-0 {key} is not run-owned")
    try:
        freeze.verify_completed_artifact(plan)
    except (OSError, ValueError) as exc:
        raise SpecError(f"completed Phase-0 frozen source changed: {exc}") from exc


def _verify_completed_generation(plan: dict, generated: Path) -> None:
    """Reopen committed generation and capture issuer evidence without recapturing."""
    from merlin.development.testgen.capture_execution_attestation import (
        AttestationNotVerified,
        require_verified_execution,
    )
    from merlin.development.testgen.evidence import load_exported_evidence
    from merlin.development.testgen.evidence_status import capture_attestation_diagnostics
    from merlin.development.testgen.sealed_generation import verified_capture_failure

    lineage = generation_lineage(plan, generated)
    bundle = plan.get("phase0_evidence_bundle")
    if lineage is None or not isinstance(bundle, str):
        raise SpecError("completed frozen Phase-0 run has no selected generation lineage")
    evidence = load_exported_evidence(bundle)
    selected = plan["phases"]["0"].get("phase0_evidence") or {}
    if selected != {
        "status": evidence.status,
        "raw_facts_sha256": evidence.raw_facts_sha256,
        "views_sha256": hashlib.sha256(evidence.views_json).hexdigest(),
        "diagnostics": evidence.diagnostics,
    }:
        raise SpecError("completed Phase-0 evidence differs from its frozen selection")
    if evidence.status != "verified":
        return  # Diagnostic corpus inspection remains possible, never promoted here.
    views = json.loads(evidence.views_json)
    applications = (views.get("application_inventory") or {}).get("applications")
    attestations = views.get("capture_execution_attestations")
    if (
        not isinstance(applications, dict)
        or not isinstance(attestations, dict)
        or set(applications) != set(attestations)
    ):
        raise SpecError("verified Phase-0 capture attestations do not cover the selected applications")
    if failures := capture_attestation_diagnostics(applications, attestations):
        raise SpecError(f"verified Phase-0 capture execution changed: {failures[0]}")
    for member, (_path, capsule) in _members(generated).items():
        for attestation in capsule.get("capture_execution_attestations") or []:
            try:
                require_verified_execution(attestation)
            except AttestationNotVerified as exc:
                raise SpecError(f"generation-time capture attestation changed: {member}: {exc}") from exc
        if failure := verified_capture_failure(capsule):
            raise SpecError(f"generation-time capture admission changed: {member}: {failure}")
    # Last, once the capture evidence is known to be the evidence the corpus was generated from: a
    # verified corpus whose sealed instruction policy forbids nothing is not released.
    _require_enforceable_instruction_policy(plan, generated)


def _require_enforceable_instruction_policy(plan: dict, generated: Path) -> None:
    """Refuse to release a verified corpus whose sealed instruction policy cannot refuse anything.

    The roles the Phase 0 command declared (else the ones the manifest records) must each resolve to at
    least one of the target's instructions. A corpus once sealed ``status: resolved`` beside
    ``vacuous_roles: [loop_descriptor]`` -- a no-FSM rule that matched nothing -- and every later phase
    read that as enforced."""
    from merlin.development.testgen.instruction_roles import enforcement_problems

    policy = read_yaml(generated / "MANIFEST.yaml").get("instruction_policy")
    declared = ((plan["phases"]["0"].get("instruction_policy") or {}).get("prohibited_instruction_roles")) or []
    roles = list(declared) or list((policy or {}).get("prohibited_instruction_roles") or ())
    if roles and (problems := enforcement_problems(policy, roles)):
        raise SpecError(f"verified Phase-0 corpus carries an unenforceable instruction policy: {'; '.join(problems)}")


def _members(root: Path) -> dict[str, tuple[Path, dict]]:
    result = {}
    names = set()
    for path in sorted(root.glob("*/*/capsule.yaml")):
        document = read_yaml(path)
        name = document.get("name")
        relative = path.parent.relative_to(root).as_posix()
        if not isinstance(name, str) or name != path.parent.name or name in names:
            raise SpecError("corpus contains duplicate or inconsistent capsule identities")
        names.add(name)
        result[relative] = (path.parent, document)
    if not result:
        raise SpecError("corpus contains no capsule members")
    return result


def generation_lineage(plan: dict, generated: Path) -> dict | None:
    """Reconcile the selected Phase 0 receipt with the corpus being released.

    The run output receipt binds these bytes as a whole; this projection makes
    the requirement, evidence, omissions, coverage and capsule membership
    individually inspectable in the operator's private preparation record.
    """
    from ..runner import fingerprint

    bundle_name = plan.get("phase0_evidence_bundle")
    if bundle_name is None:
        return None  # Historical runs have no selected evidence or generation receipt.
    bundle = Path(bundle_name)
    receipt_path = bundle / "coverage" / "generation.json"
    manifest_path = bundle / "evidence-manifest.json"
    for path in (receipt_path, manifest_path):
        ordinary_tree(path)
        if not path.is_file():
            raise SpecError("selected Phase-0 lineage member is not a file")
    receipt = json.loads(receipt_path.read_bytes())
    provenance = read_yaml(generated / "MANIFEST.yaml")
    if (
        receipt.get("schema") != "merlin.phase0_generation.v1"
        or receipt.get("target") != plan["target"]
        or receipt.get("corpus_manifest") != str(generated / "MANIFEST.yaml")
        or (provenance.get("phase0_evidence") or {}).get("generation_receipt") != str(receipt_path)
        or (provenance.get("phase0_evidence") or {}).get("manifest") != str(manifest_path)
    ):
        raise SpecError("Phase-0 generation receipt does not identify the selected corpus and evidence")
    from merlin.development.testgen.evidence import load_exported_evidence

    evidence = load_exported_evidence(bundle)
    if evidence.target != plan["target"] or receipt.get("evidence_status") != evidence.status:
        raise SpecError("Phase-0 generation receipt differs from selected evidence")
    selected = plan["phases"]["0"]["inputs"]
    requirement = selected.get("conformance_spec")
    if requirement is None:
        raise SpecError("selected Phase-0 lineage has no conformance requirement")
    requirement_sha = fingerprint(Path(requirement))
    source_requirements = [row for row in evidence.source_snapshots if row.role == "conformance-spec"]
    if len(source_requirements) != 1 or source_requirements[0].sha256 != requirement_sha:
        raise SpecError("generated corpus requirement differs from selected evidence")
    coverage = receipt.get("coverage_inputs")
    if coverage != provenance.get("coverage_inputs"):
        raise SpecError("generated corpus coverage inputs differ from generation receipt")
    from merlin.development.testgen.coverage_commitment import read_inputs

    inputs = read_inputs(generated, provenance)
    if inputs is None or inputs.get("conformance") != read_yaml(Path(requirement)):
        raise SpecError("generated corpus coverage requirement differs from frozen selection")
    members = _members(generated)
    commitments = receipt.get("capsule_commitments")
    if not isinstance(commitments, list) or len(commitments) != len(members):
        raise SpecError("Phase-0 generation receipt has incomplete capsule membership")
    committed = {}
    for row in commitments:
        if not isinstance(row, dict) or not isinstance(row.get("member"), str):
            raise SpecError("Phase-0 generation receipt has malformed capsule membership")
        name = row["member"]
        if name in committed or name not in members or row.get("sha256") != fingerprint(members[name][0]):
            raise SpecError("Phase-0 generation receipt differs from emitted capsule bytes")
        committed[name] = row["sha256"]
    if set(committed) != set(members) or receipt.get("capsules_written") != len(members):
        raise SpecError("Phase-0 generation receipt omits emitted capsules")
    for phase, record in (receipt.get("cohort_coverage") or {}).items():
        if phase not in {"phase1", "phase2"} or not isinstance(record, dict):
            raise SpecError("Phase-0 generation coverage record is malformed")
        path = bundle / "coverage" / f"{phase}-capsule-coverage.json"
        ordinary_tree(path)
        if record.get("path") != str(path) or fingerprint(path) != record.get("sha256"):
            raise SpecError("Phase-0 generation coverage report changed")
        report = json.loads(path.read_bytes())
        if record.get("status") != report.get("status") or record.get("n_capsules") != (report.get("cohort") or {}).get(
            "n_capsules"
        ):
            raise SpecError("Phase-0 generation coverage summary differs from report")
    if set(receipt.get("cohort_coverage") or {}) != {"phase1", "phase2"}:
        raise SpecError("Phase-0 generation requires both cohort coverage reports")
    omitted = receipt.get("omitted")
    if not isinstance(omitted, list):
        raise SpecError("Phase-0 generation receipt has no omission accounting")

    def digest(value) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    return {
        "requirement_sha256": requirement_sha,
        "evidence_manifest_sha256": fingerprint(manifest_path),
        "generation_receipt_sha256": fingerprint(receipt_path),
        "generated_manifest_sha256": fingerprint(generated / "MANIFEST.yaml"),
        "capsules": len(members),
        "capsule_roster_sha256": digest(sorted(committed.items())),
        "omissions": {"count": len(omitted), "sha256": digest(omitted)},
        "coverage": {
            phase: {"sha256": row["sha256"], "status": row["status"]}
            for phase, row in sorted(receipt["cohort_coverage"].items())
        },
    }

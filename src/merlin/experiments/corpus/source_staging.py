"""Private record publication and ordinary source-tree checks for corpus assembly."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

from merlin.experiments.definition import SpecError, read_yaml

# ---- preparation_io ----------------------------------------------------------------------------


def private_json(path: Path, document: dict) -> None:
    """Publish a host-only record without an intermediate world-readable file."""
    temporary = path.with_name(path.name + ".pending")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def ordinary_tree(path: Path) -> None:
    """A release has no live indirection or unclassified filesystem entries."""
    if path.is_symlink() or not path.exists():
        raise SpecError("corpus release source is absent or symlinked")
    for member in [path, *path.rglob("*")] if path.is_dir() else [path]:
        if member.is_symlink() or not (member.is_file() or member.is_dir()):
            raise SpecError("corpus release source contains symlinked or nonregular entries")


# ---- preparation_generation --------------------------------------------------------------------
# Verify completed corpus generation and its immutable source lineage.


def _verify_completed_frozen_sources(plan: dict, run_dir: Path) -> None:
    """Admit only recorded run-owned source bytes, not a live producer/runtime."""
    from merlin.development.testgen.generation import freeze
    from merlin.experiments.runs.adapters import PHASE0_MODULE
    from merlin.experiments.runs.plan_sources import _command_value, _phase0_input_paths, fingerprint

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
    from merlin.development.testgen.evidence.evidence import load_exported_evidence
    from merlin.development.testgen.evidence.evidence_status import capture_attestation_diagnostics
    from merlin.development.testgen.sealed_capture.attestation import (
        AttestationNotVerified,
        require_verified_execution,
    )
    from merlin.development.testgen.sealed_capture.generation import verified_capture_failure

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
    from merlin.development.testgen.performance.scope import enforcement_problems

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
    from merlin.experiments.runs.plan_sources import fingerprint

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
    from merlin.development.testgen.evidence.evidence import load_exported_evidence

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
    from merlin.development.testgen.evidence.coverage_obligations_coverage_commitment import read_inputs

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


# ---- preparation -------------------------------------------------------------------------------
# Private source assembly for reviewed corpus releases; never a grading implementation.


def retain_deferred_performance(generated: Path, destination: Path, provenance: dict, *, target: str) -> None:
    """Keep the byte-bound deferred Phase 2 roster with the prepared functional release."""
    preparation = ((provenance.get("performance_generation") or {}).get(target) or {}).get("preparation")
    if preparation is None:
        return
    digest = preparation.get("sha256")
    if (
        preparation.get("scope") != "phase1-functional"
        or preparation.get("status") != "deferred_not_prepared"
        or preparation.get("ledger") != "phase2-deferred.json"
        or type(preparation.get("deferred_members")) is not int
        or preparation["deferred_members"] < 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise SpecError("invalid deferred Phase 2 preparation record")
    source = generated / "phase2-deferred.json"
    ordinary_tree(source)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise SpecError("deferred Phase 2 roster changed")
    try:
        ledger = json.loads(raw)
    except ValueError as exc:
        raise SpecError("invalid deferred Phase 2 roster") from exc
    entries = ledger.get("entries")
    if (
        ledger.get("generation_scope") != "phase1-functional"
        or ledger.get("status") != "deferred_not_prepared"
        or ledger.get("phase") != "phase2"
        or not isinstance(entries, list)
        or len(entries) != preparation["deferred_members"]
        or any(
            not isinstance(entry, dict)
            or entry.get("cat") != "_perf"
            or not isinstance(entry.get("performance"), dict)
            or not entry["performance"]
            for entry in entries
        )
    ):
        raise SpecError("deferred Phase 2 roster does not retain its complete declared claims")
    copy_input(source, destination / "phase2-deferred.json", expected_sha256=digest)


def copy_input(source: Path, destination: Path, *, private: bool = False, expected_sha256: str | None = None) -> str:
    from merlin.common.storage import content_store
    from merlin.experiments.runs.plan_sources import fingerprint

    ordinary_tree(source)
    before = fingerprint(source)
    if expected_sha256 is not None and before != expected_sha256:
        raise SpecError("selected corpus input differs from verified evidence; prepare a new release")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Private and public inputs may independently contain identical bytes. Do
    # not merge their storage identities: an inode alias is an answer surface,
    # while byte equality alone does not make a declared public input private.
    store = None if private else content_store.store_root()
    if source.is_dir():
        content_store.place_tree(source, destination, store)
    else:
        content_store.place_file(source, destination, store)
    if fingerprint(source) != before or fingerprint(destination) != before:
        raise SpecError("corpus release source changed while copying; prepare a new release")
    return before


def stage_phase1_policy_descriptor(source_descriptor: Path, selected: Path, private: Path) -> tuple[dict, dict]:
    """Retain an explicit Phase-1-only policy overlay without changing Phase-0 source identity."""
    selected = selected.expanduser().absolute()
    ordinary_tree(selected)
    if not selected.is_file():
        raise SpecError("selected Phase 1 policy descriptor must be an ordinary file")
    source_document = read_yaml(source_descriptor)
    selected_document = read_yaml(selected)
    if not isinstance(source_document, dict) or not isinstance(selected_document, dict):
        raise SpecError("Phase 1 policy descriptors must be mappings")

    def without_gates(document: dict) -> dict:
        return {key: value for key, value in document.items() if key != "phase1_gates"}

    if without_gates(source_document) != without_gates(selected_document):
        raise SpecError("selected Phase 1 policy descriptor may differ from Phase 0 only in phase1_gates")
    gates = selected_document.get("phase1_gates")
    if not isinstance(gates, dict) or not gates:
        raise SpecError("selected Phase 1 policy descriptor has no phase1_gates")
    retained = private / "phase1-policy-descriptor.yaml"
    digest = copy_input(selected, retained, private=True)
    retained.chmod(0o400)
    if read_yaml(retained) != selected_document:
        raise SpecError("retained Phase 1 policy descriptor differs from selected bytes")
    return {"source_path": str(selected), "sha256": digest}, copy.deepcopy(gates)


def copy_curated_harness(source: Path, destination: Path) -> tuple[str, list[str]]:
    """Freeze contained file links as bytes while binding their authored spellings.

    Some reviewed toolchain trees alias one linker script from several environment
    directories. A release cannot retain live links, including a link outside its
    declared harness. The source fingerprint binds each link target and its bytes;
    the destination fingerprint binds the fully materialized ordinary tree.
    """
    from merlin.common.digest import sha256_file
    from merlin.common.storage import content_store
    from merlin.experiments.runs.plan_sources import fingerprint

    if source.is_symlink() or not source.is_dir():
        raise SpecError("curated harness source is absent, symlinked or not a directory")
    root = source.resolve(strict=True)
    expected = hashlib.sha256()
    links = []
    for member in sorted(source.rglob("*")):
        relative = member.relative_to(source).as_posix()
        if member.is_symlink():
            if not member.is_file():
                raise SpecError("curated harness contains a directory or broken symlink")
            if not member.resolve(strict=True).is_relative_to(root):
                raise SpecError("curated harness file link points outside the declared harness")
            links.append(relative)
        elif not (member.is_file() or member.is_dir()):
            raise SpecError("curated harness contains a nonregular entry")
        row = [relative, sha256_file(member)] if member.is_file() else [relative, "directory"]
        expected.update(json.dumps(row).encode())
    before = fingerprint(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def observe(_lexical: Path, canonical: Path, _destination: Path) -> None:
        if not canonical.is_relative_to(root):
            raise SpecError("curated harness file link points outside the declared harness")

    content_store.place_tree(source, destination, content_store.store_root(), observe=observe)
    ordinary_tree(destination)
    if fingerprint(source) != before or fingerprint(destination) != expected.hexdigest():
        raise SpecError("curated harness changed while copying; prepare a new release")
    return before, links


def copy_public_hardware(source: Path, destination: Path) -> tuple[str, list[dict[str, str]]]:
    """Freeze an already-declared public hardware grant without retaining live links."""
    from merlin.common.digest import sha256_file
    from merlin.common.storage import content_store
    from merlin.experiments.runs.plan_sources import fingerprint

    if source.is_symlink() or not source.is_dir():
        raise SpecError("public hardware source must be an ordinary directory")

    def observed_digest() -> tuple[str, list[dict[str, str]]]:
        rows: list[tuple[Path, str]] = []
        links: list[dict[str, str]] = []

        def visit(path: Path, relative: Path, ancestry: tuple[tuple[int, int], ...]) -> None:
            canonical = path.resolve(strict=True)
            if path.is_symlink():
                links.append({"path": relative.as_posix(), "target": os.readlink(path), "canonical": str(canonical)})
            if canonical.is_dir():
                stat = canonical.stat()
                key = (stat.st_dev, stat.st_ino)
                if key in ancestry:
                    raise SpecError("public hardware source contains a directory-link cycle")
                if relative != Path("."):
                    rows.append((relative, "directory"))
                for child in canonical.iterdir():
                    visit(path / child.name, relative / child.name, ancestry + (key,))
            elif canonical.is_file():
                rows.append((relative, sha256_file(canonical)))
            else:
                raise SpecError("public hardware source contains a nonregular entry")

        visit(source, Path("."), ())
        digest = hashlib.sha256()
        for relative, value in sorted(rows, key=lambda row: row[0]):
            digest.update(json.dumps([relative.as_posix(), value]).encode())
        return digest.hexdigest(), sorted(links, key=lambda row: row["path"])

    before, links = observed_digest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    content_store.place_tree(source, destination, content_store.store_root())
    ordinary_tree(destination)
    if observed_digest() != (before, links) or fingerprint(destination) != before:
        raise SpecError("public hardware source changed while copying; prepare a new release")
    return before, links


def source_run(run_dir: Path) -> tuple[dict, dict, Path]:
    """Consume a completed historical producer, not certify a current execution.

    The run's unsigned attempt/output receipt and sealed source copies provide
    recorded byte consistency; a new Phase-1 tool bundle has its own source owner.
    """
    from merlin.experiments.runs import plan_execution as _merlin_plan_execution_1
    from merlin.experiments.runs import plan_sources as runner
    from merlin.experiments.runs import plans as _merlin_plans_1

    source = run_dir.expanduser().absolute()
    if source.is_symlink() or source != source.resolve(strict=True):
        raise SpecError("Phase-0 source run is indirect; select its ordinary run directory")
    record = _merlin_plans_1.status(source)
    plan = runner._read_json(source / "resolved-plan.json")
    if plan.get("run_dir") != str(source):
        raise SpecError("completed Phase-0 plan does not identify its selected run")
    phase = plan["phases"].get("0")
    attempts = [row for row in record["attempts"] if row["phase"] == "0"]
    if not phase or phase["adapter"] != "capsule_derivation" or not attempts:
        raise SpecError("release preparation requires a completed capsule-derivation phase")
    latest = attempts[-1]
    if (
        latest["state"] != "execution_succeeded"
        or type(latest.get("returncode")) is not int
        or latest["returncode"] != 0
        or latest.get("adapter") != phase["adapter"]
        or latest.get("argv") != phase["argv"]
        or not latest.get("output_sha256")
    ):
        raise SpecError("phase 0 has no successful immutable output receipt; derive a new corpus")
    output = Path(phase["engine_output"])
    if (
        not output.is_absolute()
        or output != output.resolve(strict=True)
        or latest.get("engine_output") != str(output)
        or not output.is_relative_to(source)
    ):
        raise SpecError("phase-0 output receipt does not bind its run-owned corpus")
    ordinary_tree(output)
    if runner.fingerprint(output) != latest["output_sha256"]:
        raise SpecError("phase-0 output changed after successful derivation")
    if plan.get("phase0_source_snapshot"):
        _verify_completed_frozen_sources(plan, source)
        _verify_completed_generation(plan, output)
    else:
        # Old diagnostic runs retain their original execution-source checks;
        # this path never upgrades them to the completed frozen-source policy.
        _merlin_plan_execution_1._verify_inputs(plan)
    provenance = read_yaml(output / "MANIFEST.yaml")
    if phase.get("module"):
        from merlin.experiments.runs.adapters import PHASE0_MODULE, _legacy_script, _relative_entrypoint

        # The frozen source seal and recorded execution command bind the
        # historical producer; current installed source is a different owner.
        if plan.get("phase0_source_snapshot"):
            binding = Path(plan["input_paths"]["phase0:startup:provenance_binding"])
            historical = json.loads(binding.read_bytes())
            cited = _relative_entrypoint(historical["entrypoints"]["capsule_derivation"]["default"])
        else:
            cited = _legacy_script("capsule_derivation")
        if phase["module"] != PHASE0_MODULE or provenance.get("generated_by") != cited:
            raise SpecError("phase-0 provenance does not identify the frozen derivation implementation")
        if latest.get("argv") != phase["argv"]:
            raise SpecError("phase-0 execution receipt differs from its frozen module command")
    else:
        source = Path(phase["env"]["MERLIN_REPO_ROOT"]) / str(provenance.get("generated_by", ""))
        if source.resolve() != Path(phase["entrypoint"]).resolve() or phase["argv"][1:2] != [phase["entrypoint"]]:
            raise SpecError("phase-0 provenance does not identify the frozen derivation entrypoint")
    return plan, latest, output


def _reviewed_retirements(path: Path | None) -> tuple[dict[str, str], str | None]:
    """Read an explicit public-only retirement decision, bound by its source bytes."""
    from merlin.experiments.runs.plan_sources import fingerprint

    if path is None:
        return {}, None
    ordinary_tree(path)
    if not path.is_file():
        raise SpecError("retirement review input must be an ordinary file")
    document = read_yaml(path)
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or not isinstance(document.get("retired"), dict)
    ):
        raise SpecError("retirement review input requires schema_version 1 and retired mapping")
    retired = document["retired"]
    for member, reason in retired.items():
        if (
            not isinstance(member, str)
            or len(Path(member).parts) != 2
            or Path(member).is_absolute()
            or ".." in Path(member).parts
            or member.startswith("hidden/")
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise SpecError("retirement decisions require public category/member paths and nonempty reasons")
    return retired, fingerprint(path)


def publish_loader_profile(source: Path, target: Path, document: dict, *, profile_sources: tuple[Path, ...]):
    """Publish a receipt-bound existing loader dependency only in a fresh release."""
    capture = document.get("materialized_capture")
    if not isinstance(capture, dict):
        return None

    def read_bound(path, expected_sha, expected_bytes=None):
        path = Path(path)
        if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise SpecError("loader profile dependency is absent or indirect")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha or (
            expected_bytes is not None and len(raw) != expected_bytes
        ):
            raise SpecError("loader profile dependency differs from authenticated capture evidence")
        return raw

    receipt = source / "source-capture-receipt.json"
    receipt_raw = read_bound(receipt.absolute(), capture.get("receipt_sha256"))
    original = json.loads(receipt_raw)
    origin = Path(capture.get("path", "")).parent
    meta_pin = (original.get("artifacts") or {}).get("meta.json")
    if not isinstance(meta_pin, dict):
        raise SpecError("loader profile has no capture metadata commitment")
    meta_path = origin / "meta.json"
    meta_raw = read_bound(meta_path, meta_pin.get("sha256"), meta_pin.get("bytes"))
    meta = json.loads(meta_raw)
    digest = (meta.get("loader_provenance") or {}).get("profile_sha256")
    if digest is None:
        if (target / "profile.json").exists() or ((document.get("operation") or {}).get("attributes") or {}).get(
            "loader_profile"
        ):
            raise SpecError("published loader profile has no authenticated capture identity")
        return None
    if not isinstance(digest, str) or len(digest) != 64:
        raise SpecError("capture profile digest is malformed")
    if capture.get("profile_sha256") not in (None, digest):
        raise SpecError("selected profile differs from authenticated capture metadata")
    loader_sha = capture.get("loader_sha256")
    if loader_sha != (original.get("source") or {}).get("sha256"):
        raise SpecError("loader profile owner differs from captured source")
    loader_name = (document.get("pytorch_ref") or {}).get("loader")
    if not isinstance(loader_name, str) or Path(loader_name).name != loader_name:
        raise SpecError("loader profile needs an ordinary capsule-local owner")
    read_bound((source / loader_name).absolute(), loader_sha)
    selected = []
    for candidate in profile_sources:
        candidate = Path(candidate).absolute()
        if any(p.is_symlink() for p in (candidate, *candidate.parents)) or not candidate.is_file():
            raise SpecError("selected loader profile source is absent or indirect")
        raw = candidate.read_bytes()
        if hashlib.sha256(raw).hexdigest() == digest:
            read_bound(candidate, digest)
            read_bound(candidate.with_name("loader.py"), loader_sha)
            selected.append(candidate)
    if len(selected) > 1:
        raise SpecError("loader profile dependency selection is ambiguous")
    local = source / "profile.json"
    if local.exists():
        raw = read_bound(local.absolute(), digest)
        chosen = local.absolute()
    elif selected:
        chosen = selected[0]
        raw = read_bound(chosen, digest)
    else:
        raise SpecError("authenticated loader profile is missing; select its retained source snapshot")
    profile = target / "profile.json"
    if profile.exists():
        read_bound(profile.absolute(), digest)
    else:
        copy_input(chosen, profile, private=document.get("label") == "hidden", expected_sha256=digest)
    updated = copy.deepcopy(document)
    attrs = updated.setdefault("operation", {}).setdefault("attributes", {})
    binding = {"loader_profile": "profile.json", "loader_profile_sha256": digest}
    if any(attrs.get(k) not in (None, v) for k, v in binding.items()):
        raise SpecError("published loader profile declaration conflicts with capture evidence")
    attrs.update(loader_profile="profile.json", loader_profile_sha256=digest)
    # Public corpus copies can share content-store inodes with historical inputs.
    # Replace the release-local directory entry, never truncate that shared inode.
    staged = target / ".capsule.profile.yaml"
    staged.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")
    staged.replace(target / "capsule.yaml")
    if chosen != local.absolute():
        read_bound(chosen.with_name("loader.py"), loader_sha)
    read_bound(chosen, digest)
    read_bound(receipt.absolute(), capture.get("receipt_sha256"))
    read_bound(meta_path, meta_pin.get("sha256"), meta_pin.get("bytes"))
    return {
        "profile": {"path": str(chosen), "sha256": digest, "bytes": len(raw)},
        "loader_sha256": loader_sha,
        "capture_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "capture_metadata": {"path": str(meta_path), **meta_pin},
    }

"""Private source assembly for reviewed corpus releases; never a grading implementation."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import shutil
from contextlib import redirect_stdout
from pathlib import Path

import yaml

from ..spec import SpecError, read_yaml
from .phase_selection import validate_phase_selections


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


def copy_input(source: Path, destination: Path, *, private: bool = False, expected_sha256: str | None = None) -> str:
    from merlin.common import content_store

    from ..runner import fingerprint

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
    from merlin.common import content_store
    from merlin.common.digest import sha256_file

    from ..runner import fingerprint

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
    from merlin.common import content_store
    from merlin.common.digest import sha256_file

    from ..runner import fingerprint

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
    from .. import runner

    source = run_dir.expanduser().absolute()
    if source.is_symlink() or source != source.resolve(strict=True):
        raise SpecError("Phase-0 source run is indirect; select its ordinary run directory")
    record = runner.status(source)
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
        runner._verify_inputs(plan)
    provenance = read_yaml(output / "MANIFEST.yaml")
    if phase.get("module"):
        from ..adapters import PHASE0_MODULE, _legacy_script, _relative_entrypoint

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


def _verify_completed_frozen_sources(plan: dict, run_dir: Path) -> None:
    """Admit only recorded run-owned source bytes, not a live producer/runtime."""
    from ..adapters import PHASE0_MODULE
    from ..phase0 import freeze
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
    from ..phase0.capture_execution_attestation import AttestationNotVerified, require_verified_execution
    from ..phase0.evidence import load_exported_evidence
    from ..phase0.evidence_status import capture_attestation_diagnostics
    from ..phase0.sealed_generation import verified_capture_failure

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
    from ..phase0.instruction_roles import enforcement_problems

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
    from ..phase0.evidence import load_exported_evidence

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
    from ..phase0.coverage_commitment import read_inputs

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


def _reviewed_retirements(path: Path | None) -> tuple[dict[str, str], str | None]:
    """Read an explicit public-only retirement decision, bound by its source bytes."""
    from ..runner import fingerprint

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


def assemble(
    te,
    generated: Path,
    destination: Path,
    *,
    private_baseline: Path | None = None,
    retirements: Path | None = None,
    generated_only: bool = True,
    capsule_scope: str = "all",
    loader_profile_sources: tuple[Path, ...] | None = None,
) -> dict:
    """Assemble generated public members, with an explicit safe historical overlay."""
    from merlin.targetgen.sandbox.bwrap import _validate_host_sources

    from ..runner import fingerprint
    from .accelerator_scope import select_members, validate_scope

    validate_scope(capsule_scope, generated_only=generated_only)
    if capsule_scope != "all" and private_baseline is not None:
        raise SpecError("accelerator selection cannot mix a historical private baseline")

    source_parent = te.capsule_corpus.parent
    if generated_only and retirements is not None:
        raise SpecError("generated-only release has no historical members to retire")
    private_roots = [] if generated_only else te.hidden_roots()
    if private_baseline is not None:
        if private_roots:
            raise SpecError("private baseline is already present in the descriptor corpus; do not replace it")
        ordinary_tree(private_baseline)
        if not private_baseline.is_dir():
            raise SpecError("private baseline must be a directory")
        private_roots = [private_baseline]
    roots = (
        list(private_roots)
        if generated_only
        else list(dict.fromkeys([*te.graded_roots(), *private_roots, *te.perf_roots(), *te.model_layer_roots()]))
    )
    # Copying private files into independent inodes must not launder an existing
    # public alias. Reuse the native snapshot's authoritative privacy admission.
    generated_private = generated / "hidden"
    # Check all source surfaces together. A private external baseline aliased to
    # a generated public file is just as unsafe as an alias inside either tree.
    _validate_host_sources(
        [(str(root), root) for root in roots if root not in private_roots]
        + [(str(root), root) for root in generated.iterdir() if root != generated_private],
        [(str(root), root) for root in private_roots]
        + ([(str(generated_private), generated_private)] if generated_private.exists() else []),
    )
    from .publication import require_public_capsules

    require_public_capsules(te, [root for root in roots if root not in private_roots])
    require_public_capsules(te, [root for root in generated.iterdir() if root != generated_private])
    baseline = {}
    for root in roots:
        private_external = root == private_baseline
        if root.parent != source_parent and not private_external:
            raise SpecError("descriptor corpus categories do not share one source root")
        category = "hidden" if private_external else root.name
        baseline[category] = {
            "path": str(root),
            "sha256": copy_input(root, destination / category, private=root in private_roots),
        }
    before = _members(destination) if roots else {}
    emitted = _members(generated)
    provenance = read_yaml(generated / "MANIFEST.yaml")
    declared = set(provenance.get("generated") or [])
    public_emitted = {key for key in emitted if not key.startswith("hidden/")}
    hidden_emitted = len(emitted) - len(public_emitted)
    if declared != public_emitted or (provenance.get("held_out") or {}).get("n_generated", 0) != hidden_emitted:
        raise SpecError("phase-0 provenance does not account for every generated capsule")
    phase_corpora = provenance.get("phase_corpora")
    if phase_corpora is not None:
        record = (provenance.get("performance_generation") or {}).get(te.target) or {}
        category = (record.get("phase") or {}).get("category") or "_perf"
        try:
            validate_phase_selections(
                phase_corpora.get(te.target) if isinstance(phase_corpora, dict) else None,
                performance_category=category,
                generated_members=declared,
                complete=True,
            )
        except ValueError as exc:
            raise SpecError(f"phase-0 corpus selections do not match emitted capsules: {exc}") from exc
    scope_record = None
    if capsule_scope == "accelerator":
        emitted, scope_record = select_members(emitted, fingerprint=fingerprint)
        declared = {key for key in emitted if not key.startswith("hidden/")}
    original = (
        {
            "generated": [],
            "hand_authored": [],
            "held_out": {"n_generated": 0, "n_hand_authored": len(before)},
        }
        if generated_only
        else read_yaml(source_parent / "MANIFEST.yaml")
    )
    prior_generated = set(original.get("generated") or [])
    classified = prior_generated | set(original.get("hand_authored") or [])
    if {key for key in before if not key.startswith("hidden/")} - classified:
        raise SpecError("baseline provenance does not classify every retained public corpus member")
    hidden_counts = original.get("held_out") or {}
    if sum(hidden_counts.get(key, 0) for key in ("n_generated", "n_hand_authored")) != sum(
        key.startswith("hidden/") for key in before
    ):
        raise SpecError("baseline provenance does not account for its private corpus")
    missing = (prior_generated & set(before)) - set(emitted)
    retired, retirement_digest = _reviewed_retirements(retirements)
    # Historical hand-authored public members are normally preserved.  An
    # operator may explicitly retire one (for example, a validation model that
    # must move behind the post-freeze boundary), but the review file must not
    # name a new, private, or freshly generated member.
    hand_authored = set(original.get("hand_authored") or []) & set(before)
    reviewed_hand_authored = set(retired) - missing
    if not missing.issubset(retired) or not reviewed_hand_authored.issubset(hand_authored - set(emitted)):
        raise SpecError(
            "retirement review must account exactly for absent baseline generated members "
            "and name only existing, unreplaced hand-authored public members "
            f"(missing={len(missing)}, declared={len(retired)})"
        )
    retired_receipts = []
    for key in sorted(retired):
        target = destination / key
        retired_receipts.append({"member": key, "previous_sha256": fingerprint(target), "reason": retired[key]})
        # Only the release-local copy is removed; source and frozen run stay untouched.
        shutil.rmtree(target)
    existing_names = {value[1]["name"]: key for key, value in before.items()}
    replacements = []
    profile_dependencies = []
    for key, (source, document) in emitted.items():
        previous = before.get(key)
        if document["name"] in existing_names and existing_names[document["name"]] != key:
            raise SpecError("generated capsule collides with a different source category")
        if previous and any(previous[1].get(field) != document.get(field) for field in ("name", "label", "kind")):
            raise SpecError("generated replacement changes capsule identity, kind, or visibility")
        target = destination / key
        previous_sha = fingerprint(target) if target.exists() else None
        if target.exists():
            # This is only the fresh release's copy, never its source or historical evidence.
            shutil.rmtree(target)
        digest = copy_input(source, target, private=key.startswith("hidden/"))
        if loader_profile_sources is not None:
            dependency = publish_loader_profile(source, target, document, profile_sources=loader_profile_sources)
            if dependency is not None:
                profile_dependencies.append({"member": key, **dependency})
                digest = fingerprint(target)
        replacements.append({"member": key, "previous_sha256": previous_sha, "sha256": digest})
    final_members = _members(destination)
    from .publication import require_accelerator_placement

    # Every graded public member, generated or retained, must place work on the accelerator.
    require_accelerator_placement(
        sorted({destination / key.split("/", 1)[0] for key in final_members if not key.startswith(("hidden/", "_"))})
    )
    # Only this derivation may supply new completeness inputs. A historical
    # baseline sidecar cannot fill an absent source trace in the selected run.
    from ..phase0.coverage_commitment import INPUT_PATH, read_inputs

    coverage_inputs = read_inputs(generated, provenance)
    if coverage_inputs is not None and scope_record is None:
        copy_input(generated / INPUT_PATH, destination / INPUT_PATH, private=True)
    retain_deferred_performance(generated, destination, provenance, target=te.target)
    # The promoted descriptor points at this release, so Phase 2 must see the
    # same generated provenance as Phase 0, not a live checkout's MANIFEST.
    # Functional grading still discovers only non-underscore categories.
    merged = copy.deepcopy(provenance)
    if scope_record is not None:
        from ..phase0.coverage_commitment import write_inputs
        from .phase_selection import generate_phase_selections

        if coverage_inputs is None:
            raise SpecError("accelerator selection requires original manifest-bound coverage inputs")
        # Source application obligations stay unchanged; only standalone test scope is selected.
        public_scope = {key: value for key, value in scope_record.items() if key != "members"}
        public_scope["source_coverage_inputs_sha256"] = provenance["coverage_inputs"]["sha256"]
        merged["coverage_inputs"] = write_inputs(destination, {**coverage_inputs, "capsule_scope": public_scope})
        merged["capsule_scope"] = public_scope
        merged["generated"] = sorted(declared)
        if phase_corpora is not None:
            merged["phase_corpora"][te.target] = generate_phase_selections(
                sorted(declared), performance_category=category
            )
    if retired_receipts:
        # The public manifest needs an audit commitment, not a list of removed
        # claim-model identities or reviewer prose. Detailed decisions stay in
        # the release-private preparation record.
        for key, members in (("retired_generated", missing), ("retired_hand_authored", reviewed_hand_authored)):
            if members:
                selected = [row for row in retired_receipts if row["member"] in members]
                encoded = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")
                merged[key] = {"count": len(selected), "sha256": hashlib.sha256(encoded).hexdigest()}
    merged["hand_authored"] = sorted((set(original.get("hand_authored") or []) - declared) & set(final_members))
    previous_hidden = sum((original.get("held_out") or {}).get(key, 0) for key in ("n_generated", "n_hand_authored"))
    new_hidden = {key for key in emitted if key.startswith("hidden/")} - set(before)
    merged["held_out"] = {
        "n_generated": (original.get("held_out") or {}).get("n_generated", 0) + len(new_hidden),
        "n_hand_authored": (original.get("held_out") or {}).get("n_hand_authored", 0),
    }
    if previous_hidden + len(new_hidden) != sum(key.startswith("hidden/") for key in final_members):
        raise SpecError("promoted hidden corpus count disagrees with provenance")
    (destination / "MANIFEST.yaml").write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    return {
        "mode": "generated_only" if generated_only else "historical_overlay",
        "baseline": baseline,
        "retirements": {
            "source": str(retirements) if retirements else None,
            "sha256": retirement_digest,
            "members": retired_receipts,
        },
        "generated_manifest_sha256": fingerprint(generated / "MANIFEST.yaml"),
        "replacements": replacements,
        **({"capsule_scope": scope_record} if scope_record is not None else {}),
        **({"loader_profile_dependencies": profile_dependencies} if loader_profile_sources is not None else {}),
    }


#: A descriptor's ``grading.resource_bound.derive`` value that asks for the model policy to be derived.
DERIVED_RESOURCE_BOUND = "phase0_qualified_models_v1"


def derive_resource_bound(corpus: Path) -> dict:
    """The model resource policy the staged corpus itself states, never a hand-maintained name list.

    Every public whole-model capsule Phase 0 GENERATED and ADMITTED is a required capstone: an
    integer model is admitted only with a qualified ``model_qualification`` record, and a model whose
    arithmetic needs no such bound was admitted by being written. Any other public model (a retained,
    hand-authored or unqualified one) is excluded by name. Both lists name only capsules present.
    """
    from merlin.targetgen import capsule_runner

    manifest = read_yaml(corpus / "MANIFEST.yaml")
    generated = {str(member) for member in manifest.get("generated") or ()}
    required, excluded = [], []
    for cap in capsule_runner.discover_capsules(corpus, labels={"public", "dev"}):
        if cap.get("kind") != "model":
            continue
        name = str(cap["name"])
        member = Path(str(cap.get("__dir__") or "")).resolve()
        relative = f"{member.parent.name}/{member.name}"
        if member.parent.name.startswith("_"):
            continue
        qualification = cap.get("model_qualification") or {}
        needs_bound = "integer_partial_sum_bound" in cap or qualification
        admitted = relative in generated and (not needs_bound or qualification.get("status") == "qualified")
        (required if admitted else excluded).append(name)
    if not required:
        raise SpecError("derived resource policy found no admitted whole-model capstone in the staged corpus")
    return {
        "derive": DERIVED_RESOURCE_BOUND,
        "required_admitted_models": sorted(required),
        "exclude_capsules": sorted(excluded),
    }


def derive_release_admission(staged) -> dict:
    """Seal a staged corpus's capability and cardinality decisions for review.

    Model resource decisions are authored policy until comparable cost evidence
    exists. Require one explicit decision per model so corpus growth cannot
    silently expand the expensive formal denominator.
    """
    from merlin.targetgen import capsule_runner, eligibility

    if not eligibility.capability_map_for_target(staged.target):
        raise SpecError("release admission requires a resolvable nonempty hardware capability contract")
    public = capsule_runner.discover_capsules(staged.graded_roots(), labels={"public", "dev"})
    public_names = [str(cap.get("name")) for cap in public]
    if len(public_names) != len(set(public_names)):
        raise SpecError("release public corpus has duplicate capsule names")
    models = {str(cap["name"]) for cap in public if cap.get("kind") == "model"}
    resource = set(staged.graded_resource_exclude)
    required = set(staged.graded_required_models)
    policy = resource | required
    issues = []
    if overlap := sorted(resource & required):
        issues.append("models both excluded and required admitted: " + ", ".join(overlap))
    if unclassified := sorted(models - policy):
        issues.append("unclassified public models: " + ", ".join(unclassified))
    if stale := sorted(policy - models):
        issues.append("policy names absent from staged public models: " + ", ".join(stale))
    operations = [cap for cap in public if cap.get("kind") != "model"]
    _, withheld = capsule_runner._split_ineligible(operations, staged.target)
    capability = sorted({str(row["capsule"]) for row in withheld})
    if set(capability) & models:
        raise SpecError("model capability admission cannot be inferred from operation admission")
    hidden = capsule_runner.discover_capsules(staged.hidden_roots(), labels={"hidden"})
    hidden_ops = [cap for cap in hidden if cap.get("kind") != "model"]
    _, hidden_withheld = capsule_runner._split_ineligible(hidden_ops, staged.target)
    hidden_excluded = {str(row["capsule"]) for row in hidden_withheld}
    if not hidden:
        issues.append("hidden grading cohort is empty; supply an operator-owned private baseline")
    elif len(hidden) == len(hidden_excluded):
        issues.append("hidden grading cohort has no capability-admitted capsules")
    if issues:
        raise SpecError("release admission blocked: " + "; ".join(issues))
    return {
        "capability_exclude_capsules": capability,
        "expected_cohort": {
            "source_capsules": len(public),
            "admitted_capsules": len(public) - len(capability) - len(resource),
        },
        "hidden_capability_admission": {
            "source_capsules": len(hidden),
            "admitted_capsules": len(hidden) - len(hidden_excluded),
        },
        "resource_decisions": len(resource) + len(required),
    }


def _bundle_selection(optional_levels, selected_levels, selected_variants):
    """Select declarations only; corpus and qualification policy stay unchanged."""
    from merlin.targetgen.generate_bundles import _ARMS

    from ..phase1.levels import LEVELS

    historical = tuple(_ARMS) + tuple({"EL5": "merlin_isa", "EL6": "merlin_lean"}[level] for level in optional_levels)
    arms = historical
    if selected_levels is not None:
        available = {row["id"]: row["bundle_arm"] for row in LEVELS if row["bundle_arm"] in historical}
        if (
            not isinstance(selected_levels, tuple)
            or not selected_levels
            or len(set(selected_levels)) != len(selected_levels)
            or any(level not in available for level in selected_levels)
        ):
            raise SpecError("selected corpus levels must be a nonempty unique tuple of enabled levels")
        arms = tuple(available[level] for level in selected_levels)
    variants = ("public_v0", "realistic_v0", "hwbringup_v0")
    if selected_variants is not None:
        if (
            not isinstance(selected_variants, tuple)
            or not selected_variants
            or len(set(selected_variants)) != len(selected_variants)
            or any(variant not in variants for variant in selected_variants)
        ):
            raise SpecError("selected corpus variants must be a nonempty unique tuple of supported variants")
        variants = selected_variants
    return arms, variants


def _stage_capability_contract(
    source,
    expected_sha256,
    destination,
    native_timeout_s,
    gsim_readback_transport=None,
    *,
    model_spike_timeout_s=None,
    operand_selection=None,
    kernel_timing_selection=None,
):
    """Record closed execution selections separately from original Phase 0 bytes."""
    from merlin.common.digest import sha256_file
    from merlin.runtime.backends import chipyard_rocc_bound as runtime
    from merlin.runtime.backends import rocc_selection

    if (
        native_timeout_s is None
        and model_spike_timeout_s is None
        and gsim_readback_transport is None
        and operand_selection is None
        and kernel_timing_selection is None
    ):
        return {"path": str(source), "sha256": copy_input(source, destination, expected_sha256=expected_sha256)}
    ordinary_tree(source)
    if sha256_file(source) != expected_sha256:
        raise SpecError("selected corpus input differs from verified evidence; prepare a new release")
    try:
        selected = read_yaml(source)
        timing_record = None
        if kernel_timing_selection is not None:
            from .kernel_timing_selection import select

            selected, timing_record = select(selected, kernel_timing_selection)
        if native_timeout_s is not None:
            selected = rocc_selection.select_native_timeout(selected, native_timeout_s)
        if model_spike_timeout_s is not None:
            selected = rocc_selection.select_model_spike_timeout(selected, model_spike_timeout_s)
        if gsim_readback_transport is not None:
            selected = rocc_selection.select_gsim_readback(selected, gsim_readback_transport)
        if operand_selection is not None:
            selected = rocc_selection.select_operand_protocol(selected, operand_selection)
    except (runtime.RoCCExecutionError, ValueError) as error:
        raise SpecError(str(error)) from error
    native_engines = set(selected["runner"]["chipyard_rocc"]["engines"]) & {"gsim", "verilator"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        yaml.safe_dump(selected, output, sort_keys=False)
    if sha256_file(source) != expected_sha256:
        raise SpecError("selected original contract changed during execution selection")
    record = {"path": str(source), "sha256": expected_sha256}
    selected_sha256 = sha256_file(destination)
    if timing_record is not None:
        record["kernel_timing_selection"] = {**timing_record, "selected_contract_sha256": selected_sha256}
    if native_timeout_s is not None:
        record["execution_resource_selection"] = {
            "max_native_timeout_s": native_timeout_s,
            "engines": sorted(native_engines),
            "selected_contract_sha256": selected_sha256,
            "scope": "host process resources only; original semantic and numerical declarations retained",
        }
    if model_spike_timeout_s is not None:
        record["model_spike_resource_selection"] = {
            "max_model_spike_timeout_s": model_spike_timeout_s,
            "engine": "spike",
            "selected_contract_sha256": selected_sha256,
            "scope": "whole-model host process budget only; original semantic and numerical declarations retained",
        }
    if gsim_readback_transport is not None:
        record["execution_transport_selection"] = {
            "engine": "gsim",
            "readback_transport": gsim_readback_transport,
            "selected_contract_sha256": selected_sha256,
            "scope": "complete output values only; original ABI, instruction and numerical declarations retained",
        }
    if operand_selection is not None:
        record["operand_protocol_selection"] = {
            **copy.deepcopy(operand_selection),
            "selected_contract_sha256": selected_sha256,
            "scope": "source-derived operand observations only; original facts and semantic declarations retained",
        }
    return record


def scaffold(
    te,
    corpus: Path,
    experiment: Path,
    *,
    private: Path,
    selected_facts: Path | None = None,
    selected_contract: tuple[Path, str] | None = None,
    phase1_gates: dict | None = None,
    optional_levels: tuple[str, ...] = (),
    selected_levels: tuple[str, ...] | None = None,
    selected_variants: tuple[str, ...] | None = None,
    lean_toolchain_manifest: Path | None = None,
    host_index_bits: int | None = None,
    native_timeout_s: int | None = None,
    model_spike_timeout_s: int | None = None,
    gsim_readback_transport: str | None = None,
    operand_protocol_selection: tuple[Path, str] | None = None,
    kernel_timing_selection: tuple[Path, str] | None = None,
) -> dict:
    """Stage only declared task/selfcheck/harness resources, never prior runs or bundles."""
    from merlin.common.digest import sha256_file
    from merlin.targetgen.sandbox.bwrap import resolve_grant
    from merlin.targetgen.target_experiment import load_target_experiment, observed_experiment_contract

    arms, variants = _bundle_selection(optional_levels, selected_levels, selected_variants)
    experiment.mkdir(parents=True)
    inputs = {}
    for relative in ("task",):
        source = te.resource_path(relative)
        try:
            source.lstat()
        except FileNotFoundError:
            # Prompt rendering is shared code, not a target-authored placeholder file.
            # Keep a release-local directory for its generated bundle grant and bind
            # the source's absence explicitly in the private preparation record.
            (experiment / relative).mkdir()
            inputs[relative] = {"path": str(source), "present": False, "sha256": None}
        else:
            if not source.is_dir():
                raise SpecError("authored task resource must be a directory")
            inputs[relative] = {
                "path": str(source),
                "present": True,
                "sha256": copy_input(source, experiment / relative),
            }
    # Sealed admission requires bwrap. Its native broker replaces this entry with
    # the same public shim at launch; staging the shim now also makes the prepared
    # resource independently relocatable without copying grader dependencies.
    from merlin.common.paths import module_source_path

    shim = module_source_path("merlin_experiments.phase1.tools.selfcheck")
    inputs["scripts/agent_selfcheck.py"] = {
        "path": str(shim),
        "sha256": copy_input(shim, experiment / "scripts/agent_selfcheck.py"),
    }
    document = copy.deepcopy(read_yaml(te.path))
    if phase1_gates is not None:
        document["phase1_gates"] = copy.deepcopy(phase1_gates)
    # A release owns the resources copied below; never retain a pointer to live authored inputs.
    document.pop("resources_root", None)
    document.pop("task_root", None)
    document.pop("contracts_root", None)
    document["capsule_corpus"] = str(corpus / te.capsule_corpus.name)
    hardware = document.get("hardware_spec") or {}
    facts_root = None
    if selected_facts is not None:
        facts_root = experiment / "rtl_facts"
        digest = copy_input(selected_facts, facts_root / "facts.json")
        inputs["rtl_facts"] = {"path": str(selected_facts), "sha256": digest}
    operand_selection = None
    checker_inputs = ()
    if operand_protocol_selection is not None:
        from merlin.targetgen.rocc.selected_operand_protocol import selected_inputs

        if (
            type(operand_protocol_selection) is not tuple
            or len(operand_protocol_selection) != 2
            or not isinstance(operand_protocol_selection[0], Path)
            or facts_root is None
        ):
            raise SpecError("operand protocol selection requires a pinned sidecar and unchanged selected facts")
        sidecar, sidecar_sha256 = operand_protocol_selection
        staged_sidecar = experiment / "contracts/operand-protocol.json"
        digest = copy_input(sidecar, staged_sidecar, expected_sha256=sidecar_sha256)
        inputs["operand_protocol"] = {"path": str(sidecar), "sha256": digest}
        operand_selection = {
            "original_facts": {"path": str(facts_root / "facts.json"), "sha256": inputs["rtl_facts"]["sha256"]},
            "sidecar": {"path": str(staged_sidecar), "sha256": digest},
        }
        checker_inputs = selected_inputs(operand_selection)
    if selected_contract is not None:
        contract_source, contract_sha256 = selected_contract
        staged_contract = experiment / "contracts/target_contract.yaml"
        contract_record = _stage_capability_contract(
            contract_source,
            contract_sha256,
            staged_contract,
            native_timeout_s,
            gsim_readback_transport,
            model_spike_timeout_s=model_spike_timeout_s,
            operand_selection=operand_selection,
            kernel_timing_selection=kernel_timing_selection,
        )
        hardware["target_contract"] = str(staged_contract)
        inputs["capability_contract"] = contract_record
    elif te.declared_contract:
        raise SpecError("descriptor declares a capability contract without a selected Phase 0 export; freeze a new run")
    elif (
        native_timeout_s is not None
        or model_spike_timeout_s is not None
        or gsim_readback_transport is not None
        or operand_selection is not None
        or kernel_timing_selection is not None
    ):
        raise SpecError("execution selection requires an original Phase 0 contract")
    hardware_root = None
    if te.hwbringup_set:
        source = resolve_grant(te.hwbringup_set)
        hardware_root = experiment / "hardware_spec/hwbringup"
        digest, links = copy_public_hardware(source, hardware_root)
        hardware["hwbringup_set"] = str(hardware_root)
        inputs["hardware_spec/hwbringup"] = {"path": str(source), "sha256": digest, "materialized_links": links}
    if te.isa_headers:
        staged_headers = []
        source_root = resolve_grant(te.hwbringup_set) if te.hwbringup_set else None
        for index, declared in enumerate(te.isa_headers):
            source = resolve_grant(declared)
            if source_root is not None and source.is_relative_to(source_root):
                staged = hardware_root / source.relative_to(source_root)
            else:
                staged = experiment / "hardware_spec/isa_headers" / f"{index:02d}-{source.name}"
                if not source.is_file():
                    raise SpecError(f"declared ISA header is not a regular file: {source}")
                before = sha256_file(source)
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, staged, follow_symlinks=True)
                if sha256_file(source) != before or sha256_file(staged) != before:
                    raise SpecError("ISA header changed while copying; prepare a new release")
            if not staged.is_file() or staged.is_symlink():
                raise SpecError(f"staged ISA header is not an ordinary file: {staged}")
            staged_headers.append(str(staged))
        hardware["isa_headers"] = staged_headers
    if hardware:
        document["hardware_spec"] = hardware
    if te.curated_harness:
        relative = Path(te.curated_harness)
        if relative.is_absolute() or ".." in relative.parts:
            raise SpecError("curated harness must be a declared contained experiment resource")
        source = te.resource_path(relative)
        digest, links = copy_curated_harness(source, experiment / relative)
        inputs[relative.as_posix()] = {"path": str(source), "sha256": digest, "materialized_file_links": links}
    resource_bound = (document.get("grading") or {}).get("resource_bound") or {}
    if resource_bound.get("derive") is not None:
        if resource_bound.get("derive") != DERIVED_RESOURCE_BOUND:
            raise SpecError(f"unknown resource_bound derivation {resource_bound.get('derive')!r}")
        if resource_bound.get("required_admitted_models") or resource_bound.get("exclude_capsules"):
            raise SpecError("a derived resource policy cannot also carry hand-maintained model names")
        derived_bound = derive_resource_bound(corpus)
        document["grading"]["resource_bound"] = {
            key: value for key, value in {**resource_bound, **derived_bound}.items() if key != "derive" and value != []
        }
        inputs["resource_bound"] = {"policy": DERIVED_RESOURCE_BOUND, **derived_bound}
    descriptor = experiment / "target_experiment.yaml"
    descriptor.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    if te.graded_release_admission:
        staged = load_target_experiment(descriptor)
        with observed_experiment_contract(staged):
            derived = derive_release_admission(staged)
        grading = document["grading"]
        grading.pop("release_admission")
        grading.update({key: value for key, value in derived.items() if key != "resource_decisions"})
        descriptor.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        inputs["release_admission"] = {
            "policy": "derive_from_corpus_v1",
            "public_source": derived["expected_cohort"]["source_capsules"],
            "public_admitted": derived["expected_cohort"]["admitted_capsules"],
            "hidden_source": derived["hidden_capability_admission"]["source_capsules"],
            "hidden_admitted": derived["hidden_capability_admission"]["admitted_capsules"],
            "resource_decisions": derived["resource_decisions"],
        }
    # Generate fresh declarations, not copies that might still point at the old corpus.
    from merlin.common.paths import python_source_dir
    from merlin.targetgen.generate_bundles import materialize_bundles
    from merlin.targetgen.sandbox.toolchain import ToolchainPaths
    from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V3

    promoted = load_target_experiment(descriptor)
    toolchain = ToolchainPaths.from_checkout()
    llvm_root = Path(toolchain.llvm)
    if not llvm_root.is_dir():
        raise SpecError(
            f"selected LLVM/MLIR toolchain is absent: {llvm_root}; "
            "select an installed clang-23 with MERLIN_CLANG before preparing a release"
        )
    checkout_llvm = toolchain.repo / "third_party/llvm-install"
    # Native generation can explain unavailable contracts. Keep those diagnostics
    # in the private preparation record, never the public inspect response.
    diagnostics = io.StringIO()
    with redirect_stdout(diagnostics), observed_experiment_contract(promoted):
        materialize_bundles(
            promoted,
            experiment / "input_bundles",
            variants=variants,
            host_inputs=(
                str(private),
                *checker_inputs,
                *([str(corpus / "_phase0")] if (corpus / "_phase0").is_dir() else []),
            ),
            python_source_root=python_source_dir(),
            llvm_toolchain_root=llvm_root if llvm_root != checkout_llvm else None,
            rtl_facts_root=facts_root,
            common_substrate=COMMON_SUBSTRATE_V3,
            lean_toolchain_manifest=lean_toolchain_manifest,
            host_index_bits=host_index_bits,
            arms=arms,
        )
    inputs["bundle_generation"] = {"diagnostics": diagnostics.getvalue()}
    if selected_levels is not None or selected_variants is not None:
        inputs["bundle_generation"]["selection"] = {"arms": list(arms), "variants": list(variants)}
    return inputs


def admission(descriptor: Path, *, coverage_output: Path | None = None) -> dict:
    """Reuse native discovery/admission without launching an oracle or changing policy."""
    from merlin.targetgen.target_experiment import load_target_experiment, observed_experiment_contract

    te = load_target_experiment(descriptor)
    with observed_experiment_contract(te):
        return _admission(te, coverage_output=coverage_output)


def _admission(te, *, coverage_output: Path | None) -> dict:
    from merlin.targetgen import capsule_runner
    from merlin.targetgen.contract.materialize import (
        _TIER_ORDER,
        materialize_public_cohort,
        pin_cohort_builds,
        resolve_published_cohort,
        validate_materialized_cohort,
    )

    from ..phase0.certification_floor import require_direct_tier, selected_floor

    from .publication import require_accelerator_placement

    # Re-checked here, not only at assembly: sealing re-runs this admission over the reviewed payload.
    require_accelerator_placement(te.graded_roots())
    source_public = capsule_runner.discover_capsules(te.graded_roots(), labels={"public", "dev"})
    source_hidden = None
    if (te.workload_spec or {}).get("certification_floor") is not None:
        source_hidden = capsule_runner.discover_capsules(te.hidden_roots(), labels={"hidden"})
        declared_tiers = sorted(
            {tier for cap in [*source_public, *source_hidden] for tier in (cap.get("required_oracle_tiers") or [])}
        )
        try:
            floor = selected_floor(te.workload_spec, declared_tiers)
            require_direct_tier([*source_public, *source_hidden], floor=floor)
        except ValueError as exc:
            raise SpecError(f"release certification-floor preflight failed: {exc}") from exc
    # Read the immutable build behind the published per-target link: coverage observation refuses
    # to traverse links, and the link is this module's own swap point, not a corpus input.
    materialized = resolve_published_cohort(materialize_public_cohort(te, tier_ceiling=_TIER_ORDER[-1]))
    pin_cohort_builds(materialized)
    public = validate_materialized_cohort(materialized, te)
    from ..phase0.coverage_commitment import (
        admitted_source_roots,
        build_phase0_readiness,
        observe_cohort,
        phase0_readiness_identity,
        read_inputs,
        requires_workload_coverage,
    )

    coverage_inputs = read_inputs(te.capsule_corpus.parent)
    from .accelerator_scope import has_accelerator_work, selected_scope
    completeness = observe_cohort(
        coverage_inputs,
        admitted_source_roots(te.graded_roots(), materialized),
        target=te.target,
    )
    readiness = build_phase0_readiness(completeness)
    workload_required = requires_workload_coverage(te, coverage_inputs)
    if coverage_output is not None:
        private_json(coverage_output, completeness)
    hidden = (
        source_hidden
        if source_hidden is not None
        else capsule_runner.discover_capsules(te.hidden_roots(), labels={"hidden"})
    )
    if selected_scope(coverage_inputs) is not None:
        for capsule in [*source_public, *hidden]:
            if not has_accelerator_work(capsule):
                raise SpecError("accelerator-only release contains a host-only capsule")
    _, withheld = capsule_runner._split_ineligible([cap for cap in hidden if cap.get("kind") != "model"], te.target)
    excluded = {row["capsule"] for row in withheld}
    count = len(hidden)
    admitted = sum(cap["name"] not in excluded for cap in hidden)
    if public["n_admitted_capsules"] <= 0 or count <= 0 or admitted <= 0:
        raise SpecError("reviewable corpus must have nonempty public and hidden grading cohorts")
    if (te.hidden_expected_source_capsules is not None and count != te.hidden_expected_source_capsules) or (
        te.hidden_expected_admitted_capsules is not None and admitted != te.hidden_expected_admitted_capsules
    ):
        raise SpecError("hidden corpus differs from descriptor-frozen admission counts")
    return {
        "public_source": public["n_source_capsules"],
        "public_admitted": public["n_admitted_capsules"],
        "hidden_source": count,
        "hidden_admitted": admitted,
        "public_commitment": public["admitted_name_set_sha256"],
        "scope": "native cohort admission only; numerical and hardware readiness not executed",
        "phase0_readiness": phase0_readiness_identity(readiness, required=workload_required),
        "whole_workload_phase1": {
            "status": completeness["status"],
            "cohort_sha256": completeness["cohort"]["sha256"],
            "inputs_sha256": completeness["inputs_sha256"],
            "n_blockers": len(completeness["blockers"]),
            "required": workload_required,
            "report_sha256": hashlib.sha256(
                json.dumps(completeness, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    }

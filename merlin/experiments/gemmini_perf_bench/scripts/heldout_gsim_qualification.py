#!/usr/bin/env python3
"""Host-only, post-seal GSIM qualification for a revealed performance holdout.

The performance agent never imports or invokes this module.  The experiment orchestrator calls it
only after all candidate trees have been sealed and the committed holdout has been revealed.  Each
manifest-declared capsule is lowered by the immutable functional baseline, compiled once, and that
same ELF is checked on the pinned Verilator and GSIM engines before the tuning certificate is extended.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

import perf_gsim_gate as GATE
import produce_gsim_certificate as PRODUCER
import perf_campaign as CAMPAIGN
from merlin.benchharness import hash_tree


SCHEMA = "merlin.heldout-gsim-qualification.v1"


class QualificationError(RuntimeError):
    """The post-seal qualification boundary could not establish its claim."""


@dataclass(frozen=True)
class RevealedMember:
    name: str
    family: str
    cohort: str
    source_dir: Path
    manifest: Path
    workload: Mapping[str, Any]
    workload_sha256: str


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_file(path: Path, *, label: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"{label} is absent or linked: {path}")
    return path.resolve()


def _safe_relative(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise QualificationError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise QualificationError(f"{label} is unsafe or noncanonical: {value!r}")
    return path


def _tree_without_manifest(root: Path, manifest: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise QualificationError(f"revealed corpus contains a symlink: {path}")
        if path.is_file() and path != manifest:
            rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                         "sha256": _sha_file(path)})
    return {"files": rows, "sha256": _sha_bytes(_canonical(rows))}


def load_revealed_members(
        manifest_path: str | Path, *, expected_manifest_sha256: str | None = None,
        expected_corpus_sha256: str | None = None, expected_target: str | None = None,
        require_frozen: bool = True) -> tuple[RevealedMember, ...]:
    """Validate a v2 reveal and resolve only its explicitly declared member paths."""
    manifest = _plain_file(Path(manifest_path), label="revealed holdout manifest")
    if expected_manifest_sha256 is not None and _sha_file(manifest) != expected_manifest_sha256:
        raise QualificationError("revealed holdout manifest digest changed")
    root = manifest.parent
    if require_frozen:
        for path in (root, *root.rglob("*")):
            if path.is_symlink() or path.stat().st_mode & 0o222:
                raise QualificationError(f"revealed holdout is linked or writable: {path}")
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QualificationError("revealed holdout manifest is not valid JSON") from exc
    if (not isinstance(document, Mapping) or document.get("schema_version") != 2
            or document.get("kind") != "generated_performance_holdout_reveal"):
        raise QualificationError("revealed holdout is not the required v2 commit/reveal manifest")
    target = (document.get("domain") or {}).get("target")
    if expected_target is not None and target != expected_target:
        raise QualificationError("revealed holdout target differs from the experiment target")
    actual_tree = _tree_without_manifest(root, manifest)
    declared_tree = document.get("corpus")
    if not isinstance(declared_tree, Mapping) or actual_tree != dict(declared_tree):
        raise QualificationError("revealed corpus bytes differ from their committed tree")
    if expected_corpus_sha256 is not None and actual_tree["sha256"] != expected_corpus_sha256:
        raise QualificationError("revealed corpus digest differs from the orchestrator checkpoint")

    cohorts = document.get("cohorts")
    rows = document.get("members")
    if not isinstance(cohorts, Mapping) or not isinstance(rows, list) or not rows:
        raise QualificationError("revealed holdout has no declared cohorts/members")
    members: list[RevealedMember] = []
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    seen_workloads: set[str] = set()
    cohort_counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise QualificationError(f"revealed member {index} is malformed")
        name, family, cohort = row.get("name"), row.get("family"), row.get("cohort")
        if (not isinstance(name, str) or Path(name).name != name or name in ("", ".", "..")
                or not isinstance(family, str) or not family
                or not isinstance(cohort, str) or cohort not in cohorts):
            raise QualificationError(f"revealed member {index} has an invalid identity")
        declaration = cohorts[cohort]
        if not isinstance(declaration, Mapping) or declaration.get("family") != family:
            raise QualificationError(f"revealed member {name} disagrees with its cohort")
        relative = _safe_relative(row.get("path"), label=f"revealed member {name} path")
        source = (root / relative).resolve(strict=True)
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise QualificationError(f"revealed member escapes the corpus: {name}") from exc
        if source.is_symlink() or not source.is_dir() or source.name != name:
            raise QualificationError(f"revealed member directory is absent or substituted: {name}")
        capsule_manifest = _plain_file(source / "capsule.yaml", label=f"{name} descriptor")
        try:
            descriptor = yaml.safe_load(capsule_manifest.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise QualificationError(f"revealed member descriptor is unreadable: {name}") from exc
        if not isinstance(descriptor, Mapping) or descriptor.get("name") != name:
            raise QualificationError(f"revealed descriptor identity differs: {name}")
        coordinates = (row.get("M"), row.get("N"), row.get("K"))
        workload = PRODUCER.derive_workload(capsule_manifest)
        shape = workload.get("shape") or {}
        if coordinates != (shape.get("m"), shape.get("n"), shape.get("k")):
            raise QualificationError(f"revealed member coordinates differ from descriptor: {name}")
        identity = GATE.workload_sha256(workload)
        rel_text = relative.as_posix()
        if name in seen_names or rel_text in seen_paths or identity in seen_workloads:
            raise QualificationError("revealed holdout contains duplicate names, paths, or workloads")
        seen_names.add(name)
        seen_paths.add(rel_text)
        seen_workloads.add(identity)
        cohort_counts[cohort] = cohort_counts.get(cohort, 0) + 1
        members.append(RevealedMember(name, family, cohort, source, capsule_manifest,
                                      workload, identity))
    for cohort, declaration in cohorts.items():
        count = declaration.get("member_count") if isinstance(declaration, Mapping) else None
        if cohort_counts.get(str(cohort), 0) != count:
            raise QualificationError(f"revealed cohort count differs: {cohort}")
    return tuple(members)


def _assert_readonly_tree(root: Path, *, expected_sha256: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise QualificationError(f"functional baseline is absent or linked: {root}")
    for path in (root, *root.rglob("*")):
        if path.is_symlink() or path.stat().st_mode & 0o222:
            raise QualificationError(f"functional baseline is linked or writable: {path}")
    if str(hash_tree(root)["sha256"]) != expected_sha256:
        raise QualificationError("functional baseline compiler digest differs from its sealed run")


def lower_with_functional_baseline(
        functional_base: Path, member: RevealedMember, artifact_dir: Path, timeout: int) -> Path:
    """Run the immutable package ABI through K2-K6 and emit only into ``artifact_dir``."""
    from merlin.targetgen import capsule_common as COMMON
    from merlin.targetgen import oot_runner as OOT

    package = OOT.load_package(functional_base)
    OOT.integrity_scan(package)
    build = package.manifest.get("build") or {}
    if any(build.get(key) for key in ("configure", "command")):
        raise QualificationError(
            "frozen functional baseline still requires an in-tree build; qualification cannot write it")
    if not package.tool.is_file():
        raise QualificationError("frozen functional baseline tool is absent")
    artifact_dir.mkdir(parents=True, exist_ok=False)
    capsule = COMMON.load_capsule(member.source_dir)
    paths = SimpleNamespace(generated=artifact_dir)
    try:
        COMMON.run_entrypoints(
            package, functional_base, capsule, paths, contract=None, timeout=timeout,
            fourth_output_name="lowered.llvm.mlir")
    except Exception as exc:  # the qualification boundary turns every lowering issue into refusal
        raise QualificationError(
            f"functional baseline could not lower revealed capsule {member.name}: "
            f"{type(exc).__name__}: {str(exc)[-600:]}") from exc
    for required in (artifact_dir / "command_buffer.json", artifact_dir / "lowered.llvm.mlir"):
        _plain_file(required, label=f"{member.name} lowering artifact")
    return artifact_dir


def _validate_gsim_max_cycles(value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise QualificationError("GSIM max cycles must be a positive integer or null")


@contextlib.contextmanager
def _pinned_runtime(certificate: GATE.CertificateRecord, *, gsim_max_cycles: int | None):
    """Bind the certificate's GSIM and the declared cap, independent of ambient state."""
    _validate_gsim_max_cycles(gsim_max_cycles)
    binary_key = "MERLIN_GEMMINI_GSIM_EMU"
    cycles_key = "MERLIN_GEMMINI_GSIM_MAXCYCLES"
    previous_binary = os.environ.get(binary_key)
    previous_cycles = os.environ.get(cycles_key)
    os.environ[binary_key] = certificate.pins["gsim_binary"]["path"]
    if gsim_max_cycles is None:
        os.environ.pop(cycles_key, None)
    else:
        os.environ[cycles_key] = str(gsim_max_cycles)
    try:
        from merlin.runtime.backends import base as backends
        backend = backends.get_backend(certificate.target)
        for engine in ("gsim", "verilator"):
            resolver = getattr(backend, f"{engine}_path", None)
            if not callable(resolver):
                raise QualificationError(f"backend does not expose its {engine} binary identity")
            actual = Path(resolver()).resolve(strict=True)
            if _sha_file(actual) != certificate.pins[f"{engine}_binary"]["sha256"]:
                raise QualificationError(f"runtime {engine} binary differs from the tuning pin")
        yield backend
    finally:
        if previous_binary is None:
            os.environ.pop(binary_key, None)
        else:
            os.environ[binary_key] = previous_binary
        if previous_cycles is None:
            os.environ.pop(cycles_key, None)
        else:
            os.environ[cycles_key] = previous_cycles


def _exclusive_json(root: Path, stem: str, document: object) -> tuple[Path, str]:
    payload = _canonical(document)
    digest = _sha_bytes(payload)
    path = root / f"{stem}.{digest}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return path, digest


def _artifact_paths(certificate: GATE.CertificateRecord) -> PRODUCER.ArtifactPaths:
    return PRODUCER.ArtifactPaths(*(Path(certificate.pins[name]["path"])
                                    for name in ("gsim_firrtl", "verilator_firrtl", "gsim_model",
                                                 "gsim_binary", "verilator_binary")))


def qualify_revealed_holdout(
        reveal_manifest: Path, qualification_root: Path, tuning: GATE.CertificateRecord, *,
        functional_base: Path, functional_base_sha256: str, reveal_manifest_sha256: str,
        reveal_corpus_sha256: str, timeout: int, gsim_max_cycles: int | None,
        lowerer: Callable[[Path, RevealedMember, Path, int], Path] = lower_with_functional_baseline,
        capturer: Callable[..., Mapping[str, Any]] = PRODUCER.capture_case,
        backend: Any | None = None, target_experiment: Any | None = None) -> tuple[Path, str]:
    """Produce an exact tuning+reveal certificate inside a fresh host-only root."""
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise QualificationError("qualification timeout must be a positive integer")
    _validate_gsim_max_cycles(gsim_max_cycles)
    root = Path(qualification_root)
    if root.exists() or root.is_symlink():
        raise QualificationError(f"qualification root must be fresh: {root}")
    if root.parent.is_symlink() or not root.parent.is_dir():
        raise QualificationError("qualification root parent is absent or linked")
    members = load_revealed_members(
        reveal_manifest, expected_manifest_sha256=reveal_manifest_sha256,
        expected_corpus_sha256=reveal_corpus_sha256, expected_target=tuning.target)
    _assert_readonly_tree(Path(functional_base), expected_sha256=functional_base_sha256)
    tuning_members = set(tuning.members)
    revealed_identities = {member.workload_sha256 for member in members}
    if tuning_members & revealed_identities:
        raise QualificationError("revealed workloads overlap the tuning certificate")
    root.mkdir(mode=0o700)
    captures_dir = root / "captures"
    artifacts_dir = root / "lowered"
    work_dir = root / "work"
    for directory in (captures_dir, artifacts_dir, work_dir):
        directory.mkdir(mode=0o700)
    artifacts = _artifact_paths(tuning)
    captures: list[Mapping[str, Any]] = []
    capture_evidence: list[dict[str, Any]] = []
    runtime_context = (contextlib.nullcontext(backend) if backend is not None
                       else _pinned_runtime(tuning, gsim_max_cycles=gsim_max_cycles))
    try:
        with runtime_context as selected_backend:
            for index, member in enumerate(members):
                member_workspace = work_dir / f"m{index:03d}_{member.name}"
                member_workspace.mkdir()
                lower_context: Any = contextlib.nullcontext()
                if lowerer is lower_with_functional_baseline:
                    if target_experiment is None:
                        raise QualificationError(
                            "default baseline lowering requires the target experiment sandbox policy")
                    policy = CAMPAIGN.package_sandbox_policy(
                        target_experiment, member_workspace, Path(functional_base))
                    lower_context = CAMPAIGN.boxed_entrypoints(policy)
                with lower_context:
                    lowered = lowerer(
                        Path(functional_base), member,
                        artifacts_dir / f"m{index:03d}_{member.name}", timeout)
                _assert_readonly_tree(Path(functional_base), expected_sha256=functional_base_sha256)
                capture = dict(capturer(
                    target=tuning.target, capsule_manifest=member.manifest,
                    artifact_dir=lowered, workdir=member_workspace / "elf",
                    artifacts=artifacts, timeout=timeout, backend=selected_backend))
                if (capture.get("workload") != member.workload
                        or capture.get("workload_sha256") != member.workload_sha256):
                    raise QualificationError(f"capture workload differs from reveal: {member.name}")
                capture_path, capture_sha = _exclusive_json(
                    captures_dir, f"capture.{index:03d}.{member.name}", capture)
                PRODUCER.validate_capture(
                    capture_path, target=tuning.target, pins=tuning.pins)
                captures.append(capture)
                capture_evidence.append({"name": member.name, "family": member.family,
                                         "cohort": member.cohort,
                                         "workload_sha256": member.workload_sha256,
                                         "path": str(capture_path.resolve()),
                                         "sha256": capture_sha})
        extension = dict(tuning.document)
        original_rows = tuning.document.get("members")
        if not isinstance(original_rows, list) or len(original_rows) != len(tuning.members):
            raise QualificationError("validated tuning certificate lost its member rows")
        if tuning.document.get("unresolved") != []:
            raise QualificationError("tuning certificate has unresolved workloads")
        # The extension lives under a new content-addressed host root.  Re-emit the already validated
        # pins as absolute paths so a relative path in the tuning document cannot silently retarget when
        # its containing directory changes.  Digests and build-command commitments remain identical.
        extension["pins"] = {name: dict(tuning.pins[name]) for name in sorted(GATE.REQUIRED_PINS)}
        raw_binding = tuning.document.get("build_binding")
        if not isinstance(raw_binding, Mapping):
            raise QualificationError("validated tuning certificate lost its build binding")
        binding = dict(raw_binding)
        binding_path = Path(str(binding.get("path") or ""))
        if not binding_path.is_absolute():
            binding_path = tuning.path.parent / binding_path
        binding_path = _plain_file(binding_path, label="tuning build receipt")
        if _sha_file(binding_path) != binding.get("sha256"):
            raise QualificationError("tuning build receipt changed before envelope extension")
        binding["path"] = str(binding_path)
        extension["build_binding"] = binding
        extension["members"] = sorted(
            [*original_rows, *captures], key=lambda row: str(row["workload_sha256"]))
        extension["unresolved"] = []
        certificate_path, certificate_sha = _exclusive_json(root, "certificate", extension)
        extended = GATE.load_certificate(certificate_path, expected_sha256=certificate_sha)
        expected = tuning_members | revealed_identities
        if set(extended.members) != expected:
            raise QualificationError("extension certificate is not the exact tuning+reveal envelope")
        if any(extended.pins[name]["sha256"] != tuning.pins[name]["sha256"]
               for name in GATE.REQUIRED_PINS):
            raise QualificationError("extension certificate changed a tuning artifact pin")
        qualification = {
            "schema": SCHEMA, "status": "complete", "target": tuning.target,
            "reveal_manifest": {"path": str(Path(reveal_manifest).resolve()),
                                "sha256": reveal_manifest_sha256,
                                "corpus_sha256": reveal_corpus_sha256},
            "functional_baseline": {"path": str(Path(functional_base).resolve()),
                                    "sha256": functional_base_sha256},
            "tuning_certificate": {"path": str(tuning.path.resolve()),
                                   "sha256": tuning.sha256,
                                   "workload_sha256": sorted(tuning_members)},
            "captures": capture_evidence,
            "extension_certificate": {"path": str(certificate_path.resolve()),
                                      "sha256": certificate_sha,
                                      "workload_sha256": sorted(expected)},
            "execution": {"timeout_seconds": timeout,
                          "gsim_max_cycles": gsim_max_cycles,
                          "same_elf_engines": ["verilator", "gsim"]},
            "ordering": "all_candidates_sealed_then_reveal_then_host_qualification",
            "agent_visibility": "none",
        }
        _exclusive_json(root, "qualification", qualification)
        for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            path.chmod(0o500 if path.is_dir() else 0o400)
        root.chmod(0o500)
        return certificate_path.resolve(), certificate_sha
    except Exception:
        # A partial root is deliberately not reused; resume either adopts a complete sealed receipt or
        # refuses it.  Keeping the bytes makes an interrupted simulator run auditable.
        raise


def load_completed_qualification(
        qualification_root: Path, *, tuning: GATE.CertificateRecord,
        reveal_manifest_sha256: str, reveal_corpus_sha256: str,
        functional_base_sha256: str, gsim_max_cycles: int | None) -> tuple[Path, str]:
    """Adopt a completed uncheckpointed qualification after validating every content address."""
    _validate_gsim_max_cycles(gsim_max_cycles)
    root = Path(qualification_root)
    if root.is_symlink() or not root.is_dir():
        raise QualificationError("qualification root is absent or linked")
    receipts = sorted(root.glob("qualification.*.json"))
    if len(receipts) != 1:
        raise QualificationError("partial qualification has no unique completion receipt")
    receipt_path = _plain_file(receipts[0], label="qualification completion receipt")
    receipt_sha = _sha_file(receipt_path)
    if receipt_path.name != f"qualification.{receipt_sha}.json":
        raise QualificationError("qualification receipt filename is not content-addressed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    execution = receipt.get("execution") or {}
    if (receipt.get("schema") != SCHEMA or receipt.get("status") != "complete"
            or (receipt.get("tuning_certificate") or {}).get("sha256") != tuning.sha256
            or (receipt.get("reveal_manifest") or {}).get("sha256") != reveal_manifest_sha256
            or (receipt.get("reveal_manifest") or {}).get("corpus_sha256") != reveal_corpus_sha256
            or (receipt.get("functional_baseline") or {}).get("sha256")
            != functional_base_sha256
            or execution.get("gsim_max_cycles") != gsim_max_cycles
            or execution.get("same_elf_engines") != ["verilator", "gsim"]):
        raise QualificationError("qualification completion receipt differs from this experiment")
    for capture in receipt.get("captures") or []:
        path = _plain_file(Path(str(capture.get("path") or "")), label="qualification capture")
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise QualificationError("qualification capture is outside its host root") from exc
        if _sha_file(path) != capture.get("sha256"):
            raise QualificationError("qualification capture changed after completion")
        PRODUCER.validate_capture(path, target=tuning.target, pins=tuning.pins)
    extension = receipt.get("extension_certificate") or {}
    certificate_path = _plain_file(
        Path(str(extension.get("path") or "")), label="extension certificate")
    try:
        certificate_path.relative_to(root.resolve())
    except ValueError as exc:
        raise QualificationError("extension certificate is outside its host root") from exc
    certificate_sha = str(extension.get("sha256") or "")
    record = GATE.load_certificate(certificate_path, expected_sha256=certificate_sha)
    if set(record.members) != set(extension.get("workload_sha256") or []):
        raise QualificationError("extension certificate workload set changed after completion")
    return certificate_path, certificate_sha


__all__ = [
    "QualificationError", "RevealedMember", "load_completed_qualification",
    "load_revealed_members", "lower_with_functional_baseline", "qualify_revealed_holdout",
]

#!/usr/bin/env python3
"""Build the exact prelaunch functional-suite GSIM equivalence certificate.

This is a host-only qualification tool.  It derives the public and hidden cohort through the same
descriptor-driven policy as the formal grader, folds duplicate semantic workloads deterministically,
lowers each representative through the immutable functional compiler, and captures one ELF on both
the pinned Verilator and GSIM engines.  Every attempt is append-only.  A resumed invocation adopts
only fully validated captures and starts a new directory for an interrupted case.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import heldout_gsim_qualification as HQUAL
import perf_campaign as CAMPAIGN
import perf_gsim_gate as GATE
import produce_gsim_certificate as PRODUCER
import run_agentic_perf_experiment as ORCH
from merlin.benchharness import hash_tree
from merlin.targetgen.target_experiment import load_target_experiment


SCHEMA = "merlin.functional-gsim-qualification.v1"
POLICY = "formal-public-plus-hidden-admission-distinct-workloads.v1"


class FunctionalQualificationError(RuntimeError):
    """The exact functional cohort could not be qualified without weakening the claim."""


@dataclass(frozen=True)
class WorkloadCase:
    identity: str
    manifest: Path
    manifest_sha256: str
    capsule_names: tuple[str, ...]
    cohorts: tuple[str, ...]


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_file(path: Path, *, label: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FunctionalQualificationError(f"{label} is absent or linked: {path}")
    return path.resolve()


def _write_content_addressed(root: Path, stem: str, document: object) -> tuple[Path, str]:
    payload = _canonical(document)
    digest = _sha_bytes(payload)
    path = root / f"{stem}.{digest}.json"
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FunctionalQualificationError(f"content-addressed evidence is inconsistent: {path}")
        return path.resolve(), digest
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return path.resolve(), digest


def derive_cases(cohort: ORCH.FunctionalGradeCohort) -> tuple[WorkloadCase, ...]:
    """Fold the exact grader-admitted single-ELF envelope by canonical workload identity.

    Whole-model descriptors remain in ``cohort`` and in the formal regrade, where their dynamic tile
    GSIM execution ledger is checked.  They cannot truthfully be represented by this certificate's
    one-member/one-ELF evidence schema, so the orchestrator owns the shared non-model selection.
    """
    grouped: dict[str, dict[str, Any]] = {}
    public = {(capsule.name, str(capsule.manifest)) for capsule in cohort.public}
    for capsule in ORCH._functional_gsim_cases(cohort):
        cohort_name = ("public" if (capsule.name, str(capsule.manifest)) in public else "hidden")
        manifest = _plain_file(Path(capsule.manifest), label=f"{capsule.name} descriptor")
        digest = _sha_file(manifest)
        workload = PRODUCER.derive_workload(manifest)
        identity = GATE.workload_sha256(workload)
        if digest != capsule.manifest_sha256 or identity != capsule.workload_sha256:
            raise FunctionalQualificationError(
                f"canonical cohort descriptor changed while deriving {capsule.name}")
        row = grouped.setdefault(identity, {
            "manifests": [], "names": [], "cohorts": []})
        row["manifests"].append((str(manifest), digest, manifest))
        row["names"].append(capsule.name)
        row["cohorts"].append(cohort_name)
    if not grouped:
        raise FunctionalQualificationError("canonical functional grade cohort is empty")
    cases = []
    for identity, row in sorted(grouped.items()):
        # Representative choice is stable across discovery ordering and does not inspect outputs/timing.
        _path, digest, manifest = min(row["manifests"], key=lambda item: item[0])
        cases.append(WorkloadCase(
            identity=identity, manifest=manifest, manifest_sha256=digest,
            capsule_names=tuple(sorted(set(row["names"]))),
            cohorts=tuple(sorted(set(row["cohorts"])))))
    return tuple(cases)


def _artifacts(certificate: GATE.CertificateRecord) -> PRODUCER.ArtifactPaths:
    return PRODUCER.ArtifactPaths(*(Path(certificate.pins[name]["path"])
                                    for name in ("gsim_firrtl", "verilator_firrtl", "gsim_model",
                                                 "gsim_binary", "verilator_binary")))


def _build_receipt(certificate: GATE.CertificateRecord) -> Path:
    binding = certificate.document.get("build_binding")
    if not isinstance(binding, Mapping):
        raise FunctionalQualificationError("source certificate lost its GSIM build binding")
    path = Path(str(binding.get("path") or ""))
    if not path.is_absolute():
        path = certificate.path.parent / path
    path = _plain_file(path, label="GSIM build receipt")
    if _sha_file(path) != binding.get("sha256"):
        raise FunctionalQualificationError("GSIM build receipt differs from the source certificate")
    PRODUCER.validate_build_receipt(path, pins=certificate.pins)
    return path


def _readonly_baseline(path: Path, expected_sha256: str) -> Path:
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise FunctionalQualificationError(f"functional baseline is absent or linked: {root}")
    for item in (root, *root.rglob("*")):
        if item.is_symlink() or item.stat().st_mode & 0o222:
            raise FunctionalQualificationError(f"functional baseline is linked or writable: {item}")
    if str(hash_tree(root)["sha256"]) != expected_sha256:
        raise FunctionalQualificationError("functional baseline compiler digest differs from its pin")
    return root.resolve()


def _declaration(*, target: Any, descriptor: Path, functional_base: Path,
                 functional_base_sha256: str, source: GATE.CertificateRecord,
                 cohort: ORCH.FunctionalGradeCohort, cases: Sequence[WorkloadCase], timeout: int,
                 workers: int, gsim_max_cycles: int | None, reuse_source_captures: bool) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "policy": POLICY, "target": source.target,
        "target_descriptor": {"path": str(descriptor), "sha256": _sha_file(descriptor)},
        "functional_baseline": {"path": str(functional_base),
                                "sha256": functional_base_sha256},
        "source_certificate": {"path": str(source.path), "sha256": source.sha256,
                               "pins": {name: source.pins[name]["sha256"]
                                        for name in sorted(GATE.REQUIRED_PINS)}},
        "cohort": {"public_source_descriptors": cohort.public_source_count,
                   "public_descriptors": len(cohort.public),
                   "hidden_source_descriptors": cohort.hidden_source_count,
                   "hidden_descriptors": len(cohort.hidden),
                   "same_elf_certificate_descriptors": len(
                       ORCH._functional_gsim_cases(cohort)),
                   "dynamic_model_regrade_descriptors": sum(
                       capsule.kind == "model" for capsule in (*cohort.public, *cohort.hidden)),
                   "distinct_workloads": len(cases)},
        "cases": [{"workload_sha256": case.identity,
                   "representative_manifest": str(case.manifest),
                   "representative_manifest_sha256": case.manifest_sha256,
                   "capsules": list(case.capsule_names), "cohorts": list(case.cohorts)}
                  for case in cases],
        "execution": {"timeout_seconds": timeout, "workers": workers,
                      "gsim_max_cycles": gsim_max_cycles,
                      "reuse_identical_source_captures": reuse_source_captures,
                      "same_elf_engines": [GATE.REFERENCE_ENGINE, GATE.GSIM_ENGINE]},
    }


def _load_declaration(root: Path, expected: Mapping[str, Any]) -> tuple[Path, str]:
    paths = sorted(root.glob("declaration.*.json"))
    if len(paths) != 1:
        raise FunctionalQualificationError("resume root has no unique qualification declaration")
    path = _plain_file(paths[0], label="qualification declaration")
    digest = _sha_file(path)
    if path.name != f"declaration.{digest}.json":
        raise FunctionalQualificationError("qualification declaration filename is not content-addressed")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FunctionalQualificationError("qualification declaration is unreadable") from exc
    if document != expected:
        raise FunctionalQualificationError("resume inputs differ from the sealed declaration")
    return path, digest


def _capture_paths(root: Path, source: GATE.CertificateRecord,
                   expected: set[str]) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in sorted(root.glob("captures/*.json")) + sorted(root.glob("attempts/*/*/capture.*.json")):
        member = PRODUCER.validate_capture(path, target=source.target, pins=source.pins)
        identity = str(member["workload_sha256"])
        if identity not in expected:
            raise FunctionalQualificationError(
                f"qualification root contains an out-of-cohort capture: {identity}")
        selected.setdefault(identity, path.resolve())
    return selected


def _seed_captures(root: Path, source: GATE.CertificateRecord,
                   expected: set[str]) -> set[str]:
    reused: set[str] = set()
    captures = root / "captures"
    captures.mkdir(exist_ok=True)
    for identity in sorted(expected & set(source.members)):
        document = source.members[identity]
        path, _digest = _write_content_addressed(captures, f"seed.{identity}", document)
        member = PRODUCER.validate_capture(path, target=source.target, pins=source.pins)
        if member["workload_sha256"] != identity:
            raise FunctionalQualificationError("source capture changed identity during reuse")
        reused.add(identity)
    return reused


def _next_attempt(root: Path, case: WorkloadCase) -> Path:
    parent = root / "attempts" / case.identity
    parent.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in parent.glob("attempt-*" ):
        if path.is_dir() and path.name[8:].isdigit():
            indices.append(int(path.name[8:]))
    attempt = parent / f"attempt-{(max(indices, default=-1) + 1):03d}"
    attempt.mkdir(mode=0o700)
    _write_content_addressed(attempt, "attempt", {
        "schema": SCHEMA, "workload_sha256": case.identity,
        "manifest": str(case.manifest), "manifest_sha256": case.manifest_sha256})
    return attempt


@contextlib.contextmanager
def _runtime(certificate: GATE.CertificateRecord, max_cycles: int | None, backend: Any | None):
    if backend is not None:
        yield backend
        return
    with HQUAL._pinned_runtime(certificate, gsim_max_cycles=max_cycles) as selected:
        yield selected


def _lower_case(*, functional_base: Path, case: WorkloadCase, attempt: Path, timeout: int,
                target_experiment: Any, lowerer: Callable[..., Path]) -> Path:
    lowered = attempt / "lowered"
    if lowerer is HQUAL.lower_with_functional_baseline:
        member = HQUAL.RevealedMember(
            name=case.capsule_names[0], family="functional", cohort="functional",
            source_dir=case.manifest.parent, manifest=case.manifest,
            workload=PRODUCER.derive_workload(case.manifest), workload_sha256=case.identity)
        policy = CAMPAIGN.package_sandbox_policy(target_experiment, attempt, functional_base)
        with CAMPAIGN.boxed_entrypoints(policy):
            return lowerer(functional_base, member, lowered, timeout)
    return lowerer(functional_base, case, lowered, timeout)


def _capture_case(*, source: GATE.CertificateRecord, artifacts: PRODUCER.ArtifactPaths,
                  case: WorkloadCase, attempt: Path, lowered: Path, timeout: int,
                  backend: Any, capturer: Callable[..., Mapping[str, Any]]) -> Path:
    try:
        document = dict(capturer(
            target=source.target, capsule_manifest=case.manifest, artifact_dir=lowered,
            workdir=attempt / "elf", artifacts=artifacts, timeout=timeout, backend=backend))
        if (document.get("workload_sha256") != case.identity
                or document.get("workload") != PRODUCER.derive_workload(case.manifest)):
            raise FunctionalQualificationError("capture workload differs from its canonical case")
        path, _digest = _write_content_addressed(attempt, "capture", document)
        PRODUCER.validate_capture(path, target=source.target, pins=source.pins)
        return path
    except Exception as exc:
        _write_content_addressed(attempt, "failure", {
            "schema": SCHEMA, "status": "failed", "phase": "same_elf_capture",
            "workload_sha256": case.identity,
            "exception": type(exc).__name__, "message": str(exc)[-2000:]})
        raise


def _completion(root: Path, declaration_sha256: str, source: GATE.CertificateRecord,
                expected: set[str]) -> tuple[Path, str] | None:
    receipts = sorted(root.glob("completion.*.json"))
    if not receipts:
        return None
    if len(receipts) != 1:
        raise FunctionalQualificationError("qualification root has multiple completion receipts")
    receipt_path = _plain_file(receipts[0], label="functional qualification completion")
    receipt_sha = _sha_file(receipt_path)
    if receipt_path.name != f"completion.{receipt_sha}.json":
        raise FunctionalQualificationError("completion receipt filename is not content-addressed")
    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (document.get("schema") != SCHEMA or document.get("status") != "complete"
            or document.get("declaration_sha256") != declaration_sha256
            or (document.get("source_certificate") or {}).get("sha256") != source.sha256):
        raise FunctionalQualificationError("completion receipt differs from this qualification")
    selected = document.get("selected_captures")
    if not isinstance(selected, list) or len(selected) != len(expected):
        raise FunctionalQualificationError("completion receipt lost its exact capture set")
    selected_identities = set()
    for row in selected:
        if not isinstance(row, Mapping):
            raise FunctionalQualificationError("completion receipt has a malformed capture row")
        capture_path = _plain_file(
            Path(str(row.get("path") or "")), label="selected functional capture")
        try:
            capture_path.relative_to(root.resolve())
        except ValueError as exc:
            raise FunctionalQualificationError(
                "selected functional capture is outside its host root") from exc
        if _sha_file(capture_path) != row.get("sha256"):
            raise FunctionalQualificationError("selected functional capture changed after completion")
        member = PRODUCER.validate_capture(
            capture_path, target=source.target, pins=source.pins)
        if member["workload_sha256"] != row.get("workload_sha256"):
            raise FunctionalQualificationError("selected capture identity changed after completion")
        selected_identities.add(str(member["workload_sha256"]))
    if selected_identities != expected:
        raise FunctionalQualificationError("completion receipt captures are not the exact cohort")
    cert = document.get("functional_certificate") or {}
    path = _plain_file(Path(str(cert.get("path") or "")), label="functional GSIM certificate")
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise FunctionalQualificationError(
            "functional GSIM certificate is outside its host root") from exc
    digest = str(cert.get("sha256") or "")
    record = GATE.load_certificate(path, expected_sha256=digest)
    if (set(record.members) != expected
            or set(cert.get("workload_sha256") or []) != expected):
        raise FunctionalQualificationError("completed certificate is not the exact functional cohort")
    return path, digest


def produce_functional_certificate(
        *, descriptor: Path, functional_base: Path, functional_base_sha256: str,
        source_certificate: Path, source_certificate_sha256: str, root: Path,
        timeout: int = 3600, workers: int = 2, gsim_max_cycles: int | None = None,
        reuse_source_captures: bool = True, target_experiment: Any | None = None,
        cohort: ORCH.FunctionalGradeCohort | None = None,
        lowerer: Callable[..., Path] = HQUAL.lower_with_functional_baseline,
        capturer: Callable[..., Mapping[str, Any]] = PRODUCER.capture_case,
        backend: Any | None = None) -> tuple[Path, str]:
    """Create or safely resume the exact public+hidden functional certificate."""
    if (isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0
            or isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0):
        raise FunctionalQualificationError("timeout and workers must be positive integers")
    if (gsim_max_cycles is not None and (isinstance(gsim_max_cycles, bool)
            or not isinstance(gsim_max_cycles, int) or gsim_max_cycles <= 0)):
        raise FunctionalQualificationError("GSIM max cycles must be a positive integer")
    descriptor = _plain_file(Path(descriptor), label="target descriptor")
    functional_base = _readonly_baseline(Path(functional_base), functional_base_sha256)
    source = GATE.load_certificate(
        source_certificate, expected_sha256=source_certificate_sha256)
    _build_receipt(source)
    target_experiment = target_experiment or load_target_experiment(descriptor)
    if getattr(target_experiment, "target", None) != source.target:
        raise FunctionalQualificationError("target descriptor differs from source certificate")
    cohort = cohort or ORCH._functional_grade_cohort(target_experiment)
    cases = derive_cases(cohort)
    expected = {case.identity for case in cases}
    declaration = _declaration(
        target=target_experiment, descriptor=descriptor, functional_base=functional_base,
        functional_base_sha256=functional_base_sha256, source=source, cohort=cohort, cases=cases,
        timeout=timeout, workers=workers, gsim_max_cycles=gsim_max_cycles,
        reuse_source_captures=reuse_source_captures)
    root = Path(root)
    if root.is_symlink():
        raise FunctionalQualificationError("qualification root may not be a symlink")
    if root.exists():
        if not root.is_dir():
            raise FunctionalQualificationError("qualification root is not a directory")
        _declaration_path, declaration_sha = _load_declaration(root, declaration)
    else:
        root.mkdir(parents=True, mode=0o700)
        _declaration_path, declaration_sha = _write_content_addressed(
            root, "declaration", declaration)
    completed = _completion(root, declaration_sha, source, expected)
    if completed is not None:
        return completed
    for directory in (root / "captures", root / "attempts"):
        directory.mkdir(exist_ok=True)
    if reuse_source_captures:
        _seed_captures(root, source, expected)
    selected = _capture_paths(root, source, expected)
    pending = [case for case in cases if case.identity not in selected]
    artifacts = _artifacts(source)
    attempts: list[tuple[WorkloadCase, Path, Path]] = []
    failures: list[tuple[str, str, str]] = []
    for case in pending:
        attempt = _next_attempt(root, case)
        try:
            lowered = _lower_case(
                functional_base=functional_base, case=case, attempt=attempt, timeout=timeout,
                target_experiment=target_experiment, lowerer=lowerer)
            _readonly_baseline(functional_base, functional_base_sha256)
            attempts.append((case, attempt, lowered))
        except Exception as exc:
            _write_content_addressed(attempt, "failure", {
                "schema": SCHEMA, "status": "failed", "phase": "lowering",
                "workload_sha256": case.identity, "exception": type(exc).__name__,
                "message": str(exc)[-2000:]})
            failures.append((case.identity, type(exc).__name__, str(exc)))
    with _runtime(source, gsim_max_cycles, backend) as selected_backend:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="functional-gsim") as pool:
            futures = {pool.submit(
                _capture_case, source=source, artifacts=artifacts, case=case, attempt=attempt,
                lowered=lowered, timeout=timeout, backend=selected_backend,
                capturer=capturer): case for case, attempt, lowered in attempts}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    selected[case.identity] = future.result()
                except Exception as exc:  # every other submitted attempt is still allowed to finish
                    failures.append((case.identity, type(exc).__name__, str(exc)))
    if failures:
        detail = "; ".join(f"{identity}: {kind}: {message[-300:]}"
                           for identity, kind, message in sorted(failures))
        raise FunctionalQualificationError(
            "one or more qualification cases failed; attempts were retained for audit: " + detail)
    selected = _capture_paths(root, source, expected)
    if set(selected) != expected:
        raise FunctionalQualificationError("not every declared functional workload has a capture")
    receipt = _build_receipt(source)
    certificate = PRODUCER.produce_certificate(
        target=source.target, captures=[selected[key] for key in sorted(expected)],
        artifacts=artifacts, build_receipt=receipt)
    certificate_path, certificate_sha = _write_content_addressed(
        root, "functional-certificate", certificate)
    record = GATE.load_certificate(certificate_path, expected_sha256=certificate_sha)
    if set(record.members) != expected:
        raise FunctionalQualificationError("assembled certificate is not the exact functional cohort")
    if any(record.pins[name]["sha256"] != source.pins[name]["sha256"]
           for name in GATE.REQUIRED_PINS):
        raise FunctionalQualificationError("functional certificate changed a source artifact pin")
    completion = {
        "schema": SCHEMA, "status": "complete", "declaration_sha256": declaration_sha,
        "source_certificate": {"path": str(source.path), "sha256": source.sha256},
        "selected_captures": [{"workload_sha256": identity,
                               "path": str(selected[identity]),
                               "sha256": _sha_file(selected[identity])}
                              for identity in sorted(expected)],
        "functional_certificate": {"path": str(certificate_path),
                                   "sha256": certificate_sha,
                                   "workload_sha256": sorted(expected)},
    }
    _write_content_addressed(root, "completion", completion)
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o500 if path.is_dir() else 0o400)
    root.chmod(0o500)
    return certificate_path, certificate_sha


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--functional-base", required=True)
    parser.add_argument("--functional-base-sha256", required=True)
    parser.add_argument("--source-certificate", required=True)
    parser.add_argument("--source-certificate-sha256", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gsim-max-cycles", type=int)
    parser.add_argument("--no-reuse-source-captures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path, digest = produce_functional_certificate(
        descriptor=Path(args.descriptor), functional_base=Path(args.functional_base),
        functional_base_sha256=args.functional_base_sha256,
        source_certificate=Path(args.source_certificate),
        source_certificate_sha256=args.source_certificate_sha256, root=Path(args.root),
        timeout=args.timeout, workers=args.workers, gsim_max_cycles=args.gsim_max_cycles,
        reuse_source_captures=not args.no_reuse_source_captures)
    print(GATE.canonical_json({"path": str(path), "sha256": digest}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

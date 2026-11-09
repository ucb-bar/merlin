"""Byte-verified reading of explicit immutable performance holdout reveals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import gsim_gate as GATE
from . import gsim_workload as WORKLOAD


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


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def plain_file(path: Path, *, label: str) -> Path:
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


def tree_without_manifest(root: Path, manifest: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise QualificationError(f"revealed corpus contains a symlink: {path}")
        if path.is_file() and path != manifest:
            rows.append(
                {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)}
            )
    return {"files": rows, "sha256": sha_bytes(canonical(rows))}


def load_revealed_members(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_corpus_sha256: str | None = None,
    expected_target: str | None = None,
    require_frozen: bool = True,
) -> tuple[RevealedMember, ...]:
    """Validate a v2 reveal and resolve only its explicitly declared member paths."""
    manifest = plain_file(Path(manifest_path), label="revealed holdout manifest")
    manifest_bytes = manifest.read_bytes()
    if expected_manifest_sha256 is not None and sha_bytes(manifest_bytes) != expected_manifest_sha256:
        raise QualificationError("revealed holdout manifest digest changed")
    root = manifest.parent
    if require_frozen:
        for path in (root, *root.rglob("*")):
            if path.is_symlink() or path.stat().st_mode & 0o222:
                raise QualificationError(f"revealed holdout is linked or writable: {path}")
    try:
        document = json.loads(manifest_bytes)
    except (OSError, ValueError) as exc:
        raise QualificationError("revealed holdout manifest is not valid JSON") from exc
    if (
        not isinstance(document, Mapping)
        or document.get("schema_version") != 2
        or document.get("kind") != "generated_performance_holdout_reveal"
    ):
        raise QualificationError("revealed holdout is not the required v2 commit/reveal manifest")
    target = (document.get("domain") or {}).get("target")
    if expected_target is not None and target != expected_target:
        raise QualificationError("revealed holdout target differs from the experiment target")
    actual_tree = tree_without_manifest(root, manifest)
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
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in ("", ".", "..")
            or not isinstance(family, str)
            or not family
            or not isinstance(cohort, str)
            or cohort not in cohorts
        ):
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
        capsule_manifest = plain_file(source / "capsule.yaml", label=f"{name} descriptor")
        try:
            descriptor = yaml.safe_load(capsule_manifest.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise QualificationError(f"revealed member descriptor is unreadable: {name}") from exc
        if not isinstance(descriptor, Mapping) or descriptor.get("name") != name:
            raise QualificationError(f"revealed descriptor identity differs: {name}")
        coordinates = (row.get("M"), row.get("N"), row.get("K"))
        workload = WORKLOAD.derive_workload(capsule_manifest)
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
        members.append(RevealedMember(name, family, cohort, source, capsule_manifest, workload, identity))
    for cohort, declaration in cohorts.items():
        count = declaration.get("member_count") if isinstance(declaration, Mapping) else None
        if cohort_counts.get(str(cohort), 0) != count:
            raise QualificationError(f"revealed cohort count differs: {cohort}")
    return tuple(members)

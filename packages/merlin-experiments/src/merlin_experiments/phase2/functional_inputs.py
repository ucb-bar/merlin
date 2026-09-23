"""Frozen Phase 1 input admission and private/public transport for Phase 2.

Verifies existing host provenance and reifies immutable grants, without launching
agents or substituting live input data.
"""

from __future__ import annotations

import json
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.common.paths import repo_root
from merlin.targetgen.sandbox import bwrap as BW

from . import campaign as PC
from . import contracts as CONTRACTS
from .broker_evidence import _is_sha256
from .contracts import StageGateError
from .contracts import canonical_json as _canonical_json
from .contracts import sha256_file as _sha256_file


@dataclass(frozen=True)
class StageFunctionalRun:
    """Current campaign verdict plus the authoring inputs it did not expose."""

    run_dir: Path
    submission_dir: Path
    run_id: str
    digest: str
    public_capsules: int
    hidden_capsules: int
    public_score: dict[str, Any]
    hidden_score: dict[str, Any]
    frozen_at: str
    bundle_input_snapshot: dict[str, Any]
    model_host_lane_snapshot: dict[str, Any]
    model_host_package: Path


@dataclass(frozen=True)
class FrozenGrant:
    declared_path: str
    destination: Path
    source: Path
    source_sha256: str


@dataclass(frozen=True)
class FrozenFunctionalInputs:
    root: Path
    marker: Path
    marker_sha256: str
    content_sha256: str
    grants: tuple[FrozenGrant, ...]
    public_marker: Path | None = None
    public_marker_sha256: str | None = None
    public_content_sha256: str | None = None
    host_provenance: dict[str, Any] | None = None


def _safe_relative(value: object, *, label: str) -> Path:
    if not isinstance(value, str):
        raise StageGateError(f"{label} must be repository-relative")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise StageGateError(f"{label} must be a safe repository-relative path")
    return path


def _require_read_only_tree(root: Path, *, label: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise StageGateError(f"{label} is absent or linked: {root}")
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            raise StageGateError(f"{label} contains a symlink: {path}")
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise StageGateError(f"{label} is writable: {path}")


def _functional_input_snapshot(base: PC.FunctionalRun) -> StageFunctionalRun:
    """Join the current functional gate to its sealed authoring/host snapshot."""
    environment = CONTRACTS.mapping_file(base.run_dir / "environment.yaml", yaml_file=True)
    snapshot = environment.get("bundle_input_snapshot")
    host = environment.get("model_host_lane_snapshot")
    if (
        not isinstance(snapshot, Mapping)
        or snapshot.get("version") != 4
        or not _is_sha256(snapshot.get("content_sha256"))
        or not isinstance(host, Mapping)
        or host.get("run_snapshot") != dict(snapshot)
    ):
        raise StageGateError(
            "functional verdict requires exact V4 authoring and host-lane ownership; create a newly frozen run"
        )
    if snapshot["version"] == 4:
        PC.verify_private_input_snapshot(base.run_dir, environment)
    raw_root = snapshot.get("path")
    if not isinstance(raw_root, str) or not Path(raw_root).is_absolute():
        raise StageGateError("functional input snapshot path is not absolute")
    root = Path(raw_root)
    _require_read_only_tree(root, label="functional input snapshot")
    marker = CONTRACTS.mapping_file(root / "snapshot.json")
    for attribute in ("content_sha256", "n_files", "n_bytes"):
        if marker.get(attribute) != snapshot.get(attribute):
            raise StageGateError("functional input snapshot marker disagrees with the run record")
    package_rel = _safe_relative(host.get("package"), label="model host-lane package")
    repo = (root / "repo").resolve(strict=True)
    package = (repo / package_rel).resolve(strict=True)
    try:
        package.relative_to(repo)
    except ValueError as exc:
        raise StageGateError("model host-lane package escapes the frozen repository") from exc
    if package.is_symlink() or not package.is_dir():
        raise StageGateError("model host-lane package is absent or linked")
    _require_read_only_tree(package, label="model host-lane package")
    if host.get("resolved_package") != str(package):
        raise StageGateError("model host-lane record names different frozen bytes")
    package_record = hash_tree(package)
    if host.get("package_sha256") != package_record.get("sha256") or host.get("n_files") != package_record.get(
        "n_files"
    ):
        raise StageGateError("model host-lane digest disagrees with its frozen package")
    return StageFunctionalRun(
        base.run_dir,
        base.submission_dir,
        base.run_id,
        base.digest,
        base.public_capsules,
        base.hidden_capsules,
        base.public_score,
        base.hidden_score,
        base.frozen_at,
        dict(snapshot),
        dict(host),
        package,
    )


def inspect_stage_functional_run(
    run_root: Path, run_id: str, expected_digest: str, *, waive: frozenset[str] | tuple[str, ...] | None = None
) -> StageFunctionalRun:
    """The stage's view of the functional baseline.

    ``waive`` is passed straight through to :func:`campaign.inspect_functional_run`, which decides
    what may be waived at all -- integrity predicates refuse the waiver itself. Threading it rather
    than re-deciding here keeps ONE place that knows which gaps are acceptable; a second opinion in
    this file is how the two would drift.
    """
    return _functional_input_snapshot(PC.inspect_functional_run(run_root, run_id, expected_digest, waive=waive))


def verify_functional_host_lane_snapshot(host: Mapping[str, Any]) -> None:
    run_snapshot = host.get("run_snapshot")
    if not isinstance(run_snapshot, Mapping):
        raise StageGateError("model host-lane record omits its run snapshot")
    raw_root = run_snapshot.get("path")
    if not isinstance(raw_root, str):
        raise StageGateError("model host-lane run snapshot has no path")
    root = Path(raw_root)
    _require_read_only_tree(root, label="functional input snapshot")
    package = Path(str(host.get("resolved_package") or ""))
    if package.is_symlink() or not package.is_dir():
        raise StageGateError("model host-lane package is absent or linked")
    if hash_tree(package).get("sha256") != host.get("package_sha256"):
        raise StageGateError("model host-lane package digest changed")


def load_frozen_functional_inputs(
    functional: StageFunctionalRun, *, public_manifest_path: Path | None = None
) -> FrozenFunctionalInputs:
    """Reify V4 grants with captured support ownership and private host provenance."""
    root = Path(functional.bundle_input_snapshot["path"])
    marker = root / "snapshot.json"
    if root.is_symlink() or not root.is_dir() or marker.is_symlink() or not marker.is_file():
        raise StageGateError("functional input snapshot or marker is absent")
    document = json.loads(marker.read_text(encoding="utf-8"))
    if document.get("version") != 4:
        raise StageGateError("verified execution requires bundle snapshot V4 ownership; create a newly frozen run")
    if document.get("content_sha256") != functional.bundle_input_snapshot.get("content_sha256"):
        raise StageGateError("functional input snapshot marker identity changed")
    if public_manifest_path is None:
        raise StageGateError("V4 functional inputs require a separate public manifest projection")
    environment_path = functional.run_dir / "environment.yaml"
    environment = CONTRACTS.mapping_file(environment_path, yaml_file=True)
    try:
        private = PC.verify_private_input_snapshot(functional.run_dir, environment)
    except PC.CampaignGateError as exc:
        raise StageGateError(str(exc)) from exc
    if environment["bundle_input_snapshot"] != functional.bundle_input_snapshot:
        raise StageGateError("functional snapshot identity changed after admission")
    records = document.get("grants")
    if not isinstance(records, list) or not records:
        raise StageGateError("functional input snapshot has no exact grant table")
    recorded_repo = Path(str(document.get("repo") or ""))
    if not recorded_repo.is_absolute() or ".." in recorded_repo.parts:
        raise StageGateError("functional input snapshot has no safe recorded repo root")
    grants: list[FrozenGrant] = []
    resolved_root = root.resolve(strict=True)
    repo = repo_root().absolute()
    for row in records:
        if not isinstance(row, Mapping):
            raise StageGateError("functional input snapshot contains a malformed grant")
        declared = str(row.get("path") or "")
        destination = Path(str(row.get("destination") or ""))
        relative = Path(str(row.get("snapshot") or ""))
        declared_path = Path(declared)
        if (
            not declared
            or ".." in declared_path.parts
            or not destination.is_absolute()
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise StageGateError("functional input snapshot contains an unsafe grant")
        # The existing host marker pins the original exact destination. OOT
        # grants stay absolute; only original-repo grants relocate with code.
        if destination.is_relative_to(recorded_repo):
            destination = repo / destination.relative_to(recorded_repo)
        source = root / relative
        try:
            resolved_source = source.resolve(strict=True)
        except OSError as exc:
            raise StageGateError(f"functional frozen grant is absent: {declared}") from exc
        if resolved_source != resolved_root and resolved_root not in resolved_source.parents:
            raise StageGateError(f"functional frozen grant escapes snapshot: {declared}")
        if source.is_symlink():
            raise StageGateError(f"functional frozen grant is linked: {declared}")
        digest = CONTRACTS.exact_tree_record(source)["sha256"] if source.is_dir() else _sha256_file(source)
        grants.append(FrozenGrant(declared, destination, source, str(digest)))
    host_source = Path(functional.model_host_package)
    if (
        host_source.is_symlink()
        or not host_source.is_dir()
        or resolved_root not in host_source.resolve(strict=True).parents
    ):
        raise StageGateError("functional frozen host lane is absent from the input snapshot")
    grants.append(
        FrozenGrant(
            "__model_host_lane_snapshot__",
            host_source,
            host_source,
            str(CONTRACTS.exact_tree_record(host_source)["sha256"]),
        )
    )
    public_marker = public_marker_digest = public_digest = provenance = None
    if private is not None:
        private_sources = [root / row["snapshot"] for row in document["host_records"]]
        support = BW.snapshot_support_surfaces(private["workspace"], document)
        public_rows = []
        for grant in grants:
            rows = []
            members = sorted(grant.source.rglob("*")) if grant.source.is_dir() else [grant.source]
            for member in members:
                if any(member == hidden or hidden in member.parents for hidden in private_sources):
                    continue
                if not BW.snapshot_public_member(member, grant.source, support):
                    continue
                if member.is_file():
                    rows.append(
                        [str(member.relative_to(grant.source)) if grant.source.is_dir() else ".", _sha256_file(member)]
                    )
            public_rows.append(
                {
                    "path": grant.declared_path,
                    "destination": str(grant.destination),
                    "sha256": CONTRACTS.document_sha256(rows),
                }
            )
        public_digest = CONTRACTS.document_sha256(public_rows)
        projection = {
            "version": 1,
            "kind": "public_functional_input_projection",
            "content_sha256": public_digest,
            "grants": public_rows,
        }
        public_marker = Path(public_manifest_path)
        public_marker.parent.mkdir(parents=True, exist_ok=True)
        with public_marker.open("xb") as stream:
            stream.write(_canonical_json(projection))
        public_marker.chmod(0o444)
        public_marker_digest = _sha256_file(public_marker)
        provenance = {
            "environment": str(environment_path),
            "environment_sha256": _sha256_file(environment_path),
            "bundle_manifest": str(private["bundle_path"]),
            "bundle_manifest_sha256": private["bundle_sha256"],
        }
    return FrozenFunctionalInputs(
        root,
        marker,
        _sha256_file(marker),
        str(document["content_sha256"]),
        tuple(grants),
        public_marker,
        public_marker_digest,
        public_digest,
        provenance,
    )


def _verify_private_functional_provenance(provenance: Mapping[str, Any]) -> dict:
    environment_path = Path(str(provenance.get("environment") or ""))
    environment = CONTRACTS.mapping_file(environment_path, yaml_file=True)
    if _sha256_file(environment_path) != provenance.get("environment_sha256"):
        raise StageGateError("functional host provenance changed after admission")
    try:
        verified = PC.verify_private_input_snapshot(environment_path.parent, environment)
    except PC.CampaignGateError as exc:
        raise StageGateError(str(exc)) from exc
    if str(verified["bundle_path"]) != provenance.get("bundle_manifest") or verified["bundle_sha256"] != provenance.get(
        "bundle_manifest_sha256"
    ):
        raise StageGateError("functional original bundle identity changed")
    return verified


def _private_functional_surfaces(argv: list[str], inputs: FrozenFunctionalInputs | None) -> list:
    if inputs is None:
        return []
    if inputs.host_provenance is None:
        raise StageGateError("verified execution requires bound V4 functional inputs; create a newly frozen run")
    verified = _verify_private_functional_provenance(inputs.host_provenance)
    if (
        inputs.public_marker is None
        or inputs.public_marker.is_symlink()
        or _sha256_file(inputs.public_marker) != inputs.public_marker_sha256
    ):
        raise StageGateError("functional public manifest projection changed")
    surfaces = BW.host_input_surfaces(
        argv, verified["workspace"], verified["bundle"], repo=verified["repo"], grant_repo=repo_root()
    )
    return [
        *surfaces,
        *BW.snapshot_support_surfaces(verified["workspace"], verified["marker"]),
        *BW.private_file_surfaces(argv, [Path(inputs.host_provenance["environment"]), verified["bundle_path"]]),
    ]


def frozen_grant_mounts(inputs: FrozenFunctionalInputs) -> list[str]:
    argv: list[str] = []
    for grant in inputs.grants:
        argv += ["--ro-bind", str(grant.source), str(grant.destination)]
    return argv


def _frozen_path_for_destination(inputs: FrozenFunctionalInputs, destination: Path) -> Path:
    destination = destination.absolute()
    candidates: list[tuple[int, Path]] = []
    for grant in inputs.grants:
        if destination == grant.destination or grant.destination in destination.parents:
            candidates.append((len(grant.destination.parts), grant.source / destination.relative_to(grant.destination)))
    if not candidates:
        raise StageGateError(f"functional snapshot did not grant required path: {destination}")
    frozen = max(candidates, key=lambda row: row[0])[1]
    if frozen.is_symlink() or not frozen.exists():
        raise StageGateError(f"functional frozen required path is absent: {destination}")
    return frozen

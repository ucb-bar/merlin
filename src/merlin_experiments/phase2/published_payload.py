"""Select only the graded compiler bytes from an independently published tree.

Publication metadata is a locator, never a replacement for the Phase 1 grade.
The caller must first admit the exact frozen functional run. Extra publication
files are deliberately excluded from the performance compiler snapshot.
"""

from __future__ import annotations

import hashlib
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from merlin.benchharness import hash_tree
from merlin.targetgen import package_records
from merlin_experiments.phase2.contracts import StageGateError


@dataclass(frozen=True)
class PublishedPayload:
    root: Path
    target: str
    graded_sha256: str
    source_inventory: dict
    provenance_sha256: str

    def identity(self) -> dict:
        return {
            "root": str(self.root),
            "source_payload_sha256": self.source_inventory["sha256"],
            "provenance_sha256": self.provenance_sha256,
            "graded_submission_sha256": self.graded_sha256,
        }


def _provenance(root: Path) -> tuple[dict, bytes]:
    path = root / ".merlin/provenance.yaml"
    if (
        path.parent.is_symlink()
        or path.is_symlink()
        or not path.is_file()
        or not stat.S_ISREG(path.lstat().st_mode)
    ):
        raise StageGateError("published compiler lacks an ordinary Merlin provenance file")
    raw = path.read_bytes()
    try:
        document = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise StageGateError("published compiler provenance is invalid YAML") from exc
    if not isinstance(document, dict):
        raise StageGateError("published compiler provenance must be a mapping")
    return document, raw


def selection_paths(root: Path) -> dict[str, str]:
    """Return the exact published source members for orchestration input pins."""
    root = Path(root).absolute()
    try:
        package_records.safe_path(root)
        provenance, _ = _provenance(root)
    except ValueError as exc:
        raise StageGateError(str(exc)) from exc
    inventory = provenance.get("source_payload")
    if not package_records.valid_inventory(inventory):
        raise StageGateError("published compiler has no valid source-payload inventory")
    paths = {"phase2:published:provenance": str(root / ".merlin/provenance.yaml")}
    for index, member in enumerate(inventory["members"]):
        if member["path"].split("/")[0] in {".merlin", ".git"}:
            raise StageGateError("publication metadata cannot be a compiler source member")
        if member["kind"] == "file":
            paths[f"phase2:published:member:{index:04d}"] = str(root / member["path"])
    return paths


def inspect(functional, root: Path, *, target: str) -> PublishedPayload:
    """Prove that the relocated source members equal a fully graded submission."""
    root = Path(root).absolute()
    try:
        package_records.safe_path(root)
        provenance, raw = _provenance(root)
        frozen = package_records.payload_inventory(functional.submission_dir)
    except (OSError, ValueError) as exc:
        raise StageGateError(f"published compiler selection is invalid: {exc}") from exc
    source = provenance.get("source_payload")
    if package_records.valid_inventory(source) and any(
        member["path"].split("/")[0] in {".merlin", ".git"} for member in source["members"]
    ):
        raise StageGateError("publication metadata cannot be a compiler source member")
    if (
        not package_records.valid_inventory(source)
        or source != frozen
        or provenance.get("target") != target
        or provenance.get("provider_role") != "candidate_compiler"
    ):
        raise StageGateError("published compiler is not the exact graded Phase 1 source payload")
    if hash_tree(functional.submission_dir)["sha256"] != functional.digest:
        raise StageGateError("functional submission changed after Phase 1 admission")
    if root.stat().st_mode & 0o111 != source["root_executable"]:
        raise StageGateError("published compiler root permissions differ from graded payload")
    for member in source["members"]:
        path = root / member["path"]
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
            raise StageGateError("published compiler source contains a symlink")
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise StageGateError("published compiler source member is absent") from exc
        if mode & 0o111 != member["executable"]:
            raise StageGateError("published compiler source permissions changed")
        if member["kind"] == "directory":
            if not stat.S_ISDIR(mode):
                raise StageGateError("published compiler source directory changed kind")
        else:
            if not stat.S_ISREG(mode):
                raise StageGateError("published compiler source file changed kind")
            with path.open("rb") as stream:
                observed = hashlib.file_digest(stream, "sha256").hexdigest()
            if observed != member["sha256"]:
                raise StageGateError("published compiler source bytes differ from graded payload")
    return PublishedPayload(root, target, functional.digest, source, hashlib.sha256(raw).hexdigest())


def materialize(functional, published: PublishedPayload, destination: Path) -> Path:
    """Copy only selected source members into an immutable Phase 2 workspace."""
    from merlin_experiments.phase2.campaign import materialize_readonly_tree

    current = inspect(functional, published.root, target=published.target)
    if current != published:
        raise StageGateError("published compiler identity changed before materialization")
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise StageGateError("published compiler destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="published-source-", dir=destination.parent) as temporary:
        projection = Path(temporary)
        directories = []
        try:
            for member in published.source_inventory["members"]:
                source = published.root / member["path"]
                output = projection / member["path"]
                if member["kind"] == "directory":
                    output.mkdir(parents=True, exist_ok=True)
                    directories.append((output, member["executable"]))
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, output, follow_symlinks=False)
            for path, executable in reversed(directories):
                path.chmod(0o400 | executable)
            projection.chmod(0o400 | published.source_inventory["root_executable"])
            if package_records.payload_inventory(projection) != published.source_inventory:
                raise StageGateError("published compiler projection differs from graded payload")
            if hash_tree(projection)["sha256"] != functional.digest:
                raise StageGateError("published compiler projection differs from functional grade")
            materialize_readonly_tree(projection, destination)
        finally:
            for path, _ in sorted(directories, key=lambda row: len(row[0].parts), reverse=True):
                if path.exists():
                    path.chmod(0o700)
            projection.chmod(0o700)
    if package_records.payload_inventory(destination) != published.source_inventory:
        raise StageGateError("immutable Phase 2 compiler snapshot differs from published source")
    return destination


def verify_snapshot(functional, published: PublishedPayload, destination: Path) -> Path:
    """Recheck the immutable projected payload when a checkpoint resumes."""
    try:
        observed = package_records.payload_inventory(destination)
    except (OSError, ValueError) as exc:
        raise StageGateError(f"published compiler snapshot is invalid: {exc}") from exc
    if observed != published.source_inventory or hash_tree(destination)["sha256"] != functional.digest:
        raise StageGateError("published compiler snapshot changed across resume")
    return Path(destination)

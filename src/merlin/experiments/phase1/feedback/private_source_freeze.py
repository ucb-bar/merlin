"""Run-owned copies of the exact authored sources selected by a private model spec.

The exported Phase 0 manifest supplies both original path and semantic role.
Content equality alone never selects an alternate source.  A fresh run checks
the complete export; resume checks its already frozen manifest and copies.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from merlin.experiments.corpus.preparation import copy_input
from merlin.experiments.phase0.evidence import load_exported_evidence
from merlin.runtime.modelinputs import file_sha256

SCHEMA = "merlin.phase1.private_source_freeze.v1"
_FIELDS = ("software_spec", "host_capabilities")


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _ordinary_file(path: Path, *, expected_sha256: str) -> bytes:
    if path.is_symlink() or not path.is_file() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("frozen private source is absent or indirect")
    raw = path.read_bytes()
    if _digest(raw) != expected_sha256:
        raise ValueError("frozen private source bytes changed")
    return raw


def _spec(spec: Path, target: str) -> tuple[dict, list[dict]]:
    if spec.is_symlink() or not spec.is_file() or any(parent.is_symlink() for parent in spec.parents):
        raise ValueError("operator-private source specification is absent or indirect")
    document = yaml.safe_load(spec.read_bytes())
    rows = document.get("models") if isinstance(document, Mapping) else None
    if (
        not isinstance(document, dict)
        or document.get("schema") != "merlin.phase1.private_full_models.v1"
        or document.get("target") != target
        or not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("operator-private source freeze has no valid model declaration")
    return document, rows


def _pins(rows: list[dict]) -> tuple[dict[tuple[str, str, str], dict], dict[str, str]]:
    wanted: dict[tuple[str, str, str], dict] = {}
    owned_bytes: dict[tuple[str, str], str] = {}
    archives: dict[str, str] = {}
    for row in rows:
        root = row.get("recipe_derivation_root")
        manifest_sha = row.get("recipe_derivation_manifest_sha256")
        if (
            not isinstance(root, str)
            or not Path(root).is_absolute()
            or ".." in Path(root).parts
            or not _is_sha256(manifest_sha)
        ):
            raise ValueError("private source has no pinned derivation archive")
        if root in archives and archives[root] != manifest_sha:
            raise ValueError("private source archive has conflicting manifest pins")
        archives[root] = manifest_sha
        for field in _FIELDS:
            original, digest = row.get(field), row.get(field + "_sha256")
            if (
                not isinstance(original, str)
                or not Path(original).is_absolute()
                or ".." in Path(original).parts
                or not _is_sha256(digest)
            ):
                raise ValueError("private authored source lacks an absolute path and digest")
            owner = (original, field)
            if owner in owned_bytes and owned_bytes[owner] != digest:
                raise ValueError("one authored source has conflicting model byte pins")
            owned_bytes[owner] = digest
            key = (original, field, root)
            pin = {
                "original_path": original,
                "field": field,
                "sha256": digest,
                "archive_root": root,
                "manifest_sha256": manifest_sha,
            }
            if key in wanted and wanted[key] != pin:
                raise ValueError("private authored source has conflicting model pins")
            wanted[key] = pin
    return wanted, archives


def _selected_source(manifest: Mapping[str, Any], pin: Mapping[str, str]) -> dict:
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError("private derivation has no source ownership inventory")
    candidates = [
        source
        for source in sources
        if isinstance(source, dict)
        and source.get("source") == pin["original_path"]
        and source.get("sha256") == pin["sha256"]
        and (
            source.get("role") == "software-spec"
            if pin["field"] == "software_spec"
            else isinstance(source.get("role"), str)
            and source["role"].startswith("host-capability-spec:")
            and bool(source["role"].partition(":")[2])
        )
    ]
    if len(candidates) != 1:
        raise ValueError("private authored source has no unique original-path/role/content owner")
    source = candidates[0]
    member = source.get("path")
    if (
        not isinstance(member, str)
        or not member.startswith("software/source-snapshots/")
        or Path(member).is_absolute()
        or ".." in Path(member).parts
        or not isinstance(source.get("size_bytes"), int)
    ):
        raise ValueError("private source archive member is malformed")
    artifact = (manifest.get("artifacts") or {}).get(member)
    if not isinstance(artifact, dict) or artifact != {"sha256": pin["sha256"], "size_bytes": source["size_bytes"]}:
        raise ValueError("private source is not a verified archive member")
    return source


def stage(spec: str | Path, root: str | Path, *, target: str) -> dict:
    """Freeze only the declared authored sources from verified export ownership."""
    spec, root = Path(spec).absolute(), Path(root).absolute()
    _, rows = _spec(spec, target)
    wanted, archives = _pins(rows)
    if root.exists():
        raise ValueError("private source freeze already exists; resume must verify its record")
    archive_rows = []
    manifests = {}
    for origin, digest in sorted(archives.items()):
        original_manifest = Path(origin) / "evidence-manifest.json"
        raw = _ordinary_file(original_manifest, expected_sha256=digest)
        evidence = load_exported_evidence(origin)
        if evidence.target != target:
            raise ValueError("private source derivation names another target")
        manifest = json.loads(raw)
        manifests[origin] = manifest
        destination = root / f"manifest-{digest}.json"
        if not destination.exists():
            copy_input(original_manifest, destination, private=True, expected_sha256=digest)
            destination.chmod(0o400)
        archive_rows.append({"original_root": origin, "manifest_sha256": digest, "frozen_path": str(destination)})
    source_rows = []
    for pin in sorted(wanted.values(), key=lambda item: (item["original_path"], item["field"], item["archive_root"])):
        source = _selected_source(manifests[pin["archive_root"]], pin)
        source_path = Path(pin["archive_root"]) / source["path"]
        destination = root / f"{pin['field']}-{pin['sha256']}.bin"
        if not destination.exists():
            copy_input(source_path, destination, private=True, expected_sha256=pin["sha256"])
            destination.chmod(0o400)
        source_rows.append({**pin, "role": source["role"], "member": source["path"], "frozen_path": str(destination)})
    root.chmod(0o500)
    record = {
        "schema": SCHEMA,
        "target": target,
        "spec_sha256": _digest(spec.read_bytes()),
        "root": str(root),
        "archives": archive_rows,
        "sources": source_rows,
    }
    verify(spec, record, root=root, target=target)
    return record


def verify(
    spec: str | Path, record: Mapping[str, Any], *, root: str | Path, target: str
) -> dict[tuple[str, str], Path]:
    """Resolve a run's recorded copies without consulting live authored files."""
    spec, root = Path(spec).absolute(), Path(root).absolute()
    _, rows = _spec(spec, target)
    wanted, archives = _pins(rows)
    if (
        not isinstance(record, Mapping)
        or set(record) != {"schema", "target", "spec_sha256", "root", "archives", "sources"}
        or record.get("schema") != SCHEMA
        or record.get("target") != target
        or record.get("spec_sha256") != _digest(spec.read_bytes())
        or record.get("root") != str(root)
        or not isinstance(record.get("archives"), list)
        or not isinstance(record.get("sources"), list)
        or root.is_symlink()
        or not root.is_dir()
    ):
        raise ValueError("frozen private authored-source record is absent or changed")
    expected_archives = [
        {"original_root": origin, "manifest_sha256": digest, "frozen_path": str(root / f"manifest-{digest}.json")}
        for origin, digest in sorted(archives.items())
    ]
    if record["archives"] != expected_archives:
        raise ValueError("frozen private source archive mapping changed")
    manifests = {}
    for archive in expected_archives:
        raw = _ordinary_file(Path(archive["frozen_path"]), expected_sha256=archive["manifest_sha256"])
        manifest = json.loads(raw)
        if manifest.get("schema") != "phase0_evidence_v1" or manifest.get("target") != target:
            raise ValueError("frozen private source archive is malformed")
        manifests[archive["original_root"]] = manifest
    expected_sources = []
    resolved = {}
    for pin in sorted(wanted.values(), key=lambda item: (item["original_path"], item["field"], item["archive_root"])):
        source = _selected_source(manifests[pin["archive_root"]], pin)
        destination = root / f"{pin['field']}-{pin['sha256']}.bin"
        raw = _ordinary_file(destination, expected_sha256=pin["sha256"])
        if len(raw) != source["size_bytes"]:
            raise ValueError("frozen private authored source length changed")
        expected_sources.append(
            {**pin, "role": source["role"], "member": source["path"], "frozen_path": str(destination)}
        )
        resolved[(pin["original_path"], pin["field"])] = destination
    if record["sources"] != expected_sources:
        raise ValueError("frozen private authored-source mapping changed")
    return resolved


def resolve_optional(
    spec: Path, record: Mapping[str, Any] | None, root: str | Path | None, *, target: str
) -> dict[tuple[str, str], Path] | None:
    """Resolve only a complete run-owned record; retain legacy diagnostic reads separately."""
    if record is None:
        if root is not None:
            raise ValueError("private authored source root has no frozen record")
        return None
    if root is None:
        raise ValueError("private authored source freeze has no run-owned root")
    return verify(spec, record, root=root, target=target)


def pinned_file(spec: Path, value: Any, digest: Any) -> Path:
    """Read an exact path/hash-pinned private input in both build and diagnostic modes."""
    if not isinstance(value, str) or not value or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("private model input needs a path and SHA256")
    path = Path(value)
    path = path if path.is_absolute() else spec.parent / path
    if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
        raise ValueError(f"private model input is absent, indirect, or changed: {path}")
    return path.resolve(strict=True)


def authored_file(
    spec: Path, row: Mapping[str, Any], field: str, resolved: Mapping[tuple[str, str], Path] | None
) -> Path:
    """Select a recorded run-owned copy, not any same-hash ambient authored file."""
    if resolved is None:
        return pinned_file(spec, row.get(field), row.get(field + "_sha256"))
    source = row.get(field)
    path = resolved.get((source, field)) if isinstance(source, str) else None
    digest = row.get(field + "_sha256")
    if path is None or not _is_sha256(digest):
        raise ValueError("frozen authored source differs from the operator-private declaration")
    _ordinary_file(path, expected_sha256=digest)
    return path


def binding(spec: Path, record: Mapping[str, Any], *, root: Path, target: str) -> dict[str, str]:
    """Report the exact verified run-owned source inventory, never a caller claim."""
    verify(spec, record, root=root, target=target)
    return {
        "schema": SCHEMA,
        "spec_sha256": _digest(spec.read_bytes()),
        "record_sha256": _digest(json.dumps(dict(record), sort_keys=True, separators=(",", ":")).encode()),
        "root": str(root.absolute()),
    }


def claim_binding(
    spec: Path, record: Mapping[str, Any] | None, root: str | Path | None, *, target: str
) -> dict[str, dict[str, str]]:
    """Recheck complete frozen ownership after builds, before publishing any result."""
    if record is None:
        return {}
    if root is None:
        raise ValueError("private authored source freeze has no run-owned root")
    return {"authored_source_freeze": binding(spec, record, root=Path(root), target=target)}


def valid_binding(value: Any, spec_sha256: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"schema", "spec_sha256", "record_sha256", "root"}
        and value.get("schema") == SCHEMA
        and _is_sha256(spec_sha256)
        and value.get("spec_sha256") == spec_sha256
        and isinstance(value.get("root"), str)
        and Path(value["root"]).is_absolute()
        and _is_sha256(value.get("record_sha256"))
    )

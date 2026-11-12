"""Positively reviewed reference origins staged before fresh release admission.

Historical v1 manifests have byte pins but no publication/permission identity.
They remain historical evidence; fresh preparation refuses to restage them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.development.backend.feedback import reference_baselines as baselines
from merlin.development.backend.reference_inputs import (
    ReferenceFile,
    ReferenceOrigin,
    _ordinary,
    _relative,
    stage_references,
)
from merlin.experiments.definition import SpecError

SCHEMA = "operator.allowed_reference_sources.v2"


def stage(manifest: Path, destination: Path) -> dict:
    """The normal release path: exact origins plus individually selected files.

    The operator writes this reviewed manifest explicitly. No automatic v1
    conversion treats an arbitrary source directory or archive as public.
    Containers are relative to the manifest, and are never themselves exported.
    """
    try:
        manifest = _ordinary(manifest.expanduser().absolute())
        if not manifest.is_file() or manifest.stat().st_size > 4 << 20:
            raise ValueError("bounded regular reference review manifest required")
        original = manifest.read_bytes()
        document = json.loads(original)
        if (
            type(document) is not dict
            or document.get("schema") != SCHEMA
            or set(document) != {"schema", "review_id", "origins"}
            or type(document["origins"]) is not list
            or not document["origins"]
        ):
            raise ValueError("fresh reference preparation requires v2 positive origin review; v1 cannot be restaged")
        origins = []
        for item in document["origins"]:
            common = {"name", "kind", "container", "identity", "license", "allowed_purpose", "files"}
            if type(item) is not dict:
                raise ValueError("reference origin must be an explicit reviewed object")
            expected = common | ({"public_url"} if item.get("kind") == "public_repository" else {"user_authorization"})
            if set(item) - {"content_kind"} != expected or type(item["files"]) is not list:
                raise ValueError("reference origin requires exact publication/permission and source fields")
            relative = item["container"]
            if relative == "." and item["kind"] == "public_repository":
                container = manifest.parent
            else:
                container = manifest.parent / _relative(relative)
            files = []
            for member in item["files"]:
                if type(member) is not dict or set(member) not in (
                    {"member", "destination", "sha256"},
                    {"member", "destination", "sha256", "size_bytes"},
                ):
                    raise ValueError("reference origin requires individual source/destination/hash pins")
                files.append(ReferenceFile(**member))
            origins.append(
                ReferenceOrigin(
                    item["name"],
                    item["kind"],
                    container,
                    item["identity"],
                    item["license"],
                    tuple(files),
                    public_url=item.get("public_url", ""),
                    user_authorization=item.get("user_authorization", ""),
                    allowed_purpose=item["allowed_purpose"],
                    content_kind=item.get("content_kind", "source_text"),
                )
            )
        selected = stage_references(tuple(origins), destination, review_id=document["review_id"])
        if manifest.read_bytes() != original:
            raise ValueError("reference review changed during staging")
        staged_baselines = baselines.staged_file(destination)
        if staged_baselines is not None:
            # Performance baselines are numbers and labels only: refuse a file that carries anything else.
            try:
                baselines.validate(json.loads(staged_baselines.read_text(encoding="utf-8")))
            except ValueError as exc:
                raise SpecError(f"performance baselines member is malformed: {exc}") from exc
        record = destination / "reference-manifest.json"
        record.write_bytes(original)
        for leaf in destination.rglob("*"):
            if leaf.is_file():
                leaf.chmod(0o444)
        return {
            "path": str(manifest),
            "sha256": hashlib.sha256(original).hexdigest(),
            "members": sum(len(origin.files) for origin in origins),
            "origin_manifest_sha256": selected["manifest_sha256"],
            "origins": selected["origins"],
            "role": "readonly reviewed references; no import, execution, compiler seed or performance grant",
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SpecError(str(error)) from error


def declare(document: dict, *, destination: Path, variants, source_sha256: str, repo: Path):
    try:
        relative = destination.relative_to(repo).as_posix()
    except ValueError as exc:
        raise SpecError("reference payload must have a repository-contained grant path") from exc
    information = document.setdefault("information_sets", {})
    for variant in variants:
        item = information.setdefault(variant, {"condition": "explicit_public_references"})
        item.setdefault("allowed", []).append(
            {
                "path": relative + "/",
                "as": "reference-kernels",
                "mode": "ro",
                "note": "explicit pinned examples; compiler starts empty",
            }
        )
        item.setdefault("source_pins", []).append("public_reference_manifest:" + source_sha256)


BANK_GRANT = "cca-evidence"


def stage_bank(source: Path, sha256: str, destination: Path) -> dict:
    """Copy one whole pinned precomputed CCA bank (a ``reference_subsets`` condition payload).

    The source must verify against its bank.json pin before and after the copy; nothing else
    (other conditions, the corpus, its index or audit) is staged. The copy is read-only.
    """
    import shutil

    try:
        from merlin.mining.reference_subsets import verify_bank
    except ImportError as exc:
        raise SpecError("CCA bank staging requires the mining workflow from 'merlin[mining]'") from exc
    try:
        source = _ordinary(Path(source).expanduser().absolute())
        if destination.exists() or destination.is_symlink():
            raise ValueError("CCA bank destination must be absent")
        if not source.is_dir() or any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("CCA bank source must be one ordinary directory tree")
        if hashlib.sha256((source / "bank.json").read_bytes()).hexdigest() != sha256:
            raise ValueError("CCA bank source differs from its declared pin")
        bank = verify_bank(source)
        if not bank.get("cca_enabled") or not bank.get("units"):
            raise ValueError("CCA bank has no admitted units")
        shutil.copytree(source, destination, symlinks=False)
        if hashlib.sha256((destination / "bank.json").read_bytes()).hexdigest() != sha256:
            raise ValueError("staged CCA bank changed during copy")
        verify_bank(destination)
        for path in sorted(destination.rglob("*"), key=lambda p: -len(p.parts)):
            path.chmod(0o555 if path.is_dir() else 0o444)
        destination.chmod(0o555)
        return {
            "path": str(source),
            "sha256": sha256,
            "label": bank.get("label"),
            "units": [unit["kernel_id"] for unit in bank["units"]],
            "role": "readonly precomputed CCA bank; no import, execution, compiler seed or performance grant",
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SpecError(str(error)) from error


def declare_bank(document: dict, *, destination: Path, variants, sha256: str, repo: Path):
    """Grant the staged bank read-only as ``cca-evidence`` beside the reference grant."""
    try:
        relative = destination.relative_to(repo).as_posix()
    except ValueError as exc:
        raise SpecError("CCA bank payload must have a repository-contained grant path") from exc
    information = document.setdefault("information_sets", {})
    for variant in variants:
        item = information.get(variant)
        if item is None:
            raise SpecError("a CCA bank requires the variant's explicit public references")
        item.setdefault("allowed", []).append(
            {
                "path": relative + "/",
                "as": BANK_GRANT,
                "mode": "ro",
                "note": "pinned precomputed CCA bank; static evidence only",
            }
        )
        item.setdefault("source_pins", []).append("cca_bank:" + sha256)

"""Positively reviewed reference origins staged before fresh release admission.

Historical v1 manifests have byte pins but no publication/permission identity.
They remain historical evidence; fresh preparation refuses to restage them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..phase1.feedback import reference_baselines as baselines
from ..phase1.reference_inputs import ReferenceFile, ReferenceOrigin, _ordinary, _relative, stage_references
from ..spec import SpecError

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
        if (type(document) is not dict or document.get("schema") != SCHEMA
                or set(document) != {"schema", "review_id", "origins"}
                or type(document["origins"]) is not list or not document["origins"]):
            raise ValueError("fresh reference preparation requires v2 positive origin review; v1 cannot be restaged")
        origins = []
        for item in document["origins"]:
            common = {"name", "kind", "container", "identity", "license", "allowed_purpose", "files"}
            if type(item) is not dict:
                raise ValueError("reference origin must be an explicit reviewed object")
            expected = common | ({"public_url"} if item.get("kind") == "public_repository" else {"user_authorization"})
            if set(item) != expected or type(item["files"]) is not list:
                raise ValueError("reference origin requires exact publication/permission and source fields")
            relative = item["container"]
            if relative == "." and item["kind"] == "public_repository":
                container = manifest.parent
            else:
                container = manifest.parent / _relative(relative)
            files = []
            for member in item["files"]:
                if type(member) is not dict or set(member) != {"member", "destination", "sha256"}:
                    raise ValueError("reference origin requires individual source/destination/hash pins")
                files.append(ReferenceFile(**member))
            origins.append(ReferenceOrigin(item["name"], item["kind"], container, item["identity"], item["license"],
                tuple(files), public_url=item.get("public_url", ""),
                user_authorization=item.get("user_authorization", ""), allowed_purpose=item["allowed_purpose"]))
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
        return {"path": str(manifest), "sha256": hashlib.sha256(original).hexdigest(),
                "members": sum(len(origin.files) for origin in origins),
                "origin_manifest_sha256": selected["manifest_sha256"], "origins": selected["origins"],
                "role": "readonly positively reviewed references; no compiler seed or performance grant"}
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

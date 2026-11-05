"""Trusted operator-only provenance intake for hash exclusions.

Only explicitly pinned original inventories are opened. Protected tensor values
are never decoded. This validates the source inventory, not V23/V25 compiler or
model numerics; it does not replace a full numerics-bundle verification.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath

import yaml

from merlin.common import strict_json

SCHEMA = "merlin.phase1.protected_inventory_pointers.v1"


def digest(value):
    import json

    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def ordinary(path):
    path = Path(path)
    if (not path.is_absolute() or ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents))
            or not path.is_file()):
        raise ValueError("protected inventory has no exact ordinary original file")
    return path


def pinned(pin, *, max_bytes=64 * 1024 * 1024):
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
        raise ValueError("protected inventory needs exact original path and byte pin")
    path = ordinary(pin["path"])
    if path.stat().st_size > max_bytes:
        raise ValueError("protected inventory metadata exceeds its intake budget")
    raw = path.read_bytes()
    if len(raw) > max_bytes or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("protected inventory original byte pin changed")
    return path, raw


def _json(pin):
    return strict_json.loads(pinned(pin)[1])


def _sums(raw):
    result = {}
    for line in raw.decode("utf-8").splitlines():
        sha, marker, value = line.partition("  ")
        path = PurePosixPath(value)
        if (not marker or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
                or not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value
                or "\\" in value or "\x00" in value or value in result):
            raise ValueError("protected complete leaf inventory is malformed or ambiguous")
        result[value] = sha
    if not result:
        raise ValueError("protected complete leaf inventory is empty")
    return result


def _validation_sources(section, *, allow_catalog_aliases):
    """Reopen all source IR leaves committed by the original complete seal.

    This source-only intake checks the complete original leaf index and required
    source/catalog/census identities. It intentionally does not rehash or read
    unrelated operand/result arrays and never calls this a fresh bundle seal.
    """
    manifest_path, raw = pinned(section["manifest"])
    manifest = strict_json.loads(raw)
    sums_path, sums_raw = pinned(section["sums"])
    producer = _json(section["producer_receipt"])
    sums = _sums(sums_raw)
    seal = manifest.get("seal") or {}
    if (
        manifest.get("schema") != "merlin.model_check.bundle.v1"
        or producer.get("status") != "sealed_verified_and_protected"
        or producer.get("manifest_sha256") != section["manifest"]["sha256"]
        or not producer.get("finished_at")
        or sums_path.parent != manifest_path.parent or sums_path.name != seal.get("sums")
        or section["sums"]["sha256"] != seal.get("sums_sha256") or seal.get("files") != len(sums)
        or not isinstance(manifest.get("models"), dict) or not manifest["models"]
        or set(manifest.get("selected_capture_roots", {})) != set(manifest["models"])
    ):
        raise ValueError("protected source producer or complete original seal is not bound")
    required = set()
    for name, model in manifest["models"].items():
        programs = model.get("programs") or {}
        if not programs or model.get("model") != name:
            raise ValueError("protected model source roster is incomplete")
        for program in programs:
            for leaf in ("catalog.json", "census.json"):
                required.add(f"{name}/programs/{program}/{leaf}")
            source_prefix = f"{name}/programs/{program}/"
            if not any(member.startswith(source_prefix) and member.endswith("/model.prepared.mlir") for member in sums):
                raise ValueError("protected program has no complete prepared source in its original seal")
    sources = {member: sha for member, sha in sums.items() if member.endswith("/model.prepared.mlir")}
    if not sources or not required <= set(sums):
        raise ValueError("protected complete source or program-metadata roster is absent")
    catalogs = {}
    for member in required:
        path = ordinary(manifest_path.parent / member)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sums[member]:
            raise ValueError("protected original source ownership metadata changed")
        if member.endswith("/catalog.json"):
            catalogs[path] = (strict_json.loads(data), sums[member])
    # Reuse the original seal's exact namespace convention without array reads.
    from merlin.model_check.seal import _files

    if set(_files(manifest_path.parent)) != set(sums):
        raise ValueError("protected complete leaf namespace changed")
    # Unrelated operand/result aliases retain their original seal semantics;
    # neither their bytes nor their link targets enter this source-only grant.
    # Every required source and source-ownership leaf is ordinary and pinned.
    aliases = {}
    for member in sources:
        path = manifest_path.parent / member
        if not allow_catalog_aliases or not path.is_symlink():
            ordinary(path)
            continue
        # The original seal hashes source contents through leaf aliases. The
        # selected catalog must own this exact source namespace and role. The
        # current resolved target is bound separately and its COMPLETE contents
        # must still match the original source hash in the streaming issuer.
        ordinary(path.parent.parent / "catalog.json")
        role = path.parent.name
        catalog_path = path.parent.parent / "catalog.json"
        catalog, catalog_sha = catalogs.get(catalog_path, ({}, None))
        if (role not in {"ref", "dev"} or not isinstance(catalog, dict)
                or catalog.get("schema") != "merlin.model_check.catalog.v1"
                or catalog.get(role + "_prepared") != str(path)
                or any(parent.is_symlink() for parent in path.parents)):
            raise ValueError("protected source alias has no exact original catalog role")
        literal_target = os.readlink(path)
        target = Path(literal_target)
        if not target.is_absolute():
            target = path.parent / target
        target = ordinary(Path(os.path.abspath(target)))
        if path.resolve(strict=True) != target:
            raise ValueError("protected source alias changed during original catalog intake")
        aliases[member] = dict(read_path=str(target), literal_target=literal_target,
                               catalog=dict(path=str(catalog_path), sha256=catalog_sha), role=role)
    return manifest_path.parent, sources, aliases


def validation_sources(section):
    """Original ordinary-source intake; aliases retain their historical refusal."""
    root, sources, _ = _validation_sources(section, allow_catalog_aliases=False)
    return root, sources


def validation_sources_v2(section):
    """Explicit original sealed/catalog alias composition; no content-hash waiver.

    Every selected identity stays distinct. The returned read path is current
    operator custody, not a claim that the original seal recorded link spelling.
    A caller must hash complete source bytes and re-resolve this whole binding.
    """
    if set(section) == {'prospective_legacy_source_review'}:
        from . import operator_legacy_source_review as legacy
        return legacy.resolve(section['prospective_legacy_source_review'])
    root, sources, aliases = _validation_sources(section, allow_catalog_aliases=True)
    return {
        str(root / member): dict(sha256=sha,
            read_path=aliases[member]["read_path"] if member in aliases else str(ordinary(root / member)),
            alias=aliases.get(member))
        for member, sha in sources.items()
    }


def capsule_sources(pointer, *, protected_only=False):
    """Validate successful original Phase0 lineage, then all committed source IRs."""
    from merlin.common.digest import sha256_file
    from merlin_experiments.corpus import release
    from merlin_experiments.phase1.source_inputs import fingerprint

    section = pointer["protected_capsule_inventory"]
    manifest_path, raw = pinned(section["manifest"])
    manifest = yaml.safe_load(raw)
    generation = _json(section["generation_receipt"])
    producer = _json(section["producer_receipt"])
    preparation = _json(pointer["selected_release"]["preparation"])
    seal_path, _ = pinned(pointer["selected_release"]["seal"])
    # The original owner verifies its entire selected release. No synthetic
    # pointer flags or copied public overlap report can replace this check.
    release.verify(seal_path, seal_path.parent.parent / "payload/experiment/target_experiment.yaml")
    lineage = preparation["generation_lineage"]
    if (
        generation.get("schema") != "merlin.phase0_generation.v1"
        or producer.get("state") != "execution_succeeded"
        or any(generation.get(field) for field in ("failures", "unbuilt_roster", "performance_materialization_errors"))
        or generation.get("corpus_manifest") != str(manifest_path)
        or section["manifest"]["sha256"] != lineage.get("generated_manifest_sha256")
        or section["generation_receipt"]["sha256"] != lineage.get("generation_receipt_sha256")
        or section["source_evidence_manifest"]["sha256"] != lineage.get("evidence_manifest_sha256")
        or (manifest.get("phase0_evidence") or {}).get("manifest") != section["source_evidence_manifest"]["path"]
        or (manifest.get("phase0_evidence") or {}).get("generation_receipt") != section["generation_receipt"]["path"]
    ):
        raise ValueError("protected Phase0 generation is failed, incomplete or not release-bound")
    pinned(section["source_evidence_manifest"])
    commitments = generation.get("capsule_commitments")
    if not isinstance(commitments, list):
        raise ValueError("protected generation has no complete committed capsule roster")
    recorded = {}
    for row in commitments:
        member = row.get("member") if isinstance(row, dict) else None
        relative = PurePosixPath(member) if isinstance(member, str) else None
        if (relative is None or relative.is_absolute() or ".." in relative.parts
                or relative.as_posix() != member or member in recorded):
            raise ValueError("protected generation member roster is ambiguous")
        recorded[member] = row.get("sha256")
    root = manifest_path.parent
    actual = {str(path.parent.relative_to(root)) for path in root.rglob("capsule.yaml")}
    if (actual != set(recorded) or generation.get("capsules_written") != len(recorded)
            or lineage.get("capsules") != len(recorded)
            or digest(sorted(recorded.items())) != lineage.get("capsule_roster_sha256")):
        raise ValueError("protected generation capsule membership changed")
    sources = {}
    for member, sha in recorded.items():
        directory = root / member
        if fingerprint(directory) != sha:
            raise ValueError("protected generation committed capsule bytes changed")
        if protected_only:
            capsule = yaml.safe_load(ordinary(directory / "capsule.yaml").read_bytes())
            label = capsule.get("label") if isinstance(capsule, dict) else None
            if label not in {"public", "dev", "hidden"}:
                raise ValueError("original committed capsule has no recognized protected/public role")
            if label != "hidden":
                continue
        path = ordinary(directory / "capsule.interface.mlir")
        sources[str(path)] = sha256_file(path)
    return sources


def resolve(path, sha256):
    """Host-only exact selection; return private source pins, never public data."""
    pointer = strict_json.loads(pinned({"path": str(path), "sha256": sha256})[1])
    if not isinstance(pointer, dict) or pointer.get("schema") != SCHEMA:
        raise ValueError("protected inventory pointer has unsupported original ownership")
    sources = capsule_sources(pointer)
    root, validation = validation_sources(pointer["protected_validation_source_inventory"])
    sources.update({str(root / member): sha for member, sha in validation.items()})
    if not sources:
        raise ValueError("protected source inventory is empty")
    return dict(sorted(sources.items()))


def resolve_streaming_v2(path, sha256):
    """Explicit source-hash/canonical-target composition for bounded streaming."""
    pointer = strict_json.loads(pinned({"path": str(path), "sha256": sha256})[1])
    if not isinstance(pointer, dict) or pointer.get("schema") != SCHEMA:
        raise ValueError("protected inventory pointer has unsupported original ownership")
    sources = {path: dict(sha256=sha, read_path=path, alias=None)
               for path, sha in capsule_sources(pointer, protected_only=True).items()}
    validation = validation_sources_v2(pointer["protected_validation_source_inventory"])
    if set(sources) & set(validation):
        raise ValueError("protected selected source identities overlap")
    sources.update(validation)
    if not sources:
        raise ValueError("protected source inventory is empty")
    return dict(sorted(sources.items()))

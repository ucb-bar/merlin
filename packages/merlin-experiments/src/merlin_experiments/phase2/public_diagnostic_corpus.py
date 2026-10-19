"""Readonly published source witnesses; never Phase 0 performance admission."""

from __future__ import annotations

import shutil
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common.cpu_pool import shared_compute_pool
from merlin_experiments.phase0 import operator_exclusion as EXCLUSION
from merlin_experiments.phase0 import public_programs as PUBLIC

from .contracts import StageGateError, canonical_json, document_sha256, exact_tree_record, mapping_file, sha256_file
from .corpus import PerformanceCapsule
from .retained_revisions import source_inventory

SCHEMA = "merlin.frozen_public_diagnostic_corpus.v1"
_ISSUER = object()
_ISSUED = weakref.WeakKeyDictionary()


def _leaf(path):
    path = Path(path)
    if path.is_symlink() or path.resolve(strict=True) != path or not path.is_file():
        raise StageGateError("diagnostic roster needs original ordinary source leaves")
    return {"path": str(path), "sha256": sha256_file(path)}


def _published(directory, selection):
    EXCLUSION.require_selection(selection, directory)
    EXCLUSION.check_publication(selection, directory)
    descriptor = mapping_file(directory / "capsule.yaml", yaml_file=True)
    if descriptor.get("kind") != "model" or descriptor.get("label") not in {"public", "dev"}:
        raise StageGateError("diagnostic roster requires original published public source programs")
    PUBLIC.verify_written(descriptor, directory)
    trace = mapping_file(directory / "frontend-trace.json")
    mode = (descriptor.get("materialized_capture") or {}).get("source_mode")
    if not PUBLIC.original_contractions(trace, source_mode=mode):
        raise StageGateError("diagnostic source roster cannot admit host-only programs")
    return descriptor


@dataclass(frozen=True, eq=False)
class FrozenPublicDiagnosticCorpus:
    root: Path
    capsules_root: Path
    manifest_path: Path
    manifest_sha256: str
    capsules_sha256: str
    capsules: tuple[PerformanceCapsule, ...]
    _operator: object
    _planning_inventory: dict
    _sources: tuple
    _snapshot_inventory: dict
    _functions: tuple
    _issuer: object

    def verify(self):
        if type(self) is not FrozenPublicDiagnosticCorpus or self not in _ISSUED or self._issuer is not _ISSUER:
            raise StageGateError("diagnostic source corpus was not issued by its original host")
        with shared_compute_pool(timeout_s=120):
            if (
                type(self._operator) is not EXCLUSION.OperatorPrivateExclusion
                or any(_leaf(row["path"]) != row for row in self._sources)
                or any(getattr(owner, name) is not function for owner, name, function in self._functions)
                or _leaf(self.manifest_path)["sha256"] != self.manifest_sha256
                or _leaf(self._planning_inventory["path"]) != self._planning_inventory
                or self.root.stat().st_mode & 0o222
            ):
                raise StageGateError("diagnostic source manifest or original verifier changed")
            if (
                source_inventory(self.root, max_bytes=65 << 20, max_files=16384, readonly=True)
                != self._snapshot_inventory
            ):
                raise StageGateError("diagnostic snapshot membership or readonly ownership changed")
            document = mapping_file(self.manifest_path)
            if canonical_json(document) != _ISSUED[self] or document.get("schema") != SCHEMA:
                raise StageGateError("diagnostic source roster changed its original smoke-only scope")
            rows = []
            for member in self.capsules:
                source_inventory(member.source_dir, max_bytes=32 << 20, max_files=8192, readonly=True)
                descriptor = _published(member.source_dir, self._operator)
                actual = exact_tree_record(member.source_dir)
                if descriptor != member.descriptor or actual != {
                    "sha256": member.source_sha256,
                    "n_files": member.n_files,
                    "n_bytes": member.n_bytes,
                }:
                    raise StageGateError("diagnostic source or complete reference bytes changed")
                rows.append(_row(member))
            if document["members"] != rows or document_sha256(rows) != self.capsules_sha256:
                raise StageGateError("diagnostic source roster omitted or substituted an original member")
        return document


def _row(member):
    return {
        "family": member.family,
        "capsule": member.capsule,
        "snapshot_relative_path": "capsules/" + member.source_relative_path,
        "sha256": member.source_sha256,
        "n_files": member.n_files,
        "n_bytes": member.n_bytes,
    }


def freeze_public_diagnostic_corpus(
    *,
    sources,
    destination,
    target,
    operator_exclusion,
    planning_inventory,
    planning_inventory_sha256,
):
    """Freeze published public witnesses under an explicit private exclusion pin.

    The planning inventory remains an external exact pin, not a sealed/admitted
    performance cohort. No generation provenance, certificate or objective is
    invented for these smoke witnesses.
    """
    if (
        type(sources) is not tuple
        or not 1 <= len(sources) <= 32
        or type(operator_exclusion) is not EXCLUSION.OperatorPrivateExclusion
        or type(target) is not str
        or not target
    ):
        raise StageGateError("diagnostic corpus requires explicit bounded public sources and a live private selection")
    destination = Path(destination)
    inputs = tuple(Path(path) for path in sources)
    planning = _leaf(planning_inventory)
    if planning["sha256"] != planning_inventory_sha256:
        raise StageGateError("diagnostic roster changed its external planning inventory")
    if (
        destination.exists()
        or destination.is_symlink()
        or destination.resolve() != destination
        or len(set(inputs)) != len(inputs)
        or any(destination.is_relative_to(path) or path.is_relative_to(destination) for path in inputs)
    ):
        raise StageGateError("diagnostic source snapshot requires a fresh disjoint canonical directory")
    EXCLUSION.require_selection(operator_exclusion, destination, *inputs)
    with shared_compute_pool(timeout_s=120):
        descriptors, inventories = [], []
        for path in inputs:
            inventories.append(source_inventory(path, max_bytes=32 << 20, max_files=8192))
            descriptors.append(_published(path, operator_exclusion))
        if (
            sum(row["n_bytes"] for row in inventories) > 64 << 20
            or sum(row["n_files"] for row in inventories) > 16383
        ):
            raise StageGateError("diagnostic source roster exceeds its aggregate copy budget")
        capsules = destination / "capsules"
        capsules.mkdir(parents=True)
        members = []
        for index, (path, descriptor, inventory) in enumerate(zip(inputs, descriptors, inventories, strict=True)):
            relative = f"member_{index:03d}"
            copied = capsules / relative
            shutil.copytree(path, copied, symlinks=False)
            if (
                _published(path, operator_exclusion) != descriptor
                or _published(copied, operator_exclusion) != descriptor
                or source_inventory(path, max_bytes=32 << 20, max_files=8192) != inventory
                or source_inventory(copied, max_bytes=32 << 20, max_files=8192) != inventory
            ):
                raise StageGateError("published source changed during diagnostic freezing")
            actual = exact_tree_record(copied)
            name = descriptor.get("name")
            if type(name) is not str or not name or Path(name).name != name:
                raise StageGateError("diagnostic source has an invalid original public identity")
            members.append(
                PerformanceCapsule(
                    "public_source_diagnostic",
                    name,
                    copied,
                    relative,
                    descriptor,
                    actual["sha256"],
                    actual["n_files"],
                    actual["n_bytes"],
                )
            )
        if len({row.capsule for row in members}) != len(members):
            raise StageGateError("diagnostic source identities are repeated")
        rows = [_row(member) for member in members]
        document = {
            "schema": SCHEMA,
            "scope": "public source smoke witnesses only",
            "target": target,
            "members": rows,
            "capsules_sha256": document_sha256(rows),
            "planning_inventory": planning,
            "performance_cohort_admitted": False,
            "formal_admission": False,
            "promotion_authority": "NONE",
            "timing": "UNKNOWN",
        }
        manifest = destination / "public_diagnostic_corpus_manifest.json"
        manifest.write_bytes(canonical_json(document))
        for path in reversed((destination, *destination.rglob("*"))):
            path.chmod(path.stat().st_mode & ~0o222)
        EXCLUSION.require_selection(operator_exclusion, destination, *inputs)
        if _leaf(planning_inventory) != planning:
            raise StageGateError("diagnostic planning inventory changed during freezing")
        owner = FrozenPublicDiagnosticCorpus(
            destination,
            capsules,
            manifest,
            sha256_file(manifest),
            document["capsules_sha256"],
            tuple(members),
            operator_exclusion,
            planning,
            tuple(_leaf(Path(module.__file__).resolve()) for module in (PUBLIC, EXCLUSION)) + (_leaf(__file__),),
            source_inventory(destination, max_bytes=65 << 20, max_files=16384, readonly=True),
            tuple(
                (module, name, getattr(module, name))
                for module, name in (
                    (PUBLIC, "verify_written"),
                    (PUBLIC, "original_contractions"),
                    (EXCLUSION, "check_publication"),
                )
            ),
            _ISSUER,
        )
        _ISSUED[owner] = canonical_json(document)
        try:
            owner.verify()
        except BaseException:
            del _ISSUED[owner]
            raise
        return owner

"""Generated development-corpus admission, immutable copies and exact result cells.

This owner never imports native controllers or executes an analyzer/simulator. Model
portfolio and measured-claim decisions remain distinct consumers of frozen inputs.
"""

from __future__ import annotations

import copy
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.targetgen.target_experiment import TargetExperiment
from merlin_experiments.phase2 import contracts as CONTRACTS
from merlin_experiments.phase2 import prompt as PP
from merlin_experiments.phase2.contracts import StageGateError


@dataclass(frozen=True)
class PerformanceCapsule:
    """Exact capsule view shared by authoring, admission and measurement consumers."""

    family: str
    capsule: str
    source_dir: Path
    source_relative_path: str
    descriptor: dict[str, Any]
    source_sha256: str
    n_files: int
    n_bytes: int


@dataclass(frozen=True)
class FrozenPerformanceCorpus:
    """Content-addressed performance inputs needed by the agent-only view."""

    root: Path
    capsules_root: Path
    manifest_path: Path
    manifest_sha256: str
    capsules_sha256: str
    capsules: tuple[PerformanceCapsule, ...]


@dataclass(frozen=True)
class PerformanceCorpus:
    target: str
    corpus_root: Path
    phase_root: Path
    provenance_manifest: Path
    provenance_sha256: str
    performance_generation: dict[str, Any]
    capsules: tuple[PerformanceCapsule, ...]


def _selected_names(value: str | Sequence[str] | None, *, label: str) -> tuple[str, ...]:
    if value is None or value == "all":
        return ()
    raw = value.split(",") if isinstance(value, str) else value
    names = tuple(CONTRACTS.safe_component(str(item).strip(), label=label) for item in raw if str(item).strip())
    if not names or len(names) != len(set(names)):
        raise StageGateError(f"{label} selection must contain unique names or 'all'")
    return names


def discover_performance_corpus(
    target_experiment: TargetExperiment,
    *,
    families: str | Sequence[str] | None = None,
    capsules: str | Sequence[str] | None = None,
) -> PerformanceCorpus:
    """Admit only generated dev capsules from the descriptor-derived phase."""
    target = str(target_experiment.target or "").strip()
    if not target:
        raise StageGateError("target experiment has no target identity")
    corpus_root = Path(target_experiment.capsule_corpus).resolve().parent
    provenance = corpus_root / "MANIFEST.yaml"
    manifest = CONTRACTS.mapping_file(provenance, yaml_file=True)
    generations = manifest.get("performance_generation")
    generation = generations.get(target) if isinstance(generations, Mapping) else None
    if not isinstance(generation, Mapping) or generation.get("errors") != []:
        raise StageGateError(f"corpus provenance has no clean generated performance record for {target!r}")
    phase = generation.get("phase")
    if (
        not isinstance(phase, Mapping)
        or phase.get("included_in_functional_grade") is not False
        or phase.get("label") != "dev"
    ):
        raise StageGateError("performance provenance does not prove a dev-only phase")
    category = CONTRACTS.safe_component(str(phase.get("category") or ""), label="performance phase category")
    if not category.startswith("_"):
        raise StageGateError("performance phase is not excluded from functional discovery")
    phase_root = (corpus_root / category).resolve()
    if corpus_root not in phase_root.parents or phase_root.is_symlink() or not phase_root.is_dir():
        raise StageGateError("descriptor-derived performance phase is absent or unsafe")
    if phase_root in {Path(path).resolve() for path in target_experiment.graded_roots()}:
        raise StageGateError("performance phase leaks into functional graded roots")
    generated = manifest.get("generated")
    hand_authored = manifest.get("hand_authored")
    if not isinstance(generated, list) or not isinstance(hand_authored, list):
        raise StageGateError("corpus provenance lacks generated/manual classification")
    generated_paths = {str(value) for value in generated}
    manual_paths = {str(value) for value in hand_authored}
    phase_generated = {path for path in generated_paths if Path(path).parts and Path(path).parts[0] == category}
    if any(Path(path).parts and Path(path).parts[0] == category for path in manual_paths):
        raise StageGateError("performance phase contains manually classified capsules")

    found: list[PerformanceCapsule] = []
    for descriptor_path in sorted(phase_root.glob("*/capsule.yaml")):
        source = descriptor_path.parent
        relative = source.relative_to(corpus_root).as_posix()
        if relative not in phase_generated or relative in manual_paths:
            raise StageGateError(f"performance capsule lacks generator provenance: {relative}")
        descriptor = CONTRACTS.mapping_file(descriptor_path, yaml_file=True)
        name = CONTRACTS.safe_component(str(descriptor.get("name") or ""), label="performance capsule")
        performance = descriptor.get("performance")
        if (
            source.name != name
            or descriptor.get("label") != "dev"
            or descriptor.get("source_role") != "derived_sweep"
            or not isinstance(performance, Mapping)
        ):
            raise StageGateError(f"performance capsule {name!r} is not a generated dev member")
        family = CONTRACTS.safe_component(str(performance.get("family") or ""), label="performance family")
        claim = performance.get("claim")
        if claim not in ("RECOVERS", "PREDICTS", "DIFFERENTIAL"):
            raise StageGateError(f"performance capsule {name!r} has no canonical claim")
        if claim == "PREDICTS" and not isinstance(performance.get("acceptance"), Mapping):
            raise StageGateError(f"predictive performance capsule {name!r} has no frozen acceptance contract")
        tree = CONTRACTS.exact_tree_record(source)
        found.append(
            PerformanceCapsule(
                family,
                name,
                source.resolve(),
                relative,
                descriptor,
                str(tree["sha256"]),
                int(tree["n_files"]),
                int(tree["n_bytes"]),
            )
        )
    if not found or {row.source_relative_path for row in found} != phase_generated:
        raise StageGateError("generated performance phase is empty or stale versus provenance")
    wanted_families = set(_selected_names(families, label="performance family"))
    wanted_capsules = set(_selected_names(capsules, label="performance capsule"))
    known_families = {row.family for row in found}
    known_capsules = {row.capsule for row in found}
    if wanted_families - known_families or wanted_capsules - known_capsules:
        raise StageGateError("performance selection names an unknown generated member")
    selected = tuple(
        row
        for row in found
        if (not wanted_families or row.family in wanted_families)
        and (not wanted_capsules or row.capsule in wanted_capsules)
    )
    if not selected:
        raise StageGateError("performance selection contains zero capsules")
    return PerformanceCorpus(
        target, corpus_root, phase_root, provenance, CONTRACTS.sha256_file(provenance), dict(generation), selected
    )


def freeze_performance_corpus(corpus: PerformanceCorpus, snapshot_root: Path) -> FrozenPerformanceCorpus:
    snapshot_root = Path(snapshot_root).resolve()
    if snapshot_root.exists() or snapshot_root.is_symlink():
        raise StageGateError(f"performance snapshot is not fresh: {snapshot_root}")
    if CONTRACTS.sha256_file(corpus.provenance_manifest) != corpus.provenance_sha256:
        raise StageGateError("performance provenance changed before freeze")
    capsules_root = snapshot_root / "capsules"
    capsules_root.mkdir(parents=True)
    frozen: list[PerformanceCapsule] = []
    rows: list[dict[str, Any]] = []
    for member in corpus.capsules:
        before = CONTRACTS.exact_tree_record(member.source_dir)
        if before["sha256"] != member.source_sha256:
            raise StageGateError(f"performance capsule changed before freeze: {member.capsule}")
        destination = capsules_root / member.source_relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(member.source_dir, destination, symlinks=False)
        copied = CONTRACTS.exact_tree_record(destination)
        if copied != before or CONTRACTS.exact_tree_record(member.source_dir) != before:
            raise StageGateError(f"performance capsule changed during freeze: {member.capsule}")
        frozen.append(
            PerformanceCapsule(
                member.family,
                member.capsule,
                destination,
                member.source_relative_path,
                copy.deepcopy(member.descriptor),
                str(copied["sha256"]),
                int(copied["n_files"]),
                int(copied["n_bytes"]),
            )
        )
        rows.append(
            {
                "family": member.family,
                "capsule": member.capsule,
                "source_relative_path": member.source_relative_path,
                "snapshot_relative_path": destination.relative_to(snapshot_root).as_posix(),
                "snapshot_sha256": copied["sha256"],
                "n_files": copied["n_files"],
                "n_bytes": copied["n_bytes"],
                "performance": member.descriptor["performance"],
                "performance_sha256": CONTRACTS.document_sha256(member.descriptor["performance"]),
            }
        )
    aggregate = CONTRACTS.exact_tree_record(capsules_root)
    document = {
        "schema_version": 1,
        "target": corpus.target,
        "source": {
            "provenance_manifest": str(corpus.provenance_manifest),
            "provenance_sha256": corpus.provenance_sha256,
            "performance_generation_sha256": CONTRACTS.document_sha256(corpus.performance_generation),
        },
        "capsules_sha256": aggregate["sha256"],
        "capsules": rows,
    }
    manifest = snapshot_root / "performance_corpus_manifest.json"
    CONTRACTS.write_json(manifest, document)
    for path in sorted(snapshot_root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    snapshot_root.chmod(0o555)
    result = FrozenPerformanceCorpus(
        snapshot_root, capsules_root, manifest, CONTRACTS.sha256_file(manifest), str(aggregate["sha256"]), tuple(frozen)
    )
    verify_frozen_performance_corpus(result)
    return result


def load_frozen_performance_corpus(
    root: Path, *, manifest_sha256: str, capsules_sha256: str, expected_target: str | None = None
) -> FrozenPerformanceCorpus:
    root = CONTRACTS.require_real_directory(root, label="frozen performance corpus")
    manifest = root / "performance_corpus_manifest.json"
    if CONTRACTS.sha256_file(manifest) != manifest_sha256:
        raise StageGateError("frozen performance manifest digest changed")
    document = CONTRACTS.mapping_file(manifest)
    if (
        document.get("schema_version") != 1
        or document.get("capsules_sha256") != capsules_sha256
        or (expected_target is not None and document.get("target") != expected_target)
    ):
        raise StageGateError("frozen performance manifest identity changed")
    capsules_root = root / "capsules"
    members: list[PerformanceCapsule] = []
    for row in document.get("capsules") or []:
        if not isinstance(row, Mapping):
            raise StageGateError("frozen performance manifest contains a malformed member")
        relative = Path(str(row.get("snapshot_relative_path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise StageGateError("frozen performance member path is unsafe")
        source = (root / relative).resolve(strict=True)
        source.relative_to(capsules_root.resolve(strict=True))
        descriptor = CONTRACTS.mapping_file(source / "capsule.yaml", yaml_file=True)
        members.append(
            PerformanceCapsule(
                str(row.get("family")),
                str(row.get("capsule")),
                source,
                str(row.get("source_relative_path")),
                descriptor,
                str(row.get("snapshot_sha256")),
                int(row.get("n_files") or 0),
                int(row.get("n_bytes") or 0),
            )
        )
    result = FrozenPerformanceCorpus(root, capsules_root, manifest, manifest_sha256, capsules_sha256, tuple(members))
    verify_frozen_performance_corpus(result)
    return result


def verify_frozen_performance_corpus(corpus: FrozenPerformanceCorpus) -> None:
    if CONTRACTS.sha256_file(corpus.manifest_path) != corpus.manifest_sha256:
        raise StageGateError("frozen performance manifest bytes changed")
    if CONTRACTS.exact_tree_record(corpus.capsules_root)["sha256"] != corpus.capsules_sha256:
        raise StageGateError("frozen performance capsule bytes changed")
    document = CONTRACTS.mapping_file(corpus.manifest_path)
    rows = document.get("capsules")
    if not isinstance(rows, list) or len(rows) != len(corpus.capsules) or not rows:
        raise StageGateError("frozen performance manifest has an incomplete member set")
    indexed = {(member.family, member.capsule): member for member in corpus.capsules}
    for row in rows:
        identity = (str(row.get("family")), str(row.get("capsule")))
        member = indexed.get(identity)
        if member is None:
            raise StageGateError(f"frozen performance manifest has unknown member {identity}")
        observed = CONTRACTS.exact_tree_record(member.source_dir)
        if any(observed[key] != row.get(key) for key in ("n_files", "n_bytes")) or observed["sha256"] != row.get(
            "snapshot_sha256"
        ):
            raise StageGateError(f"frozen performance member changed: {identity}")


def expected_perf_cells(
    capsules: Sequence[PerformanceCapsule], replicates: int, timing_simulator: str = "gsim"
) -> tuple[PP.PerfCell, ...]:
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise StageGateError("performance replicate count must be positive")
    if timing_simulator not in ("gsim", "verilator"):
        raise StageGateError("performance timing simulator must be gsim or verilator")
    cells = tuple(
        PP.PerfCell(member.family, member.capsule, simulator, f"r{index:03d}")
        for member in capsules
        for index in range(replicates)
        for simulator in ("spike", timing_simulator)
    )
    if not cells or len(cells) != len(set(cells)):
        raise StageGateError("performance cell schedule is empty or duplicated")
    return cells

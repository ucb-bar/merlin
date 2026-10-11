"""Generated development-corpus admission, immutable copies and exact result cells.

This owner never imports native controllers or executes an analyzer/simulator. Model
portfolio and measured-claim decisions remain distinct consumers of frozen inputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common.digest import is_sha256
from merlin.targetgen.target_experiment import TargetExperiment
from merlin_experiments.corpus.phase_selection import validate_phase_selections
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
    test_justification: dict[str, Any] | None = None
    representative_selection: dict[str, Any] | None = None


def _selected_names(value: str | Sequence[str] | None, *, label: str) -> tuple[str, ...]:
    if value is None or value == "all":
        return ()
    raw = value.split(",") if isinstance(value, str) else value
    names = tuple(CONTRACTS.safe_component(str(item).strip(), label=label) for item in raw if str(item).strip())
    if not names or len(names) != len(set(names)):
        raise StageGateError(f"{label} selection must contain unique names or 'all'")
    return names


def _component_objectives(record, members):
    """Check pinned host-issued generation metadata before existing phase admission.

    These hashes bind the recorded selection; they are neither signatures of
    review authority nor a fresh re-read of the recorded generator/source paths.
    Capsule bytes and objective bindings themselves are rechecked here.
    """
    if record is None:
        if any(
            "component_generation_sha256" in member.descriptor.get("performance", {})
            or "objective" in member.descriptor.get("performance", {})
            for member in members
        ):
            raise StageGateError("component objective lacks its generator declaration identity")
        return {}
    from merlin.targetgen import phase_policy as policy
    from merlin_experiments.phase0.component_generation import SCHEMA, digest, require_written

    if (
        not isinstance(record, Mapping)
        or record.get("schema") != SCHEMA
        or record.get("status") != "generated_from_reviewed_component_declarations"
        or record.get("scope") != "independent_components_no_application_inputs"
        or record.get("whole_coverage_verified") is not False
        or record.get("timing_verified") is not False
        or not isinstance(record.get("families"), Mapping)
    ):
        raise StageGateError("component generation identity is malformed")
    pins = [record.get("recipe"), record.get("shared_template"), *(record.get("generator_sources") or [])]
    if len(pins) < 3 or any(
        not isinstance(pin, Mapping) or not is_sha256(pin.get("sha256")) or not pin.get("path") for pin in pins
    ):
        raise StageGateError("component generation lacks exact source/template identities")
    if (
        not is_sha256(record.get("software_spec_sha256"))
        or not is_sha256(record.get("declaration_sha256"))
        or not isinstance(record.get("hardware"), Mapping)
        or set(record["hardware"]) != {"contract_sha256", "raw_facts_sha256"}
        or any(not is_sha256(value) for value in record["hardware"].values())
    ):
        raise StageGateError("component generation lacks selected reviewed HW/SW identities")
    result = {}
    for member in members:
        if (
            CONTRACTS.mapping_file(member.source_dir / "capsule.yaml", yaml_file=True) != member.descriptor
            or CONTRACTS.exact_tree_record(member.source_dir)["sha256"] != member.source_sha256
        ):
            raise StageGateError("component generated source/descriptor changed")
        performance = member.descriptor.get("performance") or {}
        family = performance.get("family")
        declaration = record["families"].get(family)
        if performance.get("component_generation_sha256") != digest(record):
            raise StageGateError("component member differs from its generated source/contract identity")
        objective = performance.get("objective")
        if objective != (declaration.get("objective") if isinstance(declaration, Mapping) else None):
            raise StageGateError("component objective differs from its reviewed generated family binding")
        try:
            prepared = record.get("source_performance_preparation") or {}
            if prepared.get("schema") == "merlin.phase0.source_performance_preparation.v2":
                from merlin_experiments.phase2.test_justification import _host_basis

                if _host_basis(member.descriptor, member.source_dir, performance.get("gate") or {}) is None:
                    raise ValueError("prepared source curriculum has no independently bound source basis")
                original = record.get("source_semantics_admission") or {}
                screen = member.descriptor.get("source_semantics_screen") or {}
                if any(
                    screen.get(key) != original.get(key) for key in ("hardware_intake_sha256", "software_intake_sha256")
                ):
                    raise ValueError("prepared source curriculum differs from original live HW/SW selectors")
            else:
                require_written(member.descriptor)
            if objective is not None:
                if not isinstance(declaration.get("operations"), list) or not declaration["operations"]:
                    raise ValueError("component objective has no reviewed operation owners")
                value = dict(objective)
                value["provenance"] = tuple(value["provenance"])
                result[member.capsule] = policy.PerformanceObjective(**value)
        except (TypeError, ValueError, KeyError) as exc:
            raise StageGateError("component phase admission refused: " + str(exc)) from exc
    return result


def performance_objectives(corpus: PerformanceCorpus | FrozenPerformanceCorpus):
    """Typed name bindings for phase_policy.split_report/anchors; no cycle claims."""
    if isinstance(corpus, FrozenPerformanceCorpus):
        verify_frozen_performance_corpus(corpus)
        record = CONTRACTS.mapping_file(corpus.manifest_path).get("component_generation")
    elif isinstance(corpus, PerformanceCorpus):
        if CONTRACTS.sha256_file(corpus.provenance_manifest) != corpus.provenance_sha256:
            raise StageGateError("component generation provenance changed")
        record = corpus.performance_generation.get("component_generation")
    else:
        raise TypeError("typed performance corpus required")
    return _component_objectives(record, corpus.capsules)


def discover_performance_corpus(
    target_experiment: TargetExperiment,
    *,
    families: str | Sequence[str] | None = None,
    capsules: str | Sequence[str] | None = None,
) -> PerformanceCorpus:
    """Admit only generated dev capsules from the descriptor-derived phase."""
    representative = capsules == "representative"
    if representative and families not in (None, "all"):
        raise StageGateError("representative selection requires the complete generated family roster")
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
    selections = manifest.get("phase_corpora")
    if selections is not None:
        selected_roles = selections.get(target) if isinstance(selections, Mapping) else None
        try:
            roles = validate_phase_selections(
                selected_roles, performance_category=category, generated_members=generated_paths
            )
        except ValueError as exc:
            raise StageGateError(f"performance phase selections are invalid: {exc}") from exc
        if set(roles["phase2"]) != phase_generated:
            raise StageGateError("performance phase differs from Phase 0's explicit Phase 2 selection")

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
        if claim not in PP.CANONICAL_CLAIMS:
            raise StageGateError(f"performance capsule {name!r} has no canonical claim")
        if claim in ("PREDICTS", "EMITS") and not isinstance(performance.get("acceptance"), Mapping):
            raise StageGateError(f"{claim} performance capsule {name!r} has no frozen acceptance contract")
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
    _component_objectives(generation.get("component_generation"), found)
    wanted_families = set(_selected_names(families, label="performance family"))
    wanted_capsules = set(_selected_names(None if representative else capsules, label="performance capsule"))
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
    identity = CONTRACTS.sha256_file(provenance)
    justification = None
    from merlin_experiments.phase2 import test_justification as TJ

    if TJ.requires_justification(manifest):
        complete = PerformanceCorpus(
            target, corpus_root, phase_root, provenance, identity, dict(generation), tuple(found)
        )
        justification = TJ.derive_justification(complete)
    result = PerformanceCorpus(
        target, corpus_root, phase_root, provenance, identity, dict(generation), selected, justification
    )
    if representative:
        from merlin_experiments.phase2 import representative_selection as RS

        if justification is None:
            raise StageGateError("representative selection requires current generated Phase 0 evidence")
        scope = TJ.verify_live_justification(result, justification)
        selection = RS.derive(justification, scope)
        chosen = {(row["family"], row["capsule"]) for row in selection["selected_members"]}
        result = PerformanceCorpus(
            target,
            corpus_root,
            phase_root,
            provenance,
            identity,
            dict(generation),
            tuple(row for row in selected if (row.family, row.capsule) in chosen),
            justification,
            selection,
        )
    return result


def freeze_performance_corpus(corpus: PerformanceCorpus, snapshot_root: Path) -> FrozenPerformanceCorpus:
    from merlin_experiments.phase2 import test_justification as TJ

    snapshot_root = Path(snapshot_root).resolve()
    if snapshot_root.exists() or snapshot_root.is_symlink():
        raise StageGateError(f"performance snapshot is not fresh: {snapshot_root}")
    if CONTRACTS.sha256_file(corpus.provenance_manifest) != corpus.provenance_sha256:
        raise StageGateError("performance provenance changed before freeze")
    selected_scope = None
    if corpus.test_justification is not None:
        selected_scope = TJ.verify_live_justification(corpus, corpus.test_justification)
    elif TJ.requires_justification(CONTRACTS.mapping_file(corpus.provenance_manifest, yaml_file=True)):
        raise StageGateError("current generated performance corpus lacks its test justification")
    if corpus.representative_selection is not None:
        from merlin_experiments.phase2 import representative_selection as RS

        if (
            corpus.test_justification is None
            or RS.derive(corpus.test_justification, selected_scope) != corpus.representative_selection
        ):
            raise StageGateError("representative selection changed before freeze")
        chosen = {(row["family"], row["capsule"]) for row in corpus.representative_selection["selected_members"]}
        if {(row.family, row.capsule) for row in corpus.capsules} != chosen:
            raise StageGateError("representative selection differs from frozen members")
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
    component = corpus.performance_generation.get("component_generation")
    if component is not None:
        _component_objectives(component, corpus.capsules)
        document["component_generation"] = copy.deepcopy(component)
    if corpus.test_justification is not None:
        receipt_path = snapshot_root / "test_justification.json"
        CONTRACTS.write_json(receipt_path, corpus.test_justification)
        copied = snapshot_root / "test_justification_inputs"
        copied.mkdir()
        source_root = TJ.selected_evidence_root(corpus)
        for name, relative in (
            ("evidence-manifest.json", "evidence-manifest.json"),
            ("performance-facts.json", "hardware/effective-views/performance-facts.json"),
            ("operation-accounting.json", "coverage/operation-accounting.json"),
            ("performance-basis.json", "coverage/performance-basis.json"),
        ):
            shutil.copyfile(source_root / relative, copied / name)
        if selected_scope is not None:
            CONTRACTS.write_json(copied / "performance-scope.json", selected_scope)
        document["test_justification"] = {
            "path": receipt_path.name,
            "sha256": CONTRACTS.sha256_file(receipt_path),
            **(
                {"selected_requirement_sha256": corpus.test_justification["source"]["selected_requirement_sha256"]}
                if selected_scope is not None
                else {}
            ),
            "inputs": {
                "evidence-manifest.json": corpus.test_justification["source"]["manifest_sha256"],
                "performance-facts.json": corpus.test_justification["source"]["performance_facts_sha256"],
                "operation-accounting.json": corpus.test_justification["source"]["operation_accounting_sha256"],
                "performance-basis.json": corpus.test_justification["source"]["performance_basis_sha256"],
                **(
                    {"performance-scope.json": corpus.test_justification["source"]["selected_scope_sha256"]}
                    if selected_scope is not None
                    else {}
                ),
            },
        }
    if corpus.representative_selection is not None:
        receipt_path = snapshot_root / "representative_selection.json"
        CONTRACTS.write_json(receipt_path, corpus.representative_selection)
        document["representative_selection"] = {
            "path": receipt_path.name,
            "sha256": CONTRACTS.sha256_file(receipt_path),
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
    _component_objectives(document.get("component_generation"), corpus.capsules)
    justification = document.get("test_justification")
    if justification is not None:
        if not isinstance(justification, Mapping) or justification.get("path") != "test_justification.json":
            raise StageGateError("frozen performance test justification is malformed")
        receipt_path = corpus.root / "test_justification.json"
        if CONTRACTS.sha256_file(receipt_path) != justification.get("sha256"):
            raise StageGateError("frozen performance test justification bytes changed")
        receipt = CONTRACTS.mapping_file(receipt_path)
        if (
            receipt.get("schema") not in {"merlin.phase2.test_justification.v1", "merlin.phase2.test_justification.v2"}
            or receipt.get("status") != "diagnostic_unmeasured"
        ):
            raise StageGateError("frozen performance test justification claims an invalid status")
        if receipt.get("target") != document.get("target") or receipt.get("source", {}).get(
            "provenance_sha256"
        ) != document.get("source", {}).get("provenance_sha256"):
            raise StageGateError("frozen performance test justification identity differs")
        source = receipt.get("source") or {}
        inputs = justification.get("inputs") or {}
        for name, field in (
            ("evidence-manifest.json", "manifest_sha256"),
            ("performance-facts.json", "performance_facts_sha256"),
            ("operation-accounting.json", "operation_accounting_sha256"),
            ("performance-basis.json", "performance_basis_sha256"),
        ):
            digest = CONTRACTS.sha256_file(corpus.root / "test_justification_inputs" / name)
            if digest != inputs.get(name) or digest != source.get(field):
                raise StageGateError(f"frozen performance test justification input changed: {name}")
        scope_sha = source.get("selected_scope_sha256")
        requirement_sha = source.get("selected_requirement_sha256")
        if (scope_sha is None) != (requirement_sha is None):
            raise StageGateError("frozen performance scope source binding is incomplete")
        if scope_sha is not None:
            if justification.get("selected_requirement_sha256") != requirement_sha:
                raise StageGateError("frozen performance scope selected-source identity changed")
            scope_path = corpus.root / "test_justification_inputs" / "performance-scope.json"
            if (
                any(path.is_symlink() for path in (scope_path, scope_path.parent, corpus.root))
                or not scope_path.is_file()
            ):
                raise StageGateError("frozen performance scope bytes changed")
            scope_bytes = scope_path.read_bytes()
            if (
                hashlib.sha256(scope_bytes).hexdigest() != scope_sha
                or inputs.get("performance-scope.json") != scope_sha
            ):
                raise StageGateError("frozen performance scope bytes changed")
            from merlin_experiments.phase0.performance_scope import validate_performance_scope

            try:
                scope = json.loads(scope_bytes)
                if not isinstance(scope, dict):
                    raise ValueError("projection must be a mapping")
                validate_performance_scope(scope)
            except (ValueError, TypeError) as exc:
                raise StageGateError(f"frozen performance scope is invalid: {exc}") from exc
        elif (
            "selected_requirement_sha256" in justification
            or "performance-scope.json" in inputs
            or (corpus.root / "test_justification_inputs/performance-scope.json").exists()
        ):
            raise StageGateError("frozen performance scope lacks a selected-source binding")
        planned = {str(row.get("relative_path")): row for row in receipt.get("members") or []}
        if not planned or len(planned) != len(receipt.get("members") or []):
            raise StageGateError("frozen performance test justification has duplicate or empty members")
        for member in corpus.capsules:
            plan = planned.get(member.source_relative_path)
            if (
                not isinstance(plan, Mapping)
                or plan.get("capsule_tree_sha256") != member.source_sha256
                or plan.get("performance_sha256") != CONTRACTS.document_sha256(member.descriptor["performance"])
                or (plan.get("measurement") or {}).get("status") != "unmeasured"
            ):
                raise StageGateError(f"frozen performance member lacks an unmeasured test plan: {member.capsule}")
    selection_context = document.get("representative_selection")
    if selection_context is not None:
        from merlin_experiments.phase2 import representative_selection as RS

        if (
            not isinstance(selection_context, Mapping)
            or selection_context.get("path") != "representative_selection.json"
        ):
            raise StageGateError("frozen representative selection declaration is malformed")
        selection_path = corpus.root / "representative_selection.json"
        if selection_path.is_symlink() or CONTRACTS.sha256_file(selection_path) != selection_context.get("sha256"):
            raise StageGateError("frozen representative selection bytes changed")
        if justification is None:
            raise StageGateError("frozen representative selection lacks test justification")
        scope_path = corpus.root / "test_justification_inputs/performance-scope.json"
        scope = CONTRACTS.mapping_file(scope_path) if scope_path.is_file() else None
        expected = RS.derive(CONTRACTS.mapping_file(corpus.root / "test_justification.json"), scope)
        if CONTRACTS.mapping_file(selection_path) != expected:
            raise StageGateError("frozen representative selection differs from its source")
        chosen = {(row["family"], row["capsule"]) for row in expected["selected_members"]}
        if chosen != {(row.family, row.capsule) for row in corpus.capsules}:
            raise StageGateError("frozen representative members differ from the selected plan")


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

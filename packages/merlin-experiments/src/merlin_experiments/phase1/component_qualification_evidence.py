"""Reopen the private compiler, actual invocation closure and evaluated stage joins.

This owner preserves previously evaluated evidence. It cannot manufacture fresh
origin, numerical agreement, physical effects or an independent runtime role.
"""

from pathlib import Path

from merlin.common import invocation_record
from merlin_experiments.phase2 import contracts as C

from .component_qualification_members import obligations as selected_obligations
from .component_witness import (
    REQUIRED_EXECUTION_EFFECTS,
    ComponentStageFile,
    ComponentStageWitness,
    verify_component_stage_witness,
)


def evaluator_distribution() -> dict:
    """Require installed AET metadata and bind its actual imported implementation."""
    from importlib import metadata

    from merlin.common.paths import module_source_path
    from merlin.common.source_membership import python_members

    try:
        distribution = metadata.distribution("aet")
        metadata_files = {
            str(distribution.locate_file(path).resolve()): C.sha256_file(distribution.locate_file(path))
            for path in distribution.files or ()
            if ".dist-info" in str(path) and distribution.locate_file(path).is_file()
        }
        package = module_source_path("aet").resolve().parent
        sources = {
            name: {"path": str(path), "sha256": C.sha256_file(path)}
            for name, path in python_members(package, label="installed AET").items()
        }
    except (metadata.PackageNotFoundError, OSError, ImportError) as error:
        raise C.StageGateError("component qualification requires the actual installed AET distribution") from error
    if not metadata_files or not sources:
        raise C.StageGateError("component qualification has no installed AET source/metadata membership")
    return {"version": distribution.version, "metadata": metadata_files, "sources": sources}


def _direct_directory(path):
    path = Path(path).absolute()
    if not path.is_dir() or path.resolve() != path or any(owner.is_symlink() for owner in (path, *path.parents)):
        raise C.StageGateError("private component evidence owner is absent or indirect")
    return path


def invocation_members(root):
    """Inventory and reopen the complete actual records, including dependencies."""
    root = _direct_directory(root)
    records = []
    for path in sorted(root.rglob("invocation.json")):
        if any(owner.is_symlink() for owner in (path, *path.parents)) or path.resolve() != path:
            raise C.StageGateError("component invocation evidence contains indirect paths")
        invocation_record.verify(path)
        records.append({"path": str(path), "sha256": C.sha256_file(path)})
    if not records:
        raise C.StageGateError("component qualification lacks actual invocation evidence")
    return records


def require_member_invocations(report, grade_root, records, *, preparation=None, original_members=None):
    """Each original mandatory case must have its own actual observed execution path."""
    grade_root = _direct_directory(grade_root)
    paths = tuple(Path(row["path"]) for row in records)
    for obligation in selected_obligations(report, preparation=preparation, original_members=original_members):
        for member in obligation["members"]:
            name = Path(member["name"])
            if name.is_absolute() or len(name.parts) != 1 or name.parts[0] in {".", ".."}:
                raise C.StageGateError("component member invocation owner is not a direct case")
            owner = grade_root / name
            if not any(path.is_relative_to(owner) for path in paths):
                raise C.StageGateError("component member lacks actual invocation evidence: " + member["name"])


def replay_stage_witnesses(
    *,
    rows,
    report,
    corpus_root,
    grade_root,
    candidate_sha256,
    descriptor_sha256,
    preparation=None,
    original_members=None,
):
    """Recheck every complete original admitted member and source/effect join."""
    declarations = {row["id"]: row for row in report.get("declaration", {}).get("obligations", [])}
    selected = {}
    for obligation in selected_obligations(report, preparation=preparation, original_members=original_members):
        if obligation["expectation"] == "unsupported_program":
            continue
        for member in obligation["members"]:
            selected[member["name"]] = (member, declarations.get(obligation["id"], {}).get("frontend", "mlir"))
    index = {row["member_name"]: row for row in rows}
    if len(index) != len(rows) or set(index) != set(selected):
        raise C.StageGateError("component stage evidence lost its exact original member roster")
    for name, (member, frontend) in selected.items():
        row = index[name]
        capsule_root = Path(corpus_root) / member["member"]
        capsule = C.mapping_file(capsule_root / "capsule.yaml", yaml_file=True)
        effects = tuple(
            sorted(
                set(REQUIRED_EXECUTION_EFFECTS)
                | set((capsule.get("component_coverage") or {}).get("generated_effects", []))
            )
        )
        witness = ComponentStageWitness(
            row["source_program_sha256"],
            row["compiler_sha256"],
            row["member_sha256"],
            row["target_descriptor_sha256"],
            row["frontend"],
            tuple(
                ComponentStageFile(
                    item["stage"],
                    Path(item["path"]),
                    item["sha256"],
                    tuple(tuple(parent) for parent in item["parent_sha256"]),
                    tuple(item.get("facets", ())),
                )
                for item in row["stages"]
            ),
            tuple(row["output_roster"]),
            tuple(row["effect_roster"]),
            row["reference_authority"],
        )
        observed = verify_component_stage_witness(
            witness,
            member=member,
            candidate_sha256=candidate_sha256,
            target_descriptor_sha256=descriptor_sha256,
            frontend=frontend,
            capsule_root=capsule_root,
            evidence_root=Path(grade_root),
            required_effects=effects,
        )
        if {"member_name": name, **observed} != row:
            raise C.StageGateError("component stage evidence differs from its actual source/effect joins")


def verify_execution_evidence(
    *,
    document,
    compiler_root,
    grade_root,
    candidate_sha256,
    report,
    corpus_root,
    descriptor_sha256,
    preparation=None,
    original_members=None,
):
    """Require current source and all actual recorded inputs/products to match issuance."""
    if (
        document.get("compiler_snapshot_sha256") != candidate_sha256
        or C.exact_tree_record(_direct_directory(compiler_root))["sha256"] != candidate_sha256
    ):
        raise C.StageGateError("private component compiler snapshot changed after qualification")
    try:
        records = invocation_members(grade_root)
        if records != document.get("invocation_evidence"):
            raise C.StageGateError("component actual invocation membership changed after qualification")
        require_member_invocations(
            report, grade_root, records, preparation=preparation, original_members=original_members
        )
        replay_stage_witnesses(
            rows=document["stage_witnesses"],
            report=report,
            corpus_root=corpus_root,
            grade_root=grade_root,
            candidate_sha256=candidate_sha256,
            descriptor_sha256=descriptor_sha256,
            preparation=preparation,
            original_members=original_members,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise C.StageGateError("component actual execution evidence changed: " + str(error)) from error


def component_sources() -> dict:
    """Complete current Python membership of the connected implementation owners.

    The legacy Phase1 record covers named startup clients. Component grading
    additionally invokes the ordinary core compiler/grader and optional package
    owners; discover their entire current source membership, including additions.
    """
    from merlin.common.access import PYTHON_SOURCE_ROOTS
    from merlin.common.paths import python_import_roots
    from merlin.common.source_membership import python_members

    namespaces = {row.namespace for row in PYTHON_SOURCE_ROOTS}
    owners = {
        root / namespace for root in python_import_roots() for namespace in namespaces if (root / namespace).is_dir()
    }
    return {
        str(root): {
            name: {"path": str(path), "sha256": C.sha256_file(path)}
            for name, path in python_members(root, label="component qualification").items()
        }
        for root in sorted(owners)
    }

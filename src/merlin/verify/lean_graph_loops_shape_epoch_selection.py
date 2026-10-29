"""Pinned source/object/import context for the typed shape-to-epoch adapter.

The standalone shape selection remains immutable and supplies original SDK and
provider identities. This extension adds checked project objects and hosts;
selection is not independent semantic review or native allocator qualification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .lean_graph_loops import LeanSelection, _reference
from .lean_graph_loops_memory_scf_selection import load_selection as load_memory
from .lean_graph_loops_shape_selection import ShapeSelection
from .lean_graph_loops_shape_selection import load_selection as load_shape

ALLOCATOR_POLICY = "total_private_unobservable"
DESCRIPTOR_POLICY = "typed_static_identity_layout"
ENTRY_POLICY = "private_logical_epochs_disjoint_from_canonical_caller_views"
REQUIRED_ROOTS = {
    "GraphShapeEpochObligation.complete_adapter",
    "GraphShapeEpochObligation.complete_chain",
    "Merlin.GraphLoopsShape.Epoch.Kernel.checked_chain",
    "Merlin.GraphLoopsShape.Epoch.Kernel.complete_equivalent",
    "Merlin.GraphLoopsShape.Epoch.seed_eval",
    "Merlin.GraphLoopsShape.Epoch.Command.correspondence",
    "Merlin.GraphLoopsShape.Epoch.commands_equivalent",
}


def exact(row, label):
    path = Path(row["path"])
    data, actual = ShapeSelection.checked(path, row["sha256"], label)
    if actual != row:
        raise ValueError(label + " byte count/path changed")
    return data


@dataclass(frozen=True)
class EpochSelection:
    proof: LeanSelection
    parent: Path
    parent_sha256: str
    contract: Path
    contract_sha256: str

    @property
    def shape(self):
        ShapeSelection.checked(self.parent, self.parent_sha256, "original standalone selection")
        return load_shape(self.parent)

    @property
    def source(self):
        return self.shape.source

    @property
    def revision(self):
        return self.shape.revision

    def dependencies(self):
        contract, contract_ref = ShapeSelection.checked(self.contract, self.contract_sha256, "epoch contract")
        selected = dict(
            parent=str(self.parent.resolve()),
            parent_sha256=self.parent_sha256,
            proof={
                name: str(value.resolve()) if isinstance(value, Path) else value
                for name, value in vars(self.proof).items()
            },
        )
        if contract["schema"] != "merlin.graph.shape_epoch.context.v1" or contract["selected"] != selected:
            raise ValueError("replacement adapter proof/source context differs from original contract")
        if contract["policies"] != dict(allocator=ALLOCATOR_POLICY, descriptor=DESCRIPTOR_POLICY, entry=ENTRY_POLICY):
            raise ValueError("explicit adapter allocator/descriptor/entry policies required")
        shape = self.shape
        if self.proof.executable != shape.proof.executable or self.proof.sha256 != shape.proof.sha256:
            raise ValueError("adapter compiler differs from original standalone selection")
        inherited = shape.dependencies()
        if inherited != contract["parent_dependencies"] or _reference(self.parent) != contract["parent"]:
            raise ValueError("original parent dependencies differ")
        assessment = exact(contract["assessment"], "original adapter import assessment")
        if (
            not assessment["verified_unaffected"]
            or not assessment["original_bindings_exact"]
            or assessment["suspect_selected"]
            or assessment["missing"]
            or assessment["changed"]
            or assessment["returncode"] != 0
            or assessment["toolchain_written"]
            or assessment["rehash_to_promote"]
        ):
            raise ValueError("affected or unconfirmed adapter context")
        audited = exact(contract["root_audit"], "original adapter root audit")
        if audited["assessment"] != contract["assessment"]:
            raise ValueError("import assessment differs from the actual audited query context")
        if (
            audited["returncode"] != 0
            or not audited["bindings_unchanged"]
            or list(audited["audit"]) != contract["roots"]
            or not REQUIRED_ROOTS <= set(contract["roots"])
            or any(
                not set(axioms) <= {"propext", "Classical.choice", "Quot.sound"} for axioms in audited["audit"].values()
            )
        ):
            raise ValueError("complete ordered standard-three adapter audit required")
        build = json.loads(self.proof.build_record.read_bytes())
        imports = sorted(row["module"] for row in assessment["actual_imports"] if row["module"].startswith("Merlin."))
        modules = sorted(row["module"] for row in build["attempts"])
        if modules != imports or len(modules) != len(set(modules)) or modules != contract["project_modules"]:
            raise ValueError("complete actual imported project closure required")
        original_build = json.loads(shape.proof.build_record.read_bytes())
        original_modules = {row["module"]: row for row in original_build["attempts"]}
        original_modules.update({row["module"]: row for row in audited["project_modules"]})
        for row in build["attempts"]:
            original = original_modules.get(row["module"])
            if original is None or any(
                row[field][key] != original[field][key] for field in ("source", "object") for key in ("sha256", "bytes")
            ):
                raise ValueError("adapter object/source differs from originally audited bytes")
        original_fixed = json.loads(shape.fixed_context.read_bytes())
        memory = load_memory(Path(original_fixed["prior_selection"]["path"]))
        runtime = {row["path"]: row for row in json.loads(memory.fixed_context.read_bytes())["files"]}
        for row in assessment["actual_imports"]:
            if not row["module"].startswith("Merlin.") and row["path"] not in runtime:
                raise ValueError("actual SDK import has no original unaffected binding")
        manifests = [json.loads(path.read_bytes()) for path in (shape.manifest, memory.manifest)]
        for field in ("producer_source", "xdsl_loaded_source"):
            known = {}
            for manifest in manifests:
                for row in manifest[field]:
                    if row["path"] in known and known[row["path"]] != row:
                        raise ValueError("original provider/parser manifests disagree")
                    known[row["path"]] = row
            if contract[field] != [known[path] for path in sorted(known)]:
                raise ValueError("replacement provider/parser inventory differs from original selections")
        if any(runtime.get(row["path"]) != row for row in contract["additional_original_runtime_refs"]):
            raise ValueError("runtime reference is not an original binding")
        rows = contract["host_sources"] + contract["additional_original_runtime_refs"]
        for row in rows:
            path = Path(row["path"])
            first = path.stat()
            actual = _reference(path)
            last = path.stat()
            if actual != row or (first.st_ino, first.st_size, first.st_mtime_ns) != (
                last.st_ino,
                last.st_size,
                last.st_mtime_ns,
            ):
                raise ValueError("original adapter host/runtime binding changed: " + row["path"])
        return dict(
            contract=contract_ref,
            parent=_reference(self.parent),
            parent_dependencies=inherited,
            proof=self.proof.dependencies(),
            assessment=contract["assessment"],
            root_audit=contract["root_audit"],
            checked_sources=rows,
            policies=contract["policies"],
            independent_review_complete=False,
            native_model_fidelity_proved=False,
        )


def load_selection(path):
    data = json.loads(Path(path).read_bytes())
    for name in ("executable", "library", "build_record"):
        data["proof"][name] = Path(data["proof"][name])
    data["proof"] = LeanSelection(**data["proof"])
    for name in ("parent", "contract"):
        data[name] = Path(data[name])
    return EpochSelection(**data)

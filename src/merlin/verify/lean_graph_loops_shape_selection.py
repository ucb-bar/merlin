"""Original source/object/SDK selection for the standalone shape validator.

The inherited memory packet supplies fixed bytes, not a semantic allocator bridge.
A contract digest binds a selection; independent/native review is still separate.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .lean_graph_loops import LeanSelection, _reference
from .lean_graph_loops_memory_scf_selection import load_selection as load_parent
from .lean_graph_loops_shape_model import SHAPE_ROOTS


@dataclass(frozen=True)
class ShapeSelection:
    proof: LeanSelection
    source: Path
    revision: str
    manifest: Path
    manifest_sha256: str
    fixed_context: Path
    fixed_context_sha256: str
    contract: Path
    contract_sha256: str

    @staticmethod
    def checked(path, digest, label):
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("originally selected " + label + " changed")
        return json.loads(data), _reference(path, data)

    def dependencies(self):
        contract, contract_ref = self.checked(self.contract, self.contract_sha256, "shape contract")
        context = {
            name: str(value.resolve()) if isinstance(value, Path) else value
            for name, value in vars(self).items()
            if name not in ("proof", "contract", "contract_sha256")
        }
        proof = {
            name: str(value.resolve()) if isinstance(value, Path) else value for name, value in vars(self.proof).items()
        }
        if contract["selected_context"] != dict(context, proof=proof):
            raise ValueError("replacement shape proof/source/tool context differs from original policy")
        if (
            contract["descriptor_policy"] != "typed_static_identity_layout"
            or contract["allocator_policy"] != "retained_partial_observable"
        ):
            raise ValueError("unsupported shape descriptor/operator/allocator policy")
        fixed, fixed_ref = self.checked(self.fixed_context, self.fixed_context_sha256, "original unaffected context")
        parent = load_parent(Path(fixed["prior_selection"]["path"]))
        if (
            _reference(Path(fixed["prior_selection"]["path"])) != fixed["prior_selection"]
            or parent.dependencies() != fixed["prior_dependencies"]
        ):
            raise ValueError("original parent source/object/import/host bindings changed")
        manifest, manifest_ref = self.checked(
            self.manifest, self.manifest_sha256, "actual shape provider/parser manifest"
        )
        observed = subprocess.check_output(["git", "-C", str(self.source), "rev-parse", "HEAD"], text=True).strip()
        if observed != self.revision or manifest["producer_revision"] != self.revision:
            raise ValueError("selected actual shape provider revision changed")
        build = json.loads(self.proof.build_record.read_bytes())
        if sorted(row["module"] for row in build["attempts"]) != contract["proof_modules"]:
            raise ValueError("complete selected shape source/object closure required")
        assessment_ref, audit_ref = contract["unaffected_assessment"], contract["root_audit"]
        assessment, _ = self.checked(Path(assessment_ref["path"]), assessment_ref["sha256"], "import assessment")
        audit, _ = self.checked(Path(audit_ref["path"]), audit_ref["sha256"], "root audit")
        if (
            not assessment["verified_unaffected"]
            or assessment["suspect_selected"]
            or assessment["rehash_to_promote"]
            or assessment["toolchain_written"]
            or assessment["returncode"] != 0
            or assessment["missing_original_bindings"]
            or assessment["changed"]
            or not assessment["original_bindings_preserved"]
            or assessment["original_context"] != fixed_ref
        ):
            raise ValueError("shape context affected/unconfirmed or repaired")
        project_modules = sorted(
            row["module"] for row in assessment["actual_imports"] if row["module"].startswith("Merlin.")
        )
        if project_modules != contract["proof_modules"]:
            raise ValueError("selected project closure differs from the actual inspected imports")
        if audit["returncode"] != 0 or not audit["bindings_unchanged"] or set(audit["audit"]) != set(SHAPE_ROOTS):
            raise ValueError("successful exact-root transitive shape audit required")
        if any(not set(names) <= {"propext", "Classical.choice", "Quot.sound"} for names in audit["audit"].values()):
            raise ValueError("unapproved shape axiom")
        if not manifest["producer_source"] or not manifest["xdsl_loaded_source"]:
            raise ValueError("actual loaded shape parser/provider inventory missing")
        rows = (
            [assessment_ref, audit_ref]
            + manifest["producer_source"]
            + manifest["xdsl_loaded_source"]
            + contract["host_sources"]
        )
        rows += [row["object"] for row in fixed["copied_objects"]]
        for row in rows:
            expected = {name: row[name] for name in ("path", "sha256", "bytes")}
            path = Path(row["path"])
            before = path.stat()
            actual = _reference(path)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ) or actual != expected:
                raise ValueError("original shape source/host/object binding changed: " + row["path"])
        return dict(
            proof=self.proof.dependencies(),
            contract=contract_ref,
            provider_manifest=manifest_ref,
            original_fixed_context=fixed_ref,
            original_parent_bindings=fixed["prior_dependencies"],
            checked_sources=rows,
            allocator_policy=contract["allocator_policy"],
            descriptor_fidelity_and_review_pending=True,
            original_imports_verified_unaffected=True,
            old_allocator_premise_not_assumed_for_standalone_shape=True,
        )


def load_selection(path):
    data = json.loads(Path(path).read_bytes())
    for name in ("executable", "library", "build_record"):
        data["proof"][name] = Path(data["proof"][name])
    data["proof"] = LeanSelection(**data["proof"])
    for name in ("source", "manifest", "fixed_context", "contract"):
        data[name] = Path(data[name])
    return ShapeSelection(**data)

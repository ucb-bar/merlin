"""Pinned independent proof/parser/host context for the memory-SCF validator.

The selected contract is a policy binding, not proof of native conformance or
independent review. Every invocation checks original source/object/import pins.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .lean_graph_loops import LeanSelection, _reference
from .lean_graph_loops_memory_scf_model import MEMORY_SCF_ROOTS


@dataclass(frozen=True)
class MemorySCFSelection:
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
    def checked_json(path, digest, label):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("originally selected " + label + " changed")
        return json.loads(data), _reference(path, data)

    def dependencies(self):
        contract, contract_ref = self.checked_json(self.contract, self.contract_sha256, "memory-SCF contract")
        if contract["allocator_mode"] != "total_private_unobservable":
            raise ValueError("unsupported allocator contract; finite-capacity counterexample remains")
        context = {
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(self).items()
            if key not in ("contract", "contract_sha256", "proof")
        }
        proof = {
            key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(self.proof).items()
        }
        if contract["selected_context"] != dict(context, proof=proof):
            raise ValueError("replacement proof/source/tool selection differs from original contract")
        manifest, manifest_ref = self.checked_json(self.manifest, self.manifest_sha256, "provider/parser manifest")
        if (
            manifest["producer_revision"] != self.revision
            or subprocess.check_output(["git", "-C", str(self.source), "rev-parse", "HEAD"], text=True).strip()
            != self.revision
        ):
            raise ValueError("selected actual provider revision changed")
        if not manifest["producer_source"] or not manifest["xdsl_loaded_source"]:
            raise ValueError("actual provider/parser inventory missing")
        fixed, fixed_ref = self.checked_json(self.fixed_context, self.fixed_context_sha256, "unaffected fixed context")
        if (
            not fixed["fixed_context_verified_unaffected"]
            or fixed["suspect_in_actual_closure"]
            or fixed["toolchain_written"]
            or fixed["rehash_to_promote"]
        ):
            raise ValueError("fixed context is affected, unconfirmed or repaired")
        build = json.loads(self.proof.build_record.read_bytes())
        if sorted(row["module"] for row in build["attempts"]) != contract["proof_modules"]:
            raise ValueError("complete selected project proof/module closure required")
        audit_ref = contract["root_audit"]
        audit = json.loads(Path(audit_ref["path"]).read_bytes())
        if audit["returncode"] != 0 or not audit["bindings_unchanged"] or set(audit["audit"]) != set(MEMORY_SCF_ROOTS):
            raise ValueError("successful exact-root transitive standard3 audit required")
        if any(not set(names) <= {"propext", "Classical.choice", "Quot.sound"} for names in audit["audit"].values()):
            raise ValueError("unapproved audited axiom")
        rows = (
            [audit_ref]
            + manifest["producer_source"]
            + manifest["xdsl_loaded_source"]
            + fixed["files"]
            + fixed["prior_dependencies"]
            + contract["owned_host_sources"]
        )
        for row in fixed["copied_objects"]:
            rows.append(row["object"])
        checked = []
        for row in rows:
            expected = {key: row[key] for key in ("path", "sha256", "bytes")}
            path = Path(row["path"])
            before = path.stat()
            actual = _reference(path)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ) or actual != expected:
                raise ValueError("original source/object/import/host binding changed: " + row["path"])
            checked.append(actual)
        return dict(
            proof=self.proof.dependencies(),
            contract=contract_ref,
            provider_manifest=manifest_ref,
            fixed_context=fixed_ref,
            original_bindings=checked,
            review_status=contract["review_status"],
            allocator_mode=contract["allocator_mode"],
            fixed_context_verified_unaffected=True,
        )


def load_selection(path):
    data = json.loads(Path(path).read_bytes())
    for key in ("executable", "library", "build_record"):
        data["proof"][key] = Path(data["proof"][key])
    data["proof"] = LeanSelection(**data["proof"])
    for key in ("source", "manifest", "fixed_context", "contract"):
        data[key] = Path(data[key])
    return MemorySCFSelection(**data)

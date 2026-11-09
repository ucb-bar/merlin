"""Retain compiler variants observed inside the controller's frozen edit scope.

These snapshots are issued from the current authorized compiler, never imported
from a compiler directory/report supplied as a tuning answer. Retaining several
legal emissions lets an independent measurement cohort compare schedules while
holding every variant of the same mathematical workload together.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

from merlin.benchharness import hash_tree

from .component_execution import ComponentIndependentExecution
from .contracts import StageGateError, document_sha256, exact_tree_record

_ISSUED = WeakKeyDictionary()


@dataclass(frozen=True, eq=False)
class ComponentVariantSnapshot:
    execution: ComponentIndependentExecution
    compiler: Path
    compiler_sha256: str
    membership_sha256: str
    execution_sha256: str
    baseline_admission_sha256: str
    edit_authority_sha256: str

    @property
    def sha256(self):
        return document_sha256({
            "schema": "merlin.component_variant_snapshot.v1",
            "compiler_sha256": self.compiler_sha256, "membership_sha256": self.membership_sha256,
            "execution_sha256": self.execution_sha256,
            "baseline_admission_sha256": self.baseline_admission_sha256,
            "edit_authority_sha256": self.edit_authority_sha256,
        })

    def verify(self):
        issued = _ISSUED.get(self)
        if issued is None or issued != self.sha256:
            raise StageGateError("component variant requires live controller-owned source capture")
        self.execution.verify()
        if (self.execution.sha256 != self.execution_sha256
            or self.execution.baseline_admission.sha256 != self.baseline_admission_sha256
            or self.execution.edit_authority_sha256 != self.edit_authority_sha256):
            raise StageGateError("component variant belongs to another fresh source/edit authority")
        if (self.compiler.resolve() != self.compiler or self.compiler.is_symlink()
            or str(hash_tree(self.compiler)["sha256"]) != self.compiler_sha256
            or exact_tree_record(self.compiler)["sha256"] != self.membership_sha256):
            raise StageGateError("component variant source membership changed after capture")
        for path in (self.compiler, *self.compiler.rglob("*")):
            if path.is_symlink() or path.stat().st_mode & 0o222:
                raise StageGateError("component variant is not a retained immutable source snapshot")
        self.execution.edit_authority.validate_candidate(self.compiler)
        return self.sha256


def capture_component_variant(*, execution, output):
    """Capture only the actual current source after fresh-origin/edit inspection."""
    if type(execution) is not ComponentIndependentExecution:
        raise StageGateError("component variant capture requires independent normal execution authority")
    execution.verify()
    output = Path(output)
    if (output.exists() or output.is_symlink() or output.resolve() != output
        or not output.is_relative_to(execution.output)):
        raise StageGateError("component variant requires a fresh controller-owned private output")
    before = exact_tree_record(execution.candidate)["sha256"]
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    shutil.copytree(execution.candidate, output)
    if (exact_tree_record(output)["sha256"] != before
        or exact_tree_record(execution.candidate)["sha256"] != before):
        raise StageGateError("component source changed during controlled variant capture")
    for path in output.rglob("*"):
        path.chmod(path.stat().st_mode & ~0o222)
    output.chmod(0o555)
    execution.verify()
    variant = ComponentVariantSnapshot(
        execution, output, str(hash_tree(output)["sha256"]), before, execution.sha256,
        execution.baseline_admission.sha256, execution.edit_authority_sha256,
    )
    _ISSUED[variant] = variant.sha256
    variant.verify()
    return variant
